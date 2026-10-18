# [MODULE DOC]
# Base tests for item assignment on extension arrays and on series and frames holding them:
# scalars, sequences, masks, integer arrays, slices, expansion of frames, deletion, read-only
# behavior. Public contents: BaseSetitemTests mixin, with two helper fixtures. Uses data,
# na_value, invalid_scalar, box_in_series, as_array and dtype fixtures from the conftest.
# Many tests mutate the data fixture in place.
import numpy as np
import pytest

from pandas.core.dtypes.common import is_hashable

import pandas as pd
import pandas._testing as tm


class BaseSetitemTests:
    # [CLASS DOC]
    # Mixin with set-item tests. Contains the full_indexer fixture and an autouse fixture that
    # skips tests for immutable dtypes.
    @pytest.fixture(
        params=[
            lambda x: x.index,
            lambda x: list(x.index),
            lambda x: slice(None),
            lambda x: slice(0, len(x)),
            lambda x: range(len(x)),
            lambda x: list(range(len(x))),
            lambda x: np.ones(len(x), dtype=bool),
        ],
        ids=[
            "index",
            "list[index]",
            "null_slice",
            "full_slice",
            "range",
            "list(range)",
            "mask",
        ],
    )
    def full_indexer(self, request):
        # [FUNCTION DOC]
        # HELPER: parametrized fixture returning a function that maps an object to an indexer covering
        # its full length. Variants: the index itself, a list of the index, a null slice, a full-range
        # slice, a range, a list of positions and an all-true boolean mask (ids index, list[index],
        # null_slice, full_slice, range, list(range), mask). Some variants assume a default range
        # index.
        """
        Fixture for an indexer to pass to obj.loc to get/set the full length of the
        object.

        In some cases, assumes that obj.index is the default RangeIndex.
        """
        return request.param

    @pytest.fixture(autouse=True)
    def skip_if_immutable(self, dtype, request):
        # [FUNCTION DOC]
        # HELPER: autouse fixture that skips tests when the dtype is immutable. Inputs: dtype and the
        # request fixture. It never skips the immutability test itself. When this class is mixed into the
        # combined extension test class, only tests whose function's qualified name starts with this
        # class's name are skipped; tests defined elsewhere are left alone.
        if dtype._is_immutable:
            node = request.node
            if node.name.split("[")[0] == "test_is_immutable":
                # This fixture is auto-used, but we want to not-skip
                # test_is_immutable.
                return

            # When BaseSetitemTests is mixed into ExtensionTests, we only
            #  want this fixture to operate on the tests defined in this
            #  class/file.
            defined_in = node.function.__qualname__.split(".")[0]
            if defined_in == "BaseSetitemTests":
                pytest.skip("__setitem__ test not applicable with immutable dtype")

    def test_is_immutable(self, data):
        # [FUNCTION DOC]
        # If the dtype is immutable, assigning an element must raise a type error; otherwise assigning
        # element one to position zero works and position zero equals element one afterward.
        if data.dtype._is_immutable:
            with pytest.raises(TypeError):
                data[0] = data[0]
        else:
            data[0] = data[1]
            assert data[0] == data[1]

    def test_setitem_scalar_series(self, data, box_in_series):
        # [FUNCTION DOC]
        # Uses box_in_series. Sets position zero to element one, on the array or on a series of it,
        # and checks the value was set.
        if box_in_series:
            data = pd.Series(data)
        data[0] = data[1]
        assert data[0] == data[1]

    def test_setitem_sequence(self, data, box_in_series):
        # [FUNCTION DOC]
        # Uses box_in_series. Assigning a two-element list of swapped elements into positions zero
        # and one swaps them (checked against a copy made beforehand).
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()

        data[[0, 1]] = [data[1], data[0]]
        assert data[0] == original[1]
        assert data[1] == original[0]

    def test_setitem_sequence_mismatched_length_raises(self, data, as_array):
        # [FUNCTION DOC]
        # Uses as_array (value given as plain list or as an extension array). Assigning a one-element
        # value to a two-position list indexer must raise a value error saying it cannot set using a
        # list-like indexer with a different length; the same with a three-position slice says slice
        # indexer. The series must be unchanged after each failure.
        ser = pd.Series(data)
        original = ser.copy()
        value = [data[0]]
        if as_array:
            value = data._from_sequence(value, dtype=data.dtype)

        xpr = "cannot set using a {} indexer with a different length"
        with pytest.raises(ValueError, match=xpr.format("list-like")):
            ser[[0, 1]] = value
        # Ensure no modifications made before the exception
        tm.assert_series_equal(ser, original)

        with pytest.raises(ValueError, match=xpr.format("slice")):
            ser[slice(3)] = value
        tm.assert_series_equal(ser, original)

    def test_setitem_empty_indexer(self, data, box_in_series):
        # [FUNCTION DOC]
        # Uses box_in_series. Assigning an empty list through an empty integer indexer leaves the data
        # unchanged.
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()
        data[np.array([], dtype=int)] = []
        tm.assert_equal(data, original)

    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        # [FUNCTION DOC]
        # Uses box_in_series. Assigning a scalar element through a two-position list indexer sets
        # both positions to it.
        if box_in_series:
            data = pd.Series(data)
        data[[0, 1]] = data[2]
        assert data[0] == data[2]
        assert data[1] == data[2]

    @pytest.mark.parametrize("setter", ["loc", "iloc"])
    def test_setitem_scalar(self, data, setter):
        # [FUNCTION DOC]
        # Parametrized over label-based and position-based setters. Sets position zero of a series to
        # element one through the accessor and checks it.
        arr = pd.Series(data)
        setter = getattr(arr, setter)
        setter[0] = data[1]
        assert arr[0] == data[1]

    def test_setitem_loc_scalar_mixed(self, data):
        # [FUNCTION DOC]
        # Frame with an integer column and an extension column: label-based scalar assignment to
        # the extension cell works.
        df = pd.DataFrame({"A": np.arange(len(data)), "B": data})
        df.loc[0, "B"] = data[1]
        assert df.loc[0, "B"] == data[1]

    def test_setitem_loc_scalar_single(self, data):
        # [FUNCTION DOC]
        # Single-extension-column frame: label-based assignment to a new row label (ten) expands the
        # frame and the cell has the assigned value.
        df = pd.DataFrame({"B": data})
        df.loc[10, "B"] = data[1]
        assert df.loc[10, "B"] == data[1]

    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        # [FUNCTION DOC]
        # Two extension columns: label-based assignment of a new row label (ten) in one column works.
        df = pd.DataFrame({"A": data, "B": data})
        df.loc[10, "B"] = data[1]
        assert df.loc[10, "B"] == data[1]

    def test_setitem_iloc_scalar_mixed(self, data):
        # [FUNCTION DOC]
        # Frame with an integer column and an extension column: position-based scalar assignment to
        # the extension cell works.
        df = pd.DataFrame({"A": np.arange(len(data)), "B": data})
        df.iloc[0, 1] = data[1]
        assert df.loc[0, "B"] == data[1]

    def test_setitem_iloc_scalar_single(self, data):
        # [FUNCTION DOC]
        # Single-extension-column frame: position-based assignment to row nine works (requires at
        # least ten elements).
        df = pd.DataFrame({"B": data})
        df.iloc[9, 0] = data[1]
        assert df.loc[9, "B"] == data[1]

    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        # [FUNCTION DOC]
        # Two extension columns: position-based assignment to row nine of the second column works.
        df = pd.DataFrame({"A": data, "B": data})
        df.iloc[9, 1] = data[1]
        assert df.loc[9, "B"] == data[1]

    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        # [FUNCTION DOC]
        # Parametrized over a numpy boolean mask, a nullable boolean mask, and one with missing values
        # (treated as false), and over box_in_series. Assigning element zero where the mask is true on a
        # five-element copy gives the first three positions equal to element zero.
        arr = data[:5].copy()
        expected = arr.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)
        arr[mask] = data[0]
        tm.assert_equal(expected, arr)

    def test_setitem_mask_raises(self, data, box_in_series):
        # [FUNCTION DOC]
        # Uses box_in_series. A mask of the wrong length (two) as numpy and as nullable boolean array
        # must raise an index error mentioning wrong length when used for assignment.
        # wrong length
        mask = np.array([True, False])

        if box_in_series:
            data = pd.Series(data)

        with pytest.raises(IndexError, match="wrong length"):
            data[mask] = data[0]

        mask = pd.array(mask, dtype="boolean")
        with pytest.raises(IndexError, match="wrong length"):
            data[mask] = data[0]

    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        # [FUNCTION DOC]
        # Uses box_in_series. Nullable boolean mask with true in the first three and missing in the
        # next two; assigning element zero sets the first three positions to it.
        mask = pd.array(np.zeros(data.shape, dtype="bool"), dtype="boolean")
        mask[:3] = True
        mask[3:5] = pd.NA

        if box_in_series:
            data = pd.Series(data)

        data[mask] = data[0]

        assert (data[:3] == data[0]).all()

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series):
        # [FUNCTION DOC]
        # Parametrized over a list, a nullable integer array and numpy array of positions 0-2 and
        # box_in_series. Assigning element zero through the integer indexer into a five-element copy
        # equals the take of positions 0,0,0,3,4.
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])

        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        arr[idx] = arr[0]
        tm.assert_equal(arr, expected)

    @pytest.mark.parametrize(
        "idx",
        [[0, 0, 1], pd.array([0, 0, 1], dtype="Int64"), np.array([0, 0, 1])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array_with_repeats(self, data, idx, box_in_series):
        # [FUNCTION DOC]
        # Parametrized over indexer kinds with repeated positions (0,0,1) and box_in_series. Assigning
        # three values to positions 0,0,1 yields the last-write-wins arrangement 2,3,2,3,4 (after the
        # take).
        arr = data[:5].copy()
        expected = data.take([2, 3, 2, 3, 4])

        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        arr[idx] = [arr[2], arr[2], arr[3]]
        tm.assert_equal(arr, expected)

    @pytest.mark.parametrize(
        "idx",
        [
            [0, 1, 2, pd.NA],
            pd.array([0, 1, 2, pd.NA], dtype="Int64"),
        ],
        ids=["list", "integer-array"],
    )
    @pytest.mark.parametrize("box_in_series", [True, False])
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        # [FUNCTION DOC]
        # Parametrized over a list and nullable integer array with a missing entry, and over
        # box_in_series (the parameter overrides the fixture). For a bare array, assignment must raise a
        # value error saying integer indexers cannot contain NA values. For a series with string index,
        # assignment raises a key error whose message is the string zero (labels are not found).
        arr = data.copy()

        msg = "Cannot index with an integer indexer containing NA values"
        err = ValueError

        if box_in_series:
            # The integer labels are not present in the (string) index, so
            #  we get KeyErrors
            arr = pd.Series(data, index=[chr(100 + i) for i in range(len(data))])
            msg = "0"
            err = KeyError

        with pytest.raises(err, match=msg):
            arr[idx] = arr[0]

    @pytest.mark.parametrize("as_callable", [True, False])
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_aligned(self, data, as_callable, setter):
        # [FUNCTION DOC]
        # Parametrized over mask given as a callable or not, and setter label-based or plain.
        # Assigns two values (elements 5 and 6) to the two masked positions and asserts they were set.
        # KNOWN ISSUE: after assigning through the chosen target, the test assigns again directly on
        # the series with plain indexing and checks that result, so the label-based variant of the
        # setter is not what is asserted on (both parametrizations end up exercising the plain
        # assignment).
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        if as_callable:
            mask2 = lambda x: mask
        else:
            mask2 = mask

        if setter:
            # loc
            target = getattr(ser, setter)
        else:
            # Series.__setitem__
            target = ser

        target[mask2] = data[5:7]

        ser[mask2] = data[5:7]
        assert ser[0] == data[5]
        assert ser[1] == data[6]

    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        # [FUNCTION DOC]
        # Parametrized over label-based or plain setter. Assigning a scalar element through a boolean
        # mask for the first two positions sets both of them.
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        if setter:  # loc
            target = getattr(ser, setter)
        else:  # __setitem__
            target = ser

        target[mask] = data[9]
        assert ser[0] == data[9]
        assert ser[1] == data[9]

    def test_setitem_expand_columns(self, data):
        # [FUNCTION DOC]
        # Adding a scalar column by plain assignment and via label-based full-column assignment gives
        # the expected frame; then overwriting the new column with the extension data gives a frame with
        # two extension columns.
        df = pd.DataFrame({"A": data})
        result = df.copy()
        result["B"] = 1
        expected = pd.DataFrame({"A": data, "B": [1] * len(data)})
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.loc[:, "B"] = 1
        tm.assert_frame_equal(result, expected)

        # overwrite with new type
        result["B"] = data
        expected = pd.DataFrame({"A": data, "B": data})
        tm.assert_frame_equal(result, expected)

    def test_setitem_expand_with_extension(self, data):
        # [FUNCTION DOC]
        # Adding an extension column to a frame with an integer column, by plain assignment and via
        # label-based full-column assignment, gives the expected frame.
        df = pd.DataFrame({"A": [1] * len(data)})
        result = df.copy()
        result["B"] = data
        expected = pd.DataFrame({"A": [1] * len(data), "B": data})
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.loc[:, "B"] = data
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_with_expansion_preserves_ea_index_dtype(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Skips for non-hashable elements. Uses the unique values as
        # the index of a frame and series with one entry missing; label-based assignment of the missing
        # label adds the row/entry and the resulting index keeps the extension dtype.
        # GH#41626 retain index.dtype in setitem-with-expansion
        if not is_hashable(data[0]):
            pytest.skip("Test does not apply to non-hashable data.")
        data = data.unique()
        expected = pd.DataFrame({"A": range(len(data))}, index=data)
        df = expected.iloc[:-1]
        ser = df["A"]
        item = data[-1]

        df.loc[item] = len(data) - 1
        tm.assert_frame_equal(df, expected)

        ser.loc[item] = len(data) - 1
        tm.assert_series_equal(ser, expected["A"])

    def test_setitem_frame_invalid_length(self, data):
        # [FUNCTION DOC]
        # Assigning a five-element array as a new column of a longer frame must raise a value error
        # that states the lengths of values and index.
        df = pd.DataFrame({"A": [1] * len(data)})
        xpr = (
            rf"Length of values \({len(data[:5])}\) "
            rf"does not match length of index \({len(df)}\)"
        )
        with pytest.raises(ValueError, match=xpr):
            df["B"] = data[:5]

    def test_setitem_tuple_index(self, data):
        # [FUNCTION DOC]
        # A series with tuple labels: assigning through a tuple key sets that element.
        ser = pd.Series(data[:2], index=[(0, 0), (0, 1)])
        expected = pd.Series(data.take([1, 1]), index=ser.index)
        ser[(0, 0)] = data[1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_slice(self, data, box_in_series):
        # [FUNCTION DOC]
        # Uses box_in_series. Assigning element zero to the first three entries of a five-element copy
        # equals the take of positions 0,0,0,3,4.
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        arr[:3] = data[0]
        tm.assert_equal(arr, expected)

    def test_setitem_loc_iloc_slice(self, data):
        # [FUNCTION DOC]
        # Series with string index: position-slice assignment and label-slice assignment (inclusive end)
        # of element zero over the first three entries both produce the expected series.
        arr = data[:5].copy()
        s = pd.Series(arr, index=["a", "b", "c", "d", "e"])
        expected = pd.Series(data.take([0, 0, 0, 3, 4]), index=s.index)

        result = s.copy()
        result.iloc[:3] = data[0]
        tm.assert_equal(result, expected)

        result = s.copy()
        result.loc[:"c"] = data[0]
        tm.assert_equal(result, expected)

    def test_setitem_slice_mismatch_length_raises(self, data):
        # [FUNCTION DOC]
        # Assigning a two-element array to a one-element slice must raise a value error originating
        # outside pandas.
        arr = data[:5]
        with tm.external_error_raised(ValueError):
            arr[:1] = arr[:2]

    def test_setitem_slice_array(self, data):
        # [FUNCTION DOC]
        # Assigning the last five elements into the first five via a slice produces an array equal
        # to those elements.
        arr = data[:5].copy()
        arr[:5] = data[-5:]
        tm.assert_extension_array_equal(arr, data[-5:])

    def test_setitem_scalar_key_sequence_raise(self, data):
        # [FUNCTION DOC]
        # Assigning a two-element array to a single position must raise a value error originating
        # outside pandas.
        arr = data[:5].copy()
        with tm.external_error_raised(ValueError):
            arr[0] = arr[[0, 1]]

    def test_setitem_preserves_views(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue: assignment must not swap the underlying data. A view and a
        # full slice taken before assignment both see the new value afterwards.
        # GH#28150 setitem shouldn't swap the underlying data
        view1 = data.view()
        view2 = data[:]

        data[0] = data[1]
        assert view1[0] == data[1]
        assert view2[0] == data[1]

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Using the full_indexer fixture: starting from an empty frame
        # with the data's index, assigning the full extension column through label-based indexing with
        # the full-length indexer produces a frame equal to the original.
        # https://github.com/pandas-dev/pandas/issues/32395
        df = expected = pd.DataFrame(pd.Series(data))
        result = pd.DataFrame(index=df.index)

        key = full_indexer(df)
        result.loc[key, 0] = df[0]

        tm.assert_frame_equal(result, expected)

    def test_setitem_with_expansion_row(self, data, na_value):
        # [FUNCTION DOC]
        # Assigning to a new row label in a one-row frame extends the extension column; assigning
        # the missing value to another new row extends it again with the missing value (regression for
        # an upstream issue). Dtype is preserved.
        df = pd.DataFrame({"data": data[:1]})

        df.loc[1, "data"] = data[1]
        expected = pd.DataFrame({"data": data[:2]})
        tm.assert_frame_equal(df, expected)

        # https://github.com/pandas-dev/pandas/issues/47284
        df.loc[2, "data"] = na_value
        expected = pd.DataFrame(
            {"data": pd.Series([data[0], data[1], na_value], dtype=data.dtype)}
        )
        tm.assert_frame_equal(df, expected)

    def test_setitem_series(self, data, full_indexer):
        # [FUNCTION DOC]
        # Using the full_indexer fixture: assigning a series of extension data through a full-length
        # label-based indexer into an object-dtype series keeps the object dtype (setting happens in
        # place) and the values equal the data cast to object.
        # https://github.com/pandas-dev/pandas/issues/32395
        ser = pd.Series(data, name="data")
        result = pd.Series(index=ser.index, dtype=object, name="data")

        # because result has object dtype, the attempt to do setting inplace
        #  is successful, and object dtype is retained
        key = full_indexer(ser)
        result.loc[key] = ser

        expected = pd.Series(
            data.astype(object), index=ser.index, name="data", dtype=object
        )
        tm.assert_series_equal(result, expected)

    def test_setitem_frame_2d_values(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Assigning a frame, a partial frame, the 2D values array and a
        # partial values array back into a single-column frame via position-based indexing leaves the frame
        # unchanged.
        # GH#44514
        df = pd.DataFrame({"A": data})
        orig = df.copy()

        df.iloc[:] = df.copy()
        tm.assert_frame_equal(df, orig)

        df.iloc[:-1] = df.iloc[:-1].copy()
        tm.assert_frame_equal(df, orig)

        df.iloc[:] = df.values
        tm.assert_frame_equal(df, orig)

        df.iloc[:-1] = df.values[:-1]
        tm.assert_frame_equal(df, orig)

    def test_delitem_series(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Deleting the entry with label one removes the second
        # element and equals selecting the other positions.
        # GH#40763
        ser = pd.Series(data, name="data")

        taker = np.arange(len(ser))
        taker = np.delete(taker, 1)

        expected = ser[taker]
        del ser[1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_invalid(self, data, invalid_scalar):
        # [FUNCTION DOC]
        # Using the invalid_scalar fixture: assigning it to one position or to everything must raise a
        # value error or a type error (messages vary by subclass and are not checked).
        # messages vary by subclass, so we do not test it
        with pytest.raises((ValueError, TypeError), match=None):
            data[0] = invalid_scalar

        with pytest.raises((ValueError, TypeError), match=None):
            data[:] = invalid_scalar

    def test_setitem_2d_values(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Assigning the transposed-rows values array of a two-column
        # frame to rows zero and one in swapped order swaps those rows.
        # GH50085
        original = data.copy()
        df = pd.DataFrame({"a": data, "b": data})
        df.loc[[0, 1], :] = df.loc[[1, 0], :].values
        assert (df.loc[0, :] == original[1]).all()
        assert (df.loc[1, :] == original[0]).all()

    def test_readonly_property(self, data):
        # [FUNCTION DOC]
        # The private read-only flag starts false; after setting it true on the fixture, a copy is
        # writable (false), and assigning to a position, a slice, or a boolean mask raises a value error
        # saying the array is read-only, leaving the data unchanged. Mutates the fixture.
        assert data._readonly is False

        data._readonly = True
        assert data._readonly is True

        data_orig = data.copy()
        assert data_orig._readonly is False

        with pytest.raises(ValueError, match="Cannot modify read-only array"):
            data[0] = data[1]

        with pytest.raises(ValueError, match="Cannot modify read-only array"):
            data[0:3] = data[1]

        with pytest.raises(ValueError, match="Cannot modify read-only array"):
            data[np.array([True] * len(data))] = data[1]

        tm.assert_extension_array_equal(data, data_orig)

    def test_readonly_propagates_to_numpy_array(self, data):
        # [FUNCTION DOC]
        # With the array flagged read-only: a numpy conversion that copies is writable. Two conversions
        # without copy requested: if they share memory (zero-copy) the result is not writable; if not,
        # it is writable.
        data._readonly = True

        # when we ask for a copy, the result should never be readonly
        arr = np.array(data)
        assert arr.flags.writeable

        # when we don't ask for a copy -> if the conversion is zero-copy,
        # the result should be readonly
        arr1 = np.asarray(data)
        arr2 = np.asarray(data)
        if np.shares_memory(arr1, arr2):
            assert not arr1.flags.writeable
        else:
            assert arr1.flags.writeable

    def test_readonly_propagates_to_numpy_array_method(self, data):
        # [FUNCTION DOC]
        # With the array flagged read-only: the to-numpy method with copy requested yields a writable
        # array; with copy disabled the result is not writable when two calls share memory, else writable.
        # Then, if the data has missing values, converting with copy disabled and a non-missing fill
        # value always copies and yields a writable array, except for float pyarrow-backed dtypes where
        # the post-fill conversion is zero-copy and therefore not writable.
        data._readonly = True

        # when we ask for a copy, the result should never be readonly
        arr = data.to_numpy(copy=True)
        assert arr.flags.writeable

        # when we don't ask for a copy -> if the conversion is zero-copy,
        # the result should be readonly
        arr1 = data.to_numpy(copy=False)
        arr2 = data.to_numpy(copy=False)
        if np.shares_memory(arr1, arr2):
            assert not arr1.flags.writeable
        else:
            assert arr1.flags.writeable

        # non-NA fill value should always result in a copy
        if data.isna().any():
            arr = data.to_numpy(copy=False, na_value=data[0])
            if isinstance(data.dtype, pd.ArrowDtype) and data.dtype.kind == "f":
                # for float dtype, after the fillna, the conversion from pyarrow to
                # numpy is zero-copy, and pyarrow will mark the array as readonly
                assert not arr.flags.writeable
            else:
                assert arr.flags.writeable
