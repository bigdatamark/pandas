# [MODULE DOC]
# Base tests for item access (getitem, take, reindex, loc/iloc, item) on extension arrays and
# on series and frames holding them. Public contents: BaseGetitemTests mixin. Uses the data,
# data_missing, na_value and na_cmp fixtures. In most tests the expected result is a slice or
# take of the fixture data.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class BaseGetitemTests:
    # [CLASS DOC]
    # Mixin with tests of positional and label indexing and related selection methods for an
    # extension array.
    """Tests for ExtensionArray.__getitem__."""

    def test_iloc_series(self, data):
        # [FUNCTION DOC]
        # Positional selection on a series of the data: a leading slice of four and a list of the
        # first four positions both give a series equal to the first four elements.
        ser = pd.Series(data)
        result = ser.iloc[:4]
        expected = pd.Series(data[:4])
        tm.assert_series_equal(result, expected)

        result = ser.iloc[[0, 1, 2, 3]]
        tm.assert_series_equal(result, expected)

    def test_iloc_frame(self, data):
        # [FUNCTION DOC]
        # Two-column frame (extension data plus an integer range). Checks positional frame
        # selection by slice and by list of rows on the extension column returns a one-column frame of
        # the first four elements; selecting the column by scalar position returns a named series; and
        # column slicing with a step (including after reordering columns) keeps the right columns.
        # KNOWN ISSUE: the part commented as selecting by sequence into a series repeats the slice-based
        # selection, so the sequence form is not actually exercised.
        df = pd.DataFrame({"A": data, "B": np.arange(len(data), dtype="int64")})
        expected = pd.DataFrame({"A": data[:4]})

        # slice -> frame
        result = df.iloc[:4, [0]]
        tm.assert_frame_equal(result, expected)

        # sequence -> frame
        result = df.iloc[[0, 1, 2, 3], [0]]
        tm.assert_frame_equal(result, expected)

        expected = pd.Series(data[:4], name="A")

        # slice -> series
        result = df.iloc[:4, 0]
        tm.assert_series_equal(result, expected)

        # sequence -> series
        result = df.iloc[:4, 0]
        tm.assert_series_equal(result, expected)

        # GH#32959 slice columns with step
        result = df.iloc[:, ::2]
        tm.assert_frame_equal(result, df[["A"]])
        result = df[["B", "A"]].iloc[:, ::2]
        tm.assert_frame_equal(result, df[["B"]])

    def test_iloc_frame_single_block(self, data):
        # [FUNCTION DOC]
        # Single-extension-column frame: null slices, column slices that cover the column (leading
        # one or two, step two, last one), all return an equal frame; a slice past the column
        # (from position one to two) returns an empty-column frame.
        # GH#32959 null slice along index, slice along columns with single-block
        df = pd.DataFrame({"A": data})

        result = df.iloc[:, :]
        tm.assert_frame_equal(result, df)

        result = df.iloc[:, :1]
        tm.assert_frame_equal(result, df)

        result = df.iloc[:, :2]
        tm.assert_frame_equal(result, df)

        result = df.iloc[:, ::2]
        tm.assert_frame_equal(result, df)

        result = df.iloc[:, 1:2]
        tm.assert_frame_equal(result, df.iloc[:, :0])

        result = df.iloc[:, -1:]
        tm.assert_frame_equal(result, df)

    def test_loc_series(self, data):
        # [FUNCTION DOC]
        # Label-based selection on a series: an inclusive slice up to label three and a list of the
        # first four labels both give the first four elements.
        ser = pd.Series(data)
        result = ser.loc[:3]
        expected = pd.Series(data[:4])
        tm.assert_series_equal(result, expected)

        result = ser.loc[[0, 1, 2, 3]]
        tm.assert_series_equal(result, expected)

    def test_loc_frame(self, data):
        # [FUNCTION DOC]
        # Label-based counterpart of the frame iloc test: slice and list rows with a column list give
        # a one-column frame; a single column label gives a named series.
        # KNOWN ISSUE: the part commented as selecting by sequence into a series repeats the slice-based
        # selection, so the sequence form is not actually exercised.
        df = pd.DataFrame({"A": data, "B": np.arange(len(data), dtype="int64")})
        expected = pd.DataFrame({"A": data[:4]})

        # slice -> frame
        result = df.loc[:3, ["A"]]
        tm.assert_frame_equal(result, expected)

        # sequence -> frame
        result = df.loc[[0, 1, 2, 3], ["A"]]
        tm.assert_frame_equal(result, expected)

        expected = pd.Series(data[:4], name="A")

        # slice -> series
        result = df.loc[:3, "A"]
        tm.assert_series_equal(result, expected)

        # sequence -> series
        result = df.loc[:3, "A"]
        tm.assert_series_equal(result, expected)

    def test_loc_iloc_frame_single_dtype(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream bug where block-level access returned a scalar. Single-column
        # frame: label row two and positional row minus one must give a series (index is the column
        # name, name is the row label) with the extension dtype.
        # GH#27110 bug in ExtensionBlock.iget caused df.iloc[n] to incorrectly
        #  return a scalar
        df = pd.DataFrame({"A": data})
        expected = pd.Series([data[2]], index=["A"], name=2, dtype=data.dtype)

        result = df.loc[2]
        tm.assert_series_equal(result, expected)

        expected = pd.Series(
            [data[-1]], index=["A"], name=len(data) - 1, dtype=data.dtype
        )
        result = df.iloc[-1]
        tm.assert_series_equal(result, expected)

    def test_getitem_scalar(self, data):
        # [FUNCTION DOC]
        # Integer indexing the array and the series returns an instance of the dtype's scalar type.
        result = data[0]
        assert isinstance(result, data.dtype.type)

        result = pd.Series(data)[0]
        assert isinstance(result, data.dtype.type)

    def test_getitem_invalid(self, data):
        # [FUNCTION DOC]
        # Indexing the array with a string or a float must raise an index error with numpy's
        # message about valid index kinds. Indexing one past the end or one before the start must raise
        # an index error; the accepted message is an alternation covering the messages of the
        # various array implementations (python list, pyarrow, sparse, numpy).
        # TODO: box over scalar, [scalar], (scalar,)?

        msg = (
            r"only integers, slices \(`:`\), ellipsis \(`...`\), numpy.newaxis "
            r"\(`None`\) and integer or boolean arrays are valid indices"
        )
        with pytest.raises(IndexError, match=msg):
            data["foo"]
        with pytest.raises(IndexError, match=msg):
            data[2.5]

        ub = len(data)
        msg = "|".join(
            [
                "list index out of range",  # json
                "index out of bounds",  # pyarrow
                "Out of bounds access",  # Sparse
                f"loc must be an integer between -{ub} and {ub}",  # Sparse
                f"index {ub + 1} is out of bounds for axis 0 with size {ub}",
                f"index -{ub + 1} is out of bounds for axis 0 with size {ub}",
            ]
        )
        with pytest.raises(IndexError, match=msg):
            data[ub + 1]
        with pytest.raises(IndexError, match=msg):
            data[-ub - 1]

    def test_getitem_scalar_na(self, data_missing, na_cmp, na_value):
        # [FUNCTION DOC]
        # The first element of the data_missing fixture must be the missing value according to the
        # na_cmp comparator and na_value.
        result = data_missing[0]
        assert na_cmp(result, na_value)

    def test_getitem_empty(self, data):
        # [FUNCTION DOC]
        # Indexing with an empty list returns an empty array of the same class, equal to indexing
        # with an empty integer numpy array.
        # Indexing with empty list
        result = data[[]]
        assert len(result) == 0
        assert isinstance(result, type(data))

        expected = data[np.array([], dtype="int64")]
        tm.assert_extension_array_equal(result, expected)

    def test_getitem_mask(self, data):
        # [FUNCTION DOC]
        # Boolean numpy mask: an all-false mask gives an empty result of the same class (array) or
        # dtype (series); after setting the first entry true the result has one element, again for both
        # array and series.
        # Empty mask, raw array
        mask = np.zeros(len(data), dtype=bool)
        result = data[mask]
        assert len(result) == 0
        assert isinstance(result, type(data))

        # Empty mask, in series
        mask = np.zeros(len(data), dtype=bool)
        result = pd.Series(data)[mask]
        assert len(result) == 0
        assert result.dtype == data.dtype

        # non-empty mask, raw array
        mask[0] = True
        result = data[mask]
        assert len(result) == 1
        assert isinstance(result, type(data))

        # non-empty mask, in series
        result = pd.Series(data)[mask]
        assert len(result) == 1
        assert result.dtype == data.dtype

    def test_getitem_mask_raises(self, data):
        # [FUNCTION DOC]
        # A boolean mask of the wrong length (two) must raise an index error stating the expected
        # and actual lengths, for both a numpy mask and a nullable boolean array mask.
        mask = np.array([True, False])
        msg = f"Boolean index has wrong length: 2 instead of {len(data)}"
        with pytest.raises(IndexError, match=msg):
            data[mask]

        mask = pd.array(mask, dtype="boolean")
        with pytest.raises(IndexError, match=msg):
            data[mask]

    def test_getitem_boolean_array_mask(self, data):
        # [FUNCTION DOC]
        # Nullable boolean array mask: all false gives empty results (array and series). With the
        # first five set true the result equals taking positions zero to four, for both array and
        # series.
        mask = pd.array(np.zeros(data.shape, dtype="bool"), dtype="boolean")
        result = data[mask]
        assert len(result) == 0
        assert isinstance(result, type(data))

        result = pd.Series(data)[mask]
        assert len(result) == 0
        assert result.dtype == data.dtype

        mask[:5] = True
        expected = data.take([0, 1, 2, 3, 4])
        result = data[mask]
        tm.assert_extension_array_equal(result, expected)

        expected = pd.Series(expected)
        result = pd.Series(data)[mask]
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean_na_treated_as_false(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. A nullable boolean mask with missing values in the first
        # two entries and true in the next two selects the same as the mask with missing values filled
        # with false, for both array and series.
        # https://github.com/pandas-dev/pandas/issues/31503
        mask = pd.array(np.zeros(data.shape, dtype="bool"), dtype="boolean")
        mask[:2] = pd.NA
        mask[2:4] = True

        result = data[mask]
        expected = data[mask.fillna(False)]

        tm.assert_extension_array_equal(result, expected)

        s = pd.Series(data)

        result = s[mask]
        expected = s[mask.fillna(False)]

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_getitem_integer_array(self, data, idx):
        # [FUNCTION DOC]
        # Parametrized over a list, a nullable integer array and a numpy array of positions zero to
        # two. Indexing returns an array of the same class with three elements equal to the take of
        # those positions; the series version is equal too.
        result = data[idx]
        assert len(result) == 3
        assert isinstance(result, type(data))
        expected = data.take([0, 1, 2])
        tm.assert_extension_array_equal(result, expected)

        expected = pd.Series(expected)
        result = pd.Series(data)[idx]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2, pd.NA], pd.array([0, 1, 2, pd.NA], dtype="Int64")],
        ids=["list", "integer-array"],
    )
    def test_getitem_integer_with_missing_raises(self, data, idx):
        # [FUNCTION DOC]
        # Parametrized over a list and a nullable integer array containing a missing value. Indexing
        # the array must raise a value error saying integer indexers cannot contain missing values.
        msg = "Cannot index with an integer indexer containing NA values"
        with pytest.raises(ValueError, match=msg):
            data[idx]

    @pytest.mark.xfail(
        reason="Tries label-based and raises KeyError; "
        "in some cases raises when calling np.asarray"
    )
    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2, pd.NA], pd.array([0, 1, 2, pd.NA], dtype="Int64")],
        ids=["list", "integer-array"],
    )
    def test_getitem_series_integer_with_missing_raises(self, data, idx):
        # [FUNCTION DOC]
        # Expected-failure test (marked xfail with no strict argument, reason: it tries label based
        # lookup and raises a key error, or fails in array conversion). Intended behavior: indexing a
        # series that has string index labels with an integer indexer containing a missing value raises
        # a value error with the missing-values message.
        msg = "Cannot index with an integer indexer containing NA values"
        # TODO: this raises KeyError about labels not found (it tries label-based)

        ser = pd.Series(data, index=[chr(100 + i) for i in range(len(data))])
        with pytest.raises(ValueError, match=msg):
            ser[idx]

    def test_getitem_slice(self, data):
        # [FUNCTION DOC]
        # Slicing the array with an empty slice and a one-element slice both return the array
        # class (a one-element slice is still an array, not a scalar).
        # getitem[slice] should return an array
        result = data[slice(0)]  # empty
        assert isinstance(result, type(data))

        result = data[slice(1)]  # scalar
        assert isinstance(result, type(data))

    def test_getitem_ellipsis_and_slice(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue (called when slicing block rows). Indexing with an
        # ellipsis combined with a slice, in either order, for a full slice, a leading slice and a stepped
        # slice, equals plain slicing.
        # GH#40353 this is called from slice_block_rows
        result = data[..., :]
        tm.assert_extension_array_equal(result, data)

        result = data[:, ...]
        tm.assert_extension_array_equal(result, data)

        result = data[..., :3]
        tm.assert_extension_array_equal(result, data[:3])

        result = data[:3, ...]
        tm.assert_extension_array_equal(result, data[:3])

        result = data[..., ::2]
        tm.assert_extension_array_equal(result, data[::2])

        result = data[::2, ...]
        tm.assert_extension_array_equal(result, data[::2])

    def test_get(self, data):
        # [FUNCTION DOC]
        # Regression tests for the series get method with an extension array. With an even-integer
        # index: scalar get by existing label equals positional element; list get and slice get return
        # series equal to the corresponding positions; labels missing (including negative and past the
        # end) give none. With a string index: get by label, by label slice, missing label behave
        # similarly; and in this version integer keys on a string index are treated as labels and give
        # none. Finally a step slice of the series must not emit a spurious future warning and getting
        # a missing label from it gives none.
        # GH 20882
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        assert s.get(4) == s.iloc[2]

        result = s.get([4, 6])
        expected = s.iloc[[2, 3]]
        tm.assert_series_equal(result, expected)

        result = s.get(slice(2))
        expected = s.iloc[[0, 1]]
        tm.assert_series_equal(result, expected)

        assert s.get(-1) is None
        assert s.get(s.index.max() + 1) is None

        s = pd.Series(data[:6], index=list("abcdef"))
        assert s.get("c") == s.iloc[2]

        result = s.get(slice("b", "d"))
        expected = s.iloc[[1, 2, 3]]
        tm.assert_series_equal(result, expected)

        result = s.get("Z")
        assert result is None

        # As of 3.0, getitem with int keys treats them as labels
        assert s.get(4) is None
        assert s.get(-1) is None
        assert s.get(len(s)) is None

        # GH 21257
        s = pd.Series(data)
        with tm.assert_produces_warning(None):
            # GH#45324 make sure we aren't giving a spurious FutureWarning
            s2 = s[::2]
        assert s2.get(1) is None

    def test_take_sequence(self, data):
        # [FUNCTION DOC]
        # Selecting positions zero, one and three from a series by list returns elements equal to
        # the corresponding array elements, in that order.
        result = pd.Series(data)[[0, 1, 3]]
        assert result.iloc[0] == data[0]
        assert result.iloc[1] == data[1]
        assert result.iloc[2] == data[3]

    def test_take(self, data, na_value, na_cmp):
        # [FUNCTION DOC]
        # Positional take. Without fill, taking the first and last positions keeps the dtype
        # and returns the matching elements. With fill allowed and the missing fill value, the
        # first result is the first element and taking -1 yields the missing value (compared
        # with the fixture comparator). Taking a position past the end raises an index error
        # whose message mentions out of bounds.
        result = data.take([0, -1])
        assert result.dtype == data.dtype
        assert result[0] == data[0]
        assert result[1] == data[-1]

        result = data.take([0, -1], allow_fill=True, fill_value=na_value)
        assert result[0] == data[0]
        assert na_cmp(result[1], na_value)

        with pytest.raises(IndexError, match="out of bounds"):
            data.take([len(data) + 1])

    def test_take_empty(self, data, na_value, na_cmp):
        # [FUNCTION DOC]
        # Taking -1 from an empty array with fill allowed returns the missing value. Without fill,
        # taking -1 raises an index error (message about non-empty take from empty axes, or out of
        # bounds); taking positions zero and one raises an index error saying a non-empty take cannot
        # be done.
        empty = data[:0]

        result = empty.take([-1], allow_fill=True)
        assert na_cmp(result[0], na_value)

        msg = "cannot do a non-empty take from an empty axes|out of bounds"

        with pytest.raises(IndexError, match=msg):
            empty.take([-1])

        with pytest.raises(IndexError, match="cannot do a non-empty take"):
            empty.take([0, 1])

    def test_take_negative(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue: negative positions count from the end, so taking
        # position zero, minus length, last, and minus one equals taking zero, zero, last, last.
        # https://github.com/pandas-dev/pandas/issues/20640
        n = len(data)
        result = data.take([0, -n, n - 1, -1])
        expected = data.take([0, 0, n - 1, n - 1])
        tm.assert_extension_array_equal(result, expected)

    def test_take_non_na_fill_value(self, data_missing):
        # [FUNCTION DOC]
        # Using a valid element as the fill value: for an array of missing, valid, missing, taking -1
        # and 1 with fill allowed and that valid fill value equals taking the valid element twice.
        fill_value = data_missing[1]  # valid
        na = data_missing[0]

        arr = data_missing._from_sequence(
            [na, fill_value, na], dtype=data_missing.dtype
        )
        result = arr.take([-1, 1], fill_value=fill_value, allow_fill=True)
        expected = arr.take([1, 1])
        tm.assert_extension_array_equal(result, expected)

    def test_take_pandas_style_negative_raises(self, data, na_value):
        # [FUNCTION DOC]
        # With fill allowed, taking a position below -1 (here -2) must raise a value error that
        # comes from outside pandas (error type check only).
        with tm.external_error_raised(ValueError):
            data.take([0, -2], fill_value=na_value, allow_fill=True)

    @pytest.mark.parametrize("allow_fill", [True, False])
    def test_take_out_of_bounds_raises(self, data, allow_fill):
        # [FUNCTION DOC]
        # Parametrized over allow_fill. Taking positions zero and three from a three-element array
        # raises an index error mentioning out of bounds in both modes.
        arr = data[:3]

        with pytest.raises(IndexError, match="out of bounds|out-of-bounds"):
            arr.take(np.asarray([0, 3]), allow_fill=allow_fill)

    def test_take_series(self, data):
        # [FUNCTION DOC]
        # Series take of first and last positions equals a series built from those elements with the
        # index taken at the same positions.
        s = pd.Series(data)
        result = s.take([0, -1])
        expected = pd.Series(
            data._from_sequence([data[0], data[len(data) - 1]], dtype=s.dtype),
            index=s.index.take([0, -1]),
        )
        tm.assert_series_equal(result, expected)

    def test_reindex(self, data, na_value):
        # [FUNCTION DOC]
        # Series reindex: existing labels give the matching elements; a mix of missing labels (negative,
        # beyond the end) gives missing values at those places; labels all outside gives all missing
        # with a range index. Missing values are the fixture na_value.
        s = pd.Series(data)
        result = s.reindex([0, 1, 3])
        expected = pd.Series(data.take([0, 1, 3]), index=[0, 1, 3])
        tm.assert_series_equal(result, expected)

        n = len(data)
        result = s.reindex([-1, 0, n])
        expected = pd.Series(
            data._from_sequence([na_value, data[0], na_value], dtype=s.dtype),
            index=[-1, 0, n],
        )
        tm.assert_series_equal(result, expected)

        result = s.reindex([n, n + 1])
        expected = pd.Series(
            data._from_sequence([na_value, na_value], dtype=s.dtype),
            index=range(n, n + 2, 1),
        )
        tm.assert_series_equal(result, expected)

    def test_reindex_non_na_fill_value(self, data_missing):
        # [FUNCTION DOC]
        # Series of a missing and a valid element reindexed to three labels with the valid element as
        # fill value gives missing, valid, valid.
        valid = data_missing[1]
        na = data_missing[0]

        arr = data_missing._from_sequence([na, valid], dtype=data_missing.dtype)
        ser = pd.Series(arr)
        result = ser.reindex([0, 1, 2], fill_value=valid)
        expected = pd.Series(
            data_missing._from_sequence([na, valid, valid], dtype=data_missing.dtype)
        )

        tm.assert_series_equal(result, expected)

    def test_loc_len1(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue (take with a length-one indexer yielded wrong ndim).
        # Selecting one label from a frame column returns a one-dimensional series whose block is
        # one-dimensional.
        # KNOWN ISSUE: the block dimensionality is accessed through the blocks list unconditionally just
        # before the guarded check on whether the manager has blocks, so the guard does not protect
        # that access.
        # see GH-27785 take_nd with indexer of len 1 resulting in wrong ndim
        df = pd.DataFrame({"A": data})
        res = df.loc[[0], "A"]
        assert res.ndim == 1
        assert res._mgr.blocks[0].ndim == 1
        if hasattr(res._mgr, "blocks"):
            assert res._mgr._block.ndim == 1

    def test_item(self, data):
        # [FUNCTION DOC]
        # Series item method: a one-element slice returns the first scalar; an empty slice and the full
        # series each raise a value error saying only a size-1 array can be converted to a scalar.
        # https://github.com/pandas-dev/pandas/pull/30175
        s = pd.Series(data)
        result = s[:1].item()
        assert result == data[0]

        msg = "can only convert an array of size 1 to a Python scalar"
        with pytest.raises(ValueError, match=msg):
            s[:0].item()

        with pytest.raises(ValueError, match=msg):
            s.item()

    def test_array_item(self, data):
        # [FUNCTION DOC]
        # Array item method with no argument: a one-element slice returns the scalar; a slice of two
        # and an empty slice raise a value error about converting size-1 arrays.
        # GH#63876, GH#64129
        arr = data[:1]
        assert arr.item() == data[0]

        msg = "can only convert an array of size 1 to a Python scalar"
        with pytest.raises(ValueError, match=msg):
            data[:2].item()
        with pytest.raises(ValueError, match=msg):
            data[:0].item()

    def test_array_item_with_index(self, data):
        # [FUNCTION DOC]
        # Array item method with an index argument: positions zero and minus one return the matching
        # scalars; an out of range position raises an index error (from outside pandas); a list index
        # raises a type error saying the index must be an integer.
        # GH#63876, GH#64129
        assert data.item(0) == data[0]
        assert data.item(-1) == data[-1]

        with tm.external_error_raised(IndexError):
            data.item(len(data))

        msg = "index must be an integer"
        with pytest.raises(TypeError, match=msg):
            data.item([0])

    def test_getitem_propagates_readonly_property(self, data):
        # [FUNCTION DOC]
        # Sets the private read-only flag on the array (mutating the fixture) and asserts that a full
        # slice, which may be a view, is also read-only.
        # ensure read-only propagates if getitem returns view
        data._readonly = True

        result = data[:]
        assert result._readonly
