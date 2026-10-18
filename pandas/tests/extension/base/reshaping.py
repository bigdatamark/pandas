# [MODULE DOC]
# Base tests for reshaping and combining operations with extension arrays: concat, align,
# setting columns, merge, stack/unstack, ravel and transpose. Public contents:
# BaseReshapingTests mixin. Uses the data, data_missing and na_value fixtures.
import itertools

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import NumpyEADtype

import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock


class BaseReshapingTests:
    # [CLASS DOC]
    # Mixin with reshaping and concatenation tests.
    """Tests for reshaping and concatenation."""

    @pytest.mark.parametrize("in_frame", [True, False])
    def test_concat(self, data, in_frame):
        # [FUNCTION DOC]
        # Parametrized over frame vs series. Concatenating a wrapped array with itself (index
        # ignored) has twice the length, keeps the extension dtype and (for block-based managers) stores
        # an extension-backed block holding an extension array.
        wrapped = pd.Series(data)
        if in_frame:
            wrapped = pd.DataFrame(wrapped)
        result = pd.concat([wrapped, wrapped], ignore_index=True)

        assert len(result) == len(data) * 2

        if in_frame:
            dtype = result.dtypes[0]
        else:
            dtype = result.dtype

        assert dtype == data.dtype
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        assert isinstance(result._mgr.blocks[0].values, ExtensionArray)

    @pytest.mark.parametrize("in_frame", [True, False])
    def test_concat_all_na_block(self, data_missing, in_frame):
        # [FUNCTION DOC]
        # Parametrized over frame vs series. Concatenates a block of valid values with a block of
        # all-missing values (distinct indexes) and expects the extension dtype to be preserved with
        # valid, valid, missing, missing.
        valid_block = pd.Series(data_missing.take([1, 1]), index=range(2))
        na_block = pd.Series(data_missing.take([0, 0]), index=range(2, 4))
        if in_frame:
            valid_block = pd.DataFrame({"a": valid_block})
            na_block = pd.DataFrame({"a": na_block})
        result = pd.concat([valid_block, na_block])
        if in_frame:
            expected = pd.DataFrame({"a": data_missing.take([1, 1, 0, 0])})
            tm.assert_frame_equal(result, expected)
        else:
            expected = pd.Series(data_missing.take([1, 1, 0, 0]))
            tm.assert_series_equal(result, expected)

    def test_concat_mixed_dtypes(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Concatenates frames (and then series) holding the
        # extension array, an integer column and a categorical column; the result must equal the
        # concatenation after casting everything to object. Also checks the simple case of the
        # extension frame with an object-cast integer frame.
        # https://github.com/pandas-dev/pandas/issues/20762
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ["a", "b", "c"]}).astype("category")
        dfs = [df1, df2, df3]

        # dataframes
        result = pd.concat(dfs)
        expected = pd.concat([x.astype(object) for x in dfs])
        tm.assert_frame_equal(result, expected)

        # series
        result = pd.concat([x["A"] for x in dfs])
        expected = pd.concat([x["A"].astype(object) for x in dfs])
        tm.assert_series_equal(result, expected)

        # simple test for just EA and one other
        result = pd.concat([df1, df2.astype(object)])
        expected = pd.concat([df1.astype("object"), df2.astype("object")])
        tm.assert_frame_equal(result, expected)

        result = pd.concat([df1["A"], df2["A"].astype(object)])
        expected = pd.concat([df1["A"].astype("object"), df2["A"].astype("object")])
        tm.assert_series_equal(result, expected)

    def test_concat_columns(self, data, na_value):
        # [FUNCTION DOC]
        # Column-wise concatenation of frames (and series) with an extension column and an integer
        # column equals the combined frame. With a non-aligned index (second starts at 1) the result has
        # a missing value appended to the extension column and NaN at the front of the integer column.
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"B": [1, 2, 3]})

        expected = pd.DataFrame({"A": data[:3], "B": [1, 2, 3]})
        result = pd.concat([df1, df2], axis=1)
        tm.assert_frame_equal(result, expected)
        result = pd.concat([df1["A"], df2["B"]], axis=1)
        tm.assert_frame_equal(result, expected)

        # non-aligned
        df2 = pd.DataFrame({"B": [1, 2, 3]}, index=[1, 2, 3])
        expected = pd.DataFrame(
            {
                "A": data._from_sequence([*list(data[:3]), na_value], dtype=data.dtype),
                "B": [np.nan, 1, 2, 3],
            }
        )

        result = pd.concat([df1, df2], axis=1)
        tm.assert_frame_equal(result, expected)
        result = pd.concat([df1["A"], df2["B"]], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_extension_arrays_copy_false(self, data, na_value):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Column-wise concatenation of an extension column of three
        # elements and one of four starting at index zero: the shorter is padded with a missing value;
        # no copying behavior is asserted here despite the name.
        # GH 20756
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"B": data[3:7]})
        expected = pd.DataFrame(
            {
                "A": data._from_sequence([*list(data[:3]), na_value], dtype=data.dtype),
                "B": data[3:7],
            }
        )
        result = pd.concat([df1, df2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_with_reindex(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Row-wise concatenation of two frames with different
        # columns (each holding five elements) produces extension columns filled with missing values
        # for the absent half, via take with fill.
        # GH-33027
        a = pd.DataFrame({"a": data[:5]})
        b = pd.DataFrame({"b": data[:5]})
        result = pd.concat([a, b], ignore_index=True)
        expected = pd.DataFrame(
            {
                "a": data.take(list(range(5)) + ([-1] * 5), allow_fill=True),
                "b": data.take(([-1] * 5) + list(range(5)), allow_fill=True),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_align(self, data, na_value):
        # [FUNCTION DOC]
        # Aligning series of elements 0-2 (default index) and 2-4 (index 1-3) yields length-four series
        # with a missing value at the end of the first and the beginning of the second. Assumes the
        # sequence constructor takes a list of scalars.
        a = data[:3]
        b = data[2:5]
        r1, r2 = pd.Series(a).align(pd.Series(b, index=[1, 2, 3]))

        # Assumes that the ctor can take a list of scalars of the type
        e1 = pd.Series(data._from_sequence([*list(a), na_value], dtype=data.dtype))
        e2 = pd.Series(data._from_sequence([na_value, *list(b)], dtype=data.dtype))
        tm.assert_series_equal(r1, e1)
        tm.assert_series_equal(r2, e2)

    def test_align_frame(self, data, na_value):
        # [FUNCTION DOC]
        # Frame version of the alignment test, with one extension column each.
        a = data[:3]
        b = data[2:5]
        r1, r2 = pd.DataFrame({"A": a}).align(pd.DataFrame({"A": b}, index=[1, 2, 3]))

        # Assumes that the ctor can take a list of scalars of the type
        e1 = pd.DataFrame(
            {"A": data._from_sequence([*list(a), na_value], dtype=data.dtype)}
        )
        e2 = pd.DataFrame(
            {"A": data._from_sequence([na_value, *list(b)], dtype=data.dtype)}
        )
        tm.assert_frame_equal(r1, e1)
        tm.assert_frame_equal(r2, e2)

    def test_align_series_frame(self, data, na_value):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Aligning a named series with a frame that has one more row
        # extends the series with a missing value (name preserved) and leaves the frame unchanged.
        # https://github.com/pandas-dev/pandas/issues/20576
        ser = pd.Series(data, name="a")
        df = pd.DataFrame({"col": np.arange(len(ser) + 1)})
        r1, r2 = ser.align(df)

        e1 = pd.Series(
            data._from_sequence([*list(data), na_value], dtype=data.dtype),
            name=ser.name,
        )

        tm.assert_series_equal(r1, e1)
        tm.assert_frame_equal(r2, df)

    def test_set_frame_expand_regular_with_extension(self, data):
        # [FUNCTION DOC]
        # Adding an extension-array column to a frame that has an integer column yields the expected
        # two-column frame.
        df = pd.DataFrame({"A": [1] * len(data)})
        df["B"] = data
        expected = pd.DataFrame({"A": [1] * len(data), "B": data})
        tm.assert_frame_equal(df, expected)

    def test_set_frame_expand_extension_with_regular(self, data):
        # [FUNCTION DOC]
        # Adding an integer list column to a frame with an extension column yields the expected frame.
        df = pd.DataFrame({"A": data})
        df["B"] = [1] * len(data)
        expected = pd.DataFrame({"A": data, "B": [1] * len(data)})
        tm.assert_frame_equal(df, expected)

    def test_set_frame_overwrite_object(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Overwriting an object-dtype column with the extension array
        # makes the column dtype the extension dtype.
        # https://github.com/pandas-dev/pandas/issues/20555
        df = pd.DataFrame({"A": [1] * len(data)}, dtype=object)
        df["A"] = data
        assert df.dtypes["A"] == data.dtype

    def test_merge(self, data, na_value):
        # [FUNCTION DOC]
        # Regression for an upstream issue. An inner merge on the key of a frame with the extension
        # column and a frame with repeated keys expands the extension values accordingly (0,0,1); an
        # outer merge adds the unmatched keys with a missing extension value for the right-only key
        # and NaN in the integer columns. Column order is extension, int1, key, int2.
        # GH-20743
        df1 = pd.DataFrame({"ext": data[:3], "int1": [1, 2, 3], "key": [0, 1, 2]})
        df2 = pd.DataFrame({"int2": [1, 2, 3, 4], "key": [0, 0, 1, 3]})

        res = pd.merge(df1, df2)
        exp = pd.DataFrame(
            {
                "int1": [1, 1, 2],
                "int2": [1, 2, 3],
                "key": [0, 0, 1],
                "ext": data._from_sequence(
                    [data[0], data[0], data[1]], dtype=data.dtype
                ),
            }
        )
        tm.assert_frame_equal(res, exp[["ext", "int1", "key", "int2"]])

        res = pd.merge(df1, df2, how="outer")
        exp = pd.DataFrame(
            {
                "int1": [1, 1, 2, 3, np.nan],
                "int2": [1, 2, 3, np.nan, 4],
                "key": [0, 0, 1, 2, 3],
                "ext": data._from_sequence(
                    [data[0], data[0], data[1], data[2], na_value], dtype=data.dtype
                ),
            }
        )
        tm.assert_frame_equal(res, exp[["ext", "int1", "key", "int2"]])

    def test_merge_on_extension_array(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Self-merging a frame on an extension key column gives the
        # expected frame with suffixed value columns; merging a row-permuted frame with the original
        # keeps the left order.
        # GH 23020
        a, b = data[:2]
        key = type(data)._from_sequence([a, b], dtype=data.dtype)

        df = pd.DataFrame({"key": key, "val": [1, 2]})
        result = pd.merge(df, df, on="key")
        expected = pd.DataFrame({"key": key, "val_x": [1, 2], "val_y": [1, 2]})
        tm.assert_frame_equal(result, expected)

        # order
        result = pd.merge(df.iloc[[1, 0]], df, on="key")
        expected = expected.iloc[[1, 0]].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_on_extension_array_duplicates(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Merging two frames on a key column with a duplicated key
        # (a, b, a) yields all pairings: keys a,a,b,a,a with the expected left and right values.
        # GH 23020
        a, b = data[:2]
        key = type(data)._from_sequence([a, b, a], dtype=data.dtype)
        df1 = pd.DataFrame({"key": key, "val": [1, 2, 3]})
        df2 = pd.DataFrame({"key": key, "val": [1, 2, 3]})

        result = pd.merge(df1, df2, on="key")
        expected = pd.DataFrame(
            {
                "key": key.take([0, 0, 1, 2, 2]),
                "val_x": [1, 1, 2, 3, 3],
                "val_y": [1, 3, 2, 1, 3],
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize(
        "columns",
        [
            ["A", "B"],
            pd.MultiIndex.from_tuples(
                [("A", "a"), ("A", "b")], names=["outer", "inner"]
            ),
        ],
    )
    @pytest.mark.parametrize("future_stack", [True, False])
    def test_stack(self, data, columns, future_stack):
        # [FUNCTION DOC]
        # Parametrized over flat and two-level column labels and over the new and legacy stack
        # implementation (a deprecation warning is ignored). Stacks a two-column frame of the same
        # extension data and compares to the stack of the frame cast to object, which is then cast to
        # object again (some constructors infer specialized types from object). The result dtype must
        # equal the column dtype (for the new implementation with a numpy-wrapping dtype, the plain
        # numpy dtype instead), or each result column's dtype for frames. Final comparison is done after
        # casting the result to object.
        df = pd.DataFrame({"A": data[:5], "B": data[:5]})
        df.columns = columns
        result = df.stack(future_stack=future_stack)
        expected = df.astype(object).stack(future_stack=future_stack)
        # we need a second astype(object), in case the constructor inferred
        # object -> specialized, as is done for period.
        expected = expected.astype(object)

        if isinstance(expected, pd.Series):
            if future_stack and isinstance(data.dtype, NumpyEADtype):
                # GH#58817 future_stack=True constructs the result specifying the dtype
                # using the dtype of the input; we thus get the underlying
                # NumPy dtype as the result instead of the NumpyExtensionArray
                assert result.dtype == df.iloc[:, 0].to_numpy().dtype
            else:
                assert result.dtype == df.iloc[:, 0].dtype
        else:
            assert all(result.dtypes == df.iloc[:, 0].dtype)

        result = result.astype(object)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            # Two levels, uniform.
            pd.MultiIndex.from_product(([["A", "B"], ["a", "b"]]), names=["a", "b"]),
            # non-uniform
            pd.MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "b")]),
            # three levels, non-uniform
            pd.MultiIndex.from_product([("A", "B"), ("a", "b", "c"), (0, 1, 2)]),
            pd.MultiIndex.from_tuples(
                [
                    ("A", "a", 1),
                    ("A", "b", 0),
                    ("A", "a", 0),
                    ("B", "a", 0),
                    ("B", "c", 1),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize("obj", ["series", "frame"])
    def test_unstack(self, data, index, obj):
        # [FUNCTION DOC]
        # Parametrized over four multi-level indexes (uniform two-level, non-uniform two-level, and
        # two three-level ones) and over series vs frame. Truncates index and data to the shorter
        # length, then unstacks every ordered selection of fewer-than-all index levels. Each result
        # column must hold the array class. For series, the result must equal unstacking the series
        # converted to a frame and dropping the top column level. The result cast to object must equal
        # the unstack of the object-cast data using the dtype's missing value as fill value; for series
        # the expected frame must have only object columns.
        final_length = min(len(index), len(data))
        index = index[:final_length]
        data = data[:final_length]
        if obj == "series":
            ser = pd.Series(data, index=index)
        else:
            ser = pd.DataFrame({"A": data, "B": data}, index=index)

        n = index.nlevels
        levels = list(range(n))
        # [0, 1, 2]
        # [(0,), (1,), (2,), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        combinations = itertools.chain.from_iterable(
            itertools.permutations(levels, i) for i in range(1, n)
        )

        for level in combinations:
            result = ser.unstack(level=level)
            assert all(
                isinstance(result[col].array, type(data)) for col in result.columns
            )

            if obj == "series":
                # We should get the same result with to_frame+unstack+droplevel
                df = ser.to_frame()

                alt = df.unstack(level=level).droplevel(0, axis=1)
                tm.assert_frame_equal(result, alt)

            obj_ser = ser.astype(object)

            expected = obj_ser.unstack(level=level, fill_value=data.dtype.na_value)
            if obj == "series":
                assert (expected.dtypes == object).all()

            result = result.astype(object)
            tm.assert_frame_equal(result, expected)

    def test_ravel(self, data):
        # [FUNCTION DOC]
        # Ravel on a one-dimensional array is a no-op: returns the same class and (unless the dtype is
        # immutable, then the test is skipped) a view, so mutating the result changes the original. Mutates
        # the data fixture.
        # as long as EA is 1D-only, ravel is a no-op
        result = data.ravel()
        assert type(result) == type(data)

        if data.dtype._is_immutable:
            pytest.skip(f"test_ravel assumes mutability and {data.dtype} is immutable")

        # Check that we have a view, not a copy
        result[0] = result[1]
        assert data[0] == data[1]

    def test_transpose(self, data):
        # [FUNCTION DOC]
        # Array transpose returns the same class, a new object with the reversed shape, and (unless
        # immutable, then the test is skipped) a view: mutating the result changes the original.
        result = data.transpose()
        assert type(result) == type(data)

        # check we get a new object
        assert result is not data

        # If we ever _did_ support 2D, shape should be reversed
        assert result.shape == data.shape[::-1]

        if data.dtype._is_immutable:
            pytest.skip(
                f"test_transpose assumes mutability and {data.dtype} is immutable"
            )

        # Check that we have a view, not a copy
        result[0] = result[1]
        assert data[0] == data[1]

    def test_transpose_frame(self, data):
        # [FUNCTION DOC]
        # Transposing a two-column frame of the same data with a four-row string index gives four
        # columns each holding two copies of the corresponding element and a two-label index. A double
        # numpy transpose returns the original frame, also for a single column.
        df = pd.DataFrame({"A": data[:4], "B": data[:4]}, index=["a", "b", "c", "d"])
        result = df.T
        expected = pd.DataFrame(
            {
                "a": type(data)._from_sequence([data[0]] * 2, dtype=data.dtype),
                "b": type(data)._from_sequence([data[1]] * 2, dtype=data.dtype),
                "c": type(data)._from_sequence([data[2]] * 2, dtype=data.dtype),
                "d": type(data)._from_sequence([data[3]] * 2, dtype=data.dtype),
            },
            index=["A", "B"],
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(np.transpose(np.transpose(df)), df)
        tm.assert_frame_equal(np.transpose(np.transpose(df[["A"]])), df[["A"]])
