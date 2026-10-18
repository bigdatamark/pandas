# [MODULE DOC]
# Tests for the dummy (one-hot) encoding function applied to lists, series, categoricals and frames.
# Covers dense and sparse output, requested dtypes (including extension and arrow dtypes), null handling, prefixes and separators, column subsets, drop-first, categorical preservation, unicode names, duplicate column names and argument validation.
# Contains one test class with its fixtures. Optionally imports pyarrow at import time (falls back to none when unavailable); no other import-time side effects.
import re
import unicodedata

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
from pandas import (
    ArrowDtype,
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    RangeIndex,
    Series,
    SparseDtype,
    get_dummies,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray

try:
    import pyarrow as pa
except ImportError:
    pa = None


class TestGetDummies:
    @pytest.fixture
    # [CLASS DOC]
    # Groups the dummy-encoding tests, parametrized through shared fixtures for the output dtype and for dense versus sparse results.
    def df(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a three-row frame with string columns A (a, b, a) and B (b, b, c) and an integer column C (one to three).
        return DataFrame({"A": ["a", "b", "a"], "B": ["b", "b", "c"], "C": [1, 2, 3]})

    @pytest.fixture(params=["uint8", "i8", np.float64, bool, None])
    def dtype(self, request):
        # [FUNCTION DOC]
        # HELPER: parametrized fixture yielding numpy dtypes unsigned 8-bit, signed 64-bit, 64-bit float, boolean and the default (none, which numpy converts to its default float type).
        # KNOWN ISSUE: the none parameter is converted by numpy into its default float dtype, so the fixture never yields none and the none branch of the effective-dtype helper is unreachable; that case actually runs as 64-bit float.
        return np.dtype(request.param)

    @pytest.fixture(params=["dense", "sparse"])
    def sparse(self, request):
        # params are strings to simplify reading test results,
        # e.g. TestGetDummies::test_basic[uint8-sparse] instead of [uint8-True]
        # [FUNCTION DOC]
        # HELPER: fixture yielding a boolean for sparse output; the parameter ids are the strings dense and sparse for readable test names.
        return request.param == "sparse"

    def effective_dtype(self, dtype):
        # [FUNCTION DOC]
        # HELPER: plain method (not a fixture) mapping a requested dtype to the one actually expected: returns unsigned 8-bit integer when the dtype is none, otherwise the dtype unchanged.
        if dtype is None:
            return np.uint8
        return dtype

    def test_get_dummies_raises_on_dtype_object(self, df):
        # [FUNCTION DOC]
        # Requesting the object dtype raises a value error saying it is not a valid dtype for dummy encoding.
        msg = "dtype=object is not a valid dtype for get_dummies"
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, dtype="object")

    def test_get_dummies_basic(self, sparse, dtype):
        # [FUNCTION DOC]
        # Encoding a list, a series and an indexed series of three distinct strings gives an identity-like frame of the requested dtype (sparse arrays with fill zero, or false for booleans); the series index is preserved.
        s_list = list("abc")
        s_series = Series(s_list)
        s_series_index = Series(s_list, list("ABC"))

        expected = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]},
            dtype=self.effective_dtype(dtype),
        )
        if sparse:
            if dtype.kind == "b":
                expected = expected.apply(SparseArray, fill_value=False)
            else:
                expected = expected.apply(SparseArray, fill_value=0.0)
        result = get_dummies(s_list, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        expected.index = list("ABC")
        result = get_dummies(s_series_index, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_types(self, sparse, dtype, using_infer_string):
        # GH 10531
        # [FUNCTION DOC]
        # GH 10531. Same inputs plus a frame with all columns encoded: the dtype counts of the result columns are eight of the expected (sparse or dense) dtype. With only one column encoded, remaining columns keep their integer and string (object, or str when string inference is on) dtypes and three new columns have the requested dtype.
        s_list = list("abc")
        s_series = Series(s_list)
        s_df = DataFrame(
            {"a": [0, 1, 0, 1, 2], "b": ["A", "A", "B", "C", "C"], "c": [2, 3, 3, 3, 2]}
        )

        expected = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]},
            dtype=self.effective_dtype(dtype),
            columns=list("abc"),
        )
        if sparse:
            if is_integer_dtype(dtype):
                fill_value = 0
            elif dtype == bool:
                fill_value = False
            else:
                fill_value = 0.0

            expected = expected.apply(SparseArray, fill_value=fill_value)
        result = get_dummies(s_list, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_df, columns=s_df.columns, sparse=sparse, dtype=dtype)
        if sparse:
            dtype_name = f"Sparse[{self.effective_dtype(dtype).name}, {fill_value}]"
        else:
            dtype_name = self.effective_dtype(dtype).name

        expected = Series({dtype_name: 8}, name="count")
        result = result.dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        tm.assert_series_equal(result, expected)

        result = get_dummies(s_df, columns=["a"], sparse=sparse, dtype=dtype)

        key = "str" if using_infer_string else "object"
        expected_counts = {"int64": 1, key: 1}
        expected_counts[dtype_name] = 3 + expected_counts.get(dtype_name, 0)

        expected = Series(expected_counts, name="count").sort_index()
        result = result.dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        result = result.sort_index()
        tm.assert_series_equal(result, expected)

    def test_get_dummies_just_na(self, sparse):
        # [FUNCTION DOC]
        # Encoding input with only a missing value (list, series, indexed series) gives an empty-column frame keeping the original index.
        just_na_list = [np.nan]
        just_na_series = Series(just_na_list)
        just_na_series_index = Series(just_na_list, index=["A"])

        res_list = get_dummies(just_na_list, sparse=sparse)
        res_series = get_dummies(just_na_series, sparse=sparse)
        res_series_index = get_dummies(just_na_series_index, sparse=sparse)

        assert res_list.empty
        assert res_series.empty
        assert res_series_index.empty

        assert res_list.index.tolist() == [0]
        assert res_series.index.tolist() == [0]
        assert res_series_index.index.tolist() == ["A"]

    def test_get_dummies_include_na(self, sparse, dtype):
        # [FUNCTION DOC]
        # Nulls are ignored by default (no column, all-zero row); with dummy_na enabled a null-labelled column is added (a column order and index equality workaround is applied); a single null with dummy_na gives a one-column frame with value one. Sparse variants noted as unsupported for null labels in the original issue.
        s = ["a", "b", np.nan]
        res = get_dummies(s, sparse=sparse, dtype=dtype)
        exp = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0]}, dtype=self.effective_dtype(dtype)
        )
        if sparse:
            if dtype.kind == "b":
                exp = exp.apply(SparseArray, fill_value=False)
            else:
                exp = exp.apply(SparseArray, fill_value=0.0)
        tm.assert_frame_equal(res, exp)

        # Sparse dataframes do not allow nan labelled columns, see #GH8822
        res_na = get_dummies(s, dummy_na=True, sparse=sparse, dtype=dtype)
        exp_na = DataFrame(
            {np.nan: [0, 0, 1], "a": [1, 0, 0], "b": [0, 1, 0]},
            dtype=self.effective_dtype(dtype),
        )
        exp_na = exp_na.reindex(["a", "b", np.nan], axis=1)
        # hack (NaN handling in assert_index_equal)
        exp_na.columns = res_na.columns
        if sparse:
            if dtype.kind == "b":
                exp_na = exp_na.apply(SparseArray, fill_value=False)
            else:
                exp_na = exp_na.apply(SparseArray, fill_value=0.0)
        tm.assert_frame_equal(res_na, exp_na)

        res_just_na = get_dummies([np.nan], dummy_na=True, sparse=sparse, dtype=dtype)
        exp_just_na = DataFrame(
            Series(1, index=[0]), columns=[np.nan], dtype=self.effective_dtype(dtype)
        )
        tm.assert_numpy_array_equal(res_just_na.values, exp_just_na.values)

    def test_get_dummies_unicode(self, sparse):
        # See GH 6885 - get_dummies chokes on unicode values
        # [FUNCTION DOC]
        # See GH 6885. Non-ASCII category values and the prefix produce correctly named boolean columns (sparse variant too).
        e = "e"
        eacute = unicodedata.lookup("LATIN SMALL LETTER E WITH ACUTE")
        s = [e, eacute, eacute]
        res = get_dummies(s, prefix="letter", sparse=sparse)
        exp = DataFrame(
            {"letter_e": [True, False, False], f"letter_{eacute}": [False, True, True]}
        )
        if sparse:
            exp = exp.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(res, exp)

    def test_dataframe_dummies_all_obj(self, df, sparse):
        # [FUNCTION DOC]
        # A frame of only string columns yields boolean columns named column underscore value, sparse boolean arrays when sparse.
        df = df[["A", "B"]]
        result = get_dummies(df, sparse=sparse)
        expected = DataFrame(
            {"A_a": [1, 0, 1], "A_b": [0, 1, 0], "B_b": [1, 1, 0], "B_c": [0, 0, 1]},
            dtype=bool,
        )
        if sparse:
            expected = DataFrame(
                {
                    "A_a": SparseArray([1, 0, 1], dtype="bool"),
                    "A_b": SparseArray([0, 1, 0], dtype="bool"),
                    "B_b": SparseArray([1, 1, 0], dtype="bool"),
                    "B_c": SparseArray([0, 0, 1], dtype="bool"),
                }
            )

        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_string_dtype(self, df, any_string_dtype):
        # GH44965
        # [FUNCTION DOC]
        # GH44965. String columns of the default string and the parametrized string dtype are encoded; the result is boolean except the nullable boolean for the nullable string dtype.
        df = df[["A", "B"]]
        df = df.astype({"A": "str", "B": any_string_dtype})
        result = get_dummies(df)
        expected = DataFrame(
            {
                "A_a": [1, 0, 1],
                "A_b": [0, 1, 0],
                "B_b": [1, 1, 0],
                "B_c": [0, 0, 1],
            },
            dtype=bool,
        )
        if any_string_dtype == "string" and any_string_dtype.na_value is pd.NA:
            expected[["B_b", "B_c"]] = expected[["B_b", "B_c"]].astype("boolean")
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_mix_default(self, df, sparse, dtype):
        # [FUNCTION DOC]
        # Mixed frame: the numeric column is passed through first and the string columns are expanded, in the requested dtype (sparse with fill zero or false).
        result = get_dummies(df, sparse=sparse, dtype=dtype)
        if sparse:
            arr = SparseArray
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "A_a": arr([1, 0, 1], dtype=typ),
                "A_b": arr([0, 1, 0], dtype=typ),
                "B_b": arr([1, 1, 0], dtype=typ),
                "B_c": arr([0, 0, 1], dtype=typ),
            }
        )
        expected = expected[["C", "A_a", "A_b", "B_b", "B_c"]]
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_list(self, df, sparse):
        # [FUNCTION DOC]
        # A list of prefixes (one per encoded column) replaces the column names in the output; the numeric column comes first. Result columns are boolean (sparse arrays if requested).
        prefixes = ["from_A", "from_B"]
        result = get_dummies(df, prefix=prefixes, sparse=sparse)
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "from_A_a": [True, False, True],
                "from_A_b": [False, True, False],
                "from_B_b": [True, True, False],
                "from_B_c": [False, False, True],
            },
        )
        expected[["C"]] = df[["C"]]
        cols = ["from_A_a", "from_A_b", "from_B_b", "from_B_c"]
        expected = expected[["C", *cols]]

        typ = SparseArray if sparse else Series
        expected[cols] = expected[cols].apply(lambda x: typ(x))
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_str(self, df, sparse):
        # not that you should do this...
        # [FUNCTION DOC]
        # A single string prefix is applied to every encoded column, producing duplicate output names; sparse expected is built by concatenation to work around a duplicate-column limitation (issue 14427).
        result = get_dummies(df, prefix="bad", sparse=sparse)
        bad_columns = ["bad_a", "bad_b", "bad_b", "bad_c"]
        expected = DataFrame(
            [
                [1, True, False, True, False],
                [2, False, True, True, False],
                [3, True, False, False, True],
            ],
            columns=["C", *bad_columns],
        )
        expected = expected.astype({"C": np.int64})
        if sparse:
            # work around astyping & assigning with duplicate columns
            # https://github.com/pandas-dev/pandas/issues/14427
            expected = pd.concat(
                [
                    Series([1, 2, 3], name="C"),
                    Series([True, False, True], name="bad_a", dtype="Sparse[bool]"),
                    Series([False, True, False], name="bad_b", dtype="Sparse[bool]"),
                    Series([True, True, False], name="bad_b", dtype="Sparse[bool]"),
                    Series([False, False, True], name="bad_c", dtype="Sparse[bool]"),
                ],
                axis=1,
            )

        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_subset(self, df, sparse):
        # [FUNCTION DOC]
        # Using a columns list restricts encoding to those columns; the others are kept (not encoded) and appear first, with the given prefix on the new columns.
        result = get_dummies(df, prefix=["from_A"], columns=["A"], sparse=sparse)
        expected = DataFrame(
            {
                "B": ["b", "b", "c"],
                "C": [1, 2, 3],
                "from_A_a": [1, 0, 1],
                "from_A_b": [0, 1, 0],
            },
        )
        cols = expected.columns
        expected[cols[1:]] = expected[cols[1:]].astype(bool)
        expected[["C"]] = df[["C"]]
        if sparse:
            cols = ["from_A_a", "from_A_b"]
            expected[cols] = expected[cols].astype(SparseDtype("bool", False))
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_sep(self, df, sparse):
        # [FUNCTION DOC]
        # A custom separator as a string, as a list (per column) and as a dict keyed by column name determines the output column names.
        result = get_dummies(df, prefix_sep="..", sparse=sparse)
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "A..a": [True, False, True],
                "A..b": [False, True, False],
                "B..b": [True, True, False],
                "B..c": [False, False, True],
            },
        )
        expected[["C"]] = df[["C"]]
        expected = expected[["C", "A..a", "A..b", "B..b", "B..c"]]
        if sparse:
            cols = ["A..a", "A..b", "B..b", "B..c"]
            expected[cols] = expected[cols].astype(SparseDtype("bool", False))

        tm.assert_frame_equal(result, expected)

        result = get_dummies(df, prefix_sep=["..", "__"], sparse=sparse)
        expected = expected.rename(columns={"B..b": "B__b", "B..c": "B__c"})
        tm.assert_frame_equal(result, expected)

        result = get_dummies(df, prefix_sep={"A": "..", "B": "__"}, sparse=sparse)
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_bad_length(self, df, sparse):
        # [FUNCTION DOC]
        # A prefix list whose length differs from the number of encoded columns raises a value error describing the length mismatch.
        msg = re.escape(
            "Length of 'prefix' (1) did not match the length of the columns being "
            "encoded (2)"
        )
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, prefix=["too few"], sparse=sparse)

    def test_dataframe_dummies_prefix_sep_bad_length(self, df, sparse):
        # [FUNCTION DOC]
        # A separator list of the wrong length raises the corresponding value error.
        msg = re.escape(
            "Length of 'prefix_sep' (1) did not match the length of the columns being "
            "encoded (2)"
        )
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, prefix_sep=["bad"], sparse=sparse)

    def test_dataframe_dummies_prefix_dict(self, sparse):
        # [FUNCTION DOC]
        # A dict of prefixes keyed by column name renames the output columns; result is boolean (sparse if requested).
        prefixes = {"A": "from_A", "B": "from_B"}
        df = DataFrame({"C": [1, 2, 3], "A": ["a", "b", "a"], "B": ["b", "b", "c"]})
        result = get_dummies(df, prefix=prefixes, sparse=sparse)

        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "from_A_a": [1, 0, 1],
                "from_A_b": [0, 1, 0],
                "from_B_b": [1, 1, 0],
                "from_B_c": [0, 0, 1],
            }
        )

        columns = ["from_A_a", "from_A_b", "from_B_b", "from_B_c"]
        expected[columns] = expected[columns].astype(bool)
        if sparse:
            expected[columns] = expected[columns].astype(SparseDtype("bool", False))

        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_with_na(self, df, sparse, dtype):
        # [FUNCTION DOC]
        # A row of nulls added to the frame: with dummy_na enabled each encoded column gets a null indicator, the numeric column holds a null; with dummy_na disabled the null indicator columns are absent. Columns are sorted for comparison; dtype and sparsity follow fixtures.
        df.loc[3, :] = [np.nan, np.nan, np.nan]
        result = get_dummies(df, dummy_na=True, sparse=sparse, dtype=dtype).sort_index(
            axis=1
        )

        if sparse:
            arr = SparseArray
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype

        expected = DataFrame(
            {
                "C": [1, 2, 3, np.nan],
                "A_a": arr([1, 0, 1, 0], dtype=typ),
                "A_b": arr([0, 1, 0, 0], dtype=typ),
                "A_nan": arr([0, 0, 0, 1], dtype=typ),
                "B_b": arr([1, 1, 0, 0], dtype=typ),
                "B_c": arr([0, 0, 1, 0], dtype=typ),
                "B_nan": arr([0, 0, 0, 1], dtype=typ),
            }
        ).sort_index(axis=1)

        tm.assert_frame_equal(result, expected)

        result = get_dummies(df, dummy_na=False, sparse=sparse, dtype=dtype)
        expected = expected[["C", "A_a", "A_b", "B_b", "B_c"]]
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_with_categorical(self, df, sparse, dtype):
        # [FUNCTION DOC]
        # Categorical columns are encoded along with string columns and appear sorted alongside them; dtype and sparsity follow fixtures.
        df["cat"] = Categorical(["x", "y", "y"])
        result = get_dummies(df, sparse=sparse, dtype=dtype).sort_index(axis=1)
        if sparse:
            arr = SparseArray
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype

        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "A_a": arr([1, 0, 1], dtype=typ),
                "A_b": arr([0, 1, 0], dtype=typ),
                "B_b": arr([1, 1, 0], dtype=typ),
                "B_c": arr([0, 0, 1], dtype=typ),
                "cat_x": arr([1, 0, 0], dtype=typ),
                "cat_y": arr([0, 1, 1], dtype=typ),
            }
        ).sort_index(axis=1)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "get_dummies_kwargs,expected",
        [
            (
                {"data": DataFrame({"ä": ["a"]})},
                "ä_a",
            ),
            (
                {"data": DataFrame({"x": ["ä"]})},
                "x_ä",
            ),
            (
                {"data": DataFrame({"x": ["a"]}), "prefix": "ä"},
                "ä_a",
            ),
            (
                {"data": DataFrame({"x": ["a"]}), "prefix_sep": "ä"},
                "xäa",
            ),
        ],
    )
    def test_dataframe_dummies_unicode(self, get_dummies_kwargs, expected):
        # GH22084 get_dummies incorrectly encodes unicode characters
        # in dataframe column names
        # [FUNCTION DOC]
        # GH22084. Non-ASCII characters in column names, values, prefix and separator are preserved in output column names (a single true cell is expected).
        result = get_dummies(**get_dummies_kwargs)
        expected = DataFrame({expected: [True]})
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_drop_first(self, sparse):
        # GH12402 Add a new parameter `drop_first` to avoid collinearity
        # Basic case
        # [FUNCTION DOC]
        # GH12402. Dropping the first level of each variable leaves only columns for the remaining levels (boolean; sparse with fill false); index preserved for indexed series.
        s_list = list("abc")
        s_series = Series(s_list)
        s_series_index = Series(s_list, list("ABC"))

        expected = DataFrame({"b": [0, 1, 0], "c": [0, 0, 1]}, dtype=bool)

        result = get_dummies(s_list, drop_first=True, sparse=sparse)
        if sparse:
            expected = expected.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

        expected.index = list("ABC")
        result = get_dummies(s_series_index, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_drop_first_one_level(self, sparse):
        # Test the case that categorical variable only has one level.
        # [FUNCTION DOC]
        # A variable with a single level and drop-first yields a frame with no columns but the right index (default range or the given labels).
        s_list = list("aaa")
        s_series = Series(s_list)
        s_series_index = Series(s_list, list("ABC"))

        expected = DataFrame(index=RangeIndex(3))

        result = get_dummies(s_list, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(index=list("ABC"))
        result = get_dummies(s_series_index, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_drop_first_NA(self, sparse):
        # Test NA handling together with drop_first
        # [FUNCTION DOC]
        # Drop-first with nulls: the null is ignored by default; with dummy_na the null column is kept (only the first non-null level dropped); a lone null with drop-first gives an empty-column frame.
        s_NA = ["a", "b", np.nan]
        res = get_dummies(s_NA, drop_first=True, sparse=sparse)
        exp = DataFrame({"b": [0, 1, 0]}, dtype=bool)
        if sparse:
            exp = exp.apply(SparseArray, fill_value=False)

        tm.assert_frame_equal(res, exp)

        res_na = get_dummies(s_NA, dummy_na=True, drop_first=True, sparse=sparse)
        exp_na = DataFrame({"b": [0, 1, 0], np.nan: [0, 0, 1]}, dtype=bool).reindex(
            ["b", np.nan], axis=1
        )
        if sparse:
            exp_na = exp_na.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(res_na, exp_na)

        res_just_na = get_dummies(
            [np.nan], dummy_na=True, drop_first=True, sparse=sparse
        )
        exp_just_na = DataFrame(index=RangeIndex(1))
        tm.assert_frame_equal(res_just_na, exp_just_na)

    def test_dataframe_dummies_drop_first(self, df, sparse):
        # [FUNCTION DOC]
        # Drop-first on a two-string-column frame removes the first level of each column.
        df = df[["A", "B"]]
        result = get_dummies(df, drop_first=True, sparse=sparse)
        expected = DataFrame({"A_b": [0, 1, 0], "B_c": [0, 0, 1]}, dtype=bool)
        if sparse:
            expected = expected.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_drop_first_with_categorical(self, df, sparse, dtype):
        # [FUNCTION DOC]
        # Drop-first on a frame including a categorical column; passes the numeric column through and drops the first category of each encoded column. Sparse variant wraps the boolean columns as sparse arrays.
        # KNOWN ISSUE: the dtype fixture parameter is accepted but never used, so the test is run redundantly for every dtype.
        df["cat"] = Categorical(["x", "y", "y"])
        result = get_dummies(df, drop_first=True, sparse=sparse)
        expected = DataFrame(
            {"C": [1, 2, 3], "A_b": [0, 1, 0], "B_c": [0, 0, 1], "cat_y": [0, 1, 1]}
        )
        cols = ["A_b", "B_c", "cat_y"]
        expected[cols] = expected[cols].astype(bool)
        expected = expected[["C", "A_b", "B_c", "cat_y"]]
        if sparse:
            for col in cols:
                expected[col] = SparseArray(expected[col])
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_drop_first_with_na(self, df, sparse):
        # [FUNCTION DOC]
        # Drop-first together with null indicators: with dummy_na enabled the null indicator columns remain; with it disabled they are absent. Columns sorted for comparison.
        df.loc[3, :] = [np.nan, np.nan, np.nan]
        result = get_dummies(
            df, dummy_na=True, drop_first=True, sparse=sparse
        ).sort_index(axis=1)
        expected = DataFrame(
            {
                "C": [1, 2, 3, np.nan],
                "A_b": [0, 1, 0, 0],
                "A_nan": [0, 0, 0, 1],
                "B_c": [0, 0, 1, 0],
                "B_nan": [0, 0, 0, 1],
            }
        )
        cols = ["A_b", "A_nan", "B_c", "B_nan"]
        expected[cols] = expected[cols].astype(bool)
        expected = expected.sort_index(axis=1)
        if sparse:
            for col in cols:
                expected[col] = SparseArray(expected[col])

        tm.assert_frame_equal(result, expected)

        result = get_dummies(df, dummy_na=False, drop_first=True, sparse=sparse)
        expected = expected[["C", "A_b", "B_c"]]
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_int_int(self):
        # [FUNCTION DOC]
        # Integer series values become integer column labels; a categorical series gives categorical column labels. Both results boolean.
        data = Series([1, 2, 1])
        result = get_dummies(data)
        expected = DataFrame([[1, 0], [0, 1], [1, 0]], columns=[1, 2], dtype=bool)
        tm.assert_frame_equal(result, expected)

        data = Series(Categorical(["a", "b", "a"]))
        result = get_dummies(data)
        expected = DataFrame(
            [[1, 0], [0, 1], [1, 0]], columns=Categorical(["a", "b"]), dtype=bool
        )
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_int_df(self, dtype):
        # [FUNCTION DOC]
        # A frame with integer, categorical and float columns, with only some columns requested: the remaining columns are unchanged and new columns have the requested dtype.
        data = DataFrame(
            {
                "A": [1, 2, 1],
                "B": Categorical(["a", "b", "a"]),
                "C": [1, 2, 1],
                "D": [1.0, 2.0, 1.0],
            }
        )
        columns = ["C", "D", "A_1", "A_2", "B_a", "B_b"]
        expected = DataFrame(
            [[1, 1.0, 1, 0, 1, 0], [2, 2.0, 0, 1, 0, 1], [1, 1.0, 1, 0, 1, 0]],
            columns=columns,
        )
        expected[columns[2:]] = expected[columns[2:]].astype(dtype)
        result = get_dummies(data, columns=["A", "B"], dtype=dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_dataframe_dummies_preserve_categorical_dtype(self, dtype, ordered):
        # GH13854
        # [FUNCTION DOC]
        # GH13854. Encoding a categorical (ordered or not) with an unobserved category gives columns that are a categorical index with the same categories and orderedness; includes the unobserved category as an all-zero column.
        cat = Categorical(list("xy"), categories=list("xyz"), ordered=ordered)
        result = get_dummies(cat, dtype=dtype)

        data = np.array([[1, 0, 0], [0, 1, 0]], dtype=self.effective_dtype(dtype))
        cols = CategoricalIndex(
            cat.categories, categories=cat.categories, ordered=ordered
        )
        expected = DataFrame(data, columns=cols, dtype=self.effective_dtype(dtype))

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_get_dummies_dont_sparsify_all_columns(self, sparse):
        # GH18914
        # [FUNCTION DOC]
        # GH18914. Even with sparse output the non-encoded numeric column stays dense.
        df = DataFrame.from_dict({"GDP": [1, 2], "Nation": ["AB", "CD"]})
        df = get_dummies(df, columns=["Nation"], sparse=sparse)
        df2 = df.reindex(columns=["GDP"])

        tm.assert_frame_equal(df[["GDP"]], df2)

    def test_get_dummies_duplicate_columns(self, df):
        # GH20839
        # [FUNCTION DOC]
        # GH20839. A frame with duplicate column labels is encoded; the output has the numeric duplicated name column plus boolean indicator columns, all sorted for comparison.
        df.columns = ["A", "A", "A"]
        result = get_dummies(df).sort_index(axis=1)

        expected = DataFrame(
            [
                [1, True, False, True, False],
                [2, False, True, True, False],
                [3, True, False, False, True],
            ],
            columns=["A", "A_a", "A_b", "A_b", "A_c"],
        ).sort_index(axis=1)

        expected = expected.astype({"A": np.int64})

        tm.assert_frame_equal(result, expected)

    def test_get_dummies_all_sparse(self):
        # [FUNCTION DOC]
        # When every column is encoded sparsely, all result columns are sparse boolean arrays with fill false.
        df = DataFrame({"A": [1, 2]})
        result = get_dummies(df, columns=["A"], sparse=True)
        dtype = SparseDtype("bool", False)
        expected = DataFrame(
            {
                "A_1": SparseArray([1, 0], dtype=dtype),
                "A_2": SparseArray([0, 1], dtype=dtype),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("values", ["baz"])
    def test_get_dummies_with_string_values(self, values):
        # issue #28383
        # [FUNCTION DOC]
        # Issue #28383. Passing the columns argument as a plain string raises a type error saying columns must be list-like. Parametrized on a single value.
        df = DataFrame(
            {
                "bar": [1, 2, 3, 4, 5, 6],
                "foo": ["one", "one", "one", "two", "two", "two"],
                "baz": ["A", "B", "C", "A", "B", "C"],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )

        msg = "Input must be a list-like for parameter `columns`"

        with pytest.raises(TypeError, match=msg):
            get_dummies(df, columns=values)

    def test_get_dummies_ea_dtype_series(self, any_numeric_ea_and_arrow_dtype):
        # GH#32430
        # [FUNCTION DOC]
        # GH#32430. A series encoded with any numeric extension or arrow dtype yields columns of that dtype.
        ser = Series(list("abca"))
        result = get_dummies(ser, dtype=any_numeric_ea_and_arrow_dtype)
        expected = DataFrame(
            {"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]},
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_ea_dtype_dataframe(self, any_numeric_ea_and_arrow_dtype):
        # GH#32430
        # [FUNCTION DOC]
        # GH#32430. Same for frame input, with prefixed column names.
        df = DataFrame({"x": list("abca")})
        result = get_dummies(df, dtype=any_numeric_ea_and_arrow_dtype)
        expected = DataFrame(
            {"x_a": [1, 0, 0, 1], "x_b": [0, 1, 0, 0], "x_c": [0, 0, 1, 0]},
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype_type", ["string", "category"])
    def test_get_dummies_ea_dtype(self, dtype_type, string_dtype_no_object):
        # GH#56273
        # [FUNCTION DOC]
        # GH#56273. String extension columns (or categoricals built from string-typed categories), parametrized by kind: output indicators are nullable boolean when the string dtype uses the missing marker, else plain boolean.
        dtype = string_dtype_no_object
        exp_dtype = "boolean" if dtype.na_value is pd.NA else "bool"
        if dtype_type == "category":
            dtype = CategoricalDtype(Index(["a"], dtype))
        df = DataFrame({"name": Series(["a"], dtype=dtype), "x": 1})
        result = get_dummies(df)
        expected = DataFrame({"x": 1, "name_a": Series([True], dtype=exp_dtype)})
        tm.assert_frame_equal(result, expected)

    @td.skip_if_no("pyarrow")
    def test_get_dummies_arrow_dtype(self):
        # GH#56273
        # [FUNCTION DOC]
        # GH#56273. Requires pyarrow. Arrow string columns, and categoricals with arrow string categories, yield arrow-backed boolean indicator columns.
        df = DataFrame({"name": Series(["a"], dtype=ArrowDtype(pa.string())), "x": 1})
        result = get_dummies(df)
        expected = DataFrame({"x": 1, "name_a": Series([True], dtype="bool[pyarrow]")})
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {
                "name": Series(
                    ["a"],
                    dtype=CategoricalDtype(Index(["a"], dtype=ArrowDtype(pa.string()))),
                ),
                "x": 1,
            }
        )
        result = get_dummies(df)
        tm.assert_frame_equal(result, expected)
