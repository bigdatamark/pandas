# [MODULE DOC]
# Tests for DataFrame.sort_values: sorting by one or several columns (or rows when sorting along axis 1),
# ascending flags, na_position, stability, inplace, ignore_index, the key option, datetimelike and
# categorical data, and sorting by a mixture of index level names and column labels. Defines fixtures used
# by the last test class (a frame, an index-based variant and a set of sort keys).

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    NaT,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameSortValues:
    # [CLASS DOC]
    # Test container for sort_values with the standard options.
    @pytest.mark.parametrize("dtype", [np.uint8, bool])
    def test_sort_values_sparse_no_warning(self, dtype):
        # [FUNCTION DOC]
        # Regression test (GH 45618). Parametrized over unsigned 8-bit and boolean dtypes. Sorting a frame of sparse dummy columns by all its columns emits no warning (in particular none about building an index from a sparse array).
        # GH#45618
        ser = pd.Series(Categorical(["a", "b", "a"], categories=["a", "b", "c"]))
        df = pd.get_dummies(ser, dtype=dtype, sparse=True)

        with tm.assert_produces_warning(None):
            # No warnings about constructing Index from SparseArray
            df.sort_values(by=df.columns.tolist())

    def test_sort_values(self):
        # [FUNCTION DOC]
        # Covers the basics on a 3x3 frame: sorting by one column ascending and descending (also with a one-element ascending list, GH 4839), by two columns with single or per-column ascending flags, an invalid axis raising a value error that no axis named 2 exists, sorting columns by a row label (axis 1, GH 10806) for single and multiple rows and orders, and a length mismatch between ascending and by raising a value error stating both lengths.
        frame = DataFrame(
            [[1, 1, 2], [3, 1, 0], [4, 5, 6]], index=[1, 2, 3], columns=list("ABC")
        )

        # by column (axis=0)
        sorted_df = frame.sort_values(by="A")
        indexer = frame["A"].argsort().values
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        indexer = indexer[::-1]
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        # GH4839
        sorted_df = frame.sort_values(by=["A"], ascending=[False])
        tm.assert_frame_equal(sorted_df, expected)

        # multiple bys
        sorted_df = frame.sort_values(by=["B", "C"])
        expected = frame.loc[[2, 1, 3]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=["B", "C"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected[::-1])

        sorted_df = frame.sort_values(by=["B", "A"], ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=2, inplace=True)

        # by row (axis=1): GH#10806
        sorted_df = frame.sort_values(by=3, axis=1)
        expected = frame
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=3, axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 2], axis="columns")
        expected = frame.reindex(columns=["B", "A", "C"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        msg = r"Length of ascending \(5\) != length of by \(2\)"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=0, ascending=[True] * 5)

    def test_sort_values_by_empty_list(self):
        # [FUNCTION DOC]
        # Regression test for issue 40258. Sorting by an empty list returns an equal frame that is a new object.
        # https://github.com/pandas-dev/pandas/issues/40258
        expected = DataFrame({"a": [1, 4, 2, 5, 3, 6]})
        result = expected.sort_values(by=[])
        tm.assert_frame_equal(result, expected)
        assert result is not expected

    def test_sort_values_inplace(self):
        # [FUNCTION DOC]
        # Inplace sorting returns nothing and equals the non-inplace result for sorting by one column, by a row label along axis 1, by one column descending, and by two columns descending.
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by="A", inplace=True)
        assert return_value is None
        expected = frame.sort_values(by="A")
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by=1, axis=1, inplace=True)
        assert return_value is None
        expected = frame.sort_values(by=1, axis=1)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by="A", ascending=False, inplace=True)
        assert return_value is None
        expected = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by=["A", "B"], ascending=False, inplace=True
        )
        assert return_value is None
        expected = frame.sort_values(by=["A", "B"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_multicolumn(self):
        # [FUNCTION DOC]
        # Sorting 100 rows by two columns holding shuffled integers equals a lexicographic reference ordering, for ascending, for descending (ranks reversed) and for swapped key order.
        A = np.arange(5).repeat(20)
        B = np.tile(np.arange(5), 20)
        np.random.default_rng(2).shuffle(A)
        np.random.default_rng(2).shuffle(B)
        frame = DataFrame(
            {"A": A, "B": B, "C": np.random.default_rng(2).standard_normal(100)}
        )

        result = frame.sort_values(by=["A", "B"])
        indexer = np.lexsort((frame["B"], frame["A"]))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

        result = frame.sort_values(by=["A", "B"], ascending=False)
        indexer = np.lexsort(
            (frame["B"].rank(ascending=False), frame["A"].rank(ascending=False))
        )
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

        result = frame.sort_values(by=["B", "A"])
        indexer = np.lexsort((frame["A"], frame["B"]))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_multicolumn_uint64(self):
        # [FUNCTION DOC]
        # Regression test (GH 9918). Two-column sort where the first column holds unsigned 64-bit values above the signed range orders the rows correctly (second row first).
        # GH#9918
        # uint64 multicolumn sort

        df = DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            }
        )
        df["a"] = df["a"].astype(np.uint64)
        result = df.sort_values(["a", "b"])

        expected = DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            },
            index=range(1, -1, -1),
        )

        tm.assert_frame_equal(result, expected)

    def test_sort_values_nan(self):
        # [FUNCTION DOC]
        # Regression test (GH 3917). Missing values in one or two columns are placed first or last as requested with ascending, descending and mixed per-column ordering, and sorting along axis 1 by a row with missing values placed first.
        # GH#3917
        df = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]}
        )

        # sort one column only
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        expected = DataFrame(
            {"A": [np.nan, 8, 6, 4, 2, 1, 1], "B": [5, 4, 5, 5, np.nan, 9, 2]},
            index=[2, 5, 4, 6, 1, 0, 3],
        )
        sorted_df = df.sort_values(["A"], na_position="first", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        expected = df.reindex(columns=["B", "A"])
        sorted_df = df.sort_values(by=1, axis=1, na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last', order
        expected = DataFrame(
            {"A": [1, 1, 2, 4, 6, 8, np.nan], "B": [2, 9, np.nan, 5, 5, 4, 5]},
            index=[3, 0, 1, 6, 4, 5, 2],
        )
        sorted_df = df.sort_values(["A", "B"])
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first', order
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 2, 9, np.nan, 5, 5, 4]},
            index=[2, 3, 0, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first', not order
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[1, 0], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last', not order
        expected = DataFrame(
            {"A": [8, 6, 4, 2, 1, 1, np.nan], "B": [4, 5, 5, np.nan, 2, 9, 5]},
            index=[5, 4, 6, 1, 3, 0, 2],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[0, 1], na_position="last")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_descending_sort(self):
        # [FUNCTION DOC]
        # Regression test (GH 6399). A descending merge sort on a column with ties keeps the original relative order of tied rows, so an already ordered frame is unchanged.
        # GH#6399
        df = DataFrame(
            [[2, "first"], [2, "second"], [1, "a"], [1, "b"]],
            columns=["sort_col", "order"],
        )
        sorted_df = df.sort_values(by="sort_col", kind="mergesort", ascending=False)
        tm.assert_frame_equal(df, sorted_df)

    @pytest.mark.parametrize(
        "expected_idx_non_na, ascending",
        [
            [
                [3, 4, 5, 0, 1, 8, 6, 9, 7, 10, 13, 14],
                [True, True],
            ],
            [
                [0, 3, 4, 5, 1, 8, 6, 7, 10, 13, 14, 9],
                [True, False],
            ],
            [
                [9, 7, 10, 13, 14, 6, 8, 1, 3, 4, 5, 0],
                [False, True],
            ],
            [
                [7, 10, 13, 14, 9, 6, 8, 1, 0, 3, 4, 5],
                [False, False],
            ],
        ],
    )
    @pytest.mark.parametrize("na_position", ["first", "last"])
    def test_sort_values_stable_multicolumn_sort(
        self, expected_idx_non_na, ascending, na_position
    ):
        # [FUNCTION DOC]
        # Regression test (GH 38426). Parametrized over four expected orderings (from combinations of ascending flags for the two columns) and the na_position choice. A 15-row frame with ties and missing values is sorted by two columns; the result must equal taking the rows in the expected order (rows with missing values in the first column come first or last in original order).
        # GH#38426 Clarify sort_values with mult. columns / labels is stable
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 1, 1, 1, 6, 8, 4, 8, 8, np.nan, np.nan, 8, 8],
                "B": [9, np.nan, 5, 2, 2, 2, 5, 4, 5, 3, 4, np.nan, np.nan, 4, 4],
            }
        )
        # All rows with NaN in col "B" only have unique values in "A", therefore,
        # only the rows with NaNs in "A" have to be treated individually:
        expected_idx = (
            [11, 12, 2, *expected_idx_non_na]
            if na_position == "first"
            else [*expected_idx_non_na, 2, 11, 12]
        )
        expected = df.take(expected_idx)
        sorted_df = df.sort_values(
            ["A", "B"], ascending=ascending, na_position=na_position
        )
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_categorial(self):
        # [FUNCTION DOC]
        # Regression test (GH 16793). Merge sorting an already sorted ordered categorical column leaves the frame unchanged.
        # GH#16793
        df = DataFrame({"x": Categorical(np.repeat([1, 2, 3, 4], 5), ordered=True)})
        expected = df.copy()
        sorted_df = df.sort_values("x", kind="mergesort")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_datetimes(self):
        # [FUNCTION DOC]
        # Regression test (GH 3461). Sorting by a string column, a datetime column, or a constant float column followed by the datetime column gives the same result whether the key is given as a label or a one-element list, and sorting by the constant plus datetime equals sorting by the datetime alone.
        # GH#3461, argsort / lexsort differences for a datetime column
        df = DataFrame(
            ["a", "a", "a", "b", "c", "d", "e", "f", "g"],
            columns=["A"],
            index=date_range("20130101", periods=9),
        )
        dts = [
            Timestamp(x)
            for x in [
                "2004-02-11",
                "2004-01-21",
                "2004-01-26",
                "2005-09-20",
                "2010-10-04",
                "2009-05-12",
                "2008-11-12",
                "2010-09-28",
                "2010-09-28",
            ]
        ]
        df["B"] = dts[::2] + dts[1::2]
        df["C"] = 2.0
        df["A1"] = 3.0

        df1 = df.sort_values(by="A")
        df2 = df.sort_values(by=["A"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")
        df2 = df.sort_values(by=["B"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")

        df2 = df.sort_values(by=["C", "B"])
        tm.assert_frame_equal(df1, df2)

    def test_sort_values_frame_column_inplace_sort_exception(self, float_frame):
        # [FUNCTION DOC]
        # Sorting a column extracted from a frame in place changes only the extracted series (it is a new object under copy-on-write), not the frame; a copy of that series can then also be sorted.
        # KNOWN ISSUE: the name refers to an exception but nothing here raises or expects an exception.
        s = float_frame["A"]
        float_frame_orig = float_frame.copy()
        # INFO(CoW) Series is a new object, so can be changed inplace
        # without modifying original datafame
        s.sort_values(inplace=True)
        tm.assert_series_equal(s, float_frame_orig["A"].sort_values())
        # column in dataframe is not changed
        tm.assert_frame_equal(float_frame, float_frame_orig)

        cp = s.copy()
        cp.sort_values()  # it works!

    def test_sort_values_nat_values_in_int_column(self):
        # [FUNCTION DOC]
        # Regression test (GH 14922). The raw integer value of NaT in an integer column must not be treated as missing, so na_position does not change the order; with a datetime column the NaT is still treated as missing, going first or last as requested and ignoring the ascending flag when it is last.
        # GH#14922: "sorting with large float and multiple columns incorrect"

        # cause was that the int64 value NaT was considered as "na". Which is
        # only correct for datetime64 columns.

        int_values = (2, int(NaT._value))
        float_values = (2.0, -1.797693e308)

        df = DataFrame(
            {"int": int_values, "float": float_values}, columns=["int", "float"]
        )

        df_reversed = DataFrame(
            {"int": int_values[::-1], "float": float_values[::-1]},
            columns=["int", "float"],
            index=range(1, -1, -1),
        )

        # NaT is not a "na" for int64 columns, so na_position must not
        # influence the result:
        df_sorted = df.sort_values(["int", "float"], na_position="last")
        tm.assert_frame_equal(df_sorted, df_reversed)

        df_sorted = df.sort_values(["int", "float"], na_position="first")
        tm.assert_frame_equal(df_sorted, df_reversed)

        # reverse sorting order
        df_sorted = df.sort_values(["int", "float"], ascending=False)
        tm.assert_frame_equal(df_sorted, df)

        # and now check if NaT is still considered as "na" for datetime64
        # columns:
        df = DataFrame(
            {"datetime": [Timestamp("2016-01-01"), NaT], "float": float_values},
            columns=["datetime", "float"],
        )

        df_reversed = DataFrame(
            {"datetime": [NaT, Timestamp("2016-01-01")], "float": float_values[::-1]},
            columns=["datetime", "float"],
            index=range(1, -1, -1),
        )

        df_sorted = df.sort_values(["datetime", "float"], na_position="first")
        tm.assert_frame_equal(df_sorted, df_reversed)

        df_sorted = df.sort_values(["datetime", "float"], na_position="last")
        tm.assert_frame_equal(df_sorted, df)

        # Ascending should not affect the results.
        df_sorted = df.sort_values(["datetime", "float"], ascending=False)
        tm.assert_frame_equal(df_sorted, df)

    def test_sort_nat(self):
        # [FUNCTION DOC]
        # Regression test (GH 16836). Sorting by two datetime columns where one value is missing puts the row with the missing value last in the first column order.
        # GH 16836

        d1 = [Timestamp(x) for x in ["2016-01-01", "2015-01-01", np.nan, "2016-01-01"]]
        d2 = [
            Timestamp(x)
            for x in ["2017-01-01", "2014-01-01", "2016-01-01", "2015-01-01"]
        ]
        df = DataFrame({"a": d1, "b": d2}, index=[0, 1, 2, 3])

        d3 = [Timestamp(x) for x in ["2015-01-01", "2016-01-01", "2016-01-01", np.nan]]
        d4 = [
            Timestamp(x)
            for x in ["2014-01-01", "2015-01-01", "2017-01-01", "2016-01-01"]
        ]
        expected = DataFrame({"a": d3, "b": d4}, index=[1, 3, 0, 2])
        sorted_df = df.sort_values(by=["a", "b"])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_na_position_with_categories(self):
        # [FUNCTION DOC]
        # Regression test (GH 22556). Missing values in an ordered categorical column are placed first or last as requested, for ascending and descending, with the non-missing values in category order.
        # GH#22556
        # Positioning missing value properly when column is Categorical.
        categories = ["A", "B", "C"]
        category_indices = [0, 2, 4]
        list_of_nans = [np.nan, np.nan]
        na_indices = [1, 3]
        na_position_first = "first"
        na_position_last = "last"
        column_name = "c"

        reversed_categories = sorted(categories, reverse=True)
        reversed_category_indices = sorted(category_indices, reverse=True)
        reversed_na_indices = sorted(na_indices)

        df = DataFrame(
            {
                column_name: Categorical(
                    ["A", np.nan, "B", np.nan, "C"], categories=categories, ordered=True
                )
            }
        )
        # sort ascending with na first
        result = df.sort_values(
            by=column_name, ascending=True, na_position=na_position_first
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    list_of_nans + categories, categories=categories, ordered=True
                )
            },
            index=na_indices + category_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort ascending with na last
        result = df.sort_values(
            by=column_name, ascending=True, na_position=na_position_last
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    categories + list_of_nans, categories=categories, ordered=True
                )
            },
            index=category_indices + na_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort descending with na first
        result = df.sort_values(
            by=column_name, ascending=False, na_position=na_position_first
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    list_of_nans + reversed_categories,
                    categories=categories,
                    ordered=True,
                )
            },
            index=reversed_na_indices + reversed_category_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort descending with na last
        result = df.sort_values(
            by=column_name, ascending=False, na_position=na_position_last
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    reversed_categories + list_of_nans,
                    categories=categories,
                    ordered=True,
                )
            },
            index=reversed_category_indices + reversed_na_indices,
        )

        tm.assert_frame_equal(result, expected)

    def test_sort_values_nat(self):
        # [FUNCTION DOC]
        # Regression test (GH 16836).
        # KNOWN ISSUE: the body is identical to the earlier test_sort_nat (same data, same call, same expectation), so it is a duplicate test.
        # Sorting by two datetime columns with a missing value puts the row with the missing value last.
        # GH#16836

        d1 = [Timestamp(x) for x in ["2016-01-01", "2015-01-01", np.nan, "2016-01-01"]]
        d2 = [
            Timestamp(x)
            for x in ["2017-01-01", "2014-01-01", "2016-01-01", "2015-01-01"]
        ]
        df = DataFrame({"a": d1, "b": d2}, index=[0, 1, 2, 3])

        d3 = [Timestamp(x) for x in ["2015-01-01", "2016-01-01", "2016-01-01", np.nan]]
        d4 = [
            Timestamp(x)
            for x in ["2014-01-01", "2015-01-01", "2017-01-01", "2016-01-01"]
        ]
        expected = DataFrame({"a": d3, "b": d4}, index=[1, 3, 0, 2])
        sorted_df = df.sort_values(by=["a", "b"])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_na_position_with_categories_raises(self):
        # [FUNCTION DOC]
        # An invalid na_position string with a categorical column raises a value error naming the invalid position.
        df = DataFrame(
            {
                "c": Categorical(
                    ["A", np.nan, "B", np.nan, "C"],
                    categories=["A", "B", "C"],
                    ordered=True,
                )
            }
        )

        with pytest.raises(ValueError, match="invalid na_position: bad_position"):
            df.sort_values(by="c", ascending=False, na_position="bad_position")

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_dict, sorted_dict, ignore_index, output_index",
        [
            ({"A": [1, 2, 3]}, {"A": [3, 2, 1]}, True, range(3)),
            ({"A": [1, 2, 3]}, {"A": [3, 2, 1]}, False, range(2, -1, -1)),
            (
                {"A": [1, 2, 3], "B": [2, 3, 4]},
                {"A": [3, 2, 1], "B": [4, 3, 2]},
                True,
                range(3),
            ),
            (
                {"A": [1, 2, 3], "B": [2, 3, 4]},
                {"A": [3, 2, 1], "B": [4, 3, 2]},
                False,
                range(2, -1, -1),
            ),
        ],
    )
    def test_sort_values_ignore_index(
        self, inplace, original_dict, sorted_dict, ignore_index, output_index
    ):
        # [FUNCTION DOC]
        # Regression test (GH 30114). Parametrized over inplace, one or two columns, and ignore_index. Sorting descending by column A gives the expected values with either a default integer index (ignore_index) or the original reversed labels; the source frame is unchanged for the non-inplace case.
        # GH 30114
        df = DataFrame(original_dict)
        expected = DataFrame(sorted_dict, index=output_index)
        kwargs = {"ignore_index": ignore_index, "inplace": inplace}

        if inplace:
            result_df = df.copy()
            result_df.sort_values("A", ascending=False, **kwargs)
        else:
            result_df = df.sort_values("A", ascending=False, **kwargs)

        tm.assert_frame_equal(result_df, expected)
        tm.assert_frame_equal(df, DataFrame(original_dict))

    def test_sort_values_nat_na_position_default(self):
        # [FUNCTION DOC]
        # Regression test (GH 13230). Sorting by an integer column and a datetime column where the last row has NaT leaves an already ordered frame unchanged, i.e. NaT sorts last by default.
        # GH 13230
        expected = DataFrame(
            {
                "A": [1, 2, 3, 4, 4],
                "date": pd.DatetimeIndex(
                    [
                        "2010-01-01 09:00:00",
                        "2010-01-01 09:00:01",
                        "2010-01-01 09:00:02",
                        "2010-01-01 09:00:03",
                        "NaT",
                    ]
                ),
            }
        )
        result = expected.sort_values(["A", "date"])
        tm.assert_frame_equal(result, expected)

    def test_sort_values_reshaping(self):
        # [FUNCTION DOC]
        # Regression test (GH 39426). Sorting a one-row frame of 21 columns along axis 1 by its row with ignore_index leaves it unchanged.
        # GH 39426
        values = list(range(21))
        expected = DataFrame([values], columns=values)
        df = expected.sort_values(expected.index[0], axis=1, ignore_index=True)

        tm.assert_frame_equal(df, expected)

    def test_sort_values_no_by_inplace(self):
        # [FUNCTION DOC]
        # Regression test (GH 50643). Inplace sorting by an empty list returns nothing and leaves the frame unchanged.
        # GH#50643
        df = DataFrame({"a": [1, 2, 3]})
        expected = df.copy()
        result = df.sort_values(by=[], inplace=True)
        tm.assert_frame_equal(df, expected)
        assert result is None

    def test_sort_values_no_op_reset_index(self):
        # [FUNCTION DOC]
        # Regression test (GH 52553). An already sorted frame sorted with ignore_index gets a default integer index.
        # GH#52553
        df = DataFrame({"A": [10, 20], "B": [1, 5]}, index=[2, 3])
        result = df.sort_values(by="A", ignore_index=True)
        expected = DataFrame({"A": [10, 20], "B": [1, 5]})
        tm.assert_frame_equal(result, expected)

    def test_sort_by_column_named_none(self):
        # [FUNCTION DOC]
        # Regression test (GH 61512). When a column label is None, sorting with by set to None sorts by that column (rows reorder to put 2 before 3).
        # GH#61512
        df = DataFrame([[3, 1], [2, 2]], columns=[None, "C1"])
        result = df.sort_values(by=None)
        expected = DataFrame([[2, 2], [3, 1]], columns=[None, "C1"], index=[1, 0])
        tm.assert_frame_equal(result, expected)


class TestDataFrameSortKey:  # test key sorting (issue 27237)
    # [CLASS DOC]
    # Test container for sort_values with the key option (a function applied to the values before sorting); see issue 27237. Uses the sort_by_key fixture from the shared test configuration.
    def test_sort_values_inplace_key(self, sort_by_key):
        # [FUNCTION DOC]
        # Inplace variants (single column, row on axis 1, descending, two columns) with the parametrized key function give nothing returned and the same result as the non-inplace sort.
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by="A", inplace=True, key=sort_by_key)
        assert return_value is None
        expected = frame.sort_values(by="A", key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by=1, axis=1, inplace=True, key=sort_by_key
        )
        assert return_value is None
        expected = frame.sort_values(by=1, axis=1, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by="A", ascending=False, inplace=True, key=sort_by_key
        )
        assert return_value is None
        expected = frame.sort_values(by="A", ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        sorted_df.sort_values(
            by=["A", "B"], ascending=False, inplace=True, key=sort_by_key
        )
        expected = frame.sort_values(by=["A", "B"], ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_key(self):
        # [FUNCTION DOC]
        # Sorting a single column with missing values: default, with a shifting key (order unchanged) and with a negating key plus descending order give the same row order, with missing values last.
        df = DataFrame(np.array([0, 5, np.nan, 3, 2, np.nan]))

        result = df.sort_values(0)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(0, key=lambda x: x + 5)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(0, key=lambda x: -x, ascending=False)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key(self):
        # [FUNCTION DOC]
        # Sorting by one and by two columns with a negating key, ascending and descending, gives the expected row orders with missing rows last.
        df = DataFrame(
            {
                "a": np.array([0, 3, np.nan, 3, 2, np.nan]),
                "b": np.array([0, 2, np.nan, 5, 2, np.nan]),
            }
        )

        result = df.sort_values("a", key=lambda x: -x)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by=["a", "b"], key=lambda x: -x)
        expected = df.iloc[[3, 1, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by=["a", "b"], key=lambda x: -x, ascending=False)
        expected = df.iloc[[0, 4, 1, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key_by_name(self):
        # [FUNCTION DOC]
        # Key function receives each column separately and can depend on its name. A key that negates only column a: sorting by a or by a list with a, by b (unaffected) and by both give the expected row orders.
        df = DataFrame(
            {
                "a": np.array([0, 3, np.nan, 3, 2, np.nan]),
                "b": np.array([0, 2, np.nan, 5, 2, np.nan]),
            }
        )

        def key(col):
            # [FUNCTION DOC]
            # HELPER: key function for the by-name test. Takes a column (series); returns its negation when the column is named a, otherwise returns it unchanged.
            if col.name == "a":
                return -col
            else:
                return col

        result = df.sort_values(by="a", key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by=["a"], key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by="b", key=key)
        expected = df.iloc[[0, 1, 4, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by=["a", "b"], key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_string(self):
        # [FUNCTION DOC]
        # Sorting string data: default order by the second column; case-insensitive key over two columns leaves the frame as is; with descending order equals sorting by the second column descending with the same key.
        df = DataFrame(np.array([["hello", "goodbye"], ["hello", "Hello"]]))

        result = df.sort_values(1)
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values([0, 1], key=lambda col: col.str.lower())
        tm.assert_frame_equal(result, df)

        result = df.sort_values(
            [0, 1], key=lambda col: col.str.lower(), ascending=False
        )
        expected = df.sort_values(1, key=lambda col: col.str.lower(), ascending=False)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_empty(self, sort_by_key):
        # [FUNCTION DOC]
        # Sorting an empty frame by a label with the key fixture, and sorting its index with the key, does not raise (no assertions).
        df = DataFrame(np.array([]))

        df.sort_values(0, key=sort_by_key)
        df.sort_index(key=sort_by_key)

    def test_changes_length_raises(self):
        # [FUNCTION DOC]
        # A key function that returns fewer elements than the column raises a value error saying the key must not change the shape.
        df = DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError, match="change the shape"):
            df.sort_values("A", key=lambda x: x[:1])

    def test_sort_values_key_axes(self):
        # [FUNCTION DOC]
        # Key applied along rows: lower-casing a string column and negating a numeric column both reverse the two-row frame.
        df = DataFrame({0: ["Hello", "goodbye"], 1: [0, 1]})

        result = df.sort_values(0, key=lambda col: col.str.lower())
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(1, key=lambda col: -col)
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_dict_axis(self):
        # [FUNCTION DOC]
        # Key applied when sorting columns along axis 1 by a row (lower-casing a string row; negating a numeric row) reverses the two columns.
        df = DataFrame({0: ["Hello", 0], 1: ["goodbye", 1]})

        result = df.sort_values(0, key=lambda col: col.str.lower(), axis=1)
        expected = df.loc[:, ::-1]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(1, key=lambda col: -col, axis=1)
        expected = df.loc[:, ::-1]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_sort_values_key_casts_to_categorical(self, ordered):
        # [FUNCTION DOC]
        # Parametrized over ordered and unordered. Regression for issue 36383. A key that converts the string column to a categorical with reversed category order is respected: sorting by an all-equal column then the string column gives reverse alphabetical row order. Other columns are returned unchanged by the key.
        # https://github.com/pandas-dev/pandas/issues/36383
        categories = ["c", "b", "a"]
        df = DataFrame({"x": [1, 1, 1], "y": ["a", "b", "c"]})

        def sorter(key):
            # [FUNCTION DOC]
            # HELPER: key function for the categorical test. Takes a column (series); for the column named y returns a series of categorical values with the reversed category list and the given ordered flag, otherwise returns the column unchanged.
            if key.name == "y":
                return pd.Series(
                    Categorical(key, categories=categories, ordered=ordered)
                )
            return key

        result = df.sort_values(by=["x", "y"], key=sorter)
        expected = DataFrame(
            {"x": [1, 1, 1], "y": ["c", "b", "a"]}, index=pd.Index([2, 1, 0])
        )

        tm.assert_frame_equal(result, expected)


@pytest.fixture
def df_none():
    # [FUNCTION DOC]
    # Fixture: returns a six-row frame with a string column outer, an integer column inner, a descending integer column A and a column labeled by a tuple holding strings. Used as the reference for sorting by index levels and columns.
    return DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 2, 2, 1, 1],
            "A": np.arange(6, 0, -1),
            ("B", 5): ["one", "one", "two", "two", "one", "one"],
        }
    )


@pytest.fixture(params=[["outer"], ["outer", "inner"]])
def df_idx(request, df_none):
    # [FUNCTION DOC]
    # Parametrized fixture (index of just outer, or outer plus inner). Returns the df_none frame with those columns set as the (multi-)index.
    levels = request.param
    return df_none.set_index(levels)


@pytest.fixture(
    params=[
        "inner",  # index level
        ["outer"],  # list of index level
        "A",  # column
        [("B", 5)],  # list of column
        ["inner", "outer"],  # two index levels
        [("B", 5), "outer"],  # index level and column
        ["A", ("B", 5)],  # Two columns
        ["inner", "outer"],  # two index levels and column
    ]
)
def sort_names(request):
    # [FUNCTION DOC]
    # Parametrized fixture providing eight sort-key specifications: an index level name, a list with one level, a column name, a list with a tuple column label, two level names, a level and a column, two columns, and a repeat of two levels.
    # KNOWN ISSUE: the last parameter duplicates the fifth (two index levels) although its comment says it combines two index levels and a column.
    return request.param


class TestSortValuesLevelAsStr:
    # [CLASS DOC]
    # Test container for sorting by a mixture of index level names and column labels (GH 14353).
    def test_sort_index_level_and_column_label(
        self, df_none, df_idx, sort_names, ascending, request
    ):
        # [FUNCTION DOC]
        # Uses the reference frame, its indexed variant, the sort key fixture and the ascending fixture. For the case with id df_idx0-inner-True the test is marked as an allowed failure (non-strict) because of an unstable sort of duplicates with newer numpy. Sorting the indexed frame by the given names gives the same as sorting the reference by those columns then setting the index.
        # GH#14353
        if request.node.callspec.id == "df_idx0-inner-True":
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "pandas default unstable sorting of duplicates"
                        "issue with numpy>=1.25 with AVX instructions"
                    ),
                    strict=False,
                )
            )

        # Get index levels from df_idx
        levels = df_idx.index.names

        # Compute expected by sorting on columns and the setting index
        expected = df_none.sort_values(
            by=sort_names, ascending=ascending, axis=0
        ).set_index(levels)

        # Compute result sorting on mix on columns and index levels
        result = df_idx.sort_values(by=sort_names, ascending=ascending, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_sort_column_level_and_index_label(
        self, df_none, df_idx, sort_names, ascending, request
    ):
        # [FUNCTION DOC]
        # Same idea transposed: sorting the transposed indexed frame along axis 1 by names equals sorting the reference by columns, setting the index and transposing. The test unconditionally marks itself as a non-strict expected failure (unstable sorting of duplicates) before comparing.
        # GH#14353

        # Get levels from df_idx
        levels = df_idx.index.names

        # Compute expected by sorting on axis=0, setting index levels, and then
        # transposing. For some cases this will result in a frame with
        # multiple column levels
        expected = (
            df_none.sort_values(by=sort_names, ascending=ascending, axis=0)
            .set_index(levels)
            .T
        )

        # Compute result by transposing and sorting on axis=1.
        result = df_idx.T.sort_values(by=sort_names, ascending=ascending, axis=1)

        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )

        tm.assert_frame_equal(result, expected)

    def test_sort_values_validate_ascending_for_value_error(self):
        # [FUNCTION DOC]
        # Regression test (GH 41634). Passing a string as the ascending flag raises a value error saying the argument expects a boolean and received a string.
        # GH41634
        df = DataFrame({"D": [23, 7, 21]})

        msg = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            df.sort_values(by="D", ascending="False")

    def test_sort_values_validate_ascending_functional(self, ascending):
        # [FUNCTION DOC]
        # Using the ascending fixture, sorting a single column gives the same as selecting rows by the (reversed when descending) argsort order.
        df = DataFrame({"D": [23, 7, 21]})
        indexer = df["D"].argsort().values

        if not ascending:
            indexer = indexer[::-1]

        expected = df.loc[df.index[indexer]]
        result = df.sort_values(by="D", ascending=ascending)
        tm.assert_frame_equal(result, expected)
