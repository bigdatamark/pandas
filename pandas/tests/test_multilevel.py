# [MODULE DOC]
# Tests for behavior of objects with hierarchical (multi-level) indexes: reindexing by level, grouping, alignment,
# tuple-valued levels, missing values in index levels, and sorting of non-lexicographically-sorted indexes. Many
# tests use fixtures supplying frames with a multi-level index.
import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    ArrowDtype,
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestMultiLevel:
    # [CLASS DOC]
    # Collection of miscellaneous multi-level index tests, mostly fixture-driven; no state.
    def test_reindex_level(self, multiindex_year_month_day_dataframe_random_data):
        # [FUNCTION DOC]
        # Using the year/month/day fixture frame. Verifies that reindexing per-month sums onto the full index by matching
        # level one equals the grouped transform of the sum by month, for a frame and for a single column Series (names ignored).
        # axis=0
        ymd = multiindex_year_month_day_dataframe_random_data

        month_sums = ymd.groupby("month").sum()
        result = month_sums.reindex(ymd.index, level=1)
        expected = ymd.groupby(level="month").transform("sum")

        tm.assert_frame_equal(result, expected)

        # Series
        result = month_sums["A"].reindex(ymd.index, level=1)
        expected = ymd["A"].groupby(level="month").transform("sum")
        tm.assert_series_equal(result, expected, check_names=False)

    def test_reindex(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Verifies selecting two label tuples from the multi-level fixture frame gives the first and fourth rows, in that order.
        frame = multiindex_dataframe_random_data

        expected = frame.iloc[[0, 3]]
        reindexed = frame.loc[[("foo", "one"), ("bar", "one")]]
        tm.assert_frame_equal(reindexed, expected)

    def test_reindex_preserve_levels(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        # [FUNCTION DOC]
        # Verifies reindexing or label-selecting a frame with a subset of its own multi-level index (every tenth entry), on rows
        # and, after transposing, on columns, preserves the exact index object identity for reindex and equality for label selection.
        ymd = multiindex_year_month_day_dataframe_random_data

        new_index = ymd.index[::10]
        chunk = ymd.reindex(new_index)
        assert chunk.index.is_(new_index)

        chunk = ymd.loc[new_index]
        assert chunk.index.equals(new_index)

        ymdT = ymd.T
        chunk = ymdT.reindex(columns=new_index)
        assert chunk.columns.is_(new_index)

        chunk = ymdT.loc[:, new_index]
        assert chunk.columns.equals(new_index)

    def test_groupby_transform(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Verifies that grouping a Series by the first index level, applying a doubling function and reindexing to the transform
        # index gives the same values as the group transform with the same function.
        frame = multiindex_dataframe_random_data

        s = frame["A"]
        grouper = s.index.get_level_values(0)

        grouped = s.groupby(grouper, group_keys=False)

        applied = grouped.apply(lambda x: x * 2)
        expected = grouped.transform(lambda x: x * 2)
        result = applied.reindex(expected.index)
        tm.assert_series_equal(result, expected, check_names=False)

    def test_groupby_corner(self):
        # [FUNCTION DOC]
        # Verifies that creating a groupby on a level of a single-row frame with a three-level index does not raise (no assertion on the result).
        midx = MultiIndex(
            levels=[["foo"], ["bar"], ["baz"]],
            codes=[[0], [0], [0]],
            names=["one", "two", "three"],
        )
        df = DataFrame(
            [np.random.default_rng(2).random(4)],
            columns=["a", "b", "c", "d"],
            index=midx,
        )
        # should work
        df.groupby(level="three")

    def test_setitem_with_expansion_multiindex_columns(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        # [FUNCTION DOC]
        # Verifies adding a new column by tuple key to a frame with multi-level columns keeps the columns a MultiIndex and the new
        # column equals the copied one.
        ymd = multiindex_year_month_day_dataframe_random_data

        df = ymd[:5].T
        df[2000, 1, 10] = df[2000, 1, 7]
        assert isinstance(df.columns, MultiIndex)
        assert (df[2000, 1, 10] == df[2000, 1, 7]).all()

    def test_alignment(self):
        # [FUNCTION DOC]
        # Verifies arithmetic subtraction between two Series with partly overlapping multi-level indexes equals the subtraction after
        # reindexing both to the union of the indexes; repeated with reversed (non-monotonic) order to hit that code path.
        x = Series(
            data=[1, 2, 3], index=MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3)])
        )

        y = Series(
            data=[4, 5, 6], index=MultiIndex.from_tuples([("Z", 1), ("Z", 2), ("B", 3)])
        )

        res = x - y
        exp_index = x.index.union(y.index)
        exp = x.reindex(exp_index) - y.reindex(exp_index)
        tm.assert_series_equal(res, exp)

        # hit non-monotonic code path
        res = x[::-1] - y[::-1]
        exp_index = x.index.union(y.index)
        exp = x.reindex(exp_index) - y.reindex(exp_index)
        tm.assert_series_equal(res, exp)

    def test_groupby_multilevel(self, multiindex_year_month_day_dataframe_random_data):
        # [FUNCTION DOC]
        # Verifies grouping by the first two levels by position gives the same mean as grouping by the corresponding level value arrays,
        # that the result index names are the first two names, and that grouping by level names gives the same result.
        ymd = multiindex_year_month_day_dataframe_random_data

        result = ymd.groupby(level=[0, 1]).mean()

        k1 = ymd.index.get_level_values(0)
        k2 = ymd.index.get_level_values(1)

        expected = ymd.groupby([k1, k2]).mean()

        tm.assert_frame_equal(result, expected)
        assert result.index.names == ymd.index.names[:2]

        result2 = ymd.groupby(level=ymd.index.names[:2]).mean()
        tm.assert_frame_equal(result, result2)

    def test_multilevel_consolidate(self):
        # [FUNCTION DOC]
        # Builds a frame with matching multi-level rows and columns, adds a total column with a two-part key, then consolidates.
        # KNOWN ISSUE: the test makes no assertions; it only checks that these steps do not raise.
        index = MultiIndex.from_tuples(
            [("foo", "one"), ("foo", "two"), ("bar", "one"), ("bar", "two")]
        )
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), index=index, columns=index
        )
        df["Totals", ""] = df.sum(axis=1)
        df = df._consolidate()

    def test_level_with_tuples(self):
        # [FUNCTION DOC]
        # Verifies lookups and cross-section selection on Series and frames whose first index level holds tuple values (three-element
        # tuples, then two-element tuples): a tuple key selects the matching rows with that level dropped; adding a second key part
        # beyond the tuple raises a key error showing the nested key.
        index = MultiIndex(
            levels=[[("foo", "bar", 0), ("foo", "baz", 0), ("foo", "qux", 0)], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        series = Series(np.random.default_rng(2).standard_normal(6), index=index)
        frame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)

        result = series[("foo", "bar", 0)]
        result2 = series.loc[("foo", "bar", 0)]
        expected = series[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        with pytest.raises(KeyError, match=r"^\(\('foo', 'bar', 0\), 2\)$"):
            series[("foo", "bar", 0), 2]

        result = frame.loc[("foo", "bar", 0)]
        result2 = frame.xs(("foo", "bar", 0))
        expected = frame[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        index = MultiIndex(
            levels=[[("foo", "bar"), ("foo", "baz"), ("foo", "qux")], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        series = Series(np.random.default_rng(2).standard_normal(6), index=index)
        frame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)

        result = series[("foo", "bar")]
        result2 = series.loc[("foo", "bar")]
        expected = series[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        result = frame.loc[("foo", "bar")]
        result2 = frame.xs(("foo", "bar"))
        expected = frame[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_reindex_level_partial_selection(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Verifies selecting the outer labels foo and qux via level reindexing, label-based indexing, on frames, transposed frames
        # (columns) and Series gives the same rows as selecting by position.
        frame = multiindex_dataframe_random_data

        result = frame.reindex(["foo", "qux"], level=0)
        expected = frame.iloc[[0, 1, 2, 7, 8, 9]]
        tm.assert_frame_equal(result, expected)

        result = frame.T.reindex(["foo", "qux"], axis=1, level=0)
        tm.assert_frame_equal(result, expected.T)

        result = frame.loc[["foo", "qux"]]
        tm.assert_frame_equal(result, expected)

        result = frame["A"].loc[["foo", "qux"]]
        tm.assert_series_equal(result, expected["A"])

        result = frame.T.loc[:, ["foo", "qux"]]
        tm.assert_frame_equal(result, expected.T)

    @pytest.mark.parametrize("d", [4, "d"])
    def test_empty_frame_groupby_dtypes_consistency(self, d):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over a numeric and a string last column value. Verifies the first-per-group
        # result of an empty group selection has an empty MultiIndex with the three key level names and the levels still holding values.
        # GH 20888
        group_keys = ["a", "b", "c"]
        df = DataFrame({"a": [1], "b": [2], "c": [3], "d": [d]})

        g = df[df.a == 2].groupby(group_keys)
        result = g.first().index
        expected = MultiIndex(
            levels=[[1], [2], [3]], codes=[[], [], []], names=["a", "b", "c"]
        )

        tm.assert_index_equal(result, expected)

    def test_duplicate_groupby_issues(self):
        # [FUNCTION DOC]
        # Verifies grouping a Series with a duplicated multi-level index entry by its own index yields three groups (duplicates merge).
        idx_tp = [
            ("600809", "20061231"),
            ("600809", "20070331"),
            ("600809", "20070630"),
            ("600809", "20070331"),
        ]
        dt = ["demo", "demo", "demo", "demo"]

        idx = MultiIndex.from_tuples(idx_tp, names=["STK_ID", "RPT_Date"])
        s = Series(dt, index=idx)

        result = s.groupby(s.index).first()
        assert len(result) == 3

    def test_subsets_multiindex_dtype(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies attribute-style subset access of a frame with multi-level columns containing NaN and
        # float labels gives the same dtypes as selecting from the dtypes Series.
        # GH 20757
        data = [["x", 1]]
        columns = [("a", "b", np.nan), ("a", "c", 0.0)]
        df = DataFrame(data, columns=MultiIndex.from_tuples(columns))
        expected = df.dtypes.a.b
        result = df.a.b.dtypes
        tm.assert_series_equal(result, expected)

    def test_datetime_object_multiindex(self):
        # [FUNCTION DOC]
        # Verifies building a frame from a dictionary keyed by (integer, date object) tuples with the index orientation produces a two-level index
        # of integers and date objects with the right columns.
        data_dic = {
            (0, datetime.date(2018, 3, 3)): {"A": 1, "B": 10},
            (0, datetime.date(2018, 3, 4)): {"A": 2, "B": 11},
            (1, datetime.date(2018, 3, 3)): {"A": 3, "B": 12},
            (1, datetime.date(2018, 3, 4)): {"A": 4, "B": 13},
        }
        result = DataFrame.from_dict(data_dic, orient="index")
        data = {"A": [1, 2, 3, 4], "B": [10, 11, 12, 13]}
        index = [
            [0, 0, 1, 1],
            [
                datetime.date(2018, 3, 3),
                datetime.date(2018, 3, 4),
                datetime.date(2018, 3, 3),
                datetime.date(2018, 3, 4),
            ],
        ]
        expected = DataFrame(data=data, index=index)

        tm.assert_frame_equal(result, expected)

    def test_multiindex_with_na(self):
        # [FUNCTION DOC]
        # Verifies that setting a value by a new two-part label on a frame indexed by two columns, one containing NaN, appends a row and
        # fills other columns with NaN.
        df = DataFrame(
            [
                ["A", np.nan, 1.23, 4.56],
                ["A", "G", 1.23, 4.56],
                ["A", "D", 9.87, 10.54],
            ],
            columns=["pivot_0", "pivot_1", "col_1", "col_2"],
        ).set_index(["pivot_0", "pivot_1"])

        df.at[("A", "F"), "col_2"] = 0.0

        expected = DataFrame(
            [
                ["A", np.nan, 1.23, 4.56],
                ["A", "G", 1.23, 4.56],
                ["A", "D", 9.87, 10.54],
                ["A", "F", np.nan, 0.0],
            ],
            columns=["pivot_0", "pivot_1", "col_1", "col_2"],
        ).set_index(["pivot_0", "pivot_1"])

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("na", [None, np.nan])
    def test_multiindex_insert_level_with_na(self, na):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over None and NaN as the top-level key. Verifies adding a column with that missing
        # key and a second-level label, then selecting by the missing key, returns the added column.
        # GH 59003
        df = DataFrame([0], columns=[["A"], ["B"]])
        df[na, "B"] = 1
        tm.assert_frame_equal(df[na], DataFrame([1], columns=["B"]))

    def test_multiindex_dt_with_nan(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies that after resetting the index of a frame with datetime second-level columns and NaN data,
        # selecting the first column gives the Series of the original index labels, named by a tuple of the old index name and NaN.
        # GH#60388
        df = DataFrame(
            [
                [1, np.nan, 5, np.nan],
                [2, np.nan, 6, np.nan],
                [np.nan, 3, np.nan, 7],
                [np.nan, 4, np.nan, 8],
            ],
            index=Series(["a", "b", "c", "d"], dtype=object, name="sub"),
            columns=MultiIndex.from_product(
                [
                    ["value1", "value2"],
                    [datetime.datetime(2024, 11, 1), datetime.datetime(2024, 11, 2)],
                ],
                names=[None, "Date"],
            ),
        )
        df = df.reset_index()
        result = df[df.columns[0]]
        expected = Series(["a", "b", "c", "d"], name=("sub", np.nan))
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
    def test_multiindex_with_pyarrow_categorical(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Filters the block-manager deprecation warning; skips without pyarrow. Verifies a frame whose key columns are an
        # Arrow dictionary-typed string and an Arrow float, set as a two-level index, equals a column-less frame with the equivalent plain index when index
        # and column type checks are relaxed.
        # GH#53051
        pa = pytest.importorskip("pyarrow")

        df = DataFrame(
            {"string_column": ["A", "B", "C"], "number_column": [1, 2, 3]}
        ).astype(
            {
                "string_column": ArrowDtype(pa.dictionary(pa.int32(), pa.string())),
                "number_column": "float[pyarrow]",
            }
        )

        df = df.set_index(["string_column", "number_column"])

        df_expected = DataFrame(
            index=MultiIndex.from_arrays(
                [["A", "B", "C"], [1, 2, 3]], names=["string_column", "number_column"]
            )
        )
        tm.assert_frame_equal(
            df,
            df_expected,
            check_index_type=False,
            check_column_type=False,
        )


class TestSorted:
    # [CLASS DOC]
    # Collection of tests about sorting multi-level indexes (docstring: everything you wanted to test about sorting).
    """everything you wanted to test about sorting"""

    def test_sort_non_lexsorted(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Builds an index whose codes are neither lexicographically sorted nor monotonic; verifies it is not monotonic,
        # that sorting the index makes it monotonic, and that a label slice on both levels of the sorted frame returns the expected four rows.
        # degenerate case where we sort but don't
        # have a satisfying result :<
        # GH 15797
        idx = MultiIndex(
            [["A", "B", "C"], ["c", "b", "a"]], [[0, 1, 2, 0, 1, 2], [0, 2, 1, 1, 0, 2]]
        )

        df = DataFrame({"col": range(len(idx))}, index=idx, dtype="int64")
        assert df.index.is_monotonic_increasing is False

        sorted = df.sort_index()
        assert sorted.index.is_monotonic_increasing is True

        expected = DataFrame(
            {"col": [1, 4, 5, 2]},
            index=MultiIndex.from_tuples(
                [("B", "a"), ("B", "c"), ("C", "a"), ("C", "b")]
            ),
            dtype="int64",
        )
        result = sorted.loc[pd.IndexSlice["B":"C", "a":"c"], :]
        tm.assert_frame_equal(result, expected)
