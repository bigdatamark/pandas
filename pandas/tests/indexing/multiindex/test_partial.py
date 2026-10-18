# [MODULE DOC]
# Tests for partial indexing (supplying only leading levels of the key) of MultiIndex
# frames: getitem, xs, slicing and partial setitem. Uses shared MultiIndex fixtures.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    date_range,
)
import pandas._testing as tm


class TestMultiIndexPartial:
    # [CLASS DOC]
    # Test collection for partial-key indexing behaviors on MultiIndex.
    def test_getitem_partial_int(self):
        # GH 12416
        # with single item
        # [FUNCTION DOC]
        # GH 12416. On a frame with two-level columns of integer and string levels: getitem with a
        # scalar first-level label returns the sub-frame with the level dropped; a list gives the
        # selected frame with levels kept; missing scalar raises a key error and a missing list raises
        # a key error saying the label is not in the index.
        l1 = [10, 20]
        l2 = ["a", "b"]
        df = DataFrame(index=range(2), columns=MultiIndex.from_product([l1, l2]))
        expected = DataFrame(index=range(2), columns=l2)
        result = df[20]
        tm.assert_frame_equal(result, expected)

        # with list
        expected = DataFrame(
            index=range(2), columns=MultiIndex.from_product([l1[1:], l2])
        )
        result = df[[20]]
        tm.assert_frame_equal(result, expected)

        # missing item:
        with pytest.raises(KeyError, match="1"):
            df[1]
        with pytest.raises(KeyError, match=r"'\[1\] not in index'"):
            df[[1]]

    def test_series_slice_partial(self):
        # [FUNCTION DOC]
        # KNOWN ISSUE: empty placeholder test; the body does nothing and asserts nothing.
        pass

    def test_xs_partial(
        self,
        multiindex_dataframe_random_data,
        multiindex_year_month_day_dataframe_random_data,
    ):
        # [FUNCTION DOC]
        # Cross-section by partial key equals loc by the same partial key (and transposed getitem) for
        # the standard frame and year/month/day frame; also an example from issue 1796 where a
        # two-label cross-section equals loc with the same two labels.
        frame = multiindex_dataframe_random_data
        ymd = multiindex_year_month_day_dataframe_random_data
        result = frame.xs("foo")
        result2 = frame.loc["foo"]
        expected = frame.T["foo"].T
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result, result2)

        result = ymd.xs((2000, 4))
        expected = ymd.loc[2000, 4]
        tm.assert_frame_equal(result, expected)

        # ex from #1796
        index = MultiIndex(
            levels=[["foo", "bar"], ["one", "two"], [-1, 1]],
            codes=[
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
        )
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            index=index,
            columns=list("abcd"),
        )

        result = df.xs(("foo", "one"))
        expected = df.loc["foo", "one"]
        tm.assert_frame_equal(result, expected)

    def test_getitem_partial(self, multiindex_year_month_day_dataframe_random_data):
        # [FUNCTION DOC]
        # On a transposed frame (MultiIndex columns), getitem with a two-label partial key returns the
        # columns with matching codes for the second level and drops the first two levels.
        ymd = multiindex_year_month_day_dataframe_random_data
        ymd = ymd.T
        result = ymd[2000, 2]

        expected = ymd.reindex(columns=ymd.columns[ymd.columns.codes[1] == 1])
        expected.columns = expected.columns.droplevel(0).droplevel(0)
        tm.assert_frame_equal(result, expected)

    def test_fancy_slice_partial(
        self,
        multiindex_dataframe_random_data,
        multiindex_year_month_day_dataframe_random_data,
    ):
        # [FUNCTION DOC]
        # Label slicing over the first level and over (year, month) tuple bounds returns the same rows
        # as boolean-mask or positional selection on the underlying codes.
        frame = multiindex_dataframe_random_data
        result = frame.loc["bar":"baz"]
        expected = frame[3:7]
        tm.assert_frame_equal(result, expected)

        ymd = multiindex_year_month_day_dataframe_random_data
        result = ymd.loc[(2000, 2) : (2000, 4)]
        lev = ymd.index.codes[1]
        expected = ymd[(lev >= 1) & (lev <= 3)]
        tm.assert_frame_equal(result, expected)

    def test_getitem_partial_column_select(self):
        # [FUNCTION DOC]
        # Row partial tuple with all columns and with a reordered integer column list equals indexing
        # by the tuple alone and then selecting columns; a partial tuple with a missing label raises a
        # key error showing it.
        idx = MultiIndex(
            codes=[[0, 0, 0], [0, 1, 1], [1, 0, 1]],
            levels=[["a", "b"], ["x", "y"], ["p", "q"]],
        )
        df = DataFrame(np.random.default_rng(2).random((3, 2)), index=idx)

        result = df.loc[("a", "y"), :]
        expected = df.loc[("a", "y")]
        tm.assert_frame_equal(result, expected)

        result = df.loc[("a", "y"), [1, 0]]
        expected = df.loc[("a", "y")][[1, 0]]
        tm.assert_frame_equal(result, expected)

        with pytest.raises(KeyError, match=r"\('a', 'foo'\)"):
            df.loc[("a", "foo"), :]

    def test_partial_set(
        self,
        multiindex_year_month_day_dataframe_random_data,
    ):
        # GH #397
        # [FUNCTION DOC]
        # GH 397. Setting values with partial keys (scalar year-month, single year) writes the same
        # positions as positional slice assignment on a copy. Chained assignment through a series
        # view raises the chained-assignment error context and doesn't change the frame; setting a
        # column through a full tuple works.
        ymd = multiindex_year_month_day_dataframe_random_data
        df = ymd.copy()
        exp = ymd.copy()
        df.loc[2000, 4] = 0
        exp.iloc[65:85] = 0
        tm.assert_frame_equal(df, exp)

        with tm.raises_chained_assignment_error():
            df["A"].loc[2000, 4] = 1
        df.loc[(2000, 4), "A"] = 1
        exp.iloc[65:85, 0] = 1
        tm.assert_frame_equal(df, exp)

        df.loc[2000] = 5
        exp.iloc[:100] = 5
        tm.assert_frame_equal(df, exp)

        # this works...for now
        with tm.raises_chained_assignment_error():
            df["A"].iloc[14] = 5
        assert df["A"].iloc[14] == exp["A"].iloc[14]

    @pytest.mark.parametrize("dtype", [int, float])
    def test_getitem_intkey_leading_level(
        self, multiindex_year_month_day_dataframe_random_data, dtype
    ):
        # GH#33355 dont fall-back to positional when leading level is int
        # [FUNCTION DOC]
        # GH 33355. Parametrized over leading-level dtype (int, float). When the leading level is
        # numeric, a missing integer key must not fall back to positional indexing: both the level and
        # index report no positional fallback, and getitem with an absent key raises a key error.
        ymd = multiindex_year_month_day_dataframe_random_data
        levels = ymd.index.levels
        ymd.index = ymd.index.set_levels([levels[0].astype(dtype), *levels[1:]])
        ser = ymd["A"]
        mi = ser.index
        assert isinstance(mi, MultiIndex)
        if dtype is int:
            assert mi.levels[0].dtype == np.dtype(int)
        else:
            assert mi.levels[0].dtype == np.float64

        assert 14 not in mi.levels[0]
        assert not mi.levels[0]._should_fallback_to_positional
        assert not mi._should_fallback_to_positional

        with pytest.raises(KeyError, match="14"):
            ser[14]

    # ---------------------------------------------------------------------

    def test_setitem_multiple_partial(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Setting zero through a list of first-level labels or a label slice, on a frame and a
        # series, equals setting each label separately.
        frame = multiindex_dataframe_random_data
        expected = frame.copy()
        result = frame.copy()
        result.loc[["foo", "bar"]] = 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        tm.assert_frame_equal(result, expected)

        expected = frame.copy()
        result = frame.copy()
        result.loc["foo":"bar"] = 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        tm.assert_frame_equal(result, expected)

        expected = frame["A"].copy()
        result = frame["A"].copy()
        result.loc[["foo", "bar"]] = 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        tm.assert_series_equal(result, expected)

        expected = frame["A"].copy()
        result = frame["A"].copy()
        result.loc["foo":"bar"] = 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer, exp_idx, exp_values",
        [
            (
                slice("2019-2", None),
                DatetimeIndex(["2019-02-01"], dtype="M8[ns]"),
                [2, 3],
            ),
            (
                slice(None, "2019-2"),
                date_range("2019", periods=2, freq="MS", unit="ns"),
                [0, 1, 2, 3],
            ),
        ],
    )
    def test_partial_getitem_loc_datetime(self, indexer, exp_idx, exp_values):
        # GH: 25165
        # [FUNCTION DOC]
        # GH 25165. Parametrized over a date-string slice (open start, or open end) with the expected
        # index and values. Getitem, loc, loc with axis argument, and loc with trailing all-columns all
        # give the expected partial range; after swapping levels and sorting, a three-part
        # indexer on the second axis position gives the swapped expected frame.
        date_idx = date_range("2019", periods=2, freq="MS", unit="ns")
        df = DataFrame(
            list(range(4)),
            index=MultiIndex.from_product([date_idx, [0, 1]], names=["x", "y"]),
        )
        expected = DataFrame(
            exp_values,
            index=MultiIndex.from_product([exp_idx, [0, 1]], names=["x", "y"]),
        )
        result = df[indexer]
        tm.assert_frame_equal(result, expected)
        result = df.loc[indexer]
        tm.assert_frame_equal(result, expected)

        result = df.loc(axis=0)[indexer]
        tm.assert_frame_equal(result, expected)

        result = df.loc[indexer, :]
        tm.assert_frame_equal(result, expected)

        df2 = df.swaplevel(0, 1).sort_index()
        expected = expected.swaplevel(0, 1).sort_index()

        result = df2.loc[:, indexer, :]
        tm.assert_frame_equal(result, expected)


def test_loc_getitem_partial_both_axis():
    # gh-12660
    # [FUNCTION DOC]
    # GH 12660. A row label and a column label on frames with named two-level axes return the
    # sub-block with those first levels dropped on both axes.
    iterables = [["a", "b"], [2, 1]]
    columns = MultiIndex.from_product(iterables, names=["col1", "col2"])
    rows = MultiIndex.from_product(iterables, names=["row1", "row2"])
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)), index=rows, columns=columns
    )
    expected = df.iloc[:2, 2:].droplevel("row1").droplevel("col1", axis=1)
    result = df.loc["a", "b"]
    tm.assert_frame_equal(result, expected)
