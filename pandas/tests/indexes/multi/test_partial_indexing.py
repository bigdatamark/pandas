# [MODULE DOC]
# Tests for partial string indexing on a MultiIndex whose level includes
# datetimes: single-level partial strings, get_loc with partial timestamps,
# slices with IndexSlice and resolution rules.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    IndexSlice,
    MultiIndex,
    date_range,
)
import pandas._testing as tm


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture for the tests in this module.
    # Returns a frame with one integer column holding the numbers zero to fourteen,
    # indexed by the product of twelve-hourly timestamps from 2016-01-01 to 2016-01-03
    # and the labels "a", "b", "c" (15 rows).
    #                        c1
    # 2016-01-01 00:00:00 a   0
    #                     b   1
    #                     c   2
    # 2016-01-01 12:00:00 a   3
    #                     b   4
    #                     c   5
    # 2016-01-02 00:00:00 a   6
    #                     b   7
    #                     c   8
    # 2016-01-02 12:00:00 a   9
    #                     b  10
    #                     c  11
    # 2016-01-03 00:00:00 a  12
    #                     b  13
    #                     c  14
    dr = date_range("2016-01-01", "2016-01-03", freq="12h")
    abc = ["a", "b", "c"]
    mi = MultiIndex.from_product([dr, abc])
    frame = DataFrame({"c1": range(15)}, index=mi)
    return frame


def test_partial_string_matching_single_index(df):
    # [FUNCTION DOC]
    # Swapping levels in several ways and sorting, then selecting the label "a" and
    # a partial date string, gives the same rows as selecting the corresponding part of
    # the original frame.
    # partial string matching on a single index
    for df_swap in [df.swaplevel(), df.swaplevel(0), df.swaplevel(0, 1)]:
        df_swap = df_swap.sort_index()
        just_a = df_swap.loc["a"]
        result = just_a.loc["2016-01-01"]
        expected = df.loc[IndexSlice[:, "a"], :].iloc[0:2]
        expected.index = expected.index.droplevel(1)
        tm.assert_frame_equal(result, expected)


def test_get_loc_partial_timestamp_multiindex(df):
    # [FUNCTION DOC]
    # get_loc with (partial date string, label) tuples returns boolean masks marking the
    # matching rows, for day, month and year strings, also on a non-monotonic
    # (reordered)
    # index.
    mi = df.index
    key = ("2016-01-01", "a")
    loc = mi.get_loc(key)

    expected = np.zeros(len(mi), dtype=bool)
    expected[[0, 3]] = True
    tm.assert_numpy_array_equal(loc, expected)

    key2 = ("2016-01-02", "a")
    loc2 = mi.get_loc(key2)
    expected2 = np.zeros(len(mi), dtype=bool)
    expected2[[6, 9]] = True
    tm.assert_numpy_array_equal(loc2, expected2)

    key3 = ("2016-01", "a")
    loc3 = mi.get_loc(key3)
    expected3 = np.zeros(len(mi), dtype=bool)
    expected3[mi.get_level_values(1).get_loc("a")] = True
    tm.assert_numpy_array_equal(loc3, expected3)

    key4 = ("2016", "a")
    loc4 = mi.get_loc(key4)
    expected4 = expected3
    tm.assert_numpy_array_equal(loc4, expected4)

    # non-monotonic
    taker = np.arange(len(mi), dtype=np.intp)
    taker[::2] = taker[::-2]
    mi2 = mi.take(taker)
    loc5 = mi2.get_loc(key)
    expected5 = np.zeros(len(mi2), dtype=bool)
    expected5[[3, 14]] = True
    tm.assert_numpy_array_equal(loc5, expected5)


def test_partial_string_timestamp_multiindex(df):
    # [FUNCTION DOC]
    # Many partial-string selections on the datetime level (slice, year, day, hour,
    # tuple with label, secondary level after level swap). Verifies expected row subsets
    # and
    # that hour resolution drops the level, and that a date key on the wrong level
    # raises a
    # key error (GH10331).
    # KNOWN ISSUE: the expected value in the tuple-selector case is assigned twice; the first assignment (without dropping a level) is dead code.
    # GH10331
    df_swap = df.swaplevel(0, 1).sort_index()
    SLC = IndexSlice

    # indexing with IndexSlice
    result = df.loc[SLC["2016-01-01":"2016-02-01", :], :]
    expected = df
    tm.assert_frame_equal(result, expected)

    # match on secondary index
    result = df_swap.loc[SLC[:, "2016-01-01":"2016-01-01"], :]
    expected = df_swap.iloc[[0, 1, 5, 6, 10, 11]]
    tm.assert_frame_equal(result, expected)

    # partial string match on year only
    result = df.loc["2016"]
    expected = df
    tm.assert_frame_equal(result, expected)

    # partial string match on date
    result = df.loc["2016-01-01"]
    expected = df.iloc[0:6]
    tm.assert_frame_equal(result, expected)

    # partial string match on date and hour, from middle
    result = df.loc["2016-01-02 12"]
    # hourly resolution, same as index.levels[0], so we are _not_ slicing on
    #  that level, so that level gets dropped
    expected = df.iloc[9:12].droplevel(0)
    tm.assert_frame_equal(result, expected)

    # partial string match on secondary index
    result = df_swap.loc[SLC[:, "2016-01-02"], :]
    expected = df_swap.iloc[[2, 3, 7, 8, 12, 13]]
    tm.assert_frame_equal(result, expected)

    # tuple selector with partial string match on date
    # "2016-01-01" has daily resolution, so _is_ a slice on the first level.
    result = df.loc[("2016-01-01", "a"), :]
    expected = df.iloc[[0, 3]]
    expected = df.iloc[[0, 3]].droplevel(1)
    tm.assert_frame_equal(result, expected)

    # Slicing date on first level should break (of course) bc the DTI is the
    #  second level on df_swap
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df_swap.loc["2016-01-01"]


def test_partial_string_timestamp_multiindex_str_key_raises(df):
    # [FUNCTION DOC]
    # Selecting a column-style date string directly on the frame raises a key error
    # (deliberately not supported for MultiIndex).
    # Even though this syntax works on a single index, this is somewhat
    # ambiguous and we don't want to extend this behavior forward to work
    # in multi-indexes. This would amount to selecting a scalar from a
    # column.
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df["2016-01-01"]


def test_partial_string_timestamp_multiindex_daily_resolution(df):
    # [FUNCTION DOC]
    # Slices the frame by a March 2013 month range and compares with rows from a
    # position range (GH12685).
    # KNOWN ISSUE: the frame fixture only covers three days of January 2016, so the selection and the expected position range are both empty and the test is effectively vacuous.
    # GH12685 (partial string with daily resolution or below)
    result = df.loc[IndexSlice["2013-03":"2013-03", :], :]
    expected = df.iloc[118:180]
    tm.assert_frame_equal(result, expected)
