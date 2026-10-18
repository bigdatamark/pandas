# [MODULE DOC]
# Tests for unique, nunique and is_unique on DatetimeIndex.
from datetime import (
    datetime,
    timedelta,
)

from pandas import (
    DatetimeIndex,
    NaT,
    Timestamp,
)
import pandas._testing as tm


def test_unique(tz_naive_fixture):
    # [FUNCTION DOC]
    # Uses tz_naive_fixture. Verifies unique of two identical stamps returns a one-element index equal
    # to the first, with a consistent underlying scalar.
    idx = DatetimeIndex(["2017"] * 2, tz=tz_naive_fixture)
    expected = idx[:1]

    result = idx.unique()
    tm.assert_index_equal(result, expected)
    # GH#21737
    # Ensure the underlying data is consistent
    assert result[0] == expected[0]


def test_index_unique(rand_series_with_duplicate_datetimeindex):
    # [FUNCTION DOC]
    # Uses a fixture providing a random Series with duplicated datetime index labels. Verifies unique
    # gives the four distinct dates in the original dtype, nunique is four, the result type is
    # DatetimeIndex, and after localizing to US/Eastern and naming the index, unique keeps timezone and
    # name.
    dups = rand_series_with_duplicate_datetimeindex
    index = dups.index

    uniques = index.unique()
    expected = DatetimeIndex(
        [
            datetime(2000, 1, 2),
            datetime(2000, 1, 3),
            datetime(2000, 1, 4),
            datetime(2000, 1, 5),
        ],
        dtype=index.dtype,
    )
    assert uniques.dtype == index.dtype  # sanity
    tm.assert_index_equal(uniques, expected)
    assert index.nunique() == 4

    # GH#2563
    assert isinstance(uniques, DatetimeIndex)

    dups_local = index.tz_localize("US/Eastern")
    dups_local.name = "foo"
    result = dups_local.unique()
    expected = DatetimeIndex(expected, name="foo")
    expected = expected.tz_localize("US/Eastern")
    assert result.tz is not None
    assert result.name == "foo"
    tm.assert_index_equal(result, expected)


def test_index_unique2():
    # [FUNCTION DOC]
    # Verifies unique on integer-based datetime values with NaT repeated three times: unique equals
    # the distinct values and nunique is 20 by default (NaT dropped) and 21 when dropna is false.
    # NaT, note this is excluded
    arr = [1370745748 + t for t in range(20)] + [NaT._value]
    idx = DatetimeIndex(arr * 3)
    tm.assert_index_equal(idx.unique(), DatetimeIndex(arr))
    assert idx.nunique() == 20
    assert idx.nunique(dropna=False) == 21


def test_index_unique3():
    # [FUNCTION DOC]
    # Same as the previous test but constructed from Timestamps with a NaT.
    arr = [
        Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)
    ] + [NaT]
    idx = DatetimeIndex(arr * 3)
    tm.assert_index_equal(idx.unique(), DatetimeIndex(arr))
    assert idx.nunique() == 20
    assert idx.nunique(dropna=False) == 21


def test_is_unique_monotonic(rand_series_with_duplicate_datetimeindex):
    # [FUNCTION DOC]
    # Uses the duplicated-datetimeindex Series fixture; verifies the index is not unique.
    index = rand_series_with_duplicate_datetimeindex.index
    assert not index.is_unique
