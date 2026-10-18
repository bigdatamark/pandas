# [MODULE DOC]
# Tests for sort_values, argmin and argmax on DatetimeIndex, TimedeltaIndex and PeriodIndex,
# including the effect on freq for indexes that are (or are not) range-like, NaT placement,
# return_indexer, and stability relative to plain Index. Module-level helpers check the expected
# freq; the test class has helper methods shared across index types.
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    PeriodIndex,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


def check_freq_ascending(ordered, orig, ascending):
    # [FUNCTION DOC]
    # HELPER: for sort_values tests where the original index is range-like. Takes the sorted result
    # (ordered), the original index (orig) and a boolean ascending flag. For a PeriodIndex asserts the
    # freq equals the original's; for DatetimeIndex or TimedeltaIndex asserts the freq multiplier
    # equals the original's when ascending and its negative otherwise. Returns nothing.
    """
    Check the expected freq on a PeriodIndex/DatetimeIndex/TimedeltaIndex
    when the original index is generated (or generate-able) with
    period_range/date_range/timedelta_range.
    """
    if isinstance(ordered, PeriodIndex):
        assert ordered.freq == orig.freq
    elif isinstance(ordered, (DatetimeIndex, TimedeltaIndex)):
        if ascending:
            assert ordered.freq.n == orig.freq.n
        else:
            assert ordered.freq.n == -1 * orig.freq.n


def check_freq_nonmonotonic(ordered, orig):
    # [FUNCTION DOC]
    # HELPER: for sort_values tests where the original index is not range-like. For a PeriodIndex
    # asserts the freq equals the original's; for DatetimeIndex or TimedeltaIndex asserts the
    # freq is None. Returns nothing.
    """
    Check the expected freq on a PeriodIndex/DatetimeIndex/TimedeltaIndex
    when the original index is _not_ generated (or generate-able) with
    period_range/date_range//timedelta_range.
    """
    if isinstance(ordered, PeriodIndex):
        assert ordered.freq == orig.freq
    elif isinstance(ordered, (DatetimeIndex, TimedeltaIndex)):
        assert ordered.freq is None


class TestSortValues:
    # [CLASS DOC]
    # Groups sort_values/argmin/argmax tests for the datetime-like index types. Holds a fixture of
    # non-monotonic indexes and helper check methods (named check, not test) invoked by test methods.
    @pytest.fixture(params=[DatetimeIndex, TimedeltaIndex, PeriodIndex])
    def non_monotonic_idx(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over DatetimeIndex, TimedeltaIndex and PeriodIndex classes;
        # returns a three-element non-monotonic index of that kind where the first value is the largest,
        # the second the smallest and the third in the middle.
        if request.param is DatetimeIndex:
            return DatetimeIndex(["2000-01-04", "2000-01-01", "2000-01-02"])
        elif request.param is PeriodIndex:
            dti = DatetimeIndex(["2000-01-04", "2000-01-01", "2000-01-02"])
            return dti.to_period("D")
        else:
            return TimedeltaIndex(
                ["1 day 00:00:05", "1 day 00:00:01", "1 day 00:00:02"]
            )

    def test_argmin_argmax(self, non_monotonic_idx):
        # [FUNCTION DOC]
        # Verifies argmin returns position one and argmax returns position zero for the non-monotonic
        # index fixture.
        assert non_monotonic_idx.argmin() == 1
        assert non_monotonic_idx.argmax() == 0

    def test_sort_values(self, non_monotonic_idx):
        # [FUNCTION DOC]
        # Verifies sort_values ascending and descending produce monotonic results, and with
        # return_indexer the sorting positions are [1, 2, 0] ascending and [0, 2, 1] descending
        # (platform integers).
        idx = non_monotonic_idx
        ordered = idx.sort_values()
        assert ordered.is_monotonic_increasing
        ordered = idx.sort_values(ascending=False)
        assert ordered[::-1].is_monotonic_increasing

        ordered, dexer = idx.sort_values(return_indexer=True)
        assert ordered.is_monotonic_increasing
        tm.assert_numpy_array_equal(dexer, np.array([1, 2, 0], dtype=np.intp))

        ordered, dexer = idx.sort_values(return_indexer=True, ascending=False)
        assert ordered[::-1].is_monotonic_increasing
        tm.assert_numpy_array_equal(dexer, np.array([0, 2, 1], dtype=np.intp))

    def check_sort_values_with_freq(self, idx):
        # [FUNCTION DOC]
        # HELPER: shared checker for already-sorted range-like indexes. Verifies ascending sort returns
        # the same index and the freq check, descending sort returns the reversed index with negated freq,
        # and the same with return_indexer (positions 0,1,2 and 2,1,0).
        ordered = idx.sort_values()
        tm.assert_index_equal(ordered, idx)
        check_freq_ascending(ordered, idx, True)

        ordered = idx.sort_values(ascending=False)
        expected = idx[::-1]
        tm.assert_index_equal(ordered, expected)
        check_freq_ascending(ordered, idx, False)

        ordered, indexer = idx.sort_values(return_indexer=True)
        tm.assert_index_equal(ordered, idx)
        tm.assert_numpy_array_equal(indexer, np.array([0, 1, 2], dtype=np.intp))
        check_freq_ascending(ordered, idx, True)

        ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
        expected = idx[::-1]
        tm.assert_index_equal(ordered, expected)
        tm.assert_numpy_array_equal(indexer, np.array([2, 1, 0], dtype=np.intp))
        check_freq_ascending(ordered, idx, False)

    @pytest.mark.parametrize("freq", ["D", "h"])
    def test_sort_values_with_freq_timedeltaindex(self, freq):
        # [FUNCTION DOC]
        # Parametrized over daily and hourly frequency. Runs the shared range-like checker on a
        # three-element timedelta range named idx.
        # GH#10295
        idx = timedelta_range(start=f"1{freq}", periods=3, freq=freq).rename("idx")

        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize(
        "idx",
        [
            DatetimeIndex(
                ["2011-01-01", "2011-01-02", "2011-01-03"], freq="D", name="idx"
            ),
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
                freq="h",
                name="tzidx",
                tz="Asia/Tokyo",
            ),
        ],
    )
    def test_sort_values_with_freq_datetimeindex(self, idx):
        # [FUNCTION DOC]
        # Parametrized over a naive daily DatetimeIndex and a Tokyo-timezone hourly DatetimeIndex with
        # explicit frequencies. Runs the shared range-like checker.
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize("freq", ["D", "2D", "4D"])
    def test_sort_values_with_freq_periodindex(self, freq):
        # [FUNCTION DOC]
        # Parametrized over frequencies daily, two-day and four-day. Runs the shared range-like checker
        # on a three-element PeriodIndex.
        # here with_freq refers to being period_range-like
        idx = PeriodIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03"], freq=freq, name="idx"
        )
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize(
        "idx",
        [
            PeriodIndex(["2011", "2012", "2013"], name="pidx", freq="Y"),
            Index([2011, 2012, 2013], name="idx"),  # for compatibility check
        ],
    )
    def test_sort_values_with_freq_periodindex2(self, idx):
        # [FUNCTION DOC]
        # Parametrized over an annual PeriodIndex and a plain integer Index (compatibility check). Runs
        # the shared range-like checker.
        # here with_freq indicates this is period_range-like
        self.check_sort_values_with_freq(idx)

    def check_sort_values_without_freq(self, idx, expected):
        # [FUNCTION DOC]
        # HELPER: shared checker for unsorted, non-range indexes. Takes the index and the expected
        # ascending-sorted index (5 elements, possibly with duplicates and NaT at the start). Checks
        # sort_values with NaT first, default ordering (only when there is no NaN), descending, and
        # return_indexer variants, comparing indexers to fixed expected position arrays (ascending
        # [0, 4, 3, 1, 2], descending [2, 1, 3, 0, 4]) and calling the non-range freq check.
        ordered = idx.sort_values(na_position="first")
        tm.assert_index_equal(ordered, expected)
        check_freq_nonmonotonic(ordered, idx)

        if not idx.isna().any():
            ordered = idx.sort_values()
            tm.assert_index_equal(ordered, expected)
            check_freq_nonmonotonic(ordered, idx)

        ordered = idx.sort_values(ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])
        check_freq_nonmonotonic(ordered, idx)

        ordered, indexer = idx.sort_values(return_indexer=True, na_position="first")
        tm.assert_index_equal(ordered, expected)

        exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, exp)
        check_freq_nonmonotonic(ordered, idx)

        if not idx.isna().any():
            ordered, indexer = idx.sort_values(return_indexer=True)
            tm.assert_index_equal(ordered, expected)

            exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, exp)
            check_freq_nonmonotonic(ordered, idx)

        ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])

        exp = np.array([2, 1, 3, 0, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, exp)
        check_freq_nonmonotonic(ordered, idx)

    def test_sort_values_without_freq_timedeltaindex(self):
        # [FUNCTION DOC]
        # Runs the unsorted-index checker on a five-element TimedeltaIndex of hours with a duplicate.
        # GH#10295

        idx = TimedeltaIndex(
            ["1 hour", "3 hour", "5 hour", "2 hour ", "1 hour"], name="idx1"
        )
        expected = TimedeltaIndex(
            ["1 hour", "1 hour", "2 hour", "3 hour", "5 hour"], name="idx1"
        )
        self.check_sort_values_without_freq(idx, expected)

    @pytest.mark.parametrize(
        "index_dates,expected_dates",
        [
            (
                ["2011-01-01", "2011-01-03", "2011-01-05", "2011-01-02", "2011-01-01"],
                ["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
            (
                [NaT, "2011-01-03", "2011-01-05", "2011-01-02", NaT],
                [NaT, NaT, "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
        ],
    )
    def test_sort_values_without_freq_datetimeindex(
        self, index_dates, expected_dates, tz_naive_fixture
    ):
        # [FUNCTION DOC]
        # Parametrized over date lists (with and without NaT) and the tz_naive_fixture. Builds the
        # unsorted and expected-sorted DatetimeIndex with the timezone and runs the unsorted-index
        # checker.
        tz = tz_naive_fixture

        # without freq
        idx = DatetimeIndex(index_dates, tz=tz, name="idx")
        expected = DatetimeIndex(expected_dates, tz=tz, name="idx")

        self.check_sort_values_without_freq(idx, expected)

    @pytest.mark.parametrize(
        "idx,expected",
        [
            (
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-03",
                        "2011-01-05",
                        "2011-01-02",
                        "2011-01-01",
                    ],
                    freq="D",
                    name="idx1",
                ),
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-01",
                        "2011-01-02",
                        "2011-01-03",
                        "2011-01-05",
                    ],
                    freq="D",
                    name="idx1",
                ),
            ),
            (
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-03",
                        "2011-01-05",
                        "2011-01-02",
                        "2011-01-01",
                    ],
                    freq="D",
                    name="idx2",
                ),
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-01",
                        "2011-01-02",
                        "2011-01-03",
                        "2011-01-05",
                    ],
                    freq="D",
                    name="idx2",
                ),
            ),
            (
                PeriodIndex(
                    [NaT, "2011-01-03", "2011-01-05", "2011-01-02", NaT],
                    freq="D",
                    name="idx3",
                ),
                PeriodIndex(
                    [NaT, NaT, "2011-01-02", "2011-01-03", "2011-01-05"],
                    freq="D",
                    name="idx3",
                ),
            ),
            (
                PeriodIndex(
                    ["2011", "2013", "2015", "2012", "2011"], name="pidx", freq="Y"
                ),
                PeriodIndex(
                    ["2011", "2011", "2012", "2013", "2015"], name="pidx", freq="Y"
                ),
            ),
            (
                # For compatibility check
                Index([2011, 2013, 2015, 2012, 2011], name="idx"),
                Index([2011, 2011, 2012, 2013, 2015], name="idx"),
            ),
        ],
    )
    def test_sort_values_without_freq_periodindex(self, idx, expected):
        # [FUNCTION DOC]
        # Parametrized over five (index, expected sorted index) pairs: daily PeriodIndexes (with a
        # duplicate, and with NaT), an annual PeriodIndex, and a plain integer Index. Runs the
        # unsorted-index checker.
        # here without_freq means not generateable by period_range
        self.check_sort_values_without_freq(idx, expected)

    def test_sort_values_without_freq_periodindex_nat(self):
        # [FUNCTION DOC]
        # Verifies sorting a PeriodIndex with a missing value: NaT first for ascending with
        # na_position first, and the reverse ordering for descending; freq is retained.
        # doesn't quite fit into check_sort_values_without_freq
        idx = PeriodIndex(["2011", "2013", "NaT", "2011"], name="pidx", freq="D")
        expected = PeriodIndex(["NaT", "2011", "2011", "2013"], name="pidx", freq="D")

        ordered = idx.sort_values(na_position="first")
        tm.assert_index_equal(ordered, expected)
        check_freq_nonmonotonic(ordered, idx)

        ordered = idx.sort_values(ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])
        check_freq_nonmonotonic(ordered, idx)


def test_order_stability_compat():
    # [FUNCTION DOC]
    # Regression test: descending sort_values with return_indexer on a PeriodIndex and on an
    # equivalent integer Index give identical indexers (stable sort for ties).
    # GH#35922. sort_values is stable both for normal and datetime-like Index
    pidx = PeriodIndex(["2011", "2013", "2015", "2012", "2011"], name="pidx", freq="Y")
    iidx = Index([2011, 2013, 2015, 2012, 2011], name="idx")
    ordered1, indexer1 = pidx.sort_values(return_indexer=True, ascending=False)
    ordered2, indexer2 = iidx.sort_values(return_indexer=True, ascending=False)
    tm.assert_numpy_array_equal(indexer1, indexer2)
