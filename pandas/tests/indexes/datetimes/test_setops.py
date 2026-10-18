# [MODULE DOC]
# Tests for set operations (union, intersection, difference, symmetric
# difference) on DatetimeIndex: ordering via the sort argument, frequency
# retention or inference, name handling, timezone and unit mixes, daylight saving
# edge cases, and business/custom-business ranges. Defines module-level START
# and END datetimes (2009-01-01 and 2010-01-01).
# Uses the sort and tz_aware_fixture fixtures from the shared conftest.

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2_6
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timestamp,
    bdate_range,
    date_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    BMonthEnd,
    Minute,
    MonthEnd,
)

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestDatetimeIndexSetOps:
    # [CLASS DOC]
    # Groups general set-operation tests. Has a class attribute listing timezones
    # (none, UTC, Tokyo, US/Eastern, two dateutil zones) used as a parametrization
    # source by several methods.
    tz = [
        None,
        "UTC",
        "Asia/Tokyo",
        "US/Eastern",
        "dateutil/Asia/Singapore",
        "dateutil/US/Pacific",
    ]

    # TODO: moved from test_datetimelike; dedup with version below
    def test_union2(self, sort):
        # [FUNCTION DOC]
        # Note: moved from another test module, and the in-source TODO says to
        # de-duplicate it with the later version. Parametrized over sort. Verifies the
        # union of two adjacent halves of a daily range equals the full range.
        everything = date_range("2020-01-01", periods=10)
        first = everything[:5]
        second = everything[5:]
        union = first.union(second, sort=sort)
        tm.assert_index_equal(union, everything)

    @pytest.mark.parametrize("box", [np.array, Series, list])
    def test_union3(self, sort, box):
        # [FUNCTION DOC]
        # Parametrized over sort and container type (array, Series, list). Verifies
        # (GH 10149) that a union with the second half given as list-like equals the union with an Index.
        everything = date_range("2020-01-01", periods=10)
        first = everything[:5]
        second = everything[5:]

        # GH 10149 support listlike inputs other than Index objects
        expected = first.union(second, sort=sort)
        case = box(second.values)
        result = first.union(case, sort=sort)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", tz)
    def test_union(self, tz, sort):
        # [FUNCTION DOC]
        # Parametrized over timezone and sort. For three overlap cases (disjoint,
        # overlapping, other empty) verifies the union equals the full sorted range;
        # and for the reverse order, sorted when sort is none, otherwise concatenation
        # order (other first, then the original).
        rng1 = date_range("1/1/2000", freq="D", periods=5, tz=tz, unit="ns")
        other1 = date_range("1/6/2000", freq="D", periods=5, tz=tz, unit="ns")
        expected1 = date_range("1/1/2000", freq="D", periods=10, tz=tz, unit="ns")
        expected1_notsorted = DatetimeIndex(list(other1) + list(rng1))

        rng2 = date_range("1/1/2000", freq="D", periods=5, tz=tz, unit="ns")
        other2 = date_range("1/4/2000", freq="D", periods=5, tz=tz, unit="ns")
        expected2 = date_range("1/1/2000", freq="D", periods=8, tz=tz, unit="ns")
        expected2_notsorted = DatetimeIndex(list(other2) + list(rng2[:3]))

        rng3 = date_range("1/1/2000", freq="D", periods=5, tz=tz, unit="ns")
        other3 = DatetimeIndex([], tz=tz).as_unit("ns")
        expected3 = date_range("1/1/2000", freq="D", periods=5, tz=tz, unit="ns")
        expected3_notsorted = rng3

        for rng, other, exp, exp_notsorted in [
            (rng1, other1, expected1, expected1_notsorted),
            (rng2, other2, expected2, expected2_notsorted),
            (rng3, other3, expected3, expected3_notsorted),
        ]:
            result_union = rng.union(other, sort=sort)
            tm.assert_index_equal(result_union, exp)

            result_union = other.union(rng, sort=sort)
            if sort is None:
                tm.assert_index_equal(result_union, exp)
            else:
                tm.assert_index_equal(result_union, exp_notsorted)

    def test_union_coverage(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies the union of an unordered index with its
        # ordered inferred-frequency version equals the ordered one, and union with an
        # empty slice keeps the frequency.
        idx = DatetimeIndex(["2000-01-03", "2000-01-01", "2000-01-02"])
        ordered = DatetimeIndex(idx.sort_values(), freq="infer")
        result = ordered.union(idx, sort=sort)
        tm.assert_index_equal(result, ordered)

        result = ordered[:0].union(ordered, sort=sort)
        tm.assert_index_equal(result, ordered)
        assert result.freq == ordered.freq

    def test_union_bug_1730(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies (bug 1730) the union of three-hour and
        # four-hour ranges is the concatenation of the two minus the shared first value, sorted when sort is none.
        rng_a = date_range("1/1/2012", periods=4, freq="3h")
        rng_b = date_range("1/1/2012", periods=4, freq="4h")

        result = rng_a.union(rng_b, sort=sort)
        exp = list(rng_a) + list(rng_b[1:])
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        tm.assert_index_equal(result, exp)

    def test_union_bug_1745(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies (bug 1745) a union of unordered timestamps
        # gives the left value followed by right values, sorted when sort is none.
        left = DatetimeIndex(["2012-05-11 15:19:49.695000"])
        right = DatetimeIndex(
            [
                "2012-05-29 13:04:21.322000",
                "2012-05-11 15:27:24.873000",
                "2012-05-11 15:31:05.350000",
            ]
        )

        result = left.union(right, sort=sort)
        exp = DatetimeIndex(
            [
                "2012-05-11 15:19:49.695000",
                "2012-05-29 13:04:21.322000",
                "2012-05-11 15:27:24.873000",
                "2012-05-11 15:31:05.350000",
            ]
        )
        if sort is None:
            exp = exp.sort_values()
        tm.assert_index_equal(result, exp)

    def test_union_bug_4564(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies (bug 4564) the union of a daily range with a
        # copy shifted by fifteen minutes equals the concatenation, sorted when sort is none.
        from pandas import DateOffset

        left = date_range("2013-01-01", "2013-02-01")
        right = left + DateOffset(minutes=15)

        result = left.union(right, sort=sort)
        exp = list(left) + list(right)
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        tm.assert_index_equal(result, exp)

    def test_union_freq_both_none(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies (GH11086) the union of an index with no
        # frequency with itself stays equal with no frequency.
        # GH11086
        expected = bdate_range("20150101", periods=10)
        expected._data.freq = None

        result = expected.union(expected, sort=sort)
        tm.assert_index_equal(result, expected)
        assert result.freq is None

    def test_union_freq_infer(self):
        # [FUNCTION DOC]
        # Verifies the union of two frequency-less indexes whose union is regular
        # infers a daily frequency and equals the full range.
        # When taking the union of two DatetimeIndexes, we infer
        #  a freq even if the arguments don't have freq.  This matches
        #  TimedeltaIndex behavior.
        dti = date_range("2016-01-01", periods=5)
        left = dti[[0, 1, 3, 4]]
        right = dti[[2, 3, 1]]

        assert left.freq is None
        assert right.freq is None

        result = left.union(right)
        tm.assert_index_equal(result, dti)
        assert result.freq == "D"

    def test_union_dataframe_index(self):
        # [FUNCTION DOC]
        # Verifies that assembling a frame from two month-start series gives an index
        # equal to the full month range covering both.
        rng1 = date_range("1/1/1999", "1/1/2012", freq="MS")
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)

        rng2 = date_range("1/1/1980", "12/1/2001", freq="MS")
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        df = DataFrame({"s1": s1, "s2": s2})

        exp = date_range("1/1/1980", "1/1/2012", freq="MS")
        tm.assert_index_equal(df.index, exp)

    def test_union_with_DatetimeIndex(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Smoke test: the union of an integer index and a
        # datetime index must not raise in either order.
        i1 = Index(np.arange(0, 20, 2, dtype=np.int64))
        i2 = date_range(start="2012-01-03 00:00:00", periods=10, freq="D")
        # Works
        i1.union(i2, sort=sort)
        # Fails with "AttributeError: can't set attribute"
        i2.union(i1, sort=sort)

    def test_union_same_timezone_different_units(self):
        # [FUNCTION DOC]
        # Verifies (GH 55238) the union of same-zone indexes with millisecond and
        # microsecond units yields the finer (microsecond) unit.
        # GH 55238
        idx1 = date_range("2000-01-01", periods=3, tz="UTC").as_unit("ms")
        idx2 = date_range("2000-01-01", periods=3, tz="UTC").as_unit("us")
        result = idx1.union(idx2)
        expected = date_range("2000-01-01", periods=3, tz="UTC").as_unit("us")
        tm.assert_index_equal(result, expected)

    def test_union_same_nonzero_timezone_different_units(self):
        # [FUNCTION DOC]
        # Verifies (GH 60080) the union of same-zone, non-UTC indexes with different units
        # keeps the zone and gives the finer unit.
        # GH 60080 - fix timezone being changed to UTC when units differ
        # but timezone is the same
        tz = "UTC+05:00"
        idx1 = date_range("2000-01-01", periods=3, tz=tz).as_unit("us")
        idx2 = date_range("2000-01-01", periods=3, tz=tz).as_unit("ns")

        # Check pre-conditions
        assert idx1.tz == idx2.tz
        assert idx1.dtype != idx2.dtype  # Different units

        # Test union preserves timezone when units differ
        result = idx1.union(idx2)
        expected = date_range("2000-01-01", periods=3, tz=tz).as_unit("ns")
        tm.assert_index_equal(result, expected)

    def test_union_different_dates_same_timezone_different_units(self):
        # [FUNCTION DOC]
        # Verifies (GH 60080) the union of different-date, same non-UTC zone, same-unit
        # indexes gives all five consecutive dates in that zone.
        # GH 60080 - fix timezone being changed to UTC when units differ
        # but timezone is the same
        tz = "UTC+05:00"
        idx1 = date_range("2000-01-01", periods=3, tz=tz).as_unit("us")
        idx3 = date_range("2000-01-03", periods=3, tz=tz).as_unit("us")

        # Test with different dates to ensure it's not just returning one of the inputs
        result = idx1.union(idx3)
        expected = DatetimeIndex(
            ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05"],
            tz=tz,
        ).as_unit("us")
        tm.assert_index_equal(result, expected)

    def test_intersection_same_timezone_different_units(self):
        # [FUNCTION DOC]
        # Verifies (GH 60080) the intersection of same-zone indexes with different units
        # keeps the zone and takes the unit of the left operand, so it is not commutative
        # in the unit.
        # GH 60080 - fix timezone being changed to UTC when units differ
        # but timezone is the same
        tz = "UTC+05:00"
        idx1 = date_range("2000-01-01", periods=3, tz=tz).as_unit("us")
        idx2 = date_range("2000-01-01", periods=3, tz=tz).as_unit("ns")

        # Check pre-conditions
        assert idx1.tz == idx2.tz
        assert idx1.dtype != idx2.dtype  # Different units

        # Test intersection
        result = idx1.intersection(idx2)
        expected = date_range("2000-01-01", periods=3, tz=tz)
        tm.assert_index_equal(result, expected)

        # Intersection dtype is not commutative
        result2 = idx2.intersection(idx1)
        tm.assert_index_equal(result2, expected.as_unit("ns"))

    def test_symmetric_difference_same_timezone_different_units(self):
        # [FUNCTION DOC]
        # Verifies (GH 60080) the symmetric difference of same-zone indexes with
        # different units yields the first and last dates in the finer unit and keeps the zone.
        # GH 60080 - fix timezone being changed to UTC when units differ
        # but timezone is the same
        tz = "UTC+05:00"
        idx1 = date_range("2000-01-01", periods=3, tz=tz).as_unit("us")
        idx4 = date_range("2000-01-02", periods=3, tz=tz).as_unit("ns")

        # Check pre-conditions
        assert idx1.tz == idx4.tz
        assert idx1.dtype != idx4.dtype  # Different units

        # Test symmetric_difference
        result = idx1.symmetric_difference(idx4)
        expected = DatetimeIndex(["2000-01-01", "2000-01-04"], tz=tz).as_unit("ns")
        tm.assert_index_equal(result, expected)

    # TODO: moved from test_datetimelike; de-duplicate with version below
    def test_intersection2(self):
        # [FUNCTION DOC]
        # Note: moved from another test module, and the in-source TODO says to
        # de-duplicate it with the later version. Verifies the intersection of a range
        # with its tail equals the tail, with list-like targets (array, Series, list)
        # accepted (GH 10149), and that the intersection with a string index is an empty
        # object index.
        first = date_range("2020-01-01", periods=10)
        second = first[5:]
        intersect = first.intersection(second)
        tm.assert_index_equal(intersect, second)

        # GH 10149
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            result = first.intersection(case)
            tm.assert_index_equal(result, second)

        third = Index(["a", "b", "c"])
        result = first.intersection(third)
        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "tz", [None, "Asia/Tokyo", "US/Eastern", "dateutil/US/Pacific"]
    )
    def test_intersection(self, tz, sort):
        # [FUNCTION DOC]
        # Parametrized over timezone and sort. Verifies (GH 4690) the intersection of a
        # daily range with overlapping, differently-named and disjoint ranges: result
        # equality, name kept only when equal, and frequency; then the same for
        # non-monotonic indexes (sorted result when sort is none), including empty results
        # with (GH 7880) no frequency.
        # GH 4690 (with tz)
        base = date_range("6/1/2000", "6/30/2000", freq="D", name="idx", unit="ns")

        # if target has the same name, it is preserved
        rng2 = date_range("5/15/2000", "6/20/2000", freq="D", name="idx", unit="ns")
        expected2 = date_range("6/1/2000", "6/20/2000", freq="D", name="idx", unit="ns")

        # if target name is different, it will be reset
        rng3 = date_range("5/15/2000", "6/20/2000", freq="D", name="other", unit="ns")
        expected3 = date_range("6/1/2000", "6/20/2000", freq="D", name=None, unit="ns")

        rng4 = date_range("7/1/2000", "7/31/2000", freq="D", name="idx", unit="ns")
        expected4 = DatetimeIndex([], freq="D", name="idx", dtype="M8[ns]")

        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            result = base.intersection(rng)
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

        # non-monotonic
        base = DatetimeIndex(
            ["2011-01-05", "2011-01-04", "2011-01-02", "2011-01-03"], tz=tz, name="idx"
        ).as_unit("ns")

        rng2 = DatetimeIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"], tz=tz, name="idx"
        ).as_unit("ns")
        expected2 = DatetimeIndex(
            ["2011-01-04", "2011-01-02"], tz=tz, name="idx"
        ).as_unit("ns")

        rng3 = DatetimeIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"],
            tz=tz,
            name="other",
        ).as_unit("ns")
        expected3 = DatetimeIndex(
            ["2011-01-04", "2011-01-02"], tz=tz, name=None
        ).as_unit("ns")

        # GH 7880
        rng4 = date_range(
            "7/1/2000", "7/31/2000", freq="D", tz=tz, unit="ns", name="idx"
        )
        expected4 = DatetimeIndex([], tz=tz, name="idx").as_unit("ns")
        assert expected4.freq is None

        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            result = base.intersection(rng, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

    # parametrize over both anchored and non-anchored freqs, as they
    #  have different code paths
    @pytest.mark.parametrize("freq", ["min", "B"])
    def test_intersection_empty(self, tz_aware_fixture, freq):
        # [FUNCTION DOC]
        # Parametrized over frequency (non-anchored minute, anchored business) and
        # tz_aware_fixture. Verifies (GH2129) intersection with an empty index in either order
        # keeps frequency and is empty, and (GH#33604) intersection of non-overlapping
        # slices in both orders equals an empty slice, with the frequency checked only for
        # the business frequency.
        # empty same freq GH2129
        tz = tz_aware_fixture
        rng = date_range("6/1/2000", "6/15/2000", freq=freq, tz=tz)
        result = rng[0:0].intersection(rng)
        assert len(result) == 0
        assert result.freq == rng.freq

        result = rng.intersection(rng[0:0])
        assert len(result) == 0
        assert result.freq == rng.freq

        # no overlap GH#33604
        check_freq = freq != "min"  # We don't preserve freq on non-anchored offsets
        result = rng[:3].intersection(rng[-3:])
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            # We don't preserve freq on non-anchored offsets
            assert result.freq == rng.freq

        # swapped left and right
        result = rng[-3:].intersection(rng[:3])
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            # We don't preserve freq on non-anchored offsets
            assert result.freq == rng.freq

    def test_intersection_bug_1708(self):
        # [FUNCTION DOC]
        # Verifies (bug 1708) the intersection of an index with a copy shifted by one
        # hour is empty.
        from pandas import DateOffset

        index_1 = date_range("1/1/2012", periods=4, freq="12h")
        index_2 = index_1 + DateOffset(hours=1)

        result = index_1.intersection(index_2)
        assert len(result) == 0

    @pytest.mark.parametrize("tz", tz)
    def test_difference(self, tz, sort):
        # [FUNCTION DOC]
        # Parametrized over timezone and sort. Verifies difference with a disjoint range
        # (unchanged), an overlapping range (first three elements), and an empty
        # target (unchanged); sorted when sort is none, except with an empty target
        # (GH#24959 where no sort happens).
        rng_dates = ["1/2/2000", "1/3/2000", "1/1/2000", "1/4/2000", "1/5/2000"]

        rng1 = DatetimeIndex(rng_dates, tz=tz)
        other1 = date_range("1/6/2000", freq="D", periods=5, tz=tz)
        expected1 = DatetimeIndex(rng_dates, tz=tz)

        rng2 = DatetimeIndex(rng_dates, tz=tz)
        other2 = date_range("1/4/2000", freq="D", periods=5, tz=tz)
        expected2 = DatetimeIndex(rng_dates[:3], tz=tz)

        rng3 = DatetimeIndex(rng_dates, tz=tz)
        other3 = DatetimeIndex([], tz=tz)
        expected3 = DatetimeIndex(rng_dates, tz=tz)

        for rng, other, expected in [
            (rng1, other1, expected1),
            (rng2, other2, expected2),
            (rng3, other3, expected3),
        ]:
            result_diff = rng.difference(other, sort)
            if sort is None and len(other):
                # We dont sort (yet?) when empty GH#24959
                expected = expected.sort_values()
            tm.assert_index_equal(result_diff, expected)

    def test_difference_freq(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies (GH14323) the difference drops frequency when
        # removing interior elements, but preserves it when the removed part leaves a
        # contiguous subset of the original.
        # GH14323: difference of DatetimeIndex should not preserve frequency

        index = date_range("20160920", "20160925", freq="D", unit="ns")
        other = date_range("20160921", "20160924", freq="D", unit="ns")
        expected = DatetimeIndex(["20160920", "20160925"], dtype="M8[ns]", freq=None)
        idx_diff = index.difference(other, sort)
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

        # preserve frequency when the difference is a contiguous
        # subset of the original range
        other = date_range("20160922", "20160925", freq="D", unit="ns")
        idx_diff = index.difference(other, sort)
        expected = DatetimeIndex(["20160920", "20160921"], dtype="M8[ns]", freq="D")
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

    def test_datetimeindex_diff(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies the difference of 100 and 98 quarter-end
        # periods has length two.
        dti1 = date_range(freq="QE-JAN", start=datetime(1997, 12, 31), periods=100)
        dti2 = date_range(freq="QE-JAN", start=datetime(1997, 12, 31), periods=98)
        assert len(dti1.difference(dti2, sort)) == 2

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "US/Eastern"])
    def test_setops_preserve_freq(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone. Verifies name, timezone and frequency behavior of
        # set operations: adjacent and overlapping unions keep frequency; a gap union
        # loses it; intersection keeps the daily frequency string; and with an unnamed
        # frequency-less operand the name is dropped and the union and intersection regain the frequency.
        rng = date_range("1/1/2000", "1/1/2002", name="idx", tz=tz)

        result = rng[:50].union(rng[50:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].union(rng[30:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].union(rng[60:100])
        assert result.name == rng.name
        assert result.freq is None
        assert result.tz == rng.tz

        result = rng[:50].intersection(rng[25:75])
        assert result.name == rng.name
        assert result.freqstr == "D"
        assert result.tz == rng.tz

        nofreq = DatetimeIndex(list(rng[25:75]), name="other")
        result = rng[:50].union(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].intersection(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

    def test_intersection_non_tick_no_fastpath(self):
        # [FUNCTION DOC]
        # Verifies (GH#42104) the intersection of interleaved quarter-end elements,
        # which have no overlap, equals an empty slice (no fast path used for non-tick offsets).
        # GH#42104
        dti = DatetimeIndex(
            [
                "2018-12-31",
                "2019-03-31",
                "2019-06-30",
                "2019-09-30",
                "2019-12-31",
                "2020-03-31",
            ],
            freq="QE-DEC",
        )
        result = dti[::2].intersection(dti[1::2])
        expected = dti[:0]
        tm.assert_index_equal(result, expected)

    def test_dti_intersection(self):
        # [FUNCTION DOC]
        # Verifies the intersection of two reversed, UTC-aware slices keeps the UTC timezone.
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")

        left = rng[10:90][::-1]
        right = rng[20:80][::-1]

        assert left.tz == rng.tz
        result = left.intersection(right)
        assert result.tz == left.tz

    # Note: not difference, as there is no symmetry requirement there
    @pytest.mark.parametrize("setop", ["union", "intersection", "symmetric_difference"])
    def test_dti_setop_aware(self, setop):
        # [FUNCTION DOC]
        # Parametrized over set operation name (union, intersection, symmetric
        # difference; difference excluded as it has no symmetry requirement).
        # Verifies (GH#39328) that for non-overlapping aware indexes in different zones
        # the result equals the same operation on both converted to UTC, with zone UTC.
        # non-overlapping
        # GH#39328 as of 2.0 we cast these to UTC instead of object
        rng = date_range("2012-11-15 00:00:00", periods=6, freq="h", tz="US/Central")

        rng2 = date_range("2012-11-15 12:00:00", periods=6, freq="h", tz="US/Eastern")

        result = getattr(rng, setop)(rng2)

        left = rng.tz_convert("UTC")
        right = rng2.tz_convert("UTC")
        expected = getattr(left, setop)(right)
        tm.assert_index_equal(result, expected)
        assert result.tz == left.tz
        if len(result):
            assert result[0].tz is timezone.utc
            assert result[-1].tz is timezone.utc

    def test_dti_union_mixed(self):
        # [FUNCTION DOC]
        # Verifies (GH#21671) the union of a naive index containing a missing value with
        # an aware index is an object-dtype index in input order; with numpy newer than
        # 2.6 a runtime warning that the sort order is undefined is expected.
        # GH#21671
        rng = DatetimeIndex([Timestamp("2011-01-01"), pd.NaT])
        rng2 = DatetimeIndex(["2012-01-01", "2012-01-02"], tz="Asia/Tokyo")
        # numpy>=2.6 object-dtype sort compares the tz-naive and tz-aware
        #  Timestamps directly, so safe_sort fails and union warns
        warn = RuntimeWarning if np_version_gt2_6 else None
        with tm.assert_produces_warning(warn, match="sort order is undefined"):
            result = rng.union(rng2)
        expected = Index(
            [
                Timestamp("2011-01-01"),
                pd.NaT,
                Timestamp("2012-01-01", tz="Asia/Tokyo"),
                Timestamp("2012-01-02", tz="Asia/Tokyo"),
            ],
            dtype=object,
        )
        tm.assert_index_equal(result, expected)


class TestBusinessDatetimeIndex:
    # [CLASS DOC]
    # Groups set-operation tests on business-day ranges.
    def test_union(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies union of overlapping, gapped (generic Index
        # result) and adjacent slices; that argument order does not matter when sorting
        # (otherwise the concatenation order is kept); and a union with itself for a
        # different offset still gives a DatetimeIndex.
        rng = bdate_range(START, END)
        # overlapping
        left = rng[:10]
        right = rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # non-overlapping, gap in middle
        left = rng[:5]
        right = rng[10:]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, Index)

        # non-overlapping, no gap
        left = rng[:5]
        right = rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # order does not matter
        if sort is None:
            tm.assert_index_equal(right.union(left, sort=sort), the_union)
        else:
            expected = DatetimeIndex(list(right) + list(left))
            tm.assert_index_equal(right.union(left, sort=sort), expected)

        # overlapping, but different offset
        rng = date_range(START, END, freq=BMonthEnd())

        the_union = rng.union(rng, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

    def test_union_not_cacheable(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Verifies the union of minute-range slices of an
        # overlapping split equals the full range (sorted) or the rotated concatenation
        # (unsorted), and that an overlapping contained slice gives the larger slice.
        rng = date_range("1/1/2000", periods=50, freq=Minute())
        rng1 = rng[10:]
        rng2 = rng[:25]
        the_union = rng1.union(rng2, sort=sort)
        if sort is None:
            tm.assert_index_equal(the_union, rng)
        else:
            expected = DatetimeIndex(list(rng[10:]) + list(rng[:10]))
            tm.assert_index_equal(the_union, expected)

        rng1 = rng[10:]
        rng2 = rng[15:35]
        the_union = rng1.union(rng2, sort=sort)
        expected = rng[10:]
        tm.assert_index_equal(the_union, expected)

    def test_intersection(self):
        # [FUNCTION DOC]
        # Verifies intersection of overlapping minute-range slices equals the overlap
        # with frequency kept, the repeated call is the same, and a non-overlapping
        # intersection is an empty nanosecond-unit index.
        rng = date_range("1/1/2000", periods=50, freq=Minute(), unit="ns")
        rng1 = rng[10:]
        rng2 = rng[:25]
        the_int = rng1.intersection(rng2)
        expected = rng[10:25]
        tm.assert_index_equal(the_int, expected)
        assert isinstance(the_int, DatetimeIndex)
        assert the_int.freq == rng.freq

        the_int = rng1.intersection(rng2)
        tm.assert_index_equal(the_int, expected)

        # non-overlapping
        the_int = rng[:10].intersection(rng[10:])
        expected = DatetimeIndex([]).as_unit("ns")
        tm.assert_index_equal(the_int, expected)

    def test_intersection_bug(self):
        # [FUNCTION DOC]
        # Verifies (GH #771) the intersection of two business ranges equals the
        # smaller one with its frequency.
        # GH #771
        a = bdate_range("11/30/2011", "12/31/2011")
        b = bdate_range("12/10/2011", "12/20/2011")
        result = a.intersection(b)
        tm.assert_index_equal(result, b)
        assert result.freq == b.freq

    def test_intersection_list(self):
        # [FUNCTION DOC]
        # Verifies (GH#35876) the intersection of a named index with a list of its own
        # values equals the index and retains its name.
        # GH#35876
        # values is not an Index -> no name -> retain "a"
        values = [Timestamp("2020-01-01"), Timestamp("2020-02-01")]
        idx = DatetimeIndex(values, name="a")
        res = idx.intersection(values)
        tm.assert_index_equal(res, idx)

    def test_month_range_union_tz_pytz(self, sort):
        # [FUNCTION DOC]
        # Requires pytz. Smoke test: union of month-end ranges built with a pytz
        # Eastern zone must not raise.
        pytz = pytest.importorskip("pytz")
        tz = pytz.timezone("US/Eastern")

        early_start = datetime(2011, 1, 1)
        early_end = datetime(2011, 3, 1)

        late_start = datetime(2011, 3, 1)
        late_end = datetime(2011, 5, 1)

        early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
        late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())

        early_dr.union(late_dr, sort=sort)

    @td.skip_if_windows
    def test_month_range_union_tz_dateutil(self, sort):
        # [FUNCTION DOC]
        # Skipped on Windows. Smoke test: union of month-end ranges built with a
        # dateutil Eastern zone must not raise.
        from pandas._libs.tslibs.timezones import dateutil_gettz

        tz = dateutil_gettz("US/Eastern")

        early_start = datetime(2011, 1, 1)
        early_end = datetime(2011, 3, 1)

        late_start = datetime(2011, 3, 1)
        late_end = datetime(2011, 5, 1)

        early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
        late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())

        early_dr.union(late_dr, sort=sort)

    @pytest.mark.parametrize("sort", [False, None])
    def test_intersection_duplicates(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort (false or none). Verifies (GH#38196) that intersection
        # of an index having duplicate elements with itself removes duplicates.
        # GH#38196
        idx1 = Index(
            [
                Timestamp("2019-12-13"),
                Timestamp("2019-12-12"),
                Timestamp("2019-12-12"),
            ]
        )
        result = idx1.intersection(idx1, sort=sort)
        expected = Index([Timestamp("2019-12-13"), Timestamp("2019-12-12")])
        tm.assert_index_equal(result, expected)


class TestCustomDatetimeIndex:
    # [CLASS DOC]
    # Groups set-operation tests on custom business-day ranges.
    def test_union(self, sort):
        # [FUNCTION DOC]
        # Parametrized over sort. Same structure as the business-range union test:
        # overlapping, gapped and adjacent unions give the expected kinds, ordering
        # is only compared when sort is none, and a union with itself works.
        # overlapping
        rng = bdate_range(START, END, freq="C")
        left = rng[:10]
        right = rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # non-overlapping, gap in middle
        left = rng[:5]
        right = rng[10:]

        the_union = left.union(right, sort)
        assert isinstance(the_union, Index)

        # non-overlapping, no gap
        left = rng[:5]
        right = rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # order does not matter
        if sort is None:
            tm.assert_index_equal(right.union(left, sort=sort), the_union)

        # overlapping, but different offset
        rng = date_range(START, END, freq=BMonthEnd())

        the_union = rng.union(rng, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

    def test_intersection_bug(self):
        # [FUNCTION DOC]
        # Verifies (GH #771) the intersection of two custom-business ranges equals the
        # smaller one with its frequency.
        # GH #771
        a = bdate_range("11/30/2011", "12/31/2011", freq="C")
        b = bdate_range("12/10/2011", "12/20/2011", freq="C")
        result = a.intersection(b)
        tm.assert_index_equal(result, b)
        assert result.freq == b.freq

    @pytest.mark.parametrize(
        "tz", [None, "UTC", "Europe/Berlin", timezone(timedelta(hours=-1))]
    )
    def test_intersection_dst_transition(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone including a daylight-saving zone and a fixed
        # offset. Verifies (GH 46702) the intersection across a DST transition and
        # (GH#45863) the union of daily London ranges across the clock change equal the expected daily ranges.
        # GH 46702: Europe/Berlin has DST transition
        idx1 = date_range("2020-03-27", periods=5, freq="D", tz=tz)
        idx2 = date_range("2020-03-30", periods=5, freq="D", tz=tz)
        result = idx1.intersection(idx2)
        expected = date_range("2020-03-30", periods=2, freq="D", tz=tz)
        tm.assert_index_equal(result, expected)

        # GH#45863 same problem for union
        index1 = date_range("2021-10-28", periods=3, freq="D", tz="Europe/London")
        index2 = date_range("2021-10-30", periods=4, freq="D", tz="Europe/London")
        result = index1.union(index2)
        expected = date_range("2021-10-28", periods=6, freq="D", tz="Europe/London")
        tm.assert_index_equal(result, expected)


def test_union_non_nano_rangelike():
    # [FUNCTION DOC]
    # Verifies (GH 59036) the union of two adjacent daily microsecond-unit ranges with
    # the same name equals the combined range, keeping name, unit and frequency.
    # GH 59036
    l1 = DatetimeIndex(
        ["2024-05-11", "2024-05-12"], dtype="datetime64[us]", name="Date", freq="D"
    )
    l2 = DatetimeIndex(["2024-05-13"], dtype="datetime64[us]", name="Date", freq="D")
    result = l1.union(l2)
    expected = DatetimeIndex(
        ["2024-05-11", "2024-05-12", "2024-05-13"],
        dtype="datetime64[us]",
        name="Date",
        freq="D",
    )
    tm.assert_index_equal(result, expected)


def test_intersection_non_nano_rangelike():
    # [FUNCTION DOC]
    # Verifies (GH 59271) the intersection of two overlapping second-unit daily
    # ranges equals the overlap with daily frequency and second unit.
    # GH 59271
    l1 = date_range("2024-01-01", "2024-01-03", unit="s")
    l2 = date_range("2024-01-02", "2024-01-04", unit="s")
    result = l1.intersection(l2)
    expected = DatetimeIndex(
        ["2024-01-02", "2024-01-03"],
        dtype="datetime64[s]",
        freq="D",
    )
    tm.assert_index_equal(result, expected)


def test_union_across_dst_boundary():
    # [FUNCTION DOC]
    # Verifies (GH#62915) the union of two daily Helsinki ranges, one ending at the
    # daylight saving boundary and the other extending past it, equals the longer range.
    # GH#62915 union should work when one index ends at DST boundary
    # and the other extends past it
    index1 = date_range("2025-10-25", "2025-10-26", freq="D", tz="Europe/Helsinki")
    index2 = date_range("2025-10-25", "2025-10-28", freq="D", tz="Europe/Helsinki")
    result = index1.union(index2)
    expected = date_range("2025-10-25", "2025-10-28", freq="D", tz="Europe/Helsinki")
    tm.assert_index_equal(result, expected)
