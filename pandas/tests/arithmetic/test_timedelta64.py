# [MODULE DOC]
# Arithmetic and comparison tests for timedelta64 dtype across arrays, indexes, series, and frames that should behave identically.
# Covers comparisons (scalars, invalid operands, NaT, list-likes), arithmetic with datetimes and timestamps, additions and subtractions of timedelta-likes, NaT handling, multiplication and division by scalars and arrays (including numexpr and unit handling), floor division and modulo, divmod, and offset arrays. Uses shared fixtures for box types, timedelta-like scalars, names, and warnings, plus the shared invalid-comparison and invalid-add/subtract helpers.
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs import timezones
from pandas.compat import WASM
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    offsets,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)


def assert_dtype(obj, expected_dtype):
    # [FUNCTION DOC]
    # HELPER: asserts the dtype of a series, index, or single-column frame (found through the testing module's dtype lookup) equals the expected dtype.
    """
    Helper to check the dtype for a Series, Index, or single-column DataFrame.
    """
    dtype = tm.get_dtype(obj)

    assert dtype == expected_dtype


def get_expected_name(box, names):
    # [FUNCTION DOC]
    # HELPER: picks the expected result name when an operation mixes boxes. Takes a box type and a sequence of three names. If the box is a frame the first name is returned (the non-frame operand is cast to a series and its name ignored); if the box is an array type (the array helper or array factory) the second name is returned; otherwise the third.
    if box is DataFrame:
        # Since we are operating with a DataFrame and a non-DataFrame,
        # the non-DataFrame is cast to Series and its name ignored.
        exname = names[0]
    elif box in [tm.to_array, pd.array]:
        exname = names[1]
    else:
        exname = names[2]
    return exname


# ------------------------------------------------------------------
# Timedelta64[ns] dtype Comparisons


class TestTimedelta64ArrayLikeComparisons:
    # [CLASS DOC]
    # Comparison tests for timedelta64 vectors, parametrized over frame, series, timedelta index, and timedelta array boxes. Intended eventually to hold all timedelta comparison tests.
    # Comparison tests for timedelta64[ns] vectors fully parametrized over
    #  DataFrame/Series/TimedeltaIndex/TimedeltaArray.  Ideally all comparison
    #  tests will eventually end up here.

    def test_compare_timedelta64_zerodim(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies a boxed timedelta range compared by less-or-equal with a zero-dimensional NumPy array holding its first element unboxes it: true for the first element only, boxed as an array for index and array boxes.
        # GH#26689 should unbox when comparing with zerodim array
        box = box_with_array
        xbox = box_with_array if box_with_array not in [Index, pd.array] else np.ndarray

        tdi = timedelta_range("2h", periods=4)
        other = np.array(tdi.to_numpy()[0])

        tdi = tm.box_expected(tdi, box)
        res = tdi <= other
        expected = np.array([True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize(
        "td_scalar",
        [
            timedelta(days=1),
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            offsets.Hour(24),
        ],
    )
    def test_compare_timedeltalike_scalar(self, box_with_array, td_scalar):
        # [FUNCTION DOC]
        # Regression test for an issue. Parametrized over equivalent one-day scalars (Python timedelta, Timedelta, NumPy timedelta, hour offset of 24). Verifies a boxed series of one and two days compared greater-than the scalar gives false, true.
        # regression test for GH#5963
        box = box_with_array
        xbox = box if box not in [Index, pd.array] else np.ndarray

        ser = Series([timedelta(days=1), timedelta(days=2)])
        ser = tm.box_expected(ser, box)
        actual = ser > td_scalar
        expected = Series([False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "invalid",
        [
            345600000000000,
            "a",
            Timestamp("2021-01-01"),
            Timestamp("2021-01-01").now("UTC"),
            Timestamp("2021-01-01").now().to_datetime64(),
            Timestamp("2021-01-01").now().to_pydatetime(),
            Timestamp("2021-01-01").date(),
            np.array(4),  # zero-dim mismatched dtype
        ],
    )
    def test_td64_comparisons_invalid(self, box_with_array, invalid):
        # [FUNCTION DOC]
        # Parametrized over invalid operands (an integer count of nanoseconds, a string, timestamps in several forms, a date, a zero-dimensional integer array). Verifies the shared invalid-comparison helper's expectations on a boxed ten-day timedelta range (equality false, inequality true, ordering raises).
        # GH#13624 for str
        box = box_with_array

        rng = timedelta_range("1 days", periods=10)
        obj = tm.box_expected(rng, box)

        assert_invalid_comparison(obj, invalid, box)

    @pytest.mark.parametrize(
        "other",
        [
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.date_range("1970-01-01", periods=10, tz="UTC").array,
            np.array(pd.date_range("1970-01-01", periods=10)),
            list(pd.date_range("1970-01-01", periods=10)),
            pd.date_range("1970-01-01", periods=10).astype(object),
            pd.period_range("1971-01-01", freq="D", periods=10).array,
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),
        ],
    )
    def test_td64arr_cmp_arraylike_invalid(self, other, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over list-likes of integers, floats, objects, datetimes (aware and naive, various containers), and periods. Verifies the invalid-comparison helper on a boxed timedelta array; not parametrized over lists in other boxes because list-likes interact poorly with the helper's reversed checks.
        # We don't parametrize this over box_with_array because listlike
        #  other plays poorly with assert_invalid_comparison reversed checks

        rng = timedelta_range("1 days", periods=10)._data
        rng = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(rng, other, box_with_array)

    def test_td64arr_cmp_mixed_invalid(self):
        # [FUNCTION DOC]
        # Verifies comparing a timedelta array with an object array holding integers, a matching timedelta, and a timestamp: equality is true only at the matching element, inequality is the negation, and ordering comparisons raise a type error ("Invalid comparison", "Cannot compare type", or "not supported between").
        rng = timedelta_range("1 days", periods=5)._data
        other = np.array([0, 1, 2, rng[3], Timestamp("2021-01-01")])

        result = rng == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = rng != other
        tm.assert_numpy_array_equal(result, ~expected)

        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):
            rng < other
        with pytest.raises(TypeError, match=msg):
            rng > other
        with pytest.raises(TypeError, match=msg):
            rng <= other
        with pytest.raises(TypeError, match=msg):
            rng >= other


class TestTimedelta64ArrayComparisons:
    # [CLASS DOC]
    # Test class for comparisons of timedelta vectors; a TODO says all need parametrization over boxes.
    # TODO: All of these need to be parametrized over box

    @pytest.mark.parametrize("dtype", [None, object])
    def test_comp_nat(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype none or object. Verifies equality, inequality and ordering between timedelta indexes with NaT in different places, and against the NaT scalar on either side: NaT is never equal and always unequal, ordering with NaT is all false.
        left = TimedeltaIndex([Timedelta("1 days"), NaT, Timedelta("3 days")])
        right = TimedeltaIndex([NaT, NaT, Timedelta("3 days")])

        lhs, rhs = left, right
        if dtype is object:
            lhs, rhs = left.astype(object), right.astype(object)

        result = rhs == lhs
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = rhs != lhs
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs == NaT, expected)
        tm.assert_numpy_array_equal(NaT == rhs, expected)

        expected = np.array([True, True, True])
        tm.assert_numpy_array_equal(lhs != NaT, expected)
        tm.assert_numpy_array_equal(NaT != lhs, expected)

        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs < NaT, expected)
        tm.assert_numpy_array_equal(NaT > lhs, expected)

    @pytest.mark.parametrize(
        "idx2",
        [
            TimedeltaIndex(
                ["2 day", "2 day", NaT, NaT, "1 day 00:00:02", "5 days 00:00:03"]
            ),
            np.array(
                [
                    np.timedelta64(2, "D"),
                    np.timedelta64(2, "D"),
                    np.timedelta64("NaT", "ns"),
                    np.timedelta64("NaT", "ns"),
                    np.timedelta64(1, "D") + np.timedelta64(2, "s"),
                    np.timedelta64(5, "D") + np.timedelta64(3, "s"),
                ]
            ),
        ],
    )
    def test_comparisons_nat(self, idx2):
        # [FUNCTION DOC]
        # Parametrized over a timedelta index and a NumPy timedelta array containing NaT. Verifies all six comparison operators (including reversed forms) between a timedelta index with NaT and such an operand give the expected boolean arrays, with NaT handled like NaN (comparisons false, inequality true).
        idx1 = TimedeltaIndex(
            [
                "1 day",
                NaT,
                "1 day 00:00:01",
                NaT,
                "1 day 00:00:01",
                "5 day 00:00:03",
            ]
        )
        # Check pd.NaT is handles as the same as np.nan
        result = idx1 < idx2
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = idx2 > idx1
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = idx1 <= idx2
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)

        result = idx2 >= idx1
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)

        result = idx1 == idx2
        expected = np.array([False, False, False, False, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = idx1 != idx2
        expected = np.array([True, True, True, True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    # TODO: better name
    def test_comparisons_coverage(self):
        # [FUNCTION DOC]
        # A TODO says to find a better name. Verifies a ten-day timedelta range compared less-than its fourth element gives three true then false, and equality with the same values as a list equals equality with the range.
        rng = timedelta_range("1 days", periods=10)

        result = rng < rng[3]
        expected = np.array([True, True, True] + [False] * 7)
        tm.assert_numpy_array_equal(result, expected)

        result = rng == list(rng)
        exp = rng == rng
        tm.assert_numpy_array_equal(result, exp)


# ------------------------------------------------------------------
# Timedelta64[ns] dtype Arithmetic Operations


class TestTimedelta64ArithmeticUnsorted:
    # [CLASS DOC]
    # Test class for timedelta arithmetic tests moved from type-specific files but not yet sorted, parametrized, or de-duplicated.
    # Tests moved from type-specific test files but not
    #  yet sorted/parametrized/de-duplicated

    def test_td64_op_with_list(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies adding a list of timestamps to a boxed timedelta index (either order) gives a boxed microsecond-unit datetime index of the sums.
        # GH#62353
        box = box_with_array

        left = TimedeltaIndex(["2D", "4D"])
        left = tm.box_expected(left, box)

        right = [Timestamp("2016-01-01"), Timestamp("2016-02-01")]

        result = left + right
        expected = DatetimeIndex(["2016-01-03", "2016-02-05"], dtype="M8[us]")
        expected = tm.box_expected(expected, box)
        tm.assert_equal(result, expected)

        result2 = right + left
        tm.assert_equal(result2, expected)

    def test_ufunc_coercions(self):
        # [FUNCTION DOC]
        # Verifies that multiplying a timedelta index by two (operator and NumPy multiply), dividing by two (operator and NumPy divide), negating (operator and NumPy negative), and absolute value (built-in and NumPy) return timedelta indexes with the expected values and name, and the frequency scaled accordingly: four hours, one hour, minus two hours, and none for absolute value.
        # normal ops are also tested in tseries/test_timedeltas.py
        idx = TimedeltaIndex(["2h", "4h", "6h", "8h", "10h"], freq="2h", name="x")

        for result in [idx * 2, np.multiply(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["4h", "8h", "12h", "16h", "20h"], freq="4h", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "4h"

        for result in [idx / 2, np.divide(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["1h", "2h", "3h", "4h", "5h"], freq="h", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "h"

        for result in [-idx, np.negative(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(
                ["-2h", "-4h", "-6h", "-8h", "-10h"], freq="-2h", name="x"
            )
            tm.assert_index_equal(result, exp)
            assert result.freq == "-2h"

        idx = TimedeltaIndex(["-2h", "-1h", "0h", "1h", "2h"], freq="h", name="x")
        for result in [abs(idx), np.absolute(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["2h", "1h", "0h", "1h", "2h"], freq=None, name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq is None

    def test_subtraction_ops(self):
        # [FUNCTION DOC]
        # Verifies subtraction rules among timedelta index, datetime index, timedelta scalar, and timestamp: timedelta index minus a timestamp or datetime index raises a type error about subtracting a datelike from a timedelta array; a timedelta minus a timestamp raises an unsupported-operand error; a timedelta minus a datetime index raises "cannot subtract DatetimeArray from Timedelta"; a timestamp minus a datetime index and the reverse give the timedelta indexes with the datetime index's name; timedelta index minus or minus from a timedelta keeps the first operand's name and NaT; datetime index minus a timedelta gives a daily-frequency datetime index; a timestamp minus a timedelta index gives a datetime index with NaT.
        # with datetimes/timedelta and tdi/dti
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        td = Timedelta("1 days")
        dt = Timestamp("20130101")

        msg = "cannot subtract a datelike from a TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi - dt
        with pytest.raises(TypeError, match=msg):
            tdi - dti

        msg = r"unsupported operand type\(s\) for -"
        with pytest.raises(TypeError, match=msg):
            td - dt

        msg = "cannot subtract DatetimeArray from Timedelta"
        with pytest.raises(TypeError, match=msg):
            td - dti

        result = dt - dti
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"], name="bar")
        tm.assert_index_equal(result, expected)

        result = dti - dt
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"], name="bar")
        tm.assert_index_equal(result, expected)

        result = tdi - td
        expected = TimedeltaIndex(["0 days", NaT, "1 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = td - tdi
        expected = TimedeltaIndex(["0 days", NaT, "-1 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = dti - td
        expected = DatetimeIndex(
            ["20121231", "20130101", "20130102"], dtype="M8[us]", freq="D", name="bar"
        )
        tm.assert_index_equal(result, expected)

        result = dt - tdi
        expected = DatetimeIndex(
            ["20121231", NaT, "20121230"], dtype="M8[us]", name="foo"
        )
        tm.assert_index_equal(result, expected)

    def test_subtraction_ops_with_tz(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies timezone validation in subtraction: scalar subtractions among naive and aware timestamps and Python datetimes give zero timedeltas when the awareness matches and raise type errors (with several messages for Timestamp versus datetime) when it differs or the zones differ; subtracting an aware timestamp from a naive boxed datetime index and the reverse raises a type error; aware index minus aware datetime or timestamp (and the reverse) gives boxed timedelta indexes of zero, one, two days (or negated); timedelta minus timedelta gives zero; aware index minus a timedelta shifts back a day keeping the zone.
        # check that dt/dti subtraction ops with tz are validated
        dti = pd.date_range("20130101", periods=3)
        dti = tm.box_expected(dti, box_with_array)
        ts = Timestamp("20130101")
        dt = ts.to_pydatetime()
        dti_tz = pd.date_range("20130101", periods=3).tz_localize("US/Eastern")
        dti_tz = tm.box_expected(dti_tz, box_with_array)
        ts_tz = Timestamp("20130101").tz_localize("US/Eastern")
        ts_tz2 = Timestamp("20130101").tz_localize("CET")
        dt_tz = ts_tz.to_pydatetime()
        td = Timedelta("1 days")

        def _check(result, expected):
            # [FUNCTION DOC]
            # HELPER: nested assertion helper that checks a scalar result equals the expected timedelta and is an instance of the Timedelta class.
            assert result == expected
            assert isinstance(result, Timedelta)

        # scalars
        result = ts - ts
        expected = Timedelta("0 days")
        _check(result, expected)

        result = dt_tz - ts_tz
        expected = Timedelta("0 days")
        _check(result, expected)

        result = ts_tz - dt_tz
        expected = Timedelta("0 days")
        _check(result, expected)

        # tz mismatches
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects."
        with pytest.raises(TypeError, match=msg):
            dt_tz - ts
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt_tz - dt
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt - dt_tz
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects."
        with pytest.raises(TypeError, match=msg):
            ts - dt_tz
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - ts
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - dt

        msg = "Cannot subtract tz-naive and tz-aware"
        # with dti
        with pytest.raises(TypeError, match=msg):
            dti - ts_tz
        with pytest.raises(TypeError, match=msg):
            dti_tz - ts

        result = dti_tz - dt_tz
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = dt_tz - dti_tz
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = dti_tz - ts_tz
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = ts_tz - dti_tz
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = td - td
        expected = Timedelta("0 days")
        _check(result, expected)

        result = dti_tz - td
        expected = DatetimeIndex(["20121231", "20130101", "20130102"], tz="US/Eastern")
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dti_tdi_numeric_ops(self):
        # [FUNCTION DOC]
        # Verifies that adding, subtracting two timedelta indexes (with NaT) is elementwise arithmetic keeping the name, not set operations, and that a datetime index minus a timedelta index gives a datetime index with NaT in the matching place and no name (names differ so it is reset).
        # These are normally union/diff set-like ops
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")

        result = tdi - tdi
        expected = TimedeltaIndex(["0 days", NaT, "0 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = tdi + tdi
        expected = TimedeltaIndex(["2 days", NaT, "4 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = dti - tdi  # name will be reset
        expected = DatetimeIndex(["20121231", NaT, "20130101"], dtype="M8[us]")
        tm.assert_index_equal(result, expected)

    def test_addition_ops(self):
        # [FUNCTION DOC]
        # Verifies addition among timedelta index (with NaT, named), datetime index, timestamp, and timedelta: tdi plus timestamp (either order) gives a microsecond-unit datetime index with the tdi's name and NaT; timedelta plus tdi gives a timedelta index keeping name; adding indexes of unequal length raises a value error "cannot add indices of unequal length"; adding an integer index raises a type error about integer addition/subtraction; tdi plus dti (either order) gives a datetime index with name reset; timestamp plus timedelta (either order) gives the shifted timestamp.
        # with datetimes/timedelta and tdi/dti
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        td = Timedelta("1 days")
        dt = Timestamp("20130101")

        result = tdi + dt
        expected = DatetimeIndex(
            ["20130102", NaT, "20130103"], dtype="M8[us]", name="foo"
        )
        tm.assert_index_equal(result, expected)

        result = dt + tdi
        expected = DatetimeIndex(
            ["20130102", NaT, "20130103"], dtype="M8[us]", name="foo"
        )
        tm.assert_index_equal(result, expected)

        result = td + tdi
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = tdi + td
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        tm.assert_index_equal(result, expected)

        # unequal length
        msg = "cannot add indices of unequal length"
        with pytest.raises(ValueError, match=msg):
            tdi + dti[0:1]
        with pytest.raises(ValueError, match=msg):
            tdi[0:1] + dti

        # random indexes
        msg = "Addition/subtraction of integers and integer-arrays"
        with pytest.raises(TypeError, match=msg):
            tdi + Index([1, 2, 3], dtype=np.int64)

        # this is a union!
        # FIXME: don't leave commented-out
        # pytest.raises(TypeError, lambda : Index([1,2,3]) + tdi)

        result = tdi + dti  # name will be reset
        expected = DatetimeIndex(["20130102", NaT, "20130105"], dtype="M8[us]")
        tm.assert_index_equal(result, expected)

        result = dti + tdi  # name will be reset
        expected = DatetimeIndex(["20130102", NaT, "20130105"], dtype="M8[us]")
        tm.assert_index_equal(result, expected)

        result = dt + td
        expected = Timestamp("20130102")
        assert result == expected

        result = td + dt
        expected = Timestamp("20130102")
        assert result == expected

    # TODO: Needs more informative name, probably split up into
    # more targeted tests
    @pytest.mark.parametrize("freq", ["D", "B"])
    def test_timedelta(self, freq):
        # [FUNCTION DOC]
        # A comment says it needs a more informative name. Parametrized over frequencies D and B. For a 50-element nanosecond-unit date range, adding then subtracting a one-day timedelta returns the original (after re-inferring the frequency); for D the frequency is a day throughout; for B the shifted index has no frequency and the original and round-tripped have a business-day frequency; subtracting a timedelta equals adding its negation.
        index = pd.date_range("1/1/2000", periods=50, freq=freq, unit="ns")

        shifted = index + timedelta(1)
        back = shifted + timedelta(-1)
        back = back._with_freq("infer")
        tm.assert_index_equal(index, back)

        if freq == "D":
            expected = pd.tseries.offsets.Day(1)
            assert index.freq == expected
            assert shifted.freq == expected
            assert back.freq == expected
        else:  # freq == 'B'
            assert index.freq == pd.tseries.offsets.BusinessDay(1)
            assert shifted.freq is None
            assert back.freq == pd.tseries.offsets.BusinessDay(1)

        result = index - timedelta(1)
        expected = index + timedelta(-1)
        tm.assert_index_equal(result, expected)

    def test_timedelta_tick_arithmetic(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies subtraction of a one-hour offset or a 100-million-nanosecond timedelta from a daily date range or from a series of it: results from the range keep its frequency, and after clearing it equal the results built from the series.
        # GH#4134, buggy with timedeltas
        rng = pd.date_range("2013", "2014")
        s = Series(rng)
        result1 = rng - offsets.Hour(1)
        result2 = DatetimeIndex(s - np.timedelta64(100000000, "ns"))
        result3 = rng - np.timedelta64(100000000, "ns")
        result4 = DatetimeIndex(s - offsets.Hour(1))

        assert result1.freq == rng.freq
        result1 = result1._with_freq(None)
        tm.assert_index_equal(result1, result4)

        assert result3.freq == rng.freq
        result3 = result3._with_freq(None)
        tm.assert_index_equal(result2, result3)

    def test_tda_add_sub_index(self):
        # [FUNCTION DOC]
        # Verifies a timedelta array defers to the index operation: adding a datetime index, adding a timedelta index, and subtracting a timedelta index from a timedelta array equal the corresponding index operations.
        # Check that TimedeltaArray defers to Index on arithmetic ops
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"])
        tda = tdi.array

        dti = pd.date_range("1999-12-31", periods=3, freq="D")

        result = tda + dti
        expected = tdi + dti
        tm.assert_index_equal(result, expected)

        result = tda + tdi
        expected = tdi + tdi
        tm.assert_index_equal(result, expected)

        result = tda - tdi
        expected = tdi - tdi
        tm.assert_index_equal(result, expected)

    def test_tda_add_dt64_object_array(
        self, performance_warning, box_with_array, tz_naive_fixture
    ):
        # [FUNCTION DOC]
        # Uses the performance-warning, box and naive-timezone fixtures. Builds a timedelta object as the difference of a datetime index (with no frequency) with itself and adds the datetimes cast to object; verifies a performance warning is emitted and the result equals the object-cast datetimes.
        # Result should be cast back to DatetimeArray
        box = box_with_array

        dti = pd.date_range("2016-01-01", periods=3, tz=tz_naive_fixture)
        dti = dti._with_freq(None)
        tdi = dti - dti

        obj = tm.box_expected(tdi, box)
        other = tm.box_expected(dti, box)

        with tm.assert_produces_warning(performance_warning):
            result = obj + other.astype(object)
        tm.assert_equal(result, other.astype(object))

    # -------------------------------------------------------------
    # Binary operations TimedeltaIndex and timedelta-like

    def test_tdi_iadd_timedeltalike(self, two_hours, box_with_array):
        # [FUNCTION DOC]
        # Uses the two-hours fixture and box fixture. Verifies in-place addition of the timedelta-like to a boxed daily timedelta range gives the expected shifted range (nanosecond unit if the scalar is a nanosecond Timedelta and the box is not the array factory, because an extension array operates truly in place and keeps its dtype); and that for non-index boxes the original object itself changed.
        # only test adding/sub offsets as + is now numeric
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")
        if (
            isinstance(two_hours, Timedelta)
            and two_hours.unit == "ns"
            and box_with_array is not pd.array
        ):
            # The EA op has to be _actually_ inplace so does not cast to a
            #  new dtype.  For the others, the op can assign a new array
            #  and get the dtype that normally results from `rng + two_hours`
            expected = expected.as_unit("ns")

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        orig_rng = rng
        rng += two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            # Check that operation is actually inplace
            tm.assert_equal(orig_rng, expected)

    def test_tdi_isub_timedeltalike(self, two_hours, box_with_array):
        # [FUNCTION DOC]
        # Uses the two-hours fixture and box fixture. Same as the in-place add test but with in-place subtraction: two hours earlier, same unit rules, original object modified for non-index boxes.
        # only test adding/sub offsets as - is now numeric
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")
        if (
            isinstance(two_hours, Timedelta)
            and two_hours.unit == "ns"
            and box_with_array is not pd.array
        ):
            # The EA op has to be _actually_ inplace so does not cast to a
            #  new dtype.  For the others, the op can assign a new array
            #  and get the dtype that normally results from `rng - two_hours`
            expected = expected.as_unit("ns")

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        orig_rng = rng
        rng -= two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            # Check that operation is actually inplace
            tm.assert_equal(orig_rng, expected)

    # -------------------------------------------------------------

    def test_tdi_ops_attributes(self):
        # [FUNCTION DOC]
        # Verifies attributes after timedelta index operations on a two-day-frequency range named x: adding one frequency unit keeps two-day frequency; subtracting two units keeps it; multiplying by two doubles the frequency to four days; dividing by two halves it to one day; negation gives frequency minus two days; absolute value of a range from minus two days drops the frequency to none.
        rng = timedelta_range("2 days", periods=5, freq="2D", name="x")

        result = rng + 1 * rng.freq
        exp = timedelta_range("4 days", periods=5, freq="2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "2D"

        result = rng - 2 * rng.freq
        exp = timedelta_range("-2 days", periods=5, freq="2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "2D"

        result = rng * 2
        exp = timedelta_range("4 days", periods=5, freq="4D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "4D"

        result = rng / 2
        exp = timedelta_range("1 days", periods=5, freq="D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "D"

        result = -rng
        exp = timedelta_range("-2 days", periods=5, freq="-2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "-2D"

        rng = timedelta_range("-2 days", periods=5, freq="D", name="x")

        result = abs(rng)
        exp = TimedeltaIndex(
            ["2 days", "1 days", "0 days", "1 days", "2 days"], name="x"
        )
        tm.assert_index_equal(result, exp)
        assert result.freq is None


class TestAddSubNaTMasking:
    # [CLASS DOC]
    # Test class for NaT masking and overflow handling in timedelta/timestamp addition and subtraction; a TODO says to parametrize over boxes.
    # TODO: parametrize over boxes

    @pytest.mark.parametrize("str_ts", ["1950-01-01", "1980-01-01"])
    def test_tdarr_add_timestamp_nat_masking(self, box_with_array, str_ts):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over two date strings; uses the box fixture. Adds to a boxed timedelta pair (24658 days 11:15 and NaT) each of four variants of a timestamp (timestamp, Python datetime, NumPy nanosecond and day datetimes) and verifies the NaT element stays NaT (checking the frame cell for frames) without overflow wraparound.
        # GH#17991 checking for overflow-masking with NaT
        tdinat = pd.to_timedelta(["24658 days 11:15:00", "NaT"])
        tdobj = tm.box_expected(tdinat, box_with_array)

        ts = Timestamp(str_ts)
        ts_variants = [
            ts,
            ts.to_pydatetime(),
            ts.to_datetime64().astype("datetime64[ns]"),
            ts.to_datetime64().astype("datetime64[D]"),
        ]

        for variant in ts_variants:
            res = tdobj + variant
            if box_with_array is DataFrame:
                assert res.iloc[1, 1] is NaT
            else:
                assert res[1] is NaT

    def test_tdi_add_overflow(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies overflow handling: adding a 106580-day timedelta to a nanosecond timestamp in either order raises an out-of-bounds-datetime error naming the overflowed integer; a vector of that size plus a timestamp, and subtraction of timedeltas involving the internal NaT integer plus one, raise an overflow error "Overflow in int64 addition"; but operations with real NaT do not overflow and give expected NaT-containing timedelta indexes.
        # See GH#14068
        # preliminary test scalar analogue of vectorized tests below
        # TODO: Make raised error message more informative and test
        ts = Timestamp("2000").as_unit("ns")
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            pd.to_timedelta(106580, "D") + ts
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            ts + pd.to_timedelta(106580, "D")

        _NaT = NaT._value + 1
        td = pd.to_timedelta([106580], "D").as_unit("ns")
        msg = "Overflow in int64 addition"
        with pytest.raises(OverflowError, match=msg):
            td + Timestamp("2000")
        with pytest.raises(OverflowError, match=msg):
            Timestamp("2000") + td
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([_NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta(["5 days", _NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            (
                pd.to_timedelta([_NaT, "5 days", "1 hours"])
                - pd.to_timedelta(["7 seconds", _NaT, "4 hours"])
            )

        # These should not overflow!
        exp = TimedeltaIndex([NaT], dtype="m8[us]")
        result = pd.to_timedelta([NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)

        exp = TimedeltaIndex(["4 days", NaT])
        result = pd.to_timedelta(["5 days", NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)

        exp = TimedeltaIndex([NaT, NaT, "5 hours"])
        result = pd.to_timedelta([NaT, "5 days", "1 hours"]) + pd.to_timedelta(
            ["7 seconds", NaT, "4 hours"]
        )
        tm.assert_index_equal(result, exp)


class TestTimedeltaArraylikeAddSubOps:
    # [CLASS DOC]
    # Test class for addition and subtraction on timedelta64 vectors (operators and reflected forms).
    # Tests for timedelta64[ns] __add__, __sub__, __radd__, __rsub__

    def test_sub_nat_retain_unit(self):
        # [FUNCTION DOC]
        # Verifies that subtracting NaT from a series of second-unit timedeltas yields a series of NaT retaining the second unit.
        ser = pd.to_timedelta(Series(["00:00:01"])).astype("m8[s]")

        result = ser - NaT
        expected = Series([NaT], dtype="m8[s]")
        tm.assert_series_equal(result, expected)

    # TODO: moved from tests.indexes.timedeltas.test_arithmetic; needs
    #  parametrization+de-duplication
    def test_timedelta_ops_with_missing_values(self):
        # [FUNCTION DOC]
        # Moved from the timedelta index tests; a TODO says it needs parametrization and de-duplication. Checks arithmetic among scalar timedeltas, series, frames, NaT, and NaN: scalar add and subtract; series with series, with scalars, in both orders and with unary negation; series with a timedelta NaT or NaT gives an all-NaT series of microsecond unit; adding or subtracting NaN raises a type error for unsupported operand; series with frames in both orders; frames with frames and scalars; frame plus or minus a timedelta NaT or NaT gives an all-NaT frame; frame with NaN raises a type error.
        # setup
        s1 = pd.to_timedelta(Series(["00:00:01"]))
        s2 = pd.to_timedelta(Series(["00:00:02"]))

        sn = pd.to_timedelta(Series([NaT], dtype="m8[us]"))

        df1 = DataFrame(["00:00:01"]).apply(pd.to_timedelta)
        df2 = DataFrame(["00:00:02"]).apply(pd.to_timedelta)

        dfn = DataFrame([NaT._value]).apply(pd.to_timedelta).astype("m8[us]")

        scalar1 = pd.to_timedelta("00:00:01")
        scalar2 = pd.to_timedelta("00:00:02")
        timedelta_NaT = pd.to_timedelta("NaT")

        actual = scalar1 + scalar1
        assert actual == scalar2
        actual = scalar2 - scalar1
        assert actual == scalar1

        actual = s1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - s1
        tm.assert_series_equal(actual, s1)

        actual = s1 + scalar1
        tm.assert_series_equal(actual, s2)
        actual = scalar1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - scalar1
        tm.assert_series_equal(actual, s1)
        actual = -scalar1 + s2
        tm.assert_series_equal(actual, s1)

        actual = s1 + timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        actual = s1 - timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = -timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            s1 + np.nan
        with pytest.raises(TypeError, match=msg):
            np.nan + s1
        with pytest.raises(TypeError, match=msg):
            s1 - np.nan
        with pytest.raises(TypeError, match=msg):
            -np.nan + s1

        actual = s1 + NaT
        tm.assert_series_equal(actual, sn)
        actual = s2 - NaT
        tm.assert_series_equal(actual, sn)

        actual = s1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = s2 - df1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + s1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - s1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - df1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + scalar1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - scalar1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + timedelta_NaT
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - timedelta_NaT
        tm.assert_frame_equal(actual, dfn)

        msg = "cannot subtract a datelike from|unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            df1 + np.nan
        with pytest.raises(TypeError, match=msg):
            df1 - np.nan

        actual = df1 + NaT  # NaT is datetime, not timedelta
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - NaT
        tm.assert_frame_equal(actual, dfn)

    # TODO: moved from tests.series.test_operators, needs splitting, cleanup,
    # de-duplication, box-parametrization...
    def test_operators_timedelta64(self):
        # [FUNCTION DOC]
        # Moved from the series operator tests; a TODO says it needs splitting, cleanup, de-duplication, and box parametrization. Checks datetime/timedelta series arithmetic dtypes and values: the difference of two datetime series is a timedelta64 series of one day; frame column minus its shifted self is timedelta64; column plus a timedelta series is datetime64; column minus its max is timedelta64; that result plus the column gives expected datetimes; column minus a Python datetime gives expected day timedeltas and the round trip adding it back restores the column; adding and subtracting Python timedeltas round-trips the datetime column; and an in-place add on one element of the difference series stores the expected value.
        # series ops
        v1 = pd.date_range("2012-1-1", periods=3, freq="D", unit="ns")
        v2 = pd.date_range("2012-1-2", periods=3, freq="D", unit="ns")
        rs = Series(v2) - Series(v1)
        xp = Series(1e9 * 3600 * 24, rs.index).astype("int64").astype("timedelta64[ns]")
        tm.assert_series_equal(rs, xp)
        assert rs.dtype == "timedelta64[ns]"

        df = DataFrame({"A": v1})
        td = Series([timedelta(days=i) for i in range(3)], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        # series on the rhs
        result = df["A"] - df["A"].shift()
        assert result.dtype == "timedelta64[ns]"

        result = df["A"] + td
        assert result.dtype == "M8[ns]"

        # scalar Timestamp on rhs
        maxa = df["A"].max()
        assert isinstance(maxa, Timestamp)

        resultb = df["A"] - df["A"].max()
        assert resultb.dtype == "timedelta64[ns]"

        # timestamp on lhs
        result = resultb + df["A"]
        values = [Timestamp("20111230"), Timestamp("20120101"), Timestamp("20120103")]
        expected = Series(values, dtype="M8[ns]", name="A")
        tm.assert_series_equal(result, expected)

        # datetimes on rhs
        result = df["A"] - datetime(2001, 1, 1)
        expected = Series(
            [timedelta(days=4017 + i) for i in range(3)], name="A", dtype="m8[ns]"
        )
        tm.assert_series_equal(result, expected)
        assert result.dtype == "m8[ns]"

        d = datetime(2001, 1, 1, 3, 4)
        resulta = df["A"] - d
        assert resulta.dtype == "m8[ns]"

        # roundtrip
        resultb = resulta + d
        tm.assert_series_equal(df["A"], resultb)

        # timedeltas on rhs
        td = timedelta(days=1)
        resulta = df["A"] + td
        resultb = resulta - td
        tm.assert_series_equal(resultb, df["A"])
        assert resultb.dtype == "M8[ns]"

        # roundtrip
        td = timedelta(minutes=5, seconds=3)
        resulta = df["A"] + td
        resultb = resulta - td
        tm.assert_series_equal(df["A"], resultb)
        assert resultb.dtype == "M8[ns]"

        # inplace
        value = rs[2] + np.timedelta64(timedelta(minutes=5, seconds=1))
        rs[2] += np.timedelta64(timedelta(minutes=5, seconds=1))
        assert rs[2] == value

    def test_timedelta64_ops_nat(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies NaT handling in timedelta series: subtracting NaT or a one-element NaT timedelta series (also reflected forms) from a series [NaT, one second] gives an all-NaT microsecond-unit series; addition with NaT or NaT series in both orders likewise; multiplication of NaT series by one float, of the series by one, one and a half, and NaN; and division by two, two float, and NaN.
        # KNOWN ISSUE: some addition assertions with the NaT series appear twice with identical operands.
        # GH 11349
        timedelta_series = Series([NaT, Timedelta("1s")])
        nat_series_dtype_timedelta = Series([NaT, NaT], dtype="timedelta64[us]")
        single_nat_dtype_timedelta = Series([NaT], dtype="timedelta64[us]")

        # subtraction
        tm.assert_series_equal(timedelta_series - NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(-NaT + timedelta_series, nat_series_dtype_timedelta)

        tm.assert_series_equal(
            timedelta_series - single_nat_dtype_timedelta, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta
        )

        # addition
        tm.assert_series_equal(
            nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_timedelta,
            nat_series_dtype_timedelta,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timedelta,
            nat_series_dtype_timedelta,
        )

        tm.assert_series_equal(timedelta_series + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + timedelta_series, nat_series_dtype_timedelta)

        tm.assert_series_equal(
            timedelta_series + single_nat_dtype_timedelta, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_timedelta,
            nat_series_dtype_timedelta,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timedelta,
            nat_series_dtype_timedelta,
        )

        # multiplication
        tm.assert_series_equal(
            nat_series_dtype_timedelta * 1.0, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            1.0 * nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(timedelta_series * 1, timedelta_series)
        tm.assert_series_equal(1 * timedelta_series, timedelta_series)

        tm.assert_series_equal(timedelta_series * 1.5, Series([NaT, Timedelta("1.5s")]))
        tm.assert_series_equal(1.5 * timedelta_series, Series([NaT, Timedelta("1.5s")]))

        tm.assert_series_equal(timedelta_series * np.nan, nat_series_dtype_timedelta)
        tm.assert_series_equal(np.nan * timedelta_series, nat_series_dtype_timedelta)

        # division
        tm.assert_series_equal(timedelta_series / 2, Series([NaT, Timedelta("0.5s")]))
        tm.assert_series_equal(timedelta_series / 2.0, Series([NaT, Timedelta("0.5s")]))
        tm.assert_series_equal(timedelta_series / np.nan, nat_series_dtype_timedelta)

    # -------------------------------------------------------------
    # Binary operations td64 arraylike and datetime-like

    @pytest.mark.parametrize("cls", [Timestamp, datetime, np.datetime64])
    def test_td64arr_add_sub_datetimelike_scalar(
        self, cls, box_with_array, tz_naive_fixture
    ):
        # [FUNCTION DOC]
        # Linked to issues. Parametrized over scalar type (Timestamp, Python datetime, NumPy datetime64); uses the box and naive-timezone fixtures (NumPy datetime64 skipped for tz-aware). Verifies a three-day timedelta range plus the datetime-like scalar in either order gives the boxed expected date range starting the day after; frequency is dropped for non-UTC zones because day is now calendar-day; datetime minus the timedelta range gives dates counting backwards (equal to adding the negation); timedelta range minus the datetime raises a type error "cannot subtract a datelike".
        # GH#11925, GH#29558, GH#23215
        tz = tz_naive_fixture

        dt_scalar = Timestamp("2012-01-01", tz=tz)
        if cls is datetime:
            ts = dt_scalar.to_pydatetime()
        elif cls is np.datetime64:
            if tz_naive_fixture is not None:
                pytest.skip(f"{cls} doesn't support {tz_naive_fixture}")
            ts = dt_scalar.to_datetime64()
        else:
            ts = dt_scalar

        tdi = timedelta_range("1 day", periods=3)
        expected = pd.date_range("2012-01-02", periods=3, tz=tz)
        if tz is not None and not timezones.is_utc(expected.tz):
            # Day is no longer preserved by timedelta add/sub in pandas3 because
            #  it represents Calendar-Day instead of 24h
            expected = expected._with_freq(None)

        tdarr = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        tm.assert_equal(ts + tdarr, expected)
        tm.assert_equal(tdarr + ts, expected)

        expected2 = pd.date_range("2011-12-31", periods=3, freq="-1D", tz=tz)
        if tz is not None and not timezones.is_utc(expected2.tz):
            # Day is no longer preserved by timedelta add/sub in pandas3 because
            #  it represents Calendar-Day instead of 24h
            expected2 = expected2._with_freq(None)
        expected2 = tm.box_expected(expected2, box_with_array)

        tm.assert_equal(ts - tdarr, expected2)
        tm.assert_equal(ts + (-tdarr), expected2)

        msg = "cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            tdarr - ts

    @pytest.mark.filterwarnings("ignore:.*generic.*unit.*:DeprecationWarning")
    def test_td64arr_add_datetime64_nat(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Ignores a deprecation warning about generic units. Verifies that adding the bare NumPy datetime64 NaT (with generic unit) to a boxed timedelta range in either order gives a microsecond-unit all-NaT datetime index; the bare form is needed so the expected unit comes from NumPy's default promotion.
        # GH#23215
        # The bare np.datetime64("NaT") is load-bearing here: the expected
        # M8[us] dtype comes from numpy's default unit promotion from the
        # generic-unit scalar. NumPy nightly emits a DeprecationWarning for
        # the bare form, which the filterwarnings marker ignores.
        other = np.datetime64("NaT")

        tdi = timedelta_range("1 day", periods=3)
        expected = DatetimeIndex(["NaT", "NaT", "NaT"], dtype="M8[us]")

        tdser = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        tm.assert_equal(tdser + other, expected)
        tm.assert_equal(other + tdser, expected)

    def test_td64arr_sub_dt64_array(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies subtracting a NumPy datetime64 array from a boxed timedelta index raises a type error "cannot subtract a datelike from", while the reverse (datetime array minus timedelta index) equals the datetime index minus the timedelta index.
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = TimedeltaIndex(["-1 Day"] * 3)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) - tdi

        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        msg = "cannot subtract a datelike from"
        with pytest.raises(TypeError, match=msg):
            tdi - dtarr

        # TimedeltaIndex.__rsub__
        result = dtarr - tdi
        tm.assert_equal(result, expected)

    def test_td64arr_add_dt64_array(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies adding a NumPy datetime64 array to a boxed timedelta index, in either order, equals the datetime index plus the timedelta index, boxed.
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = TimedeltaIndex(["-1 Day"] * 3)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) + tdi

        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdi + dtarr
        tm.assert_equal(result, expected)
        result = dtarr + tdi
        tm.assert_equal(result, expected)

    # ------------------------------------------------------------------
    # Invalid __add__/__sub__ operations

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("tdi_freq", [None, "h"])
    def test_td64arr_sub_periodlike(
        self, box_with_array, box_with_array2, tdi_freq, pi_freq
    ):
        # [FUNCTION DOC]
        # Linked to issues. Parametrized over timedelta index frequency (none or hourly) and period frequency (D, W, Q, h); uses two box fixtures. Verifies subtracting a boxed period index or a Period scalar from a boxed timedelta index raises a type error matching one of several messages.
        # GH#20049 subtracting PeriodIndex should raise TypeError
        tdi = TimedeltaIndex(["1 hours", "2 hours"], freq=tdi_freq)
        dti = Timestamp("2018-03-07 17:16:40") + tdi
        pi = dti.to_period(pi_freq)
        per = pi[0]

        tdi = tm.box_expected(tdi, box_with_array)
        pi = tm.box_expected(pi, box_with_array2)
        msg = "|".join(
            [
                "cannot subtract",
                "unsupported operand type",
                r"bad operand type for unary \-: 'PeriodArray'",
                r"Input has different freq=-1h from PeriodArray\(.*\)",
                "Cannot add/subtract timedelta-like from PeriodArray",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            tdi - pi

        # GH#13078 subtraction of Period scalar not supported
        with pytest.raises(TypeError, match=msg):
            tdi - per

    @pytest.mark.parametrize(
        "other",
        [
            # GH#12624 for str case
            "a",
            # GH#19123
            1,
            1.5,
            np.array(2),
        ],
    )
    def test_td64arr_addsub_numeric_scalar_invalid(self, box_with_array, other):
        # [FUNCTION DOC]
        # Parametrized over invalid scalar operands (string, integer, float, zero-dimensional integer array); uses the box fixture. Verifies the shared invalid-add/subtract helper's expectations for a boxed timedelta series with a missing value (add and subtract in both orders raise). Vector-like operands are tested in the next test.
        # vector-like others are tested in test_td64arr_add_sub_numeric_arr_invalid
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        tdarr = tm.box_expected(tdser, box_with_array)

        assert_invalid_addsub_type(tdarr, other)

    @pytest.mark.parametrize(
        "vec",
        [
            np.array([1, 2, 3]),
            Index([1, 2, 3]),
            Series([1, 2, 3]),
            DataFrame([[1, 2, 3]]),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_addsub_numeric_arr_invalid(
        self, box_with_array, vec, any_real_numpy_dtype
    ):
        # [FUNCTION DOC]
        # Parametrized over numeric vectors (NumPy array, index, series, frame) and the any-real-dtype fixture; uses the box fixture. Casts each vector to the dtype and verifies the invalid-add/subtract helper's expectations for a boxed timedelta series.
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        tdarr = tm.box_expected(tdser, box_with_array)

        vector = vec.astype(any_real_numpy_dtype)
        assert_invalid_addsub_type(tdarr, vector)

    def test_td64arr_add_sub_int(self, box_with_array, one):
        # [FUNCTION DOC]
        # Uses the box fixture and the one-fixture of integer representations (deprecated behavior, linked to issues). Verifies adding or subtracting the integer to a boxed timedelta range raises a type error "Addition/subtraction of integers", including in-place forms. A TODO notes the in-place operators are not yet covered by the helper.
        # Variants of `one` for #19012, deprecated GH#22535
        rng = timedelta_range("1 days 09:00:00", freq="h", periods=10)
        tdarr = tm.box_expected(rng, box_with_array)

        msg = "Addition/subtraction of integers"
        assert_invalid_addsub_type(tdarr, one, msg)

        # TODO: get inplace ops into assert_invalid_addsub_type
        with pytest.raises(TypeError, match=msg):
            tdarr += one
        with pytest.raises(TypeError, match=msg):
            tdarr -= one

    def test_td64arr_add_sub_integer_array(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to issues. Uses the box fixture. Verifies adding or subtracting a boxed integer array to a boxed timedelta range raises a type error "Addition/subtraction of integers and integer-arrays", and that frames do not dispatch to NumPy's implementation that would treat int64 as timedelta.
        # GH#19959, deprecated GH#22535
        # GH#22696 for DataFrame case, check that we don't dispatch to numpy
        #  implementation, which treats int64 as m8[ns]
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days 09:00:00", freq="h", periods=3)
        tdarr = tm.box_expected(rng, box)
        other = tm.box_expected([4, 3, 2], xbox)

        msg = "Addition/subtraction of integers and integer-arrays"
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_addsub_integer_array_no_freq(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Same as the integer array test but with a timedelta index that has NaT and no frequency, and a different integer array; expects the integer addition/subtraction type error.
        # GH#19959
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        tdi = TimedeltaIndex(["1 Day", "NaT", "3 Hours"])
        tdarr = tm.box_expected(tdi, box)
        other = tm.box_expected([14, -1, 16], xbox)

        msg = "Addition/subtraction of integers"
        assert_invalid_addsub_type(tdarr, other, msg)

    # ------------------------------------------------------------------
    # Operations with timedelta-like others

    def test_td64arr_add_sub_td64_array(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies adding a NumPy timedelta array to a boxed timedelta index (either order) doubles it, and subtracting (either order) gives all zeros.
        box = box_with_array
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = dti - dti.shift(1)
        tdarr = tdi.values

        expected = 2 * tdi
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        result = tdi + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + tdi
        tm.assert_equal(result, expected)

        expected_sub = 0 * tdi
        result = tdi - tdarr
        tm.assert_equal(result, expected_sub)
        result = tdarr - tdi
        tm.assert_equal(result, expected_sub)

    def test_td64arr_add_sub_tdi(self, box_with_array, names):
        # [FUNCTION DOC]
        # Linked to issues. Uses the box and names fixtures (three names). Verifies timedelta index plus/minus a series in either order, with result names given by the expected-name helper, dtype timedelta64[ns], and the expected values (sum of hours and days; differences and the negation for the reversed order).
        # GH#17250 make sure result dtype is correct
        # GH#19043 make sure names are propagated correctly
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["0 days", "1 day"], name=names[1])
        tdi = np.array(tdi) if box in [tm.to_array, pd.array] else tdi
        ser = Series(
            [Timedelta(hours=3), Timedelta(hours=4)], name=names[0], dtype="m8[ns]"
        )
        expected = Series(
            [Timedelta(hours=3), Timedelta(days=1, hours=4)],
            name=exname,
            dtype="m8[ns]",
        )

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = tdi + ser
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        result = ser + tdi
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        expected = Series(
            [Timedelta(hours=-3), Timedelta(days=1, hours=-4)],
            name=exname,
            dtype="m8[ns]",
        )
        expected = tm.box_expected(expected, box)

        result = tdi - ser
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        result = ser - tdi
        tm.assert_equal(result, -expected)
        assert_dtype(result, "timedelta64[ns]")

    @pytest.mark.parametrize("tdnat", [np.timedelta64("NaT", "us"), NaT])
    def test_td64arr_add_sub_td64_nat(self, box_with_array, tdnat):
        # [FUNCTION DOC]
        # Linked to issues. Parametrized over NumPy timedelta NaT with microsecond unit and the NaT singleton; uses the box fixture. Verifies add and subtract in all operand orders between a boxed timedelta index (NaT, one second) and the NaT scalar give an all-NaT microsecond-unit result.
        # GH#18808, GH#23320 special handling for timedelta64("NaT")
        box = box_with_array
        tdi = TimedeltaIndex([NaT, Timedelta("1s")])
        expected = TimedeltaIndex(["NaT"] * 2).as_unit("us")

        obj = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        result = obj + tdnat
        tm.assert_equal(result, expected)
        result = tdnat + obj
        tm.assert_equal(result, expected)
        result = obj - tdnat
        tm.assert_equal(result, expected)
        result = tdnat - obj
        tm.assert_equal(result, expected)

    def test_td64arr_add_timedeltalike(self, two_hours, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the two-hours and box fixtures. Verifies adding the timedelta-like to a boxed daily timedelta range, in either order, shifts by two hours (nanosecond unit if the scalar is a nanosecond Timedelta).
        # only test adding/sub offsets as + is now numeric
        # GH#10699 for Tick cases
        box = box_with_array
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")
        if isinstance(two_hours, Timedelta) and two_hours.unit == "ns":
            expected = expected.as_unit("ns")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)

        result = rng + two_hours
        tm.assert_equal(result, expected)

        result = two_hours + rng
        tm.assert_equal(result, expected)

    def test_td64arr_sub_timedeltalike(self, two_hours, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the two-hours and box fixtures. Verifies subtracting the timedelta-like from a boxed daily timedelta range shifts back by two hours, and the reversed subtraction gives the negation, with nanosecond unit for a nanosecond Timedelta scalar.
        # only test adding/sub offsets as - is now numeric
        # GH#10699 for Tick cases
        box = box_with_array
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")
        if isinstance(two_hours, Timedelta) and two_hours.unit == "ns":
            expected = expected.as_unit("ns")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)

        result = rng - two_hours
        tm.assert_equal(result, expected)

        result = two_hours - rng
        tm.assert_equal(result, -expected)

    # ------------------------------------------------------------------
    # __add__/__sub__ with DateOffsets and arrays of DateOffsets

    def test_td64arr_add_sub_offset_index(
        self, performance_warning, names, box_with_array
    ):
        # [FUNCTION DOC]
        # Linked to issues. Uses the performance-warning, names, and box fixtures. Starts from a timedelta index (named) and an index of an hour offset and minus-two-minutes offset (converted to a NumPy array for array boxes). Verifies adding (either order) and subtracting produce object-dtype results equal to the elementwise sums or differences with the expected name, and that each emits the performance warning.
        # GH#18849, GH#19744
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = Index([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        other = np.array(other) if box in [tm.to_array, pd.array] else other

        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer", name=exname
        )
        expected_sub = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer", name=exname
        )

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box).astype(object)
        expected_sub = tm.box_expected(expected_sub, box).astype(object)

        with tm.assert_produces_warning(performance_warning):
            res = tdi + other
        tm.assert_equal(res, expected)

        with tm.assert_produces_warning(performance_warning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)

        with tm.assert_produces_warning(performance_warning):
            res_sub = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_add_sub_offset_array(self, performance_warning, box_with_array):
        # [FUNCTION DOC]
        # Linked to issues. Uses the performance-warning and box fixtures. Same as the offset-index test but with an unnamed NumPy array of offsets: addition (either order) and subtraction give boxed object-dtype elementwise results, each with the performance warning.
        # GH#18849, GH#18824
        box = box_with_array
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        other = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])

        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer"
        )
        expected_sub = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer"
        )

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box).astype(object)

        with tm.assert_produces_warning(performance_warning):
            res = tdi + other
        tm.assert_equal(res, expected)

        with tm.assert_produces_warning(performance_warning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)

        expected_sub = tm.box_expected(expected_sub, box_with_array).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res_sub = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_with_offset_series(
        self, performance_warning, names, box_with_array
    ):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the performance-warning, names, and box fixtures. Adds and subtracts a series of offsets to a boxed timedelta index (index, array, and array-factory boxes expect a series result): results are object-dtype elementwise sums or differences with expected name, each emitting the performance warning.
        # GH#18849
        box = box_with_array
        box2 = Series if box in [Index, tm.to_array, pd.array] else box
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = Series([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])

        expected_add = Series(
            [tdi[n] + other[n] for n in range(len(tdi))], name=exname, dtype=object
        )
        obj = tm.box_expected(tdi, box)
        expected_add = tm.box_expected(expected_add, box2).astype(object)

        with tm.assert_produces_warning(performance_warning):
            res = obj + other
        tm.assert_equal(res, expected_add)

        with tm.assert_produces_warning(performance_warning):
            res2 = other + obj
        tm.assert_equal(res2, expected_add)

        expected_sub = Series(
            [tdi[n] - other[n] for n in range(len(tdi))], name=exname, dtype=object
        )
        expected_sub = tm.box_expected(expected_sub, box2).astype(object)

        with tm.assert_produces_warning(performance_warning):
            res3 = obj - other
        tm.assert_equal(res3, expected_sub)

    @pytest.mark.parametrize("obox", [np.array, Index, Series])
    def test_td64arr_addsub_anchored_offset_arraylike(
        self, performance_warning, obox, box_with_array
    ):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over offset container (array, index, series); uses the performance-warning and box fixtures. Verifies that adding or subtracting (both orders) an anchored month-end offset and a two-day offset to a boxed timedelta index emits the performance warning and then raises a type error ("has incorrect type" or cannot add the month-end type).
        # GH#18824
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        tdi = tm.box_expected(tdi, box_with_array)

        anchored = obox([offsets.MonthEnd(), offsets.Day(n=2)])

        # addition/subtraction ops with anchored offsets should issue
        # a PerformanceWarning and _then_ raise a TypeError.
        msg = "has incorrect type|cannot add the type MonthEnd"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdi + anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                anchored + tdi
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdi - anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                anchored - tdi

    # ------------------------------------------------------------------
    # Unsorted

    def test_td64arr_add_sub_object_array(self, performance_warning, box_with_array):
        # [FUNCTION DOC]
        # Uses the performance-warning and box fixtures. With an object array of a timedelta, a day offset, and a timestamp: adding it to a boxed timedelta range emits the warning and gives object results (two days, four days, and a timestamp); subtracting it from the range raises a type error (unsupported operand or datelike) after the warning; subtracting the range from the object array gives zero, zero and the timestamp.
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        tdi = timedelta_range("1 day", periods=3, freq="D")
        tdarr = tm.box_expected(tdi, box)

        other = np.array([Timedelta(days=1), offsets.Day(2), Timestamp("2000-01-04")])

        with tm.assert_produces_warning(performance_warning):
            result = tdarr + other

        expected = Index(
            [Timedelta(days=2), Timedelta(days=4), Timestamp("2000-01-07")]
        )
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)

        msg = "unsupported operand type|cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdarr - other

        with tm.assert_produces_warning(performance_warning):
            result = other - tdarr

        expected = Index([Timedelta(0), Timedelta(0), Timestamp("2000-01-01")])
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)


class TestTimedeltaArraylikeMulDivOps:
    # [CLASS DOC]
    # Test class for multiplication, division, and floor division on timedelta64 vectors.
    # Tests for timedelta64[ns]
    # __mul__, __rmul__, __div__, __rdiv__, __floordiv__, __rfloordiv__

    # ------------------------------------------------------------------
    # Multiplication
    # organized with scalar others first, then array-like

    def test_td64arr_mul_int(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies multiplying a boxed timedelta index of zero to four nanoseconds by integer one on either side returns the same values.
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)

        result = idx * 1
        tm.assert_equal(result, idx)

        result = 1 * idx
        tm.assert_equal(result, idx)

    def test_td64arr_mul_tdlike_scalar_raises(self, two_hours, box_with_array):
        # [FUNCTION DOC]
        # Uses the two-hours and box fixtures. Verifies multiplying a boxed timedelta range by a timedelta-like scalar raises a type error matching one of several messages.
        rng = timedelta_range("1 days", "10 days", name="foo")
        rng = tm.box_expected(rng, box_with_array)
        msg = "|".join(
            [
                "argument must be an integer",
                "cannot use operands with types dtype",
                "Cannot multiply with",
                r"unsupported operand type\(s\) for \*",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            rng * two_hours

    def test_tdi_mul_int_array_zerodim(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies multiplying a boxed timedelta index by a zero-dimensional int64 array of five multiplies each value by five.
        rng5 = np.arange(5, dtype="int64")
        idx = TimedeltaIndex(rng5)
        expected = TimedeltaIndex(rng5 * 5)

        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = idx * np.array(5, dtype="int64")
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_array(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies multiplying a boxed timedelta index of zero to four by the NumPy integer array zero to four squares each value.
        rng5 = np.arange(5, dtype="int64")
        idx = TimedeltaIndex(rng5)
        expected = TimedeltaIndex(rng5**2)

        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = idx * rng5
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_series(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies multiplying a boxed timedelta index of zero to four by an int64 series zero to four gives a series (for index and array boxes) of the squares.
        box = box_with_array
        xbox = Series if box in [Index, tm.to_array, pd.array] else box

        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        expected = TimedeltaIndex(np.arange(5, dtype="int64") ** 2)

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, xbox)

        result = idx * Series(np.arange(5, dtype="int64"))
        tm.assert_equal(result, expected)

    def test_tdi_mul_float_series(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies multiplying a boxed timedelta index of zero to four by a float series of one to five gives a series of the products as timedeltas.
        box = box_with_array
        xbox = Series if box in [Index, tm.to_array, pd.array] else box

        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box)

        rng5f = np.arange(5, dtype="float64")
        expected = TimedeltaIndex(rng5f * (rng5f + 1.0))
        expected = tm.box_expected(expected, xbox)

        result = idx * Series(rng5f + 1.0)
        tm.assert_equal(result, expected)

    # TODO: Put Series/DataFrame in others?
    @pytest.mark.parametrize(
        "other",
        [
            np.arange(1, 11),
            Index(np.arange(1, 11), np.int64),
            Index(range(1, 11), np.uint64),
            Index(range(1, 11), np.float64),
            pd.RangeIndex(1, 11),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_tdi_rmul_arraylike(self, other, box_with_array):
        # [FUNCTION DOC]
        # A TODO asks about including series and frames. Parametrized over five numeric array-likes one to ten (NumPy array, int64, uint64, float64 indexes, range index); uses the box fixture. Verifies multiplying the array-like by a boxed timedelta index of ten one-day values, in either order, gives one-day to ten-day timedeltas without a frequency, boxed in the upcast box.
        box = box_with_array

        tdi = TimedeltaIndex(["1 Day"] * 10)
        expected = timedelta_range("1 days", "10 days")._with_freq(None)

        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, other)

        expected = tm.box_expected(expected, xbox)

        result = other * tdi
        tm.assert_equal(result, expected)
        commute = tdi * other
        tm.assert_equal(commute, expected)

    def test_td64arr_mul_bool_scalar_raises(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies multiplying a boxed timedelta by a Python or NumPy bool scalar, in either order, raises a type error "Cannot multiply 'timedelta64[ns]' by bool".
        # GH#58054
        ser = Series(np.arange(5) * timedelta(hours=1), dtype="m8[ns]")
        obj = tm.box_expected(ser, box_with_array)

        msg = r"Cannot multiply 'timedelta64\[ns\]' by bool"
        with pytest.raises(TypeError, match=msg):
            True * obj
        with pytest.raises(TypeError, match=msg):
            obj * True
        with pytest.raises(TypeError, match=msg):
            np.True_ * obj
        with pytest.raises(TypeError, match=msg):
            obj * np.True_

    @pytest.mark.parametrize(
        "dtype",
        [
            bool,
            "boolean",
            pytest.param("bool[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_td64arr_mul_bool_raises(self, dtype, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over boolean dtypes (NumPy bool, nullable boolean, pyarrow bool when installed); uses the box fixture. Verifies multiplying a boxed timedelta by a boxed boolean series raises a type error "Cannot multiply timedelta64[ns] by bool"; the reflected form raises the same error, except for pyarrow booleans which raise an error saying the multiply operation is not supported for the pyarrow bool dtype with timedelta64.
        # KNOWN ISSUE: the variable for the reflected-case message is derived from the first message by replacing the text "rmul" with "mul", but that text never occurs in it, so the replacement is a no-op and both messages are identical (outside the pyarrow case).
        # GH#58054
        ser = Series(np.arange(5) * timedelta(hours=1), dtype="m8[ns]")
        obj = tm.box_expected(ser, box_with_array)

        other = Series(np.arange(5) < 0.5, dtype=dtype)
        other = tm.box_expected(other, box_with_array)

        msg = r"Cannot multiply 'timedelta64\[ns\]' by bool"
        with pytest.raises(TypeError, match=msg):
            obj * other

        msg2 = msg.replace("rmul", "mul")
        if dtype == "bool[pyarrow]":
            # We go through ArrowEA.__mul__ which gives a different message
            msg2 = (
                r"operation 'mul' not supported for dtype 'bool\[pyarrow\]' "
                r"with dtype 'timedelta64\[ns\]'"
            )
        with pytest.raises(TypeError, match=msg2):
            other * obj

    @pytest.mark.parametrize(
        "dtype",
        [
            "Int64",
            "Float64",
            pytest.param("int64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_td64arr_mul_masked(self, dtype, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over Int64, Float64, and pyarrow int64 (when installed); uses the box fixture. Verifies multiplying a boxed timedelta series of zero to four hours by a masked-integer or masked-float series of zero to four, in either order, gives timedeltas of squares of hours (duration pyarrow dtype for pyarrow input).
        ser = Series(np.arange(5) * timedelta(hours=1), dtype="m8[ns]")
        obj = tm.box_expected(ser, box_with_array)

        other = Series(np.arange(5), dtype=dtype)
        other = tm.box_expected(other, box_with_array)

        expected = Series([Timedelta(hours=n**2) for n in range(5)], dtype="m8[ns]")
        expected = tm.box_expected(expected, box_with_array)
        if dtype == "int64[pyarrow]":
            expected = expected.astype("duration[ns][pyarrow]")

        result = obj * other
        tm.assert_equal(result, expected)
        result = other * obj
        tm.assert_equal(result, expected)

    # ------------------------------------------------------------------
    # __div__, __rdiv__

    def test_td64arr_div_nat_invalid(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies dividing a boxed timedelta by NaT raises a type error (unsupported operand), NaT divided by it raises "Cannot divide NaTType by", and division by or of a NumPy datetime64 NaT raises a type error with one of several messages.
        # don't allow division by NaT (maybe could in the future)
        rng = timedelta_range("1 days", "10 days", name="foo")
        rng = tm.box_expected(rng, box_with_array)

        with pytest.raises(TypeError, match="unsupported operand type"):
            rng / NaT
        with pytest.raises(TypeError, match="Cannot divide NaTType by"):
            NaT / rng

        dt64nat = np.datetime64("NaT", "ns")
        msg = "|".join(
            [
                # 'divide' on npdev as of 2021-12-18
                "ufunc '(true_divide|divide)' cannot use operands",
                "cannot perform __r?truediv__",
                "Cannot divide datetime64 by TimedeltaArray",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            rng / dt64nat
        with pytest.raises(TypeError, match=msg):
            dt64nat / rng

    def test_td64arr_div_td64nat(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies dividing a boxed timedelta range by a NumPy timedelta NaT, and the reverse, gives an all-NaN float result.
        # GH#23829
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days", "10 days")
        rng = tm.box_expected(rng, box)

        other = np.timedelta64("NaT", "ns")

        expected = np.array([np.nan] * 10)
        expected = tm.box_expected(expected, xbox)

        result = rng / other
        tm.assert_equal(result, expected)

        result = other / rng
        tm.assert_equal(result, expected)

    def test_td64arr_div_int(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies dividing a boxed timedelta index by integer one returns the same values, and integer one divided by the index raises a type error "Cannot divide".
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)

        result = idx / 1
        tm.assert_equal(result, idx)

        with pytest.raises(TypeError, match="Cannot divide"):
            # GH#23829
            1 / idx

    def test_td64arr_div_tdlike_scalar(self, two_hours, box_with_array):
        # [FUNCTION DOC]
        # Linked to issues (ensures frames return the correct dtype). Uses the two-hours and box fixtures. Verifies a boxed ten-day timedelta range divided by two hours gives float values 12, 24, ..., 120 with the name kept (array factory boxes give a NumPy array), and the reversed division gives the reciprocal.
        # GH#20088, GH#22163 ensure DataFrame returns correct dtype
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days", "10 days", name="foo")
        expected = Index((np.arange(10) + 1) * 12, dtype=np.float64, name="foo")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        result = rng / two_hours
        tm.assert_equal(result, expected)

        result = two_hours / rng
        expected = 1 / expected
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("m", [1, 3, 10])
    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_td64arr_div_td64_scalar(self, m, unit, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over multiples one, three, ten and units D, h, m, s, ms, us, ns; uses the box fixture. Verifies dividing a series of 59-day timedeltas (last value NaN) by a NumPy timedelta scalar gives the elementwise float quotients, and the reflected division gives the elementwise reverse quotient.
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        ser = Series([Timedelta(days=59)] * 3)
        ser[2] = np.nan
        flat = ser
        ser = tm.box_expected(ser, box)

        # op
        expected = Series([x / np.timedelta64(m, unit) for x in flat])
        expected = tm.box_expected(expected, xbox)
        result = ser / np.timedelta64(m, unit)
        tm.assert_equal(result, expected)

        # reverse op
        expected = Series([Timedelta(np.timedelta64(m, unit)) / x for x in flat])
        expected = tm.box_expected(expected, xbox)
        result = np.timedelta64(m, unit) / ser
        tm.assert_equal(result, expected)

    def test_td64arr_div_tdlike_scalar_with_nat(self, two_hours, box_with_array):
        # [FUNCTION DOC]
        # Uses the two-hours and box fixtures. Verifies dividing a timedelta index with NaT by two hours gives twelve, NaN, twenty-four as float with the name, and the reversed division gives the reciprocal.
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        expected = Index([12, np.nan, 24], dtype=np.float64, name="foo")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        result = rng / two_hours
        tm.assert_equal(result, expected)

        result = two_hours / rng
        expected = 1 / expected
        tm.assert_equal(result, expected)

    def test_td64arr_div_td64_ndarray(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies dividing a boxed timedelta index with NaT by a NumPy timedelta array (hours), a boxed version, an object-dtype version (object result), and a list gives twelve, NaN, twenty-four as floats; the reversed division gives the reciprocal.
        # GH#22631
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = TimedeltaIndex(["1 days", NaT, "2 days"])
        expected = Index([12, np.nan, 24], dtype=np.float64)

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        other = np.array([2, 4, 2], dtype="m8[h]")
        result = rng / other
        tm.assert_equal(result, expected)

        result = rng / tm.box_expected(other, box)
        tm.assert_equal(result, expected)

        result = rng / other.astype(object)
        tm.assert_equal(result, expected.astype(object))

        result = rng / list(other)
        tm.assert_equal(result, expected)

        # reversed op
        expected = 1 / expected
        result = other / rng
        tm.assert_equal(result, expected)

        result = tm.box_expected(other, box) / rng
        tm.assert_equal(result, expected)

        result = other.astype(object) / rng
        tm.assert_equal(result, expected)

        result = list(other) / rng
        tm.assert_equal(result, expected)

    def test_tdarr_div_length_mismatch(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. For a boxed timedelta index of three, verifies dividing by (or being divided by) a list, NumPy array, or index of the wrong length (four elements, or two) raises a value error saying vectors cannot be divided or cannot be coerced to a series.
        rng = TimedeltaIndex(["1 days", NaT, "2 days"])
        mismatched = [1, 2, 3, 4]

        rng = tm.box_expected(rng, box_with_array)
        msg = "Cannot divide vectors|Unable to coerce to Series"
        for obj in [mismatched, mismatched[:2]]:
            # one shorter, one longer
            for other in [obj, np.array(obj), Index(obj)]:
                with pytest.raises(ValueError, match=msg):
                    rng / other
                with pytest.raises(ValueError, match=msg):
                    other / rng

    def test_td64_div_object_mixed_result(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies division and floor division of a boxed timedelta index (with NaT) by an object array of a timedelta, floats, and a timedelta gives an object result whose NaT is a microsecond NumPy timedelta NaT (not the NaT singleton): quotient values one, NaT, a timedelta, one and a half for division, and one, NaT, a timedelta, one for floor division; for frames it also checks the element type.
        # Case where we having a NaT in the result inseat of timedelta64("NaT")
        #  is misleading
        orig = timedelta_range("1 Day", periods=3).insert(1, NaT)
        tdi = tm.box_expected(orig, box_with_array, transpose=False)

        other = np.array([orig[0], 1.5, 2.0, orig[2]], dtype=object)
        other = tm.box_expected(other, box_with_array, transpose=False)

        res = tdi / other

        expected = Index([1.0, np.timedelta64("NaT", "us"), orig[0], 1.5], dtype=object)
        expected = tm.box_expected(expected, box_with_array, transpose=False)
        if isinstance(expected, NumpyExtensionArray):
            expected = expected.to_numpy()
        tm.assert_equal(res, expected)
        if box_with_array is DataFrame:
            # We have an np.timedelta64(NaT), not pd.NaT
            assert isinstance(res.iloc[1, 0], np.timedelta64)

        res = tdi // other

        expected = Index([1, np.timedelta64("NaT", "us"), orig[0], 1], dtype=object)
        expected = tm.box_expected(expected, box_with_array, transpose=False)
        if isinstance(expected, NumpyExtensionArray):
            expected = expected.to_numpy()
        tm.assert_equal(res, expected)
        if box_with_array is DataFrame:
            # We have an np.timedelta64(NaT), not pd.NaT
            assert isinstance(res.iloc[1, 0], np.timedelta64)

    # ------------------------------------------------------------------
    # __floordiv__, __rfloordiv__

    @pytest.mark.skipif(WASM, reason="no fp exception support in wasm")
    def test_td64arr_floordiv_td64arr_with_nat(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Skipped on WebAssembly (no floating-point exceptions). Uses the box fixture. Verifies floor dividing a timedelta series by one with NaT gives one, one, NaN as floats (a runtime warning is expected for the array-factory box), including the reflected path through a NumPy array of the left side.
        # GH#35529
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        left = Series([1000, 222330, 30], dtype="timedelta64[ns]")
        right = Series([1000, 222330, None], dtype="timedelta64[ns]")

        left = tm.box_expected(left, box)
        right = tm.box_expected(right, box)

        expected = np.array([1.0, 1.0, np.nan], dtype=np.float64)
        expected = tm.box_expected(expected, xbox)

        with tm.maybe_produces_warning(
            RuntimeWarning, box is pd.array, check_stacklevel=False
        ):
            result = left // right

        tm.assert_equal(result, expected)

        # case that goes through __rfloordiv__ with arraylike
        with tm.maybe_produces_warning(
            RuntimeWarning, box is pd.array, check_stacklevel=False
        ):
            result = np.asarray(left) // right
        tm.assert_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_td64arr_floordiv_tdscalar(self, box_with_array, scalar_td):
        # [FUNCTION DOC]
        # Linked to issues. Ignores a runtime warning about invalid values. Uses the box and timedelta-scalar fixtures. For a series of 5 minutes 3 seconds (twice) and NaT, floor dividing by the scalar gives zero, zero, NaN, the reflected division gives two, two, NaN (the scalar being about twice the value plus one second), and calling the reflected method explicitly gives the same.
        # GH#18831, GH#19125
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        td = Timedelta("5m3s")  # i.e. (scalar_td - 1sec) / 2

        td1 = Series([td, td, NaT], dtype="m8[ns]")
        td1 = tm.box_expected(td1, box, transpose=False)

        expected = Series([0, 0, np.nan])
        expected = tm.box_expected(expected, xbox, transpose=False)

        result = td1 // scalar_td
        tm.assert_equal(result, expected)

        # Reversed op
        expected = Series([2, 2, np.nan])
        expected = tm.box_expected(expected, xbox, transpose=False)

        result = scalar_td // td1
        tm.assert_equal(result, expected)

        # same thing buts let's be explicit about calling __rfloordiv__
        result = td1.__rfloordiv__(scalar_td)
        tm.assert_equal(result, expected)

    def test_td64arr_floordiv_int(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies floor dividing a boxed timedelta index by integer one returns the same values, and one floor-divided by the index raises a type error ("floor_divide cannot use operands" or cannot divide int by Timedelta).
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        result = idx // 1
        tm.assert_equal(result, idx)

        pattern = "floor_divide cannot use operands|Cannot divide int by Timedelta*"
        with pytest.raises(TypeError, match=pattern):
            1 // idx

    # ------------------------------------------------------------------
    # mod, divmod
    # TODO: operations with timedelta-like arrays, numeric arrays,
    #  reversed ops

    def test_td64arr_mod_tdscalar(
        self, performance_warning, box_with_array, three_days
    ):
        # [FUNCTION DOC]
        # A TODO says operations with timedelta-like arrays, numeric arrays, and reversed ops are untested. Uses the performance-warning, box, and three-days fixtures. For a range of one to nine days: if the scalar is a Day offset, modulo, divmod, and floor division raise a type error about unsupported operand; otherwise modulo by the scalar gives one, two, zero days repeating; divmod's remainder equals that (frames with a date offset expect object dtype) and its quotient equals the floor division, with the performance warning expected only in the frame and date-offset case.
        # KNOWN ISSUE: a TODO in the body notes that making the expected result object dtype for frames is an artifact of the frame divmod method being implemented naively without dispatching to the underlying array.
        tdi = timedelta_range("1 Day", "9 days")
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = TimedeltaIndex(["1 Day", "2 Days", "0 Days"] * 3)
        expected = tm.box_expected(expected, box_with_array)

        if isinstance(three_days, offsets.Day):
            msg = "unsupported operand type"
            with pytest.raises(TypeError, match=msg):
                tdarr % three_days
            with pytest.raises(TypeError, match=msg):
                divmod(tdarr, three_days)
            with pytest.raises(TypeError, match=msg):
                tdarr // three_days
            return

        result = tdarr % three_days
        tm.assert_equal(result, expected)

        if box_with_array is DataFrame and isinstance(three_days, pd.DateOffset):
            # TODO: making expected be object here a result of DataFrame.__divmod__
            #  being defined in a naive way that does not dispatch to the underlying
            #  array's __divmod__
            expected = expected.astype(object)
        else:
            performance_warning = False

        with tm.assert_produces_warning(performance_warning):
            result = divmod(tdarr, three_days)

        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // three_days)

    def test_td64arr_mod_int(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies a nanosecond timedelta range modulo 2 gives one, zero nanoseconds repeating; integer modulo timedelta raises a type error "Cannot divide int by"; divmod by 2 gives remainder as above and quotient equal to floor division.
        tdi = timedelta_range("1 ns", "10 ns", periods=10)
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = TimedeltaIndex(["1 ns", "0 ns"] * 5)
        expected = tm.box_expected(expected, box_with_array)

        result = tdarr % 2
        tm.assert_equal(result, expected)

        msg = "Cannot divide int by"
        with pytest.raises(TypeError, match=msg):
            2 % tdarr

        result = divmod(tdarr, 2)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // 2)

    def test_td64arr_rmod_tdscalar(self, box_with_array, three_days):
        # [FUNCTION DOC]
        # Uses the box and three-days fixtures. For a range of one to nine days, three-days scalar modulo the range gives zero, one, zero days then three days six times; a Day offset instead raises a type error "Cannot divide Day by TimedeltaArray"; the divmod remainder equals the modulo and the quotient equals the floor division.
        tdi = timedelta_range("1 Day", "9 days")
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = ["0 Days", "1 Day", "0 Days"] + ["3 Days"] * 6
        expected = TimedeltaIndex(expected)
        expected = tm.box_expected(expected, box_with_array)

        if isinstance(three_days, offsets.Day):
            msg = "Cannot divide Day by TimedeltaArray"
            with pytest.raises(TypeError, match=msg):
                three_days % tdarr
            return

        result = three_days % tdarr
        tm.assert_equal(result, expected)

        result = divmod(three_days, tdarr)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], three_days // tdarr)

    # ------------------------------------------------------------------
    # Operations with invalid others

    def test_td64arr_mul_tdscalar_invalid(self, box_with_array, scalar_td):
        # [FUNCTION DOC]
        # Uses the box and timedelta-scalar fixtures. Verifies multiplying a boxed timedelta series (with NaN) by a timedelta-like scalar in either order raises a type error matching operate/unsupported/cannot/not supported.
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        td1 = tm.box_expected(td1, box_with_array)

        # check that we are getting a TypeError
        # with 'operate' (from core/ops.py) for the ops that are not
        # defined
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            td1 * scalar_td
        with pytest.raises(TypeError, match=pattern):
            scalar_td * td1

    def test_td64arr_mul_too_short_raises(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies multiplying a boxed timedelta index of five by an index of three raises a type error (length check happens before dtype check), and by a two-element NumPy array raises a value error; the messages come from a set of accepted alternatives.
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        msg = "|".join(
            [
                "cannot use operands with types dtype",
                "Cannot multiply with unequal lengths",
                "Unable to coerce to Series",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            # length check before dtype check
            idx * idx[:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    def test_td64arr_mul_td64arr_raises(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies multiplying a boxed timedelta index by itself raises a type error "cannot use operands with types dtype".
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            idx * idx

    # ------------------------------------------------------------------
    # Operations with numeric others

    def test_td64arr_mul_numeric_scalar(self, box_with_array, one):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture and the one-fixture of integer representations. Verifies multiplying a boxed timedelta series (59 days twice and NaT) by minus one and by two (either order) gives minus 59 days and 118 days with NaT preserved.
        # GH#4521
        # divide/multiply by integers
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        expected = Series(["-59 Days", "-59 Days", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdser * (-one)
        tm.assert_equal(result, expected)
        result = (-one) * tdser
        tm.assert_equal(result, expected)

        expected = Series(["118 Days", "118 Days", "NaT"], dtype="timedelta64[ns]")
        expected = tm.box_expected(expected, box_with_array)

        result = tdser * (2 * one)
        tm.assert_equal(result, expected)
        result = (2 * one) * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("two", [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_div_numeric_scalar(self, box_with_array, two):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over two as an integer, float, and zero-dimensional arrays of each; uses the box fixture. Verifies dividing a boxed series of 59 days twice and NaT by two gives 29.5 days twice and NaT, while the scalar divided by the series raises a type error "Cannot divide".
        # GH#4521
        # divide/multiply by integers
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        expected = Series(["29.5D", "29.5D", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdser / two
        tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match="Cannot divide"):
            two / tdser

    @pytest.mark.parametrize("two", [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_floordiv_numeric_scalar(self, box_with_array, two):
        # [FUNCTION DOC]
        # Parametrized over two as an integer, float, and zero-dimensional arrays. Verifies floor dividing the 59-day series by two gives 29.5 days twice and NaT (same result as true division), and two floor-divided by the series raises a type error "Cannot divide".
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        expected = Series(["29.5D", "29.5D", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdser // two
        tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match="Cannot divide"):
            two // tdser

    @pytest.mark.parametrize(
        "klass",
        [np.array, Index, Series],
        ids=lambda x: x.__name__,
    )
    def test_td64arr_rmul_numeric_array(
        self,
        box_with_array,
        klass,
        any_real_numpy_dtype,
    ):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over vector container (array, index, series) and the any-real-dtype fixture; uses the box fixture. Verifies multiplying the boxed 59-day series by the vector of twenty, thirty, forty (cast to the dtype), in either order, gives 1180, 1770 days and NaT in the upcast box.
        # GH#4521
        # divide/multiply by integers

        vector = klass([20, 30, 40])
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        vector = vector.astype(any_real_numpy_dtype)

        expected = Series(["1180 Days", "1770 Days", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        xbox = get_upcast_box(tdser, vector)

        expected = tm.box_expected(expected, xbox)

        result = tdser * vector
        tm.assert_equal(result, expected)

        result = vector * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "klass",
        [np.array, Index, Series],
        ids=lambda x: x.__name__,
    )
    def test_td64arr_div_numeric_array(
        self, box_with_array, klass, any_real_numpy_dtype
    ):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over vector container and the any-real-dtype fixture; uses the box fixture. Verifies dividing the 59-day series (with NaT) by the vector gives 2.95 days, 1 day 23:12, and NaT; vector divided by the series raises a type error from several accepted messages; dividing by an object-dtype version gives an object result of elementwise quotients with a NumPy NaT; and an object vector divided by the series raises a type error.
        # GH#4521
        # divide/multiply by integers

        vector = klass([20, 30, 40])
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        vector = vector.astype(any_real_numpy_dtype)

        expected = Series(["2.95D", "1D 23h 12m", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        xbox = get_upcast_box(tdser, vector)
        expected = tm.box_expected(expected, xbox)

        result = tdser / vector
        tm.assert_equal(result, expected)

        pattern = "|".join(
            [
                "true_divide'? cannot use operands",
                "cannot perform __div__",
                "cannot perform __truediv__",
                "unsupported operand",
                "Cannot divide",
                "ufunc 'divide' cannot use operands with types",
            ]
        )
        with pytest.raises(TypeError, match=pattern):
            vector / tdser

        result = tdser / vector.astype(object)
        if box_with_array is DataFrame:
            expected = [tdser.iloc[0, n] / vector[n] for n in range(len(vector))]
            expected = tm.box_expected(expected, xbox).astype(object)
            expected[2] = expected[2].fillna(np.timedelta64("NaT", "ns"))
        else:
            expected = [tdser[n] / vector[n] for n in range(len(tdser))]
            expected = [
                x if x is not NaT else np.timedelta64("NaT", "ns") for x in expected
            ]
            if xbox is tm.to_array:
                expected = tm.to_array(expected).astype(object)
            else:
                expected = xbox(expected, dtype=object)

        tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match=pattern):
            vector.astype(object) / tdser

    def test_td64arr_mul_int_series(self, box_with_array, names):
        # [FUNCTION DOC]
        # Linked to an issue about correct name attachment. Uses the box and names fixtures. Multiplies a named timedelta index (zero to four days) by a named int64 series zero to four, in either order; verifies the result equals the expected microsecond-unit timedelta series of squares in days with the name given by the name helper, in the upcast box. A TODO asks about parametrizing over series types.
        # GH#19042 test for correct name attachment
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(
            ["0days", "1day", "2days", "3days", "4days"], name=names[0]
        )
        # TODO: Should we be parametrizing over types for `ser` too?
        ser = Series([0, 1, 2, 3, 4], dtype=np.int64, name=names[1])

        expected = Series(
            ["0days", "1day", "4days", "9days", "16days"],
            dtype="timedelta64[us]",
            name=exname,
        )

        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, ser)

        expected = tm.box_expected(expected, xbox)

        result = ser * tdi
        tm.assert_equal(result, expected)

        result = tdi * ser
        tm.assert_equal(result, expected)

    # TODO: Should we be parametrizing over types for `ser` too?
    def test_float_series_rdiv_td64arr(self, box_with_array, names):
        # [FUNCTION DOC]
        # Linked to an issue. A TODO asks about parametrizing over series types. Uses the box and names fixtures. Verifies that the series reflected true-division method applied to a boxed timedelta index gives not-implemented for frames, otherwise the expected series of timedelta divided by float elements with microsecond unit and the name chosen by box (the series' name for array boxes, else the third name).
        # GH#19042 test for correct name attachment
        box = box_with_array
        tdi = TimedeltaIndex(
            ["0days", "1day", "2days", "3days", "4days"], name=names[0]
        )
        ser = Series([1.5, 3, 4.5, 6, 7.5], dtype=np.float64, name=names[1])

        xname = names[2] if box not in [tm.to_array, pd.array] else names[1]
        expected = Series(
            [tdi[n] / ser[n] for n in range(len(ser))],
            dtype="timedelta64[us]",
            name=xname,
        )

        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, ser)
        expected = tm.box_expected(expected, xbox)

        result = ser.__rtruediv__(tdi)
        if box is DataFrame:
            assert result is NotImplemented
        else:
            tm.assert_equal(result, expected)

    def test_td64arr_all_nat_div_object_dtype_numeric(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies dividing and floor dividing a boxed all-NaT timedelta index by an object array of an integer and a float gives an object result of NumPy timedelta NaT values (inferred as timedelta results; filled in for series and frames).
        # GH#39750 make sure we infer the result as td64
        tdi = TimedeltaIndex([NaT, NaT], dtype="m8[ns]")

        left = tm.box_expected(tdi, box_with_array)
        right = np.array([2, 2.0], dtype=object)

        tdnat = np.timedelta64("NaT", "ns")
        expected = Index([tdnat] * 2, dtype=object)
        if box_with_array is not Index:
            expected = tm.box_expected(expected, box_with_array).astype(object)
            if box_with_array in [Series, DataFrame]:
                expected = expected.fillna(tdnat)  # GH#18463

        result = left / right
        tm.assert_equal(result, expected)

        result = left // right
        tm.assert_equal(result, expected)


class TestTimedelta64ArrayLikeArithmetic:
    # [CLASS DOC]
    # Arithmetic tests for timedelta64 vectors fully parametrized over frame, series, timedelta index, and timedelta array; intended to eventually hold all arithmetic tests.
    # Arithmetic tests for timedelta64[ns] vectors fully parametrized over
    #  DataFrame/Series/TimedeltaIndex/TimedeltaArray.  Ideally all arithmetic
    #  tests will eventually end up here.

    def test_td64arr_pow_invalid(self, scalar_td, box_with_array):
        # [FUNCTION DOC]
        # Uses the timedelta-scalar and box fixtures. Verifies raising a timedelta-like scalar to the boxed timedelta series, and the series to the scalar, raises a type error matching operate/unsupported/cannot/not supported.
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        td1 = tm.box_expected(td1, box_with_array)

        # check that we are getting a TypeError
        # with 'operate' (from core/ops.py) for the ops that are not
        # defined
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            scalar_td**td1

        with pytest.raises(TypeError, match=pattern):
            td1**scalar_td


def test_add_timestamp_to_timedelta():
    # [FUNCTION DOC]
    # Linked to an issue. Verifies adding a timestamp to a timedelta range of 31 periods between zero and one second gives a datetime index equal to adding to the timestamp an expected timedelta per position (about a thirty-third of a second per step plus a microsecond added once every three steps), i.e. rounding of the stepped durations is handled consistently.
    # GH: 35897
    timestamp = Timestamp("2021-01-01")
    result = timestamp + timedelta_range("0s", "1s", periods=31)
    expected = DatetimeIndex(
        [
            timestamp
            + (
                pd.to_timedelta("0.033333s") * i
                + pd.to_timedelta("0.000001s") * divmod(i, 3)[0]
            )
            for i in range(31)
        ]
    )
    tm.assert_index_equal(result, expected)
