# [MODULE DOC]
# Arithmetic and comparison tests for datetime64 and timezone-aware datetime64 dtypes across arrays, indexes, series, and frames that should behave identically.
# Covers comparisons (scalars, invalid operands, NaT, arrays, tz mismatches), date range and index arithmetic with integers, timedelta-likes, offsets and periods, subtraction semantics (including time zone and unit rules), reflected operations, NumPy datetime64 arrays, DST handling, offset-array operations with performance warnings, business offsets, and Timestamp-specific behaviors. Uses shared fixtures for timezones (naive and aware), box types, performance warnings, and names, plus shared helper functions for invalid comparison and invalid add/subtract assertions.
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for datetime64 and datetime64tz dtypes
from datetime import (
    datetime,
    time,
    timedelta,
    timezone,
)
from itertools import (
    product,
)
import operator

import numpy as np
import pytest

from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months

import pandas as pd
from pandas import (
    DateOffset,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
    assert_cannot_add,
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)

# ------------------------------------------------------------------
# Comparisons


class TestDatetime64ArrayLikeComparisons:
    # [CLASS DOC]
    # Comparison tests for datetime64 vectors fully parametrized over frame, series, datetime index, and datetime array boxes. Intended eventually to hold all datetime comparison tests.
    # Comparison tests for datetime64 vectors fully parametrized over
    #  DataFrame/Series/DatetimeIndex/DatetimeArray.  Ideally all comparison
    #  tests will eventually end up here.

    def test_compare_zerodim(self, tz_naive_fixture, box_with_array):
        # [FUNCTION DOC]
        # Uses the timezone fixture (naive and aware) and the box fixture. Verifies a boxed three-day date range compared by less-or-equal with a zero-dimensional NumPy array holding its first element unboxes the array: true only for the first element.
        # Test comparison with zero-dimensional array is unboxed
        tz = tz_naive_fixture
        box = box_with_array
        dti = date_range("20130101", periods=3, tz=tz)

        other = np.array(dti.to_numpy()[0])

        dtarr = tm.box_expected(dti, box)
        xbox = get_upcast_box(dtarr, other, True)
        result = dtarr <= other
        expected = np.array([True, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            "foo",
            -1,
            99,
            4.0,
            object(),
            timedelta(days=2),
            # GH#19800, GH#19301 datetime.date comparison raises to
            #  match DatetimeIndex/Timestamp.  This also matches the behavior
            #  of stdlib datetime.datetime
            datetime(2001, 1, 1).date(),
            # GH#19301 None and NaN are *not* cast to NaT for comparisons
            None,
            np.nan,
        ],
    )
    def test_dt64arr_cmp_scalar_invalid(self, other, tz_naive_fixture, box_with_array):
        # [FUNCTION DOC]
        # Linked to issues. Parametrized over invalid scalars: a string, integers, a float, a plain object, a timedelta, a date (comparison with a stdlib date raises to match timestamps), None and NaN (which are not cast to NaT for comparison). Uses the timezone and box fixtures. Verifies the shared invalid-comparison helper's expectations for a boxed ten-day date range (equality false, inequality true, ordering raises).
        # GH#22074, GH#15966
        tz = tz_naive_fixture

        rng = date_range("1/1/2000", periods=10, tz=tz)
        dtarr = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(dtarr, other, box_with_array)

    @pytest.mark.parametrize(
        "other",
        [
            # GH#4968 invalid date/int comparisons
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.timedelta_range("1ns", periods=10).array,
            np.array(pd.timedelta_range("1ns", periods=10)),
            list(pd.timedelta_range("1ns", periods=10)),
            pd.timedelta_range("1 Day", periods=10).astype(object),
            pd.period_range("1971-01-01", freq="D", periods=10).array,
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),
        ],
    )
    def test_dt64arr_cmp_arraylike_invalid(
        self, other, tz_naive_fixture, box_with_array
    ):
        # [FUNCTION DOC]
        # Parametrized over invalid list-likes (integer lists and arrays, float and object arrays, timedelta arrays in several containers, period arrays and objects). Uses the timezone and box fixtures. Verifies the invalid-comparison helper's expectations on a boxed nanosecond-frequency datetime array.
        tz = tz_naive_fixture

        dta = date_range("2000-01-01", freq="ns", periods=10, tz=tz)._data
        obj = tm.box_expected(dta, box_with_array)
        assert_invalid_comparison(obj, other, box_with_array)

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Uses the timezone fixture. Verifies comparing an hourly datetime array to an object array holding integers, a matching timestamp, and a timedelta: equality true only at the matching element, inequality the negation, and ordering comparisons raise a type error (Invalid comparison, cannot compare type, or not supported between).
        tz = tz_naive_fixture

        dta = date_range("2000-01-01", freq="h", periods=5, tz=tz)._data
        other = np.array([0, 1, 2, dta[3], Timedelta(days=1)])
        result = dta == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = dta != other
        tm.assert_numpy_array_equal(result, ~expected)

        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):
            dta < other
        with pytest.raises(TypeError, match=msg):
            dta > other
        with pytest.raises(TypeError, match=msg):
            dta <= other
        with pytest.raises(TypeError, match=msg):
            dta >= other

    def test_dt64arr_nat_comparison(self, tz_naive_fixture, box_with_array):
        # [FUNCTION DOC]
        # Linked to issues (frames once wrongly considered NaT equal to a timestamp). Uses the timezone and box fixtures. Verifies equality of a boxed series of a timestamp and NaT with the timestamp is true then false.
        # GH#22242, GH#22163 DataFrame considered NaT == ts incorrectly
        tz = tz_naive_fixture
        box = box_with_array

        ts = Timestamp("2021-01-01", tz=tz)
        ser = Series([ts, NaT])

        obj = tm.box_expected(ser, box)
        xbox = get_upcast_box(obj, ts, True)

        expected = Series([True, False], dtype=np.bool_)
        expected = tm.box_expected(expected, xbox)

        result = obj == ts
        tm.assert_equal(result, expected)


class TestDatetime64SeriesComparison:
    # [CLASS DOC]
    # Test class for comparison tests moved from the series operator tests; a TODO says it needs cleanup.
    # TODO: moved from tests.series.test_operators; needs cleanup

    @pytest.mark.parametrize(
        "pair",
        [
            (
                [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],
                [NaT, NaT, Timestamp("2011-01-03")],
            ),
            (
                [Timedelta("1 days"), NaT, Timedelta("3 days")],
                [NaT, NaT, Timedelta("3 days")],
            ),
            (
                [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],
                [NaT, NaT, Period("2011-03", freq="M")],
            ),
        ],
    )
    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("dtype", [None, object])
    @pytest.mark.parametrize(
        "op, expected",
        [
            (operator.eq, [False, False, True]),
            (operator.ne, [True, True, False]),
            (operator.lt, [False, False, False]),
            (operator.gt, [False, False, False]),
            (operator.ge, [False, False, True]),
            (operator.le, [False, False, True]),
        ],
    )
    def test_nat_comparisons(
        self,
        dtype,
        index_or_series,
        reverse,
        pair,
        op,
        expected,
    ):
        # [FUNCTION DOC]
        # Parametrized over three kinds of pairs (timestamps, timedeltas, periods, each with NaT), reverse flag, dtype none or object, the six comparison operators with expected booleans, and the index-or-series fixture for the right side. Verifies the operator on a left series and a right container gives the expected booleans, where NaT is never equal and compares false for ordering, and is unequal to everything.
        box = index_or_series
        lhs, rhs = pair
        if reverse:
            # add lhs / rhs switched data
            lhs, rhs = rhs, lhs

        left = Series(lhs, dtype=dtype)
        right = box(rhs, dtype=dtype)

        result = op(left, right)

        tm.assert_series_equal(result, Series(expected))

    @pytest.mark.parametrize(
        "data",
        [
            [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],
            [Timedelta("1 days"), NaT, Timedelta("3 days")],
            [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],
        ],
    )
    @pytest.mark.parametrize("dtype", [None, object])
    def test_nat_comparisons_scalar(self, dtype, data, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over data kinds (timestamps, timedeltas, periods, with NaT) and dtype none or object; uses the box fixture. Verifies comparing a boxed series to the NaT scalar on either side: equality false, inequality true, all ordering comparisons false in both directions; for the array factory with object dtype the expected is a plain bool array.
        box = box_with_array

        left = Series(data, dtype=dtype)
        left = tm.box_expected(left, box)
        xbox = get_upcast_box(left, NaT, True)

        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")

        tm.assert_equal(left == NaT, expected)
        tm.assert_equal(NaT == left, expected)

        expected = [True, True, True]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")
        tm.assert_equal(left != NaT, expected)
        tm.assert_equal(NaT != left, expected)

        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")
        tm.assert_equal(left < NaT, expected)
        tm.assert_equal(NaT > left, expected)
        tm.assert_equal(left <= NaT, expected)
        tm.assert_equal(NaT >= left, expected)

        tm.assert_equal(left > NaT, expected)
        tm.assert_equal(NaT < left, expected)
        tm.assert_equal(left >= NaT, expected)
        tm.assert_equal(NaT <= left, expected)

    @pytest.mark.parametrize("val", [datetime(2000, 1, 4), datetime(2000, 1, 5)])
    def test_series_comparison_scalars(self, val):
        # [FUNCTION DOC]
        # Parametrized over two datetime scalars. Verifies a ten-day date series compared greater-than the scalar equals the elementwise comparison of each element.
        series = Series(date_range("1/1/2000", periods=10))

        result = series > val
        expected = Series([x > val for x in series])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "left,right", [("lt", "gt"), ("le", "ge"), ("eq", "eq"), ("ne", "ne")]
    )
    def test_timestamp_compare_series(self, left, right):
        # [FUNCTION DOC]
        # See an issue about comparing timestamps on both sides. Parametrized over operator name pairs (less/greater, less-or-equal/greater-or-equal, equal, not equal). Verifies a series with NaT compared with a timestamp or NaT on the left gives the same result as the mirrored operator with the scalar on the right, for a series with and without NaT in the first and fourth positions.
        # see gh-4982
        # Make sure we can compare Timestamps on the right AND left hand side.
        ser = Series(date_range("20010101", periods=10), name="dates")
        s_nat = ser.copy(deep=True)

        ser[0] = Timestamp("nat")
        ser[3] = Timestamp("nat")

        left_f = getattr(operator, left)
        right_f = getattr(operator, right)

        # No NaT
        expected = left_f(ser, Timestamp("20010109"))
        result = right_f(Timestamp("20010109"), ser)
        tm.assert_series_equal(result, expected)

        # NaT
        expected = left_f(ser, Timestamp("nat"))
        result = right_f(Timestamp("nat"), ser)
        tm.assert_series_equal(result, expected)

        # Compare to Timestamp with series containing NaT
        expected = left_f(s_nat, Timestamp("20010109"))
        result = right_f(Timestamp("20010109"), s_nat)
        tm.assert_series_equal(result, expected)

        # Compare to NaT with series containing NaT
        expected = left_f(s_nat, NaT)
        result = right_f(NaT, s_nat)
        tm.assert_series_equal(result, expected)

    def test_dt64arr_timestamp_equality(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies that comparing a boxed series of two timestamps and NaT with itself gives not-equal false, false, true and equal true, true, false; comparisons with a single element of the series (a timestamp, or NaT) give the expected broadcast results, except for frames where comparing with a series raises a value error "not aligned".
        # GH#11034
        box = box_with_array

        ser = Series([Timestamp("2000-01-29 01:59:00"), Timestamp("2000-01-30"), NaT])
        ser = tm.box_expected(ser, box)
        xbox = get_upcast_box(ser, ser, True)

        result = ser != ser
        expected = tm.box_expected([False, False, True], xbox)
        tm.assert_equal(result, expected)

        if box is pd.DataFrame:
            # alignment for frame vs series comparisons deprecated
            #  in GH#46795 enforced 2.0
            with pytest.raises(ValueError, match="not aligned"):
                ser != ser[0]

        else:
            result = ser != ser[0]
            expected = tm.box_expected([False, True, True], xbox)
            tm.assert_equal(result, expected)

        if box is pd.DataFrame:
            # alignment for frame vs series comparisons deprecated
            #  in GH#46795 enforced 2.0
            with pytest.raises(ValueError, match="not aligned"):
                ser != ser[2]
        else:
            result = ser != ser[2]
            expected = tm.box_expected([True, True, True], xbox)
            tm.assert_equal(result, expected)

        result = ser == ser
        expected = tm.box_expected([True, True, False], xbox)
        tm.assert_equal(result, expected)

        if box is pd.DataFrame:
            # alignment for frame vs series comparisons deprecated
            #  in GH#46795 enforced 2.0
            with pytest.raises(ValueError, match="not aligned"):
                ser == ser[0]
        else:
            result = ser == ser[0]
            expected = tm.box_expected([True, False, False], xbox)
            tm.assert_equal(result, expected)

        if box is pd.DataFrame:
            # alignment for frame vs series comparisons deprecated
            #  in GH#46795 enforced 2.0
            with pytest.raises(ValueError, match="not aligned"):
                ser == ser[2]
        else:
            result = ser == ser[2]
            expected = tm.box_expected([False, False, False], xbox)
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "datetimelike",
        [
            Timestamp("20130101"),
            datetime(2013, 1, 1),
            np.datetime64("2013-01-01T00:00", "ns"),
        ],
    )
    @pytest.mark.parametrize(
        "op,expected",
        [
            (operator.lt, [True, False, False, False]),
            (operator.le, [True, True, False, False]),
            (operator.eq, [False, True, False, False]),
            (operator.gt, [False, False, False, True]),
        ],
    )
    def test_dt64_compare_datetime_scalar(self, datetimelike, op, expected):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over the scalar kinds (Timestamp, Python datetime, NumPy datetime64) and operators (less, less-or-equal, equal, greater) with expected booleans. Verifies comparison of a named datetime series (with a missing value) with the scalar gives the expected named boolean series.
        # GH#17965, test for ability to compare datetime64[ns] columns
        #  to datetimelike
        ser = Series(
            [
                Timestamp("20120101"),
                Timestamp("20130101"),
                np.nan,
                Timestamp("20130103"),
            ],
            name="A",
        )
        result = op(ser, datetimelike)
        expected = Series(expected, name="A")
        tm.assert_series_equal(result, expected)

    def test_ts_series_numpy_maximum(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies the NumPy maximum of a timestamp and a microsecond-unit datetime series works and gives the elementwise maximum series.
        # GH#50864, test numpy.maximum does not fail
        # given a TimeStamp and Series(with dtype datetime64) comparison
        ts = Timestamp("2024-07-01")
        ts_series = Series(
            ["2024-06-01", "2024-07-01", "2024-08-01"],
            dtype="datetime64[us]",
        )

        expected = Series(
            ["2024-07-01", "2024-07-01", "2024-08-01"],
            dtype="datetime64[us]",
        )

        tm.assert_series_equal(expected, np.maximum(ts, ts_series))


class TestDatetimeIndexComparisons:
    # [CLASS DOC]
    # Test class for comparisons of datetime indexes; a TODO says it holds tests moved from the index tests needing parametrization and de-duplication.
    # TODO: moved from tests.indexes.test_base; parametrize and de-duplicate
    def test_comparators(self, comparison_op):
        # [FUNCTION DOC]
        # Uses the comparison-operator fixture. Verifies comparing a date range and the equivalent NumPy datetime array with the NumPy datetime64 middle element gives equal NumPy boolean arrays (the index result is a NumPy array).
        index = date_range("2020-01-01", periods=10)
        element = index[len(index) // 2]
        element = Timestamp(element).to_datetime64()

        arr = np.array(index)
        arr_result = comparison_op(arr, element)
        index_result = comparison_op(index, element)

        assert isinstance(index_result, np.ndarray)
        tm.assert_numpy_array_equal(arr_result, index_result)

    @pytest.mark.parametrize(
        "other",
        [datetime(2016, 1, 1), Timestamp("2016-01-01"), np.datetime64("2016-01-01")],
    )
    def test_dti_cmp_datetimelike(self, other, tz_naive_fixture):
        # [FUNCTION DOC]
        # Parametrized over a Python datetime, a Timestamp, and a NumPy datetime64; uses the timezone fixture (NumPy scalar skipped for aware zones; others localized to the index's zone). Verifies the five comparison operators between a two-element date range and the scalar give the expected arrays.
        tz = tz_naive_fixture
        dti = date_range("2016-01-01", periods=2, tz=tz)
        if tz is not None:
            if isinstance(other, np.datetime64):
                pytest.skip(f"{type(other).__name__} is not tz aware")
            other = localize_pydatetime(other, dti.tzinfo)

        result = dti == other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = dti > other
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = dti >= other
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        result = dti < other
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = dti <= other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, object])
    def test_dti_cmp_nat(self, dtype, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over dtype none or object; uses the box fixture. Verifies equality and inequality between two datetime vectors with NaT (NaT never equal), and against the NaT scalar on either side, boxed in the upcast box.
        left = DatetimeIndex([Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")])
        right = DatetimeIndex([NaT, NaT, Timestamp("2011-01-03")])

        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        xbox = get_upcast_box(left, right, True)

        lhs, rhs = left, right
        if dtype is object:
            lhs, rhs = left.astype(object), right.astype(object)

        result = rhs == lhs
        expected = np.array([False, False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

        result = lhs != rhs
        expected = np.array([True, True, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs == NaT, expected)
        tm.assert_equal(NaT == rhs, expected)

        expected = np.array([True, True, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs != NaT, expected)
        tm.assert_equal(NaT != lhs, expected)

        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs < NaT, expected)
        tm.assert_equal(NaT > lhs, expected)

    def test_dti_cmp_nat_behaves_like_float_cmp_nan(self):
        # [FUNCTION DOC]
        # Verifies NaT behaves like NaN in comparisons: for float indexes, datetime indexes, and datetime index versus NumPy datetime array with NaT in different places, the six comparison operators (including reflected) give the same expected arrays (NaT/NaN positions false except for not-equal); comparing with the scalar NaN or NaT gives all false (all true for not-equal); and comparing with a number three or a datetime of March first gives the expected arrays. Each block asserts no warnings are emitted.
        fidx1 = pd.Index([1.0, np.nan, 3.0, np.nan, 5.0, 7.0])
        fidx2 = pd.Index([2.0, 3.0, np.nan, np.nan, 6.0, 7.0])

        didx1 = DatetimeIndex(
            ["2014-01-01", NaT, "2014-03-01", NaT, "2014-05-01", "2014-07-01"]
        )
        didx2 = DatetimeIndex(
            ["2014-02-01", "2014-03-01", NaT, NaT, "2014-06-01", "2014-07-01"]
        )
        darr = np.array(
            [
                np.datetime64("2014-02-01 00:00"),
                np.datetime64("2014-03-01 00:00"),
                np.datetime64("nat", "ns"),
                np.datetime64("nat", "ns"),
                np.datetime64("2014-06-01 00:00"),
                np.datetime64("2014-07-01 00:00"),
            ]
        )

        cases = [(fidx1, fidx2), (didx1, didx2), (didx1, darr)]

        # Check pd.NaT is handles as the same as np.nan
        with tm.assert_produces_warning(None):
            for idx1, idx2 in cases:
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

        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, np.nan), (didx1, NaT)]:
                result = idx1 < val
                expected = np.array([False, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 > val
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 <= val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 >= val
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 == val
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 != val
                expected = np.array([True, True, True, True, True, True])
                tm.assert_numpy_array_equal(result, expected)

        # Check pd.NaT is handles as the same as np.nan
        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, 3), (didx1, datetime(2014, 3, 1))]:
                result = idx1 < val
                expected = np.array([True, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 > val
                expected = np.array([False, False, False, False, True, True])
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 <= val
                expected = np.array([True, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 >= val
                expected = np.array([False, False, True, False, True, True])
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 == val
                expected = np.array([False, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 != val
                expected = np.array([True, True, False, True, True, True])
                tm.assert_numpy_array_equal(result, expected)

    def test_comparison_tzawareness_compat(self, comparison_op, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the comparison-operator and box fixtures. Verifies that comparing naive and aware date ranges (boxed) with an ordering operator raises a type error "Invalid comparison between dtype=datetime64" for the vector, for lists of timestamps, and for object arrays in either order; equality and inequality do not raise. Aware compared to aware and naive to naive (vector, list, object array, any order) must all compare equal.
        # GH#18162
        op = comparison_op
        box = box_with_array

        dr = date_range("2016-01-01", periods=6, unit="ns")
        dz = dr.tz_localize("US/Pacific")

        dr = tm.box_expected(dr, box)
        dz = tm.box_expected(dz, box)

        if box is pd.DataFrame:
            tolist = lambda x: x.astype(object).values.tolist()[0]
        else:
            tolist = list

        if op not in [operator.eq, operator.ne]:
            msg = (
                r"Invalid comparison between dtype=datetime64\[ns.*\] "
                "and (Timestamp|DatetimeArray|list|ndarray)"
            )
            with pytest.raises(TypeError, match=msg):
                op(dr, dz)

            with pytest.raises(TypeError, match=msg):
                op(dr, tolist(dz))
            with pytest.raises(TypeError, match=msg):
                op(dr, np.array(tolist(dz), dtype=object))
            with pytest.raises(TypeError, match=msg):
                op(dz, dr)

            with pytest.raises(TypeError, match=msg):
                op(dz, tolist(dr))
            with pytest.raises(TypeError, match=msg):
                op(dz, np.array(tolist(dr), dtype=object))

        # The aware==aware and naive==naive comparisons should *not* raise
        assert np.all(dr == dr)
        assert np.all(dr == tolist(dr))
        assert np.all(tolist(dr) == dr)
        assert np.all(np.array(tolist(dr), dtype=object) == dr)
        assert np.all(dr == np.array(tolist(dr), dtype=object))

        assert np.all(dz == dz)
        assert np.all(dz == tolist(dz))
        assert np.all(tolist(dz) == dz)
        assert np.all(np.array(tolist(dz), dtype=object) == dz)
        assert np.all(dz == np.array(tolist(dz), dtype=object))

    def test_comparison_tzawareness_compat_scalars(self, comparison_op, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the comparison-operator and box fixtures. Verifies a naive boxed range is greater than a naive timestamp and an aware range greater than an aware timestamp, and that ordering operators between mixed awareness (range with timestamp, either order) raise a type error "Invalid comparison between dtype=datetime64 and Timestamp".
        # GH#18162
        op = comparison_op

        dr = date_range("2016-01-01", periods=6, unit="ns")
        dz = dr.tz_localize("US/Pacific")

        dr = tm.box_expected(dr, box_with_array)
        dz = tm.box_expected(dz, box_with_array)

        # Check comparisons against scalar Timestamps
        ts = Timestamp("2000-03-14 01:59")
        ts_tz = Timestamp("2000-03-14 01:59", tz="Europe/Amsterdam")

        assert np.all(dr > ts)
        msg = r"Invalid comparison between dtype=datetime64\[ns.*\] and Timestamp"
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dr, ts_tz)

        assert np.all(dz > ts_tz)
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dz, ts)

        if op not in [operator.eq, operator.ne]:
            # GH#12601: Check comparison against Timestamps and DatetimeIndex
            with pytest.raises(TypeError, match=msg):
                op(ts, dz)

    @pytest.mark.parametrize(
        "other",
        [datetime(2016, 1, 1), Timestamp("2016-01-01"), np.datetime64("2016-01-01")],
    )
    # Bug in NumPy? https://github.com/numpy/numpy/issues/13841
    # Raising in __eq__ will fallback to NumPy, which warns, fails,
    # then re-raises the original exception. So we just need to ignore.
    @pytest.mark.filterwarnings("ignore:elementwise comp:DeprecationWarning")
    def test_scalar_comparison_tzawareness(
        self, comparison_op, other, tz_aware_fixture, box_with_array
    ):
        # [FUNCTION DOC]
        # Parametrized over a Python datetime, Timestamp, and NumPy datetime64 as naive scalars; uses the comparison-operator, aware-timezone, and box fixtures. Ignores a NumPy deprecation warning about elementwise comparison (NumPy falls back and warns when the first attempt raises). Verifies an aware boxed date range compared to the naive scalar: equality is false and inequality true (either operand order); ordering raises a type error naming the scalar's type.
        op = comparison_op
        tz = tz_aware_fixture
        dti = date_range("2016-01-01", periods=2, tz=tz, unit="ns")

        dtarr = tm.box_expected(dti, box_with_array)
        xbox = get_upcast_box(dtarr, other, True)
        if op in [operator.eq, operator.ne]:
            exbool = op is operator.ne
            expected = np.array([exbool, exbool], dtype=bool)
            expected = tm.box_expected(expected, xbox)

            result = op(dtarr, other)
            tm.assert_equal(result, expected)

            result = op(other, dtarr)
            tm.assert_equal(result, expected)
        else:
            msg = (
                r"Invalid comparison between dtype=datetime64\[ns, .*\] "
                f"and {type(other).__name__}"
            )
            with pytest.raises(TypeError, match=msg):
                op(dtarr, other)
            with pytest.raises(TypeError, match=msg):
                op(other, dtarr)

    def test_nat_comparison_tzawareness(self, comparison_op):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the comparison-operator fixture. Verifies comparing a datetime index (naive and tz-localized) with NaT does not raise and gives all false (all true for not-equal).
        # GH#19276
        # tzaware DatetimeIndex should not raise when compared to NaT
        op = comparison_op

        dti = DatetimeIndex(
            ["2014-01-01", NaT, "2014-03-01", NaT, "2014-05-01", "2014-07-01"]
        )
        expected = np.array([op == operator.ne] * len(dti))
        result = op(dti, NaT)
        tm.assert_numpy_array_equal(result, expected)

        result = op(dti.tz_localize("US/Pacific"), NaT)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_str(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the timezone fixture. Verifies a ten-day range compared with the string "1/1/2000" is valid regardless of zone: equality true only for the first, inequality the opposite, and ordering operators as expected.
        # GH#22074
        # regardless of tz, we expect these comparisons are valid
        tz = tz_naive_fixture
        rng = date_range("1/1/2000", periods=10, tz=tz)
        other = "1/1/2000"

        result = rng == other
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)

        result = rng != other
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)

        result = rng < other
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)

        result = rng <= other
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)

        result = rng > other
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)

        result = rng >= other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_list(self):
        # [FUNCTION DOC]
        # Verifies a date range compared for equality to a list of its own values equals the comparison with itself (all true).
        rng = date_range("1/1/2000", periods=10)

        result = rng == list(rng)
        expected = rng == rng
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            pd.timedelta_range("1D", periods=10),
            pd.timedelta_range("1D", periods=10).to_series(),
            pd.timedelta_range("1D", periods=10).asi8.view("m8[ns]"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_dti_cmp_tdi_tzawareness(self, other):
        # [FUNCTION DOC]
        # Linked to an issue (a reversion test that the timezone-awareness check is not run against timedeltas). Parametrized over a timedelta index, a series of one, and a timedelta64 NumPy array. Verifies a tz-aware date range equals none (all false) and differs from all (all true), with series index preserved; ordering comparisons raise a type error "Invalid comparison between".
        # GH#22074
        # reversion test that we _don't_ call _assert_tzawareness_compat
        # when comparing against TimedeltaIndex
        dti = date_range("2000-01-01", periods=10, tz="Asia/Tokyo")

        result = dti == other
        expected = np.array([False] * 10)
        if isinstance(other, Series):
            tm.assert_series_equal(result, Series(expected, index=other.index))
        else:
            tm.assert_numpy_array_equal(result, expected)

        result = dti != other
        expected = np.array([True] * 10)
        if isinstance(other, Series):
            tm.assert_series_equal(result, Series(expected, index=other.index))
        else:
            tm.assert_numpy_array_equal(result, expected)

        msg = "Invalid comparison between"
        with pytest.raises(TypeError, match=msg):
            dti < other
        with pytest.raises(TypeError, match=msg):
            dti <= other
        with pytest.raises(TypeError, match=msg):
            dti > other
        with pytest.raises(TypeError, match=msg):
            dti >= other

    def test_dti_cmp_object_dtype(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies an aware date range compared with its object-cast version is all equal; with its naive version is all unequal; and with an object array of five of its elements plus five timedeltas, equality is true for the first five, false for the rest, and greater-or-equal raises a type error (not supported between Timestamp and Timedelta).
        # GH#22074
        dti = date_range("2000-01-01", periods=10, tz="Asia/Tokyo")

        other = dti.astype("O")

        result = dti == other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

        other = dti.tz_localize(None)
        result = dti != other
        tm.assert_numpy_array_equal(result, expected)

        other = np.array(list(dti[:5]) + [Timedelta(days=1)] * 5)
        result = dti == other
        expected = np.array([True] * 5 + [False] * 5)
        tm.assert_numpy_array_equal(result, expected)
        msg = ">=' not supported between instances of 'Timestamp' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            dti >= other


# ------------------------------------------------------------------
# Arithmetic


class TestDatetime64Arithmetic:
    # [CLASS DOC]
    # Test class for datetime arithmetic; "finished" tests fully parametrized over frame, series, index, and datetime array.
    # This class is intended for "finished" tests that are fully parametrized
    #  over DataFrame/Series/Index/DatetimeArray

    # -------------------------------------------------------------
    # Addition/Subtraction of timedelta-like

    @pytest.mark.arm_slow
    def test_dt64arr_add_timedeltalike_scalar(
        self, tz_naive_fixture, two_hours, box_with_array
    ):
        # [FUNCTION DOC]
        # Marked slow on ARM. Linked to issues. Uses the timezone, two-hours, and box fixtures. Verifies adding the timedelta-like scalar to a boxed month-long hourly-unit date range (and the reflected form and in-place add) shifts all by two hours; for aware ranges the frequency is dropped.
        # GH#22005, GH#22163 check DataFrame doesn't raise TypeError
        tz = tz_naive_fixture

        rng = date_range("2000-01-01", "2000-02-01", tz=tz, unit="ns")
        expected = date_range("2000-01-01 02:00", "2000-02-01 02:00", tz=tz, unit="ns")
        if tz is not None:
            expected = expected._with_freq(None)

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = rng + two_hours
        tm.assert_equal(result, expected)

        result = two_hours + rng
        tm.assert_equal(result, expected)

        rng += two_hours
        tm.assert_equal(rng, expected)

    def test_dt64arr_sub_timedeltalike_scalar(
        self, tz_naive_fixture, two_hours, box_with_array
    ):
        # [FUNCTION DOC]
        # Uses the timezone, two-hours, and box fixtures. Verifies subtracting the timedelta-like from a boxed month-long nanosecond date range, and in-place subtraction, shifts everything two hours earlier; aware ranges lose the frequency.
        tz = tz_naive_fixture

        rng = date_range("2000-01-01", "2000-02-01", tz=tz, unit="ns")
        expected = date_range("1999-12-31 22:00", "2000-01-31 22:00", tz=tz, unit="ns")
        if tz is not None:
            expected = expected._with_freq(None)

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = rng - two_hours
        tm.assert_equal(result, expected)

        rng -= two_hours
        tm.assert_equal(rng, expected)

    def test_dt64_array_sub_dt_with_different_timezone(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies subtracting an aware timestamp in another zone (CET) from a boxed aware range (US/Eastern) gives timedelta indexes of six hours plus whole days (and the negated order gives negative values); subtracting a naive timestamp from an aware range (either order) raises a type error "Cannot subtract tz-naive and tz-aware datetime-like objects".
        t1 = date_range("20130101", periods=3).tz_localize("US/Eastern")
        t1 = tm.box_expected(t1, box_with_array)
        t2 = Timestamp("20130101").tz_localize("CET")
        tnaive = Timestamp(20130101)

        result = t1 - t2
        expected = TimedeltaIndex(
            ["0 days 06:00:00", "1 days 06:00:00", "2 days 06:00:00"]
        )
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = t2 - t1
        expected = TimedeltaIndex(
            ["-1 days +18:00:00", "-2 days +18:00:00", "-3 days +18:00:00"]
        )
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive

        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64_array_sub_dt64_array_with_different_timezone(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies subtracting two boxed aware date ranges in different zones gives six hours for each element (and negative eighteen-hour-offset values for the reverse); subtracting a naive range from an aware one (either order) raises a type error about tz-naive and tz-aware objects.
        t1 = date_range("20130101", periods=3).tz_localize("US/Eastern")
        t1 = tm.box_expected(t1, box_with_array)
        t2 = date_range("20130101", periods=3).tz_localize("CET")
        t2 = tm.box_expected(t2, box_with_array)
        tnaive = date_range("20130101", periods=3)

        result = t1 - t2
        expected = TimedeltaIndex(
            ["0 days 06:00:00", "0 days 06:00:00", "0 days 06:00:00"]
        )
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = t2 - t1
        expected = TimedeltaIndex(
            ["-1 days +18:00:00", "-1 days +18:00:00", "-1 days +18:00:00"]
        )
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive

        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64arr_add_sub_td64_nat(self, box_with_array, tz_naive_fixture):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box and timezone fixtures. Verifies a NumPy timedelta NaT with nanosecond unit added to (either order) or subtracted from a boxed quarterly date range gives an all-NaT datetime result in the same zone, and subtracting the datetimes from the NaT raises a type error "cannot subtract".
        # GH#23320 special handling for timedelta64("NaT")
        tz = tz_naive_fixture

        dti = date_range("1994-04-01", periods=9, tz=tz, freq="QS", unit="ns")
        other = np.timedelta64("NaT", "ns")
        expected = DatetimeIndex(["NaT"] * 9, tz=tz).as_unit("ns")

        obj = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        msg = "cannot subtract"
        with pytest.raises(TypeError, match=msg):
            other - obj

    def test_dt64arr_add_sub_td64ndarray(self, tz_naive_fixture, box_with_array):
        # [FUNCTION DOC]
        # Uses the timezone and box fixtures. Verifies adding a NumPy timedelta array of minus one day to a boxed datetime range (either order) shifts one day earlier; subtracting shifts one day later; and subtracting the datetime from the timedelta array raises a type error with one of several messages.
        tz = tz_naive_fixture
        dti = date_range("2016-01-01", periods=3, tz=tz)
        tdi = TimedeltaIndex(["-1 Day", "-1 Day", "-1 Day"])
        tdarr = tdi.values

        expected = date_range("2015-12-31", "2016-01-02", periods=3, tz=tz)

        dtarr = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = dtarr + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + dtarr
        tm.assert_equal(result, expected)

        expected = date_range("2016-01-02", "2016-01-04", periods=3, tz=tz)
        expected = tm.box_expected(expected, box_with_array)

        result = dtarr - tdarr
        tm.assert_equal(result, expected)
        msg = "|".join(
            [
                "cannot subtract DatetimeArray from ndarray",
                "cannot subtract a datelike from a TimedeltaArray",
                "cannot subtract DatetimeArray from Timedelta",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            tdarr - dtarr

    # -----------------------------------------------------------------
    # Subtraction of datetime-like scalars

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-01").to_pydatetime(),
            Timestamp("2013-01-01").to_datetime64(),
            # GH#7996, GH#22163 ensure non-nano datetime64 is converted to nano
            #  for DataFrame operation
            np.datetime64("2013-01-01", "D"),
        ],
    )
    def test_dt64arr_sub_dtscalar(self, box_with_array, ts):
        # [FUNCTION DOC]
        # Linked to issues. Parametrized over four equivalent datetime scalars (Timestamp, Python datetime, NumPy nanosecond datetime64, and NumPy day-unit datetime64, which must be converted to nanosecond for frame operations); uses the box fixture. Verifies a boxed three-day range without frequency minus the scalar gives a boxed timedelta index of zero, one, two days (not a datetime dtype), and the scalar minus the range gives the negation.
        # KNOWN ISSUE: the final equality assertion for the reflected subtraction is written twice.
        # GH#8554, GH#22163 DataFrame op should _not_ return dt64 dtype
        idx = date_range("2013-01-01", periods=3)._with_freq(None)
        idx = tm.box_expected(idx, box_with_array)

        expected = TimedeltaIndex(["0 Days", "1 Day", "2 Days"])
        expected = tm.box_expected(expected, box_with_array)

        result = idx - ts
        tm.assert_equal(result, expected)

        result = ts - idx
        tm.assert_equal(result, -expected)
        tm.assert_equal(result, -expected)

    def test_dt64arr_sub_timestamp_tzaware(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies subtracting the first element (an aware timestamp) from a boxed aware two-day range gives timedeltas of zero and one day, and the reverse the negation.
        ser = date_range("2014-03-17", periods=2, freq="D", tz="US/Eastern", unit="ns")
        ser = ser._with_freq(None)
        ts = ser[0]

        ser = tm.box_expected(ser, box_with_array)

        delta_series = Series(
            [np.timedelta64(0, "D"), np.timedelta64(1, "D")], dtype="m8[ns]"
        )
        expected = tm.box_expected(delta_series, box_with_array)

        tm.assert_equal(ser - ts, expected)
        tm.assert_equal(ts - ser, -expected)

    def test_dt64arr_sub_NaT(self, box_with_array, unit):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box and datetime-unit fixtures. Verifies subtracting NaT from a boxed datetime vector (naive and tz-aware) gives an all-NaT timedelta series in the same unit.
        # GH#18808
        dti = DatetimeIndex([NaT, Timestamp("19900315")]).as_unit(unit)
        ser = tm.box_expected(dti, box_with_array)

        result = ser - NaT
        expected = Series([NaT, NaT], dtype=f"timedelta64[{unit}]")
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        dti_tz = dti.tz_localize("Asia/Tokyo")
        ser_tz = tm.box_expected(dti_tz, box_with_array)

        result = ser_tz - NaT
        expected = Series([NaT, NaT], dtype=f"timedelta64[{unit}]")
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    # -------------------------------------------------------------
    # Subtraction of datetime-like array-like

    def test_dt64arr_sub_dt64object_array(
        self, performance_warning, box_with_array, tz_naive_fixture
    ):
        # [FUNCTION DOC]
        # Uses the performance-warning, box, and naive-timezone fixtures. Verifies subtracting an object-cast copy of a boxed datetime range from itself emits the performance warning and returns the zero timedeltas as object dtype.
        dti = date_range("2016-01-01", periods=3, tz=tz_naive_fixture)
        expected = dti - dti

        obj = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array).astype(object)

        with tm.assert_produces_warning(performance_warning):
            result = obj - obj.astype(object)
        tm.assert_equal(result, expected)

    def test_dt64arr_naive_sub_dt64ndarray(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies subtracting a naive NumPy datetime64 array from a boxed naive datetime range, and the reverse, equals the range minus itself.
        dti = date_range("2016-01-01", periods=3, tz=None)
        dt64vals = dti.values

        dtarr = tm.box_expected(dti, box_with_array)

        expected = dtarr - dtarr
        result = dtarr - dt64vals
        tm.assert_equal(result, expected)
        result = dt64vals - dtarr
        tm.assert_equal(result, expected)

    def test_dt64arr_aware_sub_dt64ndarray_raises(
        self, tz_aware_fixture, box_with_array
    ):
        # [FUNCTION DOC]
        # Uses the aware-timezone and box fixtures. Verifies subtracting a naive NumPy datetime array from a boxed aware range, and the reverse, raises a type error about subtracting tz-naive and tz-aware datetimes.
        tz = tz_aware_fixture
        dti = date_range("2016-01-01", periods=3, tz=tz)
        dt64vals = dti.values

        dtarr = tm.box_expected(dti, box_with_array)
        msg = "Cannot subtract tz-naive and tz-aware datetime"
        with pytest.raises(TypeError, match=msg):
            dtarr - dt64vals
        with pytest.raises(TypeError, match=msg):
            dt64vals - dtarr

    # -------------------------------------------------------------
    # Addition of datetime-like others (invalid)

    def test_dt64arr_add_dtlike_raises(self, tz_naive_fixture, box_with_array):
        # [FUNCTION DOC]
        # Linked to issues (frames must not cast a timestamp to its integer value). Uses the naive-timezone and box fixtures. Verifies the shared cannot-add helper for adding a boxed datetime range to NumPy datetime arrays, indexes, itself, timestamps (naive and aware), Python datetimes, and NumPy day datetimes, all of which must raise.
        # GH#22163 ensure DataFrame doesn't cast Timestamp to i8
        # GH#9631
        tz = tz_naive_fixture

        dti = date_range("2016-01-01", periods=3, tz=tz)
        if tz is None:
            dti2 = dti.tz_localize("US/Eastern")
        else:
            dti2 = dti.tz_localize(None)
        dtarr = tm.box_expected(dti, box_with_array)

        assert_cannot_add(dtarr, dti.values)
        assert_cannot_add(dtarr, dti)
        assert_cannot_add(dtarr, dtarr)
        assert_cannot_add(dtarr, dti[0])
        assert_cannot_add(dtarr, dti[0].to_pydatetime())
        assert_cannot_add(dtarr, dti[0].to_datetime64())
        assert_cannot_add(dtarr, dti2[0])
        assert_cannot_add(dtarr, dti2[0].to_pydatetime())
        assert_cannot_add(dtarr, np.datetime64("2011-01-01", "D"))

    # -------------------------------------------------------------
    # Other Invalid Addition/Subtraction

    # Note: freq here includes both Tick and non-Tick offsets; this is
    #  relevant because historically integer-addition was allowed if we had
    #  a freq.
    @pytest.mark.parametrize("freq", ["h", "D", "W", "2ME", "MS", "QE", "B", None])
    @pytest.mark.parametrize("dtype", [None, "uint8"])
    def test_dt64arr_addsub_intlike(
        self, dtype, index_or_series_or_array, freq, tz_naive_fixture
    ):
        # [FUNCTION DOC]
        # Linked to issues. Parametrized over frequencies (hourly, daily, weekly, 2ME, MS, QE, B, none) with dtype none or uint8 for the integer values; uses the index-series-or-array and timezone fixtures (not frame, because frame alignment makes it inapplicable). Verifies that adding or subtracting integers (python, NumPy scalar, zero-dimensional array, integer arrays in many container types) to a datetime container raises a type error matching one of several messages, via the shared invalid-add/subtract helper.
        # GH#19959, GH#19123, GH#19012
        # GH#55860 use index_or_series_or_array instead of box_with_array
        #  bc DataFrame alignment makes it inapplicable
        tz = tz_naive_fixture

        if freq is None:
            dti = DatetimeIndex(["NaT", "2017-04-05 06:07:08"], tz=tz)
        else:
            dti = date_range("2016-01-01", periods=2, freq=freq, tz=tz)

        obj = index_or_series_or_array(dti)
        other = np.array([4, -1])
        if dtype is not None:
            other = other.astype(dtype)

        msg = "|".join(
            [
                "Addition/subtraction of integers",
                "cannot subtract DatetimeArray from",
                # IntegerArray
                "can only perform ops with numeric values",
                "unsupported operand type.*Categorical",
                r"unsupported operand type\(s\) for -: 'int' and 'Timestamp'",
            ]
        )
        assert_invalid_addsub_type(obj, 1, msg)
        assert_invalid_addsub_type(obj, np.int64(2), msg)
        assert_invalid_addsub_type(obj, np.array(3, dtype=np.int64), msg)
        assert_invalid_addsub_type(obj, other, msg)
        assert_invalid_addsub_type(obj, np.array(other), msg)
        assert_invalid_addsub_type(obj, pd.array(other), msg)
        assert_invalid_addsub_type(obj, pd.Categorical(other), msg)
        assert_invalid_addsub_type(obj, pd.Index(other), msg)
        assert_invalid_addsub_type(obj, Series(other), msg)

    @pytest.mark.parametrize(
        "other",
        [
            3.14,
            np.array([2.0, 3.0]),
            # GH#13078 datetime +/- Period is invalid
            Period("2011-01-01", freq="D"),
            # https://github.com/pandas-dev/pandas/issues/10329
            time(1, 2, 3),
        ],
    )
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_invalid(self, dti_freq, other, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over frequency none or daily and invalid operands: a float, a float array, a Period (invalid), and a time of day. Uses the box fixture. Verifies the invalid-add/subtract helper's expectations for a boxed two-date datetime index.
        dti = DatetimeIndex(["2011-01-01", "2011-01-02"], freq=dti_freq)
        dtarr = tm.box_expected(dti, box_with_array)
        msg = "|".join(
            [
                "unsupported operand type",
                "cannot (add|subtract)",
                "cannot use operands with types",
                "ufunc '?(add|subtract)'? cannot use operands with types",
                "Concatenation operation is not implemented for NumPy arrays",
            ]
        )
        assert_invalid_addsub_type(dtarr, other, msg)

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_parr(
        self, dti_freq, pi_freq, box_with_array, box_with_array2
    ):
        # [FUNCTION DOC]
        # Parametrized over the datetime index's frequency (none or daily) and the period frequency (D, W, Q, h); uses two box fixtures. Linked to an issue about subtracting period indexes. Verifies adding or subtracting a boxed period index to a boxed datetime index raises a type error matching one of several messages.
        # GH#20049 subtracting PeriodIndex should raise TypeError
        dti = DatetimeIndex(["2011-01-01", "2011-01-02"], freq=dti_freq)
        pi = dti.to_period(pi_freq)

        dtarr = tm.box_expected(dti, box_with_array)
        parr = tm.box_expected(pi, box_with_array2)
        msg = "|".join(
            [
                "cannot (add|subtract)",
                "unsupported operand",
                "descriptor.*requires",
                "ufunc.*cannot use operands",
            ]
        )
        assert_invalid_addsub_type(dtarr, parr, msg)

    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    def test_dt64arr_addsub_time_objects_raises(self, box_with_array, tz_naive_fixture):
        # [FUNCTION DOC]
        # Linked to an issue. Ignores the performance warning. Uses the box and naive-timezone fixtures. Verifies adding or subtracting a boxed list of time-of-day objects to a boxed datetime range raises a type error matching unsupported-operand or cannot-subtract messages, per the shared helper.
        # https://github.com/pandas-dev/pandas/issues/10329

        tz = tz_naive_fixture

        obj1 = date_range("2012-01-01", periods=3, tz=tz)
        obj2 = [time(i, i, i) for i in range(3)]

        obj1 = tm.box_expected(obj1, box_with_array)
        obj2 = tm.box_expected(obj2, box_with_array)

        msg = "|".join(
            [
                "unsupported operand",
                "cannot subtract DatetimeArray from ndarray",
            ]
        )
        # pandas.errors.PerformanceWarning: Non-vectorized DateOffset being
        # applied to Series or DatetimeIndex
        # we aren't testing that here, so ignore.
        assert_invalid_addsub_type(obj1, obj2, msg=msg)

    # -------------------------------------------------------------
    # Other invalid operations

    @pytest.mark.parametrize(
        "dt64_series",
        [
            Series([Timestamp("19900315"), Timestamp("19900315")]),
            Series([NaT, Timestamp("19900315")]),
            Series([NaT, NaT], dtype="datetime64[ns]"),
        ],
    )
    @pytest.mark.parametrize("one", [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(self, one, dt64_series, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over three datetime series (valid, with NaT, all NaT) and numeric one (integer, float, zero-dimensional array); uses the box fixture. Verifies multiplying and dividing a boxed datetime series by one, in either order, raises a type error "cannot perform ... with this index type".
        obj = tm.box_expected(dt64_series, box_with_array)

        msg = "cannot perform .* with this index type"

        # multiplication
        with pytest.raises(TypeError, match=msg):
            obj * one
        with pytest.raises(TypeError, match=msg):
            one * obj

        # division
        with pytest.raises(TypeError, match=msg):
            obj / one
        with pytest.raises(TypeError, match=msg):
            one / obj


class TestDatetime64DateOffsetArithmetic:
    # [CLASS DOC]
    # Test class for datetime arithmetic with date offsets (tick, relative-delta, and calendar offsets).
    # -------------------------------------------------------------
    # Tick DateOffsets

    # TODO: parametrize over timezone?
    def test_dt64arr_series_add_tick_DateOffset(self, box_with_array, unit):
        # [FUNCTION DOC]
        # A TODO asks about parametrizing over timezone. Linked to an issue. Uses the box and datetime-unit fixtures. Verifies adding a five-second offset to a boxed series of two timestamps shifts them by five seconds, in either order, preserving the unit.
        # GH#4532
        # operate with pd.offsets
        ser = Series(
            [Timestamp("20130101 9:01"), Timestamp("20130101 9:02")]
        ).dt.as_unit(unit)
        expected = Series(
            [Timestamp("20130101 9:01:05"), Timestamp("20130101 9:02:05")]
        ).dt.as_unit(unit)

        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = ser + pd.offsets.Second(5)
        tm.assert_equal(result, expected)

        result2 = pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)

    def test_dt64arr_series_sub_tick_DateOffset(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies subtracting a five-second offset from a boxed timestamp series shifts back five seconds; negated offset plus series gives the same; and the offset minus the series raises a type error "cannot subtract DatetimeArray from Second".
        # GH#4532
        # operate with pd.offsets
        ser = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])
        expected = Series(
            [Timestamp("20130101 9:00:55"), Timestamp("20130101 9:01:55")]
        )

        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = ser - pd.offsets.Second(5)
        tm.assert_equal(result, expected)

        result2 = -pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)
        msg = "cannot subtract DatetimeArray from Second"
        with pytest.raises(TypeError, match=msg):
            pd.offsets.Second(5) - ser

    @pytest.mark.parametrize(
        "cls_name", ["Day", "Hour", "Minute", "Second", "Milli", "Micro", "Nano"]
    )
    def test_dt64arr_add_sub_tick_DateOffset_smoke(self, cls_name, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over offset class names (Day, Hour, Minute, Second, Milli, Micro, Nano); uses the box fixture. Smoke test: applies addition (both orders) and subtraction of the offset with value five to a boxed timestamp series; asserts nothing about the results.
        # GH#4532
        # smoke tests for valid DateOffsets
        ser = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])
        ser = tm.box_expected(ser, box_with_array)

        offset_cls = getattr(pd.offsets, cls_name)
        ser + offset_cls(5)
        offset_cls(5) + ser
        ser - offset_cls(5)

    def test_dti_add_tick_tzaware(self, tz_aware_fixture, box_with_array):
        # [FUNCTION DOC]
        # Linked to issues (frames must not return object dtype). Uses the aware-timezone and box fixtures. For the Pacific zone checks that adding an hour offset to the whole range matches adding it to the first element. Then, for an hourly range, verifies adding or reverse-adding a five-hour offset, NumPy timedelta, or Python timedelta gives the expected hourly range in the zone (nanosecond unit), the round trip by subtraction restores the range, and scalar minus range raises a type error "cannot subtract DatetimeArray from".
        # GH#21610, GH#22163 ensure DataFrame doesn't return object-dtype
        tz = tz_aware_fixture
        if tz == "US/Pacific":
            dates = date_range("2012-11-01", periods=3, tz=tz, unit="ns")
            offset = dates + pd.offsets.Hour(5)
            assert dates[0] + pd.offsets.Hour(5) == offset[0]

        dates = date_range("2010-11-01 00:00", periods=3, tz=tz, freq="h", unit="ns")
        expected = DatetimeIndex(
            ["2010-11-01 05:00", "2010-11-01 06:00", "2010-11-01 07:00"],
            freq="h",
            tz=tz,
        ).as_unit("ns")

        dates = tm.box_expected(dates, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        for scalar in [pd.offsets.Hour(5), np.timedelta64(5, "h"), timedelta(hours=5)]:
            offset = dates + scalar
            tm.assert_equal(offset, expected)
            offset = scalar + dates
            tm.assert_equal(offset, expected)

            roundtrip = offset - scalar
            tm.assert_equal(roundtrip, dates)

            msg = "cannot subtract DatetimeArray from"
            with pytest.raises(TypeError, match=msg):
                scalar - dates

    # -------------------------------------------------------------
    # RelativeDelta DateOffsets

    def test_dt64arr_add_sub_relativedelta_offsets(self, box_with_array, unit):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box and datetime-unit fixtures. For eight datetimes, tries relative-delta date offsets with years, months, days, hours, minutes, seconds, and microseconds (single and cumulative combinations). Verifies adding and subtracting each offset to the boxed vector equals the pointwise operation on each element, with unit rule that microsecond offsets upgrade non-nanosecond units to microseconds, and that offset minus vector raises a type error "cannot subtract DatetimeArray from". A TODO notes the unit conversion in the expectation should become unnecessary.
        # GH#10699
        vec = DatetimeIndex(
            [
                Timestamp("2000-01-05 00:15:00"),
                Timestamp("2000-01-31 00:23:00"),
                Timestamp("2000-01-01"),
                Timestamp("2000-03-31"),
                Timestamp("2000-02-29"),
                Timestamp("2000-12-31"),
                Timestamp("2000-05-15"),
                Timestamp("2001-06-15"),
            ]
        ).as_unit(unit)
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec

        # DateOffset relativedelta fastpath
        relative_kwargs = [
            ("years", 2),
            ("months", 5),
            ("days", 3),
            ("hours", 5),
            ("minutes", 10),
            ("seconds", 2),
            ("microseconds", 5),
        ]
        for i, (offset_unit, value) in enumerate(relative_kwargs):
            off = DateOffset(**{offset_unit: value})

            exp_unit = unit
            if offset_unit == "microseconds" and unit != "ns":
                exp_unit = "us"

            # TODO(GH#55564): as_unit will be unnecessary
            expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)

            expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)

            off = DateOffset(**dict(relative_kwargs[: i + 1]))

            expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)

            expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)
            msg = "cannot subtract DatetimeArray from"
            with pytest.raises(TypeError, match=msg):
                off - vec

    # -------------------------------------------------------------
    # Non-Tick, Non-RelativeDelta DateOffsets

    # TODO: redundant with test_dt64arr_add_sub_DateOffset?  that includes
    #  tz-aware cases which this does not
    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    @pytest.mark.parametrize(
        "cls_and_kwargs",
        [
            "YearBegin",
            ("YearBegin", {"month": 5}),
            "YearEnd",
            ("YearEnd", {"month": 5}),
            "MonthBegin",
            "MonthEnd",
            "SemiMonthEnd",
            "SemiMonthBegin",
            "Week",
            ("Week", {"weekday": 3}),
            ("Week", {"weekday": 6}),
            "BusinessDay",
            "BDay",
            "QuarterEnd",
            "QuarterBegin",
            "CustomBusinessDay",
            "CDay",
            "CBMonthEnd",
            "CBMonthBegin",
            "BMonthBegin",
            "BMonthEnd",
            "BusinessHour",
            "BYearBegin",
            "BYearEnd",
            "BQuarterBegin",
            ("LastWeekOfMonth", {"weekday": 2}),
            (
                "FY5253Quarter",
                {
                    "qtr_with_extra_week": 1,
                    "startingMonth": 1,
                    "weekday": 2,
                    "variation": "nearest",
                },
            ),
            ("FY5253", {"weekday": 0, "startingMonth": 2, "variation": "nearest"}),
            ("WeekOfMonth", {"weekday": 2, "week": 2}),
            "Easter",
            ("DateOffset", {"day": 4}),
            ("DateOffset", {"month": 5}),
        ],
    )
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("n", [0, 5])
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_dt64arr_add_sub_DateOffsets(
        self, box_with_array, n, normalize, cls_and_kwargs, unit, tz
    ):
        # [FUNCTION DOC]
        # A TODO says it may be redundant with the next-named generic offset test, which includes aware cases this one lacks. Ignores the performance warning. Parametrized over about thirty offset kinds (with optional constructor keyword arguments), normalize flag, multiple n of zero or five, timezone (naive or US/Central), and uses the box and unit fixtures. Skips combinations where n zero is invalid for the offset (week-of-month, last-week-of-month, and the fiscal calendars). Builds eight datetimes (localized if requested), then verifies that vector plus offset (both orders) and vector minus offset equal the pointwise results in the same unit, and that offset minus vector raises a type error "cannot subtract DatetimeArray from".
        # GH#10699
        # assert vectorized operation matches pointwise operations

        if isinstance(cls_and_kwargs, tuple):
            # If cls_name param is a tuple, then 2nd entry is kwargs for
            # the offset constructor
            cls_name, kwargs = cls_and_kwargs
        else:
            cls_name = cls_and_kwargs
            kwargs = {}

        if n == 0 and cls_name in [
            "WeekOfMonth",
            "LastWeekOfMonth",
            "FY5253Quarter",
            "FY5253",
        ]:
            # passing n = 0 is invalid for these offset classes
            return

        vec = (
            DatetimeIndex(
                [
                    Timestamp("2000-01-05 00:15:00"),
                    Timestamp("2000-01-31 00:23:00"),
                    Timestamp("2000-01-01"),
                    Timestamp("2000-03-31"),
                    Timestamp("2000-02-29"),
                    Timestamp("2000-12-31"),
                    Timestamp("2000-05-15"),
                    Timestamp("2001-06-15"),
                ]
            )
            .as_unit(unit)
            .tz_localize(tz)
        )
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec

        offset_cls = getattr(pd.offsets, cls_name)
        offset = offset_cls(n, normalize=normalize, **kwargs)

        # TODO(GH#55564): as_unit will be unnecessary
        expected = DatetimeIndex([x + offset for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec + offset)
        tm.assert_equal(expected, offset + vec)

        expected = DatetimeIndex([x - offset for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec - offset)

        expected = DatetimeIndex([offset + x for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, offset + vec)
        msg = "cannot subtract DatetimeArray from"
        with pytest.raises(TypeError, match=msg):
            offset - vec

    @pytest.mark.parametrize(
        "other",
        [
            [pd.offsets.MonthEnd(), pd.offsets.Day(n=2)],
            [pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()],
            # matching offsets
            [pd.offsets.DateOffset(years=1), pd.offsets.DateOffset(years=1)],
        ],
    )
    @pytest.mark.parametrize("op", [operator.add, roperator.radd, operator.sub])
    def test_dt64arr_add_sub_offset_array(
        self, performance_warning, tz_naive_fixture, box_with_array, op, other
    ):
        # [FUNCTION DOC]
        # Linked to issues. Parametrized over three arrays of two offsets (month-end and two-day; year and month-end offsets; matching year offsets) and add, reflected add, and subtract; uses the performance-warning, naive-timezone, and box fixtures. Verifies operating a boxed two-date range with the array of offsets (both as NumPy array and as a boxed array) gives an object result equal to the pointwise operation and emits the performance warning; for the array factory with reflected add the expected is a NumPy-extension object array.
        # GH#18849
        # GH#10699 array of offsets

        tz = tz_naive_fixture
        dti = date_range("2017-01-01", periods=2, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)
        other = np.array(other)
        expected = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))])
        expected = tm.box_expected(expected, box_with_array).astype(object)

        with tm.assert_produces_warning(performance_warning):
            res = op(dtarr, other)
        tm.assert_equal(res, expected)

        # Same thing but boxing other
        other = tm.box_expected(other, box_with_array)
        if box_with_array is pd.array and op is roperator.radd:
            # We expect a NumpyExtensionArray, not ndarray[object] here
            expected = pd.array(expected, dtype=object)
        with tm.assert_produces_warning(performance_warning):
            res = op(dtarr, other)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize(
        "op, offset, exp, exp_freq",
        [
            (
                "__add__",
                DateOffset(months=3, days=10),
                [
                    Timestamp("2014-04-11"),
                    Timestamp("2015-04-11"),
                    Timestamp("2016-04-11"),
                    Timestamp("2017-04-11"),
                ],
                None,
            ),
            (
                "__add__",
                DateOffset(months=3),
                [
                    Timestamp("2014-04-01"),
                    Timestamp("2015-04-01"),
                    Timestamp("2016-04-01"),
                    Timestamp("2017-04-01"),
                ],
                "YS-APR",
            ),
            (
                "__sub__",
                DateOffset(months=3, days=10),
                [
                    Timestamp("2013-09-21"),
                    Timestamp("2014-09-21"),
                    Timestamp("2015-09-21"),
                    Timestamp("2016-09-21"),
                ],
                None,
            ),
            (
                "__sub__",
                DateOffset(months=3),
                [
                    Timestamp("2013-10-01"),
                    Timestamp("2014-10-01"),
                    Timestamp("2015-10-01"),
                    Timestamp("2016-10-01"),
                ],
                "YS-OCT",
            ),
        ],
    )
    def test_dti_add_sub_nonzero_mth_offset(
        self, op, offset, exp, exp_freq, tz_aware_fixture, box_with_array
    ):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over add/subtract special-method names with month-based offsets (three months plus ten days, and three months alone) and the expected dates with the expected frequency (none for mixed offsets, anchored-year-starting-April or October for month-only); uses the aware-timezone and box fixtures. Verifies a yearly-start aware nanosecond range operated with the offset gives the expected datetime index (frequency check performed by the boxed comparison).
        # GH 26258
        tz = tz_aware_fixture
        date = date_range(
            start="01 Jan 2014", end="01 Jan 2017", freq="YS", tz=tz, unit="ns"
        )
        date = tm.box_expected(date, box_with_array, False)
        mth = getattr(date, op)
        result = mth(offset)

        expected = DatetimeIndex(exp, tz=tz).as_unit("ns")
        expected = tm.box_expected(expected, box_with_array, False)
        tm.assert_equal(result, expected)

    def test_dt64arr_series_add_DateOffset_with_milli(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies adding a four-millisecond offset, and a one-day-plus-four-milliseconds offset, to a nanosecond datetime index with sub-millisecond fractions adds exactly 4 ms (preserving the nanosecond remainder) and days correctly across month ends.
        # GH 57529
        dti = DatetimeIndex(
            [
                "2000-01-01 00:00:00.012345678",
                "2000-01-31 00:00:00.012345678",
                "2000-02-29 00:00:00.012345678",
            ],
            dtype="datetime64[ns]",
        )
        result = dti + DateOffset(milliseconds=4)
        expected = DatetimeIndex(
            [
                "2000-01-01 00:00:00.016345678",
                "2000-01-31 00:00:00.016345678",
                "2000-02-29 00:00:00.016345678",
            ],
            dtype="datetime64[ns]",
        )
        tm.assert_index_equal(result, expected)

        result = dti + DateOffset(days=1, milliseconds=4)
        expected = DatetimeIndex(
            [
                "2000-01-02 00:00:00.016345678",
                "2000-02-01 00:00:00.016345678",
                "2000-03-01 00:00:00.016345678",
            ],
            dtype="datetime64[ns]",
        )
        tm.assert_index_equal(result, expected)


class TestDatetime64OverflowHandling:
    # [CLASS DOC]
    # Test class for overflow handling in datetime arithmetic; a TODO says to box and de-duplicate.
    # TODO: box + de-duplicate

    def test_dt64_overflow_masking(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies a boxed datetime near the epoch minus a boxed NaT series gives a boxed timedelta NaT without overflow.
        # GH#25317
        left = Series([Timestamp("1969-12-31")], dtype="M8[ns]")
        right = Series([NaT])

        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)

        expected = TimedeltaIndex([NaT], dtype="m8[ns]")
        expected = tm.box_expected(expected, box_with_array)

        result = left - right
        tm.assert_equal(result, expected)

    def test_dt64_series_arith_overflow(self):
        # [FUNCTION DOC]
        # Linked to issues. Verifies that for a series of 100-year-end dates spanning large ranges, subtracting or being subtracted from a far-past timestamp, and adding a 20000-day timedelta in either order, raises an overflow error "Overflow in int64 addition"; with the last element set to NaT the sums succeed with expected dates and NaT; with all but the first set to NaT, the differences with the timestamp give 91279 days and NaT and the reverse is the negation.
        # GH#12534, fixed by GH#19024
        dt = Timestamp("1700-01-31")
        td = Timedelta("20000 Days")
        dti = date_range("1949-09-30", freq="100YE", periods=4, unit="ns")
        ser = Series(dti)
        msg = "Overflow in int64 addition"
        with pytest.raises(OverflowError, match=msg):
            ser - dt
        with pytest.raises(OverflowError, match=msg):
            dt - ser
        with pytest.raises(OverflowError, match=msg):
            ser + td
        with pytest.raises(OverflowError, match=msg):
            td + ser

        ser.iloc[-1] = NaT
        expected = Series(
            ["2004-10-03", "2104-10-04", "2204-10-04", "NaT"], dtype="datetime64[ns]"
        )
        res = ser + td
        tm.assert_series_equal(res, expected)
        res = td + ser
        tm.assert_series_equal(res, expected)

        ser.iloc[1:] = NaT
        expected = Series(["91279 Days", "NaT", "NaT", "NaT"], dtype="timedelta64[ns]")
        res = ser - dt
        tm.assert_series_equal(res, expected)
        res = dt - ser
        tm.assert_series_equal(res, -expected)

    def test_datetimeindex_sub_timestamp_overflow(self):
        # [FUNCTION DOC]
        # Verifies for datetime indexes with the maximum or minimum timestamp that subtracting far-past or far-future timestamps (in several scalar forms) either raises an overflow error "Overflow in int64 addition" or produces a result whose last element's integer value equals the expected difference of the integer representations.
        dtimax = pd.to_datetime(["2021-12-28 17:19", Timestamp.max]).as_unit("ns")
        dtimin = pd.to_datetime(["2021-12-28 17:19", Timestamp.min]).as_unit("ns")

        tsneg = Timestamp("1950-01-01").as_unit("ns")
        ts_neg_variants = [
            tsneg,
            tsneg.to_pydatetime(),
            tsneg.to_datetime64().astype("datetime64[ns]"),
            tsneg.to_datetime64().astype("datetime64[D]"),
        ]

        tspos = Timestamp("1980-01-01").as_unit("ns")
        ts_pos_variants = [
            tspos,
            tspos.to_pydatetime(),
            tspos.to_datetime64().astype("datetime64[ns]"),
            tspos.to_datetime64().astype("datetime64[D]"),
        ]
        msg = "Overflow in int64 addition"
        for variant in ts_neg_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimax - variant

        expected = Timestamp.max._value - tspos._value
        for variant in ts_pos_variants:
            res = dtimax - variant
            assert res[1]._value == expected

        expected = Timestamp.min._value - tsneg._value
        for variant in ts_neg_variants:
            res = dtimin - variant
            assert res[1]._value == expected

        for variant in ts_pos_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimin - variant

    def test_datetimeindex_sub_datetimeindex_overflow(self):
        # [FUNCTION DOC]
        # Linked to issues. Verifies subtraction of datetime indexes at the extreme timestamps: the regular combinations produce the exact integer differences, the overflowing combinations raise an overflow error, and two edge cases (the minimum timestamp plus the maximum timedelta plus a microsecond, and the maximum minus the minimum timedelta plus a microsecond) also raise.
        # GH#22492, GH#22508
        dtimax = pd.to_datetime(["2021-12-28 17:19", Timestamp.max]).as_unit("ns")
        dtimin = pd.to_datetime(["2021-12-28 17:19", Timestamp.min]).as_unit("ns")

        ts_neg = pd.to_datetime(["1950-01-01", "1950-01-01"]).as_unit("ns")
        ts_pos = pd.to_datetime(["1980-01-01", "1980-01-01"]).as_unit("ns")

        # General tests
        expected = Timestamp.max._value - ts_pos[1]._value
        result = dtimax - ts_pos
        assert result[1]._value == expected

        expected = Timestamp.min._value - ts_neg[1]._value
        result = dtimin - ts_neg
        assert result[1]._value == expected
        msg = "Overflow in int64 addition"
        with pytest.raises(OverflowError, match=msg):
            dtimax - ts_neg

        with pytest.raises(OverflowError, match=msg):
            dtimin - ts_pos

        # Edge cases
        tmin = pd.to_datetime([Timestamp.min])
        t1 = tmin + Timedelta.max + Timedelta("1us")
        with pytest.raises(OverflowError, match=msg):
            t1 - tmin

        tmax = pd.to_datetime([Timestamp.max])
        t2 = tmax + Timedelta.min - Timedelta("1us")
        with pytest.raises(OverflowError, match=msg):
            tmax - t2


class TestTimestampSeriesArithmetic:
    # [CLASS DOC]
    # Test class for arithmetic on series of timestamps, containing tests moved from the series operator tests.
    def test_empty_series_add_sub(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies that, for empty datetime and timedelta containers built by the box fixture, addition of a timedelta to the datetime in either order, and subtraction, return the datetime container unchanged, while timedelta minus datetime raises a type error "cannot subtract".
        # GH#13844
        a = Series(dtype="M8[ns]")
        b = Series(dtype="m8[ns]")
        a = box_with_array(a)
        b = box_with_array(b)
        tm.assert_equal(a, a + b)
        tm.assert_equal(a, a - b)
        tm.assert_equal(a, b + a)
        msg = "cannot subtract"
        with pytest.raises(TypeError, match=msg):
            b - a

    def test_operators_datetimelike(self):
        # [FUNCTION DOC]
        # Smoke test (no assertions): evaluates differences of two datetime series, and addition and subtraction of a timedelta series with a datetime series (including a missing value), merely checking that none of them raise.
        # ## timedelta64 ###
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        # ## datetime64 ###
        dt1 = Series(
            [
                Timestamp("20111230"),
                Timestamp("20120101"),
                Timestamp("20120103"),
            ]
        )
        dt1.iloc[2] = np.nan
        dt2 = Series(
            [
                Timestamp("20111231"),
                Timestamp("20120102"),
                Timestamp("20120104"),
            ]
        )
        dt1 - dt2
        dt2 - dt1

        # datetime64 with timetimedelta
        dt1 + td1
        td1 + dt1
        dt1 - td1

        # timetimedelta with datetime64
        td1 + dt1
        dt1 + td1

    def test_dt64ser_sub_datetime_dtype(self, unit):
        # [FUNCTION DOC]
        # Uses the datetime-unit fixture. Verifies the dtype of a series of a timestamp minus a Python datetime is timedelta64 in the finer of the series unit and microseconds (the unit imputed to the Python datetime).
        ts = Timestamp(datetime(1993, 1, 7, 13, 30, 00))
        dt = datetime(1993, 6, 22, 13, 30)
        ser = Series([ts], dtype=f"M8[{unit}]")
        result = ser - dt

        # the expected unit is the max of `unit` and the unit imputed to `dt`,
        #  which is "us"
        exp_unit = tm.get_finest_unit(unit, "us")
        assert result.dtype == f"timedelta64[{exp_unit}]"

    # -------------------------------------------------------------
    # TODO: This next block of tests came from tests.series.test_operators,
    # needs to be de-duplicated and parametrized over `box` classes

    @pytest.mark.parametrize(
        "left, right, op_fail",
        [
            [
                [Timestamp("20111230"), Timestamp("20120101"), NaT],
                [Timestamp("20111231"), Timestamp("20120102"), Timestamp("20120104")],
                ["__sub__", "__rsub__"],
            ],
            [
                [Timestamp("20111230"), Timestamp("20120101"), NaT],
                [timedelta(minutes=5, seconds=3), timedelta(minutes=5, seconds=3), NaT],
                ["__add__", "__radd__", "__sub__"],
            ],
            [
                [
                    Timestamp("20111230", tz="US/Eastern"),
                    Timestamp("20111230", tz="US/Eastern"),
                    NaT,
                ],
                [timedelta(minutes=5, seconds=3), NaT, timedelta(minutes=5, seconds=3)],
                ["__add__", "__radd__", "__sub__"],
            ],
        ],
    )
    def test_operators_datetimelike_invalid(
        self, left, right, op_fail, all_arithmetic_operators
    ):
        # [FUNCTION DOC]
        # A TODO says this block of tests was moved from the series tests and needs de-duplication and box parametrization. Parametrized over three datetime-like pairs of series with the list of operator names that are expected to be valid for each, and the all-arithmetic-operators fixture. Applies the named operator of the first series to the second: for names not in the valid list expects a type error matching "operate", "cannot", or unsupported operand; for names in the list it is a smoke test (no raise expected).
        # these are all TypeError ops
        op_str = all_arithmetic_operators
        arg1 = Series(left)
        arg2 = Series(right)
        # check that we are getting a TypeError
        # with 'operate' (from core/ops.py) for the ops that are not
        # defined
        op = getattr(arg1, op_str, None)
        # Previously, _validate_for_numeric_binop in core/indexes/base.py
        # did this for us.
        if op_str not in op_fail:
            with pytest.raises(
                TypeError, match="operate|[cC]annot|unsupported operand"
            ):
                op(arg2)
        else:
            # Smoke test
            op(arg2)

    def test_sub_single_tz(self, unit):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the unit fixture. Verifies subtracting two single-element tz-aware series in the same zone gives a timedelta series of two days (or minus two days reversed) in the same unit.
        # GH#12290
        s1 = Series([Timestamp("2016-02-10", tz="America/Sao_Paulo")]).dt.as_unit(unit)
        s2 = Series([Timestamp("2016-02-08", tz="America/Sao_Paulo")]).dt.as_unit(unit)
        result = s1 - s2
        expected = Series([Timedelta("2days")]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)
        result = s2 - s1
        expected = Series([Timedelta("-2days")]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)

    def test_dt64tz_series_sub_dtitz(self):
        # [FUNCTION DOC]
        # Linked to an issue (subtracting a tz-aware index from a tz-aware series in the same zone used to raise). Verifies the datetime index minus the series and the series minus the index both give a series of zero timedeltas.
        # GH#19071 subtracting tzaware DatetimeIndex from tzaware Series
        # (with same tz) raises, fixed by #19024
        dti = date_range("1999-09-30", periods=10, tz="US/Pacific")
        ser = Series(dti)
        expected = Series(TimedeltaIndex(["0days"] * 10))

        res = dti - ser
        tm.assert_series_equal(res, expected)
        res = ser - dti
        tm.assert_series_equal(res, expected)

    def test_sub_datetime_compat(self, unit):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the unit fixture. Verifies a UTC-aware series with NaT minus an aware Python datetime (and minus the equivalent Timestamp) gives a one-day timedelta and NaT in the finer of the series unit and microseconds.
        # see GH#14088
        ser = Series([datetime(2016, 8, 23, 12, tzinfo=timezone.utc), NaT]).dt.as_unit(
            unit
        )
        dt = datetime(2016, 8, 22, 12, tzinfo=timezone.utc)
        # The datetime object has "us" so we upcast lower units
        exp_unit = tm.get_finest_unit(unit, "us")
        exp = Series([Timedelta("1 days"), NaT]).dt.as_unit(exp_unit)
        result = ser - dt
        tm.assert_series_equal(result, exp)
        result2 = ser - Timestamp(dt)
        tm.assert_series_equal(result2, exp)

    def test_dt64_series_add_mixed_tick_DateOffset(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies a timestamp series plus (either order) a five-millisecond offset shifts by five milliseconds, and adding a five-minute offset and a five-millisecond offset in sequence gives the cumulative shift.
        # GH#4532
        # operate with pd.offsets
        s = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])

        result = s + pd.offsets.Milli(5)
        result2 = pd.offsets.Milli(5) + s
        expected = Series(
            [Timestamp("20130101 9:01:00.005"), Timestamp("20130101 9:02:00.005")]
        )
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        result = s + pd.offsets.Minute(5) + pd.offsets.Milli(5)
        expected = Series(
            [Timestamp("20130101 9:06:00.005"), Timestamp("20130101 9:07:00.005")]
        )
        tm.assert_series_equal(result, expected)

    def test_datetime64_ops_nat(self, unit):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the unit fixture. Verifies arithmetic with NaT: negated NaT plus a datetime series (with NaT) gives an all-NaT datetime series in the same unit; negating a NaT datetime series raises a type error about unary minus on DatetimeArray; and NaT added to or from an all-NaT series gives all NaT.
        # KNOWN ISSUE: the two addition assertions are repeated verbatim.
        # GH#11349
        datetime_series = Series([NaT, Timestamp("19900315")]).dt.as_unit(unit)
        nat_series_dtype_timestamp = Series([NaT, NaT], dtype=f"datetime64[{unit}]")
        single_nat_dtype_datetime = Series([NaT], dtype=f"datetime64[{unit}]")

        # subtraction
        tm.assert_series_equal(-NaT + datetime_series, nat_series_dtype_timestamp)
        msg = "bad operand type for unary -: 'DatetimeArray'"
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + datetime_series

        tm.assert_series_equal(
            -NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + nat_series_dtype_timestamp

        # addition
        tm.assert_series_equal(
            nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )

        tm.assert_series_equal(
            nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )

    # -------------------------------------------------------------
    # Timezone-Centric Tests

    def test_operators_datetimelike_with_timezones(self):
        # [FUNCTION DOC]
        # Verifies datetime-with-timezone series arithmetic with timedeltas: a US/Eastern series plus or minus a scalar timedelta, plus the scalar on the left, and plus a timedelta series (including a copy with a missing value) all equal the tz-naive operation relocalized to the zone; timedelta scalar minus a datetime series raises a type error "cannot subtract DatetimeArray from". Also asserts the timedelta copy with a missing value has no frequency.
        tz = "US/Eastern"
        dt1 = Series(date_range("2000-01-01 09:00:00", periods=5, tz=tz), name="foo")
        dt2 = dt1.copy()
        dt2.iloc[2] = np.nan

        td1 = Series(pd.timedelta_range("1 days 1 min", periods=5, freq="h"))
        td2 = td1.copy()
        td2.iloc[1] = np.nan
        assert td2._values.freq is None

        result = dt1 + td1[0]
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt2 + td2[0]
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        # odd numpy behavior with scalar timedeltas
        result = td1[0] + dt1
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = td2[0] + dt2
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt1 - td1[0]
        exp = (dt1.dt.tz_localize(None) - td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg = "cannot subtract DatetimeArray from"
        with pytest.raises(TypeError, match=msg):
            td1[0] - dt1

        result = dt2 - td2[0]
        exp = (dt2.dt.tz_localize(None) - td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        with pytest.raises(TypeError, match=msg):
            td2[0] - dt2

        result = dt1 + td1
        exp = (dt1.dt.tz_localize(None) + td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt2 + td2
        exp = (dt2.dt.tz_localize(None) + td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt1 - td1
        exp = (dt1.dt.tz_localize(None) - td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt2 - td2
        exp = (dt2.dt.tz_localize(None) - td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg = "cannot (add|subtract)"
        with pytest.raises(TypeError, match=msg):
            td1 - dt1
        with pytest.raises(TypeError, match=msg):
            td2 - dt2


class TestDatetimeIndexArithmetic:
    # [CLASS DOC]
    # Test class for arithmetic between datetime indexes and timedelta indexes, arrays, Series, and offset array-likes.
    # -------------------------------------------------------------
    # Binary operations DatetimeIndex and TimedeltaIndex/array

    def test_dti_add_tdi(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the timezone fixture. Verifies adding a timedelta index of zero to nine days to a datetime index of ten copies of one date (naive or aware), in either order, and adding the equivalent timedelta64 NumPy array in either order, gives a ten-day date range without frequency.
        # GH#17558
        tz = tz_naive_fixture
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        tdi = pd.timedelta_range("0 days", periods=10)
        expected = date_range("2017-01-01", periods=10, tz=tz)
        expected = expected._with_freq(None)

        # add with TimedeltaIndex
        result = dti + tdi
        tm.assert_index_equal(result, expected)

        result = tdi + dti
        tm.assert_index_equal(result, expected)

        # add with timedelta64 array
        result = dti + tdi.values
        tm.assert_index_equal(result, expected)

        result = tdi.values + dti
        tm.assert_index_equal(result, expected)

    def test_dti_iadd_tdi(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the timezone fixture. Verifies in-place addition: a datetime index plus-equals a timedelta index or timedelta64 array gives the ten-day range without frequency; a timedelta index plus-equals a datetime index likewise gives that datetime index.
        # KNOWN ISSUE: the block for the timedelta64 array repeats the timedelta-index-plus-datetime-index in-place addition rather than testing the array on the left.
        # GH#17558
        tz = tz_naive_fixture
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        tdi = pd.timedelta_range("0 days", periods=10)
        expected = date_range("2017-01-01", periods=10, tz=tz)
        expected = expected._with_freq(None)

        # iadd with TimedeltaIndex
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        result += tdi
        tm.assert_index_equal(result, expected)

        result = pd.timedelta_range("0 days", periods=10)
        result += dti
        tm.assert_index_equal(result, expected)

        # iadd with timedelta64 array
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        result += tdi.values
        tm.assert_index_equal(result, expected)

        result = pd.timedelta_range("0 days", periods=10)
        result += dti
        tm.assert_index_equal(result, expected)

    def test_dti_sub_tdi(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the timezone fixture. Verifies a datetime index minus a timedelta index (or its timedelta64 values) gives dates counting backwards without frequency; the reverse subtractions (timedelta index minus datetime index, array minus datetime index) raise type errors about subtracting from TimedeltaArray or a datelike.
        # GH#17558
        tz = tz_naive_fixture
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        tdi = pd.timedelta_range("0 days", periods=10)
        expected = date_range("2017-01-01", periods=10, tz=tz, freq="-1D")
        expected = expected._with_freq(None)

        # sub with TimedeltaIndex
        result = dti - tdi
        tm.assert_index_equal(result, expected)

        msg = "cannot subtract .*TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi - dti

        # sub with timedelta64 array
        result = dti - tdi.values
        tm.assert_index_equal(result, expected)

        msg = "cannot subtract a datelike from a TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi.values - dti

    def test_dti_isub_tdi(self, tz_naive_fixture, unit):
        # [FUNCTION DOC]
        # Linked to issues. Uses the timezone and datetime-unit fixtures. Verifies in-place subtraction of a timedelta index (and timedelta64 values) from a datetime index in the given unit gives dates counting backwards; also checks the in-place subtraction on the underlying datetime array and the NumPy subtract ufunc with an output argument match the expectation; and in-place subtraction of the datetime index from the timedelta index or values raises a type error about subtracting a datelike from TimedeltaArray.
        # GH#17558
        tz = tz_naive_fixture
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10).as_unit(unit)
        tdi = pd.timedelta_range("0 days", periods=10, unit=unit)
        expected = date_range("2017-01-01", periods=10, tz=tz, freq="-1D", unit=unit)
        expected = expected._with_freq(None)

        # isub with TimedeltaIndex
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10).as_unit(unit)
        result -= tdi
        tm.assert_index_equal(result, expected)

        # DTA.__isub__ GH#43904
        dta = dti._data.copy()
        dta -= tdi
        tm.assert_datetime_array_equal(dta, expected._data)

        out = dti._data.copy()
        np.subtract(out, tdi, out=out)
        tm.assert_datetime_array_equal(out, expected._data)

        msg = "cannot subtract a datelike from a TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi -= dti

        # isub with timedelta64 array
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10).as_unit(unit)
        result -= tdi.values
        tm.assert_index_equal(result, expected)

        with pytest.raises(TypeError, match=msg):
            tdi.values -= dti

        with pytest.raises(TypeError, match=msg):
            tdi._values -= dti

    # -------------------------------------------------------------
    # Binary Operations DatetimeIndex and datetime-like
    # TODO: A couple other tests belong in this section.  Move them in
    # A PR where there isn't already a giant diff.

    # -------------------------------------------------------------

    def test_dta_add_sub_index(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Uses the timezone fixture. Verifies the datetime array defers to the index implementation: array minus index equals the index difference; array plus or minus the resulting timedelta index equals the corresponding index operation.
        # Check that DatetimeArray defers to Index classes
        dti = date_range("20130101", periods=3, tz=tz_naive_fixture)
        dta = dti.array
        result = dta - dti
        expected = dti - dti
        tm.assert_index_equal(result, expected)

        tdi = result
        result = dta + tdi
        expected = dti + tdi
        tm.assert_index_equal(result, expected)

        result = dta - tdi
        expected = dti - tdi
        tm.assert_index_equal(result, expected)

    def test_sub_dti_dti(self, unit):
        # [FUNCTION DOC]
        # Uses the datetime-unit fixture. Verifies subtraction of datetime indexes: naive minus naive and aware minus aware give zero timedeltas in the unit; mixing awareness raises a type error about tz-naive and tz-aware objects; in-place subtraction of itself gives zeros; different lengths raise a value error "cannot add indices of unequal length"; NaN/NaT propagate in the elementwise difference. A comment notes that this operation used to be a set operation.
        # previously performed setop (deprecated in 0.16.0), now changed to
        # return subtraction -> TimeDeltaIndex (GH ...)

        dti = date_range("20130101", periods=3, unit=unit)
        dti_tz = date_range("20130101", periods=3, unit=unit).tz_localize("US/Eastern")
        expected = TimedeltaIndex([0, 0, 0]).as_unit(unit)

        result = dti - dti
        tm.assert_index_equal(result, expected)

        result = dti_tz - dti_tz
        tm.assert_index_equal(result, expected)
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            dti_tz - dti

        with pytest.raises(TypeError, match=msg):
            dti - dti_tz

        # isub
        dti -= dti
        tm.assert_index_equal(dti, expected)

        # different length raises ValueError
        dti1 = date_range("20130101", periods=3, unit=unit)
        dti2 = date_range("20130101", periods=4, unit=unit)
        msg = "cannot add indices of unequal length"
        with pytest.raises(ValueError, match=msg):
            dti1 - dti2

        # NaN propagation
        dti1 = DatetimeIndex(["2012-01-01", np.nan, "2012-01-03"]).as_unit(unit)
        dti2 = DatetimeIndex(["2012-01-02", "2012-01-03", np.nan]).as_unit(unit)
        expected = TimedeltaIndex(["1 days", np.nan, np.nan]).as_unit(unit)
        result = dti2 - dti1
        tm.assert_index_equal(result, expected)

    # -------------------------------------------------------------------
    # TODO: Most of this block is moved from series or frame tests, needs
    # cleanup, box-parametrization, and de-duplication

    @pytest.mark.parametrize("op", [operator.add, operator.sub])
    def test_timedelta64_equal_timedelta_supported_ops(self, op, box_with_array):
        # [FUNCTION DOC]
        # A TODO says most of this block was moved from series or frame tests and needs cleanup, box parametrization, and de-duplication. Parametrized over add and subtract; uses the box fixture. For all 32 combinations of zero or one in each of day, hour, minute, second, microsecond, builds equivalent NumPy and Python timedeltas and verifies the operator on a boxed series of four timestamps gives equal results for both.
        ser = Series(
            [
                Timestamp("20130301"),
                Timestamp("20130228 23:00:00"),
                Timestamp("20130228 22:00:00"),
                Timestamp("20130228 21:00:00"),
            ]
        )
        obj = box_with_array(ser)

        intervals = ["D", "h", "m", "s", "us"]

        def timedelta64(*args):
            # [FUNCTION DOC]
            # HELPER: nested function that builds a single NumPy timedelta by summing the per-unit NumPy timedeltas for the given tuple of values (days, hours, minutes, seconds, microseconds). Sums element types as the NumPy casting note requires.
            # see casting notes in NumPy gh-12927
            return np.sum(list(map(np.timedelta64, args, intervals)))

        for d, h, m, s, us in product(*([range(2)] * 5)):
            nptd = timedelta64(d, h, m, s, us)
            pytd = timedelta(days=d, hours=h, minutes=m, seconds=s, microseconds=us)
            lhs = op(obj, nptd)
            rhs = op(obj, pytd)

            tm.assert_equal(lhs, rhs)

    def test_ops_nat_mixed_datetime64_timedelta64(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies NaT arithmetic mixing datetime and timedelta series: datetime series minus an NaT datetime series gives an all-NaT timedelta series; minus an NaT timedelta series gives all-NaT datetimes; negation of an NaT timedelta series plus a datetime series likewise; an all-NaT datetime series minus a single NaT datetime series gives timedelta NaT (the ambiguous bare case defaults to timedelta); a timedelta series minus an NaT datetime series raises a type error "cannot subtract a datelike"; and additions of NaT timedelta or datetime series give NaT datetimes (with repeated assertions).
        # GH#11349
        timedelta_series = Series([NaT, Timedelta("1s")])
        datetime_series = Series([NaT, Timestamp("19900315")])
        nat_series_dtype_timedelta = Series([NaT, NaT], dtype="timedelta64[ns]")
        nat_series_dtype_timestamp = Series([NaT, NaT], dtype="datetime64[ns]")
        single_nat_dtype_datetime = Series([NaT], dtype="datetime64[ns]")
        single_nat_dtype_timedelta = Series([NaT], dtype="timedelta64[ns]")

        # subtraction
        tm.assert_series_equal(
            datetime_series - single_nat_dtype_datetime, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            datetime_series - single_nat_dtype_timedelta, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + datetime_series, nat_series_dtype_timestamp
        )

        # without a Series wrapping the NaT, it is ambiguous
        # whether it is a datetime64 or timedelta64
        # defaults to interpreting it as timedelta64
        tm.assert_series_equal(
            nat_series_dtype_timestamp - single_nat_dtype_datetime,
            nat_series_dtype_timedelta,
        )

        tm.assert_series_equal(
            nat_series_dtype_timestamp - single_nat_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + nat_series_dtype_timestamp,
            nat_series_dtype_timestamp,
        )
        msg = "cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            timedelta_series - single_nat_dtype_datetime

        # addition
        tm.assert_series_equal(
            nat_series_dtype_timestamp + single_nat_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timestamp,
            nat_series_dtype_timestamp,
        )

        tm.assert_series_equal(
            nat_series_dtype_timestamp + single_nat_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timestamp,
            nat_series_dtype_timestamp,
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_datetime,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            single_nat_dtype_datetime + nat_series_dtype_timedelta,
            nat_series_dtype_timestamp,
        )

    def test_ufunc_coercions(self, unit):
        # [FUNCTION DOC]
        # Uses the datetime-unit fixture. Verifies adding or subtracting a day NumPy timedelta (operator and ufunc) on a named two-day-frequency range returns a datetime index with the expected values and the same frequency; with a NumPy array of three timedeltas on a frequency-less index, addition and subtraction (operator and ufunc) give the expected dates and no inferred frequency.
        idx = date_range("2011-01-01", periods=3, freq="2D", name="x", unit=unit)

        delta = np.timedelta64(1, "D")
        exp = date_range("2011-01-02", periods=3, freq="2D", name="x", unit=unit)
        for result in [idx + delta, np.add(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == "2D"

        exp = date_range("2010-12-31", periods=3, freq="2D", name="x", unit=unit)

        for result in [idx - delta, np.subtract(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == "2D"

        # When adding/subtracting an ndarray (which has no .freq), the result
        #  does not infer freq
        idx = idx._with_freq(None)
        delta = np.array(
            [np.timedelta64(1, "D"), np.timedelta64(2, "D"), np.timedelta64(3, "D")]
        )
        exp = DatetimeIndex(
            ["2011-01-02", "2011-01-05", "2011-01-08"], name="x"
        ).as_unit(unit)

        for result in [idx + delta, np.add(idx, delta)]:
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

        exp = DatetimeIndex(
            ["2010-12-31", "2011-01-01", "2011-01-02"], name="x"
        ).as_unit(unit)
        for result in [idx - delta, np.subtract(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

    def test_dti_add_series(self, tz_naive_fixture, names):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the timezone and names fixtures. Verifies a datetime index plus a series of five-second timedeltas (indexed by the same index), in either order, returns a series of shifted datetimes with index preserved and name chosen by the names fixture, in the index's dtype; adding the series' raw values returns an index of shifted datetimes.
        # GH#13905
        tz = tz_naive_fixture
        index = DatetimeIndex(
            ["2016-06-28 05:30", "2016-06-28 05:31"], tz=tz, name=names[0]
        ).as_unit("ns")
        ser = Series([Timedelta(seconds=5)] * 2, index=index, name=names[1])
        expected = Series(index + Timedelta(seconds=5), index=index, name=names[2])

        # passing name arg isn't enough when names[2] is None
        expected.name = names[2]
        assert expected.dtype == index.dtype
        result = ser + index
        tm.assert_series_equal(result, expected)
        result2 = index + ser
        tm.assert_series_equal(result2, expected)

        expected = index + Timedelta(seconds=5)
        result3 = ser.values + index
        tm.assert_index_equal(result3, expected)
        result4 = index + ser.values
        tm.assert_index_equal(result4, expected)

    @pytest.mark.parametrize("op", [operator.add, roperator.radd, operator.sub])
    def test_dti_addsub_offset_arraylike(
        self, performance_warning, tz_naive_fixture, names, op, index_or_series
    ):
        # [FUNCTION DOC]
        # Linked to issues. Parametrized over add, reflected add, and subtract; uses the performance-warning, timezone, names, and index-or-series fixtures. Operates a named two-date range with an index or series of a month-end offset and a two-day offset. Verifies the result is an object container (index or series per the upcast rule) of the pointwise results with inferred frequency and the expected name, and the performance warning is emitted.
        # GH#18849, GH#19744
        other_box = index_or_series

        tz = tz_naive_fixture
        dti = date_range("2017-01-01", periods=2, tz=tz, name=names[0])
        other = other_box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)], name=names[1])

        xbox = get_upcast_box(dti, other)

        with tm.assert_produces_warning(performance_warning):
            res = op(dti, other)

        expected = DatetimeIndex(
            [op(dti[n], other[n]) for n in range(len(dti))], name=names[2], freq="infer"
        )
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize("other_box", [pd.Index, np.array])
    def test_dti_addsub_object_arraylike(
        self, performance_warning, tz_naive_fixture, box_with_array, other_box
    ):
        # [FUNCTION DOC]
        # Parametrized over index or NumPy array holding a month-end offset and a four-day timedelta; uses the performance-warning, timezone, and box fixtures. Verifies adding it to a boxed two-day range gives the object results January 31 and January 6, and subtracting gives December 31 and December 29, each with a performance warning.
        tz = tz_naive_fixture

        dti = date_range("2017-01-01", periods=2, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)
        other = other_box([pd.offsets.MonthEnd(), Timedelta(days=4)])
        xbox = get_upcast_box(dtarr, other)

        expected = DatetimeIndex(["2017-01-31", "2017-01-06"], tz=tz_naive_fixture)
        expected = tm.box_expected(expected, xbox).astype(object)

        with tm.assert_produces_warning(performance_warning):
            result = dtarr + other
        tm.assert_equal(result, expected)

        expected = DatetimeIndex(["2016-12-31", "2016-12-29"], tz=tz_naive_fixture)
        expected = tm.box_expected(expected, xbox).astype(object)

        with tm.assert_produces_warning(performance_warning):
            result = dtarr - other
        tm.assert_equal(result, expected)


@pytest.mark.parametrize("years", [-1, 0, 1])
@pytest.mark.parametrize("months", [-2, 0, 2])
def test_shift_months(years, months, unit):
    # [FUNCTION DOC]
    # Parametrized over years minus one, zero, one and months minus two, zero, two; uses the datetime-unit fixture. Verifies the internal month-shifting helper, given the integer representation of five dates (including month ends and a leap day) and the combined shift in months with the resolution of the index, produces dates equal to adding the equivalent date offset to each date, in the same unit.
    dti = DatetimeIndex(
        [
            Timestamp("2000-01-05 00:15:00"),
            Timestamp("2000-01-31 00:23:00"),
            Timestamp("2000-01-01"),
            Timestamp("2000-02-29"),
            Timestamp("2000-12-31"),
        ]
    ).as_unit(unit)
    shifted = shift_months(dti.asi8, years * 12 + months, reso=dti._data._creso)
    shifted_dt64 = shifted.view(f"M8[{dti.unit}]")
    actual = DatetimeIndex(shifted_dt64)

    raw = [x + pd.offsets.DateOffset(years=years, months=months) for x in dti]
    expected = DatetimeIndex(raw).as_unit(dti.unit)
    tm.assert_index_equal(actual, expected)


def test_dt64arr_addsub_object_dtype_2d(performance_warning):
    # [FUNCTION DOC]
    # Uses the performance-warning fixture. Verifies two-dimensional datetime arrays (required for block-wise frame operations) work with object arrays: adding a column array of day offsets gives the same as the one-dimensional result reshaped (with a warning); subtracting the same array cast to object gives a two-dimensional result of zero timedeltas, also with a warning.
    # block-wise DataFrame operations will require operating on 2D
    #  DatetimeArray/TimedeltaArray, so check that specifically.
    dti = date_range("1994-02-13", freq="2W", periods=4)
    dta = dti._data.reshape((4, 1))

    other = np.array([[pd.offsets.Day(n)] for n in range(4)])
    assert other.shape == dta.shape

    with tm.assert_produces_warning(performance_warning):
        result = dta + other
    with tm.assert_produces_warning(performance_warning):
        expected = (dta[:, 0] + other[:, 0]).reshape(-1, 1)

    tm.assert_numpy_array_equal(result, expected)

    with tm.assert_produces_warning(performance_warning):
        # Case where we expect to get a TimedeltaArray back
        result2 = dta - dta.astype(object)

    assert result2.shape == (4, 1)
    assert all(td._value == 0 for td in result2.ravel())


def test_non_nano_dt64_addsub_np_nat_scalars():
    # [FUNCTION DOC]
    # Linked to an issue. Verifies subtracting a NumPy millisecond datetime NaT from a millisecond-unit datetime series gives an all-NaT millisecond timedelta series, and adding a millisecond timedelta NaT gives an all-NaT millisecond datetime series.
    # GH 52295
    ser = Series([1233242342344, 232432434324, 332434242344], dtype="datetime64[ms]")
    result = ser - np.datetime64("nat", "ms")
    expected = Series([NaT] * 3, dtype="timedelta64[ms]")
    tm.assert_series_equal(result, expected)

    result = ser + np.timedelta64("nat", "ms")
    expected = Series([NaT] * 3, dtype="datetime64[ms]")
    tm.assert_series_equal(result, expected)


def test_non_nano_dt64_addsub_np_nat_scalars_unitless():
    # [FUNCTION DOC]
    # Linked to an issue. A TODO asks whether the unit of the series could be the default.
    # KNOWN ISSUE: despite the name mentioning a unitless scalar, the test uses the same millisecond-unit NaT scalars as the previous test (only the case of the timedelta NaT string differs).
    # Verifies the same results as the previous test: all-NaT millisecond timedelta for datetime NaT subtraction, all-NaT millisecond datetime for timedelta NaT addition.
    # GH 52295
    # TODO: Can we default to the ser unit?
    ser = Series([1233242342344, 232432434324, 332434242344], dtype="datetime64[ms]")
    result = ser - np.datetime64("nat", "ms")
    expected = Series([NaT] * 3, dtype="timedelta64[ms]")
    tm.assert_series_equal(result, expected)

    result = ser + np.timedelta64("NaT", "ms")
    expected = Series([NaT] * 3, dtype="datetime64[ms]")
    tm.assert_series_equal(result, expected)


def test_non_nano_dt64_addsub_np_nat_scalars_unsupported_unit():
    # [FUNCTION DOC]
    # Linked to an issue. Verifies subtracting a day-unit NumPy datetime NaT from a second-unit datetime series gives an all-NaT second-unit timedelta series, and adding a day-unit timedelta NaT gives an all-NaT second-unit datetime series (the finer unit is kept for the unsupported coarser unit).
    # GH 52295
    ser = Series([12332, 23243, 33243], dtype="datetime64[s]")
    result = ser - np.datetime64("nat", "D")
    expected = Series([NaT] * 3, dtype="timedelta64[s]")
    tm.assert_series_equal(result, expected)

    result = ser + np.timedelta64("nat", "D")
    expected = Series([NaT] * 3, dtype="datetime64[s]")
    tm.assert_series_equal(result, expected)
