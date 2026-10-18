# [MODULE DOC]
# Tests for Series.isin and related algorithm behavior: string scalars, large
# Series, datetime/timedelta with different resolutions, empty and read-only
# arguments, dtype and timezone mismatch, period frequency mismatch, masked and
# Arrow dtypes with NA, complex numbers, mixed objects and iterators. Some tests
# patch the minimum comparison array length in the algorithms module with
# monkeypatch. Some need pyarrow. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray


class TestSeriesIsIn:
    # [CLASS DOC]
    # Groups the main Series.isin tests. Stateless.
    def test_isin(self):
        # [FUNCTION DOC]
        # Strings membership gives the expected boolean mask. GH#16012: a million-element
        # string Series tested against a list of 78 mixed int/str items must count 200000
        # matches (the numeric items must not cause an exception in NumPy comparisons).
        s = Series(["A", "B", "C", "a", "B", "B", "A", "C"])

        result = s.isin(["A", "C"])
        expected = Series([True, False, True, False, False, False, True, True])
        tm.assert_series_equal(result, expected)

        # GH#16012
        # This specific issue has to have a series over 1e6 in len, but the
        # comparison array (in_list) must be large enough so that numpy doesn't
        # do a manual masking trick that will avoid this issue altogether
        s = Series(list("abcdefghijk" * 10**5))
        # If numpy doesn't do the manual comparison/mask, these
        # unorderable mixed types are what cause the exception in numpy
        in_list = [-1, "a", "b", "G", "Y", "Z", "E", "K", "E", "S", "I", "R", "R"] * 6

        assert s.isin(in_list).sum() == 200000

    def test_isin_with_string_scalar(self):
        # [FUNCTION DOC]
        # Regression test (GH#4763): passing a string instead of a list-like raises a
        # type error that only list-like objects are allowed.
        # GH#4763
        s = Series(["A", "B", "C", "a", "B", "B", "A", "C"])
        msg = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a `str`"
        )
        with pytest.raises(TypeError, match=msg):
            s.isin("a")

        s = Series(["aaa", "b", "c"])
        with pytest.raises(TypeError, match=msg):
            s.isin("aaa")

    def test_isin_datetimelike_mismatched_reso(self):
        # [FUNCTION DOC]
        # A datetime Series tested against day-resolution NumPy dates and against a
        # second-resolution array finds the matching first two entries.
        expected = Series([True, True, False, False, False])

        ser = Series(date_range("jan-01-2013", "jan-05-2013"))

        # fails on dtype conversion in the first place
        day_values = np.asarray(ser[0:2].values).astype("datetime64[D]")
        result = ser.isin(day_values)
        tm.assert_series_equal(result, expected)

        dta = ser[:2]._values.astype("M8[s]")
        result = ser.isin(dta)
        tm.assert_series_equal(result, expected)

    def test_isin_datetimelike_mismatched_reso_list(self):
        # [FUNCTION DOC]
        # A datetime Series tested against a Python list of second-resolution datetimes
        # finds the first two entries.
        expected = Series([True, True, False, False, False])

        ser = Series(date_range("jan-01-2013", "jan-05-2013"))

        dta = ser[:2]._values.astype("M8[s]")
        result = ser.isin(list(dta))
        tm.assert_series_equal(result, expected)

    def test_isin_with_i8(self):
        # [FUNCTION DOC]
        # Regression test (GH#5021): datetime Series membership against a Series, an
        # array, a list of one Timestamp or NumPy datetime, and a set; and timedelta
        # Series membership against a Series.
        # GH#5021

        expected = Series([True, True, False, False, False])
        expected2 = Series([False, True, False, False, False])

        # datetime64[ns]
        s = Series(date_range("jan-01-2013", "jan-05-2013"))

        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

        result = s.isin(s[0:2].values)
        tm.assert_series_equal(result, expected)

        result = s.isin([s[1]])
        tm.assert_series_equal(result, expected2)

        result = s.isin([np.datetime64(s[1])])
        tm.assert_series_equal(result, expected2)

        result = s.isin(set(s[0:2]))
        tm.assert_series_equal(result, expected)

        # timedelta64[ns]
        s = Series(pd.to_timedelta(range(5), unit="D"))
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # [FUNCTION DOC]
        # Regression test (GH#16991), parametrized over an empty list, empty Series and
        # empty array: all results are False.
        # see GH#16991
        s = Series(["a", "b"])
        expected = Series([False, False])

        result = s.isin(empty)
        tm.assert_series_equal(expected, result)

    def test_isin_read_only(self):
        # [FUNCTION DOC]
        # Regression test (GitHub issue 37174): a read-only NumPy array can be used as
        # values.
        # https://github.com/pandas-dev/pandas/issues/37174
        arr = np.array([1, 2, 3])
        arr.setflags(write=False)
        s = Series([1, 2, 3])
        result = s.isin(arr)
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [object, None])
    def test_isin_dt64_values_vs_ints(self, dtype):
        # [FUNCTION DOC]
        # Regression test (GH#36621), parametrized over object or default dtype: integer
        # nanosecond values must not match datetimes (all False) for an index, a Series
        # and the algorithms-level function.
        # GH#36621 dont cast integers to datetimes for isin
        dti = date_range("2013-01-01", "2013-01-05")
        ser = Series(dti)

        comps = np.asarray([1356998400000000000], dtype=dtype)

        res = dti.isin(comps)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(comps)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, comps)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_tzawareness_mismatch(self):
        # [FUNCTION DOC]
        # Naive datetimes tested against the same instants localized to UTC are not
        # members (all False) for an index, a Series and the algorithms-level function.
        dti = date_range("2013-01-01", "2013-01-05")
        ser = Series(dti)

        other = dti.tz_localize("UTC")

        res = dti.isin(other)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_period_freq_mismatch(self):
        # [FUNCTION DOC]
        # Monthly periods tested against a period array of a different frequency
        # (same integer values, built through a private constructor) are not members,
        # for an index, a Series and the algorithms-level function.
        dti = date_range("2013-01-01", "2013-01-05")
        pi = dti.to_period("M")
        ser = Series(pi)

        # We construct another PeriodIndex with the same i8 values
        #  but different dtype
        dtype = dti.to_period("Y").dtype
        other = PeriodArray._simple_new(pi.asi8, dtype=dtype)

        res = pi.isin(other)
        expected = np.array([False] * len(pi), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize("values", [[-9.0, 0.0], [-9, 0]])
    def test_isin_float_in_int_series(self, values):
        # [FUNCTION DOC]
        # Regression test (GH#19356, GH#21804), parametrized over int or float data:
        # membership of float values (-9, -0.5) works across numeric types; only -9 matches.
        # GH#19356 GH#21804
        ser = Series(values)
        result = ser.isin([-9, -0.5])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
    @pytest.mark.parametrize(
        "data,values,expected",
        [
            ([0, 1, 0], [1], [False, True, False]),
            ([0, 1, 0], [1, pd.NA], [False, True, False]),
            ([0, pd.NA, 0], [1, 0], [True, False, True]),
            ([0, 1, pd.NA], [1, pd.NA], [False, True, True]),
            ([0, 1, pd.NA], [1, np.nan], [False, True, False]),
            ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None], [False, False, False]),
        ],
    )
    def test_isin_masked_types(self, dtype, data, values, expected):
        # [FUNCTION DOC]
        # Regression test (GH#42405), parametrized over boolean/Int64/Float64 and six
        # data/values/expected tuples including NA, NaN, NaT and None in the values. The
        # result has nullable boolean dtype; NA in the data matches only if NA is in the
        # values; NaN, NaT or None in the values never match.
        # GH#42405
        ser = Series(data, dtype=dtype)

        result = ser.isin(values)
        expected = Series(expected, dtype="boolean")

        tm.assert_series_equal(result, expected)


def test_isin_large_series_mixed_dtypes_and_nan(monkeypatch):
    # [FUNCTION DOC]
    # GitHub issue 37094: patches the minimum comparison array length to 5 so the
    # "large series" path runs for a 15-element float Series with NaN against a set
    # of strings; the result is all False.
    # https://github.com/pandas-dev/pandas/issues/37094
    # combination of object dtype for the values
    # and > _MINIMUM_COMP_ARR_LEN elements
    min_isin_comp = 5
    ser = Series([1, 2, np.nan] * min_isin_comp)
    with monkeypatch.context() as m:
        m.setattr(algorithms, "_MINIMUM_COMP_ARR_LEN", min_isin_comp)
        result = ser.isin({"foo", "bar"})
    expected = Series([False] * 3 * min_isin_comp)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype, data, values, expected",
    [
        ("boolean", [pd.NA, False, True], [False, pd.NA], [True, True, False]),
        ("Int64", [pd.NA, 2, 1], [1, pd.NA], [True, False, True]),
        ("boolean", [pd.NA, False, True], [pd.NA, True, "a", 20], [True, False, True]),
        ("boolean", [pd.NA, False, True], [], [False, False, False]),
        ("Float64", [20.0, 30.0, pd.NA], [pd.NA], [False, False, True]),
    ],
)
def test_isin_large_series_and_pdNA(dtype, data, values, expected, monkeypatch):
    # [FUNCTION DOC]
    # GitHub issue 60678: with the minimum comparison array length patched to 2,
    # nullable Series against values that contain NA (and other values) give a
    # nullable boolean result in which NA matches NA.
    # https://github.com/pandas-dev/pandas/issues/60678
    # combination of  large series (> _MINIMUM_COMP_ARR_LEN elements) and
    # values contains pdNA
    min_isin_comp = 2
    ser = Series(data, dtype=dtype)
    expected = Series(expected, dtype="boolean")

    with monkeypatch.context() as m:
        m.setattr(algorithms, "_MINIMUM_COMP_ARR_LEN", min_isin_comp)
        result = ser.isin(values)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype, data, values, expected",
    [
        ("int64[pyarrow]", [1, None], [1, pd.NA], [True, True]),
        ("binary[pyarrow]", [None, b"\xe3"], [pd.NA, b"\xe3"], [True, True]),
    ],
)
def test_isin_arrow_dtype_with_na_value(dtype, data, values, expected):
    # [FUNCTION DOC]
    # GH#63304, needs pyarrow: pyarrow-backed integer and binary Series tested
    # against values containing NA match the missing entries as well as the real values.
    # GH#63304
    pytest.importorskip("pyarrow")

    result = Series(data, dtype=dtype).isin(values)
    expected = Series(expected)

    tm.assert_series_equal(result, expected)


def test_isin_complex_numbers():
    # [FUNCTION DOC]
    # GH 17927: membership of complex numbers uses equality across all parts.
    # GH 17927
    array = [0, 1j, 1j, 1, 1 + 1j, 1 + 2j, 1 + 1j]
    result = Series(array).isin([1j, 1 + 1j, 1 + 2j])
    expected = Series([False, True, True, False, True, True, True], dtype=bool)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data,is_in",
    [([1, [2]], [1]), (["simple str", [{"values": 3}]], ["simple str"])],
)
def test_isin_filtering_with_mixed_object_types(data, is_in):
    # [FUNCTION DOC]
    # GH 20883, parametrized over two inputs: object Series with an unhashable
    # list element works; the unhashable item is not a member while the matching
    # scalar is.
    # GH 20883

    ser = Series(data)
    result = ser.isin(is_in)
    expected = Series([True, False])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize("isin", [[1, 2], [1.0, 2.0]])
def test_isin_filtering_on_iterable(data, isin):
    # [FUNCTION DOC]
    # GH 50234, parametrized over integer or float data and values: a generator can be
    # passed as the values; 1 and 2 are found.
    # GH 50234

    ser = Series(data)
    result = ser.isin(i for i in isin)
    expected_result = Series([True, True, False])

    tm.assert_series_equal(result, expected_result)
