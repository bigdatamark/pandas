# [MODULE DOC]
# Tests for missing-value detection in core dtype utilities: the public missing and not-missing
# predicates (isna, isnull, notna, notnull), the array equivalence function, the per-dtype missing
# value lookup, the low-level object missing check and scalar check from the missing-value library
# module, the matching-missing comparison, and the validity check of a missing marker for a dtype.
# Module-level data: two fixed timestamps (naive and UTC); lists of missing-like values (na_vals,
# including None, NaT, NaN in several float and complex types and NaT of every datetime/timedelta
# unit), infinite values, integer values equal to the NaT sentinel (not missing), a decimal NaN
# (sometimes missing), and negative-zero values that must never be missing. Uses the nulls_fixture
# and nulls_fixture2 fixtures from the shared conftest. No import-time side effects.
from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT

from pandas.core.dtypes.common import (
    is_float,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import (
    array_equivalent,
    is_valid_na_for_dtype,
    isna,
    isnull,
    na_value_for_dtype,
    notna,
    notnull,
)

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
)
import pandas._testing as tm

fix_now = pd.Timestamp("2021-01-01")
fix_utcnow = pd.Timestamp("2021-01-01", tz="UTC")


@pytest.mark.parametrize("notna_f", [notna, notnull])
def test_notna_notnull(notna_f):
    # [FUNCTION DOC]
    # Parametrized over the two names notna and notnull. Asserts a float is not missing while none
    # and NaN are missing.
    assert notna_f(1.0)
    assert not notna_f(None)
    assert not notna_f(np.nan)


@pytest.mark.parametrize("null_func", [notna, notnull, isna, isnull])
@pytest.mark.parametrize(
    "ser",
    [
        Series(
            [str(i) for i in range(5)],
            index=Index([str(i) for i in range(5)], dtype=object),
            dtype=object,
        ),
        Series(range(5), date_range("2020-01-01", periods=5)),
        Series(range(5), period_range("2020-01-01", periods=5)),
    ],
)
def test_null_check_is_series(null_func, ser):
    # [FUNCTION DOC]
    # Parametrized over all four predicates and three series (object with string index; integer
    # with datetime index; integer with period index). Asserts each predicate applied to a series
    # returns a series.
    assert isinstance(null_func(ser), Series)


class TestIsNA:
    # [CLASS DOC]
    # Tests of the missing-value predicates on scalars, arrays, lists, datetime-like and period
    # data, and decimals.
    def test_0d_array(self):
        # [FUNCTION DOC]
        # Zero-dimensional arrays: NaN is missing, zero (float or int) is not, both for numeric and
        # object dtype.
        assert isna(np.array(np.nan))
        assert not isna(np.array(0.0))
        assert not isna(np.array(0))
        # test object dtype
        assert isna(np.array(np.nan, dtype=object))
        assert not isna(np.array(0.0, dtype=object))
        assert not isna(np.array(0, dtype=object))

    @pytest.mark.parametrize("shape", [(4, 0), (4,)])
    def test_empty_object(self, shape):
        # [FUNCTION DOC]
        # Parametrized over shapes (4 by 0, and 4). The missing check on an empty-initialized object array
        # returns a same-shape all-true boolean array (empty-initialized object arrays hold none).
        arr = np.empty(shape=shape, dtype=object)
        result = isna(arr)
        expected = np.ones(shape=shape, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("isna_f", [isna, isnull])
    def test_isna_isnull(self, isna_f):
        # [FUNCTION DOC]
        # Parametrized over the names isna and isnull. A float is not missing; none and NaN are;
        # positive and negative infinity are not; the series and frame classes themselves (as objects)
        # are not missing.
        # KNOWN ISSUE: contains a stray assertion that the float NaN constructed from the text is
        # truthy, which is always true (NaN is truthy) and tests nothing about the predicate.
        assert not isna_f(1.0)
        assert isna_f(None)
        assert isna_f(np.nan)
        assert float("nan")
        assert not isna_f(np.inf)
        assert not isna_f(-np.inf)

        # type
        assert not isna_f(type(Series(dtype=object)))
        assert not isna_f(type(Series(dtype=np.float64)))
        assert not isna_f(type(pd.DataFrame()))

    @pytest.mark.parametrize("isna_f", [isna, isnull])
    @pytest.mark.parametrize(
        "data",
        [
            np.arange(4, dtype=float),
            [0.0, 1.0, 0.0, 1.0],
            Series(list("abcd"), dtype=object),
            date_range("2020-01-01", periods=4),
        ],
    )
    @pytest.mark.parametrize(
        "index",
        [
            date_range("2020-01-01", periods=4),
            range(4),
            period_range("2020-01-01", periods=4),
        ],
    )
    def test_isna_isnull_frame(self, isna_f, data, index):
        # [FUNCTION DOC]
        # Parametrized over the predicate name, four kinds of data (float array, float list, object
        # series, date range) and three kinds of index (date range, range, period range). The predicate
        # applied to a frame must equal applying it column by column.
        # frame
        df = pd.DataFrame(data, index=index)
        result = isna_f(df)
        expected = df.apply(isna_f)
        tm.assert_frame_equal(result, expected)

    def test_isna_lists(self):
        # [FUNCTION DOC]
        # The predicate on lists: nested lists of boolean or integers give same-shape false arrays;
        # lists of strings give false; a list with NaN and a string gives true then false (regression for
        # an upstream issue).
        # KNOWN ISSUE: the list-of-strings check is written twice identically.
        result = isna([[False]])
        exp = np.array([[False]])
        tm.assert_numpy_array_equal(result, exp)

        result = isna([[1], [2]])
        exp = np.array([[False], [False]])
        tm.assert_numpy_array_equal(result, exp)

        # list of strings / unicode
        result = isna(["foo", "bar"])
        exp = np.array([False, False])
        tm.assert_numpy_array_equal(result, exp)

        result = isna(["foo", "bar"])
        exp = np.array([False, False])
        tm.assert_numpy_array_equal(result, exp)

        # GH20675
        result = isna([np.nan, "world"])
        exp = np.array([True, False])
        tm.assert_numpy_array_equal(result, exp)

    def test_isna_nat(self):
        # [FUNCTION DOC]
        # A list with NaT, and an object array with NaT, are both detected as missing.
        result = isna([NaT])
        exp = np.array([True])
        tm.assert_numpy_array_equal(result, exp)

        result = isna(np.array([NaT], dtype=object))
        exp = np.array([True])
        tm.assert_numpy_array_equal(result, exp)

    def test_isna_numpy_nat(self):
        # [FUNCTION DOC]
        # An array mixing pandas NaT and numpy datetime/timedelta NaT values in several units is all
        # missing.
        arr = np.array(
            [
                NaT,
                np.datetime64("NaT", "ns"),
                np.timedelta64("NaT", "ns"),
                np.datetime64("NaT", "s"),
            ]
        )
        result = isna(arr)
        expected = np.array([True] * 4)
        tm.assert_numpy_array_equal(result, expected)

    def test_isna_datetime(self):
        # [FUNCTION DOC]
        # A current datetime is not missing (and notna is true). A 20-day date range is all not missing.
        # Setting the first entry of a copy of its values to the integer NaT sentinel makes the first entry
        # missing and the rest not. The same holds for the monthly period conversion (regression for an
        # upstream issue), and a slice without the first entry has no missing values.
        assert not isna(datetime.now())
        assert notna(datetime.now())

        idx = date_range("1/1/1990", periods=20)
        exp = np.ones(len(idx), dtype=bool)
        tm.assert_numpy_array_equal(notna(idx), exp)

        idx = np.asarray(idx)
        idx[0] = iNaT
        idx = DatetimeIndex(idx)
        mask = isna(idx)
        assert mask[0]
        exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

        # GH 9129
        pidx = idx.to_period(freq="M")
        mask = isna(pidx)
        assert mask[0]
        exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

        mask = isna(pidx[1:])
        exp = np.zeros(len(mask), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

    def test_isna_old_datetimelike(self):
        # [FUNCTION DOC]
        # For a three-element datetime array whose last entry is set to NaT (mutating the underlying
        # array of a date range), the predicate returns false, false, true for the naive array, the
        # time-zone localized array, the difference array (timedelta) and the period conversion.
        # isna_old should work for dt64tz, td64, and period, not just tznaive
        dti = date_range("2016-01-01", periods=3)
        dta = dti._data
        dta[-1] = NaT
        expected = np.array([False, False, True], dtype=bool)

        objs = [dta, dta.tz_localize("US/Eastern"), dta - dta, dta.to_period("D")]

        for obj in objs:
            result = isna(obj)

            tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (np.complex128(np.nan), True),
            (np.float64(1), False),
            (np.array([1, 1 + 0j, np.nan, 3]), np.array([False, False, True, False])),
            (
                np.array([1, 1 + 0j, np.nan, 3], dtype=object),
                np.array([False, False, True, False]),
            ),
            (
                np.array([1, 1 + 0j, np.nan, 3]).astype(object),
                np.array([False, False, True, False]),
            ),
        ],
    )
    def test_complex(self, value, expected):
        # [FUNCTION DOC]
        # Parametrized over complex scalars and arrays (numeric and object dtype, with NaN). For a scalar
        # result asserts identity with the expected boolean; for arrays asserts array equality.
        result = isna(value)
        if is_scalar(result):
            assert result is expected
        else:
            tm.assert_numpy_array_equal(result, expected)

    def test_datetime_other_units(self):
        # [FUNCTION DOC]
        # For a datetime index with one NaT: the predicate and its negation, on the index and on its raw
        # values, match the expected mask.
        idx = DatetimeIndex(["2011-01-01", "NaT", "2011-01-02"])
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        tm.assert_numpy_array_equal(isna(idx.values), exp)
        tm.assert_numpy_array_equal(notna(idx.values), ~exp)

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[D]",
            "datetime64[h]",
            "datetime64[m]",
            "datetime64[s]",
            "datetime64[ms]",
            "datetime64[us]",
            "datetime64[ns]",
        ],
    )
    def test_datetime_other_units_astype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over datetime64 units from days to nanoseconds. The raw values converted to that
        # unit give the expected mask for both predicates; a series from them, and an object-dtype series
        # from them, give the expected boolean series.
        idx = DatetimeIndex(["2011-01-01", "NaT", "2011-01-02"])
        values = idx.values.astype(dtype)

        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(values), exp)
        tm.assert_numpy_array_equal(notna(values), ~exp)

        exp = Series([False, True, False])
        s = Series(values)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(values, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_timedelta_other_units(self):
        # [FUNCTION DOC]
        # Timedelta counterpart of the datetime other-units test using a timedelta index.
        idx = TimedeltaIndex(["1 days", "NaT", "2 days"])
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        tm.assert_numpy_array_equal(isna(idx.values), exp)
        tm.assert_numpy_array_equal(notna(idx.values), ~exp)

    @pytest.mark.parametrize(
        "dtype",
        [
            "timedelta64[D]",
            "timedelta64[h]",
            "timedelta64[m]",
            "timedelta64[s]",
            "timedelta64[ms]",
            "timedelta64[us]",
            "timedelta64[ns]",
        ],
    )
    def test_timedelta_other_units_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over timedelta64 units; timedelta counterpart of the datetime astype test.
        idx = TimedeltaIndex(["1 days", "NaT", "2 days"])
        values = idx.values.astype(dtype)

        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(values), exp)
        tm.assert_numpy_array_equal(notna(values), ~exp)

        exp = Series([False, True, False])
        s = Series(values)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(values, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_period(self):
        # [FUNCTION DOC]
        # A period index with one NaT: both predicates give the expected mask for the index, for a series
        # and for an object-dtype series.
        idx = pd.PeriodIndex(["2011-01", "NaT", "2012-01"], freq="M")
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)

        exp = Series([False, True, False])
        s = Series(idx)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(idx, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_decimal(self):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Decimal scalars: 1.0 is not missing, NaN is, and the results
        # are real booleans (identity checks). For an object array, series and index holding both, the
        # predicates give false, true and the negation.
        # scalars GH#23530
        a = Decimal("1.0")
        assert isna(a) is False
        assert notna(a) is True

        b = Decimal("NaN")
        assert isna(b) is True
        assert notna(b) is False

        # array
        arr = np.array([a, b])
        expected = np.array([False, True])
        result = isna(arr)
        tm.assert_numpy_array_equal(result, expected)

        result = notna(arr)
        tm.assert_numpy_array_equal(result, ~expected)

        # series
        ser = Series(arr)
        expected = Series(expected)
        result = isna(ser)
        tm.assert_series_equal(result, expected)

        result = notna(ser)
        tm.assert_series_equal(result, ~expected)

        # index
        idx = Index(arr)
        expected = np.array([False, True])
        result = isna(idx)
        tm.assert_numpy_array_equal(result, expected)

        result = notna(idx)
        tm.assert_numpy_array_equal(result, ~expected)


@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent(dtype_equal):
    # [FUNCTION DOC]
    # Parametrized over dtype_equal. Checks array equivalence is true for identical arrays containing
    # NaN (float), NaN and a number, NaN and none in object dtype, nested object arrays, complex
    # values; false when a value differs, when lengths differ, and for indexes with differing
    # values; true for equal indexes.
    assert array_equivalent(
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), dtype_equal=dtype_equal
    )
    assert array_equivalent(
        np.array([np.nan, 1, np.nan]),
        np.array([np.nan, 1, np.nan]),
        dtype_equal=dtype_equal,
    )
    assert array_equivalent(
        np.array([np.nan, None], dtype="object"),
        np.array([np.nan, None], dtype="object"),
        dtype_equal=dtype_equal,
    )
    # Check the handling of nested arrays in array_equivalent_object
    assert array_equivalent(
        np.array([np.array([np.nan, None], dtype="object"), None], dtype="object"),
        np.array([np.array([np.nan, None], dtype="object"), None], dtype="object"),
        dtype_equal=dtype_equal,
    )
    assert array_equivalent(
        np.array([np.nan, 1 + 1j], dtype="complex"),
        np.array([np.nan, 1 + 1j], dtype="complex"),
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        np.array([np.nan, 1 + 1j], dtype="complex"),
        np.array([np.nan, 1 + 2j], dtype="complex"),
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        np.array([np.nan, 1, np.nan]),
        np.array([np.nan, 2, np.nan]),
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        np.array(["a", "b", "c", "d"]), np.array(["e", "e"]), dtype_equal=dtype_equal
    )
    assert array_equivalent(
        Index([0, np.nan]), Index([0, np.nan]), dtype_equal=dtype_equal
    )
    assert not array_equivalent(
        Index([0, np.nan]), Index([1, np.nan]), dtype_equal=dtype_equal
    )


@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent_tdi(dtype_equal):
    # [FUNCTION DOC]
    # Parametrized over dtype_equal. Equal timedelta indexes with NaT are equivalent; ones with
    # different values are not.
    assert array_equivalent(
        TimedeltaIndex([0, np.nan]),
        TimedeltaIndex([0, np.nan]),
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        TimedeltaIndex([0, np.nan]),
        TimedeltaIndex([1, np.nan]),
        dtype_equal=dtype_equal,
    )


@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent_dti(dtype_equal):
    # [FUNCTION DOC]
    # Parametrized over dtype_equal. Datetime indexes with NaT: equal are equivalent, differing are
    # not; time-zone aware indexes with the same zone behave likewise. Without the dtype_equal
    # shortcut: a naive index is not equivalent to a zone-aware one; two zone-aware indexes with
    # different zones but the same instants are equivalent; a datetime index is not equivalent to a
    # timedelta index.
    assert array_equivalent(
        DatetimeIndex([0, np.nan]), DatetimeIndex([0, np.nan]), dtype_equal=dtype_equal
    )
    assert not array_equivalent(
        DatetimeIndex([0, np.nan]), DatetimeIndex([1, np.nan]), dtype_equal=dtype_equal
    )

    dti1 = DatetimeIndex([0, np.nan], tz="US/Eastern")
    dti2 = DatetimeIndex([0, np.nan], tz="CET")
    dti3 = DatetimeIndex([1, np.nan], tz="US/Eastern")

    assert array_equivalent(
        dti1,
        dti1,
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        dti1,
        dti3,
        dtype_equal=dtype_equal,
    )
    # The rest are not dtype_equal
    assert not array_equivalent(DatetimeIndex([0, np.nan]), dti1)
    assert array_equivalent(
        dti2,
        dti1,
    )

    assert not array_equivalent(DatetimeIndex([0, np.nan]), TimedeltaIndex([0, np.nan]))


@pytest.mark.parametrize(
    "val", [1, 1.1, 1 + 1j, True, "abc", [1, 2], (1, 2), {1, 2}, {"a": 1}, None]
)
def test_array_equivalent_series(val):
    # [FUNCTION DOC]
    # Parametrized over a variety of second elements (numbers, bool, string, list, tuple, set, dict,
    # none). A series of two arrays is not equivalent to a series whose second element is the
    # parametrized value.
    arr = np.array([1, 2])
    assert not array_equivalent(Series([arr, arr]), Series([arr, val]))


def test_array_equivalent_array_mismatched_shape():
    # [FUNCTION DOC]
    # Regression for an upstream bug. Two object series whose first elements are arrays of
    # different lengths (that share their leading entries) are not equivalent.
    # to trigger the motivating bug, the first N elements of the arrays need
    #  to match
    first = np.array([1, 2, 3])
    second = np.array([1, 2])

    left = Series([first, "a"], dtype=object)
    right = Series([second, "a"], dtype=object)
    assert not array_equivalent(left, right)


def test_array_equivalent_array_mismatched_dtype():
    # [FUNCTION DOC]
    # Two object series whose first elements are float and integer arrays with equal values are
    # equivalent (same shape, different dtype can still be equivalent).
    # same shape, different dtype can still be equivalent
    first = np.array([1, 2], dtype=np.float64)
    second = np.array([1, 2])

    left = Series([first, "a"], dtype=object)
    right = Series([second, "a"], dtype=object)
    assert array_equivalent(left, right)


def test_array_equivalent_different_dtype_but_equal():
    # [FUNCTION DOC]
    # An integer array and a float array with equal values are equivalent (the code notes it is unclear
    # whether this is part of the public interface).
    # Unclear if this is exposed anywhere in the public-facing API
    assert array_equivalent(np.array([1, 2]), np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "lvalue, rvalue",
    [
        # There are 3 variants for each of lvalue and rvalue. We include all
        #  three for the tz-naive `now` and exclude the datetim64 variant
        #  for utcnow because it drops tzinfo.
        (fix_now, fix_utcnow),
        (fix_now.to_datetime64(), fix_utcnow),
        (fix_now.to_pydatetime(), fix_utcnow),
        (fix_now.to_datetime64(), fix_utcnow.to_pydatetime()),
        (fix_now.to_pydatetime(), fix_utcnow.to_pydatetime()),
    ],
)
def test_array_equivalent_tzawareness(lvalue, rvalue):
    # [FUNCTION DOC]
    # Parametrized over pairs of a naive and a UTC timestamp in several representations (timestamp,
    # numpy datetime, python datetime). Comparing object arrays holding a naive and a zone-aware
    # value must not raise and must report not equivalent for both strict and non-strict NaN
    # handling.
    # we shouldn't raise if comparing tzaware and tznaive datetimes
    left = np.array([lvalue], dtype=object)
    right = np.array([rvalue], dtype=object)

    assert not array_equivalent(left, right, strict_nan=True)
    assert not array_equivalent(left, right, strict_nan=False)


def test_array_equivalent_compat():
    # [FUNCTION DOC]
    # Regression for an upstream issue. Structured arrays: equal arrays are equivalent, arrays with
    # differing values are not, and arrays with differing field names are not, each for strict and
    # non-strict NaN handling.
    # see gh-13388
    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    assert array_equivalent(m, n, strict_nan=True)
    assert array_equivalent(m, n, strict_nan=False)

    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (4, 3)], dtype=[("a", int), ("b", float)])
    assert not array_equivalent(m, n, strict_nan=True)
    assert not array_equivalent(m, n, strict_nan=False)

    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (3, 4)], dtype=[("b", int), ("a", float)])
    assert not array_equivalent(m, n, strict_nan=True)
    assert not array_equivalent(m, n, strict_nan=False)


@pytest.mark.parametrize("dtype", ["O", "S", "U"])
def test_array_equivalent_str(dtype):
    # [FUNCTION DOC]
    # Parametrized over object, byte string and unicode dtypes: arrays of two strings are equivalent
    # when equal and not equivalent when one element differs.
    assert array_equivalent(
        np.array(["A", "B"], dtype=dtype), np.array(["A", "B"], dtype=dtype)
    )
    assert not array_equivalent(
        np.array(["A", "B"], dtype=dtype), np.array(["A", "X"], dtype=dtype)
    )


@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested(strict_nan):
    # [FUNCTION DOC]
    # Parametrized over strict_nan. Object arrays of numpy arrays (as in groupby aggregations): equal
    # ones are equivalent and the reversed one is not (checking the truthiness is done with an
    # any-reduction); the same with elements assigned into an empty object array; an object array of
    # arrays versus a plain integer array is not equivalent.
    # reached in groupby aggregations, make sure we use np.any when checking
    #  if the comparison is truthy
    left = np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object)
    right = np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object)

    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.empty(2, dtype=object)
    left[:] = [np.array([50, 70, 90]), np.array([20, 30, 40])]
    right = np.empty(2, dtype=object)
    right[:] = [np.array([50, 70, 90]), np.array([20, 30, 40])]
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([np.array([50, 50, 50]), np.array([40, 40])], dtype=object)
    right = np.array([50, 40])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested2(strict_nan):
    # [FUNCTION DOC]
    # Parametrized over strict_nan; elementwise comparison deprecation warnings are ignored.
    # Object arrays with two levels of nested arrays: equal are equivalent, reversed are not, and
    # compared to a plain integer array they are not.
    # more than one level of nesting
    left = np.array(
        [
            np.array([np.array([50, 70]), np.array([90])], dtype=object),
            np.array([np.array([20, 30])], dtype=object),
        ],
        dtype=object,
    )
    right = np.array(
        [
            np.array([np.array([50, 70]), np.array([90])], dtype=object),
            np.array([np.array([20, 30])], dtype=object),
        ],
        dtype=object,
    )
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([np.array([np.array([50, 50, 50])], dtype=object)], dtype=object)
    right = np.array([50])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested_list(strict_nan):
    # [FUNCTION DOC]
    # Parametrized over strict_nan. Object arrays whose elements are lists: equal ones are equivalent,
    # reversed are not, and versus an integer array they are not.
    left = np.array([[50, 70, 90], [20, 30]], dtype=object)
    right = np.array([[50, 70, 90], [20, 30]], dtype=object)

    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([[50, 50, 50], [40, 40]], dtype=object)
    right = np.array([50, 40])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
@pytest.mark.xfail(reason="failing")
@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested_mixed_list(strict_nan):
    # [FUNCTION DOC]
    # Marked as an expected failure (reason stated only as failing; upstream issue). Parametrized
    # over strict_nan; elementwise comparison deprecation warnings are ignored. Intended behavior:
    # object arrays mixing numpy arrays and lists at the same nesting positions (one level, several
    # levels, and same-length lists) are equivalent to each other and not equivalent to reversed
    # versions.
    # mixed arrays / lists in left and right
    # https://github.com/pandas-dev/pandas/issues/50360
    left = np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object)
    right = np.array([[1, 2, 3], [4, 5]], dtype=object)

    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    # multiple levels of nesting
    left = np.array(
        [
            np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
            np.array([np.array([6]), np.array([7, 8]), np.array([9])], dtype=object),
        ],
        dtype=object,
    )
    right = np.array([[[1, 2, 3], [4, 5]], [[6], [7, 8], [9]]], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    # same-length lists
    subarr = np.empty(2, dtype=object)
    subarr[:] = [
        np.array([None, "b"], dtype=object),
        np.array(["c", "d"], dtype=object),
    ]
    left = np.array([subarr, None], dtype=object)
    right = np.array([[[None, "b"], ["c", "d"]], None], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)


@pytest.mark.xfail(reason="failing")
@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested_dicts(strict_nan):
    # [FUNCTION DOC]
    # Marked as an expected failure (reason stated only as failing). Parametrized over strict_nan.
    # Intended behavior: object arrays of dicts holding arrays are equivalent when equal, also
    # when the inner arrays are replaced by lists, and the reversed arrays are not.
    left = np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object)
    right = np.array(
        [{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object
    )
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    right2 = np.array([{"f1": 1, "f2": ["a", "b"]}], dtype=object)
    assert array_equivalent(left, right2, strict_nan=strict_nan)
    assert not array_equivalent(left, right2[::-1], strict_nan=strict_nan)


def test_array_equivalent_index_with_tuples():
    # [FUNCTION DOC]
    # Regression for an upstream issue. Indexes of tuples containing the NA singleton in different
    # positions are not equivalent when their order differs, in both directions, and the index
    # equality method agrees (also not equal).
    # GH#48446
    idx1 = Index(np.array([(pd.NA, 4), (1, 1)], dtype="object"))
    idx2 = Index(np.array([(1, 1), (pd.NA, 4)], dtype="object"))
    assert not array_equivalent(idx1, idx2)
    assert not idx1.equals(idx2)
    assert not array_equivalent(idx2, idx1)
    assert not idx2.equals(idx1)

    idx1 = Index(np.array([(4, pd.NA), (1, 1)], dtype="object"))
    idx2 = Index(np.array([(1, 1), (4, pd.NA)], dtype="object"))
    assert not array_equivalent(idx1, idx2)
    assert not idx1.equals(idx2)
    assert not array_equivalent(idx2, idx1)
    assert not idx2.equals(idx1)


@pytest.mark.parametrize(
    "dtype, na_value",
    [
        # Datetime-like
        (np.dtype("M8[ns]"), np.datetime64("NaT", "ns")),
        (np.dtype("m8[ns]"), np.timedelta64("NaT", "ns")),
        (DatetimeTZDtype.construct_from_string("datetime64[ns, US/Eastern]"), NaT),
        (PeriodDtype("M"), NaT),
        # Integer
        ("u1", 0),
        ("u2", 0),
        ("u4", 0),
        ("u8", 0),
        ("i1", 0),
        ("i2", 0),
        ("i4", 0),
        ("i8", 0),
        # Bool
        ("bool", False),
        # Float
        ("f2", np.nan),
        ("f4", np.nan),
        ("f8", np.nan),
        # Complex
        ("c8", np.nan),
        ("c16", np.nan),
        # Object
        ("O", np.nan),
        # Interval
        (IntervalDtype(), np.nan),
    ],
)
def test_na_value_for_dtype(dtype, na_value):
    # [FUNCTION DOC]
    # Parametrized over dtypes and expected missing markers: datetime and timedelta nanoseconds
    # (typed NaT), time-zone aware datetime and period (NaT), every integer width (zero), boolean
    # (false), floats and complex (NaN), object (NaN) and interval (NaN). Asserts that the result is
    # the same object as expected or, for the non-singleton numpy NaT values, that both are missing
    # and have the same type.
    result = na_value_for_dtype(pandas_dtype(dtype))
    # identify check doesn't work for datetime64/timedelta64("NaT") bc they
    #  are not singletons
    assert result is na_value or (
        isna(result) and isna(na_value) and type(result) is type(na_value)
    )


class TestNAObj:
    # [CLASS DOC]
    # Tests of the low-level object-array missing check from the missing-value library module.
    def _check_behavior(self, arr, expected):
        # [FUNCTION DOC]
        # HELPER: runs the object missing check on an array and compares with the expected boolean array,
        # then repeats for the two-dimensional version of both and again after converting to Fortran
        # memory order. Inputs: arr and expected.
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)

        arr = np.atleast_2d(arr)
        expected = np.atleast_2d(expected)

        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)

        # Test fortran order
        arr = arr.copy(order="F")
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)

    def test_basic(self):
        # [FUNCTION DOC]
        # A mixed array of integer, none, string, float, NaT and NaN is detected as false, true, false,
        # false, true, true.
        arr = np.array([1, None, "foo", -5.1, NaT, np.nan])
        expected = np.array([False, True, False, False, True, True])

        self._check_behavior(arr, expected)

    def test_non_obj_dtype(self):
        # [FUNCTION DOC]
        # A float array with a NaN is handled and gives the expected mask.
        arr = np.array([1, 3, np.nan, 5], dtype=float)
        expected = np.array([False, False, True, False])

        self._check_behavior(arr, expected)

    def test_empty_arr(self):
        # [FUNCTION DOC]
        # An empty array gives an empty boolean array.
        arr = np.array([])
        expected = np.array([], dtype=bool)

        self._check_behavior(arr, expected)

    def test_empty_str_inp(self):
        # [FUNCTION DOC]
        # An array holding an empty string gives false (empty is not missing).
        arr = np.array([""])  # empty but not na
        expected = np.array([False])

        self._check_behavior(arr, expected)

    def test_empty_like(self):
        # [FUNCTION DOC]
        # Regression for an upstream issue (no segmentation fault): an array made with the empty-like
        # constructor from a list with none gives true.
        # see gh-13717: no segfaults!
        arr = np.empty_like([None])
        expected = np.array([True])

        self._check_behavior(arr, expected)


m8_units = ["as", "ps", "ns", "us", "ms", "s", "m", "h", "D", "W", "M", "Y"]

na_vals = (
    [
        None,
        NaT,
        float("NaN"),
        complex("NaN"),
        np.nan,
        np.float64("NaN"),
        np.float32("NaN"),
        np.complex64(np.nan),
        np.complex128(np.nan),
        np.datetime64("NaT", "ns"),
        np.timedelta64("NaT", "ns"),
    ]
    + [np.datetime64("NaT", unit) for unit in m8_units]  # type: ignore[call-overload]
    + [np.timedelta64("NaT", unit) for unit in m8_units]  # type: ignore[call-overload]
)

inf_vals = [
    float("inf"),
    float("-inf"),
    complex("inf"),
    complex("-inf"),
    np.inf,
    -np.inf,
]

int_na_vals = [
    # Values that match iNaT, which we treat as null in specific cases
    np.int64(NaT._value),
    int(NaT._value),
]

sometimes_na_vals = [Decimal("NaN")]

never_na_vals = [
    # float/complex values that when viewed as int64 match iNaT
    -0.0,
    np.float64("-0.0"),
    -0j,
    np.complex64(-0j),
]


class TestLibMissing:
    # [CLASS DOC]
    # Tests of the scalar missing check and the matching-missing comparison from the missing-value
    # library module, using the module-level value lists.
    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize(
        "value",
        na_vals + sometimes_na_vals,  # type: ignore[operator]
    )
    def test_checknull_na_vals(self, func, value):
        # [FUNCTION DOC]
        # Parametrized over the library check and the public predicate, and over every missing-like value
        # including decimal NaN. Each function reports missing.
        assert func(value)

    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize("value", inf_vals)
    def test_checknull_inf_vals(self, func, value):
        # [FUNCTION DOC]
        # Parametrized over both functions and the infinite values; none is reported missing.
        assert not func(value)

    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize("value", int_na_vals)
    def test_checknull_intna_vals(self, func, value):
        # [FUNCTION DOC]
        # Parametrized over both functions and integers equal to the NaT sentinel; none is reported
        # missing.
        assert not func(value)

    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize("value", never_na_vals)
    def test_checknull_never_na_vals(self, func, value):
        # [FUNCTION DOC]
        # Parametrized over both functions and negative-zero float and complex values; none is reported
        # missing.
        assert not func(value)

    @pytest.mark.parametrize(
        "value",
        na_vals + sometimes_na_vals,  # type: ignore[operator]
    )
    def test_checknull_old_na_vals(self, value):
        # [FUNCTION DOC]
        # Parametrized over the missing-like values including decimal NaN; the library check reports
        # missing.
        # KNOWN ISSUE: it duplicates the library-function half of the earlier parametrization of the
        # same values (the old-versus-new distinction in the name no longer exists).
        assert libmissing.checknull(value)

    @pytest.mark.parametrize("value", int_na_vals)
    def test_checknull_old_intna_vals(self, value):
        # [FUNCTION DOC]
        # Parametrized over the NaT-sentinel integers; the library check reports not missing.
        # KNOWN ISSUE: it duplicates the library-function half of the earlier parametrization of the
        # same values (the old-versus-new distinction in the name no longer exists).
        assert not libmissing.checknull(value)

    def test_is_matching_na(self, nulls_fixture, nulls_fixture2):
        # [FUNCTION DOC]
        # Uses the two null fixtures. A missing value matches itself. Otherwise two values match if
        # they are the same object, if both are floats (a float NaN matches another NaN), or if they have
        # the same type (such as two decimal NaN); in all other combinations they do not match.
        left = nulls_fixture
        right = nulls_fixture2

        assert libmissing.is_matching_na(left, left)

        if left is right:
            assert libmissing.is_matching_na(left, right)
        elif is_float(left) and is_float(right):
            # np.nan vs float("NaN") we consider as matching
            assert libmissing.is_matching_na(left, right)
        elif type(left) is type(right):
            # e.g. both Decimal("NaN")
            assert libmissing.is_matching_na(left, right)
        else:
            assert not libmissing.is_matching_na(left, right)

    def test_is_matching_na_nan_matches_none(self):
        # [FUNCTION DOC]
        # By default none and NaN do not match in either order; with the nan_matches_none option enabled
        # they match in both orders.
        assert not libmissing.is_matching_na(None, np.nan)
        assert not libmissing.is_matching_na(np.nan, None)

        assert libmissing.is_matching_na(None, np.nan, nan_matches_none=True)
        assert libmissing.is_matching_na(np.nan, None, nan_matches_none=True)


class TestIsValidNAForDtype:
    # [CLASS DOC]
    # Tests of whether a missing marker is valid for a given dtype.
    def test_is_valid_na_for_dtype_interval(self):
        # [FUNCTION DOC]
        # For interval dtypes (integer left-closed and datetime both-closed), NaT is not a valid missing
        # marker.
        dtype = IntervalDtype("int64", "left")
        assert not is_valid_na_for_dtype(NaT, dtype)

        dtype = IntervalDtype("datetime64[ns]", "both")
        assert not is_valid_na_for_dtype(NaT, dtype)

    def test_is_valid_na_for_dtype_categorical(self):
        # [FUNCTION DOC]
        # For a categorical dtype with categories 0, 1, 2: NaN is a valid missing marker; NaT and numpy
        # datetime and timedelta NaT are not.
        dtype = CategoricalDtype(categories=[0, 1, 2])
        assert is_valid_na_for_dtype(np.nan, dtype)

        assert not is_valid_na_for_dtype(NaT, dtype)
        assert not is_valid_na_for_dtype(np.datetime64("NaT", "ns"), dtype)
        assert not is_valid_na_for_dtype(np.timedelta64("NaT", "ns"), dtype)
