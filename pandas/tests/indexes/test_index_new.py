"""
Tests for the Index constructor conducting inference.
"""

# [MODULE DOC]
# Tests of the Index constructor: dtype inference from various inputs (all-bool, all-complex, all-missing, tuples, periods, datetime and timedelta with missing values), enforcement of an explicitly requested dtype, unwrapping of array-like inputs such as series and objects implementing the array protocol, and construction errors. Organized in four classes: inference, dtype enforcement, unwrapping, and errors.

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import numpy as np
import pytest

from pandas._libs.tslibs.timezones import maybe_get_tz

from pandas import (
    NA,
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    array,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestIndexConstructorInference:
    # [CLASS DOC]
    # Tests of the dtype chosen when no dtype is given.
    def test_object_all_bools(self):
        # [FUNCTION DOC]
        # An object array of only booleans stays object dtype in an index, matching a series.
        # GH#49594 match Series behavior on ndarray[object] of all bools
        arr = np.array([True, False], dtype=object)
        res = Index(arr)
        assert res.dtype == object

        # since the point is matching Series behavior, let's double check
        assert Series(arr).dtype == object

    def test_object_all_complex(self):
        # [FUNCTION DOC]
        # An object array of only complex numbers stays object dtype in an index, matching a series.
        # GH#49594 match Series behavior on ndarray[object] of all complex
        arr = np.array([complex(1), complex(2)], dtype=object)
        res = Index(arr)
        assert res.dtype == object

        # since the point is matching Series behavior, let's double check
        assert Series(arr).dtype == object

    @pytest.mark.parametrize("val", [NaT, None, np.nan, float("nan")])
    def test_infer_nat(self, val):
        # [FUNCTION DOC]
        # Parametrized over a second missing value (NaT, None, NaN, float NaN).
        # KNOWN ISSUE: the in-source comment claims nanosecond resolution but the assertions expect second resolution.
        # A pair of NaT and another missing marker, in either order, from a list or an object array, infers a datetime index with second resolution where every entry is missing.
        # GH#49340 all NaT/None/nan and at least 1 NaT -> datetime64[ns],
        #  matching Series behavior
        values = [NaT, val]

        idx = Index(values)
        assert idx.dtype == "datetime64[s]" and idx.isna().all()

        idx = Index(values[::-1])
        assert idx.dtype == "datetime64[s]" and idx.isna().all()

        idx = Index(np.array(values, dtype=object))
        assert idx.dtype == "datetime64[s]" and idx.isna().all()

        idx = Index(np.array(values, dtype=object)[::-1])
        assert idx.dtype == "datetime64[s]" and idx.isna().all()

    @pytest.mark.parametrize("na_value", [None, np.nan])
    @pytest.mark.parametrize("vtype", [list, tuple, iter])
    def test_construction_list_tuples_nan(self, na_value, vtype):
        # [FUNCTION DOC]
        # Parametrized over missing marker (None, NaN) and container type (list, tuple, iterator). A sequence of tuples containing a missing value builds a multi-level index equal to constructing it from tuples.
        # GH#18505 : valid tuples containing NaN
        values = [(1, "two"), (3.0, na_value)]
        result = Index(vtype(values))
        expected = MultiIndex.from_tuples(values)
        tm.assert_index_equal(result, expected)

    def test_constructor_int_dtype_float(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Uses the integer numpy dtype fixture. Floats that are whole numbers constructed with an integer dtype equal the integers constructed directly.
        # GH#18400
        expected = Index([0, 1, 2, 3], dtype=any_int_numpy_dtype)
        result = Index([0.0, 1.0, 2.0, 3.0], dtype=any_int_numpy_dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize(
        "vals", [[True, False, True], np.array([True, False, True], dtype=bool)]
    )
    def test_constructor_dtypes_to_object(self, cast_index, vals):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the name says object but the test asserts the result is boolean dtype.
        # Parametrized over cast flag and booleans as list or array. An index of booleans (with or without explicit bool dtype) is a plain Index (not a subclass) with boolean dtype.
        if cast_index:
            index = Index(vals, dtype=bool)
        else:
            index = Index(vals)

        assert type(index) is Index
        assert index.dtype == bool

    def test_constructor_categorical_to_object(self):
        # [FUNCTION DOC]
        # Constructing from a categorical index with dtype object yields an index that is no longer categorical.
        # GH#32167 Categorical data and dtype=object should return object-dtype
        ci = CategoricalIndex(range(5))
        result = Index(ci, dtype=object)
        assert not isinstance(result, CategoricalIndex)

    def test_constructor_infer_periodindex(self):
        # [FUNCTION DOC]
        # Constructing from a period index returns an equal period index of the same class.
        xp = period_range("2012-1-1", freq="M", periods=3)
        rs = Index(xp)
        tm.assert_index_equal(rs, xp)
        assert isinstance(rs, PeriodIndex)

    def test_from_list_of_periods(self):
        # [FUNCTION DOC]
        # Constructing from a list of period scalars infers a period index.
        rng = period_range("1/1/2000", periods=20, freq="D")
        periods = list(rng)

        result = Index(periods)
        assert isinstance(result, PeriodIndex)

    @pytest.mark.parametrize("pos", [0, 1])
    @pytest.mark.parametrize(
        "klass,dtype,ctor",
        [
            (DatetimeIndex, "datetime64[ns]", np.datetime64("nat", "ns")),
            (TimedeltaIndex, "timedelta64[ns]", np.timedelta64("NaT", "ns")),
        ],
    )
    def test_constructor_infer_nat_dt_like(
        self, pos, klass, dtype, ctor, nulls_fixture, request
    ):
        # [FUNCTION DOC]
        # Parametrized over position (0 or 1), the class/dtype/NaT-constructor triples (datetime64 and timedelta64 NaT), and the nulls fixture. Skips Decimal missing values (not cast). Builds a list with the typed NaT and the null at the given position; the index built from the list or object array equals an all-NaT index of the expected nanosecond datetime or timedelta type. For the NA marker the expectation becomes an object index of NA and NaT and the test is marked expected-fail (known issue with numpy NaT constructor).
        # KNOWN ISSUE: the comments in the body say the all-NaT index is inferred with second resolution but the code casts the expectation to nanosecond units.
        if isinstance(nulls_fixture, Decimal):
            # We dont cast these to datetime64/timedelta64
            pytest.skip(
                f"We don't cast {type(nulls_fixture).__name__} to "
                "datetime64/timedelta64"
            )

        expected = klass([NaT, NaT])
        if dtype[0] == "d":
            # we infer all-NaT as second resolution
            expected = expected.astype("M8[ns]")
        if dtype[0] == "t":
            # we infer all-NaT as second resolution
            expected = expected.astype("m8[ns]")
        assert expected.dtype == dtype
        data = [ctor]
        data.insert(pos, nulls_fixture)

        if nulls_fixture is NA:
            expected = Index([NA, NaT])
            mark = pytest.mark.xfail(reason="Broken with np.NaT ctor; see GH 31884")
            request.applymarker(mark)

        result = Index(data)

        tm.assert_index_equal(result, expected)

        result = Index(np.array(data, dtype=object))

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("swap_objs", [True, False])
    def test_constructor_mixed_nat_objs_infers_object(self, swap_objs):
        # [FUNCTION DOC]
        # Parametrized over order. A numpy datetime NaT mixed with a numpy timedelta NaT infers object dtype, as list or object array.
        # mixed np.datetime64/timedelta64 nat results in object
        data = [np.datetime64("nat", "ns"), np.timedelta64("NaT", "ns")]
        if swap_objs:
            data = data[::-1]

        expected = Index(data, dtype=object)
        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    @pytest.mark.parametrize("swap_objs", [True, False])
    def test_constructor_datetime_and_datetime64(self, swap_objs):
        # [FUNCTION DOC]
        # Parametrized over order. A pandas timestamp and a numpy datetime built from list or object array infer a datetime index.
        data = [Timestamp(2021, 6, 8, 9, 42), np.datetime64("now")]
        if swap_objs:
            data = data[::-1]
        expected = DatetimeIndex(data)

        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    def test_constructor_datetimes_mixed_tzs(self):
        # [FUNCTION DOC]
        # Two datetimes with different time zones (Central and UTC) produce an object index containing the datetimes rather than being coerced to a single timezone.
        # https://github.com/pandas-dev/pandas/pull/55793/files#r1383719998
        tz = maybe_get_tz("US/Central")
        dt1 = datetime(2020, 1, 1, tzinfo=tz)
        dt2 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = Index([dt1, dt2])
        expected = Index([dt1, dt2], dtype=object)
        tm.assert_index_equal(result, expected)


class TestDtypeEnforced:
    # [CLASS DOC]
    # Tests that an explicitly passed dtype is never silently ignored.
    # check we don't silently ignore the dtype keyword

    def test_constructor_object_dtype_with_ea_data(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Uses the numeric extension dtype fixture. An index from nullable-array data with dtype object has object dtype.
        # GH#45206
        arr = array([0], dtype=any_numeric_ea_dtype)

        idx = Index(arr, dtype=object)
        assert idx.dtype == object

    @pytest.mark.parametrize("dtype", [object, "float64", "uint64", "category"])
    def test_constructor_range_values_mismatched_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype (object, float64, uint64, category). Constructing from an integer range index or range object with that dtype yields that dtype.
        rng = Index(range(5))

        result = Index(rng, dtype=dtype)
        assert result.dtype == dtype

        result = Index(range(5), dtype=dtype)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [object, "float64", "uint64", "category"])
    def test_constructor_categorical_values_mismatched_non_ea_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype. Constructing from a categorical with that dtype gives that dtype.
        cat = Categorical([1, 2, 3])

        result = Index(cat, dtype=dtype)
        assert result.dtype == dtype

    def test_constructor_categorical_values_mismatched_dtype(self):
        # [FUNCTION DOC]
        # A categorical of datetimes (naive and tz-aware) or intervals, with the underlying dtype passed explicitly, produces the plain datetime, tz-aware or interval index.
        dti = date_range("2016-01-01", periods=3)
        cat = Categorical(dti)
        result = Index(cat, dti.dtype)
        tm.assert_index_equal(result, dti)

        dti2 = dti.tz_localize("Asia/Tokyo")
        cat2 = Categorical(dti2)
        result = Index(cat2, dti2.dtype)
        tm.assert_index_equal(result, dti2)

        ii = IntervalIndex.from_breaks(range(5))
        cat3 = Categorical(ii)
        result = Index(cat3, dtype=ii.dtype)
        tm.assert_index_equal(result, ii)

    def test_constructor_ea_values_mismatched_categorical_dtype(self):
        # [FUNCTION DOC]
        # A naive or tz-aware datetime index with category dtype gives a categorical index of the same values.
        dti = date_range("2016-01-01", periods=3)
        result = Index(dti, dtype="category")
        expected = CategoricalIndex(dti)
        tm.assert_index_equal(result, expected)

        dti2 = date_range("2016-01-01", periods=3, tz="US/Pacific")
        result = Index(dti2, dtype="category")
        expected = CategoricalIndex(dti2)
        tm.assert_index_equal(result, expected)

    def test_constructor_period_values_mismatched_dtype(self):
        # [FUNCTION DOC]
        # A period index with category dtype gives the equivalent categorical index.
        pi = period_range("2016-01-01", periods=3, freq="D")
        result = Index(pi, dtype="category")
        expected = CategoricalIndex(pi)
        tm.assert_index_equal(result, expected)

    def test_constructor_timedelta64_values_mismatched_dtype(self):
        # [FUNCTION DOC]
        # A timedelta index with category dtype gives the equivalent categorical index.
        # check we don't silently ignore the dtype keyword
        tdi = timedelta_range("4 Days", periods=5)
        result = Index(tdi, dtype="category")
        expected = CategoricalIndex(tdi)
        tm.assert_index_equal(result, expected)

    def test_constructor_interval_values_mismatched_dtype(self):
        # [FUNCTION DOC]
        # An interval index with category dtype gives the equivalent categorical index.
        dti = date_range("2016-01-01", periods=3)
        ii = IntervalIndex.from_breaks(dti)
        result = Index(ii, dtype="category")
        expected = CategoricalIndex(ii)
        tm.assert_index_equal(result, expected)

    def test_constructor_datetime64_values_mismatched_period_dtype(self):
        # [FUNCTION DOC]
        # A datetime index with a daily period dtype is converted to the daily period index.
        dti = date_range("2016-01-01", periods=3)
        result = Index(dti, dtype="Period[D]")
        expected = dti.to_period("D")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "uint64"])
    def test_constructor_int_dtype_nan_raises(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over int64 and uint64. NaN with an integer dtype raises a value error that it cannot convert.
        # see GH#15187
        data = [np.nan]
        msg = "cannot convert"
        with pytest.raises(ValueError, match=msg):
            Index(data, dtype=dtype)

    @pytest.mark.parametrize(
        "vals",
        [
            [1, 2, 3],
            np.array([1, 2, 3]),
            np.array([1, 2, 3], dtype=int),
            # below should coerce
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0], dtype=float),
        ],
    )
    def test_constructor_dtypes_to_int(self, vals, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Parametrized over integer or whole-float values as list or array and the integer dtype fixture. The result has the requested integer dtype.
        dtype = any_int_numpy_dtype
        index = Index(vals, dtype=dtype)
        assert index.dtype == dtype

    @pytest.mark.parametrize(
        "vals",
        [
            [1, 2, 3],
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            np.array([1, 2, 3], dtype=int),
            np.array([1.0, 2.0, 3.0], dtype=float),
        ],
    )
    def test_constructor_dtypes_to_float(self, vals, float_numpy_dtype):
        # [FUNCTION DOC]
        # Parametrized over similar values and the float dtype fixture. The result has the requested float dtype.
        dtype = float_numpy_dtype
        index = Index(vals, dtype=dtype)
        assert index.dtype == dtype

    @pytest.mark.parametrize(
        "vals",
        [
            [1, 2, 3],
            np.array([1, 2, 3], dtype=int),
            np.array(["2011-01-01", "2011-01-02"], dtype="datetime64[ns]"),
            [datetime(2011, 1, 1), datetime(2011, 1, 2)],
        ],
    )
    def test_constructor_dtypes_to_categorical(self, vals):
        # [FUNCTION DOC]
        # Parametrized over integers, a numpy integer array, numpy datetimes and standard datetimes. With category dtype the result is a categorical index.
        index = Index(vals, dtype="category")
        assert isinstance(index, CategoricalIndex)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize(
        "vals",
        [
            np.array([np.datetime64("2011-01-01"), np.datetime64("2011-01-02")]),
            [datetime(2011, 1, 1), datetime(2011, 1, 2)],
        ],
    )
    def test_constructor_dtypes_to_datetime(self, cast_index, vals):
        # [FUNCTION DOC]
        # Parametrized over cast flag and numpy or standard datetimes. The values are first built into an index; with the cast flag and dtype object the result is an object-dtype index; otherwise it is a datetime index.
        vals = Index(vals)
        if cast_index:
            index = Index(vals, dtype=object)
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            index = Index(vals)
            assert isinstance(index, DatetimeIndex)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize(
        "vals",
        [
            np.array([np.timedelta64(1, "D"), np.timedelta64(1, "D")]),
            [timedelta(1), timedelta(1)],
        ],
    )
    def test_constructor_dtypes_to_timedelta(self, cast_index, vals):
        # [FUNCTION DOC]
        # Parametrized over cast flag and numpy or standard timedeltas. With the cast flag and object dtype the result is object dtype; otherwise a timedelta index.
        if cast_index:
            index = Index(vals, dtype=object)
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            index = Index(vals)
            assert isinstance(index, TimedeltaIndex)

    def test_pass_timedeltaindex_to_index(self):
        # [FUNCTION DOC]
        # A timedelta range as an index with dtype object has the same values as the index of its standard timedelta conversions.
        rng = timedelta_range("1 days", "10 days")
        idx = Index(rng, dtype=object)

        expected = Index(rng.to_pytimedelta(), dtype=object)

        tm.assert_numpy_array_equal(idx.values, expected.values)

    def test_pass_datetimeindex_to_index(self):
        # [FUNCTION DOC]
        # A datetime range as an index with dtype object has the same values as the index built from its standard datetime conversions.
        # GH#1396
        rng = date_range("1/1/2000", "3/1/2000")
        idx = Index(rng, dtype=object)

        expected = Index(rng.to_pydatetime(), dtype=object)

        tm.assert_numpy_array_equal(idx.values, expected.values)


class TestIndexConstructorUnwrapping:
    # [CLASS DOC]
    # Tests that array-like inputs are unwrapped into index values.
    # Test passing different arraylike values to pd.Index

    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    def test_constructor_from_series_dt64(self, klass):
        # [FUNCTION DOC]
        # Parametrized over the base class and datetime class. A series of timestamps converts to the equal datetime index.
        stamps = [Timestamp("20110101"), Timestamp("20120101"), Timestamp("20130101")]
        expected = DatetimeIndex(stamps)
        ser = Series(stamps)
        result = klass(ser)
        tm.assert_index_equal(result, expected)

    def test_constructor_no_pandas_array(self):
        # [FUNCTION DOC]
        # Passing the array of a series (the pandas array wrapper) gives an ordinary integer index.
        ser = Series([1, 2, 3])
        result = Index(ser.array)
        expected = Index([1, 2, 3])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "array",
        [
            np.arange(5),
            np.array(["a", "b", "c"]),
            date_range("2000-01-01", periods=3).values,
        ],
    )
    def test_constructor_ndarray_like(self, array):
        # [FUNCTION DOC]
        # Parametrized over an integer array, a string array and a datetime array. Defines a local wrapper class implementing the array protocol; constructing an index from the wrapper equals constructing from the raw array.
        # GH#5460#issuecomment-44474502
        # it should be possible to convert any object that satisfies the numpy
        # ndarray interface directly into an Index
        class ArrayLike:
            # [CLASS DOC]
            # HELPER: minimal wrapper object defined in the test to simulate an object satisfying the numpy array interface by returning a stored array.
            def __init__(self, array) -> None:
                # [FUNCTION DOC]
                # HELPER: stores the given array on the instance. No return value.
                self.array = array

            def __array__(self, dtype=None, copy=None) -> np.ndarray:
                # [FUNCTION DOC]
                # HELPER: array-protocol hook; accepts optional dtype and copy arguments (both ignored) and returns the stored array unchanged.
                return self.array

        expected = Index(array)
        result = Index(ArrayLike(array))
        tm.assert_index_equal(result, expected)


class TestIndexConstructionErrors:
    # [CLASS DOC]
    # Tests of construction failures.
    def test_constructor_overflow_int64(self):
        # [FUNCTION DOC]
        # A value that does not fit in a signed 64-bit integer (one below the maximum unsigned 64-bit value) with dtype int64 raises an overflow error stating the elements cannot all be cast to the dtype.
        # see GH#15832
        msg = (
            "The elements provided in the data cannot all be casted to the dtype int64"
        )
        with pytest.raises(OverflowError, match=msg):
            Index([np.iinfo(np.uint64).max - 1], dtype="int64")
