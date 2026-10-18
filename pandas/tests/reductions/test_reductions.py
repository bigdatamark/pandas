# [MODULE DOC]
# Tests for reductions (minimum, maximum, argument/index of min and max, sum, product,
# mean, variance, any, all, mode) that must behave consistently across index, series and
# data frame objects of many dtypes (numeric, boolean, datetime, timezone-aware, timedelta,
# period, categorical, string, object, nullable). Contains a module helper that builds a
# collection of test objects and six test classes: general reductions, reductions moved from
# index tests, from series tests, datetime64 series, categorical series, and series mode.
# Relies on shared fixtures (index-or-series factory, skipna flag, time unit, timezone,
# string dtypes, boolean reduction names and others) from the global test configuration.
# Importing the module builds the objects used for parametrization at collection time.

from datetime import (
    datetime,
    timedelta,
)
from decimal import Decimal

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
    isna,
    period_range,
    timedelta_range,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArray


def get_objs():
    # [FUNCTION DOC]
    # HELPER: builds the objects that parametrize the general min/max test. Takes no arguments.
    # Creates seven named indexes of length ten (booleans, 64-bit integers, 64-bit floats,
    # nanosecond datetimes, nanosecond datetimes localized to US/Eastern, daily periods, string
    # numerals), then one series of the same random float values per index (using that index and
    # the name a). Returns a list with the seven indexes followed by the seven series.
    indexes = [
        Index([True, False] * 5, name="a"),
        Index(np.arange(10), dtype=np.int64, name="a"),
        Index(np.arange(10), dtype=np.float64, name="a"),
        DatetimeIndex(date_range("2020-01-01", periods=10, unit="ns"), name="a"),
        DatetimeIndex(
            date_range("2020-01-01", periods=10, unit="ns"), name="a"
        ).tz_localize(tz="US/Eastern"),
        PeriodIndex(period_range("2020-01-01", periods=10, freq="D"), name="a"),
        Index([str(i) for i in range(10)], name="a"),
    ]

    arr = np.random.default_rng(2).standard_normal(10)
    series = [Series(arr, index=idx, name="a") for idx in indexes]

    objs = indexes + series
    return objs


class TestReductions:
    # [CLASS DOC]
    # General reduction tests applied to many index and series objects at once, plus a few
    # single-case regressions (empty/all-missing min and max, argument versions, timezone
    # handling, numpy interaction). Stateless.
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize("obj", get_objs())
    def test_ops(self, opname, obj):
        # [FUNCTION DOC]
        # Parametrized by maximum or minimum and by each helper-built object (index or series;
        # a deprecation warning about business-day periods is ignored). Computes the reduction
        # and compares it to the equivalent computed on the underlying values (via a numpy array for
        # Arrow-backed strings, via the ordinal integers for period indexes, and via the integer
        # value of the epoch-based result for timezone-aware data after dropping the zone).
        result = getattr(obj, opname)()
        if not isinstance(obj, PeriodIndex):
            if isinstance(obj.values, ArrowStringArray):
                # max not on the interface
                expected = getattr(np.array(obj.values), opname)()
            else:
                expected = getattr(obj.values, opname)()
        else:
            expected = Period(ordinal=getattr(obj.asi8, opname)(), freq=obj.freq)

        if getattr(obj, "tz", None) is not None:
            # We need to de-localize before comparing to the numpy-produced result
            expected = expected.astype("M8[ns]").astype("int64")
            assert result._value == expected
        else:
            assert result == expected

    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize(
        "dtype, val",
        [
            ("object", 2.0),
            ("float64", 2.0),
            ("datetime64[ns]", datetime(2011, 11, 1)),
            ("Int64", 2),
            ("boolean", True),
        ],
    )
    def test_nanminmax(self, opname, dtype, val, index_or_series):
        # [FUNCTION DOC]
        # Regression test (GH#7261), parametrized by max or min, by dtype with a sample value
        # (object, float, datetime, nullable integer, nullable boolean) and by index-or-series
        # factory. Checks that an all-missing or empty container gives the appropriate missing marker
        # for the dtype (not-a-time for datetimes, the nullable missing marker for masked dtypes,
        # a generic missing test otherwise), with and without skipping. For non-object dtypes also
        # checks that containers holding one valid value and missing values return that value when
        # skipping and a missing result otherwise.
        # GH#7261
        klass = index_or_series

        def check_missing(res):
            # [FUNCTION DOC]
            # HELPER: nested in the min/max missing-value test. Takes a reduction result and returns
            # whether it is the right missing marker for the current dtype: identity with not-a-time
            # for datetime, identity with the nullable missing marker for nullable integer or boolean, and
            # a generic missing check otherwise.
            if dtype == "datetime64[ns]":
                return res is NaT
            elif dtype in ["Int64", "boolean"]:
                return res is pd.NA
            else:
                return isna(res)

        obj = klass([None], dtype=dtype)
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))

        obj = klass([], dtype=dtype)
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))

        if dtype == "object":
            # generic test with object only works for empty / all NaN
            return

        obj = klass([None, val], dtype=dtype)
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))

        obj = klass([None, val, None], dtype=dtype)
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))

    @pytest.mark.parametrize("opname", ["max", "min"])
    def test_nanargminmax(self, opname, index_or_series):
        # [FUNCTION DOC]
        # Regression test (GH#7261), parametrized by max or min and index-or-series factory. For a
        # datetime container with a missing value the position reduction returns the position of the
        # valid value, and with skipping disabled raises a value error saying an NA value was
        # encountered; same for a non-monotonic container with two missing values. The operation is
        # named differently for indexes (argument) and series (index label).
        # GH#7261
        klass = index_or_series
        arg_op = "arg" + opname if klass is Index else "idx" + opname

        obj = klass([NaT, datetime(2011, 11, 1)])
        assert getattr(obj, arg_op)() == 1

        with pytest.raises(ValueError, match="Encountered an NA value"):
            getattr(obj, arg_op)(skipna=False)

        obj = klass([NaT, datetime(2011, 11, 1), NaT])
        # check DatetimeIndex non-monotonic path
        assert getattr(obj, arg_op)() == 1
        with pytest.raises(ValueError, match="Encountered an NA value"):
            getattr(obj, arg_op)(skipna=False)

    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize("dtype", ["M8[ns]", "datetime64[ns, UTC]"])
    def test_nanops_empty_object(self, opname, index_or_series, dtype):
        # [FUNCTION DOC]
        # Parametrized by max or min, index-or-series factory and two datetime dtypes (naive,
        # UTC). Empty containers give the missing time marker for the value reduction, with or
        # without skipping, and the position/label reductions raise a value error mentioning an empty
        # sequence.
        klass = index_or_series
        arg_op = "arg" + opname if klass is Index else "idx" + opname

        obj = klass([], dtype=dtype)

        assert getattr(obj, opname)() is NaT
        assert getattr(obj, opname)(skipna=False) is NaT

        with pytest.raises(ValueError, match="empty sequence"):
            getattr(obj, arg_op)()
        with pytest.raises(ValueError, match="empty sequence"):
            getattr(obj, arg_op)(skipna=False)

    def test_argminmax(self):
        # [FUNCTION DOC]
        # Tests index argument-of-min and argument-of-max: plain integer index; float index with
        # missing values (skipping gives the valid positions, not skipping raises a value error about
        # an NA value); a single missing value (raises about all NA values, and about an NA value
        # when not skipping); and datetime indexes with missing values behaving the same.
        obj = Index(np.arange(5, dtype="int64"))
        assert obj.argmin() == 0
        assert obj.argmax() == 4

        obj = Index([np.nan, 1, np.nan, 2])
        assert obj.argmin() == 1
        assert obj.argmax() == 3
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

        obj = Index([np.nan])
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmin()
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmax()
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

        obj = Index([NaT, datetime(2011, 11, 1), datetime(2011, 11, 2), NaT])
        assert obj.argmin() == 1
        assert obj.argmax() == 2
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

        obj = Index([NaT])
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmin()
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmax()
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

    @pytest.mark.parametrize("op, expected_col", [["max", "a"], ["min", "b"]])
    def test_same_tz_min_max_axis_1(self, op, expected_col):
        # [FUNCTION DOC]
        # Regression test (GH 10390), parametrized by operation and expected column: for a
        # data frame with a UTC datetime column and a second column one hour earlier, row-wise
        # maximum equals the first column and minimum the second (result name dropped).
        # GH 10390
        df = DataFrame(
            date_range("2016-01-01 00:00:00", periods=3, tz="UTC", unit="ns"),
            columns=["a"],
        )
        df["b"] = df.a.subtract(Timedelta(seconds=3600))
        result = getattr(df, op)(axis=1)
        expected = df[expected_col].rename(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", ["maximum", "minimum"])
    def test_numpy_reduction_with_tz_aware_dtype(self, tz_aware_fixture, func):
        # [FUNCTION DOC]
        # Regression test (GH 15552), parametrized by numpy elementwise maximum or minimum and by
        # timezone-aware fixture. Applying the numpy function to a timezone-aware datetime series with
        # itself returns an equal series.
        # GH 15552
        tz = tz_aware_fixture
        arg = pd.to_datetime(["2019"]).tz_localize(tz)
        expected = Series(arg)
        result = getattr(np, func)(expected, expected)
        tm.assert_series_equal(result, expected)

    def test_nan_int_timedelta_sum(self):
        # [FUNCTION DOC]
        # Regression test (GH 27185): summing a data frame with a timedelta column and a nullable
        # integer column containing missing values gives a series with the timedelta of three
        # nanoseconds and the integer three.
        # GH 27185
        df = DataFrame(
            {
                "A": Series([1, 2, NaT], dtype="timedelta64[ns]"),
                "B": Series([1, 2, pd.NA], dtype="Int64"),
            }
        )
        expected = Series({"A": Timedelta(3), "B": 3})
        result = df.sum()
        tm.assert_series_equal(result, expected)


class TestIndexReductions:
    # [CLASS DOC]
    # Reduction tests moved from an index-specific test file (not intended to be index specific
    # long term): range, timedelta, datetime, period and categorical indexes, and numpy function
    # compatibility. Stateless.
    # Note: the name TestIndexReductions indicates these tests
    #  were moved from an Index-specific test file, _not_ that these tests are
    #  intended long-term to be Index-specific

    @pytest.mark.parametrize(
        "start,stop,step",
        [
            (0, 400, 3),
            (500, 0, -6),
            (-(10**6), 10**6, 4),
            (10**6, -(10**6), -4),
            (0, 10, 20),
        ],
    )
    def test_max_min_range(self, start, stop, step):
        # [FUNCTION DOC]
        # Regression test (GH#17607), parametrized by start, stop and step (positive, negative,
        # huge, and step larger than the range). Compares range index maximum and minimum to the
        # underlying array result, checks skipping is irrelevant, and checks an empty range (reversed
        # step) gives missing results.
        # GH#17607
        idx = RangeIndex(start, stop, step)
        expected = idx._values.max()
        result = idx.max()
        assert result == expected

        # skipna should be irrelevant since RangeIndex should never have NAs
        result2 = idx.max(skipna=False)
        assert result2 == expected

        expected = idx._values.min()
        result = idx.min()
        assert result == expected

        # skipna should be irrelevant since RangeIndex should never have NAs
        result2 = idx.min(skipna=False)
        assert result2 == expected

        # empty
        idx = RangeIndex(start, stop, -step)
        assert isna(idx.max())
        assert isna(idx.min())

    def test_minmax_timedelta64(self):
        # [FUNCTION DOC]
        # Verifies timedelta index min, max, argument of min and argument of max for a monotonic index
        # and for a non-monotonic one containing missing values.
        # monotonic
        idx1 = TimedeltaIndex(["1 days", "2 days", "3 days"])
        assert idx1.is_monotonic_increasing

        # non-monotonic
        idx2 = TimedeltaIndex(["1 days", np.nan, "3 days", "NaT"])
        assert not idx2.is_monotonic_increasing

        for idx in [idx1, idx2]:
            assert idx.min() == Timedelta("1 days")
            assert idx.max() == Timedelta("3 days")
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_timedelta_empty_or_na(self, op):
        # [FUNCTION DOC]
        # Parametrized by min or max. Empty, single-missing and all-missing timedelta indexes
        # return the missing time marker.
        # Return NaT
        obj = TimedeltaIndex([])
        assert getattr(obj, op)() is NaT

        obj = TimedeltaIndex([NaT])
        assert getattr(obj, op)() is NaT

        obj = TimedeltaIndex([NaT, NaT, NaT])
        assert getattr(obj, op)() is NaT

    def test_numpy_minmax_timedelta64(self):
        # [FUNCTION DOC]
        # Verifies numpy minimum, maximum, argument-of-min and argument-of-max on a timedelta range
        # index give the expected first/last values and positions, and that supplying the unsupported
        # output parameter raises a value error saying it is not supported.
        td = timedelta_range("16815 days", "16820 days", freq="D")

        assert np.min(td) == Timedelta("16815 days")
        assert np.max(td) == Timedelta("16820 days")

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(td, out=0)

        assert np.argmin(td) == 0
        assert np.argmax(td) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(td, out=0)

    def test_timedelta_ops(self):
        # [FUNCTION DOC]
        # Regression tests (GH#4984, GH#6462, GH#10040) that timedelta reductions return timedelta
        # results consistently for series and single-column frames: mean, quantile, median, sum,
        # standard deviation, and median with missing values from date differences.
        # GH#4984
        # make sure ops return Timedelta
        s = Series(
            [Timestamp("20130101") + timedelta(seconds=i * i) for i in range(10)]
        )
        td = s.diff()

        result = td.mean()
        expected = to_timedelta(timedelta(seconds=9))
        assert result == expected

        result = td.to_frame().mean()
        assert result[0] == expected

        result = td.quantile(0.1)
        expected = Timedelta(np.timedelta64(2600, "ms"))
        assert result == expected

        result = td.median()
        expected = to_timedelta("00:00:09")
        assert result == expected

        result = td.to_frame().median()
        assert result[0] == expected

        # GH#6462
        # consistency in returned values for sum
        result = td.sum()
        expected = to_timedelta("00:01:21")
        assert result == expected

        result = td.to_frame().sum()
        assert result[0] == expected

        # std
        result = td.std()
        expected = to_timedelta(Series(td.dropna().values).std())
        assert result == expected

        result = td.to_frame().std()
        assert result[0] == expected

        # GH#10040
        # make sure NaT is properly handled by median()
        s = Series([Timestamp("2015-02-03"), Timestamp("2015-02-07")])
        assert s.diff().median() == timedelta(days=4)

        s = Series(
            [Timestamp("2015-02-03"), Timestamp("2015-02-07"), Timestamp("2015-02-15")]
        )
        assert s.diff().median() == timedelta(days=6)

    @pytest.mark.parametrize("opname", ["skew", "kurt", "sem", "prod", "var"])
    def test_invalid_td64_reductions(self, opname):
        # [FUNCTION DOC]
        # Parametrized by skew, kurtosis, standard error, product and variance. Verifies that these
        # reductions on a timedelta series (and its frame with numeric-only disabled) raise a type
        # error matching one of three messages used by different code paths.
        s = Series(
            [Timestamp("20130101") + timedelta(seconds=i * i) for i in range(10)]
        )
        td = s.diff()

        msg = "|".join(
            [
                f"reduction operation '{opname}' not allowed for this dtype",
                rf"cannot perform {opname} with type timedelta64\[ns\]",
                f"does not support operation '{opname}'",
            ]
        )

        with pytest.raises(TypeError, match=msg):
            getattr(td, opname)()

        with pytest.raises(TypeError, match=msg):
            getattr(td.to_frame(), opname)(numeric_only=False)

    def test_minmax_tz(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Parametrized by naive/timezone fixture. For monotonic and non-monotonic datetime indexes
        # (with missing values) checks min, max, and positions of min and max.
        tz = tz_naive_fixture
        # monotonic
        idx1 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], tz=tz)
        assert idx1.is_monotonic_increasing

        # non-monotonic
        idx2 = DatetimeIndex(
            ["2011-01-01", NaT, "2011-01-03", "2011-01-02", NaT], tz=tz
        )
        assert not idx2.is_monotonic_increasing

        for idx in [idx1, idx2]:
            assert idx.min() == Timestamp("2011-01-01", tz=tz)
            assert idx.max() == Timestamp("2011-01-03", tz=tz)
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_nat_datetime64(self, op):
        # [FUNCTION DOC]
        # Parametrized by min or max. Empty, single-missing and all-missing datetime indexes
        # give missing results.
        # Return NaT
        obj = DatetimeIndex([])
        assert isna(getattr(obj, op)())

        obj = DatetimeIndex([NaT])
        assert isna(getattr(obj, op)())

        obj = DatetimeIndex([NaT, NaT, NaT])
        assert isna(getattr(obj, op)())

    def test_numpy_minmax_integer(self):
        # [FUNCTION DOC]
        # Regression test (GH#26125): numpy maximum, minimum, argument-of-max and argument-of-min
        # on an integer index equal the underlying-array results, and the unsupported output
        # parameter raises a value error.
        # GH#26125
        idx = Index([1, 2, 3])

        expected = idx.values.max()
        result = np.max(idx)
        assert result == expected

        expected = idx.values.min()
        result = np.min(idx)
        assert result == expected

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)

        expected = idx.values.argmax()
        result = np.argmax(idx)
        assert result == expected

        expected = idx.values.argmin()
        result = np.argmin(idx)
        assert result == expected

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(idx, out=0)

    def test_numpy_minmax_range(self):
        # [FUNCTION DOC]
        # Regression test (GH#26125): numpy maximum and minimum on a range index; the unsupported
        # output parameter raises a value error. Position functions are not retested because they
        # share the integer index implementation.
        # GH#26125
        idx = RangeIndex(0, 10, 3)

        result = np.max(idx)
        assert result == 9

        result = np.min(idx)
        assert result == 0

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)

        # No need to test again argmax/argmin compat since the implementation
        # is the same as basic integer index

    def test_numpy_minmax_datetime64(self):
        # [FUNCTION DOC]
        # Verifies numpy minimum, maximum and position functions on a daily datetime index, and the
        # unsupported output parameter raising a value error.
        dr = date_range(start="2016-01-15", end="2016-01-20")

        assert np.min(dr) == Timestamp("2016-01-15 00:00:00")
        assert np.max(dr) == Timestamp("2016-01-20 00:00:00")

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(dr, out=0)

        with pytest.raises(ValueError, match=errmsg):
            np.max(dr, out=0)

        assert np.argmin(dr) == 0
        assert np.argmax(dr) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(dr, out=0)

        with pytest.raises(ValueError, match=errmsg):
            np.argmax(dr, out=0)

    def test_minmax_period(self):
        # [FUNCTION DOC]
        # Verifies period index min, max, argument-of-min and argument-of-max for a daily index
        # with a leading missing value and for a non-monotonic one with missing values.
        # monotonic
        idx1 = PeriodIndex([NaT, "2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        assert not idx1.is_monotonic_increasing
        assert idx1[1:].is_monotonic_increasing

        # non-monotonic
        idx2 = PeriodIndex(
            ["2011-01-01", NaT, "2011-01-03", "2011-01-02", NaT], freq="D"
        )
        assert not idx2.is_monotonic_increasing

        for idx in [idx1, idx2]:
            assert idx.min() == Period("2011-01-01", freq="D")
            assert idx.max() == Period("2011-01-03", freq="D")
        assert idx1.argmin() == 1
        assert idx2.argmin() == 0
        assert idx1.argmax() == 3
        assert idx2.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    @pytest.mark.parametrize("data", [[], [NaT], [NaT, NaT, NaT]])
    def test_minmax_period_empty_nat(self, op, data):
        # [FUNCTION DOC]
        # Parametrized by min or max and by empty, single-missing and all-missing data. A monthly
        # period index returns the missing time marker.
        # Return NaT
        obj = PeriodIndex(data, freq="M")
        result = getattr(obj, op)()
        assert result is NaT

    def test_numpy_minmax_period(self):
        # [FUNCTION DOC]
        # Verifies numpy minimum, maximum and position functions on a daily period range, and the
        # unsupported output parameter raising a value error.
        pr = period_range(start="2016-01-15", end="2016-01-20")

        assert np.min(pr) == Period("2016-01-15", freq="D")
        assert np.max(pr) == Period("2016-01-20", freq="D")

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(pr, out=0)

        assert np.argmin(pr) == 0
        assert np.argmax(pr) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(pr, out=0)

    def test_min_max_categorical(self):
        # [FUNCTION DOC]
        # Verifies that min and max of an unordered categorical index raise a type error saying the
        # categorical is not ordered for the operation (with a hint to make it ordered), while an
        # ordered categorical gives the first and last categories used by the data.
        ci = pd.CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)
        msg = (
            r"Categorical is not ordered for operation min\n"
            r"you can use .as_ordered\(\) to change the Categorical to an ordered one\n"
        )
        with pytest.raises(TypeError, match=msg):
            ci.min()
        msg = (
            r"Categorical is not ordered for operation max\n"
            r"you can use .as_ordered\(\) to change the Categorical to an ordered one\n"
        )
        with pytest.raises(TypeError, match=msg):
            ci.max()

        ci = pd.CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=True)
        assert ci.min() == "c"
        assert ci.max() == "b"


class TestSeriesReductions:
    # [CLASS DOC]
    # Reduction tests moved from a series-specific test file (not intended to be series
    # specific long term): sum/product edge cases, overflow, empty and all-missing behavior,
    # mean/variance on masked types, numpy functions, index-of-min and index-of-max, and
    # any/all in many dtypes. Stateless.
    # Note: the name TestSeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def test_sum_inf(self):
        # [FUNCTION DOC]
        # Verifies that a series with infinite values sums to an infinite value, and that the
        # internal nan-aware sum on a 100 by 100 single-precision array containing a column of
        # infinities gives infinite row sums. A copy of the series with the same entries set to
        # missing is created.
        # KNOWN ISSUE: the copy with missing values is created and modified but never used afterwards.
        s = Series(np.random.default_rng(2).standard_normal(10))
        s2 = s.copy()

        s[5:8] = np.inf
        s2[5:8] = np.nan

        assert np.isinf(s.sum())

        arr = np.random.default_rng(2).standard_normal((100, 100)).astype("f4")
        arr[:, 2] = np.inf

        res = nanops.nansum(arr, axis=1)
        assert np.isinf(res).all()

    @pytest.mark.parametrize(
        "dtype", ["float64", "Float32", "Int64", "boolean", "object"]
    )
    @pytest.mark.parametrize("use_bottleneck", [True, False])
    @pytest.mark.parametrize("method, unit", [("sum", 0.0), ("prod", 1.0)])
    def test_empty(self, method, unit, use_bottleneck, dtype):
        # [FUNCTION DOC]
        # Regression test (GH#9422, GH#18921, GH#844), parametrized by sum or product with its unit
        # (zero or one), bottleneck use, and dtype (float, nullable float, nullable integer, nullable
        # boolean, object). With the bottleneck option set accordingly, verifies the result on an empty
        # series, an all-missing series and a series of one missing and one valid value, over
        # the min_count and skipna argument combinations: the unit is returned when the minimum
        # count is zero, a missing value when the minimum count is one and nothing is valid, and the
        # valid value when present. Also checks a data frame with no columns reduces row-wise to the
        # unit, and that a min_count above the number of valid values gives a missing result.
        # KNOWN ISSUE: in the empty-series and all-missing sections the comparison for the default skipna reduction is written without an assertion, so those two checks never verify anything.
        item = pd.NA if dtype in ["Float32", "Int64"] else np.nan
        with pd.option_context("use_bottleneck", use_bottleneck):
            # GH#9422 / GH#18921
            # Entirely empty
            s = Series([], dtype=dtype)
            # NA by default
            result = getattr(s, method)()
            assert result == unit

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == unit

            result = getattr(s, method)(min_count=1)
            assert isna(result)

            # Skipna, default
            result = getattr(s, method)(skipna=True)
            result == unit

            # Skipna, explicit
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == unit

            result = getattr(s, method)(skipna=True, min_count=1)
            assert isna(result)

            result = getattr(s, method)(skipna=False, min_count=0)
            assert result == unit

            result = getattr(s, method)(skipna=False, min_count=1)
            assert isna(result)

            # All-NA
            s = Series([item], dtype=dtype)
            # NA by default
            result = getattr(s, method)()
            assert result == unit

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == unit

            result = getattr(s, method)(min_count=1)
            assert isna(result)

            # Skipna, default
            result = getattr(s, method)(skipna=True)
            result == unit

            # skipna, explicit
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == unit

            result = getattr(s, method)(skipna=True, min_count=1)
            assert isna(result)

            # Mix of valid, empty
            s = Series([item, 1], dtype=dtype)
            # Default
            result = getattr(s, method)()
            assert result == 1.0

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == 1.0

            result = getattr(s, method)(min_count=1)
            assert result == 1.0

            # Skipna
            result = getattr(s, method)(skipna=True)
            assert result == 1.0

            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == 1.0

            # GH#844 (changed in GH#9422)
            df = DataFrame(np.empty((10, 0)), dtype=dtype)
            assert (getattr(df, method)(axis=1) == unit).all()

            s = Series([1], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)

            result = getattr(s, method)(skipna=False, min_count=2)
            assert isna(result)

            s = Series([item], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)

            s = Series([item, 1], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)

    @pytest.mark.parametrize("method", ["mean", "var"])
    @pytest.mark.parametrize("dtype", ["Float64", "Int64", "boolean"])
    def test_ops_consistency_on_empty_nullable(self, method, dtype):
        # [FUNCTION DOC]
        # Regression test (GH#34814), parametrized by mean or variance and nullable dtypes (float,
        # integer, boolean): both an empty and an all-missing series give the nullable missing marker
        # by identity.
        # GH#34814
        # consistency for nullable dtypes on empty or ALL-NA mean

        # empty series
        eser = Series([], dtype=dtype)
        result = getattr(eser, method)()
        assert result is pd.NA

        # ALL-NA series
        nser = Series([pd.NA], dtype=dtype)
        result = getattr(nser, method)()
        assert result is pd.NA

    @pytest.mark.parametrize("method", ["mean", "median", "std", "var"])
    def test_ops_consistency_on_empty(self, method):
        # [FUNCTION DOC]
        # Regression test (GH#7869), parametrized by mean, median, standard deviation and variance.
        # An empty float series gives a missing result. An empty timedelta series gives the
        # missing time marker, except variance which raises a type error matching one of three
        # messages.
        # GH#7869
        # consistency on empty

        # float
        result = getattr(Series(dtype=float), method)()
        assert isna(result)

        # timedelta64[ns]
        tdser = Series([], dtype="m8[ns]")
        if method == "var":
            msg = "|".join(
                [
                    "operation 'var' not allowed",
                    r"cannot perform var with type timedelta64\[ns\]",
                    "does not support operation 'var'",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                getattr(tdser, method)()
        else:
            result = getattr(tdser, method)()
            assert result is NaT

    def test_nansum_buglet(self):
        # [FUNCTION DOC]
        # Verifies numpy's nan-aware sum of a series with one valid and one missing value gives one.
        ser = Series([1.0, np.nan], index=[0, 1])
        result = np.nansum(ser)
        tm.assert_almost_equal(result, 1)

    @pytest.mark.parametrize("use_bottleneck", [True, False])
    @pytest.mark.parametrize("dtype", ["int32", "int64"])
    def test_sum_overflow_int(self, use_bottleneck, dtype):
        # [FUNCTION DOC]
        # Regression test (GH#6915), parametrized by bottleneck use and 32/64-bit integer dtype.
        # For a series of five million consecutive integers (no skipping), sum equals the 64-bit sum
        # of the values (no overflow), min is zero and max is the last value.
        with pd.option_context("use_bottleneck", use_bottleneck):
            # GH#6915
            # overflowing on the smaller int dtypes
            v = np.arange(5000000, dtype=dtype)
            s = Series(v)

            result = s.sum(skipna=False)
            assert int(result) == v.sum(dtype="int64")
            result = s.min(skipna=False)
            assert int(result) == 0
            result = s.max(skipna=False)
            assert int(result) == v[-1]

    @pytest.mark.parametrize("use_bottleneck", [True, False])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_sum_overflow_float(self, use_bottleneck, dtype):
        # [FUNCTION DOC]
        # Parametrized by bottleneck use and 32/64-bit float dtype. For a series of five million
        # consecutive floats: sum equals the sum computed in the same dtype, min is close to zero and
        # max close to the last value.
        with pd.option_context("use_bottleneck", use_bottleneck):
            v = np.arange(5000000, dtype=dtype)
            s = Series(v)

            result = s.sum(skipna=False)
            assert result == v.sum(dtype=dtype)
            result = s.min(skipna=False)
            assert np.allclose(float(result), 0.0)
            result = s.max(skipna=False)
            assert np.allclose(float(result), v[-1])

    def test_mean_masked_overflow(self):
        # [FUNCTION DOC]
        # Regression test (GH#48378): the mean of 100 copies of 10 to the 17 in a nullable integer
        # series equals the numpy mean (difference zero) and equals 1e17, i.e. no overflow.
        # GH#48378
        val = 100_000_000_000_000_000
        n_elements = 100
        na = np.array([val] * n_elements)
        ser = Series([val] * n_elements, dtype="Int64")

        result_numpy = np.mean(na)
        result_masked = ser.mean()
        assert result_masked - result_numpy == 0
        assert result_masked == 1e17

    @pytest.mark.parametrize("ddof, exp", [(1, 2.5), (0, 2.0)])
    def test_var_masked_array(self, ddof, exp):
        # [FUNCTION DOC]
        # Regression test (GH#48379), parametrized by delta degrees of freedom and expected value:
        # variance of a nullable integer series equals that of the plain integer series and the
        # expected number.
        # GH#48379
        ser = Series([1, 2, 3, 4, 5], dtype="Int64")
        ser_numpy_dtype = Series([1, 2, 3, 4, 5], dtype="int64")
        result = ser.var(ddof=ddof)
        result_numpy_dtype = ser_numpy_dtype.var(ddof=ddof)
        assert result == result_numpy_dtype
        assert result == exp

    @pytest.mark.parametrize(
        "values,expected", [([-1j, 0j, 1j], 1.0), ([1 + 2j, 2 + 3j, 3 + 4j], 2.0)]
    )
    def test_var_complex_array(self, values, expected):
        # [FUNCTION DOC]
        # Regression test (GH 61645, GH 62421), parametrized by complex data and expected variance:
        # variance, standard deviation and standard error of the mean of a complex series with one
        # delta degree of freedom are almost equal to the expected values (variance is real).
        # GH 61645, 62421
        ser = Series(values, dtype=complex)
        tm.assert_almost_equal(ser.var(ddof=1), expected)
        tm.assert_almost_equal(ser.std(ddof=1), np.sqrt(expected))
        tm.assert_almost_equal(ser.sem(ddof=1), np.sqrt(expected / len(values)))

    @pytest.mark.parametrize("dtype", ("m8[ns]", "M8[ns]", "M8[ns, UTC]"))
    def test_empty_timeseries_reductions_return_nat(self, dtype, skipna):
        # [FUNCTION DOC]
        # Parametrized by timedelta, datetime and timezone-aware datetime dtypes and the skipna
        # fixture. Empty series min and max return the missing time marker by identity (covers
        # GH#11245).
        # covers GH#11245
        assert Series([], dtype=dtype).min(skipna=skipna) is NaT
        assert Series([], dtype=dtype).max(skipna=skipna) is NaT

    def test_numpy_argmin(self):
        # [FUNCTION DOC]
        # Regression test (GH#16830): numpy argument-of-min on a series equals that of the data, the
        # method gives the same result, and the unsupported output parameter raises a value error.
        # See GH#16830
        data = np.arange(1, 11)

        s = Series(data, index=data)
        result = np.argmin(s)

        expected = np.argmin(data)
        assert result == expected

        result = s.argmin()

        assert result == expected

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmin(s, out=data)

    def test_numpy_argmax(self):
        # [FUNCTION DOC]
        # Regression test (GH#16830): same as the argument-of-min test but for argument-of-max.
        # See GH#16830
        data = np.arange(1, 11)

        ser = Series(data, index=data)
        result = np.argmax(ser)
        expected = np.argmax(data)
        assert result == expected

        result = ser.argmax()

        assert result == expected

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmax(ser, out=data)

    def test_idxmin_dt64index(self, unit):
        # [FUNCTION DOC]
        # Regression test (GH#43587), parametrized by time unit fixture. For a series with a
        # datetime index containing missing values, index-of-min and index-of-max with no skipping
        # raise a value error about an NA value, as do the data frame equivalents.
        # GH#43587 should have NaT instead of NaN
        dti = DatetimeIndex(["NaT", "2015-02-08", "NaT"]).as_unit(unit)
        ser = Series([1.0, 2.0, np.nan], index=dti)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            ser.idxmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            ser.idxmax(skipna=False)

        df = ser.to_frame()
        with pytest.raises(ValueError, match="Encountered an NA value"):
            df.idxmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            df.idxmax(skipna=False)

    def test_idxmin(self):
        # [FUNCTION DOC]
        # Verifies index-of-minimum on a float series with a block of missing values: label lookup
        # equals the minimum; not skipping raises a value error about an NA value; after dropping
        # missing values the position agrees with the array result; an all-missing series raises a
        # value error about all NA values; and a datetime series gives zero, then one after the first
        # element is made missing.
        # test idxmin
        # _check_stat_op approach can not be used here because of isna check.
        string_series = Series(range(20), dtype=np.float64, name="series")

        # add some NaNs
        string_series[5:15] = np.nan

        # skipna or no
        assert string_series[string_series.idxmin()] == string_series.min()
        with pytest.raises(ValueError, match="Encountered an NA value"):
            string_series.idxmin(skipna=False)

        # no NaNs
        nona = string_series.dropna()
        assert nona[nona.idxmin()] == nona.min()
        assert nona.index.values.tolist().index(nona.idxmin()) == nona.values.argmin()

        # all NaNs
        allna = string_series * np.nan
        msg = "Encountered all NA values"
        with pytest.raises(ValueError, match=msg):
            allna.idxmin()

        # datetime64[ns]
        s = Series(date_range("20130102", periods=6))
        result = s.idxmin()
        assert result == 0

        s[0] = np.nan
        result = s.idxmin()
        assert result == 1

    def test_idxmax(self):
        # [FUNCTION DOC]
        # Verifies index-of-maximum with the same structure as the minimum test, also checking a
        # datetime series (position five, then four after missing) and, for a series with a
        # float index (GH#5914), the label 3.1 for maximum and 1.1 for minimum, including a series
        # built from its own index.
        # KNOWN ISSUE: an assertion is placed inside the block that expects a value error, so it is never evaluated; the check only needs the call to raise.
        # test idxmax
        # _check_stat_op approach can not be used here because of isna check.
        string_series = Series(range(20), dtype=np.float64, name="series")

        # add some NaNs
        string_series[5:15] = np.nan

        # skipna or no
        assert string_series[string_series.idxmax()] == string_series.max()
        with pytest.raises(ValueError, match="Encountered an NA value"):
            assert isna(string_series.idxmax(skipna=False))

        # no NaNs
        nona = string_series.dropna()
        assert nona[nona.idxmax()] == nona.max()
        assert nona.index.values.tolist().index(nona.idxmax()) == nona.values.argmax()

        # all NaNs
        allna = string_series * np.nan
        msg = "Encountered all NA values"
        with pytest.raises(ValueError, match=msg):
            allna.idxmax()

        s = Series(date_range("20130102", periods=6))
        result = s.idxmax()
        assert result == 5

        s[5] = np.nan
        result = s.idxmax()
        assert result == 4

        # Index with float64 dtype
        # GH#5914
        s = Series([1, 2, 3], [1.1, 2.1, 3.1])
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1

        s = Series(s.index, s.index)
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1

    def test_all_any(self):
        # [FUNCTION DOC]
        # Verifies all and any on a float series (all is false after zeros, any is true) and on an
        # object series mixing a string and a boolean (any is true).
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        bool_series = ts > 0
        assert not bool_series.all()
        assert bool_series.any()

        # Alternative types, with implicit 'object' dtype.
        s = Series(["abc", True])
        assert s.any()

    def test_numpy_all_any(self, index_or_series):
        # [FUNCTION DOC]
        # Regression test (GH#40180): numpy all and any on an index or series of integers
        # including zero; and numpy all on an index with no zeros.
        # GH#40180
        idx = index_or_series([0, 1, 2])
        assert not np.all(idx)
        assert np.any(idx)
        idx = Index([1, 2, 3])
        assert np.all(idx)

    def test_all_any_skipna(self):
        # [FUNCTION DOC]
        # Verifies skipna handling of all and any in object series with a missing value: with
        # skipping disabled, missing counts as true (so all with a true value and any with a false
        # value are true); with skipping, missing is ignored.
        # Check skipna, with implicit 'object' dtype.
        s1 = Series([np.nan, True])
        s2 = Series([np.nan, False])
        assert s1.all(skipna=False)  # nan && True => True
        assert s1.all(skipna=True)
        assert s2.any(skipna=False)
        assert not s2.any(skipna=True)

    def test_all_any_bool_only(self):
        # [FUNCTION DOC]
        # Regression test (GH#47500): any and all with the boolean-only flag on a boolean series
        # with duplicated index labels give true and false respectively.
        s = Series([False, False, True, True, False, True], index=[0, 0, 1, 1, 2, 2])

        # GH#47500 - test bool_only works
        assert s.any(bool_only=True)
        assert not s.all(bool_only=True)

    def test_any_all_object_dtype(self, all_boolean_reductions, skipna):
        # [FUNCTION DOC]
        # Regression test (GH#12863), parametrized by all/any (fixture) and skipna fixture: a
        # series of non-empty strings with object dtype yields true.
        # GH#12863
        ser = Series(["a", "b", "c", "d", "e"], dtype=object)
        result = getattr(ser, all_boolean_reductions)(skipna=skipna)
        expected = True

        assert result == expected

    @pytest.mark.parametrize(
        "data", [[False, None], [None, False], [False, np.nan], [np.nan, False]]
    )
    def test_any_all_object_dtype_missing(self, data, all_boolean_reductions):
        # [FUNCTION DOC]
        # Regression test (GH#27709), parametrized by data pairs mixing a false value with None or
        # missing float, and by all/any. With skipping disabled None is treated as false and a
        # missing float as true, so the expected result is true only for the any reduction without
        # None present.
        # GH#27709
        ser = Series(data)
        result = getattr(ser, all_boolean_reductions)(skipna=False)

        # None is treated is False, but np.nan is treated as True
        expected = all_boolean_reductions == "any" and None not in data
        assert result == expected

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "UInt64", "Float64"])
    @pytest.mark.parametrize(
        # expected_data indexed as [[skipna=False/any, skipna=False/all],
        #                           [skipna=True/any, skipna=True/all]]
        "data,expected_data",
        [
            ([0, 0, 0], [[False, False], [False, False]]),
            ([1, 1, 1], [[True, True], [True, True]]),
            ([pd.NA, pd.NA, pd.NA], [[pd.NA, pd.NA], [False, True]]),
            ([0, pd.NA, 0], [[pd.NA, False], [False, False]]),
            ([1, pd.NA, 1], [[True, pd.NA], [True, True]]),
            ([1, pd.NA, 0], [[True, False], [True, False]]),
        ],
    )
    def test_any_all_nullable_kleene_logic(
        self, all_boolean_reductions, skipna, data, dtype, expected_data
    ):
        # [FUNCTION DOC]
        # Regression tests (GH-37506, GH-41967) for three-valued (Kleene) logic in nullable dtypes
        # (boolean, integer, unsigned, float), parametrized by data and a nested expected table
        # indexed by skipna then by any/all. Verifies any and all results, comparing the nullable
        # missing marker by identity.
        # GH-37506, GH-41967
        ser = Series(data, dtype=dtype)
        expected = expected_data[skipna][all_boolean_reductions == "all"]

        result = getattr(ser, all_boolean_reductions)(skipna=skipna)
        assert (result is pd.NA and expected is pd.NA) or result == expected

    def test_any_axis1_bool_only(self):
        # [FUNCTION DOC]
        # Regression test (GH#32432): row-wise any with the boolean-only flag on a frame with a
        # boolean and an integer column considers only the boolean column.
        # GH#32432
        df = DataFrame({"A": [True, False], "B": [1, 2]})
        result = df.any(axis=1, bool_only=True)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_any_all_datetimelike(self):
        # [FUNCTION DOC]
        # Regression test (GH#38723, GH#34479): any and all on naive and timezone-aware datetime
        # arrays, series and frames raise a type error saying the operation is unsupported for
        # datetime; on timedelta data they work (any true, all false). The current behavior may not be
        # the desired long-term behavior but must be consistent.
        # GH#38723 these may not be the desired long-term behavior (GH#34479)
        #  but in the interim should be internally consistent
        dta = date_range("1995-01-02", periods=3)._data
        ser = Series(dta)
        df = DataFrame(ser)

        # GH#34479
        msg = "datetime64 type does not support operation '(any|all)'"
        with pytest.raises(TypeError, match=msg):
            dta.all()
        with pytest.raises(TypeError, match=msg):
            dta.any()

        with pytest.raises(TypeError, match=msg):
            ser.all()
        with pytest.raises(TypeError, match=msg):
            ser.any()

        with pytest.raises(TypeError, match=msg):
            df.any().all()
        with pytest.raises(TypeError, match=msg):
            df.all().all()

        dta = dta.tz_localize("UTC")
        ser = Series(dta)
        df = DataFrame(ser)
        # GH#34479
        with pytest.raises(TypeError, match=msg):
            dta.all()
        with pytest.raises(TypeError, match=msg):
            dta.any()

        with pytest.raises(TypeError, match=msg):
            ser.all()
        with pytest.raises(TypeError, match=msg):
            ser.any()

        with pytest.raises(TypeError, match=msg):
            df.any().all()
        with pytest.raises(TypeError, match=msg):
            df.all().all()

        tda = dta - dta[0]
        ser = Series(tda)
        df = DataFrame(ser)

        assert tda.any()
        assert not tda.all()

        assert ser.any()
        assert not ser.all()

        assert df.any().all()
        assert not df.all().any()

    def test_any_all_string_dtype(self, any_string_dtype):
        # [FUNCTION DOC]
        # Regression test (GH#54591) with the string dtype fixture. For the nullable string dtype
        # (missing marker is the NA scalar) any and all still raise a type error. For other string
        # dtypes verifies truthiness semantic for many series (empty strings false, missing values
        # treated as true, empty series any false and all true), with and without skipping.
        # GH#54591
        if (
            isinstance(any_string_dtype, pd.StringDtype)
            and any_string_dtype.na_value is pd.NA
        ):
            # the nullable string dtype currently still raise an error
            # https://github.com/pandas-dev/pandas/issues/51939
            ser = Series(["a", "b"], dtype=any_string_dtype)
            with pytest.raises(TypeError):
                ser.any()
            with pytest.raises(TypeError):
                ser.all()
            return

        ser = Series(["", "a"], dtype=any_string_dtype)
        assert ser.any()
        assert not ser.all()
        assert ser.any(skipna=False)
        assert not ser.all(skipna=False)

        ser = Series([np.nan, "a"], dtype=any_string_dtype)
        assert ser.any()
        assert ser.all()
        assert ser.any(skipna=False)
        assert ser.all(skipna=False)  # NaN is considered truthy

        ser = Series([np.nan, ""], dtype=any_string_dtype)
        assert not ser.any()
        assert not ser.all()
        assert ser.any(skipna=False)  # NaN is considered truthy
        assert not ser.all(skipna=False)

        ser = Series(["a", "b"], dtype=any_string_dtype)
        assert ser.any()
        assert ser.all()
        assert ser.any(skipna=False)
        assert ser.all(skipna=False)

        ser = Series([], dtype=any_string_dtype)
        assert not ser.any()
        assert ser.all()
        assert not ser.any(skipna=False)
        assert ser.all(skipna=False)

        ser = Series([""], dtype=any_string_dtype)
        assert not ser.any()
        assert not ser.all()
        assert not ser.any(skipna=False)
        assert not ser.all(skipna=False)

        ser = Series([np.nan], dtype=any_string_dtype)
        assert not ser.any()
        assert ser.all()
        assert ser.any(skipna=False)  # NaN is considered truthy
        assert ser.all(skipna=False)  # NaN is considered truthy

    def test_timedelta64_analytics(self):
        # [FUNCTION DOC]
        # Tests timedelta reductions on series: index-of-min and index-of-max (including after a
        # missing value, GH#2982), absolute value via numpy and method, and max and min
        # returning timedeltas.
        # index min/max
        dti = date_range("2012-1-1", periods=3, freq="D")
        td = Series(dti) - Timestamp("20120101")

        result = td.idxmin()
        assert result == 0

        result = td.idxmax()
        assert result == 2

        # GH#2982
        # with NaT
        td[0] = np.nan

        result = td.idxmin()
        assert result == 1

        result = td.idxmax()
        assert result == 2

        # abs
        s1 = Series(date_range("20120101", periods=3))
        s2 = Series(date_range("20120102", periods=3))
        expected = Series(s2 - s1)

        result = np.abs(s1 - s2)
        tm.assert_series_equal(result, expected)

        result = (s1 - s2).abs()
        tm.assert_series_equal(result, expected)

        # max/min
        result = td.max()
        expected = Timedelta("2 days")
        assert result == expected

        result = td.min()
        expected = Timedelta("1 days")
        assert result == expected

    def test_assert_idxminmax_empty_raises(self):
        # [FUNCTION DOC]
        # Verifies that index-of-minimum and index-of-maximum on an empty float series raise a
        # value error saying an attempt was made to get the argmin or argmax of an empty sequence,
        # with and without skipping.
        """
        Cases where ``Series.argmax`` and related should raise an exception
        """
        test_input = Series([], dtype="float64")
        msg = "attempt to get argmin of an empty sequence"
        with pytest.raises(ValueError, match=msg):
            test_input.idxmin()
        with pytest.raises(ValueError, match=msg):
            test_input.idxmin(skipna=False)
        msg = "attempt to get argmax of an empty sequence"
        with pytest.raises(ValueError, match=msg):
            test_input.idxmax()
        with pytest.raises(ValueError, match=msg):
            test_input.idxmax(skipna=False)

    def test_idxminmax_object_dtype(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies index-of-min/max on object series: strings and one-tuples are comparable and give
        # the expected labels with or without skipping. Unless the inferred-string option is on, a
        # series with strings plus missing markers raises a type error about comparing a float with a
        # string. Previously object dtype was disallowed (before version 2.1).
        # pre-2.1 object-dtype was disallowed for argmin/max
        ser = Series(["foo", "bar", "baz"])
        assert ser.idxmax() == 0
        assert ser.idxmax(skipna=False) == 0
        assert ser.idxmin() == 1
        assert ser.idxmin(skipna=False) == 1

        ser2 = Series([(1,), (2,)])
        assert ser2.idxmax() == 1
        assert ser2.idxmax(skipna=False) == 1
        assert ser2.idxmin() == 0
        assert ser2.idxmin(skipna=False) == 0

        if not using_infer_string:
            # attempting to compare np.nan with string raises
            ser3 = Series(["foo", "foo", "bar", "bar", None, np.nan, "baz"])
            msg = "'>' not supported between instances of 'float' and 'str'"
            with pytest.raises(TypeError, match=msg):
                ser3.idxmax()
            with pytest.raises(TypeError, match=msg):
                ser3.idxmax(skipna=False)
            msg = "'<' not supported between instances of 'float' and 'str'"
            with pytest.raises(TypeError, match=msg):
                ser3.idxmin()
            with pytest.raises(TypeError, match=msg):
                ser3.idxmin(skipna=False)

    def test_idxminmax_object_frame(self):
        # [FUNCTION DOC]
        # Regression test (GH#4279): index-of-maximum of an object-and-float frame gives the label
        # positions zero and two.
        # GH#4279
        df = DataFrame([["zimm", 2.5], ["biff", 1.0], ["bid", 12.0]])
        res = df.idxmax()
        exp = Series([0, 2])
        tm.assert_series_equal(res, exp)

    def test_idxminmax_object_tuples(self):
        # [FUNCTION DOC]
        # Regression test (GH#43697): index-of-max and min on a series of tuples give two and
        # zero, with and without skipping.
        # GH#43697
        ser = Series([(1, 3), (2, 2), (3, 1)])
        assert ser.idxmax() == 2
        assert ser.idxmin() == 0
        assert ser.idxmax(skipna=False) == 2
        assert ser.idxmin(skipna=False) == 0

    def test_idxminmax_object_decimals(self):
        # [FUNCTION DOC]
        # Regression test (GH#40685): index-of-max and min on a frame with decimal columns give the
        # expected labels per column.
        # GH#40685
        df = DataFrame(
            {
                "idx": [0, 1],
                "x": [Decimal("8.68"), Decimal("42.23")],
                "y": [Decimal("7.11"), Decimal("79.61")],
            }
        )
        res = df.idxmax()
        exp = Series({"idx": 1, "x": 1, "y": 1})
        tm.assert_series_equal(res, exp)

        res2 = df.idxmin()
        exp2 = exp - 1
        tm.assert_series_equal(res2, exp2)

    def test_argminmax_object_ints(self):
        # [FUNCTION DOC]
        # Regression test (GH#18021): argument-of-max and min on an object series of integers give
        # one and zero with or without skipping.
        # GH#18021
        ser = Series([0, 1], dtype="object")
        assert ser.argmax() == 1
        assert ser.argmin() == 0
        assert ser.argmax(skipna=False) == 1
        assert ser.argmin(skipna=False) == 0

    def test_idxminmax_with_inf(self):
        # [FUNCTION DOC]
        # Regression test (GH #13595): index-of-min and max on numeric data with negative infinity,
        # positive infinity and a missing value give the infinity labels; not skipping raises a value
        # error about an NA value.
        # For numeric data with NA and Inf (GH #13595)
        s = Series([0, -np.inf, np.inf, np.nan])

        assert s.idxmin() == 1
        with pytest.raises(ValueError, match="Encountered an NA value"):
            s.idxmin(skipna=False)

        assert s.idxmax() == 2
        with pytest.raises(ValueError, match="Encountered an NA value"):
            s.idxmax(skipna=False)

    def test_sum_uint64(self):
        # [FUNCTION DOC]
        # Regression test (GH 53401): sum of an unsigned 64-bit series with a value above the signed
        # range equals that value as an unsigned 64-bit integer.
        # GH 53401
        s = Series([10000000000000000000], dtype="uint64")
        result = s.sum()
        expected = np.uint64(10000000000000000000)
        tm.assert_almost_equal(result, expected)

    def test_signedness_preserved_after_sum(self, using_python_scalars):
        # [FUNCTION DOC]
        # Regression test (GH 37491): the sum of an unsigned 8-bit series is an unsigned 64-bit numpy
        # integer, or a Python integer when the python-scalars fixture flag is on.
        # GH 37491
        ser = Series([1, 2, 3, 4])

        result = ser.astype("uint8").sum()
        if using_python_scalars:
            assert isinstance(result, int)
        else:
            assert isinstance(result, np.uint64)


class TestDatetime64SeriesReductions:
    # [CLASS DOC]
    # Reduction tests for datetime64 series, moved from a series-specific test file.
    # Stateless.
    # Note: the name TestDatetime64SeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize(
        "nat_ser",
        [
            Series([NaT, NaT]),
            Series([NaT, Timedelta("nat")]),
            Series([Timedelta("nat"), Timedelta("nat")]),
        ],
    )
    def test_minmax_nat_series(self, nat_ser):
        # [FUNCTION DOC]
        # Regression test (GH#23282), parametrized by series of not-a-time and not-a-timedelta
        # combinations. Min and max return the missing time marker, with and without skipping.
        # GH#23282
        assert nat_ser.min() is NaT
        assert nat_ser.max() is NaT
        assert nat_ser.min(skipna=False) is NaT
        assert nat_ser.max(skipna=False) is NaT

    @pytest.mark.parametrize(
        "nat_df",
        [
            [NaT, NaT],
            [NaT, Timedelta("nat")],
            [Timedelta("nat"), Timedelta("nat")],
        ],
    )
    def test_minmax_nat_dataframe(self, nat_df):
        # [FUNCTION DOC]
        # Regression test (GH#23282): the same as the series test but for a one-column data frame
        # built from lists of not-a-time or not-a-timedelta values.
        # GH#23282
        nat_df = DataFrame(nat_df)
        assert nat_df.min()[0] is NaT
        assert nat_df.max()[0] is NaT
        assert nat_df.min(skipna=False)[0] is NaT
        assert nat_df.max(skipna=False)[0] is NaT

    def test_min_max(self):
        # [FUNCTION DOC]
        # Verifies min and max of a shuffled daily datetime index are timestamps equal to the
        # first and last values, and the same for the sorted index.
        rng = date_range("1/1/2000", "12/31/2000")
        rng2 = rng.take(np.random.default_rng(2).permutation(len(rng)))

        the_min = rng2.min()
        the_max = rng2.max()
        assert isinstance(the_min, Timestamp)
        assert isinstance(the_max, Timestamp)
        assert the_min == rng[0]
        assert the_max == rng[-1]

        assert rng.min() == rng[0]
        assert rng.max() == rng[-1]

    def test_min_max_series(self):
        # [FUNCTION DOC]
        # Verifies max and min of a datetime column of a frame (with other columns) are timestamps
        # equal to the last and first values.
        rng = date_range("1/1/2000", periods=10, freq="4h")
        lvls = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]
        df = DataFrame(
            {
                "TS": rng,
                "V": np.random.default_rng(2).standard_normal(len(rng)),
                "L": lvls,
            }
        )

        result = df.TS.max()
        exp = Timestamp(df.TS.iat[-1])
        assert isinstance(result, Timestamp)
        assert result == exp

        result = df.TS.min()
        exp = Timestamp(df.TS.iat[0])
        assert isinstance(result, Timestamp)
        assert result == exp


class TestCategoricalSeriesReductions:
    # [CLASS DOC]
    # Reduction tests for categorical series (min and max), moved from a series-specific test
    # file. Stateless.
    # Note: the name TestCategoricalSeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_unordered_raises(self, function):
        # [FUNCTION DOC]
        # Parametrized by min or max. Unordered categorical series raise a type error saying the
        # categorical is not ordered for the operation.
        # unordered cats have no min/max
        cat = Series(Categorical(["a", "b", "c", "d"], ordered=False))
        msg = f"Categorical is not ordered for operation {function}"
        with pytest.raises(TypeError, match=msg):
            getattr(cat, function)()

    @pytest.mark.parametrize(
        "values, categories",
        [
            (list("abc"), list("abc")),
            (list("abc"), list("cba")),
            ([*list("abc"), np.nan], list("cba")),
            ([1, 2, 3], [3, 2, 1]),
            ([1, 2, 3, np.nan], [3, 2, 1]),
        ],
    )
    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_ordered(self, values, categories, function):
        # [FUNCTION DOC]
        # Regression test (GH 25303), parametrized by values with categories (letters or numbers,
        # in forward or reversed order, with or without a missing value) and by min/max. For an
        # ordered categorical, min is the first category and max is the last of three categories.
        # GH 25303
        cat = Series(Categorical(values, categories=categories, ordered=True))
        result = getattr(cat, function)(skipna=True)
        expected = categories[0] if function == "min" else categories[2]
        assert result == expected

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_ordered_with_nan_only(self, function, skipna):
        # [FUNCTION DOC]
        # Parametrized by min or max and skipna fixture. An ordered categorical series with only a
        # missing value gives the float missing value by identity regardless of skipping.
        # https://github.com/pandas-dev/pandas/issues/33450
        cat = Series(Categorical([np.nan], categories=[1, 2], ordered=True))
        result = getattr(cat, function)(skipna=skipna)
        assert result is np.nan

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_skipna(self, function, skipna):
        # [FUNCTION DOC]
        # Parametrized by min/max and skipna fixture. For an ordered categorical with a missing
        # value: skipping gives the smallest (b) or largest (a) category; not skipping gives the
        # float missing value by identity.
        cat = Series(
            Categorical(["a", "b", np.nan, "a"], categories=["b", "a"], ordered=True)
        )
        result = getattr(cat, function)(skipna=skipna)

        if skipna is True:
            expected = "b" if function == "min" else "a"
            assert result == expected
        else:
            assert result is np.nan


class TestSeriesMode:
    # [CLASS DOC]
    # Tests for the mode of a series for many dtypes, moved from a series-specific test file.
    # Stateless.
    # Note: the name TestSeriesMode indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def test_mode_empty(self, dropna):
        # [FUNCTION DOC]
        # Parametrized by the dropna fixture: the mode of an empty float series is an equal empty
        # series.
        s = Series([], dtype=np.float64)
        result = s.mode(dropna)
        tm.assert_series_equal(result, s)

    @pytest.mark.parametrize(
        "dropna, data, expected",
        [
            (True, [1, 1, 1, 2], [1]),
            (True, [1, 1, 1, 2, 3, 3, 3], [1, 3]),
            (False, [1, 1, 1, 2], [1]),
            (False, [1, 1, 1, 2, 3, 3, 3], [1, 3]),
        ],
    )
    def test_mode_numerical(self, dropna, data, expected, any_real_numpy_dtype):
        # [FUNCTION DOC]
        # Parametrized by dropna flag, data and expected modes plus the numpy real dtype fixture. The
        # mode (with ties returning all values sorted) of the series has the same dtype and values
        # as the expected series.
        s = Series(data, dtype=any_real_numpy_dtype)
        result = s.mode(dropna)
        expected = Series(expected, dtype=any_real_numpy_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dropna, expected", [(True, [1.0]), (False, [1, np.nan])])
    def test_mode_numerical_nan(self, dropna, expected):
        # [FUNCTION DOC]
        # Parametrized by dropna. Series of floats with two missing values: the mode is one when
        # dropping missing, and one plus missing when not dropping.
        s = Series([1, 1, 2, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(expected)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [
            (True, ["b"], ["bar"]),
            (False, ["b"], [np.nan]),
        ],
    )
    def test_mode_object(self, dropna, expected1, expected2):
        # [FUNCTION DOC]
        # Parametrized by dropna and two expected value lists. Checks the mode of a character-typed
        # string series (b) and of an object series with repeated missing values: bar when dropping,
        # missing marker when not.
        # Test string and object types.
        data = ["a"] * 2 + ["b"] * 3

        s = Series(data, dtype="c")
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="c")
        tm.assert_series_equal(result, expected1)

        data = ["foo", "bar", "bar", np.nan, np.nan, np.nan]

        s = Series(data, dtype=object)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=object)
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [
            (True, ["b"], ["bar"]),
            (False, ["b"], [np.nan]),
        ],
    )
    def test_mode_string(self, dropna, expected1, expected2, any_string_dtype):
        # [FUNCTION DOC]
        # Same data as the object mode test but using the string dtype fixture for the series
        # and the expected results.
        # Test string and object types.
        data = ["a"] * 2 + ["b"] * 3

        s = Series(data, dtype=any_string_dtype)
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype=any_string_dtype)
        tm.assert_series_equal(result, expected1)

        data = ["foo", "bar", "bar", np.nan, np.nan, np.nan]

        s = Series(data, dtype=any_string_dtype)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=any_string_dtype)
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [(True, ["foo"], ["foo"]), (False, ["foo"], [np.nan])],
    )
    def test_mode_mixeddtype(self, dropna, expected1, expected2):
        # [FUNCTION DOC]
        # Parametrized by dropna. Mode of a mixed integer/string object series (foo) and of one
        # with repeated missing values (foo when dropping, missing when not); results are object
        # dtype.
        s = Series([1, "foo", "foo"])
        result = s.mode(dropna)
        expected = Series(expected1, dtype=object)
        tm.assert_series_equal(result, expected)

        s = Series([1, "foo", "foo", np.nan, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(expected2, dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [
            (
                True,
                ["1900-05-03", "2011-01-03", "2013-01-02"],
                ["2011-01-03", "2013-01-02"],
            ),
            (False, [np.nan], [np.nan, "2011-01-03", "2013-01-02"]),
        ],
    )
    def test_mode_datetime(self, dropna, expected1, expected2):
        # [FUNCTION DOC]
        # Parametrized by dropna and expected values. Datetime mode of a series with unique values
        # and missing values: all values when dropping, only the missing value when not; and of a
        # series with repeated dates: the repeated dates, plus the missing marker when not
        # dropping. All results have nanosecond datetime dtype.
        s = Series(
            ["2011-01-03", "2013-01-02", "1900-05-03", "nan", "nan"], dtype="M8[ns]"
        )
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="M8[ns]")
        tm.assert_series_equal(result, expected1)

        s = Series(
            [
                "2011-01-03",
                "2013-01-02",
                "1900-05-03",
                "2011-01-03",
                "2013-01-02",
                "nan",
                "nan",
            ],
            dtype="M8[ns]",
        )
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="M8[ns]")
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [
            (True, ["-1 days", "0 days", "1 days"], ["2 min", "1 day"]),
            (False, [np.nan], [np.nan, "2 min", "1 day"]),
        ],
    )
    def test_mode_timedelta(self, dropna, expected1, expected2):
        # [FUNCTION DOC]
        # GH 5986. Same structure as the datetime mode test but with timedelta dtype: all unique
        # values when dropping missing, only missing when not; and the repeated durations, with
        # missing included when not dropping.
        # gh-5986: Test timedelta types.

        s = Series(
            ["1 days", "-1 days", "0 days", "nan", "nan"], dtype="timedelta64[ns]"
        )
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="timedelta64[ns]")
        tm.assert_series_equal(result, expected1)

        s = Series(
            [
                "1 day",
                "1 day",
                "-1 day",
                "-1 day 2 min",
                "2 min",
                "2 min",
                "nan",
                "nan",
            ],
            dtype="timedelta64[ns]",
        )
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="timedelta64[ns]")
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2, expected3",
        [
            (
                True,
                Categorical([1, 2], categories=[1, 2]),
                Categorical(["a"], categories=[1, "a"]),
                Categorical([3, 1], categories=[3, 2, 1], ordered=True),
            ),
            (
                False,
                Categorical([np.nan], categories=[1, 2]),
                Categorical([np.nan, "a"], categories=[1, "a"]),
                Categorical([np.nan, 3, 1], categories=[3, 2, 1], ordered=True),
            ),
        ],
    )
    def test_mode_category(self, dropna, expected1, expected2, expected3):
        # [FUNCTION DOC]
        # Parametrized by dropna and three expected categoricals. Checks mode of categorical
        # series: one with ties on integer categories and missing values, one with a string mode
        # among mixed categories, and an ordered one with tied modes returned in category order. When
        # not dropping, missing values are in the mode. Results have category dtype.
        s = Series(Categorical([1, 2, np.nan, np.nan]))
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="category")
        tm.assert_series_equal(result, expected1)

        s = Series(Categorical([1, "a", "a", np.nan, np.nan]))
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="category")
        tm.assert_series_equal(result, expected2)

        s = Series(
            Categorical(
                [1, 1, 2, 3, 3, np.nan, np.nan], categories=[3, 2, 1], ordered=True
            )
        )
        result = s.mode(dropna)
        expected3 = Series(expected3, dtype="category")
        tm.assert_series_equal(result, expected3)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [(True, [2**63], [1, 2**63]), (False, [2**63], [1, 2**63])],
    )
    def test_mode_intoverflow(self, dropna, expected1, expected2):
        # [FUNCTION DOC]
        # Parametrized by dropna. Mode of unsigned 64-bit data containing a value above the signed
        # range does not overflow: the repeated large value alone, or both values when tied.
        # Test for uint64 overflow.
        s = Series([1, 2**63, 2**63], dtype=np.uint64)
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype=np.uint64)
        tm.assert_series_equal(result, expected1)

        s = Series([1, 2**63], dtype=np.uint64)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=np.uint64)
        tm.assert_series_equal(result, expected2)

    def test_mode_sort_with_na(self):
        # [FUNCTION DOC]
        # Verifies that without dropping missing values the mode of a mixed object series puts the
        # string first and the missing value after it (sorting with missing values works).
        s = Series([1, "foo", "foo", np.nan, np.nan])
        expected = Series(["foo", np.nan], dtype=object)
        result = s.mode(dropna=False)
        tm.assert_series_equal(result, expected)

    def test_mode_boolean_with_na(self):
        # [FUNCTION DOC]
        # Regression test (GH#42107): the mode of a nullable boolean series with a missing value
        # is a nullable boolean series holding true.
        # GH#42107
        ser = Series([True, False, True, pd.NA], dtype="boolean")
        result = ser.mode()
        expected = Series([True], dtype="boolean")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "array,expected,dtype",
        [
            (
                [0, 1j, 1, 1, 1 + 1j, 1 + 2j],
                [1],
                np.complex128,
            ),
            (
                [0, 1j, 1, 1, 1 + 1j, 1 + 2j],
                [1],
                np.complex64,
            ),
            (
                [1 + 1j, 2j, 1 + 1j],
                [1 + 1j],
                np.complex128,
            ),
        ],
    )
    def test_single_mode_value_complex(self, array, expected, dtype):
        # [FUNCTION DOC]
        # Parametrized by complex data, expected single mode and complex dtype (128 or 64 bits):
        # the mode of a complex series has the same dtype and expected value.
        result = Series(array, dtype=dtype).mode()
        expected = Series(expected, dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "array,expected,dtype",
        [
            (
                # no modes
                [0, 1j, 1, 1 + 1j, 1 + 2j],
                [0j, 1j, 1 + 0j, 1 + 1j, 1 + 2j],
                np.complex128,
            ),
            (
                [1 + 1j, 2j, 1 + 1j, 2j, 3],
                [2j, 1 + 1j],
                np.complex64,
            ),
        ],
    )
    def test_multimode_complex(self, array, expected, dtype):
        # [FUNCTION DOC]
        # Regression test (GH 17927), parametrized by array, expected sorted modes and dtype:
        # for a multi-modal complex series, the modes are returned sorted by magnitude; when there is
        # no repeated value all values are the modes.
        # GH 17927
        # mode tries to sort multimodal series.
        # Complex numbers are sorted by their magnitude
        result = Series(array, dtype=dtype).mode()
        expected = Series(expected, dtype=dtype)
        tm.assert_series_equal(result, expected)
