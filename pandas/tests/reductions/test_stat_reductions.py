"""
Tests for statistical reductions of 2nd moment or higher: var, skew, kurt, ...
"""

# [MODULE DOC]
# Tests for statistical reductions (mean, sum, product, min, max, median, variance, standard
# deviation, standard error of the mean, skew, kurtosis) on datetime-like arrays/indexes/series
# and on plain numeric series. Contains two test classes: one for datetime-like mean behavior
# and one (moved from a series-specific file) for numeric series reductions driven by a shared
# checking helper. Uses shared fixtures for timezone and for constructing an index, series or
# array. Scipy is imported lazily and the relevant tests are skipped if it is missing.

import inspect

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDatetimeLikeStatReductions:
    # [CLASS DOC]
    # Tests of the mean reduction on datetime64, period and timedelta64 containers. Stateless;
    # each method builds its own data.
    def test_dt64_mean(self, tz_naive_fixture, index_or_series_or_array):
        # [FUNCTION DOC]
        # Verifies the mean of a shuffled datetime array wrapped as an index, series or array
        # (via the shared fixture), for naive and timezone-aware data (fixture-parametrized).
        # Without missing values the mean (with or without skipping missing) is the middle date.
        # After one element is set to missing in the underlying array, the default mean skips it
        # and gives the mean of the remaining dates, while disabling skipping returns the missing
        # timestamp marker.
        tz = tz_naive_fixture

        dti = date_range("2001-01-01", periods=11, tz=tz)
        # shuffle so that we are not just working with monotone-increasing
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
        dtarr = dti._data

        obj = index_or_series_or_array(dtarr)
        assert obj.mean() == pd.Timestamp("2001-01-06", tz=tz)
        assert obj.mean(skipna=False) == pd.Timestamp("2001-01-06", tz=tz)

        # dtarr[-2] will be the first date 2001-01-1
        dtarr[-2] = pd.NaT

        obj = index_or_series_or_array(dtarr)
        assert obj.mean() == pd.Timestamp("2001-01-06 07:12:00", tz=tz)
        assert obj.mean(skipna=False) is pd.NaT

    @pytest.mark.parametrize("freq", ["s", "h", "D", "W", "B"])
    def test_period_mean(self, index_or_series_or_array, freq):
        # [FUNCTION DOC]
        # Verifies that taking the mean of period-dtype data raises a type error mentioning
        # ambiguity, with or without missing skipping, both before and after one element is set to
        # missing. Parametrized by period frequency (second, hour, day, week, business day) and the
        # container fixture. The business-day frequency additionally emits a deprecation warning
        # on conversion to periods, which is asserted. (GH#24757)
        # GH#24757
        dti = date_range("2001-01-01", periods=11)
        # shuffle so that we are not just working with monotone-increasing
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])

        warn = FutureWarning if freq == "B" else None
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            parr = dti._data.to_period(freq)
        obj = index_or_series_or_array(parr)
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

        # parr[-2] will be the first date 2001-01-1
        parr[-2] = pd.NaT

        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

    def test_td64_mean(self, index_or_series_or_array):
        # [FUNCTION DOC]
        # Verifies the mean of timedelta data in an index/series/array container (fixture).
        # Without missing values it equals the numpy mean. After setting the first element missing
        # in the shared underlying array, a non-skipping mean returns the missing marker and a
        # skipping mean equals the mean of the remaining elements; the skipping mean is also checked
        # against the original mean rescaled, after rounding to microseconds because exact equality
        # differs by one nanosecond.
        m8values = np.array([0, 3, -2, -7, 1, 2, -1, 3, 5, -2, 4], "m8[D]")
        tdi = pd.TimedeltaIndex(m8values).as_unit("ns")

        tdarr = tdi._data
        obj = index_or_series_or_array(tdarr, copy=False)

        result = obj.mean()
        expected = np.array(tdarr).mean()
        assert result == expected

        tdarr[0] = pd.NaT
        assert obj.mean(skipna=False) is pd.NaT

        result2 = obj.mean(skipna=True)
        assert result2 == tdi[1:].mean()

        # exact equality fails by 1 nanosecond
        assert result2.round("us") == (result * 11.0 / 10).round("us")


class TestSeriesStatReductions:
    # [CLASS DOC]
    # Tests of numeric statistical reductions on Series, moved from a series-specific test file
    # (not intended to be series-specific long term). Each test builds its own series and
    # delegates the common checks to a shared helper method. Stateless.
    # Note: the name TestSeriesStatReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def _check_stat_op(
        self, name, alternate, string_series_, check_objects=False, check_allna=False
    ):
        # [FUNCTION DOC]
        # HELPER: shared checker for the statistical reduction tests of this class.
        # Parameters: name is the Series method name to test; alternate is a reference function
        # operating on a raw numeric array; string_series_ is a float series that is MUTATED in place
        # (elements 5 through 14 set to missing); check_objects (default False) additionally compares
        # the reduction on a business-date series against the reference; check_allna (default False)
        # additionally asserts that an all-missing series gives a missing result.
        # Runs with the optional bottleneck acceleration disabled via an option context.
        # Checks: for names other than max, min, mean, median and std a datetime series raises a type
        # error saying the operation is unsupported; skipping missing values gives a non-missing
        # result while not skipping gives missing; results match the reference on the values with
        # missing dropped; a series with a missing entry in an integer-like list does not fail; a
        # large-integer int64 series (GH#2888) matches the reference as floats; string data raises a
        # type error unless the name is sum, min or max; axis 1 raises a value error for a series;
        # and, if the method accepts a numeric-only argument, calling with it enabled works.
        # KNOWN ISSUE: the inline comment about an object dtype series with None is misleading: the list of numbers with a missing entry yields a float series, not an object series.
        # Returns nothing.
        with pd.option_context("use_bottleneck", False):
            f = getattr(Series, name)

            # add some NaNs
            string_series_[5:15] = np.nan

            # mean, idxmax, idxmin, min, and max are valid for dates
            if name not in ["max", "min", "mean", "median", "std"]:
                ds = Series(date_range("1/1/2001", periods=10))
                msg = f"does not support operation '{name}'"
                with pytest.raises(TypeError, match=msg):
                    f(ds)

            # skipna or no
            assert pd.notna(f(string_series_))
            assert pd.isna(f(string_series_, skipna=False))

            # check the result is correct
            nona = string_series_.dropna()
            tm.assert_almost_equal(f(nona), alternate(nona.values))
            tm.assert_almost_equal(f(string_series_), alternate(nona.values))

            allna = string_series_ * np.nan

            if check_allna:
                assert np.isnan(f(allna))

            # dtype=object with None, it works!
            s = Series([1, 2, 3, None, 5])
            f(s)

            # GH#2888
            items = [0]
            items.extend(range(2**40, 2**40 + 1000))
            s = Series(items, dtype="int64")
            tm.assert_almost_equal(float(f(s)), float(alternate(s.values)))

            # check date range
            if check_objects:
                s = Series(pd.bdate_range("1/1/2000", periods=10))
                res = f(s)
                exp = alternate(s)
                assert res == exp

            # check on string data
            if name not in ["sum", "min", "max"]:
                with pytest.raises(TypeError, match=None):
                    f(Series(list("abc")))

            # Invalid axis.
            msg = "No axis named 1 for object type Series"
            with pytest.raises(ValueError, match=msg):
                f(string_series_, axis=1)

            if "numeric_only" in inspect.getfullargspec(f).args:
                # only the index is string; dtype is float
                f(string_series_, numeric_only=True)

    def test_sum(self):
        # [FUNCTION DOC]
        # Runs the shared checker for the sum reduction on a 20-element float series, with the
        # reference being the numpy sum and the all-missing check disabled.
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("sum", np.sum, string_series, check_allna=False)

    def test_mean(self):
        # [FUNCTION DOC]
        # Runs the shared checker for the mean reduction on a 20-element float series against the
        # numpy mean.
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("mean", np.mean, string_series)

    def test_median(self):
        # [FUNCTION DOC]
        # Runs the shared checker for the median reduction against the numpy median, then verifies
        # that the median of an integer series of ones matches numpy's median.
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("median", np.median, string_series)

        # test with integers, test failure
        int_ts = Series(np.ones(10, dtype=int), index=range(10))
        tm.assert_almost_equal(np.median(int_ts), int_ts.median())

    def test_prod(self):
        # [FUNCTION DOC]
        # Runs the shared checker for the product reduction against the numpy product.
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("prod", np.prod, string_series)

    def test_min(self):
        # [FUNCTION DOC]
        # Runs the shared checker for the minimum reduction against the numpy minimum, with the
        # object (business-date) check enabled.
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("min", np.min, string_series, check_objects=True)

    def test_max(self):
        # [FUNCTION DOC]
        # Runs the shared checker for the maximum reduction against the numpy maximum, with the
        # object (business-date) check enabled.
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("max", np.max, string_series, check_objects=True)

    def test_var_std(self):
        # [FUNCTION DOC]
        # Verifies standard deviation and variance: the shared checker is run with reference
        # functions using one delta degree of freedom; then a date-indexed series is checked with
        # four delta degrees of freedom against numpy; finally a single-element series gives a
        # missing result for both with one delta degree of freedom.
        string_series = Series(range(20), dtype=np.float64, name="series")
        datetime_series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        alt = lambda x: np.std(x, ddof=1)
        self._check_stat_op("std", alt, string_series)

        alt = lambda x: np.var(x, ddof=1)
        self._check_stat_op("var", alt, string_series)

        result = datetime_series.std(ddof=4)
        expected = np.std(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        result = datetime_series.var(ddof=4)
        expected = np.var(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.var(ddof=1)
        assert pd.isna(result)

        result = s.std(ddof=1)
        assert pd.isna(result)

    def test_sem(self):
        # [FUNCTION DOC]
        # Verifies the standard error of the mean: the shared checker with a reference of standard
        # deviation (one delta degree of freedom) divided by the square root of the length; a
        # date-indexed series with four delta degrees of freedom against the same formula; and a
        # single-element series giving a missing result.
        string_series = Series(range(20), dtype=np.float64, name="series")
        datetime_series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        alt = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        self._check_stat_op("sem", alt, string_series)

        result = datetime_series.sem(ddof=4)
        expected = np.std(datetime_series.values, ddof=4) / np.sqrt(
            len(datetime_series.values)
        )
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.sem(ddof=1)
        assert pd.isna(result)

    def test_skew(self, using_python_scalars):
        # [FUNCTION DOC]
        # Verifies skew against scipy's unbiased skew via the shared checker (skipped if scipy is
        # missing). Then corner cases for sizes one through three of constant series and data frames:
        # fewer than three values give missing results; with three values the skew is zero, the
        # scalar type being a Python float or numpy 64-bit float depending on the fixture flag
        # using_python_scalars (GH53482), and a data frame gives zeros for all columns.
        sp_stats = pytest.importorskip("scipy.stats")

        string_series = Series(range(20), dtype=np.float64, name="series")

        alt = lambda x: sp_stats.skew(x, bias=False)
        self._check_stat_op("skew", alt, string_series)

        # test corner cases, skew() returns NaN unless there's at least 3
        # values
        min_N = 3
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.skew())
                assert np.isnan(df.skew()).all()
            else:
                assert 0 == s.skew()
                if using_python_scalars:
                    assert type(s.skew()) == float
                else:
                    assert isinstance(s.skew(), np.float64)  # GH53482
                assert (df.skew() == 0).all()

    def test_kurt(self):
        # [FUNCTION DOC]
        # Verifies kurtosis against scipy's unbiased kurtosis via the shared checker (skipped if
        # scipy is missing).
        sp_stats = pytest.importorskip("scipy.stats")

        string_series = Series(range(20), dtype=np.float64, name="series")

        alt = lambda x: sp_stats.kurtosis(x, bias=False)
        self._check_stat_op("kurt", alt, string_series)

    def test_kurt_corner(self, using_python_scalars):
        # [FUNCTION DOC]
        # Corner cases for kurtosis on constant series and data frames of sizes one through four:
        # fewer than four values give missing results; with four values the kurtosis is zero, with
        # scalar type a Python float or numpy 64-bit float depending on the fixture flag
        # using_python_scalars (GH53482), and a data frame gives zeros for all columns.
        # test corner cases, kurt() returns NaN unless there's at least 4
        # values
        min_N = 4
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.kurt())
                assert np.isnan(df.kurt()).all()
            else:
                assert 0 == s.kurt()
                if using_python_scalars:
                    assert type(s.kurt()) == float
                else:
                    assert isinstance(s.kurt(), np.float64)  # GH53482
                assert (df.kurt() == 0).all()
