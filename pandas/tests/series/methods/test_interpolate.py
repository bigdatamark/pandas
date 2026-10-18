# [MODULE DOC]
# Tests for Series.interpolate: linear, time, index/values, scipy-based methods
# (nearest, slinear, zero, quadratic, cubic, barycentric, krogh, polynomial,
# spline, piecewise polynomial, from_derivatives, pchip, akima, cubicspline),
# limit, limit_direction and limit_area handling, invalid arguments, several
# index types (datetime, timedelta, multi, string, period, interval), and
# fill_value. Many tests skip when scipy is missing. Defines two fixtures that
# enumerate interpolation methods with their required keyword arguments.
# No import-time side effects.

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
    date_range,
    isna,
)
import pandas._testing as tm


@pytest.fixture(
    params=[
        "linear",
        "index",
        "values",
        "nearest",
        "slinear",
        "zero",
        "quadratic",
        "cubic",
        "barycentric",
        "krogh",
        "polynomial",
        "spline",
        "piecewise_polynomial",
        "from_derivatives",
        "pchip",
        "akima",
        "cubicspline",
    ]
)
def nontemporal_method(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the non-temporal interpolation method names
    # (everything except the time-weighted method). Returns a pair of the method name
    # and a dictionary of required keyword arguments (order 1 for polynomial and
    # spline, otherwise empty).
    """Fixture that returns an (method name, required kwargs) pair.

    This fixture does not include method 'time' as a parameterization; that
    method requires a Series with a DatetimeIndex, and is generally tested
    separately from these non-temporal methods.
    """
    method = request.param
    kwargs = {"order": 1} if method in ("spline", "polynomial") else {}
    return method, kwargs


@pytest.fixture(
    params=[
        "linear",
        "slinear",
        "zero",
        "quadratic",
        "cubic",
        "barycentric",
        "krogh",
        "polynomial",
        "spline",
        "piecewise_polynomial",
        "from_derivatives",
        "pchip",
        "akima",
        "cubicspline",
    ]
)
def interp_methods_ind(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the interpolation methods that work for
    # non-numeric index types (excluding time, index, nearest and values). Returns a
    # pair of the method name and a dictionary of required keyword arguments (order 1
    # for polynomial and spline, otherwise empty).
    """Fixture that returns a (method name, required kwargs) pair to
    be tested for various Index types.

    This fixture does not include methods - 'time', 'index', 'nearest',
    'values' as a parameterization
    """
    method = request.param
    kwargs = {"order": 1} if method in ("spline", "polynomial") else {}
    return method, kwargs


class TestSeriesInterpolateData:
    # [CLASS DOC]
    # Groups the data-driven tests of Series.interpolate. Stateless.
    @pytest.mark.xfail(reason="EA.fillna does not handle 'linear' method")
    def test_interpolate_period_values(self):
        # [FUNCTION DOC]
        # Expected-failure test (the fill implementation for extension arrays does not
        # handle the linear method): interpolating a period-cast datetime Series with a
        # missing value should give back the original periods.
        orig = Series(date_range("2012-01-01", periods=5))
        ser = orig.copy()
        ser[2] = pd.NaT

        # period cast
        ser_per = ser.dt.to_period("D")
        res_per = ser_per.interpolate()
        expected_per = orig.dt.to_period("D")
        tm.assert_series_equal(res_per, expected_per)

    def test_interpolate(self, datetime_series):
        # [FUNCTION DOC]
        # On the datetime_series fixture (equally spaced within each week): linear
        # interpolation restores values blanked out in two ranges; time-weighted
        # interpolation restores ordinal date values blanked in the middle.
        ts = Series(np.arange(len(datetime_series), dtype=float), datetime_series.index)

        ts_copy = ts.copy()

        # Set data between Tuesday and Thursday to NaN for 2 consecutive weeks.
        # Linear interpolation should fill in the missing values correctly,
        # as the index is equally-spaced within each week.
        ts_copy[1:4] = np.nan
        ts_copy[6:9] = np.nan

        linear_interp = ts_copy.interpolate(method="linear")
        tm.assert_series_equal(linear_interp, ts)

        ord_ts = Series(
            [d.toordinal() for d in datetime_series.index], index=datetime_series.index
        ).astype(float)

        ord_ts_copy = ord_ts.copy()
        ord_ts_copy[5:10] = np.nan

        time_interp = ord_ts_copy.interpolate(method="time")
        tm.assert_series_equal(time_interp, ord_ts)

    def test_interpolate_time_raises_for_non_timeseries(self):
        # [FUNCTION DOC]
        # Time interpolation on a Series without a DatetimeIndex raises a value error
        # saying it only works with a DatetimeIndex.
        # When method='time' is used on a non-TimeSeries that contains a null
        # value, a ValueError should be raised.
        non_ts = Series([0, 1, 2, np.nan])
        msg = "time-weighted interpolation only works on Series.* with a DatetimeIndex"
        with pytest.raises(ValueError, match=msg):
            non_ts.interpolate(method="time")

    def test_interpolate_cubicspline(self):
        # [FUNCTION DOC]
        # Needs scipy. Interpolating a reindexed integer Series onto quarter steps with
        # the cubicspline method gives the straight-line values between 1 and 3.
        pytest.importorskip("scipy")
        ser = Series([10, 11, 12, 13])

        expected = Series(
            [11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        result = ser.reindex(new_index).interpolate(method="cubicspline").loc[1:3]
        tm.assert_series_equal(result, expected)

    def test_interpolate_pchip(self):
        # [FUNCTION DOC]
        # Needs scipy. Interpolating onto quarter steps with pchip on a sorted random
        # Series does not raise and the slice can be read (GH5977); no values are checked.
        pytest.importorskip("scipy")
        ser = Series(np.sort(np.random.default_rng(2).uniform(size=100)))

        # interpolate at new_index
        new_index = ser.index.union(
            Index([49.25, 49.5, 49.75, 50.25, 50.5, 50.75])
        ).astype(float)
        interp_s = ser.reindex(new_index).interpolate(method="pchip")
        # does not blow up, GH5977
        interp_s.loc[49:51]

    def test_interpolate_akima(self):
        # [FUNCTION DOC]
        # Needs scipy. Akima interpolation of a linear Series gives the straight-line
        # values; with a derivative order argument of 1 the filled values are the constant
        # slope of 1.0.
        pytest.importorskip("scipy")
        ser = Series([10, 11, 12, 13])

        # interpolate at new_index where `der` is zero
        expected = Series(
            [11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        interp_s = ser.reindex(new_index).interpolate(method="akima")
        tm.assert_series_equal(interp_s.loc[1:3], expected)

        # interpolate at new_index where `der` is a non-zero int
        expected = Series(
            [11.0, 1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0, 13.0],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        interp_s = ser.reindex(new_index).interpolate(method="akima", der=1)
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    def test_interpolate_piecewise_polynomial(self):
        # [FUNCTION DOC]
        # Needs scipy. Piecewise-polynomial interpolation onto quarter steps gives the
        # straight-line values.
        pytest.importorskip("scipy")
        ser = Series([10, 11, 12, 13])

        expected = Series(
            [11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        interp_s = ser.reindex(new_index).interpolate(method="piecewise_polynomial")
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    def test_interpolate_from_derivatives(self):
        # [FUNCTION DOC]
        # Needs scipy. Interpolation by derivatives onto quarter steps gives the
        # straight-line values.
        pytest.importorskip("scipy")
        ser = Series([10, 11, 12, 13])

        expected = Series(
            [11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        interp_s = ser.reindex(new_index).interpolate(method="from_derivatives")
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            pytest.param(
                {"method": "polynomial", "order": 1}, marks=td.skip_if_no("scipy")
            ),
        ],
    )
    def test_interpolate_corners(self, kwargs):
        # [FUNCTION DOC]
        # Parametrized over default arguments or (with scipy) polynomial order 1. A
        # Series of only NaN is unchanged, and so is an empty Series.
        s = Series([np.nan, np.nan])
        tm.assert_series_equal(s.interpolate(**kwargs), s)

        s = Series([], dtype=object).interpolate()
        tm.assert_series_equal(s.interpolate(**kwargs), s)

    def test_interpolate_index_values(self):
        # [FUNCTION DOC]
        # Series with a random sorted float index and every third value known: the index
        # method fills missing entries as numpy linear interpolation on the index
        # values; the "values" method is a synonym of "index".
        s = Series(np.nan, index=np.sort(np.random.default_rng(2).random(30)))
        s.loc[::3] = np.random.default_rng(2).standard_normal(10)

        vals = s.index.values.astype(float)

        result = s.interpolate(method="index")

        bad = isna(s)
        good = ~bad
        expected = Series(
            np.interp(vals[bad], vals[good], s.values[good]), index=s.index[bad]
        )

        tm.assert_series_equal(result[bad], expected)

        # 'values' is synonymous with 'index' for the method kwarg
        other_result = s.interpolate(method="values")

        tm.assert_series_equal(other_result, result)
        tm.assert_series_equal(other_result[bad], expected)

    def test_interpolate_non_ts(self):
        # [FUNCTION DOC]
        # Time interpolation on a Series with a default integer index raises a value
        # error saying the index must be a DatetimeIndex.
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        msg = (
            "time-weighted interpolation only works on Series or DataFrames "
            "with a DatetimeIndex"
        )
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="time")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            pytest.param(
                {"method": "polynomial", "order": 1}, marks=td.skip_if_no("scipy")
            ),
        ],
    )
    def test_nan_interpolate(self, kwargs):
        # [FUNCTION DOC]
        # Parametrized over default arguments or polynomial order 1 (needs scipy).
        # A single interior NaN is replaced by the linear midpoint and the result is float.
        s = Series([0, 1, np.nan, 3])
        result = s.interpolate(**kwargs)
        expected = Series([0.0, 1.0, 2.0, 3.0])
        tm.assert_series_equal(result, expected)

    def test_nan_irregular_index(self):
        # [FUNCTION DOC]
        # Linear interpolation ignores the spacing of an irregular integer index and
        # uses positions, so a middle gap gets the average of neighbors.
        s = Series([1, 2, np.nan, 4], index=[1, 3, 5, 9])
        result = s.interpolate()
        expected = Series([1.0, 2.0, 3.0, 4.0], index=[1, 3, 5, 9])
        tm.assert_series_equal(result, expected)

    def test_nan_str_index(self):
        # [FUNCTION DOC]
        # A Series with a string index: the trailing NaN is filled by repeating the last
        # value (default linear method with forward direction).
        s = Series([0, 1, 2, np.nan], index=list("abcd"))
        result = s.interpolate()
        expected = Series([0.0, 1.0, 2.0, 2.0], index=list("abcd"))
        tm.assert_series_equal(result, expected)

    def test_interp_quad(self):
        # [FUNCTION DOC]
        # Needs scipy. Quadratic interpolation of 1, 4, NaN, 16 on index 1 to 4
        # yields 9.
        pytest.importorskip("scipy")
        sq = Series([1, 4, np.nan, 16], index=[1, 2, 3, 4])
        result = sq.interpolate(method="quadratic")
        expected = Series([1.0, 4.0, 9.0, 16.0], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)

    def test_interp_scipy_basic(self):
        # [FUNCTION DOC]
        # Needs scipy. Checks the slinear, nearest, zero, quadratic (GH #15662) and cubic
        # methods on a Series with two gaps against fixed expected values.
        # KNOWN ISSUE: each method's call and comparison is repeated (the same
        # interpolate call and assertion appear twice for slinear, nearest, zero and
        # quadratic), so the repeats add no coverage.
        pytest.importorskip("scipy")
        s = Series([1, 3, np.nan, 12, np.nan, 25])
        # slinear
        expected = Series([1.0, 3.0, 7.5, 12.0, 18.5, 25.0])
        result = s.interpolate(method="slinear")
        tm.assert_series_equal(result, expected)
        result = s.interpolate(method="slinear")
        tm.assert_series_equal(result, expected)
        # nearest
        expected = Series([1, 3, 3, 12, 12, 25.0])
        result = s.interpolate(method="nearest")
        tm.assert_series_equal(result, expected.astype("float"))

        result = s.interpolate(method="nearest")
        tm.assert_series_equal(result, expected)
        # zero
        expected = Series([1, 3, 3, 12, 12, 25.0])
        result = s.interpolate(method="zero")
        tm.assert_series_equal(result, expected.astype("float"))

        result = s.interpolate(method="zero")
        tm.assert_series_equal(result, expected)
        # quadratic
        # GH #15662.
        expected = Series([1, 3.0, 6.823529, 12.0, 18.058824, 25.0])
        result = s.interpolate(method="quadratic")
        tm.assert_series_equal(result, expected)

        result = s.interpolate(method="quadratic")
        tm.assert_series_equal(result, expected)
        # cubic
        expected = Series([1.0, 3.0, 6.8, 12.0, 18.2, 25.0])
        result = s.interpolate(method="cubic")
        tm.assert_series_equal(result, expected)

    def test_interp_limit(self):
        # [FUNCTION DOC]
        # Linear interpolation with a limit of 2 over a gap of three fills only the first
        # two NaN values.
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        expected = Series([1.0, 3.0, 5.0, 7.0, np.nan, 11.0])
        result = s.interpolate(method="linear", limit=2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("limit", [-1, 0])
    def test_interpolate_invalid_nonpositive_limit(self, nontemporal_method, limit):
        # [FUNCTION DOC]
        # Regression test (GH 9217), parametrized over limit values -1 and 0 and all
        # non-temporal methods: a non-positive limit raises a value error ("Limit must be
        # greater than 0").
        # GH 9217: make sure limit is greater than zero.
        s = Series([1, 2, np.nan, 4])
        method, kwargs = nontemporal_method
        with pytest.raises(ValueError, match="Limit must be greater than 0"):
            s.interpolate(limit=limit, method=method, **kwargs)

    def test_interpolate_invalid_float_limit(self, nontemporal_method):
        # [FUNCTION DOC]
        # Regression test (GH 9217), parametrized over non-temporal methods: a float
        # limit (2.0) raises a value error ("Limit must be an integer").
        # GH 9217: make sure limit is an integer.
        s = Series([1, 2, np.nan, 4])
        method, kwargs = nontemporal_method
        limit = 2.0
        with pytest.raises(ValueError, match="Limit must be an integer"):
            s.interpolate(limit=limit, method=method, **kwargs)

    @pytest.mark.parametrize("invalid_method", [None, "nonexistent_method"])
    def test_interp_invalid_method(self, invalid_method):
        # [FUNCTION DOC]
        # Parametrized over None and an unknown method name. Interpolation raises a value
        # error: for None, saying the method must be a string; otherwise saying the method
        # cannot be used. The same error is raised even when the limit is also invalid
        # (-1), so method validation comes first.
        s = Series([1, 3, np.nan, 12, np.nan, 25])

        msg = "Can not interpolate with method=nonexistent_method"
        if invalid_method is None:
            msg = "'method' should be a string, not None"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=invalid_method)

        # When an invalid method and invalid limit (such as -1) are
        # provided, the error message reflects the invalid method.
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=invalid_method, limit=-1)

    def test_interp_limit_forward(self):
        # [FUNCTION DOC]
        # Limit of 2 with the forward limit direction, written lower case and upper
        # case (case-insensitive), fills only the first two NaN values of the gap.
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        # Provide 'forward' (the default) explicitly here.
        expected = Series([1.0, 3.0, 5.0, 7.0, np.nan, 11.0])

        result = s.interpolate(method="linear", limit=2, limit_direction="forward")
        tm.assert_series_equal(result, expected)

        result = s.interpolate(method="linear", limit=2, limit_direction="FORWARD")
        tm.assert_series_equal(result, expected)

    def test_interp_unlimited(self):
        # [FUNCTION DOC]
        # Regression test (issue #16282): with no limit given the fill is unlimited. Both
        # directions fill leading and trailing NaN; forward leaves the leading NaN;
        # backward leaves the trailing NaN.
        # these test are for issue #16282 default Limit=None is unlimited
        s = Series([np.nan, 1.0, 3.0, np.nan, np.nan, np.nan, 11.0, np.nan])
        expected = Series([1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 11.0])
        result = s.interpolate(method="linear", limit_direction="both")
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 11.0])
        result = s.interpolate(method="linear", limit_direction="forward")
        tm.assert_series_equal(result, expected)

        expected = Series([1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, np.nan])
        result = s.interpolate(method="linear", limit_direction="backward")
        tm.assert_series_equal(result, expected)

    def test_interp_limit_bad_direction(self):
        # [FUNCTION DOC]
        # An invalid limit_direction value raises a value error listing the allowed
        # values, whether or not a limit is given.
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        msg = (
            r"Invalid limit_direction: expecting one of \['forward', "
            r"'backward', 'both'\], got 'abc'"
        )
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="linear", limit=2, limit_direction="abc")

        # raises an error even if no limit is specified.
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="linear", limit_direction="abc")

    # limit_area introduced GH #16284
    def test_interp_limit_area(self):
        # [FUNCTION DOC]
        # Covers limit_area "inside" (only fill between valid values) and "outside"
        # (only fill beyond the first/last valid value), alone and combined with limit
        # and limit_direction, using a nine-element Series (GH #9218, GH #16284). An
        # invalid limit_area raises a value error listing the allowed values.
        # These tests are for issue #9218 -- fill NaNs in both directions.
        s = Series([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan])

        expected = Series([np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan])
        result = s.interpolate(method="linear", limit_area="inside")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [np.nan, np.nan, 3.0, 4.0, np.nan, np.nan, 7.0, np.nan, np.nan]
        )
        result = s.interpolate(method="linear", limit_area="inside", limit=1)
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, np.nan, np.nan])
        result = s.interpolate(
            method="linear", limit_area="inside", limit_direction="both", limit=1
        )
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0])
        result = s.interpolate(method="linear", limit_area="outside")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan]
        )
        result = s.interpolate(method="linear", limit_area="outside", limit=1)
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan])
        result = s.interpolate(
            method="linear", limit_area="outside", limit_direction="both", limit=1
        )
        tm.assert_series_equal(result, expected)

        expected = Series([3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan])
        result = s.interpolate(
            method="linear", limit_area="outside", limit_direction="backward"
        )
        tm.assert_series_equal(result, expected)

        # raises an error even if limit type is wrong.
        msg = r"Invalid limit_area: expecting one of \['inside', 'outside'\], got abc"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="linear", limit_area="abc")

    @pytest.mark.parametrize(
        "data, kwargs",
        (
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                {"method": "pad", "limit_area": "inside"},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                {"method": "pad", "limit_area": "inside", "limit": 1},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                {"method": "pad", "limit_area": "outside"},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                {"method": "pad", "limit_area": "outside", "limit": 1},
            ),
            (
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                {"method": "pad", "limit_area": "outside", "limit": 1},
            ),
            (
                range(5),
                {"method": "pad", "limit_area": "outside", "limit": 1},
            ),
        ),
    )
    def test_interp_limit_area_with_pad(self, data, kwargs):
        # [FUNCTION DOC]
        # Regression test (GH26796), parametrized over data and keywords combining the
        # pad method with limit_area/limit: interpolate raises a value error that the
        # pad method cannot be used.
        # GH26796

        s = Series(data)
        msg = "Can not interpolate with method=pad"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(**kwargs)

    @pytest.mark.parametrize(
        "data, kwargs",
        (
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                {"method": "bfill", "limit_area": "inside"},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                {"method": "bfill", "limit_area": "inside", "limit": 1},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                {"method": "bfill", "limit_area": "outside"},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                {"method": "bfill", "limit_area": "outside", "limit": 1},
            ),
        ),
    )
    def test_interp_limit_area_with_backfill(self, data, kwargs):
        # [FUNCTION DOC]
        # Regression test (GH26796), parametrized over the backfill method with
        # limit_area/limit: interpolate raises a value error that bfill cannot be used.
        # GH26796
        s = Series(data)

        msg = "Can not interpolate with method=bfill"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(**kwargs)

    def test_interp_limit_direction(self):
        # [FUNCTION DOC]
        # Issue #9218: backward direction fills up to two NaN before the next valid value;
        # both directions with limit 1 fill one from each side. A longer Series is also
        # checked with limits of 2 and 1 in both directions.
        # These tests are for issue #9218 -- fill NaNs in both directions.
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        expected = Series([1.0, 3.0, np.nan, 7.0, 9.0, 11.0])
        result = s.interpolate(method="linear", limit=2, limit_direction="backward")
        tm.assert_series_equal(result, expected)

        expected = Series([1.0, 3.0, 5.0, np.nan, 9.0, 11.0])
        result = s.interpolate(method="linear", limit=1, limit_direction="both")
        tm.assert_series_equal(result, expected)

        # Check that this works on a longer series of nans.
        s = Series([1, 3, np.nan, np.nan, np.nan, 7, 9, np.nan, np.nan, 12, np.nan])

        expected = Series([1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0])
        result = s.interpolate(method="linear", limit=2, limit_direction="both")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [1.0, 3.0, 4.0, np.nan, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0]
        )
        result = s.interpolate(method="linear", limit=1, limit_direction="both")
        tm.assert_series_equal(result, expected)

    def test_interp_limit_direction_limit_exceeds_length(self):
        # [FUNCTION DOC]
        # Regression test (GH#64322): a limit larger than the Series length behaves like
        # no limit for forward, backward, and both directions.
        # GH#64322
        s = Series([np.nan, 1, 1, 1, 1, np.nan])
        result = s.interpolate(limit_direction="both", limit=10)
        expected = Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        tm.assert_series_equal(result, expected)

        s2 = Series([1, np.nan, np.nan, 4])
        result_fwd = s2.interpolate(limit_direction="forward", limit=100)
        expected_fwd = Series([1.0, 2.0, 3.0, 4.0])
        tm.assert_series_equal(result_fwd, expected_fwd)

        result_bwd = s2.interpolate(limit_direction="backward", limit=100)
        expected_bwd = Series([1.0, 2.0, 3.0, 4.0])
        tm.assert_series_equal(result_bwd, expected_bwd)

    def test_interp_limit_to_ends(self):
        # [FUNCTION DOC]
        # Issue #10420: with leading NaN, backward direction fills them from the first
        # valid value; both directions fills leading and trailing.
        # These test are for issue #10420 -- flow back to beginning.
        s = Series([np.nan, np.nan, 5, 7, 9, np.nan])

        expected = Series([5.0, 5.0, 5.0, 7.0, 9.0, np.nan])
        result = s.interpolate(method="linear", limit=2, limit_direction="backward")
        tm.assert_series_equal(result, expected)

        expected = Series([5.0, 5.0, 5.0, 7.0, 9.0, 9.0])
        result = s.interpolate(method="linear", limit=2, limit_direction="both")
        tm.assert_series_equal(result, expected)

    def test_interp_limit_before_ends(self):
        # [FUNCTION DOC]
        # Issue #11115: with a limit of 1, forward, backward and both directions each
        # fill one position beside the valid run and stop (not at the ends).
        # These test are for issue #11115 -- limit ends properly.
        s = Series([np.nan, np.nan, 5, 7, np.nan, np.nan])

        expected = Series([np.nan, np.nan, 5.0, 7.0, 7.0, np.nan])
        result = s.interpolate(method="linear", limit=1, limit_direction="forward")
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, 5.0, 5.0, 7.0, np.nan, np.nan])
        result = s.interpolate(method="linear", limit=1, limit_direction="backward")
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, 5.0, 5.0, 7.0, 7.0, np.nan])
        result = s.interpolate(method="linear", limit=1, limit_direction="both")
        tm.assert_series_equal(result, expected)

    def test_interp_all_good(self):
        # [FUNCTION DOC]
        # Needs scipy. A Series without missing values is returned unchanged by
        # polynomial and default interpolation.
        pytest.importorskip("scipy")
        s = Series([1, 2, 3])
        result = s.interpolate(method="polynomial", order=1)
        tm.assert_series_equal(result, s)

        # non-scipy
        result = s.interpolate()
        tm.assert_series_equal(result, s)

    @pytest.mark.parametrize(
        "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
    )
    def test_interp_multiIndex(self, check_scipy):
        # [FUNCTION DOC]
        # Parametrized over whether scipy is available: for a MultiIndex Series the linear
        # default fills the trailing NaN with the last value; with scipy, other methods
        # raise a value error that only the linear method is supported for MultiIndex.
        idx = MultiIndex.from_tuples([(0, "a"), (1, "b"), (2, "c")])
        s = Series([1, 2, np.nan], index=idx)

        expected = s.copy()
        expected.loc[2] = 2
        result = s.interpolate()
        tm.assert_series_equal(result, expected)

        msg = "Only `method=linear` interpolation is supported on MultiIndexes"
        if check_scipy:
            with pytest.raises(ValueError, match=msg):
                s.interpolate(method="polynomial", order=1)

    def test_interp_nonmono_raise(self):
        # [FUNCTION DOC]
        # Needs scipy. The krogh method on a non-monotonic index raises a value error
        # that the index must be monotonic.
        pytest.importorskip("scipy")
        s = Series([1, np.nan, 3], index=[0, 2, 1])
        msg = "krogh interpolation requires that the index be monotonic"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="krogh")

    @pytest.mark.parametrize("method", ["nearest", "pad"])
    def test_interp_datetime64(self, method, tz_naive_fixture):
        # [FUNCTION DOC]
        # Needs scipy; parametrized over the nearest and pad methods and the tz fixture.
        # For a Series with a datetime index, nearest fills with the neighbor; pad raises
        # a value error that it cannot be used with interpolate. (The local variable is named as
        # a frame but is a Series.)
        pytest.importorskip("scipy")
        df = Series(
            [1, np.nan, 3], index=date_range("1/1/2000", periods=3, tz=tz_naive_fixture)
        )

        if method == "nearest":
            result = df.interpolate(method=method)
            expected = Series(
                [1.0, 1.0, 3.0],
                index=date_range("1/1/2000", periods=3, tz=tz_naive_fixture),
            )
            tm.assert_series_equal(result, expected)
        else:
            msg = "Can not interpolate with method=pad"
            with pytest.raises(ValueError, match=msg):
                df.interpolate(method=method)

    def test_interp_pad_datetime64tz_values(self):
        # [FUNCTION DOC]
        # Regression test (GH#27628): interpolating tz-aware datetime values with the
        # pad method raises a value error that the method cannot be used.
        # GH#27628 missing.interpolate_2d should handle datetimetz values
        dti = date_range("2015-04-05", periods=3, tz="US/Central")
        ser = Series(dti)
        ser[1] = pd.NaT

        msg = "Can not interpolate with method=pad"
        with pytest.raises(ValueError, match=msg):
            ser.interpolate(method="pad")

    def test_interp_limit_no_nans(self):
        # [FUNCTION DOC]
        # Regression test (GH 7173): a limit on a Series with no missing values is
        # a no-op.
        # GH 7173
        s = Series([1.0, 2.0, 3.0])
        result = s.interpolate(limit=1)
        expected = s
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["polynomial", "spline"])
    def test_no_order(self, method):
        # [FUNCTION DOC]
        # Needs scipy; parametrized over polynomial and spline. Omitting the order raises
        # a value error requiring the order (GH-10633, GH-24014).
        # see GH-10633, GH-24014
        pytest.importorskip("scipy")
        s = Series([0, 1, np.nan, 3])
        msg = "You must specify the order of the spline or polynomial"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=method)

    @pytest.mark.parametrize("order", [-1, -1.0, 0, 0.0, np.nan])
    def test_interpolate_spline_invalid_order(self, order):
        # [FUNCTION DOC]
        # Needs scipy; parametrized over invalid orders (negative, zero, NaN as int or
        # float). Spline interpolation raises a value error saying the order must be
        # specified and greater than 0.
        pytest.importorskip("scipy")
        s = Series([0, 1, np.nan, 3])
        msg = "order needs to be specified and greater than 0"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="spline", order=order)

    def test_spline(self):
        # [FUNCTION DOC]
        # Needs scipy. Linear (order 1) spline fills interior gaps of a Series with
        # the straight-line values.
        pytest.importorskip("scipy")
        s = Series([1, 2, np.nan, 4, 5, np.nan, 7])
        result = s.interpolate(method="spline", order=1)
        expected = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        tm.assert_series_equal(result, expected)

    def test_spline_extrapolate(self):
        # [FUNCTION DOC]
        # Needs scipy. Linear spline with the extrapolation option: mode 3 repeats the
        # last value for a trailing NaN, mode 0 extrapolates the line.
        pytest.importorskip("scipy")
        s = Series([1, 2, 3, 4, np.nan, 6, np.nan])
        result3 = s.interpolate(method="spline", order=1, ext=3)
        expected3 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0])
        tm.assert_series_equal(result3, expected3)

        result1 = s.interpolate(method="spline", order=1, ext=0)
        expected1 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        tm.assert_series_equal(result1, expected1)

    def test_spline_smooth(self):
        # [FUNCTION DOC]
        # Needs scipy. A cubic spline with smoothing parameter zero gives a different
        # value at a gap than one with the default smoothing.
        pytest.importorskip("scipy")
        s = Series([1, 2, np.nan, 4, 5.1, np.nan, 7])
        assert (
            s.interpolate(method="spline", order=3, s=0)[5]
            != s.interpolate(method="spline", order=3)[5]
        )

    def test_spline_interpolation(self):
        # [FUNCTION DOC]
        # Needs scipy. Intended to check spline interpolation on a quadratic sequence
        # with random NaN values.
        # KNOWN ISSUE: result and expected are computed by the identical call, so the
        # assertion is a tautology and verifies nothing about interpolation values.
        # Explicit cast to float to avoid implicit cast when setting np.nan
        pytest.importorskip("scipy")
        s = Series(np.arange(10) ** 2, dtype="float")
        s[np.random.default_rng(2).integers(0, 9, 3)] = np.nan
        result1 = s.interpolate(method="spline", order=1)
        expected1 = s.interpolate(method="spline", order=1)
        tm.assert_series_equal(result1, expected1)

    def test_interp_timedelta64(self):
        # [FUNCTION DOC]
        # Regression test (GH 6424): time interpolation on a timedelta index fills the
        # middle value, including for non-uniformly spaced timedeltas (1.666667).
        # GH 6424
        df = Series([1, np.nan, 3], index=pd.to_timedelta([1, 2, 3]))
        result = df.interpolate(method="time")
        expected = Series([1.0, 2.0, 3.0], index=pd.to_timedelta([1, 2, 3]))
        tm.assert_series_equal(result, expected)

        # test for non uniform spacing
        df = Series([1, np.nan, 3], index=pd.to_timedelta([1, 2, 4]))
        result = df.interpolate(method="time")
        expected = Series([1.0, 1.666667, 3.0], index=pd.to_timedelta([1, 2, 4]))
        tm.assert_series_equal(result, expected)

    def test_series_interpolate_method_values(self):
        # [FUNCTION DOC]
        # Regression test (GH#1646): the "values" method on a daily datetime index equals
        # the default interpolation.
        # GH#1646
        rng = date_range("1/1/2000", "1/20/2000", freq="D")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        ts[::2] = np.nan

        result = ts.interpolate(method="values")
        exp = ts.interpolate()
        tm.assert_series_equal(result, exp)

    def test_series_interpolate_intraday(self):
        # [FUNCTION DOC]
        # Regression test (issue 1698): time interpolation on a 12-hour grid with offsets
        # of one hour gives the same fill values as a 12-day grid with offsets of one day
        # (proportional spacing), so the values are equal.
        # #1698
        index = date_range("1/1/2012", periods=4, freq="12D")
        ts = Series([0, 12, 24, 36], index)
        new_index = index.append(index + pd.DateOffset(days=1)).sort_values()

        exp = ts.reindex(new_index).interpolate(method="time")

        index = date_range("1/1/2012", periods=4, freq="12h")
        ts = Series([0, 12, 24, 36], index)
        new_index = index.append(index + pd.DateOffset(hours=1)).sort_values()
        result = ts.reindex(new_index).interpolate(method="time")

        tm.assert_numpy_array_equal(result.values, exp.values)

    @pytest.mark.parametrize(
        "ind",
        [
            ["a", "b", "c", "d"],
            pd.period_range(start="2019-01-01", periods=4),
            pd.interval_range(start=0, end=4),
        ],
    )
    def test_interp_non_timedelta_index(self, interp_methods_ind, ind):
        # [FUNCTION DOC]
        # Regression test (gh 21662), parametrized over a string, period and interval
        # index and over the interpolation methods fixture. For the linear method the
        # result fills the gap; for other methods it raises a value error saying the index
        # column must be numeric or datetime when using a non-linear method. The pchip
        # method needs scipy.
        # gh 21662
        df = pd.DataFrame([0, 1, np.nan, 3], index=ind)

        method, kwargs = interp_methods_ind
        if method == "pchip":
            pytest.importorskip("scipy")

        if method == "linear":
            result = df[0].interpolate(**kwargs)
            expected = Series([0.0, 1.0, 2.0, 3.0], name=0, index=ind)
            tm.assert_series_equal(result, expected)
        else:
            expected_error = (
                "Index column must be numeric or datetime type when "
                f"using {method} method other than linear. "
                "Try setting a numeric or datetime index column before "
                "interpolating."
            )
            with pytest.raises(ValueError, match=expected_error):
                df[0].interpolate(method=method, **kwargs)

    def test_interpolate_timedelta_index(self, request, interp_methods_ind):
        # [FUNCTION DOC]
        # Needs scipy. Interpolates a frame column with a timedelta index using each
        # method of the fixture; cubic and zero are marked expected failures for the
        # timedelta index. The gap value must equal the linear one.
        # KNOWN ISSUE: the docstring says the test covers object, period and timedelta
        # index types, but only a timedelta index is used here (the other types are
        # tested elsewhere).
        """
        Tests for non numerical index types  - object, period, timedelta
        Note that all methods except time, index, nearest and values
        are tested here.
        """
        # gh 21662
        pytest.importorskip("scipy")
        ind = pd.timedelta_range(start=1, periods=4)
        df = pd.DataFrame([0, 1, np.nan, 3], index=ind)

        method, kwargs = interp_methods_ind

        if method in {"cubic", "zero"}:
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"{method} interpolation is not supported for TimedeltaIndex"
                )
            )
        result = df[0].interpolate(method=method, **kwargs)
        expected = Series([0.0, 1.0, 2.0, 3.0], name=0, index=ind)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending, expected_values",
        [(True, [1, 2, 3, 9, 10]), (False, [10, 9, 3, 2, 1])],
    )
    def test_interpolate_unsorted_index(self, ascending, expected_values):
        # [FUNCTION DOC]
        # Regression test (GH 21037), parametrized over ascending order: after sorting a
        # Series with an unsorted integer index in either direction, the index method
        # fills the gap and the result has a float dtype with the sorted index.
        # GH 21037
        ts = Series(data=[10, 9, np.nan, 2, 1], index=[10, 9, 3, 2, 1])
        result = ts.sort_index(ascending=ascending).interpolate(method="index")
        expected = Series(data=expected_values, index=expected_values, dtype=float)
        tm.assert_series_equal(result, expected)

    def test_interpolate_asfreq_raises(self):
        # [FUNCTION DOC]
        # Interpolating an object Series with the asfreq method raises a value error that
        # the method cannot be used.
        ser = Series(["a", None, "b"], dtype=object)
        msg = "Can not interpolate with method=asfreq"
        with pytest.raises(ValueError, match=msg):
            ser.interpolate(method="asfreq")

    def test_interpolate_fill_value(self):
        # [FUNCTION DOC]
        # Regression test (GH#54920), needs scipy: the nearest method with a fill value
        # of 0 uses that value for the trailing extrapolated position, while the
        # leading missing value (before the first valid one) stays missing.
        # GH#54920
        pytest.importorskip("scipy")
        ser = Series([np.nan, 0, 1, np.nan, 3, np.nan])
        result = ser.interpolate(method="nearest", fill_value=0)
        expected = Series([np.nan, 0, 1, 1, 3, 0])
        tm.assert_series_equal(result, expected)
