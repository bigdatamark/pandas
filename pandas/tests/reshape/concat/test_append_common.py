# [MODULE DOC]
# Tests for dtype coercion rules shared by index append, series append (via the
# internal single-object append helper) and the concatenate function. Defines a
# parametrized fixture providing (dtype label, three-element value list) pairs
# for bool, integer, float, categorical, object, datetime (naive and tz-aware),
# timedelta and period data, and one test class exercising same-dtype and mixed
# dtype concatenation, datetimelike/period fallbacks to object, and categorical
# coercion rules. Uses shared fixtures index_or_series, tz_aware_fixture and
# using_infer_string from conftest.
import zoneinfo

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


@pytest.fixture(
    params=list(
        {
            "bool": [True, False, True],
            "int64": [1, 2, 3],
            "float64": [1.1, np.nan, 3.3],
            "category": Categorical(["X", "Y", "Z"]),
            "object": ["a", "b", "c"],
            "datetime64[s]": [
                pd.Timestamp("2011-01-01").as_unit("s"),
                pd.Timestamp("2011-01-02").as_unit("s"),
                pd.Timestamp("2011-01-03").as_unit("s"),
            ],
            "datetime64[s, US/Eastern]": [
                pd.Timestamp("2011-01-01", tz="US/Eastern").as_unit("s"),
                pd.Timestamp("2011-01-02", tz="US/Eastern").as_unit("s"),
                pd.Timestamp("2011-01-03", tz="US/Eastern").as_unit("s"),
            ],
            "timedelta64[us]": [
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
                pd.Timedelta("3 days"),
            ],
            "period[M]": [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Period("2011-03", freq="M"),
            ],
        }.items()
    )
)
def item(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture for this module's tests. Parametrized over a
    # mapping of dtype label to a list of three values (the labels cover bool,
    # int64, float64, category, object, datetime64 second resolution naive and
    # US/Eastern, timedelta64 microsecond, and monthly period). Returns the pair
    # of label and values for the current parameter.
    key, data = request.param
    return key, data


@pytest.fixture
def item2(item):
    # [FUNCTION DOC]
    # HELPER: fixture returning the same pair as the item fixture; it exists so a
    # test can request two independently parametrized items (cross product of
    # dtypes).
    return item


class TestConcatAppendCommon:
    # [CLASS DOC]
    # Collects tests of common dtype coercion rules between concatenate and the
    # append operations on indexes and series. Stateless.
    """
    Test common dtype coercion rules between concat and append.
    """

    def test_dtypes(self, item, index_or_series, using_infer_string):
        # [FUNCTION DOC]
        # Sanity check that each item fixture value, wrapped in an index or series (per
        # the index_or_series fixture), has the dtype named by the label; object label
        # maps to string when the infer-string mode is active, and series of periods are
        # checked against monthly period dtype.
        # to confirm test case covers intended dtypes
        typ, vals = item
        obj = index_or_series(vals)
        if typ == "object" and using_infer_string:
            typ = "string"
        if isinstance(obj, Index):
            assert obj.dtype == typ
        elif isinstance(obj, Series):
            if typ.startswith("period"):
                assert obj.dtype == "Period[M]"
            else:
                assert obj.dtype == typ

    def test_concatlike_same_dtypes(self, item):
        # [FUNCTION DOC]
        # Verifies, for each dtype in the item fixture, that appending or concatenating
        # objects of the same dtype keeps the dtype: index append with two and three
        # elements, name handling (names kept only if equal), series append and
        # concatenate with two or three elements and name handling, and type errors
        # raised for non-index inputs to index append and for non-series inputs to
        # concatenate (GH 13660).
        # GH 13660
        typ1, vals1 = item

        vals2 = vals1
        vals3 = vals1

        if typ1 == "category":
            exp_data = Categorical(list(vals1) + list(vals2))
            exp_data3 = Categorical(list(vals1) + list(vals2) + list(vals3))
        else:
            exp_data = vals1 + vals2
            exp_data3 = vals1 + vals2 + vals3

        # ----- Index ----- #

        # index.append
        res = Index(vals1).append(Index(vals2))
        exp = Index(exp_data)
        tm.assert_index_equal(res, exp)

        # 3 elements
        res = Index(vals1).append([Index(vals2), Index(vals3)])
        exp = Index(exp_data3)
        tm.assert_index_equal(res, exp)

        # index.append name mismatch
        i1 = Index(vals1, name="x")
        i2 = Index(vals2, name="y")
        res = i1.append(i2)
        exp = Index(exp_data)
        tm.assert_index_equal(res, exp)

        # index.append name match
        i1 = Index(vals1, name="x")
        i2 = Index(vals2, name="x")
        res = i1.append(i2)
        exp = Index(exp_data, name="x")
        tm.assert_index_equal(res, exp)

        # cannot append non-index
        with pytest.raises(TypeError, match="all inputs must be Index"):
            Index(vals1).append(vals2)

        with pytest.raises(TypeError, match="all inputs must be Index"):
            Index(vals1).append([Index(vals2), vals3])

        # ----- Series ----- #

        # series.append
        res = Series(vals1)._append_internal(Series(vals2), ignore_index=True)
        exp = Series(exp_data)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # concat
        res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # 3 elements
        exp = Series(exp_data3)
        res = pd.concat(
            [Series(vals1), Series(vals2), Series(vals3)],
            ignore_index=True,
        )
        tm.assert_series_equal(res, exp)

        # name mismatch
        s1 = Series(vals1, name="x")
        s2 = Series(vals2, name="y")
        res = s1._append_internal(s2, ignore_index=True)
        exp = Series(exp_data)
        tm.assert_series_equal(res, exp, check_index_type=True)

        res = pd.concat([s1, s2], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # name match
        s1 = Series(vals1, name="x")
        s2 = Series(vals2, name="x")
        res = s1._append_internal(s2, ignore_index=True)
        exp = Series(exp_data, name="x")
        tm.assert_series_equal(res, exp, check_index_type=True)

        res = pd.concat([s1, s2], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # cannot append non-index
        msg = (
            r"cannot concatenate object of type '.+'; "
            "only Series and DataFrame objs are valid"
        )
        with pytest.raises(TypeError, match=msg):
            pd.concat([Series(vals1), vals2])

        with pytest.raises(TypeError, match=msg):
            pd.concat([Series(vals1), Series(vals2), vals3])

    def test_concatlike_dtypes_coercion(self, item, item2, request):
        # [FUNCTION DOC]
        # Cross product of two item fixture dtypes. Skips same-dtype pairs and any
        # categorical pair. Verifies index append and series append/concatenate of two
        # and three elements produce the inferred common dtype, or object for pairs
        # expected to fall back; bool mixed with numeric is marked as an expected failure
        # because the series result is expected to be numeric (GH 39187).
        # KNOWN ISSUE: the object-fallback branch compares against dtype labels with
        # nanosecond units, but the fixture labels use second and microsecond units, so
        # that branch can never be taken and the expected dtype stays inferred.
        # GH 13660
        typ1, vals1 = item
        typ2, vals2 = item2

        vals3 = vals2

        # basically infer
        exp_index_dtype = None
        exp_series_dtype = None

        if typ1 == typ2:
            pytest.skip("same dtype is tested in test_concatlike_same_dtypes")
        elif typ1 == "category" or typ2 == "category":
            pytest.skip("categorical type tested elsewhere")

        # specify expected dtype
        if typ1 == "bool" and typ2 in ("int64", "float64"):
            # series coerces to numeric based on numpy rule
            # index doesn't because bool is object dtype
            exp_series_dtype = typ2
            mark = pytest.mark.xfail(reason="GH#39187 casting to object")
            request.applymarker(mark)
        elif typ2 == "bool" and typ1 in ("int64", "float64"):
            exp_series_dtype = typ1
            mark = pytest.mark.xfail(reason="GH#39187 casting to object")
            request.applymarker(mark)
        elif typ1 in {"datetime64[ns, US/Eastern]", "timedelta64[ns]"} or typ2 in {
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        }:
            exp_index_dtype = object
            exp_series_dtype = object

        exp_data = vals1 + vals2
        exp_data3 = vals1 + vals2 + vals3

        # ----- Index ----- #

        # index.append
        # GH#39817
        res = Index(vals1).append(Index(vals2))
        exp = Index(exp_data, dtype=exp_index_dtype)
        tm.assert_index_equal(res, exp)

        # 3 elements
        res = Index(vals1).append([Index(vals2), Index(vals3)])
        exp = Index(exp_data3, dtype=exp_index_dtype)
        tm.assert_index_equal(res, exp)

        # ----- Series ----- #

        # series._append
        # GH#39817
        res = Series(vals1)._append_internal(Series(vals2), ignore_index=True)
        exp = Series(exp_data, dtype=exp_series_dtype)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # concat
        # GH#39817
        res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # 3 elements
        # GH#39817
        exp = Series(exp_data3, dtype=exp_series_dtype)
        res = pd.concat(
            [Series(vals1), Series(vals2), Series(vals3)],
            ignore_index=True,
        )
        tm.assert_series_equal(res, exp)

    def test_concatlike_common_coerce_to_pandas_object(self):
        # [FUNCTION DOC]
        # Verifies that appending a datetime index and a timedelta index (and the series
        # equivalents, via append helper and concatenate) gives an object result whose
        # elements are pandas Timestamp and Timedelta instances, not stdlib types
        # (GH 13626).
        # GH 13626
        # result must be Timestamp/Timedelta, not datetime.datetime/timedelta
        dti = pd.DatetimeIndex(["2011-01-01", "2011-01-02"])
        tdi = pd.TimedeltaIndex(["1 days", "2 days"])

        exp = Index(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
            ]
        )

        res = dti.append(tdi)
        tm.assert_index_equal(res, exp)
        assert isinstance(res[0], pd.Timestamp)
        assert isinstance(res[-1], pd.Timedelta)

        dts = Series(dti)
        tds = Series(tdi)
        res = dts._append_internal(tds)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        assert isinstance(res.iloc[0], pd.Timestamp)
        assert isinstance(res.iloc[-1], pd.Timedelta)

        res = pd.concat([dts, tds])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        assert isinstance(res.iloc[0], pd.Timestamp)
        assert isinstance(res.iloc[-1], pd.Timedelta)

    def test_concatlike_datetimetz(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Parametrized by tz_aware_fixture. Verifies that same-timezone datetime indexes
        # and series append/concatenate to a tz-aware datetime result (GH 7795).
        tz = tz_aware_fixture
        # GH 7795
        dti1 = pd.DatetimeIndex(["2011-01-01", "2011-01-02"], tz=tz)
        dti2 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"], tz=tz)

        exp = pd.DatetimeIndex(
            ["2011-01-01", "2011-01-02", "2012-01-01", "2012-01-02"], tz=tz
        )

        res = dti1.append(dti2)
        tm.assert_index_equal(res, exp)

        dts1 = Series(dti1)
        dts2 = Series(dti2)
        res = dts1._append_internal(dts2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([dts1, dts2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo", "EST5EDT"])
    def test_concatlike_datetimetz_short(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone names. Verifies that concatenating two frames with
        # tz-aware datetime indexes of the same zone keeps the zone and nanosecond unit,
        # with rows in input order (GH 7795).
        # GH#7795
        ix1 = pd.date_range(
            start="2014-07-15", end="2014-07-17", freq="D", tz=tz, unit="ns"
        )
        ix2 = pd.DatetimeIndex(["2014-07-11", "2014-07-21"], tz=tz)
        df1 = DataFrame(0, index=ix1, columns=["A", "B"])
        df2 = DataFrame(0, index=ix2, columns=["A", "B"])

        exp_idx = pd.DatetimeIndex(
            ["2014-07-15", "2014-07-16", "2014-07-17", "2014-07-11", "2014-07-21"],
            tz=tz,
        ).as_unit("ns")
        exp = DataFrame(0, index=exp_idx, columns=["A", "B"])

        tm.assert_frame_equal(pd.concat([df1, df2]), exp)

    def test_concatlike_datetimetz_to_object(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Parametrized by tz_aware_fixture. Verifies that appending a tz-aware datetime
        # index to a naive one, or to one with a different timezone, gives an object
        # index of Timestamps; same for series append and concatenate (GH 13660).
        tz = tz_aware_fixture
        # GH 13660

        # different tz coerces to object
        dti1 = pd.DatetimeIndex(["2011-01-01", "2011-01-02"], tz=tz)
        dti2 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"])

        exp = Index(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.Timestamp("2011-01-02", tz=tz),
                pd.Timestamp("2012-01-01"),
                pd.Timestamp("2012-01-02"),
            ],
            dtype=object,
        )

        res = dti1.append(dti2)
        tm.assert_index_equal(res, exp)

        dts1 = Series(dti1)
        dts2 = Series(dti2)
        res = dts1._append_internal(dts2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([dts1, dts2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # different tz
        tz_diff = zoneinfo.ZoneInfo("US/Hawaii")
        dti3 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"], tz=tz_diff)

        exp = Index(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.Timestamp("2011-01-02", tz=tz),
                pd.Timestamp("2012-01-01", tz=tz_diff),
                pd.Timestamp("2012-01-02", tz=tz_diff),
            ],
            dtype=object,
        )

        res = dti1.append(dti3)
        tm.assert_index_equal(res, exp)

        dts1 = Series(dti1)
        dts3 = Series(dti3)
        res = dts1._append_internal(dts3)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([dts1, dts3])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period(self):
        # [FUNCTION DOC]
        # Verifies that period indexes/series of the same frequency append and
        # concatenate to a period result (GH 13660).
        # GH 13660
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        pi2 = pd.PeriodIndex(["2012-01", "2012-02"], freq="M")

        exp = pd.PeriodIndex(["2011-01", "2011-02", "2012-01", "2012-02"], freq="M")

        res = pi1.append(pi2)
        tm.assert_index_equal(res, exp)

        ps1 = Series(pi1)
        ps2 = Series(pi2)
        res = ps1._append_internal(ps2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([ps1, ps2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period_diff_freq_to_object(self):
        # [FUNCTION DOC]
        # Verifies that period objects with different frequencies (monthly and daily)
        # append/concatenate to an object result holding Period elements (GH 13221).
        # GH 13221
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        pi2 = pd.PeriodIndex(["2012-01-01", "2012-02-01"], freq="D")

        exp = Index(
            [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Period("2012-01-01", freq="D"),
                pd.Period("2012-02-01", freq="D"),
            ],
            dtype=object,
        )

        res = pi1.append(pi2)
        tm.assert_index_equal(res, exp)

        ps1 = Series(pi1)
        ps2 = Series(pi2)
        res = ps1._append_internal(ps2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([ps1, ps2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period_mixed_dt_to_object(self):
        # [FUNCTION DOC]
        # Verifies that mixing periods with timedeltas, in both orders, gives an object
        # result for index append, series append and concatenate (GH 13221).
        # GH 13221
        # different datetimelike
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        tdi = pd.TimedeltaIndex(["1 days", "2 days"])
        exp = Index(
            [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
            ],
            dtype=object,
        )

        res = pi1.append(tdi)
        tm.assert_index_equal(res, exp)

        ps1 = Series(pi1)
        tds = Series(tdi)
        res = ps1._append_internal(tds)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([ps1, tds])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # inverse
        exp = Index(
            [
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
            ],
            dtype=object,
        )

        res = tdi.append(pi1)
        tm.assert_index_equal(res, exp)

        ps1 = Series(pi1)
        tds = Series(tdi)
        res = tds._append_internal(ps1)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([tds, ps1])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concat_categorical(self):
        # [FUNCTION DOC]
        # Verifies categorical concatenation: same categories keep categorical dtype;
        # partially or fully different categories (same underlying dtype) fall back to
        # the underlying dtype (integer or float when missing values exist), for both
        # concatenate and series append (GH 13524).
        # GH 13524

        # same categories -> category
        s1 = Series([1, 2, np.nan], dtype="category")
        s2 = Series([2, 1, 2], dtype="category")

        exp = Series([1, 2, np.nan, 2, 1, 2], dtype="category")
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

        # partially different categories => not-category
        s1 = Series([3, 2], dtype="category")
        s2 = Series([2, 1], dtype="category")

        exp = Series([3, 2, 2, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

        # completely different categories (same dtype) => not-category
        s1 = Series([10, 11, np.nan], dtype="category")
        s2 = Series([np.nan, 1, 3, 2], dtype="category")

        exp = Series([10, 11, np.nan, np.nan, 1, 3, 2], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

    def test_union_categorical_same_categories_different_order(self):
        # [FUNCTION DOC]
        # Verifies that concatenating categorical series having the same categories in a
        # different order keeps categorical dtype with the first series' category order
        # (GH 19096).
        # https://github.com/pandas-dev/pandas/issues/19096
        a = Series(Categorical(["a", "b", "c"], categories=["a", "b", "c"]))
        b = Series(Categorical(["a", "b", "c"], categories=["b", "a", "c"]))
        result = pd.concat([a, b], ignore_index=True)
        expected = Series(
            Categorical(["a", "b", "c", "a", "b", "c"], categories=["a", "b", "c"])
        )
        tm.assert_series_equal(result, expected)

    def test_concat_categorical_coercion(self):
        # [FUNCTION DOC]
        # Verifies that mixing a categorical series with a non-categorical one (numeric,
        # string, or all-missing) falls back to a non-categorical dtype, regardless of
        # operand order, for both concatenate and series append (GH 13524).
        # GH 13524

        # category + not-category => not-category
        s1 = Series([1, 2, np.nan], dtype="category")
        s2 = Series([2, 1, 2])

        exp = Series([1, 2, np.nan, 2, 1, 2], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

        # result shouldn't be affected by 1st elem dtype
        exp = Series([2, 1, 2, 1, 2, np.nan], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append_internal(s1, ignore_index=True), exp)

        # all values are not in category => not-category
        s1 = Series([3, 2], dtype="category")
        s2 = Series([2, 1])

        exp = Series([3, 2, 2, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

        exp = Series([2, 1, 3, 2])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append_internal(s1, ignore_index=True), exp)

        # completely different categories => not-category
        s1 = Series([10, 11, np.nan], dtype="category")
        s2 = Series([1, 3, 2])

        exp = Series([10, 11, np.nan, 1, 3, 2], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

        exp = Series([1, 3, 2, 10, 11, np.nan], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append_internal(s1, ignore_index=True), exp)

        # different dtype => not-category
        s1 = Series([10, 11, np.nan], dtype="category")
        s2 = Series(["a", "b", "c"])

        exp = Series([10, 11, np.nan, "a", "b", "c"])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

        exp = Series(["a", "b", "c", 10, 11, np.nan])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append_internal(s1, ignore_index=True), exp)

        # if normal series only contains NaN-likes => not-category
        s1 = Series([10, 11], dtype="category")
        s2 = Series([np.nan, np.nan, np.nan])

        exp = Series([10, 11, np.nan, np.nan, np.nan])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

        exp = Series([np.nan, np.nan, np.nan, 10, 11])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append_internal(s1, ignore_index=True), exp)

    def test_concat_categorical_3elem_coercion(self):
        # [FUNCTION DOC]
        # Verifies the same fallback rules with three inputs in different orders:
        # categorical, categorical and non-categorical combinations yield non-
        # categorical results of the expected values/dtype (GH 13524). The last two
        # sections repeat the same comment text for different data.
        # GH 13524

        # mixed dtypes => not-category
        s1 = Series([1, 2, np.nan], dtype="category")
        s2 = Series([2, 1, 2], dtype="category")
        s3 = Series([1, 2, 1, 2, np.nan])

        exp = Series([1, 2, np.nan, 2, 1, 2, 1, 2, 1, 2, np.nan], dtype="float")
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)

        exp = Series([1, 2, 1, 2, np.nan, 1, 2, np.nan, 2, 1, 2], dtype="float")
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)

        # values are all in either category => not-category
        s1 = Series([4, 5, 6], dtype="category")
        s2 = Series([1, 2, 3], dtype="category")
        s3 = Series([1, 3, 4])

        exp = Series([4, 5, 6, 1, 2, 3, 1, 3, 4])
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)

        exp = Series([1, 3, 4, 4, 5, 6, 1, 2, 3])
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)

        # values are all in either category => not-category
        s1 = Series([4, 5, 6], dtype="category")
        s2 = Series([1, 2, 3], dtype="category")
        s3 = Series([10, 11, 12])

        exp = Series([4, 5, 6, 1, 2, 3, 10, 11, 12])
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)

        exp = Series([10, 11, 12, 4, 5, 6, 1, 2, 3])
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)

    def test_concat_categorical_multi_coercion(self):
        # [FUNCTION DOC]
        # Verifies concatenating six series (mixed categorical and plain) falls back to
        # non-categorical values in both forward and reverse order (GH 13524).
        # GH 13524

        s1 = Series([1, 3], dtype="category")
        s2 = Series([3, 4], dtype="category")
        s3 = Series([2, 3])
        s4 = Series([2, 2], dtype="category")
        s5 = Series([1, np.nan])
        s6 = Series([1, 3, 2], dtype="category")

        # mixed dtype, values are all in categories => not-category
        exp = Series([1, 3, 3, 4, 2, 3, 2, 2, 1, np.nan, 1, 3, 2])
        res = pd.concat([s1, s2, s3, s4, s5, s6], ignore_index=True)
        tm.assert_series_equal(res, exp)

        exp = Series([1, 3, 2, 1, np.nan, 2, 2, 2, 3, 3, 4, 1, 3])
        res = pd.concat([s6, s5, s4, s3, s2, s1], ignore_index=True)
        tm.assert_series_equal(res, exp)

    def test_concat_categorical_ordered(self):
        # [FUNCTION DOC]
        # Verifies that concatenating ordered categoricals with identical categories
        # keeps ordered categorical dtype, also with three inputs (GH 13524).
        # GH 13524

        s1 = Series(Categorical([1, 2, np.nan], ordered=True))
        s2 = Series(Categorical([2, 1, 2], ordered=True))

        exp = Series(Categorical([1, 2, np.nan, 2, 1, 2], ordered=True))
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)

        exp = Series(Categorical([1, 2, np.nan, 2, 1, 2, 1, 2, np.nan], ordered=True))
        tm.assert_series_equal(pd.concat([s1, s2, s1], ignore_index=True), exp)

    def test_concat_categorical_coercion_nan(self):
        # [FUNCTION DOC]
        # Edge cases with all-missing categorical data: mixing with plain series gives a
        # non-categorical result, while two all-missing categoricals stay categorical
        # (GH 13524).
        # GH 13524

        # some edge cases
        # category + not-category => not category
        s1 = Series(np.array([np.nan, np.nan], dtype=np.float64), dtype="category")
        s2 = Series([np.nan, 1])

        exp = Series([np.nan, np.nan, np.nan, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)

        s1 = Series([1, np.nan], dtype="category")
        s2 = Series([np.nan, np.nan])

        exp = Series([1, np.nan, np.nan, np.nan], dtype="float")
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)

        # mixed dtype, all nan-likes => not-category
        s1 = Series([np.nan, np.nan], dtype="category")
        s2 = Series([np.nan, np.nan])

        exp = Series([np.nan, np.nan, np.nan, np.nan])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)

        # all category nan-likes => category
        s1 = Series([np.nan, np.nan], dtype="category")
        s2 = Series([np.nan, np.nan], dtype="category")

        exp = Series([np.nan, np.nan, np.nan, np.nan], dtype="category")

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)

    def test_concat_categorical_empty(self):
        # [FUNCTION DOC]
        # Verifies concatenation of empty categorical series: with a non-empty
        # categorical the result is converted to object, two empty categoricals stay
        # categorical, an empty categorical with an empty object series gives object,
        # and with an all-missing float series gives object (GH 13524).
        # GH 13524

        s1 = Series([], dtype="category")
        s2 = Series([1, 2], dtype="category")
        exp = s2.astype(object)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), exp)

        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append_internal(s1, ignore_index=True), exp)

        s1 = Series([], dtype="category")
        s2 = Series([], dtype="category")

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), s2)

        s1 = Series([], dtype="category")
        s2 = Series([], dtype="object")

        # different dtype => not-category
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        tm.assert_series_equal(s1._append_internal(s2, ignore_index=True), s2)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), s2)
        tm.assert_series_equal(s2._append_internal(s1, ignore_index=True), s2)

        s1 = Series([], dtype="category")
        s2 = Series([np.nan, np.nan])

        exp = Series([np.nan, np.nan], dtype=object)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)

        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)

    def test_categorical_concat_append(self):
        # [FUNCTION DOC]
        # Verifies that concatenating frames with a categorical column keeps the dtype
        # when categories match and that concatenating with a frame whose categorical
        # has extra categories yields an object column (GH 13524).
        cat = Categorical(["a", "b"], categories=["a", "b"])
        vals = [1, 2]
        df = DataFrame({"cats": cat, "vals": vals})
        cat2 = Categorical(["a", "b", "a", "b"], categories=["a", "b"])
        vals2 = [1, 2, 1, 2]
        exp = DataFrame({"cats": cat2, "vals": vals2}, index=Index([0, 1, 0, 1]))

        tm.assert_frame_equal(pd.concat([df, df]), exp)

        # GH 13524 can concat different categories
        cat3 = Categorical(["a", "b"], categories=["a", "b", "c"])
        vals3 = [1, 2]
        df_different_categories = DataFrame({"cats": cat3, "vals": vals3})

        res = pd.concat([df, df_different_categories], ignore_index=True)
        exp = DataFrame({"cats": list("abab"), "vals": [1, 2, 1, 2]})
        tm.assert_frame_equal(res, exp)
