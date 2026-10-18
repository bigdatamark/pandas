# [MODULE DOC]
# Tests for the DatetimeIndex constructor and the base Index constructor when given datetime
# data: timezone handling (explicit tz, dtype, mixed zones, fixed offsets, DST edge cases),
# frequency validation, unit and out-of-bounds handling, integer and float inputs, parsing
# options (dayfirst, yearfirst), copy semantics and error cases. Uses the shared timezone fixtures.
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from functools import partial
from operator import attrgetter
import zoneinfo

import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest

from pandas._libs.tslibs import (
    astype_overflowsafe,
    timezones,
)

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    offsets,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import period_array


class TestDatetimeIndex:
    # [CLASS DOC]
    # Main group of DatetimeIndex constructor tests; no state.
    def test_from_dt64_unsupported_unit(self):
        # [FUNCTION DOC]
        # Verifies a datetime64 scalar in an unsupported unit (days) with a timezone is converted to the
        # nearest supported unit (seconds) in the result.
        # GH#49292
        val = np.datetime64(1, "D")
        result = DatetimeIndex([val], tz="US/Pacific")

        expected = DatetimeIndex([val.astype("M8[s]")], tz="US/Pacific")
        tm.assert_index_equal(result, expected)

    def test_explicit_tz_none(self):
        # [FUNCTION DOC]
        # Verifies passing tz as None with timezone-aware data raises a value error (data incompatible
        # with tz none), for index and numpy array inputs, and passing a timezone-aware dtype together
        # with tz None raises a value error.
        # GH#48659
        dti = date_range("2016-01-01", periods=10, tz="UTC", unit="ns")

        msg = "Passed data is timezone-aware, incompatible with 'tz=None'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(dti, tz=None)

        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(np.array(dti), tz=None)

        msg = "Cannot pass both a timezone-aware dtype and tz=None"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([], dtype="M8[ns, UTC]", tz=None)

    def test_freq_validation_with_nat(self):
        # [FUNCTION DOC]
        # Verifies an explicit freq conflicting with data that contains NaT raises a value error that the
        # inferred frequency (none) does not conform to the passed frequency, for Timestamps and
        # integers.
        # GH#11587 make sure we get a useful error message when generate_range
        #  raises
        msg = (
            "Inferred frequency None from passed values does not conform "
            "to passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([pd.NaT, Timestamp("2011-01-01")], freq="D")
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([pd.NaT, Timestamp("2011-01-01")._value], freq="D")

    # TODO: better place for tests shared by DTI/TDI?
    @pytest.mark.parametrize(
        "index",
        [
            date_range("2016-01-01", periods=5, tz="US/Pacific", unit="ns"),
            pd.timedelta_range("1 Day", periods=5),
        ],
    )
    def test_shallow_copy_inherits_array_freq(self, index):
        # [FUNCTION DOC]
        # Parametrized over a US/Pacific DatetimeIndex and a TimedeltaIndex (a TODO notes a better place
        # may exist for tests shared with TimedeltaIndex). Verifies the private shallow-copy helper,
        # given an array whose freq is none (from taking elements out of order), yields an index with
        # freq none rather than inheriting the original index freq.
        # If we pass a DTA/TDA to shallow_copy and dont specify a freq,
        #  we should inherit the array's freq, not our own.
        array = index._data

        arr = array[[0, 3, 2, 4, 1]]
        assert arr.freq is None

        result = index._shallow_copy(arr)
        assert result.freq is None

    def test_categorical_preserves_tz(self):
        # [FUNCTION DOC]
        # Verifies a timezone-aware DatetimeIndex round trips through CategoricalIndex, Categorical and
        # Series of categoricals back to a DatetimeIndex equal to the original (timezone kept), for both
        # index and array sources.
        # GH#18664 retain tz when going DTI-->Categorical-->DTI
        dti = DatetimeIndex(
            [pd.NaT, "2015-01-01", "1999-04-06 15:14:13", "2015-01-01"], tz="US/Eastern"
        )

        for dtobj in [dti, dti._data]:
            # works for DatetimeIndex or DatetimeArray

            ci = pd.CategoricalIndex(dtobj)
            carr = pd.Categorical(dtobj)
            cser = pd.Series(ci)

            for obj in [ci, carr, cser]:
                result = DatetimeIndex(obj)
                tm.assert_index_equal(result, dti)

    def test_dti_with_period_data_raises(self):
        # [FUNCTION DOC]
        # Verifies constructing a DatetimeIndex or calling to_datetime on period data (index or array)
        # raises a type error saying period data is invalid.
        # GH#23675
        data = pd.PeriodIndex(["2016Q1", "2016Q2"], freq="Q")

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            DatetimeIndex(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            to_datetime(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            DatetimeIndex(period_array(data))

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            to_datetime(period_array(data))

    def test_dti_with_timedelta64_data_raises(self):
        # [FUNCTION DOC]
        # Verifies constructing a DatetimeIndex or calling to_datetime on timedelta data (array or index)
        # raises a type error saying timedelta cannot be converted to datetime.
        # GH#23675 deprecated, enforced in GH#29794
        data = np.array([0], dtype="m8[ns]")
        msg = r"timedelta64\[ns\] cannot be converted to datetime64"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(data)

        with pytest.raises(TypeError, match=msg):
            to_datetime(data)

        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(pd.TimedeltaIndex(data))

        with pytest.raises(TypeError, match=msg):
            to_datetime(pd.TimedeltaIndex(data))

    def test_constructor_from_sparse_array(self):
        # [FUNCTION DOC]
        # Regression test: Index built from a sparse array of Timestamps stays a base Index with the
        # sparse dtype.
        # https://github.com/pandas-dev/pandas/issues/35843
        values = [
            Timestamp("2012-05-01T01:00:00.000000"),
            Timestamp("2016-05-01T01:00:00.000000"),
        ]
        arr = pd.arrays.SparseArray(values)
        result = Index(arr)
        assert type(result) is Index
        assert result.dtype == arr.dtype

    def test_construction_caching(self):
        # [FUNCTION DOC]
        # Regression test: building a frame with naive, timezone-aware, null-containing and
        # nanosecond-frequency datetime columns preserves the zoneinfo key of the aware column.
        df = pd.DataFrame(
            {
                "dt": date_range("20130101", periods=3, unit="ns"),
                "dttz": date_range(
                    "20130101", periods=3, tz=zoneinfo.ZoneInfo("US/Eastern"), unit="ns"
                ),
                "dt_with_null": [
                    Timestamp("20130101"),
                    pd.NaT,
                    Timestamp("20130103"),
                ],
                "dtns": date_range("20130101", periods=3, freq="ns", unit="ns"),
            }
        )
        assert df.dttz.dtype.tz.key == "US/Eastern"

    @pytest.mark.parametrize(
        "kwargs",
        [{"tz": "dtype.tz"}, {"dtype": "dtype"}, {"dtype": "dtype", "tz": "dtype.tz"}],
    )
    def test_construction_with_alt(self, kwargs, tz_aware_fixture):
        # [FUNCTION DOC]
        # Parametrized over keyword sets (tz from the index, dtype from the index, both) and
        # tz_aware_fixture. Verifies re-constructing from an index with those keywords yields an equal index.
        tz = tz_aware_fixture
        i = date_range("20130101", periods=5, freq="h", tz=tz, unit="ns")
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
        result = DatetimeIndex(i, **kwargs)
        tm.assert_index_equal(i, result)

    @pytest.mark.parametrize(
        "kwargs",
        [{"tz": "dtype.tz"}, {"dtype": "dtype"}, {"dtype": "dtype", "tz": "dtype.tz"}],
    )
    def test_construction_with_alt_tz_localize(self, kwargs, tz_aware_fixture):
        # [FUNCTION DOC]
        # Same parametrization as the previous test with freq removed. Verifies constructing from UTC
        # integers with a timezone equals converting; localizing into a given timezone; and that passing
        # both a tz and a conflicting tz-aware dtype raises a value error.
        tz = tz_aware_fixture
        i = date_range("20130101", periods=5, freq="h", tz=tz, unit="ns")
        i = i._with_freq(None)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}

        if "tz" in kwargs:
            result = DatetimeIndex(i.asi8, tz="UTC").tz_convert(kwargs["tz"])

            expected = DatetimeIndex(i, **kwargs)
            tm.assert_index_equal(result, expected)

        # localize into the provided tz
        i2 = DatetimeIndex(i.tz_localize(None).asi8, tz="UTC")
        expected = i.tz_localize(None).tz_localize("UTC")
        tm.assert_index_equal(i2, expected)

        # incompat tz/dtype
        msg = "cannot supply both a tz and a dtype with a tz"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(
                i.tz_localize(None).asi8,
                dtype=i.dtype,
                tz=zoneinfo.ZoneInfo("US/Hawaii"),
            )

    def test_construction_index_with_mixed_timezones(self):
        # [FUNCTION DOC]
        # Verifies base Index construction from Timestamps: no timezone gives a DatetimeIndex; the same
        # timezone (also across DST, and length one) gives a timezone-aware DatetimeIndex; mixed naive and
        # aware or different timezones give an object Index; and constructing a DatetimeIndex from strings
        # with different offsets raises a value error about mixed timezones.
        # gh-11488: no tz results in DatetimeIndex
        result = Index([Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx")
        exp = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # same tz results in DatetimeIndex
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-02 10:00")],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # same tz results in DatetimeIndex (DST)
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # Different tz results in Index(dtype=object)
        result = Index(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2013-11-02 22:00-05:00", "2013-11-03 22:00-06:00"])

        # length = 1
        result = Index([Timestamp("2011-01-01")], name="idx")
        exp = DatetimeIndex([Timestamp("2011-01-01")], name="idx")
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # length = 1 with tz
        result = Index([Timestamp("2011-01-01 10:00", tz="Asia/Tokyo")], name="idx")
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00")], tz="Asia/Tokyo", name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

    def test_construction_index_with_mixed_timezones_with_NaT(self):
        # [FUNCTION DOC]
        # Same as the previous test with NaT present in the data: same-zone data with NaT gives a
        # timezone-aware DatetimeIndex; mixed zones give object dtype; all NaT gives a naive DatetimeIndex.
        # see gh-11488
        result = Index(
            [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
            name="idx",
        )
        exp = DatetimeIndex(
            [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # Same tz results in DatetimeIndex
        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00"),
            ],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # same tz results in DatetimeIndex (DST)
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                pd.NaT,
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), pd.NaT, Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # different tz results in Index(dtype=object)
        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        # all NaT
        result = Index([pd.NaT, pd.NaT], name="idx")
        exp = DatetimeIndex([pd.NaT, pd.NaT], name="idx")
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

    def test_construction_dti_with_mixed_timezones(self):
        # [FUNCTION DOC]
        # Verifies the DatetimeIndex constructor with mixed timezones: same-zone and naive data are fine;
        # different-zone Timestamps without tz argument raise a value error (cannot be converted to
        # datetime64); with a tz argument the values are converted pointwise (naive treated as wall time
        # in the requested zone), also when given as a dtype.
        # GH 11488 (not changed, added explicit tests)

        # no tz results in DatetimeIndex
        result = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # same tz results in DatetimeIndex
        result = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-02 10:00")],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # same tz results in DatetimeIndex (DST)
        result = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # tz mismatch affecting to tz-aware raises TypeError/ValueError

        msg = "cannot be converted to datetime64"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(
                [
                    Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )

        # pre-2.0 this raised bc of awareness mismatch. in 2.0 with a tz#
        #  specified we behave as if this was called pointwise, so
        #  the naive Timestamp is treated as a wall time.
        dti = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            tz="Asia/Tokyo",
            name="idx",
        )
        expected = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern").tz_convert("Asia/Tokyo"),
            ],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(dti, expected)

        # pre-2.0 mixed-tz scalars raised even if a tz/dtype was specified.
        #  as of 2.0 we successfully return the requested tz/dtype
        dti = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            tz="US/Eastern",
            name="idx",
        )
        expected = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo").tz_convert("US/Eastern"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(dti, expected)

        # same thing but pass dtype instead of tz
        dti = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="M8[us, US/Eastern]",
            name="idx",
        )
        tm.assert_index_equal(dti, expected)

    def test_construction_base_constructor(self):
        # [FUNCTION DOC]
        # Verifies Index and DatetimeIndex agree when built from lists or arrays of Timestamps and NaT or
        # NaN.
        arr = [Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-03")]
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

        arr = [np.nan, pd.NaT, Timestamp("2011-01-03")]
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

    def test_construction_outofbounds(self):
        # [FUNCTION DOC]
        # Regression test: Index from python datetimes in years 3000 to 6000 gives microsecond
        # datetime64 (beyond nanosecond bounds) and the DatetimeIndex constructor accepts them.
        # GH 13663
        dates = [
            datetime(3000, 1, 1),
            datetime(4000, 1, 1),
            datetime(5000, 1, 1),
            datetime(6000, 1, 1),
        ]
        exp = Index(dates, dtype="M8[us]")
        res = Index(dates)
        tm.assert_index_equal(res, exp)

        DatetimeIndex(dates)

    @pytest.mark.parametrize("data", [["1400-01-01"], [datetime(1400, 1, 1)]])
    def test_dti_date_out_of_range(self, data):
        # [FUNCTION DOC]
        # Parametrized over a date string and a datetime from the year 1400. Verifies construction does not
        # raise.
        # GH#1475
        DatetimeIndex(data)

    def test_construction_with_ndarray(self):
        # [FUNCTION DOC]
        # Verifies constructing from a raw numpy datetime array with a business-day freq gives the expected
        # microsecond index with that freq.
        # GH 5152
        dates = [datetime(2013, 10, 7), datetime(2013, 10, 8), datetime(2013, 10, 9)]
        data = DatetimeIndex(dates, freq=offsets.BDay()).values
        result = DatetimeIndex(data, freq=offsets.BDay())
        expected = DatetimeIndex(
            ["2013-10-07", "2013-10-08", "2013-10-09"], dtype="M8[us]", freq="B"
        )
        tm.assert_index_equal(result, expected)

    def test_integer_values_and_tz_interpreted_as_utc(self):
        # [FUNCTION DOC]
        # Regression test: integer values interpreted as UTC wall values when a timezone is given;
        # localizing to US/Central works, and passing tz UTC directly produces no warning.
        # GH-24559
        val = np.datetime64("2000-01-01 00:00:00", "ns")
        values = np.array([val.view("i8")])

        result = DatetimeIndex(values).tz_localize("US/Central")

        expected = DatetimeIndex(["2000-01-01T00:00:00"], dtype="M8[ns, US/Central]")
        tm.assert_index_equal(result, expected)

        # but UTC is *not* deprecated.
        with tm.assert_produces_warning(None):
            result = DatetimeIndex(values, tz="UTC")
        expected = DatetimeIndex(["2000-01-01T00:00:00"], dtype="M8[ns, UTC]")
        tm.assert_index_equal(result, expected)

    def test_constructor_coverage(self):
        # [FUNCTION DOC]
        # Verifies a scalar string raises a type error requiring a collection; a generator expression and
        # numpy string arrays (with and without NaT) are accepted, matching lists of objects and integers;
        # and values not conforming to a daily freq raise a value error.
        msg = r"DatetimeIndex\(\.\.\.\) must be called with a collection"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex("1/1/2000")

        # generator expression
        gen = (datetime(2000, 1, 1) + timedelta(i) for i in range(10))
        result = DatetimeIndex(gen)
        expected = DatetimeIndex(
            [datetime(2000, 1, 1) + timedelta(i) for i in range(10)]
        )
        tm.assert_index_equal(result, expected)

        # NumPy string array
        strings = np.array(["2000-01-01", "2000-01-02", "2000-01-03"])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype("O"))
        tm.assert_index_equal(result, expected)

        from_ints = DatetimeIndex(expected.as_unit("ns").asi8).as_unit("us")
        tm.assert_index_equal(from_ints, expected)

        # string with NaT
        strings = np.array(["2000-01-01", "2000-01-02", "NaT"])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype("O"))
        tm.assert_index_equal(result, expected)

        from_ints = DatetimeIndex(expected.as_unit("ns").asi8).as_unit("us")
        tm.assert_index_equal(from_ints, expected)

        # non-conforming
        msg = (
            "Inferred frequency None from passed values does not conform "
            "to passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"], freq="D")

    @pytest.mark.parametrize("freq", ["YS", "W-SUN"])
    def test_constructor_datetime64_tzformat(self, freq):
        # [FUNCTION DOC]
        # Parametrized over YS and weekly Sunday freq. Verifies ISO-style and non-ISO date strings with
        # UTC offsets create ranges whose timezone is a fixed offset, matching the equivalent ranges
        # and integer values of Lima and Tokyo ranges.
        # see GH#6572: ISO 8601 format results in stdlib timezone object
        idx = date_range(
            "2013-01-01T00:00:00-05:00", "2016-01-01T23:59:59-05:00", freq=freq
        )
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=-300)),
        )
        tm.assert_index_equal(idx, expected)
        # Unable to use `US/Eastern` because of DST
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="America/Lima"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range(
            "2013-01-01T00:00:00+09:00", "2016-01-01T23:59:59+09:00", freq=freq
        )
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=540)),
        )
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="Asia/Tokyo"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        # Non ISO 8601 format results in dateutil.tz.tzoffset
        idx = date_range("2013/1/1 0:00:00-5:00", "2016/1/1 23:59:59-5:00", freq=freq)
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=-300)),
        )
        tm.assert_index_equal(idx, expected)
        # Unable to use `US/Eastern` because of DST
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="America/Lima"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range("2013/1/1 0:00:00+9:00", "2016/1/1 23:59:59+09:00", freq=freq)
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=540)),
        )
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="Asia/Tokyo"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

    def test_constructor_dtype(self):
        # [FUNCTION DOC]
        # Verifies a timezone-aware dtype with naive strings localizes them, same as passing tz.
        # passing a dtype with a tz should localize
        idx = DatetimeIndex(
            ["2013-01-01", "2013-01-02"], dtype="datetime64[ns, US/Eastern]"
        )
        expected = (
            DatetimeIndex(["2013-01-01", "2013-01-02"])
            .as_unit("ns")
            .tz_localize("US/Eastern")
        )
        tm.assert_index_equal(idx, expected)

        idx = DatetimeIndex(["2013-01-01", "2013-01-02"], tz="US/Eastern").as_unit("ns")
        tm.assert_index_equal(idx, expected)

    def test_constructor_dtype_tz_mismatch_raises(self):
        # [FUNCTION DOC]
        # Verifies conflicts: a naive dtype or conflicting tz on a timezone-aware index raises value
        # errors (cannot supply both) or a type error (data already aware in another zone); the same dtype
        # is accepted.
        # if we already have a tz and its not the same, then raise
        idx = DatetimeIndex(
            ["2013-01-01", "2013-01-02"], dtype="datetime64[ns, US/Eastern]"
        )

        msg = (
            "cannot supply both a tz and a timezone-naive dtype "
            r"\(i\.e\. datetime64\[ns\]\)"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(idx, dtype="datetime64[ns]")

        # this is effectively trying to convert tz's
        msg = "data is already tz-aware US/Eastern, unable to set specified tz: CET"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(idx, dtype="datetime64[ns, CET]")
        msg = "cannot supply both a tz and a dtype with a tz"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(idx, tz="CET", dtype="datetime64[ns, US/Eastern]")

        result = DatetimeIndex(idx, dtype="datetime64[ns, US/Eastern]")
        tm.assert_index_equal(idx, result)

    @pytest.mark.parametrize("dtype", [object, np.int32, np.int64])
    def test_constructor_invalid_dtype_raises(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over object, int32, int64 dtype. Verifies integer data with an invalid dtype raises
        # a value error saying the dtype value is unexpected.
        # GH 23986
        msg = "Unexpected value for 'dtype'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([1, 2], dtype=dtype)

    def test_000constructor_resolution(self):
        # [FUNCTION DOC]
        # Regression test: nanosecond component is preserved when building an index from a Timestamp with
        # nanoseconds.
        # 2252
        t1 = Timestamp((1352934390 * 1000000000) + 1000000 + 1000 + 1)
        idx = DatetimeIndex([t1])

        assert idx.nanosecond[0] == t1.nanosecond

    def test_disallow_setting_tz(self):
        # [FUNCTION DOC]
        # Verifies assigning to the tz attribute raises an attribute error saying the timezone cannot be set
        # directly.
        # GH 3746
        dti = DatetimeIndex(["2010"], tz="UTC")
        msg = "Cannot directly set timezone"
        with pytest.raises(AttributeError, match=msg):
            dti.tz = zoneinfo.ZoneInfo("US/Pacific")

    @pytest.mark.parametrize(
        "tz",
        [
            None,
            "America/Los_Angeles",
            zoneinfo.ZoneInfo("America/Los_Angeles"),
            Timestamp("2000", tz="America/Los_Angeles").tz,
        ],
    )
    def test_constructor_start_end_with_tz(self, tz):
        # [FUNCTION DOC]
        # Parametrized over tz given as none, name, zoneinfo object, and Timestamp-derived tz. Regression
        # test: date_range with timezone-aware start and end uses the zoneinfo timezone consistently
        # (identity check).
        # GH 18595
        start = Timestamp("2013-01-01 06:00:00", tz="America/Los_Angeles")
        end = Timestamp("2013-01-02 06:00:00", tz="America/Los_Angeles")
        result = date_range(freq="D", start=start, end=end, tz=tz, unit="ns")
        expected = DatetimeIndex(
            ["2013-01-01 06:00:00", "2013-01-02 06:00:00"],
            dtype="M8[ns, America/Los_Angeles]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)
        # Especially assert that the timezone is consistent for zoneinfo
        assert zoneinfo.ZoneInfo("America/Los_Angeles") is result.tz

    @pytest.mark.parametrize("tz", ["US/Pacific", "US/Eastern", "Asia/Tokyo"])
    def test_constructor_with_non_normalized_pytz(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezones. Skipped when pytz is unavailable. Verifies constructing with a
        # non-normalized pytz timezone gives the normalized zone object.
        # GH 18595
        pytz = pytest.importorskip("pytz")
        tz_in = pytz.timezone(tz)
        non_norm_tz = Timestamp("2010", tz=tz_in).tz
        result = DatetimeIndex(["2010"], tz=non_norm_tz)
        assert pytz.timezone(tz) is result.tz

    def test_constructor_timestamp_near_dst(self):
        # [FUNCTION DOC]
        # Verifies constructing from two Helsinki Timestamps at the ambiguous hour of the DST end equals
        # constructing from their python datetimes.
        # GH 20854
        ts = [
            Timestamp("2016-10-30 03:00:00+0300", tz="Europe/Helsinki"),
            Timestamp("2016-10-30 03:00:00+0200", tz="Europe/Helsinki"),
        ]
        result = DatetimeIndex(ts).as_unit("ns")
        expected = DatetimeIndex(
            [ts[0].to_pydatetime(), ts[1].to_pydatetime()]
        ).as_unit("ns")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    @pytest.mark.parametrize("box", [np.array, partial(np.array, dtype=object), list])
    @pytest.mark.parametrize(
        "tz, dtype",
        [("US/Pacific", "datetime64[ns, US/Pacific]"), (None, "datetime64[ns]")],
    )
    def test_constructor_with_int_tz(self, klass, box, tz, dtype):
        # [FUNCTION DOC]
        # Parametrized over Index/DatetimeIndex, array/object-array/list boxes, and (tz, dtype). Verifies
        # constructing from integer nanosecond values with a datetime dtype equals constructing from
        # the Timestamp.
        # GH 20997, 20964
        ts = Timestamp("2018-01-01", tz=tz).as_unit("ns")
        result = klass(box([ts._value]), dtype=dtype)
        expected = klass([ts])
        assert result == expected

    def test_construction_int_rountrip(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Uses tz_naive_fixture. Verifies an integer constant equals the integer representation of the
        # stamp after constructing an index from it.
        # GH 12619, GH#24559
        tz = tz_naive_fixture

        result = 1293858000000000000
        expected = DatetimeIndex([result], tz=tz).asi8[0]
        assert result == expected

    def test_construction_from_replaced_timestamps_with_dst(self):
        # [FUNCTION DOC]
        # Regression test: constructing from Timestamps in Melbourne time that were modified with replace
        # across DST gives the expected wall times.
        # GH 18785
        index = date_range(
            Timestamp(2000, 12, 31),
            Timestamp(2005, 12, 31),
            freq="YE-DEC",
            tz="Australia/Melbourne",
            unit="ns",
        )
        result = DatetimeIndex([x.replace(month=6, day=1) for x in index])
        expected = DatetimeIndex(
            [
                "2000-06-01 00:00:00",
                "2001-06-01 00:00:00",
                "2002-06-01 00:00:00",
                "2003-06-01 00:00:00",
                "2004-06-01 00:00:00",
                "2005-06-01 00:00:00",
            ],
            tz="Australia/Melbourne",
        ).as_unit("ns")
        tm.assert_index_equal(result, expected)

    def test_construction_with_tz_and_tz_aware_dti(self):
        # [FUNCTION DOC]
        # Verifies passing a different tz with a timezone-aware index raises a type error saying the data is
        # already tz-aware.
        # GH 23579
        dti = date_range("2016-01-01", periods=3, tz="US/Central")
        msg = "data is already tz-aware US/Central, unable to set specified tz"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(dti, tz="Asia/Tokyo")

    def test_construction_with_nat_and_tzlocal(self):
        # [FUNCTION DOC]
        # Verifies strings and NaT with local timezone equal the Timestamp-based construction.
        tz = dateutil.tz.tzlocal()
        result = DatetimeIndex(["2018", "NaT"], tz=tz).as_unit("ns")
        expected = DatetimeIndex([Timestamp("2018", tz=tz), pd.NaT]).as_unit("ns")
        tm.assert_index_equal(result, expected)

    def test_constructor_with_ambiguous_keyword_arg(self):
        # [FUNCTION DOC]
        # Regression test: date_range accepts the ambiguous keyword when the start or the end is an
        # ambiguous New York time, producing the expected range.
        # GH 35297

        expected = DatetimeIndex(
            ["2020-11-01 01:00:00", "2020-11-02 01:00:00"],
            dtype="datetime64[ns, America/New_York]",
            freq="D",
            ambiguous=False,
        )

        # ambiguous keyword in start
        timezone = "America/New_York"
        start = Timestamp(year=2020, month=11, day=1, hour=1).tz_localize(
            timezone, ambiguous=False
        )
        result = date_range(start=start, periods=2, ambiguous=False, unit="ns")
        tm.assert_index_equal(result, expected)

        # ambiguous keyword in end
        timezone = "America/New_York"
        end = Timestamp(year=2020, month=11, day=2, hour=1).tz_localize(
            timezone, ambiguous=False
        )
        result = date_range(end=end, periods=2, ambiguous=False, unit="ns")
        tm.assert_index_equal(result, expected)

    def test_constructor_with_nonexistent_keyword_arg(self, warsaw):
        # [FUNCTION DOC]
        # Uses the warsaw timezone fixture. Regression test: date_range with a nonexistent start (shifted
        # forward) or end gives the expected hourly stamps around the gap.
        # GH 35297
        timezone = warsaw

        # nonexistent keyword in start
        start = Timestamp("2015-03-29 02:30:00").tz_localize(
            timezone, nonexistent="shift_forward"
        )
        result = date_range(start=start, periods=2, freq="h", unit="ns")
        expected = DatetimeIndex(
            [
                Timestamp("2015-03-29 03:00:00+02:00", tz=timezone),
                Timestamp("2015-03-29 04:00:00+02:00", tz=timezone),
            ]
        ).as_unit("ns")

        tm.assert_index_equal(result, expected)

        # nonexistent keyword in end
        end = start
        result = date_range(end=end, periods=2, freq="h", unit="ns")
        expected = DatetimeIndex(
            [
                Timestamp("2015-03-29 01:00:00+01:00", tz=timezone),
                Timestamp("2015-03-29 03:00:00+02:00", tz=timezone),
            ]
        ).as_unit("ns")

        tm.assert_index_equal(result, expected)

    def test_constructor_no_precision_raises(self):
        # [FUNCTION DOC]
        # Verifies datetime64 with no precision raises value errors (not allowed for the index constructor;
        # dtype has no unit for the base Index).
        # GH-24753, GH-24739

        msg = "with no precision is not allowed"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000"], dtype="datetime64")

        msg = "The 'datetime64' dtype has no unit. Please pass in"
        with pytest.raises(ValueError, match=msg):
            Index(["2000"], dtype="datetime64")

    def test_constructor_wrong_precision_raises(self):
        # [FUNCTION DOC]
        # Verifies that a microsecond dtype is honored (dtype and value), despite the test name suggesting
        # an error.
        # KNOWN ISSUE: the test name says the constructor raises for the wrong precision, but the body
        # asserts that construction succeeds and keeps the microsecond unit; the name is misleading.
        dti = DatetimeIndex(["2000"], dtype="datetime64[us]")
        assert dti.dtype == "M8[us]"
        assert dti[0] == Timestamp(2000, 1, 1)

    def test_index_constructor_with_numpy_object_array_and_timestamp_tz_with_nan(self):
        # [FUNCTION DOC]
        # Regression test: Index from an object array holding a UTC Timestamp and NaN gives a timezone-aware
        # DatetimeIndex with NaT.
        # GH 27011
        result = Index(np.array([Timestamp("2019", tz="UTC"), np.nan], dtype=object))
        expected = DatetimeIndex([Timestamp("2019", tz="UTC"), pd.NaT])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "tz", [zoneinfo.ZoneInfo("US/Eastern"), gettz("US/Eastern")]
    )
    def test_dti_from_tzaware_datetime(self, tz):
        # [FUNCTION DOC]
        # Parametrized over zoneinfo and dateutil Eastern timezones. Verifies a datetime with tzinfo yields
        # an index with an equal timezone.
        d = [datetime(2012, 8, 19, tzinfo=tz)]

        index = DatetimeIndex(d)
        assert timezones.tz_compare(index.tz, tz)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_constructors(self, tzstr):
        # [FUNCTION DOC]
        # Parametrized over timezone spelling. Verifies localization, date_range, string-list and
        # numpy-array construction with tz all give equal indexes.
        """Test different DatetimeIndex constructions with timezone
        Follow-up of GH#4229
        """
        arr = ["11/10/2005 08:00:00", "11/10/2005 09:00:00"]

        idx1 = to_datetime(arr).tz_localize(tzstr)
        idx2 = date_range(
            start="2005-11-10 08:00:00", freq="h", periods=2, tz=tzstr, unit="us"
        )
        idx2 = idx2._with_freq(None)  # the others all have freq=None
        idx3 = DatetimeIndex(arr, tz=tzstr)
        idx4 = DatetimeIndex(np.array(arr), tz=tzstr)

        tm.assert_index_equal(idx1, idx2)
        tm.assert_index_equal(idx1, idx3)
        tm.assert_index_equal(idx1, idx4)

    def test_dti_construction_idempotent(self, unit):
        # [FUNCTION DOC]
        # Uses unit. Verifies constructing from a timezone-aware index with the same tz returns an equal
        # index.
        rng = date_range(
            "03/12/2012 00:00", periods=10, freq="W-FRI", tz="US/Eastern", unit=unit
        )
        rng2 = DatetimeIndex(data=rng, tz="US/Eastern")
        tm.assert_index_equal(rng, rng2)

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_constructor_static_tzinfo(self, prefix):
        # [FUNCTION DOC]
        # Parametrized over name prefix. Smoke test: a fixed-offset static timezone (EST) works for
        # construction, hour access and element access; there are no assertions.
        # it works!
        index = DatetimeIndex([datetime(2012, 1, 1)], tz=prefix + "EST")
        index.hour
        index[0]

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_convert_datetime_list(self, tzstr):
        # [FUNCTION DOC]
        # Parametrized over timezone spelling. Verifies a list of Timestamps with explicit daily freq
        # round trips to an equal index.
        dr = date_range("2012-06-02", periods=10, tz=tzstr, name="foo")
        dr2 = DatetimeIndex(list(dr), name="foo", freq="D")
        tm.assert_index_equal(dr, dr2)

    @pytest.mark.parametrize(
        "tz",
        [
            "pytz/US/Eastern",
            gettz("US/Eastern"),
        ],
    )
    @pytest.mark.parametrize("use_str", [True, False])
    @pytest.mark.parametrize("box_cls", [Timestamp, DatetimeIndex])
    def test_dti_ambiguous_matches_timestamp(self, tz, use_str, box_cls, request):
        # [FUNCTION DOC]
        # Parametrized over timezones (pytz, dateutil), string versus datetime input, and Timestamp
        # versus DatetimeIndex target; uses request to mark expected failures. Verifies construction at an
        # ambiguous time raises a value error mentioning the string, consistently between Timestamp and
        # index. When a datetime input is combined with a dateutil timezone the case is marked as expected
        # failure since the Timestamp constructor implicitly uses a fold of zero.
        # GH#47471 check that we get the same raising behavior in the DTI
        # constructor and Timestamp constructor
        if isinstance(tz, str) and tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        dtstr = "2013-11-03 01:59:59.999999"
        item = dtstr
        if not use_str:
            item = Timestamp(dtstr).to_pydatetime()
        if box_cls is not Timestamp:
            item = [item]

        if not use_str and isinstance(tz, dateutil.tz.tzfile):
            # FIXME: The Timestamp constructor here behaves differently than all
            #  the other cases bc with dateutil/zoneinfo tzinfos we implicitly
            #  get fold=0. Having this raise is not important, but having the
            #  behavior be consistent across cases is.
            mark = pytest.mark.xfail(reason="We implicitly get fold=0.")
            request.applymarker(mark)

        with pytest.raises(ValueError, match=dtstr):
            box_cls(item, tz=tz)

    @pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
    def test_dti_constructor_with_non_nano_dtype(self, tz):
        # [FUNCTION DOC]
        # Parametrized over tz (none, UTC, US/Pacific). Verifies a mixed list of a Timestamp, a string
        # and an integer with a microsecond dtype is parsed pointwise (integer treated as microseconds)
        # and equals the expected index, also for an object array input.
        # GH#55756, GH#54620
        ts = Timestamp("2999-01-01")
        dtype = "M8[us]"
        if tz is not None:
            dtype = f"M8[us, {tz}]"
        vals = [ts, "2999-01-02 03:04:05.678910", 2500]
        result = DatetimeIndex(vals, dtype=dtype)
        # The 2500 is interpreted as microseconds, consistent with what
        #  we would get if we created DatetimeIndexes from vals[:2] and vals[2:]
        #  and concated the results.
        pointwise = [
            vals[0].tz_localize(tz),
            Timestamp(vals[1], tz=tz),
            to_datetime(vals[2], unit="us", utc=True).tz_convert(tz),
        ]
        exp_vals = [x.as_unit("us").asm8 for x in pointwise]
        exp_arr = np.array(exp_vals, dtype="M8[us]")
        expected = DatetimeIndex(exp_arr, dtype="M8[us]")
        if tz is not None:
            expected = expected.tz_localize("UTC").tz_convert(tz)
        tm.assert_index_equal(result, expected)

        result2 = DatetimeIndex(np.array(vals, dtype=object), dtype=dtype)
        tm.assert_index_equal(result2, expected)

    def test_dti_constructor_with_non_nano_now_today(self, request):
        # [FUNCTION DOC]
        # Uses request to mark a flaky tolerance check. Verifies the strings now and today with second
        # unit produce second-unit dtype and values not earlier than the time before construction; the
        # final tolerance assertions (within one second) are allowed to fail without failing the test.
        # KNOWN ISSUE: the failure messages of the assertions on the second difference (today) report
        # the first difference (now) because they use the wrong variable, so they are misleading
        # when the second difference is the one that fails.
        # GH#55756
        now = Timestamp.now()
        today = Timestamp.today()
        result = DatetimeIndex(["now", "today"], dtype="M8[s]")
        assert result.dtype == "M8[s]"

        diff0 = result[0] - now.as_unit("s")
        diff1 = result[1] - today.as_unit("s")
        assert diff1 >= pd.Timedelta(0), f"The difference is {diff0}"
        assert diff0 >= pd.Timedelta(0), f"The difference is {diff0}"

        # result may not exactly match [now, today] so we'll test it up to a tolerance.
        #  (it *may* match exactly due to rounding)
        # GH 57535
        request.applymarker(
            pytest.mark.xfail(
                reason="result may not exactly match [now, today]", strict=False
            )
        )
        tolerance = pd.Timedelta(seconds=1)
        assert diff0 < tolerance, f"The difference is {diff0}"
        assert diff1 < tolerance, f"The difference is {diff0}"

    def test_dti_constructor_object_float_matches_float_dtype(self):
        # [FUNCTION DOC]
        # Regression test: float array with NaN and its object-dtype copy give equal timezone-aware
        # indexes.
        # GH#55780
        arr = np.array([0, np.nan], dtype=np.float64)
        arr2 = arr.astype(object)

        dti1 = DatetimeIndex(arr, tz="CET")
        dti2 = DatetimeIndex(arr2, tz="CET")
        tm.assert_index_equal(dti1, dti2)

    @pytest.mark.parametrize("dtype", ["M8[us]", "M8[us, US/Pacific]"])
    def test_dti_constructor_with_dtype_object_int_matches_int_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over microsecond dtypes with and without timezone. Verifies integer, float, and
        # object-typed integer or float arrays with NaT/NaN all give the same expected index when a dtype
        # is passed.
        # Going through the object path should match the non-object path

        vals1 = np.arange(5, dtype="i8") * 1000
        vals1[0] = pd.NaT.value

        vals2 = vals1.astype(np.float64)
        vals2[0] = np.nan

        vals3 = vals1.astype(object)
        # change lib.infer_dtype(vals3) from "integer" so we go through
        #  array_to_datetime in _sequence_to_dt64
        vals3[0] = pd.NaT

        vals4 = vals2.astype(object)

        res1 = DatetimeIndex(vals1, dtype=dtype)
        res2 = DatetimeIndex(vals2, dtype=dtype)
        res3 = DatetimeIndex(vals3, dtype=dtype)
        res4 = DatetimeIndex(vals4, dtype=dtype)

        expected = DatetimeIndex(vals1.view("M8[us]"))
        if res1.tz is not None:
            expected = expected.tz_localize("UTC").tz_convert(res1.tz)
        tm.assert_index_equal(res1, expected)
        tm.assert_index_equal(res2, expected)
        tm.assert_index_equal(res3, expected)
        tm.assert_index_equal(res4, expected)


class TestTimeSeries:
    # [CLASS DOC]
    # Time-series oriented constructor tests for DatetimeIndex; no state.
    def test_dti_constructor_preserve_dti_freq(self):
        # [FUNCTION DOC]
        # Verifies constructing from an existing range preserves its freq.
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")

        rng2 = DatetimeIndex(rng)
        assert rng.freq == rng2.freq

    def test_explicit_none_freq(self):
        # [FUNCTION DOC]
        # Verifies explicitly passing freq as none drops the freq for both index and array inputs.
        # Explicitly passing freq=None is respected
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")

        result = DatetimeIndex(rng, freq=None)
        assert result.freq is None

        result = DatetimeIndex(rng._data, freq=None)
        assert result.freq is None

    def test_dti_constructor_small_int(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Uses any_int_numpy_dtype. Regression test: small integers of any integer dtype are interpreted as
        # nanoseconds from the epoch.
        # see gh-13721
        exp = DatetimeIndex(
            [
                "1970-01-01 00:00:00.00000000",
                "1970-01-01 00:00:00.00000001",
                "1970-01-01 00:00:00.00000002",
            ]
        )

        arr = np.array([0, 10, 20], dtype=any_int_numpy_dtype)
        tm.assert_index_equal(DatetimeIndex(arr), exp)

    def test_ctor_str_intraday(self):
        # [FUNCTION DOC]
        # Verifies a string with only time-of-day information is parsed with the seconds component.
        rng = DatetimeIndex(["1-1-2000 00:00:01"])
        assert rng[0].second == 1

    def test_index_cast_datetime64_other_units(self):
        # [FUNCTION DOC]
        # Verifies an Index built from a day-unit datetime64 array holds values equal to the nanosecond
        # cast of the array.
        arr = np.arange(0, 100, 10, dtype=np.int64).view("M8[D]")
        idx = Index(arr)

        assert (idx.values == astype_overflowsafe(arr, dtype=np.dtype("M8[ns]"))).all()

    def test_constructor_int64_nocopy(self):
        # [FUNCTION DOC]
        # Regression test: with copy false the index shares memory with the integer array, so mutating the
        # array changes the index; with copy true it does not.
        # GH#1624
        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr, copy=False)

        arr[50:100] = -1
        assert (index.asi8[50:100] == -1).all()

        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr, copy=True)

        arr[50:100] = -1
        assert (index.asi8[50:100] != -1).all()

    @pytest.mark.parametrize(
        "freq",
        ["ME", "QE", "YE", "D", "B", "bh", "min", "s", "ms", "us", "h", "ns", "C"],
    )
    def test_from_freq_recreate_from_data(self, freq):
        # [FUNCTION DOC]
        # Parametrized over thirteen frequencies. Verifies re-creating an index from a one-element range with
        # the same freq (with and without timezone) gives an equal index.
        org = date_range(start="2001/02/01 09:00", freq=freq, periods=1)
        idx = DatetimeIndex(org, freq=freq)
        tm.assert_index_equal(idx, org)

        org = date_range(
            start="2001/02/01 09:00", freq=freq, tz="US/Pacific", periods=1
        )
        idx = DatetimeIndex(org, freq=freq, tz="US/Pacific")
        tm.assert_index_equal(idx, org)

    def test_datetimeindex_constructor_misc(self):
        # [FUNCTION DOC]
        # Verifies an unparsable date string raises a value error about unknown format, various mixes of
        # strings and datetimes parse identically, and day-first and year-first parsing options give equal
        # results.
        arr = ["1/1/2005", "1/2/2005", "Jn 3, 2005", "2005-01-04"]
        msg = r"(\(')?Unknown datetime string format(:', 'Jn 3, 2005'\))?"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(arr)

        arr = ["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"]
        idx1 = DatetimeIndex(arr)

        arr = [datetime(2005, 1, 1), "1/2/2005", "1/3/2005", "2005-01-04"]
        idx2 = DatetimeIndex(arr)

        arr = [Timestamp(datetime(2005, 1, 1)), "1/2/2005", "1/3/2005", "2005-01-04"]
        idx3 = DatetimeIndex(arr)

        arr = np.array(["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"], dtype="O")
        idx4 = DatetimeIndex(arr)

        idx5 = DatetimeIndex(["12/05/2007", "25/01/2008"], dayfirst=True)
        idx6 = DatetimeIndex(
            ["2007/05/12", "2008/01/25"], dayfirst=False, yearfirst=True
        )
        tm.assert_index_equal(idx5, idx6)

        for other in [idx2, idx3, idx4]:
            assert (idx1.values == other.values).all()

    def test_dti_constructor_object_dtype_dayfirst_yearfirst_with_tz(self):
        # [FUNCTION DOC]
        # Regression test: a day-ambiguous string with a timezone parses as day-first (October 5, 2016) or
        # year-first (October 16, 2005) according to the option.
        # GH#55813
        val = "5/10/16"

        dfirst = Timestamp(2016, 10, 5, tz="US/Pacific")
        yfirst = Timestamp(2005, 10, 16, tz="US/Pacific")

        result1 = DatetimeIndex([val], tz="US/Pacific", dayfirst=True)
        expected1 = DatetimeIndex([dfirst])
        tm.assert_index_equal(result1, expected1)

        result2 = DatetimeIndex([val], tz="US/Pacific", yearfirst=True)
        expected2 = DatetimeIndex([yfirst])
        tm.assert_index_equal(result2, expected2)
