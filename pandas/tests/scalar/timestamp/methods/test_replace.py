# [MODULE DOC]
# Tests for the Timestamp replace method, which returns a new timestamp with selected
# calendar, clock, timezone or fold fields substituted. Covers out-of-bounds years for
# different units, non-nanosecond resolution, naive and aware timestamps, nanosecond
# preservation, invalid arguments, daylight-saving transitions, fold handling and unit
# promotion.
from datetime import datetime
import zoneinfo

from dateutil.tz import gettz
import numpy as np
import pytest

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timestamp,
    conversion,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import WASM
import pandas.util._test_decorators as td

import pandas._testing as tm


class TestTimestampReplace:
    # [CLASS DOC]
    # Groups the tests of the Timestamp replace method. Stateless.
    def test_replace_out_of_pydatetime_bounds(self):
        # [FUNCTION DOC]
        # Regression test (GH 50348). Replacing the year with a value beyond what
        # nanosecond resolution can hold raises an out-of-bounds datetime error with a
        # message naming the value and frequency. After converting to millisecond
        # resolution, the same replacement succeeds, giving the requested year and the
        # same integer as a millisecond datetime64 for that date.
        # GH#50348
        ts = Timestamp("2016-01-01").as_unit("ns")

        msg = "Out of bounds timestamp: 99999-01-01 00:00:00 with frequency 'ns'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            ts.replace(year=99_999)

        ts = ts.as_unit("ms")
        result = ts.replace(year=99_999)
        assert result.year == 99_999
        assert result._value == Timestamp(np.datetime64("99999-01-01", "ms"))._value

    def test_replace_non_nano(self):
        # [FUNCTION DOC]
        # Verifies replace on a microsecond-resolution timestamp far in the future (year
        # 4869 built through the private constructor). The result keeps the same
        # resolution code and converts to the python datetime with the year replaced by
        # 4900 and same month and day.
        ts = Timestamp._from_value_and_reso(
            91514880000000000, NpyDatetimeUnit.NPY_FR_us.value, None
        )
        assert ts.to_pydatetime() == datetime(4869, 12, 28)

        result = ts.replace(year=4900)
        assert result._creso == ts._creso
        assert result.to_pydatetime() == datetime(4900, 12, 28)

    def test_replace_naive(self):
        # [FUNCTION DOC]
        # Verifies replacing the hour on a tz-naive timestamp gives the expected
        # timestamp with the hour set to zero (GH 14621, GH 7825).
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00")
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00")
        assert result == expected

    def test_replace_aware(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Verifies replacing the hour on a tz-aware timestamp (for each aware timezone
        # fixture) gives the expected timestamp with zero hour in the same timezone.
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        # replacing datetime components with and w/o presence of a timezone
        ts = Timestamp("2016-01-01 09:00:00", tz=tz)
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00", tz=tz)
        assert result == expected

    def test_replace_preserves_nanos(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Verifies replacing the hour on an aware timestamp that has nonzero nanoseconds
        # keeps the nanoseconds.
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00.000000123", tz=tz)
        assert result == expected

    def test_replace_multiple(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Verifies replacing all of year, month, day, hour, minute, second, microsecond
        # and nanosecond at once on an aware timestamp gives the fully replaced
        # timestamp.
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        # replacing datetime components with and w/o presence of a timezone
        # test all
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        result = ts.replace(
            year=2015,
            month=2,
            day=2,
            hour=0,
            minute=5,
            second=5,
            microsecond=5,
            nanosecond=5,
        )
        expected = Timestamp("2015-02-02 00:05:05.000005005", tz=tz)
        assert result == expected

    def test_replace_invalid_kwarg(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Verifies that an unknown keyword argument to replace raises a type error whose
        # message says replace got an unexpected keyword argument.
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        msg = r"replace\(\) got an unexpected keyword argument"
        with pytest.raises(TypeError, match=msg):
            ts.replace(foo=5)

    def test_replace_integer_args(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Verifies that a non-integer (float) value for the hour raises a value error
        # saying the value must be an integer and naming the received type and the
        # field.
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        msg = "value must be an integer, received <class 'float'> for hour"
        with pytest.raises(ValueError, match=msg):
            ts.replace(hour=0.1)

    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    def test_replace_tzinfo_equiv_tz_localize_none(self):
        # [FUNCTION DOC]
        # Verifies that replacing the timezone info with none gives the same result as
        # localizing to none, for a time near a daylight-saving boundary in US/Eastern.
        # Skipped on WASM (timezone reset not available).
        # GH#14621, GH#7825
        # assert conversion to naive is the same as replacing tzinfo with None
        ts = Timestamp("2013-11-03 01:59:59.999999-0400", tz="US/Eastern")
        assert ts.tz_localize(None) == ts.replace(tzinfo=None)

    @td.skip_if_windows
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    def test_replace_tzinfo(self):
        # [FUNCTION DOC]
        # Regression test (GH 15683). Skipped on Windows and WASM. A datetime with fold
        # set in the Berlin spring-forward gap has its tzinfo replaced by a Berlin
        # tzinfo object, and the same is done to the equivalent Timestamp. Expected:
        # with the process timezone set to UTC both give the same POSIX seconds; both
        # are equal to each other and to the python datetime of the Timestamp. The same
        # checks are repeated after replacing tzinfo again with none.
        # GH#15683
        dt = datetime(2016, 3, 27, 1, fold=1)
        tzinfo = dt.astimezone(zoneinfo.ZoneInfo("Europe/Berlin")).tzinfo

        result_dt = dt.replace(tzinfo=tzinfo)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo)

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            assert result_dt.timestamp() == result_pd.timestamp()

        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

        result_dt = dt.replace(tzinfo=tzinfo).replace(tzinfo=None)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo).replace(tzinfo=None)

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            assert result_dt.timestamp() == result_pd.timestamp()

        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

    @pytest.mark.parametrize(
        "tz, normalize",
        [
            ("pytz/US/Eastern", lambda x: x.tzinfo.normalize(x)),
            (gettz("US/Eastern"), lambda x: x),
        ],
    )
    def test_replace_across_dst(self, tz, normalize):
        # [FUNCTION DOC]
        # Regression test (GH 18319). Parametrized over a pytz zone (with a normalize
        # function) and a dateutil zone (identity normalize). A naive timestamp is
        # localized with the low-level helper, the sanity check confirms the normalize
        # function does not change it, then the month is replaced with June (across the
        # daylight-saving boundary). Expected: hour and minute stay the same and the
        # result is unchanged by normalization.
        # GH#18319 check that 1) timezone is correctly normalized and
        # 2) that hour is not incorrectly changed by this normalization
        if isinstance(tz, str) and tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        ts_naive = Timestamp("2017-12-03 16:03:30")
        ts_aware = conversion.localize_pydatetime(ts_naive, tz)

        # Preliminary sanity-check
        assert ts_aware == normalize(ts_aware)

        # Replace across DST boundary
        ts2 = ts_aware.replace(month=6)

        # Check that `replace` preserves hour literal
        assert (ts2.hour, ts2.minute) == (ts_aware.hour, ts_aware.minute)

        # Check that post-replace object is appropriately normalized
        ts2b = normalize(ts2)
        assert ts2 == ts2b

    def test_replace_dst_border(self, unit):
        # [FUNCTION DOC]
        # Regression test (GH 7825). Replacing the hour with 3 on a midnight Chicago
        # timestamp on the day of a fall-back transition, for each unit fixture, gives
        # the expected 03:00 timestamp in Chicago and keeps the unit's resolution code.
        # Gh 7825
        t = Timestamp("2013-11-3", tz="America/Chicago").as_unit(unit)
        result = t.replace(hour=3)
        expected = Timestamp("2013-11-3 03:00:00", tz="America/Chicago")
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    @pytest.mark.parametrize("fold", [0, 1])
    @pytest.mark.parametrize("tz", ["dateutil/Europe/London", "Europe/London"])
    def test_replace_dst_fold(self, fold, tz, unit):
        # [FUNCTION DOC]
        # Regression test (GH 25017). Parametrized over fold zero/one, timezone
        # (dateutil or default London zone) and unit. Replacing the hour with 1 and the
        # fold on an ambiguous London time must equal the naive 01:30 localized with the
        # ambiguous flag set to the opposite truth value of fold; the resolution code is
        # preserved.
        # GH 25017
        d = datetime(2019, 10, 27, 2, 30)
        ts = Timestamp(d, tz=tz).as_unit(unit)
        result = ts.replace(hour=1, fold=fold)
        expected = Timestamp(datetime(2019, 10, 27, 1, 30)).tz_localize(
            tz, ambiguous=not fold
        )
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    @pytest.mark.parametrize("fold", [0, 1])
    def test_replace_preserves_fold(self, fold):
        # [FUNCTION DOC]
        # Regression test (GH 37610). Parametrized over fold zero/one. Replacing the
        # second on a Moscow timestamp that has a given fold must leave the fold
        # unchanged.
        # GH#37610. Check that replace preserves Timestamp fold property
        tz = gettz("Europe/Moscow")

        ts = Timestamp(
            year=2009, month=10, day=25, hour=2, minute=30, fold=fold, tzinfo=tz
        )
        ts_replaced = ts.replace(second=1)

        assert ts_replaced.fold == fold

    def test_replace_updates_unit(self):
        # [FUNCTION DOC]
        # Regression test (GH 57749): replace promotes the unit when the new value needs
        # finer resolution. A microsecond-resolution timestamp replaced with the
        # microsecond and nanosecond of a nanosecond timestamp becomes nanosecond unit
        # and equals it. A second-resolution timestamp replaced with a microsecond value
        # becomes microsecond unit and equals the microsecond reference.
        # GH#57749
        ts = Timestamp("2023-07-15 23:08:12.134567123")
        ts2 = Timestamp("2023-07-15 23:08:12.000000")
        assert ts2.unit == "us"
        result = ts2.replace(microsecond=ts.microsecond, nanosecond=ts.nanosecond)
        assert result.unit == "ns"
        assert result == ts

        ts3 = Timestamp("2023-07-15 23:08:12").as_unit("s")
        result = ts3.replace(microsecond=ts2.microsecond)
        assert result.unit == "us"
        assert result == ts2
