# [MODULE DOC]
# Tests for the time-of-day row selection method of frames and series (at_time):
# selecting entries at an exact time of day from a datetime index, with time objects and
# strings, time zones, either axis, empty results, error cases and deprecation of
# ambiguous string formats. Uses frame-or-series and axis fixtures plus the testing
# helpers module. No import-time side effects.

from datetime import (
    time,
    timezone,
)
import zoneinfo

import numpy as np
import pytest

from pandas._libs.tslibs import timezones
from pandas.errors import Pandas4Warning

from pandas import (
    DataFrame,
    date_range,
)
import pandas._testing as tm


class TestAtTime:
    # [CLASS DOC]
    # Stateless pytest class of at_time tests.
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_localized_at_time(self, tzstr, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over a pytz and a dateutil US/Eastern zone; takes frame-or-series
        # fixture. Selecting 10:00 from a localized object equals selecting from the
        # naive one and then localizing; the result index keeps the zone.
        tz = timezones.maybe_get_tz(tzstr)

        rng = date_range("4/16/2012", "5/1/2012", freq="h")
        ts = frame_or_series(
            np.random.default_rng(2).standard_normal(len(rng)), index=rng
        )

        ts_local = ts.tz_localize(tzstr)

        result = ts_local.at_time(time(10, 0))
        expected = ts.at_time(time(10, 0)).tz_localize(tzstr)
        tm.assert_equal(result, expected)
        assert timezones.tz_compare(result.index.tz, tz)

    def test_at_time(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Selecting by a timestamp's time gives rows
        # whose hour, minute and second all match; string input equals time-object
        # input.
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)
        rs = ts.at_time(rng[1])
        assert (rs.index.hour == rng[1].hour).all()
        assert (rs.index.minute == rng[1].minute).all()
        assert (rs.index.second == rng[1].second).all()

        result = ts.at_time("9:30")
        expected = ts.at_time(time(9, 30))
        tm.assert_equal(result, expected)

    def test_at_time_midnight(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. On a daily midnight index, selecting midnight
        # returns the whole object.
        # midnight, everything
        rng = date_range("1/1/2000", "1/31/2000")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)

        result = ts.at_time(time(0, 0))
        tm.assert_equal(result, ts)

    def test_at_time_nonexistent(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Selecting a time that never occurs in a
        # 23-minute-frequency index returns an empty object.
        # time doesn't exist
        rng = date_range("1/1/2012", freq="23Min", periods=384)
        ts = DataFrame(np.random.default_rng(2).standard_normal(len(rng)), rng)
        ts = tm.get_obj(ts, frame_or_series)
        rs = ts.at_time("16:00")
        assert len(rs) == 0

    @pytest.mark.parametrize(
        "hour", ["1:00", "1:00AM", time(1), time(1, tzinfo=timezone.utc)]
    )
    def test_at_time_errors(self, hour):
        # [FUNCTION DOC]
        # Parametrized over string and time-object inputs, with and without tz. Naive
        # inputs select the 01:00 row; a tz-aware time on a naive index raises a value
        # error saying the index must be timezone-aware.
        # GH#24043
        dti = date_range("2018", periods=3, freq="h")
        df = DataFrame(list(range(len(dti))), index=dti)
        if getattr(hour, "tzinfo", None) is None:
            result = df.at_time(hour)
            expected = df.iloc[1:2]
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="Index must be timezone"):
                df.at_time(hour)

    def test_at_time_tz(self):
        # [FUNCTION DOC]
        # A tz-aware time object (Eastern) on a Pacific tz-aware index selects the row
        # at the equivalent instant time.
        # GH#24043
        dti = date_range("2018", periods=3, freq="h", tz="US/Pacific")
        df = DataFrame(list(range(len(dti))), index=dti)
        result = df.at_time(time(4, tzinfo=zoneinfo.ZoneInfo("US/Eastern")))
        expected = df.iloc[1:2]
        tm.assert_frame_equal(result, expected)

    def test_at_time_raises(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. A non-datetime index raises a type error saying
        # the index must be a datetime index.
        # GH#20725
        obj = DataFrame([[1, 2, 3], [4, 5, 6]])
        obj = tm.get_obj(obj, frame_or_series)
        msg = "Index must be DatetimeIndex"
        with pytest.raises(TypeError, match=msg):  # index is not a DatetimeIndex
            obj.at_time("00:00")

    def test_at_time_axis(self, axis):
        # [FUNCTION DOC]
        # Takes axis fixture (names and numbers). Selecting 9:30 along rows or columns
        # of a square frame indexed by dates on both axes equals label-based selection
        # along that axis; frequency is cleared before comparison.
        # KNOWN ISSUE: the explanatory comment mentions a frequency string from an old
        # alias (1440T) that no longer exists, but the freq-clearing step still applies.
        # issue 8839
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")
        ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), len(rng))))
        ts.index, ts.columns = rng, rng

        indices = rng[(rng.hour == 9) & (rng.minute == 30) & (rng.second == 0)]

        if axis in ["index", 0]:
            expected = ts.loc[indices, :]
        elif axis in ["columns", 1]:
            expected = ts.loc[:, indices]

        result = ts.at_time("9:30", axis=axis)

        # Without clearing freq, result has freq 1440T and expected 5T
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        tm.assert_frame_equal(result, expected)

    def test_at_time_datetimeindex(self):
        # [FUNCTION DOC]
        # At 12:00 on a half-hourly index returns four rows, equal to label-based time
        # slicing and to positional selection.
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        akey = time(12, 0, 0)
        ainds = [24, 72, 120, 168]

        result = df.at_time(akey)
        expected = df.loc[akey]
        expected2 = df.iloc[ainds]
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result, expected2)
        assert len(result) == 4

    def test_at_time_ambiguous_format_deprecation(self):
        # [FUNCTION DOC]
        # Strings with a date part passed as the time emit Pandas4Warning stating the
        # string cannot be parsed (three variants asserted); formats that dateutil
        # rejects (compact digits with and without am/pm) are accepted silently.
        # GH#50839
        rng = date_range("1/1/2000", "1/5/2000", freq="125min")
        ts = DataFrame(list(range(len(rng))), index=rng)

        msg1 = "The string '.*' cannot be parsed"
        with tm.assert_produces_warning(Pandas4Warning, match=msg1):
            ts.at_time("2022-12-12 00:00:00")
        with tm.assert_produces_warning(Pandas4Warning, match=msg1):
            ts.at_time("2022-12-12 00:00:00 +09:00")
        with tm.assert_produces_warning(Pandas4Warning, match=msg1):
            ts.at_time("2022-12-12 00:00:00.000000")

        # The dateutil parser raises on these, so we can give the future behavior
        #  immediately using pd.core.tools.to_time
        ts.at_time("235500")
        ts.at_time("115500PM")
