# [MODULE DOC]
# Tests for the time-window row selection method of frames and series (between_time):
# start/end given as time objects or strings in many formats, inclusive modes, windows
# crossing midnight, time zones, axis arguments and error cases. Uses frame-or-series,
# axis and inclusive-endpoint fixtures, locale decorator and testing helpers. No
# import-time side effects.

from datetime import (
    datetime,
    time,
)

import numpy as np
import pytest

from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestBetweenTime:
    # [CLASS DOC]
    # Stateless pytest class of between_time tests.
    @td.skip_if_not_us_locale
    def test_between_time_formats(self, frame_or_series):
        # [FUNCTION DOC]
        # Runs only in a US locale. Eight equivalent spellings of a 2:00 to 2:30 window
        # (24h/12h, with/without colons, with seconds) each select 28 rows of a 5-minute
        # index over several days.
        # GH#11818
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)

        strings = [
            ("2:00", "2:30"),
            ("0200", "0230"),
            ("2:00am", "2:30am"),
            ("0200am", "0230am"),
            ("2:00:00", "2:30:00"),
            ("020000", "023000"),
            ("2:00:00am", "2:30:00am"),
            ("020000am", "023000am"),
        ]
        expected_length = 28

        for time_string in strings:
            assert len(ts.between_time(*time_string)) == expected_length

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_localized_between_time(self, tzstr, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over a pytz and a dateutil US/Eastern zone with frame-or-series
        # fixture. Between-time on a localized object equals the naive result localized;
        # zone preserved.
        tz = timezones.maybe_get_tz(tzstr)

        rng = date_range("4/16/2012", "5/1/2012", freq="h")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        if frame_or_series is DataFrame:
            ts = ts.to_frame()

        ts_local = ts.tz_localize(tzstr)

        t1, t2 = time(10, 0), time(11, 0)
        result = ts_local.between_time(t1, t2)
        expected = ts.between_time(t1, t2).tz_localize(tzstr)
        tm.assert_equal(result, expected)
        assert timezones.tz_compare(result.index.tz, tz)

    def test_between_time_types(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Passing datetime objects (not times) raises a
        # value error saying the argument cannot be converted to a time.
        # GH11818
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        obj = DataFrame({"A": 0}, index=rng)
        obj = tm.get_obj(obj, frame_or_series)

        msg = r"Cannot convert arg \[datetime\.datetime\(2010, 1, 2, 1, 0\)\] to a time"
        with pytest.raises(ValueError, match=msg):
            obj.between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))

    def test_between_time(self, inclusive_endpoints_fixture, frame_or_series):
        # [FUNCTION DOC]
        # Takes inclusive-endpoint fixture and frame-or-series. Verifies the number of
        # rows for a 00:00 to 01:00 window under each inclusive mode and that every
        # selected time respects the boundaries; string and time-object bounds agree;
        # repeats for a window across midnight (22:00 to 09:00) with its own expected
        # counts and boundary checks.
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)

        stime = time(0, 0)
        etime = time(1, 0)
        inclusive = inclusive_endpoints_fixture

        filtered = ts.between_time(stime, etime, inclusive=inclusive)
        exp_len = 13 * 4 + 1

        if inclusive in ["right", "neither"]:
            exp_len -= 5
        if inclusive in ["left", "neither"]:
            exp_len -= 4

        assert len(filtered) == exp_len
        for rs in filtered.index:
            t = rs.time()
            if inclusive in ["left", "both"]:
                assert t >= stime
            else:
                assert t > stime

            if inclusive in ["right", "both"]:
                assert t <= etime
            else:
                assert t < etime

        result = ts.between_time("00:00", "01:00")
        expected = ts.between_time(stime, etime)
        tm.assert_equal(result, expected)

        # across midnight
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)
        stime = time(22, 0)
        etime = time(9, 0)

        filtered = ts.between_time(stime, etime, inclusive=inclusive)
        exp_len = (12 * 11 + 1) * 4 + 1
        if inclusive in ["right", "neither"]:
            exp_len -= 4
        if inclusive in ["left", "neither"]:
            exp_len -= 4

        assert len(filtered) == exp_len
        for rs in filtered.index:
            t = rs.time()
            if inclusive in ["left", "both"]:
                assert (t >= stime) or (t <= etime)
            else:
                assert (t > stime) or (t <= etime)

            if inclusive in ["right", "both"]:
                assert (t <= etime) or (t >= stime)
            else:
                assert (t < etime) or (t >= stime)

    def test_between_time_raises(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Non-datetime index raises a type error saying
        # the index must be a datetime index.
        # GH#20725
        obj = DataFrame([[1, 2, 3], [4, 5, 6]])
        obj = tm.get_obj(obj, frame_or_series)

        msg = "Index must be DatetimeIndex"
        with pytest.raises(TypeError, match=msg):  # index is not a DatetimeIndex
            obj.between_time(start_time="00:00", end_time="12:00")

    def test_between_time_axis(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Default and explicit row axis give seven rows
        # for an hour window of 10-minute data; an axis equal to the object dimension
        # raises a value error saying no such axis exists.
        # GH#8839
        rng = date_range("1/1/2000", periods=100, freq="10min")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        if frame_or_series is DataFrame:
            ts = ts.to_frame()

        stime, etime = ("08:00:00", "09:00:00")
        expected_length = 7

        assert len(ts.between_time(stime, etime)) == expected_length
        assert len(ts.between_time(stime, etime, axis=0)) == expected_length
        msg = f"No axis named {ts.ndim} for object type {type(ts).__name__}"
        with pytest.raises(ValueError, match=msg):
            ts.between_time(stime, etime, axis=ts.ndim)

    def test_between_time_axis_aliases(self, axis):
        # [FUNCTION DOC]
        # Takes axis fixture. With dates on the rows or the columns of a square frame,
        # selection along the matching axis (names or numbers) gives seven entries.
        # GH#8839
        rng = date_range("1/1/2000", periods=100, freq="10min")
        ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), len(rng))))
        stime, etime = ("08:00:00", "09:00:00")
        exp_len = 7

        if axis in ["index", 0]:
            ts.index = rng
            assert len(ts.between_time(stime, etime)) == exp_len
            assert len(ts.between_time(stime, etime, axis=0)) == exp_len

        if axis in ["columns", 1]:
            ts.columns = rng
            selected = ts.between_time(stime, etime, axis=1).columns
            assert len(selected) == exp_len

    def test_between_time_axis_raises(self, axis):
        # [FUNCTION DOC]
        # Takes axis fixture. When the relevant axis is not a datetime index (replaced
        # by integers), selection on that axis raises a type error about the index being
        # a datetime index.
        # issue 8839
        rng = date_range("1/1/2000", periods=100, freq="10min")
        mask = np.arange(0, len(rng))
        rand_data = np.random.default_rng(2).standard_normal((len(rng), len(rng)))
        ts = DataFrame(rand_data, index=rng, columns=rng)
        stime, etime = ("08:00:00", "09:00:00")

        msg = "Index must be DatetimeIndex"
        if axis in ["columns", 1]:
            ts.index = mask
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime)
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime, axis=0)

        if axis in ["index", 0]:
            ts.columns = mask
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime, axis=1)

    def test_between_time_datetimeindex(self):
        # [FUNCTION DOC]
        # A 13:00 to 14:00 window on a half-hourly index gives 12 rows equal to
        # label-based time slicing and positional selection.
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        bkey = slice(time(13, 0, 0), time(14, 0, 0))
        binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]

        result = df.between_time(bkey.start, bkey.stop)
        expected = df.loc[bkey]
        expected2 = df.iloc[binds]
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result, expected2)
        assert len(result) == 12

    def test_between_time_incorrect_arg_inclusive(self):
        # [FUNCTION DOC]
        # An unknown inclusive string raises a value error listing the allowed values.
        # GH40245
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )

        stime = time(0, 0)
        etime = time(1, 0)
        inclusive = "bad_string"
        msg = "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
        with pytest.raises(ValueError, match=msg):
            ts.between_time(stime, etime, inclusive=inclusive)
