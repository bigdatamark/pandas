# [MODULE DOC]
# Tests for the string forms of Timestamp: ISO formatting with a time specification,
# repr (including evaluability and timezone display), pretty printing and string
# conversion compared with the standard library datetime. Defines module-level sample
# timestamps used by the ISO test (without nanoseconds, year 1, with nanoseconds, aware
# with nanoseconds, nanoseconds without microseconds).
from datetime import (
    datetime,
    timezone,
)
import pprint

import dateutil.tz
import pytest

from pandas.compat import WASM

from pandas import Timestamp

ts_no_ns = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
)
ts_no_ns_year1 = Timestamp(
    year=1,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
)
ts_ns = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
    nanosecond=123,
)
ts_ns_tz = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
    nanosecond=123,
    tz="UTC",
)
ts_no_us = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=0,
    nanosecond=123,
)


@pytest.mark.parametrize(
    "ts, timespec, expected_iso",
    [
        (ts_no_ns, "auto", "2019-05-18T15:17:08.132263"),
        (ts_no_ns, "seconds", "2019-05-18T15:17:08"),
        (ts_no_ns, "nanoseconds", "2019-05-18T15:17:08.132263000"),
        (ts_no_ns_year1, "seconds", "0001-05-18T15:17:08"),
        (ts_no_ns_year1, "nanoseconds", "0001-05-18T15:17:08.132263000"),
        (ts_ns, "auto", "2019-05-18T15:17:08.132263123"),
        (ts_ns, "hours", "2019-05-18T15"),
        (ts_ns, "minutes", "2019-05-18T15:17"),
        (ts_ns, "seconds", "2019-05-18T15:17:08"),
        (ts_ns, "milliseconds", "2019-05-18T15:17:08.132"),
        (ts_ns, "microseconds", "2019-05-18T15:17:08.132263"),
        (ts_ns, "nanoseconds", "2019-05-18T15:17:08.132263123"),
        (ts_ns_tz, "auto", "2019-05-18T15:17:08.132263123+00:00"),
        (ts_ns_tz, "hours", "2019-05-18T15+00:00"),
        (ts_ns_tz, "minutes", "2019-05-18T15:17+00:00"),
        (ts_ns_tz, "seconds", "2019-05-18T15:17:08+00:00"),
        (ts_ns_tz, "milliseconds", "2019-05-18T15:17:08.132+00:00"),
        (ts_ns_tz, "microseconds", "2019-05-18T15:17:08.132263+00:00"),
        (ts_ns_tz, "nanoseconds", "2019-05-18T15:17:08.132263123+00:00"),
        (ts_no_us, "auto", "2019-05-18T15:17:08.000000123"),
    ],
)
def test_isoformat(ts, timespec, expected_iso):
    # [FUNCTION DOC]
    # Parametrized over a timestamp, a time specification (auto, hours through
    # nanoseconds) and the expected ISO string. The ISO format output with that
    # specification equals the expected string, including timezone suffix for aware
    # inputs and truncation or zero padding of fractions.
    assert ts.isoformat(timespec=timespec) == expected_iso


class TestTimestampRendering:
    # [CLASS DOC]
    # Groups the Timestamp repr and string rendering tests. Stateless.
    @pytest.mark.parametrize(
        "tz", ["UTC", "Asia/Tokyo", "US/Eastern", "dateutil/America/Los_Angeles"]
    )
    @pytest.mark.parametrize("freq", ["D", "M", "S", "N"])
    @pytest.mark.parametrize(
        "date", ["2014-03-07", "2014-01-01 09:00", "2014-01-01 00:00:00.000000001"]
    )
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    def test_repr(self, date, freq, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone, a frequency letter and a date string; skipped on
        # WASM. A naive timestamp repr contains the date but not the timezone name or
        # quoted frequency letter, and evaluating the repr reproduces an equal
        # timestamp; the aware repr contains the timezone name (with the dateutil prefix
        # removed), not the frequency, and also evaluates back to an equal timestamp.
        # avoid to match with timezone name
        freq_repr = f"'{freq}'"
        if tz.startswith("dateutil"):
            tz_repr = tz.replace("dateutil", "")
        else:
            tz_repr = tz

        date_only = Timestamp(date)
        assert date in repr(date_only)
        assert tz_repr not in repr(date_only)
        assert freq_repr not in repr(date_only)
        assert date_only == eval(repr(date_only))

        date_tz = Timestamp(date, tz=tz)
        assert date in repr(date_tz)
        assert tz_repr in repr(date_tz)
        assert freq_repr not in repr(date_tz)
        assert date_tz == eval(repr(date_tz))

    def test_repr_utcoffset(self):
        # [FUNCTION DOC]
        # Verifies a timestamp parsed with a numeric UTC offset shows the date with
        # offset and the fixed-offset name in its repr, not the internal tz offset class
        # name, and evaluates back to an equal timestamp.
        # This can cause the tz field to be populated, but it's redundant to
        # include this information in the date-string.
        date_with_utc_offset = Timestamp("2014-03-13 00:00:00-0400")
        assert "2014-03-13 00:00:00-0400" in repr(date_with_utc_offset)
        assert "tzoffset" not in repr(date_with_utc_offset)
        assert "UTC-04:00" in repr(date_with_utc_offset)
        expr = repr(date_with_utc_offset)
        assert date_with_utc_offset == eval(expr)

    def test_timestamp_repr_pre1900(self):
        # [FUNCTION DOC]
        # Verifies the repr of a timestamp before 1900 in US/Eastern works, and that the
        # repr of a timestamp with fractional seconds contains the ISO string given.
        # pre-1900
        stamp = Timestamp("1850-01-01", tz="US/Eastern")
        repr(stamp)

        iso8601 = "1850-01-01 01:23:45.012345"
        stamp = Timestamp(iso8601, tz="US/Eastern")
        result = repr(stamp)
        assert iso8601 in result

    def test_pprint(self):
        # [FUNCTION DOC]
        # Regression test (GH 12622). Pretty printing a nested dict and list structure
        # containing timestamps with a width of 50 gives the expected wrapped text with
        # each timestamp shown as a constructor call.
        # GH#12622
        nested_obj = {"foo": 1, "bar": [{"w": {"a": Timestamp("2011-01-01")}}] * 10}
        result = pprint.pformat(nested_obj, width=50)
        expected = r"""{'bar': [{'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}}],
 'foo': 1}"""
        assert result == expected

    def test_to_timestamp_repr_is_code(self):
        # [FUNCTION DOC]
        # Verifies the repr of timestamps (a two-digit year, naive, UTC and Los Angeles)
        # can be evaluated to recreate an equal timestamp.
        # KNOWN ISSUE: the name mentions to_timestamp but the test only exercises the
        # repr of Timestamp objects; no conversion method is called.
        zs = [
            Timestamp("99-04-17 00:00:00", tz="UTC"),
            Timestamp("2001-04-17 00:00:00", tz="UTC"),
            Timestamp("2001-04-17 00:00:00", tz="America/Los_Angeles"),
            Timestamp("2001-04-17 00:00:00", tz=None),
        ]
        for z in zs:
            assert eval(repr(z)) == z

    def test_repr_matches_pydatetime_no_tz(self):
        # [FUNCTION DOC]
        # Verifies that the string conversion of a naive Timestamp equals that of the
        # same datetime for dates, times and microseconds, and for nanosecond-only
        # values shows nine fractional digits.
        dt_date = datetime(2013, 1, 2)
        assert str(dt_date) == str(Timestamp(dt_date))

        dt_datetime = datetime(2013, 1, 2, 12, 1, 3)
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45)
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))

        ts_nanos_only = Timestamp(200)
        assert str(ts_nanos_only) == "1970-01-01 00:00:00.000000200"

        ts_nanos_micros = Timestamp(1200)
        assert str(ts_nanos_micros) == "1970-01-01 00:00:00.000001200"

    def test_repr_matches_pydatetime_tz_stdlib(self):
        # [FUNCTION DOC]
        # Verifies the string conversion of a Timestamp built from a datetime with the
        # standard library UTC timezone equals the datetime's own string for date, time
        # and microsecond cases.
        dt_date = datetime(2013, 1, 2, tzinfo=timezone.utc)
        assert str(dt_date) == str(Timestamp(dt_date))

        dt_datetime = datetime(2013, 1, 2, 12, 1, 3, tzinfo=timezone.utc)
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45, tzinfo=timezone.utc)
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))

    def test_repr_matches_pydatetime_tz_dateutil(self):
        # [FUNCTION DOC]
        # Same as the previous test but with the dateutil UTC timezone object.
        utc = dateutil.tz.tzutc()

        dt_date = datetime(2013, 1, 2, tzinfo=utc)
        assert str(dt_date) == str(Timestamp(dt_date))

        dt_datetime = datetime(2013, 1, 2, 12, 1, 3, tzinfo=utc)
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45, tzinfo=utc)
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))
