# [MODULE DOC]
# Tests for the Timestamp method converting to a standard library datetime: preserving
# the fold flag, warning on nanosecond loss, preserving timezone objects for several
# timezone flavors, and round trips at the bounds.
from datetime import (
    datetime,
    timedelta,
)

import pytest

from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
import pandas.util._test_decorators as td

from pandas import Timestamp
import pandas._testing as tm


class TestTimestampToPyDatetime:
    # [CLASS DOC]
    # Groups the Timestamp to python-datetime conversion tests. Stateless.
    def test_to_pydatetime_fold(self):
        # [FUNCTION DOC]
        # Verifies the fold attribute (GH 45087) survives conversion: an ambiguous
        # Chicago time with fold set to one gives a datetime whose fold is one.
        # GH#45087
        tzstr = "dateutil/usr/share/zoneinfo/America/Chicago"
        ts = Timestamp(year=2013, month=11, day=3, hour=1, minute=0, fold=1, tz=tzstr)
        dt = ts.to_pydatetime()
        assert dt.fold == 1

    def test_to_pydatetime_nonzero_nano(self):
        # [FUNCTION DOC]
        # Verifies conversion of a timestamp with nonzero nanoseconds emits a user
        # warning about discarding nanoseconds, and that the result is the datetime
        # truncated to microseconds.
        ts = Timestamp("2011-01-01 9:00:00.123456789")

        # Warn the user of data loss (nanoseconds).
        msg = "Discarding nonzero nanoseconds in conversion"
        with tm.assert_produces_warning(UserWarning, match=msg):
            expected = datetime(2011, 1, 1, 9, 0, 0, 123456)
            result = ts.to_pydatetime()
            assert result == expected

    def test_timestamp_to_datetime(self):
        # [FUNCTION DOC]
        # Verifies a US/Eastern timestamp (string timezone) converts to a datetime that
        # is equal and has an equal timezone object.
        stamp = Timestamp("20090415", tz="US/Eastern")
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_timestamp_to_pydatetime_dateutil(self):
        # [FUNCTION DOC]
        # Verifies the same equality and timezone preservation for a timestamp using a
        # dateutil timezone given by string.
        stamp = Timestamp("20090415", tz="dateutil/US/Eastern")
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_timestamp_to_pydatetime_explicit_pytz(self):
        # [FUNCTION DOC]
        # Verifies equality and timezone preservation when the timezone is an explicit
        # pytz object. Skipped if pytz is not installed.
        pytz = pytest.importorskip("pytz")
        stamp = Timestamp("20090415", tz=pytz.timezone("US/Eastern"))
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    @td.skip_if_windows
    def test_timestamp_to_pydatetime_explicit_dateutil(self):
        # [FUNCTION DOC]
        # Verifies equality and timezone preservation when the timezone is an explicit
        # dateutil object obtained through the helper getter. Skipped on Windows.
        stamp = Timestamp("20090415", tz=gettz("US/Eastern"))
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_to_pydatetime_bijective(self):
        # [FUNCTION DOC]
        # Verifies that converting the maximum and minimum timestamps to datetimes and
        # back only loses sub-microsecond precision. A user warning is expected only if
        # the bound has nonzero nanoseconds. Checks the round trip (compared after
        # dividing the integer values by one thousand) for the maximum, and for the
        # minimum after adding one microsecond (which must exceed the minimum).
        # Ensure that converting to datetime and back only loses precision
        # by going from nanoseconds to microseconds.
        exp_warning = None if Timestamp.max.nanosecond == 0 else UserWarning
        with tm.assert_produces_warning(exp_warning):
            pydt_max = Timestamp.max.to_pydatetime()

        assert (
            Timestamp(pydt_max).as_unit("ns")._value / 1000
            == Timestamp.max._value / 1000
        )

        exp_warning = None if Timestamp.min.nanosecond == 0 else UserWarning
        with tm.assert_produces_warning(exp_warning):
            pydt_min = Timestamp.min.to_pydatetime()

        # The next assertion can be enabled once GH#39221 is merged
        #  assert pydt_min < Timestamp.min  # this is bc nanos are dropped
        tdus = timedelta(microseconds=1)
        assert pydt_min + tdus > Timestamp.min

        assert (
            Timestamp(pydt_min + tdus).as_unit("ns")._value / 1000
            == Timestamp.min._value / 1000
        )
