"""
Tests for Timestamp timezone-related methods
"""
# [MODULE DOC]
# Tests for timezone-related operations of Timestamp (the timetz method).

from datetime import datetime

from pandas._libs.tslibs import timezones

from pandas import Timestamp


class TestTimestampTZOperations:
    # [CLASS DOC]
    # Groups the Timestamp timezone operation tests. Stateless.
    # ------------------------------------------------------------------

    def test_timestamp_timetz_equivalent_with_datetime_tz(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Regression test (GH 21358). Over every naive-or-timezone fixture, the
        # time-with-timezone component of a Timestamp equals that of the equivalent
        # python datetime built with the same timezone.
        # GH21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)

        stamp = Timestamp("2018-06-04 10:20:30", tz=tz)
        _datetime = datetime(2018, 6, 4, hour=10, minute=20, second=30, tzinfo=tz)

        result = stamp.timetz()
        expected = _datetime.timetz()

        assert result == expected
