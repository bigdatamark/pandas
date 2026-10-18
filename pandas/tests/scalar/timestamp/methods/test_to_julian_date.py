# [MODULE DOC]
# Tests for the Timestamp method that returns the Julian date as a float, checked
# against known reference values in several centuries and at several hours of the day.
from pandas import Timestamp


class TestTimestampToJulianDate:
    # [CLASS DOC]
    # Groups the Julian date tests for Timestamp. Stateless.
    def test_compare_1700(self):
        # [FUNCTION DOC]
        # Verifies the Julian date of midnight on a day in 1700; expected exactly the
        # known half-integer day number.
        ts = Timestamp("1700-06-23")
        res = ts.to_julian_date()
        assert res == 2_342_145.5

    def test_compare_2000(self):
        # [FUNCTION DOC]
        # Verifies the Julian date of midnight on a day in 2000; expected the known
        # half-integer day number.
        ts = Timestamp("2000-04-12")
        res = ts.to_julian_date()
        assert res == 2_451_646.5

    def test_compare_2100(self):
        # [FUNCTION DOC]
        # Verifies the Julian date of midnight on a day in 2100; expected the known
        # half-integer day number.
        ts = Timestamp("2100-08-12")
        res = ts.to_julian_date()
        assert res == 2_488_292.5

    def test_compare_hour01(self):
        # [FUNCTION DOC]
        # Verifies the Julian date at 01:00 on a day in 2000; expected the half-integer
        # day number plus one twenty-fourth of a day, compared with exact float
        # equality.
        ts = Timestamp("2000-08-12T01:00:00")
        res = ts.to_julian_date()
        assert res == 2_451_768.5416666666666666

    def test_compare_hour13(self):
        # [FUNCTION DOC]
        # Verifies the Julian date at 13:00 on a day in 2000; expected the next whole
        # day number plus one twenty-fourth of a day (since Julian days start at noon),
        # compared with exact float equality.
        ts = Timestamp("2000-08-12T13:00:00")
        res = ts.to_julian_date()
        assert res == 2_451_769.0416666666666666
