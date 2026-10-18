# [MODULE DOC]
# Tests for individual US federal holiday rules and calendar results: Martin Luther
# King Jr. day start year, Memorial Day dates, and result type stability for empty
# queries. No import-time side effects.
from datetime import datetime

from pandas import DatetimeIndex
import pandas._testing as tm

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    USFederalHolidayCalendar,
    USMartinLutherKingJr,
    USMemorialDay,
)


def test_no_mlk_before_1986():
    # [FUNCTION DOC]
    # see gh-10278. A calendar holding only the MLK rule queried from 1984 to 1988 returns only
    # 1986-01-20 and 1987-01-19, since the holiday is not observed before 1986.
    # see gh-10278
    class MLKCalendar(AbstractHolidayCalendar):
        # [CLASS DOC]
        # HELPER: local calendar subclass with the MLK rule as its only rule.
        rules = [USMartinLutherKingJr]

    holidays = MLKCalendar().holidays(start="1984", end="1988").to_pydatetime().tolist()

    # Testing to make sure holiday is not incorrectly observed before 1986.
    assert holidays == [datetime(1986, 1, 20, 0, 0), datetime(1987, 1, 19, 0, 0)]


def test_memorial_day():
    # [FUNCTION DOC]
    # Memorial Day rule queried from 1971 to 1980 returns the last Monday of May for 1971 through
    # 1979 as datetimes (the 1980 query end is the start of that year, so 1980 is excluded).
    class MemorialDay(AbstractHolidayCalendar):
        # [CLASS DOC]
        # HELPER: local calendar subclass with Memorial Day as its only rule.
        rules = [USMemorialDay]

    holidays = MemorialDay().holidays(start="1971", end="1980").to_pydatetime().tolist()

    # Fixes 5/31 error and checked manually against Wikipedia.
    assert holidays == [
        datetime(1971, 5, 31, 0, 0),
        datetime(1972, 5, 29, 0, 0),
        datetime(1973, 5, 28, 0, 0),
        datetime(1974, 5, 27, 0, 0),
        datetime(1975, 5, 26, 0, 0),
        datetime(1976, 5, 31, 0, 0),
        datetime(1977, 5, 30, 0, 0),
        datetime(1978, 5, 29, 0, 0),
        datetime(1979, 5, 28, 0, 0),
    ]


def test_federal_holiday_inconsistent_returntype():
    # [FUNCTION DOC]
    # GH 49075. Uses two separate federal calendar instances (to avoid the shared cache) for August 2018 and
    # August 2019, months without federal holidays. Both results must equal an empty microsecond datetime index with no frequency.
    # GH 49075 test case
    # Instantiate two calendars to rule out _cache
    cal1 = USFederalHolidayCalendar()
    cal2 = USFederalHolidayCalendar()

    results_2018 = cal1.holidays(start=datetime(2018, 8, 1), end=datetime(2018, 8, 31))
    results_2019 = cal2.holidays(start=datetime(2019, 8, 1), end=datetime(2019, 8, 31))
    expected_results = DatetimeIndex([], dtype="datetime64[us]", freq=None)

    # Check against expected results to ensure both date
    # ranges generate expected results as per GH49075 submission
    tm.assert_index_equal(results_2018, expected_results)
    tm.assert_index_equal(results_2019, expected_results)
