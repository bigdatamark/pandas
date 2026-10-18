"""
Tests for offsets.CustomBusinessDay / CDay
"""

# [MODULE DOC]
# Tests for the custom business day offset (CDay): repr, custom holidays given in several types, week masks in string, bit-string and list forms,
# combined masks and holidays, holiday calendar objects, pickling round trips and compatibility with an old pickled file. Uses the shared offsets assertion helper and the US federal holiday calendar.
# No import-time side effects.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs.offsets import CDay

from pandas import (
    _testing as tm,
    read_pickle,
)
from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries.holiday import USFederalHolidayCalendar


@pytest.fixture
def offset():
    # [FUNCTION DOC]
    # HELPER: fixture returning a default custom business day offset.
    return CDay()


@pytest.fixture
def offset2():
    # [FUNCTION DOC]
    # HELPER: fixture returning a custom business day offset with step two.
    return CDay(2)


class TestCustomBusinessDay:
    # [CLASS DOC]
    # Test collection for the custom business day offset.
    def test_repr(self, offset, offset2):
        # [FUNCTION DOC]
        # Default prints as a custom business day, step two as two custom business days. Adding a one-day timedelta gives a representation that starts with the business day class name and its offset field.
            assert repr(offset) == "<CustomBusinessDay>"
        assert repr(offset2) == "<2 * CustomBusinessDays>"

        expected = "<BusinessDay: offset=datetime.timedelta(days=1)>"
        assert repr(offset + timedelta(1)) == expected

    def test_holidays(self):
        # [FUNCTION DOC]
        # Builds an offset with three holidays given as a string, a datetime and a NumPy datetime64 (May 1 of 2012-2014). For each year, April 30 plus the offset skips the holiday and lands on May 2.
        # Define a TradingDay offset
        holidays = ["2012-05-01", datetime(2013, 5, 1), np.datetime64("2014-05-01")]
        tday = CDay(holidays=holidays)
        for year in range(2012, 2015):
            dt = datetime(year, 4, 30)
            xp = datetime(year, 5, 2)
            rs = dt + tday
            assert rs == xp

    def test_weekmask(self):
        # [FUNCTION DOC]
        # Three week masks all describing a Friday-Saturday weekend (Saudi-style Thursday-Friday weekend given by names, UAE bit string, Egypt list of flags). Adding one offset to 2013-05-01 gives the expected day (2013-05-04 for the Saudi mask, 2013-05-02 for the other two), and adding two gives 2013-05-05 for all three.
            weekmask_saudi = "Sat Sun Mon Tue Wed"  # Thu-Fri Weekend
        weekmask_uae = "1111001"  # Fri-Sat Weekend
        weekmask_egypt = [1, 1, 1, 1, 0, 0, 1]  # Fri-Sat Weekend
        bday_saudi = CDay(weekmask=weekmask_saudi)
        bday_uae = CDay(weekmask=weekmask_uae)
        bday_egypt = CDay(weekmask=weekmask_egypt)
        dt = datetime(2013, 5, 1)
        xp_saudi = datetime(2013, 5, 4)
        xp_uae = datetime(2013, 5, 2)
        xp_egypt = datetime(2013, 5, 2)
        assert xp_saudi == dt + bday_saudi
        assert xp_uae == dt + bday_uae
        assert xp_egypt == dt + bday_egypt
        xp2 = datetime(2013, 5, 5)
        assert xp2 == dt + 2 * bday_saudi
        assert xp2 == dt + 2 * bday_uae
        assert xp2 == dt + 2 * bday_egypt

    def test_weekmask_and_holidays(self):
        # [FUNCTION DOC]
        # Combines a Sunday-to-Thursday work week with the three holidays. Adding two offsets to 2013-04-30 skips the weekend and May 1 holiday and lands on 2013-05-05.
        weekmask_egypt = "Sun Mon Tue Wed Thu"  # Fri-Sat Weekend
        holidays = ["2012-05-01", datetime(2013, 5, 1), np.datetime64("2014-05-01")]
        bday_egypt = CDay(holidays=holidays, weekmask=weekmask_egypt)
        dt = datetime(2013, 4, 30)
        xp_egypt = datetime(2013, 5, 5)
        assert xp_egypt == dt + 2 * bday_egypt

    @pytest.mark.filterwarnings("ignore:Non:pandas.errors.PerformanceWarning")
    def test_calendar(self):
        # [FUNCTION DOC]
        # Filtered for a PerformanceWarning about non-vectorized operations. Using the US federal holiday calendar, adding one offset to Friday 2014-01-17 gives Tuesday 2014-01-21 (skipping the weekend and Martin Luther King day on the 20th).
        calendar = USFederalHolidayCalendar()
        dt = datetime(2014, 1, 17)
        assert_offset_equal(CDay(calendar=calendar), dt, datetime(2014, 1, 21))

    def test_roundtrip_pickle(self, offset, offset2, temp_file):
        # [FUNCTION DOC]
        # Uses the temp_file fixture. Serializes and restores the default offset, step two offset and a doubled offset through the round trip helper; each restored object equals the original.
        def _check_roundtrip(obj):
            # [FUNCTION DOC]
            # HELPER: nested function for test_roundtrip_pickle; round-trips the given object through pickle using the enclosing temp file and asserts equality.
            unpickled = tm.round_trip_pickle(obj, temp_file)
            assert unpickled == obj

        _check_roundtrip(offset)
        _check_roundtrip(offset2)
        _check_roundtrip(offset * 2)

    def test_pickle_compat_0_14_1(self, datapath):
        # [FUNCTION DOC]
        # Loads a stored pickle written by version 0.14.1 (found via the data-path fixture) and checks it equals a freshly built offset with four identical holidays (2013-01-01).
        hdays = [datetime(2013, 1, 1) for ele in range(4)]
        pth = datapath("tseries", "offsets", "data", "cday-0.14.1.pickle")
        cday0_14_1 = read_pickle(pth)
        cday = CDay(holidays=hdays)
        assert cday == cday0_14_1
