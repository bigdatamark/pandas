"""
Tests for the following offsets:
- SemiMonthBegin
- SemiMonthEnd
- MonthBegin
- MonthEnd
"""

# [MODULE DOC]
# Tests for the semi-monthly offsets (begin and end, with a configurable day of month) and the
# monthly offsets (month begin and month end). Covers walking through a whole year, table-driven
# application of offsets with many counts, vectorized addition to indexes and series (including
# timezone-aware values and absence of warnings), on-offset checks, and normalization.
# Uses shared helpers from the offsets common test module; no import-time side effects other than
# building case tables on the test classes.

from __future__ import annotations

from datetime import datetime

import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    MonthBegin,
    MonthEnd,
    SemiMonthBegin,
    SemiMonthEnd,
)

from pandas import (
    DatetimeIndex,
    Series,
    _testing as tm,
)
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)


class TestSemiMonthEnd:
    # [CLASS DOC]
    # Tests for the semi-monthly end offset, which anchors on a configurable mid-month day (default the 15th) and the last day of the month.
    def test_offset_whole_year(self):
        # [FUNCTION DOC]
        # Checks that applying the default offset to each date in a hand-written chronological series of mid-month
        # and month-end dates for 2008 gives the next date in the series. Then adds the offset to an index of all but
        # the last date and asserts, with no warning emitted (GH#22535), that the result equals an index of all but the first date.
        dates = (
            datetime(2007, 12, 31),
            datetime(2008, 1, 15),
            datetime(2008, 1, 31),
            datetime(2008, 2, 15),
            datetime(2008, 2, 29),
            datetime(2008, 3, 15),
            datetime(2008, 3, 31),
            datetime(2008, 4, 15),
            datetime(2008, 4, 30),
            datetime(2008, 5, 15),
            datetime(2008, 5, 31),
            datetime(2008, 6, 15),
            datetime(2008, 6, 30),
            datetime(2008, 7, 15),
            datetime(2008, 7, 31),
            datetime(2008, 8, 15),
            datetime(2008, 8, 31),
            datetime(2008, 9, 15),
            datetime(2008, 9, 30),
            datetime(2008, 10, 15),
            datetime(2008, 10, 31),
            datetime(2008, 11, 15),
            datetime(2008, 11, 30),
            datetime(2008, 12, 15),
            datetime(2008, 12, 31),
        )

        for base, exp_date in zip(dates[:-1], dates[1:], strict=True):
            assert_offset_equal(SemiMonthEnd(), base, exp_date)

        # ensure .apply_index works as expected
        shift = DatetimeIndex(dates[:-1])
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = SemiMonthEnd() + shift

        exp = DatetimeIndex(dates[1:])
        tm.assert_index_equal(result, exp)

    offset_cases = []
    offset_cases.append(
        (
            SemiMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 15),
                datetime(2008, 1, 15): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 15),
                datetime(2006, 12, 14): datetime(2006, 12, 15),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 15),
                datetime(2007, 1, 1): datetime(2007, 1, 15),
                datetime(2006, 12, 1): datetime(2006, 12, 15),
                datetime(2006, 12, 15): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(day_of_month=20),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 20),
                datetime(2008, 1, 15): datetime(2008, 1, 20),
                datetime(2008, 1, 21): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 20),
                datetime(2006, 12, 14): datetime(2006, 12, 20),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 20),
                datetime(2007, 1, 1): datetime(2007, 1, 20),
                datetime(2006, 12, 1): datetime(2006, 12, 20),
                datetime(2006, 12, 15): datetime(2006, 12, 20),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 15),
                datetime(2008, 1, 16): datetime(2008, 1, 31),
                datetime(2008, 1, 15): datetime(2008, 1, 15),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 15),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(0, day_of_month=16),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 16),
                datetime(2008, 1, 16): datetime(2008, 1, 16),
                datetime(2008, 1, 15): datetime(2008, 1, 16),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 16),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2007, 1, 15),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2007, 1, 16): datetime(2007, 2, 15),
                datetime(2006, 11, 1): datetime(2006, 11, 30),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 6, 15),
                datetime(2008, 12, 31): datetime(2008, 12, 15),
                datetime(2006, 12, 29): datetime(2006, 12, 15),
                datetime(2006, 12, 30): datetime(2006, 12, 15),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(-1, day_of_month=4),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2007, 1, 4): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 6, 4),
                datetime(2008, 12, 31): datetime(2008, 12, 4),
                datetime(2006, 12, 5): datetime(2006, 12, 4),
                datetime(2006, 12, 30): datetime(2006, 12, 4),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 15),
                datetime(2008, 6, 30): datetime(2008, 5, 31),
                datetime(2008, 3, 15): datetime(2008, 2, 15),
                datetime(2008, 12, 31): datetime(2008, 11, 30),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 14): datetime(2006, 11, 15),
                datetime(2007, 1, 1): datetime(2006, 12, 15),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the class-level case table (default, custom day of month, n of 0, 2, minus 1, minus 2 and
        # combinations). For each base date in the mapping, asserts the offset gives the expected date.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("case", offset_cases)
    def test_apply_index(self, case):
        # [FUNCTION DOC]
        # Parametrized over the same case table. Builds an index from the base dates, adds the offset to it
        # (no warning allowed), and asserts the result equals the index of expected dates.
        # https://github.com/pandas-dev/pandas/issues/34580
        offset, cases = case
        shift = DatetimeIndex(cases.keys())
        exp = DatetimeIndex(cases.values())

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = offset + shift
        tm.assert_index_equal(result, exp)

    on_offset_cases = [
        (datetime(2007, 12, 31), True),
        (datetime(2007, 12, 15), True),
        (datetime(2007, 12, 14), False),
        (datetime(2007, 12, 1), False),
        (datetime(2008, 2, 29), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over (date, expected) pairs; asserts the default semi-month-end offset's on-offset check matches.
        dt, expected = case
        assert_is_on_offset(SemiMonthEnd(), dt, expected)

    @pytest.mark.parametrize("klass", [Series, DatetimeIndex])
    def test_vectorized_offset_addition(self, klass):
        # [FUNCTION DOC]
        # Parametrized over series and datetime index. For two timezone-aware named inputs (mid-month and
        # early-month values with and without a time of day), adds the offset on either side (no warning
        # allowed) and asserts both orders give the expected shifted values (to month end, or to mid month) with the name preserved.
        shift = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = shift + SemiMonthEnd()
            result2 = SemiMonthEnd() + shift

        exp = klass(
            [
                Timestamp("2000-01-31 00:15:00", tz="US/Central"),
                Timestamp("2000-02-29", tz="US/Central"),
            ],
            name="a",
        )
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

        shift = klass(
            [
                Timestamp("2000-01-01 00:15:00", tz="US/Central"),
                Timestamp("2000-02-01", tz="US/Central"),
            ],
            name="a",
        )

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = shift + SemiMonthEnd()
            result2 = SemiMonthEnd() + shift

        exp = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)


class TestSemiMonthBegin:
    # [CLASS DOC]
    # Tests for the semi-monthly begin offset, which anchors on the first of the month and a configurable day (default the 15th).
    def test_offset_whole_year(self):
        # [FUNCTION DOC]
        # Same structure as the end-offset version: stepping through a chronological series of first-of-month and
        # mid-month dates for 2008 one step at a time, and then checking vectorized addition to an index without warnings.
        dates = (
            datetime(2007, 12, 15),
            datetime(2008, 1, 1),
            datetime(2008, 1, 15),
            datetime(2008, 2, 1),
            datetime(2008, 2, 15),
            datetime(2008, 3, 1),
            datetime(2008, 3, 15),
            datetime(2008, 4, 1),
            datetime(2008, 4, 15),
            datetime(2008, 5, 1),
            datetime(2008, 5, 15),
            datetime(2008, 6, 1),
            datetime(2008, 6, 15),
            datetime(2008, 7, 1),
            datetime(2008, 7, 15),
            datetime(2008, 8, 1),
            datetime(2008, 8, 15),
            datetime(2008, 9, 1),
            datetime(2008, 9, 15),
            datetime(2008, 10, 1),
            datetime(2008, 10, 15),
            datetime(2008, 11, 1),
            datetime(2008, 11, 15),
            datetime(2008, 12, 1),
            datetime(2008, 12, 15),
        )

        for base, exp_date in zip(dates[:-1], dates[1:], strict=True):
            assert_offset_equal(SemiMonthBegin(), base, exp_date)

        # ensure .apply_index works as expected
        shift = DatetimeIndex(dates[:-1])
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = SemiMonthBegin() + shift

        exp = DatetimeIndex(dates[1:])
        tm.assert_index_equal(result, exp)

    offset_cases = [
        (
            SemiMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 15),
                datetime(2008, 1, 15): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 14): datetime(2006, 12, 15),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2007, 1, 1): datetime(2007, 1, 15),
                datetime(2006, 12, 1): datetime(2006, 12, 15),
                datetime(2006, 12, 15): datetime(2007, 1, 1),
            },
        ),
        (
            SemiMonthBegin(day_of_month=20),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 20),
                datetime(2008, 1, 15): datetime(2008, 1, 20),
                datetime(2008, 1, 21): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 14): datetime(2006, 12, 20),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2007, 1, 1): datetime(2007, 1, 20),
                datetime(2006, 12, 1): datetime(2006, 12, 20),
                datetime(2006, 12, 15): datetime(2006, 12, 20),
            },
        ),
        (
            SemiMonthBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 16): datetime(2008, 2, 1),
                datetime(2008, 1, 15): datetime(2008, 1, 15),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 2): datetime(2006, 12, 15),
                datetime(2007, 1, 1): datetime(2007, 1, 1),
            },
        ),
        (
            SemiMonthBegin(0, day_of_month=16),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 16): datetime(2008, 1, 16),
                datetime(2008, 1, 15): datetime(2008, 1, 16),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2007, 1, 5): datetime(2007, 1, 16),
                datetime(2007, 1, 1): datetime(2007, 1, 1),
            },
        ),
        (
            SemiMonthBegin(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 15),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 15),
                datetime(2006, 12, 15): datetime(2007, 1, 15),
                datetime(2007, 1, 1): datetime(2007, 2, 1),
                datetime(2007, 1, 16): datetime(2007, 2, 15),
                datetime(2006, 11, 1): datetime(2006, 12, 1),
            },
        ),
        (
            SemiMonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 15),
                datetime(2008, 6, 30): datetime(2008, 6, 15),
                datetime(2008, 6, 14): datetime(2008, 6, 1),
                datetime(2008, 12, 31): datetime(2008, 12, 15),
                datetime(2006, 12, 29): datetime(2006, 12, 15),
                datetime(2006, 12, 15): datetime(2006, 12, 1),
                datetime(2007, 1, 1): datetime(2006, 12, 15),
            },
        ),
        (
            SemiMonthBegin(-1, day_of_month=4),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 4),
                datetime(2007, 1, 4): datetime(2007, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 6, 4),
                datetime(2008, 12, 31): datetime(2008, 12, 4),
                datetime(2006, 12, 5): datetime(2006, 12, 4),
                datetime(2006, 12, 30): datetime(2006, 12, 4),
                datetime(2006, 12, 2): datetime(2006, 12, 1),
                datetime(2007, 1, 1): datetime(2006, 12, 4),
            },
        ),
        (
            SemiMonthBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 6, 30): datetime(2008, 6, 1),
                datetime(2008, 6, 14): datetime(2008, 5, 15),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 12, 15): datetime(2006, 11, 15),
                datetime(2007, 1, 1): datetime(2006, 12, 1),
            },
        ),
    ]

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the class-level case table (default, day of month 20, n of 0 with day 16, n of 2, minus 1,
        # minus 1 with day 4, minus 2); asserts each base date maps to the expected date.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("case", offset_cases)
    def test_apply_index(self, case):
        # [FUNCTION DOC]
        # Parametrized over the case table; adds the offset to an index of base dates (no warning allowed) and asserts the expected index.
        offset, cases = case
        shift = DatetimeIndex(cases.keys())

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = offset + shift

        exp = DatetimeIndex(cases.values())
        tm.assert_index_equal(result, exp)

    on_offset_cases = [
        (datetime(2007, 12, 1), True),
        (datetime(2007, 12, 15), True),
        (datetime(2007, 12, 14), False),
        (datetime(2007, 12, 31), False),
        (datetime(2008, 2, 15), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over (date, expected) pairs; asserts the default semi-month-begin offset's on-offset check matches.
        dt, expected = case
        assert_is_on_offset(SemiMonthBegin(), dt, expected)

    @pytest.mark.parametrize("klass", [Series, DatetimeIndex])
    def test_vectorized_offset_addition(self, klass):
        # [FUNCTION DOC]
        # Parametrized over series and datetime index. Adds the offset on either side to timezone-aware named
        # inputs (no warning allowed) and asserts both orders give the expected next anchor dates with name preserved.
        shift = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = shift + SemiMonthBegin()
            result2 = SemiMonthBegin() + shift

        exp = klass(
            [
                Timestamp("2000-02-01 00:15:00", tz="US/Central"),
                Timestamp("2000-03-01", tz="US/Central"),
            ],
            name="a",
        )
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

        shift = klass(
            [
                Timestamp("2000-01-01 00:15:00", tz="US/Central"),
                Timestamp("2000-02-01", tz="US/Central"),
            ],
            name="a",
        )
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = shift + SemiMonthBegin()
            result2 = SemiMonthBegin() + shift

        exp = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)


class TestMonthBegin:
    # [CLASS DOC]
    # Tests for the month-begin offset. Class attribute holds the case table (n of 1, 0, 2, minus 1).
    offset_cases = []
    # NOTE: I'm not entirely happy with the logic here for Begin -ss
    # see thread 'offset conventions' on the ML
    offset_cases.append(
        (
            MonthBegin(),
            {
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 2, 1): datetime(2008, 3, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
                datetime(2007, 1, 31): datetime(2007, 2, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(0),
            {
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2006, 12, 3): datetime(2007, 1, 1),
                datetime(2007, 1, 31): datetime(2007, 2, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(2),
            {
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 1, 31): datetime(2008, 3, 1),
                datetime(2006, 12, 31): datetime(2007, 2, 1),
                datetime(2007, 12, 28): datetime(2008, 2, 1),
                datetime(2007, 1, 1): datetime(2007, 3, 1),
                datetime(2006, 11, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 5, 31): datetime(2008, 5, 1),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 1, 2): datetime(2006, 1, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the case table; asserts each base date maps to the expected month start.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestMonthEnd:
    # [CLASS DOC]
    # Tests for the month-end offset. Class attributes hold the case table and the on-offset table.
    def test_day_of_month(self):
        # [FUNCTION DOC]
        # Adding the default offset to the first of January 2007 gives January 31, and adding it again gives February 28.
        dt = datetime(2007, 1, 1)
        offset = MonthEnd()

        result = dt + offset
        assert result == Timestamp(2007, 1, 31)

        result = result + offset
        assert result == Timestamp(2007, 2, 28)

    def test_normalize(self):
        # [FUNCTION DOC]
        # Adding an offset with normalization enabled to a datetime with a time of day equals adding the plain
        # offset to the same datetime with the hour reset to midnight.
        dt = datetime(2007, 1, 1, 3)

        result = dt + MonthEnd(normalize=True)
        expected = dt.replace(hour=0) + MonthEnd()
        assert result == expected

    offset_cases = []
    offset_cases.append(
        (
            MonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2006, 12, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 3, 31),
                datetime(2006, 12, 29): datetime(2007, 1, 31),
                datetime(2006, 12, 31): datetime(2007, 2, 28),
                datetime(2007, 1, 1): datetime(2007, 2, 28),
                datetime(2006, 11, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 5, 31),
                datetime(2008, 12, 31): datetime(2008, 11, 30),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 30): datetime(2006, 11, 30),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the case table (n of 1, 0, 2, minus 1); asserts each base date maps to the expected month end.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (MonthEnd(), datetime(2007, 12, 31), True),
        (MonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over (offset, date, expected); asserts the on-offset check matches the expected boolean.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
