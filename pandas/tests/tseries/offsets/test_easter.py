"""
Tests for the following offsets:
- Easter
"""

# [MODULE DOC]
# Tests for the Easter offset, which moves a date to the next or previous Easter Sunday. Covers the default method and the
# explicitly specified Western and Orthodox calendars, with positive and negative step counts. Uses the shared offsets assertion helper.
# No import-time side effects.
from __future__ import annotations

from datetime import datetime

from dateutil.easter import (
    EASTER_ORTHODOX,
    EASTER_WESTERN,
)
import pytest

from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries.offsets import Easter


class TestEaster:
    # [CLASS DOC]
    # Test collection for the Easter offset; three parametrized tests with identical shape.
    @pytest.mark.parametrize(
        "offset,date,expected",
        [
            (Easter(), datetime(2010, 1, 1), datetime(2010, 4, 4)),
            (Easter(), datetime(2010, 4, 5), datetime(2011, 4, 24)),
            (Easter(2), datetime(2010, 1, 1), datetime(2011, 4, 24)),
            (Easter(), datetime(2010, 4, 4), datetime(2011, 4, 24)),
            (Easter(2), datetime(2010, 4, 4), datetime(2012, 4, 8)),
            (-Easter(), datetime(2011, 1, 1), datetime(2010, 4, 4)),
            (-Easter(), datetime(2010, 4, 5), datetime(2010, 4, 4)),
            (-Easter(2), datetime(2011, 1, 1), datetime(2009, 4, 12)),
            (-Easter(), datetime(2010, 4, 4), datetime(2009, 4, 12)),
            (-Easter(2), datetime(2010, 4, 4), datetime(2008, 3, 23)),
        ],
    )
    def test_offset(self, offset, date, expected):
        # [FUNCTION DOC]
        # Parametrized over (offset, start date, expected date) for the default method (Western). Checks forward steps one and two, backward steps, and dates falling on or just after an Easter (April 4 2010 is Easter Sunday 2010, so adding one jumps to the 2011 Easter on April 24).
        assert_offset_equal(offset, date, expected)

    @pytest.mark.parametrize(
        "offset,date,expected",
        [
            (Easter(method=EASTER_WESTERN), datetime(2010, 1, 1), datetime(2010, 4, 4)),
            (
                Easter(method=EASTER_WESTERN),
                datetime(2010, 4, 5),
                datetime(2011, 4, 24),
            ),
            (
                Easter(2, method=EASTER_WESTERN),
                datetime(2010, 1, 1),
                datetime(2011, 4, 24),
            ),
            (
                Easter(method=EASTER_WESTERN),
                datetime(2010, 4, 4),
                datetime(2011, 4, 24),
            ),
            (
                Easter(2, method=EASTER_WESTERN),
                datetime(2010, 4, 4),
                datetime(2012, 4, 8),
            ),
            (
                -Easter(method=EASTER_WESTERN),
                datetime(2011, 1, 1),
                datetime(2010, 4, 4),
            ),
            (
                -Easter(method=EASTER_WESTERN),
                datetime(2010, 4, 5),
                datetime(2010, 4, 4),
            ),
            (
                -Easter(2, method=EASTER_WESTERN),
                datetime(2011, 1, 1),
                datetime(2009, 4, 12),
            ),
            (
                -Easter(method=EASTER_WESTERN),
                datetime(2010, 4, 4),
                datetime(2009, 4, 12),
            ),
            (
                -Easter(2, method=EASTER_WESTERN),
                datetime(2010, 4, 4),
                datetime(2008, 3, 23),
            ),
        ],
    )
    def test_western_easter_offset(self, offset, date, expected):
        # [FUNCTION DOC]
        # Same cases as the default test but with the Western method specified explicitly; results must be identical to the default (Easter dates 2008-03-23, 2009-04-12, 2010-04-04, 2011-04-24, 2012-04-08).
        assert_offset_equal(offset, date, expected)

    @pytest.mark.parametrize(
        "offset,date,expected",
        [
            (
                Easter(method=EASTER_ORTHODOX),
                datetime(2010, 1, 1),
                datetime(2010, 4, 4),
            ),
            (
                Easter(method=EASTER_ORTHODOX),
                datetime(2010, 4, 5),
                datetime(2011, 4, 24),
            ),
            (
                Easter(2, method=EASTER_ORTHODOX),
                datetime(2010, 1, 1),
                datetime(2011, 4, 24),
            ),
            (
                Easter(method=EASTER_ORTHODOX),
                datetime(2010, 4, 4),
                datetime(2011, 4, 24),
            ),
            (
                Easter(2, method=EASTER_ORTHODOX),
                datetime(2010, 4, 4),
                datetime(2012, 4, 15),
            ),
            (
                -Easter(method=EASTER_ORTHODOX),
                datetime(2011, 1, 1),
                datetime(2010, 4, 4),
            ),
            (
                -Easter(method=EASTER_ORTHODOX),
                datetime(2010, 4, 5),
                datetime(2010, 4, 4),
            ),
            (
                -Easter(2, method=EASTER_ORTHODOX),
                datetime(2011, 1, 1),
                datetime(2009, 4, 19),
            ),
            (
                -Easter(method=EASTER_ORTHODOX),
                datetime(2010, 4, 4),
                datetime(2009, 4, 19),
            ),
            (
                -Easter(2, method=EASTER_ORTHODOX),
                datetime(2010, 4, 4),
                datetime(2008, 4, 27),
            ),
        ],
    )
    def test_orthodox_easter_offset(self, offset, date, expected):
        # [FUNCTION DOC]
        # Same shape with the Orthodox method. Easter dates differ from Western in some years: 2008-04-27, 2009-04-19, 2010-04-04, 2011-04-24, 2012-04-15.
        assert_offset_equal(offset, date, expected)
