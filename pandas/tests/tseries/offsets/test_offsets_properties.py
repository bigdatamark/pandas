"""
Behavioral based tests for offsets and date_range.

This file is adapted from https://github.com/pandas-dev/pandas/pull/18761 -
which was more ambitious but less idiomatic in its use of Hypothesis.

You may wish to consult the previous version for inspiration on further
tests, or when trying to pin down the bugs exposed by the tests below.
"""

# [MODULE DOC]
# Property-based (Hypothesis) tests of offsets: agreement of class-specific on-offset implementations
# with the generic definition, and preservation of wall-clock hour when shifting a timezone-aware range across
# a daylight saving transition. Uses strategies for datetimes from 1900 with optional time zones and for
# year/quarter/month offsets from the pandas testing hypothesis module. Both tests are marked slow.

import zoneinfo

from hypothesis import (
    assume,
    given,
)
import pytest

from pandas.compat import WASM

import pandas as pd
from pandas._testing._hypothesis import (
    DATETIME_JAN_1_1900_OPTIONAL_TZ,
    YQM_OFFSET,
)

# ----------------------------------------------------------------
# Offset-specific behaviour tests


@pytest.mark.slow
@pytest.mark.arm_slow
@given(DATETIME_JAN_1_1900_OPTIONAL_TZ, YQM_OFFSET)
def test_on_offset_implementations(dt, offset):
    # [FUNCTION DOC]
    # Property test over a datetime (1900 onward, optional time zone) and a year/quarter/month offset. Rejects
    # normalizing offsets, and a known flaky combination (WASM builds, the Indian/Cocos zone, month begin offset).
    # Computes the value obtained by adding then subtracting the offset; if that raises a value error (nonexistent
    # or ambiguous local time, GH41906) the example is discarded as invalid. Asserts the offset's on-offset check
    # equals whether that round trip returns the original datetime. Marked slow and slow-on-ARM.
    assume(not offset.normalize)
    # This case is flaky in CI 2024-11-04
    assume(
        not (
            WASM
            and isinstance(dt.tzinfo, zoneinfo.ZoneInfo)
            and dt.tzinfo.key == "Indian/Cocos"
            and isinstance(offset, pd.offsets.MonthBegin)
        )
    )
    # check that the class-specific implementations of is_on_offset match
    # the general case definition:
    #   (dt + offset) - offset == dt
    try:
        compare = (dt + offset) - offset
    except ValueError:
        # When dt + offset does not exist or is DST-ambiguous, assume(False) to
        # indicate to hypothesis that this is not a valid test case
        # DST-ambiguous example (GH41906):
        # dt = datetime.datetime(1900, 1, 1, tzinfo=ZoneInfo('Africa/Kinshasa'))
        # offset = MonthBegin(66)
        assume(False)

    assert offset.is_on_offset(dt) == (compare == dt)


@pytest.mark.slow
@given(YQM_OFFSET)
def test_shift_across_dst(offset):
    # [FUNCTION DOC]
    # Regression test (GH#18319) as a property test over year/quarter/month offsets (non-normalizing only). Builds a daily
    # US/Eastern range at noon from 2017-10-30 to 2017-11-06 crossing the DST end, asserts all hours are 12, then adds the offset
    # and asserts all resulting hours are still 12 (timezone correctly normalized, hour not changed). Marked slow.
    # GH#18319 check that 1) timezone is correctly normalized and
    # 2) that hour is not incorrectly changed by this normalization
    assume(not offset.normalize)

    # Note that dti includes a transition across DST boundary
    dti = pd.date_range(
        start="2017-10-30 12:00:00", end="2017-11-06", freq="D", tz="US/Eastern"
    )
    assert (dti.hour == 12).all()  # we haven't screwed up yet

    res = dti + offset
    assert (res.hour == 12).all()
