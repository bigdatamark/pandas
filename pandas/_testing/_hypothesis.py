"""
Hypothesis data generator helpers.
"""

# [MODULE DOC]
# HELPER: test support module (internal testing package) that defines reusable
# Hypothesis strategies for property-based tests; it contains only module-level
# constants and no functions or classes. Requires the hypothesis library.
# Public contents: OPTIONAL_INTS, OPTIONAL_FLOATS, OPTIONAL_TEXT,
# OPTIONAL_DICTS and OPTIONAL_LISTS (each a strategy for a list of between 3
# and 10 items that may be none: integers, floats, text, dictionaries mapping
# text to integers, and lists of text respectively); OPTIONAL_ONE_OF_ALL (one
# of those five); DATETIME_NO_TZ (naive datetimes; on Windows limited to start
# at 1900-01-01); DATETIME_JAN_1_1900_OPTIONAL_TZ (datetimes whose minimum and
# maximum are both 1900-01-01, with no timezone or a dateutil or zoneinfo
# timezone); DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ (naive datetimes within the
# range of the pandas timestamp, converted without warnings); INT_NEG_999_TO_POS_999
# (integers from minus 999 to 999); and YQM_OFFSET (one of the month, quarter and
# year begin or end offsets, plain and business variants).
# Relationships: the strategies for the offset types are registered in the
# test configuration module, because the types do not carry enough information
# to infer how to build them; this module only requests them by type.
# Import-time side effects: builds the strategy objects.
from datetime import datetime

from hypothesis import strategies as st
from hypothesis.extra.dateutil import timezones as dateutil_timezones

from pandas.compat import is_platform_windows

import pandas as pd

from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
)

OPTIONAL_INTS = st.lists(st.one_of(st.integers(), st.none()), max_size=10, min_size=3)

OPTIONAL_FLOATS = st.lists(st.one_of(st.floats(), st.none()), max_size=10, min_size=3)

OPTIONAL_TEXT = st.lists(st.one_of(st.none(), st.text()), max_size=10, min_size=3)

OPTIONAL_DICTS = st.lists(
    st.one_of(st.none(), st.dictionaries(st.text(), st.integers())),
    max_size=10,
    min_size=3,
)

OPTIONAL_LISTS = st.lists(
    st.one_of(st.none(), st.lists(st.text(), max_size=10, min_size=3)),
    max_size=10,
    min_size=3,
)

OPTIONAL_ONE_OF_ALL = st.one_of(
    OPTIONAL_DICTS, OPTIONAL_FLOATS, OPTIONAL_INTS, OPTIONAL_LISTS, OPTIONAL_TEXT
)

if is_platform_windows():
    DATETIME_NO_TZ = st.datetimes(min_value=datetime(1900, 1, 1))
else:
    DATETIME_NO_TZ = st.datetimes()

DATETIME_JAN_1_1900_OPTIONAL_TZ = st.datetimes(  # pyright: ignore[reportCallIssue]
    min_value=pd.Timestamp(1900, 1, 1).to_pydatetime(),  # pyright: ignore[reportArgumentType]
    max_value=pd.Timestamp(1900, 1, 1).to_pydatetime(),  # pyright: ignore[reportArgumentType]
    timezones=st.one_of(st.none(), dateutil_timezones(), st.timezones()),
)

DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ = st.datetimes(
    min_value=pd.Timestamp.min.to_pydatetime(warn=False),
    max_value=pd.Timestamp.max.to_pydatetime(warn=False),
)

INT_NEG_999_TO_POS_999 = st.integers(-999, 999)

# The strategy for each type is registered in conftest.py, as they don't carry
# enough runtime information (e.g. type hints) to infer how to build them.
YQM_OFFSET = st.one_of(
    *map(
        st.from_type,
        [
            MonthBegin,
            MonthEnd,
            BMonthBegin,
            BMonthEnd,
            QuarterBegin,
            QuarterEnd,
            BQuarterBegin,
            BQuarterEnd,
            YearBegin,
            YearEnd,
            BYearBegin,
            BYearEnd,
        ],
    )
)
