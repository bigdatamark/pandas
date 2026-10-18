# [MODULE DOC]
# Pytest fixtures local to the DataFrame tests: random-valued frames of various column
# dtypes and indexes (datetime, float plus string, mixed float, mixed int, time zones).
# All random data come from a generator seeded with a fixed value, so fixtures are
# deterministic. No import-time side effects.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    NaT,
    date_range,
)


@pytest.fixture
def datetime_frame() -> DataFrame:
    # [FUNCTION DOC]
    # Fixture returning a ten-row, four-column (labels A to D) frame of standard normal floats
    # indexed by a business-day date range starting 2000-01-01.
    """
    Fixture for DataFrame of floats with DatetimeIndex

    Columns are ['A', 'B', 'C', 'D']
    """
    return DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )


@pytest.fixture
def float_string_frame():
    # [FUNCTION DOC]
    # Fixture returning a thirty-row frame with float columns A to D (standard normal data)
    # indexed by object-dtype strings foo_0 through foo_29, plus a fifth column named foo whose
    # every value is the string bar.
    """
    Fixture for DataFrame of floats and strings with index of unique strings

    Columns are ['A', 'B', 'C', 'D', 'foo'].
    """
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD")),
    )
    df["foo"] = "bar"
    return df


@pytest.fixture
def mixed_float_frame():
    # [FUNCTION DOC]
    # Fixture returning a thirty-row frame of columns A to D with random floats in the unit
    # interval, string index foo_0 to foo_29. Columns A to C start as 32-bit floats and D as
    # 64-bit; column C is then cast to 16-bit float (numpy random cannot generate it).
    """
    Fixture for DataFrame of different float types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    df = DataFrame(
        {
            col: np.random.default_rng(2).random(30, dtype=dtype)
            for col, dtype in zip(
                list("ABCD"), ["float32", "float32", "float32", "float64"]
            )
        },
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
    )
    # not supported by numpy random
    df["C"] = df["C"].astype("float16")
    return df


@pytest.fixture
def mixed_int_frame():
    # [FUNCTION DOC]
    # Fixture returning a thirty-row frame of columns A to D filled with ones, with dtypes
    # 32-bit int, 64-bit unsigned, 8-bit unsigned and 64-bit int respectively, and string
    # index foo_0 to foo_29.
    """
    Fixture for DataFrame of different int types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    return DataFrame(
        {
            col: np.ones(30, dtype=dtype)
            for col, dtype in zip(list("ABCD"), ["int32", "uint64", "uint8", "int64"])
        },
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
    )


@pytest.fixture
def timezone_frame():
    # [FUNCTION DOC]
    # Fixture returning a three-row frame with date columns A (naive), B (US/Eastern) and C
    # (CET), all daily from 2013-01-01 at nanosecond unit. The second-row values of B and C
    # are then set to missing (NaT).
    """
    Fixture for DataFrame of date_range Series with different time zones

    Columns are ['A', 'B', 'C']; some entries are missing

               A                         B                         C
    0 2013-01-01 2013-01-01 00:00:00-05:00 2013-01-01 00:00:00+01:00
    1 2013-01-02                       NaT                       NaT
    2 2013-01-03 2013-01-03 00:00:00-05:00 2013-01-03 00:00:00+01:00
    """
    df = DataFrame(
        {
            "A": date_range("20130101", periods=3, unit="ns"),
            "B": date_range("20130101", periods=3, tz="US/Eastern", unit="ns"),
            "C": date_range("20130101", periods=3, tz="CET", unit="ns"),
        }
    )
    df.iloc[1, 1] = NaT
    df.iloc[1, 2] = NaT
    return df
