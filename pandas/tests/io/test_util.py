# [MODULE DOC]
# Tests for the arrow table conversion utility in the io utilities module.
# Covers timezone normalization when converting an Arrow table to a DataFrame:
# timezone-aware data in columns, the row index, the column labels and a
# multi-level index must all end up with standard-library zone info timezone objects.
# Requires pyarrow and pytz (skipped otherwise). A module-level mark ignores the
# deprecation warning about passing a block manager to the frame constructor.
import zoneinfo

import pytest

from pandas.compat import (
    pa_version_under18p0,
    pa_version_under23p0,
)

import pandas as pd
import pandas._testing as tm

from pandas.io._util import arrow_table_to_pandas

pa = pytest.importorskip("pyarrow")
pytz = pytest.importorskip("pytz")


pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


def test_arrow_table_to_pandas_normalize_timezones():
    # [FUNCTION DOC]
    # Verifies that converting an Arrow table built from a frame with a timezone-aware
    # column and a timezone-aware datetime index yields a frame equal to the original
    # (index frequency cleared), and that both timezones are standard-library zone info objects.
    df = pd.DataFrame(
        {"ts": pd.date_range("2024-03-01", periods=2, tz="America/New_York")},
        index=pd.date_range("2024-01-01", periods=2, tz="US/Eastern"),
    )
    expected = df.copy()
    expected.index = expected.index._with_freq(None)

    table = pa.Table.from_pandas(df)
    result = arrow_table_to_pandas(table)

    tm.assert_frame_equal(result, expected)
    assert isinstance(result["ts"].dtype.tz, zoneinfo.ZoneInfo)
    assert isinstance(result.index.tz, zoneinfo.ZoneInfo)


def test_arrow_table_to_pandas_normalize_timezones_columns():
    # [FUNCTION DOC]
    # Verifies normalization when the timezone-aware values are the column labels:
    # the round-tripped frame equals the original (columns cast to nanosecond unit when
    # the installed pyarrow is in the range that loses the unit) and the columns timezone
    # is a zone info object.
    df = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=pd.date_range("2024-02-01", periods=2, tz="Europe/Berlin"),
    )
    expected = df.copy()

    table = pa.Table.from_pandas(df)
    result = arrow_table_to_pandas(table)

    if pa_version_under23p0 and not pa_version_under18p0:
        expected.columns = expected.columns.as_unit("ns")

    tm.assert_frame_equal(result, expected)
    assert isinstance(result.columns.tz, zoneinfo.ZoneInfo)


def test_arrow_table_to_pandas_normalize_timezones_multiindex():
    # [FUNCTION DOC]
    # Verifies normalization with a two-level index whose second level is timezone-aware
    # and also kept as a column: the round trip equals the original frame, and both the
    # column dtype timezone and the index level timezone are zone info objects.
    df = pd.DataFrame(
        {"ts": pd.date_range("2024-03-01", periods=2, tz="America/New_York")},
        index=pd.Index([1, 2], name="index"),
    ).set_index("ts", append=True, drop=False)
    expected = df.copy()

    table = pa.Table.from_pandas(df)
    result = arrow_table_to_pandas(table)

    tm.assert_frame_equal(result, expected)
    assert isinstance(result["ts"].dtype.tz, zoneinfo.ZoneInfo)
    assert isinstance(result.index.get_level_values("ts").tz, zoneinfo.ZoneInfo)
