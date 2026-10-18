# [MODULE DOC]
# Tests for multi-level indexes with period or datetime elements: label lookup with a period level and datetime keys, and construction of an empty frame with datetime-valued column levels.

from datetime import datetime

import numpy as np

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    period_range,
    to_datetime,
)
import pandas._testing as tm


def test_multiindex_period_datetime():
    # [FUNCTION DOC]
    # A series on a two-level index (strings, monthly periods): selecting with a string and a period gives the first element; selecting with the string and a standard datetime for the same month gives the same element (no exception).
    # GH4861, using datetime in period of multiindex raises exception

    idx1 = Index(["a", "a", "a", "b", "b"])
    idx2 = period_range("2012-01", periods=len(idx1), freq="M")
    s = Series(np.random.default_rng(2).standard_normal(len(idx1)), [idx1, idx2])

    # try Period as index
    expected = s.iloc[0]
    result = s.loc["a", Period("2012-01")]
    assert result == expected

    # try datetime as index
    result = s.loc["a", datetime(2012, 1, 1)]
    assert result == expected


def test_multiindex_datetime_columns():
    # [FUNCTION DOC]
    # An empty frame whose columns are a two-level index built from tuples of two datetimes equals one whose columns come from separate arrays of the same datetimes, with the same level names.
    # GH35015, using datetime as column indices raises exception

    mi = MultiIndex.from_tuples(
        [(to_datetime("02/29/2020"), to_datetime("03/01/2020"))], names=["a", "b"]
    )

    df = DataFrame([], columns=mi)

    expected_df = DataFrame(
        [],
        columns=MultiIndex.from_arrays(
            [[to_datetime("02/29/2020")], [to_datetime("03/01/2020")]], names=["a", "b"]
        ),
    )

    tm.assert_frame_equal(df, expected_df)
