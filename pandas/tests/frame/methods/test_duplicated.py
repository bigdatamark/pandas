# [MODULE DOC]
# Tests for the duplicate-row flagging method of the frame class (duplicated): subset
# validation, keep modes, missing values, wide frames, empty frames and datetime data.
# Uses the keep fixture and the testing helpers module. All tests are module-level
# functions. No import-time side effects.

import re
import sys

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


@pytest.mark.parametrize("subset", ["a", ["a"], ["a", "B"]])
def test_duplicated_with_misspelled_column_name(subset):
    # [FUNCTION DOC]
    # Parametrized over a string, a list and a mixed list naming a nonexistent
    # lower-case column. Raises a key error whose message shows the index of missing
    # labels.
    # GH 19730
    df = DataFrame({"A": [0, 0, 1], "B": [0, 0, 1], "C": [0, 0, 1]})
    msg = re.escape("Index(['a'], dtype=")

    with pytest.raises(KeyError, match=msg):
        df.duplicated(subset)


def test_duplicated_implemented_no_recursion():
    # [FUNCTION DOC]
    # Regression for wide frames: with a very low recursion limit (restored afterwards),
    # flagging duplicates on a 10 by 1000 integer frame must not fail and returns a
    # boolean series.
    # gh-21524
    # Ensure duplicated isn't implemented using recursion that
    # can fail on wide frames
    df = DataFrame(np.random.default_rng(2).integers(0, 1000, (10, 1000)))
    rec_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        result = df.duplicated()
    finally:
        sys.setrecursionlimit(rec_limit)

    # Then duplicates produce the bool Series as a result and don't fail during
    # calculation. Actual values doesn't matter here, though usually it's all
    # False in this case
    assert isinstance(result, Series)
    assert result.dtype == np.bool_


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, False, True])),
        ("last", Series([True, True, False, False, False])),
        (False, Series([True, True, True, False, True])),
    ],
)
def test_duplicated_keep(keep, expected):
    # [FUNCTION DOC]
    # Parametrized over keep (first, last, false) with expected boolean series. A frame
    # with an integer and a string column marks the expected positions.
    df = DataFrame({"A": [0, 1, 1, 2, 0], "B": ["a", "b", "b", "c", "a"]})

    result = df.duplicated(keep=keep)
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(reason="GH#21720; nan/None falsely considered equal")
@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, False, True])),
        ("last", Series([True, True, False, False, False])),
        (False, Series([True, True, True, False, True])),
    ],
)
def test_duplicated_nan_none(keep, expected):
    # [FUNCTION DOC]
    # Expected failure (known issue about missing values being considered equal).
    # Parametrized over keep modes with expected series; frame of object dtype with NaN
    # and None mixed.
    df = DataFrame({"C": [np.nan, 3, 3, None, np.nan], "x": 1}, dtype=object)

    result = df.duplicated(keep=keep)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("subset", [None, ["A", "B"], "A"])
def test_duplicated_subset(subset, keep):
    # [FUNCTION DOC]
    # Parametrized over subset (none, list, string) and takes keep fixture. Flagging
    # with a subset equals flagging on the sub-frame selected by those columns (a string
    # subset is converted to a one-element list for the expected frame).
    df = DataFrame(
        {
            "A": [0, 1, 1, 2, 0],
            "B": ["a", "b", "b", "c", "a"],
            "C": [np.nan, 3, 3, None, np.nan],
        }
    )

    if subset is None:
        subset = list(df.columns)
    elif isinstance(subset, str):
        # need to have a DataFrame, not a Series
        # -> select columns with singleton list, not string
        subset = [subset]

    expected = df[subset].duplicated(keep=keep)
    result = df.duplicated(keep=keep, subset=subset)
    tm.assert_series_equal(result, expected)


def test_duplicated_on_empty_frame():
    # [FUNCTION DOC]
    # A frame with columns but no rows: boolean mask from duplicated can filter the
    # frame and gives an equal empty frame.
    # GH 25184

    df = DataFrame(columns=["a", "b"])
    dupes = df.duplicated("a")

    result = df[dupes]
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


def test_frame_datetime64_duplicated():
    # [FUNCTION DOC]
    # A frame with a datetime column and a constant column has no duplicates; the series
    # version on the datetime column has no duplicates either (checked by negating the
    # boolean series).
    dates = date_range("2010-07-01", end="2010-08-05")

    tst = DataFrame({"symbol": "AAA", "date": dates})
    result = tst.duplicated(["date", "symbol"])
    assert (-result).all()

    tst = DataFrame({"date": dates})
    result = tst.date.duplicated()
    assert (-result).all()
