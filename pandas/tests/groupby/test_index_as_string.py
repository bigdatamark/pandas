# [MODULE DOC]
# Tests that index level names can be used as grouping keys by string (alone, in lists, or mixed with column names) and give the same result as explicit level groupers, for frames and series.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "key_strs,groupers",
    [
        ("inner", pd.Grouper(level="inner")),  # Index name
        (["inner"], [pd.Grouper(level="inner")]),  # List of index name
        (["B", "inner"], ["B", pd.Grouper(level="inner")]),  # Column and index
        (["inner", "B"], [pd.Grouper(level="inner"), "B"]),  # Index and column
    ],
)
@pytest.mark.parametrize("levels", [["inner"], ["inner", "outer"]])
def test_grouper_index_level_as_string(levels, key_strs, groupers):
    # [FUNCTION DOC]
    # Parametrized over index levels and keys: grouping a frame by string keys naming index levels and/or columns gives the same mean as using Grouper objects on the level; numeric_only is applied when a non-numeric column would be involved.
    frame = pd.DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 3, 1, 2, 3],
            "A": np.arange(6),
            "B": ["one", "one", "two", "two", "one", "one"],
        }
    )
    frame = frame.set_index(levels)
    if "B" not in key_strs or "outer" in frame.columns:
        result = frame.groupby(key_strs).mean(numeric_only=True)
        expected = frame.groupby(groupers).mean(numeric_only=True)
    else:
        result = frame.groupby(key_strs).mean()
        expected = frame.groupby(groupers).mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "levels",
    [
        "inner",
        "outer",
        "B",
        ["inner"],
        ["outer"],
        ["B"],
        ["inner", "outer"],
        ["outer", "inner"],
        ["inner", "outer", "B"],
        ["B", "outer", "inner"],
    ],
)
def test_grouper_index_level_as_string_series(levels):
    # [FUNCTION DOC]
    # Grouping a three-level-indexed series by each level name or lists of them (including column-like name B) gives the same mean as using level groupers.
    # Compute expected result
    df = pd.DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 3, 1, 2, 3],
            "A": np.arange(6),
            "B": ["one", "one", "two", "two", "one", "one"],
        }
    )
    series = df.set_index(["outer", "inner", "B"])["A"]
    if isinstance(levels, list):
        groupers = [pd.Grouper(level=lv) for lv in levels]
    else:
        groupers = pd.Grouper(level=levels)

    expected = series.groupby(groupers).mean()

    # Compute and check result
    result = series.groupby(levels).mean()
    tm.assert_series_equal(result, expected)
