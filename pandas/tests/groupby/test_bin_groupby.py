# [MODULE DOC]
# Tests for lower-level groupby support: manager block locations after aggregation and the binning function that computes bin edges for sorted datetime-like integer values.
import numpy as np
import pytest

from pandas._libs import lib

import pandas as pd
import pandas._testing as tm


def assert_block_lengths(x):
    # [FUNCTION DOC]
    # HELPER: aggregation function used by the manager-locations test. Asserts that the number of rows of the group chunk equals the length of the manager locations of its first block, then returns zero.
    assert len(x) == len(x._mgr.blocks[0].mgr_locs)
    return 0


def cumsum_max(x):
    # [FUNCTION DOC]
    # HELPER: aggregation function used by the manager-locations test. Computes the cumulative sum of the chunk and its maximum (result discarded, only exercising block creation) and returns zero.
    x.cumsum().max()
    return 0


@pytest.mark.parametrize(
    "func",
    [
        cumsum_max,
        assert_block_lengths,
    ],
)
def test_mgr_locs_updated(func):
    # [FUNCTION DOC]
    # Issue 31802: parametrized over the two helper functions. Aggregating a frame grouped by two string keys must work and return zero for the C column in each group, with a MultiIndex of (A, B); operations that create new blocks need valid manager locations.
    # https://github.com/pandas-dev/pandas/issues/31802
    # Some operations may require creating new blocks, which requires
    # valid mgr_locs
    df = pd.DataFrame({"A": ["a", "a", "a"], "B": ["a", "b", "b"], "C": [1, 1, 1]})
    result = df.groupby(["A", "B"]).agg(func)
    expected = pd.DataFrame(
        {"C": [0, 0]},
        index=pd.MultiIndex.from_product([["a"], ["a", "b"]], names=["A", "B"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "binner,closed,expected",
    [
        (
            [0, 3, 6, 9],
            "left",
            [2, 5, 6],
        ),
        (
            [0, 3, 6, 9],
            "right",
            [3, 6, 6],
        ),
        ([0, 3, 6], "left", [2, 5]),
        (
            [0, 3, 6],
            "right",
            [3, 6],
        ),
    ],
)
def test_generate_bins(binner, closed, expected):
    # [FUNCTION DOC]
    # Parametrized over bin edges and closed side. Calls the low-level function that generates bin end positions for the sorted values one to six against the integer edges; the result must be an int64 array of the number of values falling at or before each edge (one entry per bin, i.e. one fewer than the number of edges; left closed gives 2, 5, 6 for four edges, right closed gives 3, 6, 6).
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    result = lib.generate_bins_dt64(
        values, np.array(binner, dtype=np.int64), closed=closed
    )
    expected = np.array(expected, dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)
