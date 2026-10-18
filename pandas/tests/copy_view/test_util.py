# [MODULE DOC]
# Tests for the get_array helper in the sibling util module: it returns a plain
# buffer for numpy-backed and masked columns.

import numpy as np

from pandas import DataFrame
from pandas.tests.copy_view.util import get_array


def test_get_array_numpy():
    # [FUNCTION DOC]
    # Two calls to get_array for the same numpy-backed column return arrays that share memory.
    df = DataFrame({"a": [1, 2, 3]})
    assert np.shares_memory(get_array(df, "a"), get_array(df, "a"))


def test_get_array_masked():
    # [FUNCTION DOC]
    # Two calls to get_array for the same nullable Int64 column return arrays that
    # share memory (the underlying data of the masked array is returned).
    df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    assert np.shares_memory(get_array(df, "a"), get_array(df, "a"))
