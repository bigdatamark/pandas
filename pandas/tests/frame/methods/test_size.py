# [MODULE DOC]
# Tests for the size attribute of DataFrame (number of elements, rows times columns).
# Contains a single parametrized test function.

import numpy as np
import pytest

from pandas import DataFrame


@pytest.mark.parametrize(
    "data, index, expected",
    [
        ({"col1": [1], "col2": [3]}, None, 2),
        ({}, None, 0),
        ({"col1": [1, np.nan], "col2": [3, 4]}, None, 4),
        ({"col1": [1, 2], "col2": [3, 4]}, [["a", "b"], [1, 2]], 4),
        ({"col1": [1, 2, 3, 4], "col2": [3, 4, 5, 6]}, ["x", "y", "a", "b"], 8),
    ],
)
def test_size(data, index, expected):
    # [FUNCTION DOC]
    # Verifies that the size attribute equals the number of cells of the frame.
    # Parametrized over data mappings and optional index (None, nested list for a
    # multi-level index, or flat labels); expected sizes include an empty frame (0),
    # a frame with a missing value (still counted) and a multi-index frame.
    # Asserts the value equals the expectation and that its type is the builtin integer.
    # GH#52897
    df = DataFrame(data, index=index)
    assert df.size == expected
    assert isinstance(df.size, int)
