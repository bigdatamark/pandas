# [MODULE DOC]
# Tests for helper functions in the internal indexers module: length of indexer, scalar indexer
# detection, and index validation.
# Tests aimed at pandas.core.indexers
import numpy as np
import pytest

from pandas.core.indexers import (
    is_scalar_indexer,
    length_of_indexer,
    validate_indices,
)


def test_length_of_indexer():
    # [FUNCTION DOC]
    # A boolean mask with a single true element has length 1 as an indexer.
    arr = np.zeros(4, dtype=bool)
    arr[0] = 1
    result = length_of_indexer(arr)
    assert result == 1


def test_is_scalar_indexer():
    # [FUNCTION DOC]
    # A tuple of two integers (for ndim 2) is a scalar indexer; the bare first integer is not for
    # ndim 2; tuples containing arrays, and a slice, are not; a bare integer or one-element tuple is
    # scalar for ndim 1.
    indexer = (0, 1)
    assert is_scalar_indexer(indexer, 2)
    assert not is_scalar_indexer(indexer[0], 2)

    indexer = (np.array([2]), 1)
    assert not is_scalar_indexer(indexer, 2)

    indexer = (np.array([2]), np.array([3]))
    assert not is_scalar_indexer(indexer, 2)

    indexer = (np.array([2]), np.array([3, 4]))
    assert not is_scalar_indexer(indexer, 2)

    assert not is_scalar_indexer(slice(None), 1)

    indexer = 0
    assert is_scalar_indexer(indexer, 1)

    indexer = (0,)
    assert is_scalar_indexer(indexer, 1)


class TestValidateIndices:
    # [CLASS DOC]
    # Test collection for the function that validates an integer array of positions for a given
    # length.
    def test_validate_indices_ok(self):
        # [FUNCTION DOC]
        # Valid positions (within range, empty array, and all -1 markers with length zero) pass
        # silently with no return value checked.
        indices = np.asarray([0, 1])
        validate_indices(indices, 2)
        validate_indices(indices[:0], 0)
        validate_indices(np.array([-1, -1]), 0)

    def test_validate_indices_low(self):
        # [FUNCTION DOC]
        # A position below -1 raises a value error saying the indices contain such values.
        indices = np.asarray([0, -2])
        with pytest.raises(ValueError, match="'indices' contains"):
            validate_indices(indices, 2)

    def test_validate_indices_high(self):
        # [FUNCTION DOC]
        # A position equal to or greater than the length raises an index error saying positions are out
        # of bounds.
        indices = np.asarray([0, 1, 2])
        with pytest.raises(IndexError, match="indices are out"):
            validate_indices(indices, 2)

    def test_validate_indices_empty(self):
        # [FUNCTION DOC]
        # Any non-negative positions with length zero raise an out-of-bounds index error.
        with pytest.raises(IndexError, match="indices are out"):
            validate_indices(np.array([0, 1]), 0)
