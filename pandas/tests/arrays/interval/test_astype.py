# [MODULE DOC]
# Test of astype from IntervalArray and IntervalIndex to categorical dtypes. No import-time side effects.

import pytest

from pandas import (
    Categorical,
    CategoricalDtype,
    Index,
    IntervalIndex,
)
import pandas._testing as tm


class TestAstype:
    # [CLASS DOC]
    # Groups astype tests for interval containers. Stateless.
    @pytest.mark.parametrize("ordered", [True, False])
    def test_astype_categorical_retains_ordered(self, ordered):
        # [FUNCTION DOC]
        # Parametrized over ordered true and false. Casting an interval array to a categorical dtype with no categories builds a Categorical of its
        # intervals with the requested ordered flag; casting the interval index gives the corresponding index of that Categorical.
        index = IntervalIndex.from_breaks(range(5))
        arr = index._data

        dtype = CategoricalDtype(None, ordered=ordered)

        expected = Categorical(list(arr), ordered=ordered)
        result = arr.astype(dtype)
        assert result.ordered is ordered
        tm.assert_categorical_equal(result, expected)

        # test IntervalIndex.astype while we're at it.
        result = index.astype(dtype)
        expected = Index(expected)
        tm.assert_index_equal(result, expected)
