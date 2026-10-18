# [MODULE DOC]
# Tests for the private scalar-lookup method on DataFrame, including multi-index failure
# and agreement with ordinary lookup. Uses the float_frame fixture. No import-time side effects.
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)


class TestGetValue:
    # [CLASS DOC]
    # Test class for scalar value lookup.
    def test_get_set_value_no_partial_indexing(self):
        # [FUNCTION DOC]
        # Verifies that with a two-level row index, a partial (first level only) lookup through the private
        # scalar getter raises a key error whose message is exactly the first-level value.
        # partial w/ MultiIndex raise exception
        index = MultiIndex.from_tuples([(0, 1), (0, 2), (1, 1), (1, 2)])
        df = DataFrame(index=index, columns=range(4))
        with pytest.raises(KeyError, match=r"^0$"):
            df._get_value(0, 1)

    def test_get_value(self, float_frame):
        # [FUNCTION DOC]
        # Verifies that for every row and column label of the float frame, the private scalar getter returns
        # the same value as column-then-row Series lookup.
        for idx in float_frame.index:
            for col in float_frame.columns:
                result = float_frame._get_value(idx, col)
                expected = float_frame[col][idx]
                assert result == expected
