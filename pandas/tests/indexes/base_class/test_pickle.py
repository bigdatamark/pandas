# [MODULE DOC]
# Pickle round-trip test for the base Index.
from pandas import Index
import pandas._testing as tm


def test_pickle_preserves_object_dtype(temp_file):
    # [FUNCTION DOC]
    # Uses the temp_file fixture as a pickle path. Verifies an object-dtype Index of integers keeps
    # object dtype (no numeric inference) after a pickle round trip and equals the original.
    # GH#43188, GH#43155 don't infer numeric dtype
    index = Index([1, 2, 3], dtype=object)

    result = tm.round_trip_pickle(index, temp_file)
    assert result.dtype == object
    tm.assert_index_equal(index, result)
