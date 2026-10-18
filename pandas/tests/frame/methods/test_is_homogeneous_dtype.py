# [MODULE DOC]
# Tests for the DataFrame private property that reports whether all columns share one dtype.

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
)


@pytest.mark.parametrize(
    "data, expected",
    [
        # empty
        (DataFrame(), True),
        # multi-same
        (DataFrame({"A": [1, 2], "B": [1, 2]}), True),
        # multi-object
        (
            DataFrame(
                {
                    "A": np.array([1, 2], dtype=object),
                    "B": np.array(["a", "b"], dtype=object),
                },
                dtype="object",
            ),
            True,
        ),
        # multi-extension
        (
            DataFrame({"A": Categorical(["a", "b"]), "B": Categorical(["a", "b"])}),
            True,
        ),
        # differ types
        (DataFrame({"A": [1, 2], "B": [1.0, 2.0]}), False),
        # differ sizes
        (
            DataFrame(
                {
                    "A": np.array([1, 2], dtype=np.int32),
                    "B": np.array([1, 2], dtype=np.int64),
                }
            ),
            False,
        ),
        # multi-extension differ
        (
            DataFrame({"A": Categorical(["a", "b"]), "B": Categorical(["b", "c"])}),
            False,
        ),
    ],
)
def test_is_homogeneous_type(data, expected):
    # [FUNCTION DOC]
    # Parametrized over frames and expected booleans: empty, same-dtype numeric, object, and identical categorical frames are homogeneous; mixed int and float, int32 and int64, and categoricals with different categories are not. Verifies the property is exactly the expected boolean.
    assert data._is_homogeneous_type is expected
