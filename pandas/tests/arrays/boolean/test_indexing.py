# [MODULE DOC]
# Tests for item assignment on the nullable boolean array.
# Uses the public array constructor and the extension-array equality assertion helper from the testing module.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("na", [None, np.nan, pd.NA])
def test_setitem_missing_values(na):
    # [FUNCTION DOC]
    # Verifies that assigning any missing-value marker (None, NaN, or the NA singleton) to a position in a boolean array stores a missing value there.
    # Parametrized over the three markers; starts from an array of true, false, missing, assigns at position one, and expects the result to hold true, missing, missing.
    arr = pd.array([True, False, None], dtype="boolean")
    expected = pd.array([True, None, None], dtype="boolean")
    arr[1] = na
    tm.assert_extension_array_equal(arr, expected)
