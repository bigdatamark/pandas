# [MODULE DOC]
# Test of membership checks on the nullable floating array when it contains a real NaN. Uses the
# using_nan_is_na fixture. No import-time side effects.

import numpy as np

import pandas as pd


def test_contains_nan(using_nan_is_na):
    # [FUNCTION DOC]
    # Dividing an integer array by zero gives a NaN in the data array. Whether that entry is reported missing depends on
    # the NaN-is-NA mode; in both modes a NaN is reported as contained in the array.
    # GH#52840
    arr = pd.array(range(5)) / 0

    assert np.isnan(arr._data[0])
    if using_nan_is_na:
        assert arr.isna()[0]
    else:
        assert not arr.isna()[0]
    assert np.nan in arr
