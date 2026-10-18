# [MODULE DOC]
# Tests that building frames and Series from mixed objects does not wrongly infer
# datetime-like dtypes.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timestamp,
)


@pytest.mark.parametrize(
    "data,exp_size",
    [
        # see gh-16362.
        ([[NaT, "a", "b", 0], [NaT, "b", "c", 1]], 8),
        ([[NaT, "a", 0], [NaT, "b", 1]], 6),
    ],
)
def test_maybe_infer_to_datetimelike_df_construct(data, exp_size):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; parametrized over nested lists mixing NaT
    # and strings with integers and the expected element count. A frame built from the
    # object array has the expected total size (no failure during inference).
    result = DataFrame(np.array(data))
    assert result.size == exp_size


def test_maybe_infer_to_datetimelike_ser_construct():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: a Series from a string that looks
    # datetime-like plus a Timestamp keeps object dtype.
    # see gh-19671.
    result = Series(["M1701", Timestamp("20130101")])
    assert result.dtype.kind == "O"
