# [MODULE DOC]
# Tests for concatenating Series of nullable floating dtypes. No import-time side effects.

import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        (["Float64", "Float64"], "Float64"),
        (["Float32", "Float64"], "Float64"),
        (["Float32", "Float32"], "Float32"),
    ],
)
def test_concat_series(to_concat_dtypes, result_dtype):
    # [FUNCTION DOC]
    # Parametrized over lists of dtype names and the expected result dtype. Concatenating two Series (with a missing value)
    # of 64/64, 32/64 and 32/32-bit nullable floats gives the wider dtype (or same), equal to concatenating object
    # Series and casting.
    result = pd.concat([pd.Series([1, 2, pd.NA], dtype=t) for t in to_concat_dtypes])
    expected = pd.concat([pd.Series([1, 2, pd.NA], dtype=object)] * 2).astype(
        result_dtype
    )
    tm.assert_series_equal(result, expected)
