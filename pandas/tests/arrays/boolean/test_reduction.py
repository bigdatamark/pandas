# [MODULE DOC]
# Tests for reductions on the nullable boolean array and series: any and all with missing-value handling, and the scalar return types of numeric reductions.
import numpy as np
import pytest

import pandas as pd


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # Fixture returning an eleven-element boolean array of alternating true and false with missing values at two positions.
    """Fixture returning boolean array, with valid and missing values."""
    return pd.array(
        [True, False] * 2 + [np.nan] + [True, False] + [np.nan] + [True, False],
        dtype="boolean",
    )


@pytest.mark.parametrize(
    "values, exp_any, exp_all, exp_any_noskip, exp_all_noskip",
    [
        ([True, pd.NA], True, True, True, pd.NA),
        ([False, pd.NA], False, False, pd.NA, False),
        ([pd.NA], False, True, pd.NA, pd.NA),
        ([], False, True, False, True),
        # GH-33253: all True / all False values buggy with skipna=False
        ([True, True], True, True, True, True),
        ([False, False], False, False, False, False),
    ],
)
@pytest.mark.parametrize("con", [pd.array, pd.Series])
def test_any_all(
    values, exp_any, exp_all, exp_any_noskip, exp_all_noskip, using_python_scalars, con
):
    # [FUNCTION DOC]
    # Parametrized over value lists with expected results for any and all, both skipping missing values (default) and not skipping them, and over the container (array constructor or series). Covers Kleene logic: a missing value only yields a missing result when the outcome is undetermined; empty input gives false for any and true for all; all-true and all-false inputs also covered.
    # Expected values are NumPy bool scalars (or the NA singleton) unless the python-scalars option is on and the container is a series, in which case they are plain Python values. Identity (not just equality) with the expected object is asserted.
    # the methods return numpy scalars
    if not using_python_scalars or con is pd.array:
        exp_any = pd.NA if exp_any is pd.NA else np.bool_(exp_any)
        exp_all = pd.NA if exp_all is pd.NA else np.bool_(exp_all)
        exp_any_noskip = pd.NA if exp_any_noskip is pd.NA else np.bool_(exp_any_noskip)
        exp_all_noskip = pd.NA if exp_all_noskip is pd.NA else np.bool_(exp_all_noskip)

    a = con(values, dtype="boolean")
    assert a.any() is exp_any
    assert a.all() is exp_all
    assert a.any(skipna=False) is exp_any_noskip
    assert a.all(skipna=False) is exp_all_noskip


def test_reductions_return_types(
    dropna, data, all_numeric_reductions, using_python_scalars
):
    # [FUNCTION DOC]
    # Uses a shared fixture of numeric reduction names, a dropna flag fixture, and the data fixture. Builds a series (dropping missing values if requested), runs the named reduction, and asserts the result is an instance of the expected scalar type: python int for sum, prod, and count, bool for min and max, float otherwise when the python-scalars option is on; the NumPy counterparts (integer types, bool, float64) otherwise.
    op = all_numeric_reductions
    s = pd.Series(data)
    if dropna:
        s = s.dropna()

    if using_python_scalars:
        expected = {
            "sum": int,
            "prod": int,
            "count": int,
            "min": bool,
            "max": bool,
        }.get(op, float)
    else:
        expected = {
            "sum": np.int_,
            "prod": np.int_,
            "count": np.integer,
            "min": np.bool_,
            "max": np.bool_,
        }.get(op, np.float64)
    result = getattr(s, op)()
    assert isinstance(result, expected), f"{type(result)} vs {expected}"
