# [MODULE DOC]
# Tests for Series.duplicated: keep options, object data with NaN and None,
# categorical with NA, and nullable integer (masked) arrays. Uses the keep and
# nulls_fixture fixtures from the shared conftest. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    NA,
    Categorical,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", [False, False, True, False, True]),
        ("last", [True, True, False, False, False]),
        (False, [True, True, True, False, True]),
    ],
)
def test_duplicated_keep(keep, expected):
    # [FUNCTION DOC]
    # Parametrized over keep (first, last, False) and the expected boolean mask for a
    # string Series with repeats; the result is a boolean Series keeping the name.
    ser = Series(["a", "b", "b", "c", "a"], name="name")

    result = ser.duplicated(keep=keep)
    expected = Series(expected, name="name")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", [False, False, True, False, True]),
        ("last", [True, True, False, False, False]),
        (False, [True, True, True, False, True]),
    ],
)
def test_duplicated_nan_none(keep, expected):
    # [FUNCTION DOC]
    # Parametrized over keep with the expected mask for an object Series of NaN,
    # numbers and None. Repeated NaN values are duplicates of each other, but None is
    # distinct from NaN.
    ser = Series([np.nan, 3, 3, None, np.nan], dtype=object)

    result = ser.duplicated(keep=keep)
    expected = Series(expected)
    tm.assert_series_equal(result, expected)


def test_duplicated_categorical_bool_na(nulls_fixture):
    # [FUNCTION DOC]
    # Regression test (GH#44351), parametrized over null scalars: in an ordered
    # boolean categorical, a missing value is not a duplicate of the others (default keep).
    # GH#44351
    ser = Series(
        Categorical(
            [True, False, True, False, nulls_fixture],
            categories=[True, False],
            ordered=True,
        )
    )
    result = ser.duplicated()
    expected = Series([False, False, True, True, False])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "keep, vals",
    [
        ("last", [True, True, False]),
        ("first", [False, True, True]),
        (False, [True, True, True]),
    ],
)
def test_duplicated_mask(keep, vals):
    # [FUNCTION DOC]
    # Regression test (GH#48150), parametrized over keep: in a nullable Int64 Series
    # with three NA values, NA values are treated as equal to each other and the
    # mask follows the keep option; the unique values 1 and 2 are not duplicates.
    # GH#48150
    ser = Series([1, 2, NA, NA, NA], dtype="Int64")
    result = ser.duplicated(keep=keep)
    expected = Series([False, False, *vals])
    tm.assert_series_equal(result, expected)


def test_duplicated_mask_no_duplicated_na(keep):
    # [FUNCTION DOC]
    # Regression test (GH#48150), using the keep fixture: with a single NA in a
    # nullable Int64 Series nothing is flagged as a duplicate.
    # GH#48150
    ser = Series([1, 2, NA], dtype="Int64")
    result = ser.duplicated(keep=keep)
    expected = Series([False, False, False])
    tm.assert_series_equal(result, expected)
