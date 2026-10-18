# [MODULE DOC]
# Tests for the interval-closedness validator in the private validators module
# of the util package: invalid inputs raise, valid strings map to a pair of
# booleans (left inclusive, right inclusive).
import numpy as np
import pytest

from pandas.util._validators import validate_inclusive

import pandas as pd


@pytest.mark.parametrize(
    "invalid_inclusive",
    (
        "ccc",
        2,
        object(),
        None,
        np.nan,
        pd.NA,
        pd.DataFrame(),
    ),
)
def test_invalid_inclusive(invalid_inclusive):
    # [FUNCTION DOC]
    # Verifies that any value other than the four allowed strings (an unknown
    # string, integer, plain object, None, NaN, NA, or an empty frame) makes
    # the validator raise a value error stating the inclusive option has to be
    # both, neither, left or right.
    with pytest.raises(
        ValueError,
        match="Inclusive has to be either 'both', 'neither', 'left' or 'right'",
    ):
        validate_inclusive(invalid_inclusive)


@pytest.mark.parametrize(
    "valid_inclusive, expected_tuple",
    (
        ("left", (True, False)),
        ("right", (False, True)),
        ("both", (True, True)),
        ("neither", (False, False)),
    ),
)
def test_valid_inclusive(valid_inclusive, expected_tuple):
    # [FUNCTION DOC]
    # Verifies each valid option maps to the expected pair of flags (left
    # closed, right closed): left gives true/false, right false/true, both
    # true/true, neither false/false.
    resultant_tuple = validate_inclusive(valid_inclusive)
    assert expected_tuple == resultant_tuple
