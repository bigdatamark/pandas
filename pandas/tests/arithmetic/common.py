"""
Assertion helpers for arithmetic tests.
"""

# [MODULE DOC]
# HELPER: assertion helpers shared by the arithmetic test modules.
# Public contents: functions asserting that addition fails, that addition and subtraction both fail,
# that comparisons between mismatched types behave properly, and one choosing the expected result container.
# Depends on the pandas containers, the testing module and the boolean and numpy-backed extension arrays.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    array,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    NumpyExtensionArray,
)


def assert_cannot_add(left, right, msg="cannot add"):
    # [FUNCTION DOC]
    # HELPER: asserts that adding left and right in both orders raises a type error whose message matches
    # the regular expression msg (default is the phrase cannot add). Returns nothing.
    """
    Helper function to assert that two objects cannot be added.

    Parameters
    ----------
    left : object
        The first operand.
    right : object
        The second operand.
    msg : str, default "cannot add"
        The error message expected in the TypeError.
    """
    with pytest.raises(TypeError, match=msg):
        left + right
    with pytest.raises(TypeError, match=msg):
        right + left


def assert_invalid_addsub_type(left, right, msg=None):
    # [FUNCTION DOC]
    # HELPER: asserts that left plus right, right plus left, left minus right and right minus left each
    # raise a type error whose message matches msg. When msg is none any message matches. Returns nothing.
    """
    Helper function to assert that two objects can
    neither be added nor subtracted.

    Parameters
    ----------
    left : object
        The first operand.
    right : object
        The second operand.
    msg : str or None, default None
        The error message expected in the TypeError.
    """
    with pytest.raises(TypeError, match=msg):
        left + right
    with pytest.raises(TypeError, match=msg):
        right + left
    with pytest.raises(TypeError, match=msg):
        left - right
    with pytest.raises(TypeError, match=msg):
        right - left


def get_upcast_box(left, right, is_cmp: bool = False):
    # [FUNCTION DOC]
    # HELPER: chooses the container type to use when building the expected result of an arithmetic or
    # comparison operation between left and right. is_cmp (default false) says the operation is a comparison.
    # Returns the DataFrame class if either operand is a DataFrame; else the Series class if either is a
    # Series, except that a comparison between an Index on the left and a Series gives the numpy array
    # constructor since Index does not defer; else for an Index operand the numpy array constructor for
    # comparisons and the Index class otherwise; else the testing module's to-array function.
    """
    Get the box to use for 'expected' in an arithmetic or comparison operation.

    Parameters
    left : Any
    right : Any
    is_cmp : bool, default False
        Whether the operation is a comparison method.
    """

    if isinstance(left, DataFrame) or isinstance(right, DataFrame):
        return DataFrame
    if isinstance(left, Series) or isinstance(right, Series):
        if is_cmp and isinstance(left, Index):
            # Index does not defer for comparisons
            return np.array
        return Series
    if isinstance(left, Index) or isinstance(right, Index):
        if is_cmp:
            return np.array
        return Index
    return tm.to_array


def assert_invalid_comparison(left, right, box):
    # [FUNCTION DOC]
    # HELPER: asserts that comparing left and right of mismatched types behaves correctly. box is the
    # container type used for the expected result (a frame, series, index, pandas array or to-array function).
    # Equality must be all-False and inequality all-True, in both operand orders, with Index and pandas array
    # boxes expecting numpy arrays. Ordering comparisons (less, less-equal, greater, greater-equal) in both
    # orders must raise a type error matching any of several known messages. Returns nothing.
    """
    Assert that comparison operations with mismatched types behave correctly.

    Parameters
    ----------
    left : np.ndarray, ExtensionArray, Index, or Series
    right : object
    box : {pd.DataFrame, pd.Series, pd.Index, pd.array, tm.to_array}
    """
    # Not for tznaive-tzaware comparison

    # Note: not quite the same as how we do this for tm.box_expected
    xbox = box if box not in [Index, array] else np.array

    def xbox2(x):
        # [FUNCTION DOC]
        # HELPER: normalizes an equality result before comparing: a numpy-backed extension array is replaced by
        # its underlying ndarray and a boolean extension array is converted to a plain bool array (assuming no
        # NA values); anything else is returned unchanged.
        # Eventually we'd like this to be tighter, but for now we'll
        #  just exclude NumpyExtensionArray[bool]
        if isinstance(x, NumpyExtensionArray):
            return x._ndarray
        if isinstance(x, BooleanArray):
            # NB: we are assuming no pd.NAs for now
            return x.astype(bool)
        return x

    result = xbox2(left == right)
    expected = xbox(np.zeros(result.shape, dtype=np.bool_))

    tm.assert_equal(result, expected)

    result = xbox2(right == left)
    tm.assert_equal(result, xbox(expected))

    result = xbox2(left != right)
    tm.assert_equal(result, ~expected)

    result = xbox2(right != left)
    tm.assert_equal(result, xbox(~expected))

    msg = "|".join(
        [
            "Invalid comparison between",
            "Cannot compare type",
            "not supported between",
            "invalid type promotion",
            (
                # GH#36706 npdev 1.20.0 2020-09-28
                r"The DTypes <class 'numpy.dtype\[datetime64\]'> and "
                r"<class 'numpy.dtype\[int64\]'> do not have a common DType. "
                "For example they cannot be stored in a single array unless the "
                "dtype is `object`."
            ),
        ]
    )
    with pytest.raises(TypeError, match=msg):
        left < right
    with pytest.raises(TypeError, match=msg):
        left <= right
    with pytest.raises(TypeError, match=msg):
        left > right
    with pytest.raises(TypeError, match=msg):
        left >= right
    with pytest.raises(TypeError, match=msg):
        right < left
    with pytest.raises(TypeError, match=msg):
        right <= left
    with pytest.raises(TypeError, match=msg):
        right > left
    with pytest.raises(TypeError, match=msg):
        right >= left
