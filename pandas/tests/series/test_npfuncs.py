"""
Tests for np.foo applied to Series, not necessarily ufuncs.
"""
# [MODULE DOC]
# Tests for NumPy functions applied to Series, not necessarily universal functions: peak-to-peak, unique, argwhere and a logarithm on pyarrow-backed data (skipped without pyarrow).
# No import-time side effects.


import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import Series
import pandas._testing as tm


class TestPtp:
    # [CLASS DOC]
    # Groups tests of the NumPy peak-to-peak function applied to a Series. Stateless.
    def test_ptp(self):
        # [FUNCTION DOC]
        # Verifies (issue 21614) that the NumPy peak-to-peak function gives the same scalar for a thousand-element random Series as for the underlying array.
        # GH#21614
        N = 1000
        arr = np.random.default_rng(2).standard_normal(N)
        ser = Series(arr)
        assert np.ptp(ser) == np.ptp(arr)


def test_numpy_unique(datetime_series):
    # [FUNCTION DOC]
    # Smoke test: the NumPy unique function accepts the datetime fixture series without raising. No assertion.
    # it works!
    np.unique(datetime_series)


@pytest.mark.parametrize("index", [["a", "b", "c", "d", "e"], None])
def test_numpy_argwhere(index):
    # [FUNCTION DOC]
    # Verifies (issue 35331) that the NumPy argwhere function applied to a boolean Series (values above two) returns a column of positions three and four as 64-bit integers, for both a string index and the default index.
    # GH#35331

    s = Series(range(5), index=index, dtype=np.int64)

    result = np.argwhere(s > 2).astype(np.int64)
    expected = np.array([[3], [4]], dtype=np.int64)

    tm.assert_numpy_array_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_log_arrow_backed_missing_value(using_nan_is_na):
    # [FUNCTION DOC]
    # Verifies (issue 56285) applying the NumPy log to a pyarrow-backed float Series with a missing value. When the nan-is-na option fixture is on, the result equals the log applied to an equal series. Otherwise the Series is converted to object, and a type error saying the ufunc loop does not support the argument is raised.
    # KNOWN ISSUE: in the first branch the expected value is computed by the same operation on an identical series, so the assertion compares the function with itself and cannot detect an incorrect result.
    # GH#56285
    ser = Series([1, 2, None], dtype="float64[pyarrow]")
    if using_nan_is_na:
        result = np.log(ser)
        expected = np.log(Series([1, 2, None], dtype="float64[pyarrow]"))
        tm.assert_series_equal(result, expected)
    else:
        # we get cast to object which raises
        msg = "loop of ufunc does not support argument"
        with pytest.raises(TypeError, match=msg):
            np.log(ser)
