# [MODULE DOC]
# Module-level tests for converting a Series to a NumPy array and for a related nullable-integer cast to timedelta/datetime types.
# One test requires pyarrow and is skipped when it is not installed. No import-time side effects.

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    NA,
    Series,
    Timedelta,
)
import pandas._testing as tm


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_to_numpy_na_value(dtype):
    # [FUNCTION DOC]
    # Verifies that missing values in a Series are replaced by the given fill value (zero) when converting to an array of the requested dtype; parametrized over int64 and float64 targets (issue 48951).
    # GH#48951
    ser = Series([1, 2, NA, 4])
    result = ser.to_numpy(dtype=dtype, na_value=0)
    expected = np.array([1, 2, 0, 4], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_cast_before_setting_na():
    # [FUNCTION DOC]
    # Verifies that for an integer Series converted to float64 with a NaN fill value, the cast happens before the fill is applied, so no error occurs and the result is a float array (issue 50600).
    # GH#50600
    ser = Series([1])
    result = ser.to_numpy(dtype=np.float64, na_value=np.nan)
    expected = np.array([1.0])
    tm.assert_numpy_array_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_to_numpy_arrow_dtype_given():
    # [FUNCTION DOC]
    # Verifies that a pyarrow-backed integer Series containing a missing value converts to a float64 array with NaN for the missing entry when the dtype is given (issue 57121). Skipped without pyarrow.
    # GH#57121
    ser = Series([1, NA], dtype="int64[pyarrow]")
    result = ser.to_numpy(dtype="float64")
    expected = np.array([1.0, np.nan])
    tm.assert_numpy_array_equal(result, expected)


def test_astype_ea_int_to_td_ts():
    # [FUNCTION DOC]
    # Verifies that casting a nullable Int64 Series with a missing value to nanosecond timedelta and to nanosecond datetime64 dtypes works, giving NaT for the missing entry (issue 57093).
    # KNOWN ISSUE: the expected Series for the datetime64 case is built with a timedelta missing marker rather than a datetime one; harmless because the missing markers compare equal, but misleading.
    # GH#57093
    ser = Series([1, None], dtype="Int64")
    result = ser.astype("m8[ns]")
    expected = Series([1, Timedelta("nat")], dtype="m8[ns]")
    tm.assert_series_equal(result, expected)

    result = ser.astype("M8[ns]")
    expected = Series([1, Timedelta("nat")], dtype="M8[ns]")
    tm.assert_series_equal(result, expected)
