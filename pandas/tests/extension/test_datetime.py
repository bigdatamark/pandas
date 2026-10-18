"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
# [MODULE DOC]
# Compliance tests for the time-zone aware datetime extension array (US/Central, nanosecond unit)
# run through the shared extension-array test suite. Provides the fixtures the suite expects and
# a test class that adjusts expected exceptions, reduction and accumulation support and
# reduction checks. Full datetime array tests live elsewhere. No import-time side effects.

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.tests.extension import base


@pytest.fixture
def dtype():
    # [FUNCTION DOC]
    # HELPER: fixture returning the datetime dtype with nanosecond unit and US/Central time zone.
    return DatetimeTZDtype(unit="ns", tz="US/Central")


@pytest.fixture
def data(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a ten-element datetime array of daily dates starting 2000 in the dtype's
    # time zone, built through the private sequence constructor with the dtype.
    data = DatetimeArray._from_sequence(
        pd.date_range("2000", periods=10, tz=dtype.tz), dtype=dtype
    )
    return data


@pytest.fixture
def data_missing(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element array with NaT first and 2000-01-01 second, in the dtype.
    return DatetimeArray._from_sequence(
        np.array(["NaT", "2000-01-01"], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def data_for_sorting(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning an array of three timestamps in the order 2000-01-02, 2000-01-03,
    # 2000-01-01 for sorting tests.
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    c = pd.Timestamp("2000-01-03")
    return DatetimeArray._from_sequence(
        np.array([b, c, a], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def data_missing_for_sorting(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning an array of 2000-01-02, NaT, 2000-01-01 for sorting with missing values.
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    return DatetimeArray._from_sequence(
        np.array([b, "NaT", a], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def data_for_grouping(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning an eight-element array [B, B, NA, NA, A, A, B, C] for group-by tests,
    # where A, B and C are three consecutive days from 2000-01-01 and NA is NaT.
    """
    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    c = pd.Timestamp("2000-01-03")
    na = "NaT"
    return DatetimeArray._from_sequence(
        np.array([b, b, na, na, a, a, b, c], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def na_cmp():
    # [FUNCTION DOC]
    # HELPER: fixture returning a comparison function used by the suite to compare missing values.
    def cmp(a, b):
        # [FUNCTION DOC]
        # HELPER: returns true only when the first argument is NaT and the two arguments are the same object.
        return a is pd.NaT and a is b

    return cmp


# ----------------------------------------------------------------------------
class TestDatetimeArray(base.ExtensionTests):
    # [CLASS DOC]
    # Extension-suite test class for the time-zone aware datetime array; overrides expected
    # exceptions, supported operations and several reduction and construction tests.
    def _get_expected_exception(self, op_name, obj, other):
        # [FUNCTION DOC]
        # HELPER: returns no expected exception for subtraction and reflected subtraction (they work for
        # datetimes); otherwise defers to the parent implementation.
        if op_name in ["__sub__", "__rsub__"]:
            return None
        return super()._get_expected_exception(op_name, obj, other)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: expected result dtype of a reduction. Returns nanosecond timedelta dtype for the standard
        # deviation reduction and the array's own dtype for all other reductions. skipna is unused.
        if op_name == "std":
            return "timedelta64[ns]"
        return arr.dtype

    def _supports_accumulation(self, ser, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: reports whether an accumulation is supported: only running minimum and maximum.
        return op_name in ["cummin", "cummax"]

    def _supports_reduction(self, obj, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: reports whether a reduction is supported: min, max, median, mean, std, any and all.
        return op_name in ["min", "max", "median", "mean", "std", "any", "all"]

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        # [FUNCTION DOC]
        # Parametrized over skipna. Boolean reductions are not supported for datetimes, so the inherited test
        # is expected to raise a type error saying the datetime type does not support the operation.
        meth = all_boolean_reductions
        msg = f"datetime64 type does not support operation '{meth}'"
        with pytest.raises(TypeError, match=msg):
            super().test_reduce_series_boolean(data, all_boolean_reductions, skipna)

    def test_series_constructor(self, data):
        # [FUNCTION DOC]
        # Clears the frequency attribute of the data before calling the inherited Series constructor test,
        # since constructing a Series drops the frequency.
        # Series construction drops any .freq attr
        data = data._with_freq(None)
        super().test_series_constructor(data)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        # [FUNCTION DOC]
        # Parametrized over missing-value action (none, ignore). Verifies mapping the identity function
        # gives an equal extension array.
        result = data.map(lambda x: x, na_action=na_action)
        tm.assert_extension_array_equal(result, data)

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: overrides the reduction check. For median, mean and std computes the expected value from the
        # series cast to 64-bit integers (nanosecond counts); for mean and median converts it to a timestamp in the
        # series time zone, for std converts it to a timedelta; compares with almost-equal. Other reductions defer
        # to the parent check.
        if op_name in ["median", "mean", "std"]:
            alt = ser.astype("int64")

            res_op = getattr(ser, op_name)
            exp_op = getattr(alt, op_name)
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
            if op_name in ["mean", "median"]:
                # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype"
                # has no attribute "tz"
                tz = ser.dtype.tz  # type: ignore[union-attr]
                expected = pd.Timestamp(expected, tz=tz)
            else:
                expected = pd.Timedelta(expected)
            tm.assert_almost_equal(result, expected)

        else:
            return super().check_reduce(ser, op_name, skipna)


class Test2DCompat(base.NDArrayBacked2DTests):
    # [CLASS DOC]
    # Runs the shared two-dimensional compatibility tests for array-backed extension arrays against the
    # datetime array without changes.
    pass
