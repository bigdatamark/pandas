# [MODULE DOC]
# Tests for the compiled numpy-datetime helper library: detection of unitless datetime/timedelta dtypes, mapping from dtype to unit code,
# conversion of integer timedeltas to component dictionaries at several units, and overflow-safe unit casting (type errors for bad input,
# out-of-bounds errors, matching numpy on safe casts, and refusing lossy rounding). Depends on numpy version flags to decide how numpy itself
# behaves on overflow.

import numpy as np
import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    astype_overflowsafe,
    is_unitless,
    py_get_unit_from_dtype,
    py_td64_to_tdstruct,
)
from pandas.compat.numpy import (
    is_numpy_dev,
    np_version_gt2_5,
)

import pandas._testing as tm


@pytest.mark.filterwarnings("ignore:.*'generic' unit:DeprecationWarning")
def test_is_unitless():
    # [FUNCTION DOC]
    # Ignoring the deprecation warning about the generic unit. Nanosecond datetime and timedelta dtypes are not unitless, the generic ones are; a non-datetime dtype raises a value error saying dtype must be datetime64 or timedelta64; a non-dtype argument (a string) raises a type error about an incorrect argument type.
    dtype = np.dtype("M8[ns]")
    assert not is_unitless(dtype)

    dtype = np.dtype("datetime64")
    assert is_unitless(dtype)

    dtype = np.dtype("m8[ns]")
    assert not is_unitless(dtype)

    dtype = np.dtype("timedelta64")
    assert is_unitless(dtype)

    msg = "dtype must be datetime64 or timedelta64"
    with pytest.raises(ValueError, match=msg):
        is_unitless(np.dtype(np.int64))

    msg = "Argument 'dtype' has incorrect type"
    with pytest.raises(TypeError, match=msg):
        is_unitless("foo")


def test_get_unit_from_dtype():
    # [FUNCTION DOC]
    # For each datetime64 and timedelta64 unit from year through attosecond (the removed business-day unit is skipped), the helper returns the integer code equal to the matching member of the numpy unit enumeration.
    # datetime64
    assert py_get_unit_from_dtype(np.dtype("M8[Y]")) == NpyDatetimeUnit.NPY_FR_Y.value
    assert py_get_unit_from_dtype(np.dtype("M8[M]")) == NpyDatetimeUnit.NPY_FR_M.value
    assert py_get_unit_from_dtype(np.dtype("M8[W]")) == NpyDatetimeUnit.NPY_FR_W.value
    # B has been deprecated and removed -> no 3
    assert py_get_unit_from_dtype(np.dtype("M8[D]")) == NpyDatetimeUnit.NPY_FR_D.value
    assert py_get_unit_from_dtype(np.dtype("M8[h]")) == NpyDatetimeUnit.NPY_FR_h.value
    assert py_get_unit_from_dtype(np.dtype("M8[m]")) == NpyDatetimeUnit.NPY_FR_m.value
    assert py_get_unit_from_dtype(np.dtype("M8[s]")) == NpyDatetimeUnit.NPY_FR_s.value
    assert py_get_unit_from_dtype(np.dtype("M8[ms]")) == NpyDatetimeUnit.NPY_FR_ms.value
    assert py_get_unit_from_dtype(np.dtype("M8[us]")) == NpyDatetimeUnit.NPY_FR_us.value
    assert py_get_unit_from_dtype(np.dtype("M8[ns]")) == NpyDatetimeUnit.NPY_FR_ns.value
    assert py_get_unit_from_dtype(np.dtype("M8[ps]")) == NpyDatetimeUnit.NPY_FR_ps.value
    assert py_get_unit_from_dtype(np.dtype("M8[fs]")) == NpyDatetimeUnit.NPY_FR_fs.value
    assert py_get_unit_from_dtype(np.dtype("M8[as]")) == NpyDatetimeUnit.NPY_FR_as.value

    # timedelta64
    assert py_get_unit_from_dtype(np.dtype("m8[Y]")) == NpyDatetimeUnit.NPY_FR_Y.value
    assert py_get_unit_from_dtype(np.dtype("m8[M]")) == NpyDatetimeUnit.NPY_FR_M.value
    assert py_get_unit_from_dtype(np.dtype("m8[W]")) == NpyDatetimeUnit.NPY_FR_W.value
    # B has been deprecated and removed -> no 3
    assert py_get_unit_from_dtype(np.dtype("m8[D]")) == NpyDatetimeUnit.NPY_FR_D.value
    assert py_get_unit_from_dtype(np.dtype("m8[h]")) == NpyDatetimeUnit.NPY_FR_h.value
    assert py_get_unit_from_dtype(np.dtype("m8[m]")) == NpyDatetimeUnit.NPY_FR_m.value
    assert py_get_unit_from_dtype(np.dtype("m8[s]")) == NpyDatetimeUnit.NPY_FR_s.value
    assert py_get_unit_from_dtype(np.dtype("m8[ms]")) == NpyDatetimeUnit.NPY_FR_ms.value
    assert py_get_unit_from_dtype(np.dtype("m8[us]")) == NpyDatetimeUnit.NPY_FR_us.value
    assert py_get_unit_from_dtype(np.dtype("m8[ns]")) == NpyDatetimeUnit.NPY_FR_ns.value
    assert py_get_unit_from_dtype(np.dtype("m8[ps]")) == NpyDatetimeUnit.NPY_FR_ps.value
    assert py_get_unit_from_dtype(np.dtype("m8[fs]")) == NpyDatetimeUnit.NPY_FR_fs.value
    assert py_get_unit_from_dtype(np.dtype("m8[as]")) == NpyDatetimeUnit.NPY_FR_as.value


def test_td64_to_tdstruct():
    # [FUNCTION DOC]
    # Converts an arbitrary integer (12454636234) interpreted in nanoseconds, microseconds, milliseconds and seconds into a component dictionary and compares with exact expected dictionaries (days, hours, minutes, seconds, ms, us, ns, plus whole-second, microsecond and nanosecond remainder totals). The seconds case is out of bounds for a nanosecond timedelta but still works.
    val = 12454636234  # arbitrary value

    res1 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_ns.value)
    exp1 = {
        "days": 0,
        "hrs": 0,
        "min": 0,
        "sec": 12,
        "ms": 454,
        "us": 636,
        "ns": 234,
        "seconds": 12,
        "microseconds": 454636,
        "nanoseconds": 234,
    }
    assert res1 == exp1

    res2 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_us.value)
    exp2 = {
        "days": 0,
        "hrs": 3,
        "min": 27,
        "sec": 34,
        "ms": 636,
        "us": 234,
        "ns": 0,
        "seconds": 12454,
        "microseconds": 636234,
        "nanoseconds": 0,
    }
    assert res2 == exp2

    res3 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_ms.value)
    exp3 = {
        "days": 144,
        "hrs": 3,
        "min": 37,
        "sec": 16,
        "ms": 234,
        "us": 0,
        "ns": 0,
        "seconds": 13036,
        "microseconds": 234000,
        "nanoseconds": 0,
    }
    assert res3 == exp3

    # Note this out of bounds for nanosecond Timedelta
    res4 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_s.value)
    exp4 = {
        "days": 144150,
        "hrs": 21,
        "min": 10,
        "sec": 34,
        "ms": 0,
        "us": 0,
        "ns": 0,
        "seconds": 76234,
        "microseconds": 0,
        "nanoseconds": 0,
    }
    assert res4 == exp4


class TestAstypeOverflowSafe:
    # [CLASS DOC]
    # Tests for the overflow-safe unit cast of datetime and timedelta arrays.
    def test_pass_non_dt64_array(self):
        # [FUNCTION DOC]
        # Passing an integer array with a datetime target dtype raises a type error saying both must be datetime64 or both timedelta64 (not a crash), for copy true and false.
        # check that we raise, not segfault
        arr = np.arange(5)
        dtype = np.dtype("M8[ns]")

        msg = (
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64"
        )
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=True)

        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=False)

    def test_pass_non_dt64_dtype(self):
        # [FUNCTION DOC]
        # Passing a datetime64 day array with a timedelta64 target dtype raises the same type error, for copy true and false.
        # check that we raise, not segfault
        arr = np.arange(5, dtype="i8").view("M8[D]")
        dtype = np.dtype("m8[ns]")

        msg = (
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64"
        )
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=True)

        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=False)

    def test_astype_overflowsafe_dt64(self):
        # [FUNCTION DOC]
        # Builds ten dates starting 2262-04-05 (near the nanosecond upper limit). Numpy's own cast to nanoseconds either raises an overflow error (development numpy or numpy newer than 2.5) or silently overflows (a round trip then differs). The safe cast to nanoseconds raises the out-of-bounds datetime error; the cast to microseconds works and equals numpy's result.
        dtype = np.dtype("M8[ns]")

        dt = np.datetime64("2262-04-05", "D")
        arr = dt + np.arange(10, dtype="m8[D]")

        if is_numpy_dev or np_version_gt2_5:
            # numpy now raises instead of silently overflowing
            with pytest.raises(OverflowError, match="Overflow"):
                arr.astype(dtype)
        else:
            # arr.astype silently overflows, so this
            wrong = arr.astype(dtype)
            roundtrip = wrong.astype(arr.dtype)
            assert not (wrong == roundtrip).all()

        msg = "Out of bounds nanosecond timestamp"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            astype_overflowsafe(arr, dtype)

        # But converting to microseconds is fine, and we match numpy's results.
        dtype2 = np.dtype("M8[us]")
        result = astype_overflowsafe(arr, dtype2)
        expected = arr.astype(dtype2)
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_overflowsafe_td64(self):
        # [FUNCTION DOC]
        # Same as the datetime test but with the dates reinterpreted as day timedeltas. The safe cast to nanoseconds raises the out-of-bounds timedelta error saying 106752 days cannot be converted without overflow; microsecond conversion equals numpy's.
        dtype = np.dtype("m8[ns]")

        dt = np.datetime64("2262-04-05", "D")
        arr = dt + np.arange(10, dtype="m8[D]")
        arr = arr.view("m8[D]")

        if is_numpy_dev or np_version_gt2_5:
            # numpy now raises instead of silently overflowing
            with pytest.raises(OverflowError, match="Overflow"):
                arr.astype(dtype)
        else:
            # arr.astype silently overflows, so this
            wrong = arr.astype(dtype)
            roundtrip = wrong.astype(arr.dtype)
            assert not (wrong == roundtrip).all()

        msg = r"Cannot convert 106752 days to timedelta64\[ns\] without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            astype_overflowsafe(arr, dtype)

        # But converting to microseconds is fine, and we match numpy's results.
        dtype2 = np.dtype("m8[us]")
        result = astype_overflowsafe(arr, dtype2)
        expected = arr.astype(dtype2)
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_overflowsafe_disallow_rounding(self):
        # [FUNCTION DOC]
        # Casting nanosecond values minus 1500 and 1500 to microseconds with rounding disallowed raises a value error that it cannot losslessly cast minus 1500 ns to us; with rounding allowed the result equals numpy's cast.
        arr = np.array([-1500, 1500], dtype="M8[ns]")
        dtype = np.dtype("M8[us]")

        msg = "Cannot losslessly cast '-1500 ns' to us"
        with pytest.raises(ValueError, match=msg):
            astype_overflowsafe(arr, dtype, round_ok=False)

        result = astype_overflowsafe(arr, dtype, round_ok=True)
        expected = arr.astype(dtype)
        tm.assert_numpy_array_equal(result, expected)
