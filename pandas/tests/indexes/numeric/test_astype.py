# [MODULE DOC]
# Tests for astype on floating point Index objects: conversion to unsigned and signed
# integers, object, other floats and datetime-like types, and the errors for
# non-finite or negative values.
import numpy as np
import pytest

from pandas import (
    Index,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm


class TestAstype:
    # [CLASS DOC]
    # Groups astype tests for floating indexes.
    def test_astype_float64_to_uint64(self):
        # [FUNCTION DOC]
        # Casting a float index of non-negative whole numbers to 64-bit unsigned gives
        # an
        # unsigned index of identical values (GH#45309); casting after subtracting so
        # that
        # negatives appear raises a value error mentioning lossless conversion.
        # GH#45309 used to incorrectly return Index with int64 dtype
        idx = Index([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)
        result = idx.astype("u8")
        expected = Index([0, 5, 10, 15, 20], dtype=np.uint64)
        tm.assert_index_equal(result, expected, exact=True)

        idx_with_negatives = idx - 10
        with pytest.raises(ValueError, match="losslessly"):
            idx_with_negatives.astype(np.uint64)

    def test_astype_float64_to_object(self):
        # [FUNCTION DOC]
        # Casting to object gives an equal index of object dtype.
        float_index = Index([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float64)
        result = float_index.astype(object)
        assert result.equals(float_index)
        assert float_index.equals(result)
        assert isinstance(result, Index) and result.dtype == object

    def test_astype_float64_mixed_to_object(self):
        # [FUNCTION DOC]
        # Same for a float index with fractional and whole values and a name.
        # mixed int-float
        idx = Index([1.5, 2, 3, 4, 5], dtype=np.float64)
        idx.name = "foo"
        result = idx.astype(object)
        assert result.equals(idx)
        assert idx.equals(result)
        assert isinstance(result, Index) and result.dtype == object

    @pytest.mark.parametrize("dtype", ["int16", "int32", "int64"])
    def test_astype_float64_to_int_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over three signed integer dtypes. Whole floats cast exactly;
        # floats with
        # fractions are truncated (GH#12881).
        # GH#12881
        # a float astype int
        idx = Index([0, 1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        expected = Index([0, 1, 2], dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

        idx = Index([0, 1.1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        expected = Index([0, 1, 2], dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_astype_float64_to_float_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over 32/64-bit floats. Result is an Index of the requested dtype
        # (GH#12881).
        # GH#12881
        # a float astype int
        idx = Index([0, 1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        assert isinstance(result, Index) and result.dtype == dtype

    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_astype_float_to_datetimelike(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over datetime and timedelta nanosecond dtypes. Result equals the
        # conversion of the float values and matches Series behavior (GH#49660).
        # GH#49660 pre-2.0 Index.astype from floating to M8/m8/Period raised,
        #  inconsistent with Series.astype
        idx = Index([0, 1.1, 2], dtype=np.float64)

        result = idx.astype(dtype)
        if dtype[0] == "M":
            expected = to_datetime(idx.values)
        else:
            expected = to_timedelta(idx.values)
        tm.assert_index_equal(result, expected)

        # check that we match Series behavior
        result = idx.to_series().set_axis(range(3)).astype(dtype)
        expected = expected.to_series().set_axis(range(3))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [int, "int16", "int32", "int64"])
    @pytest.mark.parametrize("non_finite", [np.inf, np.nan])
    def test_cannot_cast_inf_to_int(self, non_finite, dtype):
        # [FUNCTION DOC]
        # Parametrized over non-finite value (infinity, NaN) and integer dtypes. Casting
        # raises a value error saying non-finite values cannot be converted to integer
        # (GH#13149).
        # GH#13149
        idx = Index([1, 2, non_finite], dtype=np.float64)

        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(ValueError, match=msg):
            idx.astype(dtype)

    def test_astype_from_object(self):
        # [FUNCTION DOC]
        # An object index of floats and NaN cast to float yields a float64 index.
        index = Index([1.0, np.nan, 0.2], dtype="object")
        result = index.astype(float)
        expected = Index([1.0, np.nan, 0.2], dtype=np.float64)
        assert result.dtype == expected.dtype
        tm.assert_index_equal(result, expected)
