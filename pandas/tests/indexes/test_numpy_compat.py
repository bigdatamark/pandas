# [MODULE DOC]
# Tests of interoperability of index objects with numpy universal functions: output-parameter support, element-wise math functions, floating-point predicate functions, binary reductions, and bitwise operations. All except the bitwise test use the shared index fixture so they exercise every index type; outcomes depend on dtype (numeric returns indexes, datetime-like raises).

import numpy as np
import pytest

from pandas import (
    BooleanDtype,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    PeriodIndex,
    TimedeltaIndex,
    isna,
)
import pandas._testing as tm
from pandas.api.types import (
    is_complex_dtype,
    is_numeric_dtype,
)
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin


def test_numpy_ufuncs_out(index):
    # [FUNCTION DOC]
    # Comparing an index to itself with the numpy equality ufunc and an output array fills the array with the same boolean values as the element-wise equality operator; the same for the underlying extension array when the index is not multi-level.
    result = index == index

    out = np.empty(index.shape, dtype=bool)
    np.equal(index, index, out=out)
    tm.assert_numpy_array_equal(out, result)

    if not index._is_multi:
        # same thing on the ExtensionArray
        out = np.empty(index.shape, dtype=bool)
        np.equal(index.array, index.array, out=out)
        tm.assert_numpy_array_equal(out, result)


@pytest.mark.parametrize(
    "func",
    [
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
        np.sqrt,
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.deg2rad,
        np.rad2deg,
    ],
    ids=lambda x: x.__name__,
)
def test_numpy_ufuncs_basic(index, func):
    # [FUNCTION DOC]
    # Parametrized over 22 numpy math ufuncs (exponentials, logarithms, roots, trigonometric, hyperbolic, angle conversions). Datetime-like indexes raise a type or attribute error from the external library; numeric indexes (except complex with angle conversions) return an index equal to applying the function to the values, keeping the name; the result dtype is checked per input dtype (complex preserved; small integer and bool give float16 or float32; 16-bit integer and float32 give float32; others float64), and extension-dtype numeric inputs return a plain Index. Empty non-numeric indexes pass; other non-numeric indexes raise type or attribute error.
    # test ufuncs of numpy, see:
    # https://numpy.org/doc/stable/reference/ufuncs.html

    if isinstance(index, DatetimeIndexOpsMixin):
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all="ignore"):
                func(index)
    elif is_numeric_dtype(index) and not (
        is_complex_dtype(index) and func in [np.deg2rad, np.rad2deg]
    ):
        # coerces to float (e.g. np.sin)
        with np.errstate(all="ignore"):
            result = func(index)
            arr_result = func(index.values)
            if arr_result.dtype == np.float16:
                arr_result = arr_result.astype(np.float32)
            exp = Index(arr_result, name=index.name)

        tm.assert_index_equal(result, exp)
        if isinstance(index.dtype, np.dtype) and is_numeric_dtype(index):
            if is_complex_dtype(index):
                assert result.dtype == index.dtype
            elif index.dtype in ["bool", "int8", "uint8"]:
                assert result.dtype in ["float16", "float32"]
            elif index.dtype in ["int16", "uint16", "float32"]:
                assert result.dtype == "float32"
            else:
                assert result.dtype == "float64"
        else:
            # e.g. np.exp with Int64 -> Float64
            assert type(result) is Index
    # raise AttributeError or TypeError
    elif len(index) == 0:
        pass
    else:
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all="ignore"):
                func(index)


@pytest.mark.parametrize(
    "func", [np.isfinite, np.isinf, np.isnan, np.signbit], ids=lambda x: x.__name__
)
def test_numpy_ufuncs_other(index, func):
    # [FUNCTION DOC]
    # Parametrized over isfinite, isinf, isnan, signbit. For datetime and timedelta indexes the first three return results (also fillable via an output array) while signbit raises a type error; for period indexes all raise a type error; numeric indexes (except complex with signbit) return a boolean index (nullable boolean for extension dtypes) equal to the output array version; empty others pass; remaining types raise a type error.
    # test ufuncs of numpy, see:
    # https://numpy.org/doc/stable/reference/ufuncs.html
    if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
        if func in (np.isfinite, np.isinf, np.isnan):
            # numpy 1.18 changed isinf and isnan to not raise on dt64/td64
            result = func(index)

            out = np.empty(index.shape, dtype=bool)
            func(index, out=out)
            tm.assert_index_equal(Index(out), result)
        else:
            with tm.external_error_raised(TypeError):
                func(index)

    elif isinstance(index, PeriodIndex):
        with tm.external_error_raised(TypeError):
            func(index)

    elif is_numeric_dtype(index) and not (
        is_complex_dtype(index) and func is np.signbit
    ):
        # Results in bool array
        result = func(index)
        assert isinstance(result, Index)
        if not isinstance(index.dtype, np.dtype):
            # e.g. Int64 we expect to get BooleanArray back
            assert isinstance(result.dtype, BooleanDtype)
        else:
            assert isinstance(result.dtype, np.dtype)

        out = np.empty(index.shape, dtype=bool)
        func(index, out=out)

        if not isinstance(index.dtype, np.dtype):
            tm.assert_index_equal(result, Index(out, dtype="boolean"))
        else:
            tm.assert_index_equal(result, Index(out))

    elif len(index) == 0:
        pass
    else:
        with tm.external_error_raised(TypeError):
            func(index)


@pytest.mark.parametrize("func", [np.maximum, np.minimum])
def test_numpy_ufuncs_reductions(index, func):
    # [FUNCTION DOC]
    # Parametrized over maximum and minimum. Skips empty indexes. Reducing an unordered categorical index raises a type error saying it is not ordered. Otherwise the result has the same type as the index's own max or min (without skipping missing values) and either both are missing or equal.
    # TODO: overlap with tests.series.test_ufunc.test_reductions
    if len(index) == 0:
        pytest.skip("Test doesn't make sense for empty index.")

    if isinstance(index, CategoricalIndex) and index.dtype.ordered is False:
        with pytest.raises(TypeError, match="is not ordered for"):
            func.reduce(index)
        return
    else:
        result = func.reduce(index)

    if func is np.maximum:
        expected = index.max(skipna=False)
    else:
        expected = index.min(skipna=False)
        # TODO: do we have cases both with and without NAs?

    assert type(result) is type(expected)
    if isna(result):
        assert isna(expected)
    else:
        assert result == expected


@pytest.mark.parametrize("func", [np.bitwise_and, np.bitwise_or, np.bitwise_xor])
def test_numpy_ufuncs_bitwise(func):
    # [FUNCTION DOC]
    # Parametrized over bitwise and, or, xor. Applied to two int64 indexes it emits no warnings and equals an index of the function applied to the underlying values.
    # https://github.com/pandas-dev/pandas/issues/46769
    idx1 = Index([1, 2, 3, 4], dtype="int64")
    idx2 = Index([3, 4, 5, 6], dtype="int64")

    with tm.assert_produces_warning(None):
        result = func(idx1, idx2)

    expected = Index(func(idx1.values, idx2.values))
    tm.assert_index_equal(result, expected)
