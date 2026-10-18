# [MODULE DOC]
# Tests for numpy ufuncs, statistical methods, value counts and reductions (sum, min, max, prod) on the
# nullable floating array and Series built from it. Uses fixtures skipna, dtype, using_nan_is_na and
# using_python_scalars. Some tests are skipped on 32-bit platforms. No import-time side effects.

import numpy as np
import pytest

from pandas.compat import IS64

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("ufunc", [np.abs, np.sign])
# np.sign emits a warning with nans, <https://github.com/numpy/numpy/issues/15127>
@pytest.mark.filterwarnings("ignore:invalid value encountered in sign:RuntimeWarning")
def test_ufuncs_single(ufunc, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over absolute value and sign (sign warnings for NaN filtered). The ufunc applied to an array with a
    # missing entry equals the numpy result on the float conversion with the missing position restored to NA, for
    # the array and for a Series.
    a = pd.array([1, 2, -3, pd.NA], dtype="Float64")
    result = ufunc(a)
    np_res = ufunc(a.astype(float))
    np_res = np_res.astype(object)
    np_res[a.isna()] = pd.NA
    expected = pd.array(np_res, dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = ufunc(s)
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.log, np.exp, np.sin, np.cos, np.sqrt])
def test_ufuncs_single_float(ufunc, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over log, exp, sin, cos and sqrt with invalid-value errors suppressed. Same check as the single ufunc test for arrays and Series.
    a = pd.array([1.0, 0.2, 3.0, pd.NA], dtype="Float64")
    with np.errstate(invalid="ignore"):
        result = ufunc(a)
        np_res = ufunc(a.astype(float))
        np_res = np_res.astype(object)
        np_res[a.isna()] = pd.NA
        expected = pd.array(np_res, dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    with np.errstate(invalid="ignore"):
        result = ufunc(s)
        np_res = ufunc(s.astype(float))
        np_res = np_res.astype(object)
        np_res[a.isna()] = pd.NA
        expected = pd.Series(np_res, dtype="Float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.add, np.subtract])
def test_ufuncs_binary_float(ufunc, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over add and subtract. Applied to two nullable arrays, to an array and a numpy array (both orders) and to an
    # array and a scalar (both orders): the result equals the numpy computation with missing positions set to NA.
    # two FloatingArrays
    a = pd.array([1, 0.2, -3, pd.NA], dtype="Float64")
    result = ufunc(a, a)
    np_res = ufunc(a.astype(float), a.astype(float))
    np_res = np_res.astype(object)
    np_res[a.isna()] = pd.NA
    expected = pd.array(np_res, dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    # FloatingArray with numpy array
    arr = np.array([1, 2, 3, 4])
    result = ufunc(a, arr)
    np_res = ufunc(a.astype(float), arr)
    np_res = np_res.astype(object)
    np_res[a.isna()] = pd.NA
    expected = pd.array(np_res, dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(arr, a)
    np_res = ufunc(arr, a.astype(float))
    np_res = np_res.astype(object)
    np_res[a.isna()] = pd.NA
    expected = pd.array(np_res, dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    # FloatingArray with scalar
    result = ufunc(a, 1)
    np_res = ufunc(a.astype(float), 1)
    np_res = np_res.astype(object)
    np_res[a.isna()] = pd.NA
    expected = pd.array(np_res, dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(1, a)
    np_res = ufunc(1, a.astype(float))
    np_res = np_res.astype(object)
    np_res[a.isna()] = pd.NA
    expected = pd.array(np_res, dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("values", [[0, 1], [0, None]])
def test_ufunc_reduce_raises(values):
    # [FUNCTION DOC]
    # Parametrized over a list without and with a missing value. Add-reduce on the array gives the same value as the sum with
    # skipna disabled (so NA when a value is missing).
    # KNOWN ISSUE: the name says reduce raises, but nothing is expected to raise; the test compares the reduction result with the sum.
    arr = pd.array(values, dtype="Float64")

    res = np.add.reduce(arr)
    expected = arr.sum(skipna=False)
    tm.assert_almost_equal(res, expected)


@pytest.mark.skipif(not IS64, reason="GH 36579: fail on 32-bit system")
@pytest.mark.parametrize(
    "pandasmethname, kwargs",
    [
        ("var", {"ddof": 0}),
        ("var", {"ddof": 1}),
        ("std", {"ddof": 0}),
        ("std", {"ddof": 1}),
        ("kurtosis", {}),
        ("skew", {}),
        ("sem", {}),
    ],
)
def test_stat_method(pandasmethname, kwargs):
    # [FUNCTION DOC]
    # Parametrized over variance and standard deviation (ddof zero and one), kurtosis, skew and standard error. A Series of floats with
    # trailing NA gives exactly the same result as the same data in a plain 64-bit float Series without them. Skipped on 32-bit.
    s = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, pd.NA, pd.NA], dtype="Float64")
    pandasmeth = getattr(s, pandasmethname)
    result = pandasmeth(**kwargs)
    s2 = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype="float64")
    pandasmeth = getattr(s2, pandasmethname)
    expected = pandasmeth(**kwargs)
    assert expected == result


def test_value_counts_na():
    # [FUNCTION DOC]
    # Value counts with dropna disabled includes a missing row counted once; with dropna enabled it excludes it. Result is a Series named count of nullable integers indexed by a Float64 index.
    arr = pd.array([0.1, 0.2, 0.1, pd.NA], dtype="Float64")
    result = arr.value_counts(dropna=False)
    idx = pd.Index([0.1, 0.2, pd.NA], dtype=arr.dtype)
    assert idx.dtype == arr.dtype
    expected = pd.Series([2, 1, 1], index=idx, dtype="Int64", name="count")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=idx[:-1], dtype="Int64", name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_empty():
    # [FUNCTION DOC]
    # Value counts of an empty Float64 Series is an empty Int64 Series named count with a Float64 index.
    ser = pd.Series([], dtype="Float64")
    result = ser.value_counts()
    idx = pd.Index([], dtype="Float64")
    assert idx.dtype == "Float64"
    expected = pd.Series([], index=idx, dtype="Int64", name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize():
    # [FUNCTION DOC]
    # Normalized value counts divide the counts by the number of non-missing values; the result is named proportion and has Float64 index.
    ser = pd.Series([0.1, 0.2, 0.1, pd.NA], dtype="Float64")
    result = ser.value_counts(normalize=True)
    expected = pd.Series([2, 1], index=ser[:2], dtype="Float64", name="proportion") / 3
    assert expected.index.dtype == ser.dtype
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("min_count", [0, 4])
def test_floating_array_sum(skipna, min_count, dtype):
    # [FUNCTION DOC]
    # Parametrized over min_count (zero and four) with the skipna fixture. Sum is six when skipping missing and min_count is zero, otherwise the NA singleton.
    arr = pd.array([1, 2, 3, None], dtype=dtype)
    result = arr.sum(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 6.0
    else:
        assert result is pd.NA


@pytest.mark.parametrize(
    "values, expected", [([1, 2, 3], 6.0), ([1, 2, 3, None], 6.0), ([None], 0.0)]
)
def test_floating_array_numpy_sum(values, expected):
    # [FUNCTION DOC]
    # The numpy sum function on nullable floats skips missing values (all-missing gives zero).
    arr = pd.array(values, dtype="Float64")
    result = np.sum(arr)
    assert result == expected


@pytest.mark.parametrize("op", ["sum", "min", "max", "prod"])
def test_preserve_dtypes(op, using_python_scalars):
    # [FUNCTION DOC]
    # Parametrized over sum, min, max and prod. A column of nullable floats reduces to a numpy float scalar (or python float when
    # python scalars are enabled); a groupby reduction keeps the nullable Float64 dtype for that column.
    df = pd.DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": pd.array([0.1, None, 3.0], dtype="Float64"),
        }
    )

    # op
    result = getattr(df.C, op)()
    if using_python_scalars:
        assert type(result) == float
    else:
        assert isinstance(result, np.float64)

    # groupby
    result = getattr(df.groupby("A"), op)()

    expected = pd.DataFrame(
        {"B": np.array([1.0, 3.0]), "C": pd.array([0.1, 3], dtype="Float64")},
        index=pd.Index(["a", "b"], name="A"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["min", "max"])
def test_floating_array_min_max(skipna, method, dtype):
    # [FUNCTION DOC]
    # Parametrized over min and max with skipna and dtype fixtures. Skipping missing returns the extreme (zero or one); otherwise NA.
    arr = pd.array([0.0, 1.0, None], dtype=dtype)
    func = getattr(arr, method)
    result = func(skipna=skipna)
    if skipna:
        assert result == (0 if method == "min" else 1)
    else:
        assert result is pd.NA


@pytest.mark.parametrize("min_count", [0, 9])
def test_floating_array_prod(skipna, min_count, dtype):
    # [FUNCTION DOC]
    # Parametrized over min_count (zero and nine). Product skipping missing is two when min_count is zero, else NA.
    arr = pd.array([1.0, 2.0, None], dtype=dtype)
    result = arr.prod(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 2
    else:
        assert result is pd.NA
