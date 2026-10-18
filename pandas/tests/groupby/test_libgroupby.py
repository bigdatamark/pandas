# [MODULE DOC]
# Tests for the low-level compiled groupby kernels (variance, OHLC, cumulative sum/product, mean, sum) called directly with prepared output, count, value and label arrays instead of going through the groupby API.
# Imports the compiled groupby module and its kernel functions, so requires the compiled extension to be built.
import numpy as np
import pytest

from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
    group_cumprod,
    group_cumsum,
    group_mean,
    group_sum,
    group_var,
)

from pandas.core.dtypes.common import ensure_platform_int

from pandas import isna
import pandas._testing as tm


@pytest.mark.parametrize("dtype, rtol", [("float32", 1e-2), ("float64", 1e-5)])
class TestGroupVar:
    # [CLASS DOC]
    # Tests of the group variance kernel, parametrized at class level over float32 (loose tolerance) and float64 (tight tolerance); each test fills an output array of NaN, counts and checks against a reference standard deviation computed with numpy.
    def test_group_var_generic_1d(self, dtype, rtol):
        # [FUNCTION DOC]
        # Variance for one value column and five groups of three values each: matches the sample variance and counts of three per group.
        prng = np.random.default_rng(2)

        out = (np.nan * np.ones((5, 1))).astype(dtype)
        counts = np.zeros(5, dtype="int64")
        values = 10 * prng.random((15, 1)).astype(dtype)
        labels = np.tile(np.arange(5), (3,)).astype("intp")

        expected_out = (
            np.squeeze(values).reshape((5, 3), order="F").std(axis=1, ddof=1) ** 2
        )[:, np.newaxis]
        expected_counts = counts + 3

        group_var(out, counts, values, labels)
        assert np.allclose(out, expected_out, rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_1d_flat_labels(self, dtype, rtol):
        # [FUNCTION DOC]
        # Variance for one group of five values: matches the overall sample variance and counts five.
        prng = np.random.default_rng(2)

        out = (np.nan * np.ones((1, 1))).astype(dtype)
        counts = np.zeros(1, dtype="int64")
        values = 10 * prng.random((5, 1)).astype(dtype)
        labels = np.zeros(5, dtype="intp")

        expected_out = np.array([[values.std(ddof=1) ** 2]])
        expected_counts = counts + 5

        group_var(out, counts, values, labels)

        assert np.allclose(out, expected_out, rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_2d_all_finite(self, dtype, rtol):
        # [FUNCTION DOC]
        # Variance for two finite columns and five groups of two: matches reference and counts two per group.
        prng = np.random.default_rng(2)

        out = (np.nan * np.ones((5, 2))).astype(dtype)
        counts = np.zeros(5, dtype="int64")
        values = 10 * prng.random((10, 2)).astype(dtype)
        labels = np.tile(np.arange(5), (2,)).astype("intp")

        expected_out = np.std(values.reshape(2, 5, 2), ddof=1, axis=0) ** 2
        expected_counts = counts + 2

        group_var(out, counts, values, labels)
        assert np.allclose(out, expected_out, rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_2d_some_nan(self, dtype, rtol):
        # [FUNCTION DOC]
        # Variance when the second column is all NaN: the first column matches reference and the second yields NaN for every group; counts remain two.
        prng = np.random.default_rng(2)

        out = (np.nan * np.ones((5, 2))).astype(dtype)
        counts = np.zeros(5, dtype="int64")
        values = 10 * prng.random((10, 2)).astype(dtype)
        values[:, 1] = np.nan
        labels = np.tile(np.arange(5), (2,)).astype("intp")

        expected_out = np.vstack(
            [
                values[:, 0].reshape(5, 2, order="F").std(ddof=1, axis=1) ** 2,
                np.nan * np.ones(5),
            ]
        ).T.astype(dtype)
        expected_counts = counts + 2

        group_var(out, counts, values, labels)
        tm.assert_almost_equal(out, expected_out, rtol=0.5e-06)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_constant(self, dtype, rtol):
        # [FUNCTION DOC]
        # Regression test (issue 10448): variance of a constant-valued group is non-negative and approximately zero, with count three.
        # Regression test from GH 10448.

        out = np.array([[np.nan]], dtype=dtype)
        counts = np.array([0], dtype="int64")
        values = 0.832845131556193 * np.ones((3, 1), dtype=dtype)
        labels = np.zeros(3, dtype="intp")

        group_var(out, counts, values, labels)

        assert counts[0] == 3
        assert out[0, 0] >= 0
        tm.assert_almost_equal(out[0, 0], 0.0)


def test_group_var_large_inputs():
    # [FUNCTION DOC]
    # Variance of a million values with a large offset (numerical stability): the result is close to one twelfth and the count is a million.
    dtype = np.float64
    prng = np.random.default_rng(2)

    out = np.array([[np.nan]], dtype=dtype)
    counts = np.array([0], dtype="int64")
    values = (prng.random((10**6, 1)) + 10**12).astype(dtype)
    labels = np.zeros(10**6, dtype="intp")

    group_var(out, counts, values, labels)

    assert counts[0] == 10**6
    tm.assert_almost_equal(out[0, 0], 1.0 / 12, rtol=0.5e-3)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_group_ohlc(dtype):
    # [FUNCTION DOC]
    # OHLC kernel for float32/float64 over three bins of size 6, 6, 8: output rows are open, high, low, close per bin and counts match; after setting the first bin to NaN the first output row becomes NaN.
    obj = np.array(np.random.default_rng(2).standard_normal(20), dtype=dtype)

    bins = np.array([6, 12, 20])
    out = np.zeros((3, 4), dtype)
    counts = np.zeros(len(out), dtype=np.int64)
    labels = ensure_platform_int(np.repeat(np.arange(3), np.diff(np.r_[0, bins])))

    func = libgroupby.group_ohlc
    func(out, counts, obj[:, None], labels)

    def _ohlc(group):
        # [FUNCTION DOC]
        # HELPER: reference OHLC for a group (first, maximum, minimum, last), or four NaN when all values are missing.
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group[0], group.max(), group.min(), group[-1]]

    expected = np.array([_ohlc(obj[:6]), _ohlc(obj[6:12]), _ohlc(obj[12:])])

    tm.assert_almost_equal(out, expected)
    tm.assert_numpy_array_equal(counts, np.array([6, 6, 8], dtype=np.int64))

    obj[:6] = np.nan
    func(out, counts, obj[:, None], labels)
    expected[0] = np.nan
    tm.assert_almost_equal(out, expected)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize(
    "pd_op, np_op",
    [
        (group_cumsum, np.cumsum),
        (group_cumprod, np.cumprod),
    ],
)
def test_cython_group_transform(dtype, pd_op, np_op):
    # [FUNCTION DOC]
    # Cumulative sum and cumulative product kernels on a single group of integer or float data (issue 4095) match the numpy cumulative functions; parametrized over dtype and operation.
    # see gh-4095
    is_datetimelike = False

    data = np.array([[1], [2], [3], [4]], dtype=dtype)
    answer = np.zeros_like(data)

    labels = np.array([0, 0, 0, 0], dtype=np.intp)
    ngroups = 1
    pd_op(answer, data, labels, ngroups, is_datetimelike)

    tm.assert_numpy_array_equal(np_op(data), answer[:, 0], check_dtype=False)


def test_cython_group_transform_algos():
    # [FUNCTION DOC]
    # Cumulative product and sum kernels with a NaN in the data propagate NaN only at that position (floats), and the cumulative sum on timedelta data viewed as integers with the datetime-like flag gives running totals.
    # see gh-4095
    is_datetimelike = False

    # with nans
    labels = np.array([0, 0, 0, 0, 0], dtype=np.intp)
    ngroups = 1

    data = np.array([[1], [2], [3], [np.nan], [4]], dtype="float64")
    actual = np.zeros_like(data)
    actual.fill(np.nan)
    group_cumprod(actual, data, labels, ngroups, is_datetimelike)
    expected = np.array([1, 2, 6, np.nan, 24], dtype="float64")
    tm.assert_numpy_array_equal(actual[:, 0], expected)

    actual = np.zeros_like(data)
    actual.fill(np.nan)
    group_cumsum(actual, data, labels, ngroups, is_datetimelike)
    expected = np.array([1, 3, 6, np.nan, 10], dtype="float64")
    tm.assert_numpy_array_equal(actual[:, 0], expected)

    # timedelta
    is_datetimelike = True
    data = np.array([np.timedelta64(1, "ns")] * 5, dtype="m8[ns]")[:, None]
    actual = np.zeros_like(data, dtype="int64")
    group_cumsum(actual, data.view("int64"), labels, ngroups, is_datetimelike)
    expected = np.array(
        [
            np.timedelta64(1, "ns"),
            np.timedelta64(2, "ns"),
            np.timedelta64(3, "ns"),
            np.timedelta64(4, "ns"),
            np.timedelta64(5, "ns"),
        ]
    )
    tm.assert_numpy_array_equal(actual[:, 0].view("m8[ns]"), expected)


def test_cython_group_mean_datetimelike():
    # [FUNCTION DOC]
    # Mean kernel with the datetime-like flag ignores NaT values (stored as the minimum integer) and returns the mean of the rest.
    actual = np.zeros(shape=(1, 1), dtype="float64")
    counts = np.array([0], dtype="int64")
    data = (
        np.array(
            [
                np.timedelta64(2, "ns"),
                np.timedelta64(4, "ns"),
                np.timedelta64("NaT", "ns"),
            ],
            dtype="m8[ns]",
        )[:, None]
        .view("int64")
        .astype("float64")
    )
    labels = np.zeros(len(data), dtype=np.intp)

    group_mean(actual, counts, data, labels, is_datetimelike=True)

    tm.assert_numpy_array_equal(actual[:, 0], np.array([3], dtype="float64"))


def test_cython_group_mean_wrong_min_count():
    # [FUNCTION DOC]
    # Mean kernel called with a minimum count of zero while the datetime-like flag is true raises an assertion error mentioning the minimum count.
    actual = np.zeros(shape=(1, 1), dtype="float64")
    counts = np.zeros(1, dtype="int64")
    data = np.zeros(1, dtype="float64")[:, None]
    labels = np.zeros(1, dtype=np.intp)

    with pytest.raises(AssertionError, match="min_count"):
        group_mean(actual, counts, data, labels, is_datetimelike=True, min_count=0)


def test_cython_group_mean_not_datetimelike_but_has_NaT_values():
    # [FUNCTION DOC]
    # Mean kernel without the datetime-like flag treats NaT-valued integers as ordinary numbers and returns their numeric average.
    actual = np.zeros(shape=(1, 1), dtype="float64")
    counts = np.array([0], dtype="int64")
    data = (
        np.array(
            [np.timedelta64("NaT", "ns"), np.timedelta64("NaT", "ns")],
            dtype="m8[ns]",
        )[:, None]
        .view("int64")
        .astype("float64")
    )
    labels = np.zeros(len(data), dtype=np.intp)

    group_mean(actual, counts, data, labels, is_datetimelike=False)

    tm.assert_numpy_array_equal(
        actual[:, 0], np.array(np.divide(np.add(data[0], data[1]), 2), dtype="float64")
    )


def test_cython_group_mean_Inf_at_beginning_and_end():
    # [FUNCTION DOC]
    # Issue 50367: mean kernel with infinite values at the start and end of groups gives infinity rather than NaN.
    # GH 50367
    actual = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype="float64")
    counts = np.array([0, 0], dtype="int64")
    data = np.array(
        [[np.inf, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5, np.inf]],
        dtype="float64",
    )
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)

    group_mean(actual, counts, data, labels, is_datetimelike=False)

    expected = np.array([[np.inf, 3], [3, np.inf]], dtype="float64")

    tm.assert_numpy_array_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "values, out",
    [
        ([[np.inf], [np.inf], [np.inf]], [[np.inf], [np.inf]]),
        ([[np.inf], [np.inf], [-np.inf]], [[np.inf], [np.nan]]),
        ([[np.inf], [-np.inf], [np.inf]], [[np.inf], [np.nan]]),
        ([[np.inf], [-np.inf], [-np.inf]], [[np.inf], [-np.inf]]),
    ],
)
def test_cython_group_sum_Inf_at_beginning_and_end(values, out):
    # [FUNCTION DOC]
    # Issue 53606: sum kernel with infinity and negative infinity combinations: matching signs give the infinity, opposite signs give NaN; parametrized over input values and expected output.
    # GH #53606
    actual = np.array([[np.nan], [np.nan]], dtype="float64")
    counts = np.array([0, 0], dtype="int64")
    data = np.array(values, dtype="float64")
    labels = np.array([0, 1, 1], dtype=np.intp)

    group_sum(actual, counts, data, labels, None, is_datetimelike=False)

    expected = np.array(out, dtype="float64")

    tm.assert_numpy_array_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "values, expected_values",
    [
        (np.finfo(np.float64).max, [[np.inf]]),
        (np.finfo(np.float64).min, [[-np.inf]]),
        (
            np.complex128(np.finfo(np.float64).min + np.finfo(np.float64).max * 1j),
            [[complex(-np.inf, np.inf)]],
        ),
        (
            np.complex128(np.finfo(np.float64).max + np.finfo(np.float64).min * 1j),
            [[complex(np.inf, -np.inf)]],
        ),
        (
            np.complex128(np.finfo(np.float64).max + np.finfo(np.float64).max * 1j),
            [[complex(np.inf, np.inf)]],
        ),
        (
            np.complex128(np.finfo(np.float64).min + np.finfo(np.float64).min * 1j),
            [[complex(-np.inf, -np.inf)]],
        ),
        (
            np.complex128(3.0 + np.finfo(np.float64).min * 1j),
            [[complex(9.0, -np.inf)]],
        ),
        (
            np.complex128(np.finfo(np.float64).max + 3 * 1j),
            [[complex(np.inf, 9.0)]],
        ),
    ],
)
def test_cython_group_sum_overflow(values, expected_values):
    # [FUNCTION DOC]
    # Issue 60303: sum kernel overflow of float and complex values saturates to the signed infinity per component (and keeps finite parts finite); parametrized over values and expected result.
    # GH-60303
    data = np.array([[values] for _ in range(3)])
    labels = np.array([0, 0, 0], dtype=np.intp)
    counts = np.array([0], dtype="int64")

    expected = np.array(expected_values, dtype=values.dtype)
    actual = np.zeros_like(expected)

    group_sum(actual, counts, data, labels, None, is_datetimelike=False)

    tm.assert_numpy_array_equal(actual, expected)
