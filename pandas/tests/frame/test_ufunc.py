# [MODULE DOC]
# Tests for applying numpy universal functions (ufuncs) to DataFrame objects:
# unary, binary and many-input ufuncs, argument pass-through, alignment of
# frames, and the unsupported frame/series combinations.
# Module-level data: a list of three dtype specs (numpy int, nullable Int, and a
# per-column mapping) used for parametrization.
from functools import partial
import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype

dtypes = [
    "int64",
    "Int64",
    {"A": "int64", "B": "Int64"},
]


@pytest.mark.parametrize("dtype", dtypes)
def test_unary_unary(dtype):
    # [FUNCTION DOC]
    # Verifies a unary ufunc returning one output (positive) applied to a frame
    # yields a frame with the same labels, for each dtype spec. Expected is the
    # numpy result wrapped in a frame and cast to the dtype.
    # unary input, unary output
    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    result = np.positive(df)
    expected = pd.DataFrame(
        np.positive(values), index=df.index, columns=df.columns
    ).astype(dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
def test_unary_binary(request, dtype):
    # [FUNCTION DOC]
    # Verifies a unary ufunc with two outputs (modf) returns a tuple of two frames,
    # each equal to the matching numpy output with same index and columns.
    # Parametrized over dtype specs; extension or per-column dtypes are marked as
    # expected failures because multiple outputs are not implemented for them.
    # unary input, binary output
    if is_extension_array_dtype(dtype) or isinstance(dtype, dict):
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple outputs not implemented."
            )
        )

    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    result_pandas = np.modf(df)
    assert isinstance(result_pandas, tuple)
    assert len(result_pandas) == 2
    expected_numpy = np.modf(values)

    for result, b in zip(result_pandas, expected_numpy):
        expected = pd.DataFrame(b, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
def test_binary_input_dispatch_binop(dtype):
    # [FUNCTION DOC]
    # Verifies that a binary ufunc (add) called with the same frame twice is
    # dispatched to the frame arithmetic and equals the numpy result cast to the
    # dtype. Parametrized over dtype specs.
    # binop ufuncs are dispatched to our dunder methods.
    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    result = np.add(df, df)
    expected = pd.DataFrame(
        np.add(values, values), index=df.index, columns=df.columns
    ).astype(dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func,arg,expected",
    [
        (np.add, 1, [2, 3, 4, 5]),
        (
            partial(np.add, where=[[False, True], [True, False]]),
            np.array([[1, 1], [1, 1]]),
            [0, 3, 4, 0],
        ),
        (np.power, np.array([[1, 1], [2, 2]]), [1, 2, 9, 16]),
        (np.subtract, 2, [-1, 0, 1, 2]),
        (
            partial(np.negative, where=np.array([[False, True], [True, False]])),
            None,
            [0, -2, -3, 0],
        ),
    ],
)
def test_ufunc_passes_args(func, arg, expected):
    # [FUNCTION DOC]
    # Verifies extra ufunc keyword arguments (output buffer, where mask) and
    # second operands are forwarded (GH#40662). Parametrized over the callable, the
    # second operand (or none for a one-argument ufunc) and the flattened expected
    # values.
    # Both the preallocated output array and the returned frame must hold the expected 2
    # by 2 result.
    # GH#40662
    arr = np.array([[1, 2], [3, 4]])
    df = pd.DataFrame(arr)
    result_inplace = np.zeros_like(arr)
    # 1-argument ufunc
    if arg is None:
        result = func(df, out=result_inplace)
    else:
        result = func(df, arg, out=result_inplace)

    expected = np.array(expected).reshape(2, 2)
    tm.assert_numpy_array_equal(result_inplace, expected)

    expected = pd.DataFrame(expected)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype_a", dtypes)
@pytest.mark.parametrize("dtype_b", dtypes)
def test_binary_input_aligns_columns(request, dtype_a, dtype_b):
    # [FUNCTION DOC]
    # Verifies a binary ufunc (heaviside) on two frames first aligns them on columns,
    # producing the union of columns with missing entries, and that with a plain
    # array as second operand no alignment occurs. Parametrized over two dtype
    # specs; any extension or per-column dtype is marked an expected failure. When
    # both specs are mappings the second frame's column B is renamed to C.
    if (
        is_extension_array_dtype(dtype_a)
        or isinstance(dtype_a, dict)
        or is_extension_array_dtype(dtype_b)
        or isinstance(dtype_b, dict)
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple inputs not implemented."
            )
        )

    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}).astype(dtype_a)

    if isinstance(dtype_a, dict) and isinstance(dtype_b, dict):
        dtype_b = dtype_b.copy()
        dtype_b["C"] = dtype_b.pop("B")
    df2 = pd.DataFrame({"A": [1, 2], "C": [3, 4]}).astype(dtype_b)
    # As of 2.0, align first before applying the ufunc
    result = np.heaviside(df1, df2)
    expected = np.heaviside(
        np.array([[1, 3, np.nan], [2, 4, np.nan]]),
        np.array([[1, np.nan, 3], [2, np.nan, 4]]),
    )
    expected = pd.DataFrame(expected, index=[0, 1], columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)

    result = np.heaviside(df1, df2.values)
    expected = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
def test_binary_input_aligns_index(request, dtype):
    # [FUNCTION DOC]
    # Verifies a binary ufunc on two frames aligns on the index (union with
    # missing rows) and that an array second operand skips alignment. Parametrized
    # over dtype specs; extension or per-column dtypes are expected failures.
    if is_extension_array_dtype(dtype) or isinstance(dtype, dict):
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple inputs not implemented."
            )
        )
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).astype(dtype)
    df2 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "c"]).astype(dtype)
    result = np.heaviside(df1, df2)
    expected = np.heaviside(
        np.array([[1, 3], [3, 4], [np.nan, np.nan]]),
        np.array([[1, 3], [np.nan, np.nan], [3, 4]]),
    )
    # TODO(FloatArray): this will be Float64Dtype.
    expected = pd.DataFrame(expected, index=["a", "b", "c"], columns=["A", "B"])
    tm.assert_frame_equal(result, expected)

    result = np.heaviside(df1, df2.values)
    expected = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]], columns=["A", "B"], index=["a", "b"]
    )
    tm.assert_frame_equal(result, expected)


def test_binary_frame_series_raises():
    # [FUNCTION DOC]
    # Verifies that a binary ufunc mixing a frame and a series, in either order,
    # raises a not-implemented error mentioning the ufunc name.
    # We don't currently implement
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(NotImplementedError, match="logaddexp"):
        np.logaddexp(df, df["A"])

    with pytest.raises(NotImplementedError, match="logaddexp"):
        np.logaddexp(df["A"], df)


def test_unary_accumulate_axis():
    # [FUNCTION DOC]
    # Verifies the accumulate method of a reducing ufunc (maximum) on frames
    # (GH#39259): default and explicit axis zero accumulate down columns (int
    # column becomes float when mixed with a float column), axis one accumulates across
    # columns.
    # https://github.com/pandas-dev/pandas/issues/39259
    df = pd.DataFrame({"a": [1, 3, 2, 4]})
    result = np.maximum.accumulate(df)
    expected = pd.DataFrame({"a": [1, 3, 3, 4]})
    tm.assert_frame_equal(result, expected)

    df = pd.DataFrame({"a": [1, 3, 2, 4], "b": [0.1, 4.0, 3.0, 2.0]})
    result = np.maximum.accumulate(df)
    # in theory could preserve int dtype for default axis=0
    expected = pd.DataFrame({"a": [1.0, 3.0, 3.0, 4.0], "b": [0.1, 4.0, 4.0, 4.0]})
    tm.assert_frame_equal(result, expected)

    result = np.maximum.accumulate(df, axis=0)
    tm.assert_frame_equal(result, expected)

    result = np.maximum.accumulate(df, axis=1)
    expected = pd.DataFrame({"a": [1.0, 3.0, 2.0, 4.0], "b": [1.0, 4.0, 3.0, 4.0]})
    tm.assert_frame_equal(result, expected)


def test_frame_outer_disallowed():
    # [FUNCTION DOC]
    # Verifies the outer method of a binary ufunc applied to frames raises a
    # not-implemented error with an empty message (deprecation enforced in 2.0).
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(NotImplementedError, match="^$"):
        # deprecation enforced in 2.0
        np.subtract.outer(df, df)


def test_alignment_deprecation_enforced():
    # [FUNCTION DOC]
    # Verifies enforced alignment behavior (2.0, GH#39184) for add on frames and series:
    # aligned inputs
    # give no warning; a frame with an array uses positional values; misaligned
    # frames give the union with missing values; a series aligns on columns; and mixing
    # a series first with a frame raises a not-implemented error about mixed inputs.
    # Enforced in 2.0
    # https://github.com/pandas-dev/pandas/issues/39184
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]})
    s1 = pd.Series([1, 2], index=["a", "b"])
    s2 = pd.Series([1, 2], index=["b", "c"])

    # binary dataframe / dataframe
    expected = pd.DataFrame({"a": [2, 4, 6], "b": [8, 10, 12]})

    with tm.assert_produces_warning(None):
        # aligned -> no warning!
        result = np.add(df1, df1)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, df2.values)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, df2)
    expected = pd.DataFrame({"a": [np.nan] * 3, "b": [5, 7, 9], "c": [np.nan] * 3})
    tm.assert_frame_equal(result, expected)

    result = np.add(df1.values, df2)
    expected = pd.DataFrame({"b": [2, 4, 6], "c": [8, 10, 12]})
    tm.assert_frame_equal(result, expected)

    # binary dataframe / series
    expected = pd.DataFrame({"a": [2, 3, 4], "b": [6, 7, 8]})

    with tm.assert_produces_warning(None):
        # aligned -> no warning!
        result = np.add(df1, s1)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, s2.values)
    tm.assert_frame_equal(result, expected)

    expected = pd.DataFrame(
        {"a": [np.nan] * 3, "b": [5.0, 6.0, 7.0], "c": [np.nan] * 3}
    )
    result = np.add(df1, s2)
    tm.assert_frame_equal(result, expected)

    msg = "Cannot apply ufunc <ufunc 'add'> to mixed DataFrame and Series inputs."
    with pytest.raises(NotImplementedError, match=msg):
        np.add(s2, df1)


@pytest.mark.single_cpu
def test_alignment_deprecation_many_inputs_enforced():
    # [FUNCTION DOC]
    # Verifies alignment enforcement for ufuncs with more than two inputs, using a
    # numba vectorized three-argument addition (skipped if numba is missing; single
    # cpu marker). Three misaligned frames give all-missing union; identical
    # frames give plain sums with no warning; a frame mixed with plain arrays raises a
    # value
    # error about broadcasting when shapes disagree, and index/columns come from the
    # first frame.
    # Enforced in 2.0
    # https://github.com/pandas-dev/pandas/issues/39184
    # test that the deprecation also works with > 2 inputs -> using a numba
    # written ufunc for this because numpy itself doesn't have such ufuncs
    numba = pytest.importorskip("numba")

    @numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
    def my_ufunc(x, y, z):
        # [FUNCTION DOC]
        # HELPER: numba-compiled elementwise ufunc taking three float64 values and
        # returning their sum; used by the
        # many-inputs alignment test.
        return x + y + z

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]})
    df3 = pd.DataFrame({"a": [1, 2, 3], "c": [4, 5, 6]})

    result = my_ufunc(df1, df2, df3)
    expected = pd.DataFrame(np.full((3, 3), np.nan), columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)

    # all aligned -> no warning
    with tm.assert_produces_warning(None):
        result = my_ufunc(df1, df1, df1)
    expected = pd.DataFrame([[3.0, 12.0], [6.0, 15.0], [9.0, 18.0]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # mixed frame / arrays
    msg = (
        r"operands could not be broadcast together with shapes \(3,3\) \(3,3\) \(3,2\)"
    )
    with pytest.raises(ValueError, match=msg):
        my_ufunc(df1, df2, df3.values)

    # single frame -> no warning
    with tm.assert_produces_warning(None):
        result = my_ufunc(df1, df2.values, df3.values)
    tm.assert_frame_equal(result, expected)

    # takes indices of first frame
    msg = (
        r"operands could not be broadcast together with shapes \(3,2\) \(3,3\) \(3,3\)"
    )
    with pytest.raises(ValueError, match=msg):
        my_ufunc(df1.values, df2, df3)


def test_array_ufuncs_for_many_arguments():
    # [FUNCTION DOC]
    # Verifies a ufunc created from a three-argument Python function works on
    # frames (GH39853): two frames and a scalar give an object-dtype frame of sums,
    # while a frame/frame/series combination raises a not-implemented error about mixed
    # inputs.
    # GH39853
    def add3(x, y, z):
        # [FUNCTION DOC]
        # HELPER: three-argument addition used to build the ufunc in the enclosing test;
        # returns x plus y plus z.
        return x + y + z

    ufunc = np.frompyfunc(add3, 3, 1)
    df = pd.DataFrame([[1, 2], [3, 4]])

    result = ufunc(df, df, 1)
    expected = pd.DataFrame([[3, 5], [7, 9]], dtype=object)
    tm.assert_frame_equal(result, expected)

    ser = pd.Series([1, 2])
    msg = (
        "Cannot apply ufunc <ufunc 'add3 (vectorized)'> "
        "to mixed DataFrame and Series inputs."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        ufunc(df, df, ser)
