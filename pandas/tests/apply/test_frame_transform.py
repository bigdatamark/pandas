# [MODULE DOC]
# Tests for DataFrame.transform (and Series.transform where frame_or_series is parametrized): ufuncs,
# list-like and dict-like arguments, user functions, bad dtypes, failures, argument passing and empty frames.
# Module-level names: a list of kernels expected not to fail on object data and a derived list of
# kernels expected to raise. Uses the zip_frames helper from the frame tests.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames


def unpack_obj(obj, klass, axis):
    # [FUNCTION DOC]
    # HELPER: given obj (a frame), klass (DataFrame or Series class) and axis, returns the frame itself
    # when klass is DataFrame; otherwise returns the column A of the frame as a Series, and skips the test
    # when the axis is not zero because the test only applies to frames in that case.
    """
    Helper to ensure we have the right type of object for a test parametrized
    over frame_or_series.
    """
    if klass is not DataFrame:
        obj = obj["A"]
        if axis != 0:
            pytest.skip(f"Test is only for DataFrame with axis={axis}")
    return obj


def test_transform_ufunc(axis, float_frame, frame_or_series):
    # [FUNCTION DOC]
    # GH 35964: transform with a numpy ufunc (square root) along the axis must equal the ufunc applied
    # directly to the object (with floating point errors ignored). Run for frame and for series (axis zero only).
    # GH 35964
    obj = unpack_obj(float_frame, frame_or_series, axis)

    with np.errstate(all="ignore"):
        f_sqrt = np.sqrt(obj)

    # ufunc
    result = obj.transform(np.sqrt, axis=axis)
    expected = f_sqrt
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),
        (np.array([np.sqrt]), ["sqrt"]),
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),
    ],
)
def test_transform_listlike(axis, float_frame, ops, names):
    # [FUNCTION DOC]
    # GH 35964: list-like transform parametrized by ops (lists or arrays of numpy functions) and names.
    # Expected: the results of each function zipped together into a frame with MultiIndex labels on the
    # transformed axis (columns for axis zero, index otherwise) formed from the original labels and the names.
    # GH 35964
    other_axis = 1 if axis in {0, "index"} else 0
    with np.errstate(all="ignore"):
        expected = zip_frames([op(float_frame) for op in ops], axis=other_axis)
    if axis in {0, "index"}:
        expected.columns = MultiIndex.from_product([float_frame.columns, names])
    else:
        expected.index = MultiIndex.from_product([float_frame.index, names])
    result = float_frame.transform(ops, axis=axis)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ops", [[], np.array([])])
def test_transform_empty_listlike(float_frame, ops, frame_or_series):
    # [FUNCTION DOC]
    # Transform with an empty list or array of functions must raise a value error stating that no
    # transform functions were provided. Run for frame and series.
    obj = unpack_obj(float_frame, frame_or_series, 0)

    with pytest.raises(ValueError, match="No transform functions were provided"):
        obj.transform(ops)


def test_transform_listlike_func_with_args():
    # [FUNCTION DOC]
    # GH 50624: list-like transform where functions take extra arguments. An extra keyword argument that one
    # function does not accept raises a type error naming that function; a valid combination gives a frame with
    # two columns labelled by the original column and function name and values equal to the input plus extras.
    # GH 50624
    df = DataFrame({"x": [1, 2, 3]})

    def foo1(x, a=1, c=0):
        # [FUNCTION DOC]
        # HELPER: adds a and c to the input; first of the two functions under test.
        return x + a + c

    def foo2(x, b=2, c=0):
        # [FUNCTION DOC]
        # HELPER: adds b and c to the input; second of the two functions under test.
        return x + b + c

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        df.transform([foo1, foo2], 0, 3, b=3, c=4)

    result = df.transform([foo1, foo2], 0, 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],
        columns=MultiIndex.from_tuples([("x", "foo1"), ("x", "foo2")]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("box", [dict, Series])
def test_transform_dictlike(axis, float_frame, box):
    # [FUNCTION DOC]
    # GH 35964: dict-like transform with a dict or Series mapping a single label (the first column for
    # axis zero, the first row label otherwise) to the absolute function. Expected: the corresponding slice
    # of the frame transformed by the absolute function.
    # GH 35964
    if axis in (0, "index"):
        e = float_frame.columns[0]
        expected = float_frame[[e]].transform(np.abs)
    else:
        e = float_frame.index[0]
        expected = float_frame.iloc[[0]].transform(np.abs)
    result = float_frame.transform(box({e: np.abs}), axis=axis)
    tm.assert_frame_equal(result, expected)


def test_transform_dictlike_mixed():
    # [FUNCTION DOC]
    # GH 40018: dict-like transform whose values mix a list of functions and a single name; result has
    # MultiIndex columns and equals the expected frame of square roots and absolute values.
    # GH 40018 - mix of lists and non-lists in values of a dictionary
    df = DataFrame({"a": [1, 2], "b": [1, 4], "c": [1, 4]})
    result = df.transform({"b": ["sqrt", "abs"], "c": "sqrt"})
    expected = DataFrame(
        [[1.0, 1, 1.0], [2.0, 4, 2.0]],
        columns=MultiIndex([("b", "c"), ("sqrt", "abs")], [(0, 0, 1), (0, 1, 0)]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {},
        {"A": []},
        {"A": [], "B": "cumsum"},
        {"A": "cumsum", "B": []},
        {"A": [], "B": ["cumsum"]},
        {"A": ["cumsum"], "B": []},
    ],
)
def test_transform_empty_dictlike(float_frame, ops, frame_or_series):
    # [FUNCTION DOC]
    # Parametrized over dict-like arguments that are empty or contain empty lists for a key; transform
    # must raise a value error that no transform functions were provided. Run for frame and series.
    obj = unpack_obj(float_frame, frame_or_series, 0)

    with pytest.raises(ValueError, match="No transform functions were provided"):
        obj.transform(ops)


@pytest.mark.parametrize("use_apply", [True, False])
def test_transform_udf(axis, float_frame, use_apply, frame_or_series):
    # [FUNCTION DOC]
    # GH 35964: user function transform, via apply or on the whole object depending on use_apply.
    # The function raises a value error whenever it receives the "wrong" kind of input so transform has to
    # fall back to the other path; otherwise it adds one. The result must equal the object plus one.
    # GH 35964
    obj = unpack_obj(float_frame, frame_or_series, axis)

    # transform uses UDF either via apply or passing the entire DataFrame
    def func(x):
        # [FUNCTION DOC]
        # HELPER: adds one, but raises a value error when the argument's type does not match the path under test
        # (a whole container when use_apply is true, an element otherwise), forcing the fallback path.
        # transform is using apply iff x is not a DataFrame
        if use_apply == isinstance(x, frame_or_series):
            # Force transform to fallback
            raise ValueError
        return x + 1

    result = obj.transform(func, axis=axis)
    expected = obj + 1
    tm.assert_equal(result, expected)


wont_fail = ["ffill", "bfill", "fillna", "pad", "backfill", "shift"]
frame_kernels_raise = [x for x in frame_transform_kernels if x not in wont_fail]


@pytest.mark.parametrize("op", [*frame_kernels_raise, lambda x: x + 1])
def test_transform_bad_dtype(op, frame_or_series, request):
    # [FUNCTION DOC]
    # GH 35964: parametrized over transformation kernels expected to raise plus a lambda adding one.
    # Uses a frame (or series) of object values. The ngroup kernel is marked an expected failure (value error).
    # Every form of the argument (bare op, list, dict of op, dict of list) must raise a type error with a
    # message about unsupported operand types or comparison of types.
    # GH 35964
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )

    obj = DataFrame({"A": 3 * [object]})  # DataFrame that will fail on most transforms
    obj = tm.get_obj(obj, frame_or_series)
    error = TypeError
    msg = "|".join(
        [
            "not supported between instances of 'type' and 'type'",
            "unsupported operand type",
        ]
    )

    with pytest.raises(error, match=msg):
        obj.transform(op)
    with pytest.raises(error, match=msg):
        obj.transform([op])
    with pytest.raises(error, match=msg):
        obj.transform({"A": op})
    with pytest.raises(error, match=msg):
        obj.transform({"A": [op]})


@pytest.mark.parametrize("op", frame_kernels_raise)
def test_transform_failure_typeerror(request, op):
    # [FUNCTION DOC]
    # GH 35964: for each kernel expected to raise, a frame with an object column and an integer column
    # must raise a type error with the unsupported operand message when transformed with a list, a dict, a dict
    # of lists, and a dict of lists where one list also includes the shift kernel. The ngroup kernel is an expected failure.
    # GH 35964

    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )

    # Using object makes most transform kernels fail
    df = DataFrame({"A": 3 * [object], "B": [1, 2, 3]})
    error = TypeError
    msg = "|".join(
        [
            "not supported between instances of 'type' and 'type'",
            "unsupported operand type",
        ]
    )

    with pytest.raises(error, match=msg):
        df.transform([op])

    with pytest.raises(error, match=msg):
        df.transform({"A": op, "B": op})

    with pytest.raises(error, match=msg):
        df.transform({"A": [op], "B": [op]})

    with pytest.raises(error, match=msg):
        df.transform({"A": [op, "shift"], "B": [op]})


def test_transform_failure_valueerror():
    # [FUNCTION DOC]
    # GH 40211: when the user function raises a value error (sum of all values below ten), transform with list,
    # dict, dict of lists or dict of lists with shift must raise a value error with message that the transform
    # function failed.
    # GH 40211
    def op(x):
        # [FUNCTION DOC]
        # HELPER: raises a value error when the total of the input is below ten, otherwise returns it unchanged.
        if np.sum(np.sum(x)) < 10:
            raise ValueError
        return x

    df = DataFrame({"A": [1, 2, 3], "B": [400, 500, 600]})
    msg = "Transform function failed"

    with pytest.raises(ValueError, match=msg):
        df.transform([op])

    with pytest.raises(ValueError, match=msg):
        df.transform({"A": op, "B": op})

    with pytest.raises(ValueError, match=msg):
        df.transform({"A": [op], "B": [op]})

    with pytest.raises(ValueError, match=msg):
        df.transform({"A": [op, "shift"], "B": [op]})


@pytest.mark.parametrize("use_apply", [True, False])
def test_transform_passes_args(use_apply, frame_or_series):
    # [FUNCTION DOC]
    # GH 35964: transform passes extra positional and keyword arguments through to the user function.
    # Parametrized by use_apply; the function raises to force fallback when given the wrong input kind,
    # and asserts that it received the expected two positional values and the keyword value.
    # GH 35964
    # transform uses UDF either via apply or passing the entire DataFrame
    expected_args = [1, 2]
    expected_kwargs = {"c": 3}

    def f(x, a, b, c):
        # [FUNCTION DOC]
        # HELPER: user function that asserts the extra arguments equal expected values and returns the
        # input unchanged, raising a value error when the input kind is the wrong one for the path under test.
        # transform is using apply iff x is not a DataFrame
        if use_apply == isinstance(x, frame_or_series):
            # Force transform to fallback
            raise ValueError
        assert [a, b] == expected_args
        assert c == expected_kwargs["c"]
        return x

    frame_or_series([1]).transform(f, 0, *expected_args, **expected_kwargs)


def test_transform_empty_dataframe():
    # [FUNCTION DOC]
    # Issue 39636: transforming an empty frame (two named columns) with a function adding ten gives a
    # frame equal to the input, and the same for a single column as a series.
    # https://github.com/pandas-dev/pandas/issues/39636
    df = DataFrame([], columns=["col1", "col2"])
    result = df.transform(lambda x: x + 10)
    tm.assert_frame_equal(result, df)

    result = df["col1"].transform(lambda x: x + 10)
    tm.assert_series_equal(result, df["col1"])
