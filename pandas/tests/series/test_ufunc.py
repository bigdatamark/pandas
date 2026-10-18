# [MODULE DOC]
# Tests for NumPy universal functions (ufuncs) applied to Series: unary, binary (with arrays, indexes, Series, scalars, other sequence types), multi-output, name handling, object dtype, reductions via reduce, outer, matrix multiplication, many-argument ufuncs and truncation. Cases are run for sparse and dense data where applicable.
# Defines fixtures: a choice of binary ufunc, a sparse/dense switch, a pair of random mostly-zero integer arrays, and a collection of array-likes of many dtypes for reduction tests. No import-time side effects beyond constructing the fixture parameter values.

from collections import deque
import re
import string

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray


@pytest.fixture(params=[np.add, np.logaddexp])
def ufunc(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over two binary ufuncs (addition and log-add-exp); returns the chosen ufunc. The comment calls it the dunder op.
    # dunder op
    return request.param


@pytest.fixture(
    params=[pytest.param(True, marks=pytest.mark.fails_arm_wheels), False],
    ids=["sparse", "dense"],
)
def sparse(request):
    # [FUNCTION DOC]
    # HELPER: fixture returning true or false for sparse or dense data (ids sparse, dense); the sparse case is marked as failing on ARM wheels.
    return request.param


@pytest.fixture
def arrays_for_binary_ufunc():
    # [FUNCTION DOC]
    # HELPER: fixture returning a pair of length-100 int64 arrays of random integers below ten (seeded identically, so the underlying draws coincide), with every third element of the first and every fourth of the second set to zero.
    # KNOWN ISSUE: both arrays are drawn from separately created generators with the same seed, so before zeroing they are identical, and the pair is less random than the docstring suggests.
    """
    A pair of random, length-100 integer-dtype arrays, that are mostly 0.
    """
    a1 = np.random.default_rng(2).integers(0, 10, 100, dtype="int64")
    a2 = np.random.default_rng(2).integers(0, 10, 100, dtype="int64")
    a1[::3] = 0
    a2[::4] = 0
    return a1, a2


@pytest.mark.parametrize("ufunc", [np.positive, np.floor, np.exp])
def test_unary_ufunc(ufunc, sparse):
    # [FUNCTION DOC]
    # Verifies that applying a unary ufunc (positive, floor or exponential, parametrized) to a Series of ten small ints (every other zero; a sparse array of int64 with fill zero when the sparse fixture is true) with letter index and a name equals a Series built from the ufunc on the raw data with the same index and name.
    # Test that ufunc(pd.Series) == pd.Series(ufunc)
    arr = np.random.default_rng(2).integers(0, 10, 10, dtype="int64")
    arr[::2] = 0
    if sparse:
        arr = SparseArray(arr, dtype=pd.SparseDtype("int64", 0))

    index = list(string.ascii_letters[:10])
    name = "name"
    series = pd.Series(arr, index=index, name=name)

    result = ufunc(series)
    expected = pd.Series(ufunc(arr), index=index, name=name)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_array(flip, sparse, ufunc, arrays_for_binary_ufunc):
    # [FUNCTION DOC]
    # Verifies a binary ufunc between a Series and a plain array (sparse or dense, fixtures), in straight order and flipped order (array first): the result equals a Series of the ufunc on the raw arrays and keeps the Series' name.
    # Test that ufunc(pd.Series(a), array) == pd.Series(ufunc(a, b))
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    name = "name"  # op(pd.Series, array) preserves the name.
    series = pd.Series(a1, name=name)
    other = a2

    array_args = (a1, a2)
    series_args = (series, other)  # ufunc(series, array)

    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)  # ufunc(array, series)

    expected = pd.Series(ufunc(*array_args), name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_index(flip, sparse, ufunc, arrays_for_binary_ufunc):
    # [FUNCTION DOC]
    # Verifies a binary ufunc between a Series and an int64 Index (straight or flipped): the result is a Series, even when the Index is the first argument, equal to the ufunc on the raw arrays with the name preserved.
    # KNOWN ISSUE: the comment at the top of the test mentions two Series arguments, but the second operand in the body is an Index.
    # Test that
    #   * func(pd.Series(a), pd.Series(b)) == pd.Series(ufunc(a, b))
    #   * ufunc(Index, pd.Series) dispatches to pd.Series (returns a pd.Series)
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    name = "name"  # op(pd.Series, array) preserves the name.
    series = pd.Series(a1, name=name)

    other = pd.Index(a2, name=name).astype("int64")

    array_args = (a1, a2)
    series_args = (series, other)  # ufunc(series, array)

    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)  # ufunc(array, series)

    expected = pd.Series(ufunc(*array_args), name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("shuffle", [True, False], ids=["unaligned", "aligned"])
@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_series(
    flip, shuffle, sparse, ufunc, arrays_for_binary_ufunc
):
    # [FUNCTION DOC]
    # Verifies a binary ufunc between two Series with equal names, either aligned or with the second shuffled by a random permutation (unaligned), straight or flipped. The expected result is built from the raw arrays and an index taken from aligning the two Series (in the order matching the operand order), so misaligned labels are aligned before the operation.
    # Test that
    #   * func(pd.Series(a), pd.Series(b)) == pd.Series(ufunc(a, b))
    #   with alignment between the indices
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    name = "name"  # op(pd.Series, array) preserves the name.
    series = pd.Series(a1, name=name)
    other = pd.Series(a2, name=name)

    idx = np.random.default_rng(2).permutation(len(a1))

    if shuffle:
        other = other.take(idx)
        if flip:
            index = other.align(series)[0].index
        else:
            index = series.align(other)[0].index
    else:
        index = series.index

    array_args = (a1, a2)
    series_args = (series, other)  # ufunc(series, array)

    if flip:
        array_args = tuple(reversed(array_args))
        series_args = tuple(reversed(series_args))  # ufunc(array, series)

    expected = pd.Series(ufunc(*array_args), index=index, name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flip", [True, False])
def test_binary_ufunc_scalar(ufunc, sparse, flip, arrays_for_binary_ufunc):
    # [FUNCTION DOC]
    # Verifies a binary ufunc between a Series and the scalar two, in both operand orders (parametrized flip, sparse and dense): result equals the ufunc on the raw array with the scalar, with the name preserved.
    # Test that
    #   * ufunc(pd.Series, scalar) == pd.Series(ufunc(array, scalar))
    #   * ufunc(pd.Series, scalar) == ufunc(scalar, pd.Series)
    arr, _ = arrays_for_binary_ufunc
    if sparse:
        arr = SparseArray(arr)
    other = 2
    series = pd.Series(arr, name="name")

    series_args = (series, other)
    array_args = (arr, other)

    if flip:
        series_args = tuple(reversed(series_args))
        array_args = tuple(reversed(array_args))

    expected = pd.Series(ufunc(*array_args), name="name")
    result = ufunc(*series_args)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.divmod])  # TODO: np.modf, np.frexp
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_multiple_output_binary_ufuncs(ufunc, sparse, shuffle, arrays_for_binary_ufunc):
    # [FUNCTION DOC]
    # Verifies a multiple-output binary ufunc (integer divmod; the TODO lists modf and frexp as untested) between two Series, aligned or with the second randomly sampled-and-shuffled (so alignment is needed). Zeros in the input arrays are replaced with ones in place to avoid a known divide-by-zero issue (26987); divide-by-zero warnings are filtered. The result must be a tuple of two Series equal to the quotient and remainder of the raw arrays.
    # Test that
    #  the same conditions from binary_ufunc_scalar apply to
    #  ufuncs with multiple outputs.

    a1, a2 = arrays_for_binary_ufunc
    # work around https://github.com/pandas-dev/pandas/issues/26987
    a1[a1 == 0] = 1
    a2[a2 == 0] = 1

    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    s1 = pd.Series(a1)
    s2 = pd.Series(a2)

    if shuffle:
        # ensure we align before applying the ufunc
        s2 = s2.sample(frac=1)

    expected = ufunc(a1, a2)
    assert isinstance(expected, tuple)

    result = ufunc(s1, s2)
    assert isinstance(result, tuple)
    tm.assert_series_equal(result[0], pd.Series(expected[0]))
    tm.assert_series_equal(result[1], pd.Series(expected[1]))


def test_multiple_output_ufunc(sparse, arrays_for_binary_ufunc):
    # [FUNCTION DOC]
    # Verifies that a multiple-output unary ufunc (splitting into fractional and integral parts) applied to a Series (sparse or dense) returns a tuple of two named Series equal to the ufunc on the raw array.
    # Test that the same conditions from unary input apply to multi-output
    # ufuncs
    arr, _ = arrays_for_binary_ufunc

    if sparse:
        arr = SparseArray(arr)

    series = pd.Series(arr, name="name")
    result = np.modf(series)
    expected = np.modf(arr)

    assert isinstance(result, tuple)
    assert isinstance(expected, tuple)

    tm.assert_series_equal(result[0], pd.Series(expected[0], name="name"))
    tm.assert_series_equal(result[1], pd.Series(expected[1], name="name"))


def test_binary_ufunc_drops_series_name(ufunc, sparse, arrays_for_binary_ufunc):
    # [FUNCTION DOC]
    # Verifies that a binary ufunc between two Series with different names (a and b) gives a result with no name.
    # Drop the names when they differ.
    a1, a2 = arrays_for_binary_ufunc
    s1 = pd.Series(a1, name="a")
    s2 = pd.Series(a2, name="b")

    result = ufunc(s1, s2)
    assert result.name is None


def test_object_series_ok():
    # [FUNCTION DOC]
    # Verifies that the add ufunc works on an object-dtype Series of instances of a locally defined class whose addition returns the sum of their wrapped integer values: adding the Series to itself equals adding the Series to the raw object array, and adding it to a single instance equals the same operation computed again.
    class Dummy:
        # [CLASS DOC]
        # HELPER: local class wrapping a single value attribute, whose addition returns the plain sum of the two wrapped values (a number, not an instance).
        def __init__(self, value) -> None:
            # [FUNCTION DOC]
            # HELPER: stores the wrapped value.
            self.value = value

        def __add__(self, other):
            # [FUNCTION DOC]
            # HELPER: returns the sum of this object's value and the other object's value.
            return self.value + other.value

    arr = np.array([Dummy(0), Dummy(1)])
    ser = pd.Series(arr)
    tm.assert_series_equal(np.add(ser, ser), pd.Series(np.add(ser, arr)))
    tm.assert_series_equal(np.add(ser, Dummy(1)), pd.Series(np.add(ser, Dummy(1))))


@pytest.fixture(
    params=[
        pd.array([1, 3, 2], dtype=np.int64),
        pd.array([1, 3, 2], dtype="Int64"),
        pd.array([1, 3, 2], dtype="Float32"),
        pd.array([1, 10, 2], dtype="Sparse[int]"),
        pd.to_datetime(["2000", "2010", "2001"]),
        pd.to_datetime(["2000", "2010", "2001"]).tz_localize("CET"),
        pd.to_datetime(["2000", "2010", "2001"]).to_period(freq="D"),
        pd.to_timedelta(["1 Day", "3 Days", "2 Days"]),
        pd.IntervalIndex([pd.Interval(0, 1), pd.Interval(2, 3), pd.Interval(1, 2)]),
    ],
    ids=lambda x: str(x.dtype),
)
def values_for_np_reduce(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over nine array-likes (int64 NumPy, nullable Int64, nullable Float32, sparse int, naive datetimes, tz-aware datetimes in CET, daily periods, timedeltas, intervals; ids are dtype names) returning one. Used by the reduction tests, which assume the values are monotonic increasing.
    # min/max tests assume that these are monotonic increasing
    return request.param


class TestNumpyReductions:
    # [CLASS DOC]
    # Groups tests of NumPy reduce methods for the add, multiply, maximum and minimum ufuncs applied to Series, Index and DataFrame containers. Each test is run for every container from the array-box fixture and every dtype from the reduction values fixture. A TODO notes missing cases with missing values and the axis argument for frames.
    # TODO: cases with NAs, axis kwarg for DataFrame

    def test_multiply(self, values_for_np_reduce, box_with_array, request):
        # [FUNCTION DOC]
        # Verifies the multiply-reduce of each container. Constructing the container must raise no warnings; sparse arrays are marked expected-to-fail since they have no product. For integer, unsigned and float values the result equals the container's own product (frame: per-column Series without numeric-only restriction; Index: product of its underlying values). For other dtypes the call raises a type error (does not support operation, unsupported operand, or ufunc cannot use operands).
        box = box_with_array
        values = values_for_np_reduce

        with tm.assert_produces_warning(None):
            obj = box(values)

        if isinstance(values, pd.core.arrays.SparseArray):
            mark = pytest.mark.xfail(reason="SparseArray has no 'prod'")
            request.applymarker(mark)

        if values.dtype.kind in "iuf":
            result = np.multiply.reduce(obj)
            if box is pd.DataFrame:
                expected = obj.prod(numeric_only=False)
                tm.assert_series_equal(result, expected)
            elif box is pd.Index:
                # Index has no 'prod'
                expected = obj._values.prod()
                assert result == expected
            else:
                expected = obj.prod()
                assert result == expected
        else:
            msg = "|".join(
                [
                    "does not support operation",
                    "unsupported operand type",
                    "ufunc 'multiply' cannot use operands",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                np.multiply.reduce(obj)

    def test_add(self, values_for_np_reduce, box_with_array):
        # [FUNCTION DOC]
        # Verifies the add-reduce of each container: constructing it raises no warnings; for timedelta, integer, unsigned and float values the result equals the container's own sum (frame without numeric-only restriction, Index via its underlying values); other dtypes raise a type error with one of the standard unsupported-operation messages.
        box = box_with_array
        values = values_for_np_reduce

        with tm.assert_produces_warning(None):
            obj = box(values)

        if values.dtype.kind in "miuf":
            result = np.add.reduce(obj)
            if box is pd.DataFrame:
                expected = obj.sum(numeric_only=False)
                tm.assert_series_equal(result, expected)
            elif box is pd.Index:
                # Index has no 'sum'
                expected = obj._values.sum()
                assert result == expected
            else:
                expected = obj.sum()
                assert result == expected
        else:
            msg = "|".join(
                [
                    "does not support operation",
                    "unsupported operand type",
                    "ufunc 'add' cannot use operands",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                np.add.reduce(obj)

    def test_max(self, values_for_np_reduce, box_with_array, using_python_scalars):
        # [FUNCTION DOC]
        # Verifies the maximum-reduce for each container: for a frame, equals the frame maximum without numeric-only restriction; otherwise equals the second element of the values (the maximum, since they are increasing), converted to a Python scalar when the Python-scalar option fixture is on and values are integer or float, and, when the container is not an Index of integers or floats (which become Python numbers), has exactly the same type as that expected value (for example Timestamp rather than a datetime64).
        box = box_with_array
        values = values_for_np_reduce

        same_type = True
        if box is pd.Index and values.dtype.kind in "if":
            # ATM Index casts to object, so we get python ints/floats
            same_type = False

        with tm.assert_produces_warning(None):
            obj = box(values)

        result = np.maximum.reduce(obj)
        if box is pd.DataFrame:
            # TODO: cases with axis kwarg
            expected = obj.max(numeric_only=False)
            tm.assert_series_equal(result, expected)
        else:
            expected = values[1]
            if using_python_scalars and values.dtype.kind in "if":
                expected = expected.item()
            assert result == expected
            if same_type:
                # check we have e.g. Timestamp instead of dt64
                assert type(result) == type(expected)

    def test_min(self, values_for_np_reduce, box_with_array, using_python_scalars):
        # [FUNCTION DOC]
        # Verifies the minimum-reduce for each container, the mirror of the maximum test: expected is the first element of the values, with the same Python-scalar conversion and type-equality rules.
        box = box_with_array
        values = values_for_np_reduce

        same_type = True
        if box is pd.Index and values.dtype.kind in "if":
            # ATM Index casts to object, so we get python ints/floats
            same_type = False

        with tm.assert_produces_warning(None):
            obj = box(values)

        result = np.minimum.reduce(obj)
        if box is pd.DataFrame:
            expected = obj.min(numeric_only=False)
            tm.assert_series_equal(result, expected)
        else:
            expected = values[0]
            if using_python_scalars and values.dtype.kind in ["i", "f"]:
                expected = expected.item()
            assert result == expected
            if same_type:
                # check we have e.g. Timestamp instead of dt64
                assert type(result) == type(expected)


@pytest.mark.parametrize("type_", [list, deque, tuple])
def test_binary_ufunc_other_types(type_):
    # [FUNCTION DOC]
    # Verifies that a binary ufunc (addition) between a named Series and a list, deque or tuple of integers gives a Series with the same name equal to the sum with the Series' NumPy values. Parametrized over the sequence type.
    a = pd.Series([1, 2, 3], name="name")
    b = type_([3, 4, 5])

    result = np.add(a, b)
    expected = pd.Series(np.add(a.to_numpy(), b), name="name")
    tm.assert_series_equal(result, expected)


def test_object_dtype_ok():
    # [FUNCTION DOC]
    # Verifies that the add ufunc between an object-dtype Series of instances of a local class and one such instance gives a Series of new instances with each value incremented, compared using the class' equality and representation.
    class Thing:
        # [CLASS DOC]
        # HELPER: local value-wrapping class whose addition unwraps the other operand's value if it has one and returns a new instance; equality requires exactly this class and equal values; it has a readable representation. Used as object data.
        def __init__(self, value) -> None:
            # [FUNCTION DOC]
            # HELPER: stores the wrapped value.
            self.value = value

        def __add__(self, other):
            # [FUNCTION DOC]
            # HELPER: returns a new instance of the same class holding the sum of this value and the other operand (using its value attribute if it has one, otherwise the operand itself).
            other = getattr(other, "value", other)
            return type(self)(self.value + other)

        def __eq__(self, other) -> bool:
            # [FUNCTION DOC]
            # HELPER: returns true only when the other object is exactly of this class and holds an equal value.
            return type(other) is Thing and self.value == other.value

        def __repr__(self) -> str:
            # [FUNCTION DOC]
            # HELPER: returns a text representation naming the class and its value.
            return f"Thing({self.value})"

    s = pd.Series([Thing(1), Thing(2)])
    result = np.add(s, Thing(1))
    expected = pd.Series([Thing(2), Thing(3)])
    tm.assert_series_equal(result, expected)


def test_outer():
    # [FUNCTION DOC]
    # Verifies (issue 27186) that the outer-product method of a ufunc applied to a Series and an array raises a not-implemented error with an empty message.
    # https://github.com/pandas-dev/pandas/issues/27186
    ser = pd.Series([1, 2, 3])
    obj = np.array([1, 2, 3])

    with pytest.raises(NotImplementedError, match="^$"):
        np.subtract.outer(ser, obj)


def test_np_matmul():
    # [FUNCTION DOC]
    # Verifies (issue 26650) that NumPy matrix multiplication of a one-row frame by a one-column frame gives a one-by-one frame holding the dot product.
    # GH26650
    df1 = pd.DataFrame(data=[[-1, 1, 10]])
    df2 = pd.DataFrame(data=[-1, 1, 10])
    expected = pd.DataFrame(data=[102])

    result = np.matmul(df1, df2)
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("box", [pd.Index, pd.Series])
def test_np_matmul_1D(box, using_python_scalars):
    # [FUNCTION DOC]
    # Verifies matrix multiplication of two one-dimensional Index or Series objects gives eight; its type is Python int when the Python-scalar option fixture is on, and a NumPy 64-bit integer otherwise.
    result = np.matmul(box([1, 2]), box([2, 3]))
    assert result == 8
    if using_python_scalars:
        assert type(result) == int, type(result)
    else:
        assert type(result) == np.int64, type(result)


def test_array_ufuncs_for_many_arguments():
    # [FUNCTION DOC]
    # Verifies (issue 39853) a ufunc built from a three-argument Python function (via frompyfunc): called with two Series and a scalar it returns an object-dtype Series of elementwise sums; called with two Series and a frame it raises a not-implemented error about mixed frame and Series inputs.
    # GH39853
    def add3(x, y, z):
        # [FUNCTION DOC]
        # HELPER: nested function taking three values and returning their sum, wrapped into a ufunc by the test.
        return x + y + z

    ufunc = np.frompyfunc(add3, 3, 1)
    ser = pd.Series([1, 2])

    result = ufunc(ser, ser, 1)
    expected = pd.Series([3, 5], dtype=object)
    tm.assert_series_equal(result, expected)

    df = pd.DataFrame([[1, 2]])

    msg = (
        "Cannot apply ufunc <ufunc 'add3 (vectorized)'> "
        "to mixed DataFrame and Series inputs."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        ufunc(ser, ser, df)


def test_np_trunc():
    # [FUNCTION DOC]
    # Verifies that the NumPy truncate function (a true ufunc, replacing an older fix function that relied on out and where keywords) rounds floats toward zero for a float Series.
    # This used to test np.fix, which is not a ufunc but is composed of
    # several ufunc calls under the hood with `out` and `where` keywords. But numpy
    # is deprecating that (or at least discussing deprecating) in favor of np.trunc,
    # which _is_ a ufunc without the out keyword usage.
    ser = pd.Series([-1.5, -0.5, 0.5, 1.5])
    result = np.trunc(ser)
    expected = pd.Series([-1.0, -0.0, 0.0, 1.0])
    tm.assert_series_equal(result, expected)
