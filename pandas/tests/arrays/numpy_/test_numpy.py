"""
Additional tests for NumpyExtensionArray that aren't covered by
the interface tests.
"""

# [MODULE DOC]
# Additional tests for NumpyExtensionArray and NumpyEADtype that are not covered by the shared extension interface tests:
# dtype classification and repr, construction (no coercion, copy, dtype, Series), conversion to numpy including read-only handling, setitem and views,
# reductions, ufuncs, take, quantile, factorize and repr. Defines a fixture of numpy arrays of several dtypes. Uses fixtures any_numpy_dtype
# and any_numpy_array. No import-time side effects.

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes.dtypes import NumpyEADtype

import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray


@pytest.fixture(
    params=[
        np.array(["a", "b"], dtype=object),
        np.array([0, 1], dtype=float),
        np.array([0, 1], dtype=int),
        np.array([0, 1 + 2j], dtype=complex),
        np.array([True, False], dtype=bool),
        np.array([0, 1], dtype="datetime64[ns]"),
        np.array([0, 1], dtype="timedelta64[ns]"),
    ],
)
def any_numpy_array(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over small numpy arrays of object, float, int, complex, bool, datetime and timedelta dtypes (excluding string and bytes); returns a copy so tests may mutate it.
    """
    Parametrized fixture for NumPy arrays with different dtypes.

    This excludes string and bytes.
    """
    return request.param.copy()


# ----------------------------------------------------------------------------
# NumpyEADtype


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("bool", True),
        ("int", True),
        ("uint", True),
        ("float", True),
        ("complex", True),
        ("str", False),
        ("bytes", False),
        ("datetime64[ns]", False),
        ("object", False),
        ("void", False),
    ],
)
def test_is_numeric(dtype, expected):
    # [FUNCTION DOC]
    # Parametrized over dtype names. The dtype's numeric flag is true for bool, int, uint, float and complex and false for text, bytes, datetime, object and void.
    dtype = NumpyEADtype(dtype)
    assert dtype._is_numeric is expected


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("bool", True),
        ("int", False),
        ("uint", False),
        ("float", False),
        ("complex", False),
        ("str", False),
        ("bytes", False),
        ("datetime64[ns]", False),
        ("object", False),
        ("void", False),
    ],
)
def test_is_boolean(dtype, expected):
    # [FUNCTION DOC]
    # Parametrized over dtype names. The dtype's boolean flag is true only for bool.
    dtype = NumpyEADtype(dtype)
    assert dtype._is_boolean is expected


def test_repr():
    # [FUNCTION DOC]
    # Repr of the 64-bit integer wrapper dtype shows the class name with the numpy dtype name in quotes.
    dtype = NumpyEADtype(np.dtype("int64"))
    assert repr(dtype) == "NumpyEADtype('int64')"


def test_constructor_from_string():
    # [FUNCTION DOC]
    # Constructing the dtype from the string int64 equals constructing from the numpy dtype.
    result = NumpyEADtype.construct_from_string("int64")
    expected = NumpyEADtype(np.dtype("int64"))
    assert result == expected


def test_dtype_idempotent(any_numpy_dtype):
    # [FUNCTION DOC]
    # Constructing a wrapper dtype from another wrapper dtype returns an equal dtype for any numpy dtype.
    dtype = NumpyEADtype(any_numpy_dtype)

    result = NumpyEADtype(dtype)
    assert result == dtype


# ----------------------------------------------------------------------------
# Construction


def test_constructor_no_coercion():
    # [FUNCTION DOC]
    # Giving a list instead of a numpy array to the array constructor raises a value error mentioning NumPy array.
    with pytest.raises(ValueError, match="NumPy array"):
        NumpyExtensionArray([1, 2, 3])


def test_series_constructor_with_copy():
    # [FUNCTION DOC]
    # A Series built from the array with copy enabled does not hold the original numpy array.
    ndarray = np.array([1, 2, 3])
    ser = pd.Series(NumpyExtensionArray(ndarray), copy=True)

    assert ser.values is not ndarray


def test_series_constructor_with_astype():
    # [FUNCTION DOC]
    # A Series built from the array with a float dtype converts the values.
    ndarray = np.array([1, 2, 3])
    result = pd.Series(NumpyExtensionArray(ndarray), dtype="float64")
    expected = pd.Series([1.0, 2.0, 3.0], dtype="float64")
    tm.assert_series_equal(result, expected)


def test_from_sequence_dtype():
    # [FUNCTION DOC]
    # The private sequence constructor with an unsigned dtype converts a signed numpy array to that dtype.
    arr = np.array([1, 2, 3], dtype="int64")
    result = NumpyExtensionArray._from_sequence(arr, dtype="uint64")
    expected = NumpyExtensionArray(np.array([1, 2, 3], dtype="uint64"))
    tm.assert_extension_array_equal(result, expected)


def test_constructor_copy():
    # [FUNCTION DOC]
    # With copy enabled the array does not share memory with the input.
    arr = np.array([0, 1])
    result = NumpyExtensionArray(arr, copy=True)

    assert not tm.shares_memory(result, arr)


def test_constructor_with_data(any_numpy_array):
    # [FUNCTION DOC]
    # Using any_numpy_array. The array's numpy dtype equals the input's dtype.
    nparr = any_numpy_array
    arr = NumpyExtensionArray(nparr)
    assert arr.dtype.numpy_dtype == nparr.dtype


# ----------------------------------------------------------------------------
# Conversion


def test_to_numpy():
    # [FUNCTION DOC]
    # Default conversion returns the underlying array; copy enabled returns a different one; a dtype argument converts.
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    result = arr.to_numpy()
    assert result is arr._ndarray

    result = arr.to_numpy(copy=True)
    assert result is not arr._ndarray

    result = arr.to_numpy(dtype="f8")
    expected = np.array([1, 2, 3], dtype="f8")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_readonly():
    # [FUNCTION DOC]
    # For a read-only array, default conversion is read-only; copy or dtype conversion are writeable.
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    arr._readonly = True
    result = arr.to_numpy()
    assert not result.flags.writeable

    result = arr.to_numpy(copy=True)
    assert result.flags.writeable

    result = arr.to_numpy(dtype="f8")
    assert result.flags.writeable


@pytest.mark.skipif(not np_version_gt2, reason="copy keyword introduced in np 2.0")
@pytest.mark.parametrize("dtype", [None, "int64"])
def test_asarray_readonly(dtype):
    # [FUNCTION DOC]
    # Parametrized over dtype none and int64; skipped on old numpy. numpy array protocol on a read-only array is read-only by default and with copy disabled, writeable with copy enabled.
    arr = NumpyExtensionArray(np.array([1, 2, 3], dtype="int64"))
    arr._readonly = True
    result = np.asarray(arr, dtype=dtype)
    assert not result.flags.writeable

    result = np.asarray(arr, dtype=dtype, copy=True)
    assert result.flags.writeable

    result = np.asarray(arr, dtype=dtype, copy=False)
    assert not result.flags.writeable


# ----------------------------------------------------------------------------
# Setitem


def test_setitem(any_numpy_array):
    # [FUNCTION DOC]
    # Using any_numpy_array. Assigning one element to another position gives the same result as doing so on the plain numpy array.
    nparr = any_numpy_array
    arr = NumpyExtensionArray(nparr, copy=True)

    arr[0] = arr[1]
    nparr[0] = nparr[1]

    tm.assert_numpy_array_equal(arr.to_numpy(), nparr)


# ----------------------------------------------------------------------------
# Reductions


def test_bad_reduce_raises():
    # [FUNCTION DOC]
    # An unknown reduction name raises a type error saying it cannot perform that operation with type int.
    # KNOWN ISSUE: the expected message string is itself passed as the reduction name, so the name is the sentence and the match works only because the message embeds the name; the test name and intent suggest a short method name was meant.
    arr = np.array([1, 2, 3], dtype="int64")
    arr = NumpyExtensionArray(arr)
    msg = "cannot perform not_a_method with type int"
    with pytest.raises(TypeError, match=msg):
        arr._reduce(msg)


def test_validate_reduction_keyword_args():
    # [FUNCTION DOC]
    # Passing the keepdims keyword to a reduction raises a value error saying that parameter is not supported.
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    msg = "the 'keepdims' parameter is not supported .*all"
    with pytest.raises(ValueError, match=msg):
        arr.all(keepdims=True)


def test_np_max_nested_tuples():
    # [FUNCTION DOC]
    # For an array of nested tuples, the maximum method on the array and Series and the numpy maximum reduce (array and Series) all return the third element (which is the same object for the methods).
    # case where checking in ufunc.nout works while checking for tuples
    #  does not
    vals = [
        (("j", "k"), ("l", "m")),
        (("l", "m"), ("o", "p")),
        (("o", "p"), ("j", "k")),
    ]
    ser = pd.Series(vals)
    arr = ser.array

    assert arr.max() is arr[2]
    assert ser.max() is arr[2]

    result = np.maximum.reduce(arr)
    assert result == arr[2]

    result = np.maximum.reduce(ser)
    assert result == arr[2]


def test_np_reduce_2d():
    # [FUNCTION DOC]
    # For a two-dimensional array, numpy maximum reduce and the maximum method along axis zero both return the last row as an extension array.
    raw = np.arange(12).reshape(4, 3)
    arr = NumpyExtensionArray(raw)

    res = np.maximum.reduce(arr, axis=0)
    tm.assert_extension_array_equal(res, arr[-1])

    alt = arr.max(axis=0)
    tm.assert_extension_array_equal(alt, arr[-1])


# ----------------------------------------------------------------------------
# Ops


@pytest.mark.parametrize("ufunc", [np.abs, np.negative, np.positive])
def test_ufunc_unary(ufunc):
    # [FUNCTION DOC]
    # Parametrized over absolute, negative and positive. Applying the ufunc gives a wrapped array of the numpy result; with an out argument the output array is filled in place.
    arr = NumpyExtensionArray(np.array([-1.0, 0.0, 1.0]))
    result = ufunc(arr)
    expected = NumpyExtensionArray(ufunc(arr._ndarray))
    tm.assert_extension_array_equal(result, expected)

    # same thing but with the 'out' keyword
    out = NumpyExtensionArray(np.array([-9.0, -9.0, -9.0]))
    ufunc(arr, out=out)
    tm.assert_extension_array_equal(out, expected)


def test_ufunc():
    # [FUNCTION DOC]
    # A two-output ufunc (divmod) applied to the array and an array plus two returns two wrapped arrays equal to the numpy results.
    arr = NumpyExtensionArray(np.array([-1.0, 0.0, 1.0]))

    r1, r2 = np.divmod(arr, np.add(arr, 2))
    e1, e2 = np.divmod(arr._ndarray, np.add(arr._ndarray, 2))
    e1 = NumpyExtensionArray(e1)
    e2 = NumpyExtensionArray(e2)
    tm.assert_extension_array_equal(r1, e1)
    tm.assert_extension_array_equal(r2, e2)


def test_basic_binop():
    # [FUNCTION DOC]
    # Smoke test: adding the array to itself doubles the values.
    # Just a basic smoke test. The EA interface tests exercise this
    # more thoroughly.
    x = NumpyExtensionArray(np.array([1, 2, 3]))
    result = x + x
    expected = NumpyExtensionArray(np.array([2, 4, 6]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, object])
def test_setitem_object_typecode(dtype):
    # [FUNCTION DOC]
    # Parametrized over default (string) and object numpy dtype. Setting a string element updates the array and preserves the dtype.
    arr = NumpyExtensionArray(np.array(["a", "b", "c"], dtype=dtype))
    arr[0] = "t"
    expected = NumpyExtensionArray(np.array(["t", "b", "c"], dtype=dtype))
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_no_coercion():
    # [FUNCTION DOC]
    # Assigning a string into an integer array raises a value error mentioning int; assigning a float coerces the value, not the array, so the element stays an integer type.
    # https://github.com/pandas-dev/pandas/issues/28150
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="int"):
        arr[0] = "a"

    # With a value that we do coerce, check that we coerce the value
    #  and not the underlying array.
    arr[0] = 2.5
    assert isinstance(arr[0], (int, np.integer)), type(arr[0])


def test_setitem_preserves_views():
    # [FUNCTION DOC]
    # A view, a slice and a numpy conversion all see assignments to the original array, and assigning through the view changes the original.
    # GH#28150, see also extension test of the same name
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    view1 = arr.view()
    view2 = arr[:]
    view3 = np.asarray(arr)

    arr[0] = 9
    assert view1[0] == 9
    assert view2[0] == 9
    assert view3[0] == 9

    arr[-1] = 2.5
    view1[-1] = 5
    assert arr[-1] == 5


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_quantile_empty(dtype):
    # [FUNCTION DOC]
    # Parametrized over signed and unsigned 64-bit integers. Quantiles of an empty array give NaN values (floating), not minus ones.
    # we should get back np.nans, not -1s
    arr = NumpyExtensionArray(np.array([], dtype=dtype))
    idx = pd.Index([0.0, 0.5])

    result = arr._quantile(idx, interpolation="linear")
    expected = NumpyExtensionArray(np.array([np.nan, np.nan]))
    tm.assert_extension_array_equal(result, expected)


def test_factorize_unsigned():
    # [FUNCTION DOC]
    # Factorizing an unsigned integer array does not raise and gives the same codes and uniques (wrapped) as the top-level factorize.
    # don't raise when calling factorize on unsigned int NumpyExtensionArray
    arr = np.array([1, 2, 3], dtype=np.uint64)
    obj = NumpyExtensionArray(arr)

    res_codes, res_unique = obj.factorize()
    exp_codes, exp_unique = pd.factorize(arr)

    tm.assert_numpy_array_equal(res_codes, exp_codes)

    tm.assert_extension_array_equal(res_unique, NumpyExtensionArray(exp_unique))


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ],
)
def test_take_assigns_floating_point_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over bool and integer numpy dtypes. Taking with fill allowed and a negative index promotes integers to 64-bit float and booleans to object, with default fill and with a float fill value.
    # GH#62448.
    if dtype == np.bool_:
        array = NumpyExtensionArray(np.array([False, True, False], dtype=dtype))
        expected = np.dtype(object)
    else:
        array = NumpyExtensionArray(np.array([1, 2, 3], dtype=dtype))
        expected = np.float64

    result = array.take([-1], allow_fill=True)
    assert result.dtype.numpy_dtype == expected

    result = array.take([-1], allow_fill=True, fill_value=5.0)
    assert result.dtype.numpy_dtype == expected


def test_take_preserves_boolean_arrays():
    # [FUNCTION DOC]
    # Taking without allowing fill keeps the boolean numpy dtype.
    array = NumpyExtensionArray(np.array([False, True, False], dtype=np.bool_))
    result = array.take([-1], allow_fill=False)
    assert result.dtype.numpy_dtype == np.bool_


# ----------------------------------------------------------------------------
# Output formatting


def test_array_repr(any_numpy_array):
    # [FUNCTION DOC]
    # Using any_numpy_array. Representation shows the class header, values formatted per dtype (strings quoted, complex with imaginary part, datetimes at nanosecond precision, timedeltas in nanoseconds), and a length and dtype line.
    # The assertion message shows both strings.
    # GH#61085
    nparray = any_numpy_array
    arr = NumpyExtensionArray(nparray)
    if nparray.dtype == "object":
        values = "['a', 'b']"
    elif nparray.dtype == "float64":
        values = "[0.0, 1.0]"
    elif str(nparray.dtype).startswith("int"):
        values = "[0, 1]"
    elif nparray.dtype == "complex128":
        values = "[0j, (1+2j)]"
    elif nparray.dtype == "bool":
        values = "[True, False]"
    elif nparray.dtype == "datetime64[ns]":
        values = "[1970-01-01T00:00:00.000000000, 1970-01-01T00:00:00.000000001]"
    elif nparray.dtype == "timedelta64[ns]":
        values = "[0 nanoseconds, 1 nanoseconds]"
    expected = f"<NumpyExtensionArray>\n{values}\nLength: 2, dtype: {nparray.dtype}"
    result = repr(arr)
    assert result == expected, f"{result} vs {expected}"
