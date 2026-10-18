"""
This module tests the functionality of StringArray and ArrowStringArray.
Tests for the str accessors are in pandas/tests/strings/test_string_array.py
"""
# [MODULE DOC]
# Tests for the pandas string extension arrays (StringArray with python storage
# and ArrowStringArray with pyarrow storage), parametrized over storage and
# missing-value marker (NA or NaN) through the string_dtype_arguments fixture of
# the suite conftest. Covers dtype equality and repr, construction, setitem
# validation, casting, reductions, fillna, arrow interop, value_counts, memory
# usage, numpy conversion, isin, ufuncs and views. No import-time side effects.

import numpy as np
import pytest

from pandas.compat.pyarrow import pa_version_under19p0

from pandas.core.dtypes.common import is_dtype_equal

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
    ArrowStringArray,
)


@pytest.fixture
def dtype(string_dtype_arguments):
    # [FUNCTION DOC]
    # HELPER: fixture. Builds a StringDtype from the storage and NA value pair given
    # by the string_dtype_arguments fixture and returns it.
    """Fixture giving StringDtype from parametrized storage and na_value arguments"""
    storage, na_value = string_dtype_arguments
    return pd.StringDtype(storage=storage, na_value=na_value)


@pytest.fixture
def dtype2(string_dtype_arguments2):
    # [FUNCTION DOC]
    # HELPER: fixture. Like the dtype fixture but built from the second independent
    # parametrization (string_dtype_arguments2); used by tests mixing two string
    # dtypes.
    storage, na_value = string_dtype_arguments2
    return pd.StringDtype(storage=storage, na_value=na_value)


@pytest.fixture
def cls(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns the extension array class associated with the dtype
    # fixture (python-backed or arrow-backed string array class).
    """Fixture giving array type from parametrized 'dtype'"""
    return dtype.construct_array_type()


def test_dtype_equality():
    # [FUNCTION DOC]
    # Skips without pyarrow. Verifies StringDtype equality depends on both storage
    # and NA marker: the default NA variants with the same storage are equal;
    # different storages or different NA markers are unequal; NaN markers compare
    # equal whichever NaN object is used.
    pytest.importorskip("pyarrow")

    dtype1 = pd.StringDtype("python")
    dtype2 = pd.StringDtype("pyarrow")
    dtype3 = pd.StringDtype("pyarrow", na_value=np.nan)

    assert dtype1 == pd.StringDtype("python", na_value=pd.NA)
    assert dtype1 != dtype2
    assert dtype1 != dtype3

    assert dtype2 == pd.StringDtype("pyarrow", na_value=pd.NA)
    assert dtype2 != dtype1
    assert dtype2 != dtype3

    assert dtype3 == pd.StringDtype("pyarrow", na_value=np.nan)
    assert dtype3 == pd.StringDtype("pyarrow", na_value=float("nan"))
    assert dtype3 != dtype1
    assert dtype3 != dtype2


def test_repr(dtype):
    # [FUNCTION DOC]
    # Verifies the text representation of a one-column DataFrame, of its Series and
    # of the underlying array for the dtype: NaN-variants print NaN and dtype name
    # str, NA-variants print the NA marker and dtype name string, and the array repr
    # names the arrow or python array class.
    df = pd.DataFrame({"A": pd.array(["a", pd.NA, "b"], dtype=dtype)})
    if dtype.na_value is np.nan:
        expected = "     A\n0    a\n1  NaN\n2    b"
    else:
        expected = "      A\n0     a\n1  <NA>\n2     b"
    assert repr(df) == expected

    if dtype.na_value is np.nan:
        expected = "0      a\n1    NaN\n2      b\nName: A, dtype: str"
    else:
        expected = "0       a\n1    <NA>\n2       b\nName: A, dtype: string"
    assert repr(df.A) == expected

    if dtype.storage == "pyarrow" and dtype.na_value is pd.NA:
        arr_name = "ArrowStringArray"
        expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    elif dtype.storage == "pyarrow" and dtype.na_value is np.nan:
        arr_name = "ArrowStringArray"
        expected = f"<{arr_name}>\n['a', nan, 'b']\nLength: 3, dtype: str"
    elif dtype.storage == "python" and dtype.na_value is np.nan:
        arr_name = "StringArray"
        expected = f"<{arr_name}>\n['a', nan, 'b']\nLength: 3, dtype: str"
    else:
        arr_name = "StringArray"
        expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    assert repr(df.A.array) == expected


def test_dtype_repr(dtype):
    # [FUNCTION DOC]
    # Verifies the dtype repr: pyarrow storage omits the storage in the text and
    # shows only the NA marker; python storage shows both storage and NA marker.
    if dtype.storage == "pyarrow":
        if dtype.na_value is pd.NA:
            assert repr(dtype) == "<StringDtype(na_value=<NA>)>"
        else:
            assert repr(dtype) == "<StringDtype(na_value=nan)>"
    elif dtype.na_value is pd.NA:
        assert repr(dtype) == "<StringDtype(storage='python', na_value=<NA>)>"
    else:
        assert repr(dtype) == "<StringDtype(storage='python', na_value=nan)>"


def test_none_to_nan(cls, dtype):
    # [FUNCTION DOC]
    # Verifies that a None in the input sequence is converted into the dtype's own
    # missing-value marker (and not left as None) when building the array.
    a = cls._from_sequence(["a", None, "b"], dtype=dtype)
    assert a[1] is not None
    assert a[1] is a.dtype.na_value


def test_setitem_validates(cls, dtype):
    # [FUNCTION DOC]
    # Verifies setting a non-string scalar (an integer) or an integer array into a
    # string array raises a type error whose message names the invalid value and the
    # dtype.
    arr = cls._from_sequence(["a", "b"], dtype=dtype)

    msg = "Invalid value '10' for dtype 'str"
    with pytest.raises(TypeError, match=msg):
        arr[0] = 10

    msg = "Invalid value for dtype 'str"
    with pytest.raises(TypeError, match=msg):
        arr[:] = np.array([1, 2])


def test_setitem_with_scalar_string(dtype):
    # [FUNCTION DOC]
    # Verifies assigning a one-character string such as the letter d, which some
    # float-dtype checks would misclassify, works as a normal string assignment.
    # is_float_dtype considers some strings, like 'd', to be floats
    # which can cause issues.
    arr = pd.array(["a", "c"], dtype=dtype)
    arr[0] = "d"
    expected = pd.array(["d", "c"], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_with_array_with_missing(dtype):
    # [FUNCTION DOC]
    # Verifies assigning a numpy array containing None via a list key stores a
    # missing value in the array and does not mutate the assigned value array.
    # ensure that when setting with an array of values, we don't mutate the
    # array `value` in __setitem__(self, key, value)
    arr = pd.array(["a", "b", "c"], dtype=dtype)
    value = np.array(["A", None])
    value_orig = value.copy()
    arr[[0, 1]] = value

    expected = pd.array(["A", pd.NA, "c"], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)
    tm.assert_numpy_array_equal(value, value_orig)


def test_astype_roundtrip(dtype):
    # [FUNCTION DOC]
    # Verifies casting a datetime Series with a missing value to the string dtype and
    # back to datetime round trips, and the same for timedelta values (GH#38509).
    ser = pd.Series(pd.date_range("2000", periods=12, unit="ns"))
    ser[0] = None

    casted = ser.astype(dtype)
    assert is_dtype_equal(casted.dtype, dtype)

    result = casted.astype("datetime64[ns]")
    tm.assert_series_equal(result, ser)

    # GH#38509 same thing for timedelta64
    ser2 = ser - ser.iloc[-1]
    casted2 = ser2.astype(dtype)
    assert is_dtype_equal(casted2.dtype, dtype)

    result2 = casted2.astype(ser2.dtype)
    tm.assert_series_equal(result2, ser2)


def test_constructor_raises(cls):
    # [FUNCTION DOC]
    # Verifies the array constructor rejects invalid input with a value error:
    # bytes strings, empty float arrays, and arrays with datetime/timedelta/NaT
    # missing markers. For the python array class NaN and None inside an object array
    # are accepted as valid missing values (GH#45057); for the arrow class they are
    # rejected as unsupported input.
    if cls is pd.arrays.StringArray:
        msg = "StringArray requires a sequence of strings or pandas.NA"
        kwargs = {"dtype": pd.StringDtype()}
    else:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowExtensionArray"
        kwargs = {}

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", "b"], dtype="S1"), **kwargs)

    with pytest.raises(ValueError, match=msg):
        cls(np.array([]), **kwargs)

    if cls is pd.arrays.StringArray:
        # GH#45057 np.nan and None do NOT raise, as they are considered valid NAs
        #  for string dtype
        cls(np.array(["a", np.nan], dtype=object), **kwargs)
        cls(np.array(["a", None], dtype=object), **kwargs)
    else:
        with pytest.raises(ValueError, match=msg):
            cls(np.array(["a", np.nan], dtype=object), **kwargs)
        with pytest.raises(ValueError, match=msg):
            cls(np.array(["a", None], dtype=object), **kwargs)

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", pd.NaT], dtype=object), **kwargs)

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.datetime64("NaT", "ns")], dtype=object), **kwargs)

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.timedelta64("NaT", "ns")], dtype=object), **kwargs)


@pytest.mark.parametrize("na", [np.nan, np.float64("nan"), float("nan"), None, pd.NA])
def test_constructor_nan_like(na):
    # [FUNCTION DOC]
    # Parametrized over several NaN-like markers (NaN variants, None, NA). Verifies
    # the python StringArray constructor normalizes each to NA, giving an array equal
    # to one built with NA.
    expected = pd.arrays.StringArray(np.array(["a", pd.NA]), dtype=pd.StringDtype())
    result = pd.arrays.StringArray(
        np.array(["a", na], dtype="object"), dtype=pd.StringDtype()
    )
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("copy", [True, False])
def test_from_sequence_no_mutate(copy, cls, dtype):
    # [FUNCTION DOC]
    # Parametrized over copy. Verifies the private sequence constructor builds the
    # array from an object array containing NaN with the expected missing marker and
    # does not mutate the input array.
    nan_arr = np.array(["a", np.nan], dtype=object)
    expected_input = nan_arr.copy()
    na_arr = np.array(["a", pd.NA], dtype=object)

    result = cls._from_sequence(nan_arr, dtype=dtype, copy=copy)

    if cls is ArrowStringArray:
        import pyarrow as pa

        expected = cls(
            pa.array(na_arr, type=pa.string(), from_pandas=True), dtype=dtype
        )
    elif dtype.na_value is np.nan:
        expected = cls(nan_arr, dtype=dtype)
    else:
        expected = cls(na_arr, dtype=dtype)

    tm.assert_extension_array_equal(result, expected)
    tm.assert_numpy_array_equal(nan_arr, expected_input)


def test_astype_int(dtype):
    # [FUNCTION DOC]
    # Verifies casting to 64-bit integer works for numeric strings and, with a
    # missing value present, raises: a value error about converting NaN for NaN
    # variants, a type error about the int argument type for NA variants.
    arr = pd.array(["1", "2", "3"], dtype=dtype)
    result = arr.astype("int64")
    expected = np.array([1, 2, 3], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)
    if dtype.na_value is np.nan:
        err = ValueError
        msg = "cannot convert float NaN to integer"
    else:
        err = TypeError
        msg = (
            r"int\(\) argument must be a string, a bytes-like "
            r"object or a( real)? number"
        )
    with pytest.raises(err, match=msg):
        arr.astype("int64")


def test_astype_nullable_int(dtype):
    # [FUNCTION DOC]
    # Verifies casting a string array containing a missing value to the nullable
    # Int64 dtype keeps the missing value as NA.
    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)

    result = arr.astype("Int64")
    expected = pd.array([1, pd.NA, 3], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_astype_float(dtype, any_float_dtype):
    # [FUNCTION DOC]
    # Uses the any_float_dtype fixture. Verifies casting a Series of numeric strings
    # with a missing value to each float dtype parses the values; the missing value
    # becomes NaN for numpy dtypes and NA for extension dtypes. Compares Series, not arrays.
    # Don't compare arrays (37974)
    ser = pd.Series(["1.1", pd.NA, "3.3"], dtype=dtype)
    result = ser.astype(any_float_dtype)
    item = np.nan if isinstance(result.dtype, np.dtype) else pd.NA
    expected = pd.Series([1.1, item, 3.3], dtype=any_float_dtype)
    tm.assert_series_equal(result, expected)


def test_reduce(skipna, dtype):
    # [FUNCTION DOC]
    # Uses the skipna fixture. Verifies the sum of a Series of three strings is their
    # concatenation.
    arr = pd.Series(["a", "b", "c"], dtype=dtype)
    result = arr.sum(skipna=skipna)
    assert result == "abc"


def test_reduce_missing(skipna, dtype):
    # [FUNCTION DOC]
    # Uses the skipna fixture. Verifies the sum with missing values is the
    # concatenation of present strings when skipping, otherwise a missing value.
    arr = pd.Series([None, "a", None, "b", "c", None], dtype=dtype)
    result = arr.sum(skipna=skipna)
    if skipna:
        assert result == "abc"
    else:
        assert pd.isna(result)


@pytest.mark.parametrize("min_count", [0, 1])
def test_reduce_empty(skipna, dtype, min_count):
    # [FUNCTION DOC]
    # Uses the skipna fixture, parametrized over min_count (0, 1). Verifies the sum of
    # an empty Series is the empty string for min_count zero and missing otherwise;
    # for an all-missing Series the sum is the empty string only when skipping NAs with
    # min_count zero, otherwise missing.
    arr = pd.Series([], dtype=dtype)
    result = arr.sum(skipna=skipna, min_count=min_count)
    if min_count == 0:
        assert result == ""
    else:
        assert pd.isna(result)

    # all-missing
    arr = pd.Series([None, None], dtype=dtype)
    result = arr.sum(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == ""
    else:
        assert pd.isna(result)


@pytest.mark.parametrize("method", ["min", "max"])
def test_min_max(method, skipna, dtype):
    # [FUNCTION DOC]
    # Parametrized over min and max, uses skipna fixture. Verifies the reduction
    # gives the smallest or largest string when skipping missing values and
    # otherwise returns the dtype's missing-value marker (the identical object).
    arr = pd.Series(["a", "b", "c", None], dtype=dtype)
    result = getattr(arr, method)(skipna=skipna)
    if skipna:
        expected = "a" if method == "min" else "c"
        assert result == expected
    else:
        assert result is arr.dtype.na_value


@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("box", [pd.Series, pd.array])
def test_min_max_numpy(method, box, dtype, request):
    # [FUNCTION DOC]
    # Parametrized over min/max and container (Series, array). Verifies the numpy
    # minimum or maximum functions on a string container with a missing value give
    # the smallest or largest string. Marked expected-failure (type error) for
    # pyarrow storage with a bare array container.
    # KNOWN ISSUE: the inner conditional repeats the outer test of the container
    # being an array, so the alternative branch giving the reason about a missing
    # max attribute is unreachable.
    if dtype.storage == "pyarrow" and box is pd.array:
        if box is pd.array:
            reason = "'<=' not supported between instances of 'str' and 'NoneType'"
        else:
            reason = "'ArrowStringArray' object has no attribute 'max'"
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.applymarker(mark)

    arr = box(["a", "b", "c", None], dtype=dtype)
    result = getattr(np, method)(arr)
    expected = "a" if method == "min" else "c"
    assert result == expected


def test_fillna_args(dtype):
    # [FUNCTION DOC]
    # Regression test (GH 37987). Verifies fillna with a string and with a numpy
    # string scalar fills missing values, and that filling with an integer raises a
    # type error naming the invalid value and dtype.
    # GH 37987

    arr = pd.array(["a", pd.NA], dtype=dtype)

    res = arr.fillna(value="b")
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    res = arr.fillna(value=np.str_("b"))
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    msg = "Invalid value '1' for dtype 'str"
    with pytest.raises(TypeError, match=msg):
        arr.fillna(value=1)


def test_arrow_array(dtype):
    # [FUNCTION DOC]
    # Skips without pyarrow. Verifies converting the array to a pyarrow array yields
    # a large_string array, cast to string for python storage, with the same values.
    # protocol added in 0.15.0
    pa = pytest.importorskip("pyarrow")
    import pyarrow.compute as pc

    data = pd.array(["a", "b", "c"], dtype=dtype)
    arr = pa.array(data)
    expected = pa.array(list(data), type=pa.large_string(), from_pandas=True)
    if dtype.storage == "python":
        expected = pc.cast(expected, pa.string())
    assert arr.equals(expected)


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_arrow_roundtrip(dtype, string_storage, using_infer_string):
    # [FUNCTION DOC]
    # Uses string_storage and using_infer_string fixtures. Verifies converting a
    # DataFrame with a string column to a pyarrow table gives a string type for python
    # storage and large_string for pyarrow storage; converting the table back to
    # pandas under the chosen string_storage option gives string dtype with that
    # storage and the NA marker of the original dtype, equal to the expected frame
    # (with columns also string-typed when infer-string mode is on), with the missing
    # value being the dtype's marker object. Ignores the block manager deprecation warning.
    # roundtrip possible from arrow 1.0.0
    pa = pytest.importorskip("pyarrow")

    data = pd.array(["a", "b", None], dtype=dtype)
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    if dtype.storage == "python":
        assert table.field("a").type == "string"
    else:
        assert table.field("a").type == "large_string"
    with pd.option_context("string_storage", string_storage):
        result = table.to_pandas()

    assert isinstance(result["a"].dtype, pd.StringDtype)
    expected = df.astype(pd.StringDtype(string_storage, na_value=dtype.na_value))
    if using_infer_string:
        expected.columns = expected.columns.astype(
            pd.StringDtype(string_storage, na_value=np.nan)
        )
    tm.assert_frame_equal(result, expected)
    # ensure the missing value is represented by NA and not np.nan or None
    assert result.loc[2, "a"] is result["a"].dtype.na_value


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_arrow_from_string(using_infer_string):
    # [FUNCTION DOC]
    # Starts with a plain pyarrow table of string type (no pandas metadata) and
    # verifies the dtype after conversion to pandas: object for pyarrow older than 19
    # (in either mode), otherwise a NaN-variant string dtype without infer-string mode
    # or the default str dtype with it.
    # not roundtrip,  but starting with pyarrow table without pandas metadata
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": pa.array(["a", "b", None], type=pa.string())})

    result = table.to_pandas()

    if not using_infer_string:
        if pa_version_under19p0:
            expected = pd.DataFrame({"a": ["a", "b", None]}, dtype="object")
        else:
            expected = pd.DataFrame(
                {"a": ["a", "b", None]}, dtype=pd.StringDtype(na_value=np.nan)
            )
    elif pa_version_under19p0:
        expected = pd.DataFrame({"a": ["a", "b", None]}, dtype="object")
    else:
        expected = pd.DataFrame({"a": ["a", "b", None]}, dtype="str")
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_arrow_load_from_zero_chunks(dtype, string_storage, using_infer_string):
    # [FUNCTION DOC]
    # Regression test (GH-41040). Verifies an empty-array frame converts to a
    # pyarrow table with the right string type, and that a table whose column has
    # zero chunks converts back to pandas as an empty string-dtype column under the
    # selected string_storage option, equal to the expected frame.
    # GH-41040
    pa = pytest.importorskip("pyarrow")

    data = pd.array([], dtype=dtype)
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    if dtype.storage == "python":
        assert table.field("a").type == "string"
    else:
        assert table.field("a").type == "large_string"
    # Instantiate the same table with no chunks at all
    table = pa.table([pa.chunked_array([], type=pa.string())], schema=table.schema)
    with pd.option_context("string_storage", string_storage):
        result = table.to_pandas()

    assert isinstance(result["a"].dtype, pd.StringDtype)
    expected = df.astype(pd.StringDtype(string_storage, na_value=dtype.na_value))
    if using_infer_string:
        expected.columns = expected.columns.astype(
            pd.StringDtype(string_storage, na_value=np.nan)
        )
    tm.assert_frame_equal(result, expected)


def test_value_counts_na(dtype):
    # [FUNCTION DOC]
    # Verifies value_counts with and without dropping missing values: the result
    # counts are of integer dtype (numpy for NaN variants, arrow integer for pyarrow
    # NA variants, nullable Int64 otherwise), indexed by the distinct strings, named
    # count.
    if dtype.na_value is np.nan:
        exp_dtype = "int64"
    elif dtype.storage == "pyarrow":
        exp_dtype = "int64[pyarrow]"
    else:
        exp_dtype = "Int64"
    arr = pd.array(["a", "b", "a", pd.NA], dtype=dtype)
    result = arr.value_counts(dropna=False)
    expected = pd.Series([2, 1, 1], index=arr[[0, 1, 3]], dtype=exp_dtype, name="count")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=arr[:2], dtype=exp_dtype, name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize(dtype):
    # [FUNCTION DOC]
    # Verifies normalized value_counts excludes missing values from the total
    # and returns proportions (floating point dtype matching the storage/NA
    # variant), named proportion.
    if dtype.na_value is np.nan:
        exp_dtype = np.float64
    elif dtype.storage == "pyarrow":
        exp_dtype = "double[pyarrow]"
    else:
        exp_dtype = "Float64"
    ser = pd.Series(["a", "b", "a", pd.NA], dtype=dtype)
    result = ser.value_counts(normalize=True)
    expected = pd.Series([2, 1], index=ser[:2], dtype=exp_dtype, name="proportion") / 3
    tm.assert_series_equal(result, expected)


def test_value_counts_sort_false(dtype):
    # [FUNCTION DOC]
    # Verifies value_counts with sorting disabled keeps first-appearance order and
    # yields counts of the dtype-appropriate integer type.
    if dtype.na_value is np.nan:
        exp_dtype = "int64"
    elif dtype.storage == "pyarrow":
        exp_dtype = "int64[pyarrow]"
    else:
        exp_dtype = "Int64"
    ser = pd.Series(["a", "b", "c", "b"], dtype=dtype)
    result = ser.value_counts(sort=False)
    expected = pd.Series([1, 2, 1], index=ser[:3], dtype=exp_dtype, name="count")
    tm.assert_series_equal(result, expected)


def test_memory_usage(dtype):
    # [FUNCTION DOC]
    # Regression test (GH 33963). Skipped for pyarrow storage. Verifies the byte
    # counts relation: nbytes is positive and not greater than the shallow memory usage,
    # which is smaller than the deep memory usage.
    # GH 33963

    if dtype.storage == "pyarrow":
        pytest.skip(f"not applicable for {dtype.storage}")

    series = pd.Series(["a", "b", "c"], dtype=dtype)

    assert 0 < series.nbytes <= series.memory_usage() < series.memory_usage(deep=True)


@pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
def test_astype_from_float_dtype(float_dtype, dtype):
    # [FUNCTION DOC]
    # Parametrized over float16/32/64. Regression test for GH 36451: casting a float
    # Series to a string dtype gives its short text representation.
    # https://github.com/pandas-dev/pandas/issues/36451
    ser = pd.Series([0.1], dtype=float_dtype)
    result = ser.astype(dtype)
    expected = pd.Series(["0.1"], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_astype_from_masked_float_with_nan(dtype, using_nan_is_na):
    # [FUNCTION DOC]
    # Regression test (GH#61617, GH#65227). Uses the using_nan_is_na fixture.
    # Verifies casting a nullable Float64 array holding an unmasked NaN, a masked NA
    # and a number to the string dtype: the unmasked NaN becomes NA when NaN counts as
    # missing, otherwise the text nan; the masked NA stays NA.
    # GH#61617, GH#65227 - FloatingArray.astype(str) with unmasked NaN
    arr = pd.array([np.nan, pd.NA, 3.0], dtype="Float64")
    result = arr.astype(dtype)
    if using_nan_is_na:
        expected = pd.array([pd.NA, pd.NA, "3.0"], dtype=dtype)
    else:
        expected = pd.array(["nan", pd.NA, "3.0"], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_to_numpy_returns_pdna_default(dtype):
    # [FUNCTION DOC]
    # Verifies that converting to a numpy array via the array protocol gives an object
    # array with the dtype's missing-value marker in place of missing values.
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    result = np.array(arr)
    expected = np.array(["a", dtype.na_value, "b"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_na_value(dtype, nulls_fixture):
    # [FUNCTION DOC]
    # Uses the nulls_fixture. Verifies to_numpy with the na_value argument replaces
    # missing entries by the given marker in an object array.
    na_value = nulls_fixture
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    result = arr.to_numpy(na_value=na_value)
    expected = np.array(["a", na_value, "b"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_readonly(dtype):
    # [FUNCTION DOC]
    # Sets the private read-only flag on the array and verifies the numpy result
    # of to_numpy is not writeable for python storage but is writeable for pyarrow
    # storage (a copy is produced).
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    arr._readonly = True
    result = arr.to_numpy()
    if dtype.storage == "python":
        assert not result.flags.writeable
    else:
        assert result.flags.writeable


def test_isin(dtype, fixed_now_ts):
    # [FUNCTION DOC]
    # Uses the fixed_now_ts fixture. Verifies isin on a string Series with missing
    # value: matches strings, matches missing values when the list contains NA, is all
    # false for an empty list, and ignores non-string elements such as timestamps.
    s = pd.Series(["a", "b", None], dtype=dtype)

    result = s.isin(["a", "c"])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", pd.NA])
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)

    result = s.isin([])
    expected = pd.Series([False, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", fixed_now_ts])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin([fixed_now_ts])
    expected = pd.Series([False, False, False])
    tm.assert_series_equal(result, expected)


def test_isin_string_array(dtype, dtype2):
    # [FUNCTION DOC]
    # Verifies isin accepts another string-dtype array (with a possibly different
    # storage/NA marker) and that a missing value in the list matches missing values.
    s = pd.Series(["a", "b", None], dtype=dtype)

    result = s.isin(pd.array(["a", "c"], dtype=dtype2))
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(pd.array(["a", None], dtype=dtype2))
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)


def test_isin_arrow_string_array(dtype):
    # [FUNCTION DOC]
    # Skips without pyarrow. Verifies isin accepts an arrow-string typed extension
    # array argument, including a null that matches missing values.
    pa = pytest.importorskip("pyarrow")
    s = pd.Series(["a", "b", None], dtype=dtype)

    result = s.isin(pd.array(["a", "c"], dtype=pd.ArrowDtype(pa.string())))
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(pd.array(["a", None], dtype=pd.ArrowDtype(pa.string())))
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)


def test_setitem_scalar_with_mask_validation(dtype):
    # [FUNCTION DOC]
    # Regression test (GitHub 47628). Verifies assigning None with a boolean mask
    # stores the dtype's missing marker, and assigning an integer with a mask raises a
    # type error naming the invalid value and dtype.
    # https://github.com/pandas-dev/pandas/issues/47628
    # setting None with a boolean mask (through _putmaks) should still result
    # in pd.NA values in the underlying array
    ser = pd.Series(["a", "b", "c"], dtype=dtype)
    mask = np.array([False, True, False])

    ser[mask] = None
    assert ser.array[1] is ser.dtype.na_value

    # for other non-string we should also raise an error
    ser = pd.Series(["a", "b", "c"], dtype=dtype)
    msg = "Invalid value '1' for dtype 'str"
    with pytest.raises(TypeError, match=msg):
        ser[mask] = 1


def test_from_numpy_str(dtype):
    # [FUNCTION DOC]
    # Verifies building a string array from a numpy fixed-width unicode array gives
    # the same result as building from the list of strings.
    vals = ["a", "b", "c"]
    arr = np.array(vals, dtype=np.str_)
    result = pd.array(arr, dtype=dtype)
    expected = pd.array(vals, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_tolist(dtype):
    # [FUNCTION DOC]
    # Verifies tolist returns the equivalent list of Python strings.
    vals = ["a", "b", "c"]
    arr = pd.array(vals, dtype=dtype)
    result = arr.tolist()
    expected = vals
    tm.assert_equal(result, expected)


def test_string_array_view_type_error():
    # [FUNCTION DOC]
    # Verifies that viewing a string array as another data type raises a type error
    # ("Cannot change data-type for string array").
    arr = pd.array(["a", "b", "c"], dtype="string")
    with pytest.raises(TypeError, match="Cannot change data-type for string array."):
        arr.view("i8")


@pytest.mark.parametrize("box", [pd.Series, pd.array])
def test_numpy_array_ufunc(dtype, box):
    # [FUNCTION DOC]
    # Parametrized over container (Series, array). Verifies a numpy ufunc built from
    # a Python function: returning numbers gives an object-typed container result (a
    # TODO notes integer dtype could be inferred); returning strings keeps the string
    # dtype for python storage, whereas for pyarrow storage the result is an object
    # array (for array input) or a default-dtype Series (the class/dtype is not yet
    # preserved).
    arr = box(["a", "bb", "ccc"], dtype=dtype)

    # custom ufunc that works with string (object) input -> returning numeric
    str_len_ufunc = np.frompyfunc(lambda x: len(x), 1, 1)
    result = str_len_ufunc(arr)
    expected_cls = pd.Series if box is pd.Series else np.array
    # TODO we should infer int64 dtype here?
    expected = expected_cls([1, 2, 3], dtype=object)
    tm.assert_equal(result, expected)

    # custom ufunc returning strings
    str_multiply_ufunc = np.frompyfunc(lambda x: x * 2, 1, 1)
    result = str_multiply_ufunc(arr)
    expected = box(["aa", "bbbb", "cccccc"], dtype=dtype)
    if dtype.storage == "pyarrow":
        # TODO ArrowStringArray should also preserve the class / dtype
        if box is pd.array:
            expected = np.array(["aa", "bbbb", "cccccc"], dtype=object)
        else:
            # not specifying the dtype because the exact dtype is not yet preserved
            expected = pd.Series(["aa", "bbbb", "cccccc"])

    tm.assert_equal(result, expected)


@pytest.mark.parametrize("box", [pd.Series, pd.array])
def test_numpy_random_permute(dtype, box):
    # [FUNCTION DOC]
    # Regression test (GitHub 63935). Parametrized over container. Verifies a numpy
    # random generator permutation of a string container returns a numpy array
    # containing the same strings.
    # https://github.com/pandas-dev/pandas/issues/63935
    arr = box(["a", "bb", "ccc"], dtype=dtype)

    rng = np.random.default_rng(2)
    result = rng.permutation(arr)
    assert isinstance(result, np.ndarray)
    assert sorted(result.tolist()) == ["a", "bb", "ccc"]
