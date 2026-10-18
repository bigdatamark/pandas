# [PACKAGE DOC]
# HELPER: internal testing support package used by pandas' own test suite (the
# public testing module re-exports only the assertion functions).
# Submodules: _hypothesis (property-test strategies), _io (file round-trip
# helpers), _warnings (warning assertions), asserters (equality assertion
# functions for arrays, indexes, series and frames), compat (small dtype/object
# accessors) and contexts (temporary environment context managers).
# This file defines and re-exports: lists of dtype names or objects grouped by
# kind (unsigned, signed and all integers in numpy and nullable extension
# flavors; floats; complex; strings; datetimes and timedeltas at nanosecond
# resolution; booleans; bytes; objects; combinations; and, when the arrow
# library is available, lists of arrow data types and their string forms with
# empty lists as fallbacks otherwise); narrow numpy types; builtin Python data
# types; the native byte order marker; lists of missing-value objects and numpy
# not-a-time scalars in every unit; lists of arithmetic and comparison special
# method names; helper functions (box_expected, to_array, the CSV string
# converter, error context helper, parametrization helper, operator lookup,
# indexer helpers, unit helper, memory sharing check, multithreaded runner);
# test subclasses of series and frame; and re-exports of the locale helpers,
# warning assertions, asserters, compat helpers and context managers. The names
# in __all__ are the intended public surface of this package.
# Import-time side effects: imports the pandas package and most of core;
# reads the string-dtype future option to choose the string dtype list; imports
# pyarrow if available (flag from the compat module) to build arrow dtype lists.
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import operator
import os
from sys import byteorder
import threading
from typing import (
    TYPE_CHECKING,
    ContextManager,
)

import numpy as np

from pandas._config import using_string_dtype
from pandas._config.localization import (
    can_set_locale,
    get_locales,
    set_locale,
)

from pandas.compat import HAS_PYARROW

import pandas as pd
from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
)
from pandas._testing._io import (
    round_trip_pathlib,
    round_trip_pickle,
    write_to_compressed,
)
from pandas._testing._warnings import (
    assert_produces_warning,
    maybe_produces_warning,
)
from pandas._testing.asserters import (
    assert_almost_equal,
    assert_attr_equal,
    assert_categorical_equal,
    assert_class_equal,
    assert_contains_all,
    assert_copy,
    assert_datetime_array_equal,
    assert_dict_equal,
    assert_equal,
    assert_extension_array_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_indexing_slices_equivalent,
    assert_interval_array_equal,
    assert_is_sorted,
    assert_metadata_equivalent,
    assert_numpy_array_equal,
    assert_period_array_equal,
    assert_series_equal,
    assert_sp_array_equal,
    assert_timedelta_array_equal,
    raise_assert_detail,
)
from pandas._testing.compat import (
    get_dtype,
    get_obj,
)
from pandas._testing.contexts import (
    decompress_file,
    raises_chained_assignment_error,
    set_timezone,
    with_csv_dialect,
)
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    NumpyExtensionArray,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        Dtype,
        NpDtype,
    )


UNSIGNED_INT_NUMPY_DTYPES: list[NpDtype] = ["uint8", "uint16", "uint32", "uint64"]
UNSIGNED_INT_EA_DTYPES: list[Dtype] = ["UInt8", "UInt16", "UInt32", "UInt64"]
SIGNED_INT_NUMPY_DTYPES: list[NpDtype] = [int, "int8", "int16", "int32", "int64"]
SIGNED_INT_EA_DTYPES: list[Dtype] = ["Int8", "Int16", "Int32", "Int64"]
ALL_INT_NUMPY_DTYPES = UNSIGNED_INT_NUMPY_DTYPES + SIGNED_INT_NUMPY_DTYPES
ALL_INT_EA_DTYPES = UNSIGNED_INT_EA_DTYPES + SIGNED_INT_EA_DTYPES
ALL_INT_DTYPES: list[Dtype] = [*ALL_INT_NUMPY_DTYPES, *ALL_INT_EA_DTYPES]

FLOAT_NUMPY_DTYPES: list[NpDtype] = [float, "float32", "float64"]
FLOAT_EA_DTYPES: list[Dtype] = ["Float32", "Float64"]
ALL_FLOAT_DTYPES: list[Dtype] = [*FLOAT_NUMPY_DTYPES, *FLOAT_EA_DTYPES]

COMPLEX_DTYPES: list[Dtype] = [complex, "complex64", "complex128"]
if using_string_dtype():
    STRING_DTYPES: list[Dtype] = ["U"]
else:
    STRING_DTYPES: list[Dtype] = [str, "str", "U"]  # type: ignore[no-redef]
COMPLEX_FLOAT_DTYPES: list[Dtype] = [*COMPLEX_DTYPES, *FLOAT_NUMPY_DTYPES]

DATETIME64_DTYPES: list[Dtype] = ["datetime64[ns]", "M8[ns]"]
TIMEDELTA64_DTYPES: list[Dtype] = ["timedelta64[ns]", "m8[ns]"]

BOOL_DTYPES: list[Dtype] = [bool, "bool"]
BYTES_DTYPES: list[Dtype] = [bytes, "bytes"]
OBJECT_DTYPES: list[Dtype] = [object, "object"]

ALL_REAL_NUMPY_DTYPES = FLOAT_NUMPY_DTYPES + ALL_INT_NUMPY_DTYPES
ALL_REAL_EXTENSION_DTYPES = FLOAT_EA_DTYPES + ALL_INT_EA_DTYPES
ALL_REAL_DTYPES: list[Dtype] = [*ALL_REAL_NUMPY_DTYPES, *ALL_REAL_EXTENSION_DTYPES]
ALL_NUMERIC_DTYPES: list[Dtype] = [*ALL_REAL_DTYPES, *COMPLEX_DTYPES]

ALL_NUMPY_DTYPES = (
    ALL_REAL_NUMPY_DTYPES
    + COMPLEX_DTYPES
    + STRING_DTYPES
    + DATETIME64_DTYPES
    + TIMEDELTA64_DTYPES
    + BOOL_DTYPES
    + OBJECT_DTYPES
    + BYTES_DTYPES
)

NARROW_NP_DTYPES = [
    np.float16,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.uint8,
    np.uint16,
    np.uint32,
]

PYTHON_DATA_TYPES = [
    str,
    int,
    float,
    complex,
    list,
    tuple,
    range,
    dict,
    set,
    frozenset,
    bool,
    bytes,
    bytearray,
    memoryview,
]

ENDIAN = {"little": "<", "big": ">"}[byteorder]

NULL_OBJECTS = [None, np.nan, pd.NaT, float("nan"), pd.NA, Decimal("NaN")]
NP_NAT_OBJECTS = [
    cls("NaT", unit)
    for cls in [np.datetime64, np.timedelta64]
    for unit in [
        "Y",
        "M",
        "W",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        "ps",
        "fs",
        "as",
    ]
]

if HAS_PYARROW:
    import pyarrow as pa

    UNSIGNED_INT_PYARROW_DTYPES = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
    SIGNED_INT_PYARROW_DTYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
    ALL_INT_PYARROW_DTYPES = UNSIGNED_INT_PYARROW_DTYPES + SIGNED_INT_PYARROW_DTYPES
    ALL_INT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in ALL_INT_PYARROW_DTYPES
    ]

    # pa.float16 doesn't seem supported
    # https://github.com/apache/arrow/blob/master/python/pyarrow/src/arrow/python/helpers.cc#L86
    FLOAT_PYARROW_DTYPES = [pa.float32(), pa.float64()]
    FLOAT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in FLOAT_PYARROW_DTYPES
    ]
    DECIMAL_PYARROW_DTYPES = [pa.decimal128(7, 3)]
    STRING_PYARROW_DTYPES = [pa.string()]
    BINARY_PYARROW_DTYPES = [pa.binary()]

    TIME_PYARROW_DTYPES = [
        pa.time32("s"),
        pa.time32("ms"),
        pa.time64("us"),
        pa.time64("ns"),
    ]
    DATE_PYARROW_DTYPES = [pa.date32(), pa.date64()]
    DATETIME_PYARROW_DTYPES = [
        pa.timestamp(unit=unit, tz=tz)
        for unit in ["s", "ms", "us", "ns"]
        for tz in [None, "UTC", "US/Pacific", "US/Eastern"]
    ]
    TIMEDELTA_PYARROW_DTYPES = [pa.duration(unit) for unit in ["s", "ms", "us", "ns"]]

    BOOL_PYARROW_DTYPES = [pa.bool_()]

    # TODO: Add container like pyarrow types:
    #  https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions
    ALL_PYARROW_DTYPES = (
        ALL_INT_PYARROW_DTYPES
        + FLOAT_PYARROW_DTYPES
        + DECIMAL_PYARROW_DTYPES
        + STRING_PYARROW_DTYPES
        + BINARY_PYARROW_DTYPES
        + TIME_PYARROW_DTYPES
        + DATE_PYARROW_DTYPES
        + DATETIME_PYARROW_DTYPES
        + TIMEDELTA_PYARROW_DTYPES
        + BOOL_PYARROW_DTYPES
    )
    ALL_REAL_PYARROW_DTYPES_STR_REPR = (
        ALL_INT_PYARROW_DTYPES_STR_REPR + FLOAT_PYARROW_DTYPES_STR_REPR
    )
else:
    FLOAT_PYARROW_DTYPES_STR_REPR = []
    ALL_INT_PYARROW_DTYPES_STR_REPR = []
    ALL_PYARROW_DTYPES = []
    ALL_REAL_PYARROW_DTYPES_STR_REPR = []

ALL_REAL_NULLABLE_DTYPES = (
    FLOAT_NUMPY_DTYPES + ALL_REAL_EXTENSION_DTYPES + ALL_REAL_PYARROW_DTYPES_STR_REPR
)

arithmetic_dunder_methods = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]

comparison_dunder_methods = ["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"]


# -----------------------------------------------------------------------------
# Comparators


def box_expected(expected, box_cls, transpose: bool = True):
    # [FUNCTION DOC]
    # HELPER: test helper that wraps an expected result in a requested
    # container kind. Parameters: expected (array-like result), box_cls (the
    # target kind: the array constructor function, Index, Series, DataFrame,
    # the numpy array type or constructor, or the to_array helper) and transpose
    # (default true; only for DataFrame).
    # Behavior per kind: for the array constructor, a range index is wrapped in a
    # numpy-backed extension array (to avoid becoming an integer array), anything
    # else goes through the array constructor without copying; Index is built
    # without copying; Series is built directly; DataFrame is built as a single
    # column frame and, when transpose is true, transposed to one row and then
    # stacked twice with a fresh index so that it has two rows (avoids
    # single-row special cases in datetime arithmetic); numpy array kinds produce
    # a numpy array; the to_array helper is applied for that kind.
    # Raises a not-implemented error for any other box class.
    # KNOWN ISSUE: the existing docstring lists only Index, Series and DataFrame
    # as box classes, but the code also supports the array constructor, numpy
    # array kinds and the to_array helper.
    """
    Helper function to wrap the expected output of a test in a given box_class.

    Parameters
    ----------
    expected : np.ndarray, Index, Series
    box_cls : {Index, Series, DataFrame}

    Returns
    -------
    subclass of box_cls
    """
    if box_cls is pd.array:
        if isinstance(expected, RangeIndex):
            # pd.array would return an IntegerArray
            expected = NumpyExtensionArray(np.asarray(expected._values))
        else:
            expected = pd.array(expected, copy=False)
    elif box_cls is Index:
        expected = Index(expected, copy=False)
    elif box_cls is Series:
        expected = Series(expected)
    elif box_cls is DataFrame:
        expected = Series(expected).to_frame()
        if transpose:
            # for vector operations, we need a DataFrame to be a single-row,
            #  not a single-column, in order to operate against non-DataFrame
            #  vectors of the same length. But convert to two rows to avoid
            #  single-row special cases in datetime arithmetic
            expected = expected.T
            expected = pd.concat([expected] * 2, ignore_index=True)
    elif box_cls is np.ndarray or box_cls is np.array:
        expected = np.array(expected)
    elif box_cls is to_array:
        expected = to_array(expected)
    else:
        raise NotImplementedError(box_cls)
    return expected


def to_array(obj):
    # [FUNCTION DOC]
    # HELPER: test helper. Parameter obj is any array-like. If it has no dtype
    # attribute, converts it to a numpy array; otherwise returns the underlying
    # array (numpy arrays kept as plain numpy arrays rather than wrapped) with
    # the library's array extraction helper. Unlike the array constructor, does
    # not convert numpy dtypes to nullable extension dtypes. Marked in the
    # source as a temporary implementation.
    """
    Similar to pd.array, but does not cast numpy dtypes to nullable dtypes.
    """
    # temporary implementation until we get pd.array in place
    dtype = getattr(obj, "dtype", None)

    if dtype is None:
        return np.asarray(obj)

    return extract_array(obj, extract_numpy=True)


class SubclassedSeries(Series):
    # [CLASS DOC]
    # HELPER: test subclass of the series type used to check that operations
    # preserve subclass behavior. Class attribute _metadata lists the names of
    # extra attributes propagated to results (testattr and name). Overrides the
    # constructor hooks so that results of operations are built through generic
    # callables (rather than returning the class itself), to ensure the library
    # does not depend on those properties returning a class. Expanding to
    # two dimensions yields the matching frame subclass.
    _metadata = ["testattr", "name"]

    @property
    def _constructor(self):
        # [FUNCTION DOC]
        # HELPER: property returning a callable that builds a new instance of
        # this subclass from any positional and keyword arguments. Deliberately a
        # lambda rather than the class (see the linked issues in the source).
        # For testing, those properties return a generic callable, and not
        # the actual class. In this case that is equivalent, but it is to
        # ensure we don't rely on the property returning a class
        # See https://github.com/pandas-dev/pandas/pull/46018 and
        # https://github.com/pandas-dev/pandas/issues/32638 and linked issues
        return lambda *args, **kwargs: SubclassedSeries(*args, **kwargs)

    @property
    def _constructor_expanddim(self):
        # [FUNCTION DOC]
        # HELPER: property returning a callable that builds the frame subclass,
        # used when the series is expanded to two dimensions.
        return lambda *args, **kwargs: SubclassedDataFrame(*args, **kwargs)


class SubclassedDataFrame(DataFrame):
    # [CLASS DOC]
    # HELPER: test subclass of the frame type, counterpart of the series
    # subclass. Class attribute _metadata lists propagated extra attribute names
    # (testattr). Overrides the constructor hooks to return generic callables
    # building this subclass, and the sliced hook to build the series subclass
    # (one-dimensional results).
    _metadata = ["testattr"]

    @property
    def _constructor(self):
        # [FUNCTION DOC]
        # HELPER: property returning a callable that builds a new instance of this
        # frame subclass from any arguments.
        return lambda *args, **kwargs: SubclassedDataFrame(*args, **kwargs)

    # error: Cannot override writeable attribute with read-only property
    @property
    def _constructor_sliced(self):  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: property returning a callable that builds the series subclass,
        # used for one-dimensional results such as a column or row.
        return lambda *args, **kwargs: SubclassedSeries(*args, **kwargs)


def convert_rows_list_to_csv_str(rows_list: list[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: test helper. Parameter rows_list is a list of row strings. Returns
    # a single string with the rows joined by the operating system's line
    # separator, plus a trailing separator. Used to build expected CSV output.
    """
    Convert list of CSV rows to single CSV-formatted string for current OS.

    This method is used for creating expected value of to_csv() method.

    Parameters
    ----------
    rows_list : List[str]
        Each element represents the row of csv.

    Returns
    -------
    str
        Expected output of to_csv() in current OS.
    """
    sep = os.linesep
    return sep.join(rows_list) + sep


def external_error_raised(expected_exception: type[Exception]) -> ContextManager:
    # [FUNCTION DOC]
    # HELPER: test helper. Parameter expected_exception is an exception class.
    # Returns the test framework's raises context manager for that class with
    # message matching explicitly disabled, to mark errors whose message comes
    # from an external library. Imports the test framework lazily.
    # KNOWN ISSUE: the existing docstring says it returns a callable, but a
    # context manager object is returned.
    """
    Helper function to mark pytest.raises that have an external error message.

    Parameters
    ----------
    expected_exception : Exception
        Expected error to raise.

    Returns
    -------
    Callable
        Regular `pytest.raises` function with `match` equal to `None`.
    """
    import pytest

    return pytest.raises(expected_exception, match=None)


def get_cython_table_params(ndframe, func_names_and_expected):
    # [FUNCTION DOC]
    # HELPER: test helper for parametrization. Parameters: ndframe (a frame or
    # series) and func_names_and_expected (a sequence of pairs of method name
    # and expected result). Returns a list of triples, one per pair, each holding
    # the given object, the method name and the expected result, in order.
    # KNOWN ISSUE: the existing docstring says each triple holds a function and
    # refers to a cython table of the common module, but the code just repeats
    # the name string and uses no table.
    """
    Combine frame, functions from com._cython_table
    keys and expected result.

    Parameters
    ----------
    ndframe : DataFrame or Series
    func_names_and_expected : Sequence of two items
        The first item is a name of an NDFrame method ('sum', 'prod') etc.
        The second item is the expected return value.

    Returns
    -------
    list
        List of three items (DataFrame, function, expected result)
    """
    results = []
    for func_name, expected in func_names_and_expected:
        results.append((ndframe, func_name, expected))
    return results


def get_op_from_name(op_name: str) -> Callable:
    # [FUNCTION DOC]
    # HELPER: test helper. Parameter op_name is an operator name with or without
    # surrounding underscores (add or the special add method name). Strips the
    # underscores and looks the name up in the standard operator module.
    # If absent, assumes it is a reflected name (leading letter r, e.g. radd),
    # looks up the name without its first character and returns a function that
    # calls the operator with its two arguments swapped. Returns a callable of two
    # arguments. Raises an attribute error if neither lookup succeeds.
    # (The reflected function is built by an assigned anonymous function.)
    """
    The operator function for a given op name.

    Parameters
    ----------
    op_name : str
        The op name, in form of "add" or "__add__".

    Returns
    -------
    function
        A function performing the operation.
    """
    short_opname = op_name.strip("_")
    try:
        op = getattr(operator, short_opname)
    except AttributeError:
        # Assume it is the reverse operator
        rop = getattr(operator, short_opname[1:])
        op = lambda x, y: rop(y, x)

    return op


# -----------------------------------------------------------------------------
# Indexing test helpers


def getitem(x):
    # [FUNCTION DOC]
    # HELPER: indexer selector for parametrized tests. Returns its argument
    # unchanged, so indexing the result uses plain bracket indexing for reading.
    return x


def setitem(x):
    # [FUNCTION DOC]
    # HELPER: indexer selector for parametrized tests. Returns its argument
    # unchanged, so indexing the result uses plain bracket indexing for writing
    # (identical to the get variant; the name is what distinguishes the test cases).
    return x


def loc(x):
    # [FUNCTION DOC]
    # HELPER: indexer selector. Returns the label-based indexer attribute of the
    # object.
    return x.loc


def iloc(x):
    # [FUNCTION DOC]
    # HELPER: indexer selector. Returns the position-based indexer attribute of
    # the object.
    return x.iloc


def at(x):
    # [FUNCTION DOC]
    # HELPER: indexer selector. Returns the scalar label-based indexer attribute
    # of the object.
    return x.at


def iat(x):
    # [FUNCTION DOC]
    # HELPER: indexer selector. Returns the scalar position-based indexer
    # attribute of the object.
    return x.iat


# -----------------------------------------------------------------------------

_UNITS = ["s", "ms", "us", "ns"]


def get_finest_unit(left: str, right: str) -> str:
    # [FUNCTION DOC]
    # HELPER: test helper. Parameters left and right are datetime unit names
    # from seconds, milliseconds, microseconds and nanoseconds. Returns the one
    # with the finer resolution (later in that order); on a tie returns left.
    # Raises a value error for any other unit name.
    """
    Find the higher of two datetime64 units.
    """
    if _UNITS.index(left) >= _UNITS.index(right):
        return left
    return right


def shares_memory(left, right) -> bool:
    # [FUNCTION DOC]
    # HELPER: test helper that reports whether two array-like objects share
    # memory, extending numpy's check to pandas containers. Parameters left and
    # right can be numpy arrays, indexes, series, frames, extension arrays.
    # Dispatch on left (after swapping when only right is a numpy array): both
    # numpy -> numpy's check; range index -> false; multi index -> check its
    # codes; index or series -> compare the underlying values; numpy-backed
    # extension arrays -> the backing array; sparse arrays -> the stored values;
    # interval arrays -> true if either endpoint array shares; arrow arrays ->
    # if both arrow, compare the address of the first buffer of the first chunk
    # (only chunk 0 is examined), otherwise numpy's check; masked arrays -> true if
    # either the data or the mask arrays share; frames with a single block ->
    # check that block's values. Raises a not-implemented error for any other
    # combination.
    """
    Pandas-compat for np.shares_memory.
    """
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.shares_memory(left, right)
    elif isinstance(left, np.ndarray):
        # Call with reversed args to get to unpacking logic below.
        return shares_memory(right, left)

    if isinstance(left, RangeIndex):
        return False
    if isinstance(left, MultiIndex):
        return shares_memory(left._codes, right)
    if isinstance(left, (Index, Series)):
        if isinstance(right, (Index, Series)):
            return shares_memory(left._values, right._values)
        return shares_memory(left._values, right)

    if isinstance(left, NDArrayBackedExtensionArray):
        return shares_memory(left._ndarray, right)
    if isinstance(left, pd.core.arrays.SparseArray):
        return shares_memory(left.sp_values, right)
    if isinstance(left, pd.core.arrays.IntervalArray):
        return shares_memory(left._left, right) or shares_memory(left._right, right)

    if isinstance(left, ArrowExtensionArray):
        if isinstance(right, ArrowExtensionArray):
            # https://github.com/pandas-dev/pandas/pull/43930#discussion_r736862669
            left_pa_data = left._pa_array
            right_pa_data = right._pa_array
            left_buf1 = left_pa_data.chunk(0).buffers()[1]
            right_buf1 = right_pa_data.chunk(0).buffers()[1]
            return left_buf1.address == right_buf1.address
        else:
            # if we have one one ArrowExtensionArray and one other array, assume
            # they can only share memory if they share the same numpy buffer
            return np.shares_memory(left, right)

    if isinstance(left, BaseMaskedArray) and isinstance(right, BaseMaskedArray):
        # By convention, we'll say these share memory if they share *either*
        #  the _data or the _mask
        return np.shares_memory(left._data, right._data) or np.shares_memory(
            left._mask, right._mask
        )

    if isinstance(left, DataFrame) and len(left._mgr.blocks) == 1:
        arr = left._mgr.blocks[0].values
        return shares_memory(arr, right)

    raise NotImplementedError(type(left), type(right))


def run_multithreaded(closure, max_workers, arguments=None, pass_barrier=False):
    # [FUNCTION DOC]
    # HELPER: test helper that runs one callable concurrently in several threads.
    # Parameters: closure (callable), max_workers (number of threads and of
    # submitted calls), arguments (optional iterable of positional arguments,
    # copied to a list) and pass_barrier (if true, a barrier for max_workers
    # parties is appended as the last argument so threads can synchronize).
    # Submits max_workers identical calls, then waits for all and re-raises the
    # first exception encountered from any result. If the system cannot start the
    # threads (runtime error), the barrier (if any) is aborted so started threads
    # do not hang, and the test is skipped through the test framework.
    # Returns nothing.
    with ThreadPoolExecutor(max_workers=max_workers) as tpe:
        if arguments is None:
            arguments = []
        else:
            arguments = list(arguments)

        if pass_barrier:
            barrier = threading.Barrier(max_workers)
            arguments.append(barrier)

        try:
            futures = []
            for _ in range(max_workers):
                futures.append(tpe.submit(closure, *arguments))  # noqa: PERF401
        except RuntimeError as e:
            import pytest

            pytest.skip(
                f"Spawning {max_workers} threads failed with "
                f"error {e!r} (likely due to resource limits on the "
                "system running the tests)"
            )
        finally:
            if len(futures) < max_workers and pass_barrier:
                barrier.abort()
        for f in futures:
            f.result()


__all__ = [
    "ALL_INT_EA_DTYPES",
    "ALL_INT_NUMPY_DTYPES",
    "ALL_NUMPY_DTYPES",
    "ALL_REAL_NUMPY_DTYPES",
    "BOOL_DTYPES",
    "BYTES_DTYPES",
    "COMPLEX_DTYPES",
    "DATETIME64_DTYPES",
    "ENDIAN",
    "FLOAT_EA_DTYPES",
    "FLOAT_NUMPY_DTYPES",
    "NARROW_NP_DTYPES",
    "NP_NAT_OBJECTS",
    "NULL_OBJECTS",
    "OBJECT_DTYPES",
    "SIGNED_INT_EA_DTYPES",
    "SIGNED_INT_NUMPY_DTYPES",
    "STRING_DTYPES",
    "TIMEDELTA64_DTYPES",
    "UNSIGNED_INT_EA_DTYPES",
    "UNSIGNED_INT_NUMPY_DTYPES",
    "SubclassedDataFrame",
    "SubclassedSeries",
    "assert_almost_equal",
    "assert_attr_equal",
    "assert_categorical_equal",
    "assert_class_equal",
    "assert_contains_all",
    "assert_copy",
    "assert_datetime_array_equal",
    "assert_dict_equal",
    "assert_equal",
    "assert_extension_array_equal",
    "assert_frame_equal",
    "assert_index_equal",
    "assert_indexing_slices_equivalent",
    "assert_interval_array_equal",
    "assert_is_sorted",
    "assert_metadata_equivalent",
    "assert_numpy_array_equal",
    "assert_period_array_equal",
    "assert_produces_warning",
    "assert_series_equal",
    "assert_sp_array_equal",
    "assert_timedelta_array_equal",
    "at",
    "box_expected",
    "can_set_locale",
    "convert_rows_list_to_csv_str",
    "decompress_file",
    "external_error_raised",
    "get_cython_table_params",
    "get_dtype",
    "get_finest_unit",
    "get_locales",
    "get_obj",
    "get_op_from_name",
    "getitem",
    "iat",
    "iloc",
    "loc",
    "maybe_produces_warning",
    "raise_assert_detail",
    "raises_chained_assignment_error",
    "round_trip_pathlib",
    "round_trip_pickle",
    "run_multithreaded",
    "set_locale",
    "set_timezone",
    "setitem",
    "shares_memory",
    "to_array",
    "with_csv_dialect",
    "write_to_compressed",
]
