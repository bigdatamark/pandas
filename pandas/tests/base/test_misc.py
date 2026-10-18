# [MODULE DOC]
# Miscellaneous tests of the base object behavior for Series, Index and
# DataFrame: alias docstrings, generated binary operation docstrings, numpy
# compatibility properties, memory usage, searchsorted and positional access.
# Uses the index_or_series_obj, frame_or_series and related fixtures. No import-time
# side effects.

import sys

import numpy as np
import pytest

from pandas.compat import PYPY

from pandas.core.dtypes.common import (
    is_object_dtype,
)

import pandas as pd
from pandas import (
    Index,
    Series,
)


def test_isnull_notnull_docstrings():
    # [FUNCTION DOC]
    # Regression test (GH#41855). Verifies the docstrings of isnull and notnull on
    # DataFrame and Series start with a sentence stating they are aliases of isna
    # and notna.
    # GH#41855 make sure its clear these are aliases
    doc = pd.DataFrame.notnull.__doc__
    assert doc.strip().startswith("DataFrame.notnull is an alias for DataFrame.notna.")
    doc = pd.DataFrame.isnull.__doc__
    assert doc.strip().startswith("DataFrame.isnull is an alias for DataFrame.isna.")

    doc = Series.notnull.__doc__
    assert doc.startswith("\nSeries.notnull is an alias for Series.notna.\n")
    doc = Series.isnull.__doc__
    assert doc.startswith("\nSeries.isnull is an alias for Series.isna.\n")


@pytest.mark.parametrize(
    "op_name, op",
    [
        ("add", "+"),
        ("sub", "-"),
        ("mul", "*"),
        ("mod", "%"),
        ("pow", "**"),
        ("truediv", "/"),
        ("floordiv", "//"),
    ],
)
def test_binary_ops_docstring(frame_or_series, op_name, op):
    # [FUNCTION DOC]
    # Parametrized over operation names (add, sub, mul, mod, pow, truediv, floordiv)
    # with their operator symbols; uses the frame_or_series fixture. Verifies the
    # generated docstring of the operation contains the expression with the lower
    # cased class name as left operand and other as right operand, and the reflected
    # operation contains the reversed expression.
    # not using the all_arithmetic_functions fixture with _get_opstr
    # as _get_opstr is used internally in the dynamic implementation of the docstring
    klass = frame_or_series

    operand1 = klass.__name__.lower()
    operand2 = "other"
    expected_str = " ".join([operand1, op, operand2])
    assert expected_str in getattr(klass, op_name).__doc__

    # reverse version of the binary ops
    expected_str = " ".join([operand2, op, operand1])
    assert expected_str in getattr(klass, "r" + op_name).__doc__


def test_ndarray_compat_properties(index_or_series_obj):
    # [FUNCTION DOC]
    # Verifies the numpy-compatible properties shape, dtype, transpose and nbytes exist;
    # the removed properties strides, itemsize, base and data do not exist; item on
    # a multi-element object raises a value error (only size-1 conversion allowed);
    # ndim is one and size equals the length; item works on length-one objects.
    obj = index_or_series_obj

    # Check that we work.
    for p in ["shape", "dtype", "T", "nbytes"]:
        assert getattr(obj, p, None) is not None

    # deprecated properties
    for p in ["strides", "itemsize", "base", "data"]:
        assert not hasattr(obj, p)

    msg = "can only convert an array of size 1 to a Python scalar"
    with pytest.raises(ValueError, match=msg):
        obj.item()  # len > 1

    assert obj.ndim == 1
    assert obj.size == len(obj)

    assert Index([1]).item() == 1
    assert Series([1]).item() == 1


@pytest.mark.skipif(PYPY, reason="not relevant for PyPy")
def test_memory_usage(index_or_series_memory_obj):
    # [FUNCTION DOC]
    # Skipped on PyPy. Uses index_or_series_memory_obj. Clears the index engine
    # mapping cache so empty objects report zero memory. Verifies memory usage equals
    # zero for empty objects, deep usage exceeds shallow usage when object-like
    # values (python-storage strings, object dtype, categories, levels or the Series
    # index) are present, equals it otherwise, and that deep usage is within 100 bytes
    # of the built-in size function result.
    obj = index_or_series_memory_obj
    # Clear index caches so that len(obj) == 0 report 0 memory usage
    if isinstance(obj, Series):
        is_ser = True
        obj.index._engine.clear_mapping()
    else:
        is_ser = False
        obj._engine.clear_mapping()

    res = obj.memory_usage()
    res_deep = obj.memory_usage(deep=True)

    def _is_object_dtype(obj):
        # [FUNCTION DOC]
        # HELPER: recursive predicate for the memory usage test. Input: obj (Index or
        # Series). Returns true if the values are stored as Python objects: any level of a
        # MultiIndex, the categories of a categorical, python-storage string dtype, or
        # object dtype. Returns false otherwise.
        if isinstance(obj, pd.MultiIndex):
            return any(_is_object_dtype(level) for level in obj.levels)
        elif isinstance(obj.dtype, pd.CategoricalDtype):
            return _is_object_dtype(obj.dtype.categories)
        elif isinstance(obj.dtype, pd.StringDtype):
            return obj.dtype.storage == "python"
        return is_object_dtype(obj)

    has_objects = _is_object_dtype(obj) or (is_ser and _is_object_dtype(obj.index))

    if len(obj) == 0:
        expected = 0
        assert res_deep == res == expected
    elif has_objects:
        # only deep will pick them up
        assert res_deep > res
    else:
        assert res == res_deep

    # sys.getsizeof will call the .memory_usage with
    # deep=True, and add on some GC overhead
    diff = res_deep - sys.getsizeof(obj)
    assert abs(diff) < 100


def test_memory_usage_components_series(series_with_simple_index):
    # [FUNCTION DOC]
    # Verifies a Series memory usage including the index equals the sum of its
    # usage without index and the index usage.
    series = series_with_simple_index
    total_usage = series.memory_usage(index=True)
    non_index_usage = series.memory_usage(index=False)
    index_usage = series.index.memory_usage()
    assert total_usage == non_index_usage + index_usage


def test_memory_usage_components_narrow_series(any_real_numpy_dtype):
    # [FUNCTION DOC]
    # Uses any_real_numpy_dtype. Same sum check for a Series of five values with string
    # index labels.
    series = Series(
        range(5),
        dtype=any_real_numpy_dtype,
        index=[f"i-{i}" for i in range(5)],
        name="a",
    )
    total_usage = series.memory_usage(index=True)
    non_index_usage = series.memory_usage(index=False)
    index_usage = series.index.memory_usage()
    assert total_usage == non_index_usage + index_usage


def test_searchsorted(request, index_or_series_obj):
    # [FUNCTION DOC]
    # Regression test (gh-12238). Uses the request and index_or_series_obj fixtures.
    # Marks expected failure for MultiIndex (GH 14833) and for complex Index. Verifies
    # the numpy searchsorted function with the maximum (or zero if empty) returns a
    # position between zero and the length, with and without a sorter.
    # numpy.searchsorted calls obj.searchsorted under the hood.
    # See gh-12238
    obj = index_or_series_obj

    if isinstance(obj, pd.MultiIndex):
        # See gh-14833
        request.applymarker(
            pytest.mark.xfail(
                reason="np.searchsorted doesn't work on pd.MultiIndex: GH 14833"
            )
        )
    elif obj.dtype.kind == "c" and isinstance(obj, Index):
        # TODO: Should Series cases also raise? Looks like they use numpy
        #  comparison semantics https://github.com/numpy/numpy/issues/15981
        mark = pytest.mark.xfail(reason="complex objects are not comparable")
        request.applymarker(mark)

    max_obj = max(obj, default=0)
    index = np.searchsorted(obj, max_obj)
    assert 0 <= index <= len(obj)

    index = np.searchsorted(obj, max_obj, sorter=range(len(obj)))
    assert 0 <= index <= len(obj)


def test_access_by_position(index_flat):
    # [FUNCTION DOC]
    # Uses index_flat. Skips empty. Verifies positional lookup of an index and of a
    # Series built from it agree at positions 0, 5 and -1 and that -1 equals the last
    # element; indexing past the end raises an index error (message differs for pyarrow
    # strings) as does Series iloc ("single positional indexer is out-of-bounds").
    # KNOWN ISSUE: position 5 is read without checking the length is at least six, so
    # indexes with fewer elements would raise an index error unrelated to the intended
    # checks.
    index = index_flat

    if len(index) == 0:
        pytest.skip("Test doesn't make sense on empty data")

    series = Series(index)
    assert index[0] == series.iloc[0]
    assert index[5] == series.iloc[5]
    assert index[-1] == series.iloc[-1]

    size = len(index)
    assert index[-1] == index[size - 1]

    msg = f"index {size} is out of bounds for axis 0 with size {size}"
    if isinstance(index.dtype, pd.StringDtype) and index.dtype.storage == "pyarrow":
        msg = "index out of bounds"
    with pytest.raises(IndexError, match=msg):
        index[size]
    msg = "single positional indexer is out-of-bounds"
    with pytest.raises(IndexError, match=msg):
        series.iloc[size]
