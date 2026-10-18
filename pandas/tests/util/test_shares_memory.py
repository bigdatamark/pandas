# [MODULE DOC]
# Tests for the testing-utility function that reports whether two array-like objects share memory, across interval arrays, string extension arrays, numpy arrays and range indexes.

import numpy as np

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


def test_shares_memory_interval():
    # [FUNCTION DOC]
    # For an interval range: the object shares memory with itself, its underlying array, a reversed slice and a sliced view; it does not share memory with a copy of its underlying array.
    obj = pd.interval_range(1, 5)

    assert tm.shares_memory(obj, obj)
    assert tm.shares_memory(obj, obj._data)
    assert tm.shares_memory(obj, obj[::-1])
    assert tm.shares_memory(obj, obj[:2])

    assert not tm.shares_memory(obj, obj._data.copy())


@td.skip_if_no("pyarrow")
def test_shares_memory_string():
    # [FUNCTION DOC]
    # GH#55823. Requires pyarrow. Arrow-backed string arrays with a missing-value marker of NA, of NaN, and the arrow string dtype each share memory with themselves.
    # GH#55823
    import pyarrow as pa

    obj = pd.array(["a", "b"], dtype=pd.StringDtype("pyarrow", na_value=pd.NA))
    assert tm.shares_memory(obj, obj)

    obj = pd.array(["a", "b"], dtype=pd.StringDtype("pyarrow", na_value=np.nan))
    assert tm.shares_memory(obj, obj)

    obj = pd.array(["a", "b"], dtype=pd.ArrowDtype(pa.string()))
    assert tm.shares_memory(obj, obj)


def test_shares_memory_numpy():
    # [FUNCTION DOC]
    # A numpy array shares memory with a slice view of itself and not with an independently created equal array.
    arr = np.arange(10)
    view = arr[:5]
    assert tm.shares_memory(arr, view)
    arr2 = np.arange(10)
    assert not tm.shares_memory(arr, arr2)


def test_shares_memory_rangeindex():
    # [FUNCTION DOC]
    # A range index does not share memory with a numpy array of the same values.
    idx = pd.RangeIndex(10)
    arr = np.arange(10)
    assert not tm.shares_memory(idx, arr)
