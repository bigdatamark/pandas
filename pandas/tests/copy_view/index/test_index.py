# [MODULE DOC]
# Copy-on-write tests for the generic Index: indexes derived from frames or
# Series must not see later modifications of the originals, index-returning
# operations do not alias mutable data, and constructors copy inputs by default.
# Uses the get_array helper of the copy_view utilities. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    array,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def index_view(index_data):
    # [FUNCTION DOC]
    # HELPER: used by the index operation tests. Input: index_data, values for a
    # column a. Builds a frame with that column and a float column b, takes a full
    # slice (a view of the frame), makes column a the index (dropping it) and returns
    # a tuple of the resulting index and the earlier view of the frame.
    df = DataFrame({"a": index_data, "b": 1.5})
    view = df[:]
    df = df.set_index("a", drop=True)
    idx = df.index
    # df = None
    return idx, view


def test_set_index_update_column():
    # [FUNCTION DOC]
    # Verifies an index created with set_index keeping the column does not change when
    # the column value in the frame is modified.
    df = DataFrame({"a": [1, 2], "b": 1})
    df = df.set_index("a", drop=False)
    expected = df.index.copy(deep=True)
    df.iloc[0, 0] = 100
    tm.assert_index_equal(df.index, expected)


def test_set_index_drop_update_column():
    # [FUNCTION DOC]
    # Verifies an index created from a column with set_index and dropping it does not
    # change when the earlier view of the frame is modified.
    df = DataFrame({"a": [1, 2], "b": 1.5})
    view = df[:]
    df = df.set_index("a", drop=True)
    expected = df.index.copy(deep=True)
    view.iloc[0, 0] = 100
    tm.assert_index_equal(df.index, expected)


def test_set_index_series():
    # [FUNCTION DOC]
    # Verifies an index set from a Series via set_index does not change when the
    # Series is modified.
    df = DataFrame({"a": [1, 2], "b": 1.5})
    ser = Series([10, 11])
    df = df.set_index(ser)
    expected = df.index.copy(deep=True)
    ser.iloc[0] = 100
    tm.assert_index_equal(df.index, expected)


def test_assign_index_as_series():
    # [FUNCTION DOC]
    # Verifies assigning a Series to the index attribute of a frame is isolated from
    # later changes to the Series.
    df = DataFrame({"a": [1, 2], "b": 1.5})
    ser = Series([10, 11])
    df.index = ser
    expected = df.index.copy(deep=True)
    ser.iloc[0] = 100
    tm.assert_index_equal(df.index, expected)


def test_assign_index_as_index():
    # [FUNCTION DOC]
    # Verifies assigning an Index built from a Series to the index attribute is
    # isolated from later changes to the Series (the temporary reference is cleared).
    df = DataFrame({"a": [1, 2], "b": 1.5})
    ser = Series([10, 11])
    rhs_index = Index(ser)
    df.index = rhs_index
    rhs_index = None  # overwrite to clear reference
    expected = df.index.copy(deep=True)
    ser.iloc[0] = 100
    tm.assert_index_equal(df.index, expected)


def test_index_from_series():
    # [FUNCTION DOC]
    # Verifies an Index built from a Series is not changed when the Series changes.
    ser = Series([1, 2])
    idx = Index(ser)
    expected = idx.copy(deep=True)
    ser.iloc[0] = 100
    tm.assert_index_equal(idx, expected)


def test_index_from_series_copy():
    # [FUNCTION DOC]
    # Verifies that building an Index from a Series with copy true leaves the
    # Series with its own data: modifying the Series keeps the same underlying array.
    ser = Series([1, 2])
    idx = Index(ser, copy=True)  # noqa: F841
    arr = get_array(ser)
    ser.iloc[0] = 100
    assert np.shares_memory(get_array(ser), arr)


def test_index_from_index():
    # [FUNCTION DOC]
    # Verifies an Index built from an Index built from a Series is isolated from the
    # Series.
    ser = Series([1, 2])
    idx = Index(ser)
    idx = Index(idx)
    expected = idx.copy(deep=True)
    ser.iloc[0] = 100
    tm.assert_index_equal(idx, expected)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x._shallow_copy(x._values),
        lambda x: x.view(),
        lambda x: x.take([0, 1]),
        lambda x: x.repeat([1, 1]),
        lambda x: x[slice(0, 2)],
        lambda x: x[[0, 1]],
        lambda x: x._getitem_slice(slice(0, 2)),
        lambda x: x.delete([]),
        lambda x: x.rename("b"),
        lambda x: x.astype("Int64", copy=False),
    ],
    ids=[
        "_shallow_copy",
        "view",
        "take",
        "repeat",
        "getitem_slice",
        "getitem_list",
        "_getitem_slice",
        "delete",
        "rename",
        "astype",
    ],
)
def test_index_ops(func, request):
    # [FUNCTION DOC]
    # Parametrized over ten operations on an index (shallow copy, view, take, repeat,
    # slice, list indexing, private slice, delete nothing, rename, cast to Int64
    # without copy), ids as listed; uses the request fixture. Builds an index and a
    # view of the originating frame, applies the operation, modifies the frame view,
    # and verifies the result is unchanged (equal to the expected copy, cast for the
    # astype case), ignoring names.
    idx, view_ = index_view([1, 2])
    expected = idx.copy(deep=True)
    if "astype" in request.node.callspec.id:
        expected = expected.astype("Int64")
    idx = func(idx)
    view_.iloc[0, 0] = 100
    tm.assert_index_equal(idx, expected, check_names=False)


def test_infer_objects():
    # [FUNCTION DOC]
    # Verifies infer_objects without copying returns an index not affected by later
    # modification of the originating frame view.
    idx, view_ = index_view(["a", "b"])
    expected = idx.copy(deep=True)
    idx = idx.infer_objects(copy=False)
    view_.iloc[0, 0] = "aaaa"
    tm.assert_index_equal(idx, expected, check_names=False)


def test_index_to_frame():
    # [FUNCTION DOC]
    # Verifies the frame from an Index to_frame shares memory with the index but holds
    # a reference (is not exclusively owned), and modifying the frame leaves the index
    # unchanged.
    idx = Index([1, 2, 3], name="a")
    expected = idx.copy(deep=True)
    df = idx.to_frame()
    assert np.shares_memory(get_array(df, "a"), idx._values)
    assert not df._mgr._has_no_reference(0)

    df.iloc[0, 0] = 100
    tm.assert_index_equal(idx, expected)


def test_index_values():
    # [FUNCTION DOC]
    # Verifies the values attribute of an Index is a read-only numpy array.
    idx = Index([1, 2, 3])
    result = idx.values
    assert result.flags.writeable is False


def test_constructor_copy_input_ndarray_default():
    # [FUNCTION DOC]
    # Verifies constructing an Index from a numpy array copies it by default.
    arr = np.array([0, 1])
    idx = Index(arr)
    assert not np.shares_memory(arr, get_array(idx))


def test_constructor_copy_input_ea_default():
    # [FUNCTION DOC]
    # Verifies constructing an Index from a masked extension array copies it by default.
    arr = array([0, 1], dtype="Int64")
    idx = Index(arr)
    assert not tm.shares_memory(arr, idx.array)


def test_series_from_temporary_index_readonly_data():
    # [FUNCTION DOC]
    # Regression test (GH 63370). With a read-only int8 numpy source, verifies a Series
    # built from a temporary Index does not share memory, holds no references, and
    # can be modified through a boolean mask giving the expected values.
    # GH 63370
    arr = np.array([0, 1], dtype=np.dtype(np.int8))
    arr.flags.writeable = False
    ser = Series(Index(arr))
    assert not np.shares_memory(arr, get_array(ser))
    assert ser._mgr._has_no_reference(0)
    ser[[False, True]] = np.array([0, 2], dtype=np.dtype(np.int8))
    expected = Series([0, 2], dtype=np.dtype(np.int8))
    tm.assert_series_equal(ser, expected)
