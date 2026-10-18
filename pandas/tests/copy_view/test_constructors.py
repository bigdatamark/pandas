# [MODULE DOC]
# Copy-on-write tests for the Series and DataFrame constructors: when
# constructing from another Series, Index, array, extension array, dict of
# Series or manager, the result must share memory lazily (tracking a reference)
# unless a copy is explicitly requested or required by a dtype change/reindex,
# and mutations must never propagate between source and result.
# Uses the get_array helper from the sibling util module and the fixture for
# choosing between Index and Series constructors.

import numpy as np
import pytest

from pandas._config import using_string_dtype

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

# -----------------------------------------------------------------------------
# Copy/view behaviour for Series / DataFrame constructors


@pytest.mark.parametrize("dtype", [None, "int64"])
def test_series_from_series(dtype):
    # [FUNCTION DOC]
    # Parametrized over dtype None or int64. Building a Series from a Series is a lazy
    # shallow copy: shares memory and the block has a reference. Writing to the
    # result (or to the parent) triggers copy on write and does not propagate.
    # Explicit copy=False behaves the same; copy=True produces a hard copy up front,
    # leaving the parent without references.
    # Case: constructing a Series from another Series object follows CoW rules:
    # a new object is returned and thus mutations are not propagated
    ser = Series([1, 2, 3], name="name")

    # default is copy=False -> new Series is a shallow copy / view of original
    result = Series(ser, dtype=dtype)

    # the shallow copy still shares memory
    assert np.shares_memory(get_array(ser), get_array(result))

    assert result._mgr.blocks[0].refs.has_reference()

    # mutating new series copy doesn't mutate original
    result.iloc[0] = 0
    assert ser.iloc[0] == 1
    # mutating triggered a copy-on-write -> no longer shares memory
    assert not np.shares_memory(get_array(ser), get_array(result))

    # the same when modifying the parent
    result = Series(ser, dtype=dtype)

    # mutating original doesn't mutate new series
    ser.iloc[0] = 0
    assert result.iloc[0] == 1

    # forcing copy=False still gives a CoW shallow copy
    result = Series(ser, dtype=dtype, copy=False)
    assert np.shares_memory(get_array(ser), get_array(result))
    assert result._mgr.blocks[0].refs.has_reference()

    # forcing copy=True still results in an actual hard copy up front
    result = Series(ser, dtype=dtype, copy=True)
    assert not np.shares_memory(get_array(ser), get_array(result))
    assert ser._mgr._has_no_reference(0)


def test_series_from_series_with_reindex():
    # [FUNCTION DOC]
    # Passing an index equal to the existing one (same object, copy, list, renamed)
    # still gives a lazy shallow copy; writing to the result does not change the
    # source. With copy=True a hard copy with no references is made. When the index
    # forces a real reindex, the result owns its memory and has no references.
    # Case: constructing a Series from another Series with specifying an index
    # that potentially requires a reindex of the values
    ser = Series([1, 2, 3], name="name")

    # passing an index that doesn't actually require a reindex of the values
    # -> still getting a CoW shallow copy
    for index in [
        ser.index,
        ser.index.copy(),
        list(ser.index),
        ser.index.rename("idx"),
    ]:
        result = Series(ser, index=index)
        assert np.shares_memory(ser.values, result.values)
        result.iloc[0] = 0
        assert ser.iloc[0] == 1

        # forcing copy=True still results in an actual hard copy up front
        result = Series(ser, index=index, copy=True)
        assert not np.shares_memory(ser.values, result.values)
        assert not result._mgr.blocks[0].refs.has_reference()

    # ensure that if an actual reindex is needed, we don't have any refs
    # (mutating the result wouldn't trigger CoW)
    result = Series(ser, index=[0, 1, 2, 3])
    assert not np.shares_memory(ser.values, result.values)
    assert not result._mgr.blocks[0].refs.has_reference()


@pytest.mark.parametrize("dtype", [None, "int64"])
@pytest.mark.parametrize("idx", [None, RangeIndex(start=0, stop=3, step=1)])
@pytest.mark.parametrize(
    "arr", [np.array([1, 2, 3], dtype="int64"), pd.array([1, 2, 3], dtype="Int64")]
)
def test_series_from_array(idx, dtype, arr):
    # [FUNCTION DOC]
    # Parametrized over dtype, index (none or range) and input array (numpy int64 or
    # nullable Int64). By default the Series copies the array, so later changes to the
    # array do not show in the Series. With copy=False the Series is a true view
    # that is not protected by copy on write.
    ser = Series(arr, dtype=dtype, index=idx)
    ser_orig = ser.copy()
    data = getattr(arr, "_data", arr)
    assert not np.shares_memory(get_array(ser), data)

    arr[0] = 100
    tm.assert_series_equal(ser, ser_orig)

    # if the user explicitly passes copy=False, we get an actual view
    # not protected by CoW
    ser = Series(arr, dtype=dtype, index=idx, copy=False)
    assert np.shares_memory(get_array(ser), data)
    arr[0] = 50
    assert ser.iloc[0] == 50


@pytest.mark.parametrize("copy", [True, False, None])
def test_series_from_array_different_dtype(copy):
    # [FUNCTION DOC]
    # Parametrized over copy (True, False, None). Requesting a different dtype than
    # the numpy array's forces a conversion, so the result never shares memory.
    arr = np.array([1, 2, 3], dtype="int64")
    ser = Series(arr, dtype="int32", copy=copy)
    assert not np.shares_memory(get_array(ser), arr)


@pytest.mark.parametrize(
    "idx",
    [
        Index([1, 2]),
        RangeIndex(2),
        DatetimeIndex([Timestamp("2019-12-31"), Timestamp("2020-12-31")]),
        PeriodIndex([Period("2019-12-31"), Period("2020-12-31")]),
        TimedeltaIndex([Timedelta("1 days"), Timedelta("2 days")]),
    ],
)
def test_series_from_index(idx):
    # [FUNCTION DOC]
    # Parametrized over several Index types (integer, range, datetime, period,
    # timedelta). A Series built from an Index shares memory and holds a reference;
    # writing to the Series leaves the Index and its data unchanged. Same with
    # explicit copy=False; copy=True gives an unshared, reference-free copy.
    ser = Series(idx)
    expected = idx.copy(deep=True)
    assert np.shares_memory(get_array(ser), get_array(idx))
    assert not ser._mgr._has_no_reference(0)
    ser.iloc[0] = ser.iloc[1]
    tm.assert_index_equal(idx, expected)
    tm.assert_numpy_array_equal(get_array(idx), get_array(expected))

    # forcing copy=False still gives a CoW shallow copy
    ser = Series(idx, copy=False)
    assert np.shares_memory(get_array(ser), get_array(idx))
    assert not ser._mgr._has_no_reference(0)
    ser.iloc[0] = ser.iloc[1]
    tm.assert_index_equal(idx, expected)
    tm.assert_numpy_array_equal(get_array(idx), get_array(expected))

    # forcing copy=True still results in a copy
    ser = Series(idx, copy=True)
    assert not np.shares_memory(get_array(ser), get_array(idx))
    assert ser._mgr._has_no_reference(0)


@pytest.mark.parametrize("copy", [True, False, None])
def test_series_from_index_different_dtypes(copy):
    # [FUNCTION DOC]
    # Parametrized over the copy argument used to build the Index. Converting an
    # int64 Index to int32 when building a Series produces unshared data with no
    # references.
    idx = Index([1, 2, 3], dtype="int64", copy=copy)
    ser = Series(idx, dtype="int32")
    assert not np.shares_memory(get_array(ser), get_array(idx))
    assert ser._mgr._has_no_reference(0)


def test_series_from_block_manager_different_dtype():
    # [FUNCTION DOC]
    # Constructing a Series from an internal single-block manager with a different
    # dtype emits a DeprecationWarning about passing a manager and yields data that
    # is not shared and has no references.
    ser = Series([1, 2, 3], dtype="int64")
    msg = "Passing a SingleBlockManager to Series"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser2 = Series(ser._mgr, dtype="int32")
    assert not np.shares_memory(get_array(ser), get_array(ser2))
    assert ser2._mgr._has_no_reference(0)


@pytest.mark.parametrize("use_mgr", [True, False])
@pytest.mark.parametrize("columns", [None, ["a"]])
def test_dataframe_constructor_mgr_or_df(columns, use_mgr):
    # [FUNCTION DOC]
    # Parametrized over whether the input is the frame's internal manager (which warns
    # with a DeprecationWarning about passing a manager) or the frame itself, and
    # over columns None or a subset. The new frame shares memory with the original;
    # writing to the new frame copies on write and leaves the original unchanged.
    df = DataFrame({"a": [1, 2, 3]})
    df_orig = df.copy()

    if use_mgr:
        data = df._mgr
        warn = DeprecationWarning
    else:
        data = df
        warn = None
    msg = "Passing a BlockManager to DataFrame"
    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
        new_df = DataFrame(data)

    assert np.shares_memory(get_array(df, "a"), get_array(new_df, "a"))
    new_df.iloc[0] = 100

    assert not np.shares_memory(get_array(df, "a"), get_array(new_df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("dtype", [None, "int64", "Int64"])
@pytest.mark.parametrize("index", [None, [0, 1, 2]])
@pytest.mark.parametrize("columns", [None, ["a", "b"], ["a", "b", "c"]])
def test_dataframe_from_dict_of_series(columns, index, dtype):
    # [FUNCTION DOC]
    # Parametrized over columns, index and dtype. With copy=False a frame built from
    # a dict of Series is a lazy view of the Series: memory is shared until the frame
    # or a parent Series is written to, after which the other side is unaffected and
    # the frame equals the expected frame built from plain lists.
    # Case: constructing a DataFrame from Series objects with copy=False
    # has to do a lazy following CoW rules
    # (the default for DataFrame(dict) is still to copy to ensure consolidation)
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    s1_orig = s1.copy()
    expected = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, index=index, columns=columns, dtype=dtype
    )

    result = DataFrame(
        {"a": s1, "b": s2}, index=index, columns=columns, dtype=dtype, copy=False
    )

    # the shallow copy still shares memory
    assert np.shares_memory(get_array(result, "a"), get_array(s1))

    # mutating the new dataframe doesn't mutate original
    result.iloc[0, 0] = 10
    assert not np.shares_memory(get_array(result, "a"), get_array(s1))
    tm.assert_series_equal(s1, s1_orig)

    # the same when modifying the parent series
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    result = DataFrame(
        {"a": s1, "b": s2}, index=index, columns=columns, dtype=dtype, copy=False
    )
    s1.iloc[0] = 10
    assert not np.shares_memory(get_array(result, "a"), get_array(s1))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, "int64"])
def test_dataframe_from_dict_of_series_with_reindex(dtype):
    # [FUNCTION DOC]
    # Parametrized over dtype. With copy=False and an index that forces a real
    # reindex, the frame owns its data: it does not share memory with the Series and
    # writing to it does not trigger a copy (the buffer is unchanged).
    # Case: constructing a DataFrame from Series objects with copy=False
    # and passing an index that requires an actual (no-view) reindex -> need
    # to ensure the result doesn't have refs set up to unnecessarily trigger
    # a copy on write
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    df = DataFrame({"a": s1, "b": s2}, index=[1, 2, 3], dtype=dtype, copy=False)

    # df should own its memory, so mutating shouldn't trigger a copy
    arr_before = get_array(df, "a")
    assert not np.shares_memory(arr_before, get_array(s1))
    df.iloc[0, 0] = 100
    arr_after = get_array(df, "a")
    assert np.shares_memory(arr_before, arr_after)


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, 2], "int64"),
        # 1D-only EA
        ([1, 2], "Int64"),
        pytest.param(
            ["a", "b"],
            "str",
            marks=pytest.mark.xfail(
                reason="TODO bug with infer_string=False and specifying dtype='str'"
            )
            if not using_string_dtype()
            else [],
        ),
        (["a", "b"], object),
        # 2D EA
        (
            [Timestamp("2020", tz="UTC"), Timestamp("2021", tz="UTC")],
            "datetime64[ns, UTC]",
        ),
    ],
    ids=["int", "int-ea", "str", "object", "datetime64tz"],
)
def test_dataframe_from_series_or_index(data, dtype, index_or_series):
    # [FUNCTION DOC]
    # Parametrized over (data, dtype) pairs: integer, nullable integer, string
    # (xfail when the string dtype option is off), object and tz-aware datetime;
    # fixture index_or_series chooses the container type.
    # DataFrame from the object (default, and with an identical dtype) shares memory
    # and keeps a reference; writes to the frame do not alter the source. With
    # copy=True the memory is not shared and no reference is kept, except for
    # pyarrow-backed string dtype where the check is skipped because deep copy still
    # points to the same data.
    obj = index_or_series(data, dtype=dtype)
    obj_orig = obj.copy(deep=True)  # deep=True needed for Index

    # default is copy=False -> DataFrame holds a shallow copy of original Index/Series
    df = DataFrame(obj)
    assert tm.shares_memory(get_array(obj), get_array(df, 0))
    assert not df._mgr._has_no_reference(0)

    df.iloc[0, 0] = data[-1]
    tm.assert_equal(obj, obj_orig)

    # with passing the (identical) dtype -> same
    df = DataFrame(obj, dtype=dtype)
    assert tm.shares_memory(get_array(obj), get_array(df, 0))
    assert not df._mgr._has_no_reference(0)

    df.iloc[0, 0] = data[-1]
    tm.assert_equal(obj, obj_orig)

    # forcing copy=True still results in an actual hard copy up front
    df = DataFrame(obj, copy=True)
    if not (obj.dtype == "str" and obj.dtype.storage == "pyarrow"):
        # ArrowExtensionArray deep copy still points to the same underlying data
        assert not tm.shares_memory(get_array(obj), get_array(df, 0))
        assert df._mgr._has_no_reference(0)

    df.iloc[0, 0] = data[-1]
    tm.assert_equal(obj, obj_orig)


def test_dataframe_from_series_or_index_different_dtype(index_or_series):
    # [FUNCTION DOC]
    # Fixture index_or_series gives the container type. Building a frame with a
    # different dtype (int32 from int64) copies: no sharing and no references.
    obj = index_or_series([1, 2], dtype="int64")
    df = DataFrame(obj, dtype="int32")
    assert not np.shares_memory(get_array(obj), get_array(df, 0))
    assert df._mgr._has_no_reference(0)


def test_dataframe_from_series_dont_infer_datetime():
    # [FUNCTION DOC]
    # A Series of Timestamp objects with object dtype must stay object dtype when
    # wrapped in a frame, still sharing memory and holding a reference.
    ser = Series([Timestamp("2019-12-31"), Timestamp("2020-12-31")], dtype=object)
    df = DataFrame(ser)
    assert df.dtypes.iloc[0] == np.dtype(object)
    assert np.shares_memory(get_array(ser), get_array(df, 0))
    assert not df._mgr._has_no_reference(0)


@pytest.mark.parametrize("index", [None, [0, 1, 2]])
def test_dataframe_from_dict_of_series_with_dtype(index):
    # [FUNCTION DOC]
    # Parametrized over index (none or explicit). With copy=False and a dtype that
    # causes a conversion to int64, the frame owns its data: not shared with the
    # source Series and mutation does not reallocate.
    # Variant of above, but now passing a dtype that causes a copy
    # -> need to ensure the result doesn't have refs set up to unnecessarily
    # trigger a copy on write
    s1 = Series([1.0, 2.0, 3.0])
    s2 = Series([4, 5, 6])
    df = DataFrame({"a": s1, "b": s2}, index=index, dtype="int64", copy=False)

    # df should own its memory, so mutating shouldn't trigger a copy
    arr_before = get_array(df, "a")
    assert not np.shares_memory(arr_before, get_array(s1))
    df.iloc[0, 0] = 100
    arr_after = get_array(df, "a")
    assert np.shares_memory(arr_before, arr_after)


@pytest.mark.parametrize("copy", [False, None, True])
def test_dataframe_from_numpy_array(copy):
    # [FUNCTION DOC]
    # Parametrized over copy (False, None, True). Checks whether the frame shares
    # memory with the source numpy array.
    # KNOWN ISSUE: the condition is redundant (the second operand is implied by the first), so the shared-memory branch is only taken for copy=False; copy=None is treated as copy=True.
    arr = np.array([[1, 2], [3, 4]])
    df = DataFrame(arr, copy=copy)

    if copy is not False or copy is True:
        assert not np.shares_memory(get_array(df, 0), arr)
    else:
        assert np.shares_memory(get_array(df, 0), arr)


@pytest.mark.parametrize(
    "data, dtype",
    [
        # 1D-only EA
        ([1, 2], "Int64"),
        # 2D EA
        (
            [Timestamp("2020", tz="UTC"), Timestamp("2021", tz="UTC")],
            "datetime64[ns, UTC]",
        ),
    ],
    ids=["int-ea", "datetime64tz"],
)
@pytest.mark.parametrize("copy", [False, None, True])
def test_dataframe_from_extension_array(copy, data, dtype):
    # [FUNCTION DOC]
    # Parametrized over copy and over nullable integer or tz-aware datetime arrays.
    # For copy None or True the frame must not share memory with the array; for False
    # it must. The nullable integer array is replaced by its backing numpy array
    # first so the memory-sharing helper works.
    arr = pd.array(data, dtype=dtype)
    df = DataFrame(arr, copy=copy)

    if arr.dtype == "Int64":
        # to ensure tm.shares_memory works correctly
        # TODO fix in tm.shares_memory or get_array?
        arr = arr._data

    if copy is None or copy is True:
        assert not tm.shares_memory(get_array(df, 0), arr)
    else:
        assert tm.shares_memory(get_array(df, 0), arr)


def test_frame_from_dict_of_index():
    # [FUNCTION DOC]
    # Frame built from a dict containing an Index with copy=False shares the Index's
    # values and holds a reference; writing to the frame leaves the Index unchanged.
    idx = Index([1, 2, 3])
    expected = idx.copy(deep=True)
    df = DataFrame({"a": idx}, copy=False)
    assert np.shares_memory(get_array(df, "a"), idx._values)
    assert not df._mgr._has_no_reference(0)

    df.iloc[0, 0] = 100
    tm.assert_index_equal(idx, expected)


def test_rangeindex_cached_data():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: a Series built from a RangeIndex shares the
    # index's lazily materialized cached data. After renaming the index (a view that
    # shares the cache) and deleting the original, writing to the Series must not
    # corrupt the cached data of the renamed index.
    # https://github.com/pandas-dev/pandas/issues/67055
    # creating a Series from a RangeIndex should still track a reference because
    # the RangeIndex caches the materialized _data used by the Series
    idx = RangeIndex(3)
    ser = Series(idx)

    # idx2 takes a view of idx -> shares the cache and tracks a reference
    idx2 = idx.rename("a")
    del idx
    ser.iloc[0] = 99

    tm.assert_numpy_array_equal(idx2._data, np.array([0, 1, 2], dtype="int64"))
