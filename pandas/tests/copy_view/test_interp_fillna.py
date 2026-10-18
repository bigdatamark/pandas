# [MODULE DOC]
# Copy-on-write tests for interpolate, forward/backward fill and fillna on
# DataFrame and Series: no-op calls return lazy copies that share memory, calls
# that change data copy only what is needed, in-place variants either reuse the
# buffer (no other references) or copy and drop references (when a view exists),
# and chained in-place calls emit the chained assignment warning.
# Uses the get_array helper and fixtures from the wider test suite.

import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    Interval,
    NaT,
    Series,
    Timestamp,
    interval_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


@pytest.mark.parametrize("method", ["pad", "nearest", "linear"])
def test_interpolate_no_op(method):
    # [FUNCTION DOC]
    # Parametrized over method pad, nearest, linear. The pad method raises a value
    # error ("can not interpolate with method"). The others, on data with nothing to
    # fill, return a new frame (new index and columns objects) sharing memory with
    # the original; writing to the result copies on write and leaves the original
    # unchanged.
    df = DataFrame({"a": [1, 2]})
    df_orig = df.copy()

    if method == "pad":
        msg = f"Can not interpolate with method={method}"
        with pytest.raises(ValueError, match=msg):
            df.interpolate(method=method)
    else:
        result = df.interpolate(method=method)
        assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
        assert result.index is not df.index
        assert result.columns is not df.columns

        result.iloc[0, 0] = 100

        assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
        tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("func", ["ffill", "bfill"])
def test_interp_fill_functions(func):
    # [FUNCTION DOC]
    # Parametrized over forward fill and backward fill with nothing to fill. Result
    # is a new object (new index and columns) sharing memory; writing to it copies
    # and leaves the original unchanged.
    # Check that these takes the same code paths as interpolate
    df = DataFrame({"a": [1, 2]})
    df_orig = df.copy()

    result = getattr(df, func)()

    assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    assert result.index is not df.index
    assert result.columns is not df.columns

    result.iloc[0, 0] = 100
    assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("func", ["ffill", "bfill"])
@pytest.mark.parametrize(
    "vals", [[1, np.nan, 2], [Timestamp("2019-12-31"), NaT, Timestamp("2020-12-31")]]
)
def test_interpolate_triggers_copy(vals, func):
    # [FUNCTION DOC]
    # Parametrized over fill direction and over numeric or datetime data with a
    # missing value. Filling real gaps produces a result that does not share memory
    # and has no references.
    df = DataFrame({"a": vals})
    result = getattr(df, func)()

    assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    # Check that we don't have references when triggering a copy
    assert result._mgr._has_no_reference(0)


@pytest.mark.parametrize(
    "vals", [[1, np.nan, 2], [Timestamp("2019-12-31"), NaT, Timestamp("2020-12-31")]]
)
def test_interpolate_inplace_no_reference_no_copy(vals):
    # [FUNCTION DOC]
    # Parametrized over numeric or datetime data with a gap. In-place linear
    # interpolation on a frame without other references reuses the same buffer and
    # leaves no references.
    df = DataFrame({"a": vals})
    arr = get_array(df, "a")
    df.interpolate(method="linear", inplace=True)

    assert np.shares_memory(arr, get_array(df, "a"))
    # Check that we don't have references when triggering a copy
    assert df._mgr._has_no_reference(0)


@pytest.mark.parametrize(
    "vals", [[1, np.nan, 2], [Timestamp("2019-12-31"), NaT, Timestamp("2020-12-31")]]
)
def test_interpolate_inplace_with_refs(vals):
    # [FUNCTION DOC]
    # Takes a vals parameter (numeric or datetime) which is not used.
    # KNOWN ISSUE: the frame is always built from the fixed numeric list instead of the parametrized values, so the datetime case is not actually tested.
    # With a view alive, in-place linear interpolation copies: the frame gets a new
    # buffer, the view is unchanged, and neither holds references.
    df = DataFrame({"a": [1, np.nan, 2]})
    df_orig = df.copy()
    arr = get_array(df, "a")
    view = df[:]
    df.interpolate(method="linear", inplace=True)
    # Check that copy was triggered in interpolate and that we don't
    # have any references left
    assert not np.shares_memory(arr, get_array(df, "a"))
    tm.assert_frame_equal(df_orig, view)
    assert df._mgr._has_no_reference(0)
    assert view._mgr._has_no_reference(0)


@pytest.mark.parametrize("func", ["ffill", "bfill"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_interp_fill_functions_inplace(func, dtype):
    # [FUNCTION DOC]
    # Parametrized over fill direction and dtype (numpy float, nullable float). With a
    # view alive, in-place fill copies the data (new buffer), the view is unchanged,
    # and neither frame nor view keeps references.
    # Check that these takes the same code paths as interpolate
    df = DataFrame({"a": [1, np.nan, 2]}, dtype=dtype)
    df_orig = df.copy()
    arr = get_array(df, "a")
    view = df[:]

    getattr(df, func)(inplace=True)

    # Check that copy was triggered in interpolate and that we don't
    # have any references left
    assert not np.shares_memory(arr, get_array(df, "a"))
    tm.assert_frame_equal(df_orig, view)
    assert df._mgr._has_no_reference(0)
    assert view._mgr._has_no_reference(0)


def test_interpolate_cannot_with_object_dtype():
    # [FUNCTION DOC]
    # Interpolating a frame holding an object-dtype column raises a type error saying
    # a frame cannot interpolate with object dtype.
    df = DataFrame({"a": ["a", np.nan, "c"], "b": 1})
    df["a"] = df["a"].astype(object)

    msg = "DataFrame cannot interpolate with object dtype"
    with pytest.raises(TypeError, match=msg):
        df.interpolate()


def test_interpolate_object_convert_no_op():
    # [FUNCTION DOC]
    # A frame with an object column converted by cast; no interpolation is done. The
    # block has no references and the buffer is the same one fetched earlier.
    # KNOWN ISSUE: the test name and the inline comment about copy on write making a copy do not match the assertions, which verify that no copy and no references occur; also no interpolation is invoked at all.
    df = DataFrame({"a": ["a", "b", "c"], "b": 1})
    df["a"] = df["a"].astype(object)
    arr_a = get_array(df, "a")

    # Now CoW makes a copy, it should not!
    assert df._mgr._has_no_reference(0)
    assert np.shares_memory(arr_a, get_array(df, "a"))


def test_interpolate_object_convert_copies():
    # [FUNCTION DOC]
    # In-place interpolation with the pad method raises a value error ("can not
    # interpolate with method"), after which the block must still have no references
    # and share the original buffer.
    # KNOWN ISSUE: the name says the conversion copies, but the test asserts that nothing was copied (the error is raised before any work).
    df = DataFrame({"a": [1, np.nan, 2.5], "b": 1})
    arr_a = get_array(df, "a")
    msg = "Can not interpolate with method=pad"
    with pytest.raises(ValueError, match=msg):
        df.interpolate(method="pad", inplace=True)

    assert df._mgr._has_no_reference(0)
    assert np.shares_memory(arr_a, get_array(df, "a"))


def test_interpolate_downcast_reference_triggers_copy():
    # [FUNCTION DOC]
    # With a view alive, in-place interpolation using the pad method raises a value
    # error; the original and view must remain equal.
    # KNOWN ISSUE: the two assertions about references and memory sharing sit inside the block that expects the error, after the call that raises, so they are never executed.
    df = DataFrame({"a": [1, np.nan, 2.5], "b": 1})
    df_orig = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]

    msg = "Can not interpolate with method=pad"
    with pytest.raises(ValueError, match=msg):
        df.interpolate(method="pad", inplace=True)
        assert df._mgr._has_no_reference(0)
        assert not np.shares_memory(arr_a, get_array(df, "a"))

    tm.assert_frame_equal(df_orig, view)


def test_fillna():
    # [FUNCTION DOC]
    # fillna with a scalar returns a new frame (new index and columns) where the
    # column that needed no filling shares memory; writing to the result leaves the
    # original unchanged.
    df = DataFrame({"a": [1.5, np.nan], "b": 1})
    df_orig = df.copy()

    df2 = df.fillna(5.5)
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert df2.index is not df.index
    assert df2.columns is not df.columns

    df2.iloc[0, 1] = 100
    tm.assert_frame_equal(df_orig, df)


def test_fillna_dict():
    # [FUNCTION DOC]
    # fillna with a per-column mapping: the unmapped column shares memory with the
    # original, the filled column does not; writes to the result leave the original
    # unchanged.
    df = DataFrame({"a": [1.5, np.nan], "b": 1})
    df_orig = df.copy()

    df2 = df.fillna({"a": 100.5})
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.iloc[0, 1] = 100
    tm.assert_frame_equal(df_orig, df)


def test_fillna_inplace():
    # [FUNCTION DOC]
    # In-place fillna without other references keeps both buffers and leaves no
    # references.
    df = DataFrame({"a": [1.5, np.nan], "b": 1})
    arr_a = get_array(df, "a")
    arr_b = get_array(df, "b")

    df.fillna(5.5, inplace=True)
    assert np.shares_memory(get_array(df, "a"), arr_a)
    assert np.shares_memory(get_array(df, "b"), arr_b)
    assert df._mgr._has_no_reference(0)
    assert df._mgr._has_no_reference(1)


def test_fillna_inplace_reference():
    # [FUNCTION DOC]
    # In-place fillna with a view alive: the column that changed gets a new buffer,
    # the untouched column keeps sharing, the view has no references any more, the
    # view is unchanged and the frame equals the expected filled frame.
    df = DataFrame({"a": [1.5, np.nan], "b": 1})
    df_orig = df.copy()
    arr_a = get_array(df, "a")
    arr_b = get_array(df, "b")
    view = df[:]

    df.fillna(5.5, inplace=True)
    assert not np.shares_memory(get_array(df, "a"), arr_a)
    assert np.shares_memory(get_array(df, "b"), arr_b)
    assert view._mgr._has_no_reference(0)
    assert df._mgr._has_no_reference(0)
    tm.assert_frame_equal(view, df_orig)
    expected = DataFrame({"a": [1.5, 5.5], "b": 1})
    tm.assert_frame_equal(df, expected)


def test_fillna_interval_inplace_reference():
    # [FUNCTION DOC]
    # Series with interval dtype (explicit dtype to avoid implicit casts) containing a
    # missing value, with a view alive. In-place fillna with an interval value must
    # not share the left-endpoint buffers with the view, and the view is unchanged.
    # Set dtype explicitly to avoid implicit cast when setting nan
    ser = Series(
        interval_range(start=0, end=5), name="a", dtype="interval[float64, right]"
    )
    ser.iloc[1] = np.nan

    ser_orig = ser.copy()
    view = ser[:]
    ser.fillna(value=Interval(left=0, right=5), inplace=True)

    assert not np.shares_memory(
        get_array(ser, "a").left.values, get_array(view, "a").left.values
    )
    tm.assert_series_equal(view, ser_orig)


def test_fillna_series_empty_arg():
    # [FUNCTION DOC]
    # Series fillna with an empty mapping is a no-op returning a lazy copy sharing
    # memory; writing to the original afterwards does not change the result.
    ser = Series([1, np.nan, 2])
    ser_orig = ser.copy()
    result = ser.fillna({})
    assert np.shares_memory(get_array(ser), get_array(result))

    ser.iloc[0] = 100.5
    tm.assert_series_equal(ser_orig, result)


def test_fillna_series_empty_arg_inplace():
    # [FUNCTION DOC]
    # In-place Series fillna with an empty mapping keeps the same buffer and the
    # block has no references.
    ser = Series([1, np.nan, 2])
    arr = get_array(ser)
    ser.fillna({}, inplace=True)

    assert np.shares_memory(get_array(ser), arr)
    assert ser._mgr._has_no_reference(0)


def test_fillna_ea_noop_shares_memory(any_numeric_ea_and_arrow_dtype):
    # [FUNCTION DOC]
    # Fixture any_numeric_ea_and_arrow_dtype supplies nullable/arrow numeric dtypes.
    # fillna on a frame where one column has a missing value: the filled column does
    # not share memory, the other column does and holds a reference; later writes
    # to the shared column copy it and drop references; the original is unchanged.
    df = DataFrame({"a": [1, NA, 3], "b": 1}, dtype=any_numeric_ea_and_arrow_dtype)
    df_orig = df.copy()
    df2 = df.fillna(100)

    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert not df2._mgr._has_no_reference(1)
    tm.assert_frame_equal(df_orig, df)

    df2.iloc[0, 1] = 100
    assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert df2._mgr._has_no_reference(1)
    assert df._mgr._has_no_reference(1)
    tm.assert_frame_equal(df_orig, df)


def test_fillna_inplace_ea_noop_shares_memory(any_numeric_ea_and_arrow_dtype):
    # [FUNCTION DOC]
    # Same dtype fixture. In-place fillna with a view alive: the filled column no
    # longer shares memory with the view, the unfilled column still shares and both
    # keep references; after writing to the frame, the view is unchanged.
    df = DataFrame({"a": [1, NA, 3], "b": 1}, dtype=any_numeric_ea_and_arrow_dtype)
    df_orig = df.copy()
    view = df[:]
    df.fillna(100, inplace=True)
    assert not np.shares_memory(get_array(df, "a"), get_array(view, "a"))

    assert np.shares_memory(get_array(df, "b"), get_array(view, "b"))
    assert not df._mgr._has_no_reference(1)
    assert not view._mgr._has_no_reference(1)

    df.iloc[0, 1] = 100
    tm.assert_frame_equal(df_orig, view)


def test_fillna_chained_assignment():
    # [FUNCTION DOC]
    # In-place fillna through a chained selection (a column Series, then a
    # single-column frame) emits the chained assignment warning and leaves the frame
    # unchanged.
    df = DataFrame({"a": [1, np.nan, 2], "b": 1})
    df_orig = df.copy()
    with tm.raises_chained_assignment_error():
        df["a"].fillna(100, inplace=True)
    tm.assert_frame_equal(df, df_orig)

    with tm.raises_chained_assignment_error():
        df[["a"]].fillna(100, inplace=True)
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("func", ["interpolate", "ffill", "bfill"])
def test_interpolate_chained_assignment(func):
    # [FUNCTION DOC]
    # Parametrized over interpolate, forward fill and backward fill. Calling the
    # chosen in-place method through a chained selection (column Series, then
    # single-column frame) emits the chained assignment warning and leaves the frame
    # unchanged.
    df = DataFrame({"a": [1, np.nan, 2], "b": 1})
    df_orig = df.copy()
    with tm.raises_chained_assignment_error():
        getattr(df["a"], func)(inplace=True)
    tm.assert_frame_equal(df, df_orig)

    with tm.raises_chained_assignment_error():
        getattr(df[["a"]], func)(inplace=True)
    tm.assert_frame_equal(df, df_orig)
