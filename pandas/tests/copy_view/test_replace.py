# [MODULE DOC]
# Copy-on-write tests for replace (also where and mask in place): unchanged
# columns are shared lazily, only modified columns are copied, in-place replace
# reuses the buffer unless other references exist, and categorical, regex, list
# and dictionary forms follow the same rules. Uses get_array from the sibling
# util module.

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


@pytest.mark.parametrize(
    "replace_kwargs",
    [
        {"to_replace": {"a": 1, "b": 4}, "value": -1},
        # Test CoW splits blocks to avoid copying unchanged columns
        {"to_replace": {"a": 1}, "value": -1},
        {"to_replace": {"b": 4}, "value": -1},
        {"to_replace": {"b": {4: 1}}},
        # TODO: Add these in a further optimization
        # We would need to see which columns got replaced in the mask
        # which could be expensive
        # {"to_replace": {"b": 1}},
        # 1
    ],
)
def test_replace(replace_kwargs):
    # [FUNCTION DOC]
    # Parametrized over several mapping-style replace argument sets (replace within
    # column a, column b, both, or nested mapping). The result shares memory for
    # unchanged column b when its values are equal, and always for untouched column
    # c; writing to column c copies on write; if column a was a target, writing to
    # it afterwards does not copy again. Original equals its saved copy.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    df_replaced = df.replace(**replace_kwargs)

    if (df_replaced["b"] == df["b"]).all():
        assert np.shares_memory(get_array(df_replaced, "b"), get_array(df, "b"))
    assert tm.shares_memory(get_array(df_replaced, "c"), get_array(df, "c"))

    # mutating squeezed df triggers a copy-on-write for that column/block
    df_replaced.loc[0, "c"] = -1
    assert not np.shares_memory(get_array(df_replaced, "c"), get_array(df, "c"))

    if "a" in replace_kwargs["to_replace"]:
        arr = get_array(df_replaced, "a")
        df_replaced.loc[0, "a"] = 100
        assert np.shares_memory(get_array(df_replaced, "a"), arr)
    tm.assert_frame_equal(df, df_orig)


def test_replace_regex_inplace_refs():
    # [FUNCTION DOC]
    # In-place regex replace on a string column with a view alive: the column gets
    # a new buffer, the frame has no references and the view is unchanged.
    df = DataFrame({"a": ["aaa", "bbb"]})
    df_orig = df.copy()
    view = df[:]
    arr = get_array(df, "a")
    df.replace(to_replace=r"^a.*$", value="new", inplace=True, regex=True)
    assert not np.shares_memory(arr, get_array(df, "a"))
    assert df._mgr._has_no_reference(0)
    tm.assert_frame_equal(view, df_orig)


def test_replace_regex_inplace():
    # [FUNCTION DOC]
    # In-place regex replace without references reuses the buffer and leaves no
    # references. A non-inplace regex replace that matches returns a result that does not
    # share memory and leaves the original unchanged.
    df = DataFrame({"a": ["aaa", "bbb"]})
    arr = get_array(df, "a")
    df.replace(to_replace=r"^a.*$", value="new", inplace=True, regex=True)
    assert df._mgr._has_no_reference(0)
    assert tm.shares_memory(arr, get_array(df, "a"))

    df_orig = df.copy()
    df2 = df.replace(to_replace=r"^b.*$", value="new", regex=True)
    tm.assert_frame_equal(df_orig, df)
    assert not tm.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_replace_regex_inplace_no_op():
    # [FUNCTION DOC]
    # Regex replace on an integer column (nothing matches): in-place keeps buffer
    # and no references; non-inplace returns a lazy copy that shares memory and
    # leaves the original unchanged.
    df = DataFrame({"a": [1, 2]})
    arr = get_array(df, "a")
    df.replace(to_replace=r"^a.$", value="new", inplace=True, regex=True)
    assert df._mgr._has_no_reference(0)
    assert np.shares_memory(arr, get_array(df, "a"))

    df_orig = df.copy()
    df2 = df.replace(to_replace=r"^x.$", value="new", regex=True)
    tm.assert_frame_equal(df_orig, df)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_replace_mask_all_false_second_block():
    # [FUNCTION DOC]
    # Replace a float that occurs only in column a of a frame with two float
    # columns and two integer columns. Column a gets new data; the integer
    # columns (a different block with no match) stay shared with the original.
    # Writing to column c then copies just that column while column d keeps
    # sharing; the original is never modified. The unchanged float column b is
    # copied too, because blocks are not split here (see the comment in the test).
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5, "c": 1, "d": 2})
    df_orig = df.copy()

    df2 = df.replace(to_replace=1.5, value=55.5)

    # TODO: Block splitting would allow us to avoid copying b
    assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.loc[0, "c"] = 1
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged

    assert not np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    assert np.shares_memory(get_array(df, "d"), get_array(df2, "d"))


def test_replace_coerce_single_column():
    # [FUNCTION DOC]
    # Replacing a float with a string in one column forces a type change for it only;
    # the other column still shares memory. Writing to the other column later copies
    # it and leaves the original unchanged.
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5})
    df_orig = df.copy()

    df2 = df.replace(to_replace=1.5, value="a")
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.loc[0, "b"] = 0.5
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged
    assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))


def test_replace_to_replace_wrong_dtype():
    # [FUNCTION DOC]
    # Replacing a string in float-only data replaces nothing: all columns still share
    # memory with the result; writing to the result copies the touched column and
    # leaves the original unchanged.
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5})
    df_orig = df.copy()

    df2 = df.replace(to_replace="xxx", value=1.5)

    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.loc[0, "b"] = 0.5
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged
    assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))


def test_replace_list_categorical():
    # [FUNCTION DOC]
    # In-place list replace on a categorical column keeps the codes buffer and leaves no
    # references. A non-inplace replace and a category rename produce new codes
    # and the original is unchanged.
    df = DataFrame({"a": ["a", "b", "c"]}, dtype="category")
    arr = get_array(df, "a")

    df.replace(["c"], value="a", inplace=True)
    assert np.shares_memory(arr.codes, get_array(df, "a").codes)
    assert df._mgr._has_no_reference(0)

    df_orig = df.copy()
    df.replace(["b"], value="a")
    df2 = df.apply(lambda x: x.cat.rename_categories({"b": "d"}))
    assert not np.shares_memory(arr.codes, get_array(df2, "a").codes)

    tm.assert_frame_equal(df, df_orig)


def test_replace_list_inplace_refs_categorical():
    # [FUNCTION DOC]
    # In-place list replace on a categorical column with a view alive: the view must
    # remain equal to the earlier copy.
    df = DataFrame({"a": ["a", "b", "c"]}, dtype="category")
    view = df[:]
    df_orig = df.copy()
    df.replace(["c"], value="a", inplace=True)
    tm.assert_frame_equal(df_orig, view)


@pytest.mark.parametrize("to_replace", [1.5, [1.5], []])
def test_replace_inplace(to_replace):
    # [FUNCTION DOC]
    # Parametrized over to_replace (scalar, list, empty list), which is not used.
    # KNOWN ISSUE: the call always passes the fixed scalar value instead of the parametrized to_replace, so the list and empty-list variants are not exercised.
    # In-place replace of a float without references keeps the buffer and leaves no
    # references.
    df = DataFrame({"a": [1.5, 2, 3]})
    arr_a = get_array(df, "a")
    df.replace(to_replace=1.5, value=15.5, inplace=True)

    assert np.shares_memory(get_array(df, "a"), arr_a)
    assert df._mgr._has_no_reference(0)


@pytest.mark.parametrize("to_replace", [1.5, [1.5]])
def test_replace_inplace_reference(to_replace):
    # [FUNCTION DOC]
    # Parametrized over to_replace (scalar or list). In-place replace with a view
    # alive moves the column to a new buffer and drops references on both objects.
    df = DataFrame({"a": [1.5, 2, 3]})
    arr_a = get_array(df, "a")
    view = df[:]
    df.replace(to_replace=to_replace, value=15.5, inplace=True)

    assert not np.shares_memory(get_array(df, "a"), arr_a)
    assert df._mgr._has_no_reference(0)
    assert view._mgr._has_no_reference(0)


@pytest.mark.parametrize("to_replace", ["a", 100.5])
def test_replace_inplace_reference_no_op(to_replace):
    # [FUNCTION DOC]
    # Parametrized over to_replace values that match nothing (a string or an absent
    # float). In-place replace with a view alive keeps the buffer and both objects
    # still hold references.
    df = DataFrame({"a": [1.5, 2, 3]})
    arr_a = get_array(df, "a")
    view = df[:]
    df.replace(to_replace=to_replace, value=15.5, inplace=True)

    assert np.shares_memory(get_array(df, "a"), arr_a)
    assert not df._mgr._has_no_reference(0)
    assert not view._mgr._has_no_reference(0)


@pytest.mark.parametrize("to_replace", [1, [1]])
def test_replace_categorical_inplace_reference(to_replace):
    # [FUNCTION DOC]
    # Parametrized over scalar and list to_replace. In-place replace on a
    # categorical column with a view alive copies the codes, drops references on
    # both, and leaves the view unchanged.
    df = DataFrame({"a": Categorical([1, 2, 3])})
    df_orig = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]
    df.replace(to_replace=to_replace, value=1, inplace=True)
    assert not np.shares_memory(get_array(df, "a").codes, arr_a.codes)
    assert df._mgr._has_no_reference(0)
    assert view._mgr._has_no_reference(0)
    tm.assert_frame_equal(view, df_orig)


def test_replace_categorical_inplace():
    # [FUNCTION DOC]
    # In-place replacement of a category with itself keeps the codes buffer,
    # leaves no references and the data equal the original categorical frame.
    df = DataFrame({"a": Categorical([1, 2, 3])})
    arr_a = get_array(df, "a")
    df.replace(to_replace=1, value=1, inplace=True)

    assert np.shares_memory(get_array(df, "a").codes, arr_a.codes)
    assert df._mgr._has_no_reference(0)

    expected = DataFrame({"a": Categorical([1, 2, 3])})
    tm.assert_frame_equal(df, expected)


def test_replace_categorical():
    # [FUNCTION DOC]
    # Non-inplace replace of a category with itself returns a copy: no references on
    # either, codes not shared and original unchanged. After setting a valid value
    # in the result, the codes buffer is not copied again.
    df = DataFrame({"a": Categorical([1, 2, 3])})
    df_orig = df.copy()
    df2 = df.replace(to_replace=1, value=1)

    assert df._mgr._has_no_reference(0)
    assert df2._mgr._has_no_reference(0)
    assert not np.shares_memory(get_array(df, "a").codes, get_array(df2, "a").codes)
    tm.assert_frame_equal(df, df_orig)

    arr_a = get_array(df2, "a").codes
    df2.iloc[0, 0] = 2.0
    assert np.shares_memory(get_array(df2, "a").codes, arr_a)


@pytest.mark.parametrize("method", ["where", "mask"])
def test_masking_inplace(method):
    # [FUNCTION DOC]
    # Parametrized over where and mask. In-place masking with a view alive moves the
    # column to a new buffer, drops references on both objects and leaves the view
    # unchanged.
    df = DataFrame({"a": [1.5, 2, 3]})
    df_orig = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]

    method = getattr(df, method)
    method(df["a"] > 1.6, -1, inplace=True)

    assert not np.shares_memory(get_array(df, "a"), arr_a)
    assert df._mgr._has_no_reference(0)
    assert view._mgr._has_no_reference(0)
    tm.assert_frame_equal(view, df_orig)


def test_replace_empty_list():
    # [FUNCTION DOC]
    # Replacing an empty list with an empty list is a no-op: the result shares
    # memory and the original holds a reference; in-place version keeps the buffer
    # and references on both frames.
    df = DataFrame({"a": [1, 2]})

    df2 = df.replace([], [])
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert not df._mgr._has_no_reference(0)
    arr_a = get_array(df, "a")
    df.replace([], [])
    assert np.shares_memory(get_array(df, "a"), arr_a)
    assert not df._mgr._has_no_reference(0)
    assert not df2._mgr._has_no_reference(0)


@pytest.mark.parametrize("value", ["d", None])
def test_replace_object_list_inplace(value):
    # [FUNCTION DOC]
    # Parametrized over replacement value (a string or none). In-place list replace
    # on an object column keeps the buffer and leaves no references.
    df = DataFrame({"a": ["a", "b", "c"]}, dtype=object)
    arr = get_array(df, "a")
    df.replace(["c"], value, inplace=True)
    assert np.shares_memory(arr, get_array(df, "a"))
    assert df._mgr._has_no_reference(0)


def test_replace_list_multiple_elements_inplace():
    # [FUNCTION DOC]
    # In-place replace of two values by a single one keeps the buffer and no
    # references.
    df = DataFrame({"a": [1, 2, 3]})
    arr = get_array(df, "a")
    df.replace([1, 2], 4, inplace=True)
    assert np.shares_memory(arr, get_array(df, "a"))
    assert df._mgr._has_no_reference(0)


def test_replace_list_none():
    # [FUNCTION DOC]
    # Replacing a listed value by none returns a result not sharing memory with the
    # original (unchanged). Replacing several absent values by none returns an equal
    # frame that still shares memory (regression test for a GitHub issue).
    df = DataFrame({"a": ["a", "b", "c"]})

    df_orig = df.copy()
    df2 = df.replace(["b"], value=None)
    tm.assert_frame_equal(df, df_orig)

    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # replace multiple values that don't actually replace anything with None
    # https://github.com/pandas-dev/pandas/issues/59770
    df3 = df.replace(["d", "e", "f"], value=None)
    tm.assert_frame_equal(df3, df_orig)
    assert tm.shares_memory(get_array(df, "a"), get_array(df3, "a"))


def test_replace_list_none_inplace_refs():
    # [FUNCTION DOC]
    # In-place replace by none with a view alive: new buffer, no references on the
    # frame, view unchanged.
    df = DataFrame({"a": ["a", "b", "c"]})
    arr = get_array(df, "a")
    df_orig = df.copy()
    view = df[:]
    df.replace(["a"], value=None, inplace=True)
    assert df._mgr._has_no_reference(0)
    assert not np.shares_memory(arr, get_array(df, "a"))
    tm.assert_frame_equal(df_orig, view)


def test_replace_columnwise_no_op_inplace():
    # [FUNCTION DOC]
    # In-place mapping replace that matches nothing keeps sharing memory with a view;
    # writing afterwards copies on write and the view stays unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    view = df[:]
    df_orig = df.copy()
    df.replace({"a": 10}, 100, inplace=True)
    assert np.shares_memory(get_array(view, "a"), get_array(df, "a"))
    df.iloc[0, 0] = 100
    tm.assert_frame_equal(view, df_orig)


def test_replace_columnwise_no_op():
    # [FUNCTION DOC]
    # Non-inplace mapping replace that matches nothing returns a result sharing
    # memory with the original; writing to it leaves the original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    df_orig = df.copy()
    df2 = df.replace({"a": 10}, 100)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    df2.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df_orig)


def test_replace_chained_assignment():
    # [FUNCTION DOC]
    # In-place replace through a chained selection (column Series, then
    # single-column frame) emits the chained assignment warning and leaves the frame
    # unchanged.
    df = DataFrame({"a": [1, np.nan, 2], "b": 1})
    df_orig = df.copy()
    with tm.raises_chained_assignment_error():
        df["a"].replace(1, 100, inplace=True)
    tm.assert_frame_equal(df, df_orig)

    with tm.raises_chained_assignment_error():
        df[["a"]].replace(1, 100, inplace=True)
    tm.assert_frame_equal(df, df_orig)


def test_replace_listlike():
    # [FUNCTION DOC]
    # List-to-list replace: no matches returns a lazy copy sharing memory; a match in
    # column a copies it.
    # KNOWN ISSUE: after writing to the result, the check compares the original frame with itself, so it asserts nothing; only the later comparison with the saved copy is meaningful.
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    df_orig = df.copy()

    result = df.replace([200, 201], [11, 11])
    assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))

    result.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df)

    result = df.replace([200, 2], [10, 10])
    assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_replace_listlike_inplace():
    # [FUNCTION DOC]
    # In-place list-to-list replace: with a match the buffer is kept (no references).
    # After taking a view, an in-place replace that matches copies to a new buffer
    # and the view stays unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    arr = get_array(df, "a")
    df.replace([200, 2], [10, 11], inplace=True)
    assert np.shares_memory(get_array(df, "a"), arr)

    view = df[:]
    df_orig = df.copy()
    df.replace([200, 3], [10, 11], inplace=True)
    assert not np.shares_memory(get_array(df, "a"), arr)
    tm.assert_frame_equal(view, df_orig)
