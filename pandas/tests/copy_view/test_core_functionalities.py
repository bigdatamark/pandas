# [MODULE DOC]
# Tests of core copy-on-write reference tracking in the block manager:
# reference removal after reassignment, splitting blocks in setitem, deletion
# and references that go out of scope.

import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def test_assigning_to_same_variable_removes_references():
    # [FUNCTION DOC]
    # Reassigning a frame to the result of resetting its index leaves the relevant
    # block without references, so a subsequent write stays in place (same buffer).
    df = DataFrame({"a": [1, 2, 3]})
    df = df.reset_index()
    assert df._mgr._has_no_reference(1)
    arr = get_array(df, "a")
    df.iloc[0, 1] = 100  # Write into a

    assert np.shares_memory(arr, get_array(df, "a"))


def test_setitem_dont_track_unnecessary_references():
    # [FUNCTION DOC]
    # Setting a column splits the block; the resulting blocks must not reference
    # each other, so a later write to another column does not copy the buffer.
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})

    df["b"] = 100
    arr = get_array(df, "a")
    # We split the block in setitem, if we are not careful the new blocks will
    # reference each other triggering a copy
    df.iloc[0, 0] = 100
    assert np.shares_memory(arr, get_array(df, "a"))


def test_setitem_with_view_copies():
    # [FUNCTION DOC]
    # With a live view of the frame, setting a column and then writing to another
    # column must copy on write (buffer no longer shared) and leave the view equal
    # to the earlier copy.
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})
    view = df[:]
    expected = df.copy()

    df["b"] = 100
    arr = get_array(df, "a")
    df.iloc[0, 0] = 100  # Check that we correctly track reference
    assert not np.shares_memory(arr, get_array(df, "a"))
    tm.assert_frame_equal(view, expected)


def test_setitem_with_view_invalidated_does_not_copy(request):
    # [FUNCTION DOC]
    # Takes the pytest request fixture. After the view is dropped, a write into a
    # column after a setitem-induced block split ought not to copy, but this is
    # marked as an expected failure because deleting a block does not track
    # references correctly. The assertion that buffers are shared is the expected
    # (currently failing) behavior.
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})
    view = df[:]

    df["b"] = 100
    arr = get_array(df, "a")
    view = None  # noqa: F841
    # TODO(CoW) block gets split because of `df["b"] = 100`
    # which introduces additional refs, even when those of `view` go out of scopes
    df.iloc[0, 0] = 100
    # Setitem split the block. Since the old block shared data with view
    # all the new blocks are referencing view and each other. When view
    # goes out of scope, they don't share data with any other block,
    # so we should not trigger a copy
    mark = pytest.mark.xfail(reason="blk.delete does not track references correctly")
    request.applymarker(mark)
    assert np.shares_memory(arr, get_array(df, "a"))


def test_out_of_scope():
    # [FUNCTION DOC]
    # HELPER: contains nested function func building a frame and returning a column
    # subset. After the frame goes out of scope, none of the result's blocks keep
    # references.
    def func():
        # [FUNCTION DOC]
        # HELPER: nested in test_out_of_scope; builds a three-column frame and returns a
        # two-column subset of it.
        df = DataFrame({"a": [1, 2], "b": 1.5, "c": 1})
        # create some subset
        result = df[["a", "b"]]
        return result

    result = func()
    assert not result._mgr.blocks[0].refs.has_reference()
    assert not result._mgr.blocks[1].refs.has_reference()


def test_delete():
    # [FUNCTION DOC]
    # Deleting a column from a frame, then selecting a column subset, leaves no
    # block with a reference.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["a", "b", "c"]
    )
    del df["b"]
    assert not df._mgr.blocks[0].refs.has_reference()
    assert not df._mgr.blocks[1].refs.has_reference()

    df = df[["a"]]
    assert not df._mgr.blocks[0].refs.has_reference()


def test_delete_reference():
    # [FUNCTION DOC]
    # With another object holding a slice of the frame, deleting a column leaves
    # both frame blocks and the slice block with references.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["a", "b", "c"]
    )
    x = df[:]
    del df["b"]
    assert df._mgr.blocks[0].refs.has_reference()
    assert df._mgr.blocks[1].refs.has_reference()
    assert x._mgr.blocks[0].refs.has_reference()
