# [MODULE DOC]
# Copy-on-write tests for clip: in-place and non-in-place variants, with and
# without other references to the data, and the chained in-place case.

import numpy as np

from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def test_clip_inplace_reference():
    # [FUNCTION DOC]
    # In-place clip on a frame that has a view referencing its data, where values
    # actually change: the frame gets a fresh buffer (no longer shared with the old
    # one), neither frame nor view keeps references, and the view keeps old values.
    df = DataFrame({"a": [1.5, 2, 3]})
    df_copy = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]
    df.clip(lower=2, inplace=True)

    assert not np.shares_memory(get_array(df, "a"), arr_a)
    assert df._mgr._has_no_reference(0)
    assert view._mgr._has_no_reference(0)
    tm.assert_frame_equal(df_copy, view)


def test_clip_inplace_reference_no_op():
    # [FUNCTION DOC]
    # In-place clip where no value needs changing (lower bound below all values)
    # while a view exists: the original buffer is kept and both objects still
    # report references; the view is unchanged.
    df = DataFrame({"a": [1.5, 2, 3]})
    df_copy = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]
    df.clip(lower=0, inplace=True)

    assert np.shares_memory(get_array(df, "a"), arr_a)

    assert not df._mgr._has_no_reference(0)
    assert not view._mgr._has_no_reference(0)
    tm.assert_frame_equal(df_copy, view)


def test_clip_inplace():
    # [FUNCTION DOC]
    # In-place clip without other references: modifies the existing buffer, which
    # stays the same, and the block has no references.
    df = DataFrame({"a": [1.5, 2, 3]})
    arr_a = get_array(df, "a")
    df.clip(lower=2, inplace=True)

    assert np.shares_memory(get_array(df, "a"), arr_a)
    assert df._mgr._has_no_reference(0)


def test_clip():
    # [FUNCTION DOC]
    # Non-in-place clip that changes values returns a result that does not share
    # memory with the source; the source keeps no references and is unchanged.
    df = DataFrame({"a": [1.5, 2, 3]})
    df_orig = df.copy()
    df2 = df.clip(lower=2)

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    assert df._mgr._has_no_reference(0)
    tm.assert_frame_equal(df_orig, df)


def test_clip_no_op():
    # [FUNCTION DOC]
    # Non-in-place clip that changes nothing returns a lazy copy: the result shares
    # memory with the source and the source then holds a reference.
    df = DataFrame({"a": [1.5, 2, 3]})
    df2 = df.clip(lower=0)

    assert not df._mgr._has_no_reference(0)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_clip_chained_inplace():
    # [FUNCTION DOC]
    # In-place clip invoked through a chained selection (a column Series, then a
    # single-column frame) must emit the chained assignment warning and leave the
    # original frame unchanged.
    df = DataFrame({"a": [1, 4, 2], "b": 1})
    df_orig = df.copy()
    with tm.raises_chained_assignment_error():
        df["a"].clip(1, 2, inplace=True)
    tm.assert_frame_equal(df, df_orig)

    with tm.raises_chained_assignment_error():
        df[["a"]].clip(1, 2, inplace=True)
    tm.assert_frame_equal(df, df_orig)
