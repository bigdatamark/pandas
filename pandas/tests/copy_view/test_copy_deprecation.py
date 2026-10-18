# [MODULE DOC]
# Tests that passing the deprecated copy keyword to DataFrame/Series methods and
# to merge and concat emits a Pandas4Warning, and that setting the copy-on-write
# option emits a warning that it can no longer be disabled.

import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import (
    concat,
    merge,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "meth, kwargs",
    [
        ("truncate", {}),
        ("tz_convert", {"tz": "UTC"}),
        ("tz_localize", {"tz": "UTC"}),
        ("infer_objects", {}),
        ("astype", {"dtype": "float64"}),
        ("reindex", {"index": [2, 0, 1]}),
        ("transpose", {}),
        ("set_axis", {"labels": [1, 2, 3]}),
        ("rename", {"index": {1: 2}}),
        ("set_flags", {}),
        ("to_period", {}),
        ("to_timestamp", {}),
        ("swaplevel", {"i": 0, "j": 1}),
    ],
)
def test_copy_deprecation(meth, kwargs):
    # [FUNCTION DOC]
    # Parametrized over method name and keyword arguments (truncate, tz_convert,
    # tz_localize, infer_objects, astype, reindex, transpose, set_axis, rename,
    # set_flags, to_period, to_timestamp, swaplevel). The frame is prepared with a
    # suitable index (datetime, period or multi-level) per method. Each method called
    # with copy set to false on the frame (except swaplevel) and on a column
    # Series (except transpose) must emit a Pandas4Warning matching the word copy.
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": 1})

    if meth in ("tz_convert", "tz_localize", "to_period"):
        tz = None if meth in ("tz_localize", "to_period") else "US/Eastern"
        df.index = pd.date_range("2020-01-01", freq="D", periods=len(df), tz=tz)
    elif meth == "to_timestamp":
        df.index = pd.period_range("2020-01-01", freq="D", periods=len(df))
    elif meth == "swaplevel":
        df = df.set_index(["b", "c"])

    if meth != "swaplevel":
        with tm.assert_produces_warning(Pandas4Warning, match="copy"):
            getattr(df, meth)(copy=False, **kwargs)

    if meth != "transpose":
        with tm.assert_produces_warning(Pandas4Warning, match="copy"):
            getattr(df.a, meth)(copy=False, **kwargs)


def test_copy_deprecation_reindex_like_align():
    # [FUNCTION DOC]
    # Calling reindex_like and align with the copy keyword on frame and Series each
    # emit a Pandas4Warning. Stack level checks are disabled because the reported
    # stack level is known to be inaccurate here.
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # Somehow the stack level check is incorrect here
    with tm.assert_produces_warning(
        Pandas4Warning, match="copy", check_stacklevel=False
    ):
        df.reindex_like(df, copy=False)

    with tm.assert_produces_warning(
        Pandas4Warning, match="copy", check_stacklevel=False
    ):
        df.a.reindex_like(df.a, copy=False)

    with tm.assert_produces_warning(
        Pandas4Warning, match="copy", check_stacklevel=False
    ):
        df.align(df, copy=False)

    with tm.assert_produces_warning(
        Pandas4Warning, match="copy", check_stacklevel=False
    ):
        df.a.align(df.a, copy=False)


def test_copy_deprecation_merge_concat():
    # [FUNCTION DOC]
    # The copy keyword on the frame merge method, the top-level merge function and
    # concat each emit a Pandas4Warning (stack level not checked).
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with tm.assert_produces_warning(
        Pandas4Warning, match="copy", check_stacklevel=False
    ):
        df.merge(df, copy=False)

    with tm.assert_produces_warning(
        Pandas4Warning, match="copy", check_stacklevel=False
    ):
        merge(df, df, copy=False)

    with tm.assert_produces_warning(
        Pandas4Warning, match="copy", check_stacklevel=False
    ):
        concat([df, df], copy=False)


@pytest.mark.parametrize("value", [False, True, "warn"])
def test_copy_on_write_deprecation_option(value):
    # [FUNCTION DOC]
    # Parametrized over option values false, true and the string warn. Entering an
    # option context for the copy-on-write mode option emits a Pandas4Warning saying
    # it can no longer be disabled. Stack level is not checked since the warning
    # points into the context manager machinery.
    msg = "Copy-on-Write can no longer be disabled"
    # stacklevel points to contextlib due to use of context manager.
    with tm.assert_produces_warning(Pandas4Warning, match=msg, check_stacklevel=False):
        with pd.option_context("mode.copy_on_write", value):
            pass
