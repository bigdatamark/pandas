"""
Tests that apply to all groupby operation methods.

The only tests that should appear here are those that use the `groupby_func` fixture.
Even if it does use that fixture, prefer a more specific test file if it available
such as:

 - test_categorical
 - test_groupby_dropna
 - test_groupby_subclass
 - test_raises
"""

# [MODULE DOC]
# Tests that apply across every groupby reduction/transform method, driven by the groupby_func fixture (one test instance per method name). Uses the helper that builds suitable arguments for a given method name. Covers empty MultiIndex frames, duplicate column labels, output shape with duplicate labels, and a non-contiguous mask on nullable data. The corrwith method is deprecated and produces a Pandas4Warning, which tests expect or skip.
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


def test_multiindex_group_all_columns_when_empty(groupby_func):
    # [FUNCTION DOC]
    # GH 32464: for an empty frame indexed by a three-level index (a, b, c) with no columns left and grouped by all three levels with group_keys true, calling each method must return something whose index equals the original (empty) index. For corrwith, the deprecation warning is asserted.
    # GH 32464
    df = DataFrame({"a": [], "b": [], "c": []}).set_index(["a", "b", "c"])
    gb = df.groupby(["a", "b", "c"], group_keys=True)
    method = getattr(gb, groupby_func)
    args = get_groupby_method_args(groupby_func, df)
    if groupby_func == "corrwith":
        warn = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = method(*args).index
    expected = df.index
    tm.assert_index_equal(result, expected)


def test_duplicate_columns(request, groupby_func, as_index):
    # [FUNCTION DOC]
    # GH#50806: parametrized over every method and as_index. A frame with duplicate column label b is grouped by a; the result must equal that from a frame with unique renamed columns (b, c) with the last label renamed back to b, except for size, ngroup and cumcount where no column renaming applies. The corrwith case is marked as an expected failure (GH#50845).
    # GH#50806
    if groupby_func == "corrwith":
        msg = "GH#50845 - corrwith fails when there are duplicate columns"
        request.applymarker(pytest.mark.xfail(reason=msg))
    df = DataFrame([[1, 3, 6], [1, 4, 7], [2, 5, 8]], columns=list("abb"))
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby("a", as_index=as_index)
    result = getattr(gb, groupby_func)(*args)

    expected_df = df.set_axis(["a", "b", "c"], axis=1)
    expected_args = get_groupby_method_args(groupby_func, expected_df)
    expected_gb = expected_df.groupby("a", as_index=as_index)
    expected = getattr(expected_gb, groupby_func)(*expected_args)
    if groupby_func not in ("size", "ngroup", "cumcount"):
        expected = expected.rename(columns={"c": "b"})
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index(["a", "a"], name="foo"),
        pd.MultiIndex.from_tuples((("a", "a"), ("a", "a")), names=["foo", "bar"]),
    ],
)
def test_dup_labels_output_shape(groupby_func, idx):
    # [FUNCTION DOC]
    # Parametrized over each method and two column index kinds (flat with duplicate labels, or a two-level MultiIndex with duplicates). Skips size, ngroup and cumcount. A one-row frame grouped by column label 0 must give a result of shape one by two with columns identical to the original column index. The corrwith deprecation warning is asserted.
    if groupby_func in {"size", "ngroup", "cumcount"}:
        pytest.skip(f"Not applicable for {groupby_func}")

    df = DataFrame([[1, 1]], columns=idx)
    grp_by = df.groupby([0])

    args = get_groupby_method_args(groupby_func, df)
    if groupby_func == "corrwith":
        warn = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = getattr(grp_by, groupby_func)(*args)

    assert result.shape == (1, 2)
    tm.assert_index_equal(result.columns, idx)


def test_not_c_contiguous_mask(groupby_func):
    # [FUNCTION DOC]
    # Issue 61031: builds a nullable Int64 frame as a reversed view so its mask is not C-contiguous (asserted), then every method on the reversed grouped frame must equal the method on the equivalent contiguous frame. Returns immediately for the deprecated corrwith. The variable named reversed shadows a builtin name locally.
    # https://github.com/pandas-dev/pandas/issues/61031
    if groupby_func == "corrwith":
        # corrwith is deprecated
        return
    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]}, dtype="Int64")
    reversed = DataFrame(
        {"a": [2, 1, 1], "b": [5, 4, 3]}, dtype="Int64", index=[2, 1, 0]
    )[::-1]
    assert not reversed["b"].array._mask.flags["C_CONTIGUOUS"]
    args = get_groupby_method_args(groupby_func, df)

    gb_reversed = reversed.groupby("a")
    result = getattr(gb_reversed, groupby_func)(*args)
    gb = df.groupby("a")
    expected = getattr(gb, groupby_func)(*args)
    tm.assert_equal(result, expected)
