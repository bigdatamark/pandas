# [MODULE DOC]
# Tests that groupby preserves pandas subclasses (the test subclassed frame and series from the testing module) and their custom metadata through grouping, group retrieval, aggregation, apply and resample.
# Module-level mark: ignores deprecation warnings about passing a block manager.
from datetime import datetime

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)


@pytest.mark.parametrize(
    "obj",
    [
        tm.SubclassedDataFrame({"A": np.arange(0, 10)}),
        tm.SubclassedSeries(np.arange(0, 10), name="A"),
    ],
)
def test_groupby_preserves_subclass(obj, groupby_func):
    # [FUNCTION DOC]
    # Issue 28330: for a subclassed frame and series grouped by a distinct key per row, parametrized over every groupby function, checks the retrieved group keeps the subclass type, the method result and the aggregate-by-name result are subclass instances (or the subclassed series for group number, cumulative count and size on frames) and equal each other. Skips the correlation-with-other function for series; expects a deprecation warning for it on frames.
    # GH28330 -- preserve subclass through groupby operations

    if isinstance(obj, Series) and groupby_func in {"corrwith"}:
        pytest.skip(f"Not applicable for Series and {groupby_func}")

    grouped = obj.groupby(np.arange(0, 10))

    # Groups should preserve subclass type
    assert isinstance(grouped.get_group(0), type(obj))

    args = get_groupby_method_args(groupby_func, obj)

    warn = Pandas4Warning if groupby_func == "corrwith" else None
    msg = f"{type(grouped).__name__}.corrwith is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        result1 = getattr(grouped, groupby_func)(*args)
    with tm.assert_produces_warning(warn, match=msg):
        result2 = grouped.agg(groupby_func, *args)

    # Reduction or transformation kernels should preserve type
    slices = {"ngroup", "cumcount", "size"}
    if isinstance(obj, DataFrame) and groupby_func in slices:
        assert isinstance(result1, tm.SubclassedSeries)
    else:
        assert isinstance(result1, type(obj))

    # Confirm .agg() groupby operations return same results
    if isinstance(result1, DataFrame):
        tm.assert_frame_equal(result1, result2)
    else:
        tm.assert_series_equal(result1, result2)


def test_groupby_preserves_metadata():
    # [FUNCTION DOC]
    # Issues 37343 and 45314: custom metadata attribute set on a subclassed frame is visible on each group when iterating and inside apply on frames, selected column lists and series, and the apply result is a subclassed series; also covers aggregate with a function on a subclassed series.
    # GH-37343
    custom_df = tm.SubclassedDataFrame({"a": [1, 2, 3], "b": [1, 1, 2], "c": [7, 8, 9]})
    assert "testattr" in custom_df._metadata
    custom_df.testattr = "hello"
    for _, group_df in custom_df.groupby("c"):
        assert group_df.testattr == "hello"

    # GH-45314
    def func(group):
        # [FUNCTION DOC]
        # HELPER: apply function asserting the group is a subclassed frame carrying the metadata attribute with the expected value, and returning that value.
        assert isinstance(group, tm.SubclassedDataFrame)
        assert hasattr(group, "testattr")
        assert group.testattr == "hello"
        return group.testattr

    result = custom_df.groupby("c").apply(func)
    expected = tm.SubclassedSeries(["hello"] * 3, index=Index([7, 8, 9], name="c"))
    tm.assert_series_equal(result, expected)

    result = custom_df.groupby("c").apply(func)
    tm.assert_series_equal(result, expected)

    # https://github.com/pandas-dev/pandas/pull/56761
    result = custom_df.groupby("c")[["a", "b"]].apply(func)
    tm.assert_series_equal(result, expected)

    def func2(group):
        # [FUNCTION DOC]
        # HELPER: same as the frame helper but for a subclassed series group.
        assert isinstance(group, tm.SubclassedSeries)
        assert hasattr(group, "testattr")
        return group.testattr

    custom_series = tm.SubclassedSeries([1, 2, 3])
    custom_series.testattr = "hello"
    result = custom_series.groupby(custom_df["c"]).apply(func2)
    tm.assert_series_equal(result, expected)
    result = custom_series.groupby(custom_df["c"]).agg(func2)
    tm.assert_series_equal(result, expected)


def test_groupby_apply_preserves_metadata():
    # [FUNCTION DOC]
    # Issue 62134: apply returning reductions of groups keeps the metadata attribute on the result for subclassed frames and series.
    # GH#62134 - Test that apply() preserves metadata when returning DataFrames/Series
    custom_df = tm.SubclassedDataFrame({"a": [1, 2, 3], "b": [1, 1, 2], "c": [7, 8, 9]})
    custom_df.testattr = "hello"

    def sum_func(group):
        # [FUNCTION DOC]
        # HELPER: asserts the group is a subclassed frame with the metadata and returns its sum.
        assert isinstance(group, tm.SubclassedDataFrame)
        assert hasattr(group, "testattr")
        assert group.testattr == "hello"
        return group.sum()

    result = custom_df.groupby("c").apply(sum_func)
    assert hasattr(result, "testattr"), "DataFrame apply() should preserve metadata"
    assert result.testattr == "hello"

    custom_series = tm.SubclassedSeries([1, 2, 3])
    custom_series.testattr = "hello"

    def sum_series_func(group):
        # [FUNCTION DOC]
        # HELPER: asserts the group is a subclassed series with the metadata and returns its sum.
        assert isinstance(group, tm.SubclassedSeries)
        assert hasattr(group, "testattr")
        assert group.testattr == "hello"
        return group.sum()

    result = custom_series.groupby(custom_df["c"]).apply(sum_series_func)
    assert hasattr(result, "testattr"), "Series apply() should preserve metadata"
    assert result.testattr == "hello"


@pytest.mark.parametrize("obj", [DataFrame, tm.SubclassedDataFrame])
def test_groupby_resample_preserves_subclass(obj):
    # [FUNCTION DOC]
    # Issue 28330: grouped resample with a five-day bin and sum preserves the frame type (plain or subclassed frame, parametrized).
    # GH28330 -- preserve subclass through groupby.resample()

    df = obj(
        {
            "Buyer": Series("Carl Carl Carl Carl Joe Carl".split(), dtype=object),
            "Quantity": [18, 3, 5, 1, 9, 3],
            "Date": [
                datetime(2013, 9, 1, 13, 0),
                datetime(2013, 9, 1, 13, 5),
                datetime(2013, 10, 1, 20, 0),
                datetime(2013, 10, 3, 10, 0),
                datetime(2013, 12, 2, 12, 0),
                datetime(2013, 9, 2, 14, 0),
            ],
        }
    )
    df = df.set_index("Date")

    # Confirm groupby.resample() preserves dataframe type
    result = df.groupby("Buyer").resample("5D").sum()
    assert isinstance(result, obj)
