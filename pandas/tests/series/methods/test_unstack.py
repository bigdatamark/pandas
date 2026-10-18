# [MODULE DOC]
# Tests for the Series method that pivots an inner index level into columns, producing a DataFrame: dtype preservation, missing labels, tuple and mixed-type level names, categorical values.
# Module-level test functions only; no import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


def test_unstack_preserves_object():
    # [FUNCTION DOC]
    # Verifies that unstacking an object-dtype Series with a two-level index keeps all resulting columns object dtype, both for the default (last) level and for level zero.
    mi = MultiIndex.from_product([["bar", "foo"], ["one", "two"]])

    ser = Series(np.arange(4.0), index=mi, dtype=object)

    res1 = ser.unstack()
    assert (res1.dtypes == object).all()

    res2 = ser.unstack(level=0)
    assert (res2.dtypes == object).all()


def test_unstack():
    # [FUNCTION DOC]
    # Verifies unstacking in several situations. (1) A two-level index with unsorted codes gives a frame with sorted row labels and columns, with missing cells as NaN; unstacking level zero gives the transpose. (2) A three-level index with a single outer label unstacked at level zero equals a frame built from the values with a two-level index, after sorting. (3) Issue 5873: an index containing NaN in a float level places NaN as a column label. (4) A three-level index with NaN in the innermost level, unstacked at level zero, gives rows indexed by the remaining (letter, number) pairs including NaN.
    index = MultiIndex(
        levels=[["bar", "foo"], ["one", "three", "two"]],
        codes=[[1, 1, 0, 0], [0, 1, 0, 2]],
    )

    s = Series(np.arange(4.0), index=index)
    unstacked = s.unstack()

    expected = DataFrame(
        [[2.0, np.nan, 3.0], [0.0, 1.0, np.nan]],
        index=["bar", "foo"],
        columns=["one", "three", "two"],
    )

    tm.assert_frame_equal(unstacked, expected)

    unstacked = s.unstack(level=0)
    tm.assert_frame_equal(unstacked, expected.T)

    index = MultiIndex(
        levels=[["bar"], ["one", "two", "three"], [0, 1]],
        codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
    )
    s = Series(np.random.default_rng(2).standard_normal(6), index=index)
    exp_index = MultiIndex(
        levels=[["one", "two", "three"], [0, 1]],
        codes=[[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
    )
    expected = DataFrame({"bar": s.values}, index=exp_index).sort_index(level=0)
    unstacked = s.unstack(0).sort_index()
    tm.assert_frame_equal(unstacked, expected)

    # GH5873
    idx = MultiIndex.from_arrays([[101, 102], [3.5, np.nan]])
    ts = Series([1, 2], index=idx)
    left = ts.unstack()
    right = DataFrame(
        [[np.nan, 1], [2, np.nan]], index=[101, 102], columns=[np.nan, 3.5]
    )
    tm.assert_frame_equal(left, right)

    idx = MultiIndex.from_arrays(
        [
            ["cat", "cat", "cat", "dog", "dog"],
            ["a", "a", "b", "a", "b"],
            [1, 2, 1, 1, np.nan],
        ]
    )
    ts = Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
    right = DataFrame(
        [[1.0, 1.3], [1.1, np.nan], [np.nan, 1.4], [1.2, np.nan]],
        columns=["cat", "dog"],
    )
    tpls = [("a", 1), ("a", 2), ("b", np.nan), ("b", 1)]
    right.index = MultiIndex.from_tuples(tpls)
    tm.assert_frame_equal(ts.unstack(level=0), right)


def test_unstack_tuplename_in_multiindex():
    # [FUNCTION DOC]
    # Verifies unstacking by a level whose name is a tuple (issue 19966): the tuple name is accepted as the level argument, columns become a one-level multi-index named by the tuple, and the remaining level becomes the row index named by its tuple name. All values are one.
    # GH 19966
    idx = MultiIndex.from_product(
        [["a", "b", "c"], [1, 2, 3]], names=[("A", "a"), ("B", "b")]
    )
    ser = Series(1, index=idx)
    result = ser.unstack(("A", "a"))

    expected = DataFrame(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        columns=MultiIndex.from_tuples([("a",), ("b",), ("c",)], names=[("A", "a")]),
        index=Index([1, 2, 3], name=("B", "b")),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "unstack_idx, expected_values, expected_index, expected_columns",
    [
        (
            ("A", "a"),
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=["B", "C"]),
            MultiIndex.from_tuples([("a",), ("b",)], names=[("A", "a")]),
        ),
        (
            (("A", "a"), "B"),
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            Index([3, 4], name="C"),
            MultiIndex.from_tuples(
                [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=[("A", "a"), "B"]
            ),
        ),
    ],
)
def test_unstack_mixed_type_name_in_multiindex(
    unstack_idx, expected_values, expected_index, expected_columns
):
    # [FUNCTION DOC]
    # Verifies unstacking with level names of mixed types in a three-level index (issue 19966). Parametrized over the level selection (a single tuple-named level, or a tuple-named plus a string-named level), expected values, row index and column index. The result is compared to a frame built from these expectations.
    # GH 19966
    idx = MultiIndex.from_product(
        [["a", "b"], [1, 2], [3, 4]], names=[("A", "a"), "B", "C"]
    )
    ser = Series(1, index=idx)
    result = ser.unstack(unstack_idx)

    expected = DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    tm.assert_frame_equal(result, expected)


def test_unstack_multi_index_categorical_values():
    # [FUNCTION DOC]
    # Verifies that unstacking a categorical-dtype Series with a two-level index (dates by letters, from a stacked random frame with levels renamed major and minor) yields columns that are each categorical with the single category, columns named by the minor level and rows named by the major level.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    mi = df.stack().index.rename(["major", "minor"])
    ser = Series(["foo"] * len(mi), index=mi, name="category", dtype="category")

    result = ser.unstack()

    dti = ser.index.levels[0]
    c = pd.Categorical(["foo"] * len(dti))
    expected = DataFrame(
        {"A": c, "B": c, "C": c, "D": c},
        columns=Index(list("ABCD"), name="minor"),
        index=dti.rename("major"),
    )
    tm.assert_frame_equal(result, expected)


def test_unstack_mixed_level_names():
    # [FUNCTION DOC]
    # Verifies unstacking by level name when level names mix strings and an integer (issue 48763): unstacking the level named x moves it to columns, and the remaining levels keep their names (the integer 0 and the string y).
    # GH#48763
    arrays = [["a", "a"], [1, 2], ["red", "blue"]]
    idx = MultiIndex.from_arrays(arrays, names=("x", 0, "y"))
    ser = Series([1, 2], index=idx)
    result = ser.unstack("x")
    expected = DataFrame(
        [[1], [2]],
        columns=Index(["a"], name="x"),
        index=MultiIndex.from_tuples([(1, "red"), (2, "blue")], names=[0, "y"]),
    )
    tm.assert_frame_equal(result, expected)
