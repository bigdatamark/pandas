# [MODULE DOC]
# Tests for DataFrame.value_counts: ordering, normalization, missing-value handling, subsets,
# empty frames and categorical columns. All tests are module-level functions; most use a small
# frame of animal leg and wing counts (falcon, dog, cat, ant). Result is a Series named count
# (or proportion when normalized) indexed by a multi-level index of the unique rows.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_data_frame_value_counts_unsorted():
    # [FUNCTION DOC]
    # Verifies sort turned off: the counts are returned in order of first appearance of each unique row (1, 2, 1).
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(sort=False)
    expected = pd.Series(
        data=[1, 2, 1],
        index=pd.MultiIndex.from_arrays(
            [(2, 4, 6), (2, 0, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_ascending():
    # [FUNCTION DOC]
    # Verifies ascending order: smallest counts first (1, 1, 2), ties in order of first appearance.
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(ascending=True)
    expected = pd.Series(
        data=[1, 1, 2],
        index=pd.MultiIndex.from_arrays(
            [(2, 6, 4), (2, 0, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_default():
    # [FUNCTION DOC]
    # Verifies the default: descending by count (2, 1, 1) with the most common row first.
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts()
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.MultiIndex.from_arrays(
            [(4, 2, 6), (0, 2, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_normalize():
    # [FUNCTION DOC]
    # Verifies normalization gives relative frequencies 0.5, 0.25, 0.25 and renames the result to proportion.
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(normalize=True)
    expected = pd.Series(
        data=[0.5, 0.25, 0.25],
        index=pd.MultiIndex.from_arrays(
            [(4, 2, 6), (0, 2, 0)], names=["num_legs", "num_wings"]
        ),
        name="proportion",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_single_col_default():
    # [FUNCTION DOC]
    # A one-column frame still gives a result indexed by a one-level multi-index named after the column.
    df = pd.DataFrame({"num_legs": [2, 4, 4, 6]})

    result = df.value_counts()
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.MultiIndex.from_arrays([[4, 2, 6]], names=["num_legs"]),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_empty():
    # [FUNCTION DOC]
    # A frame without columns gives an empty integer series named count with an empty platform-integer index.
    df_no_cols = pd.DataFrame()

    result = df_no_cols.value_counts()
    expected = pd.Series(
        [], dtype=np.int64, name="count", index=np.array([], dtype=np.intp)
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_empty_normalize():
    # [FUNCTION DOC]
    # Same as the empty-frame test but normalized: empty float series named proportion with an empty platform-integer index.
    df_no_cols = pd.DataFrame()

    result = df_no_cols.value_counts(normalize=True)
    expected = pd.Series(
        [], dtype=np.float64, name="proportion", index=np.array([], dtype=np.intp)
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_dropna_true(nulls_fixture):
    # [FUNCTION DOC]
    # Regression test (GH 41334). Parametrized over the null-value fixture placed in a string column. With the default
    # of dropping missing values, rows containing a null are excluded; only the two complete rows are counted.
    # GH 41334
    df = pd.DataFrame(
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )
    result = df.value_counts()
    expected = pd.Series(
        data=[1, 1],
        index=pd.MultiIndex.from_arrays(
            [("John", "Beth"), ("Smith", "Louise")], names=["first_name", "middle_name"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_dropna_false(nulls_fixture):
    # [FUNCTION DOC]
    # Regression test (GH 41334). With dropna off, rows with a null are included, giving four rows of count one; the index levels
    # are sorted and the null appears as a level value (compared as a missing marker whatever null fixture was used).
    # GH 41334
    df = pd.DataFrame(
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )

    result = df.value_counts(dropna=False)
    expected = pd.Series(
        data=[1, 1, 1, 1],
        index=pd.MultiIndex(
            levels=[
                pd.Index(["Anne", "Beth", "John"]),
                pd.Index(["Louise", "Smith", np.nan]),
            ],
            codes=[[2, 0, 2, 1], [1, 2, 2, 0]],
            names=["first_name", "middle_name"],
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", (["first_name", "middle_name"], [0, 1]))
def test_data_frame_value_counts_subset(nulls_fixture, columns):
    # [FUNCTION DOC]
    # Regression test (GH 50829). Parametrized over string and integer column labels, with null values in the second column.
    # Counting only the first column (given as a label subset) gives counts John 2, Anne 1, Beth 1 with a flat index named by the column.
    # GH 50829
    df = pd.DataFrame(
        {
            columns[0]: ["John", "Anne", "John", "Beth"],
            columns[1]: ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )
    result = df.value_counts(columns[0])
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.Index(["John", "Anne", "Beth"], name=columns[0]),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_value_counts_categorical_future_warning():
    # [FUNCTION DOC]
    # Regression test (GH 54775). Counting a frame with one categorical column yields count 1 for each category, indexed by a multi-index
    # with a categorical level.
    # KNOWN ISSUE: the name mentions a future warning but no warning is expected or checked.
    # GH#54775
    df = pd.DataFrame({"a": [1, 2, 3]}, dtype="category")
    result = df.value_counts()
    expected = pd.Series(
        1,
        index=pd.MultiIndex.from_arrays(
            [pd.Index([1, 2, 3], name="a", dtype="category")]
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_value_counts_with_missing_category():
    # [FUNCTION DOC]
    # Regression test (GH 54836). With a categorical column having an unused category, the unused category appears in the result with count zero, after the observed ones.
    # GH-54836
    df = pd.DataFrame({"a": pd.Categorical([1, 2, 4], categories=[1, 2, 3, 4])})
    result = df.value_counts()
    expected = pd.Series(
        [1, 1, 1, 0],
        index=pd.MultiIndex.from_arrays(
            [pd.CategoricalIndex([1, 2, 4, 3], categories=[1, 2, 3, 4], name="a")]
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)
