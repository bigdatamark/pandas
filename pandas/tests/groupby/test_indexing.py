# Test GroupBy._positional_selector positional grouped indexing GH#42864
# [MODULE DOC]
# Tests for the private positional selector on groupby objects (positional grouped indexing, issue 42864): selecting rows by position within each group using integers, slices, lists and tuples, compared against iloc, head, tail and nth.
# Uses the shared sliced test frame fixtures (a frame and its grouped form) for the basic cases.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [0, [0, 1, 4]],
        [2, [5]],
        [5, []],
        [-1, [3, 4, 7]],
        [-2, [1, 6]],
        [-6, []],
    ],
)
def test_int(slice_test_df, slice_test_grouped, arg, expected_rows):
    # [FUNCTION DOC]
    # Single integer position within every group, including negative positions and positions beyond group sizes (empty result), parametrized over the argument and the expected row positions of the original frame.
    # Test single integer
    result = slice_test_grouped._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_slice(slice_test_df, slice_test_grouped):
    # [FUNCTION DOC]
    # A single slice with a step selects matching rows from every group in original order.
    # Test single slice
    result = slice_test_grouped._positional_selector[0:3:2]
    expected = slice_test_df.iloc[[0, 1, 4, 5]]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [[0, 2], [0, 1, 4, 5]],
        [[0, 2, -1], [0, 1, 3, 4, 5, 7]],
        [range(0, 3, 2), [0, 1, 4, 5]],
        [{0, 2}, [0, 1, 4, 5]],
    ],
    ids=[
        "list",
        "negative",
        "range",
        "set",
    ],
)
def test_list(slice_test_df, slice_test_grouped, arg, expected_rows):
    # [FUNCTION DOC]
    # Lists, ranges and sets of integer positions select the same rows (set order does not matter); includes a negative position.
    # Test lists of integers and integer valued iterables
    result = slice_test_grouped._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_ints(slice_test_df, slice_test_grouped):
    # [FUNCTION DOC]
    # A tuple of integers selects the union of the positions per group, in original row order.
    # Test tuple of ints
    result = slice_test_grouped._positional_selector[0, 2, -1]
    expected = slice_test_df.iloc[[0, 1, 3, 4, 5, 7]]

    tm.assert_frame_equal(result, expected)


def test_slices(slice_test_df, slice_test_grouped):
    # [FUNCTION DOC]
    # A tuple of slices selects the union of the slice selections per group.
    # Test tuple of slices
    result = slice_test_grouped._positional_selector[:2, -2:]
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]

    tm.assert_frame_equal(result, expected)


def test_mix(slice_test_df, slice_test_grouped):
    # [FUNCTION DOC]
    # A mixed tuple of integers and slices selects the union of the selections per group.
    # Test mixed tuple of ints and slices
    result = slice_test_grouped._positional_selector[0, 1, -2:]
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [0, [0, 1, 4]],
        [[0, 2, -1], [0, 1, 3, 4, 5, 7]],
        [(slice(None, 2), slice(-2, None)), [0, 1, 2, 3, 4, 6, 7]],
    ],
)
def test_as_index(slice_test_df, arg, expected_rows):
    # [FUNCTION DOC]
    # With the default as_index and an unsorted grouping, selection results keep the original frame's index and rows; parametrized over int, list and tuple-of-slices arguments.
    # Test the default as_index behaviour
    result = slice_test_df.groupby("Group", sort=False)._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_doc_examples():
    # [FUNCTION DOC]
    # Checks the examples from the documentation for as_index False grouping: a slice and a tuple of positions.
    # Test the examples in the documentation
    df = pd.DataFrame(
        [["a", 1], ["a", 2], ["a", 3], ["b", 4], ["b", 5]], columns=["A", "B"]
    )

    grouped = df.groupby("A", as_index=False)

    result = grouped._positional_selector[1:2]
    expected = pd.DataFrame([["a", 2], ["b", 5]], columns=["A", "B"], index=[1, 4])

    tm.assert_frame_equal(result, expected)

    result = grouped._positional_selector[1, -1]
    expected = pd.DataFrame(
        [["a", 2], ["a", 3], ["b", 5]], columns=["A", "B"], index=[1, 2, 4]
    )

    tm.assert_frame_equal(result, expected)


def test_multiindex():
    # [FUNCTION DOC]
    # Uses a multiindex frame (date and item) built from random data to check that slicing off the first and last three rows per date via nth equals the manually sliced data.
    # Test the multiindex mentioned as the use-case in the documentation

    def _make_df_from_data(data):
        # [FUNCTION DOC]
        # HELPER: builds a frame with a two-level index (date, item) and columns A and B from a mapping of dates to lists of item tuples.
        rows = {}
        for date in data:
            for level in data[date]:
                rows[(date, level[0])] = {"A": level[1], "B": level[2]}

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.names = ("Date", "Item")
        return df

    rng = np.random.default_rng(2)
    ndates = 100
    nitems = 20
    dates = pd.date_range("20130101", periods=ndates, freq="D")
    items = [f"item {i}" for i in range(nitems)]

    multiindex_data = {}
    for date in dates:
        nitems_for_date = nitems - rng.integers(0, 12)
        levels = [
            (item, rng.integers(0, 10000) / 100, rng.integers(0, 10000) / 100)
            for item in items[:nitems_for_date]
        ]
        levels.sort(key=lambda x: x[1])
        multiindex_data[date] = levels

    df = _make_df_from_data(multiindex_data)
    result = df.groupby("Date", as_index=False).nth(slice(3, -3))

    sliced = {date: values[3:-3] for date, values in multiindex_data.items()}
    expected = _make_df_from_data(sliced)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("arg", [1, 5, 30, 1000, -1, -5, -30, -1000])
@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("simulated", [True, False])
def test_against_head_and_tail(arg, method, simulated):
    # [FUNCTION DOC]
    # Checks the positional selector with head and tail style slices equals grouped head and tail, or an index list simulated by hand, over many sizes (including exceeding group size and negative), 100 groups of 30 rows.
    # Test gives the same results as grouped head and tail
    n_groups = 100
    n_rows_per_group = 30

    data = {
        "group": [
            f"group {g}" for j in range(n_rows_per_group) for g in range(n_groups)
        ],
        "value": [
            f"group {g} row {j}"
            for j in range(n_rows_per_group)
            for g in range(n_groups)
        ],
    }
    df = pd.DataFrame(data)
    grouped = df.groupby("group", as_index=False)
    size = arg if arg >= 0 else n_rows_per_group + arg

    if method == "head":
        result = grouped._positional_selector[:arg]

        if simulated:
            indices = [
                j * n_groups + i
                for j in range(size)
                for i in range(n_groups)
                if j * n_groups + i < n_groups * n_rows_per_group
            ]
            expected = df.iloc[indices]

        else:
            expected = grouped.head(arg)

    else:
        result = grouped._positional_selector[-arg:]

        if simulated:
            indices = [
                (n_rows_per_group + j - size) * n_groups + i
                for j in range(size)
                for i in range(n_groups)
                if (n_rows_per_group + j - size) * n_groups + i >= 0
            ]
            expected = df.iloc[indices]

        else:
            expected = grouped.tail(arg)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("start", [None, 0, 1, 10, -1, -10])
@pytest.mark.parametrize("stop", [None, 0, 1, 10, -1, -10])
@pytest.mark.parametrize("step", [None, 1, 5])
def test_against_df_iloc(start, stop, step):
    # [FUNCTION DOC]
    # With a single group, the positional selector equals plain iloc slicing for many start, stop and step combinations (including None and negative).
    # Test that a single group gives the same results as DataFrame.iloc
    n_rows = 30

    data = {
        "group": ["group 0"] * n_rows,
        "value": list(range(n_rows)),
    }
    df = pd.DataFrame(data)
    grouped = df.groupby("group", as_index=False)

    result = grouped._positional_selector[start:stop:step]
    expected = df.iloc[start:stop:step]

    tm.assert_frame_equal(result, expected)


def test_series():
    # [FUNCTION DOC]
    # Selection by slice on a series grouped by index level returns the second element of each group as a series.
    # Test grouped Series
    ser = pd.Series([1, 2, 3, 4, 5], index=["a", "a", "a", "b", "b"])
    grouped = ser.groupby(level=0)
    result = grouped._positional_selector[1:2]
    expected = pd.Series([2, 5], index=["a", "b"])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_step(step):
    # [FUNCTION DOC]
    # Slicing with a step of 1 to 5 over groups of unequal size (5, 4, 3 rows) selects every step-th row of each group, with the original index.
    # Test slice with various step values
    data = [["x", f"x{i}"] for i in range(5)]
    data += [["y", f"y{i}"] for i in range(4)]
    data += [["z", f"z{i}"] for i in range(3)]
    df = pd.DataFrame(data, columns=["A", "B"])

    grouped = df.groupby("A", as_index=False)

    result = grouped._positional_selector[::step]

    data = [["x", f"x{i}"] for i in range(0, 5, step)]
    data += [["y", f"y{i}"] for i in range(0, 4, step)]
    data += [["z", f"z{i}"] for i in range(0, 3, step)]

    index = [0 + i for i in range(0, 5, step)]
    index += [5 + i for i in range(0, 4, step)]
    index += [9 + i for i in range(0, 3, step)]

    expected = pd.DataFrame(data, columns=["A", "B"], index=index)

    tm.assert_frame_equal(result, expected)


def test_columns_on_iter():
    # [FUNCTION DOC]
    # Issue 44821: iterating a groupby on a column selection yields groups with only the selected columns.
    # GitHub issue #44821
    df = pd.DataFrame({k: range(10) for k in "ABC"})

    # Group-by and select columns
    cols = ["A", "B"]
    for _, dg in df.groupby(df.A < 4)[cols]:
        tm.assert_index_equal(dg.columns, pd.Index(cols))
        assert "C" not in dg.columns


@pytest.mark.parametrize("func", [list, pd.Index, pd.Series, np.array])
def test_groupby_duplicated_columns(func):
    # [FUNCTION DOC]
    # Issue 44924: selecting columns with duplicates (a list, index, series or array of labels) from a groupby and averaging returns the duplicated columns as expected.
    # GH#44924
    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 3],
            "C": ["G", "G"],
        }
    )
    result = df.groupby("C")[func(["A", "B", "A"])].mean()
    expected = pd.DataFrame(
        [[1.5, 3.0, 1.5]], columns=["A", "B", "A"], index=pd.Index(["G"], name="C")
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_get_nonexisting_groups():
    # [FUNCTION DOC]
    # Issue 32492: getting a group for a key combination that does not exist (the other key is missing) raises a key error.
    # GH#32492
    df = pd.DataFrame(
        data={
            "A": ["a1", "a2", None],
            "B": ["b1", "b2", "b1"],
            "val": [1, 2, 3],
        }
    )
    grps = df.groupby(by=["A", "B"])

    msg = "('a2', 'b1')"
    with pytest.raises(KeyError, match=msg):
        grps.get_group(("a2", "b1"))
