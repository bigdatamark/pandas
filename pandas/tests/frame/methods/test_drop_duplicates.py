# [MODULE DOC]
# Tests for the duplicate-row removal method of the frame class (drop_duplicates):
# subset selection (single label, list, tuple, array, set), keep modes (first, last,
# none), missing values, empty frames, in-place mode, ignore_index, duplicate and tuple
# column labels, and comparison with the series method. Uses the keep and nulls fixtures
# plus the testing helpers module. All tests are module-level functions. No import-time
# side effects.

from datetime import datetime
import re

import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    concat,
)
import pandas._testing as tm


@pytest.mark.parametrize("subset", ["a", ["a"], ["a", "B"]])
def test_drop_duplicates_with_misspelled_column_name(subset):
    # [FUNCTION DOC]
    # Parametrized over a string, a list and a mixed list naming a nonexistent
    # (lower-case) column. Raises a key error whose message shows the index of missing
    # labels.
    # GH 19730
    df = DataFrame({"A": [0, 0, 1], "B": [0, 0, 1], "C": [0, 0, 1]})
    msg = re.escape("Index(['a'], dtype=")

    with pytest.raises(KeyError, match=msg):
        df.drop_duplicates(subset)


def test_drop_duplicates():
    # [FUNCTION DOC]
    # Large scenario with string and integer columns: single column and multiple column
    # subsets (given as string, array, list, tuple) with keep first, last and false;
    # whole-row duplicates; integer subset including a narrow integer column; several
    # regression frames of numbers (including big integers) that have no duplicates; and
    # a wide frame verifying that duplicated reports none for all keep modes.
    df = DataFrame(
        {
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates("AAA")
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep="last")
    expected = df.loc[[6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep=False)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0

    # multi column
    expected = df.loc[[0, 1, 2, 3]]
    result = df.drop_duplicates(np.array(["AAA", "B"]))
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(["AAA", "B"])
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AAA", "B"), keep="last")
    expected = df.loc[[0, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AAA", "B"), keep=False)
    expected = df.loc[[0]]
    tm.assert_frame_equal(result, expected)

    # consider everything
    df2 = df.loc[:, ["AAA", "B", "C"]]

    result = df2.drop_duplicates()
    # in this case only
    expected = df2.drop_duplicates(["AAA", "B"])
    tm.assert_frame_equal(result, expected)

    result = df2.drop_duplicates(keep="last")
    expected = df2.drop_duplicates(["AAA", "B"], keep="last")
    tm.assert_frame_equal(result, expected)

    result = df2.drop_duplicates(keep=False)
    expected = df2.drop_duplicates(["AAA", "B"], keep=False)
    tm.assert_frame_equal(result, expected)

    # integers
    result = df.drop_duplicates("C")
    expected = df.iloc[[0, 2]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates("C", keep="last")
    expected = df.iloc[[-2, -1]]
    tm.assert_frame_equal(result, expected)

    df["E"] = df["C"].astype("int8")
    result = df.drop_duplicates("E")
    expected = df.iloc[[0, 2]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates("E", keep="last")
    expected = df.iloc[[-2, -1]]
    tm.assert_frame_equal(result, expected)

    # GH 11376
    df = DataFrame({"x": [7, 6, 3, 3, 4, 8, 0], "y": [0, 6, 5, 5, 9, 1, 2]})
    expected = df.loc[df.index != 3]
    tm.assert_frame_equal(df.drop_duplicates(), expected)

    df = DataFrame([[1, 0], [0, 2]])
    tm.assert_frame_equal(df.drop_duplicates(), df)

    df = DataFrame([[-2, 0], [0, -4]])
    tm.assert_frame_equal(df.drop_duplicates(), df)

    x = np.iinfo(np.int64).max / 3 * 2
    df = DataFrame([[-x, x], [0, x + 4]])
    tm.assert_frame_equal(df.drop_duplicates(), df)

    df = DataFrame([[-x, x], [x, x + 4]])
    tm.assert_frame_equal(df.drop_duplicates(), df)

    # GH 11864
    df = DataFrame([i] * 9 for i in range(16))
    df = concat([df, DataFrame([[1] + [0] * 8])], ignore_index=True)

    for keep in ["first", "last", False]:
        assert df.duplicated(keep=keep).sum() == 0


def test_drop_duplicates_with_duplicate_column_names():
    # [FUNCTION DOC]
    # With duplicated column labels, dropping duplicates over all columns keeps all
    # rows, and subset naming the duplicated label compares both columns.
    # GH17836
    df = DataFrame([[1, 2, 5], [3, 4, 6], [3, 4, 7]], columns=["a", "a", "b"])

    result0 = df.drop_duplicates()
    tm.assert_frame_equal(result0, df)

    result1 = df.drop_duplicates("a")
    expected1 = df[:2]
    tm.assert_frame_equal(result1, expected1)


def test_drop_duplicates_for_take_all():
    # [FUNCTION DOC]
    # Same style of checks as the main test but with data where no group is entirely
    # duplicated, for single and multi-column subsets and all keep modes.
    df = DataFrame(
        {
            "AAA": ["foo", "bar", "baz", "bar", "foo", "bar", "qux", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates("AAA")
    expected = df.iloc[[0, 1, 2, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep="last")
    expected = df.iloc[[2, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep=False)
    expected = df.iloc[[2, 6]]
    tm.assert_frame_equal(result, expected)

    # multiple columns
    result = df.drop_duplicates(["AAA", "B"])
    expected = df.iloc[[0, 1, 2, 3, 4, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["AAA", "B"], keep="last")
    expected = df.iloc[[0, 1, 2, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["AAA", "B"], keep=False)
    expected = df.iloc[[0, 1, 2, 6]]
    tm.assert_frame_equal(result, expected)


def test_drop_duplicates_tuple():
    # [FUNCTION DOC]
    # Columns labeled by tuples: subset given as a tuple label (single column) and as a
    # tuple containing a tuple label (multi column) behave as for plain labels.
    df = DataFrame(
        {
            ("AA", "AB"): ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates(("AA", "AB"))
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AA", "AB"), keep="last")
    expected = df.loc[[6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AA", "AB"), keep=False)
    expected = df.loc[[]]  # empty df
    assert len(result) == 0
    tm.assert_frame_equal(result, expected)

    # multi column
    expected = df.loc[[0, 1, 2, 3]]
    result = df.drop_duplicates((("AA", "AB"), "B"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "df",
    [
        DataFrame(),
        DataFrame(columns=[]),
        DataFrame(columns=["A", "B", "C"]),
        DataFrame(index=[]),
        DataFrame(index=["A", "B", "C"]),
    ],
)
def test_drop_duplicates_empty(df):
    # [FUNCTION DOC]
    # Parametrized over five empty frames (no rows, no columns, only columns, only
    # index). Result equals the input, both returned and in-place.
    # GH 20516
    result = df.drop_duplicates()
    tm.assert_frame_equal(result, df)

    result = df.copy()
    result.drop_duplicates(inplace=True)
    tm.assert_frame_equal(result, df)


def test_drop_duplicates_NA():
    # [FUNCTION DOC]
    # Missing values (None and NaN) in the subset are considered equal to each other;
    # checks single and multi-column subsets with keep first, last and false.
    # none
    df = DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates("A")
    expected = df.loc[[0, 2, 3]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("A", keep="last")
    expected = df.loc[[1, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("A", keep=False)
    expected = df.loc[[]]  # empty df
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0

    # multi column
    result = df.drop_duplicates(["A", "B"])
    expected = df.loc[[0, 2, 3, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["A", "B"], keep="last")
    expected = df.loc[[1, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["A", "B"], keep=False)
    expected = df.loc[[6]]
    tm.assert_frame_equal(result, expected)

    # nan
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates("C")
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("C", keep="last")
    expected = df.loc[[3, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("C", keep=False)
    expected = df.loc[[]]  # empty df
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0

    # multi column
    result = df.drop_duplicates(["C", "B"])
    expected = df.loc[[0, 1, 2, 4]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["C", "B"], keep="last")
    expected = df.loc[[1, 3, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["C", "B"], keep=False)
    expected = df.loc[[1]]
    tm.assert_frame_equal(result, expected)


def test_drop_duplicates_NA_for_take_all():
    # [FUNCTION DOC]
    # Same, with data in which some missing groups are the only elements, checking
    # expected row selection.
    # none
    df = DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "baz", "bar", "qux"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 2.0, 3, 1.0],
        }
    )

    # single column
    result = df.drop_duplicates("A")
    expected = df.iloc[[0, 2, 3, 5, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("A", keep="last")
    expected = df.iloc[[1, 4, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("A", keep=False)
    expected = df.iloc[[5, 7]]
    tm.assert_frame_equal(result, expected)

    # nan

    # single column
    result = df.drop_duplicates("C")
    expected = df.iloc[[0, 1, 5, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("C", keep="last")
    expected = df.iloc[[3, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("C", keep=False)
    expected = df.iloc[[5, 6]]
    tm.assert_frame_equal(result, expected)


def test_drop_duplicates_inplace():
    # [FUNCTION DOC]
    # In-place variant of the main scenarios: all calls return none and mutate the frame
    # to the expected rows, for single and multi-column subsets and whole rows.
    orig = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    df = orig.copy()
    return_value = df.drop_duplicates("A", inplace=True)
    expected = orig[:2]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df = orig.copy()
    return_value = df.drop_duplicates("A", keep="last", inplace=True)
    expected = orig.loc[[6, 7]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df = orig.copy()
    return_value = df.drop_duplicates("A", keep=False, inplace=True)
    expected = orig.loc[[]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert len(df) == 0
    assert return_value is None

    # multi column
    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], inplace=True)
    expected = orig.loc[[0, 1, 2, 3]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], keep="last", inplace=True)
    expected = orig.loc[[0, 5, 6, 7]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], keep=False, inplace=True)
    expected = orig.loc[[0]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    # consider everything
    orig2 = orig.loc[:, ["A", "B", "C"]].copy()

    df2 = orig2.copy()
    return_value = df2.drop_duplicates(inplace=True)
    # in this case only
    expected = orig2.drop_duplicates(["A", "B"])
    result = df2
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df2 = orig2.copy()
    return_value = df2.drop_duplicates(keep="last", inplace=True)
    expected = orig2.drop_duplicates(["A", "B"], keep="last")
    result = df2
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df2 = orig2.copy()
    return_value = df2.drop_duplicates(keep=False, inplace=True)
    expected = orig2.drop_duplicates(["A", "B"], keep=False)
    result = df2
    tm.assert_frame_equal(result, expected)
    assert return_value is None


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "origin_dict, output_dict, ignore_index, output_index",
    [
        ({"A": [2, 2, 3]}, {"A": [2, 3]}, True, range(2)),
        ({"A": [2, 2, 3]}, {"A": [2, 3]}, False, range(0, 4, 2)),
        ({"A": [2, 2, 3], "B": [2, 2, 4]}, {"A": [2, 3], "B": [2, 4]}, True, range(2)),
        (
            {"A": [2, 2, 3], "B": [2, 2, 4]},
            {"A": [2, 3], "B": [2, 4]},
            False,
            range(0, 4, 2),
        ),
    ],
)
def test_drop_duplicates_ignore_index(
    inplace, origin_dict, output_dict, ignore_index, output_index
):
    # [FUNCTION DOC]
    # Parametrized over in-place flag and four (input, expected data, ignore_index,
    # expected index) cases. With ignore_index true the result gets a fresh range index;
    # otherwise original labels are kept. The input is never changed when not in place.
    # GH 30114
    df = DataFrame(origin_dict)
    expected = DataFrame(output_dict, index=output_index)

    if inplace:
        result_df = df.copy()
        result_df.drop_duplicates(ignore_index=ignore_index, inplace=inplace)
    else:
        result_df = df.drop_duplicates(ignore_index=ignore_index, inplace=inplace)

    tm.assert_frame_equal(result_df, expected)
    tm.assert_frame_equal(df, DataFrame(origin_dict))


def test_drop_duplicates_null_in_object_column(nulls_fixture):
    # [FUNCTION DOC]
    # Takes the nulls fixture. An object frame with a missing marker and a string has no
    # duplicates, so it is returned unchanged.
    # https://github.com/pandas-dev/pandas/issues/32992
    df = DataFrame([[1, nulls_fixture], [2, "a"]], dtype=object)
    result = df.drop_duplicates()
    tm.assert_frame_equal(result, df)


def test_drop_duplicates_series_vs_dataframe(keep):
    # [FUNCTION DOC]
    # Takes the keep fixture. For each column of a mixed frame, dropping duplicates on
    # the one-column frame equals the series version converted to a frame.
    # GH#14192
    df = DataFrame(
        {
            "a": [1, 1, 1, "one", "one"],
            "b": [2, 2, np.nan, np.nan, np.nan],
            "c": [3, 3, np.nan, np.nan, "three"],
            "d": [1, 2, 3, 4, 4],
            "e": [
                datetime(2015, 1, 1),
                datetime(2015, 1, 1),
                datetime(2015, 2, 1),
                NaT,
                NaT,
            ],
        }
    )
    for column in df.columns:
        dropped_frame = df[[column]].drop_duplicates(keep=keep)
        dropped_series = df[column].drop_duplicates(keep=keep)
        tm.assert_frame_equal(dropped_frame, dropped_series.to_frame())


@pytest.mark.parametrize("arg", [[1], 1, "True", [], 0])
def test_drop_duplicates_non_boolean_ignore_index(arg):
    # [FUNCTION DOC]
    # Parametrized over several non-boolean values. Passing them as ignore_index raises
    # a value error saying a bool was expected.
    # GH#38274
    df = DataFrame({"a": [1, 2, 1, 3]})
    msg = '^For argument "ignore_index" expected type bool, received type .*.$'
    with pytest.raises(ValueError, match=msg):
        df.drop_duplicates(ignore_index=arg)


def test_drop_duplicates_set():
    # [FUNCTION DOC]
    # A set is accepted as subset (single and multi element) with keep first, last and
    # false, with the same results as list subsets.
    # GH#59237
    df = DataFrame(
        {
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates({"AAA"})
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates({"AAA"}, keep="last")
    expected = df.loc[[6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates({"AAA"}, keep=False)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0

    # multi column
    expected = df.loc[[0, 1, 2, 3]]
    result = df.drop_duplicates({"AAA", "B"})
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates({"AAA", "B"}, keep="last")
    expected = df.loc[[0, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates({"AAA", "B"}, keep=False)
    expected = df.loc[[0]]
    tm.assert_frame_equal(result, expected)
