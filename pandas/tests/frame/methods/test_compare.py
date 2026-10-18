# [MODULE DOC]
# Tests for the frame method that shows differences between two identically-labeled
# frames (compare): alignment axis, keeping shape and equal values, treatment of missing
# values, multi-level labels, custom result labels, error cases and nullable/extension
# dtype combinations. Uses only the public package and the testing helpers module. No
# import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("align_axis", [0, 1, "index", "columns"])
def test_compare_axis(align_axis):
    # [FUNCTION DOC]
    # Parametrized over align_axis values (0, 1, index, columns). Two frames differing
    # in one string cell and one numeric cell: with columns alignment the result has
    # self/other pairs side by side as a second column level for the differing columns
    # and rows only; with index alignment the pairs are stacked in a second row level.
    # Compared with explicitly built expected frames.
    # GH#30429
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
        columns=["col1", "col2", "col3"],
    )
    df2 = df.copy()
    df2.loc[0, "col1"] = "c"
    df2.loc[2, "col3"] = 4.0

    result = df.compare(df2, align_axis=align_axis)

    if align_axis in (1, "columns"):
        indices = pd.RangeIndex(0, 4, 2)
        columns = pd.MultiIndex.from_product([["col1", "col3"], ["self", "other"]])
        expected = pd.DataFrame(
            [["a", "c", np.nan, np.nan], [np.nan, np.nan, 3.0, 4.0]],
            index=indices,
            columns=columns,
        )
    else:
        indices = pd.MultiIndex.from_product([range(0, 4, 2), ["self", "other"]])
        columns = pd.Index(["col1", "col3"])
        expected = pd.DataFrame(
            [["a", np.nan], ["c", np.nan], [np.nan, 3.0], [np.nan, 4.0]],
            index=indices,
            columns=columns,
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "keep_shape, keep_equal",
    [
        (True, False),
        (False, True),
        (True, True),
        # False, False case is already covered in test_compare_axis
    ],
)
def test_compare_various_formats(keep_shape, keep_equal):
    # [FUNCTION DOC]
    # Parametrized over (keep_shape, keep_equal) combinations other than both false.
    # Verifies the shape (all rows and columns vs only differing ones) and that equal
    # values are shown instead of missing markers when keep_equal is true.
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
        columns=["col1", "col2", "col3"],
    )
    df2 = df.copy()
    df2.loc[0, "col1"] = "c"
    df2.loc[2, "col3"] = 4.0

    result = df.compare(df2, keep_shape=keep_shape, keep_equal=keep_equal)

    if keep_shape:
        indices = pd.RangeIndex(3)
        columns = pd.MultiIndex.from_product(
            [["col1", "col2", "col3"], ["self", "other"]]
        )
        if keep_equal:
            expected = pd.DataFrame(
                [
                    ["a", "c", 1.0, 1.0, 1.0, 1.0],
                    ["b", "b", 2.0, 2.0, 2.0, 2.0],
                    ["c", "c", np.nan, np.nan, 3.0, 4.0],
                ],
                index=indices,
                columns=columns,
            )
        else:
            expected = pd.DataFrame(
                [
                    ["a", "c", np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 3.0, 4.0],
                ],
                index=indices,
                columns=columns,
            )
    else:
        indices = pd.RangeIndex(0, 4, 2)
        columns = pd.MultiIndex.from_product([["col1", "col3"], ["self", "other"]])
        expected = pd.DataFrame(
            [["a", "c", 1.0, 1.0], ["c", "c", 3.0, 4.0]], index=indices, columns=columns
        )
    tm.assert_frame_equal(result, expected)


def test_compare_with_equal_nulls():
    # [FUNCTION DOC]
    # Missing values in the same position in both frames are considered equal, so the
    # unchanged column and rows containing only such missing values are dropped from the
    # result.
    # We want to make sure two NaNs are considered the same
    # and dropped where applicable
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
        columns=["col1", "col2", "col3"],
    )
    df2 = df.copy()
    df2.loc[0, "col1"] = "c"

    result = df.compare(df2)
    indices = pd.Index([0])
    columns = pd.MultiIndex.from_product([["col1"], ["self", "other"]])
    expected = pd.DataFrame([["a", "c"]], index=indices, columns=columns)
    tm.assert_frame_equal(result, expected)


def test_compare_with_non_equal_nulls():
    # [FUNCTION DOC]
    # When one frame has a missing value where the other has a number, that position
    # counts as a difference; the result keeps the row and the missing entry on the self
    # or other side as appropriate.
    # We want to make sure the relevant NaNs do not get dropped
    # even if the entire row or column are NaNs
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
        columns=["col1", "col2", "col3"],
    )
    df2 = df.copy()
    df2.loc[0, "col1"] = "c"
    df2.loc[2, "col3"] = np.nan

    result = df.compare(df2)

    indices = pd.Index([0, 2])
    columns = pd.MultiIndex.from_product([["col1", "col3"], ["self", "other"]])
    expected = pd.DataFrame(
        [["a", "c", np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan]],
        index=indices,
        columns=columns,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("align_axis", [0, 1])
def test_compare_multi_index(align_axis):
    # [FUNCTION DOC]
    # Parametrized over align axis (0, 1). Frames with multi-level row and column labels
    # yield results with the extra self/other level added to rows or columns; labels of
    # the differing positions are retained.
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]}
    )
    df.columns = pd.MultiIndex.from_arrays([["a", "a", "b"], ["col1", "col2", "col3"]])
    df.index = pd.MultiIndex.from_arrays([["x", "x", "y"], [0, 1, 2]])

    df2 = df.copy()
    df2.iloc[0, 0] = "c"
    df2.iloc[2, 2] = 4.0

    result = df.compare(df2, align_axis=align_axis)

    if align_axis == 0:
        indices = pd.MultiIndex.from_arrays(
            [["x", "x", "y", "y"], [0, 0, 2, 2], ["self", "other", "self", "other"]]
        )
        columns = pd.MultiIndex.from_arrays([["a", "b"], ["col1", "col3"]])
        data = [["a", np.nan], ["c", np.nan], [np.nan, 3.0], [np.nan, 4.0]]
    else:
        indices = pd.MultiIndex.from_arrays([["x", "y"], [0, 2]])
        columns = pd.MultiIndex.from_arrays(
            [
                ["a", "a", "b", "b"],
                ["col1", "col1", "col3", "col3"],
                ["self", "other", "self", "other"],
            ]
        )
        data = [["a", "c", np.nan, np.nan], [np.nan, np.nan, 3.0, 4.0]]

    expected = pd.DataFrame(data=data, index=indices, columns=columns)
    tm.assert_frame_equal(result, expected)


def test_compare_different_indices():
    # [FUNCTION DOC]
    # Frames with different row labels raise a value error saying only
    # identically-labeled frames can be compared.
    msg = (
        r"Can only compare identically-labeled \(both index and columns\) DataFrame "
        "objects"
    )
    df1 = pd.DataFrame([1, 2, 3], index=["a", "b", "c"])
    df2 = pd.DataFrame([1, 2, 3], index=["a", "b", "d"])
    with pytest.raises(ValueError, match=msg):
        df1.compare(df2)


def test_compare_different_shapes():
    # [FUNCTION DOC]
    # Frames of different shape raise the same value error.
    msg = (
        r"Can only compare identically-labeled \(both index and columns\) DataFrame "
        "objects"
    )
    df1 = pd.DataFrame(np.ones((3, 3)))
    df2 = pd.DataFrame(np.zeros((2, 1)))
    with pytest.raises(ValueError, match=msg):
        df1.compare(df2)


def test_compare_result_names():
    # [FUNCTION DOC]
    # Custom labels for the two sides are used as the second level of the result columns
    # instead of self and other.
    # GH 44354
    df1 = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
    )
    df2 = pd.DataFrame(
        {
            "col1": ["c", "b", "c"],
            "col2": [1.0, 2.0, np.nan],
            "col3": [1.0, 2.0, np.nan],
        },
    )
    result = df1.compare(df2, result_names=("left", "right"))
    result.index = pd.Index([0, 2])
    expected = pd.DataFrame(
        {
            ("col1", "left"): {0: "a", 2: np.nan},
            ("col1", "right"): {0: "c", 2: np.nan},
            ("col3", "left"): {0: np.nan, 2: 3.0},
            ("col3", "right"): {0: np.nan, 2: np.nan},
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "result_names",
    [
        [1, 2],
        "HK",
        {"2": 2, "3": 3},
        3,
        3.0,
    ],
)
def test_invalid_input_result_names(result_names):
    # [FUNCTION DOC]
    # Parametrized over list, string, dict, int and float. Anything other than a tuple
    # for the result labels raises a type error telling to use a tuple and naming the
    # given type.
    # GH 44354
    df1 = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
    )
    df2 = pd.DataFrame(
        {
            "col1": ["c", "b", "c"],
            "col2": [1.0, 2.0, np.nan],
            "col3": [1.0, 2.0, np.nan],
        },
    )
    with pytest.raises(
        TypeError,
        match=(
            f"Passing 'result_names' as a {type(result_names)} is not "
            "supported. Provide 'result_names' as a tuple instead."
        ),
    ):
        df1.compare(df2, result_names=result_names)


@pytest.mark.parametrize(
    "val1,val2",
    [(4, pd.NA), (pd.NA, pd.NA), (pd.NA, 4)],
)
def test_compare_ea_and_np_dtype(val1, val2):
    # [FUNCTION DOC]
    # Parametrized over pairs of value and missing. Compares a numpy float column with a
    # nullable integer column, keeping shape. When the first value is a missing marker,
    # comparing raises a type error about the ambiguous truth value of NA; otherwise the
    # result matches the expected frame (with an adjustment when both are missing).
    # KNOWN ISSUE: a comment on the both-missing adjustment asks whether this is really
    # the desired behavior (TODO), and the raising branch assigns a result that is never
    # used.
    # GH 48966
    arr = [4.0, val1]
    ser = pd.Series([1, val2], dtype="Int64")

    df1 = pd.DataFrame({"a": arr, "b": [1.0, 2]})
    df2 = pd.DataFrame({"a": ser, "b": [1.0, 2]})
    expected = pd.DataFrame(
        {
            ("a", "self"): arr,
            ("a", "other"): ser,
            ("b", "self"): np.nan,
            ("b", "other"): np.nan,
        }
    )
    if val1 is pd.NA and val2 is pd.NA:
        # GH#18463 TODO: is this really the desired behavior?
        expected.loc[1, ("a", "self")] = np.nan

    if val1 is pd.NA:
        # can't compare with numpy array if it contains pd.NA
        with pytest.raises(TypeError, match="boolean value of NA is ambiguous"):
            result = df1.compare(df2, keep_shape=True)
    else:
        result = df1.compare(df2, keep_shape=True)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "df1_val,df2_val,diff_self,diff_other",
    [
        (4, 3, 4, 3),
        (4, 4, pd.NA, pd.NA),
        (4, pd.NA, 4, pd.NA),
        (pd.NA, pd.NA, pd.NA, pd.NA),
    ],
)
def test_compare_nullable_int64_dtype(df1_val, df2_val, diff_self, diff_other):
    # [FUNCTION DOC]
    # Parametrized over value combinations. Comparing frames with nullable integer
    # columns keeping shape gives nullable-integer self/other columns; equal values and
    # missing values are shown as NA in the result.
    # GH 48966
    df1 = pd.DataFrame({"a": pd.Series([df1_val, pd.NA], dtype="Int64"), "b": [1.0, 2]})
    df2 = df1.copy()
    df2.loc[0, "a"] = df2_val

    expected = pd.DataFrame(
        {
            ("a", "self"): pd.Series([diff_self, pd.NA], dtype="Int64"),
            ("a", "other"): pd.Series([diff_other, pd.NA], dtype="Int64"),
            ("b", "self"): np.nan,
            ("b", "other"): np.nan,
        }
    )
    result = df1.compare(df2, keep_shape=True)
    tm.assert_frame_equal(result, expected)
