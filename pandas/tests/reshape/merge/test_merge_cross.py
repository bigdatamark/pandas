# [MODULE DOC]
# Tests for the cross join mode of the top-level merge function and the frame join method.
# Covers output layout, column-name suffixing, dtype handling, null values, series inputs and the error raised when key arguments are combined with a cross join.
# Relies on the shared null-values fixture and the merge error class from the reshape merge module; no import-time side effects.
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.core.reshape.merge import (
    MergeError,
    merge,
)


@pytest.mark.parametrize(
    ("input_col", "output_cols"), [("b", ["a", "b"]), ("a", ["a_x", "a_y"])]
)
def test_merge_cross(input_col, output_cols):
    # GH#5401
    # [FUNCTION DOC]
    # Verifies a cross merge yields the cartesian product of two single-column frames.
    # Parametrized on the right column name: a distinct name keeps both column names, while a clashing name gets the default left and right suffixes.
    # Also checks that neither input frame is mutated.
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({input_col: [3, 4]})
    left_copy = left.copy()
    right_copy = right.copy()
    result = merge(left, right, how="cross")
    expected = DataFrame({output_cols[0]: [1, 1, 3, 3], output_cols[1]: [3, 4, 3, 4]})
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(left, left_copy)
    tm.assert_frame_equal(right, right_copy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True},
        {"right_index": True},
        {"on": "a"},
        {"left_on": "a"},
        {"right_on": "b"},
    ],
)
def test_merge_cross_error_reporting(kwargs):
    # GH#5401
    # [FUNCTION DOC]
    # Verifies a cross merge raises the merge error with the message about not passing key arguments when any of left index, right index, on, left on or right on is supplied.
    # Parametrized over those keyword arguments.
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({"b": [3, 4]})
    msg = (
        "Can not pass on, right_on, left_on or set right_index=True or left_index=True"
    )
    with pytest.raises(MergeError, match=msg):
        merge(left, right, how="cross", **kwargs)


def test_merge_cross_mixed_dtypes():
    # GH#5401
    # [FUNCTION DOC]
    # Cross merge of a string column frame with an integer range column frame gives the full product with each left row repeated for every right row; dtypes preserved.
    left = DataFrame(["a", "b", "c"], columns=["A"])
    right = DataFrame(range(2), columns=["B"])
    result = merge(left, right, how="cross")
    expected = DataFrame({"A": ["a", "a", "b", "b", "c", "c"], "B": [0, 1, 0, 1, 0, 1]})
    tm.assert_frame_equal(result, expected)


def test_merge_cross_more_than_one_column():
    # GH#5401
    # [FUNCTION DOC]
    # Cross merge of two frames with two columns each yields all four columns, left rows repeated in blocks and right rows cycled.
    left = DataFrame({"A": list("ab"), "B": [2, 1]})
    right = DataFrame({"C": range(2), "D": range(4, 6)})
    result = merge(left, right, how="cross")
    expected = DataFrame(
        {
            "A": ["a", "a", "b", "b"],
            "B": [2, 2, 1, 1],
            "C": [0, 1, 0, 1],
            "D": [4, 5, 4, 5],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_merge_cross_null_values(nulls_fixture):
    # GH#5401
    # [FUNCTION DOC]
    # Verifies null values from the nulls fixture in the left key column are carried through the cross merge unchanged (repeated per right row).
    left = DataFrame({"a": [1, nulls_fixture]})
    right = DataFrame({"b": ["a", "b"], "c": [1.0, 2.0]})
    result = merge(left, right, how="cross")
    expected = DataFrame(
        {
            "a": [1, 1, nulls_fixture, nulls_fixture],
            "b": ["a", "b", "a", "b"],
            "c": [1.0, 2.0, 1.0, 2.0],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_join_cross_error_reporting():
    # GH#5401
    # [FUNCTION DOC]
    # Verifies the frame join method with cross mode and an on argument raises the merge error with the key-arguments message.
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({"a": [3, 4]})
    msg = (
        "Can not pass on, right_on, left_on or set right_index=True or left_index=True"
    )
    with pytest.raises(MergeError, match=msg):
        left.join(right, how="cross", on="a")


def test_merge_cross_series():
    # GH#54055
    # [FUNCTION DOC]
    # Verifies cross merging two named Series gives the same result as cross merging their frame conversions.
    ls = Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="left")
    rs = Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="right")
    res = merge(ls, rs, how="cross")

    expected = merge(ls.to_frame(), rs.to_frame(), how="cross")
    tm.assert_frame_equal(res, expected)
