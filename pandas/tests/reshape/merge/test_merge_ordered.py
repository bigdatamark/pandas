# [MODULE DOC]
# Tests for the ordered merge function and a few sequence-concatenation edge cases.
# Provides small left and right fixtures keyed on a string column and a test class exercising basic ordering, forward fill, grouped merges, list-valued group arguments and argument validation.
# No import-time side effects.
import re

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    merge_ordered,
)
import pandas._testing as tm


@pytest.fixture
def left():
    # [FUNCTION DOC]
    # HELPER: fixture returning a frame with string key column (a, c, e) and a numeric lvalue column.
    return DataFrame({"key": ["a", "c", "e"], "lvalue": [1, 2.0, 3]})


@pytest.fixture
def right():
    # [FUNCTION DOC]
    # HELPER: fixture returning a frame with string key column (b, c, d, f) and a numeric rvalue column.
    return DataFrame({"key": ["b", "c", "d", "f"], "rvalue": [1, 2, 3.0, 4]})


class TestMergeOrdered:
    # [CLASS DOC]
    # Groups tests for the ordered merge function and related behaviors (subclass propagation, empty concatenation).
    def test_basic(self, left, right):
        # [FUNCTION DOC]
        # Ordered outer merge on key yields the sorted union of keys with missing values where a side lacks a key.
        result = merge_ordered(left, right, on="key")
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"],
                "lvalue": [1, np.nan, 2, np.nan, 3, np.nan],
                "rvalue": [np.nan, 1, 2, 3, np.nan, 4],
            }
        )

        tm.assert_frame_equal(result, expected)

    def test_ffill(self, left, right):
        # [FUNCTION DOC]
        # With forward fill, missing values on each side are filled from the previous row; leading missing values remain missing.
        result = merge_ordered(left, right, on="key", fill_method="ffill")
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"],
                "lvalue": [1.0, 1, 2, 2, 3, 3.0],
                "rvalue": [np.nan, 1, 2, 3, 3, 4],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_multigroup(self, left, right):
        # [FUNCTION DOC]
        # Left frame is duplicated and given a group column; merging with left_by and forward fill gives the single-group result repeated per group.
        # Swapping inputs with right_by gives the same result (column order aside). Without fill, the group column has no missing values.
        left = pd.concat([left, left], ignore_index=True)

        left["group"] = ["a"] * 3 + ["b"] * 3

        result = merge_ordered(
            left, right, on="key", left_by="group", fill_method="ffill"
        )
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"] * 2,
                "lvalue": [1.0, 1, 2, 2, 3, 3.0] * 2,
                "rvalue": [np.nan, 1, 2, 3, 3, 4] * 2,
            }
        )
        expected["group"] = ["a"] * 6 + ["b"] * 6

        tm.assert_frame_equal(result, expected.loc[:, result.columns])

        result2 = merge_ordered(
            right, left, on="key", right_by="group", fill_method="ffill"
        )
        tm.assert_frame_equal(result, result2.loc[:, result.columns])

        result = merge_ordered(left, right, on="key", left_by="group")
        assert result["group"].notna().all()

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_merge_type(self, left, right):
        # [FUNCTION DOC]
        # Verifies that the frame merge method on a DataFrame subclass returns an instance of that subclass.
        # Runs with the deprecation warning about block managers ignored.
        class NotADataFrame(DataFrame):
            @property
            # [CLASS DOC]
            # HELPER: DataFrame subclass used to check that merge preserves the subclass via the constructor hook.
            def _constructor(self):
                # [FUNCTION DOC]
                # HELPER: property returning the subclass itself so pandas operations build the same subclass.
                return NotADataFrame

        nad = NotADataFrame(left)
        result = nad.merge(right, on="key")

        assert isinstance(result, NotADataFrame)

    @pytest.mark.parametrize(
        "df_seq, pattern",
        [
            ((), "[Nn]o objects"),
            ([], "[Nn]o objects"),
            ({}, "[Nn]o objects"),
            ([None], "objects.*None"),
            ([None, None], "objects.*None"),
        ],
    )
    def test_empty_sequence_concat(self, df_seq, pattern):
        # GH 9157
        # [FUNCTION DOC]
        # GH 9157. Concatenating an empty tuple, list or dict, or a sequence containing only None, raises a value error whose message matches the parametrized pattern (no objects to concatenate, or all objects are None).
        with pytest.raises(ValueError, match=pattern):
            pd.concat(df_seq)

    @pytest.mark.parametrize(
        "arg", [[DataFrame()], [None, DataFrame()], [DataFrame(), None]]
    )
    def test_empty_sequence_concat_ok(self, arg):
        # [FUNCTION DOC]
        # Concatenating sequences that contain an empty frame (alone or with None) does not raise. No result is checked.
        pd.concat(arg)

    def test_doc_example(self):
        # [FUNCTION DOC]
        # Reproduces the documentation example: ordered merge with forward fill and left_by on a group column; checks the expected grouped, filled output.
        left = DataFrame(
            {
                "group": list("aaabbb"),
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3] * 2,
            }
        )

        right = DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})

        result = merge_ordered(left, right, fill_method="ffill", left_by="group")

        expected = DataFrame(
            {
                "group": list("aaaaabbbbb"),
                "key": ["a", "b", "c", "d", "e"] * 2,
                "lvalue": [1, 1, 2, 2, 3] * 2,
                "rvalue": [np.nan, 1, 2, 3, 3] * 2,
            }
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "left, right, on, left_by, right_by, expected",
        [
            (
                {"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]},
                {"T": [2], "E": [1]},
                ["T"],
                ["G", "H"],
                None,
                {
                    "G": ["g"] * 3,
                    "H": ["h"] * 3,
                    "T": [1, 2, 3],
                    "E": [np.nan, 1.0, np.nan],
                },
            ),
            (
                {"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]},
                {"T": [2], "E": [1]},
                "T",
                ["G", "H"],
                None,
                {
                    "G": ["g"] * 3,
                    "H": ["h"] * 3,
                    "T": [1, 2, 3],
                    "E": [np.nan, 1.0, np.nan],
                },
            ),
            (
                {"T": [2], "E": [1]},
                {"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]},
                ["T"],
                None,
                ["G", "H"],
                {
                    "T": [1, 2, 3],
                    "E": [np.nan, 1.0, np.nan],
                    "G": ["g"] * 3,
                    "H": ["h"] * 3,
                },
            ),
        ],
    )
    def test_list_type_by(self, left, right, on, left_by, right_by, expected):
        # GH 35269
        # [FUNCTION DOC]
        # GH 35269. Parametrized over dict inputs, on as list or string, and left_by or right_by given as lists of two columns.
        # The merge must work with list-valued group arguments and produce a frame sorted on the key per group with missing values for absent rows.
        left = DataFrame(left)
        right = DataFrame(right)
        result = merge_ordered(
            left=left,
            right=right,
            on=on,
            left_by=left_by,
            right_by=right_by,
        )
        expected = DataFrame(expected)

        tm.assert_frame_equal(result, expected)

    def test_left_by_length_equals_to_right_shape0(self):
        # GH 38166
        # [FUNCTION DOC]
        # GH 38166. Left_by with two columns when the right frame has a single row; the result must contain the combined keys with missing values in the right-only column.
        left = DataFrame([["g", "h", 1], ["g", "h", 3]], columns=list("GHE"))
        right = DataFrame([[2, 1]], columns=list("ET"))
        result = merge_ordered(left, right, on="E", left_by=["G", "H"])
        expected = DataFrame(
            {"G": ["g"] * 3, "H": ["h"] * 3, "E": [1, 2, 3], "T": [np.nan, 1.0, np.nan]}
        )

        tm.assert_frame_equal(result, expected)

    def test_elements_not_in_by_but_in_df(self):
        # GH 38167
        # [FUNCTION DOC]
        # GH 38167. A left_by entry that is not a column of the left frame raises a key error naming the missing column.
        left = DataFrame([["g", "h", 1], ["g", "h", 3]], columns=list("GHE"))
        right = DataFrame([[2, 1]], columns=list("ET"))
        msg = r"\{'h'\} not found in left columns"
        with pytest.raises(KeyError, match=msg):
            merge_ordered(left, right, on="E", left_by=["G", "h"])

    @pytest.mark.parametrize("invalid_method", ["linear", "carrot"])
    def test_ffill_validate_fill_method(self, left, right, invalid_method):
        # GH 55884
        # [FUNCTION DOC]
        # GH 55884. Any fill_method other than forward fill or None (linear, carrot) raises a value error saying it must be ffill or None.
        with pytest.raises(
            ValueError, match=re.escape("fill_method must be 'ffill' or None")
        ):
            merge_ordered(left, right, on="key", fill_method=invalid_method)

    def test_ffill_left_merge(self):
        # GH 57010
        # [FUNCTION DOC]
        # GH 57010. A left-type ordered merge with forward fill and left_by keeps only left keys per group and fills right values forward within each group.
        df1 = DataFrame(
            {
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3, 1, 2, 3],
                "group": ["a", "a", "a", "b", "b", "b"],
            }
        )
        df2 = DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
        result = merge_ordered(
            df1, df2, fill_method="ffill", left_by="group", how="left"
        )
        expected = DataFrame(
            {
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3, 1, 2, 3],
                "group": ["a", "a", "a", "b", "b", "b"],
                "rvalue": [np.nan, 2.0, 2.0, np.nan, 2.0, 2.0],
            }
        )
        tm.assert_frame_equal(result, expected)
