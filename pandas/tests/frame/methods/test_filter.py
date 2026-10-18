# [MODULE DOC]
# Tests for the DataFrame filter method (select index or columns by explicit items, substring, or regular expression).
# Uses the shared float_frame and float_string_frame fixtures plus small hand-built frames.

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm


class TestDataFrameFilter:
    # [CLASS DOC]
    # Groups the tests of DataFrame filter: the items, like and regex selection modes, axis handling, error cases and non-ascii or non-string labels.
    def test_filter(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # Broad behavioral test of filter using the float_frame and float_string_frame fixtures.
        # Verifies: items selection drops labels that are missing (on columns by default and on the index when the axis is given); the like mode matches substrings, including with integer column labels; the regex mode matches only string labels and keeps integer-like labels matching a numeric pattern; calling with no selector, or with items set to None, or only an axis, raises a type error mentioning that one must pass an argument; supplying more than one of items, regex, like raises a type error saying they are mutually exclusive; works on object columns and on non-ascii column labels.
        # Items
        filtered = float_frame.filter(["A", "B", "E"])
        assert len(filtered.columns) == 2
        assert "E" not in filtered

        filtered = float_frame.filter(["A", "B", "E"], axis="columns")
        assert len(filtered.columns) == 2
        assert "E" not in filtered

        # Other axis
        idx = float_frame.index[0:4]
        filtered = float_frame.filter(idx, axis="index")
        expected = float_frame.reindex(index=idx)
        tm.assert_frame_equal(filtered, expected)

        # like
        fcopy = float_frame.copy()
        fcopy["AA"] = 1

        filtered = fcopy.filter(like="A")
        assert len(filtered.columns) == 2
        assert "AA" in filtered

        # like with ints in column names
        df = DataFrame(0.0, index=[0, 1, 2], columns=[0, 1, "_A", "_B"])
        filtered = df.filter(like="_")
        assert len(filtered.columns) == 2

        # regex with ints in column names
        # from PR #10384
        df = DataFrame(0.0, index=[0, 1, 2], columns=["A1", 1, "B", 2, "C"])
        expected = DataFrame(
            0.0, index=[0, 1, 2], columns=pd.Index([1, 2], dtype=object)
        )
        filtered = df.filter(regex="^[0-9]+$")
        tm.assert_frame_equal(filtered, expected)

        expected = DataFrame(0.0, index=[0, 1, 2], columns=[0, "0", 1, "1"])
        # shouldn't remove anything
        filtered = expected.filter(regex="^[0-9]+$")
        tm.assert_frame_equal(filtered, expected)

        # pass in None
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter()
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter(items=None)
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter(axis=1)

        # test mutually exclusive arguments
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$", like="bbi")
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$", axis=1)
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$")
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], like="bbi", axis=0)
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], like="bbi")

        # objects
        filtered = float_string_frame.filter(like="foo")
        assert "foo" in filtered

        # unicode columns, won't ascii-encode
        df = float_frame.rename(columns={"B": "\u2202"})
        filtered = df.filter(like="C")
        assert "C" in filtered

    def test_filter_regex_search(self, float_frame):
        # [FUNCTION DOC]
        # Verifies the regex mode uses a search, not a full match: a pattern matches labels where it occurs anywhere, not only at the start.
        # Setup: float_frame copy with an extra column AA, plus a frame with labels containing BB at various positions. Expected: only matching columns are returned, equal to a manual column selection.
        fcopy = float_frame.copy()
        fcopy["AA"] = 1

        # regex
        filtered = fcopy.filter(regex="[A]+")
        assert len(filtered.columns) == 2
        assert "AA" in filtered

        # doesn't have to be at beginning
        df = DataFrame(
            {"aBBa": [1, 2], "BBaBB": [1, 2], "aCCa": [1, 2], "aCCaBB": [1, 2]}
        )

        result = df.filter(regex="BB")
        exp = df[[x for x in df.columns if "BB" in x]]
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize(
        "name,expected_data",
        [
            ("a", {"a": [1, 2]}),
            ("あ", {"あ": [3, 4]}),
        ],
    )
    def test_filter_unicode(self, name, expected_data):
        # [FUNCTION DOC]
        # Parametrized over a name (an ascii letter or a Japanese character) with the expected single-column data.
        # Verifies the like and regex modes both select the column whose label matches non-ascii text (GH13101); result equals a frame containing only that column.
        # GH13101
        df = DataFrame({"a": [1, 2], "あ": [3, 4]})
        expected = DataFrame(expected_data)

        tm.assert_frame_equal(df.filter(like=name), expected)
        tm.assert_frame_equal(df.filter(regex=name), expected)

    def test_filter_bytestring(self):
        # [FUNCTION DOC]
        # Verifies like and regex selection work with bytes column labels (GH13101): the string pattern matches the bytes label, returning only that column.
        # GH13101
        name = "a"
        df = DataFrame({b"a": [1, 2], b"b": [3, 4]})
        expected = DataFrame({b"a": [1, 2]})

        tm.assert_frame_equal(df.filter(like=name), expected)
        tm.assert_frame_equal(df.filter(regex=name), expected)

    def test_filter_corner(self):
        # [FUNCTION DOC]
        # Verifies filtering an empty frame, with an empty items list or a like pattern, returns a frame equal to the empty input.
        empty = DataFrame()

        result = empty.filter([])
        tm.assert_frame_equal(result, empty)

        result = empty.filter(like="foo")
        tm.assert_frame_equal(result, empty)

    def test_filter_regex_non_string(self):
        # [FUNCTION DOC]
        # Verifies a regex filter on columns containing a non-string label (an integer) simply drops that label instead of raising (GH#5798).
        # GH#5798 trying to filter on non-string columns should drop,
        #  not raise
        df = DataFrame(np.random.default_rng(2).random((3, 2)), columns=["STRING", 123])
        result = df.filter(regex="STRING")
        expected = df[["STRING"]]
        tm.assert_frame_equal(result, expected)

    def test_filter_keep_order(self):
        # [FUNCTION DOC]
        # Verifies items selection returns columns in the order given by the items, not the original order (GH#54980).
        # GH#54980
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.filter(items=["B", "A"])
        expected = df[["B", "A"]]
        tm.assert_frame_equal(result, expected)

    def test_filter_different_dtype(self):
        # [FUNCTION DOC]
        # Verifies items that do not match any label (string items on integer column labels) produce a frame with no columns but the original index (GH#54980).
        # GH#54980
        df = DataFrame({1: [1, 2, 3], 2: [4, 5, 6]})
        result = df.filter(items=["B", "A"])
        expected = df[[]]
        tm.assert_frame_equal(result, expected)
