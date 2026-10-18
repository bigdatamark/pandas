# [MODULE DOC]
# Tests for deleting columns from a DataFrame with the delete-item operation, including
# multi-index columns and missing keys. Uses the float_frame fixture. No import-time side effects.
import re

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)


class TestDataFrameDelItem:
    # [CLASS DOC]
    # Test class for column deletion on DataFrames.
    def test_delitem(self, float_frame):
        # [FUNCTION DOC]
        # Verifies deleting the column named A from the float frame removes it from the frame.
        del float_frame["A"]
        assert "A" not in float_frame

    def test_delitem_multiindex(self):
        # [FUNCTION DOC]
        # Verifies deleting a top-level key of a two-level column frame removes all its sub-columns
        # (four columns become two) after first checking the key is contained and selecting it returns a
        # frame. After deletion, the key is no longer contained (behavior changed from GH 2770 to GH 19027),
        # and deleting it again, bare or as a one-element tuple, raises a key error naming the tuple.
        midx = MultiIndex.from_product([["A", "B"], [1, 2]])
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=midx)
        assert len(df.columns) == 4
        assert ("A",) in df.columns
        assert "A" in df.columns

        result = df["A"]
        assert isinstance(result, DataFrame)
        del df["A"]

        assert len(df.columns) == 2

        # A still in the levels, BUT get a KeyError if trying
        # to delete
        assert ("A",) not in df.columns
        with pytest.raises(KeyError, match=re.escape("('A',)")):
            del df[("A",)]

        # behavior of dropped/deleted MultiIndex levels changed from
        # GH 2770 to GH 19027: MultiIndex no longer '.__contains__'
        # levels which are dropped/deleted
        assert "A" not in df.columns
        with pytest.raises(KeyError, match=re.escape("('A',)")):
            del df["A"]

    def test_delitem_corner(self, float_frame):
        # [FUNCTION DOC]
        # Verifies on a copy of the float frame that deleting column D removes it, deleting it again raises
        # a key error whose message is exactly the quoted label, and deleting B leaves two columns.
        f = float_frame.copy()
        del f["D"]
        assert len(f.columns) == 3
        with pytest.raises(KeyError, match=r"^'D'$"):
            del f["D"]
        del f["B"]
        assert len(f.columns) == 2

    def test_delitem_col_still_multiindex(self):
        # [FUNCTION DOC]
        # Verifies that deleting one full-depth tuple key from a three-level column frame (padded with
        # empty strings) leaves the columns as a multi-index.
        arrays = [["a", "b", "c", "top"], ["", "", "", "OD"], ["", "", "", "wx"]]

        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)

        df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), columns=index)
        del df[("a", "", "")]
        assert isinstance(df.columns, MultiIndex)
