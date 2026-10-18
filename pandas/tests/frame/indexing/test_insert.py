"""
test_insert is specifically for the DataFrame.insert method; not to be
confused with tests with "insert" in their names that are really testing
__setitem__.
"""
# [MODULE DOC]
# Tests for the DataFrame insert method specifically (not setitem operations that merely have
# insert in their names). Covers position handling, duplicate names, name preservation, error
# messages and regression cases. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    NaT,
    Timestamp,
)
import pandas._testing as tm


class TestDataFrameInsert:
    # [CLASS DOC]
    # Test class for the insert method.
    def test_insert(self):
        # [FUNCTION DOC]
        # Verifies inserting copies of existing columns at the start and middle places them at the given
        # position and values match their source. Inserting an existing name raises a value error saying
        # it already exists (message includes a cannot-insert text). The columns index name is preserved
        # by insertion.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=np.arange(5),
            columns=["c", "b", "a"],
        )

        df.insert(0, "foo", df["a"])
        tm.assert_index_equal(df.columns, Index(["foo", "c", "b", "a"]))
        tm.assert_series_equal(df["a"], df["foo"], check_names=False)

        df.insert(2, "bar", df["c"])
        tm.assert_index_equal(df.columns, Index(["foo", "c", "bar", "b", "a"]))
        tm.assert_almost_equal(df["c"], df["bar"], check_names=False)

        with pytest.raises(ValueError, match="already exists"):
            df.insert(1, "a", df["b"])

        msg = "cannot insert c, already exists"
        with pytest.raises(ValueError, match=msg):
            df.insert(1, "c", df["b"])

        df.columns.name = "some_name"
        # preserve columns name field
        df.insert(0, "baz", df["c"])
        assert df.columns.name == "some_name"

    def test_insert_column_bug_4032(self):
        # [FUNCTION DOC]
        # Regression test for GH#4032: inserting a new column at the front, with renames by an empty
        # mapping before and after, gives the expected frame with the columns in order, both for an
        # integer column and then a float column.
        # GH#4032, inserting a column and renaming causing errors
        df = DataFrame({"b": [1.1, 2.2]})

        df = df.rename(columns={})
        df.insert(0, "a", [1, 2])
        result = df.rename(columns={})

        expected = DataFrame([[1, 1.1], [2, 2.2]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

        df.insert(0, "c", [1.3, 2.3])
        result = df.rename(columns={})

        expected = DataFrame([[1.3, 1, 1.1], [2.3, 2, 2.2]], columns=["c", "a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_insert_with_columns_dups(self):
        # [FUNCTION DOC]
        # Regression test for GH#14291: inserting a column named A three times at position zero with
        # duplicates allowed gives three A columns with the latest inserted first.
        # GH#14291
        df = DataFrame()
        df.insert(0, "A", ["g", "h", "i"], allow_duplicates=True)
        df.insert(0, "A", ["d", "e", "f"], allow_duplicates=True)
        df.insert(0, "A", ["a", "b", "c"], allow_duplicates=True)
        exp = DataFrame(
            [["a", "d", "g"], ["b", "e", "h"], ["c", "f", "i"]], columns=["A", "A", "A"]
        )
        tm.assert_frame_equal(df, exp)

    def test_insert_EA_no_warning(self):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the name refers to insert but the operation under test is an item assignment.
        # Verifies that assigning a new numpy column to a frame of masked 64-bit integer columns does not
        # emit any warning (no fragmentation performance warning for extension arrays, GH 44098).
        # PerformanceWarning about fragmented frame should not be raised when
        # using EAs (https://github.com/pandas-dev/pandas/issues/44098)
        df = DataFrame(
            np.random.default_rng(2).integers(0, 100, size=(3, 100)), dtype="Int64"
        )
        with tm.assert_produces_warning(None):
            df["a"] = np.array([1, 2, 3])

    def test_insert_frame(self):
        # [FUNCTION DOC]
        # Regression test for GH#42403: inserting a multi-column frame as a column raises a value error
        # stating a one-dimensional object was expected and a frame with two columns was given.
        # GH#42403
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]})

        msg = (
            "Expected a one-dimensional object, got a DataFrame with 2 columns instead."
        )
        with pytest.raises(ValueError, match=msg):
            df.insert(1, "newcol", df)

    def test_insert_int64_loc(self):
        # [FUNCTION DOC]
        # Regression test for GH#53193: insert accepts a numpy 64-bit integer position, adding the
        # scalar broadcast column at the front.
        # GH#53193
        df = DataFrame({"a": [1, 2]})
        df.insert(np.int64(0), "b", 0)
        tm.assert_frame_equal(df, DataFrame({"b": [0, 0], "a": [1, 2]}))

    def test_insert_delete_mixed_multiindex_columns(self):
        # [FUNCTION DOC]
        # Regression test for GH#56853: on a frame with a two-level column label containing a timestamp,
        # inserting a plain column yields a multi-index whose second element is missing (NaT); deleting
        # that column afterwards must not raise a recursion error and gives back the original frame.
        # GH#56853

        df = DataFrame({("A", Timestamp("2024-01-01")): [0]})
        df.insert(1, "B", [1])

        expected = DataFrame(
            [[0, 1]],
            columns=MultiIndex.from_tuples(
                [("A", Timestamp("2024-01-01")), ("B", NaT)]
            ),
        )
        tm.assert_frame_equal(df, expected)

        # Should not raise RecursionError (this was the original bug)
        del df["B"]

        expected = DataFrame({("A", Timestamp("2024-01-01")): [0]})
        tm.assert_frame_equal(df, expected)
