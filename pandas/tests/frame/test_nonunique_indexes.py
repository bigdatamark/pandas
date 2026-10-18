# [MODULE DOC]
# Tests for the tabular container when column labels (and sometimes row labels)
# are not unique: assigning to the columns attribute, inserting, setting and
# deleting duplicated columns, mixed dtypes, comparisons, selection and
# internal block bookkeeping. No fixtures beyond the shared testing helpers; no
# import-time side effects.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameNonuniqueIndexes:
    # [CLASS DOC]
    # Test group for frames with duplicate labels. Stateless.
    def test_setattr_columns_vs_construct_with_columns(self):
        # [FUNCTION DOC]
        # GH 3687: a frame built with two identical column labels can have its columns
        # attribute replaced by unique labels; the result equals a frame built directly
        # with those labels.
        # assignment
        # GH 3687
        arr = np.random.default_rng(2).standard_normal((3, 2))
        idx = list(range(2))
        df = DataFrame(arr, columns=["A", "A"])
        df.columns = idx
        expected = DataFrame(arr, columns=idx)
        tm.assert_frame_equal(df, expected)

    def test_setattr_columns_vs_construct_with_columns_datetimeindx(self):
        # [FUNCTION DOC]
        # Replacing four identical column labels by a datetime index through the columns
        # attribute gives a frame equal to one constructed with that datetime index.
        idx = date_range("20130101", periods=4, freq="QE-NOV")
        df = DataFrame(
            [[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]], columns=["a", "a", "a", "a"]
        )
        df.columns = idx
        expected = DataFrame([[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]], columns=idx)
        tm.assert_frame_equal(df, expected)

    def test_insert_with_duplicate_columns(self):
        # [FUNCTION DOC]
        # Long scenario on a frame with a duplicated label: adding a scalar column
        # appends it; inserting with a too-short value sequence raises a value error
        # mentioning the length; setting a new then an existing non-duplicated column;
        # deleting non-duplicated columns (including after a block is non-consolidated)
        # and consolidating; inserting a column in the middle; inserting an existing
        # label raises a value error unless duplicates are allowed; finally deleting a
        # duplicated label removes every column carrying it. Each step compares to an
        # expected frame.
        # insert
        df = DataFrame(
            [[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]],
            columns=["foo", "bar", "foo", "hello"],
        )
        df["string"] = "bah"
        expected = DataFrame(
            [[1, 1, 1, 5, "bah"], [1, 1, 2, 5, "bah"], [2, 1, 3, 5, "bah"]],
            columns=["foo", "bar", "foo", "hello", "string"],
        )
        tm.assert_frame_equal(df, expected)
        with pytest.raises(ValueError, match="Length of value"):
            df.insert(0, "AnotherColumn", range(len(df.index) - 1))

        # insert same dtype
        df["foo2"] = 3
        expected = DataFrame(
            [[1, 1, 1, 5, "bah", 3], [1, 1, 2, 5, "bah", 3], [2, 1, 3, 5, "bah", 3]],
            columns=["foo", "bar", "foo", "hello", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # set (non-dup)
        df["foo2"] = 4
        expected = DataFrame(
            [[1, 1, 1, 5, "bah", 4], [1, 1, 2, 5, "bah", 4], [2, 1, 3, 5, "bah", 4]],
            columns=["foo", "bar", "foo", "hello", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)
        df["foo2"] = 3

        # delete (non dup)
        del df["bar"]
        expected = DataFrame(
            [[1, 1, 5, "bah", 3], [1, 2, 5, "bah", 3], [2, 3, 5, "bah", 3]],
            columns=["foo", "foo", "hello", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # try to delete again (its not consolidated)
        del df["hello"]
        expected = DataFrame(
            [[1, 1, "bah", 3], [1, 2, "bah", 3], [2, 3, "bah", 3]],
            columns=["foo", "foo", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # consolidate
        df = df._consolidate()
        expected = DataFrame(
            [[1, 1, "bah", 3], [1, 2, "bah", 3], [2, 3, "bah", 3]],
            columns=["foo", "foo", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # insert
        df.insert(2, "new_col", 5.0)
        expected = DataFrame(
            [[1, 1, 5.0, "bah", 3], [1, 2, 5.0, "bah", 3], [2, 3, 5.0, "bah", 3]],
            columns=["foo", "foo", "new_col", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # insert a dup
        with pytest.raises(ValueError, match="cannot insert"):
            df.insert(2, "new_col", 4.0)

        df.insert(2, "new_col", 4.0, allow_duplicates=True)
        expected = DataFrame(
            [
                [1, 1, 4.0, 5.0, "bah", 3],
                [1, 2, 4.0, 5.0, "bah", 3],
                [2, 3, 4.0, 5.0, "bah", 3],
            ],
            columns=["foo", "foo", "new_col", "new_col", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # delete (dup)
        del df["foo"]
        expected = DataFrame(
            [[4.0, 5.0, "bah", 3], [4.0, 5.0, "bah", 3], [4.0, 5.0, "bah", 3]],
            columns=["new_col", "new_col", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

    def test_dup_across_dtypes(self):
        # [FUNCTION DOC]
        # Duplicated labels spanning integer and float blocks: adding a float column,
        # selecting the duplicated label (returns both columns), replacing all columns
        # with the label by a string (both replaced), and deleting the label.
        # dup across dtypes
        df = DataFrame(
            [[1, 1, 1.0, 5], [1, 1, 2.0, 5], [2, 1, 3.0, 5]],
            columns=["foo", "bar", "foo", "hello"],
        )

        df["foo2"] = 7.0
        expected = DataFrame(
            [[1, 1, 1.0, 5, 7.0], [1, 1, 2.0, 5, 7.0], [2, 1, 3.0, 5, 7.0]],
            columns=["foo", "bar", "foo", "hello", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        result = df["foo"]
        expected = DataFrame([[1, 1.0], [1, 2.0], [2, 3.0]], columns=["foo", "foo"])
        tm.assert_frame_equal(result, expected)

        # multiple replacements
        df["foo"] = "string"
        expected = DataFrame(
            [
                ["string", 1, "string", 5, 7.0],
                ["string", 1, "string", 5, 7.0],
                ["string", 1, "string", 5, 7.0],
            ],
            columns=["foo", "bar", "foo", "hello", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        del df["foo"]
        expected = DataFrame(
            [[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=["bar", "hello", "foo2"]
        )
        tm.assert_frame_equal(df, expected)

    def test_column_dups_indexes(self):
        # [FUNCTION DOC]
        # Assigning an index to a duplicated column label, both when the assigned index
        # equals the frame's index and when it is the reversed order, sets every column
        # with that label; the assigned values are used positionally (no alignment since
        # a plain index is not aligned) while other columns are untouched.
        # check column dups with index equal and not equal to df's index
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["A", "B", "A"],
        )
        for index in [df.index, pd.Index(list("edcba"))]:
            this_df = df.copy()
            expected_ser = Series(index.values, index=this_df.index)
            expected_df = DataFrame(
                {"A": expected_ser, "B": this_df["B"]},
                columns=["A", "B", "A"],
            )
            this_df["A"] = index
            tm.assert_frame_equal(this_df, expected_df)

    def test_changing_dtypes_with_duplicate_columns(self):
        # [FUNCTION DOC]
        # GH 6120: assigning a scalar to a duplicated label changes dtype of all the
        # matching columns (float scalar into float data, integer scalar into float data
        # results in integer columns), even when the location is a slice.
        # multiple assignments that change dtypes
        # the location indexer is a slice
        # GH 6120
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=["that", "that"]
        )
        expected = DataFrame(1.0, index=range(5), columns=["that", "that"])

        df["that"] = 1.0
        tm.assert_frame_equal(df, expected)

        df = DataFrame(
            np.random.default_rng(2).random((5, 2)), columns=["that", "that"]
        )
        expected = DataFrame(1, index=range(5), columns=["that", "that"])

        df["that"] = 1
        tm.assert_frame_equal(df, expected)

    def test_dup_columns_comparisons(self):
        # [FUNCTION DOC]
        # Comparing frames whose labels differ raises a value error about identically
        # labeled frames; after reindexing the first like the second (which has duplicate
        # columns), equality works and gives the expected boolean frame.
        # equality
        df1 = DataFrame([[1, 2], [2, np.nan], [3, 4], [4, 4]], columns=["A", "B"])
        df2 = DataFrame([[0, 1], [2, 4], [2, np.nan], [4, 5]], columns=["A", "A"])

        # not-comparing like-labelled
        msg = (
            r"Can only compare identically-labeled \(both index and columns\) "
            "DataFrame objects"
        )
        with pytest.raises(ValueError, match=msg):
            df1 == df2

        df1r = df1.reindex_like(df2)
        result = df1r == df2
        expected = DataFrame(
            [[False, True], [True, False], [False, False], [True, False]],
            columns=["A", "A"],
        )
        tm.assert_frame_equal(result, expected)

    def test_mixed_column_selection(self):
        # [FUNCTION DOC]
        # GH 5639: selecting a list of columns with a repeated label from a frame
        # matches concatenating those columns along the column axis, preserving order
        # and duplication.
        # mixed column selection
        # GH 5639
        dfbool = DataFrame(
            {
                "one": Series([True, True, False], index=["a", "b", "c"]),
                "two": Series([False, False, True, False], index=["a", "b", "c", "d"]),
                "three": Series([False, True, True, True], index=["a", "b", "c", "d"]),
            }
        )
        expected = pd.concat([dfbool["one"], dfbool["three"], dfbool["one"]], axis=1)
        result = dfbool[["one", "three", "one"]]
        tm.assert_frame_equal(result, expected)

    def test_multi_axis_dups(self):
        # [FUNCTION DOC]
        # GH 6121: a frame with duplicated labels on both axes (selected duplicate
        # columns and then duplicated row labels) gives the same result whether the
        # column selection was copied or not.
        # multi-axis dups
        # GH 6121
        df = DataFrame(
            np.arange(25.0).reshape(5, 5),
            index=["a", "b", "c", "d", "e"],
            columns=["A", "B", "C", "D", "E"],
        )
        z = df[["A", "C", "A"]].copy()
        expected = z.loc[["a", "c", "a"]]

        df = DataFrame(
            np.arange(25.0).reshape(5, 5),
            index=["a", "b", "c", "d", "e"],
            columns=["A", "B", "C", "D", "E"],
        )
        z = df[["A", "C", "A"]]
        result = z.loc[["a", "c", "a"]]
        tm.assert_frame_equal(result, expected)

    def test_columns_with_dups(self):
        # [FUNCTION DOC]
        # GH 3468 related: replacing duplicated column labels by unique ones (suffixed
        # names) through the columns attribute works for two- and three-column
        # single-row frames.
        # GH 3468 related

        # basic
        df = DataFrame([[1, 2]], columns=["a", "a"])
        df.columns = ["a", "a.1"]
        expected = DataFrame([[1, 2]], columns=["a", "a.1"])
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[1, 2, 3]], columns=["b", "a", "a"])
        df.columns = ["b", "a", "a.1"]
        expected = DataFrame([[1, 2, 3]], columns=["b", "a", "a.1"])
        tm.assert_frame_equal(df, expected)

    def test_columns_with_dup_index(self):
        # [FUNCTION DOC]
        # Relabeling columns that were duplicated with a different duplicated label via
        # the columns attribute works.
        # with a dup index
        df = DataFrame([[1, 2]], columns=["a", "a"])
        df.columns = ["b", "b"]
        expected = DataFrame([[1, 2]], columns=["b", "b"])
        tm.assert_frame_equal(df, expected)

    def test_multi_dtype(self):
        # [FUNCTION DOC]
        # Replacing the columns attribute on a single-row frame that has duplicated
        # labels across integer, float and string data with a unique label list gives
        # the same frame as one built with those labels.
        # multi-dtype
        df = DataFrame(
            [[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]],
            columns=["a", "a", "b", "b", "d", "c", "c"],
        )
        df.columns = list("ABCDEFG")
        expected = DataFrame(
            [[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]], columns=list("ABCDEFG")
        )
        tm.assert_frame_equal(df, expected)

    def test_multi_dtype2(self):
        # [FUNCTION DOC]
        # Same relabeling as the multi-dtype test for four duplicated labels holding
        # integer and string values.
        df = DataFrame([[1, 2, "foo", "bar"]], columns=["a", "a", "a", "a"])
        df.columns = ["a", "a.1", "a.2", "a.3"]
        expected = DataFrame([[1, 2, "foo", "bar"]], columns=["a", "a.1", "a.2", "a.3"])
        tm.assert_frame_equal(df, expected)

    def test_dups_across_blocks(self):
        # [FUNCTION DOC]
        # Concatenating float, integer, boolean, object and datetime frames along columns
        # gives an internal manager whose per-column block number and block location
        # arrays each have one entry per column; positional column access with the
        # integer location accessor works for every column (no assertion on values).
        # dups across blocks
        df_float = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), dtype="float64"
        )
        df_int = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)).astype("int64")
        )
        df_bool = DataFrame(True, index=df_float.index, columns=df_float.columns)
        df_object = DataFrame("foo", index=df_float.index, columns=df_float.columns)
        df_dt = DataFrame(
            pd.Timestamp("20010101"), index=df_float.index, columns=df_float.columns
        )
        df = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1)

        assert len(df._mgr.blknos) == len(df.columns)
        assert len(df._mgr.blklocs) == len(df.columns)

        # testing iloc
        for i in range(len(df.columns)):
            df.iloc[:, i]

    def test_dup_columns_across_dtype(self):
        # [FUNCTION DOC]
        # GH 2079/2194: constructing from a list of rows with duplicated labels across
        # integer and float data equals constructing without labels then assigning the
        # duplicated labels.
        # dup columns across dtype GH 2079/2194
        vals = [[1, -1, 2.0], [2, -2, 3.0]]
        rs = DataFrame(vals, columns=["A", "A", "B"])
        xp = DataFrame(vals)
        xp.columns = ["A", "A", "B"]
        tm.assert_frame_equal(rs, xp)

    def test_set_value_by_index(self):
        # [FUNCTION DOC]
        # GH-12344: assigning a value to a column by position on a frame with duplicate
        # (or mixed-type but equal-looking) labels must not alter the other columns. The
        # expected warning is None (no warning expected), so the match message is
        # effectively unused.
        # See gh-12344
        warn = None
        msg = "will attempt to set the values inplace"

        df = DataFrame(np.arange(9).reshape(3, 3).T)
        df.columns = list("AAA")
        expected = df.iloc[:, 2].copy()

        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[:, 0] = 3
        tm.assert_series_equal(df.iloc[:, 2], expected)

        df = DataFrame(np.arange(9).reshape(3, 3).T)
        df.columns = [2, float(2), str(2)]
        expected = df.iloc[:, 1].copy()

        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[:, 0] = 3
        tm.assert_series_equal(df.iloc[:, 1], expected)
