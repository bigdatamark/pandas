"""
test setting *parts* of objects both positionally and label based

TODO: these should be split among the indexer tests
"""

# [MODULE DOC]
# Tests for setting parts of series and frames by position and by label, with an emphasis on enlargement (setting with expansion), setting on empty frames, partial selection with missing keys, and label lists given as strings for date-like indexes. A module note says these tests should eventually be split among the indexer-specific modules.
# Classes: expansion of empty frames, partial setting on non-empty objects, and string slicing on date-like indexes. Uses the shared frame-or-series fixture and the infer-string fixture.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Period,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestEmptyFrameSetitemExpansion:
    # [CLASS DOC]
    # Tests for assigning columns or rows into frames that start empty (no rows, no columns or both), verifying index handling, dtype of the new data and error messages.
    def test_empty_frame_setitem_index_name_retained(self):
        # [FUNCTION DOC]
        # Verifies that assigning a series to a new column of a frame with an empty named range index keeps the frame's index name and takes the series' length (issue 31368).
        # GH#31368 empty frame has non-None index.name -> retained
        df = DataFrame({}, index=pd.RangeIndex(0, name="df_index"))
        series = Series(1.23, index=pd.RangeIndex(4, name="series_index"))

        df["series"] = series
        expected = DataFrame(
            {"series": [1.23] * 4},
            index=pd.RangeIndex(4, name="df_index"),
            columns=Index(["series"]),
        )

        tm.assert_frame_equal(df, expected)

    def test_empty_frame_setitem_index_name_inherited(self):
        # [FUNCTION DOC]
        # Verifies that assigning a series into a completely empty frame (no index name) makes the frame take the series' index including its name (issue 36527).
        # GH#36527 empty frame has None index.name -> not retained
        df = DataFrame()
        series = Series(1.23, index=pd.RangeIndex(4, name="series_index"))
        df["series"] = series
        expected = DataFrame(
            {"series": [1.23] * 4},
            index=pd.RangeIndex(4, name="series_index"),
            columns=Index(["series"]),
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_zerolen_series_columns_align(self):
        # [FUNCTION DOC]
        # Verifies row assignment of a series into a frame with two columns and no rows aligns on columns.
        # A series with unrelated integer labels leaves the new row with missing values in both columns (float dtype); a series labelled with only the second column name sets just that column and leaves the first missing.
        # columns will align
        df = DataFrame(columns=["A", "B"])
        df.loc[0] = Series(1, index=range(4))
        expected = DataFrame(columns=["A", "B"], index=[0], dtype=np.float64)
        tm.assert_frame_equal(df, expected)

        # columns will align
        df = DataFrame(columns=["A", "B"])
        df.loc[0] = Series(1, index=["B"])

        exp = DataFrame([[np.nan, 1]], columns=["A", "B"], index=[0], dtype="float64")
        tm.assert_frame_equal(df, exp)

    def test_loc_setitem_zerolen_list_length_must_match_columns(self):
        # [FUNCTION DOC]
        # Verifies that assigning a list as a new row to a frame with columns but no rows needs the list length to equal the column count: a longer list raises a value error saying the row has mismatched columns; a list of the right length creates the row at that label with integer dtype.
        # list-like must conform
        df = DataFrame(columns=["A", "B"])

        msg = "cannot set a row with mismatched columns"
        with pytest.raises(ValueError, match=msg):
            df.loc[0] = [1, 2, 3]

        df = DataFrame(columns=["A", "B"])
        df.loc[3] = [6, 7]  # length matches len(df.columns) --> OK!

        exp = DataFrame([[6, 7]], index=[3], columns=["A", "B"], dtype=np.int64)
        tm.assert_frame_equal(df, exp)

    def test_partial_set_empty_frame(self):
        # [FUNCTION DOC]
        # Verifies error cases when label-assigning into a frame with no columns: assigning a scalar or a series to a row label raises a value error saying a frame with no defined columns cannot be set; assigning a scalar to a column label across all rows raises a value error saying a frame with no defined index cannot be set with a scalar.
        # partially set with an empty object
        # frame
        df = DataFrame()

        msg = "cannot set a frame with no defined columns"

        with pytest.raises(ValueError, match=msg):
            df.loc[1] = 1

        with pytest.raises(ValueError, match=msg):
            df.loc[1] = Series([1], index=["foo"])

        msg = "cannot set a frame with no defined index and a scalar"
        with pytest.raises(ValueError, match=msg):
            df.loc[:, 1] = 1

    def test_partial_set_empty_frame2(self):
        # [FUNCTION DOC]
        # Verifies that adding a column to a frame with an empty index works when it only changes the columns (issue 5632): an empty object series, a series built from the empty index and the empty index itself all give the same expected frame with one object column and an empty object index.
        # these work as they don't really change
        # anything but the index
        # GH#5632
        expected = DataFrame(columns=Index(["foo"]), index=Index([], dtype="object"))

        df = DataFrame(index=Index([], dtype="object"))
        df["foo"] = Series([], dtype="object")

        tm.assert_frame_equal(df, expected)

        df = DataFrame(index=Index([]))
        df["foo"] = Series(df.index)

        tm.assert_frame_equal(df, expected)

        df = DataFrame(index=Index([]))
        df["foo"] = df.index

        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame3(self):
        # [FUNCTION DOC]
        # Verifies dtype of a column added to a frame with an empty integer index: an empty list and an empty float series both give a float64 column.
        expected = DataFrame(columns=Index(["foo"]), index=Index([], dtype="int64"))
        expected["foo"] = expected["foo"].astype("float64")

        df = DataFrame(index=Index([], dtype="int64"))
        df["foo"] = []

        tm.assert_frame_equal(df, expected)

        df = DataFrame(index=Index([], dtype="int64"))
        df["foo"] = Series(np.arange(len(df)), dtype="float64")

        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame4(self):
        # [FUNCTION DOC]
        # Verifies assigning an empty range to a new column of a frame with an empty integer index gives an int64 column.
        df = DataFrame(index=Index([], dtype="int64"))
        df["foo"] = range(len(df))

        expected = DataFrame(columns=Index(["foo"]), index=Index([], dtype="int64"))
        # range is int-dtype-like, so we get int64 dtype
        expected["foo"] = expected["foo"].astype("int64")
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame5(self):
        # [FUNCTION DOC]
        # Verifies that an empty frame has a zero-length range as columns, and that setting a column by label over all rows with a one-element series (indexed by a string) gives the same frame as plain column assignment: one row indexed by that string and column labelled one.
        df = DataFrame()
        tm.assert_index_equal(df.columns, pd.RangeIndex(0))
        df2 = DataFrame()
        df2[1] = Series([1], index=["foo"])
        df.loc[:, 1] = Series([1], index=["foo"])
        tm.assert_frame_equal(df, DataFrame([[1]], index=["foo"], columns=[1]))
        tm.assert_frame_equal(df, df2)

    def test_partial_set_empty_frame_no_index(self):
        # [FUNCTION DOC]
        # Verifies adding a new column labelled zero to a frame that has two columns and no index, by plain assignment or all-rows label assignment, from a constant series of four rows: the result has four rows with the old columns missing.
        # no index to start
        expected = DataFrame({0: Series(1, index=range(4))}, columns=["A", "B", 0])

        df = DataFrame(columns=["A", "B"])
        df[0] = Series(1, index=range(4))
        tm.assert_frame_equal(df, expected)

        df = DataFrame(columns=["A", "B"])
        df.loc[:, 0] = Series(1, index=range(4))
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_row(self):
        # [FUNCTION DOC]
        # Verifies that no rows are created when assigning to empty frames (issues 5720 and 5744).
        # Cases: adding a column to an empty row selection keeps the empty frame with an integer first column, float second and a float new column; adding a scalar column to a frame with columns but no rows gives an integer column without rows while the text-named column remains an empty object series; reindexing columns of an empty selection keeps dtypes and adds a float column.
        # GH#5720, GH#5744
        # don't create rows when empty
        expected = DataFrame(columns=["A", "B", "New"], index=Index([], dtype="int64"))
        expected["A"] = expected["A"].astype("int64")
        expected["B"] = expected["B"].astype("float64")
        expected["New"] = expected["New"].astype("float64")

        df = DataFrame({"A": [1, 2, 3], "B": [1.2, 4.2, 5.2]})
        y = df[df.A > 5]
        y["New"] = np.nan
        tm.assert_frame_equal(y, expected)

        expected = DataFrame(columns=["a", "b", "c c", "d"])
        expected["d"] = expected["d"].astype("int64")
        df = DataFrame(columns=["a", "b", "c c"])
        df["d"] = 3
        tm.assert_frame_equal(df, expected)
        tm.assert_series_equal(df["c c"], Series(name="c c", dtype=object))

        # reindex columns is ok
        df = DataFrame({"A": [1, 2, 3], "B": [1.2, 4.2, 5.2]})
        y = df[df.A > 5]
        result = y.reindex(columns=["A", "B", "C"])
        expected = DataFrame(columns=["A", "B", "C"])
        expected["A"] = expected["A"].astype("int64")
        expected["B"] = expected["B"].astype("float64")
        expected["C"] = expected["C"].astype("float64")
        tm.assert_frame_equal(result, expected)

    def test_partial_set_empty_frame_set_series(self):
        # [FUNCTION DOC]
        # Verifies a frame built from an empty object series has one column (labelled zero or by the series name) and no rows (issue 5756).
        # GH#5756
        # setting with empty Series
        df = DataFrame(Series(dtype=object))
        expected = DataFrame({0: Series(dtype=object)})
        tm.assert_frame_equal(df, expected)

        df = DataFrame(Series(name="foo", dtype=object))
        expected = DataFrame({"foo": Series(dtype=object)})
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_copy_assignment(self):
        # [FUNCTION DOC]
        # Verifies that copying a frame with an index but no columns and then assigning a scalar column works and fills the existing row (issue 5932).
        # GH#5932
        # copy on empty with assignment fails
        df = DataFrame(index=[0])
        df = df.copy()
        df["a"] = 0
        expected = DataFrame(0, index=[0], columns=Index(["a"]))
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_consistencies(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies consistency of dtype when filling an empty frame with columns (issue 6171), parametrized by the infer-string fixture.
        # Cases: assigning a list of integers to one column leaves the other column missing (dtypes not compared); assigning a list of strings gives a string-typed (or object when inference is off) column and an object missing column; label-assigning a single cell in a new row gives one row with the other column missing (dtypes not compared).
        # GH#6171
        # consistency on empty frames
        df = DataFrame(columns=["x", "y"])
        df["x"] = [1, 2]
        expected = DataFrame({"x": [1, 2], "y": [np.nan, np.nan]})
        tm.assert_frame_equal(df, expected, check_dtype=False)

        df = DataFrame(columns=["x", "y"])
        df["x"] = ["1", "2"]
        expected = DataFrame(
            {
                "x": Series(
                    ["1", "2"],
                    dtype=object if not using_infer_string else "str",
                ),
                "y": Series([np.nan, np.nan], dtype=object),
            }
        )
        tm.assert_frame_equal(df, expected)

        df = DataFrame(columns=["x", "y"])
        df.loc[0, "x"] = 1
        expected = DataFrame({"x": [1], "y": [np.nan]})
        tm.assert_frame_equal(df, expected, check_dtype=False)


class TestPartialSetting:
    # [CLASS DOC]
    # Tests for setting with enlargement and partial selection on non-empty series and frames, and for label lists on date-like indexes.
    def test_partial_setting(self):
        # [FUNCTION DOC]
        # Verifies enlargement of a series by setting a missing label (issue 2578): via direct and label-based setting with an integer or float value, the series gets the new label at the end; the float value upcasts. Position-based setting out of range raises an index error saying positional setting cannot enlarge the object, and scalar position setting raises an index error about the position being out of bounds.
        # GH2578, allow ix and friends to partially set

        # series
        s_orig = Series([1, 2, 3])

        s = s_orig.copy()
        s[5] = 5
        expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        s.loc[5] = 5
        expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        s[5] = 5.0
        expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        s.loc[5] = 5.0
        expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        # iloc/iat raise
        s = s_orig.copy()

        msg = "iloc cannot enlarge its target object"
        with pytest.raises(IndexError, match=msg):
            s.iloc[3] = 5.0

        msg = "index 3 is out of bounds for axis 0 with size 3"
        with pytest.raises(IndexError, match=msg):
            s.iat[3] = 5.0

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_partial_setting_frame(self):
        # [FUNCTION DOC]
        # Verifies partial setting on frames; ignores the future warning about setting a value on a view.
        # Position-based setting out of range raises an index error (cannot enlarge; out of bounds on the scalar accessor). Row setting where it exists by position and label overwrites the row; label setting of a new row appends it and preserves integer dtypes; whole-column label setting on a single-dtype frame overwrites; on a mixed frame (second column converted to float first) it sets in place keeping the float dtype; setting a new column by label adds it.
        # KNOWN ISSUE: the final block labelled as a mixed frame uses exactly the same steps as the single-dtype partial setting block just before it (the frame is reset from the original, which has a single dtype), so it checks nothing new.
        df_orig = DataFrame(
            np.arange(6).reshape(3, 2), columns=["A", "B"], dtype="int64"
        )

        # iloc/iat raise
        df = df_orig.copy()

        msg = "iloc cannot enlarge its target object"
        with pytest.raises(IndexError, match=msg):
            df.iloc[4, 2] = 5.0

        msg = "index 2 is out of bounds for axis 0 with size 2"
        with pytest.raises(IndexError, match=msg):
            df.iat[4, 2] = 5.0

        # row setting where it exists
        expected = DataFrame({"A": [0, 4, 4], "B": [1, 5, 5]})
        df = df_orig.copy()
        df.iloc[1] = df.iloc[2]
        tm.assert_frame_equal(df, expected)

        expected = DataFrame({"A": [0, 4, 4], "B": [1, 5, 5]})
        df = df_orig.copy()
        df.loc[1] = df.loc[2]
        tm.assert_frame_equal(df, expected)

        # like 2578, partial setting with dtype preservation
        expected = DataFrame({"A": [0, 2, 4, 4], "B": [1, 3, 5, 5]})
        df = df_orig.copy()
        df.loc[3] = df.loc[2]
        tm.assert_frame_equal(df, expected)

        # single dtype frame, overwrite
        expected = DataFrame({"A": [0, 2, 4], "B": [0, 2, 4]})
        df = df_orig.copy()
        df.loc[:, "B"] = df.loc[:, "A"]
        tm.assert_frame_equal(df, expected)

        # mixed dtype frame, overwrite
        expected = DataFrame({"A": [0, 2, 4], "B": Series([0.0, 2.0, 4.0])})
        df = df_orig.copy()
        df["B"] = df["B"].astype(np.float64)
        # as of 2.0, df.loc[:, "B"] = ... attempts (and here succeeds) at
        #  setting inplace
        df.loc[:, "B"] = df.loc[:, "A"]
        tm.assert_frame_equal(df, expected)

        # single dtype frame, partial setting
        expected = df_orig.copy()
        expected["C"] = df["A"]
        df = df_orig.copy()
        df.loc[:, "C"] = df.loc[:, "A"]
        tm.assert_frame_equal(df, expected)

        # mixed frame, partial setting
        expected = df_orig.copy()
        expected["C"] = df["A"]
        df = df_orig.copy()
        df.loc[:, "C"] = df.loc[:, "A"]
        tm.assert_frame_equal(df, expected)

    def test_partial_setting2(self):
        # [FUNCTION DOC]
        # Verifies enlargement of a datetime-indexed frame by label (issue 8473).
        # Setting a missing next date in an existing column by the label setter or the scalar setter gives the same frame as concatenating a one-row frame (columns sorted); setting a new column labelled zero at the next date gives the frame concatenated along columns.
        # GH 8473
        dates = date_range("1/1/2000", periods=8)
        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            index=dates,
            columns=["A", "B", "C", "D"],
        )

        expected = pd.concat(
            [df_orig, DataFrame({"A": 7}, index=dates[-1:] + dates.freq)], sort=True
        )
        df = df_orig.copy()
        df.loc[dates[-1] + dates.freq, "A"] = 7
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.at[dates[-1] + dates.freq, "A"] = 7
        tm.assert_frame_equal(df, expected)

        exp_other = DataFrame({0: 7}, index=dates[-1:] + dates.freq)
        expected = pd.concat([df_orig, exp_other], axis=1)

        df = df_orig.copy()
        df.loc[dates[-1] + dates.freq, 0] = 7
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.at[dates[-1] + dates.freq, 0] = 7
        tm.assert_frame_equal(df, expected)

    def test_partial_setting_mixed_dtype(self):
        # [FUNCTION DOC]
        # Verifies that enlarging a mixed bool/integer frame with a row copied from another keeps the column dtypes by appending (infer objects on the transposed row), equal to concatenating the row.
        # in a mixed dtype environment, try to preserve dtypes
        # by appending
        df = DataFrame([[True, 1], [False, 2]], columns=["female", "fitness"])

        s = df.loc[1].copy()
        s.name = 2
        expected = pd.concat([df, DataFrame(s).T.infer_objects()])

        df.loc[2] = df.loc[1]
        tm.assert_frame_equal(df, expected)

    def test_series_partial_set(self):
        # [FUNCTION DOC]
        # Verifies partial label lists on a series (regression from issue 4825) across many index and key combinations.
        # For each: label-list selection with any missing key raises a key error saying labels are not in the index; the same keys with reindex produce missing values at those positions; keys all present (including repeated ones) work and keep duplicates; a list in which nothing is present raises a key error saying none of the labels are in the index (message includes the index repr with integer dtype); and position-list selection with repeats works.
        # partial set with new index
        # Regression from GH4825
        ser = Series([0.1, 0.2], index=[1, 2])

        # loc equiv to .reindex
        expected = Series([np.nan, 0.2, np.nan], index=[3, 2, 3])
        with pytest.raises(KeyError, match=r"not in index"):
            ser.loc[[3, 2, 3]]

        result = ser.reindex([3, 2, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)

        expected = Series([np.nan, 0.2, np.nan, np.nan], index=[3, 2, 3, "x"])
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[3, 2, 3, "x"]]

        result = ser.reindex([3, 2, 3, "x"])
        tm.assert_series_equal(result, expected, check_index_type=True)

        expected = Series([0.2, 0.2, 0.1], index=[2, 2, 1])
        result = ser.loc[[2, 2, 1]]
        tm.assert_series_equal(result, expected, check_index_type=True)

        expected = Series([0.2, 0.2, np.nan, 0.1], index=[2, 2, "x", 1])
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[2, 2, "x", 1]]

        result = ser.reindex([2, 2, "x", 1])
        tm.assert_series_equal(result, expected, check_index_type=True)

        # raises as nothing is in the index
        msg = (
            rf"\"None of \[Index\(\[3, 3, 3\], dtype='{np.dtype(int)}'\)\] "
            r"are in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            ser.loc[[3, 3, 3]]

        expected = Series([0.2, 0.2, np.nan], index=[2, 2, 3])
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[2, 2, 3]]

        result = ser.reindex([2, 2, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3], index=[1, 2, 3])
        expected = Series([0.3, np.nan, np.nan], index=[3, 4, 4])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[3, 4, 4]]

        result = s.reindex([3, 4, 4])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([np.nan, 0.3, 0.3], index=[5, 3, 3])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[5, 3, 3]]

        result = s.reindex([5, 3, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([np.nan, 0.4, 0.4], index=[5, 4, 4])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[5, 4, 4]]

        result = s.reindex([5, 4, 4])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3, 0.4], index=[4, 5, 6, 7])
        expected = Series([0.4, np.nan, np.nan], index=[7, 2, 2])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[7, 2, 2]]

        result = s.reindex([7, 2, 2])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([0.4, np.nan, np.nan], index=[4, 5, 5])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[4, 5, 5]]

        result = s.reindex([4, 5, 5])
        tm.assert_series_equal(result, expected, check_index_type=True)

        # iloc
        expected = Series([0.2, 0.2, 0.1, 0.1], index=[2, 2, 1, 1])
        result = ser.iloc[[1, 1, 0, 0]]
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_series_partial_set_with_name(self):
        # [FUNCTION DOC]
        # Verifies the same behaviors with a named index and series (issue 11497).
        # Missing-key errors name the missing labels; the none-found error message includes the index name; repeated present labels keep the index name and series name; position-list selection with repeats keeps names.
        # GH 11497

        idx = Index([1, 2], dtype="int64", name="idx")
        ser = Series([0.1, 0.2], index=idx, name="s")

        # loc
        with pytest.raises(KeyError, match=r"\[3\] not in index"):
            ser.loc[[3, 2, 3]]

        with pytest.raises(KeyError, match=r"not in index"):
            ser.loc[[3, 2, 3, "x"]]

        exp_idx = Index([2, 2, 1], dtype="int64", name="idx")
        expected = Series([0.2, 0.2, 0.1], index=exp_idx, name="s")
        result = ser.loc[[2, 2, 1]]
        tm.assert_series_equal(result, expected, check_index_type=True)

        with pytest.raises(KeyError, match=r"\['x'\] not in index"):
            ser.loc[[2, 2, "x", 1]]

        # raises as nothing is in the index
        msg = (
            rf"\"None of \[Index\(\[3, 3, 3\], dtype='{np.dtype(int)}', "
            r"name='idx'\)\] are in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            ser.loc[[3, 3, 3]]

        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[2, 2, 3]]

        idx = Index([1, 2, 3], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3], index=idx, name="s").loc[[3, 4, 4]]

        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[5, 3, 3]]

        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[5, 4, 4]]

        idx = Index([4, 5, 6, 7], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[7, 2, 2]]

        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[4, 5, 5]]

        # iloc
        exp_idx = Index([2, 2, 1, 1], dtype="int64", name="idx")
        expected = Series([0.2, 0.2, 0.1, 0.1], index=exp_idx, name="s")
        result = ser.iloc[[1, 1, 0, 0]]
        tm.assert_series_equal(result, expected, check_index_type=True)

    @pytest.mark.parametrize("key", [100, 100.0])
    def test_setitem_with_expansion_numeric_into_datetimeindex(self, key):
        # [FUNCTION DOC]
        # Verifies setting a numeric label (integer or float) on a datetime-indexed frame with the whole-row label setter appends a row and converts the index to object dtype keeping the index name (issue 4940). Parametrized over 100 and 100.0.
        # GH#4940 inserting non-strings
        orig = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df = orig.copy()

        df.loc[key, :] = df.iloc[0]
        ex_index = Index([*list(orig.index), key], dtype=object, name=orig.index.name)
        ex_data = np.concatenate([orig.values, df.iloc[[0]].values], axis=0)
        expected = DataFrame(ex_data, index=ex_index, columns=orig.columns)

        tm.assert_frame_equal(df, expected)

    def test_partial_set_invalid(self):
        # [FUNCTION DOC]
        # Verifies that setting a text label on a datetime-indexed frame is allowed and converts the index to object dtype (issue 4940): the new row equals the first row, the result equals concatenation with an object-inferred row.
        # GH 4940
        # allow only setting of 'valid' values

        orig = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )

        # allow object conversion here
        df = orig.copy()
        df.loc["a", :] = df.iloc[0]
        ser = Series(df.iloc[0], name="a")
        exp = pd.concat([orig, DataFrame(ser).T.infer_objects()])
        tm.assert_frame_equal(df, exp)
        tm.assert_index_equal(df.index, Index([*orig.index.tolist(), "a"]))
        assert df.index.dtype == "object"

    @pytest.mark.parametrize(
        "idx,labels,expected_idx",
        [
            (
                period_range(start="2000", periods=20, freq="D"),
                ["2000-01-04", "2000-01-08", "2000-01-12"],
                [
                    Period("2000-01-04", freq="D"),
                    Period("2000-01-08", freq="D"),
                    Period("2000-01-12", freq="D"),
                ],
            ),
            (
                date_range(start="2000", periods=20, freq="D", unit="s"),
                ["2000-01-04", "2000-01-08", "2000-01-12"],
                [
                    Timestamp("2000-01-04").as_unit("s"),
                    Timestamp("2000-01-08").as_unit("s"),
                    Timestamp("2000-01-12").as_unit("s"),
                ],
            ),
            (
                pd.timedelta_range(start="1 day", periods=20),
                ["4D", "8D", "12D"],
                [pd.Timedelta("4 day"), pd.Timedelta("8 day"), pd.Timedelta("12 day")],
            ),
        ],
    )
    def test_loc_with_list_of_strings_representing_datetimes(
        self, idx, labels, expected_idx, frame_or_series
    ):
        # [FUNCTION DOC]
        # Verifies label-list selection with strings that parse to the index type (issue 11278).
        # Parametrized over period, second-unit datetime and timedelta indexes with matching string labels and expected index values. Selection by the label indexer returns the three expected values with a typed index; for series, direct bracket selection returns the same. Uses the frame-or-series fixture.
        # GH 11278
        obj = frame_or_series(range(20), index=idx)

        expected_value = [3, 7, 11]
        expected = frame_or_series(expected_value, expected_idx)

        tm.assert_equal(expected, obj.loc[labels])
        if frame_or_series is Series:
            tm.assert_series_equal(expected, obj[labels])

    @pytest.mark.parametrize(
        "idx,labels",
        [
            (
                period_range(start="2000", periods=20, freq="D"),
                ["2000-01-04", "2000-01-30"],
            ),
            (
                date_range(start="2000", periods=20, freq="D"),
                ["2000-01-04", "2000-01-30"],
            ),
            (pd.timedelta_range(start="1 day", periods=20), ["3 day", "30 day"]),
        ],
    )
    def test_loc_with_list_of_strings_representing_datetimes_missing_value(
        self, idx, labels
    ):
        # [FUNCTION DOC]
        # Verifies a list of strings in which one is valid but outside the index range raises a key error saying labels are not in the index, for series and frame label selection and series bracket selection (issue 11278). Parametrized over period, datetime and timedelta indexes.
        # GH 11278
        ser = Series(range(20), index=idx)
        df = DataFrame(range(20), index=idx)
        msg = r"not in index"

        with pytest.raises(KeyError, match=msg):
            ser.loc[labels]
        with pytest.raises(KeyError, match=msg):
            ser[labels]
        with pytest.raises(KeyError, match=msg):
            df.loc[labels]

    @pytest.mark.parametrize(
        "idx,labels,msg",
        [
            (
                period_range(start="2000", periods=20, freq="D"),
                Index(["4D", "8D"], dtype=object),
                (
                    r"None of \[Index\(\['4D', '8D'\], dtype='object'\)\] "
                    r"are in the \[index\]"
                ),
            ),
            (
                date_range(start="2000", periods=20, freq="D"),
                Index(["4D", "8D"], dtype=object),
                (
                    r"None of \[Index\(\['4D', '8D'\], dtype='object'\)\] "
                    r"are in the \[index\]"
                ),
            ),
            (
                pd.timedelta_range(start="1 day", periods=20),
                Index(["2000-01-04", "2000-01-08"], dtype=object),
                (
                    r"None of \[Index\(\['2000-01-04', '2000-01-08'\], "
                    r"dtype='object'\)\] are in the \[index\]"
                ),
            ),
        ],
    )
    def test_loc_with_list_of_strings_representing_datetimes_not_matched_type(
        self, idx, labels, msg
    ):
        # [FUNCTION DOC]
        # Verifies that an object index of strings that cannot match the index type (period/datetime index with day-offset strings, timedelta index with date strings) raises a key error saying none of the labels are in the index (issue 11278), for series and frame.
        # GH 11278
        ser = Series(range(20), index=idx)
        df = DataFrame(range(20), index=idx)

        with pytest.raises(KeyError, match=msg):
            ser.loc[labels]
        with pytest.raises(KeyError, match=msg):
            ser[labels]
        with pytest.raises(KeyError, match=msg):
            df.loc[labels]


class TestStringSlicing:
    # [CLASS DOC]
    # Tests for slicing with string bounds on date-like indexes.
    def test_slice_irregular_datetime_index_with_nan(self):
        # [FUNCTION DOC]
        # Verifies slicing an unordered datetime index with a missing entry (issue 36953). A string slice whose upper bound is absent from the index raises a key error saying non-existing keys are not allowed (issues 53983 and 37819); a slice whose right bound is given with full nanosecond precision (so the rounded-up bound is exact) returns the first three rows.
        # GH36953
        index = pd.to_datetime(["2012-01-01", "2012-01-02", "2012-01-03", None])
        df = DataFrame(range(len(index)), index=index)
        expected = DataFrame(range(len(index[:3])), index=index[:3])
        with pytest.raises(KeyError, match="non-existing keys is not allowed"):
            # Upper bound is not in index (which is unordered)
            # GH53983
            # GH37819
            df["2012-01-01":"2012-01-04"]
        # Need this precision for right bound since the right slice
        # bound is "rounded" up to the largest timepoint smaller than
        # the next "resolution"-step of the provided point.
        # e.g. 2012-01-03 is rounded up to 2012-01-04 - 1ns
        result = df["2012-01-01":"2012-01-03 00:00:00.000000000"]
        tm.assert_frame_equal(result, expected)
