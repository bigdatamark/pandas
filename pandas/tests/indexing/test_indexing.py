"""test fancy indexing & misc"""
# [MODULE DOC]
# General fancy-indexing and miscellaneous indexing tests: setting with arrays and lists, dtype
# upcast errors, duplicate-label selection and missing-label errors, multi-assignment, index
# type coercion, label slices with negative steps, None coercion, datetime-like setitem
# dispatch, extension-array cross sections, and nested-value scalar assignment. Imports the
# object generator helper from the float indexing tests and the label maker from the shared
# indexing test utilities.

import array
from datetime import datetime
import re
import weakref

import numpy as np
import pytest

from pandas.errors import IndexingError

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    NaT,
    Series,
    date_range,
    offsets,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj

# ------------------------------------------------------------------------
# Indexing test cases


class TestFancy:
    # [CLASS DOC]
    # Test collection for pure get/set item and fancy indexing behaviors.
    """pure get/set item & fancy indexing"""

    def test_setitem_ndarray_1d(self):
        # GH5508

        # len of indexer vs length of the 1d ndarray
        # [FUNCTION DOC]
        # GH 5508. On a frame with integer index, setting a complex column slice from an array of
        # mismatching length raises a value error about equal lengths; a matching-length array sets values
        # and reads back as a complex series.
        df = DataFrame(index=Index(np.arange(1, 11), dtype=np.int64))
        df["foo"] = np.zeros(10, dtype=np.float64)
        df["bar"] = np.zeros(10, dtype=complex)

        # invalid
        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[df.index[2:5], "bar"] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])

        # valid
        df.loc[df.index[2:6], "bar"] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])

        result = df.loc[df.index[2:6], "bar"]
        expected = Series(
            [2.33j, 1.23 + 0.1j, 2.2, 1.0], index=[3, 4, 5, 6], name="bar"
        )
        tm.assert_series_equal(result, expected)

    def test_setitem_ndarray_1d_2(self):
        # GH5508

        # dtype getting changed?
        # [FUNCTION DOC]
        # GH 5508. Assigning a complex array of mismatching length to a row slice by plain setitem
        # raises a value error about equal lengths.
        # KNOWN ISSUE: a comment asks whether dtype is getting changed, but the test only checks that
        # the length error is raised.
        df = DataFrame(index=Index(np.arange(1, 11)))
        df["foo"] = np.zeros(10, dtype=np.float64)
        df["bar"] = np.zeros(10, dtype=complex)

        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df[2:5] = np.arange(1, 4) * 1j

    @pytest.mark.filterwarnings(
        "ignore:Series.__getitem__ treating keys as positions is deprecated:"
        "FutureWarning"
    )
    def test_getitem_ndarray_3d(self, index, frame_or_series, indexer_sli):
        # GH 25567
        # [FUNCTION DOC]
        # GH 25567. Parametrized over index fixture, frame-or-series and the accessor fixtures
        # (getitem, loc, iloc). Indexing with a three-dimensional integer array must raise one of an
        # index, value or not-implemented error whose message matches one of a set of messages that is
        # built according to the object type, accessor and index kind. A deprecation warning about
        # treating keys as positions is ignored.
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        msgs = []
        if frame_or_series is Series and indexer_sli in [tm.setitem, tm.iloc]:
            msgs.append(r"Wrong number of dimensions. values.ndim > ndim \[3 > 1\]")
        if frame_or_series is Series or indexer_sli is tm.iloc:
            msgs.append(r"Buffer has wrong number of dimensions \(expected 1, got 3\)")
        if indexer_sli is tm.loc or (
            frame_or_series is Series and indexer_sli is tm.setitem
        ):
            msgs.append("Cannot index with multidimensional key")
        if frame_or_series is DataFrame and indexer_sli is tm.setitem:
            msgs.append("Index data must be 1-dimensional")
        if isinstance(index, pd.IntervalIndex) and indexer_sli is tm.iloc:
            msgs.append("Index data must be 1-dimensional")
        if isinstance(index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
            msgs.append("Data must be 1-dimensional")
        if len(index) == 0 or isinstance(index, pd.MultiIndex):
            msgs.append("positional indexers are out-of-bounds")
        if type(index) is Index and not isinstance(index._values, np.ndarray):
            # e.g. Int64
            msgs.append("values must be a 1D array")

            # string[pyarrow]
            msgs.append("only handle 1-dimensional arrays")

        msg = "|".join(msgs)

        potential_errors = (IndexError, ValueError, NotImplementedError)
        with pytest.raises(potential_errors, match=msg):
            idxr[nd3]

    @pytest.mark.filterwarnings(
        "ignore:Series.__setitem__ treating keys as positions is deprecated:"
        "FutureWarning"
    )
    def test_setitem_ndarray_3d(self, index, frame_or_series, indexer_sli):
        # GH 25567
        # [FUNCTION DOC]
        # GH 25567. Same parameters. Assigning zero with a three-dimensional array key raises a value
        # error: for the positional accessor, the message says values cannot have more dimensions than
        # the object; for others, any of several messages.
        # KNOWN ISSUE: the error class is assigned the same value in both branches, so the branching only
        # selects the message.
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        if indexer_sli is tm.iloc:
            err = ValueError
            msg = f"Cannot set values with ndim > {obj.ndim}"
        else:
            err = ValueError
            msg = "|".join(
                [
                    r"Buffer has wrong number of dimensions \(expected 1, got 3\)",
                    "Cannot set values with ndim > 1",
                    "Index data must be 1-dimensional",
                    "Data must be 1-dimensional",
                    "Array conditional must be same shape as self",
                ]
            )

        with pytest.raises(err, match=msg):
            idxr[nd3] = 0

    def test_getitem_ndarray_0d(self):
        # GH#24924
        # [FUNCTION DOC]
        # GH 24924. A zero-dimensional array key picks a column on a frame and an element on a series.
        key = np.array(0)

        # dataframe __getitem__
        df = DataFrame([[1, 2], [3, 4]])
        result = df[key]
        expected = Series([1, 3], name=0)
        tm.assert_series_equal(result, expected)

        # series __getitem__
        ser = Series([1, 2])
        result = ser[key]
        assert result == 1

    def test_inf_upcast(self):
        # GH 16957
        # We should be able to use np.inf as a key
        # np.inf should cause an index to convert to float

        # Test with np.inf in rows
        # [FUNCTION DOC]
        # GH 16957. Setting a row with infinity as label converts the integer index to float and
        # permits lookup by infinity.
        df = DataFrame(columns=[0])
        df.loc[1] = 1
        df.loc[2] = 2
        df.loc[np.inf] = 3

        # make sure we can look up the value
        assert df.loc[np.inf, 0] == 3

        result = df.index
        expected = Index([1, 2, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    def test_setitem_dtype_upcast(self):
        # GH3216
        # [FUNCTION DOC]
        # GH 3216. Assigning a string to an element of a float column raises a type error about an
        # invalid value.
        df = DataFrame([{"a": 1}, {"a": 3, "b": 2}])
        df["c"] = np.nan
        assert df["c"].dtype == np.float64

        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[0, "c"] = "foo"

    @pytest.mark.parametrize("val", [3.14, "wxyz"])
    def test_setitem_dtype_upcast2(self, val):
        # GH10280
        # [FUNCTION DOC]
        # GH 10280. Parametrized over a float and a string value. Assigning to an element of an
        # integer column raises a type error about an invalid value.
        df = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3),
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )

        left = df.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            left.loc["a", "bar"] = val

    def test_setitem_dtype_upcast3(self):
        # [FUNCTION DOC]
        # Assigning a string to a float frame element raises a type error about an invalid value.
        left = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3) / 10.0,
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )
        with pytest.raises(TypeError, match="Invalid value"):
            left.loc["a", "bar"] = "wxyz"

    def test_dups_fancy_indexing(self):
        # GH 3455

        # [FUNCTION DOC]
        # GH 3455. Selecting columns by a list on a frame with duplicate labels returns the columns in
        # the order given with duplicates expanded.
        df = DataFrame(np.eye(3), columns=["a", "a", "b"])
        result = df[["b", "a"]].columns
        expected = Index(["b", "a", "a"])
        tm.assert_index_equal(result, expected)

    def test_dups_fancy_indexing_across_dtypes(self):
        # across dtypes
        # [FUNCTION DOC]
        # GH 3468, GH 3509. A mixed-dtype single-row frame constructed with duplicate column labels is
        # equal to one built then relabelled; positional access of a column works (smoke).
        df = DataFrame([[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]], columns=list("aaaaaaa"))
        result = DataFrame([[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]])
        result.columns = list("aaaaaaa")  # GH#3468

        # GH#3509 smoke tests for indexing with duplicate columns
        df.iloc[:, 4]
        result.iloc[:, 4]

        tm.assert_frame_equal(df, result)

    def test_dups_fancy_indexing_not_in_order(self):
        # GH 3561, dups not in selected order
        # [FUNCTION DOC]
        # GH 3561, GH 5553. Label lists on a frame with duplicate index selected in the list order (as list
        # or index); lists with missing labels raise a key error saying not in index.
        df = DataFrame(
            {"test": [5, 7, 9, 11], "test1": [4.0, 5, 6, 7], "other": list("abcd")},
            index=["A", "A", "B", "C"],
        )
        rows = ["C", "B"]
        expected = DataFrame(
            {"test": [11, 9], "test1": [7.0, 6], "other": ["d", "c"]}, index=rows
        )
        result = df.loc[rows]
        tm.assert_frame_equal(result, expected)

        result = df.loc[Index(rows)]
        tm.assert_frame_equal(result, expected)

        rows = ["C", "B", "E"]
        with pytest.raises(KeyError, match="not in index"):
            df.loc[rows]

        # see GH5553, make sure we use the right indexer
        rows = ["F", "G", "H", "C", "B", "E"]
        with pytest.raises(KeyError, match="not in index"):
            df.loc[rows]

    def test_dups_fancy_indexing_only_missing_label(self, using_infer_string):
        # List containing only missing label
        # [FUNCTION DOC]
        # Uses the infer-string fixture. A list that only contains a missing label raises a key error
        # saying none of the labels are in the index, with the index dtype named as string or object
        # depending on the option.
        dfnu = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), index=list("AABCD")
        )
        if using_infer_string:
            with pytest.raises(
                KeyError,
                match=re.escape(
                    "\"None of [Index(['E'], dtype='str')] are in the [index]\""
                ),
            ):
                dfnu.loc[["E"]]
        else:
            with pytest.raises(
                KeyError,
                match=re.escape(
                    "\"None of [Index(['E'], dtype='object')] are in the [index]\""
                ),
            ):
                dfnu.loc[["E"]]

    @pytest.mark.parametrize("vals", [[0, 1, 2], list("abc")])
    def test_dups_fancy_indexing_missing_label(self, vals):
        # GH 4619; duplicate indexer with missing label
        # [FUNCTION DOC]
        # GH 4619. Parametrized over integer and string column values. A list with a repeated label and
        # a missing label raises a key error.
        df = DataFrame({"A": vals})
        with pytest.raises(KeyError, match="not in index"):
            df.loc[[0, 8, 0]]

    def test_dups_fancy_indexing_non_unique(self):
        # non unique with non unique selector
        # [FUNCTION DOC]
        # Non-unique labels with a non-unique selector containing a missing label raise a key error.
        df = DataFrame({"test": [5, 7, 9, 11]}, index=["A", "A", "B", "C"])
        with pytest.raises(KeyError, match="not in index"):
            df.loc[["A", "A", "E"]]

    def test_dups_fancy_indexing2(self):
        # GH 5835
        # dups on index and missing values
        # [FUNCTION DOC]
        # GH 5835. Selecting duplicate columns plus a missing column raises a key error.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 5)),
            columns=["A", "B", "B", "B", "A"],
        )

        with pytest.raises(KeyError, match="not in index"):
            df.loc[:, ["A", "B", "C"]]

    def test_dups_fancy_indexing3(self):
        # GH 6504, multi-axis indexing
        # [FUNCTION DOC]
        # GH 6504. Multi-axis selection on a frame with duplicate row labels returns all rows for each
        # requested label.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((9, 2)),
            index=[1, 1, 1, 2, 2, 2, 3, 3, 3],
            columns=["a", "b"],
        )

        expected = df.iloc[0:6]
        result = df.loc[[1, 2]]
        tm.assert_frame_equal(result, expected)

        expected = df
        result = df.loc[:, ["a", "b"]]
        tm.assert_frame_equal(result, expected)

        expected = df.iloc[0:6, :]
        result = df.loc[[1, 2], ["a", "b"]]
        tm.assert_frame_equal(result, expected)

    def test_duplicate_int_indexing(self, indexer_sl):
        # GH 17347
        # [FUNCTION DOC]
        # GH 17347. Uses the series accessor fixture. A list selector with a duplicated integer label
        # returns every matching element.
        ser = Series(range(3), index=[1, 1, 3])
        expected = Series(range(2), index=[1, 1])
        result = indexer_sl(ser)[[1]]
        tm.assert_series_equal(result, expected)

    def test_indexing_mixed_frame_bug(self):
        # GH3492
        # [FUNCTION DOC]
        # GH 3492. Applying a function creates a column; assigning transformed values via a mask
        # updates only the masked cell.
        # KNOWN ISSUE: a comment says this operation "does not work" (the column is not changed), but the
        # assertion verifies that the value is changed.
        df = DataFrame(
            {"a": {1: "aaa", 2: "bbb", 3: "ccc"}, "b": {1: 111, 2: 222, 3: 333}}
        )

        # this works, new column is created correctly
        df["test"] = df["a"].apply(lambda x: "_" if x == "aaa" else x)

        # this does not work, ie column test is not changed
        idx = df["test"] == "_"
        temp = df.loc[idx, "a"].apply(lambda x: "-----" if x == "aaa" else x)
        df.loc[idx, "test"] = temp
        assert df.iloc[0, 2] == "-----"

    def test_multitype_list_index_access(self):
        # GH 10610
        # [FUNCTION DOC]
        # GH 10610. Selecting columns with a list containing absent labels (including a negative one)
        # raises a key error listing the missing; an existing label returns a full-length column.
        df = DataFrame(
            np.random.default_rng(2).random((10, 5)), columns=["a", 20, 21, 22, 23]
        )

        with pytest.raises(KeyError, match=re.escape("'[26, -8] not in index'")):
            df[[22, 26, -8]]
        assert df[21].shape[0] == df.shape[0]

    def test_set_index_nan(self):
        # GH 3586
        # [FUNCTION DOC]
        # GH 3586. Setting several columns as index (one with missing values) and resetting it round-trips
        # to the original frame with its original column order.
        df = DataFrame(
            {
                "PRuid": {
                    17: "nonQC",
                    18: "nonQC",
                    19: "nonQC",
                    20: "10",
                    21: "11",
                    22: "12",
                    23: "13",
                    24: "24",
                    25: "35",
                    26: "46",
                    27: "47",
                    28: "48",
                    29: "59",
                    30: "10",
                },
                "QC": {
                    17: 0.0,
                    18: 0.0,
                    19: 0.0,
                    20: np.nan,
                    21: np.nan,
                    22: np.nan,
                    23: np.nan,
                    24: 1.0,
                    25: np.nan,
                    26: np.nan,
                    27: np.nan,
                    28: np.nan,
                    29: np.nan,
                    30: np.nan,
                },
                "data": {
                    17: 7.9544899999999998,
                    18: 8.0142609999999994,
                    19: 7.8591520000000008,
                    20: 0.86140349999999999,
                    21: 0.87853110000000001,
                    22: 0.8427041999999999,
                    23: 0.78587700000000005,
                    24: 0.73062459999999996,
                    25: 0.81668560000000001,
                    26: 0.81927080000000008,
                    27: 0.80705009999999999,
                    28: 0.81440240000000008,
                    29: 0.80140849999999997,
                    30: 0.81307740000000006,
                },
                "year": {
                    17: 2006,
                    18: 2007,
                    19: 2008,
                    20: 1985,
                    21: 1985,
                    22: 1985,
                    23: 1985,
                    24: 1985,
                    25: 1985,
                    26: 1985,
                    27: 1985,
                    28: 1985,
                    29: 1985,
                    30: 1986,
                },
            }
        ).reset_index()

        result = (
            df.set_index(["year", "PRuid", "QC"])
            .reset_index()
            .reindex(columns=df.columns)
        )
        tm.assert_frame_equal(result, df)

    def test_multi_assign(self):
        # GH 3626, an assignment of a sub-df to a df
        # set float64 to avoid upcast when setting nan
        # [FUNCTION DOC]
        # GH 3626, GH 14001. Assigning a sub-frame (or its raw values) selected by a non-missing mask over
        # two columns into a copy gives the expected frame with missing preserved and float coercion.
        df = DataFrame(
            {
                "FC": ["a", "b", "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": list(range(6)),
                "col2": list(range(6, 12)),
            }
        ).astype({"col2": "float64"})
        df.iloc[1, 0] = np.nan
        df2 = df.copy()

        mask = ~df2.FC.isna()
        cols = ["col1", "col2"]

        dft = df2 * 2
        dft.iloc[3, 3] = np.nan

        expected = DataFrame(
            {
                "FC": ["a", np.nan, "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": Series([0, 1, 4, 6, 8, 10]),
                "col2": [12, 7, 16, np.nan, 20, 22],
            }
        )

        # frame on rhs
        df2.loc[mask, cols] = dft.loc[mask, cols]
        tm.assert_frame_equal(df2, expected)

        # with an ndarray on rhs
        # coerces to float64 because values has float64 dtype
        # GH 14001
        expected = DataFrame(
            {
                "FC": ["a", np.nan, "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": [0, 1, 4, 6, 8, 10],
                "col2": [12, 7, 16, np.nan, 20, 22],
            }
        )
        df2 = df.copy()
        df2.loc[mask, cols] = dft.loc[mask, cols].values
        tm.assert_frame_equal(df2, expected)

    def test_multi_assign_broadcasting_rhs(self):
        # broadcasting on the rhs is required
        # [FUNCTION DOC]
        # Assigning a series to multiple columns for masked rows broadcasts it across the columns.
        df = DataFrame(
            {
                "A": [1, 2, 0, 0, 0],
                "B": [0, 0, 0, 10, 11],
                "C": [0, 0, 0, 10, 11],
                "D": [3, 4, 5, 6, 7],
            }
        )

        expected = df.copy()
        mask = expected["A"] == 0
        for col in ["A", "B"]:
            expected.loc[mask, col] = df["D"]

        df.loc[df["A"] == 0, ["A", "B"]] = df["D"].copy()
        tm.assert_frame_equal(df, expected)

    def test_setitem_list(self):
        # GH 6043
        # iloc with a list
        # [FUNCTION DOC]
        # GH 6043. A list assigned to a single cell of an object frame, twice, equals assigning the final
        # list once.
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = [1, 2, 3]
        df.iloc[1, 0] = [1, 2]

        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[1, 0] = [1, 2]

        tm.assert_frame_equal(result, df)

    def test_string_slice(self):
        # GH 14424
        # string indexing against datetimelike with object
        # dtype should properly raises KeyError
        # [FUNCTION DOC]
        # GH 14424. A string key on a frame with an object index of timestamps raises key errors for plain
        # getitem and loc.
        df = DataFrame([1], Index([pd.Timestamp("2011-01-01")], dtype=object))
        assert df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df["2011"]

        with pytest.raises(KeyError, match="'2011'"):
            df.loc["2011", 0]

    def test_string_slice_empty(self):
        # GH 14424

        # [FUNCTION DOC]
        # GH 14424. A string key on an empty frame raises a key error naming the string for getitem;
        # for loc with a row string and a column, the error names the column 0.
        df = DataFrame()
        assert not df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df["2011"]

        with pytest.raises(KeyError, match="^0$"):
            df.loc["2011", 0]

    def test_astype_assignment(self, using_infer_string):
        # GH4312 (iloc)
        # [FUNCTION DOC]
        # Uses the infer-string fixture. GH 4312, GH 5702. Assigning the integer-cast version of object
        # string columns through positional or label setters writes in place, so object dtype is
        # preserved; compared with the expected frames of object-typed columns.
        # KNOWN ISSUE: the infer-string fixture parameter is accepted but not used.
        df_orig = DataFrame(
            [["1", "2", "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        df_orig[list("ABCDG")] = df_orig[list("ABCDG")].astype(object)

        df = df_orig.copy()

        # with the enforcement of GH#45333 in 2.0, this setting is attempted inplace,
        #  so object dtype is retained
        df.iloc[:, 0:2] = df.iloc[:, 0:2].astype(np.int64)
        expected = DataFrame(
            [[1, 2, "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        expected[list("CDG")] = expected[list("CDG")].astype(object)
        expected["A"] = expected["A"].astype(object)
        expected["B"] = expected["B"].astype(object)
        tm.assert_frame_equal(df, expected)

        # GH5702 (loc)
        df = df_orig.copy()
        df.loc[:, "A"] = df.loc[:, "A"].astype(np.int64)
        expected = DataFrame(
            [[1, "2", "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        expected[list("ABCDG")] = expected[list("ABCDG")].astype(object)
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()

        df.loc[:, ["B", "C"]] = df.loc[:, ["B", "C"]].astype(np.int64)
        expected = DataFrame(
            [["1", 2, 3, ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        expected[list("ABCDG")] = expected[list("ABCDG")].astype(object)
        tm.assert_frame_equal(df, expected)

    def test_astype_assignment_full_replacements(self):
        # full replacements / no nans
        # [FUNCTION DOC]
        # Replacing a whole float column with its integer cast, by positional or label setter, writes in
        # place and keeps float64.
        df = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})

        # With the enforcement of GH#45333 in 2.0, this assignment occurs inplace,
        #  so float64 is retained
        df.iloc[:, 0] = df["A"].astype(np.int64)
        expected = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
        df.loc[:, "A"] = df["A"].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer", [tm.getitem, tm.loc])
    def test_index_type_coercion(self, indexer):
        # GH 11836
        # if we have an index type and set it with something that looks
        # to numpy like the same, but is actually, not
        # (e.g. setting with a float or string '0')
        # then we need to coerce to object

        # integer indexes
        # [FUNCTION DOC]
        # GH 11836. Parametrized over getitem and loc accessors. Enlarging a series by setting a float
        # key converts an integer index to float; setting key 0.0 on an integer index appends or keeps the
        # integer index as appropriate; a string key makes the index object. Float indexes stay float
        # for float keys, and become object for a string key.
        for s in [Series(range(5)), Series(range(5), index=range(1, 6))]:
            assert is_integer_dtype(s.index)

            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert is_float_dtype(s2.index)
            assert indexer(s2)[0.1] == 0

            s2 = s.copy()
            indexer(s2)[0.0] = 0
            exp = s.index
            if 0 not in s:
                exp = Index([*s.index.tolist(), 0])
            tm.assert_index_equal(s2.index, exp)

            s2 = s.copy()
            indexer(s2)["0"] = 0
            assert is_object_dtype(s2.index)

        for s in [Series(range(5), index=np.arange(5.0))]:
            assert is_float_dtype(s.index)

            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert is_float_dtype(s2.index)
            assert indexer(s2)[0.1] == 0

            s2 = s.copy()
            indexer(s2)[0.0] = 0
            tm.assert_index_equal(s2.index, s.index)

            s2 = s.copy()
            indexer(s2)["0"] = 0
            assert is_object_dtype(s2.index)


class TestMisc:
    # [CLASS DOC]
    # Test collection for miscellaneous indexing behaviors.
    def test_float_index_to_mixed(self):
        # [FUNCTION DOC]
        # A frame with float column labels accepts a new string column; the frame equals the expected
        # one with mixed labels.
        df = DataFrame(
            {
                0.0: np.random.default_rng(2).random(10),
                1.0: np.random.default_rng(2).random(10),
            }
        )
        df["a"] = 10

        expected = DataFrame({0.0: df[0.0], 1.0: df[1.0], "a": [10] * 10})
        tm.assert_frame_equal(expected, df)

    def test_float_index_non_scalar_assignment(self):
        # [FUNCTION DOC]
        # Assigning a scalar through loc with a slice of the float index sets the first two rows.
        df = DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        df.loc[df.index[:2]] = 1
        expected = DataFrame({"a": [1, 1, 3], "b": [1, 1, 5]}, index=df.index)
        tm.assert_frame_equal(expected, df)

    def test_loc_setitem_fullindex_views(self):
        # [FUNCTION DOC]
        # Assigning a frame to itself through the full index leaves the frame unchanged.
        df = DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        df2 = df.copy()
        df.loc[df.index] = df.loc[df.index]
        tm.assert_frame_equal(df, df2)

    def test_rhs_alignment(self):
        # GH8258, tests that both rows & columns are aligned to what is
        # assigned to. covers both uniform data-type & multi-type cases
        # [FUNCTION DOC]
        # GH 8258. Right-hand side frames are aligned on both rows and columns when assigned with label
        # lists, but positional assignment uses the right-hand side's order. For uniform integer dtype, all
        # three setters (label, position list, position slice) give the expected results. After turning two
        # columns into float and string, the same run raises a type error about an invalid value.
        def run_tests(df, rhs, right_loc, right_iloc):
            # label, index, slice
            # [FUNCTION DOC]
            # HELPER: runs three assignments on copies of df of the right-hand side rhs into rows b-d and
            # columns joe and jolie: label lists, integer position lists, and position slices. Compares the
            # label result with right_loc and the two positional results with right_iloc. Returns nothing.
            lbl_one, idx_one, slice_one = list("bcd"), [1, 2, 3], slice(1, 4)
            lbl_two, idx_two, slice_two = ["joe", "jolie"], [1, 2], slice(1, 3)

            left = df.copy()
            left.loc[lbl_one, lbl_two] = rhs
            tm.assert_frame_equal(left, right_loc)

            left = df.copy()
            left.iloc[idx_one, idx_two] = rhs
            tm.assert_frame_equal(left, right_iloc)

            left = df.copy()
            left.iloc[slice_one, slice_two] = rhs
            tm.assert_frame_equal(left, right_iloc)

        xs = np.arange(20).reshape(5, 4)
        cols = ["jim", "joe", "jolie", "joline"]
        df = DataFrame(xs, columns=cols, index=list("abcde"), dtype="int64")

        # right hand side; permute the indices and multiply by -2
        rhs = -2 * df.iloc[3:0:-1, 2:0:-1]

        # expected `right` result; just multiply by -2
        right_iloc = df.copy()
        right_iloc["joe"] = [1, 14, 10, 6, 17]
        right_iloc["jolie"] = [2, 13, 9, 5, 18]
        right_iloc.iloc[1:4, 1:3] *= -2
        right_loc = df.copy()
        right_loc.iloc[1:4, 1:3] *= -2

        # run tests with uniform dtypes
        run_tests(df, rhs, right_loc, right_iloc)

        # make frames multi-type & re-run tests
        for frame in [df, rhs, right_loc, right_iloc]:
            frame["joe"] = frame["joe"].astype("float64")
            frame["jolie"] = frame["jolie"].map(lambda x: f"@{x}")
        right_iloc["joe"] = [1.0, "@-28", "@-20", "@-12", 17.0]
        right_iloc["jolie"] = ["@2", -26.0, -18.0, -10.0, "@18"]
        with pytest.raises(TypeError, match="Invalid value"):
            run_tests(df, rhs, right_loc, right_iloc)

    @pytest.mark.parametrize(
        "idx", [_mklbl("A", 20), np.arange(20) + 100, np.linspace(100, 150, 20)]
    )
    def test_str_label_slicing_with_negative_step(self, idx):
        # [FUNCTION DOC]
        # Parametrized over string labels, shifted integers, and linearly spaced floats. Label slices
        # with negative steps (open start, open stop, both, and reverse bounds that give an empty result)
        # are equivalent to the matching integer slices (checked by a testing helper).
        SLC = pd.IndexSlice

        idx = Index(idx)
        ser = Series(np.arange(20), index=idx)
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[9] :: -1], SLC[9::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[: idx[9] : -1], SLC[:8:-1])
        tm.assert_indexing_slices_equivalent(
            ser, SLC[idx[13] : idx[9] : -1], SLC[13:8:-1]
        )
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[9] : idx[13] : -1], SLC[:0])

    def test_slice_with_zero_step_raises(self, index, indexer_sl, frame_or_series):
        # [FUNCTION DOC]
        # Uses the index fixture, the series accessor fixture and frame-or-series. A slice with step
        # zero raises a value error that the slice step cannot be zero.
        obj = frame_or_series(np.arange(len(index)), index=index)
        with pytest.raises(ValueError, match="slice step cannot be zero"):
            indexer_sl(obj)[::0]

    def test_loc_setitem_indexing_assignment_dict_already_exists(self):
        # [FUNCTION DOC]
        # Assigning a mapping to an existing row label sets values by column label; checked with all-integer
        # and mixed dtypes (GH 38335).
        index = Index([-5, 0, 5], name="z")
        df = DataFrame({"x": [1, 2, 6], "y": [2, 2, 8]}, index=index)
        expected = df.copy()
        rhs = {"x": 9, "y": 99}
        df.loc[5] = rhs
        expected.loc[5] = [9, 99]
        tm.assert_frame_equal(df, expected)

        # GH#38335 same thing, mixed dtypes
        df = DataFrame({"x": [1, 2, 6], "y": [2.0, 2.0, 8.0]}, index=index)
        df.loc[5] = rhs
        expected = DataFrame({"x": [1, 2, 9], "y": [2.0, 2.0, 99.0]}, index=index)
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_indexing_dtypes_on_empty(self):
        # Check that .iloc returns correct dtypes GH9983
        # [FUNCTION DOC]
        # GH 9983. A positional empty row selection keeps column dtypes: label and positional column
        # selections of the empty frame are equal and integer.
        df = DataFrame({"a": [1, 2, 3], "b": ["b", "b2", "b3"]})
        df2 = df.iloc[[], :]

        assert df2.loc[:, "a"].dtype == np.int64
        tm.assert_series_equal(df2.loc[:, "a"], df2.iloc[:, 0])

    @pytest.mark.parametrize("size", [5, 999999, 1000000])
    def test_loc_range_in_series_indexing(self, size):
        # range can cause an indexing error
        # GH 11652
        # [FUNCTION DOC]
        # GH 11652. Parametrized over sizes around one million. A range key for loc assignment and
        # selection on a float series works.
        s = Series(index=range(size), dtype=np.float64)
        s.loc[range(1)] = 42
        tm.assert_series_equal(s.loc[range(1)], Series(42.0, index=range(1)))

        s.loc[range(2)] = 43
        tm.assert_series_equal(s.loc[range(2)], Series(43.0, index=range(2)))

    def test_partial_boolean_frame_indexing(self):
        # GH 17170
        # [FUNCTION DOC]
        # GH 17170. A boolean frame covering part of the rows and columns used as a key masks the rest as
        # missing in the result.
        df = DataFrame(
            np.arange(9.0).reshape(3, 3), index=list("abc"), columns=list("ABC")
        )
        index_df = DataFrame(1, index=list("ab"), columns=list("AB"))
        result = df[index_df.notnull()]
        expected = DataFrame(
            np.array([[0.0, 1.0, np.nan], [3.0, 4.0, np.nan], [np.nan] * 3]),
            index=list("abc"),
            columns=list("ABC"),
        )
        tm.assert_frame_equal(result, expected)

    def test_period_column_slicing(self):
        # GH#60273 The transpose operation creates a single 5x1 block of PeriodDtype
        # Make sure it is reindexed correctly
        # [FUNCTION DOC]
        # GH 60273. A transposed frame holding a single period block selected with a list of column
        # labels is reindexed properly.
        df = DataFrame(
            pd.period_range("2021-01-01", periods=5, freq="D"),
            columns=["A"],
        ).T
        result = df[[0, 1, 2]]
        expected = DataFrame(
            [
                [
                    pd.Period("2021-01-01", freq="D"),
                    pd.Period("2021-01-02", freq="D"),
                    pd.Period("2021-01-03", freq="D"),
                ]
            ],
            index=["A"],
            columns=[0, 1, 2],
        )
        tm.assert_frame_equal(result, expected)

    def test_no_reference_cycle(self):
        # [FUNCTION DOC]
        # Accessing the label and position accessors on a frame creates no reference cycle: after deleting
        # the frame, a weak reference to it is dead.
        df = DataFrame({"a": [0, 1], "b": [2, 3]})
        for name in ("loc", "iloc", "at", "iat"):
            getattr(df, name)
        wr = weakref.ref(df)
        del df
        assert wr() is None

    def test_label_indexing_on_nan(self, nulls_fixture):
        # GH 32431
        # [FUNCTION DOC]
        # GH 32431. Uses the missing-values fixture. Value counts including missing values can be indexed
        # by the missing value with loc and getitem, giving 1.
        df = Series([1, "{1,2}", 1, nulls_fixture])
        vc = df.value_counts(dropna=False)
        result1 = vc.loc[nulls_fixture]
        result2 = vc[nulls_fixture]

        expected = 1
        assert result1 == expected
        assert result2 == expected


class TestDataframeNoneCoercion:
    # [CLASS DOC]
    # Test collection for setting None into frames: numeric coerces to NaN, datetime to NaT, object
    # keeps None. A class-level table of (start data, expected data, expected warning) rows is
    # shared.
    EXPECTED_SINGLE_ROW_RESULTS = [
        # For numeric series, we should coerce to NaN.
        ([1, 2, 3], [np.nan, 2, 3], FutureWarning),
        ([1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], None),
        # For datetime series, we should coerce to NaT.
        (
            [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)],
            None,
        ),
        # For objects, we should preserve the None value.
        (["foo", "bar", "baz"], [None, "bar", "baz"], None),
    ]

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_loc(self, expected):
        # [FUNCTION DOC]
        # Parametrized over the table rows. Setting None through loc at a position and a list of columns
        # gives the expected coerced values.
        # KNOWN ISSUE: the third element of each row (the expected warning class) is unpacked but never
        # checked in this test.
        start_data, expected_result, warn = expected

        start_dataframe = DataFrame({"foo": start_data})
        start_dataframe.loc[0, ["foo"]] = None

        expected_dataframe = DataFrame({"foo": expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_setitem_and_dataframe(self, expected):
        # [FUNCTION DOC]
        # Parametrized over the table rows. Setting None where a boolean frame mask selects the first
        # element gives the expected coerced values.
        # KNOWN ISSUE: the expected warning in each row is unpacked but never checked.
        start_data, expected_result, warn = expected

        start_dataframe = DataFrame({"foo": start_data})
        start_dataframe[start_dataframe["foo"] == start_dataframe["foo"][0]] = None

        expected_dataframe = DataFrame({"foo": expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_none_coercion_loc_and_dataframe(self, expected):
        # [FUNCTION DOC]
        # Parametrized over the table rows. Setting None through loc with a boolean frame mask gives the
        # expected coerced values.
        # KNOWN ISSUE: the expected warning in each row is unpacked but never checked.
        start_data, expected_result, warn = expected

        start_dataframe = DataFrame({"foo": start_data})
        start_dataframe.loc[start_dataframe["foo"] == start_dataframe["foo"][0]] = None

        expected_dataframe = DataFrame({"foo": expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    def test_none_coercion_mixed_dtypes(self):
        # [FUNCTION DOC]
        # Setting None to the first row of a mixed-dtype frame yields NaN for numeric, NaT for datetime
        # and None for object columns.
        start_dataframe = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.0, 2.0, 3.0],
                "c": [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                "d": ["a", "b", "c"],
            }
        )
        start_dataframe.iloc[0] = None

        exp = DataFrame(
            {
                "a": [np.nan, 2, 3],
                "b": [np.nan, 2.0, 3.0],
                "c": [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)],
                "d": [None, "b", "c"],
            }
        )
        tm.assert_frame_equal(start_dataframe, exp)


class TestDatetimelikeCoercion:
    # [CLASS DOC]
    # Test collection for setting values into datetime-like series via accessors without changing the
    # underlying array buffer (where possible), and validating that the value is acceptable to the
    # array before assignment.
    def test_setitem_dt64_string_scalar(self, tz_naive_fixture, indexer_sli):
        # dispatching _can_hold_element to underlying DatetimeArray
        # [FUNCTION DOC]
        # Uses the timezone-naive fixture and the setter fixture. The array first validates that a date
        # string is acceptable; setting that string at position 0 keeps the dtype; for naive data the
        # underlying buffer is reused, for aware data the same array object is kept.
        # KNOWN ISSUE: a TODO says the timezone-naive case could be made not to copy.
        tz = tz_naive_fixture

        dti = date_range("2016-01-01", periods=3, tz=tz)
        ser = Series(dti.copy(deep=True))

        values = ser._values

        newval = "2018-01-01"
        values._validate_setitem_value(newval)

        indexer_sli(ser)[0] = newval

        if tz is None:
            # TODO(EA2D): we can make this no-copy in tz-naive case too
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            assert ser._values is values

    @pytest.mark.parametrize("box", [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize(
        "key", [[0, 1], slice(0, 2), np.array([True, True, False])]
    )
    def test_setitem_dt64_string_values(self, tz_naive_fixture, indexer_sli, key, box):
        # dispatching _can_hold_element to underling DatetimeArray
        # [FUNCTION DOC]
        # Parametrized over several box types and keys (list, slice, boolean array), plus the timezone
        # and setter fixtures. A slice key for the label setter is shortened to one element to account for
        # inclusive bounds. String values (after validation) are set for the selected elements and the
        # dtype and underlying buffer are retained, as for the scalar test.
        # KNOWN ISSUE: a TODO says the timezone-naive case could be made not to copy.
        tz = tz_naive_fixture

        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)

        dti = date_range("2016-01-01", periods=3, tz=tz)
        ser = Series(dti.copy(deep=True))

        values = ser._values

        newvals = box(["2019-01-01", "2010-01-02"])
        values._validate_setitem_value(newvals)

        indexer_sli(ser)[key] = newvals

        if tz is None:
            # TODO(EA2D): we can make this no-copy in tz-naive case too
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            assert ser._values is values

    @pytest.mark.parametrize("scalar", ["3 Days", offsets.Hour(4)])
    def test_setitem_td64_scalar(self, indexer_sli, scalar):
        # dispatching _can_hold_element to underling TimedeltaArray
        # [FUNCTION DOC]
        # Parametrized over a string and an offset scalar. After validation by the array, setting at
        # position 0 keeps the same underlying buffer.
        tdi = timedelta_range("1 Day", periods=3)
        ser = Series(tdi.copy(deep=True))

        values = ser._values
        values._validate_setitem_value(scalar)

        indexer_sli(ser)[0] = scalar
        assert ser._values._ndarray is values._ndarray

    @pytest.mark.parametrize("box", [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize(
        "key", [[0, 1], slice(0, 2), np.array([True, True, False])]
    )
    def test_setitem_td64_string_values(self, indexer_sli, key, box):
        # dispatching _can_hold_element to underling TimedeltaArray
        # [FUNCTION DOC]
        # Parametrized over box types and keys. Setting duration strings keeps the same underlying buffer
        # (slice key shortened for label setter).
        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)

        tdi = timedelta_range("1 Day", periods=3)
        ser = Series(tdi.copy(deep=True))

        values = ser._values

        newvals = box(["10 Days", "44 hours"])
        values._validate_setitem_value(newvals)

        indexer_sli(ser)[key] = newvals
        assert ser._values._ndarray is values._ndarray


def test_extension_array_cross_section():
    # A cross-section of a homogeneous EA should be an EA
    # [FUNCTION DOC]
    # A row of a frame whose columns are all the same nullable integer dtype is a series of that
    # extension dtype (via label and position).
    df = DataFrame(
        {
            "A": pd.array([1, 2], dtype="Int64"),
            "B": pd.array([3, 4], dtype="Int64"),
        },
        index=["a", "b"],
    )
    expected = Series(pd.array([1, 3], dtype="Int64"), index=["A", "B"], name="a")
    result = df.loc["a"]
    tm.assert_series_equal(result, expected)

    result = df.iloc[0]
    tm.assert_series_equal(result, expected)


def test_extension_array_cross_section_converts():
    # all numeric columns -> numeric series
    # [FUNCTION DOC]
    # Mixed nullable and numpy integer columns give a nullable integer row; nullable integer with
    # strings give an object row; both via label and position.
    df = DataFrame(
        {
            "A": pd.array([1, 2], dtype="Int64"),
            "B": np.array([1, 2], dtype="int64"),
        },
        index=["a", "b"],
    )
    result = df.loc["a"]
    expected = Series([1, 1], dtype="Int64", index=["A", "B"], name="a")
    tm.assert_series_equal(result, expected)

    result = df.iloc[0]
    tm.assert_series_equal(result, expected)

    # mixed columns -> object series
    df = DataFrame(
        {"A": pd.array([1, 2], dtype="Int64"), "B": np.array(["a", "b"])},
        index=["a", "b"],
    )
    result = df.loc["a"]
    expected = Series([1, "a"], dtype=object, index=["A", "B"], name="a")
    tm.assert_series_equal(result, expected)

    result = df.iloc[0]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, keys",
    [(Series([10]), (0, 0)), (Series([1, 2, 3], index=list("abc")), (0, 1))],
)
def test_ser_tup_indexer_exceeds_dimensions(ser, keys, indexer_li):
    # GH#13831
    # [FUNCTION DOC]
    # GH 13831. Parametrized over two series and tuple keys with too many entries, with the
    # accessor fixture (label or position). Getting raises the indexing error about too many
    # indexers; setting raises the same, except the positional accessor which raises numpy's index error.
    exp_err, exp_msg = IndexingError, "Too many indexers"
    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys]

    if indexer_li == tm.iloc:
        # For iloc.__setitem__ we let numpy handle the error reporting.
        exp_err, exp_msg = IndexError, "too many indices for array"

    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys] = 0


def test_ser_list_indexer_exceeds_dimensions(indexer_li):
    # GH#13831
    # Make sure an exception is raised when a tuple exceeds the dimension of the series,
    # but not list when a list is used.
    # [FUNCTION DOC]
    # GH 13831. A list of repeated positions or labels is allowed on a series, returning repeated
    # elements.
    ser = Series([10])
    res = indexer_li(ser)[[0, 0]]
    exp = Series([10, 10], index=Index([0, 0]))
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize(
    "value", [(0, 1), [0, 1], np.array([0, 1]), array.array("b", [0, 1])]
)
def test_scalar_setitem_with_nested_value(value):
    # For numeric data, we try to unpack and thus raise for mismatching length
    # [FUNCTION DOC]
    # Parametrized over tuple, list, array and typed-array values. Assigning a two-element sequence to
    # a single cell of a new numeric column, or an existing object column, raises a value error about
    # length mismatch or about setting an element with a sequence.
    # KNOWN ISSUE: a TODO asks whether nested data should be preserved for object dtype, and
    # commented-out assertions at the end are dead code.
    df = DataFrame({"A": [1, 2, 3]})
    msg = "|".join(
        [
            "Must have equal len keys and value",
            "setting an array element with a sequence",
        ]
    )
    with pytest.raises(ValueError, match=msg):
        df.loc[0, "B"] = value

    # TODO For object dtype this happens as well, but should we rather preserve
    # the nested data and set as such?
    df = DataFrame({"A": [1, 2, 3], "B": np.array([1, "a", "b"], dtype=object)})
    with pytest.raises(ValueError, match="Must have equal len keys and value"):
        df.loc[0, "B"] = value
    # if isinstance(value, np.ndarray):
    #     assert (df.loc[0, "B"] == value).all()
    # else:
    #     assert df.loc[0, "B"] == value


@pytest.mark.parametrize(
    "value", [(0, 1), [0, 1], np.array([0, 1]), array.array("b", [0, 1])]
)
def test_scalar_setitem_series_with_nested_value(value, indexer_sli):
    # For numeric data, we try to unpack and thus raise for mismatching length
    # [FUNCTION DOC]
    # Parametrized over the same values and the setter fixture. On a numeric series assignment to a
    # position raises a value error about setting an element with a sequence; on an object series the
    # nested value is stored as is.
    ser = Series([1, 2, 3])
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        indexer_sli(ser)[0] = value

    # but for object dtype we preserve the nested data and set as such
    ser = Series([1, "a", "b"], dtype=object)
    indexer_sli(ser)[0] = value
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value


@pytest.mark.parametrize(
    "value", [(0.0,), [0.0], np.array([0.0]), array.array("d", [0.0])]
)
def test_scalar_setitem_with_nested_value_length1(value):
    # https://github.com/pandas-dev/pandas/issues/46268

    # For numeric data, assigning length-1 array to scalar position gets unpacked
    # [FUNCTION DOC]
    # Issue 46268. Parametrized over one-element sequences. On a new numeric column, the single
    # element is unpacked and set at the position with missing elsewhere; on an object column, the
    # sequence is preserved.
    df = DataFrame({"A": [1, 2, 3]})
    df.loc[0, "B"] = value
    expected = DataFrame({"A": [1, 2, 3], "B": [0.0, np.nan, np.nan]})
    tm.assert_frame_equal(df, expected)

    # but for object dtype we preserve the nested data
    df = DataFrame({"A": [1, 2, 3], "B": np.array([1, "a", "b"], dtype=object)})
    df.loc[0, "B"] = value
    if isinstance(value, np.ndarray):
        assert (df.loc[0, "B"] == value).all()
    else:
        assert df.loc[0, "B"] == value


@pytest.mark.parametrize(
    "value", [(0.0,), [0.0], np.array([0.0]), array.array("d", [0.0])]
)
def test_scalar_setitem_series_with_nested_value_length1(value, indexer_sli):
    # For numeric data, assigning length-1 array to scalar position gets unpacked
    # TODO this only happens in case of ndarray, should we make this consistent
    # for all list-likes? (as happens for DataFrame.(i)loc, see test above)
    # [FUNCTION DOC]
    # Parametrized over one-element sequences and the setter fixture. For a float series, a numpy
    # array is unpacked and set, while other sequence types raise a value error; for an object
    # series the nested value is preserved.
    # KNOWN ISSUE: a TODO notes that unpacking only happens for arrays, which is inconsistent with
    # the frame behavior.
    ser = Series([1.0, 2.0, 3.0])
    if isinstance(value, np.ndarray):
        indexer_sli(ser)[0] = value
        expected = Series([0.0, 2.0, 3.0])
        tm.assert_series_equal(ser, expected)
    else:
        with pytest.raises(
            ValueError, match="setting an array element with a sequence"
        ):
            indexer_sli(ser)[0] = value

    # but for object dtype we preserve the nested data
    ser = Series([1, "a", "b"], dtype=object)
    indexer_sli(ser)[0] = value
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value


def test_object_dtype_series_set_series_element():
    # GH 48933
    # [FUNCTION DOC]
    # GH 48933. An empty series can be stored as an element of an object series by getitem-style,
    # label and positional setting, and read back as an empty series.
    s1 = Series(dtype="O", index=["a", "b"])

    s1["a"] = Series()
    s1.loc["b"] = Series()

    tm.assert_series_equal(s1.loc["a"], Series())
    tm.assert_series_equal(s1.loc["b"], Series())

    s2 = Series(dtype="O", index=["a", "b"])

    s2.iloc[1] = Series()
    tm.assert_series_equal(s2.iloc[1], Series())
