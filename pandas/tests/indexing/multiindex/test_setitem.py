# [MODULE DOC]
# Tests for setitem / loc-assignment on frames and series with a MultiIndex: scalar and
# frame-valued assignment, augmented assignment, boolean masks, column enlargement, dtype
# changes and chained-assignment behavior. Defines a small equality helper used by a
# class-level check method.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
    isna,
    notna,
)
import pandas._testing as tm


def assert_equal(a, b):
    # [FUNCTION DOC]
    # HELPER: default comparison callback for the check method; asserts that two values are
    # equal using the equality operator, failing with an assertion error otherwise.
    assert a == b


class TestMultiIndexSetItem:
    # [CLASS DOC]
    # Test collection for assignment into MultiIndex objects. Provides a check helper method
    # that assigns through loc and reads back.
    def check(self, target, indexers, value, compare_fn=assert_equal, expected=None):
        # [FUNCTION DOC]
        # HELPER: assigns value to target via label-based assignment with indexers (mutating
        # target), reads it back with the same indexers, and compares the result with expected
        # using compare_fn (default is an equality assertion). If expected is None, value is used
        # as expected.
        target.loc[indexers] = value
        result = target.loc[indexers]
        if expected is None:
            expected = value
        compare_fn(result, expected)

    def test_setitem_multiindex(self):
        # GH#7190
        # [FUNCTION DOC]
        # GH 7190, gh-7218. On a large frame with a product index, setting a single cell through a
        # (tuple, column) key and reading it back gives the assigned value for NaN-filled,
        # integer-filled and empty-object frames; assigning a zero-dimensional array reads back as its
        # scalar value.
        cols = ["A", "w", "l", "a", "x", "X", "d", "profit"]
        index = MultiIndex.from_product(
            [np.arange(0, 100), np.arange(0, 80)], names=["time", "firm"]
        )
        t, n = 0, 2

        df = DataFrame(
            np.nan,
            columns=cols,
            index=index,
        )
        self.check(target=df, indexers=((t, n), "X"), value=0)

        df = DataFrame(-999, columns=cols, index=index)
        self.check(target=df, indexers=((t, n), "X"), value=1)

        df = DataFrame(columns=cols, index=index)
        self.check(target=df, indexers=((t, n), "X"), value=2)

        # gh-7218: assigning with 0-dim arrays
        df = DataFrame(-999, columns=cols, index=index)
        self.check(
            target=df,
            indexers=((t, n), "X"),
            value=np.array(3),
            expected=3,
        )

    def test_setitem_multiindex2(self):
        # GH#5206
        # [FUNCTION DOC]
        # GH 5206. Assigning a column (with a mask over rows and a list of columns) from a series
        # sets the selected cells to that column's value; both direct check and the helper give the
        # 99.0 frame.
        df = DataFrame(
            np.arange(25).reshape(5, 5), columns="A,B,C,D,E".split(","), dtype=float
        )
        df["F"] = 99
        row_selection = df["A"] % 2 == 0
        col_selection = ["B", "C"]
        df.loc[row_selection, col_selection] = df["F"]
        output = DataFrame(99.0, index=[0, 2, 4], columns=["B", "C"])
        tm.assert_frame_equal(df.loc[row_selection, col_selection], output)
        self.check(
            target=df,
            indexers=(row_selection, col_selection),
            value=df["F"],
            compare_fn=tm.assert_frame_equal,
            expected=output,
        )

    def test_setitem_multiindex3(self):
        # GH#11372
        # [FUNCTION DOC]
        # GH 11372. Assigning a frame with a smaller MultiIndex on both axes (date levels) via loc
        # sets those cells and reads back the same frame; also with all columns, and assigning the
        # frame to itself leaves it unchanged.
        idx = MultiIndex.from_product(
            [["A", "B", "C"], date_range("2015-01-01", "2015-04-01", freq="MS")]
        )
        cols = MultiIndex.from_product(
            [["foo", "bar"], date_range("2016-01-01", "2016-02-01", freq="MS")]
        )

        df = DataFrame(
            np.random.default_rng(2).random((12, 4)), index=idx, columns=cols
        )

        subidx = MultiIndex.from_arrays(
            [["A", "A"], date_range("2015-01-01", "2015-02-01", freq="MS")]
        )
        subcols = MultiIndex.from_arrays(
            [["foo", "foo"], date_range("2016-01-01", "2016-02-01", freq="MS")]
        )

        vals = DataFrame(
            np.random.default_rng(2).random((2, 2)), index=subidx, columns=subcols
        )
        self.check(
            target=df,
            indexers=(subidx, subcols),
            value=vals,
            compare_fn=tm.assert_frame_equal,
        )
        # set all columns
        vals = DataFrame(
            np.random.default_rng(2).random((2, 4)), index=subidx, columns=cols
        )
        self.check(
            target=df,
            indexers=(subidx, slice(None, None, None)),
            value=vals,
            compare_fn=tm.assert_frame_equal,
        )
        # identity
        copy = df.copy()
        self.check(
            target=df,
            indexers=(df.index, df.columns),
            value=df,
            compare_fn=tm.assert_frame_equal,
            expected=copy,
        )

    def test_multiindex_setitem(self):
        # GH 3738
        # setting with a multi-index right hand side
        # [FUNCTION DOC]
        # GH 3738. In-place multiplication on a loc list selection of one first-level label doubles
        # those rows. In-place multiplication using the scalar label (whose result has fewer levels)
        # raises a type error about aligning on a MultiIndex without specifying join levels.
        arrays = [
            np.array(["bar", "bar", "baz", "qux", "qux", "bar"]),
            np.array(["one", "two", "one", "one", "two", "one"]),
            np.arange(0, 6, 1),
        ]

        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((6, 3)),
            index=arrays,
            columns=["A", "B", "C"],
        ).sort_index()

        expected = df_orig.loc[["bar"]] * 2
        df = df_orig.copy()
        df.loc[["bar"]] *= 2
        tm.assert_frame_equal(df.loc[["bar"]], expected)

        # raise because these have differing levels
        msg = "cannot align on a multi-index with out specifying the join levels"
        with pytest.raises(TypeError, match=msg):
            df.loc["bar"] *= 2

    def test_multiindex_setitem2(self):
        # from SO
        # https://stackoverflow.com/questions/24572040/pandas-access-the-level-of-multiindex-for-inplace-operation
        # [FUNCTION DOC]
        # From a StackOverflow question. In-place multiply on selected rows using a slice-all, slice-all,
        # label IndexSlice (with all columns, or the price column) doubles exactly the rows with that
        # last-level label.
        df_orig = DataFrame.from_dict(
            {
                "price": {
                    ("DE", "Coal", "Stock"): 2,
                    ("DE", "Gas", "Stock"): 4,
                    ("DE", "Elec", "Demand"): 1,
                    ("FR", "Gas", "Stock"): 5,
                    ("FR", "Solar", "SupIm"): 0,
                    ("FR", "Wind", "SupIm"): 0,
                }
            }
        )
        df_orig.index = MultiIndex.from_tuples(
            df_orig.index, names=["Sit", "Com", "Type"]
        )

        expected = df_orig.copy()
        expected.iloc[[0, 1, 3]] *= 2

        idx = pd.IndexSlice
        df = df_orig.copy()
        df.loc[idx[:, :, "Stock"], :] *= 2
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc[idx[:, :, "Stock"], "price"] *= 2
        tm.assert_frame_equal(df, expected)

    def test_multiindex_assignment(self):
        # GH3777 part 2

        # mixed dtype
        # [FUNCTION DOC]
        # GH 3777 part 2. Assigning an array of floats to a column (NaN-initialized float column) for
        # one first-level label sets values aligned with the second-level labels.
        df = DataFrame(
            np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3),
            columns=list("abc"),
            index=[[4, 4, 8], [8, 10, 12]],
        )
        df["d"] = np.nan
        arr = np.array([0.0, 1.0])

        df.loc[4, "d"] = arr
        tm.assert_series_equal(df.loc[4, "d"], Series(arr, index=[8, 10], name="d"))

    def test_multiindex_assignment_single_dtype(self):
        # GH3777 part 2b
        # single dtype
        # [FUNCTION DOC]
        # GH 3777 part 2b. For an all-integer frame: assigning losslessly castable floats keeps the
        # integer dtype in place; non-integer floats raise a type error (invalid value) until the
        # column is cast to float; scalar assignment works; length-mismatched lists raise a value
        # error; a length-one list for a list-like column indexer sets zero for all selected rows.
        arr = np.array([0.0, 1.0])

        df = DataFrame(
            np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3),
            columns=list("abc"),
            index=[[4, 4, 8], [8, 10, 12]],
            dtype=np.int64,
        )

        # arr can be losslessly cast to int, so this setitem is inplace
        df.loc[4, "c"] = arr
        exp = Series(arr, index=[8, 10], name="c", dtype="int64")
        result = df.loc[4, "c"]
        tm.assert_series_equal(result, exp)

        # arr + 0.5 cannot be cast losslessly to int, so we upcast
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[4, "c"] = arr + 0.5
        # Upcast so that we can add .5
        df = df.astype({"c": "float64"})
        df.loc[4, "c"] = arr + 0.5

        # scalar ok
        df.loc[4, "c"] = 10
        exp = Series(10, index=[8, 10], name="c", dtype="float64")
        tm.assert_series_equal(df.loc[4, "c"], exp)

        # invalid assignments
        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[4, "c"] = [0, 1, 2, 3]

        with pytest.raises(ValueError, match=msg):
            df.loc[4, "c"] = [0]

        # But with a length-1 listlike column indexer this behaves like
        #  `df.loc[4, "c"] = 0
        df.loc[4, ["c"]] = [0]
        assert (df.loc[4, "c"] == 0).all()

    def test_groupby_example(self):
        # groupby example
        # [FUNCTION DOC]
        # Smoke test. Builds a frame with a five-level index, groups by the first four levels, and
        # assigns per-group positions to a new column using group names as keys. No assertions;
        # checks that no exception is raised.
        NUM_ROWS = 100
        NUM_COLS = 10
        col_names = ["A" + num for num in map(str, np.arange(NUM_COLS).tolist())]
        index_cols = col_names[:5]

        df = DataFrame(
            np.random.default_rng(2).integers(5, size=(NUM_ROWS, NUM_COLS)),
            dtype=np.int64,
            columns=col_names,
        )
        df = df.set_index(index_cols).sort_index()
        grp = df.groupby(level=index_cols[:4])
        df["new_col"] = np.nan

        # we are actually operating on a copy here
        # but in this case, that's ok
        for name, df2 in grp:
            new_vals = np.arange(df2.shape[0])
            df.loc[name, "new_col"] = new_vals

    def test_series_setitem(self, multiindex_year_month_day_dataframe_random_data):
        # [FUNCTION DOC]
        # Assigning NaN with a partial key (year-month) sets that block of the series to missing and
        # leaves others valid; a full key sets one entry; an absent integer key raises a key error and
        # does not fall back to position (GH 33355).
        ymd = multiindex_year_month_day_dataframe_random_data
        s = ymd["A"]

        s[2000, 3] = np.nan
        assert isna(s.values[42:65]).all()
        assert notna(s.values[:42]).all()
        assert notna(s.values[65:]).all()

        s[2000, 3, 10] = np.nan
        assert isna(s.iloc[49])

        with pytest.raises(KeyError, match="49"):
            # GH#33355 dont fall-back to positional when leading level is int
            s[49]

    def test_frame_getitem_setitem_boolean(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Boolean-frame getitem equals where with the same mask; boolean-frame setitem with scalar
        # matches the same operation on the raw values, including a mask needing alignment (shorter
        # mask) ; a non-boolean mask raises a type error about boolean values only.
        frame = multiindex_dataframe_random_data
        df = frame.T.copy()
        values = df.values.copy()

        result = df[df > 0]
        expected = df.where(df > 0)
        tm.assert_frame_equal(result, expected)

        df[df > 0] = 5
        values[values > 0] = 5
        tm.assert_almost_equal(df.values, values)

        df[df == 5] = 0
        values[values == 5] = 0
        tm.assert_almost_equal(df.values, values)

        # a df that needs alignment first
        df[df[:-1] < 0] = 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        tm.assert_almost_equal(df.values, values)

        with pytest.raises(TypeError, match="boolean values only"):
            df[df * 0] = 2

    def test_frame_getitem_setitem_multislice(self):
        # [FUNCTION DOC]
        # Slicing all rows with a column label, a subset of rows, and all-slice yields expected
        # series / frames; assigning a scalar by loc to a column and to the whole frame matches
        # the direct assignment.
        # KNOWN ISSUE: the variable named result is an alias of the original frame, not a copy, so the
        # following equality comparisons compare the object with itself and cannot fail.
        levels = [["t1", "t2"], ["a", "b", "c"]]
        codes = [[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]]
        midx = MultiIndex(codes=codes, levels=levels, names=[None, "id"])
        df = DataFrame({"value": [1, 2, 3, 7, 8]}, index=midx)

        result = df.loc[:, "value"]
        tm.assert_series_equal(df["value"], result)

        result = df.loc[df.index[1:3], "value"]
        tm.assert_series_equal(df["value"][1:3], result)

        result = df.loc[:, :]
        tm.assert_frame_equal(df, result)

        result = df
        df.loc[:, "value"] = 10
        result["value"] = 10
        tm.assert_frame_equal(df, result)

        df.loc[:, :] = 10
        tm.assert_frame_equal(df, result)

    def test_frame_setitem_multi_column(self):
        # [FUNCTION DOC]
        # Assigning one first-level group of columns from another (as a frame or as a raw array)
        # makes both groups equal.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=[["a", "a", "b", "b"], [0, 1, 0, 1]],
        )

        cp = df.copy()
        cp["a"] = cp["b"]
        tm.assert_frame_equal(cp["a"], cp["b"])

        # set with ndarray
        cp = df.copy()
        cp["a"] = cp["b"].values
        tm.assert_frame_equal(cp["a"], cp["b"])

    def test_frame_setitem_multi_column2(self):
        # ---------------------------------------
        # GH#1803
        # [FUNCTION DOC]
        # GH 1803. Assigning a scalar to a first-level key in an empty (NaN) frame fills all its
        # subcolumns; a new full-tuple column can be set from a list; assigning that column to a group
        # broadcasts it across all subcolumns, and the column names of the selections stay the tuples.
        # KNOWN ISSUE: comments claim the first assignment "adds a column instead of updating" and
        # "doesn't work", contradicting the assertion that all values are updated to zero.
        columns = MultiIndex.from_tuples([("A", "1"), ("A", "2"), ("B", "1")])
        df = DataFrame(index=[1, 3, 5], columns=columns)

        # Works, but adds a column instead of updating the two existing ones
        df["A"] = 0.0  # Doesn't work
        assert (df["A"].values == 0).all()

        # it broadcasts
        df["B", "1"] = [1, 2, 3]
        df["A"] = df["B", "1"]

        sliced_a1 = df["A", "1"]
        sliced_a2 = df["A", "2"]
        sliced_b1 = df["B", "1"]
        tm.assert_series_equal(sliced_a1, sliced_b1, check_names=False)
        tm.assert_series_equal(sliced_a2, sliced_b1, check_names=False)
        assert sliced_a1.name == ("A", "1")
        assert sliced_a2.name == ("A", "2")
        assert sliced_b1.name == ("B", "1")

    def test_loc_getitem_tuple_plus_columns(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        # GH #1013
        # [FUNCTION DOC]
        # GH 1013. A full row tuple with a list of columns returns the same series as getting the
        # row by three scalars and then selecting the columns.
        ymd = multiindex_year_month_day_dataframe_random_data
        df = ymd[:5]

        result = df.loc[(2000, 1, 6), ["A", "B", "C"]]
        expected = df.loc[2000, 1, 6][["A", "B", "C"]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_loc_getitem_setitem_slice_integers(self, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over frame and series. Integer slicing on the first level is label-based and
        # inclusive of the end; the read returns the rows from position 2 on and assigning a scalar
        # through the slice sets them. A warning about setting on a view is filtered.
        index = MultiIndex(
            levels=[[0, 1, 2], [0, 2]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]
        )

        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 4)),
            index=index,
            columns=["a", "b", "c", "d"],
        )
        obj = tm.get_obj(obj, frame_or_series)

        res = obj.loc[1:2]
        exp = obj.reindex(obj.index[2:])
        tm.assert_equal(res, exp)

        obj.loc[1:2] = 7
        assert (obj.loc[1:2] == 7).values.all()

    def test_setitem_change_dtype(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Replacing a boolean-valued entry column on a transposed frame with a comparison result
        # changes the column's dtype and the replaced values are retrievable by getitem and after
        # reindexing columns.
        # KNOWN ISSUE: a commented-out assertion on internal block items remains in the body; it is dead
        # code.
        frame = multiindex_dataframe_random_data
        dft = frame.T
        s = dft["foo", "two"]
        dft["foo", "two"] = s > s.median()
        tm.assert_series_equal(dft["foo", "two"], s > s.median())
        # assert isinstance(dft._data.blocks[1].items, MultiIndex)

        reindexed = dft.reindex(columns=[("foo", "two")])
        tm.assert_series_equal(reindexed["foo", "two"], s > s.median())

    def test_set_column_scalar_with_loc(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Assigning a scalar to selected rows via loc sets them. Chained assignment through a column
        # object (selecting a column then assigning by subset) does not modify the original frame
        # (copy-on-write).
        frame = multiindex_dataframe_random_data
        subset = frame.index[[1, 4, 5]]

        frame.loc[subset] = 99
        assert (frame.loc[subset].values == 99).all()

        frame_original = frame.copy()
        col = frame["B"]
        col[subset] = 97
        # chained setitem doesn't work with CoW
        tm.assert_frame_equal(frame, frame_original)

    def test_nonunique_assignment_1750(self):
        # [FUNCTION DOC]
        # Issue 1750. Assigning a string through a MultiIndex key to a column of a frame with duplicate
        # index entries sets all duplicated rows.
        df = DataFrame(
            [[1, 1, "x", "X"], [1, 1, "y", "Y"], [1, 2, "z", "Z"]], columns=list("ABCD")
        )

        df = df.set_index(["A", "B"])
        mi = MultiIndex.from_tuples([(1, 1)])

        df.loc[mi, "C"] = "_"

        assert (df.xs((1, 1))["C"] == "_").all()

    def test_astype_assignment_with_dups(self):
        # GH 4686
        # assignment with dups that has a dtype change
        # [FUNCTION DOC]
        # GH 4686. Replacing duplicated-label columns with a float-cast version leaves the row index
        # unchanged.
        cols = MultiIndex.from_tuples([("A", "1"), ("B", "1"), ("A", "2")])
        df = DataFrame(np.arange(3).reshape((1, 3)), columns=cols, dtype=object)
        index = df.index.copy()

        df["A"] = df["A"].astype(np.float64)
        tm.assert_index_equal(df.index, index)

    def test_setitem_nonmonotonic(self):
        # https://github.com/pandas-dev/pandas/issues/31449
        # [FUNCTION DOC]
        # Issue 31449. On a non-monotonic index, assigning an integer array to rows for one first-level
        # label sets them in order of occurrence.
        index = MultiIndex.from_tuples(
            [("a", "c"), ("b", "x"), ("a", "d")], names=["l1", "l2"]
        )
        df = DataFrame(data=[0, 1, 2], index=index, columns=["e"])
        df.loc["a", "e"] = np.arange(99, 101, dtype="int64")
        expected = DataFrame({"e": [99, 1, 100]}, index=index)
        tm.assert_frame_equal(df, expected)


class TestSetitemWithExpansionMultiIndex:
    # [CLASS DOC]
    # Test collection for setitem that enlarges the frame or its columns with a MultiIndex.
    def test_setitem_new_column_mixed_depth(self):
        # [FUNCTION DOC]
        # Setting a new column with a short key on frame with columns of mixed tuple depth equals
        # setting using the full padded tuple with empty strings.
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        result = df.copy()
        expected = df.copy()
        result["b"] = [1, 2, 3, 4]
        expected["b", "", ""] = [1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    def test_setitem_new_column_all_na(self):
        # GH#1534
        # [FUNCTION DOC]
        # GH 1534. Setting a new column from a series whose index does not overlap the frame's index
        # produces an all-missing column.
        mix = MultiIndex.from_tuples([("1a", "2a"), ("1a", "2b"), ("1a", "2c")])
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mix)
        s = Series({(1, 1): 1, (1, 2): 2})
        df["new"] = s
        assert df["new"].isna().all()

    def test_setitem_enlargement_keep_index_names(self):
        # GH#53053
        # [FUNCTION DOC]
        # GH 53053. Enlarging a frame by assigning a row via a new full tuple preserves the index
        # level names.
        mi = MultiIndex.from_tuples([(1, 2, 3)], names=["i1", "i2", "i3"])
        df = DataFrame(data=[[10, 20, 30]], index=mi, columns=["A", "B", "C"])
        df.loc[(0, 0, 0)] = df.loc[(1, 2, 3)]
        mi_expected = MultiIndex.from_tuples(
            [(1, 2, 3), (0, 0, 0)], names=["i1", "i2", "i3"]
        )
        expected = DataFrame(
            data=[[10, 20, 30], [10, 20, 30]],
            index=mi_expected,
            columns=["A", "B", "C"],
        )
        tm.assert_frame_equal(df, expected)

    def test_setitem_enlargement_multiindex_with_none(self):
        # GH#59153
        # enlarging a DataFrame with a MultiIndex containing None values
        # [FUNCTION DOC]
        # GH 59153. Enlarging a frame whose MultiIndex contains None by assigning a row with tuple
        # containing None appends it to the end; index type is not checked strictly.
        index = MultiIndex.from_tuples(
            [("A", "a1"), ("A", "a2"), ("B", "b1"), ("B", None)]
        )
        df = DataFrame([(0.0, 6.0), (1.0, 5.0), (2.0, 4.0), (3.0, 7.0)], index=index)
        df.loc[("A", None), :] = [12.0, 13.0]
        expected_index = MultiIndex.from_tuples(
            [("A", "a1"), ("A", "a2"), ("B", "b1"), ("B", None), ("A", None)]
        )
        expected = DataFrame(
            [[0.0, 6.0], [1.0, 5.0], [2.0, 4.0], [3.0, 7.0], [12.0, 13.0]],
            index=expected_index,
            columns=[0, 1],
        )
        tm.assert_frame_equal(df, expected, check_index_type=False)


def test_frame_setitem_view_direct(multiindex_dataframe_random_data):
    # this works because we are modifying the underlying array
    # really a no-no
    # [FUNCTION DOC]
    # Writing through the values array of a selected column raises a value error about read-only
    # data, and the frame is unchanged (all remain nonzero).
    # KNOWN ISSUE: the comment says "this works because we are modifying the underlying array", but
    # the test expects a read-only error, so the comment is stale.
    df = multiindex_dataframe_random_data.T
    with pytest.raises(ValueError, match="read-only"):
        df["foo"].values[:] = 0
    assert (df["foo"].values != 0).all()


def test_frame_setitem_copy_raises(multiindex_dataframe_random_data):
    # will raise/warn as its chained assignment
    # [FUNCTION DOC]
    # Chained assignment via a column selection on a transposed frame emits the chained-assignment
    # error / warning (context manager).
    df = multiindex_dataframe_random_data.T
    with tm.raises_chained_assignment_error():
        df["foo"]["one"] = 2


def test_frame_setitem_copy_no_write(multiindex_dataframe_random_data):
    # [FUNCTION DOC]
    # The chained assignment on a copy raises the chained-assignment error and leaves the frame
    # equal to the original.
    frame = multiindex_dataframe_random_data.T
    expected = frame
    df = frame.copy()
    with tm.raises_chained_assignment_error():
        df["foo"]["one"] = 2

    tm.assert_frame_equal(df, expected)


def test_frame_setitem_partial_multiindex():
    # GH 54875
    # [FUNCTION DOC]
    # GH 54875. Setting a column from a series whose index has fewer levels (one level dropped)
    # aligns on the shared levels, giving the same result as setting a scalar 8.
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 4, 5],
            "c": 6,
            "d": 7,
        }
    ).set_index(["a", "b", "c"])
    ser = Series(8, index=df.index.droplevel("c"))
    result = df.copy()
    result["d"] = ser
    expected = df.copy()
    expected["d"] = 8
    tm.assert_frame_equal(result, expected)
