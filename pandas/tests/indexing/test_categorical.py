# [MODULE DOC]
# Tests for indexing (loc, iloc, at, iat, getitem, setitem) on objects whose index or values
# are categorical, including unused categories, non-string categories, NaN categories and
# non-unique indexes. Defines two fixtures and one test class.
import re

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    Interval,
    Series,
    Timedelta,
    Timestamp,
    option_context,
)
import pandas._testing as tm


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a one-column integer frame (values 0-5) with a categorical index
    # named B, values a a b b c a, with categories c, a, b.
    return DataFrame(
        {
            "A": np.arange(6, dtype="int64"),
        },
        index=CategoricalIndex(
            list("aabbca"), dtype=CategoricalDtype(list("cab")), name="B"
        ),
    )


@pytest.fixture
def df2():
    # [FUNCTION DOC]
    # HELPER: fixture like the primary one but the categorical index has an extra unused
    # category e (categories c, a, b, e).
    return DataFrame(
        {
            "A": np.arange(6, dtype="int64"),
        },
        index=CategoricalIndex(
            list("aabbca"), dtype=CategoricalDtype(list("cabe")), name="B"
        ),
    )


class TestCategoricalIndex:
    # [CLASS DOC]
    # Test collection for indexing behavior with a categorical index (and categorical data).
    def test_loc_scalar(self, df):
        # [FUNCTION DOC]
        # Scalar label returns the matching rows keeping the categorical index dtype; assigning a
        # scalar through the label sets all matches; an absent label raises a key error; assigning to
        # a new label outside the categories succeeds on a copy by converting the index to object.
        dtype = CategoricalDtype(list("cab"))
        result = df.loc["a"]
        bidx = Series(list("aaa"), name="B").astype(dtype)
        assert bidx.dtype == dtype

        expected = DataFrame({"A": [0, 1, 5]}, index=Index(bidx))
        tm.assert_frame_equal(result, expected)

        df = df.copy()
        df.loc["a"] = 20
        bidx2 = Series(list("aabbca"), name="B").astype(dtype)
        assert bidx2.dtype == dtype
        expected = DataFrame(
            {
                "A": [20, 20, 2, 3, 4, 20],
            },
            index=Index(bidx2),
        )
        tm.assert_frame_equal(df, expected)

        # value not in the categories
        with pytest.raises(KeyError, match=r"^'d'$"):
            df.loc["d"]

        df2 = df.copy()
        expected = df2.copy()
        expected.index = expected.index.astype(object)
        expected.loc["d"] = 10
        df2.loc["d"] = 10
        tm.assert_frame_equal(df2, expected)

    def test_loc_setitem_with_expansion_non_category(self, df):
        # Setting-with-expansion with a new key "d" that is not among caegories
        # [FUNCTION DOC]
        # Setting with expansion by a label not among the categories converts the index to object:
        # adds a row on an existing column (column becomes float) and adds both a new row and new column
        # (missing values elsewhere).
        # KNOWN ISSUE: a comment in the body misspells "categories" (caegories); harmless.
        df.loc["a"] = 20

        # Setting a new row on an existing column
        df3 = df.copy()
        df3.loc["d", "A"] = 10
        bidx3 = Index(list("aabbcad"), name="B")
        expected3 = DataFrame(
            {
                "A": [20, 20, 2, 3, 4, 20, 10.0],
            },
            index=Index(bidx3),
        )
        tm.assert_frame_equal(df3, expected3)

        # Setting a new row _and_ new column
        df4 = df.copy()
        df4.loc["d", "C"] = 10
        expected3 = DataFrame(
            {
                "A": [20, 20, 2, 3, 4, 20, np.nan],
                "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10],
            },
            index=Index(bidx3),
        )
        tm.assert_frame_equal(df4, expected3)

    def test_loc_getitem_scalar_non_category(self, df):
        # [FUNCTION DOC]
        # A scalar label of the wrong type (integer) raises a key error with just the label as message.
        with pytest.raises(KeyError, match="^1$"):
            df.loc[1]

    def test_slicing(self):
        # [FUNCTION DOC]
        # Reverse slicing of a categorical series yields reversed values. A frame with an interval
        # category column built by binning: integer positional and slice row access return the row /
        # frame holding interval values, and label access returns the right row.
        cat = Series(Categorical([1, 2, 3, 4]))
        reverse = cat[::-1]
        exp = np.array([4, 3, 2, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(reverse.__array__(), exp)

        df = DataFrame({"value": (np.arange(100) + 1).astype("int64")})
        df["D"] = pd.cut(df.value, bins=[0, 25, 50, 75, 100])

        expected = Series([11, Interval(0, 25)], index=["value", "D"], name=10)
        result = df.iloc[10]
        tm.assert_series_equal(result, expected)

        expected = DataFrame(
            {"value": np.arange(11, 21).astype("int64")},
            index=np.arange(10, 20).astype("int64"),
        )
        expected["D"] = pd.cut(expected.value, bins=[0, 25, 50, 75, 100])
        result = df.iloc[10:20]
        tm.assert_frame_equal(result, expected)

        expected = Series([9, Interval(0, 25)], index=["value", "D"], name=8)
        result = df.loc[8]
        tm.assert_series_equal(result, expected)

    def test_slicing_and_getting_ops(self):
        # systematically test the slicing operations:
        #  for all slicing ops:
        #   - returning a dataframe
        #   - returning a column
        #   - returning a row
        #   - returning a single value

        # [FUNCTION DOC]
        # Systematic check of slicing a frame with a categorical column and a plain label index with
        # positional and label variants (iloc, loc, iat, at, mask selections, integer / slice / list
        # inputs). Returned frames and columns keep categorical dtype; a row series has object dtype with
        # the category value as plain string; scalar equals the expected category.
        cats = Categorical(
            ["a", "c", "b", "c", "c", "c", "c"], categories=["a", "b", "c"]
        )
        idx = Index(["h", "i", "j", "k", "l", "m", "n"])
        values = [1, 2, 3, 4, 5, 6, 7]
        df = DataFrame({"cats": cats, "values": values}, index=idx)

        # the expected values
        cats2 = Categorical(["b", "c"], categories=["a", "b", "c"])
        idx2 = Index(["j", "k"])
        values2 = [3, 4]

        # 2:4,: | "j":"k",:
        exp_df = DataFrame({"cats": cats2, "values": values2}, index=idx2)

        # :,"cats" | :,0
        exp_col = Series(cats, index=idx, name="cats")

        # "j",: | 2,:
        exp_row = Series(["b", 3], index=["cats", "values"], dtype="object", name="j")

        # "j","cats | 2,0
        exp_val = "b"

        # iloc
        # frame
        res_df = df.iloc[2:4, :]
        tm.assert_frame_equal(res_df, exp_df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        # row
        res_row = df.iloc[2, :]
        tm.assert_series_equal(res_row, exp_row)
        assert isinstance(res_row["cats"], str)

        # col
        res_col = df.iloc[:, 0]
        tm.assert_series_equal(res_col, exp_col)
        assert isinstance(res_col.dtype, CategoricalDtype)

        # single value
        res_val = df.iloc[2, 0]
        assert res_val == exp_val

        # loc
        # frame
        res_df = df.loc["j":"k", :]
        tm.assert_frame_equal(res_df, exp_df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        # row
        res_row = df.loc["j", :]
        tm.assert_series_equal(res_row, exp_row)
        assert isinstance(res_row["cats"], str)

        # col
        res_col = df.loc[:, "cats"]
        tm.assert_series_equal(res_col, exp_col)
        assert isinstance(res_col.dtype, CategoricalDtype)

        # single value
        res_val = df.loc["j", "cats"]
        assert res_val == exp_val

        # single value
        res_val = df.loc["j", df.columns[0]]
        assert res_val == exp_val

        # iat
        res_val = df.iat[2, 0]
        assert res_val == exp_val

        # at
        res_val = df.at["j", "cats"]
        assert res_val == exp_val

        # fancy indexing
        exp_fancy = df.iloc[[2]]

        res_fancy = df[df["cats"] == "b"]
        tm.assert_frame_equal(res_fancy, exp_fancy)
        res_fancy = df[df["values"] == 3]
        tm.assert_frame_equal(res_fancy, exp_fancy)

        # get_value
        res_val = df.at["j", "cats"]
        assert res_val == exp_val

        # i : int, slice, or sequence of integers
        res_row = df.iloc[2]
        tm.assert_series_equal(res_row, exp_row)
        assert isinstance(res_row["cats"], str)

        res_df = df.iloc[slice(2, 4)]
        tm.assert_frame_equal(res_df, exp_df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        res_df = df.iloc[[2, 3]]
        tm.assert_frame_equal(res_df, exp_df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        res_col = df.iloc[:, 0]
        tm.assert_series_equal(res_col, exp_col)
        assert isinstance(res_col.dtype, CategoricalDtype)

        res_df = df.iloc[:, slice(0, 2)]
        tm.assert_frame_equal(res_df, df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        res_df = df.iloc[:, [0, 1]]
        tm.assert_frame_equal(res_df, df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

    def test_slicing_doc_examples(self):
        # GH 7918
        # [FUNCTION DOC]
        # GH 7918. Examples from the documentation: positional slices, dtype listing of a slice
        # (category and int64), label slice on a categorical column producing a series, and a label slice
        # with list of columns producing a frame.
        cats = Categorical(
            ["a", "b", "b", "b", "c", "c", "c"], categories=["a", "b", "c"]
        )
        idx = Index(["h", "i", "j", "k", "l", "m", "n"])
        values = [1, 2, 2, 2, 3, 4, 5]
        df = DataFrame({"cats": cats, "values": values}, index=idx)

        result = df.iloc[2:4, :]
        expected = DataFrame(
            {
                "cats": Categorical(["b", "b"], categories=["a", "b", "c"]),
                "values": [2, 2],
            },
            index=["j", "k"],
        )
        tm.assert_frame_equal(result, expected)

        result = df.iloc[2:4, :].dtypes
        expected = Series(["category", "int64"], ["cats", "values"], dtype=object)
        tm.assert_series_equal(result, expected)

        result = df.loc["h":"j", "cats"]
        expected = Series(
            Categorical(["a", "b", "b"], categories=["a", "b", "c"]),
            index=["h", "i", "j"],
            name="cats",
        )
        tm.assert_series_equal(result, expected)

        result = df.loc["h":"j", df.columns[0:1]]
        expected = DataFrame(
            {"cats": Categorical(["a", "b", "b"], categories=["a", "b", "c"])},
            index=["h", "i", "j"],
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_listlike_labels(self, df):
        # list of labels
        # [FUNCTION DOC]
        # A list of labels returns rows grouped in the list order (all rows for the first label, then
        # the next), with the categorical index type.
        result = df.loc[["c", "a"]]
        expected = df.iloc[[4, 0, 1, 5]]
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_loc_getitem_listlike_unused_category(self, df2):
        # GH#37901 a label that is in index.categories but not in index
        # listlike containing an element in the categories but not in the values
        # [FUNCTION DOC]
        # GH 37901. A list containing a label that is a category without matching rows raises a key error
        # that lists it as not in the index.
        with pytest.raises(KeyError, match=re.escape("['e'] not in index")):
            df2.loc[["a", "b", "e"]]

    def test_loc_getitem_label_unused_category(self, df2):
        # element in the categories but not in the values
        # [FUNCTION DOC]
        # A scalar label that is an unused category raises a key error with the label as message.
        with pytest.raises(KeyError, match=r"^'e'$"):
            df2.loc["e"]

    def test_loc_getitem_non_category(self, df2):
        # not all labels in the categories
        # [FUNCTION DOC]
        # A list containing a label not in the categories raises a key error naming it as not in the
        # index.
        with pytest.raises(KeyError, match=re.escape("['d'] not in index")):
            df2.loc[["a", "d"]]

    def test_loc_setitem_expansion_label_unused_category(self, df2):
        # assigning with a label that is in the categories but not in the index
        # [FUNCTION DOC]
        # Assigning a scalar via an unused-category label appends a row keeping the categorical index and
        # categories; reading a list containing the label returns the grouped rows including the new
        # one.
        df = df2.copy()
        df.loc["e"] = 20
        result = df.loc[["a", "b", "e"]]
        exp_index = CategoricalIndex(list("aaabbe"), categories=list("cabe"), name="B")
        expected = DataFrame({"A": [0, 1, 5, 2, 3, 20]}, index=exp_index)
        tm.assert_frame_equal(result, expected)

    def test_loc_listlike_dtypes(self):
        # GH 11586

        # unique categories and codes
        # [FUNCTION DOC]
        # GH 11586. For unique categories, selecting a list of labels (unique, or with a duplicate)
        # returns rows in list order and a categorical index with all original categories; an absent
        # label raises a key error.
        index = CategoricalIndex(["a", "b", "c"])
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=index)

        # unique slice
        res = df.loc[["a", "b"]]
        exp_index = CategoricalIndex(["a", "b"], categories=index.categories)
        exp = DataFrame({"A": [1, 2], "B": [4, 5]}, index=exp_index)
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # duplicated slice
        res = df.loc[["a", "a", "b"]]

        exp_index = CategoricalIndex(["a", "a", "b"], categories=index.categories)
        exp = DataFrame({"A": [1, 1, 2], "B": [4, 4, 5]}, index=exp_index)
        tm.assert_frame_equal(res, exp, check_index_type=True)

        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]

    def test_loc_listlike_dtypes_duplicated_categories_and_codes(self):
        # duplicated categories and codes
        # [FUNCTION DOC]
        # With a categorical index with repeated values, list selection returns all matching rows
        # grouped per list label (duplicated list labels duplicate groups); absent label raises key
        # error.
        index = CategoricalIndex(["a", "b", "a"])
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=index)

        # unique slice
        res = df.loc[["a", "b"]]
        exp = DataFrame(
            {"A": [1, 3, 2], "B": [4, 6, 5]}, index=CategoricalIndex(["a", "a", "b"])
        )
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # duplicated slice
        res = df.loc[["a", "a", "b"]]
        exp = DataFrame(
            {"A": [1, 3, 1, 3, 2], "B": [4, 6, 4, 6, 5]},
            index=CategoricalIndex(["a", "a", "a", "a", "b"]),
        )
        tm.assert_frame_equal(res, exp, check_index_type=True)

        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]

    def test_loc_listlike_dtypes_unused_category(self):
        # contains unused category
        # [FUNCTION DOC]
        # Same as the duplicated-values test, with an index having unused categories; resulting index
        # keeps all categories.
        index = CategoricalIndex(["a", "b", "a", "c"], categories=list("abcde"))
        df = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=index)

        res = df.loc[["a", "b"]]
        exp = DataFrame(
            {"A": [1, 3, 2], "B": [5, 7, 6]},
            index=CategoricalIndex(["a", "a", "b"], categories=list("abcde")),
        )
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # duplicated slice
        res = df.loc[["a", "a", "b"]]
        exp = DataFrame(
            {"A": [1, 3, 1, 3, 2], "B": [5, 7, 5, 7, 6]},
            index=CategoricalIndex(["a", "a", "a", "a", "b"], categories=list("abcde")),
        )
        tm.assert_frame_equal(res, exp, check_index_type=True)

        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]

    def test_loc_getitem_listlike_unused_category_raises_keyerror(self):
        # key that is an *unused* category raises
        # [FUNCTION DOC]
        # An unused category as a scalar or inside a list raises a key error (the list form reports it as
        # not in the index).
        index = CategoricalIndex(["a", "b", "a", "c"], categories=list("abcde"))
        df = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=index)

        with pytest.raises(KeyError, match="e"):
            # For comparison, check the scalar behavior
            df.loc["e"]

        with pytest.raises(KeyError, match=re.escape("['e'] not in index")):
            df.loc[["a", "e"]]

    def test_ix_categorical_index(self):
        # GH 12531
        # [FUNCTION DOC]
        # GH 12531. For frames whose row and column labels are both categorical, loc with scalar labels
        # returns series / frames named appropriately, and loc with lists returns the subset with a
        # categorical index that keeps all original categories.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=list("ABC"),
            columns=list("XYZ"),
        )
        cdf = df.copy()
        cdf.index = CategoricalIndex(df.index)
        cdf.columns = CategoricalIndex(df.columns)

        expect = Series(df.loc["A", :], index=cdf.columns, name="A")
        tm.assert_series_equal(cdf.loc["A", :], expect)

        expect = Series(df.loc[:, "X"], index=cdf.index, name="X")
        tm.assert_series_equal(cdf.loc[:, "X"], expect)

        exp_index = CategoricalIndex(list("AB"), categories=["A", "B", "C"])
        expect = DataFrame(df.loc[["A", "B"], :], columns=cdf.columns, index=exp_index)
        tm.assert_frame_equal(cdf.loc[["A", "B"], :], expect)

        exp_columns = CategoricalIndex(list("XY"), categories=["X", "Y", "Z"])
        expect = DataFrame(df.loc[:, ["X", "Y"]], index=cdf.index, columns=exp_columns)
        tm.assert_frame_equal(cdf.loc[:, ["X", "Y"]], expect)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    def test_ix_categorical_index_non_unique(self, infer_string):
        # non-unique
        # [FUNCTION DOC]
        # Parametrized over the infer-string option (True requires the optional pyarrow dependency,
        # otherwise the test is skipped). Same as the previous test but with duplicated labels in both
        # axes; results contain the repeated entries, with categories restricted to the values that
        # occur.
        with option_context("future.infer_string", infer_string):
            df = DataFrame(
                np.random.default_rng(2).standard_normal((3, 3)),
                index=list("ABA"),
                columns=list("XYX"),
            )
            cdf = df.copy()
            cdf.index = CategoricalIndex(df.index)
            cdf.columns = CategoricalIndex(df.columns)

            exp_index = CategoricalIndex(list("AA"), categories=["A", "B"])
            expect = DataFrame(df.loc["A", :], columns=cdf.columns, index=exp_index)
            tm.assert_frame_equal(cdf.loc["A", :], expect)

            exp_columns = CategoricalIndex(list("XX"), categories=["X", "Y"])
            expect = DataFrame(df.loc[:, "X"], index=cdf.index, columns=exp_columns)
            tm.assert_frame_equal(cdf.loc[:, "X"], expect)

            expect = DataFrame(
                df.loc[["A", "B"], :],
                columns=cdf.columns,
                index=CategoricalIndex(list("AAB")),
            )
            tm.assert_frame_equal(cdf.loc[["A", "B"], :], expect)

            expect = DataFrame(
                df.loc[:, ["X", "Y"]],
                index=cdf.index,
                columns=CategoricalIndex(list("XXY")),
            )
            tm.assert_frame_equal(cdf.loc[:, ["X", "Y"]], expect)

    def test_loc_slice(self, df):
        # GH9748
        # [FUNCTION DOC]
        # GH 9748. An integer slice on a categorical index with string categories raises a type error
        # about slice indexing with that indexer type; a label slice returns the inclusive range.
        msg = (
            "cannot do slice indexing on CategoricalIndex with these "
            r"indexers \[1\] of type int"
        )
        with pytest.raises(TypeError, match=msg):
            df.loc[1:5]

        result = df.loc["b":"c"]
        expected = df.iloc[[2, 3, 4]]
        tm.assert_frame_equal(result, expected)

    def test_loc_and_at_with_categorical_index(self):
        # GH 20629
        # [FUNCTION DOC]
        # GH 20629. Label accessors on series and frames with a categorical index return the same
        # scalar for both loc and at.
        df = DataFrame(
            [[1, 2], [3, 4], [5, 6]], index=CategoricalIndex(["A", "B", "C"])
        )

        s = df[0]
        assert s.loc["A"] == 1
        assert s.at["A"] == 1

        assert df.loc["B", 1] == 4
        assert df.at["B", 1] == 4

    @pytest.mark.parametrize(
        "idx_values",
        [
            # python types
            [1, 2, 3],
            [-1, -2, -3],
            [1.5, 2.5, 3.5],
            [-1.5, -2.5, -3.5],
            # numpy int/uint
            *(np.array([1, 2, 3], dtype=dtype) for dtype in tm.ALL_INT_NUMPY_DTYPES),
            # numpy floats
            *(np.array([1.5, 2.5, 3.5], dtype=dtyp) for dtyp in tm.FLOAT_NUMPY_DTYPES),
            # numpy object
            np.array([1, "b", 3.5], dtype=object),
            # pandas scalars
            [Interval(1, 4), Interval(4, 6), Interval(6, 9)],
            [Timestamp(2019, 1, 1), Timestamp(2019, 2, 1), Timestamp(2019, 3, 1)],
            [Timedelta(1, "D"), Timedelta(2, "D"), Timedelta(3, "D")],
            # pandas Integer arrays
            *(pd.array([1, 2, 3], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES),
            # other pandas arrays
            pd.IntervalIndex.from_breaks([1, 4, 6, 9]).array,
            pd.date_range("2019-01-01", periods=3).array,
            pd.timedelta_range(start="1D", periods=3).array,
        ],
    )
    def test_loc_getitem_with_non_string_categories(self, idx_values, ordered):
        # GH-17569
        # [FUNCTION DOC]
        # GH 17569. Parametrized over many categories value kinds (python numbers, numpy dtypes, object,
        # intervals, timestamps, timedeltas, nullable integers, other arrays) and the ordered-flag
        # fixture. On a categorical index of the values: scalar, list and slice selection return the
        # expected rows; scalar, list and slice assignment through loc set the right entries.
        cat_idx = CategoricalIndex(idx_values, ordered=ordered)
        df = DataFrame({"A": ["foo", "bar", "baz"]}, index=cat_idx)
        sl = slice(idx_values[0], idx_values[1])

        # scalar selection
        result = df.loc[idx_values[0]]
        expected = Series(["foo"], index=["A"], name=idx_values[0])
        tm.assert_series_equal(result, expected)

        # list selection
        result = df.loc[idx_values[:2]]
        expected = DataFrame(["foo", "bar"], index=cat_idx[:2], columns=["A"])
        tm.assert_frame_equal(result, expected)

        # slice selection
        result = df.loc[sl]
        expected = DataFrame(["foo", "bar"], index=cat_idx[:2], columns=["A"])
        tm.assert_frame_equal(result, expected)

        # scalar assignment
        result = df.copy()
        result.loc[idx_values[0]] = "qux"
        expected = DataFrame({"A": ["qux", "bar", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)

        # list assignment
        result = df.copy()
        result.loc[idx_values[:2], "A"] = ["qux", "qux2"]
        expected = DataFrame({"A": ["qux", "qux2", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)

        # slice assignment
        result = df.copy()
        result.loc[sl, "A"] = ["qux", "qux2"]
        expected = DataFrame({"A": ["qux", "qux2", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)

    def test_getitem_categorical_with_nan(self):
        # GH#41933
        # [FUNCTION DOC]
        # GH 41933. Missing value as a label in a categorical index can be looked up by plain getitem
        # and loc on both series and frame, returning the third element.
        ci = CategoricalIndex(["A", "B", np.nan])

        ser = Series(range(3), index=ci)

        assert ser[np.nan] == 2
        assert ser.loc[np.nan] == 2

        df = DataFrame(ser)
        assert df.loc[np.nan, 0] == 2
        assert df.loc[np.nan][0] == 2

    def test_getitem_row_categorical_with_nan(self):
        # GH#58954
        # [FUNCTION DOC]
        # GH 58954. Selecting a row of a frame having a categorical column containing a missing value
        # (by position or by label) returns an object-typed series with NaN for the missing categorical.
        df = DataFrame({"a": [1, 2], "b": CategoricalIndex([1, None])})

        res = df.iloc[1]
        expected = Series([2, np.nan], index=df.columns, name=1)
        tm.assert_series_equal(res, expected)

        res = df.loc[1]
        tm.assert_series_equal(res, expected)

    def test_getitem_row_categorical_with_nan_bool(self):
        # GH#58954
        # [FUNCTION DOC]
        # GH 58954. Same for boolean categories: the row series is object dtype containing False and NaN.
        df = DataFrame({"a": [True, False], "b": CategoricalIndex([False, None])})

        res = df.iloc[1]
        expected = Series([False, np.nan], index=df.columns, dtype=object, name=1)
        tm.assert_series_equal(res, expected)

        res = df.loc[1]
        tm.assert_series_equal(res, expected)
