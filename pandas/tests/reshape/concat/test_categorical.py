# [MODULE DOC]
# Tests for concatenating categorical data (columns, series and categorical
# indexes) with the concatenate function: dtype preservation when categories
# match, fallback to object or the underlying dtype otherwise, handling of
# timezone/datetime values, missing columns and category ordering. Takes the
# sort and using_infer_string fixtures from conftest.
from datetime import datetime

import numpy as np

from pandas.errors import Pandas4Warning

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestCategoricalConcat:
    # [CLASS DOC]
    # Test collection for categorical behavior of the concatenate function.
    # Stateless.
    def test_categorical_concat(self, sort):
        # [FUNCTION DOC]
        # Concatenates a frame with a frame having an extra categorical column along
        # rows with ignored index. Expected result keeps the categorical dtype for that
        # column, filled with missing for the first frame's rows, and missing values for
        # the column absent from the second frame. Parametrized by sort (GH 10177).
        # See GH 10177
        df1 = DataFrame(
            np.arange(18, dtype="int64").reshape(6, 3), columns=["a", "b", "c"]
        )

        df2 = DataFrame(np.arange(14, dtype="int64").reshape(7, 2), columns=["a", "c"])

        cat_values = ["one", "one", "two", "one", "two", "two", "one"]
        df2["h"] = Series(Categorical(cat_values))

        res = pd.concat((df1, df2), axis=0, ignore_index=True, sort=sort)
        exp = DataFrame(
            {
                "a": [0, 3, 6, 9, 12, 15, 0, 2, 4, 6, 8, 10, 12],
                "b": [
                    1,
                    4,
                    7,
                    10,
                    13,
                    16,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "c": [2, 5, 8, 11, 14, 17, 1, 3, 5, 7, 9, 11, 13],
                "h": [None] * 6 + cat_values,
            }
        )
        exp["h"] = exp["h"].astype(df2["h"].dtype)
        tm.assert_frame_equal(res, exp)

    def test_categorical_concat_dtypes(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies the column dtypes after concatenating categorical, string and
        # integer series side by side with keys: the dtype comparison pinpoints which
        # column is object (or the string dtype in infer-string mode), integer and
        # categorical (GH 8143).
        # GH8143
        index = ["cat", "obj", "num"]
        cat = Categorical(["a", "b", "c"])
        obj = Series(["a", "b", "c"])
        num = Series([1, 2, 3])
        df = pd.concat([Series(cat), obj, num], axis=1, keys=index)

        result = df.dtypes == (object if not using_infer_string else "str")
        expected = Series([False, True, False], index=index)
        tm.assert_series_equal(result, expected)

        result = df.dtypes == "int64"
        expected = Series([False, False, True], index=index)
        tm.assert_series_equal(result, expected)

        result = df.dtypes == "category"
        expected = Series([True, False, False], index=index)
        tm.assert_series_equal(result, expected)

    def test_concat_categoricalindex(self):
        # [FUNCTION DOC]
        # Verifies side-by-side concatenation of series indexed by categorical indexes
        # whose categories are not sorted yields a categorical union index preserving
        # the category order (GH 16111).
        # GH 16111, categories that aren't lexsorted
        categories = [9, 0, 1, 2, 3]

        a = Series(1, index=pd.CategoricalIndex([9, 0], categories=categories))
        b = Series(2, index=pd.CategoricalIndex([0, 1], categories=categories))
        c = Series(3, index=pd.CategoricalIndex([1, 2], categories=categories))

        result = pd.concat([a, b, c], axis=1)

        exp_idx = pd.CategoricalIndex([9, 0, 1, 2], categories=categories)
        exp = DataFrame(
            {
                0: [1, 1, np.nan, np.nan],
                1: [np.nan, 2, 2, np.nan],
                2: [np.nan, np.nan, 3, 3],
            },
            columns=[0, 1, 2],
            index=exp_idx,
        )
        tm.assert_frame_equal(result, exp)

    def test_categorical_concat_preserve(self):
        # [FUNCTION DOC]
        # Verifies series and frame concatenation: different categories fall back to
        # object values, identical categories keep categorical dtype, and a frame with
        # a categorical column concatenated with itself keeps the category set and
        # order (GH 8641, GH 13524).
        # GH 8641  series concat not preserving category dtype
        # GH 13524 can concat different categories
        s = Series(list("abc"), dtype="category")
        s2 = Series(list("abd"), dtype="category")

        exp = Series(list("abcabd"))
        res = pd.concat([s, s2], ignore_index=True)
        tm.assert_series_equal(res, exp)

        exp = Series(list("abcabc"), dtype="category")
        res = pd.concat([s, s], ignore_index=True)
        tm.assert_series_equal(res, exp)

        exp = Series(list("abcabc"), index=[0, 1, 2, 0, 1, 2], dtype="category")
        res = pd.concat([s, s])
        tm.assert_series_equal(res, exp)

        a = Series(np.arange(6, dtype="int64"))
        b = Series(list("aabbca"))

        df2 = DataFrame({"A": a, "B": b.astype(CategoricalDtype(list("cab")))})
        res = pd.concat([df2, df2])
        exp = DataFrame(
            {
                "A": pd.concat([a, a]),
                "B": pd.concat([b, b]).astype(CategoricalDtype(list("cab"))),
            }
        )
        tm.assert_frame_equal(res, exp)

    def test_categorical_index_preserver(self):
        # [FUNCTION DOC]
        # Verifies concatenating frames indexed by a categorical column keeps the
        # categorical index when categories are the same. With a frame whose categories
        # differ (built under an expected Pandas4Warning about values not in the dtype),
        # concatenation casts both indexes to object.
        a = Series(np.arange(6, dtype="int64"))
        b = Series(list("aabbca"))

        df2 = DataFrame(
            {"A": a, "B": b.astype(CategoricalDtype(list("cab")))}
        ).set_index("B")
        result = pd.concat([df2, df2])
        expected = DataFrame(
            {
                "A": pd.concat([a, a]),
                "B": pd.concat([b, b]).astype(CategoricalDtype(list("cab"))),
            }
        ).set_index("B")
        tm.assert_frame_equal(result, expected)

        # wrong categories -> uses concat_compat, which casts to object
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            df3 = DataFrame(
                {"A": a, "B": Categorical(b, categories=list("abe"))}
            ).set_index("B")
        result = pd.concat([df2, df3])
        expected = pd.concat(
            [
                df2.set_axis(df2.index.astype(object), axis=0),
                df3.set_axis(df3.index.astype(object), axis=0),
            ]
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_categorical_tz(self):
        # [FUNCTION DOC]
        # Verifies concatenating a tz-aware datetime series with a categorical series
        # gives an object series of Timestamps and strings (GH 23816).
        # GH-23816
        a = Series(pd.date_range("2017-01-01", periods=2, tz="US/Pacific"))
        b = Series(["a", "b"], dtype="category")
        result = pd.concat([a, b], ignore_index=True)
        expected = Series(
            [
                pd.Timestamp("2017-01-01", tz="US/Pacific"),
                pd.Timestamp("2017-01-02", tz="US/Pacific"),
                "a",
                "b",
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_concat_categorical_datetime(self):
        # [FUNCTION DOC]
        # Verifies concatenating frames whose column is a categorical of single datetime
        # values yields a datetime column (not categorical) (GH 39443).
        # GH-39443
        df1 = DataFrame(
            {"x": Series(datetime(2021, 1, 1), index=[0], dtype="category")}
        )
        df2 = DataFrame(
            {"x": Series(datetime(2021, 1, 2), index=[1], dtype="category")}
        )

        result = pd.concat([df1, df2])
        expected = DataFrame(
            {"x": Series([datetime(2021, 1, 1), datetime(2021, 1, 2)])}
        )

        tm.assert_equal(result, expected)

    def test_concat_categorical_unchanged(self):
        # [FUNCTION DOC]
        # Verifies that side-by-side concatenation of a categorical frame column and a
        # numeric series with a misaligned index keeps the categorical dtype (with
        # missing value) and the float dtype for the other column (GH 12007).
        # GH-12007
        # test fix for when concat on categorical and float
        # coerces dtype categorical -> float
        df = DataFrame(Series(["a", "b", "c"], dtype="category", name="A"))
        ser = Series([0, 1, 2], index=[0, 1, 3], name="B")
        result = pd.concat([df, ser], axis=1)
        expected = DataFrame(
            {
                "A": Series(["a", "b", "c", np.nan], dtype="category"),
                "B": Series([0, 1, np.nan, 2], dtype="float"),
            }
        )
        tm.assert_equal(result, expected)

    def test_categorical_concat_gh7864(self):
        # [FUNCTION DOC]
        # Verifies that splitting a frame with a categorical column into two parts and
        # concatenating them preserves the categories index.
        # KNOWN ISSUE: the call that
        # sets new categories on the column returns a new object that is discarded, so it
        # has no effect on the frame and the stated ordering check is not exercised.
        # GH 7864
        # make sure ordering is preserved
        df = DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": list("abbaae")})
        df["grade"] = Categorical(df["raw_grade"])
        df["grade"].cat.set_categories(["e", "a", "b"])

        df1 = df[0:3]
        df2 = df[3:]

        tm.assert_index_equal(df["grade"].cat.categories, df1["grade"].cat.categories)
        tm.assert_index_equal(df["grade"].cat.categories, df2["grade"].cat.categories)

        dfx = pd.concat([df1, df2])
        tm.assert_index_equal(df["grade"].cat.categories, dfx["grade"].cat.categories)

    def test_categorical_index_upcast(self):
        # [FUNCTION DOC]
        # Verifies that concatenating frames or series whose categorical indexes have
        # non-identical categories upcasts the index to object (GH 17629).
        # GH 17629
        # test upcasting to object when concatenating on categorical indexes
        # with non-identical categories

        a = DataFrame({"foo": [1, 2]}, index=Categorical(["foo", "bar"]))
        b = DataFrame({"foo": [4, 3]}, index=Categorical(["baz", "bar"]))

        res = pd.concat([a, b])
        exp = DataFrame({"foo": [1, 2, 4, 3]}, index=["foo", "bar", "baz", "bar"])

        tm.assert_equal(res, exp)

        a = Series([1, 2], index=Categorical(["foo", "bar"]))
        b = Series([4, 3], index=Categorical(["baz", "bar"]))

        res = pd.concat([a, b])
        exp = Series([1, 2, 4, 3], index=["foo", "bar", "baz", "bar"])

        tm.assert_equal(res, exp)

    def test_categorical_missing_from_one_frame(self):
        # [FUNCTION DOC]
        # Verifies concatenating (with sort enabled) a frame lacking a categorical column
        # with one that has it keeps the column categorical with missing codes for the
        # first frame's rows (GH 25412).
        # GH 25412
        df1 = DataFrame({"f1": [1, 2, 3]})
        df2 = DataFrame({"f1": [2, 3, 1], "f2": Series([4, 4, 4]).astype("category")})
        result = pd.concat([df1, df2], sort=True)
        dtype = CategoricalDtype([4])
        expected = DataFrame(
            {
                "f1": [1, 2, 3, 2, 3, 1],
                "f2": Categorical.from_codes([-1, -1, -1, 0, 0, 0], dtype=dtype),
            },
            index=[0, 1, 2, 0, 1, 2],
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_categorical_same_categories_different_order(self):
        # [FUNCTION DOC]
        # Verifies concatenating frames indexed by unordered categorical indexes with the
        # same categories in a different order gives a categorical index with the first
        # index's category order.
        # https://github.com/pandas-dev/pandas/issues/24845

        c1 = pd.CategoricalIndex(["a", "a"], categories=["a", "b"], ordered=False)
        c2 = pd.CategoricalIndex(["b", "b"], categories=["b", "a"], ordered=False)
        c3 = pd.CategoricalIndex(
            ["a", "a", "b", "b"], categories=["a", "b"], ordered=False
        )

        df1 = DataFrame({"A": [1, 2]}, index=c1)
        df2 = DataFrame({"A": [3, 4]}, index=c2)

        result = pd.concat((df1, df2))
        expected = DataFrame({"A": [1, 2, 3, 4]}, index=c3)
        tm.assert_frame_equal(result, expected)
