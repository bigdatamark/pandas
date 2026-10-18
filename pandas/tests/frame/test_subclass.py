# [MODULE DOC]
# Tests that DataFrame and Series subclasses keep their class (and metadata) through
# many operations: slicing, indexing, alignment, stacking, pivot, melt, apply,
# reductions, and others, using the SubclassedDataFrame and SubclassedSeries helper
# classes from the testing module. Also defines extra subclasses (with metadata, and
# with no constructor override). Import-time side effect: a module-wide marker ignores
# deprecation warnings about passing a block manager.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)


class TestDataFrameSubclassing:
    # [CLASS DOC]
    # Groups subclassing tests for DataFrame using the testing-module subclass types.
    # Stateless.
    def test_no_warning_on_mgr(self):
        # [FUNCTION DOC]
        # Verifies (GH#57032) that isna on a subclassed frame and on one of its columns
        # raises no warning (the construct-from-manager path must not hand a manager to
        # the initializer).
        # GH#57032
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [1, 2, 3]}, index=["a", "b", "c"]
        )
        with tm.assert_produces_warning(None):
            # df.isna() goes through _constructor_from_mgr, which we want to
            #  *not* pass a Manager do __init__
            df.isna()
            df["X"].isna()

    def test_frame_subclassing_and_slicing(self):
        # [FUNCTION DOC]
        # Verifies (PR 9632) that a custom frame subclass with custom series class and
        # constructors returns its own classes: selecting a column gives the custom
        # series, row slicing gives the custom frame, selecting a top-level label of a
        # multi-index column gives a custom frame, and a label with empty second level
        # gives a custom series.
        # Subclass frame and ensure it returns the right class on slicing it
        # In reference to PR 9632

        class CustomSeries(Series):
            # [CLASS DOC]
            # Series subclass used by the slicing test; its constructor property returns
            # its own class and it has one extra method returning OK.
            @property
            def _constructor(self):
                # [FUNCTION DOC]
                # HELPER: property returning the subclass itself as the constructor.
                return CustomSeries

            def custom_series_function(self):
                # [FUNCTION DOC]
                # HELPER: method returning the string OK, used to show the custom series
                # class is preserved.
                return "OK"

        class CustomDataFrame(DataFrame):
            # [CLASS DOC]
            # Frame subclass used by the slicing test: sets its own class as constructor
            # and the custom series class as sliced-constructor, with an extra method
            # returning OK. The class docstring describes simulation results and
            # plotting helpers which the class does not actually contain.
            """
            Subclasses pandas DF, fills DF with simulation results, adds some
            custom plotting functions.
            """

            def __init__(self, *args, **kw) -> None:
                # [FUNCTION DOC]
                # HELPER: initializer that only forwards all arguments to the parent
                # initializer.
                super().__init__(*args, **kw)

            @property
            def _constructor(self):
                # [FUNCTION DOC]
                # HELPER: property returning the subclass itself as the constructor.
                return CustomDataFrame

            _constructor_sliced = CustomSeries

            def custom_frame_function(self):
                # [FUNCTION DOC]
                # HELPER: method returning the string OK, used to show the custom frame
                # class is preserved.
                return "OK"

        data = {"col1": range(10), "col2": range(10)}
        cdf = CustomDataFrame(data)

        # Did we get back our own DF class?
        assert isinstance(cdf, CustomDataFrame)

        # Do we get back our own Series class after selecting a column?
        cdf_series = cdf.col1
        assert isinstance(cdf_series, CustomSeries)
        assert cdf_series.custom_series_function() == "OK"

        # Do we get back our own DF class after slicing row-wise?
        cdf_rows = cdf[1:5]
        assert isinstance(cdf_rows, CustomDataFrame)
        assert cdf_rows.custom_frame_function() == "OK"

        # Make sure sliced part of multi-index frame is custom class
        mcol = MultiIndex.from_tuples([("A", "A"), ("A", "B")])
        cdf_multi = CustomDataFrame([[0, 1], [2, 3]], columns=mcol)
        assert isinstance(cdf_multi["A"], CustomDataFrame)

        mcol = MultiIndex.from_tuples([("A", ""), ("B", "")])
        cdf_multi2 = CustomDataFrame([[0, 1], [2, 3]], columns=mcol)
        assert isinstance(cdf_multi2["A"], CustomSeries)

    def test_dataframe_metadata(self, temp_file):
        # [FUNCTION DOC]
        # Verifies custom attributes in the metadata list survive column selection,
        # label and position row selection (gh-9776) and a pickle round trip (gh-10553),
        # and that the metadata list and attribute value are equal after unpickling.
        # Uses the temp_file fixture.
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [1, 2, 3]}, index=["a", "b", "c"]
        )
        df.testattr = "XXX"

        assert df.testattr == "XXX"
        assert df[["X"]].testattr == "XXX"
        assert df.loc[["a", "b"], :].testattr == "XXX"
        assert df.iloc[[0, 1], :].testattr == "XXX"

        # see gh-9776
        assert df.iloc[0:1, :].testattr == "XXX"

        # see gh-10553
        unpickled = tm.round_trip_pickle(df, temp_file)
        tm.assert_frame_equal(df, unpickled)
        assert df._metadata == unpickled._metadata
        assert df.testattr == unpickled.testattr

    def test_indexing_sliced(self):
        # [FUNCTION DOC]
        # Verifies (GH 11559) label and position selection of a single column or row
        # from a subclassed frame returns the subclassed series with the expected
        # values, index and name.
        # GH 11559
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [4, 5, 6], "Z": [7, 8, 9]}, index=["a", "b", "c"]
        )
        res = df.loc[:, "X"]
        exp = tm.SubclassedSeries([1, 2, 3], index=list("abc"), name="X")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.iloc[:, 1]
        exp = tm.SubclassedSeries([4, 5, 6], index=list("abc"), name="Y")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.loc[:, "Z"]
        exp = tm.SubclassedSeries([7, 8, 9], index=list("abc"), name="Z")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.loc["a", :]
        exp = tm.SubclassedSeries([1, 4, 7], index=list("XYZ"), name="a")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.iloc[1, :]
        exp = tm.SubclassedSeries([2, 5, 8], index=list("XYZ"), name="b")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.loc["c", :]
        exp = tm.SubclassedSeries([3, 6, 9], index=list("XYZ"), name="c")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

    def test_subclass_attr_err_propagation(self):
        # [FUNCTION DOC]
        # Verifies (GH 11808) an attribute error raised inside a property of a frame
        # subclass propagates with the missing attribute name, rather than being masked.
        # GH 11808
        class A(DataFrame):
            # [CLASS DOC]
            # Frame subclass with one property that reads a non-existing attribute;
            # used to check that attribute errors propagate with their own message.
            @property
            def nonexistence(self):
                # [FUNCTION DOC]
                # HELPER: property that accesses a non-existing attribute, to trigger an
                # attribute error.
                return self.i_dont_exist

        with pytest.raises(AttributeError, match=".*i_dont_exist.*"):
            A().nonexistence

    def test_subclass_align(self):
        # [FUNCTION DOC]
        # Verifies (GH 12983) aligning two subclassed frames on rows, and two subclassed
        # series, returns the subclass types with union index and missing values.
        # GH 12983
        df1 = tm.SubclassedDataFrame(
            {"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE")
        )
        df2 = tm.SubclassedDataFrame(
            {"c": [1, 2, 4], "d": [1, 2, 4]}, index=list("ABD")
        )

        res1, res2 = df1.align(df2, axis=0)
        exp1 = tm.SubclassedDataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        exp2 = tm.SubclassedDataFrame(
            {"c": [1, 2, np.nan, 4, np.nan], "d": [1, 2, np.nan, 4, np.nan]},
            index=list("ABCDE"),
        )
        assert isinstance(res1, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res1, exp1)
        assert isinstance(res2, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res2, exp2)

        res1, res2 = df1.a.align(df2.c)
        assert isinstance(res1, tm.SubclassedSeries)
        tm.assert_series_equal(res1, exp1.a)
        assert isinstance(res2, tm.SubclassedSeries)
        tm.assert_series_equal(res2, exp2.c)

    def test_subclass_align_combinations(self):
        # [FUNCTION DOC]
        # Verifies (GH 12983) aligning a subclassed frame with a subclassed series, in
        # either order, returns the matching subclass types with the union index.
        # KNOWN ISSUE: the comment saying the name is lost is truncated, and the
        # expected series actually keeps its name.
        # GH 12983
        df = tm.SubclassedDataFrame({"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE"))
        s = tm.SubclassedSeries([1, 2, 4], index=list("ABD"), name="x")

        # frame + series
        res1, res2 = df.align(s, axis=0)
        exp1 = tm.SubclassedDataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        # name is lost when
        exp2 = tm.SubclassedSeries(
            [1, 2, np.nan, 4, np.nan], index=list("ABCDE"), name="x"
        )

        assert isinstance(res1, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res1, exp1)
        assert isinstance(res2, tm.SubclassedSeries)
        tm.assert_series_equal(res2, exp2)

        # series + frame
        res1, res2 = s.align(df)
        assert isinstance(res1, tm.SubclassedSeries)
        tm.assert_series_equal(res1, exp2)
        assert isinstance(res2, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res2, exp1)

    def test_subclass_iterrows(self):
        # [FUNCTION DOC]
        # Verifies (GH 13977) iterating rows of a subclassed frame yields the subclassed
        # series equal to the row selection.
        # GH 13977
        df = tm.SubclassedDataFrame({"a": [1]})
        for i, row in df.iterrows():
            assert isinstance(row, tm.SubclassedSeries)
            tm.assert_series_equal(row, df.loc[i])

    def test_subclass_stack(self):
        # [FUNCTION DOC]
        # Verifies (GH 15564) stacking a subclassed single-level frame returns the
        # subclassed series with a two-level index.
        # GH 15564
        df = tm.SubclassedDataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            index=["a", "b", "c"],
            columns=["X", "Y", "Z"],
        )

        res = df.stack()
        exp = tm.SubclassedSeries(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], index=[list("aaabbbccc"), list("XYZXYZXYZ")]
        )

        tm.assert_series_equal(res, exp)

    def test_subclass_stack_multi(self):
        # [FUNCTION DOC]
        # Verifies (GH 15564) stacking a subclassed frame with multi-index rows and
        # columns: the default level and the named level yyy give a subclassed frame
        # with the last column level moved into the index, and the level www gives the
        # other arrangement.
        # GH 15564
        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        exp = tm.SubclassedDataFrame(
            [
                [10, 12],
                [11, 13],
                [20, 22],
                [21, 23],
                [30, 32],
                [31, 33],
                [40, 42],
                [41, 43],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("yzyzyzyz"))),
                names=["aaa", "ccc", "yyy"],
            ),
            columns=Index(["W", "X"], name="www"),
        )

        res = df.stack()
        tm.assert_frame_equal(res, exp)

        res = df.stack("yyy")
        tm.assert_frame_equal(res, exp)

        exp = tm.SubclassedDataFrame(
            [
                [10, 11],
                [12, 13],
                [20, 21],
                [22, 23],
                [30, 31],
                [32, 33],
                [40, 41],
                [42, 43],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("WXWXWXWX"))),
                names=["aaa", "ccc", "www"],
            ),
            columns=Index(["y", "z"], name="yyy"),
        )

        res = df.stack("www")
        tm.assert_frame_equal(res, exp)

    def test_subclass_stack_multi_mixed(self):
        # [FUNCTION DOC]
        # Verifies (GH 15564) the same stacking cases with mixed integer and float
        # columns, keeping per-column dtypes.
        # GH 15564
        df = tm.SubclassedDataFrame(
            [
                [10, 11, 12.0, 13.0],
                [20, 21, 22.0, 23.0],
                [30, 31, 32.0, 33.0],
                [40, 41, 42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        exp = tm.SubclassedDataFrame(
            [
                [10, 12.0],
                [11, 13.0],
                [20, 22.0],
                [21, 23.0],
                [30, 32.0],
                [31, 33.0],
                [40, 42.0],
                [41, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("yzyzyzyz"))),
                names=["aaa", "ccc", "yyy"],
            ),
            columns=Index(["W", "X"], name="www"),
        )

        res = df.stack()
        tm.assert_frame_equal(res, exp)

        res = df.stack("yyy")
        tm.assert_frame_equal(res, exp)

        exp = tm.SubclassedDataFrame(
            [
                [10.0, 11.0],
                [12.0, 13.0],
                [20.0, 21.0],
                [22.0, 23.0],
                [30.0, 31.0],
                [32.0, 33.0],
                [40.0, 41.0],
                [42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("WXWXWXWX"))),
                names=["aaa", "ccc", "www"],
            ),
            columns=Index(["y", "z"], name="yyy"),
        )

        res = df.stack("www")
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self):
        # [FUNCTION DOC]
        # Verifies (GH 15564) unstacking a subclassed single-level frame returns the
        # subclassed series.
        # GH 15564
        df = tm.SubclassedDataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            index=["a", "b", "c"],
            columns=["X", "Y", "Z"],
        )

        res = df.unstack()
        exp = tm.SubclassedSeries(
            [1, 4, 7, 2, 5, 8, 3, 6, 9], index=[list("XXXYYYZZZ"), list("abcabcabc")]
        )

        tm.assert_series_equal(res, exp)

    def test_subclass_unstack_multi(self):
        # [FUNCTION DOC]
        # Verifies (GH 15564) unstacking multi-index frames: default level and named
        # levels ccc and aaa give subclassed frames with the unstacked level moved to
        # the columns.
        # GH 15564
        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        exp = tm.SubclassedDataFrame(
            [[10, 20, 11, 21, 12, 22, 13, 23], [30, 40, 31, 41, 32, 42, 33, 43]],
            index=Index(["A", "B"], name="aaa"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("cdcdcdcd"))),
                names=["www", "yyy", "ccc"],
            ),
        )

        res = df.unstack()
        tm.assert_frame_equal(res, exp)

        res = df.unstack("ccc")
        tm.assert_frame_equal(res, exp)

        exp = tm.SubclassedDataFrame(
            [[10, 30, 11, 31, 12, 32, 13, 33], [20, 40, 21, 41, 22, 42, 23, 43]],
            index=Index(["c", "d"], name="ccc"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("ABABABAB"))),
                names=["www", "yyy", "aaa"],
            ),
        )

        res = df.unstack("aaa")
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack_multi_mixed(self):
        # [FUNCTION DOC]
        # Verifies (GH 15564) the same unstack cases with mixed integer and float
        # columns.
        # GH 15564
        df = tm.SubclassedDataFrame(
            [
                [10, 11, 12.0, 13.0],
                [20, 21, 22.0, 23.0],
                [30, 31, 32.0, 33.0],
                [40, 41, 42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        exp = tm.SubclassedDataFrame(
            [
                [10, 20, 11, 21, 12.0, 22.0, 13.0, 23.0],
                [30, 40, 31, 41, 32.0, 42.0, 33.0, 43.0],
            ],
            index=Index(["A", "B"], name="aaa"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("cdcdcdcd"))),
                names=["www", "yyy", "ccc"],
            ),
        )

        res = df.unstack()
        tm.assert_frame_equal(res, exp)

        res = df.unstack("ccc")
        tm.assert_frame_equal(res, exp)

        exp = tm.SubclassedDataFrame(
            [
                [10, 30, 11, 31, 12.0, 32.0, 13.0, 33.0],
                [20, 40, 21, 41, 22.0, 42.0, 23.0, 43.0],
            ],
            index=Index(["c", "d"], name="ccc"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("ABABABAB"))),
                names=["www", "yyy", "aaa"],
            ),
        )

        res = df.unstack("aaa")
        tm.assert_frame_equal(res, exp)

    def test_subclass_pivot(self):
        # [FUNCTION DOC]
        # Verifies (GH 15564) pivot on a subclassed frame returns a subclassed frame
        # with named index and columns.
        # GH 15564
        df = tm.SubclassedDataFrame(
            {
                "index": ["A", "B", "C", "C", "B", "A"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            }
        )

        pivoted = df.pivot(index="index", columns="columns", values="values")

        expected = tm.SubclassedDataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        expected.index.name, expected.columns.name = "index", "columns"

        tm.assert_frame_equal(pivoted, expected)

    def test_subclassed_melt(self):
        # [FUNCTION DOC]
        # Verifies (GH 15564) melting a subclassed frame with identifier columns returns
        # a subclassed frame with variable and value columns.
        # GH 15564
        cheese = tm.SubclassedDataFrame(
            {
                "first": ["John", "Mary"],
                "last": ["Doe", "Bo"],
                "height": [5.5, 6.0],
                "weight": [130, 150],
            }
        )

        melted = pd.melt(cheese, id_vars=["first", "last"])

        expected = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 5.5],
                ["Mary", "Bo", "height", 6.0],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        tm.assert_frame_equal(melted, expected)

    def test_subclassed_wide_to_long(self):
        # [FUNCTION DOC]
        # Verifies (GH 9762) wide-to-long conversion of a subclassed frame returns a
        # subclassed frame with a two-level index (id and year) and the expected values.
        # GH 9762

        x = np.random.default_rng(2).standard_normal(3)
        df = tm.SubclassedDataFrame(
            {
                "A1970": {0: "a", 1: "b", 2: "c"},
                "A1980": {0: "d", 1: "e", 2: "f"},
                "B1970": {0: 2.5, 1: 1.2, 2: 0.7},
                "B1980": {0: 3.2, 1: 1.3, 2: 0.1},
                "X": dict(zip(range(3), x)),
            }
        )

        df["id"] = df.index
        exp_data = {
            "X": x.tolist() + x.tolist(),
            "A": ["a", "b", "c", "d", "e", "f"],
            "B": [2.5, 1.2, 0.7, 3.2, 1.3, 0.1],
            "year": [1970, 1970, 1970, 1980, 1980, 1980],
            "id": [0, 1, 2, 0, 1, 2],
        }
        expected = tm.SubclassedDataFrame(exp_data)
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]
        long_frame = pd.wide_to_long(df, ["A", "B"], i="id", j="year")

        tm.assert_frame_equal(long_frame, expected)

    def test_subclassed_apply(self):
        # [FUNCTION DOC]
        # Verifies (GH 19822) apply on a subclassed frame passes subclassed series rows,
        # and results: a row-modifying function (axis one) gives a subclassed frame with
        # heights increased by half; returning subclassed series or an expanded list
        # gives a subclassed frame; returning plain lists without expansion gives a
        # plain Series result that is not a subclassed frame.
        # GH 19822

        def check_row_subclass(row):
            # [FUNCTION DOC]
            # HELPER: asserts that the given row is a subclassed series.
            assert isinstance(row, tm.SubclassedSeries)

        def stretch(row):
            # [FUNCTION DOC]
            # HELPER: for a row whose variable is height, increases its value by one
            # half; returns the row.
            if row["variable"] == "height":
                row["value"] += 0.5
            return row

        df = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 5.5],
                ["Mary", "Bo", "height", 6.0],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        df.apply(lambda x: check_row_subclass(x))
        df.apply(lambda x: check_row_subclass(x), axis=1)

        expected = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 6.0],
                ["Mary", "Bo", "height", 6.5],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        result = df.apply(lambda x: stretch(x), axis=1)
        assert isinstance(result, tm.SubclassedDataFrame)
        tm.assert_frame_equal(result, expected)

        expected = tm.SubclassedDataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

        result = df.apply(lambda x: tm.SubclassedSeries([1, 2, 3]), axis=1)
        assert isinstance(result, tm.SubclassedDataFrame)
        tm.assert_frame_equal(result, expected)

        result = df.apply(lambda x: [1, 2, 3], axis=1, result_type="expand")
        assert isinstance(result, tm.SubclassedDataFrame)
        tm.assert_frame_equal(result, expected)

        expected = tm.SubclassedSeries([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

        result = df.apply(lambda x: [1, 2, 3], axis=1)
        assert not isinstance(result, tm.SubclassedDataFrame)
        tm.assert_series_equal(result, expected)

    def test_subclassed_reductions(self, all_reductions):
        # [FUNCTION DOC]
        # Verifies (GH 25596) every reduction in the all_reductions fixture returns the
        # subclassed series on a subclassed frame.
        # GH 25596

        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = getattr(df, all_reductions)()
        assert isinstance(result, tm.SubclassedSeries)

    def test_subclassed_count(self):
        # [FUNCTION DOC]
        # Verifies the count method returns a subclassed series for mixed data, numeric
        # data, multi-index frames and an empty frame.
        df = tm.SubclassedDataFrame(
            {
                "Person": ["John", "Myla", "Lewis", "John", "Myla"],
                "Age": [24.0, np.nan, 21.0, 33, 26],
                "Single": [False, True, True, True, False],
            }
        )
        result = df.count()
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame({"A": [1, 0, 3], "B": [0, 5, 6], "C": [7, 8, 0]})
        result = df.count()
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )
        result = df.count()
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame()
        result = df.count()
        assert isinstance(result, tm.SubclassedSeries)

    def test_isin(self):
        # [FUNCTION DOC]
        # Verifies membership testing on a subclassed frame returns a subclassed frame.
        df = tm.SubclassedDataFrame(
            {"num_legs": [2, 4], "num_wings": [2, 0]}, index=["falcon", "dog"]
        )
        result = df.isin([0, 2])
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_duplicated(self):
        # [FUNCTION DOC]
        # Verifies duplicated returns a subclassed series for a non-empty and an empty
        # subclassed frame.
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.duplicated()
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame()
        result = df.duplicated()
        assert isinstance(result, tm.SubclassedSeries)

    @pytest.mark.parametrize("idx_method", ["idxmax", "idxmin"])
    def test_idx(self, idx_method):
        # [FUNCTION DOC]
        # Verifies idxmax and idxmin return a subclassed series. Parametrized over the
        # method name.
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = getattr(df, idx_method)()
        assert isinstance(result, tm.SubclassedSeries)

    def test_dot(self):
        # [FUNCTION DOC]
        # Verifies the dot product of a subclassed frame with a subclassed series gives
        # a subclassed series, and with a subclassed frame gives a subclassed frame.
        df = tm.SubclassedDataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        s = tm.SubclassedSeries([1, 1, 2, 1])
        result = df.dot(s)
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        s = tm.SubclassedDataFrame([1, 1, 2, 1])
        result = df.dot(s)
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_memory_usage(self):
        # [FUNCTION DOC]
        # Verifies memory usage (with and without index) returns a subclassed series.
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.memory_usage()
        assert isinstance(result, tm.SubclassedSeries)

        result = df.memory_usage(index=False)
        assert isinstance(result, tm.SubclassedSeries)

    def test_corrwith(self):
        # [FUNCTION DOC]
        # Verifies correlation with another subclassed frame (row-wise, dropping
        # unmatched, Kendall method) returns a subclassed series. Skipped without scipy.
        pytest.importorskip("scipy")
        index = ["a", "b", "c", "d", "e"]
        columns = ["one", "two", "three", "four"]
        df1 = tm.SubclassedDataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            index=index,
            columns=columns,
        )
        df2 = tm.SubclassedDataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=index[:4],
            columns=columns,
        )
        correls = df1.corrwith(df2, axis=1, drop=True, method="kendall")

        assert isinstance(correls, (tm.SubclassedSeries))

    def test_asof(self):
        # [FUNCTION DOC]
        # Verifies as-of lookup on a subclassed frame returns a subclassed frame for a
        # list of labels and a subclassed series for scalar labels (including a label
        # before the first row).
        N = 3
        rng = pd.date_range("1/1/1990", periods=N, freq="53s")
        df = tm.SubclassedDataFrame(
            {
                "A": [np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan],
                "C": [np.nan, np.nan, np.nan],
            },
            index=rng,
        )

        result = df.asof(rng[-2:])
        assert isinstance(result, tm.SubclassedDataFrame)

        result = df.asof(rng[-2])
        assert isinstance(result, tm.SubclassedSeries)

        result = df.asof("1989-12-31")
        assert isinstance(result, tm.SubclassedSeries)

    def test_idxmin_preserves_subclass(self):
        # [FUNCTION DOC]
        # Verifies (GH 28330) idxmin returns a subclassed series.
        # GH 28330

        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.idxmin()
        assert isinstance(result, tm.SubclassedSeries)

    def test_idxmax_preserves_subclass(self):
        # [FUNCTION DOC]
        # Verifies (GH 28330) idxmax returns a subclassed series.
        # GH 28330

        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.idxmax()
        assert isinstance(result, tm.SubclassedSeries)

    def test_convert_dtypes_preserves_subclass(self):
        # [FUNCTION DOC]
        # Verifies (GH 43668) convert_dtypes returns a subclassed frame.
        # GH 43668
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.convert_dtypes()
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_convert_dtypes_preserves_subclass_with_constructor(self):
        # [FUNCTION DOC]
        # Verifies convert_dtypes preserves a locally defined subclass that only defines
        # the constructor property.
        class SubclassedDataFrame(DataFrame):
            # [CLASS DOC]
            # Local frame subclass defining only the constructor property, used by the
            # preceding test.
            @property
            def _constructor(self):
                # [FUNCTION DOC]
                # HELPER: property returning the local subclass as the constructor.
                return SubclassedDataFrame

        df = SubclassedDataFrame({"a": [1, 2, 3]})
        result = df.convert_dtypes()
        assert isinstance(result, SubclassedDataFrame)

    def test_astype_preserves_subclass(self):
        # [FUNCTION DOC]
        # Verifies (GH#40810) astype with a per-column dtype mapping returns a
        # subclassed frame.
        # GH#40810
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

        result = df.astype({"A": np.int64, "B": np.int32, "C": np.float64})
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_equals_subclass(self):
        # [FUNCTION DOC]
        # Verifies (PR 34402) equality comparison works between a plain frame and a
        # subclassed frame in both directions.
        # https://github.com/pandas-dev/pandas/pull/34402
        # allow subclass in both directions
        df1 = DataFrame({"a": [1, 2, 3]})
        df2 = tm.SubclassedDataFrame({"a": [1, 2, 3]})
        assert df1.equals(df2)
        assert df2.equals(df1)


class MySubclassWithMetadata(DataFrame):
    # [CLASS DOC]
    # Frame subclass with a metadata attribute (my_metadata) used to check
    # constructor-with-metadata behavior. Inherits from the frame; state: the attribute,
    # set at initialization to the keyword value, or copied from the first argument if
    # that is an instance of the same class. The constructor property returns this
    # class.
    _metadata = ["my_metadata"]

    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Initializer: forwards arguments to the parent initializer, then sets the
        # metadata attribute from the keyword (default none) or from the first argument
        # when it is another instance.
        # KNOWN ISSUE: the metadata keyword is removed only after the parent initializer
        # has already received it, so passing it would reach the parent initializer; the
        # removal has no effect on that call.
        super().__init__(*args, **kwargs)

        my_metadata = kwargs.pop("my_metadata", None)
        if args and isinstance(args[0], MySubclassWithMetadata):
            my_metadata = args[0].my_metadata  # type: ignore[has-type]
        self.my_metadata = my_metadata

    @property
    def _constructor(self):
        # [FUNCTION DOC]
        # HELPER: property returning this class so operations keep the subclass.
        return MySubclassWithMetadata


def test_constructor_with_metadata():
    # [FUNCTION DOC]
    # Verifies (PR 54922, issue 55120) selecting a column subset of a metadata subclass
    # frame returns the same subclass.
    # https://github.com/pandas-dev/pandas/pull/54922
    # https://github.com/pandas-dev/pandas/issues/55120
    df = MySubclassWithMetadata(
        np.random.default_rng(2).random((5, 3)), columns=["A", "B", "C"]
    )
    subset = df[["A", "B"]]
    assert isinstance(subset, MySubclassWithMetadata)


def test_constructor_with_metadata_from_records():
    # [FUNCTION DOC]
    # Verifies (GH#57008) building the metadata subclass from records yields the
    # subclass with metadata none.
    # GH#57008
    df = MySubclassWithMetadata.from_records([{"a": 1, "b": 2}])
    assert df.my_metadata is None
    assert type(df) is MySubclassWithMetadata


class SimpleDataFrameSubClass(DataFrame):
    # [CLASS DOC]
    # Frame subclass that does not define a constructor property; used to check default
    # behavior.
    """A subclass of DataFrame that does not define a constructor."""


class SimpleSeriesSubClass(Series):
    # [CLASS DOC]
    # Series subclass that does not define a constructor property; used to check default
    # behavior.
    """A subclass of Series that does not define a constructor."""


class TestSubclassWithoutConstructor:
    # [CLASS DOC]
    # Groups tests of subclasses that do not override the constructor property.
    # Stateless.
    def test_copy_df(self):
        # [FUNCTION DOC]
        # Verifies copying a constructor-less frame subclass returns a plain frame equal
        # to the original.
        expected = DataFrame({"a": [1, 2, 3]})
        result = SimpleDataFrameSubClass(expected).copy()

        assert (
            type(result) is DataFrame
        )  # assert_frame_equal only checks isinstance(lhs, type(rhs))
        tm.assert_frame_equal(result, expected)

    def test_copy_series(self):
        # [FUNCTION DOC]
        # Verifies copying a constructor-less series subclass returns a series equal to
        # the original.
        expected = Series([1, 2, 3])
        result = SimpleSeriesSubClass(expected).copy()

        tm.assert_series_equal(result, expected)

    def test_series_to_frame(self):
        # [FUNCTION DOC]
        # Verifies converting a constructor-less series subclass to a frame gives a
        # plain frame.
        orig = Series([1, 2, 3])
        expected = orig.to_frame()
        result = SimpleSeriesSubClass(orig).to_frame()

        assert (
            type(result) is DataFrame
        )  # assert_frame_equal only checks isinstance(lhs, type(rhs))
        tm.assert_frame_equal(result, expected)

    def test_groupby(self):
        # [FUNCTION DOC]
        # Verifies iterating the groups of a constructor-less frame subclass yields
        # plain frames.
        df = SimpleDataFrameSubClass(DataFrame({"a": [1, 2, 3]}))

        for _, v in df.groupby("a"):
            assert type(v) is DataFrame
