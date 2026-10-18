# [MODULE DOC]
# Tests for the cross-tabulation function (frequency tables, or aggregated tables, of factors).
# Covers single and multiple row/column factors, array and list inputs, alignment, margins and margin names, dropping of nulls, normalization options, tuple and duplicate names, categorical and nullable inputs, and argument validation errors.
# Provides one shared frame fixture; the tests live in one class plus one module-level parametrized test. No import-time side effects.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    crosstab,
)
import pandas._testing as tm


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a frame with string columns A, B, C and random normal columns D, E, F (11 rows each from seeded generators), concatenated with itself with a fresh index to give 22 rows.
    df = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    return pd.concat([df, df], ignore_index=True)


class TestCrosstab:
    # [CLASS DOC]
    # Tests of the cross-tabulation function, often compared against equivalent group-by, unstack or pivot-table computations.
    def test_crosstab_single(self, df):
        # [FUNCTION DOC]
        # Cross tabulating two columns equals counting group sizes and unstacking, with missing combinations filled with zero as 64-bit integers.
        result = crosstab(df["A"], df["C"])
        expected = df.groupby(["A", "C"]).size().unstack()
        tm.assert_frame_equal(result, expected.fillna(0).astype(np.int64))

    def test_crosstab_multiple(self, df):
        # [FUNCTION DOC]
        # Multiple column factors (list of two series) and multiple row factors give the same counts as a grouped size unstacked on the corresponding levels, zero filled.
        result = crosstab(df["A"], [df["B"], df["C"]])
        expected = df.groupby(["A", "B", "C"]).size()
        expected = expected.unstack("B").unstack("C").fillna(0).astype(np.int64)
        tm.assert_frame_equal(result, expected)

        result = crosstab([df["B"], df["C"]], df["A"])
        expected = df.groupby(["B", "C", "A"]).size()
        expected = expected.unstack("A").fillna(0).astype(np.int64)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("box", [np.array, list, tuple])
    def test_crosstab_ndarray(self, box):
        # GH 44076
        # [FUNCTION DOC]
        # GH 44076. Arrays, lists and tuples as inputs, with explicit row and column names, give the same result as using equivalent frame columns; with no names the axes are named row_0 and col_0.
        a = box(np.random.default_rng(2).integers(0, 5, size=100))
        b = box(np.random.default_rng(2).integers(0, 3, size=100))
        c = box(np.random.default_rng(2).integers(0, 10, size=100))

        df = DataFrame({"a": a, "b": b, "c": c})

        result = crosstab(a, [b, c], rownames=["a"], colnames=("b", "c"))
        expected = crosstab(df["a"], [df["b"], df["c"]])
        tm.assert_frame_equal(result, expected)

        result = crosstab([b, c], a, colnames=["a"], rownames=("b", "c"))
        expected = crosstab([df["b"], df["c"]], df["a"])
        tm.assert_frame_equal(result, expected)

        # assign arbitrary names
        result = crosstab(a, c)
        expected = crosstab(df["a"], df["c"])
        expected.index.names = ["row_0"]
        expected.columns.names = ["col_0"]
        tm.assert_frame_equal(result, expected)

    def test_crosstab_non_aligned(self):
        # GH 17005
        # [FUNCTION DOC]
        # GH 17005. Series with differing indexes are intersected on index labels, and a plain array is matched by position to the first series; both give the same small count table.
        a = Series([0, 1, 1], index=["a", "b", "c"])
        b = Series([3, 4, 3, 4, 3], index=["a", "b", "c", "d", "f"])
        c = np.array([3, 4, 3], dtype=np.int64)

        expected = DataFrame(
            [[1, 0], [1, 1]],
            index=Index([0, 1], name="row_0"),
            columns=Index([3, 4], name="col_0"),
        )

        result = crosstab(a, b)
        tm.assert_frame_equal(result, expected)

        result = crosstab(a, c)
        tm.assert_frame_equal(result, expected)

    def test_crosstab_margins(self):
        # [FUNCTION DOC]
        # With margins enabled the result gains an All row and All column holding totals; checks the index and column names and that the margin column and margin row equal the expected group size counts plus the total.
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)

        df = DataFrame({"a": a, "b": b, "c": c})

        result = crosstab(a, [b, c], rownames=["a"], colnames=("b", "c"), margins=True)

        assert result.index.names == ("a",)
        assert result.columns.names == ["b", "c"]

        all_cols = result["All", ""]
        exp_cols = df.groupby(["a"]).size().astype("i8")
        # to keep index.name
        exp_margin = Series([len(df)], index=Index(["All"], name="a"))
        exp_cols = pd.concat([exp_cols, exp_margin])
        exp_cols.name = ("All", "")

        tm.assert_series_equal(all_cols, exp_cols)

        all_rows = result.loc["All"]
        exp_rows = df.groupby(["b", "c"]).size().astype("i8")
        exp_rows = pd.concat([exp_rows, Series([len(df)], index=[("All", "")])])
        exp_rows.name = "All"

        exp_rows = exp_rows.reindex(all_rows.index)
        exp_rows = exp_rows.fillna(0).astype(np.int64)
        tm.assert_series_equal(all_rows, exp_rows)

    def test_crosstab_margins_set_margin_name(self):
        # GH 15972
        # [FUNCTION DOC]
        # GH 15972. Same as the margins test but with a custom margins_name; additionally, a margins name that is an integer, None or a list raises a value error saying it must be a string.
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)

        df = DataFrame({"a": a, "b": b, "c": c})

        result = crosstab(
            a,
            [b, c],
            rownames=["a"],
            colnames=("b", "c"),
            margins=True,
            margins_name="TOTAL",
        )

        assert result.index.names == ("a",)
        assert result.columns.names == ["b", "c"]

        all_cols = result["TOTAL", ""]
        exp_cols = df.groupby(["a"]).size().astype("i8")
        # to keep index.name
        exp_margin = Series([len(df)], index=Index(["TOTAL"], name="a"))
        exp_cols = pd.concat([exp_cols, exp_margin])
        exp_cols.name = ("TOTAL", "")

        tm.assert_series_equal(all_cols, exp_cols)

        all_rows = result.loc["TOTAL"]
        exp_rows = df.groupby(["b", "c"]).size().astype("i8")
        exp_rows = pd.concat([exp_rows, Series([len(df)], index=[("TOTAL", "")])])
        exp_rows.name = "TOTAL"

        exp_rows = exp_rows.reindex(all_rows.index)
        exp_rows = exp_rows.fillna(0).astype(np.int64)
        tm.assert_series_equal(all_rows, exp_rows)

        msg = "margins_name argument must be a string"
        for margins_name in [666, None, ["a", "b"]]:
            with pytest.raises(ValueError, match=msg):
                crosstab(
                    a,
                    [b, c],
                    rownames=["a"],
                    colnames=("b", "c"),
                    margins=True,
                    margins_name=margins_name,
                )

    def test_crosstab_pass_values(self):
        # [FUNCTION DOC]
        # Passing values with a sum aggregation, named rows and columns, equals a pivot table of the same values, index and columns.
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)
        values = np.random.default_rng(2).standard_normal(100)

        table = crosstab(
            [a, b], c, values, aggfunc="sum", rownames=["foo", "bar"], colnames=["baz"]
        )

        df = DataFrame({"foo": a, "bar": b, "baz": c, "values": values})

        expected = df.pivot_table(
            "values", index=["foo", "bar"], columns="baz", aggfunc="sum"
        )
        tm.assert_frame_equal(table, expected)

    def test_crosstab_dropna(self):
        # GH 3820
        # [FUNCTION DOC]
        # GH 3820. With dropna disabled, the column multi-index contains all four observed combinations of the two column factors.
        a = np.array(["foo", "foo", "foo", "bar", "bar", "foo", "foo"], dtype=object)
        b = np.array(["one", "one", "two", "one", "two", "two", "two"], dtype=object)
        c = np.array(
            ["dull", "dull", "dull", "dull", "dull", "shiny", "shiny"], dtype=object
        )
        res = crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"], dropna=False)
        m = MultiIndex.from_tuples(
            [("one", "dull"), ("one", "shiny"), ("two", "dull"), ("two", "shiny")],
            names=["b", "c"],
        )
        tm.assert_index_equal(res.columns, m)

    def test_crosstab_no_overlap(self):
        # GS 10291

        # [FUNCTION DOC]
        # GS 10291. Two series with no shared index labels give an empty frame with empty integer indexes named row_0 and col_0.
        s1 = Series([1, 2, 3], index=[1, 2, 3])
        s2 = Series([4, 5, 6], index=[4, 5, 6])

        actual = crosstab(s1, s2)
        expected = DataFrame(
            index=Index([], dtype="int64", name="row_0"),
            columns=Index([], dtype="int64", name="col_0"),
        )

        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna(self):
        # GH 12577
        # pivot_table counts null into margin ('All')
        # when margins=true and dropna=true

        # [FUNCTION DOC]
        # GH 12577. With margins and dropna enabled a null in the row factor is excluded from counts and totals.
        df = DataFrame({"a": [1, 2, 2, 2, 2, np.nan], "b": [3, 3, 4, 4, 4, 4]})
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        expected = DataFrame([[1, 0, 1], [1, 3, 4], [2, 3, 5]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3, 4, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna2(self):
        # [FUNCTION DOC]
        # Nulls in both factors at different positions are dropped; each row with a null in either factor is excluded.
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, 2, np.nan], "b": [3, np.nan, 4, 4, 4, 4]}
        )
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        expected = DataFrame([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3.0, 4.0, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna3(self):
        # [FUNCTION DOC]
        # Null row factors with non-null column factors are excluded from the counts and totals.
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, np.nan, 2], "b": [3, 3, 4, 4, 4, 4]}
        )
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        expected = DataFrame([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3, 4, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna4(self):
        # GH 12642
        # _add_margins raises KeyError: Level None not found
        # when margins=True and dropna=False
        # GH: 10772: Keep np.nan in result with dropna=False
        # [FUNCTION DOC]
        # GH 12642 and GH 10772. With margins and dropna disabled the null row label is kept as its own row and counted in totals (previously raised a key error).
        df = DataFrame({"a": [1, 2, 2, 2, 2, np.nan], "b": [3, 3, 4, 4, 4, 4]})
        actual = crosstab(df.a, df.b, margins=True, dropna=False)
        expected = DataFrame([[1, 0, 1], [1, 3, 4], [0, 1, 1], [2, 4, 6]])
        expected.index = Index([1.0, 2.0, np.nan, "All"], name="a")
        expected.columns = Index([3, 4, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna5(self):
        # GH: 10772: Keep np.nan in result with dropna=False
        # [FUNCTION DOC]
        # GH 10772. Both factors contain nulls with dropna disabled: null row and null column are kept; dtype is not checked.
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, 2, np.nan], "b": [3, np.nan, 4, 4, 4, 4]}
        )
        actual = crosstab(df.a, df.b, margins=True, dropna=False)
        expected = DataFrame(
            [[1, 0, 0, 1.0], [0, 1, 0, 1.0], [0, 3, 1, 4.0], [1, 4, 1, 6.0]]
        )
        expected.index = Index([1.0, 2.0, np.nan, "All"], name="a")
        expected.columns = Index([3.0, 4.0, np.nan, "All"], name="b")
        tm.assert_frame_equal(actual, expected, check_dtype=False)

    def test_margin_dropna6(self):
        # GH: 10772: Keep np.nan in result with dropna=False
        # [FUNCTION DOC]
        # GH 10772. Object arrays with a null in the middle factor, margins on and dropna off: the null is kept as a column (or row) label in a multi-index with the All margin; the final case with dropna on drops rows containing the null.
        a = np.array(["foo", "foo", "foo", "bar", "bar", "foo", "foo"], dtype=object)
        b = np.array(["one", "one", "two", "one", "two", np.nan, "two"], dtype=object)
        c = np.array(
            ["dull", "dull", "dull", "dull", "dull", "shiny", "shiny"], dtype=object
        )

        actual = crosstab(
            a, [b, c], rownames=["a"], colnames=["b", "c"], margins=True, dropna=False
        )
        m = MultiIndex.from_arrays(
            [
                ["one", "one", "two", "two", np.nan, np.nan, "All"],
                ["dull", "shiny", "dull", "shiny", "dull", "shiny", ""],
            ],
            names=["b", "c"],
        )
        expected = DataFrame(
            [[1, 0, 1, 0, 0, 0, 2], [2, 0, 1, 1, 0, 1, 5], [3, 0, 2, 1, 0, 1, 7]],
            columns=m,
        )
        expected.index = Index(["bar", "foo", "All"], name="a")
        tm.assert_frame_equal(actual, expected)

        actual = crosstab(
            [a, b], c, rownames=["a", "b"], colnames=["c"], margins=True, dropna=False
        )
        m = MultiIndex.from_arrays(
            [
                ["bar", "bar", "bar", "foo", "foo", "foo", "All"],
                ["one", "two", np.nan, "one", "two", np.nan, ""],
            ],
            names=["a", "b"],
        )
        expected = DataFrame(
            [
                [1, 0, 1.0],
                [1, 0, 1.0],
                [0, 0, np.nan],
                [2, 0, 2.0],
                [1, 1, 2.0],
                [0, 1, 1.0],
                [5, 2, 7.0],
            ],
            index=m,
        )
        expected.columns = Index(["dull", "shiny", "All"], name="c")
        tm.assert_frame_equal(actual, expected)

        actual = crosstab(
            [a, b], c, rownames=["a", "b"], colnames=["c"], margins=True, dropna=True
        )
        m = MultiIndex.from_arrays(
            [["bar", "bar", "foo", "foo", "All"], ["one", "two", "one", "two", ""]],
            names=["a", "b"],
        )
        expected = DataFrame(
            [[1, 0, 1], [1, 0, 1], [2, 0, 2], [1, 1, 2], [5, 1, 6]], index=m
        )
        expected.columns = Index(["dull", "shiny", "All"], name="c")
        tm.assert_frame_equal(actual, expected)

    def test_crosstab_normalize(self):
        # Issue 12578
        # [FUNCTION DOC]
        # Issue 12578. Normalization by all (true or the string), index, columns and the integer aliases zero and one give the expected proportion tables; with margins on, the margin row or column holds the normalized totals.
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        rindex = Index([1, 2], name="a")
        cindex = Index([3, 4], name="b")
        full_normal = DataFrame([[0.2, 0], [0.2, 0.6]], index=rindex, columns=cindex)
        row_normal = DataFrame([[1.0, 0], [0.25, 0.75]], index=rindex, columns=cindex)
        col_normal = DataFrame([[0.5, 0], [0.5, 1.0]], index=rindex, columns=cindex)

        # Check all normalize args
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="all"), full_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize=True), full_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="index"), row_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="columns"), col_normal)
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=1),
            crosstab(df.a, df.b, normalize="columns"),
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=0), crosstab(df.a, df.b, normalize="index")
        )

        row_normal_margins = DataFrame(
            [[1.0, 0], [0.25, 0.75], [0.4, 0.6]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4], name="b", dtype="object"),
        )
        col_normal_margins = DataFrame(
            [[0.5, 0, 0.2], [0.5, 1.0, 0.8]],
            index=Index([1, 2], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )

        all_normal_margins = DataFrame(
            [[0.2, 0, 0.2], [0.2, 0.6, 0.8], [0.4, 0.6, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize="index", margins=True), row_normal_margins
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize="columns", margins=True), col_normal_margins
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=True, margins=True), all_normal_margins
        )

    def test_crosstab_normalize_arrays(self):
        # GH#12578
        # [FUNCTION DOC]
        # GH#12578. Array inputs work without error; normalizing a count or sum aggregation of values over everything with margins gives the expected proportions, null values ignored.
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        # Test arrays
        crosstab(
            [np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2])], np.array([1, 2, 1, 2])
        )

        # Test with aggfunc
        norm_counts = DataFrame(
            [[0.25, 0, 0.25], [0.25, 0.5, 0.75], [0.5, 0.5, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b"),
        )
        test_case = crosstab(
            df.a, df.b, df.c, aggfunc="count", normalize="all", margins=True
        )
        tm.assert_frame_equal(test_case, norm_counts)

        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [0, 4, np.nan, 3, 3]}
        )

        norm_sum = DataFrame(
            [[0, 0, 0.0], [0.4, 0.6, 1], [0.4, 0.6, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )
        test_case = crosstab(
            df.a, df.b, df.c, aggfunc=np.sum, normalize="all", margins=True
        )
        tm.assert_frame_equal(test_case, norm_sum)

    def test_crosstab_with_empties(self):
        # Check handling of empties
        # [FUNCTION DOC]
        # Values that are all null with a count aggregation: normalizing (true, index, columns) gives all-zero float tables, while without normalization the table has a null cell where nothing was counted.
        df = DataFrame(
            {
                "a": [1, 2, 2, 2, 2],
                "b": [3, 3, 4, 4, 4],
                "c": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        empty = DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            index=Index([1, 2], name="a", dtype="int64"),
            columns=Index([3, 4], name="b"),
        )

        for i in [True, "index", "columns"]:
            calculated = crosstab(df.a, df.b, values=df.c, aggfunc="count", normalize=i)
            tm.assert_frame_equal(empty, calculated)

        nans = DataFrame(
            [[0.0, np.nan], [0.0, 0.0]],
            index=Index([1, 2], name="a", dtype="int64"),
            columns=Index([3, 4], name="b"),
        )

        calculated = crosstab(df.a, df.b, values=df.c, aggfunc="count", normalize=False)
        tm.assert_frame_equal(nans, calculated)

    def test_crosstab_errors(self):
        # Issue 12578

        # [FUNCTION DOC]
        # Issue 12578. Value errors: values without an aggregation, aggregation without values, an invalid normalize argument (string or integer), and a non-boolean margins argument with normalization.
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        error = "values cannot be used without an aggfunc."
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, values=df.c)

        error = "aggfunc cannot be used without values"
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, aggfunc=np.mean)

        error = "Not a valid normalize argument"
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize="42")

        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize=42)

        error = "Not a valid margins argument"
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize="all", margins=42)

    def test_crosstab_with_categorial_columns(self):
        # GH 8860
        # [FUNCTION DOC]
        # GH 8860. A categorical column factor keeps its full category set (including categories with zero count) as an unordered categorical column index in the result.
        df = DataFrame(
            {
                "MAKE": ["Honda", "Acura", "Tesla", "Honda", "Honda", "Acura"],
                "MODEL": ["Sedan", "Sedan", "Electric", "Pickup", "Sedan", "Sedan"],
            }
        )
        categories = ["Sedan", "Electric", "Pickup"]
        df["MODEL"] = df["MODEL"].astype("category").cat.set_categories(categories)
        result = crosstab(df["MAKE"], df["MODEL"])

        expected_index = Index(["Acura", "Honda", "Tesla"], name="MAKE")
        expected_columns = CategoricalIndex(
            categories, categories=categories, ordered=False, name="MODEL"
        )
        expected_data = [[2, 0, 0], [2, 0, 1], [0, 1, 0]]
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        tm.assert_frame_equal(result, expected)

    def test_crosstab_with_numpy_size(self):
        # GH 4003
        # [FUNCTION DOC]
        # GH 4003. Using the numpy size function as aggregation of values with margins and two row factors yields float cells with nulls for empty combinations, and an integer All column; the row multi-index ends with the All entry.
        df = DataFrame(
            {
                "A": ["one", "one", "two", "three"] * 6,
                "B": ["A", "B", "C"] * 8,
                "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
                "D": np.random.default_rng(2).standard_normal(24),
                "E": np.random.default_rng(2).standard_normal(24),
            }
        )
        result = crosstab(
            index=[df["A"], df["B"]],
            columns=[df["C"]],
            margins=True,
            aggfunc=np.size,
            values=df["D"],
        )
        expected_index = MultiIndex(
            levels=[["All", "one", "three", "two"], ["", "A", "B", "C"]],
            codes=[[1, 1, 1, 2, 2, 2, 3, 3, 3, 0], [1, 2, 3, 1, 2, 3, 1, 2, 3, 0]],
            names=["A", "B"],
        )
        expected_column = Index(["bar", "foo", "All"], name="C")
        expected_data = np.array(
            [
                [2.0, 2.0, 4.0],
                [2.0, 2.0, 4.0],
                [2.0, 2.0, 4.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [12.0, 12.0, 24.0],
            ]
        )
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_column
        )
        # aggfunc is np.size, resulting in integers
        expected["All"] = expected["All"].astype("int64")
        tm.assert_frame_equal(result, expected)

    def test_crosstab_duplicate_names(self):
        # GH 13279 / 22529

        # [FUNCTION DOC]
        # GH 13279 and GH 22529. Row and/or column factors sharing the same name still produce a table identical to one built with unique names and then relabelled, for duplicate row, duplicate column and mixed cases.
        s1 = Series(range(3), name="foo")
        s2_foo = Series(range(1, 4), name="foo")
        s2_bar = Series(range(1, 4), name="bar")
        s3 = Series(range(3), name="waldo")

        # check result computed with duplicate labels against
        # result computed with unique labels, then relabelled
        mapper = {"bar": "foo"}

        # duplicate row, column labels
        result = crosstab(s1, s2_foo)
        expected = crosstab(s1, s2_bar).rename_axis(columns=mapper, axis=1)
        tm.assert_frame_equal(result, expected)

        # duplicate row, unique column labels
        result = crosstab([s1, s2_foo], s3)
        expected = crosstab([s1, s2_bar], s3).rename_axis(index=mapper, axis=0)
        tm.assert_frame_equal(result, expected)

        # unique row, duplicate column labels
        result = crosstab(s3, [s1, s2_foo])
        expected = crosstab(s3, [s1, s2_bar]).rename_axis(columns=mapper, axis=1)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("names", [["a", ("b", "c")], [("a", "b"), "c"]])
    def test_crosstab_tuple_name(self, names):
        # [FUNCTION DOC]
        # Series named by a tuple in either position produce a table whose axis names are those tuples; compared with a unstacked series of ones filled with zero.
        s1 = Series(range(3), name=names[0])
        s2 = Series(range(1, 4), name=names[1])

        mi = MultiIndex.from_arrays([range(3), range(1, 4)], names=names)
        expected = Series(1, index=mi).unstack(1, fill_value=0)

        result = crosstab(s1, s2)
        tm.assert_frame_equal(result, expected)

    def test_crosstab_both_tuple_names(self):
        # GH 18321
        # [FUNCTION DOC]
        # GH 18321. Both series named by tuples give an identity count table with tuple-named index and columns.
        s1 = Series(range(3), name=("a", "b"))
        s2 = Series(range(3), name=("c", "d"))

        expected = DataFrame(
            np.eye(3, dtype="int64"),
            index=Index(range(3), name=("a", "b")),
            columns=Index(range(3), name=("c", "d")),
        )
        result = crosstab(s1, s2)
        tm.assert_frame_equal(result, expected)

    def test_crosstab_unsorted_order(self):
        # [FUNCTION DOC]
        # Row labels from an unsorted index and a two-level column factor come out sorted, with the row axis named row_0.
        df = DataFrame({"b": [3, 1, 2], "a": [5, 4, 6]}, index=["C", "A", "B"])
        result = crosstab(df.index, [df.b, df.a])
        e_idx = Index(["A", "B", "C"], name="row_0")
        e_columns = MultiIndex.from_tuples([(1, 4), (2, 6), (3, 5)], names=["b", "a"])
        expected = DataFrame(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], index=e_idx, columns=e_columns
        )
        tm.assert_frame_equal(result, expected)

    def test_crosstab_normalize_multiple_columns(self):
        # GH 15150
        # [FUNCTION DOC]
        # GH 15150. Two row factors, sum aggregation of an all-zero values column, normalize over all with margins: every cell is zero except the grand total, which is one.
        df = DataFrame(
            {
                "A": ["one", "one", "two", "three"] * 6,
                "B": ["A", "B", "C"] * 8,
                "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
                "D": [0] * 24,
                "E": [0] * 24,
            }
        )

        result = crosstab(
            [df.A, df.B],
            df.C,
            values=df.D,
            aggfunc=np.sum,
            normalize=True,
            margins=True,
        )
        expected = DataFrame(
            np.array([0] * 29 + [1], dtype=float).reshape(10, 3),
            columns=Index(["bar", "foo", "All"], name="C"),
            index=MultiIndex.from_tuples(
                [
                    ("one", "A"),
                    ("one", "B"),
                    ("one", "C"),
                    ("three", "A"),
                    ("three", "B"),
                    ("three", "C"),
                    ("two", "A"),
                    ("two", "B"),
                    ("two", "C"),
                    ("All", ""),
                ],
                names=["A", "B"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_margin_normalize(self):
        # GH 27500
        # [FUNCTION DOC]
        # GH 27500. Two row factors with a custom margin name and normalization by index, by columns and by everything; checks the expected proportions and the margin row or column placement (approximate float comparison).
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        # normalize on index
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=0
        )
        expected = DataFrame(
            [[0.5, 0.5], [0.5, 0.5], [0.666667, 0.333333], [0, 1], [0.444444, 0.555556]]
        )
        expected.index = MultiIndex(
            levels=[["Sub-Total", "bar", "foo"], ["", "one", "two"]],
            codes=[[1, 1, 2, 2, 0], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )
        expected.columns = Index(["large", "small"], name="C")
        tm.assert_frame_equal(result, expected)

        # normalize on columns
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=1
        )
        expected = DataFrame(
            [
                [0.25, 0.2, 0.222222],
                [0.25, 0.2, 0.222222],
                [0.5, 0.2, 0.333333],
                [0, 0.4, 0.222222],
            ]
        )
        expected.columns = Index(["large", "small", "Sub-Total"], name="C")
        expected.index = MultiIndex(
            levels=[["bar", "foo"], ["one", "two"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
            names=["A", "B"],
        )
        tm.assert_frame_equal(result, expected)

        # normalize on both index and column
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=True
        )
        expected = DataFrame(
            [
                [0.111111, 0.111111, 0.222222],
                [0.111111, 0.111111, 0.222222],
                [0.222222, 0.111111, 0.333333],
                [0.000000, 0.222222, 0.222222],
                [0.444444, 0.555555, 1],
            ]
        )
        expected.columns = Index(["large", "small", "Sub-Total"], name="C")
        expected.index = MultiIndex(
            levels=[["Sub-Total", "bar", "foo"], ["", "one", "two"]],
            codes=[[1, 1, 2, 2, 0], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )
        tm.assert_frame_equal(result, expected)

    def test_margin_normalize_multiple_columns(self):
        # GH 35144
        # use multiple columns with margins and normalization
        # [FUNCTION DOC]
        # GH 35144. Two column factors with margins named margin and normalization over everything: expected proportions table with a margin row and margin column.
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        result = crosstab(
            index=df.C,
            columns=[df.A, df.B],
            margins=True,
            margins_name="margin",
            normalize=True,
        )
        expected = DataFrame(
            [
                [0.111111, 0.111111, 0.222222, 0.000000, 0.444444],
                [0.111111, 0.111111, 0.111111, 0.222222, 0.555556],
                [0.222222, 0.222222, 0.333333, 0.222222, 1.0],
            ],
            index=["large", "small", "margin"],
        )
        expected.columns = MultiIndex(
            levels=[["bar", "foo", "margin"], ["", "one", "two"]],
            codes=[[0, 0, 1, 1, 2], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )
        expected.index.name = "C"
        tm.assert_frame_equal(result, expected)

    def test_margin_support_Float(self):
        # GH 50313
        # use Float64 formats and function aggfunc with margins
        # [FUNCTION DOC]
        # GH 50313. Nullable Float64 data with a sum aggregation and margins: result keeps the nullable dtype with missing cells as the missing marker and correct totals.
        df = DataFrame(
            {"A": [1, 2, 2, 1], "B": [3, 3, 4, 5], "C": [-1.0, 10.0, 1.0, 10.0]},
            dtype="Float64",
        )
        result = crosstab(
            df["A"],
            df["B"],
            values=df["C"],
            aggfunc="sum",
            margins=True,
        )
        expected = DataFrame(
            [
                [-1.0, pd.NA, 10.0, 9.0],
                [10.0, 1.0, pd.NA, 11.0],
                [9.0, 1.0, 10.0, 20.0],
            ],
            index=Index([1.0, 2.0, "All"], dtype="object", name="A"),
            columns=Index([3.0, 4.0, 5.0, "All"], dtype="object", name="B"),
            dtype="Float64",
        )
        tm.assert_frame_equal(result, expected)

    def test_margin_with_ordered_categorical_column(self):
        # GH 25278
        # [FUNCTION DOC]
        # GH 25278. A row factor that is an ordered categorical with a custom category order keeps that order, with the All margin row last.
        df = DataFrame(
            {
                "First": ["B", "B", "C", "A", "B", "C"],
                "Second": ["C", "B", "B", "B", "C", "A"],
            }
        )
        df["First"] = df["First"].astype(CategoricalDtype(ordered=True))
        customized_categories_order = ["C", "A", "B"]
        df["First"] = df["First"].cat.reorder_categories(customized_categories_order)
        result = crosstab(df["First"], df["Second"], margins=True)

        expected_index = Index(["C", "A", "B", "All"], name="First")
        expected_columns = Index(["A", "B", "C", "All"], name="Second")
        expected_data = [[1, 1, 0, 2], [0, 1, 0, 1], [0, 1, 2, 3], [1, 3, 2, 6]]
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("a_dtype", ["category", "int64"])
@pytest.mark.parametrize("b_dtype", ["category", "int64"])
def test_categoricals(a_dtype, b_dtype):
    # https://github.com/pandas-dev/pandas/issues/37465
    # [FUNCTION DOC]
    # Issue 37465. Parametrized over category or integer dtype for each factor: with margins and dropna disabled the counts match the expected table (index labels zero, one, two plus All).
    # Then one category value is removed from the first factor (values of one set to two): a categorical keeps a zero row for it, while integers drop that row (expected restricted accordingly, All column cast to 64-bit integer).
    g = np.random.default_rng(2)
    a = Series(g.integers(0, 3, size=100)).astype(a_dtype)
    b = Series(g.integers(0, 2, size=100)).astype(b_dtype)
    result = crosstab(a, b, margins=True, dropna=False)
    columns = Index([0, 1, "All"], dtype="object", name="col_0")
    index = Index([0, 1, 2, "All"], dtype="object", name="row_0")
    values = [[10, 18, 28], [23, 16, 39], [17, 16, 33], [50, 50, 100]]
    expected = DataFrame(values, index, columns)
    tm.assert_frame_equal(result, expected)

    # Verify when categorical does not have all values present
    a.loc[a == 1] = 2
    a_is_cat = isinstance(a.dtype, CategoricalDtype)
    assert not a_is_cat or a.value_counts().loc[1] == 0
    result = crosstab(a, b, margins=True, dropna=False)
    values = [[10, 18, 28], [0, 0, 0], [40, 32, 72], [50, 50, 100]]
    expected = DataFrame(values, index, columns)
    if not a_is_cat:
        expected = expected.loc[[0, 2, "All"]]
        expected["All"] = expected["All"].astype("int64")
    tm.assert_frame_equal(result, expected)
