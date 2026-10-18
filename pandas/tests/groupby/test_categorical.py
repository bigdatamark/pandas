# [MODULE DOC]
# Tests for groupby on categorical keys and categorical data: the observed option (cartesian product of categories versus only observed ones), ordering of ordered and unordered categories with sort, unobserved categories in results of reductions/transformations/apply/agg (with the expected fill value per function), NaN categories and dropna, categorical indexes and MultiIndexes, datetime and interval categories, as_index false, and numerous regressions.
# Module-level contents: a helper that reindexes a result to the full cartesian product of groupers, and a dictionary giving the expected value that each built-in groupby function yields for an unobserved category (shared across Series and DataFrame tests). Uses the observed, sort, dropna, as_index, df and groupby_func style fixtures and the helper building method arguments from the groupby tests package; the deprecated corrwith emits a Pandas4Warning.
from datetime import datetime

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    qcut,
)
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args


def cartesian_product_for_groupers(result, args, names, fill_value=np.nan):
    # [FUNCTION DOC]
    # HELPER: builds the expected output for unobserved-category tests. Takes a result (frame or Series), args (a list of grouper arrays, Categorical or CategoricalIndex ones among them), names (level names) and fill_value (default NaN; may be a dict mapping column names to fill values). Builds a MultiIndex as the cartesian product of the groupers, replacing each categorical grouper with the full categories in order (keeping the ordered flag), then reindexes the result to it filling missing combinations, and sorts the index. With a dict fill value it reindexes column by column and rebuilds a frame (because reindex does not accept per-column fills).
    """Reindex to a cartesian production for the groupers,
    preserving the nature (Categorical) of each grouper
    """

    def f(a):
        # [FUNCTION DOC]
        # HELPER: converts a categorical grouper to a Categorical made of all categories in order (keeping ordered); returns any other input unchanged.
        if isinstance(a, (CategoricalIndex, Categorical)):
            categories = a.categories
            a = Categorical.from_codes(
                np.arange(len(categories)), categories=categories, ordered=a.ordered
            )
        return a

    index = MultiIndex.from_product(map(f, args), names=names)
    if isinstance(fill_value, dict):
        # fill_value is a dict mapping column names to fill values
        # -> reindex column by column (reindex itself does not support this)
        res = {}
        for col in result.columns:
            res[col] = result[col].reindex(index, fill_value=fill_value[col])
        return DataFrame(res, index=index).sort_index()

    return result.reindex(index, fill_value=fill_value).sort_index()


_results_for_groupbys_with_missing_categories = {
    # This maps the builtin groupby functions to their expected outputs for
    # missing categories when they are called on a categorical grouper with
    # observed=False. Some functions are expected to return NaN, some zero.
    # These expected values can be used across several tests (i.e. they are
    # the same for SeriesGroupBy and DataFrameGroupBy) but they should only be
    # hardcoded in one place.
    "all": True,
    "any": False,
    "count": 0,
    "corrwith": np.nan,
    "first": np.nan,
    "idxmax": np.nan,
    "idxmin": np.nan,
    "last": np.nan,
    "max": np.nan,
    "mean": np.nan,
    "median": np.nan,
    "min": np.nan,
    "nth": np.nan,
    "nunique": 0,
    "prod": 1,
    "quantile": np.nan,
    "sem": np.nan,
    "size": 0,
    "skew": np.nan,
    "kurt": np.nan,
    "std": np.nan,
    "sum": 0,
    "var": np.nan,
}


def test_apply_use_categorical_name(df):
    # [FUNCTION DOC]
    # Group a column D by quantile bins of C (observed false) and apply a function returning a dict of statistics; the first name of the resulting index is the name of the binned column, C.
    cats = qcut(df.C, 4)

    def get_stats(group):
        # [FUNCTION DOC]
        # HELPER: returns a dict of min, max, count and mean of the group.
        return {
            "min": group.min(),
            "max": group.max(),
            "count": group.count(),
            "mean": group.mean(),
        }

    result = df.groupby(cats, observed=False).D.apply(get_stats)
    assert result.index.names[0] == "C"


def test_basic():
    # [FUNCTION DOC]
    # Mean of a column by an ordered categorical with an unobserved category d: the result has a CategoricalIndex named b including d, with NaN mean for d.
    cats = Categorical(
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    data = DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 4, 5], "b": cats})

    exp_index = CategoricalIndex(list("abcd"), name="b", ordered=True)
    expected = DataFrame({"a": [1, 2, 4, np.nan]}, index=exp_index)
    result = data.groupby("b", observed=False).mean()
    tm.assert_frame_equal(result, expected)


def test_basic_single_grouper():
    # [FUNCTION DOC]
    # Two ordered categorical columns with unobserved categories; grouping by one with observed false and summing numeric columns gives sum zero for the unobserved category z, with an ordered CategoricalIndex named A.
    cat1 = Categorical(["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True)
    cat2 = Categorical(["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True)
    df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})

    gb = df.groupby("A", observed=False)
    exp_idx = CategoricalIndex(["a", "b", "z"], name="A", ordered=True)
    expected = DataFrame({"values": Series([3, 7, 0], index=exp_idx)})
    result = gb.sum(numeric_only=True)
    tm.assert_frame_equal(result, expected)


def test_basic_string(using_infer_string):
    # [FUNCTION DOC]
    # GH 8623: a categorical string column on a grouped frame. Identity transform returns the column; drop_duplicates of the column equals the first two rows; apply of a function taking the first non-duplicate row gives the person_name column with index of person ids, cast to str if infer-string is on else object. Uses the using_infer_string fixture.
    # GH 8623
    x = DataFrame(
        [[1, "John P. Doe"], [2, "Jane Dove"], [1, "John P. Doe"]],
        columns=["person_id", "person_name"],
    )
    x["person_name"] = Categorical(x.person_name)

    g = x.groupby(["person_id"], observed=False)
    result = g.transform(lambda x: x)
    tm.assert_frame_equal(result, x[["person_name"]])

    result = x.drop_duplicates("person_name")
    expected = x.iloc[[0, 1]]
    tm.assert_frame_equal(result, expected)

    def f(x):
        # [FUNCTION DOC]
        # HELPER: drops duplicates on person_name and returns the first row.
        return x.drop_duplicates("person_name").iloc[0]

    result = g.apply(f)
    expected = x[["person_name"]].iloc[[0, 1]]
    expected.index = Index([1, 2], name="person_id")
    dtype = "str" if using_infer_string else object
    expected["person_name"] = expected["person_name"].astype(dtype)
    tm.assert_frame_equal(result, expected)


def test_basic_monotonic():
    # [FUNCTION DOC]
    # GH 9921: monotonic bins from cut as the grouper (observed false). Transform with the builtin sum, a numpy sum, maximum reduce variants, the filter with numpy all must all return the original column/frame unchanged since each bin contains at most one value. Some comparisons ignore dtype.
    # GH 9921
    df = DataFrame({"a": [5, 15, 25]})
    c = pd.cut(df.a, bins=[0, 10, 20, 30, 40])

    result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])

    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )
    result = df.groupby(c, observed=False).transform(sum)
    expected = df[["a"]]
    tm.assert_frame_equal(result, expected)

    gbc = df.groupby(c, observed=False)
    result = gbc.transform(lambda xs: np.max(xs, axis=0))
    tm.assert_frame_equal(result, df[["a"]])

    result2 = gbc.transform(lambda xs: np.max(xs, axis=0))
    result3 = gbc.transform(max)
    result4 = gbc.transform(np.maximum.reduce)
    result5 = gbc.transform(lambda xs: np.maximum.reduce(xs))
    tm.assert_frame_equal(result2, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result3, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result4, df[["a"]])
    tm.assert_frame_equal(result5, df[["a"]])

    # Filter
    tm.assert_series_equal(df.a.groupby(c, observed=False).filter(np.all), df["a"])
    tm.assert_frame_equal(df.groupby(c, observed=False).filter(np.all), df)


def test_basic_non_monotonic():
    # [FUNCTION DOC]
    # GH: non-monotonic variant of the cut-bin transform test: one value falls in a negative bin; transform with builtin and numpy sum returns the original column/frame.
    df = DataFrame({"a": [5, 15, 25, -5]})
    c = pd.cut(df.a, bins=[-10, 0, 10, 20, 30, 40])

    result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])

    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )
    result = df.groupby(c, observed=False).transform(sum)
    expected = df[["a"]]
    tm.assert_frame_equal(result, expected)

    tm.assert_frame_equal(
        df.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df[["a"]]
    )


def test_basic_cut_grouping():
    # [FUNCTION DOC]
    # GH 9603: cut with explicit categorical labels as the grouper, apply of length gives counts 1,0,0,0 in a CategoricalIndex named a that includes unobserved categories.
    # GH 9603
    df = DataFrame({"a": [1, 0, 0, 0]})
    c = pd.cut(df.a, [0, 1, 2, 3, 4], labels=Categorical(list("abcd")))
    result = df.groupby(c, observed=False).apply(len)

    exp_index = CategoricalIndex(c.values.categories, ordered=c.values.ordered)
    expected = Series([1, 0, 0, 0], index=exp_index)
    expected.index.name = "a"
    tm.assert_series_equal(result, expected)


def test_more_basic():
    # [FUNCTION DOC]
    # Random data grouped by an ordered categorical built from codes: mean equals grouping by the plain array values and reindexing to the category order. describe on the categorical groups equals describe on data sorted by code grouped without sorting. GH 10460: stacking the describe result gives an outer level that is categorical (each category repeated 8) and an inner level of the eight summary labels repeated four times. Uses random data.
    levels = ["foo", "bar", "baz", "qux"]
    codes = np.random.default_rng(2).integers(0, 4, size=10)

    cats = Categorical.from_codes(codes, levels, ordered=True)

    data = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

    result = data.groupby(cats, observed=False).mean()

    expected = data.groupby(np.asarray(cats), observed=False).mean()
    exp_idx = CategoricalIndex(levels, categories=cats.categories, ordered=True)
    expected = expected.reindex(exp_idx)

    tm.assert_frame_equal(result, expected)

    grouped = data.groupby(cats, observed=False)
    desc_result = grouped.describe()

    idx = cats.codes.argsort()
    ord_labels = np.asarray(cats).take(idx)
    ord_data = data.take(idx)

    exp_cats = Categorical(
        ord_labels, ordered=True, categories=["foo", "bar", "baz", "qux"]
    )
    expected = ord_data.groupby(exp_cats, sort=False, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)

    # GH 10460
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp = CategoricalIndex(expc)
    tm.assert_index_equal(desc_result.stack().index.get_level_values(0), exp)
    exp = Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4)
    tm.assert_index_equal(desc_result.stack().index.get_level_values(1), exp)


def test_level_get_group(observed):
    # [FUNCTION DOC]
    # GH15155 and GH15166: parametrized over observed. Grouping by a categorical level of a MultiIndex and calling get_group with a one-element tuple returns the rows with that category, preserving the categorical MultiIndex structure.
    # GH15155
    df = DataFrame(
        data=np.arange(2, 22, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(10)],
            codes=[[0] * 5 + [1] * 5, range(10)],
            names=["Index1", "Index2"],
        ),
    )
    g = df.groupby(level=["Index1"], observed=observed)

    # expected should equal test.loc[["a"]]
    # GH15166
    expected = DataFrame(
        data=np.arange(2, 12, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(5)],
            codes=[[0] * 5, range(5)],
            names=["Index1", "Index2"],
        ),
    )
    result = g.get_group(("a",))
    tm.assert_frame_equal(result, expected)


def test_sorting_with_different_categoricals():
    # [FUNCTION DOC]
    # GH 24271: value_counts of an ordered categorical column grouped by a string column, after sorting the index by level 0, yields counts of two for each category in a MultiIndex of group and categorical dose, in the category order low, med, high. Result Series named count.
    # GH 24271
    df = DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "dose": ["high", "med", "low"] * 4,
            "outcomes": np.arange(12.0),
        }
    )

    df.dose = Categorical(df.dose, categories=["low", "med", "high"], ordered=True)

    result = df.groupby("group")["dose"].value_counts()
    result = result.sort_index(level=0, sort_remaining=True)
    index = ["low", "med", "high", "low", "med", "high"]
    index = Categorical(index, categories=["low", "med", "high"], ordered=True)
    index = [["A", "A", "A", "B", "B", "B"], CategoricalIndex(index)]
    index = MultiIndex.from_arrays(index, names=["group", "dose"])
    expected = Series([2] * 6, index=index, name="count")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_apply(ordered):
    # [FUNCTION DOC]
    # GH 10138: parametrized over ordered. Grouping by two categoricals (one with unused category b) with observed true: apply of mean over axis 0, mean and agg with numpy mean give the same expected frame with the observed combinations only; an apply returning a constant one gives a Series of ones with the same MultiIndex (transform-like).
    # GH 10138

    dense = Categorical(list("abc"), ordered=ordered)

    # 'b' is in the categories but not in the list
    missing = Categorical(list("aaa"), categories=["a", "b"], ordered=ordered)
    values = np.arange(len(dense))
    df = DataFrame({"missing": missing, "dense": dense, "values": values})
    grouped = df.groupby(["missing", "dense"], observed=True)

    # missing category 'b' should still exist in the output index
    idx = MultiIndex.from_arrays([missing, dense], names=["missing", "dense"])
    expected = DataFrame([0, 1, 2.0], index=idx, columns=["values"])

    result = grouped.apply(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)

    result = grouped.mean()
    tm.assert_frame_equal(result, expected)

    result = grouped.agg(np.mean)
    tm.assert_frame_equal(result, expected)

    # but for transform we should still get back the original index
    idx = MultiIndex.from_arrays([missing, dense], names=["missing", "dense"])
    expected = Series(1, index=idx)
    result = grouped.apply(lambda x: 1)
    tm.assert_series_equal(result, expected)


def test_observed(observed, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over observed (and uses using_infer_string). Multiple groupers must not re-expand the output space (gh-14942, gh-10132, gh-8138, gh-8869). With three groupers including a plain string, sum matches the expected (cartesian product with zero fill when not observed). With two categorical groupers, sum of values plus string C concatenations matches the expected; when not observed missing combos are filled with zero (or with zero and the empty string per column under infer-string).
    # multiple groupers, don't re-expand the output space
    # of the grouper
    # gh-14942 (implement)
    # gh-10132 (back-compat)
    # gh-8138 (back-compat)
    # gh-8869

    cat1 = Categorical(["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True)
    cat2 = Categorical(["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True)
    df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
    df["C"] = ["foo", "bar"] * 2

    # multiple groupers with a non-cat
    gb = df.groupby(["A", "B", "C"], observed=observed)
    exp_index = MultiIndex.from_arrays(
        [cat1, cat2, ["foo", "bar"] * 2], names=["A", "B", "C"]
    )
    expected = DataFrame({"values": Series([1, 2, 3, 4], index=exp_index)}).sort_index()
    result = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [cat1, cat2, ["foo", "bar"]], list("ABC"), fill_value=0
        )

    tm.assert_frame_equal(result, expected)

    gb = df.groupby(["A", "B"], observed=observed)
    exp_index = MultiIndex.from_arrays([cat1, cat2], names=["A", "B"])
    expected = DataFrame(
        {"values": [1, 2, 3, 4], "C": ["foo", "bar", "foo", "bar"]}, index=exp_index
    )
    result = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(
            expected,
            [cat1, cat2],
            list("AB"),
            fill_value={"values": 0, "C": ""} if using_infer_string else 0,
        )

    tm.assert_frame_equal(result, expected)


def test_observed_single_column(observed):
    # [FUNCTION DOC]
    # gh-8138: mean grouped by one ordered categorical column with unobserved category c; when observed is false the result is reindexed to include c with NaN.
    # https://github.com/pandas-dev/pandas/issues/8138
    d = {
        "cat": Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True
        ),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    df = DataFrame(d)

    groups_single_key = df.groupby("cat", observed=observed)
    result = groups_single_key.mean()

    exp_index = CategoricalIndex(
        list("ab"), name="cat", categories=list("abc"), ordered=True
    )
    expected = DataFrame({"ints": [1.5, 1.5], "val": [20.0, 30]}, index=exp_index)
    if not observed:
        index = CategoricalIndex(
            list("abc"), name="cat", categories=list("abc"), ordered=True
        )
        expected = expected.reindex(index)

    tm.assert_frame_equal(result, expected)


def test_observed_two_columns(observed):
    # [FUNCTION DOC]
    # gh-8138: aggregate mean grouped by a categorical and an integer column; with observed false the expected is the cartesian product of categories and integers; and get_group for each (category, int) key returns the matching original rows (GH 10132).
    # https://github.com/pandas-dev/pandas/issues/8138
    d = {
        "cat": Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True
        ),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    df = DataFrame(d)
    groups_double_key = df.groupby(["cat", "ints"], observed=observed)
    result = groups_double_key.agg("mean")
    expected = DataFrame(
        {
            "val": [10.0, 30.0, 20.0, 40.0],
            "cat": Categorical(
                ["a", "a", "b", "b"], categories=["a", "b", "c"], ordered=True
            ),
            "ints": [1, 2, 1, 2],
        }
    ).set_index(["cat", "ints"])
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [df.cat.values, [1, 2]], ["cat", "ints"]
        )

    tm.assert_frame_equal(result, expected)

    # GH 10132
    for key in [("a", 1), ("b", 2), ("b", 1), ("a", 2)]:
        c, i = key
        result = groups_double_key.get_group(key)
        expected = df[(df.cat == c) & (df.ints == i)]
        tm.assert_frame_equal(result, expected)


def test_observed_with_as_index(observed):
    # [FUNCTION DOC]
    # gh-8869: grouping by cut bins and a string column with as_index false and with as_index true then reset_index must produce equal means.
    # gh-8869
    # with as_index
    d = {
        "foo": [10, 8, 4, 8, 4, 1, 1],
        "bar": [10, 20, 30, 40, 50, 60, 70],
        "baz": ["d", "c", "e", "a", "a", "d", "c"],
    }
    df = DataFrame(d)
    cat = pd.cut(df["foo"], np.linspace(0, 10, 3))
    df["range"] = cat
    groups = df.groupby(["range", "baz"], as_index=False, observed=observed)
    result = groups.agg("mean")

    groups2 = df.groupby(["range", "baz"], as_index=True, observed=observed)
    expected = groups2.agg("mean").reset_index()
    tm.assert_frame_equal(result, expected)


def test_observed_codes_remap(observed):
    # [FUNCTION DOC]
    # Aggregation mean grouped by categorical cut bins and an integer column whose codes need remapping; with observed false, expected is the cartesian product of the categories and the integer values.
    d = {"C1": [3, 3, 4, 5], "C2": [1, 2, 3, 4], "C3": [10, 100, 200, 34]}
    df = DataFrame(d)
    values = pd.cut(df["C1"], [1, 2, 3, 6])
    values.name = "cat"
    groups_double_key = df.groupby([values, "C2"], observed=observed)

    idx = MultiIndex.from_arrays([values, [1, 2, 3, 4]], names=["cat", "C2"])
    expected = DataFrame(
        {"C1": [3.0, 3.0, 4.0, 5.0], "C3": [10.0, 100.0, 200.0, 34.0]}, index=idx
    )
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [values.values, [1, 2, 3, 4]], ["cat", "C2"]
        )

    result = groups_double_key.agg("mean")
    tm.assert_frame_equal(result, expected)


def test_observed_perf():
    # [FUNCTION DOC]
    # gh-14942: performance guard. With 30000 random rows, a categorical of string numbers and two integer columns grouped with observed true, count must produce an index whose level sizes equal the number of unique values in each column (no cartesian product).
    # we create a cartesian product, so this is
    # non-performant if we don't use observed values
    # gh-14942
    df = DataFrame(
        {
            "cat": np.random.default_rng(2).integers(0, 255, size=30000),
            "int_id": np.random.default_rng(2).integers(0, 255, size=30000),
            "other_id": np.random.default_rng(2).integers(0, 10000, size=30000),
            "foo": 0,
        }
    )
    df["cat"] = df.cat.astype(str).astype("category")

    grouped = df.groupby(["cat", "int_id", "other_id"], observed=True)
    result = grouped.count()
    assert result.index.levels[0].nunique() == df.cat.nunique()
    assert result.index.levels[1].nunique() == df.int_id.nunique()
    assert result.index.levels[2].nunique() == df.other_id.nunique()


def test_observed_groups(observed):
    # [FUNCTION DOC]
    # gh-20583: parametrized over observed. The groups mapping of a categorical grouper with an unobserved category maps observed categories to index labels; when observed is false the unobserved category b maps to an empty int64 Index.
    # gh-20583
    # test that we have the appropriate groups

    cat = Categorical(["a", "c", "a"], categories=["a", "b", "c"])
    df = DataFrame({"cat": cat, "vals": [1, 2, 3]})
    g = df.groupby("cat", observed=observed)

    result = g.groups
    if observed:
        expected = {"a": Index([0, 2], dtype="int64"), "c": Index([1], dtype="int64")}
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "c": Index([1], dtype="int64"),
        }

    tm.assert_dict_equal(result, expected)


def test_groups_na_category(dropna, observed):
    # [FUNCTION DOC]
    # Refs GitHub issue 61356: parametrized over dropna and observed. Groups mapping for a categorical grouper with a NaN: includes the key NaN with its row labels only if dropna is false; includes unobserved categories b and d with empty Index only if observed is false.
    # https://github.com/pandas-dev/pandas/issues/61356
    df = DataFrame(
        {"cat": Categorical(["a", np.nan, "a"], categories=list("adb"))},
        index=list("xyz"),
    )
    g = df.groupby("cat", observed=observed, dropna=dropna)

    result = g.groups
    expected = {"a": Index(["x", "z"])}
    if not dropna:
        expected |= {np.nan: Index(["y"])}
    if not observed:
        expected |= {"b": Index([]), "d": Index([])}
    tm.assert_dict_equal(result, expected)


@pytest.mark.parametrize(
    "keys, expected_values, expected_index_levels",
    [
        ("a", [15, 9, 0], CategoricalIndex([1, 2, 3], name="a")),
        (
            ["a", "b"],
            [7, 8, 0, 0, 0, 9, 0, 0, 0],
            [CategoricalIndex([1, 2, 3], name="a"), Index([4, 5, 6])],
        ),
        (
            ["a", "a2"],
            [15, 0, 0, 0, 9, 0, 0, 0, 0],
            [
                CategoricalIndex([1, 2, 3], name="a"),
                CategoricalIndex([1, 2, 3], name="a"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("test_series", [True, False])
def test_unobserved_in_index(keys, expected_values, expected_index_levels, test_series):
    # [FUNCTION DOC]
    # GH#49354: parametrized over three key choices and frame or Series (c column). Grouping by categorical index levels with observed false must show unobserved categories in the result. Single key gives sums 15, 9, 0; two keys give 3x3 combinations including the second categorical level; b is dropped from the frame unless used for grouping. Expected index levels are categorical.
    # GH#49354 - ensure unobserved cats occur when grouping by index levels
    df = DataFrame(
        {
            "a": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "a2": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    ).set_index(["a", "a2"])
    if "b" not in keys:
        # Only keep b when it is used for grouping for consistent columns in the result
        df = df.drop(columns="b")

    gb = df.groupby(keys, observed=False)
    if test_series:
        gb = gb["c"]
    result = gb.sum()

    if len(keys) == 1:
        index = expected_index_levels
    else:
        codes = [[0, 0, 0, 1, 1, 1, 2, 2, 2], 3 * [0, 1, 2]]
        index = MultiIndex(
            expected_index_levels,
            codes=codes,
            names=keys,
        )
    expected = DataFrame({"c": expected_values}, index=index)
    if test_series:
        expected = expected["c"]
    tm.assert_equal(result, expected)


def test_observed_groups_with_nan(observed):
    # [FUNCTION DOC]
    # GH 24740: parametrized over observed. The groups mapping with a NaN category: NaN row excluded; unobserved categories b and d with empty index included only if observed is false.
    # GH 24740
    df = DataFrame(
        {
            "cat": Categorical(["a", np.nan, "a"], categories=["a", "b", "d"]),
            "vals": [1, 2, 3],
        }
    )
    g = df.groupby("cat", observed=observed)
    result = g.groups
    if observed:
        expected = {"a": Index([0, 2], dtype="int64")}
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "d": Index([], dtype="int64"),
        }
    tm.assert_dict_equal(result, expected)


def test_observed_nth():
    # [FUNCTION DOC]
    # GH 26385: nth at 0 of a series grouped by a categorical with NaNs and observed false returns only the first row (the first non-NaN category member).
    # GH 26385
    cat = Categorical(["a", np.nan, np.nan], categories=["a", "b", "c"])
    ser = Series([1, 2, 3])
    df = DataFrame({"cat": cat, "ser": ser})

    result = df.groupby("cat", observed=False)["ser"].nth(0)
    expected = df["ser"].iloc[[0]]
    tm.assert_series_equal(result, expected)


def test_dataframe_categorical_with_nan(observed):
    # [FUNCTION DOC]
    # GH 21151: first of a frame grouped by a categorical column with NaN and observed true or false; reset_index result has categorical s1 with only a for observed; all categories with NaN for unobserved b and c when not observed.
    # GH 21151
    s1 = Categorical([np.nan, "a", np.nan, "a"], categories=["a", "b", "c"])
    s2 = Series([1, 2, 3, 4])
    df = DataFrame({"s1": s1, "s2": s2})
    result = df.groupby("s1", observed=observed).first().reset_index()
    if observed:
        expected = DataFrame(
            {"s1": Categorical(["a"], categories=["a", "b", "c"]), "s2": [2]}
        )
    else:
        expected = DataFrame(
            {
                "s1": Categorical(["a", "b", "c"], categories=["a", "b", "c"]),
                "s2": [2, np.nan, np.nan],
            }
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_dataframe_categorical_ordered_observed_sort(ordered, observed, sort):
    # [FUNCTION DOC]
    # GH 25871 and GH 25167: parametrized over ordered, observed and sort. Aggregating first of val per label with an unobserved category: the labels in the result index must equal the aggregated values (an unobserved category's NaN aggregation is mapped to the string missing); otherwise the test fails with a message about inconsistent sorting. The check shows both sort and observed settings yield a correctly ordered result.
    # GH 25871: Fix groupby sorting on ordered Categoricals
    # GH 25167: Groupby with observed=True doesn't sort

    # Build a dataframe with cat having one unobserved category ('missing'),
    # and a Series with identical values
    label = Categorical(
        ["d", "a", "b", "a", "d", "b"],
        categories=["a", "b", "missing", "d"],
        ordered=ordered,
    )
    val = Series(["d", "a", "b", "a", "d", "b"])
    df = DataFrame({"label": label, "val": val})

    # aggregate on the Categorical
    result = df.groupby("label", observed=observed, sort=sort)["val"].aggregate("first")

    # If ordering works, we expect index labels equal to aggregation results,
    # except for 'observed=False': label 'missing' has aggregation None
    label = Series(result.index.array, dtype="object")
    aggr = Series(result.array)
    if not observed:
        aggr[aggr.isna()] = "missing"
    if not all(label == aggr):
        msg = (
            "Labels and aggregation results not consistently sorted\n"
            f"for (ordered={ordered}, observed={observed}, sort={sort})\n"
            f"Result:\n{result}"
        )
        pytest.fail(msg)


def test_datetime():
    # [FUNCTION DOC]
    # GH9049: backwards compatibility with datetime categories: mean by ordered categorical of dates equals grouping by the array and reindexing to all categories; describe equals describing the code-sorted data; stacked describe output has a categorical outer level (GH 10460) and inner statistic labels.
    # GH9049: ensure backward compatibility
    levels = pd.date_range("2014-01-01", periods=4)
    codes = np.random.default_rng(2).integers(0, 4, size=10)

    cats = Categorical.from_codes(codes, levels, ordered=True)

    data = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result = data.groupby(cats, observed=False).mean()

    expected = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )

    tm.assert_frame_equal(result, expected)

    grouped = data.groupby(cats, observed=False)
    desc_result = grouped.describe()

    idx = cats.codes.argsort()
    ord_labels = cats.take(idx)
    ord_data = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )

    # GH 10460
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp = CategoricalIndex(expc)
    tm.assert_index_equal((desc_result.stack().index.get_level_values(0)), exp)
    exp = Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4)
    tm.assert_index_equal((desc_result.stack().index.get_level_values(1)), exp)


def test_categorical_index():
    # [FUNCTION DOC]
    # Sum grouped by a categorical index level (via set_index and level 0) and by a categorical column must both give a CategoricalIndex of ordered categories named cats, equal to sums grouped by integer codes. Uses random category codes.
    s = np.random.default_rng(2)
    levels = ["foo", "bar", "baz", "qux"]
    codes = s.integers(0, 4, size=20)
    cats = Categorical.from_codes(codes, levels, ordered=True)
    df = DataFrame(np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd"))
    df["cats"] = cats

    # with a cat index
    result = df.set_index("cats").groupby(level=0, observed=False).sum()
    expected = df[list("abcd")].groupby(cats.codes, observed=False).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True), name="cats"
    )
    tm.assert_frame_equal(result, expected)

    # with a cat column, should produce a cat index
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(cats.codes, observed=False).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True), name="cats"
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns():
    # [FUNCTION DOC]
    # GH 11558: describe on groupby of a frame whose columns are an ordered CategoricalIndex; after stacking, the columns index and the categorical values equal the original categorical columns.
    # GH 11558
    cats = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 4)), columns=cats)
    result = df.groupby([1, 2, 3, 4] * 5).describe()

    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical():
    # [FUNCTION DOC]
    # GH11558: count per (artist, medium) with a categorical medium, unstacked and described: the columns are a categorical index named medium; the sum of the A and B columns gives 6 and 4 per artist.
    # GH11558 (example is taken from the original issue)
    df = DataFrame(
        {"a": range(10), "medium": ["A", "B"] * 5, "artist": list("XYXXY") * 2}
    )
    df["medium"] = df["medium"].astype("category")

    gcat = df.groupby(["artist", "medium"], observed=False)["a"].count().unstack()
    result = gcat.describe()

    exp_columns = CategoricalIndex(["A", "B"], ordered=False, name="medium")
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)

    result = gcat["A"] + gcat["B"]
    expected = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


def test_bins_unequal_len():
    # [FUNCTION DOC]
    # GH3011: grouping a Series by an array of cut bins whose length differs from the Series (NaNs were dropped) must raise a value error saying grouper and axis must be the same length.
    # GH3011
    series = Series([np.nan, np.nan, 1, 1, 2, 2, 3, 3, 4, 4])
    bins = pd.cut(series.dropna().values, 4)

    # len(bins) != len(series) here
    with pytest.raises(ValueError, match="Grouper and axis must be same length"):
        series.groupby(bins).mean()


@pytest.mark.parametrize(
    ["series", "data"],
    [
        # Group a series with length and index equal to those of the grouper.
        (Series(range(4)), {"A": [0, 3], "B": [1, 2]}),
        # Group a series with length equal to that of the grouper and index unequal to
        # that of the grouper.
        (Series(range(4)).rename(lambda idx: idx + 1), {"A": [2], "B": [0, 1]}),
        # GH44179: Group a series with length unequal to that of the grouper.
        (Series(range(7)), {"A": [0, 3], "B": [1, 2]}),
    ],
)
def test_categorical_series(series, data):
    # [FUNCTION DOC]
    # Parametrized over three Series (matching index; same length with a shifted index; longer than the grouper, GH44179). Grouping by a categorical Series of A and B and aggregating to lists gives a Series of lists per category, matched by index alignment, with a CategoricalIndex of the keys.
    # Group the given series by a series with categorical data type such that group A
    # takes indices 0 and 3 and group B indices 1 and 2, obtaining the values mapped in
    # the given data.
    groupby = series.groupby(Series(list("ABBA"), dtype="category"), observed=False)
    result = groupby.aggregate(list)
    expected = Series(data, index=CategoricalIndex(data.keys()))
    tm.assert_series_equal(result, expected)


def test_as_index():
    # [FUNCTION DOC]
    # GH13204: as_index false with categorical keys and observed true. Cases: a categorical column plus integer column sum; a function grouper (named level_1 in the output); another grouper Series sharing the name cat (conflicting names) which replaces the categorical; and whether the original index is dropped regardless of its name (None, X or B).
    # GH13204
    df = DataFrame(
        {
            "cat": Categorical([1, 2, 2], [1, 2, 3]),
            "A": [10, 11, 11],
            "B": [101, 102, 103],
        }
    )
    result = df.groupby(["cat", "A"], as_index=False, observed=True).sum()
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "A": [10, 11],
            "B": [101, 205],
        },
        columns=["cat", "A", "B"],
    )
    tm.assert_frame_equal(result, expected)

    # function grouper
    f = lambda r: df.loc[r, "A"]
    result = df.groupby(["cat", f], as_index=False, observed=True).sum()
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "level_1": [10, 11],
            "A": [10, 22],
            "B": [101, 205],
        },
    )
    tm.assert_frame_equal(result, expected)

    # another not in-axis grouper (conflicting names in index)
    s = Series(["a", "b", "b"], name="cat")
    result = df.groupby(["cat", s], as_index=False, observed=True).sum()
    expected = DataFrame(
        {
            "cat": ["a", "b"],
            "A": [10, 22],
            "B": [101, 205],
        },
    )
    tm.assert_frame_equal(result, expected)

    # is original index dropped?
    group_columns = ["cat", "A"]
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "A": [10, 11],
            "B": [101, 205],
        },
        columns=["cat", "A", "B"],
    )

    for name in [None, "X", "B"]:
        df.index = Index(list("abc"), name=name)
        result = df.groupby(group_columns, as_index=False, observed=True).sum()

        tm.assert_frame_equal(result, expected)


def test_preserve_categories():
    # [FUNCTION DOC]
    # GH-13179: for an ordered categorical grouper with observed false, sort true gives a CategoricalIndex ordered by category (a, b, c); sort false keeps first-appearance order then unobserved (b, a, c) with ordered preserved (GH#42482).
    # GH-13179
    categories = list("abc")

    # ordered=True
    df = DataFrame({"A": Categorical(list("ba"), categories=categories, ordered=True)})
    sort_index = CategoricalIndex(categories, categories, ordered=True, name="A")
    nosort_index = CategoricalIndex(list("bac"), categories, ordered=True, name="A")
    tm.assert_index_equal(
        df.groupby("A", sort=True, observed=False).first().index, sort_index
    )
    # GH#42482 - don't sort result when sort=False, even when ordered=True
    tm.assert_index_equal(
        df.groupby("A", sort=False, observed=False).first().index, nosort_index
    )


def test_preserve_categories_ordered_false():
    # [FUNCTION DOC]
    # GH-13179: same as the previous test with an unordered categorical: sorting uses the category order, while sort false keeps appearance order, and category order is not changed (GH#48749).
    # GH-13179
    categories = list("abc")
    df = DataFrame({"A": Categorical(list("ba"), categories=categories, ordered=False)})
    sort_index = CategoricalIndex(categories, categories, ordered=False, name="A")
    # GH#48749 - don't change order of categories
    # GH#42482 - don't sort result when sort=False, even when ordered=True
    nosort_index = CategoricalIndex(list("bac"), list("abc"), ordered=False, name="A")
    tm.assert_index_equal(
        df.groupby("A", sort=True, observed=False).first().index, sort_index
    )
    tm.assert_index_equal(
        df.groupby("A", sort=False, observed=False).first().index, nosort_index
    )


@pytest.mark.parametrize("col", ["C1", "C2"])
def test_preserve_categorical_dtype(col):
    # [FUNCTION DOC]
    # GH13743, GH13854: parametrized over unordered or ordered categorical columns. Mean (numeric only) with observed false and as_index false, and with as_index true followed by reset_index, both give a frame whose categorical keeps all categories (bac) with NaN for the unobserved c.
    # GH13743, GH13854
    df = DataFrame(
        {
            "A": [1, 2, 1, 1, 2],
            "B": [10, 16, 22, 28, 34],
            "C1": Categorical(list("abaab"), categories=list("bac"), ordered=False),
            "C2": Categorical(list("abaab"), categories=list("bac"), ordered=True),
        }
    )
    # single grouper
    exp_full = DataFrame(
        {
            "A": [2.0, 1.0, np.nan],
            "B": [25.0, 20.0, np.nan],
            "C1": Categorical(list("bac"), categories=list("bac"), ordered=False),
            "C2": Categorical(list("bac"), categories=list("bac"), ordered=True),
        }
    )
    result1 = df.groupby(by=col, as_index=False, observed=False).mean(numeric_only=True)
    result2 = (
        df.groupby(by=col, as_index=True, observed=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    expected = exp_full.reindex(columns=result1.columns)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        ("first", ["second", "first"]),
        ("last", ["fourth", "third"]),
        ("min", ["fourth", "first"]),
        ("max", ["second", "third"]),
    ],
)
def test_preserve_on_ordered_ops(func, values):
    # [FUNCTION DOC]
    # gh-18502: parametrized over first, last, min, max. On an ordered categorical column, these must preserve the categorical dtype, for both frame groupby and column groupby. Expected values per payload group are given for the two groups.
    # gh-18502
    # preserve the categoricals on ops
    c = Categorical(["first", "second", "third", "fourth"], ordered=True)
    df = DataFrame({"payload": [-1, -2, -1, -2], "col": c})
    g = df.groupby("payload")
    result = getattr(g, func)()
    expected = DataFrame(
        {"payload": [-2, -1], "col": Series(values, dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)

    # we should also preserve categorical for SeriesGroupBy
    sgb = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_categorical_no_compress():
    # [FUNCTION DOC]
    # Mean by categorical of random data matches grouping by codes with the categorical index; a second case where one category is unobserved (codes skip 2) gets the category reindexed with NaN. Uses random data.
    data = Series(np.random.default_rng(2).standard_normal(9))

    codes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cats = Categorical.from_codes(codes, [0, 1, 2], ordered=True)

    result = data.groupby(cats, observed=False).mean()
    exp = data.groupby(codes, observed=False).mean()

    exp.index = CategoricalIndex(
        exp.index, categories=cats.categories, ordered=cats.ordered
    )
    tm.assert_series_equal(result, exp)

    codes = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3])
    cats = Categorical.from_codes(codes, [0, 1, 2, 3], ordered=True)

    result = data.groupby(cats, observed=False).mean()
    exp = data.groupby(codes, observed=False).mean().reindex(cats.categories)
    exp.index = CategoricalIndex(
        exp.index, categories=cats.categories, ordered=cats.ordered
    )
    tm.assert_series_equal(result, exp)


def test_categorical_no_compress_string():
    # [FUNCTION DOC]
    # Mean of a numeric column grouped by an ordered categorical string column with unobserved category d returns the means 1, 2, 4 and NaN as a numpy array.
    cats = Categorical(
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    data = DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 4, 5], "b": cats})

    result = data.groupby("b", observed=False).mean()
    result = result["a"].values
    exp = np.array([1, 2, 4, np.nan])
    tm.assert_numpy_array_equal(result, exp)


def test_groupby_empty_with_category():
    # [FUNCTION DOC]
    # GH-9614: grouping by a column of all None and taking first of a categorical column must keep the categorical dtype (with categories test and train), producing an empty categorical Series with an empty object index named A.
    # GH-9614
    # test fix for when group by on None resulted in
    # coercion of dtype categorical -> float
    df = DataFrame({"A": [None] * 3, "B": Categorical(["train", "train", "test"])})
    result = df.groupby("A").first()["B"]
    expected = Series(
        Categorical([], categories=["test", "train"]),
        index=Series([], dtype="object", name="A"),
        name="B",
    )
    tm.assert_series_equal(result, expected)


def test_sort():
    # [FUNCTION DOC]
    # Counts of binned values grouped by cut bin labels (a categorical) give a result in label order (sorted by lower bound); expected is built by sorting the labels numerically. Based on a stack overflow question about sorting categorical labels. Random data.
    # https://stackoverflow.com/questions/23814368/sorting-pandas-
    #        categorical-labels-after-groupby
    # This should result in a properly sorted Series so that the plot
    # has a sorted x axis
    # self.cat.groupby(['value_group'])['value_group'].count().plot(kind='bar')

    df = DataFrame({"value": np.random.default_rng(2).integers(0, 10000, 10)})
    labels = [f"{i} - {i + 499}" for i in range(0, 10000, 500)]
    cat_labels = Categorical(labels, labels)

    df = df.sort_values(by=["value"], ascending=True)
    df["value_group"] = pd.cut(
        df.value, range(0, 10500, 500), right=False, labels=cat_labels
    )

    res = df.groupby(["value_group"], observed=False)["value_group"].count()
    exp = res[sorted(res.index, key=lambda x: float(x.split()[0]))]
    exp.index = CategoricalIndex(exp.index, name=exp.index.name)
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize("ordered", [True, False])
def test_sort2(sort, ordered):
    # [FUNCTION DOC]
    # GH 8868: parametrized over ordered and the sort fixture. First per category of a frame where the range column holds interval-like strings: with sort true the categories are sorted (alphabetical category order of the strings); with sort false the result follows appearance order. GH#48749, GH#42482: category order isn't changed and ordered doesn't force sorting.
    # dataframe groupby sort was being ignored # GH 8868
    # GH#48749 - don't change order of categories
    # GH#42482 - don't sort result when sort=False, even when ordered=True
    df = DataFrame(
        [
            ["(7.5, 10]", 10, 10],
            ["(7.5, 10]", 8, 20],
            ["(2.5, 5]", 5, 30],
            ["(5, 7.5]", 6, 40],
            ["(2.5, 5]", 4, 50],
            ["(0, 2.5]", 1, 60],
            ["(5, 7.5]", 7, 70],
        ],
        columns=["range", "foo", "bar"],
    )
    df["range"] = Categorical(df["range"], ordered=ordered)
    result = df.groupby("range", sort=sort, observed=False).first()

    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = ["(0, 2.5]", "(2.5, 5]", "(5, 7.5]", "(7.5, 10]"]
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = ["(7.5, 10]", "(2.5, 5]", "(5, 7.5]", "(0, 2.5]"]
    expected = DataFrame(
        data_values,
        columns=["foo", "bar"],
        index=CategoricalIndex(index_values, name="range", ordered=ordered),
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_sort_datetimelike(sort, ordered):
    # [FUNCTION DOC]
    # GH10505: same as the string-range sort test but with datetime categories: sort true yields chronological order, sort false yields appearance order, for ordered or not.
    # GH10505
    # GH#42482 - don't sort result when sort=False, even when ordered=True

    # use same data as test_groupby_sort_categorical, which category is
    # corresponding to datetime.month
    df = DataFrame(
        {
            "dt": [
                datetime(2011, 7, 1),
                datetime(2011, 7, 1),
                datetime(2011, 2, 1),
                datetime(2011, 5, 1),
                datetime(2011, 2, 1),
                datetime(2011, 1, 1),
                datetime(2011, 5, 1),
            ],
            "foo": [10, 8, 5, 6, 4, 1, 7],
            "bar": [10, 20, 30, 40, 50, 60, 70],
        },
        columns=["dt", "foo", "bar"],
    )

    # ordered=True
    df["dt"] = Categorical(df["dt"], ordered=ordered)
    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = [
            datetime(2011, 1, 1),
            datetime(2011, 2, 1),
            datetime(2011, 5, 1),
            datetime(2011, 7, 1),
        ]
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = [
            datetime(2011, 7, 1),
            datetime(2011, 2, 1),
            datetime(2011, 5, 1),
            datetime(2011, 1, 1),
        ]
    expected = DataFrame(
        data_values,
        columns=["foo", "bar"],
        index=CategoricalIndex(index_values, name="dt", ordered=ordered),
    )
    result = df.groupby("dt", sort=sort, observed=False).first()
    tm.assert_frame_equal(result, expected)


def test_empty_sum():
    # [FUNCTION DOC]
    # Issue 18678: sum over an unobserved category gives 0 by default and with min_count zero; with min_count 1 it gives NaN for the unobserved category; with min_count 2 the single-member category also gives NaN.
    # https://github.com/pandas-dev/pandas/issues/18678
    df = DataFrame(
        {"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"]), "B": [1, 2, 1]}
    )
    expected_idx = CategoricalIndex(["a", "b", "c"], name="A")

    # 0 by default
    result = df.groupby("A", observed=False).B.sum()
    expected = Series([3, 1, 0], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count=0
    result = df.groupby("A", observed=False).B.sum(min_count=0)
    expected = Series([3, 1, 0], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count=1
    result = df.groupby("A", observed=False).B.sum(min_count=1)
    expected = Series([3, 1, np.nan], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count>1
    result = df.groupby("A", observed=False).B.sum(min_count=2)
    expected = Series([3, np.nan, np.nan], expected_idx, name="B")
    tm.assert_series_equal(result, expected)


def test_empty_prod():
    # [FUNCTION DOC]
    # Issue 18678: product over an unobserved category gives 1 by default and with min_count 0; with min_count 1 it gives NaN.
    # https://github.com/pandas-dev/pandas/issues/18678
    df = DataFrame(
        {"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"]), "B": [1, 2, 1]}
    )

    expected_idx = CategoricalIndex(["a", "b", "c"], name="A")

    # 1 by default
    result = df.groupby("A", observed=False).B.prod()
    expected = Series([2, 1, 1], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count=0
    result = df.groupby("A", observed=False).B.prod(min_count=0)
    expected = Series([2, 1, 1], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count=1
    result = df.groupby("A", observed=False).B.prod(min_count=1)
    expected = Series([2, 1, np.nan], expected_idx, name="B")
    tm.assert_series_equal(result, expected)


def test_groupby_multiindex_categorical_datetime():
    # [FUNCTION DOC]
    # Issue 21390: grouping on two categorical columns (letters and datetimes) with observed false and taking mean yields all combinations (MultiIndex from product) with NaN for the missing combination.
    # https://github.com/pandas-dev/pandas/issues/21390

    df = DataFrame(
        {
            "key1": Categorical(list("abcbabcba")),
            "key2": Categorical(
                list(pd.date_range("2018-06-01 00", freq="1min", periods=3)) * 3
            ),
            "values": np.arange(9),
        }
    )
    result = df.groupby(["key1", "key2"], observed=False).mean()

    idx = MultiIndex.from_product(
        [
            Categorical(["a", "b", "c"]),
            Categorical(pd.date_range("2018-06-01 00", freq="1min", periods=3)),
        ],
        names=["key1", "key2"],
    )
    expected = DataFrame({"values": [0, 4, 8, 3, 4, 5, 6, np.nan, 2]}, index=idx)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "as_index, expected",
    [
        (
            True,
            Series(
                index=MultiIndex.from_arrays(
                    [Series([1, 1, 2], dtype="category"), [1, 2, 2]], names=["a", "b"]
                ),
                data=[1, 2, 3],
                name="x",
            ),
        ),
        (
            False,
            DataFrame(
                {
                    "a": Series([1, 1, 2], dtype="category"),
                    "b": [1, 2, 2],
                    "x": [1, 2, 3],
                }
            ),
        ),
    ],
)
def test_groupby_agg_observed_true_single_column(as_index, expected):
    # [FUNCTION DOC]
    # GH-23970: parametrized over as_index. Summing column x grouped by a categorical and integer column with observed true gives a Series indexed by the observed combinations (named x), or a frame with a, b and x columns for as_index false.
    # GH-23970
    df = DataFrame(
        {"a": Series([1, 1, 2], dtype="category"), "b": [1, 2, 2], "x": [1, 2, 3]}
    )

    result = df.groupby(["a", "b"], as_index=as_index, observed=True)["x"].sum()

    tm.assert_equal(result, expected)


@pytest.mark.parametrize("fill_value", [None, np.nan, pd.NaT])
def test_shift(fill_value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite living in the groupby tests, this test never uses a groupby; it exercises Categorical shift directly.
    # Parametrized over fill value None, NaN, NaT. Shifting a Categorical by one position fills the first slot with a missing value and keeps the categories and unordered flag.
    ct = Categorical(
        ["a", "b", "c", "d"], categories=["a", "b", "c", "d"], ordered=False
    )
    expected = Categorical(
        [None, "a", "b", "c"], categories=["a", "b", "c", "d"], ordered=False
    )
    res = ct.shift(1, fill_value=fill_value)
    tm.assert_equal(res, expected)


@pytest.fixture
def df_cat(df):
    # [FUNCTION DOC]
    # HELPER: pytest fixture used by the observed tests. Takes the df fixture, copies its first four rows (to leave out some groups), converts columns A and B to categorical, replaces column C with the integers one to four and drops column D. Returns this frame with two categorical columns and an integer column.
    """
    DataFrame with multiple categorical columns and a column of integers.
    Shortened so as not to contain all possible combinations of categories.
    Useful for testing `observed` kwarg functionality on GroupBy objects.

    Parameters
    ----------
    df: DataFrame
        Non-categorical, longer DataFrame from another fixture, used to derive
        this one

    Returns
    -------
    df_cat: DataFrame
    """
    df_cat = df.copy()[:4]  # leave out some groups
    df_cat["A"] = df_cat["A"].astype("category")
    df_cat["B"] = df_cat["B"].astype("category")
    df_cat["C"] = Series([1, 2, 3, 4])
    df_cat = df_cat.drop(["D"], axis=1)
    return df_cat


@pytest.mark.parametrize("operation", ["agg", "apply"])
def test_seriesgroupby_observed_true(df_cat, operation):
    # [FUNCTION DOC]
    # GH#24880 and GH#49223: parametrized over agg and apply with the builtin sum. A Series grouped by two categorical columns with observed true gives only observed combinations, correctly ordered (sorted index) as a Series named C.
    # GH#24880
    # GH#49223 - order of results was wrong when grouping by index levels
    lev_a = Index(["bar", "bar", "foo", "foo"], dtype=df_cat["A"].dtype, name="A")
    lev_b = Index(["one", "three", "one", "two"], dtype=df_cat["B"].dtype, name="B")
    index = MultiIndex.from_arrays([lev_a, lev_b])
    expected = Series(data=[2, 4, 1, 3], index=index, name="C").sort_index()

    grouped = df_cat.groupby(["A", "B"], observed=True)["C"]
    result = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("operation", ["agg", "apply"])
@pytest.mark.parametrize("observed", [False, None])
def test_seriesgroupby_observed_false_or_none(df_cat, observed, operation):
    # [FUNCTION DOC]
    # GH 24880 and GH#49223: parametrized over agg/apply and observed false or none (default). All six combinations of the two categoricals appear, with sum zero for unobserved ones, in sorted order.
    # GH 24880
    # GH#49223 - order of results was wrong when grouping by index levels
    index, _ = MultiIndex.from_product(
        [
            CategoricalIndex(["bar", "foo"], ordered=False),
            CategoricalIndex(["one", "three", "two"], ordered=False),
        ],
        names=["A", "B"],
    ).sortlevel()

    expected = Series(data=[2, 4, 0, 1, 0, 3], index=index, name="C")
    grouped = df_cat.groupby(["A", "B"], observed=observed)["C"]
    result = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(
                        ["one", "one", "three", "three", "one", "one", "two", "two"],
                        dtype="category",
                        name="B",
                    ),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(df_cat, observed, index, data):
    # [FUNCTION DOC]
    # GH 24880: parametrized over observed true/false/none. Applying a function returning a dict of min and max of each group's values yields a Series with a third index level (min, max): observed true only observed combinations, otherwise NaN for unobserved combinations.
    # GH 24880
    expected = Series(data=data, index=index, name="C")
    result = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(df_cat):
    # [FUNCTION DOC]
    # GH 20416: mean of column C grouped by two categoricals gives the same Series whether taken from the SeriesGroupBy or from the frame groupby's mean.
    # GH 20416
    expected = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    result = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    tm.assert_series_equal(result, expected)


def test_groupby_cat_preserves_structure(observed, ordered):
    # [FUNCTION DOC]
    # GH 28787: uses the observed and ordered fixtures. Aggregating sum with skipna true over each categorical group (all groups of one row) and resetting the index must reproduce the original frame (categorical dtype and order preserved).
    # GH 28787
    df = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]},
        columns=["Name", "Item"],
    )
    expected = df.copy()

    result = (
        df.groupby("Name", observed=observed)
        .agg(DataFrame.sum, skipna=True)
        .reset_index()
    )

    tm.assert_frame_equal(result, expected)


def test_get_nonexistent_category():
    # [FUNCTION DOC]
    # Accessing a nonexistent column inside an applied function raises a key error naming it (vau); not specific to categories.
    # Accessing a Category that is not in the dataframe
    df = DataFrame({"var": ["a", "a", "b", "b"], "val": range(4)})
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby("var").apply(lambda rows: DataFrame({"val": [rows.iloc[-1]["vau"]]}))


def test_series_groupby_on_2_categoricals_unobserved(reduction_func, observed):
    # [FUNCTION DOC]
    # GH 17605: uses the reduction_func and observed fixtures. Skips ngroup. For a Series grouped by two categoricals with a number of observed combinations of four (sixteen if unobserved are included), each reduction must return a result of that length; corrwith must not exist on the SeriesGroupBy (TODO GH 32293). For idxmin/idxmax with unobserved categories, a value error saying the group is empty due to unobserved categories is expected.
    # GH 17605
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")

    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABCD")),
            "cat_2": Categorical(list("AB") * 2, categories=list("ABCD")),
            "value": [0.1] * 4,
        }
    )
    args = get_groupby_method_args(reduction_func, df)

    expected_length = 4 if observed else 16

    series_groupby = df.groupby(["cat_1", "cat_2"], observed=observed)["value"]

    if reduction_func == "corrwith":
        # TODO: implemented SeriesGroupBy.corrwith. See GH 32293
        assert not hasattr(series_groupby, reduction_func)
        return

    agg = getattr(series_groupby, reduction_func)

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        # idxmin and idxmax are designed to fail on empty inputs
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            agg(*args)
        return

    result = agg(*args)

    assert len(result) == expected_length


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func, request
):
    # [FUNCTION DOC]
    # GH 17605: uses the reduction_func fixture and request. Skips ngroup; marks corrwith as an expected failure (TODO GH 32293, not implemented for SeriesGroupBy). Groups a value column by two categoricals (with unused categories) with observed false, and checks the unobserved combinations are filled with the per-function value from the module dictionary (NaN, zero, one, true or false). For idxmin and idxmax a value error about an empty group due to unobserved categories is expected instead. If the fill is zero, the result dtype must be integer for count, nunique and size; otherwise the function must be sum or any.
    # GH 17605
    # Tests whether the unobserved categories in the result contain 0 or NaN

    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")

    if reduction_func == "corrwith":  # GH 32293
        mark = pytest.mark.xfail(
            reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293"
        )
        request.applymarker(mark)

    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("AB") * 2, categories=list("ABC")),
            "value": [0.1] * 4,
        }
    )
    unobserved = [tuple("AC"), tuple("BC"), tuple("CA"), tuple("CB"), tuple("CC")]
    args = get_groupby_method_args(reduction_func, df)

    series_groupby = df.groupby(["cat_1", "cat_2"], observed=False)["value"]
    agg = getattr(series_groupby, reduction_func)

    if reduction_func in ["idxmin", "idxmax"]:
        # idxmin and idxmax are designed to fail on empty inputs
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            agg(*args)
        return

    result = agg(*args)

    missing_fillin = _results_for_groupbys_with_missing_categories[reduction_func]

    for idx in unobserved:
        val = result.loc[idx]
        assert (pd.isna(missing_fillin) and pd.isna(val)) or (val == missing_fillin)

    # If we expect unobserved values to be zero, we also expect the dtype to be int.
    # Except for .sum(). If the observed categories sum to dtype=float (i.e. their
    # sums have decimals), then the zeros for the missing categories should also be
    # floats.
    if missing_fillin == 0:
        if reduction_func in ["count", "nunique", "size"]:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ["sum", "any"]


def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(reduction_func):
    # [FUNCTION DOC]
    # GH 23865 and GH 27075: uses the reduction_func fixture. Skips ngroup. For a frame grouped by two categoricals with observed true, none of the unobserved category combinations may appear in the index of any reduction result. For corrwith the deprecation warning (Pandas4Warning) is asserted.
    # GH 23865
    # GH 27075
    # Ensure that df.groupby, when 'by' is two Categorical variables,
    # does not return the categories that are not in df when observed=True
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")

    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("1111"), categories=list("12")),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats = [("A", "2"), ("B", "2"), ("C", "1"), ("C", "2")]

    df_grp = df.groupby(["cat_1", "cat_2"], observed=True)

    args = get_groupby_method_args(reduction_func, df)
    if reduction_func == "corrwith":
        warn = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        res = getattr(df_grp, reduction_func)(*args)

    for cat in unobserved_cats:
        assert cat not in res.index


@pytest.mark.parametrize("observed", [False, None])
def test_dataframe_groupby_on_2_categoricals_when_observed_is_false(
    reduction_func, observed, using_python_scalars
):
    # [FUNCTION DOC]
    # GH 23865 and GH 27075: parametrized over observed false/none and the reduction_func and using_python_scalars fixtures. Skips ngroup. With unobserved categories present, each reduction must include unobserved combinations in the result, filled with the expected value (NaN, zero, one, true/false per function). idxmin and idxmax must raise a value error about an empty group instead. Size with python-scalar mode compares via an identity check on the all result. corrwith asserts its deprecation warning.
    # GH 23865
    # GH 27075
    # Ensure that df.groupby, when 'by' is two Categorical variables,
    # returns the categories that are not in df when observed=False/None

    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")

    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("1111"), categories=list("12")),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats = [("A", "2"), ("B", "2"), ("C", "1"), ("C", "2")]

    df_grp = df.groupby(["cat_1", "cat_2"], observed=observed)

    args = get_groupby_method_args(reduction_func, df)

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        # idxmin and idxmax are designed to fail on empty inputs
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            getattr(df_grp, reduction_func)(*args)
        return

    if reduction_func == "corrwith":
        warn = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        res = getattr(df_grp, reduction_func)(*args)

    expected = _results_for_groupbys_with_missing_categories[reduction_func]

    if using_python_scalars and reduction_func == "size":
        assert (res.loc[unobserved_cats] == expected).all() is True
    elif expected is np.nan:
        assert res.loc[unobserved_cats].isnull().all().all()
    else:
        assert (res.loc[unobserved_cats] == expected).all().all()


def test_series_groupby_categorical_aggregation_getitem():
    # [FUNCTION DOC]
    # GH 8870: the SeriesGroupBy aggregation mean on column foo grouped by cut bins and a string with observed false must equal selecting foo from the frame aggregation.
    # GH 8870
    d = {"foo": [10, 8, 4, 1], "bar": [10, 20, 30, 40], "baz": ["d", "c", "d", "c"]}
    df = DataFrame(d)
    cat = pd.cut(df["foo"], np.linspace(0, 20, 5))
    df["range"] = cat
    groups = df.groupby(["range", "baz"], as_index=True, sort=True, observed=False)
    result = groups["foo"].agg("mean")
    expected = groups.agg("mean")["foo"]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_values",
    [(Series.nunique, [1, 1, 2]), (Series.count, [1, 2, 2])],
)
def test_groupby_agg_categorical_columns(func, expected_values):
    # [FUNCTION DOC]
    # Issue 31256: parametrized over Series nunique (1, 1, 2) and Series count (1, 2, 2). Aggregating a categorical value column grouped by integers gives integer results per group.
    # 31256
    df = DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "groups": [0, 1, 1, 2, 2],
            "value": Categorical([0, 0, 0, 0, 1]),
        }
    ).set_index("id")
    result = df.groupby("groups").agg(func)

    expected = DataFrame(
        {"value": expected_values}, index=Index([0, 1, 2], name="groups")
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_agg_non_numeric():
    # [FUNCTION DOC]
    # Aggregating a categorical column with nunique, as a function and as a method, grouped by an external list gives distinct counts (2 and 1).
    df = DataFrame({"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"])})
    expected = DataFrame({"A": [2, 1]}, index=np.array([1, 2]))

    result = df.groupby([1, 2, 1]).agg(Series.nunique)
    tm.assert_frame_equal(result, expected)

    result = df.groupby([1, 2, 1]).nunique()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["first", "last"])
def test_groupby_first_returned_categorical_instead_of_dataframe(func):
    # [FUNCTION DOC]
    # GH 28641: parametrized over first and last. A single-row frame with an ordered categorical column grouped by A: first/last of column B returns a Series (not a Categorical) preserving the ordered categorical dtype, indexed by A.
    # GH 28641: groupby drops index, when grouping over categorical column with
    # first/last. Renamed Categorical instead of DataFrame previously.
    df = DataFrame({"A": [1997], "B": Series(["b"], dtype="category").cat.as_ordered()})
    df_grouped = df.groupby("A")["B"]
    result = getattr(df_grouped, func)()

    # ordered categorical dtype should be preserved
    expected = Series(
        ["b"], index=Index([1997], name="A"), name="B", dtype=df["B"].dtype
    )
    tm.assert_series_equal(result, expected)


def test_read_only_category_no_sort():
    # [FUNCTION DOC]
    # GH33410: categories backed by a read-only numpy array; mean with sort false and observed false must work and give 2.0 and 6.0 in a CategoricalIndex named b.
    # GH33410
    cats = np.array([1, 2])
    cats.flags.writeable = False
    df = DataFrame(
        {"a": [1, 3, 5, 7], "b": Categorical([1, 1, 2, 2], categories=Index(cats))}
    )
    expected = DataFrame(data={"a": [2.0, 6.0]}, index=CategoricalIndex(cats, name="b"))
    result = df.groupby("b", sort=False, observed=False).mean()
    tm.assert_frame_equal(result, expected)


def test_sorted_missing_category_values():
    # [FUNCTION DOC]
    # GH 28597: size grouped by a string column and an ordered categorical (with an unused category tiny), unstacked: gives a frame with counts for every category including zero for tiny, a categorical column index in category order named foo, and an index named bar.
    # GH 28597
    df = DataFrame(
        {
            "foo": [
                "small",
                "large",
                "large",
                "large",
                "medium",
                "large",
                "large",
                "medium",
            ],
            "bar": ["C", "A", "A", "C", "A", "C", "A", "C"],
        }
    )
    df["foo"] = (
        df["foo"]
        .astype("category")
        .cat.set_categories(["tiny", "small", "medium", "large"], ordered=True)
    )

    expected = DataFrame(
        {
            "tiny": {"A": 0, "C": 0},
            "small": {"A": 0, "C": 1},
            "medium": {"A": 1, "C": 1},
            "large": {"A": 3, "C": 2},
        }
    )
    expected = expected.rename_axis("bar", axis="index")
    expected.columns = CategoricalIndex(
        ["tiny", "small", "medium", "large"],
        categories=["tiny", "small", "medium", "large"],
        ordered=True,
        name="foo",
        dtype="category",
    )

    result = df.groupby(["bar", "foo"], observed=False).size().unstack()

    tm.assert_frame_equal(result, expected)


def test_agg_cython_category_not_implemented_fallback():
    # [FUNCTION DOC]
    # Issue 31450: first on a column with unordered categorical dtype (not implemented in the compiled path) falls back and keeps the categorical dtype both through selection and agg with a dict. The comment notes that the unordered case is a less clear one (it arguably should raise).
    # https://github.com/pandas-dev/pandas/issues/31450
    df = DataFrame({"col_num": [1, 1, 2, 3]})
    df["col_cat"] = df["col_num"].astype("category")

    result = df.groupby("col_num").col_cat.first()

    # ordered categorical dtype should definitely be preserved;
    #  this is unordered, so is less-clear case (if anything, it should raise)
    expected = Series(
        [1, 2, 3],
        index=Index([1, 2, 3], name="col_num"),
        name="col_cat",
        dtype=df["col_cat"].dtype,
    )
    tm.assert_series_equal(result, expected)

    result = df.groupby("col_num").agg({"col_cat": "first"})
    expected = expected.to_frame()
    tm.assert_frame_equal(result, expected)


def test_aggregate_categorical_with_isnan():
    # [FUNCTION DOC]
    # GH 29837: aggregating with a function that counts missing values per group on numeric, object and categorical columns gives (1, 0), (0, 0), (0, 0) for the two groups.
    # GH 29837
    df = DataFrame(
        {
            "A": [1, 1, 1, 1],
            "B": [1, 2, 1, 2],
            "numerical_col": [0.1, 0.2, np.nan, 0.3],
            "object_col": ["foo", "bar", "foo", "fee"],
            "categorical_col": ["foo", "bar", "foo", "fee"],
        }
    )

    df = df.astype({"categorical_col": "category"})

    result = df.groupby(["A", "B"]).agg(lambda df: df.isna().sum())
    index = MultiIndex.from_arrays([[1, 1], [1, 2]], names=("A", "B"))
    expected = DataFrame(
        data={
            "numerical_col": [1, 0],
            "object_col": [0, 0],
            "categorical_col": [0, 0],
        },
        index=index,
    )
    tm.assert_frame_equal(result, expected)


def test_categorical_transform():
    # [FUNCTION DOC]
    # GH 29037: transform with the builtin max on an ordered categorical column adds a column last_status; the transform result for the added column is the string Waiting for every package, with the categorical dtype preserved.
    # KNOWN ISSUE: with the ordered categories Waiting, OnTheWay, Delivered, a maximum by category order would give Delivered for package 1; the expected Waiting is only consistent with comparing the plain string labels lexicographically, so the test name and ordered dtype are misleading about what is verified.
    # GH 29037
    df = DataFrame(
        {
            "package_id": [1, 1, 1, 2, 2, 3],
            "status": [
                "Waiting",
                "OnTheWay",
                "Delivered",
                "Waiting",
                "OnTheWay",
                "Waiting",
            ],
        }
    )

    delivery_status_type = pd.CategoricalDtype(
        categories=["Waiting", "OnTheWay", "Delivered"], ordered=True
    )
    df["status"] = df["status"].astype(delivery_status_type)
    df["last_status"] = df.groupby("package_id")["status"].transform(max)
    result = df.copy()

    expected = DataFrame(
        {
            "package_id": [1, 1, 1, 2, 2, 3],
            "status": [
                "Waiting",
                "OnTheWay",
                "Delivered",
                "Waiting",
                "OnTheWay",
                "Waiting",
            ],
            "last_status": [
                "Waiting",
                "Waiting",
                "Waiting",
                "Waiting",
                "Waiting",
                "Waiting",
            ],
        }
    )

    expected["status"] = expected["status"].astype(delivery_status_type)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["first", "last"])
def test_series_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: str, observed: bool
):
    # [FUNCTION DOC]
    # GH 34951: parametrized over first/last and observed. A Series c grouped by two categorical columns: expected first values 0, NaN, NaN, 1 (and last 1, NaN, NaN, 0) over the full product index; with observed true only the observed combinations and cast to int64.
    # GH 34951
    cat = Categorical([0, 0, 1, 1])
    val = [0, 1, 1, 0]
    df = DataFrame({"a": cat, "b": cat, "c": val})

    cat2 = Categorical([0, 1])
    idx = MultiIndex.from_product([cat2, cat2], names=["a", "b"])
    expected_dict = {
        "first": Series([0, np.nan, np.nan, 1], idx, name="c"),
        "last": Series([1, np.nan, np.nan, 0], idx, name="c"),
    }

    expected = expected_dict[func]
    if observed:
        expected = expected.dropna().astype(np.int64)

    srs_grp = df.groupby(["a", "b"], observed=observed)["c"]
    result = getattr(srs_grp, func)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["first", "last"])
def test_df_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: str, observed: bool
):
    # [FUNCTION DOC]
    # GH 34951: the DataFrameGroupBy version of the preceding test: first/last of the value column on a frame, expected is the Series result converted to a frame, with the same observed handling.
    # GH 34951
    cat = Categorical([0, 0, 1, 1])
    val = [0, 1, 1, 0]
    df = DataFrame({"a": cat, "b": cat, "c": val})

    cat2 = Categorical([0, 1])
    idx = MultiIndex.from_product([cat2, cat2], names=["a", "b"])
    expected_dict = {
        "first": Series([0, np.nan, np.nan, 1], idx, name="c"),
        "last": Series([1, np.nan, np.nan, 0], idx, name="c"),
    }

    expected = expected_dict[func].to_frame()
    if observed:
        expected = expected.dropna().astype(np.int64)

    df_grp = df.groupby(["a", "b"], observed=observed)
    result = getattr(df_grp, func)()
    tm.assert_frame_equal(result, expected)


def test_groupby_categorical_indices_unused_categories():
    # [FUNCTION DOC]
    # GH#38642: the indices mapping of a categorical grouper with sort false and observed false lists the observed categories in appearance order then the unobserved category c with an empty intp array; the key order must match.
    # GH#38642
    df = DataFrame(
        {
            "key": Categorical(["b", "b", "a"], categories=["a", "b", "c"]),
            "col": range(3),
        }
    )
    grouped = df.groupby("key", sort=False, observed=False)
    result = grouped.indices
    expected = {
        "b": np.array([0, 1], dtype="intp"),
        "a": np.array([2], dtype="intp"),
        "c": np.array([], dtype="intp"),
    }
    assert result.keys() == expected.keys()
    for key in result.keys():
        tm.assert_numpy_array_equal(result[key], expected[key])


@pytest.mark.parametrize("func", ["first", "last"])
def test_groupby_last_first_preserve_categoricaldtype(func):
    # [FUNCTION DOC]
    # GH#33090: parametrized over first and last. On a column of categorical integers with each group of one row, the result Series keeps the categorical dtype, named b and indexed by a.
    # GH#33090
    df = DataFrame({"a": [1, 2, 3]})
    df["b"] = df["a"].astype("category")
    result = getattr(df.groupby("a")["b"], func)()
    expected = Series(
        Categorical([1, 2, 3]), name="b", index=Index([1, 2, 3], name="a")
    )
    tm.assert_series_equal(expected, result)


def test_groupby_categorical_observed_nunique():
    # [FUNCTION DOC]
    # GH#45128: nunique of c grouped by two categoricals with observed true gives ones indexed by a MultiIndex of categorical indexes.
    # GH#45128
    df = DataFrame({"a": [1, 2], "b": [1, 2], "c": [10, 11]})
    df = df.astype(dtype={"a": "category", "b": "category"})
    result = df.groupby(["a", "b"], observed=True).nunique()["c"]
    expected = Series(
        [1, 1],
        index=MultiIndex.from_arrays(
            [CategoricalIndex([1, 2], name="a"), CategoricalIndex([1, 2], name="b")]
        ),
        name="c",
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_aggregate_functions():
    # [FUNCTION DOC]
    # GH#37275: max of an ordered categorical column grouped by integer gives the highest category per group under category order (big for group 1, small for group 2) preserving the ordered dtype.
    # GH#37275
    dtype = pd.CategoricalDtype(categories=["small", "big"], ordered=True)
    df = DataFrame(
        [[1, "small"], [1, "big"], [2, "small"]], columns=["grp", "description"]
    ).astype({"description": dtype})

    result = df.groupby("grp")["description"].max()
    expected = Series(
        ["big", "small"],
        index=Index([1, 2], name="grp"),
        name="description",
        dtype=pd.CategoricalDtype(categories=["small", "big"], ordered=True),
    )

    tm.assert_series_equal(result, expected)


def test_groupby_categorical_dropna(observed, dropna):
    # [FUNCTION DOC]
    # GH#48645: uses the observed and dropna fixtures. When no values are NA, dropna has no effect: observed true gives only observed categories; otherwise the unobserved category 3 appears with zero. The index name is x.
    # GH#48645 - dropna should have no impact on the result when there are no NA values
    cat = Categorical([1, 2], categories=[1, 2, 3])
    df = DataFrame({"x": Categorical([1, 2], categories=[1, 2, 3]), "y": [3, 4]})
    gb = df.groupby("x", observed=observed, dropna=dropna)
    result = gb.sum()

    if observed:
        expected = DataFrame({"y": [3, 4]}, index=cat)
    else:
        index = CategoricalIndex([1, 2, 3], [1, 2, 3])
        expected = DataFrame({"y": [3, 4, 0]}, index=index)
    expected.index.name = "x"

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_reducer(
    request, as_index, sort, observed, reduction_func, index_kind, ordered
):
    # [FUNCTION DOC]
    # GH#48749: parametrized over index kind (range, single, multi), ordered, and the request, as_index, sort, observed, reduction_func fixtures. The categories of the grouping level (and of the second level a2 for multi) in the output of any reduction must remain in the original category order (1, 4, 3, 2), regardless of sorting. corrwith with as_index false and not a single index is an expected failure (GH#49950). Non-range index with as_index false is skipped. idxmin/idxmax with unobserved categories expect a value error about an empty group. corrwith asserts the deprecation warning.
    # GH#48749
    if reduction_func == "corrwith" and not as_index and index_kind != "single":
        msg = "GH#49950 - corrwith with as_index=False may not have grouping column"
        request.applymarker(pytest.mark.xfail(reason=msg))
    elif index_kind != "range" and not as_index:
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    args = get_groupby_method_args(reduction_func, df)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        # idxmin and idxmax are designed to fail on empty inputs
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            getattr(gb, reduction_func)(*args)
        return
    if reduction_func == "corrwith":
        warn = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        op_result = getattr(gb, reduction_func)(*args)
    if as_index:
        result = op_result.index.get_level_values("a").categories
    else:
        result = op_result["a"].cat.categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["single", "multi"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_transformer(
    as_index, sort, observed, transformation_func, index_kind, ordered
):
    # [FUNCTION DOC]
    # GH#48749: same category-order preservation check for transformation functions (transformation_func fixture) with single or multi index keys, ordered flag, as_index, sort and observed. The categories of the output index levels remain 1, 4, 3, 2.
    # GH#48749
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    if index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    args = get_groupby_method_args(transformation_func, df)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result = getattr(gb, transformation_func)(*args)
    result = op_result.index.get_level_values("a").categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_head_tail(
    as_index, sort, observed, method, index_kind, ordered
):
    # [FUNCTION DOC]
    # GH#48749: category order is preserved in the output of head and tail for range, single and multi index kinds, with as_index, sort, observed and ordered variants. For a range index the categories are taken from the column.
    # GH#48749
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result = getattr(gb, method)()
    if index_kind == "range":
        result = op_result["a"].cat.categories
    else:
        result = op_result.index.get_level_values("a").categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("method", ["apply", "agg", "transform"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_apply(as_index, sort, observed, method, index_kind, ordered):
    # [FUNCTION DOC]
    # GH#48749: category order is preserved in the output of apply, agg and transform with a function that sums numeric columns. Skips combos without categories in the result (transform with a range index, or as_index false with a non-range index). For transform or as_index false with a range index the categories are taken from the column, otherwise from the index levels.
    # GH#48749
    if (method == "transform" and index_kind == "range") or (
        not as_index and index_kind != "range"
    ):
        pytest.skip("No categories in result, nothing to test")
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result = getattr(gb, method)(lambda x: x.sum(numeric_only=True))
    if (method == "transform" or not as_index) and index_kind == "range":
        result = op_result["a"].cat.categories
    else:
        result = op_result.index.get_level_values("a").categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
def test_many_categories(as_index, sort, index_kind, ordered):
    # [FUNCTION DOC]
    # GH#48749: a grouper with 10000 categories in descending order; summing with observed true. The expected index holds only the three observed categories in sorted category order for sort true (3, 2, 1) or appearance order (2, 1, 3) for sort false, with ordered kept; as_index false returns the columns and multi gives a MultiIndex. Skips non-range index kinds with as_index false.
    # GH#48749 - Test when the grouper has many categories
    if index_kind != "range" and not as_index:
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    categories = np.arange(9999, -1, -1)
    grouper = Categorical([2, 1, 2, 3], categories=categories, ordered=ordered)
    df = DataFrame({"a": grouper, "b": range(4)})
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=True)
    result = gb.sum()

    # Test is setup so that data and index are the same values
    data = [3, 2, 1] if sort else [2, 1, 3]

    index = CategoricalIndex(
        data, categories=grouper.categories, ordered=ordered, name="a"
    )
    if as_index:
        expected = DataFrame({"b": data})
        if index_kind == "multi":
            expected.index = MultiIndex.from_frame(DataFrame({"a": index, "a2": index}))
        else:
            expected.index = index
    elif index_kind == "multi":
        expected = DataFrame({"a": Series(index), "a2": Series(index), "b": data})
    else:
        expected = DataFrame({"a": Series(index), "b": data})

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("test_series", [True, False])
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_agg_list(request, as_index, observed, reduction_func, test_series, keys):
    # [FUNCTION DOC]
    # GH#52760: parametrized over Series or frame groupby, one or two categorical keys, and the request, as_index, observed, reduction_func fixtures. agg with a one-element list of the reduction must equal the direct reduction, with the columns relabelled (MultiIndex with empty second level for key columns, or key columns plus the function name for Series as_index false). corrwith: not available for SeriesGroupBy (skipped) and an expected failure for frames (GH#32293). idxmin/idxmax with unobserved combinations raise a value error about an empty group.
    # GH#52760
    if test_series and reduction_func == "corrwith":
        assert not hasattr(SeriesGroupBy, "corrwith")
        pytest.skip("corrwith not implemented for SeriesGroupBy")
    elif reduction_func == "corrwith":
        msg = "GH#32293: attempts to call SeriesGroupBy.corrwith"
        request.applymarker(pytest.mark.xfail(reason=msg))

    df = DataFrame({"a1": [0, 0, 1], "a2": [2, 3, 3], "b": [4, 5, 6]})
    df = df.astype({"a1": "category", "a2": "category"})
    if "a2" not in keys:
        df = df.drop(columns="a2")
    gb = df.groupby(by=keys, as_index=as_index, observed=observed)
    if test_series:
        gb = gb["b"]
    args = get_groupby_method_args(reduction_func, df)

    if not observed and reduction_func in ["idxmin", "idxmax"] and keys == ["a1", "a2"]:
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            gb.agg([reduction_func], *args)
        return

    result = gb.agg([reduction_func], *args)
    expected = getattr(gb, reduction_func)(*args)

    if as_index and (test_series or reduction_func == "size"):
        expected = expected.to_frame(reduction_func)
    if not test_series:
        expected.columns = MultiIndex.from_tuples(
            [(ind, "") for ind in expected.columns[:-1]] + [("b", reduction_func)]
        )
    elif not as_index:
        expected.columns = [*keys, reduction_func]

    tm.assert_equal(result, expected)


def test_categorical_with_noncategorical_na(observed, sort):
    # [FUNCTION DOC]
    # Refs GitHub issue 63920: grouping by a string column and an ordered categorical column that contains NaN, parametrized over observed and sort. Sum must give the expected combinations: the NaN-categorical row is dropped; ordering and unobserved categories (zeros for all combinations with category 3) follow sort and observed. The expected index rows are taken in the order chosen by the combination of sort and observed.
    # https://github.com/pandas-dev/pandas/issues/63920
    df = DataFrame(
        {
            "dates": list("YXXYY"),
            "sector": Categorical(
                [2, 1, 2, 1, np.nan], categories=[1, 2, 3], ordered=True
            ),
            "metric": [1, 2, 3, 4, 5],
        }
    )
    gb = df.groupby(["dates", "sector"], observed=observed, sort=sort)
    # Only testing the ids/result_index, okay to just use one kernel
    result = gb.sum()

    if sort and observed:
        taker = [0, 1, 2, 3]
    elif not sort and observed:
        taker = [3, 0, 1, 2]
    elif sort and not observed:
        taker = [0, 1, 4, 2, 3, 5]
    elif not sort and not observed:
        taker = [3, 0, 1, 2, 5, 4]
    expected = (
        DataFrame(
            {
                "dates": list("XXYYXY"),
                "sector": Categorical(
                    [1, 2, 1, 2, 3, 3], categories=[1, 2, 3], ordered=True
                ),
                "metric": [2, 3, 4, 1, 0, 0],
            }
        )
        .set_index(["dates", "sector"])
        .take(taker)
    )
    tm.assert_frame_equal(result, expected)
