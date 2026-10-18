"""
these are systematically testing all of the args to value_counts
with different size combinations. This is to ensure stability of the sorting
and proper parameter handling
"""

# [MODULE DOC]
# Tests for value_counts on SeriesGroupBy and DataFrameGroupBy: comparison against applying the plain Series/DataFrame value_counts per group, normalization and sorting options, dropna handling, categorical groupers and non-groupers with observed true/false, as_index false output layout, duplicate and generated column labels, the subset argument and its validation, time groupers, integer column labels and several issue regressions.
# Many tests use the shared education_df fixture defined here and the sort, ascending, dropna, as_index, unit and nulls_fixture fixtures from the groupby conftest. Several are marked as non-strict expected failures because of unstable sorting of duplicates with some numpy builds.
import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    date_range,
    to_datetime,
)
import pandas._testing as tm


def tests_value_counts_index_names_category_column():
    # [FUNCTION DOC]
    # GH44324: grouping a frame with a categorical gender column by country and calling value_counts on the gender column must give a MultiIndex whose level names are country and gender with the categorical dtype kept, and a Series named count with the value one.
    # GH44324 Missing name of index category column
    df = DataFrame(
        {
            "gender": ["female"],
            "country": ["US"],
        }
    )
    df["gender"] = df["gender"].astype("category")
    result = df.groupby("country")["gender"].value_counts()

    # Construct expected, very specific multiindex
    df_mi_expected = DataFrame([["US", "female"]], columns=["country", "gender"])
    df_mi_expected["gender"] = df_mi_expected["gender"].astype("category")
    mi_expected = MultiIndex.from_frame(df_mi_expected)
    expected = Series([1], index=mi_expected, name="count")

    tm.assert_series_equal(result, expected)


def seed_df(seed_nans, n, m):
    # [FUNCTION DOC]
    # HELPER: data builder for the slow Series value_counts test. Takes seed_nans (bool), n (row count) and m (maximum integer). Returns a frame with a random letter column 1st, a random date column 2nd (drawn from ten days) and a random integer column 3rd between one and m. When seed_nans is true the third column is cast to float and NaN is injected at fixed strided positions of all three columns. Uses a seeded generator for reproducibility.
    days = date_range("2015-08-24", periods=10)

    frame = DataFrame(
        {
            "1st": np.random.default_rng(2).choice(list("abcd"), n),
            "2nd": np.random.default_rng(2).choice(days, n),
            "3rd": np.random.default_rng(2).integers(1, m + 1, n),
        }
    )

    if seed_nans:
        # Explicitly cast to float to avoid implicit cast when setting nan
        frame["3rd"] = frame["3rd"].astype("float")
        frame.loc[1::11, "1st"] = np.nan
        frame.loc[3::17, "2nd"] = np.nan
        frame.loc[7::19, "3rd"] = np.nan
        frame.loc[8::19, "3rd"] = np.nan
        frame.loc[9::19, "3rd"] = np.nan

    return frame


@pytest.mark.slow
@pytest.mark.parametrize("seed_nans", [True, False])
@pytest.mark.parametrize("num_rows", [10, 50])
@pytest.mark.parametrize("max_int", [5, 20])
@pytest.mark.parametrize("keys", ["1st", "2nd", ["1st", "2nd"]], ids=repr)
@pytest.mark.parametrize("bins", [None, [0, 5]], ids=repr)
@pytest.mark.parametrize("isort", [True, False])
@pytest.mark.parametrize("normalize, name", [(True, "proportion"), (False, "count")])
def test_series_groupby_value_counts(
    seed_nans,
    num_rows,
    max_int,
    keys,
    bins,
    isort,
    normalize,
    name,
    sort,
    ascending,
    dropna,
):
    # [FUNCTION DOC]
    # Slow test parametrized over seeding NaN, row count, max int, grouping keys, bins, group sort, normalize (with matching result name), plus sort, ascending and dropna fixtures. Compares value_counts on the grouped 3rd column with applying the standalone Series value_counts to each group, after renaming the last index name and result name, rebuilding the MultiIndex and sorting both by index since sort order of ties is unstable (xref GH9212, GH 49909).
    df = seed_df(seed_nans, num_rows, max_int)

    def rebuild_index(df):
        # [FUNCTION DOC]
        # HELPER: nested function that rebuilds the index of a Series as a fresh MultiIndex from the level values of its existing index, preserving the names; mutates the input's index in place and returns the same object.
        arr = list(map(df.index.get_level_values, range(df.index.nlevels)))
        df.index = MultiIndex.from_arrays(arr, names=df.index.names)
        return df

    kwargs = {
        "normalize": normalize,
        "sort": sort,
        "ascending": ascending,
        "dropna": dropna,
        "bins": bins,
    }

    gr = df.groupby(keys, sort=isort)
    left = gr["3rd"].value_counts(**kwargs)

    gr = df.groupby(keys, sort=isort)
    right = gr["3rd"].apply(Series.value_counts, **kwargs)
    right.index.names = [*right.index.names[:-1], "3rd"]
    # https://github.com/pandas-dev/pandas/issues/49909
    right = right.rename(name)

    # have to sort on index because of unstable sort on values
    left, right = map(rebuild_index, (left, right))  # xref GH9212
    tm.assert_series_equal(left.sort_index(), right.sort_index())


@pytest.mark.parametrize("utc", [True, False])
def test_series_groupby_value_counts_with_grouper(utc):
    # [FUNCTION DOC]
    # GH28479: parametrized over utc. Groups a frame with unix-second timestamps (with row 3 dropped) by a one-day time Grouper on a datetime key; value_counts of Food must equal applying the Series value_counts to each group, compared after sorting the index, with the expected renamed to count and index names copied from the result.
    # GH28479
    df = DataFrame(
        {
            "Timestamp": [
                1565083561,
                1565083561 + 86400,
                1565083561 + 86500,
                1565083561 + 86400 * 2,
                1565083561 + 86400 * 3,
                1565083561 + 86500 * 3,
                1565083561 + 86400 * 4,
            ],
            "Food": ["apple", "apple", "banana", "banana", "orange", "orange", "pear"],
        }
    ).drop([3])

    df["Datetime"] = to_datetime(df["Timestamp"], utc=utc, unit="s")
    dfg = df.groupby(Grouper(freq="1D", key="Datetime"))

    # have to sort on index because of unstable sort on values xref GH9212
    result = dfg["Food"].value_counts().sort_index()
    expected = dfg["Food"].apply(Series.value_counts).sort_index()
    expected.index.names = result.index.names
    # https://github.com/pandas-dev/pandas/issues/49909
    expected = expected.rename("count")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
def test_series_groupby_value_counts_empty(columns):
    # [FUNCTION DOC]
    # GH39172: empty frame with two or three columns, grouped by all but the last; value_counts of the last column returns an empty Series named count with a MultiIndex of empty levels named by the columns.
    # GH39172
    df = DataFrame(columns=columns)
    dfg = df.groupby(columns[:-1])

    result = dfg[columns[-1]].value_counts()
    expected = Series([], dtype=result.dtype, name="count")
    expected.index = MultiIndex.from_arrays([[]] * len(columns), names=columns)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
def test_series_groupby_value_counts_one_row(columns):
    # [FUNCTION DOC]
    # GH42618: single-row frame, grouped by all but the last column; value_counts of the last column equals the frame-level value_counts.
    # GH42618
    df = DataFrame(data=[range(len(columns))], columns=columns)
    dfg = df.groupby(columns[:-1])

    result = dfg[columns[-1]].value_counts()
    expected = df.value_counts()

    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_on_categorical():
    # [FUNCTION DOC]
    # GH38672: a categorical Series with an unobserved category grouped by a constant list; value_counts keeps the unobserved category with count zero in a MultiIndex of group key and categorical index; Series named count.
    # GH38672

    s = Series(Categorical(["a"], categories=["a", "b"]))
    result = s.groupby([0]).value_counts()

    expected = Series(
        data=[1, 0],
        index=MultiIndex.from_arrays(
            [
                np.array([0, 0]),
                CategoricalIndex(
                    ["a", "b"], categories=["a", "b"], ordered=False, dtype="category"
                ),
            ]
        ),
        name="count",
    )

    # Expected:
    # 0  a    1
    #    b    0
    # dtype: int64

    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_no_sort():
    # [FUNCTION DOC]
    # GH#50482: grouping by country and gender with group sort false, value_counts with sort false must preserve order of first appearance of the group and value combos, with count column named count and an explicit MultiIndex of three levels.
    # GH#50482
    df = DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )
    gb = df.groupby(["country", "gender"], sort=False)["education"]
    result = gb.value_counts(sort=False)
    index = MultiIndex(
        levels=[["US", "FR"], ["male", "female"], ["low", "medium", "high"]],
        codes=[[0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 1, 2, 0, 2]],
        names=["country", "gender", "education"],
    )
    expected = Series([1, 1, 1, 2, 1], index=index, name="count")
    tm.assert_series_equal(result, expected)


@pytest.fixture
def education_df():
    # [FUNCTION DOC]
    # Pytest fixture returning a small six-row frame with string columns gender, education and country used by the many value_counts tests in this module.
    return DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )


def test_bad_subset(education_df):
    # [FUNCTION DOC]
    # Grouped value_counts with a subset naming the grouping column must raise a value error mentioning subset.
    gp = education_df.groupby("country")
    with pytest.raises(ValueError, match="subset"):
        gp.value_counts(subset=["country"])


def test_basic(education_df, request):
    # [FUNCTION DOC]
    # gh43564: normalized value_counts of the gender and education columns grouped by country yields proportions within each country, with a three-level index and a Series named proportion. Marked as a non-strict expected failure for the unstable sort of ties.
    # gh43564
    request.applymarker(
        pytest.mark.xfail(
            reason=(
                "pandas default unstable sorting of duplicates"
                "issue with numpy>=1.25 with AVX instructions"
            ),
            strict=False,
        )
    )
    result = education_df.groupby("country")[["gender", "education"]].value_counts(
        normalize=True
    )
    expected = Series(
        data=[0.5, 0.25, 0.25, 0.5, 0.5],
        index=MultiIndex.from_tuples(
            [
                ("FR", "male", "low"),
                ("FR", "male", "medium"),
                ("FR", "female", "high"),
                ("US", "male", "low"),
                ("US", "female", "high"),
            ],
            names=["country", "gender", "education"],
        ),
        name="proportion",
    )
    tm.assert_series_equal(result, expected)


def _frame_value_counts(df, keys, normalize, sort, ascending):
    # [FUNCTION DOC]
    # HELPER: used by the 3-way comparison test. Takes a frame, a list of column keys, normalize, sort and ascending, and returns the frame-level value_counts of those columns with the given options.
    return df[keys].value_counts(normalize=normalize, sort=sort, ascending=ascending)


@pytest.mark.parametrize("groupby", ["column", "array", "function"])
@pytest.mark.parametrize("normalize, name", [(True, "proportion"), (False, "count")])
@pytest.mark.parametrize(
    "sort, ascending",
    [
        (False, None),
        (True, True),
        (True, False),
    ],
)
@pytest.mark.parametrize("frame", [True, False])
def test_against_frame_and_seriesgroupby(
    education_df,
    groupby,
    normalize,
    name,
    sort,
    ascending,
    as_index,
    frame,
    request,
    using_infer_string,
):
    # [FUNCTION DOC]
    # Parametrized over groupby specification (column, array or function), normalize (with matching name), sort/ascending combos, whether to compare to a frame-level apply or to a Series-level value_counts, plus as_index and using_infer_string. The grouped value_counts of gender and education must match (a) applying the frame-level value_counts per group, or (b) the Series value_counts of a combined column that is split back into gender and education. For as_index false the key columns are inserted manually (level_0 for array or function keys). Contains special handling for string-dtype inference of column names. Non-strictly xfail for frame, sort and normalize because of unstable ties ordering.
    # test all parameters:
    # - Use column, array or function as by= parameter
    # - Whether or not to normalize
    # - Whether or not to sort and how
    # - Whether or not to use the groupby as an index
    # - 3-way compare against:
    #   - apply with :meth:`~DataFrame.value_counts`
    #   - `~SeriesGroupBy.value_counts`
    if frame and sort and normalize:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    by = {
        "column": "country",
        "array": education_df["country"].values,
        "function": lambda x: education_df["country"][x] == "US",
    }[groupby]

    gp = education_df.groupby(by=by, as_index=as_index)
    result = gp[["gender", "education"]].value_counts(
        normalize=normalize, sort=sort, ascending=ascending
    )
    if frame:
        # compare against apply with DataFrame value_counts
        expected = gp.apply(
            _frame_value_counts, ["gender", "education"], normalize, sort, ascending
        )

        if as_index:
            tm.assert_series_equal(result, expected)
        else:
            name = "proportion" if normalize else "count"
            expected = expected.reset_index().rename({0: name}, axis=1)
            if groupby in ["array", "function"] and (not as_index and frame):
                expected.insert(loc=0, column="level_0", value=result["level_0"])
            else:
                expected.insert(loc=0, column="country", value=result["country"])
            tm.assert_frame_equal(result, expected)
    else:
        # compare against SeriesGroupBy value_counts
        education_df["both"] = education_df["gender"] + "-" + education_df["education"]
        expected = gp["both"].value_counts(
            normalize=normalize, sort=sort, ascending=ascending
        )
        expected.name = name
        if as_index:
            index_frame = expected.index.to_frame(index=False)
            index_frame["gender"] = index_frame["both"].str.split("-").str.get(0)
            index_frame["education"] = index_frame["both"].str.split("-").str.get(1)
            both_dtype = index_frame["both"].dtype
            index_frame = index_frame.astype(
                {"gender": both_dtype, "education": both_dtype}
            )
            del index_frame["both"]
            index_frame2 = index_frame.rename({0: None}, axis=1)
            expected.index = MultiIndex.from_frame(index_frame2)

            if index_frame2.columns.isna()[0]:
                # with using_infer_string, the columns in index_frame as string
                #  dtype, which makes the rename({0: None}) above use np.nan
                #  instead of None, so we need to set None more explicitly.
                expected.index.names = [None, *expected.index.names[1:]]
            tm.assert_series_equal(result, expected)
        else:
            expected.insert(1, "gender", expected["both"].str.split("-").str.get(0))
            expected.insert(2, "education", expected["both"].str.split("-").str.get(1))
            if using_infer_string:
                expected = expected.astype({"gender": "str", "education": "str"})
            del expected["both"]
            tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "sort, ascending, expected_rows, expected_count, expected_group_size",
    [
        (False, None, [0, 1, 2, 3, 4], [1, 1, 1, 2, 1], [1, 3, 1, 3, 1]),
        (True, False, [3, 0, 1, 2, 4], [2, 1, 1, 1, 1], [3, 1, 3, 1, 1]),
        (True, True, [0, 1, 2, 4, 3], [1, 1, 1, 1, 2], [1, 3, 1, 1, 3]),
    ],
)
def test_compound(
    education_df,
    normalize,
    sort,
    ascending,
    expected_rows,
    expected_count,
    expected_group_size,
    any_string_dtype,
    using_infer_string,
):
    # [FUNCTION DOC]
    # Parametrized over normalize and sort/ascending combos with expected row picks, counts and group sizes, and over any string dtype. Grouping by country and gender with as_index false and group sort false; value_counts on the education column returns a frame with the three key columns, plus count or proportion (counts divided by group sizes). Expected is built from original rows cast to the string dtype; pyarrow strings get nullable numeric dtypes for the count column. Handles object dtype under infer-string by casting to str.
    dtype = any_string_dtype
    education_df = education_df.astype(dtype)
    education_df.columns = education_df.columns.astype(dtype)
    # Multiple groupby keys and as_index=False
    gp = education_df.groupby(["country", "gender"], as_index=False, sort=False)
    result = gp["education"].value_counts(
        normalize=normalize, sort=sort, ascending=ascending
    )
    expected = DataFrame()
    for column in ["country", "gender", "education"]:
        expected[column] = [education_df[column][row] for row in expected_rows]
        expected = expected.astype(dtype)
        expected.columns = expected.columns.astype(dtype)
    if normalize:
        expected["proportion"] = expected_count
        expected["proportion"] /= expected_group_size
        if dtype == "string[pyarrow]":
            # TODO(nullable) also string[python] should return nullable dtypes
            expected["proportion"] = expected["proportion"].convert_dtypes()
    else:
        expected["count"] = expected_count
        if dtype == "string[pyarrow]":
            expected["count"] = expected["count"].convert_dtypes()
    if using_infer_string and dtype == object:
        expected = expected.astype(
            {"country": "str", "gender": "str", "education": "str"}
        )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort, ascending, normalize, name, expected_data, expected_index",
    [
        (False, None, False, "count", [1, 2, 1], [(1, 1, 1), (2, 4, 6), (2, 0, 0)]),
        (True, True, False, "count", [1, 1, 2], [(1, 1, 1), (2, 6, 4), (2, 0, 0)]),
        (True, False, False, "count", [2, 1, 1], [(1, 1, 1), (4, 2, 6), (0, 2, 0)]),
        (
            True,
            False,
            True,
            "proportion",
            [0.5, 0.25, 0.25],
            [(1, 1, 1), (4, 2, 6), (0, 2, 0)],
        ),
    ],
)
def test_data_frame_value_counts(
    sort, ascending, normalize, name, expected_data, expected_index
):
    # [FUNCTION DOC]
    # Parametrized over sort/ascending/normalize, expected data and index. Checks the three-way equality: DataFrame value_counts on a frame with a constant key, and groupby on that key then value_counts must produce the same Series (a MultiIndex of key, num_legs, num_wings). Mirrors tests in the frame methods value_counts tests.
    # 3-way compare with :meth:`~DataFrame.value_counts`
    # Tests from frame/methods/test_value_counts.py
    animals_df = DataFrame(
        {"key": [1, 1, 1, 1], "num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )
    result_frame = animals_df.value_counts(
        sort=sort, ascending=ascending, normalize=normalize
    )
    expected = Series(
        data=expected_data,
        index=MultiIndex.from_arrays(
            expected_index, names=["key", "num_legs", "num_wings"]
        ),
        name=name,
    )
    tm.assert_series_equal(result_frame, expected)

    result_frame_groupby = animals_df.groupby("key").value_counts(
        sort=sort, ascending=ascending, normalize=normalize
    )

    tm.assert_series_equal(result_frame_groupby, expected)


@pytest.mark.parametrize(
    "group_dropna, count_dropna, expected_rows, expected_values",
    [
        (
            False,
            False,
            [0, 1, 3, 5, 6, 7, 8, 2, 4],
            [0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0],
        ),
        (False, True, [0, 1, 3, 5, 2, 4], [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
        (True, False, [0, 1, 5, 6, 7, 8], [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]),
        (True, True, [0, 1, 5], [0.5, 0.5, 1.0]),
    ],
)
def test_dropna_combinations(
    group_dropna, count_dropna, expected_rows, expected_values, request
):
    # [FUNCTION DOC]
    # Parametrized over group-level dropna and value-level dropna with expected rows and proportions. Groups on two columns with NaNs and takes normalized sorted value_counts; the expected is built from the picked rows of the original frame as a MultiIndex of all four columns. When group dropna is false the test is marked as a non-strict xfail because of unstable sorting.
    if not group_dropna:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    nulls_df = DataFrame(
        {
            "A": [1, 1, np.nan, 4, np.nan, 6, 6, 6, 6],
            "B": [1, 1, 3, np.nan, np.nan, 6, 6, 6, 6],
            "C": [1, 2, 3, 4, 5, 6, np.nan, 8, np.nan],
            "D": [1, 2, 3, 4, 5, 6, 7, np.nan, np.nan],
        }
    )
    gp = nulls_df.groupby(["A", "B"], dropna=group_dropna)
    result = gp.value_counts(normalize=True, sort=True, dropna=count_dropna)
    columns = DataFrame()
    for column in nulls_df.columns:
        columns[column] = [nulls_df[column][row] for row in expected_rows]
    index = MultiIndex.from_frame(columns)
    expected = Series(data=expected_values, index=index, name="proportion")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dropna, expected_data, expected_index",
    [
        (
            True,
            [1, 1],
            MultiIndex.from_arrays(
                [(1, 1), ("John", "Beth"), ("Smith", "Louise")],
                names=["key", "first_name", "middle_name"],
            ),
        ),
        (
            False,
            [1, 1, 1, 1],
            MultiIndex(
                levels=[
                    Index([1]),
                    Index(["Anne", "Beth", "John"]),
                    Index(["Louise", "Smith", np.nan]),
                ],
                codes=[[0, 0, 0, 0], [2, 0, 2, 1], [1, 2, 2, 0]],
                names=["key", "first_name", "middle_name"],
            ),
        ),
    ],
)
@pytest.mark.parametrize("normalize, name", [(False, "count"), (True, "proportion")])
def test_data_frame_value_counts_dropna(
    nulls_fixture, dropna, normalize, name, expected_data, expected_index
):
    # [FUNCTION DOC]
    # GH 41334: frame-level and grouped value_counts with dropna true or false, normalize true or false and the nulls_fixture in the middle_name column. With dropna the null rows are excluded; without dropna null combinations are included as NaN level values. Normalized expected values are divided by the number of rows. Both frame-level and grouped results are compared to the same expected Series.
    # GH 41334
    # 3-way compare with :meth:`~DataFrame.value_counts`
    # Tests with nulls from frame/methods/test_value_counts.py
    names_with_nulls_df = DataFrame(
        {
            "key": [1, 1, 1, 1],
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )
    result_frame = names_with_nulls_df.value_counts(dropna=dropna, normalize=normalize)
    expected = Series(
        data=expected_data,
        index=expected_index,
        name=name,
    )
    if normalize:
        expected /= float(len(expected_data))

    tm.assert_series_equal(result_frame, expected)

    result_frame_groupby = names_with_nulls_df.groupby("key").value_counts(
        dropna=dropna, normalize=normalize
    )

    tm.assert_series_equal(result_frame_groupby, expected)


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_single_grouper_with_only_observed_categories(
    education_df, as_index, observed, normalize, name, expected_data, request
):
    # [FUNCTION DOC]
    # Single categorical grouper test with all other columns also categorical; parametrized over observed, as_index, normalize. Expected result contains all combinations of gender and education for each observed country, including zero counts, as categorical levels. For as_index false the expected is reset into a frame. Non-strict xfail for unstable sorting.
    # Test single categorical grouper with only observed grouping categories
    # when non-groupers are also categorical
    request.applymarker(
        pytest.mark.xfail(
            reason=(
                "pandas default unstable sorting of duplicates"
                "issue with numpy>=1.25 with AVX instructions"
            ),
            strict=False,
        )
    )

    gp = education_df.astype("category").groupby(
        "country", as_index=as_index, observed=observed
    )
    result = gp.value_counts(normalize=normalize)

    expected_index = MultiIndex.from_tuples(
        [
            ("FR", "male", "low"),
            ("FR", "male", "medium"),
            ("FR", "female", "high"),
            ("FR", "male", "high"),
            ("FR", "female", "low"),
            ("FR", "female", "medium"),
            ("US", "male", "low"),
            ("US", "female", "high"),
            ("US", "male", "medium"),
            ("US", "male", "high"),
            ("US", "female", "low"),
            ("US", "female", "medium"),
        ],
        names=["country", "gender", "education"],
    )

    expected_series = Series(
        data=expected_data,
        index=expected_index,
        name=name,
    )
    for i in range(3):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        tm.assert_frame_equal(result, expected)


def assert_categorical_single_grouper(
    education_df, as_index, observed, expected_index, normalize, name, expected_data
):
    # [FUNCTION DOC]
    # HELPER: shared body for the observed-true and observed-false categorical single-grouper tests. Takes the education frame, as_index, observed, the expected index tuples, normalize, the result name and the expected data. Casts everything to category, adds an unobserved category ASIA to country, groups by country, takes value_counts, and compares to an expected Series built from the tuples with categorical levels (the country level reset to the full categories); as_index false compares against the reset frame.
    # Test single categorical grouper when non-groupers are also categorical
    education_df = education_df.copy().astype("category")

    # Add non-observed grouping categories
    education_df["country"] = education_df["country"].cat.add_categories(["ASIA"])

    gp = education_df.groupby("country", as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)

    expected_series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(
            expected_index,
            names=["country", "gender", "education"],
        ),
        name=name,
    )
    for i in range(3):
        index_level = CategoricalIndex(expected_series.index.levels[i])
        if i == 0:
            index_level = index_level.set_categories(
                education_df["country"].cat.categories
            )
        expected_series.index = expected_series.index.set_levels(index_level, level=i)

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name=name)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_single_grouper_observed_true(
    education_df, as_index, normalize, name, expected_data, request
):
    # [FUNCTION DOC]
    # GH#46357: observed true with a categorical grouper that has the unobserved category ASIA; delegates to the shared assertion helper with the twelve observed-country combinations (zero counts still reported for unobserved gender and education pairs). Non-strict xfail for unstable ties order.
    # GH#46357

    request.applymarker(
        pytest.mark.xfail(
            reason=(
                "pandas default unstable sorting of duplicates"
                "issue with numpy>=1.25 with AVX instructions"
            ),
            strict=False,
        )
    )

    expected_index = [
        ("FR", "male", "low"),
        ("FR", "male", "medium"),
        ("FR", "female", "high"),
        ("FR", "male", "high"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("US", "male", "low"),
        ("US", "female", "high"),
        ("US", "male", "medium"),
        ("US", "male", "high"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
    ]

    assert_categorical_single_grouper(
        education_df=education_df,
        as_index=as_index,
        observed=True,
        expected_index=expected_index,
        normalize=normalize,
        name=name,
        expected_data=expected_data,
    )


@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array(
                [2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64
            ),
        ),
        (
            True,
            "proportion",
            np.array(
                [
                    0.5,
                    0.25,
                    0.25,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    ],
)
def test_categorical_single_grouper_observed_false(
    education_df, as_index, normalize, name, expected_data, request
):
    # [FUNCTION DOC]
    # GH#46357: same as the previous test with observed false, so the unobserved country ASIA contributes six more rows with zero counts (eighteen rows in total). Delegates to the shared helper. Non-strict xfail.
    # GH#46357

    request.applymarker(
        pytest.mark.xfail(
            reason=(
                "pandas default unstable sorting of duplicates"
                "issue with numpy>=1.25 with AVX instructions"
            ),
            strict=False,
        )
    )

    expected_index = [
        ("FR", "male", "low"),
        ("FR", "male", "medium"),
        ("FR", "female", "high"),
        ("FR", "male", "high"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("US", "male", "low"),
        ("US", "female", "high"),
        ("US", "male", "medium"),
        ("US", "male", "high"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
        ("ASIA", "male", "low"),
        ("ASIA", "male", "medium"),
        ("ASIA", "male", "high"),
        ("ASIA", "female", "low"),
        ("ASIA", "female", "medium"),
        ("ASIA", "female", "high"),
    ]

    assert_categorical_single_grouper(
        education_df=education_df,
        as_index=as_index,
        observed=False,
        expected_index=expected_index,
        normalize=normalize,
        name=name,
        expected_data=expected_data,
    )


@pytest.mark.parametrize(
    "observed, expected_index",
    [
        (
            False,
            [
                ("FR", "high", "female"),
                ("FR", "high", "male"),
                ("FR", "low", "male"),
                ("FR", "low", "female"),
                ("FR", "medium", "male"),
                ("FR", "medium", "female"),
                ("US", "high", "female"),
                ("US", "high", "male"),
                ("US", "low", "male"),
                ("US", "low", "female"),
                ("US", "medium", "male"),
                ("US", "medium", "female"),
            ],
        ),
        (
            True,
            [
                ("FR", "high", "female"),
                ("FR", "low", "male"),
                ("FR", "medium", "male"),
                ("US", "high", "female"),
                ("US", "low", "male"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            # NaN values corresponds to non-observed groups
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_multiple_groupers(
    education_df, as_index, observed, expected_index, normalize, name, expected_data
):
    # [FUNCTION DOC]
    # GH#46357: two categorical groupers (country, education) with non-categorical gender, parametrized over observed and normalize. With observed false all grouper combinations appear (including zero counts), with observed true only observed ones (positive values). Expected levels of the first two index levels are categorical. as_index false compares to the reset frame.
    # GH#46357

    # Test multiple categorical groupers when non-groupers are non-categorical
    education_df = education_df.copy()
    education_df["country"] = education_df["country"].astype("category")
    education_df["education"] = education_df["education"].astype("category")

    gp = education_df.groupby(
        ["country", "education"], as_index=as_index, observed=observed
    )
    result = gp.value_counts(normalize=normalize)

    expected_series = Series(
        data=expected_data[expected_data > 0.0] if observed else expected_data,
        index=MultiIndex.from_tuples(
            expected_index,
            names=["country", "education", "gender"],
        ),
        name=name,
    )
    for i in range(2):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            # NaN values corresponds to non-observed groups
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_non_groupers(
    education_df, as_index, observed, normalize, name, expected_data, request
):
    # [FUNCTION DOC]
    # GH#46357: non-grouper columns gender and education are categorical; non-observed categories of the non-groupers must be included regardless of observed. Parametrized on observed, normalize, as_index. Non-strict xfail for unstable ordering.
    # GH#46357 Test non-observed categories are included in the result,
    # regardless of `observed`

    request.applymarker(
        pytest.mark.xfail(
            reason=(
                "pandas default unstable sorting of duplicates"
                "issue with numpy>=1.25 with AVX instructions"
            ),
            strict=False,
        )
    )

    education_df = education_df.copy()
    education_df["gender"] = education_df["gender"].astype("category")
    education_df["education"] = education_df["education"].astype("category")

    gp = education_df.groupby("country", as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)

    expected_index = [
        ("FR", "male", "low"),
        ("FR", "male", "medium"),
        ("FR", "female", "high"),
        ("FR", "male", "high"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("US", "male", "low"),
        ("US", "female", "high"),
        ("US", "male", "medium"),
        ("US", "male", "high"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
    ]
    expected_series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(
            expected_index,
            names=["country", "gender", "education"],
        ),
        name=name,
    )
    for i in range(1, 3):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, expected_label, expected_values",
    [
        (False, "count", [1, 1, 1]),
        (True, "proportion", [0.5, 0.5, 1.0]),
    ],
)
def test_mixed_groupings(normalize, expected_label, expected_values):
    # [FUNCTION DOC]
    # Mixed groupings (a list, a column name and a function) with as_index false; value_counts with sort and normalize yields a frame with generated level_0 and level_2 columns plus A and B and a count or proportion column.
    # Test multiple groupings
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    gp = df.groupby([[4, 5, 4], "A", lambda i: 7 if i == 1 else 8], as_index=False)
    result = gp.value_counts(sort=True, normalize=normalize)
    expected = DataFrame(
        {
            "level_0": np.array([4, 4, 5], dtype=int),
            "A": [1, 1, 2],
            "level_2": [8, 8, 7],
            "B": [1, 3, 2],
            expected_label: expected_values,
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "test, columns, expected_names",
    [
        ("repeat", list("abbde"), ["a", None, "d", "b", "b", "e"]),
        ("level", [*list("abcd"), "level_1"], ["a", None, "d", "b", "c", "level_1"]),
    ],
)
def test_column_label_duplicates(test, columns, expected_names, as_index):
    # [FUNCTION DOC]
    # GH 44992: with duplicate column labels in the source frame ("repeat") or a label that clashes with the generated level_1 name ("level"), grouping by a column, an array and a column yields a MultiIndex with those names (None for the array); for as_index false the array level is named level_1 and a final count column is appended.
    # GH 44992
    # Test for duplicate input column labels and generated duplicate labels
    df = DataFrame([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], columns=columns)
    expected_data = [(1, 0, 7, 3, 5, 9), (2, 1, 8, 4, 6, 10)]
    keys = ["a", np.array([0, 1], dtype=np.int64), "d"]
    result = df.groupby(keys, as_index=as_index).value_counts()
    if as_index:
        expected = Series(
            data=(1, 1),
            index=MultiIndex.from_tuples(
                expected_data,
                names=expected_names,
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected)
    else:
        expected_data = [[*list(row), 1] for row in expected_data]
        expected_columns = list(expected_names)
        expected_columns[1] = "level_1"
        expected_columns.append("count")
        expected = DataFrame(expected_data, columns=expected_columns)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, expected_label",
    [
        (False, "count"),
        (True, "proportion"),
    ],
)
def test_result_label_duplicates(normalize, expected_label):
    # [FUNCTION DOC]
    # When the result column name (count or proportion) already exists as an input column, value_counts with as_index false must raise a value error that the column label is a duplicate of the result column.
    # Test for result column label duplicating an input column label
    gb = DataFrame([[1, 2, 3]], columns=["a", "b", expected_label]).groupby(
        "a", as_index=False
    )
    msg = f"Column label '{expected_label}' is duplicate of result column"
    with pytest.raises(ValueError, match=msg):
        gb.value_counts(normalize=normalize)


def test_ambiguous_grouping():
    # [FUNCTION DOC]
    # An array grouper whose length equals the row count must not be confused with the column key; value_counts of a one-column frame grouped by a constant array gives a MultiIndex with an unnamed level and level a, one row with count two.
    # Test that groupby is not confused by groupings length equal to row count
    df = DataFrame({"a": [1, 1]})
    gb = df.groupby(np.array([1, 1], dtype=np.int64))
    result = gb.value_counts()
    expected = Series(
        [2], index=MultiIndex.from_tuples([[1, 1]], names=[None, "a"]), name="count"
    )
    tm.assert_series_equal(result, expected)


def test_subset_overlaps_gb_key_raises():
    # [FUNCTION DOC]
    # GH 46383: subset listing the grouping column must raise a value error saying the keys in subset cannot be in the groupby column keys.
    # GH 46383
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    msg = "Keys {'c1'} in subset cannot be in the groupby column keys."
    with pytest.raises(ValueError, match=msg):
        df.groupby("c1").value_counts(subset=["c1"])


def test_subset_doesnt_exist_in_frame():
    # [FUNCTION DOC]
    # GH 46383: subset naming a column absent from the frame must raise a value error saying the keys in subset do not exist in the DataFrame.
    # GH 46383
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    msg = "Keys {'c3'} in subset do not exist in the DataFrame."
    with pytest.raises(ValueError, match=msg):
        df.groupby("c1").value_counts(subset=["c3"])


def test_subset():
    # [FUNCTION DOC]
    # GH 46383: with a duplicated index grouped by level 0, subset of c2 gives counts (1, 2) indexed by the level value and c2 (name count).
    # GH 46383
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    result = df.groupby(level=0).value_counts(subset=["c2"])
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays([[0, 1], ["x", "y"]], names=[None, "c2"]),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_subset_duplicate_columns():
    # [FUNCTION DOC]
    # GH 46383: subset naming a duplicated column c2 yields a result index with both duplicates included as separate levels named c2 and counts one and two.
    # GH 46383
    df = DataFrame(
        [["a", "x", "x"], ["b", "y", "y"], ["b", "y", "y"]],
        index=[0, 1, 1],
        columns=["c1", "c2", "c2"],
    )
    result = df.groupby(level=0).value_counts(subset=["c2"])
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays(
            [[0, 1], ["x", "y"], ["x", "y"]], names=[None, "c2", "c2"]
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("utc", [True, False])
def test_value_counts_time_grouper(utc, unit):
    # [FUNCTION DOC]
    # GH#50486: parametrized over utc and the unit fixture. Time Grouper with one day frequency on a datetime column cast to the given unit; value_counts returns count one per row in a three-level index (Datetime, Timestamp, Food) with datetime level in the specified unit.
    # GH#50486
    df = DataFrame(
        {
            "Timestamp": [
                1565083561,
                1565083561 + 86400,
                1565083561 + 86500,
                1565083561 + 86400 * 2,
                1565083561 + 86400 * 3,
                1565083561 + 86500 * 3,
                1565083561 + 86400 * 4,
            ],
            "Food": ["apple", "apple", "banana", "banana", "orange", "orange", "pear"],
        }
    ).drop([3])

    df["Datetime"] = to_datetime(df["Timestamp"], utc=utc, unit="s").dt.as_unit(unit)
    gb = df.groupby(Grouper(freq="1D", key="Datetime"))
    result = gb.value_counts()
    dates = to_datetime(
        ["2019-08-06", "2019-08-07", "2019-08-09", "2019-08-10"], utc=utc
    ).as_unit(unit)
    timestamps = df["Timestamp"].unique()
    index = MultiIndex(
        levels=[dates, timestamps, ["apple", "banana", "orange", "pear"]],
        codes=[[0, 1, 1, 2, 2, 3], range(6), [0, 0, 1, 2, 2, 3]],
        names=["Datetime", "Timestamp", "Food"],
    )
    expected = Series(1, index=index, name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_integer_columns():
    # [FUNCTION DOC]
    # GH#55627: integer column labels with as_index false and sort false; value_counts on column 3 returns a frame with integer labelled key columns plus a count column of ones.
    # GH#55627
    df = DataFrame({1: ["a", "a", "a"], 2: ["a", "a", "d"], 3: ["a", "b", "c"]})
    gp = df.groupby([1, 2], as_index=False, sort=False)
    result = gp[3].value_counts()
    expected = DataFrame(
        {1: ["a", "a", "a"], 2: ["a", "a", "d"], 3: ["a", "b", "c"], "count": 1}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("vc_sort", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts_sort(sort, vc_sort, normalize):
    # [FUNCTION DOC]
    # GH#55951: parametrized over group-level sort fixture, value_counts sort (vc_sort) and normalize. Chooses the expected row order by cases of group sort and vc_sort (sorted by group then by count desc, or first appearance etc.) and compares with the index (a, 0), named proportion or count.
    # GH#55951
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]})
    gb = df.groupby("a", sort=sort)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)

    if normalize:
        values = [2 / 3, 1 / 3, 1.0]
    else:
        values = [2, 1, 1]
    index = MultiIndex(
        levels=[[1, 2], [3, 4]], codes=[[0, 0, 1], [0, 1, 0]], names=["a", 0]
    )
    expected = Series(values, index=index, name="proportion" if normalize else "count")
    if sort and vc_sort:
        taker = [0, 1, 2]
    elif sort and not vc_sort:
        taker = [1, 0, 2]
    elif not sort and vc_sort:
        taker = [0, 2, 1]
    else:
        taker = [2, 1, 0]
    expected = expected.take(taker)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("vc_sort", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts_sort_categorical(sort, vc_sort, normalize):
    # [FUNCTION DOC]
    # GH#55951: same ordering scenario with categorical columns and observed true; the unobserved combination (2, 4) appears with zero count/proportion as the last row. Order of taker depends on sort and vc_sort.
    # GH#55951
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]}, dtype="category")
    gb = df.groupby("a", sort=sort, observed=True)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)

    if normalize:
        values = [2 / 3, 1 / 3, 1.0, 0.0]
    else:
        values = [2, 1, 1, 0]
    name = "proportion" if normalize else "count"
    expected = DataFrame(
        {
            "a": Categorical([1, 1, 2, 2]),
            0: Categorical([3, 4, 3, 4]),
            name: values,
        }
    ).set_index(["a", 0])[name]
    if sort and vc_sort:
        taker = [0, 1, 2, 3]
    elif sort and not vc_sort:
        taker = [0, 1, 2, 3]
    elif not sort and vc_sort:
        taker = [0, 2, 1, 3]
    else:
        taker = [2, 1, 0, 3]
    expected = expected.take(taker)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("groupby_sort", [True, False])
def test_value_counts_all_na(sort, dropna, groupby_sort):
    # [FUNCTION DOC]
    # GH#59989: all-NaN second column. Parametrized on groupby sort, plus sort and dropna fixtures. With dropna true the result is an empty int64 Series named count with a MultiIndex of the right names; otherwise counts are 1 and 2 (or 2 and 1) per group depending on whether the groupby is sorted and the value_counts is sorted.
    # GH#59989
    df = DataFrame({"a": [2, 1, 1], "b": np.nan})
    gb = df.groupby("a", sort=groupby_sort)
    result = gb.value_counts(sort=sort, dropna=dropna)

    kwargs = {"levels": [[1, 2], [np.nan]], "names": ["a", "b"]}
    if dropna:
        data = []
        index = MultiIndex(codes=[[], []], **kwargs)
    elif not groupby_sort and not sort:
        data = [1, 2]
        index = MultiIndex(codes=[[1, 0], [0, 0]], **kwargs)
    else:
        data = [2, 1]
        index = MultiIndex(codes=[[0, 1], [0, 0]], **kwargs)
    expected = Series(data, index=index, dtype="int64", name="count")

    tm.assert_series_equal(result, expected)
