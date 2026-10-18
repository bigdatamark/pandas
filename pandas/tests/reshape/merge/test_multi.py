# [MODULE DOC]
# Tests for merging and joining frames with multi-level indexes (multi-key joins, join on index levels, multi-index to multi-index joins).
# Defines fixtures for a string-keyed left frame, a multi-indexed right frame, and travel-style multi-indexed frames, then two test classes: one for column/multi-index merges and joins and one for joining two multi-indexed frames.
# Uses the shared join type, sort and multi-index random data fixtures; no import-time side effects.
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
    option_context,
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge


@pytest.fixture
def left():
    # [FUNCTION DOC]
    # HELPER: fixture returning a ten-row frame with two string key columns and a random normal data column (seeded generator) for multi-index join tests.
    """left dataframe (not multi-indexed) for multi-index join tests"""
    # a little relevant example with NAs
    key1 = ["bar", "bar", "bar", "foo", "foo", "baz", "baz", "qux", "qux", "snap"]
    key2 = ["two", "one", "three", "one", "two", "one", "two", "two", "three", "one"]

    data = np.random.default_rng(2).standard_normal(len(key1))
    return DataFrame({"key1": key1, "key2": key2, "data": data})


@pytest.fixture
def right(multiindex_dataframe_random_data):
    # [FUNCTION DOC]
    # HELPER: fixture returning the shared random multi-index frame after naming its index levels key1 and key2 and renaming its three columns to j_one, j_two and j_three.
    # Mutates the fixture frame it receives (index names and columns).
    """right dataframe (multi-indexed) for multi-index join tests"""
    df = multiindex_dataframe_random_data
    df.index.names = ["key1", "key2"]

    df.columns = ["j_one", "j_two", "j_three"]
    return df


@pytest.fixture
def left_multi():
    # [FUNCTION DOC]
    # HELPER: fixture returning a five-row frame indexed by origin, destination, period and trip purpose with a single trips column.
    return DataFrame(
        {
            "Origin": ["A", "A", "B", "B", "C"],
            "Destination": ["A", "B", "A", "C", "A"],
            "Period": ["AM", "AM", "IP", "AM", "OP"],
            "TripPurp": ["hbw", "nhb", "hbo", "nhb", "hbw"],
            "Trips": [1987, 3647, 2470, 4296, 4444],
        },
        columns=["Origin", "Destination", "Period", "TripPurp", "Trips"],
    ).set_index(["Origin", "Destination", "Period", "TripPurp"])


@pytest.fixture
def right_multi():
    # [FUNCTION DOC]
    # HELPER: fixture returning a seven-row frame indexed by origin, destination, period and link type with a single distance column.
    return DataFrame(
        {
            "Origin": ["A", "A", "B", "B", "C", "C", "E"],
            "Destination": ["A", "B", "A", "B", "A", "B", "F"],
            "Period": ["AM", "AM", "IP", "AM", "OP", "IP", "AM"],
            "LinkType": ["a", "b", "c", "b", "a", "b", "a"],
            "Distance": [100, 80, 90, 80, 75, 35, 55],
        },
        columns=["Origin", "Destination", "Period", "LinkType", "Distance"],
    ).set_index(["Origin", "Destination", "Period", "LinkType"])


@pytest.fixture
def on_cols_multi():
    # [FUNCTION DOC]
    # HELPER: fixture returning the three shared level names origin, destination and period used to join the multi-indexed frames.
    return ["Origin", "Destination", "Period"]


class TestMergeMulti:
    # [CLASS DOC]
    # Tests merges and joins combining a regular frame with a multi-indexed one, including null handling, ordering and the datetime-index key cases.
    def test_merge_on_multikey(self, left, right, join_type):
        # [FUNCTION DOC]
        # Joining the left frame onto the multi-indexed right frame via the two key columns must equal a column-based merge with the reset right frame, for the join type fixture, both unsorted and sorted.
        # The join result has its index reset before comparison.
        on_cols = ["key1", "key2"]
        result = left.join(right, on=on_cols, how=join_type).reset_index(drop=True)

        expected = merge(left, right.reset_index(), on=on_cols, how=join_type)

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on=on_cols, how=join_type, sort=True).reset_index(
            drop=True
        )

        expected = merge(
            left, right.reset_index(), on=on_cols, how=join_type, sort=True
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    def test_left_join_multi_index(self, sort, infer_string):
        # [FUNCTION DOC]
        # Randomized left-join check on a three-column key (letters, floats with nulls). Parametrized on the string-inference option, which is set for the duration of the test (the True case needs pyarrow).
        # Builds a left frame, a shuffled right frame whose index is the key columns, and derived fourth and fifth columns that encode the key; run twice, the second time after injecting nulls into each key column.
        with option_context("future.infer_string", infer_string):
            icols = ["1st", "2nd", "3rd"]

            def bind_cols(df):
                # [FUNCTION DOC]
                # HELPER: derives a numeric series from the three key columns of a frame: character code offsets of the first and third columns (nulls map to zero), the third scaled by one hundred, plus ten times the second column with nulls filled as zero. Returns the series.
                iord = lambda a: 0 if a != a else ord(a)
                f = lambda ts: ts.map(iord) - ord("a")
                return f(df["1st"]) + f(df["3rd"]) * 1e2 + df["2nd"].fillna(0) * 10

            def run_asserts(left, right, sort):
                # [FUNCTION DOC]
                # HELPER: checks a left join of the left and right frames on the key list with the given sort flag.
                # Asserts: result is not shorter than the left frame; fourth and fifth columns have no nulls; the fourth equals the negated fifth and equals the recomputed key encoding; the result name is unset; when sorted, the result equals itself sorted stably by the keys; and the equivalent column-based merge (with the right index reset), after resetting the join result to a default index, equals the join result.
                res = left.join(right, on=icols, how="left", sort=sort)

                assert len(left) < len(res) + 1
                assert not res["4th"].isna().any()
                assert not res["5th"].isna().any()

                tm.assert_series_equal(res["4th"], -res["5th"], check_names=False)
                result = bind_cols(res.iloc[:, :-2])
                tm.assert_series_equal(res["4th"], result, check_names=False)
                assert result.name is None

                if sort:
                    tm.assert_frame_equal(res, res.sort_values(icols, kind="mergesort"))

                out = merge(left, right.reset_index(), on=icols, sort=sort, how="left")

                res.index = RangeIndex(len(res))
                tm.assert_frame_equal(out, res)

            lc = list(map(chr, np.arange(ord("a"), ord("z") + 1)))
            left = DataFrame(
                np.random.default_rng(2).choice(lc, (50, 2)), columns=["1st", "3rd"]
            )
            # Explicit cast to float to avoid implicit cast when setting nan
            left.insert(
                1,
                "2nd",
                np.random.default_rng(2).integers(0, 10, len(left)).astype("float"),
            )
            right = left.sample(frac=1, random_state=np.random.default_rng(2))

            left["4th"] = bind_cols(left)
            right["5th"] = -bind_cols(right)
            right.set_index(icols, inplace=True)

            run_asserts(left, right, sort)

            # inject some nulls
            left.loc[1::4, "1st"] = np.nan
            left.loc[2::5, "2nd"] = np.nan
            left.loc[3::6, "3rd"] = np.nan
            left["4th"] = bind_cols(left)

            i = np.random.default_rng(2).permutation(len(left))
            right = left.iloc[i, :-1]
            right["5th"] = -bind_cols(right)
            right.set_index(icols, inplace=True)

            run_asserts(left, right, sort)

    def test_merge_right_vs_left(self, left, right, sort):
        # compare left vs right merge with multikey
        # [FUNCTION DOC]
        # A left merge using left key columns and right index must give the same frame as the mirrored right merge with swapped inputs, once columns are reordered, for each value of the sort fixture.
        on_cols = ["key1", "key2"]
        merged_left_right = left.merge(
            right, left_on=on_cols, right_index=True, how="left", sort=sort
        )

        merge_right_left = right.merge(
            left, right_on=on_cols, left_index=True, how="right", sort=sort
        )

        # Reorder columns
        merge_right_left = merge_right_left[merged_left_right.columns]

        tm.assert_frame_equal(merged_left_right, merge_right_left)

    def test_merge_multiple_cols_with_mixed_cols_index(self):
        # GH29522
        # [FUNCTION DOC]
        # GH29522. Merging a frame with a reset multi-indexed series on two key columns gives the frame columns plus the series values column.
        s = Series(
            range(6),
            MultiIndex.from_product([["A", "B"], [1, 2, 3]], names=["lev1", "lev2"]),
            name="Amount",
        )
        df = DataFrame({"lev1": list("AAABBB"), "lev2": [1, 2, 3, 1, 2, 3], "col": 0})
        result = merge(df, s.reset_index(), on=["lev1", "lev2"])
        expected = DataFrame(
            {
                "lev1": list("AAABBB"),
                "lev2": [1, 2, 3, 1, 2, 3],
                "col": [0] * 6,
                "Amount": range(6),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_compress_group_combinations(self):
        # ~ 40000000 possible unique groups
        # [FUNCTION DOC]
        # Outer merge on two string key columns of ten thousand distinct values each (roughly forty million possible key pairs) exercising the group-label compression code path. Only checks that it completes without error.
        key1 = [str(i) for i in range(10000)]
        key1 = np.tile(key1, 2)
        key2 = key1[::-1]

        df = DataFrame(
            {
                "key1": key1,
                "key2": key2,
                "value1": np.random.default_rng(2).standard_normal(20000),
            }
        )

        df2 = DataFrame(
            {
                "key1": key1[::2],
                "key2": key2[::2],
                "value2": np.random.default_rng(2).standard_normal(10000),
            }
        )

        # just to hit the label compression code path
        merge(df, df2, how="outer")

    def test_left_join_index_preserve_order(self):
        # [FUNCTION DOC]
        # Left join of a frame on two key columns onto a two-level multi-indexed frame preserves left row order and fills unmatched rows with nulls; with sort, the result equals the unsorted result stably sorted by the keys.
        # Repeated with a left frame containing mixed numeric dtypes (multiple blocks).
        on_cols = ["k1", "k2"]
        left = DataFrame(
            {
                "k1": [0, 1, 2] * 8,
                "k2": ["foo", "bar"] * 12,
                "v": np.array(np.arange(24), dtype=np.int64),
            }
        )

        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": [5, 7]}, index=index)

        result = left.join(right, on=on_cols)

        expected = left.copy()
        expected["v2"] = np.nan
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        tm.assert_frame_equal(result, expected)

        result.sort_values(on_cols, kind="mergesort", inplace=True)
        expected = left.join(right, on=on_cols, sort=True)

        tm.assert_frame_equal(result, expected)

        # test join with multi dtypes blocks
        left = DataFrame(
            {
                "k1": [0, 1, 2] * 8,
                "k2": ["foo", "bar"] * 12,
                "k3": np.array([0, 1, 2] * 8, dtype=np.float32),
                "v": np.array(np.arange(24), dtype=np.int32),
            }
        )

        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": [5, 7]}, index=index)

        result = left.join(right, on=on_cols)

        expected = left.copy()
        expected["v2"] = np.nan
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        tm.assert_frame_equal(result, expected)

        result = result.sort_values(on_cols, kind="mergesort")
        expected = left.join(right, on=on_cols, sort=True)

        tm.assert_frame_equal(result, expected)

    def test_left_join_index_multi_match_multiindex(self):
        # [FUNCTION DOC]
        # Left join on three columns against a three-level multi-index with duplicated keys: each left row is repeated per match, unmatched rows get null, left order and index labels are kept; with sort it equals the expected result stably sorted by the key columns.
        left = DataFrame(
            [
                ["X", "Y", "C", "a"],
                ["W", "Y", "C", "e"],
                ["V", "Q", "A", "h"],
                ["V", "R", "D", "i"],
                ["X", "Y", "D", "b"],
                ["X", "Y", "A", "c"],
                ["W", "Q", "B", "f"],
                ["W", "R", "C", "g"],
                ["V", "Y", "C", "j"],
                ["X", "Y", "B", "d"],
            ],
            columns=["cola", "colb", "colc", "tag"],
            index=[3, 2, 0, 1, 7, 6, 4, 5, 9, 8],
        )

        right = DataFrame(
            [
                ["W", "R", "C", 0],
                ["W", "Q", "B", 3],
                ["W", "Q", "B", 8],
                ["X", "Y", "A", 1],
                ["X", "Y", "A", 4],
                ["X", "Y", "B", 5],
                ["X", "Y", "C", 6],
                ["X", "Y", "C", 9],
                ["X", "Q", "C", -6],
                ["X", "R", "C", -9],
                ["V", "Y", "C", 7],
                ["V", "R", "D", 2],
                ["V", "R", "D", -1],
                ["V", "Q", "A", -3],
            ],
            columns=["col1", "col2", "col3", "val"],
        ).set_index(["col1", "col2", "col3"])

        result = left.join(right, on=["cola", "colb", "colc"], how="left")

        expected = DataFrame(
            [
                ["X", "Y", "C", "a", 6],
                ["X", "Y", "C", "a", 9],
                ["W", "Y", "C", "e", np.nan],
                ["V", "Q", "A", "h", -3],
                ["V", "R", "D", "i", 2],
                ["V", "R", "D", "i", -1],
                ["X", "Y", "D", "b", np.nan],
                ["X", "Y", "A", "c", 1],
                ["X", "Y", "A", "c", 4],
                ["W", "Q", "B", "f", 3],
                ["W", "Q", "B", "f", 8],
                ["W", "R", "C", "g", 0],
                ["V", "Y", "C", "j", 7],
                ["X", "Y", "B", "d", 5],
            ],
            columns=["cola", "colb", "colc", "tag", "val"],
            index=[3, 3, 2, 0, 1, 1, 7, 6, 6, 4, 4, 5, 9, 8],
        )

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on=["cola", "colb", "colc"], how="left", sort=True)

        expected = expected.sort_values(["cola", "colb", "colc"], kind="mergesort")

        tm.assert_frame_equal(result, expected)

    def test_left_join_index_multi_match(self):
        # [FUNCTION DOC]
        # Left join on a single column against a non-unique index: left rows repeat per match with the original left index labels, unmatched left rows get null; sort gives the stably key-sorted result.
        # GH7331: the equivalent left merge with the reset right frame keeps left order (result has a default index).
        left = DataFrame(
            [["c", 0], ["b", 1], ["a", 2], ["b", 3]],
            columns=["tag", "val"],
            index=[2, 0, 1, 3],
        )

        right = DataFrame(
            [
                ["a", "v"],
                ["c", "w"],
                ["c", "x"],
                ["d", "y"],
                ["a", "z"],
                ["c", "r"],
                ["e", "q"],
                ["c", "s"],
            ],
            columns=["tag", "char"],
        ).set_index("tag")

        result = left.join(right, on="tag", how="left")

        expected = DataFrame(
            [
                ["c", 0, "w"],
                ["c", 0, "x"],
                ["c", 0, "r"],
                ["c", 0, "s"],
                ["b", 1, np.nan],
                ["a", 2, "v"],
                ["a", 2, "z"],
                ["b", 3, np.nan],
            ],
            columns=["tag", "val", "char"],
            index=[2, 2, 2, 2, 0, 1, 1, 3],
        )

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on="tag", how="left", sort=True)
        expected2 = expected.sort_values("tag", kind="mergesort")

        tm.assert_frame_equal(result, expected2)

        # GH7331 - maintain left frame order in left merge
        result = merge(left, right.reset_index(), how="left", on="tag")
        expected.index = RangeIndex(len(expected))
        tm.assert_frame_equal(result, expected)

    def test_left_merge_na_buglet(self):
        # [FUNCTION DOC]
        # Left merge on an id column where the right frame has null ids and null values: null keys must not match, so the result equals joining the left frame with the right frame minus its id column.
        left = DataFrame(
            {
                "id": list("abcde"),
                "v1": np.random.default_rng(2).standard_normal(5),
                "v2": np.random.default_rng(2).standard_normal(5),
                "dummy": list("abcde"),
                "v3": np.random.default_rng(2).standard_normal(5),
            },
            columns=["id", "v1", "v2", "dummy", "v3"],
        )
        right = DataFrame(
            {
                "id": ["a", "b", np.nan, np.nan, np.nan],
                "sv3": [1.234, 5.678, np.nan, np.nan, np.nan],
            }
        )

        result = merge(left, right, on="id", how="left")

        rdf = right.drop(["id"], axis=1)
        expected = left.join(rdf)
        tm.assert_frame_equal(result, expected)

    def test_merge_na_keys(self):
        # [FUNCTION DOC]
        # Outer merge on all shared columns (year, panel, data) where the data column contains nulls: nulls in key columns must match each other. Expected is computed by filling nulls with a sentinel, merging, then replacing the sentinel with null.
        data = [
            [1950, "A", 1.5],
            [1950, "B", 1.5],
            [1955, "B", 1.5],
            [1960, "B", np.nan],
            [1970, "B", 4.0],
            [1950, "C", 4.0],
            [1960, "C", np.nan],
            [1965, "C", 3.0],
            [1970, "C", 4.0],
        ]

        frame = DataFrame(data, columns=["year", "panel", "data"])

        other_data = [
            [1960, "A", np.nan],
            [1970, "A", np.nan],
            [1955, "A", np.nan],
            [1965, "A", np.nan],
            [1965, "B", np.nan],
            [1955, "C", np.nan],
        ]
        other = DataFrame(other_data, columns=["year", "panel", "data"])

        result = frame.merge(other, how="outer")

        expected = frame.fillna(-999).merge(other.fillna(-999), how="outer")
        expected = expected.replace(-999, np.nan)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("klass", [None, np.asarray, Series, Index])
    def test_merge_datetime_index(self, klass):
        # see gh-19038
        # [FUNCTION DOC]
        # gh-19038. Merging a frame with itself on a column plus a vector of index years, with the vector passed as raw array, series or index (or unchanged). The result has an extra unnamed key column (named key_1) of 32-bit integer years.
        # Also merging on a list holding only the year vector yields key_0 plus suffixed columns.
        df = DataFrame(
            [1, 2, 3], ["2016-01-01", "2017-01-01", "2018-01-01"], columns=["a"]
        )
        df.index = pd.to_datetime(df.index)
        on_vector = df.index.year

        if klass is not None:
            on_vector = klass(on_vector)

        exp_years = np.array([2016, 2017, 2018], dtype=np.int32)
        expected = DataFrame({"a": [1, 2, 3], "key_1": exp_years})

        result = df.merge(df, on=["a", on_vector], how="inner")
        tm.assert_frame_equal(result, expected)

        expected = DataFrame({"key_0": exp_years, "a_x": [1, 2, 3], "a_y": [1, 2, 3]})

        result = df.merge(df, on=[df.index.year], how="inner")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("merge_type", ["left", "right"])
    def test_merge_datetime_multi_index_empty_df(self, merge_type):
        # see gh-36895

        # [FUNCTION DOC]
        # gh-36895. Parametrized on left or right merge type. Merging a two-row frame indexed by (date, panel) with an empty frame that has the same index names and one column state: the result keeps the two rows with null state. Checks both the merge function on the index names and the join method, which must agree with the expected frame.
        left = DataFrame(
            data={
                "data": [1.5, 1.5],
            },
            index=MultiIndex.from_tuples(
                [[Timestamp("1950-01-01"), "A"], [Timestamp("1950-01-02"), "B"]],
                names=["date", "panel"],
            ),
        )

        right = DataFrame(
            index=MultiIndex.from_tuples([], names=["date", "panel"]), columns=["state"]
        )

        expected_index = MultiIndex.from_tuples(
            [[Timestamp("1950-01-01"), "A"], [Timestamp("1950-01-02"), "B"]],
            names=["date", "panel"],
        )

        if merge_type == "left":
            expected = DataFrame(
                data={
                    "data": [1.5, 1.5],
                    "state": np.array([np.nan, np.nan], dtype=object),
                },
                index=expected_index,
            )
            results_merge = left.merge(right, how="left", on=["date", "panel"])
            results_join = left.join(right, how="left")
        else:
            expected = DataFrame(
                data={
                    "state": np.array([np.nan, np.nan], dtype=object),
                    "data": [1.5, 1.5],
                },
                index=expected_index,
            )
            results_merge = right.merge(left, how="right", on=["date", "panel"])
            results_join = right.join(left, how="right")

        tm.assert_frame_equal(results_merge, expected)
        tm.assert_frame_equal(results_join, expected)

    @pytest.fixture
    def household(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a three-row frame indexed by household id with male flag and wealth columns.
        household = DataFrame(
            {
                "household_id": [1, 2, 3],
                "male": [0, 1, 0],
                "wealth": [196087.3, 316478.7, 294750],
            },
            columns=["household_id", "male", "wealth"],
        ).set_index("household_id")
        return household

    @pytest.fixture
    def portfolio(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a seven-row frame indexed by (household id, asset id) with name and share columns; the last row has null asset id and name.
        portfolio = DataFrame(
            {
                "household_id": [1, 2, 2, 3, 3, 3, 4],
                "asset_id": [
                    "nl0000301109",
                    "nl0000289783",
                    "gb00b03mlx29",
                    "gb00b03mlx29",
                    "lu0197800237",
                    "nl0000289965",
                    np.nan,
                ],
                "name": [
                    "ABN Amro",
                    "Robeco",
                    "Royal Dutch Shell",
                    "Royal Dutch Shell",
                    "AAB Eastern Europe Equity Fund",
                    "Postbank BioTech Fonds",
                    np.nan,
                ],
                "share": [1.0, 0.4, 0.6, 0.15, 0.6, 0.25, 1.0],
            },
            columns=["household_id", "asset_id", "name", "share"],
        ).set_index(["household_id", "asset_id"])
        return portfolio

    @pytest.fixture
    def expected(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the expected inner join of household and portfolio: six rows indexed by (household id, asset id) with columns male, wealth, name and share.
        expected = (
            DataFrame(
                {
                    "male": [0, 1, 1, 0, 0, 0],
                    "wealth": [
                        196087.3,
                        316478.7,
                        316478.7,
                        294750.0,
                        294750.0,
                        294750.0,
                    ],
                    "name": [
                        "ABN Amro",
                        "Robeco",
                        "Royal Dutch Shell",
                        "Royal Dutch Shell",
                        "AAB Eastern Europe Equity Fund",
                        "Postbank BioTech Fonds",
                    ],
                    "share": [1.00, 0.40, 0.60, 0.15, 0.60, 0.25],
                    "household_id": [1, 2, 2, 3, 3, 3],
                    "asset_id": [
                        "nl0000301109",
                        "nl0000289783",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "lu0197800237",
                        "nl0000289965",
                    ],
                }
            )
            .set_index(["household_id", "asset_id"])
            .reindex(columns=["male", "wealth", "name", "share"])
        )
        return expected

    def test_join_multi_levels(self, portfolio, household, expected):
        # [FUNCTION DOC]
        # GH 3662. Inner joining the single-level household frame with the two-level portfolio frame (on the shared household level) equals the expected fixture. Inputs are copied first.
        portfolio = portfolio.copy()
        household = household.copy()

        # GH 3662
        # merge multi-levels
        result = household.join(portfolio, how="inner")
        tm.assert_frame_equal(result, expected)

    def test_join_multi_levels_merge_equivalence(self, portfolio, household, expected):
        # [FUNCTION DOC]
        # The equivalent column-based inner merge on household id, with the result indexed by household id and asset id, equals the expected fixture.
        portfolio = portfolio.copy()
        household = household.copy()

        # equivalency
        result = merge(
            household.reset_index(),
            portfolio.reset_index(),
            on=["household_id"],
            how="inner",
        ).set_index(["household_id", "asset_id"])
        tm.assert_frame_equal(result, expected)

    def test_join_multi_levels_outer(self, portfolio, household, expected):
        # [FUNCTION DOC]
        # Outer join adds the unmatched household 4 row (null asset id, share one) to the expected fixture; expected is built by concatenating with sorting and reindexing columns. Index type check is relaxed.
        portfolio = portfolio.copy()
        household = household.copy()

        result = household.join(portfolio, how="outer")
        expected = concat(
            [
                expected,
                (
                    DataFrame(
                        {"share": [1.00]},
                        index=MultiIndex.from_tuples(
                            [(4, np.nan)], names=["household_id", "asset_id"]
                        ),
                    )
                ),
            ],
            axis=0,
            sort=True,
        ).reindex(columns=expected.columns)
        tm.assert_frame_equal(result, expected, check_index_type=False)

    def test_join_multi_levels_invalid(self, portfolio, household):
        # [FUNCTION DOC]
        # Joining after renaming the household index to a name that overlaps nothing raises a value error saying there are no overlapping index names.
        # A second case joins the portfolio with a copy of itself and expects a value error about overlapping columns with no suffix.
        # KNOWN ISSUE: the set-names call on the copied portfolio index returns a new index and its result is discarded, so the intended rename never happens; the second error is raised only because the frames are identical.
        portfolio = portfolio.copy()
        household = household.copy()

        # invalid cases
        household.index.name = "foo"

        with pytest.raises(
            ValueError, match="cannot join with no overlapping index names"
        ):
            household.join(portfolio, how="inner")

        portfolio2 = portfolio.copy()
        portfolio2.index.set_names(["household_id", "foo"])

        with pytest.raises(ValueError, match="columns overlap but no suffix specified"):
            portfolio2.join(portfolio, how="inner")

    def test_join_multi_levels2(self):
        # some more advanced merges
        # GH6360
        # [FUNCTION DOC]
        # GH6360. Joins a household frame indexed by (household id, asset id) with a log return frame indexed by (asset id, time). Verifies the inner and outer column-based merges on asset id, indexed by household id, asset id and time, equal explicitly constructed expected frames (outer includes unmatched assets with null time and return).
        household = DataFrame(
            {
                "household_id": [1, 2, 2, 3, 3, 3, 4],
                "asset_id": [
                    "nl0000301109",
                    "nl0000301109",
                    "gb00b03mlx29",
                    "gb00b03mlx29",
                    "lu0197800237",
                    "nl0000289965",
                    np.nan,
                ],
                "share": [1.0, 0.4, 0.6, 0.15, 0.6, 0.25, 1.0],
            },
            columns=["household_id", "asset_id", "share"],
        ).set_index(["household_id", "asset_id"])

        log_return = DataFrame(
            {
                "asset_id": [
                    "gb00b03mlx29",
                    "gb00b03mlx29",
                    "gb00b03mlx29",
                    "lu0197800237",
                    "lu0197800237",
                ],
                "t": [233, 234, 235, 180, 181],
                "log_return": [
                    0.09604978,
                    -0.06524096,
                    0.03532373,
                    0.03025441,
                    0.036997,
                ],
            }
        ).set_index(["asset_id", "t"])

        expected = (
            DataFrame(
                {
                    "household_id": [2, 2, 2, 3, 3, 3, 3, 3],
                    "asset_id": [
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "lu0197800237",
                        "lu0197800237",
                    ],
                    "t": [233, 234, 235, 233, 234, 235, 180, 181],
                    "share": [0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.6, 0.6],
                    "log_return": [
                        0.09604978,
                        -0.06524096,
                        0.03532373,
                        0.09604978,
                        -0.06524096,
                        0.03532373,
                        0.03025441,
                        0.036997,
                    ],
                }
            )
            .set_index(["household_id", "asset_id", "t"])
            .reindex(columns=["share", "log_return"])
        )

        # this is the equivalency
        result = merge(
            household.reset_index(),
            log_return.reset_index(),
            on=["asset_id"],
            how="inner",
        ).set_index(["household_id", "asset_id", "t"])
        tm.assert_frame_equal(result, expected)

        expected = (
            DataFrame(
                {
                    "household_id": [2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 2, 4],
                    "asset_id": [
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "lu0197800237",
                        "lu0197800237",
                        "nl0000289965",
                        "nl0000301109",
                        "nl0000301109",
                        None,
                    ],
                    "t": [
                        233,
                        234,
                        235,
                        233,
                        234,
                        235,
                        180,
                        181,
                        None,
                        None,
                        None,
                        None,
                    ],
                    "share": [
                        0.6,
                        0.6,
                        0.6,
                        0.15,
                        0.15,
                        0.15,
                        0.6,
                        0.6,
                        0.25,
                        1.0,
                        0.4,
                        1.0,
                    ],
                    "log_return": [
                        0.09604978,
                        -0.06524096,
                        0.03532373,
                        0.09604978,
                        -0.06524096,
                        0.03532373,
                        0.03025441,
                        0.036997,
                        None,
                        None,
                        None,
                        None,
                    ],
                }
            )
            .set_index(["household_id", "asset_id", "t"])
            .reindex(columns=["share", "log_return"])
        )

        result = merge(
            household.reset_index(),
            log_return.reset_index(),
            on=["asset_id"],
            how="outer",
        ).set_index(["household_id", "asset_id", "t"])

        tm.assert_frame_equal(result, expected)


class TestJoinMultiMulti:
    # [CLASS DOC]
    # Tests joining two multi-indexed frames on their shared level names.
    def test_join_multi_multi(self, left_multi, right_multi, join_type, on_cols_multi):
        # [FUNCTION DOC]
        # Joining the two multi-indexed fixtures for each join type must equal the column-based merge on the shared levels, re-indexed with level order (right levels first for right joins, otherwise left levels first) and sorted by index.
        left_names = left_multi.index.names
        right_names = right_multi.index.names
        if join_type == "right":
            level_order = right_names + left_names.difference(right_names)
        else:
            level_order = left_names + right_names.difference(left_names)
        # Multi-index join tests
        expected = (
            merge(
                left_multi.reset_index(),
                right_multi.reset_index(),
                how=join_type,
                on=on_cols_multi,
            )
            .set_index(level_order)
            .sort_index()
        )

        result = left_multi.join(right_multi, how=join_type).sort_index()
        tm.assert_frame_equal(result, expected)

    def test_join_multi_empty_frames(
        self, left_multi, right_multi, join_type, on_cols_multi
    ):
        # [FUNCTION DOC]
        # Same as the multi-multi join test but with all value columns dropped from both frames, so only the indexes are joined; result must equal the sorted column-based merge.
        left_multi = left_multi.drop(columns=left_multi.columns)
        right_multi = right_multi.drop(columns=right_multi.columns)

        left_names = left_multi.index.names
        right_names = right_multi.index.names
        if join_type == "right":
            level_order = right_names + left_names.difference(right_names)
        else:
            level_order = left_names + right_names.difference(left_names)

        expected = (
            merge(
                left_multi.reset_index(),
                right_multi.reset_index(),
                how=join_type,
                on=on_cols_multi,
            )
            .set_index(level_order)
            .sort_index()
        )

        result = left_multi.join(right_multi, how=join_type).sort_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("box", [None, np.asarray, Series, Index])
    def test_merge_datetime_index(self, box):
        # see gh-19038
        # [FUNCTION DOC]
        # gh-19038. Same scenario as the datetime index test in the other class (the test body is a duplicate; parametrized on how the year vector is boxed: none, array, series, index): merging with the year vector yields a key_1 column, and merging on only the vector yields key_0 and suffixed columns.
        df = DataFrame(
            [1, 2, 3], ["2016-01-01", "2017-01-01", "2018-01-01"], columns=["a"]
        )
        df.index = pd.to_datetime(df.index)
        on_vector = df.index.year

        if box is not None:
            on_vector = box(on_vector)

        exp_years = np.array([2016, 2017, 2018], dtype=np.int32)
        expected = DataFrame({"a": [1, 2, 3], "key_1": exp_years})

        result = df.merge(df, on=["a", on_vector], how="inner")
        tm.assert_frame_equal(result, expected)

        expected = DataFrame({"key_0": exp_years, "a_x": [1, 2, 3], "a_y": [1, 2, 3]})

        result = df.merge(df, on=[df.index.year], how="inner")
        tm.assert_frame_equal(result, expected)

    def test_single_common_level(self):
        # [FUNCTION DOC]
        # Joining two multi-indexed frames sharing exactly one level name (key) gives an inner-style result indexed by key, the left-only level and the right-only level, equal to the column-based inner merge on key.
        index_left = MultiIndex.from_tuples(
            [("K0", "X0"), ("K0", "X1"), ("K1", "X2")], names=["key", "X"]
        )

        left = DataFrame(
            {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=index_left
        )

        index_right = MultiIndex.from_tuples(
            [("K0", "Y0"), ("K1", "Y1"), ("K2", "Y2"), ("K2", "Y3")], names=["key", "Y"]
        )

        right = DataFrame(
            {"C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]},
            index=index_right,
        )

        result = left.join(right)
        expected = merge(
            left.reset_index(), right.reset_index(), on=["key"], how="inner"
        ).set_index(["key", "X", "Y"])

        tm.assert_frame_equal(result, expected)

    def test_join_multi_wrong_order(self):
        # GH 25760
        # GH 28956

        # [FUNCTION DOC]
        # GH 25760 and GH 28956. Joining where the right multi-index has the same level names in a different order: levels are matched by name, so the right values align to the left index rows with null where there is no match.
        midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])
        midx3 = MultiIndex.from_tuples([(4, 1), (3, 2), (3, 1)], names=["b", "a"])

        left = DataFrame(index=midx1, data={"x": [10, 20, 30, 40]})
        right = DataFrame(index=midx3, data={"y": ["foo", "bar", "fing"]})

        result = left.join(right)

        expected = DataFrame(
            index=midx1,
            data={"x": [10, 20, 30, 40], "y": ["fing", "foo", "bar", np.nan]},
        )

        tm.assert_frame_equal(result, expected)
