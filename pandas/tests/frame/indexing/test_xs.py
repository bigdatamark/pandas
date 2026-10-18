# [MODULE DOC]
# Tests for the cross-section method of the frame class (xs): selecting rows or columns
# by label, optionally by level of a multi-level index, with or without dropping the
# selected level. Covers mixed dtypes, duplicates, error cases, copy/view behavior and
# multi-level (including four-level) indexes. Defines one fixture, uses shared fixtures
# from the frame test conftest and the testing helpers module. No import-time side
# effects.

import re

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm

from pandas.tseries.offsets import BDay


@pytest.fixture
def four_level_index_dataframe():
    # [FUNCTION DOC]
    # HELPER: fixture returning a 3-row, 5-column float frame (columns A to E) with a
    # four-level row index (levels one, two, three, four) containing mixed string, float
    # and integer labels. Used by the multi-level cross-section tests.
    arr = np.array(
        [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.364],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
            [-0.6662, -0.5243, -0.358, 0.89145, 2.5838],
        ]
    )
    index = MultiIndex(
        levels=[["a", "x"], ["b", "q"], [10.0032, 20.0, 30.0], [3, 4, 5]],
        codes=[[0, 0, 1], [0, 1, 1], [0, 1, 2], [2, 1, 0]],
        names=["one", "two", "three", "four"],
    )
    return DataFrame(arr, index=index, columns=list("ABCDE"))


class TestXS:
    # [CLASS DOC]
    # Tests of cross-section on single-level indexes; stateless pytest class.
    def test_xs(self, float_frame):
        # [FUNCTION DOC]
        # Takes a row from the float frame by label; every value (treating missing
        # values specially) equals the corresponding element looked up by column and row
        # label.
        idx = float_frame.index[5]
        xs = float_frame.xs(idx)
        for item, value in xs.items():
            if np.isnan(value):
                assert np.isnan(float_frame[item][idx])
            else:
                assert value == float_frame[item][idx]

    def test_xs_mixed(self):
        # [FUNCTION DOC]
        # For a frame with an integer-valued and a string-valued column, the row
        # cross-section is an object series with the original values.
        # mixed-type xs
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}
        frame = DataFrame(test_data)
        xs = frame.xs("1")
        assert xs.dtype == np.object_
        assert xs["A"] == 1
        assert xs["B"] == "1"

    def test_xs_dt_error(self, datetime_frame):
        # [FUNCTION DOC]
        # Looking up a timestamp not present in the datetime index (one business day
        # before the first) raises a key error whose message contains the timestamp
        # representation.
        with pytest.raises(
            KeyError, match=re.escape("Timestamp('1999-12-31 00:00:00')")
        ):
            datetime_frame.xs(datetime_frame.index[0] - BDay())

    def test_xs_other(self, float_frame):
        # [FUNCTION DOC]
        # Column cross-section equals the column; modifying the returned series must not
        # change the original frame (no view propagation under current semantics).
        float_frame_orig = float_frame.copy()
        # xs get column
        series = float_frame.xs("A", axis=1)
        expected = float_frame["A"]
        tm.assert_series_equal(series, expected)

        # view is returned if possible
        series = float_frame.xs("A", axis=1)
        series[:] = 5
        # The view shouldn't propagate mutations
        tm.assert_series_equal(float_frame["A"], float_frame_orig["A"])
        assert not (expected == 5).all()

    def test_xs_corner(self):
        # [FUNCTION DOC]
        # Row cross-section on a frame built from alternating float and string columns
        # returns an object series in column order with the row label as name; a frame
        # with rows but no columns gives an empty float series named by the row label.
        # pathological mixed-type reordering case
        df = DataFrame(index=[0], columns=Index([], dtype="str"))
        df["A"] = 1.0
        df["B"] = "foo"
        df["C"] = 2.0
        df["D"] = "bar"
        df["E"] = 3.0

        xs = df.xs(0)
        exp = Series([1.0, "foo", 2.0, "bar", 3.0], index=list("ABCDE"), name=0)
        tm.assert_series_equal(xs, exp)

        # no columns but Index(dtype=object)
        df = DataFrame(index=["a", "b", "c"])
        result = df.xs("a")
        expected = Series([], name="a", dtype=np.float64)
        tm.assert_series_equal(result, expected)

    def test_xs_duplicates(self):
        # [FUNCTION DOC]
        # Row cross-section with a unique label in an index that has duplicates
        # elsewhere returns that row as a series.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            index=["b", "b", "c", "b", "a"],
        )

        cross = df.xs("c")
        exp = df.iloc[2]
        tm.assert_series_equal(cross, exp)

    def test_xs_keep_level(self):
        # [FUNCTION DOC]
        # With a three-level row index, cross-section by level with drop_level false
        # keeps all levels in the result; works with single key/level and with key
        # tuple/level list.
        df = DataFrame(
            {
                "day": {0: "sat", 1: "sun"},
                "flavour": {0: "strawberry", 1: "strawberry"},
                "sales": {0: 10, 1: 12},
                "year": {0: 2008, 1: 2008},
            }
        ).set_index(["year", "flavour", "day"])
        result = df.xs("sat", level="day", drop_level=False)
        expected = df[:1]
        tm.assert_frame_equal(result, expected)

        result = df.xs((2008, "sat"), level=["year", "day"], drop_level=False)
        tm.assert_frame_equal(result, expected)

    def test_xs_view(self):
        # [FUNCTION DOC]
        # Chained assignment on a row cross-section must raise the chained-assignment
        # error and leave the original frame unchanged.
        # in 0.14 this will return a view if possible a copy otherwise, but
        # this is numpy dependent

        dm = DataFrame(np.arange(20.0).reshape(4, 5), index=range(4), columns=range(5))
        df_orig = dm.copy()

        with tm.raises_chained_assignment_error():
            dm.xs(2)[:] = 20
        tm.assert_frame_equal(dm, df_orig)


class TestXSWithMultiIndex:
    # [CLASS DOC]
    # Tests of cross-section on frames with multi-level row or column indexes.
    def test_xs_doc_example(self):
        # [FUNCTION DOC]
        # Based on an example from the advanced indexing user guide. Cross-section on
        # columns by a tuple of keys over levels given by name (in swapped order)
        # returns the single matching column as a frame.
        # TODO: more descriptive name
        # based on example in advanced.rst
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = list(zip(*arrays))

        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 8)),
            index=["A", "B", "C"],
            columns=index,
        )

        result = df.xs(("one", "bar"), level=("second", "first"), axis=1)

        expected = df.iloc[:, [0]]
        tm.assert_frame_equal(result, expected)

    def test_xs_integer_key(self):
        # [FUNCTION DOC]
        # Cross-section using an integer key on a named level equals label-based
        # selection of the same key.
        # see GH#2107
        dates = range(20111201, 20111205)
        ids = list("abcde")
        index = MultiIndex.from_product([dates, ids], names=["date", "secid"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 3)),
            index,
            ["X", "Y", "Z"],
        )

        result = df.xs(20111201, level="date")
        expected = df.loc[20111201, :]
        tm.assert_frame_equal(result, expected)

    def test_xs_level(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Using the multi-level random-data fixture, selecting the value two on the
        # level named second returns matching rows with the remaining level as the
        # index.
        df = multiindex_dataframe_random_data
        result = df.xs("two", level="second")
        expected = df[df.index.get_level_values(1) == "two"]
        expected.index = Index(["foo", "bar", "baz", "qux"], name="first")
        tm.assert_frame_equal(result, expected)

    def test_xs_level_eq_2(self):
        # [FUNCTION DOC]
        # Selecting on the third level by label returns the one matching row with the
        # other two levels as a new multi-level index.
        arr = np.random.default_rng(2).standard_normal((3, 5))
        index = MultiIndex(
            levels=[["a", "p", "x"], ["b", "q", "y"], ["c", "r", "z"]],
            codes=[[2, 0, 1], [2, 0, 1], [2, 0, 1]],
        )
        df = DataFrame(arr, index=index)
        expected = DataFrame(arr[1:2], index=[["a"], ["b"]])
        result = df.xs("c", level=2)
        tm.assert_frame_equal(result, expected)

    def test_xs_setting_with_copy_error(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Cross-section by level returns a copy: assigning to the result must leave the
        # original frame unchanged.
        # this is a copy in 0.14
        df = multiindex_dataframe_random_data
        df_orig = df.copy()
        result = df.xs("two", level="second")

        result[:] = 10
        tm.assert_frame_equal(df, df_orig)

    def test_xs_setting_with_copy_error_multiple(self, four_level_index_dataframe):
        # [FUNCTION DOC]
        # Same copy semantics for a cross-section using multiple keys and levels on the
        # four-level fixture.
        # this is a copy in 0.14
        df = four_level_index_dataframe
        df_orig = df.copy()
        result = df.xs(("a", 4), level=["one", "four"])

        result[:] = 10
        tm.assert_frame_equal(df, df_orig)

    @pytest.mark.parametrize("key, level", [("one", "second"), (["one"], ["second"])])
    def test_xs_with_duplicates(self, key, level, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Parametrized over (key, level) as scalars or single-element lists. On a frame
        # with non-unique multi-level index, cross-section by a level returns both
        # duplicated groups concatenated; list keys are converted to a tuple before the
        # call.
        # see GH#13719
        frame = multiindex_dataframe_random_data
        df = concat([frame] * 2)
        assert df.index.is_unique is False
        expected = concat([frame.xs("one", level="second")] * 2)

        if isinstance(key, list):
            result = df.xs(tuple(key), level=level)
        else:
            result = df.xs(key, level=level)
        tm.assert_frame_equal(result, expected)

    def test_xs_missing_values_in_index(self):
        # [FUNCTION DOC]
        # Cross-section by first level keeps missing values in the remaining level of
        # the result index (object index with a missing entry).
        # see GH#6574
        # missing values in returned index should be preserved
        acc = [
            ("a", "abcde", 1),
            ("b", "bbcde", 2),
            ("y", "yzcde", 25),
            ("z", "xbcde", 24),
            ("z", None, 26),
            ("z", "zbcde", 25),
            ("z", "ybcde", 26),
        ]
        df = DataFrame(acc, columns=["a1", "a2", "cnt"]).set_index(["a1", "a2"])
        expected = DataFrame(
            {"cnt": [24, 26, 25, 26]},
            index=Index(["xbcde", np.nan, "zbcde", "ybcde"], name="a2"),
        )

        result = df.xs("z", level="a1")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "key, level, exp_arr, exp_index",
        [
            ("a", "lvl0", lambda x: x[:, 0:2], Index(["bar", "foo"], name="lvl1")),
            ("foo", "lvl1", lambda x: x[:, 1:2], Index(["a"], name="lvl0")),
        ],
    )
    def test_xs_named_levels_axis_eq_1(self, key, level, exp_arr, exp_index):
        # [FUNCTION DOC]
        # Parametrized over (key, level, expected-array function, expected index). On a
        # frame with multi-level columns, cross-section along columns by named level
        # selects the matching columns and keeps the other level as column index.
        # see GH#2903
        arr = np.random.default_rng(2).standard_normal((4, 4))
        index = MultiIndex(
            levels=[["a", "b"], ["bar", "foo", "hello", "world"]],
            codes=[[0, 0, 1, 1], [0, 1, 2, 3]],
            names=["lvl0", "lvl1"],
        )
        df = DataFrame(arr, columns=index)
        result = df.xs(key, level=level, axis=1)
        expected = DataFrame(exp_arr(arr), columns=exp_index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer",
        [
            lambda df: df.xs(("a", 4), level=["one", "four"]),
            lambda df: df.xs("a").xs(4, level="four"),
        ],
    )
    def test_xs_level_multiple(self, indexer, four_level_index_dataframe):
        # [FUNCTION DOC]
        # Parametrized over two equivalent ways (single call with two levels, or chained
        # calls) to select a specific combination. Result must equal the single expected
        # row with a two-level index.
        df = four_level_index_dataframe
        expected_values = [[0.4473, 1.4152, 0.2834, 1.00661, 0.1744]]
        expected_index = MultiIndex(
            levels=[["q"], [20.0]], codes=[[0], [0]], names=["two", "three"]
        )
        expected = DataFrame(
            expected_values, index=expected_index, columns=list("ABCDE")
        )
        result = indexer(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer", [lambda df: df.xs("a", level=0), lambda df: df.xs("a")]
    )
    def test_xs_level0(self, indexer, four_level_index_dataframe):
        # [FUNCTION DOC]
        # Parametrized over explicit first level and implicit selection. Both return the
        # two rows for the first-level key with a three-level index.
        df = four_level_index_dataframe
        expected_values = [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.364],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
        ]
        expected_index = MultiIndex(
            levels=[["b", "q"], [10.0032, 20.0], [4, 5]],
            codes=[[0, 1], [0, 1], [1, 0]],
            names=["two", "three", "four"],
        )
        expected = DataFrame(
            expected_values, index=expected_index, columns=list("ABCDE")
        )

        result = indexer(df)
        tm.assert_frame_equal(result, expected)

    def test_xs_values(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Row cross-section with a full key tuple has values equal to the expected row
        # of the underlying array.
        df = multiindex_dataframe_random_data
        result = df.xs(("bar", "two")).values
        expected = df.values[4]
        tm.assert_almost_equal(result, expected)

    def test_xs_loc_equality(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Cross-section by full tuple equals label-based row selection with the same
        # tuple.
        df = multiindex_dataframe_random_data
        result = df.xs(("bar", "two"))
        expected = df.loc[("bar", "two")]
        tm.assert_series_equal(result, expected)

    def test_xs_IndexSlice_argument_not_implemented(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Cross-section with an IndexSlice-style key of a
        # tuple plus slice on a multi-level index whose first level contains tuples;
        # result equals label-based selection and the last two rows without the first
        # level.
        # GH#35301

        index = MultiIndex(
            levels=[[("foo", "bar", 0), ("foo", "baz", 0), ("foo", "qux", 0)], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        obj = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)
        if frame_or_series is Series:
            obj = obj[0]

        expected = obj.iloc[-2:].droplevel(0)

        result = obj.xs(IndexSlice[("foo", "qux", 0), :])
        tm.assert_equal(result, expected)

        result = obj.loc[IndexSlice[("foo", "qux", 0), :]]
        tm.assert_equal(result, expected)

    def test_xs_levels_raises(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Using a level argument on a non-multi-level
        # index raises a type error saying the index must be a multi-level index.
        obj = DataFrame({"A": [1, 2, 3]})
        if frame_or_series is Series:
            obj = obj["A"]

        msg = "Index must be a MultiIndex"
        with pytest.raises(TypeError, match=msg):
            obj.xs(0, level="as")

    def test_xs_multiindex_droplevel_false(self):
        # [FUNCTION DOC]
        # With drop_level false on columns of a multi-level column index, the selected
        # columns retain both levels and names.
        # GH#19056
        mi = MultiIndex.from_tuples(
            [("a", "x"), ("a", "y"), ("b", "x")], names=["level1", "level2"]
        )
        df = DataFrame([[1, 2, 3]], columns=mi)
        result = df.xs("a", axis=1, drop_level=False)
        expected = DataFrame(
            [[1, 2]],
            columns=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y")], names=["level1", "level2"]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_xs_droplevel_false(self):
        # [FUNCTION DOC]
        # On a flat column index, cross-section with drop_level false returns a
        # one-column frame.
        # GH#19056
        df = DataFrame([[1, 2, 3]], columns=Index(["a", "b", "c"]))
        result = df.xs("a", axis=1, drop_level=False)
        expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(result, expected)

    def test_xs_droplevel_false_view(self):
        # [FUNCTION DOC]
        # The result with drop_level false shares memory with the source for a
        # single-dtype frame, yet later in-place modification of the source must not
        # change the result (also checked for a mixed-dtype frame).
        # GH#37832
        df = DataFrame([[1, 2, 3]], columns=Index(["a", "b", "c"]))
        result = df.xs("a", axis=1, drop_level=False)
        # check that result still views the same data as df
        assert np.shares_memory(result.iloc[:, 0]._values, df.iloc[:, 0]._values)

        df.iloc[0, 0] = 2
        # The subset is never modified
        expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(result, expected)

        df = DataFrame([[1, 2.5, "a"]], columns=Index(["a", "b", "c"]))
        result = df.xs("a", axis=1, drop_level=False)
        df.iloc[0, 0] = 2
        # The subset is never modified
        expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(result, expected)

    def test_xs_list_indexer_droplevel_false(self):
        # [FUNCTION DOC]
        # A key tuple whose second element does not exist raises a key error naming the
        # missing element, with drop_level false on a three-level column index.
        # GH#41760
        mi = MultiIndex.from_tuples([("x", "m", "a"), ("x", "n", "b"), ("y", "o", "c")])
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
        with pytest.raises(KeyError, match="y"):
            df.xs(("x", "y"), drop_level=False, axis=1)
