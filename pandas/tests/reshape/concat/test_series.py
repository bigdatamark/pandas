# [MODULE DOC]
# Tests for the concatenate function on series: bool/numeric mixes, row-wise and
# column-wise concatenation, name handling and keys/names arguments, alignment
# with reindexing, and length-one reversed input. Uses the sort and
# frame_or_series fixtures from conftest.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm


class TestSeriesConcat:
    # [CLASS DOC]
    # Test collection for series concatenation. Stateless.
    @pytest.mark.parametrize("bool_dtype", [bool, "boolean"])
    @pytest.mark.parametrize("dtype", [np.int64, np.float64, "Int64", "Float64"])
    def test_concat_bool_and_numeric(self, bool_dtype, dtype):
        # [FUNCTION DOC]
        # Parametrized over bool or nullable boolean on the left and integer/float (numpy
        # or nullable) on the right. Verifies that concatenation gives object dtype where
        # booleans stay booleans and numbers are not converted to bool (GH 21108,
        # GH 45101).
        # GH#21108, GH#45101
        left = Series([True, False], dtype=bool_dtype)
        right = Series([1, 2], dtype=dtype)
        result = concat([left, right], ignore_index=True)
        expected = Series([True, False, 1, 2], dtype=object)
        assert result.iloc[0] is True
        assert type(result.iloc[2]) in [int, float]  # i.e. not bool
        tm.assert_series_equal(result, expected)

    def test_concat_series(self):
        # [FUNCTION DOC]
        # Verifies that concatenating consecutive slices of a named series restores it and
        # keeps the name, and that supplying keys builds a two-level index with the keys
        # and the original datetime index (the redundant second name assignment on the
        # source series does nothing new).
        ts = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20, unit="ns"),
            name="foo",
        )
        ts.name = "foo"

        pieces = [ts[:5], ts[5:15], ts[15:]]

        result = concat(pieces)
        tm.assert_series_equal(result, ts)
        assert result.name == ts.name

        result = concat(pieces, keys=[0, 1, 2])
        expected = ts.copy()
        exp_codes = [np.repeat([0, 1, 2], [len(x) for x in pieces]), np.arange(len(ts))]
        exp_index = MultiIndex(
            levels=[[0, 1, 2], DatetimeIndex(ts.index.to_numpy(dtype="M8[ns]"))],
            codes=exp_codes,
        )
        expected.index = exp_index
        tm.assert_series_equal(result, expected)

    def test_concat_empty_and_non_empty_series_regression(self):
        # [FUNCTION DOC]
        # Verifies that concatenating a one-element integer series with an empty object
        # series gives an object series (GH 18187).
        # GH 18187 regression test
        s1 = Series([1])
        s2 = Series([], dtype=object)

        expected = s1.astype(object)
        result = concat([s1, s2])
        tm.assert_series_equal(result, expected)

    def test_concat_series_axis1(self):
        # [FUNCTION DOC]
        # Verifies that column-wise concatenation of overlapping slices of a series equals
        # the transposed frame built from the pieces, also when keys label the columns.
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )

        pieces = [ts[:-2], ts[2:], ts[2:-2]]

        result = concat(pieces, axis=1)
        expected = DataFrame(pieces).T
        tm.assert_frame_equal(result, expected)

        result = concat(pieces, keys=["A", "B", "C"], axis=1)
        expected = DataFrame(pieces, index=["A", "B", "C"]).T
        tm.assert_frame_equal(result, expected)

    def test_concat_series_axis1_preserves_series_names(self):
        # [FUNCTION DOC]
        # Verifies that series names become column labels when concatenating along columns
        # and that an unnamed series gets position 0 as label (GH 2489).
        # preserve series names, #2489
        s = Series(np.random.default_rng(2).standard_normal(5), name="A")
        s2 = Series(np.random.default_rng(2).standard_normal(5), name="B")

        result = concat([s, s2], axis=1)
        expected = DataFrame({"A": s, "B": s2})
        tm.assert_frame_equal(result, expected)

        s2.name = None
        result = concat([s, s2], axis=1)
        tm.assert_index_equal(result.columns, Index(["A", 0], dtype="object"))

    def test_concat_series_axis1_with_reindex(self, sort):
        # [FUNCTION DOC]
        # Parametrized by sort. Verifies column-wise concatenation aligns on the union
        # index: first-appearance order, or sorted when sort is true (GH 2603).
        # must reindex, #2603
        s = Series(
            np.random.default_rng(2).standard_normal(3), index=["c", "a", "b"], name="A"
        )
        s2 = Series(
            np.random.default_rng(2).standard_normal(4),
            index=["d", "a", "b", "c"],
            name="B",
        )
        result = concat([s, s2], axis=1, sort=sort)
        expected = DataFrame({"A": s, "B": s2}, index=["c", "a", "b", "d"])
        if sort:
            expected = expected.sort_index()
        tm.assert_frame_equal(result, expected)

    def test_concat_series_axis1_names_applied(self):
        # [FUNCTION DOC]
        # Verifies that the names argument is applied to column labels when concatenating
        # along columns with keys (single name for plain keys, two names for tuple keys)
        # (GH 23490).
        # ensure names argument is not ignored on axis=1, #23490
        s = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        result = concat([s, s2], axis=1, keys=["a", "b"], names=["A"])
        expected = DataFrame(
            [[1, 4], [2, 5], [3, 6]], columns=Index(["a", "b"], name="A")
        )
        tm.assert_frame_equal(result, expected)

        result = concat([s, s2], axis=1, keys=[("a", 1), ("b", 2)], names=["A", "B"])
        expected = DataFrame(
            [[1, 4], [2, 5], [3, 6]],
            columns=MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["A", "B"]),
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_series_axis1_same_names_ignore_index(self):
        # [FUNCTION DOC]
        # Verifies that two same-named series concatenated along columns with ignored
        # index yield range-typed columns.
        dates = date_range("01-Jan-2013", "01-Jan-2014", freq="MS")[0:-1]
        s1 = Series(
            np.random.default_rng(2).standard_normal(len(dates)),
            index=dates,
            name="value",
        )
        s2 = Series(
            np.random.default_rng(2).standard_normal(len(dates)),
            index=dates,
            name="value",
        )

        result = concat([s1, s2], axis=1, ignore_index=True)
        expected = Index(range(2))

        tm.assert_index_equal(result.columns, expected, exact=True)

    @pytest.mark.parametrize("s1name", [np.int64(190), 190])
    def test_concat_series_name_npscalar_tuple(self, s1name):
        # [FUNCTION DOC]
        # Parametrized over a numpy integer name or a plain integer. Verifies row-wise
        # concatenation of series with different (scalar, tuple) names gives an unnamed
        # series (GH 21015).
        # GH21015
        s2name = (43, 0)
        s1 = Series({"a": 1, "b": 2}, name=s1name)
        s2 = Series({"c": 5, "d": 6}, name=s2name)
        result = concat([s1, s2])
        expected = Series({"a": 1, "b": 2, "c": 5, "d": 6})
        tm.assert_series_equal(result, expected)

    def test_concat_series_partial_columns_names(self):
        # [FUNCTION DOC]
        # Verifies column naming for a mix of named and unnamed series: named keeps its
        # name and unnamed get positions; keys override names; ignored index renumbers
        # (GH 10698).
        # GH10698
        named_series = Series([1, 2], name="foo")
        unnamed_series1 = Series([1, 2])
        unnamed_series2 = Series([4, 5])

        result = concat([named_series, unnamed_series1, unnamed_series2], axis=1)
        expected = DataFrame(
            {"foo": [1, 2], 0: [1, 2], 1: [4, 5]}, columns=["foo", 0, 1]
        )
        tm.assert_frame_equal(result, expected)

        result = concat(
            [named_series, unnamed_series1, unnamed_series2],
            axis=1,
            keys=["red", "blue", "yellow"],
        )
        expected = DataFrame(
            {"red": [1, 2], "blue": [1, 2], "yellow": [4, 5]},
            columns=["red", "blue", "yellow"],
        )
        tm.assert_frame_equal(result, expected)

        result = concat(
            [named_series, unnamed_series1, unnamed_series2], axis=1, ignore_index=True
        )
        expected = DataFrame({0: [1, 2], 1: [1, 2], 2: [4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_concat_series_length_one_reversed(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes the frame_or_series fixture. Verifies that concatenating a single reversed
        # length-one object returns an equal object (GH 39401).
        # GH39401
        obj = frame_or_series([100])
        result = concat([obj.iloc[::-1]])
        tm.assert_equal(result, obj)
