# [MODULE DOC]
# Tests for Series.reset_index (and the frame equivalent in a few round-trip
# checks): datetime round trip, naming, drop, levels, in place, errors,
# duplicates, dtype preservation on an empty MultiIndex and the inferred
# string option. No import-time side effects.

from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    date_range,
    option_context,
)
import pandas._testing as tm


class TestResetIndex:
    # [CLASS DOC]
    # Groups the tests of Series.reset_index. Stateless.
    def test_reset_index_dti_round_trip(self):
        # [FUNCTION DOC]
        # A datetime-indexed frame survives reset_index and set_index (dtype
        # preserved); GH#2329: a datetime index becomes a datetime column and back.
        dti = date_range(
            start="1/1/2001", end="6/1/2001", freq="D", unit="ns"
        )._with_freq(None)
        d1 = DataFrame({"v": np.random.default_rng(2).random(len(dti))}, index=dti)
        d2 = d1.reset_index()
        assert d2.dtypes.iloc[0] == np.dtype("M8[ns]")
        d3 = d2.set_index("index")
        tm.assert_frame_equal(d1, d3, check_names=False)

        # GH#2329
        stamp = datetime(2012, 11, 22)
        df = DataFrame([[stamp, 12.1]], columns=["Date", "Value"])
        df = df.set_index("Date")

        assert df.index[0] == stamp
        assert df.reset_index()["Date"].iloc[0] == stamp

    def test_reset_index(self):
        # [FUNCTION DOC]
        # Stacks a frame to a Series named "value" with named levels; reset_index uses
        # the Series name for the column or a given name; the in-place drop returns None
        # and equals the non-in-place result; a level selection drops or keeps levels as
        # specified (returns a Series with the remaining index when dropping).
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )[:5]
        ser = df.stack()
        ser.index.names = ["hash", "category"]

        ser.name = "value"
        df = ser.reset_index()
        assert "value" in df

        df = ser.reset_index(name="value2")
        assert "value2" in df

        # check inplace
        s = ser.reset_index(drop=True)
        s2 = ser
        return_value = s2.reset_index(drop=True, inplace=True)
        assert return_value is None
        tm.assert_series_equal(s, s2)

        # level
        index = MultiIndex(
            levels=[["bar"], ["one", "two", "three"], [0, 1]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
        )
        s = Series(np.random.default_rng(2).standard_normal(6), index=index)
        rs = s.reset_index(level=1)
        assert len(rs.columns) == 2

        rs = s.reset_index(level=[0, 2], drop=True)
        tm.assert_index_equal(rs.index, Index(index.get_level_values(1)))
        assert isinstance(rs, Series)

    def test_reset_index_name(self):
        # [FUNCTION DOC]
        # The result of reset_index has an unnamed index, with or without drop.
        s = Series([1, 2, 3], index=Index(range(3), name="x"))
        assert s.reset_index().index.name is None
        assert s.reset_index(drop=True).index.name is None

    def test_reset_index_level(self):
        # [FUNCTION DOC]
        # Resetting one or both levels (by name or position) of a MultiIndex Series
        # gives the expected frame or Series; a missing level raises a key error; for a
        # single index too many levels raise an index error; an empty level list with
        # drop works on a plain Series.
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        for levels in ["A", "B"], [0, 1]:
            # With MultiIndex
            s = df.set_index(["A", "B"])["C"]

            result = s.reset_index(level=levels[0])
            tm.assert_frame_equal(result, df.set_index("B"))

            result = s.reset_index(level=levels[:1])
            tm.assert_frame_equal(result, df.set_index("B"))

            result = s.reset_index(level=levels)
            tm.assert_frame_equal(result, df)

            result = df.set_index(["A", "B"]).reset_index(level=levels, drop=True)
            tm.assert_frame_equal(result, df[["C"]])

            with pytest.raises(KeyError, match="Level E "):
                s.reset_index(level=["A", "E"])

            # With single-level Index
            s = df.set_index("A")["B"]

            result = s.reset_index(level=levels[0])
            tm.assert_frame_equal(result, df[["A", "B"]])

            result = s.reset_index(level=levels[:1])
            tm.assert_frame_equal(result, df[["A", "B"]])

            result = s.reset_index(level=levels[0], drop=True)
            tm.assert_series_equal(result, df["B"])

            with pytest.raises(IndexError, match="Too many levels"):
                s.reset_index(level=[0, 1, 2])

        # Check that .reset_index([],drop=True) doesn't fail
        result = Series(range(4)).reset_index([], drop=True)
        expected = Series(range(4))
        tm.assert_series_equal(result, expected)

    def test_reset_index_range(self):
        # [FUNCTION DOC]
        # GH 12071: a Series with a range index resets to a frame with a range index and
        # the columns "index" and the Series name.
        # GH 12071
        s = Series(range(2), name="A", dtype="int64")
        series_result = s.reset_index()
        assert isinstance(series_result.index, RangeIndex)
        series_expected = DataFrame(
            [[0, 0], [1, 1]], columns=["index", "A"], index=RangeIndex(stop=2)
        )
        tm.assert_frame_equal(series_result, series_expected)

    def test_reset_index_drop_errors(self):
        # [FUNCTION DOC]
        # GH 20925: resetting an unknown level name raises a key error (for a plain
        # index, that the name does not match the index name; for a MultiIndex, that
        # the level is not found).
        #  GH 20925

        # KeyError raised for series index when passed level name is missing
        s = Series(range(4))
        with pytest.raises(KeyError, match="does not match index name"):
            s.reset_index("wrong", drop=True)
        with pytest.raises(KeyError, match="does not match index name"):
            s.reset_index("wrong")

        # KeyError raised for series when level to be dropped is missing
        s = Series(range(4), index=MultiIndex.from_product([[1, 2]] * 2))
        with pytest.raises(KeyError, match="not found"):
            s.reset_index("wrong", drop=True)

    def test_reset_index_with_drop(self):
        # [FUNCTION DOC]
        # With a MultiIndex, reset_index without drop gives a frame with a column per
        # level plus the data; with drop gives a Series; the index name is kept.
        arrays = [
            ["bar", "bar", "baz", "baz", "qux", "qux", "foo", "foo"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = zip(*arrays, strict=True)
        index = MultiIndex.from_tuples(tuples)
        data = np.random.default_rng(2).standard_normal(8)
        ser = Series(data, index=index)
        ser.iloc[3] = np.nan

        deleveled = ser.reset_index()
        assert isinstance(deleveled, DataFrame)
        assert len(deleveled.columns) == len(ser.index.levels) + 1
        assert deleveled.index.name == ser.index.name

        deleveled = ser.reset_index(drop=True)
        assert isinstance(deleveled, Series)
        assert deleveled.index.name == ser.index.name

    def test_reset_index_inplace_and_drop_ignore_name(self):
        # [FUNCTION DOC]
        # GH#44575: with both drop and in place, the name argument is ignored and the
        # Series keeps its old name.
        # GH#44575
        ser = Series(range(2), name="old")
        ser.reset_index(name="new", drop=True, inplace=True)
        expected = Series(range(2), name="old")
        tm.assert_series_equal(ser, expected)

    def test_reset_index_drop_infer_string(self):
        # [FUNCTION DOC]
        # GH#56160, requires pyarrow: reset_index with drop on an object Series under
        # the inferred-string option returns an equal Series.
        # GH#56160
        pytest.importorskip("pyarrow")
        ser = Series(["a", "b", "c"], dtype=object)
        with option_context("future.infer_string", True):
            result = ser.reset_index(drop=True)
        tm.assert_series_equal(result, ser)


@pytest.mark.parametrize(
    "array, dtype",
    [
        (["a", "b"], object),
        (
            pd.period_range("12-1-2000", periods=2, freq="Q-DEC"),
            pd.PeriodDtype(freq="Q-DEC"),
        ),
    ],
)
def test_reset_index_dtypes_on_empty_series_with_multiindex(
    array, dtype, using_infer_string
):
    # [FUNCTION DOC]
    # GH 19602, parametrized over level dtype (strings or periods): resetting the
    # index of an empty Series with a three-level MultiIndex keeps the level dtypes
    # in the resulting frame columns (string or object, period) with object data
    # column.
    # GH 19602 - Preserve dtype on empty Series with MultiIndex
    idx = MultiIndex.from_product([[0, 1], [0.5, 1.0], array])
    result = Series(dtype=object, index=idx)[:0].reset_index().dtypes
    exp = "str" if using_infer_string else object
    expected = Series(
        {
            "level_0": np.int64,
            "level_1": np.float64,
            "level_2": exp if dtype == object else dtype,
            0: object,
        }
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "names, expected_names",
    [
        (["A", "A"], ["A", "A"]),
        (["level_1", None], ["level_1", "level_1"]),
    ],
)
@pytest.mark.parametrize("allow_duplicates", [False, True])
def test_column_name_duplicates(names, expected_names, allow_duplicates):
    # [FUNCTION DOC]
    # GH#44755, parametrized over duplicate-name cases and whether duplicates are
    # allowed: with allow_duplicates the frame is built with repeated column names
    # (unnamed level named after its position); without it a value error says the
    # column cannot be inserted because it already exists.
    # GH#44755 reset_index with duplicate column labels
    s = Series([1], index=MultiIndex.from_arrays([[1], [1]], names=names))
    if allow_duplicates:
        result = s.reset_index(allow_duplicates=True)
        expected = DataFrame([[1, 1, 1]], columns=[*expected_names, 0])
        tm.assert_frame_equal(result, expected)
    else:
        with pytest.raises(ValueError, match="cannot insert"):
            s.reset_index()
