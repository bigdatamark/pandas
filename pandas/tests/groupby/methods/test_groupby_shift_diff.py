# [MODULE DOC]
# Tests for groupby shift and diff: null keys, fill values, timezone preservation, dtype
# of diff results, datetimelike and boolean diff, object dtype errors, empty inputs,
# frequency and suffix arguments, and multiple periods.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


def test_group_shift_with_null_key():
    # [FUNCTION DOC]
    # Verifies shifting by minus one with a partially missing group key works without
    # crashing (segfault regression for issue #13813): a 1200-row float frame grouped by
    # two keys, one of which has missing values; the result takes the value twelve rows
    # ahead where the key is complete and missing elsewhere.
    # This test is designed to replicate the segfault in issue #13813.
    n_rows = 1200

    # Generate a moderately large dataframe with occasional missing
    # values in column `B`, and then group by [`A`, `B`]. This should
    # force `-1` in `labels` array of `g._grouper.group_info` exactly
    # at those places, where the group-by key is partially missing.
    df = DataFrame(
        [(i % 12, i % 3 if i % 3 else np.nan, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    expected = DataFrame(
        [(i + 12 if i % 3 and i < n_rows - 12 else np.nan) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    result = g.shift(-1)

    tm.assert_frame_equal(result, expected)


def test_group_shift_with_fill_value():
    # [FUNCTION DOC]
    # Verifies shifting by minus one with fill value zero (GH #24128) on a 24-row frame
    # grouped by two keys: values come from twelve rows ahead, and the last twelve rows
    # are filled with zero.
    # GH #24128
    n_rows = 24
    df = DataFrame(
        [(i % 12, i % 3, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    expected = DataFrame(
        [(i + 12 if i < n_rows - 12 else 0) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    result = g.shift(-1, fill_value=0)

    tm.assert_frame_equal(result, expected)


def test_group_shift_lose_timezone():
    # [FUNCTION DOC]
    # Verifies shifting by zero a tz-aware datetime column keeps timezone and value (GH
    # 30134).
    # GH 30134
    now_dt = Timestamp.now("UTC").as_unit("ns")
    df = DataFrame({"a": [1, 1], "date": now_dt})
    result = df.groupby("a").shift(0).iloc[0]
    expected = Series({"date": now_dt}, name=result.name)
    tm.assert_series_equal(result, expected)


def test_group_diff_real_series(any_real_numpy_dtype):
    # [FUNCTION DOC]
    # Verifies grouped diff on a series for each real numpy dtype yields the expected
    # differences with missing values for first group elements; result dtype is float32
    # for 8-bit and 16-bit integers and float32, else float. Uses the
    # any_real_numpy_dtype fixture.
    df = DataFrame(
        {"a": [1, 2, 3, 3, 2], "b": [1, 2, 3, 4, 5]},
        dtype=any_real_numpy_dtype,
    )
    result = df.groupby("a")["b"].diff()
    exp_dtype = "float"
    if any_real_numpy_dtype in ["int8", "int16", "float32"]:
        exp_dtype = "float32"
    expected = Series([np.nan, np.nan, np.nan, 1.0, 3.0], dtype=exp_dtype, name="b")
    tm.assert_series_equal(result, expected)


def test_group_diff_real_frame(any_real_numpy_dtype):
    # [FUNCTION DOC]
    # Verifies grouped diff on a frame with two value columns, with the same dtype
    # expectations as the series version.
    df = DataFrame(
        {
            "a": [1, 2, 3, 3, 2],
            "b": [1, 2, 3, 4, 5],
            "c": [1, 2, 3, 4, 6],
        },
        dtype=any_real_numpy_dtype,
    )
    result = df.groupby("a").diff()
    exp_dtype = "float"
    if any_real_numpy_dtype in ["int8", "int16", "float32"]:
        exp_dtype = "float32"
    expected = DataFrame(
        {
            "b": [np.nan, np.nan, np.nan, 1.0, 3.0],
            "c": [np.nan, np.nan, np.nan, 1.0, 4.0],
        },
        dtype=exp_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-02"),
            Timestamp("2013-01-03"),
        ],
        [Timedelta("5 days"), Timedelta("6 days"), Timedelta("7 days")],
    ],
)
def test_group_diff_datetimelike(data, unit):
    # [FUNCTION DOC]
    # Verifies grouped diff on datetime and timedelta columns gives timedelta
    # differences with missing values for first elements, preserving the time unit.
    # Parametrized over data type and the unit fixture.
    df = DataFrame({"a": [1, 2, 2], "b": data})
    df["b"] = df["b"].dt.as_unit(unit)
    result = df.groupby("a")["b"].diff()
    expected = Series([NaT, NaT, Timedelta("1 days")], name="b").dt.as_unit(unit)
    tm.assert_series_equal(result, expected)


def test_group_diff_bool():
    # [FUNCTION DOC]
    # Verifies grouped diff of a boolean column: the first element of each group is
    # missing and the remaining entries are false (equal to the expected series with
    # those values).
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [True, True, False, False, True]})
    result = df.groupby("a")["b"].diff()
    expected = Series([np.nan, np.nan, np.nan, False, False], name="b")
    tm.assert_series_equal(result, expected)


def test_group_diff_object_raises(object_dtype):
    # [FUNCTION DOC]
    # Verifies grouped diff on an object-dtype string column raises a type error about
    # unsupported subtraction operands. Uses the object_dtype fixture.
    df = DataFrame(
        {"a": ["foo", "bar", "bar"], "b": ["baz", "foo", "foo"]}, dtype=object_dtype
    )
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for -"):
        df.groupby("a")["b"].diff()


def test_empty_shift_with_fill():
    # [FUNCTION DOC]
    # Verifies (GH 41264) shifting an empty frame grouped by one key gives the same
    # result and index with or without a fill value.
    # GH 41264, single-index check
    df = DataFrame(columns=["a", "b", "c"])
    shifted = df.groupby(["a"]).shift(1)
    shifted_with_fill = df.groupby(["a"]).shift(1, fill_value=0)
    tm.assert_frame_equal(shifted, shifted_with_fill)
    tm.assert_index_equal(shifted.index, shifted_with_fill.index)


def test_multindex_empty_shift_with_fill():
    # [FUNCTION DOC]
    # Verifies (GH 41264) the same for grouping by two keys.
    # GH 41264, multi-index check
    df = DataFrame(columns=["a", "b", "c"])
    shifted = df.groupby(["a", "b"]).shift(1)
    shifted_with_fill = df.groupby(["a", "b"]).shift(1, fill_value=0)
    tm.assert_frame_equal(shifted, shifted_with_fill)
    tm.assert_index_equal(shifted.index, shifted_with_fill.index)


def test_shift_periods_freq():
    # [FUNCTION DOC]
    # Verifies (GH 54093) grouped shift with periods minus two and a daily frequency
    # moves the index instead of the values, giving the original data on an index
    # shifted two days earlier.
    # GH 54093
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data, index=date_range(start="20100101", periods=6))
    result = df.groupby(df.index).shift(periods=-2, freq="D")
    expected = DataFrame(data, index=date_range(start="2009-12-30", periods=6))
    tm.assert_frame_equal(result, expected)


def test_shift_disallow_freq_and_fill_value():
    # [FUNCTION DOC]
    # Verifies (GH 53832) combining a frequency with a fill value in grouped shift
    # raises a value error.
    # GH 53832
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data, index=date_range(start="20100101", periods=6))
    msg = "Passing a 'freq' together with a 'fill_value'"
    with pytest.raises(ValueError, match=msg):
        df.groupby(df.index).shift(periods=-2, freq="D", fill_value="1")


def test_shift_disallow_suffix_if_periods_is_int():
    # [FUNCTION DOC]
    # Verifies (GH#44424) passing a suffix with an integer periods raises a value error.
    # GH#44424
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data)
    msg = "Cannot specify `suffix` if `periods` is an int."
    with pytest.raises(ValueError, match=msg):
        df.groupby("b").shift(1, suffix="fails")


def test_group_shift_with_multiple_periods():
    # [FUNCTION DOC]
    # Verifies (GH#44424) shifting by a list of periods (zero and one) returns a frame
    # with suffixed columns, for both frame and series selection.
    # GH#44424
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [True, True, False, False, True]})

    shifted_df = df.groupby("b")[["a"]].shift([0, 1])
    expected_df = DataFrame(
        {"a_0": [1, 2, 3, 3, 2], "a_1": [np.nan, 1.0, np.nan, 3.0, 2.0]}
    )
    tm.assert_frame_equal(shifted_df, expected_df)

    # series
    shifted_series = df.groupby("b")["a"].shift([0, 1])
    tm.assert_frame_equal(shifted_series, expected_df)


def test_group_shift_with_multiple_periods_and_freq():
    # [FUNCTION DOC]
    # Verifies (GH#44424) shifting by several periods with an hourly frequency returns a
    # frame with suffixed columns and an extended index.
    # GH#44424
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
        index=date_range("1/1/2000", periods=5, freq="h"),
    )
    shifted_df = df.groupby("b")[["a"]].shift(
        [0, 1],
        freq="h",
    )
    expected_df = DataFrame(
        {
            "a_0": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "a_1": [
                np.nan,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        },
        index=date_range("1/1/2000", periods=6, freq="h"),
    )
    tm.assert_frame_equal(shifted_df, expected_df)


def test_group_shift_with_multiple_periods_and_fill_value():
    # [FUNCTION DOC]
    # Verifies (GH#44424) shifting by several periods with a fill value fills the
    # shifted column with minus one and keeps integer dtype.
    # GH#44424
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
    )
    shifted_df = df.groupby("b")[["a"]].shift([0, 1], fill_value=-1)
    expected_df = DataFrame(
        {"a_0": [1, 2, 3, 4, 5], "a_1": [-1, 1, -1, 3, 2]},
    )
    tm.assert_frame_equal(shifted_df, expected_df)


def test_group_shift_with_multiple_periods_and_both_fill_and_freq_deprecated():
    # [FUNCTION DOC]
    # Verifies (GH#44424) giving both fill value and frequency with a list of periods
    # raises a value error.
    # KNOWN ISSUE: the test name says deprecated, but the test expects a value error, so
    # the behavior is an error, not a deprecation.
    # GH#44424
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
        index=date_range("1/1/2000", periods=5, freq="h"),
    )
    msg = "Passing a 'freq' together with a 'fill_value'"
    with pytest.raises(ValueError, match=msg):
        df.groupby("b")[["a"]].shift([1, 2], fill_value=1, freq="h")


def test_groupby_shift_multiple_periods_unsorted_index():
    # [FUNCTION DOC]
    # Verifies (pull request 62843) shifting by multiple periods with a frequency on an
    # unsorted datetime index returns the union index (original unsorted entries
    # followed by the new end) with correct values.
    # https://github.com/pandas-dev/pandas/pull/62843
    idx = date_range("1/1/2000", periods=4, freq="h")
    df = DataFrame(
        {"a": [1, 2, 3], "b": [True, True, False]},
        index=[idx[2], idx[0], idx[1]],
    )
    result = df.groupby("b")[["a"]].shift([0, 1], freq="h")
    expected = DataFrame(
        {
            "a_0": [1.0, 2.0, 3.0, np.nan],
            "a_1": [3.0, np.nan, 2.0, 1.0],
        },
        index=[idx[2], idx[0], idx[1], idx[3]],
    )
    tm.assert_frame_equal(result, expected)
