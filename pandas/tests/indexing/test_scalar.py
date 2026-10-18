"""test scalar indexing, including at and iat"""

# [MODULE DOC]
# Tests for scalar (single element) access and assignment through the label-based scalar accessor and the position-based scalar accessor, plus comparisons with the general label and position indexers.
# Contains a module-level helper that enumerates index combinations, test classes for general scalar access, at/iat specifics and multi-level index scalar access, and standalone tests for boxing and tuple-labelled indexes.
# Relies on the shared fixtures for frame-or-series and the positional indexer wrapper.

from datetime import (
    datetime,
    timedelta,
)
import itertools

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


def generate_indices(f, values=False):
    # [FUNCTION DOC]
    # HELPER: enumerates every index combination of an object, used by the scalar-access tests in this module.
    # Inputs: f is a frame or series; values (default false) selects whether to use the actual axis labels (true is documented as labels) or integer positions.
    # Returns a lazy iterator over the cartesian product of the axes, yielding tuples with one entry per axis (a one-element tuple for a series).
    # KNOWN ISSUE: the docstring says that when values is true the axis values are used and when false the range is used, but the code does the opposite: when values is true it replaces each axis with positions from zero up to the axis length, and when false it uses the real axis labels.
    """
    generate the indices
    if values is True , use the axis values
    is False, use the range
    """
    axes = f.axes
    if values:
        axes = (list(range(len(ax))) for ax in axes)

    return itertools.product(*axes)


class TestScalar:
    # [CLASS DOC]
    # Groups scalar get/set tests for the position-based and label-based scalar accessors on frames and series. Stateless; each test builds its own object.
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_iat_set_ints(self, dtype, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that position-based scalar assignment works for every position of a frame or series with an integer or unsigned integer index.
        # Parametrized by index dtype (signed 64-bit and unsigned 64-bit) and the frame-or-series fixture. Setup: a three-element object. For every position combination, sets the value one and checks the stored value read back from the underlying array is approximately one.
        f = frame_or_series(range(3), index=Index([0, 1, 2], dtype=dtype))
        indices = generate_indices(f, True)
        for i in indices:
            f.iat[i] = 1
            expected = f.values[i]
            tm.assert_almost_equal(expected, 1)

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcd"), dtype=object),
            date_range("20130101", periods=4),
            Index(range(0, 8, 2), dtype=np.float64),
        ],
    )
    def test_iat_set_other(self, index, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that position-based scalar assignment with a non-integer indexer raises.
        # Parametrized over object-dtype string, datetime and float indexes, and the frame-or-series fixture. Takes the first label combination (the real labels) and assigning one through the position accessor must raise a value error stating that position-based indexing can only have integer indexers.
        f = frame_or_series(range(len(index)), index=index)
        msg = "iAt based indexing can only have integer indexers"
        idx = next(generate_indices(f, False))
        with pytest.raises(ValueError, match=msg):
            f.iat[idx] = 1

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcd"), dtype=object),
            date_range("20130101", periods=4),
            Index(range(0, 8, 2), dtype=np.float64),
            Index(range(0, 8, 2), dtype=np.uint64),
            Index(range(0, 8, 2), dtype=np.int64),
        ],
    )
    def test_at_set_ints_other(self, index, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that label-based scalar assignment works for every label of several index types.
        # Parametrized over string, datetime, float, unsigned and signed integer indexes and the frame-or-series fixture. For every label combination, sets the value one with the label accessor and checks that the label indexer reads back approximately one.
        f = frame_or_series(range(len(index)), index=index)
        indices = generate_indices(f, False)
        for i in indices:
            f.at[i] = 1
            expected = f.loc[i]
            tm.assert_almost_equal(expected, 1)


class TestAtAndiAT:
    # [CLASS DOC]
    # Collection of tests for the label-based and position-based scalar accessors that do not need a shared base class.
    # at and iat tests that don't need Base class

    def test_float_index_at_iat(self):
        # [FUNCTION DOC]
        # Verifies that a series with a float index supports scalar get by label and by position: each label returns its item and each position returns the position plus one.
        ser = Series([1, 2, 3], index=[0.1, 0.2, 0.3])
        for el, item in ser.items():
            assert ser.at[el] == item
        for i in range(len(ser)):
            assert ser.iat[i] == i + 1

    def test_at_iat_coercion(self):
        # [FUNCTION DOC]
        # Verifies that scalar access by label on a series derived from a datetime-indexed frame returns the underlying value for a timestamp key (a timestamp must not be treated as a tuple key).
        # Setup: a random frame with eight daily rows; takes the first column and reads the element at the sixth date, comparing it with the stored array value.
        # as timestamp is not a tuple!
        dates = date_range("1/1/2000", periods=8)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            index=dates,
            columns=["A", "B", "C", "D"],
        )
        s = df["A"]

        result = s.at[dates[5]]
        xp = s.values[5]
        assert result == xp

    @pytest.mark.parametrize(
        "ser, expected",
        [
            [
                Series(["2014-01-01", "2014-02-02"], dtype="datetime64[ns]"),
                Timestamp("2014-02-02"),
            ],
            [
                Series(["1 days", "2 days"], dtype="timedelta64[ns]"),
                Timedelta("2 days"),
            ],
        ],
    )
    def test_iloc_iat_coercion_datelike(self, indexer_ial, ser, expected):
        # [FUNCTION DOC]
        # Verifies that position-based scalar access and integer location access box date-like values.
        # Parametrized over a datetime series and a timedelta series with the expected boxed scalar (timestamp or time delta), and the positional indexer fixture. Reads the second element and checks it equals the expected boxed scalar (issue 7729).
        # GH 7729
        # make sure we are boxing the returns
        result = indexer_ial(ser)[1]
        assert result == expected

    def test_imethods_with_dups(self):
        # [FUNCTION DOC]
        # Verifies integer-location and position-scalar access on objects with duplicate index labels (issue 6493).
        # Setup: integer series with index duplicates. Checks that integer location and scalar position return the same scalar, that out-of-range positions (positive and negative) raise an index error with the matching bounds message, that list-based integer location keeps the duplicate labels, and that on the frame form a row returns a series named by the position while scalar position access returns the element.
        # GH6493
        # iat/iloc with dups

        s = Series(range(5), index=[1, 1, 2, 2, 3], dtype="int64")
        result = s.iloc[2]
        assert result == 2
        result = s.iat[2]
        assert result == 2

        msg = "index 10 is out of bounds for axis 0 with size 5"
        with pytest.raises(IndexError, match=msg):
            s.iat[10]
        msg = "index -10 is out of bounds for axis 0 with size 5"
        with pytest.raises(IndexError, match=msg):
            s.iat[-10]

        result = s.iloc[[2, 3]]
        expected = Series([2, 3], [2, 2], dtype="int64")
        tm.assert_series_equal(result, expected)

        df = s.to_frame()
        result = df.iloc[2]
        expected = Series(2, index=[0], name=2)
        tm.assert_series_equal(result, expected)

        result = df.iat[2, 0]
        assert result == 2

    def test_frame_at_with_duplicate_axes(self):
        # [FUNCTION DOC]
        # Verifies label-based scalar access on a frame with duplicate column labels (issue 33041).
        # Setup: a random frame with two columns both named the same. Getting by row and column label returns the whole row as a series; on the transpose it returns the same series. Setting a scalar through the accessor on a row sets both duplicate columns, which is verified against a series of two floats.
        # GH#33041
        arr = np.random.default_rng(2).standard_normal(6).reshape(3, 2)
        df = DataFrame(arr, columns=["A", "A"])

        result = df.at[0, "A"]
        expected = df.iloc[0].copy()

        tm.assert_series_equal(result, expected)

        result = df.T.at["A", 0]
        tm.assert_series_equal(result, expected)

        # setter
        df.at[1, "A"] = 2
        expected = Series([2.0, 2.0], index=["A", "A"], name=1)
        tm.assert_series_equal(df.iloc[1], expected)

    def test_at_getitem_dt64tz_values(self):
        # [FUNCTION DOC]
        # Verifies that scalar access of a timezone-aware datetime column works through both the general label indexer and the label scalar accessor (issue 15822).
        # Setup: a frame with a name column and a date column localized to a Shanghai time zone. Both accessors must return the expected timezone-aware timestamp.
        # gh-15822
        df = DataFrame(
            {
                "name": ["John", "Anderson"],
                "date": [
                    Timestamp(2017, 3, 13, 13, 32, 56),
                    Timestamp(2017, 2, 16, 12, 10, 3),
                ],
            }
        )
        df["date"] = df["date"].dt.tz_localize("Asia/Shanghai")

        expected = Timestamp("2017-03-13 13:32:56+0800", tz="Asia/Shanghai")

        result = df.loc[0, "date"]
        assert result == expected

        result = df.at[0, "date"]
        assert result == expected

    def test_mixed_index_at_iat_loc_iloc_series(self):
        # [FUNCTION DOC]
        # Verifies label scalar access, position scalar access and general indexers agree on a series with a mixed string and integer index (issue 19860).
        # Also checks that a missing integer label raises a key error with a message of only that label for both label accessors.
        # GH 19860
        s = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        for el, item in s.items():
            assert s.at[el] == s.loc[el] == item
        for i in range(len(s)):
            assert s.iat[i] == s.iloc[i] == i + 1

        with pytest.raises(KeyError, match="^4$"):
            s.at[4]
        with pytest.raises(KeyError, match="^4$"):
            s.loc[4]

    def test_mixed_index_at_iat_loc_iloc_dataframe(self):
        # [FUNCTION DOC]
        # Same as the series version but for a frame with mixed string and integer column labels (issue 19860).
        # Checks all row and column combinations agree between label scalar, general label, position scalar and integer location, and that a missing integer column label raises a key error for both the scalar accessor and the general label indexer.
        # GH 19860
        df = DataFrame(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], columns=["a", "b", "c", 1, 2]
        )
        for rowIdx, row in df.iterrows():
            for el, item in row.items():
                assert df.at[rowIdx, el] == df.loc[rowIdx, el] == item

        for row in range(2):
            for i in range(5):
                assert df.iat[row, i] == df.iloc[row, i] == row * 5 + i

        with pytest.raises(KeyError, match="^3$"):
            df.at[0, 3]
        with pytest.raises(KeyError, match="^3$"):
            df.loc[0, 3]

    def test_iat_setter_incompatible_assignment(self):
        # [FUNCTION DOC]
        # Verifies that position-based scalar assignment of a missing value into a float column works (issue 23236).
        # Sets the first element of a float column to none and expects a frame whose first value is missing (nan) while the integer column is unchanged.
        # GH 23236
        result = DataFrame({"a": [0.0, 1.0], "b": [4, 5]})
        result.iat[0, 0] = None
        expected = DataFrame({"a": [None, 1], "b": [4, 5]})
        tm.assert_frame_equal(result, expected)


def test_iat_dont_wrap_object_datetimelike():
    # [FUNCTION DOC]
    # Verifies that scalar accessors do not box object-dtype date-like Python objects into pandas scalar types (issue 32809).
    # Setup: a frame with an object column of Python datetimes and an object column of Python timedeltas. All four access routes (label scalar, position scalar, label, integer location) must return the identical original Python object, an instance of the standard datetime or timedelta but not of the pandas timestamp or time delta.
    # GH#32809 .iat calls go through DataFrame._get_value, should not
    #  call maybe_box_datetimelike
    dti = date_range("2016-01-01", periods=3)
    tdi = dti - dti
    ser = Series(dti.to_pydatetime(), dtype=object)
    ser2 = Series(tdi.to_pytimedelta(), dtype=object)
    df = DataFrame({"A": ser, "B": ser2})
    assert (df.dtypes == object).all()

    for result in [df.at[0, "A"], df.iat[0, 0], df.loc[0, "A"], df.iloc[0, 0]]:
        assert result is ser[0]
        assert isinstance(result, datetime)
        assert not isinstance(result, Timestamp)

    for result in [df.at[1, "B"], df.iat[1, 1], df.loc[1, "B"], df.iloc[1, 1]]:
        assert result is ser2[1]
        assert isinstance(result, timedelta)
        assert not isinstance(result, Timedelta)


def test_at_with_tuple_index_get():
    # [FUNCTION DOC]
    # Verifies that scalar get by label works when the index holds tuples as plain labels (issue 26989), for both frame and series; the index must have a single level.
    # GH 26989
    # DataFrame.at getter works with Index of tuples
    df = DataFrame({"a": [1, 2]}, index=[(1, 2), (3, 4)])
    assert df.index.nlevels == 1
    assert df.at[(1, 2), "a"] == 1

    # Series.at getter works with Index of tuples
    series = df["a"]
    assert series.index.nlevels == 1
    assert series.at[(1, 2)] == 1


@pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
def test_at_with_tuple_index_set():
    # [FUNCTION DOC]
    # Verifies that scalar set by label works when the index holds tuples as labels (issue 26989), for both frame and series, with the index remaining single level. Ignores the future warning about setting a value on a view.
    # GH 26989
    # DataFrame.at setter works with Index of tuples
    df = DataFrame({"a": [1, 2]}, index=[(1, 2), (3, 4)])
    assert df.index.nlevels == 1
    df.at[(1, 2), "a"] = 2
    assert df.at[(1, 2), "a"] == 2

    # Series.at setter works with Index of tuples
    series = df["a"]
    assert series.index.nlevels == 1
    series.at[1, 2] = 3
    assert series.at[1, 2] == 3


class TestMultiIndexScalar:
    # [CLASS DOC]
    # Groups scalar access tests using a multi-level index.
    def test_multiindex_at_get(self):
        # [FUNCTION DOC]
        # Verifies that scalar get by label works with a two-level index for the label scalar accessor and the general label indexer, on both frame and series (issue 26989).
        # GH 26989
        # DataFrame.at and DataFrame.loc getter works with MultiIndex
        df = DataFrame({"a": [1, 2]}, index=[[1, 2], [3, 4]])
        assert df.index.nlevels == 2
        assert df.at[(1, 3), "a"] == 1
        assert df.loc[(1, 3), "a"] == 1

        # Series.at and Series.loc getter works with MultiIndex
        series = df["a"]
        assert series.index.nlevels == 2
        assert series.at[1, 3] == 1
        assert series.loc[1, 3] == 1

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_multiindex_at_set(self):
        # [FUNCTION DOC]
        # Verifies that scalar set by label works with a two-level index for the label scalar accessor and the general label indexer, on both frame and series, reading the new values back (issue 26989). Ignores the future warning about setting a value on a view.
        # GH 26989
        # DataFrame.at and DataFrame.loc setter works with MultiIndex
        df = DataFrame({"a": [1, 2]}, index=[[1, 2], [3, 4]])
        assert df.index.nlevels == 2
        df.at[(1, 3), "a"] = 3
        assert df.at[(1, 3), "a"] == 3
        df.loc[(1, 3), "a"] = 4
        assert df.loc[(1, 3), "a"] == 4

        # Series.at and Series.loc setter works with MultiIndex
        series = df["a"]
        assert series.index.nlevels == 2
        series.at[1, 3] = 5
        assert series.at[1, 3] == 5
        series.loc[1, 3] = 6
        assert series.loc[1, 3] == 6

    def test_multiindex_at_get_one_level(self):
        # [FUNCTION DOC]
        # Verifies that label scalar get on a series with a single-level multi-level index built from one list of booleans returns the matching element (issue 38053).
        # GH#38053
        s2 = Series((0, 1), index=[[False, True]])
        result = s2.at[False]
        assert result == 0
