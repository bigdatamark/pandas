# [MODULE DOC]
# Tests for the values attribute of DataFrame (and the private values property): dtype resolution when
# interleaving columns, datetimelike and timezone handling, duplicate columns and read-only behavior.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestDataFrameValues:
    # [CLASS DOC]
    # Test container for the public values attribute of DataFrame.
    def test_values(self, float_frame):
        # [FUNCTION DOC]
        # The array returned by values is read-only: writing to a column of it raises a value error mentioning read-only,
        # and the frame's data remain unchanged.
        with pytest.raises(ValueError, match="read-only"):
            float_frame.values[:, 0] = 5.0
        assert (float_frame.values[:, 0] != 5).all()

    def test_more_values(self, float_string_frame):
        # [FUNCTION DOC]
        # For the float and string fixture frame, the number of columns of the values array equals the number of frame columns.
        values = float_string_frame.values
        assert values.shape[1] == len(float_string_frame.columns)

    def test_values_mixed_dtypes(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # Checks that every element of the values array matches the corresponding frame cell (NaN matching NaN). For a string and float
        # column pair the first element is the string; a complex and integer frame preserves the complex value.
        frame = float_frame
        arr = frame.values

        frame_cols = frame.columns
        for i, row in enumerate(arr):
            for j, value in enumerate(row):
                col = frame_cols[j]
                if np.isnan(value):
                    assert np.isnan(frame[col].iloc[i])
                else:
                    assert value == frame[col].iloc[i]

        # mixed type
        arr = float_string_frame[["foo", "A"]].values
        assert arr[0, 0] == "bar"

        df = DataFrame({"complex": [1j, 2j, 3j], "real": [1, 2, 3]})
        arr = df.values
        assert arr[0, 0] == 1j

    def test_values_duplicates(self):
        # [FUNCTION DOC]
        # Mixed integer and string columns with duplicate labels give an object array containing the row values.
        df = DataFrame(
            [[1, 2, "a", "b"], [1, 2, "a", "b"]], columns=["one", "one", "two", "two"]
        )

        result = df.values
        expected = np.array([[1, 2, "a", "b"], [1, 2, "a", "b"]], dtype=object)

        tm.assert_numpy_array_equal(result, expected)

    def test_values_with_duplicate_columns(self):
        # [FUNCTION DOC]
        # Duplicate column labels with numeric data give a values array equal to the expected numeric array.
        df = DataFrame([[1, 2.5], [3, 4.5]], index=[1, 2], columns=["x", "x"])
        result = df.values
        expected = np.array([[1, 2.5], [3, 4.5]])
        assert (result == expected).all().all()

    @pytest.mark.parametrize("constructor", [date_range, period_range])
    def test_values_casts_datetimelike_to_object(self, constructor):
        # [FUNCTION DOC]
        # Parametrized over datetime and period range constructors. A frame with the datetimelike series next to a float column,
        # and then next to a string column, converts to an object array whose first column equals the series cast to object.
        series = Series(constructor("2000-01-01", periods=10, freq="D"))

        expected = series.astype("object")

        df = DataFrame(
            {"a": series, "b": np.random.default_rng(2).standard_normal(len(series))}
        )

        result = df.values.squeeze()
        assert (result[:, 0] == expected.values).all()

        df = DataFrame({"a": series, "b": ["foo"] * len(series)})

        result = df.values.squeeze()
        assert (result[:, 0] == expected.values).all()

    def test_frame_values_with_tz(self):
        # [FUNCTION DOC]
        # Timezone-aware column (US/Central) gives an object array of timestamps; duplicating the column keeps timestamps in both
        # columns; adding a column converted to US/Eastern gives a three-column object array in which the third column has the shifted wall-clock times and keeps its own zone.
        tz = "US/Central"
        df = DataFrame({"A": date_range("2000", periods=4, tz=tz)})
        result = df.values
        expected = np.array(
            [
                [Timestamp("2000-01-01", tz=tz)],
                [Timestamp("2000-01-02", tz=tz)],
                [Timestamp("2000-01-03", tz=tz)],
                [Timestamp("2000-01-04", tz=tz)],
            ]
        )
        tm.assert_numpy_array_equal(result, expected)

        # two columns, homogeneous

        df["B"] = df["A"]
        result = df.values
        expected = np.concatenate([expected, expected], axis=1)
        tm.assert_numpy_array_equal(result, expected)

        # three columns, heterogeneous
        est = "US/Eastern"
        df["C"] = df["A"].dt.tz_convert(est)

        new = np.array(
            [
                [Timestamp("2000-01-01T01:00:00", tz=est)],
                [Timestamp("2000-01-02T01:00:00", tz=est)],
                [Timestamp("2000-01-03T01:00:00", tz=est)],
                [Timestamp("2000-01-04T01:00:00", tz=est)],
            ]
        )
        expected = np.concatenate([expected, new], axis=1)
        result = df.values
        tm.assert_numpy_array_equal(result, expected)

    def test_interleave_with_tzaware(self, timezone_frame):
        # [FUNCTION DOC]
        # Using the mixed timezone fixture frame: adding a string column gives an object array with timestamps (None-zone, Eastern, CET with
        # missing values as NaT) and strings; without the extra column the array holds only the timestamp columns. Expected arrays are the transpose of the per-column arrays.
        # interleave with object
        result = timezone_frame.assign(D="foo").values
        expected = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
                ["foo", "foo", "foo"],
            ],
            dtype=object,
        ).T
        tm.assert_numpy_array_equal(result, expected)

        # interleave with only datetime64[ns]
        result = timezone_frame.values
        expected = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        tm.assert_numpy_array_equal(result, expected)

    def test_values_interleave_non_unique_cols(self):
        # [FUNCTION DOC]
        # A frame with duplicate column labels, a timestamp column and a float column, has values of the same shape and rows as the same frame with unique labels.
        df = DataFrame(
            [[Timestamp("20130101"), 3.5], [Timestamp("20130102"), 4.5]],
            columns=["x", "x"],
            index=[1, 2],
        )

        df_unique = df.copy()
        df_unique.columns = ["x", "y"]
        assert df_unique.values.shape == df.values.shape
        tm.assert_numpy_array_equal(df_unique.values[0], df.values[0])
        tm.assert_numpy_array_equal(df_unique.values[1], df.values[1])

    def test_values_numeric_cols(self, float_frame):
        # [FUNCTION DOC]
        # Selecting only the numeric columns from a frame that also has a string column gives a float64 array.
        float_frame["foo"] = "bar"

        values = float_frame[["A", "B", "C", "D"]].values
        assert values.dtype == np.float64

    def test_values_lcd(self, mixed_float_frame, mixed_int_frame):
        # [FUNCTION DOC]
        # Checks that the values dtype is the common (lowest common denominator) numeric dtype: for mixed float frames float64, float32 or float16 depending on selected columns;
        # for mixed integer frames (GH 10364) float64 when an unsigned 64-bit column is mixed with signed types, unsigned 64-bit for only unsigned columns, int64, int32 or
        # uint8 for single or compatible columns.
        # mixed lcd
        values = mixed_float_frame[["A", "B", "C", "D"]].values
        assert values.dtype == np.float64

        values = mixed_float_frame[["A", "B", "C"]].values
        assert values.dtype == np.float32

        values = mixed_float_frame[["C"]].values
        assert values.dtype == np.float16

        # GH#10364
        # B uint64 forces float because there are other signed int types
        values = mixed_int_frame[["A", "B", "C", "D"]].values
        assert values.dtype == np.float64

        values = mixed_int_frame[["A", "D"]].values
        assert values.dtype == np.int64

        # B uint64 forces float because there are other signed int types
        values = mixed_int_frame[["A", "B", "C"]].values
        assert values.dtype == np.float64

        # as B and C are both unsigned, no forcing to float is needed
        values = mixed_int_frame[["B", "C"]].values
        assert values.dtype == np.uint64

        values = mixed_int_frame[["A", "C"]].values
        assert values.dtype == np.int32

        values = mixed_int_frame[["C", "D"]].values
        assert values.dtype == np.int64

        values = mixed_int_frame[["A"]].values
        assert values.dtype == np.int32

        values = mixed_int_frame[["C"]].values
        assert values.dtype == np.uint8


class TestPrivateValues:
    # [CLASS DOC]
    # Test container for the private values property, which returns extension arrays for datetimelike data.
    def test_private_values_dt64tz(self):
        # [FUNCTION DOC]
        # A single timezone-aware column built from a datetime array returns an equal two-dimensional datetime array that does not share memory with the input;
        # subtracting the frame from itself gives a timedelta array.
        dta = date_range("2000", periods=4, tz="US/Central")._data.reshape(-1, 1)

        df = DataFrame(dta, columns=["A"])
        tm.assert_equal(df._values, dta)

        assert not np.shares_memory(df._values._ndarray, dta._ndarray)

        # TimedeltaArray
        tda = dta - dta
        df2 = df - df
        tm.assert_equal(df2._values, tda)

    def test_private_values_dt64tz_multicol(self):
        # [FUNCTION DOC]
        # Same as the single-column case with a two-column frame built from a reshaped datetime array.
        dta = date_range("2000", periods=8, tz="US/Central")._data.reshape(-1, 2)

        df = DataFrame(dta, columns=["A", "B"])
        tm.assert_equal(df._values, dta)

        assert not np.shares_memory(df._values._ndarray, dta._ndarray)

        # TimedeltaArray
        tda = dta - dta
        df2 = df - df
        tm.assert_equal(df2._values, tda)

    def test_private_values_dt64_multiblock(self):
        # [FUNCTION DOC]
        # A frame with two naive datetime columns added separately has two internal blocks, and its private values equal the datetime array reshaped to two columns and transposed.
        dta = date_range("2000", periods=8)._data

        df = DataFrame({"A": dta[:4]}, copy=False)
        df["B"] = dta[4:]

        assert len(df._mgr.blocks) == 2

        result = df._values
        expected = dta.reshape(2, 4).T
        tm.assert_equal(result, expected)
