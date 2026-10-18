# [MODULE DOC]
# Tests for the discrete-difference method of the frame class (diff): periods argument
# (negative, zero, float, numpy integer), both axes, integer, float, nullable, sparse,
# datetime, timedelta and period data, missing values, read-only data and result dtypes.
# Uses the datetime frame and unit fixtures and the testing helpers module. No
# import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameDiff:
    # [CLASS DOC]
    # Stateless pytest class of diff tests on frames.
    def test_diff_requires_integer(self):
        # [FUNCTION DOC]
        # A non-integer fractional periods value raises a value error saying periods
        # must be an integer.
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
        with pytest.raises(ValueError, match="periods must be an integer"):
            df.diff(1.5)

    # GH#44572 np.int64 is accepted
    @pytest.mark.parametrize("num", [1, np.int64(1)])
    def test_diff(self, datetime_frame, num):
        # [FUNCTION DOC]
        # Parametrized over a python int and a numpy int64 as periods (the latter
        # accepted per an issue). Column difference equals the column minus its shifted
        # version.
        df = datetime_frame
        the_diff = df.diff(num)

        expected = df["A"] - df["A"].shift(num)
        tm.assert_series_equal(the_diff["A"], expected)

    def test_diff_int_dtype(self):
        # [FUNCTION DOC]
        # Large integers near 1e16 keep precision: the difference of consecutive values
        # is exactly one.
        # int dtype
        a = 10_000_000_000_000_000
        b = a + 1
        ser = Series([a, b])

        rs = DataFrame({"s": ser}).diff()
        assert rs.s[1] == 1

    def test_diff_mixed_numeric(self, datetime_frame):
        # [FUNCTION DOC]
        # For a 32-bit float frame, differences equal the column minus its shift.
        # mixed numeric
        tf = datetime_frame.astype("float32")
        the_diff = tf.diff(1)
        tm.assert_series_equal(the_diff["A"], tf["A"] - tf["A"].shift(1))

    def test_diff_axis1_nonconsolidated(self):
        # [FUNCTION DOC]
        # A frame built with an inserted column (non-consolidated blocks) differenced
        # along columns gives missing for the first column and the expected differences
        # for the rest.
        # GH#10907
        df = DataFrame({"y": Series([2]), "z": Series([3])})
        df.insert(0, "x", 1)
        result = df.diff(axis=1)
        expected = DataFrame({"x": np.nan, "y": Series(1), "z": Series(1)})
        tm.assert_frame_equal(result, expected)

    def test_diff_timedelta64_with_nat(self):
        # [FUNCTION DOC]
        # Timedelta frame with a column of NaT: diff by one along rows gives NaT for
        # that column and the expected differences for the other; zero periods equals
        # frame minus frame; periods minus one along columns gives all missing.
        # GH#32441
        arr = np.arange(6).reshape(3, 2).astype("timedelta64[ns]")
        arr[:, 0] = np.timedelta64("NaT", "ns")

        df = DataFrame(arr)
        result = df.diff(1, axis=0)

        expected = DataFrame({0: df[0], 1: [pd.NaT, pd.Timedelta(2), pd.Timedelta(2)]})
        tm.assert_equal(result, expected)

        result = df.diff(0)
        expected = df - df
        assert expected[0].isna().all()
        tm.assert_equal(result, expected)

        result = df.diff(-1, axis=1)
        expected = df * np.nan
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis0_with_nat(self, tz, unit):
        # [FUNCTION DOC]
        # Parametrized over naive and UTC time zones; takes unit fixture. A datetime
        # column with leading NaT gives timedelta results with NaT for the first two
        # rows; unit preserved.
        # GH#32441
        dti = pd.DatetimeIndex(["NaT", "2019-01-01", "2019-01-02"], tz=tz).as_unit(unit)
        ser = Series(dti)

        df = ser.to_frame()

        result = df.diff()
        ex_index = pd.TimedeltaIndex([pd.NaT, pd.NaT, pd.Timedelta(days=1)]).as_unit(
            unit
        )
        expected = Series(ex_index).to_frame()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_with_nat_zero_periods(self, tz):
        # [FUNCTION DOC]
        # Parametrized over naive and UTC. With zero periods, differences of NaT are NaT
        # rather than a zero timedelta, along either axis.
        # diff on NaT values should give NaT, not timedelta64(0)
        dti = date_range("2016-01-01", periods=4, tz=tz)
        ser = Series(dti)
        df = ser.to_frame().copy()

        df[1] = ser.copy()

        df.iloc[:, 0] = pd.NaT

        expected = df - df
        assert expected[0].isna().all()

        result = df.diff(0, axis=0)
        tm.assert_frame_equal(result, expected)

        result = df.diff(0, axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis0(self, tz):
        # [FUNCTION DOC]
        # Parametrized over naive and UTC. Two identical date columns diffed along rows
        # give NaT then one day in each.
        # GH#18578
        df = DataFrame(
            {
                0: date_range("2010", freq="D", periods=2, tz=tz),
                1: date_range("2010", freq="D", periods=2, tz=tz),
            }
        )

        result = df.diff(axis=0)
        expected = DataFrame(
            {
                0: pd.TimedeltaIndex(["NaT", "1 days"]),
                1: pd.TimedeltaIndex(["NaT", "1 days"]),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis1(self, tz):
        # [FUNCTION DOC]
        # Parametrized over naive and UTC. Diff along columns yields NaT for the first
        # column (microsecond timedelta dtype) and zero days for the second.
        # GH#18578
        df = DataFrame(
            {
                0: date_range("2010", freq="D", periods=2, tz=tz),
                1: date_range("2010", freq="D", periods=2, tz=tz),
            }
        )

        result = df.diff(axis=1)
        expected = DataFrame(
            {
                0: pd.TimedeltaIndex(["NaT", "NaT"], dtype="m8[us]"),
                1: pd.TimedeltaIndex(["0 days", "0 days"]),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_diff_timedelta(self, unit):
        # [FUNCTION DOC]
        # Takes unit fixture. A time column and a value column: diff gives NaT then a
        # one-minute timedelta, and floats; unit preserved.
        # GH#4533
        df = DataFrame(
            {
                "time": [Timestamp("20130101 9:01"), Timestamp("20130101 9:02")],
                "value": [1.0, 2.0],
            }
        )
        df["time"] = df["time"].dt.as_unit(unit)

        res = df.diff()
        exp = DataFrame(
            [[pd.NaT, np.nan], [pd.Timedelta("00:01:00"), 1]], columns=["time", "value"]
        )
        exp["time"] = exp["time"].dt.as_unit(unit)
        tm.assert_frame_equal(res, exp)

    def test_diff_mixed_dtype(self):
        # [FUNCTION DOC]
        # A frame with an object-dtype column of integers alongside float columns: the
        # float columns' diff stays float64.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["A"] = np.array([1, 2, 3, 4, 5], dtype=object)

        result = df.diff()
        assert result[0].dtype == np.float64

    def test_diff_neg_n(self, datetime_frame):
        # [FUNCTION DOC]
        # Negative periods equals frame minus its negative shift.
        rs = datetime_frame.diff(-1)
        xp = datetime_frame - datetime_frame.shift(-1)
        tm.assert_frame_equal(rs, xp)

    def test_diff_float_n(self, datetime_frame):
        # [FUNCTION DOC]
        # A float periods value that is integral (1.0) is accepted and equals integer
        # one.
        rs = datetime_frame.diff(1.0)
        xp = datetime_frame.diff(1)
        tm.assert_frame_equal(rs, xp)

    def test_diff_axis(self):
        # [FUNCTION DOC]
        # Small float frame: diff along columns gives missing first column and ones;
        # along rows gives missing first row and twos.
        # GH#9727
        df = DataFrame([[1.0, 2.0], [3.0, 4.0]])
        tm.assert_frame_equal(
            df.diff(axis=1), DataFrame([[np.nan, 1.0], [np.nan, 1.0]])
        )
        tm.assert_frame_equal(
            df.diff(axis=0), DataFrame([[np.nan, np.nan], [2.0, 2.0]])
        )

    def test_diff_period(self):
        # [FUNCTION DOC]
        # A period-dtype column diffed along columns (axis one) gives frame minus NaT
        # cast to object; no wrong axis is passed to the column method.
        # GH#32995 Don't pass an incorrect axis
        pi = date_range("2016-01-01", periods=3).to_period("D")
        df = DataFrame({"A": pi})

        result = df.diff(1, axis=1)

        expected = (df - pd.NaT).astype(object)
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes(self):
        # [FUNCTION DOC]
        # With integer and float columns, diff along columns operates column-wise: first
        # column all missing, second the original difference; for float32 and float64
        # columns the second becomes zero.
        # GH#32995 operate column-wise when we have mixed dtypes and axis=1
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        expected = DataFrame({"A": [np.nan, np.nan, np.nan], "B": df["B"] / 2})

        result = df.diff(axis=1)
        tm.assert_frame_equal(result, expected)

        # GH#21437 mixed-float-dtypes
        df = DataFrame(
            {"a": np.arange(3, dtype="float32"), "b": np.arange(3, dtype="float64")}
        )
        result = df.diff(axis=1)
        expected = DataFrame({"a": df["a"] * np.nan, "b": df["b"] * 0})
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes_large_periods(self):
        # [FUNCTION DOC]
        # Periods larger than the number of columns along columns gives all missing.
        # GH#32995 operate column-wise when we have mixed dtypes and axis=1
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        expected = df * np.nan

        result = df.diff(axis=1, periods=3)
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes_negative_periods(self):
        # [FUNCTION DOC]
        # Negative periods along columns takes the difference with the following column
        # (first column result is the negative of itself minus zero here, second column
        # missing).
        # GH#32995 operate column-wise when we have mixed dtypes and axis=1
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        expected = DataFrame({"A": -1.0 * df["A"], "B": df["B"] * np.nan})

        result = df.diff(axis=1, periods=-1)
        tm.assert_frame_equal(result, expected)

    def test_diff_sparse(self):
        # [FUNCTION DOC]
        # Sparse integer frame diff returns sparse float frame with fill value zero and
        # missing first row.
        # GH#28813 .diff() should work for sparse dataframes as well
        sparse_df = DataFrame([[0, 1], [1, 0]], dtype="Sparse[int]")

        result = sparse_df.diff()
        expected = DataFrame(
            [[np.nan, np.nan], [1.0, -1.0]], dtype=pd.SparseDtype("float", 0.0)
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "axis,expected",
        [
            (
                0,
                DataFrame(
                    {
                        "a": [pd.NA, 0, 1, 0, pd.NA, pd.NA, pd.NA, 0],
                        "b": [pd.NA, 1, pd.NA, pd.NA, -2, 1, pd.NA, pd.NA],
                        "c": np.repeat(pd.NA, 8),  # type: ignore[call-overload]
                        "d": [pd.NA, 3, 5, 7, 9, 11, 13, 15],
                    },
                    dtype="Int64",
                ),
            ),
            (
                1,
                DataFrame(
                    {
                        "a": np.repeat(pd.NA, 8),  # type: ignore[call-overload]
                        "b": [0, 1, pd.NA, 1, pd.NA, pd.NA, pd.NA, 0],
                        "c": np.repeat(pd.NA, 8),  # type: ignore[call-overload]
                        "d": np.repeat(pd.NA, 8),  # type: ignore[call-overload]
                    },
                    dtype="Int64",
                ),
            ),
        ],
    )
    def test_diff_integer_na(self, axis, expected):
        # [FUNCTION DOC]
        # Parametrized over axis zero and one with expected nullable integer frames.
        # Differences with NA propagate NA where either operand is missing; all-missing
        # column stays NA.
        # GH#24171 IntegerNA Support for DataFrame.diff()
        df = DataFrame(
            {
                "a": np.repeat([0, 1, pd.NA, 2], 2),
                "b": np.tile([0, 1, pd.NA, 2], 2),
                "c": np.repeat(pd.NA, 8),
                "d": np.arange(1, 9) ** 2,
            },
            dtype="Int64",
        )

        # Test case for default behaviour of diff
        result = df.diff(axis=axis)
        tm.assert_frame_equal(result, expected)

    def test_diff_readonly(self):
        # [FUNCTION DOC]
        # Data with the writeable flag cleared can be differenced; result equals that of
        # a writable copy.
        # https://github.com/pandas-dev/pandas/issues/35559
        arr = np.random.default_rng(2).standard_normal((5, 2))
        arr.flags.writeable = False
        df = DataFrame(arr)
        result = df.diff()
        expected = DataFrame(np.array(df)).diff()
        tm.assert_frame_equal(result, expected)

    def test_diff_all_int_dtype(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Takes the integer numpy dtype fixture. Differences of integer frames are
        # float32 for 8 and 16 bit integers and float64 for the rest, with leading
        # missing.
        # GH 14773
        df = DataFrame(range(5))
        df = df.astype(any_int_numpy_dtype)
        result = df.diff()
        expected_dtype = (
            "float32" if any_int_numpy_dtype in ("int8", "int16") else "float64"
        )
        expected = DataFrame([np.nan, 1.0, 1.0, 1.0, 1.0], dtype=expected_dtype)
        tm.assert_frame_equal(result, expected)
