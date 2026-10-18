# [MODULE DOC]
# Tests for Series.describe across dtypes: integers, booleans, strings,
# timedeltas, periods, empty object data, tz-aware datetimes, all numeric
# dtypes (result dtype), and a single-element extension array. Uses the
# tz_naive_fixture and any_numeric_dtype fixtures. No import-time side effects.

import numpy as np
import pytest

from pandas.core.dtypes.common import (
    is_complex_dtype,
    is_extension_array_dtype,
)

from pandas import (
    NA,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestSeriesDescribe:
    # [CLASS DOC]
    # Groups the tests of Series.describe. Stateless.
    def test_describe_ints(self):
        # [FUNCTION DOC]
        # Integer data gives count, mean, std, min, quartiles and max in that order,
        # and keeps the Series name.
        ser = Series([0, 1, 2, 3, 4], name="int_data")
        result = ser.describe()
        expected = Series(
            [5, 2, ser.std(), 0, 1, 2, 3, 4],
            name="int_data",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    def test_describe_bools(self):
        # [FUNCTION DOC]
        # Boolean data is described like categorical data: count, unique, top and freq.
        ser = Series([True, True, False, False, False], name="bool_data")
        result = ser.describe()
        expected = Series(
            [5, 2, False, 3], name="bool_data", index=["count", "unique", "top", "freq"]
        )
        tm.assert_series_equal(result, expected)

    def test_describe_strs(self):
        # [FUNCTION DOC]
        # String data gives count, unique, top (most frequent value) and freq.
        ser = Series(["a", "a", "b", "c", "d"], name="str_data")
        result = ser.describe()
        expected = Series(
            [5, 4, "a", 2], name="str_data", index=["count", "unique", "top", "freq"]
        )
        tm.assert_series_equal(result, expected)

    def test_describe_timedelta64(self):
        # [FUNCTION DOC]
        # Timedelta data gives count, mean, std, min, quartiles and max as timedeltas
        # (count stays an integer).
        ser = Series(
            [
                Timedelta("1 days"),
                Timedelta("2 days"),
                Timedelta("3 days"),
                Timedelta("4 days"),
                Timedelta("5 days"),
            ],
            name="timedelta_data",
        )
        result = ser.describe()
        expected = Series(
            [5, ser[2], ser.std(), ser[0], ser[1], ser[2], ser[3], ser[4]],
            name="timedelta_data",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    def test_describe_period(self):
        # [FUNCTION DOC]
        # Period data is described with count, unique, top and freq.
        ser = Series(
            [Period("2020-01", "M"), Period("2020-01", "M"), Period("2019-12", "M")],
            name="period_data",
        )
        result = ser.describe()
        expected = Series(
            [3, 2, ser[0], 2],
            name="period_data",
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_series_equal(result, expected)

    def test_describe_empty_object(self):
        # [FUNCTION DOC]
        # Regression test for an object Series of only None (and for an empty slice of
        # it): the result has count and unique 0 and NaN (not None) for top and freq, with
        # object dtype (GitHub issue 27183).
        # https://github.com/pandas-dev/pandas/issues/27183
        s = Series([None, None], dtype=object)
        result = s.describe()
        expected = Series(
            [0, 0, np.nan, np.nan],
            dtype=object,
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_series_equal(result, expected)

        result = s[:0].describe()
        tm.assert_series_equal(result, expected)
        # ensure NaN, not None
        assert np.isnan(result.iloc[2])
        assert np.isnan(result.iloc[3])

    def test_describe_with_tz(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Regression test (GH 21332), parametrized over timezone fixture: tz-aware or
        # naive date range of 5 days yields count, mean, min, quartiles and max as
        # timestamps in that timezone (no std).
        # GH 21332
        tz = tz_naive_fixture
        name = str(tz_naive_fixture)
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        s = Series(date_range(start, end, tz=tz), name=name)
        result = s.describe()
        expected = Series(
            [
                5,
                Timestamp(2018, 1, 3).tz_localize(tz),
                start.tz_localize(tz),
                s[1],
                s[2],
                s[3],
                end.tz_localize(tz),
            ],
            name=name,
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    def test_describe_with_tz_numeric(self):
        # [FUNCTION DOC]
        # Same as the tz test for a fixed CET timezone, with explicitly written
        # expected timestamps.
        name = tz = "CET"
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        s = Series(date_range(start, end, tz=tz), name=name)

        result = s.describe()

        expected = Series(
            [
                5,
                Timestamp("2018-01-03 00:00:00", tz=tz),
                Timestamp("2018-01-01 00:00:00", tz=tz),
                Timestamp("2018-01-02 00:00:00", tz=tz),
                Timestamp("2018-01-03 00:00:00", tz=tz),
                Timestamp("2018-01-04 00:00:00", tz=tz),
                Timestamp("2018-01-05 00:00:00", tz=tz),
            ],
            name=name,
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    def test_datetime_is_numeric_includes_datetime(self):
        # [FUNCTION DOC]
        # A datetime Series is treated like numeric data in describe: count, mean, min,
        # quartiles and max as timestamps (no std).
        s = Series(date_range("2012", periods=3))
        result = s.describe()
        expected = Series(
            [
                3,
                Timestamp("2012-01-02"),
                Timestamp("2012-01-01"),
                Timestamp("2012-01-01T12:00:00"),
                Timestamp("2012-01-02"),
                Timestamp("2012-01-02T12:00:00"),
                Timestamp("2012-01-03"),
            ],
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:Casting complex values to real discards")
    def test_numeric_result_dtype(self, any_numeric_dtype):
        # [FUNCTION DOC]
        # GH#48340, parametrized over all numeric dtypes: describe always returns floats
        # for non-complex numeric input (Float64 for extension dtypes, float64 otherwise)
        # with the values for the two-element series zero and one. For complex dtypes it
        # raises a type error stating the input must be an array of real numbers.
        # Ignores a warning about discarding the complex part.
        # GH#48340 - describe should always return float on non-complex numeric input
        if is_extension_array_dtype(any_numeric_dtype):
            dtype = "Float64"
        else:
            dtype = "complex128" if is_complex_dtype(any_numeric_dtype) else None

        ser = Series([0, 1], dtype=any_numeric_dtype)
        if dtype == "complex128":
            with pytest.raises(
                TypeError, match=r"^a must be an array of real numbers$"
            ):
                ser.describe()
            return
        result = ser.describe()
        expected = Series(
            [
                2.0,
                0.5,
                ser.std(),
                0,
                0.25,
                0.5,
                0.75,
                1.0,
            ],
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype=dtype,
        )
        tm.assert_series_equal(result, expected)

    def test_describe_one_element_ea(self):
        # [FUNCTION DOC]
        # Regression test (GH#52515): describe of a one-element Float64 Series emits no
        # warning and returns a missing standard deviation with the other statistics
        # equal to the element.
        # GH#52515
        ser = Series([0.0], dtype="Float64")
        with tm.assert_produces_warning(None):
            result = ser.describe()
        expected = Series(
            [1, 0, NA, 0, 0, 0, 0, 0],
            dtype="Float64",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)
