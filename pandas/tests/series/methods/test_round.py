# [MODULE DOC]
# Tests for Series.round and its NumPy and builtin forms, including NaN, missing
# values, rounding datetime-likes through the dt accessor, boolean extension
# dtypes, object, empty and non-numeric dtypes. Uses the datetime_series,
# any_float_dtype and unit fixtures. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm


class TestSeriesRound:
    # [CLASS DOC]
    # Groups the tests of Series.round. Stateless.
    def test_round(self, datetime_series):
        # [FUNCTION DOC]
        # Rounding the datetime_series fixture (after naming its index) to two decimals
        # equals NumPy rounding of the values and keeps the name and index.
        datetime_series.index.name = "index_name"
        result = datetime_series.round(2)
        expected = Series(
            np.round(datetime_series.values, 2), index=datetime_series.index, name="ts"
        )
        tm.assert_series_equal(result, expected)
        assert result.name == datetime_series.name

    def test_round_numpy(self, any_float_dtype):
        # [FUNCTION DOC]
        # GH#12600, parametrized over float dtypes: NumPy's round on a Series rounds to
        # the given decimals and keeps dtype; passing an output argument raises a value
        # error that out is not supported.
        # See GH#12600
        ser = Series([1.53, 1.36, 0.06], dtype=any_float_dtype)
        out = np.round(ser, decimals=0)
        expected = Series([2.0, 1.0, 0.0], dtype=any_float_dtype)
        tm.assert_series_equal(out, expected)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.round(ser, decimals=0, out=ser)

    def test_round_numpy_with_nan(self, any_float_dtype):
        # [FUNCTION DOC]
        # GH#14197, parametrized over float dtypes: rounding a Series with NaN emits no
        # warning and keeps NaN.
        # See GH#14197
        ser = Series([1.53, np.nan, 0.06], dtype=any_float_dtype)
        with tm.assert_produces_warning(None):
            result = ser.round()
        expected = Series([2.0, np.nan, 0.0], dtype=any_float_dtype)
        tm.assert_series_equal(result, expected)

    def test_round_builtin(self, any_float_dtype):
        # [FUNCTION DOC]
        # Parametrized over float dtypes: the builtin rounding function without digits
        # rounds to whole floats and with digits to that many decimals.
        ser = Series(
            [1.123, 2.123, 3.123],
            index=range(3),
            dtype=any_float_dtype,
        )
        result = round(ser)
        expected_rounded0 = Series(
            [1.0, 2.0, 3.0], index=range(3), dtype=any_float_dtype
        )
        tm.assert_series_equal(result, expected_rounded0)

        decimals = 2
        expected_rounded = Series(
            [1.12, 2.12, 3.12], index=range(3), dtype=any_float_dtype
        )
        result = round(ser, decimals)
        tm.assert_series_equal(result, expected_rounded)

    @pytest.mark.parametrize("method", ["round", "floor", "ceil"])
    @pytest.mark.parametrize("freq", ["s", "5s", "min", "5min", "h", "5h"])
    def test_round_nat(self, method, freq, unit):
        # [FUNCTION DOC]
        # GH14940, GH#56158, parametrized over round/floor/ceil, several frequencies
        # and datetime unit: rounding NaT via the datetime accessor gives NaT of the same dtype.
        # GH14940, GH#56158
        ser = Series([pd.NaT], dtype=f"M8[{unit}]")
        expected = Series(pd.NaT, dtype=f"M8[{unit}]")
        round_method = getattr(ser.dt, method)
        result = round_method(freq)
        tm.assert_series_equal(result, expected)

    def test_round_ea_boolean(self):
        # [FUNCTION DOC]
        # GH#55936: rounding a boolean masked Series returns an equal Series that is a
        # copy (modifying the result does not change the original).
        # GH#55936
        ser = Series([True, False], dtype="boolean")
        expected = ser.copy()
        result = ser.round(2)
        tm.assert_series_equal(result, expected)
        result.iloc[0] = False
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "data,decimals,expected_data",
        [
            ([0.2], 0, [0.0]),
            ([1.234, 2.567], 1, [1.2, 2.6]),
            ([1.234, 2.567], 2, [1.23, 2.57]),
            ([1.234, 2.567], 0, [1.0, 3.0]),
        ],
    )
    def test_round_dtype_object(self, data, decimals, expected_data):
        # [FUNCTION DOC]
        # GH#63444, parametrized over decimals: round on object-dtype floats works and
        # keeps object dtype.
        # GH#63444
        ser = Series(data, dtype="object")
        result = ser.round(decimals)
        expected = Series(expected_data, dtype="object")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            None,
            "float64",
            "int64",
            "object",
        ],
    )
    def test_round_empty_series(self, dtype):
        # [FUNCTION DOC]
        # GH#63444, parametrized over dtype (default, float, int, object): rounding an
        # empty Series returns an equal empty Series.
        # GH#63444
        result = Series(dtype=dtype).round(4)
        expected = Series(dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype,data",
        [
            ("string", ["a", "b", "c"]),
            ("category", ["cat1", "cat2", "cat3"]),
            (pd.StringDtype(), ["x", "y", "z"]),
        ],
    )
    def test_round_non_numeric_dtype_noop(self, dtype, data):
        # [FUNCTION DOC]
        # GH#63444, GH#63559, parametrized over string and categorical dtypes: round
        # returns the Series unchanged (no-op for non-numeric data).
        # GH#63444, GH#63559
        ser = Series(data, dtype=dtype)
        result = ser.round(2)
        tm.assert_series_equal(result, ser)
