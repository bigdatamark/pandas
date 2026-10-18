"""
Tests for Series cumulative operations.

See also
--------
tests.frame.test_cumulative
"""
# [MODULE DOC]
# Tests for the cumulative operations of Series (cumulative sum, product, minimum, maximum) across numeric, boolean, object, datetime-like, period, ordered categorical and string extension dtypes, including missing-value skipping.
# Defines a module-level mapping from method name to the matching NumPy function (sum, product, and the minimum/maximum accumulate functions) used to compute expected results. No other import-time side effects. Inline anonymous functions used in parametrization cannot carry documentation blocks.


import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

methods = {
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "cummin": np.minimum.accumulate,
    "cummax": np.maximum.accumulate,
}


class TestSeriesCumulativeOps:
    # [CLASS DOC]
    # Groups tests of Series cumulative operations. Stateless.
    @pytest.mark.parametrize("func", [np.cumsum, np.cumprod])
    def test_datetime_series(self, datetime_series, func):
        # [FUNCTION DOC]
        # Verifies that NumPy cumulative sum and product applied to the datetime fixture Series match the same function applied to the raw array (including dtype). With every other element set to NaN, the result at the odd positions equals the function applied to the array of non-missing values (dtype not checked). Parametrized over the two functions.
        tm.assert_numpy_array_equal(
            func(datetime_series).values,
            func(np.array(datetime_series)),
            check_dtype=True,
        )

        # with missing values
        ts = datetime_series.copy()
        ts[::2] = np.nan

        result = func(ts)[1::2]
        expected = func(np.array(ts.dropna()))

        tm.assert_numpy_array_equal(result.values, expected, check_dtype=False)

    @pytest.mark.parametrize("method", ["cummin", "cummax"])
    def test_cummin_cummax(self, datetime_series, method):
        # [FUNCTION DOC]
        # Verifies the cumulative minimum and maximum methods on the datetime fixture series: values equal the NumPy accumulate function on the raw array; with every other value NaN the odd-position results equal the accumulate function applied to the non-missing Series (index frequency cleared before comparing).
        ufunc = methods[method]

        result = getattr(datetime_series, method)().values
        expected = ufunc(np.array(datetime_series))

        tm.assert_numpy_array_equal(result, expected)
        ts = datetime_series.copy()
        ts[::2] = np.nan
        result = getattr(ts, method)()[1::2]
        expected = ufunc(ts.dropna())

        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ts",
        [
            pd.Timedelta(0),
            pd.Timestamp("1999-12-31"),
            pd.Timestamp("1999-12-31").tz_localize("US/Pacific"),
        ],
    )
    @pytest.mark.parametrize(
        "method, skipna, exp_tdi",
        [
            ["cummax", True, ["NaT", "2 days", "NaT", "2 days", "NaT", "3 days"]],
            ["cummin", True, ["NaT", "2 days", "NaT", "1 days", "NaT", "1 days"]],
            [
                "cummax",
                False,
                ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT"],
            ],
            [
                "cummin",
                False,
                ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT"],
            ],
        ],
    )
    def test_cummin_cummax_datetimelike(self, ts, method, skipna, exp_tdi):
        # [FUNCTION DOC]
        # Verifies cumulative minimum and maximum for datetime-like dtypes: a timedelta Series (offset zero), a naive datetime Series and a tz-aware datetime Series are built by adding an offset to a timedelta list with missing values at positions zero, two and four. Parametrized over the offset, method, skipna flag and the expected timedelta list. With skipping on, missing entries stay missing while others hold the running extremum; with skipping off, everything is missing since the first entry is missing.
        # with ts==pd.Timedelta(0), we are testing td64; with naive Timestamp
        #  we are testing datetime64[ns]; with Timestamp[US/Pacific]
        #  we are testing dt64tz
        tdi = pd.to_timedelta(["NaT", "2 days", "NaT", "1 days", "NaT", "3 days"])
        ser = pd.Series(tdi + ts)

        exp_tdi = pd.to_timedelta(exp_tdi)
        expected = pd.Series(exp_tdi + ts)
        result = getattr(ser, method)(skipna=skipna)
        tm.assert_series_equal(expected, result)

    def test_cumsum_datetimelike(self):
        # [FUNCTION DOC]
        # Verifies cumulative sum on a frame of timedelta columns containing a missing value (issue 57956): missing entries stay missing while later entries keep accumulating over earlier valid values. The test lives in the Series file but exercises the frame method.
        # GH#57956
        df = pd.DataFrame(
            [
                [pd.Timedelta(0), pd.Timedelta(days=1)],
                [pd.Timedelta(days=2), pd.NaT],
                [pd.Timedelta(hours=-6), pd.Timedelta(hours=12)],
            ]
        )
        result = df.cumsum()
        expected = pd.DataFrame(
            [
                [pd.Timedelta(0), pd.Timedelta(days=1)],
                [pd.Timedelta(days=2), pd.NaT],
                [pd.Timedelta(days=1, hours=18), pd.Timedelta(days=1, hours=12)],
            ]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "func, exp",
        [
            ("cummin", "2012-1-1"),
            ("cummax", "2012-1-2"),
        ],
    )
    def test_cummin_cummax_period(self, func, exp):
        # [FUNCTION DOC]
        # Verifies cumulative minimum or maximum on a daily period Series with a missing middle entry (issue 28385). With skipna off, everything after the missing entry is missing; with skipna on, the missing entry stays missing and the last entry holds the running extremum (first period for minimum, second for maximum). Parametrized over method and expected final period.
        # GH#28385
        ser = pd.Series(
            [pd.Period("2012-1-1", freq="D"), pd.NaT, pd.Period("2012-1-2", freq="D")]
        )
        result = getattr(ser, func)(skipna=False)
        expected = pd.Series([pd.Period("2012-1-1", freq="D"), pd.NaT, pd.NaT])
        tm.assert_series_equal(result, expected)

        result = getattr(ser, func)(skipna=True)
        exp = pd.Period(exp, freq="D")
        expected = pd.Series([pd.Period("2012-1-1", freq="D"), pd.NaT, exp])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "arg",
        [
            [False, False, False, True, True, False, False],
            [False, False, False, False, False, False, False],
        ],
    )
    @pytest.mark.parametrize(
        "func", [lambda x: x, lambda x: ~x], ids=["identity", "inverse"]
    )
    @pytest.mark.parametrize("method", methods.keys())
    def test_cummethods_bool(self, arg, func, method):
        # [FUNCTION DOC]
        # Verifies (issue 6270) the four cumulative methods on boolean Series (two patterns, each used as-is and inverted) equal a Series built by applying the matching NumPy function to the underlying values. Parametrized over pattern, transformation and method name.
        # GH#6270
        # checking Series method vs the ufunc applied to the values

        ser = func(pd.Series(arg))
        ufunc = methods[method]

        exp_vals = ufunc(ser.values)
        expected = pd.Series(exp_vals)

        result = getattr(ser, method)()

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "method, expected",
        [
            ["cumsum", pd.Series([0, 1, np.nan, 1], dtype=object)],
            ["cumprod", pd.Series([False, 0, np.nan, 0])],
            ["cummin", pd.Series([False, False, np.nan, False])],
            ["cummax", pd.Series([False, True, np.nan, True])],
        ],
    )
    def test_cummethods_bool_in_object_dtype(self, method, expected):
        # [FUNCTION DOC]
        # Verifies cumulative methods on a Series of booleans with a missing value, which has object dtype: sum gives an object result mixing counts and NaN; product, minimum and maximum give the expected values with NaN at the missing position. Parametrized over method and expected Series.
        ser = pd.Series([False, True, np.nan, False])
        result = getattr(ser, method)()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "method, order",
        [
            ["cummax", "abc"],
            ["cummin", "cba"],
        ],
    )
    def test_cummax_cummin_on_ordered_categorical(self, method, order):
        # [FUNCTION DOC]
        # Verifies (issue 52335) cumulative maximum and minimum on an ordered categorical Series where the category order is ascending for the maximum case and reversed for the minimum case, so both yield the same running extremum list of letters.
        # GH#52335
        cat = pd.CategoricalDtype(list(order), ordered=True)
        ser = pd.Series(
            list("ababcab"),
            dtype=cat,
        )
        result = getattr(ser, method)()
        expected = pd.Series(
            list("abbbccc"),
            dtype=cat,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "skip, exp",
        [
            [True, ["a", np.nan, "b", "b", "c"]],
            [False, ["a", np.nan, np.nan, np.nan, np.nan]],
        ],
    )
    @pytest.mark.parametrize(
        "method, order",
        [
            ["cummax", "abc"],
            ["cummin", "cba"],
        ],
    )
    def test_cummax_cummin_ordered_categorical_nan(self, skip, exp, method, order):
        # [FUNCTION DOC]
        # Verifies (issue 52335) cumulative maximum and minimum on an ordered categorical with a missing value: with skipna on, the missing entry stays missing and the rest accumulate; with skipna off, all entries after the first missing one are missing. Parametrized over skipna with expected list, and method with category order.
        # GH#52335
        cat = pd.CategoricalDtype(list(order), ordered=True)
        ser = pd.Series(
            ["a", np.nan, "b", "a", "c"],
            dtype=cat,
        )
        result = getattr(ser, method)(skipna=skip)
        expected = pd.Series(
            exp,
            dtype=cat,
        )
        tm.assert_series_equal(
            result,
            expected,
        )

    def test_cumprod_timedelta(self):
        # [FUNCTION DOC]
        # Verifies (issue 48111) that cumulative product on a timedelta Series raises a type error stating cumprod is not supported for timedelta.
        # GH#48111
        ser = pd.Series([pd.Timedelta(days=1), pd.Timedelta(days=3)])
        with pytest.raises(TypeError, match="cumprod not supported for Timedelta"):
            ser.cumprod()

    @pytest.mark.parametrize(
        "data, op, skipna, expected_data",
        [
            ([], "cumsum", True, []),
            ([], "cumsum", False, []),
            (["x", "z", "y"], "cumsum", True, ["x", "xz", "xzy"]),
            (["x", "z", "y"], "cumsum", False, ["x", "xz", "xzy"]),
            (["x", pd.NA, "y"], "cumsum", True, ["x", pd.NA, "xy"]),
            (["x", pd.NA, "y"], "cumsum", False, ["x", pd.NA, pd.NA]),
            ([pd.NA, "x", "y"], "cumsum", True, [pd.NA, "x", "xy"]),
            ([pd.NA, "x", "y"], "cumsum", False, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cumsum", True, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cumsum", False, [pd.NA, pd.NA, pd.NA]),
            ([], "cummin", True, []),
            ([], "cummin", False, []),
            (["y", "z", "x"], "cummin", True, ["y", "y", "x"]),
            (["y", "z", "x"], "cummin", False, ["y", "y", "x"]),
            (["y", pd.NA, "x"], "cummin", True, ["y", pd.NA, "x"]),
            (["y", pd.NA, "x"], "cummin", False, ["y", pd.NA, pd.NA]),
            ([pd.NA, "y", "x"], "cummin", True, [pd.NA, "y", "x"]),
            ([pd.NA, "y", "x"], "cummin", False, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cummin", True, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cummin", False, [pd.NA, pd.NA, pd.NA]),
            ([], "cummax", True, []),
            ([], "cummax", False, []),
            (["x", "z", "y"], "cummax", True, ["x", "z", "z"]),
            (["x", "z", "y"], "cummax", False, ["x", "z", "z"]),
            (["x", pd.NA, "y"], "cummax", True, ["x", pd.NA, "y"]),
            (["x", pd.NA, "y"], "cummax", False, ["x", pd.NA, pd.NA]),
            ([pd.NA, "x", "y"], "cummax", True, [pd.NA, "x", "y"]),
            ([pd.NA, "x", "y"], "cummax", False, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cummax", True, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cummax", False, [pd.NA, pd.NA, pd.NA]),
        ],
    )
    def test_cum_methods_ea_strings(
        self, string_dtype_no_object, data, op, skipna, expected_data
    ):
        # [FUNCTION DOC]
        # Verifies cumulative sum (concatenation), minimum and maximum on non-object string extension dtypes (fixture string_dtype_no_object), for empty data, plain strings, and data with missing values at start, middle or everywhere, for both skipna settings. With skipping on, missing entries are kept missing and the accumulation continues; with it off, everything after the first missing entry is missing. References the pull requests that added pyarrow and Python implementations.
        # https://github.com/pandas-dev/pandas/pull/60633 - pyarrow
        # https://github.com/pandas-dev/pandas/pull/60938 - Python
        ser = pd.Series(data, dtype=string_dtype_no_object)
        method = getattr(ser, op)
        expected = pd.Series(expected_data, dtype=string_dtype_no_object)
        result = method(skipna=skipna)
        tm.assert_series_equal(result, expected)

    def test_cumprod_pyarrow_strings(self, pyarrow_string_dtype, skipna):
        # [FUNCTION DOC]
        # Verifies that cumulative product on a pyarrow string Series raises a type error saying the cumprod operation is not supported for the dtype, for both skipna settings (fixtures).
        # https://github.com/pandas-dev/pandas/pull/60633
        ser = pd.Series(list("xyz"), dtype=pyarrow_string_dtype)
        msg = re.escape(f"operation 'cumprod' not supported for dtype '{ser.dtype}'")
        with pytest.raises(TypeError, match=msg):
            ser.cumprod(skipna=skipna)
