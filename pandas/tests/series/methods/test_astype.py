# [MODULE DOC]
# Tests for Series.astype.
# Covers the argument API (unit-less datetime dtypes, the errors option, dict-like
# dtype mappings), conversions between numeric, string, datetime, timedelta,
# period, extension and categorical dtypes, NaN/NA handling, attribute
# propagation, and category-specific behavior. Defines one module-level helper
# (random string generator). Uses fixtures such as unit, any_float_dtype,
# using_infer_string and nullable_string_dtype. Some tests require pyarrow.
# No import-time side effects.

from datetime import (
    datetime,
    timedelta,
)
from importlib import reload
import string
import sys

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas.errors import Pandas4Warning
import pandas.util._test_decorators as td

from pandas import (
    NA,
    Categorical,
    CategoricalDtype,
    DatetimeTZDtype,
    Index,
    Interval,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    to_datetime,
)
import pandas._testing as tm


def rand_str(nchars: int) -> str:
    # [FUNCTION DOC]
    # HELPER: generates test strings for the astype string-conversion tests.
    # Takes nchars, the number of characters wanted, and returns a text string made of
    # that many characters drawn from ASCII letters and digits.
    # KNOWN ISSUE: the docstring says "random byte string" but the function
    # returns a text string.
    # Each call builds a fresh random generator with the same fixed seed, so repeated
    # calls with the same length return the same string and calls with different
    # lengths share a common prefix.
    """
    Generate one random byte string.
    """
    RANDS_CHARS = np.array(
        list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
    )
    return "".join(np.random.default_rng(2).choice(RANDS_CHARS, nchars))


class TestAstypeAPI:
    # [CLASS DOC]
    # Tests of the API-level behavior of Series.astype: argument validation and
    # dtype mappings, as opposed to individual dtype conversions. Stateless.
    def test_astype_unitless_dt64_raises(self):
        # [FUNCTION DOC]
        # Regression test (GH#47844): casting a datetime64 Series or its one-column frame
        # to the unit-less datetime64 type (NumPy class or string) raises a type error
        # saying casting to a unit-less dtype is not supported.
        # GH#47844
        ser = Series(["1970-01-01", "1970-01-01", "1970-01-01"], dtype="datetime64[ns]")
        df = ser.to_frame()

        msg = "Casting to unit-less dtype 'datetime64' is not supported"
        with pytest.raises(TypeError, match=msg):
            ser.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            df.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            ser.astype("datetime64")
        with pytest.raises(TypeError, match=msg):
            df.astype("datetime64")

    def test_arg_for_errors_in_astype(self):
        # [FUNCTION DOC]
        # Regression test (GH#14878): an invalid errors argument (the boolean False)
        # raises a value error listing the allowed values "raise" and "ignore"; the valid
        # value "raise" works without error.
        # see GH#14878
        ser = Series([1, 2, 3])

        msg = (
            r"Expected value of kwarg 'errors' to be one of \['raise', "
            r"'ignore'\]\. Supplied value is 'False'"
        )
        with pytest.raises(ValueError, match=msg):
            ser.astype(np.float64, errors=False)

        ser.astype(np.int8, errors="raise")

    @pytest.mark.parametrize("dtype_class", [dict, Series])
    def test_astype_dict_like(self, dtype_class):
        # [FUNCTION DOC]
        # Parametrized over the mapping type (dict or Series) used as the dtype argument.
        # A mapping keyed by the Series name converts to string and to float64 as
        # expected. A mapping with an extra key, a key that is not the name, or an empty
        # mapping (GH#16717) raises a key error saying only the Series name can be used
        # as key.
        # see GH#7271
        ser = Series(range(0, 10, 2), name="abc")

        dt1 = dtype_class({"abc": str})
        result = ser.astype(dt1)
        expected = Series(["0", "2", "4", "6", "8"], name="abc", dtype="str")
        tm.assert_series_equal(result, expected)

        dt2 = dtype_class({"abc": "float64"})
        result = ser.astype(dt2)
        expected = Series([0.0, 2.0, 4.0, 6.0, 8.0], dtype="float64", name="abc")
        tm.assert_series_equal(result, expected)

        dt3 = dtype_class({"abc": str, "def": str})
        msg = (
            "Only the Series name can be used for the key in Series dtype "
            r"mappings\."
        )
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt3)

        dt4 = dtype_class({0: str})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt4)

        # GH#16717
        # if dtypes provided is empty, it should error
        if dtype_class is Series:
            dt5 = dtype_class({}, dtype=object)
        else:
            dt5 = dtype_class({})

        with pytest.raises(KeyError, match=msg):
            ser.astype(dt5)


class TestAstype:
    # [CLASS DOC]
    # Tests of Series.astype conversions between concrete dtypes (datetime,
    # timedelta, string, numeric, object, extension types). Stateless.
    @pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
    def test_astype_object_to_dt64_non_nano(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone (none, UTC, US/Pacific). Casts an object Series that
        # mixes a Timestamp, a date string and the integer 2500 to a microsecond-resolution
        # datetime dtype (tz-aware when given; GH#55756, GH#54620). The integer is
        # interpreted as microseconds since the epoch. Expected values are built
        # pointwise from each element converted separately.
        # GH#55756, GH#54620
        ts = Timestamp("2999-01-01")
        dtype = "M8[us]"
        if tz is not None:
            dtype = f"M8[us, {tz}]"
        vals = [ts, "2999-01-02 03:04:05.678910", 2500]
        ser = Series(vals, dtype=object)
        result = ser.astype(dtype)

        # The 2500 is interpreted as microseconds, consistent with what
        #  we would get if we created DatetimeIndexes from vals[:2] and vals[2:]
        #  and concated the results.
        pointwise = [
            vals[0].tz_localize(tz),
            Timestamp(vals[1], tz=tz),
            to_datetime(vals[2], unit="us", utc=True).tz_convert(tz),
        ]
        exp_vals = [x.as_unit("us").asm8 for x in pointwise]
        exp_arr = np.array(exp_vals, dtype="M8[us]")
        expected = Series(exp_arr, dtype="M8[us]")
        if tz is not None:
            expected = expected.dt.tz_localize("UTC").dt.tz_convert(tz)
        tm.assert_series_equal(result, expected)

    def test_astype_mixed_object_to_dt64tz(self):
        # [FUNCTION DOC]
        # Casting an object Series holding timestamps in two different timezones to a
        # tz-aware dtype for a third zone succeeds and converts every value to the target
        # zone (earlier versions raised a value error; GH#32581).
        # pre-2.0 this raised ValueError bc of tz mismatch
        # xref GH#32581
        ts = Timestamp("2016-01-04 05:06:07", tz="US/Pacific")
        ts2 = ts.tz_convert("Asia/Tokyo")

        ser = Series([ts, ts2], dtype=object)
        res = ser.astype("datetime64[ns, Europe/Brussels]")
        expected = Series(
            [ts.tz_convert("Europe/Brussels"), ts2.tz_convert("Europe/Brussels")],
            dtype="datetime64[ns, Europe/Brussels]",
        )
        tm.assert_series_equal(res, expected)

    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_astype_empty_constructor_equality(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over every NumPy type code. For codes other than bytes, void and
        # the generic datetime/timedelta codes (poorly supported or tested elsewhere),
        # an empty Series created with that dtype equals an empty Series cast to it (GH#15524).
        # see GH#15524

        if dtype not in (
            "S",
            "V",  # poor support (if any) currently
            "M",
            "m",  # Generic timestamps raise a ValueError. Already tested.
        ):
            init_empty = Series([], dtype=dtype)
            as_type_empty = Series([]).astype(dtype)
            tm.assert_series_equal(init_empty, as_type_empty)

    @pytest.mark.parametrize("dtype", [str, np.str_])
    @pytest.mark.parametrize(
        "data",
        [
            [string.digits * 10, rand_str(63), rand_str(64), rand_str(1000)],
            [string.digits * 10, rand_str(63), rand_str(64), np.nan, 1.0],
        ],
    )
    def test_astype_str_map(self, dtype, data, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized over dtype (str or the NumPy string class) and two datasets of
        # long random strings (one with NaN and a float). Casting to a string type must
        # equal mapping the built-in string conversion over the values; under the
        # inferred-string option with the built-in str type, missing values stay missing
        # instead of becoming text, and otherwise the expected result is object dtype
        # (GH#4405).
        # see GH#4405
        series = Series(data)
        using_string_dtype = using_infer_string and dtype is str
        result = series.astype(dtype)
        if using_string_dtype:
            expected = series.map(lambda val: str(val) if val is not np.nan else np.nan)
        else:
            expected = series.map(str)
            if using_infer_string:
                expected = expected.astype(object)
        tm.assert_series_equal(result, expected)

    def test_astype_float_to_period(self):
        # [FUNCTION DOC]
        # Casting a float Series holding only NaN to a daily period dtype gives a period
        # Series holding a missing value.
        result = Series([np.nan]).astype("period[D]")
        expected = Series([NaT], dtype="period[D]")
        tm.assert_series_equal(result, expected)

    def test_astype_no_pandas_dtype(self):
        # [FUNCTION DOC]
        # Casting an int64 Series to its own array's dtype object (the NumPy-backed
        # extension dtype, not available in the public API) gives back an equal Series.
        # https://github.com/pandas-dev/pandas/pull/24866
        ser = Series([1, 2], dtype="int64")
        # Don't have NumpyEADtype in the public API, so we use `.array.dtype`,
        # which is a NumpyEADtype.
        result = ser.astype(ser.array.dtype)
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    def test_astype_generic_timestamp_no_frequency(self, dtype, request):
        # [FUNCTION DOC]
        # Parametrized over the generic NumPy datetime64 and timedelta64 classes. Casting
        # an integer Series to a unit-less dtype raises a value error telling the user to
        # supply a unit such as nanoseconds (GH#15524, GH#15987).
        # KNOWN ISSUE: the xfail mark is conditioned on the dtype name being outside
        # the two expected names, which can never hold for these parameters, so the
        # mark is dead code.
        # see GH#15524, GH#15987
        data = [1]
        ser = Series(data)

        if np.dtype(dtype).name not in ["timedelta64", "datetime64"]:
            mark = pytest.mark.xfail(reason="GH#33890 Is assigned ns unit")
            request.applymarker(mark)

        msg = (
            rf"The '{dtype.__name__}' dtype has no unit\. "
            rf"Please pass in '{dtype.__name__}\[ns\]' instead."
        )
        with pytest.raises(ValueError, match=msg):
            ser.astype(dtype)

    def test_astype_dt64_to_str(self):
        # [FUNCTION DOC]
        # Casting a datetime64 Series to str gives date-only strings in a string-dtype
        # Series (GH#10442).
        # GH#10442 : testing astype(str) is correct for Series/DatetimeIndex
        dti = date_range("2012-01-01", periods=3)
        result = Series(dti).astype(str)
        expected = Series(["2012-01-01", "2012-01-02", "2012-01-03"], dtype="str")
        tm.assert_series_equal(result, expected)

    def test_astype_dt64tz_to_str(self):
        # [FUNCTION DOC]
        # Casting a tz-aware datetime64 Series (US/Eastern) to str gives strings with the
        # time and UTC offset, in a string-dtype Series (GH#10442).
        # GH#10442 : testing astype(str) is correct for Series/DatetimeIndex
        dti_tz = date_range("2012-01-01", periods=3, tz="US/Eastern")
        result = Series(dti_tz).astype(str)
        expected = Series(
            [
                "2012-01-01 00:00:00-05:00",
                "2012-01-02 00:00:00-05:00",
                "2012-01-03 00:00:00-05:00",
            ],
            dtype="str",
        )
        tm.assert_series_equal(result, expected)

    def test_astype_datetime(self, unit):
        # [FUNCTION DOC]
        # Parametrized over datetime unit. Checks that all-NaT, single, and
        # NaN-containing datetime64 Series of that unit can be cast to object dtype, and
        # that assigning NaN into a datetime Series keeps its datetime64 dtype.
        ser = Series(iNaT, dtype=f"M8[{unit}]", index=range(5))

        ser = ser.astype("O")
        assert ser.dtype == np.object_

        ser = Series([datetime(2001, 1, 2, 0, 0)])

        ser = ser.astype("O")
        assert ser.dtype == np.object_

        ser = Series(
            [datetime(2001, 1, 2, 0, 0) for i in range(3)], dtype=f"M8[{unit}]"
        )

        ser[1] = np.nan
        assert ser.dtype == f"M8[{unit}]"

        ser = ser.astype("O")
        assert ser.dtype == np.object_

    def test_astype_datetime64tz(self):
        # [FUNCTION DOC]
        # Tz-aware (US/Eastern) datetime Series: casting to object matches
        # construction from objects; re-localizing the naive values round-trips to the
        # original; casting naive datetimes to a tz-aware dtype (by name or dtype object)
        # raises a type error recommending explicit localization; casting to another
        # zone (CET) converts the values to that zone's wall times.
        ser = Series(date_range("20130101", periods=3, tz="US/Eastern", unit="ns"))

        # astype
        result = ser.astype(object)
        expected = Series(ser.astype(object), dtype=object)
        tm.assert_series_equal(result, expected)

        result = Series(ser.values).dt.tz_localize("UTC").dt.tz_convert(ser.dt.tz)
        tm.assert_series_equal(result, ser)

        # astype - object, preserves on construction
        result = Series(ser.astype(object))
        expected = ser.astype(object)
        tm.assert_series_equal(result, expected)

        # astype - datetime64[ns, tz]
        msg = "Cannot use .astype to convert from timezone-naive"
        with pytest.raises(TypeError, match=msg):
            # dt64->dt64tz astype deprecated
            Series(ser.values).astype("datetime64[ns, US/Eastern]")

        with pytest.raises(TypeError, match=msg):
            # dt64->dt64tz astype deprecated
            Series(ser.values).astype(ser.dtype)

        result = ser.astype("datetime64[ns, CET]")
        expected = Series(
            date_range("20130101 06:00:00", periods=3, tz="CET", unit="ns")
        )
        tm.assert_series_equal(result, expected)

    def test_astype_str_cast_dt64(self):
        # [FUNCTION DOC]
        # Regression test (GH#9757): casting a naive timestamp Series and a tz-aware
        # (US/Eastern) one to str gives the date-only text and the text with UTC offset
        # respectively, as string dtype.
        # see GH#9757
        ts = Series([Timestamp("2010-01-04 00:00:00")])
        res = ts.astype(str)

        expected = Series(["2010-01-04"], dtype="str")
        tm.assert_series_equal(res, expected)

        ts = Series([Timestamp("2010-01-04 00:00:00", tz="US/Eastern")])
        res = ts.astype(str)

        expected = Series(["2010-01-04 00:00:00-05:00"], dtype="str")
        tm.assert_series_equal(res, expected)

    def test_astype_str_cast_td64(self):
        # [FUNCTION DOC]
        # Regression test (GH#9757): casting a one-day timedelta Series to str gives the
        # text "1 days" in a string-dtype Series.
        # see GH#9757

        td = Series([Timedelta(1, unit="D")])
        ser = td.astype(str)

        expected = Series(["1 days"], dtype="str")
        tm.assert_series_equal(ser, expected)

    def test_dt64_series_astype_object(self):
        # [FUNCTION DOC]
        # Casting a datetime64 Series to object dtype gives Python datetime elements and
        # object dtype.
        dt64ser = Series(date_range("20130101", periods=3))
        result = dt64ser.astype(object)
        assert isinstance(result.iloc[0], datetime)
        assert result.dtype == np.object_

    def test_td64_series_astype_object(self):
        # [FUNCTION DOC]
        # Casting a timedelta64 Series (including NaT) to object dtype gives Python
        # timedelta elements for the first entry and object dtype.
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="timedelta64[ns]")
        result = tdser.astype(object)
        assert isinstance(result.iloc[0], timedelta)
        assert result.dtype == np.object_

    @pytest.mark.parametrize(
        "data, dtype",
        [
            (["x", "y", "z"], "string[python]"),
            pytest.param(
                ["x", "y", "z"],
                "string[pyarrow]",
                marks=td.skip_if_no("pyarrow"),
            ),
            (["x", "y", "z"], "category"),
            (3 * [Timestamp("2020-01-01", tz="UTC")], None),
            (3 * [Interval(0, 1)], None),
        ],
    )
    @pytest.mark.parametrize("errors", ["raise", "ignore"])
    def test_astype_ignores_errors_for_extension_dtypes(self, data, dtype, errors):
        # [FUNCTION DOC]
        # Parametrized over several string/category/tz/interval datasets (pyarrow-backed
        # one skipped without pyarrow) and the errors option ("raise" or "ignore").
        # Casting to float with "ignore" returns the Series unchanged; with "raise" it
        # raises a value or type error whose message mentions failure to cast or convert
        # (GH 35471).
        # https://github.com/pandas-dev/pandas/issues/35471
        ser = Series(data, dtype=dtype)
        if errors == "ignore":
            expected = ser
            result = ser.astype(float, errors="ignore")
            tm.assert_series_equal(result, expected)
        else:
            msg = "(Cannot cast)|(could not convert)"
            with pytest.raises((ValueError, TypeError), match=msg):
                ser.astype(float, errors=errors)

    def test_astype_from_float_to_str(self, any_float_dtype):
        # [FUNCTION DOC]
        # Parametrized over float dtypes: casting a float Series to str gives the text
        # "0.1" in a string-dtype Series (GH 36451).
        # https://github.com/pandas-dev/pandas/issues/36451
        ser = Series([0.1], dtype=any_float_dtype)
        result = ser.astype(str)
        expected = Series(["0.1"], dtype="str")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "value, string_value",
        [
            (None, "None"),
            (np.nan, "nan"),
            (NA, "<NA>"),
        ],
    )
    def test_astype_to_str_preserves_na(self, value, string_value, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized over None, NaN and NA with their text forms. Casting an object
        # Series containing the missing value to str gives the text form of it, except
        # under the inferred-string option where it stays a missing value in the string
        # dtype result (GH 36904).
        # https://github.com/pandas-dev/pandas/issues/36904
        ser = Series(["a", "b", value], dtype=object)
        result = ser.astype(str)
        expected = Series(
            ["a", "b", None if using_infer_string else string_value], dtype="str"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "int32"])
    def test_astype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over four numeric dtypes: casting a random float Series gives the
        # requested dtype and preserves the Series name.
        ser = Series(np.random.default_rng(2).standard_normal(5), name="foo")
        as_typed = ser.astype(dtype)

        assert as_typed.dtype == dtype
        assert as_typed.name == ser.name

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_astype_cast_nan_inf_int(self, any_int_numpy_dtype, value):
        # [FUNCTION DOC]
        # Parametrized over NaN and infinity and every NumPy integer dtype: casting a
        # Series holding that value to an integer type raises a value error saying
        # non-finite values cannot be converted to integer (gh-14265).
        # gh-14265: check NaN and inf raise error when converting to int
        msg = "Cannot convert non-finite values \\(NA or inf\\) to integer"
        ser = Series([value])

        with pytest.raises(ValueError, match=msg):
            ser.astype(any_int_numpy_dtype)

    def test_astype_cast_object_int_fail(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Casting an object Series with a non-numeric string to a NumPy integer dtype
        # raises a value error from the failed integer parse of the string "car".
        arr = Series(["car", "house", "tree", "1"])
        msg = r"invalid literal for int\(\) with base 10: 'car'"
        with pytest.raises(ValueError, match=msg):
            arr.astype(any_int_numpy_dtype)

    def test_astype_float_to_uint_negatives_raise(
        self, float_numpy_dtype, any_unsigned_int_numpy_dtype
    ):
        # [FUNCTION DOC]
        # Regression test (GH#45151): casting a float Series that contains negative
        # values to any unsigned NumPy integer dtype raises a value error ("cannot
        # losslessly cast") for the Series, its DataFrame, an Index, and the underlying
        # array.
        # KNOWN ISSUE: a TODO in the test notes that the same check is missing for
        # extension float/unsigned dtypes and signed integers; only NumPy dtypes are
        # covered.
        # GH#45151 We don't cast negative numbers to nonsense values
        # TODO: same for EA float/uint dtypes, signed integers?
        arr = np.arange(5).astype(float_numpy_dtype) - 3  # includes negatives
        ser = Series(arr)

        msg = "Cannot losslessly cast from .* to .*"
        with pytest.raises(ValueError, match=msg):
            ser.astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            ser.to_frame().astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            # We currently catch and re-raise in Index.astype
            Index(ser).astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            ser.array.astype(any_unsigned_int_numpy_dtype)

    def test_astype_cast_object_int(self):
        # [FUNCTION DOC]
        # Casting an object Series of numeric strings to the Python int type gives an
        # integer Series 1 to 4.
        arr = Series(["1", "2", "3", "4"], dtype=object)
        result = arr.astype(int)

        tm.assert_series_equal(result, Series(np.arange(1, 5)))

    def test_astype_unicode(self, using_infer_string):
        # [FUNCTION DOC]
        # Casts Series of long ASCII/random strings and Japanese text to the NumPy string
        # class and compares against mapping the string conversion; when the default
        # encoding is utf-8, a bytes element is decoded rather than converted via its
        # repr (GH#45326).
        # KNOWN ISSUE: the former_encoding variable is always None, so the final block
        # that restores a default encoding is dead code, and the comment about
        # needing magic to set the default encoding is stale.
        # see GH#7758: A bit of magic is required to set
        # default encoding to utf-8
        digits = string.digits
        test_series = [
            Series([digits * 10, rand_str(63), rand_str(64), rand_str(1000)]),
            Series(["データーサイエンス、お前はもう死んでいる"]),
        ]

        former_encoding = None

        if sys.getdefaultencoding() == "utf-8":
            # GH#45326 as of 2.0 Series.astype matches Index.astype by handling
            #  bytes with obj.decode() instead of str(obj)
            item = "野菜食べないとやばい"
            ser = Series([item.encode()])
            result = ser.astype(np.str_)
            expected = Series([item], dtype=object)
            tm.assert_series_equal(result, expected)

        for ser in test_series:
            res = ser.astype(np.str_)
            expec = ser.map(str)
            if using_infer_string:
                expec = expec.astype(object)
            tm.assert_series_equal(res, expec)

        # Restore the former encoding
        if former_encoding is not None and former_encoding != "utf-8":
            reload(sys)
            sys.setdefaultencoding(former_encoding)

    def test_astype_bytes(self):
        # [FUNCTION DOC]
        # Regression test (GH#39474): casting a string Series to bytes gives a dtype
        # that is the 3-byte NumPy fixed-width byte string type.
        # GH#39474
        result = Series(["foo", "bar", "baz"]).astype(bytes)
        assert result.dtypes == np.dtype("S3")

    def test_astype_nan_to_bool(self):
        # [FUNCTION DOC]
        # Regression test (GH#43018): casting an object Series holding NaN to bool gives
        # True (NaN is truthy).
        # GH#43018
        ser = Series(np.nan, dtype="object")
        result = ser.astype("bool")
        expected = Series(True, dtype="bool")
        tm.assert_series_equal(result, expected)

    def test_astype_ea_to_datetimetzdtype(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Parametrized over nullable numeric extension dtypes: casting integer values
        # to a tz-aware (US/Pacific) datetime dtype interprets the integers as
        # nanoseconds since the epoch and gives tz-aware timestamps (GH37553).
        # GH37553
        ser = Series([4, 0, 9], dtype=any_numeric_ea_dtype)
        result = ser.astype(DatetimeTZDtype(tz="US/Pacific"))

        expected = Series(
            {
                0: Timestamp("1969-12-31 16:00:00.000000004-08:00", tz="US/Pacific"),
                1: Timestamp("1969-12-31 16:00:00.000000000-08:00", tz="US/Pacific"),
                2: Timestamp("1969-12-31 16:00:00.000000009-08:00", tz="US/Pacific"),
            }
        )

        tm.assert_series_equal(result, expected)

    def test_astype_retain_attrs(self, any_numpy_dtype):
        # [FUNCTION DOC]
        # Parametrized over NumPy dtypes: the attrs dictionary of the result of astype
        # equals the attrs of the source Series (GH#44414).
        # GH#44414
        ser = Series([0, 1, 2, 3])
        ser.attrs["Location"] = "Michigan"

        result = ser.astype(any_numpy_dtype).attrs
        expected = ser.attrs

        tm.assert_dict_equal(expected, result)


class TestAstypeString:
    # [CLASS DOC]
    # Tests for round trips from string dtypes back to other extension dtypes.
    # Stateless.
    @pytest.mark.parametrize(
        "data, dtype",
        [
            ([True, NA], "boolean"),
            (["A", NA], "category"),
            (["2020-10-10", "2020-10-10"], "datetime64[ns]"),
            (["2020-10-10", "2020-10-10", NaT], "datetime64[ns]"),
            (
                ["2012-01-01 00:00:00-05:00", NaT],
                "datetime64[ns, US/Eastern]",
            ),
            ([1, None], "UInt16"),
            (["1/1/2021", "2/1/2021"], "period[M]"),
            (["1/1/2021", "2/1/2021", NaT], "period[M]"),
            (["1 Day", "59 Days", NaT], "timedelta64[us]"),
            # currently no way to parse IntervalArray from a list of strings
        ],
    )
    def test_astype_string_to_extension_dtype_roundtrip(
        self, data, dtype, request, nullable_string_dtype
    ):
        # [FUNCTION DOC]
        # Parametrized over data/dtype pairs (boolean, category, datetime, tz-aware,
        # nullable integer, period, timedelta) and the nullable_string_dtype fixture.
        # Converts the Series to the string dtype and back to the original dtype and expects
        # equality (GH-40351). The boolean case is marked as an expected failure because
        # of missing-value handling in string array casting (GH40566).
        if dtype == "boolean":
            mark = pytest.mark.xfail(
                reason="TODO StringArray.astype() with missing values #GH40566"
            )
            request.applymarker(mark)
        # GH-40351
        ser = Series(data, dtype=dtype)

        # Note: just passing .astype(dtype) fails for dtype="category"
        #  with bc ser.dtype.categories will be object dtype whereas
        #  result.dtype.categories will have string dtype
        result = ser.astype(nullable_string_dtype).astype(ser.dtype)
        tm.assert_series_equal(result, ser)


class TestAstypeCategorical:
    # [CLASS DOC]
    # Tests of Series.astype involving categorical dtype. Stateless.
    def test_astype_categorical_to_other(self):
        # [FUNCTION DOC]
        # Builds a categorical Series by cutting random integers into bins. Checks that
        # casting to category (by name or dtype) is the identity, that float conversion
        # raises a value error, that conversions to str, int and object preserve values,
        # that array conversion gives the same values, and that a round trip through object
        # gives a categorical with sorted categories and unused ones removed.
        cat = Categorical([f"{i} - {i + 499}" for i in range(0, 10000, 500)])
        ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)

        expected = ser
        tm.assert_series_equal(ser.astype("category"), expected)
        tm.assert_series_equal(ser.astype(CategoricalDtype()), expected)
        msg = r"Cannot cast object|str dtype to float64"
        with pytest.raises(ValueError, match=msg):
            ser.astype("float64")

        cat = Series(Categorical(["a", "b", "b", "a", "a", "c", "c", "c"]))
        exp = Series(["a", "b", "b", "a", "a", "c", "c", "c"], dtype="str")
        tm.assert_series_equal(cat.astype("str"), exp)
        s2 = Series(Categorical(["1", "2", "3", "4"]))
        exp2 = Series([1, 2, 3, 4]).astype("int")
        tm.assert_series_equal(s2.astype("int"), exp2)

        # object don't sort correctly, so just compare that we have the same
        # values
        def cmp(a, b):
            # [FUNCTION DOC]
            # HELPER: compares two inputs by sorting their unique values and asserting they
            # are nearly equal, used because object arrays do not sort consistently.
            tm.assert_almost_equal(np.sort(np.unique(a)), np.sort(np.unique(b)))

        expected = Series(np.array(ser.values), name="value_group")
        cmp(ser.astype("object"), expected)
        cmp(ser.astype(np.object_), expected)

        # array conversion
        tm.assert_almost_equal(np.array(ser), np.array(ser.values))

        tm.assert_series_equal(ser.astype("category"), ser)
        tm.assert_series_equal(ser.astype(CategoricalDtype()), ser)

        roundtrip_expected = ser.cat.set_categories(
            ser.cat.categories.sort_values()
        ).cat.remove_unused_categories()
        result = ser.astype("object").astype("category")
        tm.assert_series_equal(result, roundtrip_expected)
        result = ser.astype("object").astype(CategoricalDtype())
        tm.assert_series_equal(result, roundtrip_expected)

    def test_astype_categorical_invalid_conversions(self):
        # [FUNCTION DOC]
        # Casting to the Categorical class itself (not a dtype) raises a type error
        # that the dtype is not understood, both directly and after casting to object.
        # invalid conversion (these are NOT a dtype)
        cat = Categorical([f"{i} - {i + 499}" for i in range(0, 10000, 500)])
        ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)

        msg = "dtype '<class 'pandas.Categorical'>' not understood"
        with pytest.raises(TypeError, match=msg):
            ser.astype(Categorical)
        with pytest.raises(TypeError, match=msg):
            ser.astype("object").astype(Categorical)

    def test_astype_categoricaldtype(self):
        # [FUNCTION DOC]
        # Casting a string Series to explicit categorical dtypes: ordered or unordered,
        # with the given categories or an additional unused category, producing the
        # expected categorical and categories index.
        ser = Series(["a", "b", "a"])
        result = ser.astype(CategoricalDtype(["a", "b"], ordered=True))
        expected = Series(Categorical(["a", "b", "a"], ordered=True))
        tm.assert_series_equal(result, expected)

        result = ser.astype(CategoricalDtype(["a", "b"], ordered=False))
        expected = Series(Categorical(["a", "b", "a"], ordered=False))
        tm.assert_series_equal(result, expected)

        result = ser.astype(CategoricalDtype(["a", "b", "c"], ordered=False))
        expected = Series(
            Categorical(["a", "b", "a"], categories=["a", "b", "c"], ordered=False)
        )
        tm.assert_series_equal(result, expected)
        tm.assert_index_equal(result.cat.categories, Index(["a", "b", "c"]))

    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("dtype_ordered", [True, False])
    @pytest.mark.parametrize("series_ordered", [True, False])
    def test_astype_categorical_to_categorical(
        self, name, dtype_ordered, series_ordered
    ):
        # [FUNCTION DOC]
        # Parametrized over name, whether the target dtype is ordered, and whether the
        # source is ordered. Casting a categorical Series to a dtype without categories
        # keeps the categories and applies the ordered flag; casting to a dtype with
        # different categories emits a Pandas4Warning about constructing a categorical
        # with values not in the dtype (GH#10696, GH#18593) and matches the equivalent
        # construction; casting to category leaves the Series unchanged.
        # GH#10696, GH#18593
        s_data = list("abcaacbab")
        s_dtype = CategoricalDtype(list("bac"), ordered=series_ordered)
        ser = Series(s_data, dtype=s_dtype, name=name)

        # unspecified categories
        dtype = CategoricalDtype(ordered=dtype_ordered)
        result = ser.astype(dtype)
        exp_dtype = CategoricalDtype(s_dtype.categories, dtype_ordered)
        expected = Series(s_data, name=name, dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

        # different categories
        dtype = CategoricalDtype(list("adc"), dtype_ordered)
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = ser.astype(dtype)
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            expected = Series(s_data, name=name, dtype=dtype)
        tm.assert_series_equal(result, expected)

        if dtype_ordered is False:
            # not specifying ordered, so only test once
            expected = ser
            result = ser.astype("category")
            tm.assert_series_equal(result, expected)

    def test_astype_bool_missing_to_categorical(self):
        # [FUNCTION DOC]
        # Regression test (GH-19182): an object Series of booleans with a missing value
        # cast to a categorical dtype with boolean categories keeps the missing value.
        # GH-19182
        ser = Series([True, False, np.nan])
        assert ser.dtypes == np.object_

        result = ser.astype(CategoricalDtype(categories=[True, False]))
        expected = Series(Categorical([True, False, np.nan], categories=[True, False]))
        tm.assert_series_equal(result, expected)

    def test_astype_categories_raises(self):
        # [FUNCTION DOC]
        # Passing categories and ordered keyword arguments directly to astype raises a
        # type error about unexpected arguments (removed feature).
        # deprecated GH#17636, removed in GH#27141
        ser = Series(["a", "b", "a"])
        with pytest.raises(TypeError, match="got an unexpected"):
            ser.astype("category", categories=["a", "b"], ordered=True)

    @pytest.mark.parametrize("items", [["a", "b", "c", "a"], [1, 2, 3, 1]])
    def test_astype_from_categorical(self, items):
        # [FUNCTION DOC]
        # Parametrized over string and integer lists with repeats: casting to category
        # equals a Series built from a categorical of the same data.
        ser = Series(items)
        exp = Series(Categorical(items))
        res = ser.astype("category")
        tm.assert_series_equal(res, exp)

    def test_astype_from_categorical_with_keywords(self):
        # [FUNCTION DOC]
        # Casting to a categorical dtype with ordered set, with and without an explicit
        # list of categories, gives the matching ordered categorical Series.
        # with keywords
        lst = ["a", "b", "c", "a"]
        ser = Series(lst)
        exp = Series(Categorical(lst, ordered=True))
        res = ser.astype(CategoricalDtype(None, ordered=True))
        tm.assert_series_equal(res, exp)

        exp = Series(Categorical(lst, categories=list("abcdef"), ordered=True))
        res = ser.astype(CategoricalDtype(list("abcdef"), ordered=True))
        tm.assert_series_equal(res, exp)

    def test_astype_timedelta64_with_np_nan(self):
        # [FUNCTION DOC]
        # Regression test (GH45798): constructing a timedelta64 Series with a Timedelta and
        # NaN gives NaT for the NaN.
        # GH45798
        result = Series([Timedelta(1), np.nan], dtype="timedelta64[ns]")
        expected = Series([Timedelta(1), NaT], dtype="timedelta64[ns]")
        tm.assert_series_equal(result, expected)

    @td.skip_if_no("pyarrow")
    def test_astype_int_na_string(self):
        # [FUNCTION DOC]
        # Regression test (GH#57418), requires pyarrow: casting an Int64 pyarrow-backed
        # Series with a missing value to the pyarrow string dtype gives a string Series
        # with the text of the number and the missing value preserved.
        # GH#57418
        ser = Series([12, NA], dtype="Int64[pyarrow]")
        result = ser.astype("string[pyarrow]")
        expected = Series(["12", NA], dtype="string[pyarrow]")
        tm.assert_series_equal(result, expected)
