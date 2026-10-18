"""
The functions benchmarked in this file depend _almost_ exclusively on
_libs, but not in a way that is easy to formalize.

If a PR does not change anything in pandas/_libs/ or pandas/core/tools/, then
it is likely that these benchmarks will be unaffected.
"""

# [MODULE DOC]
# Benchmarks for type inference and conversion routines: numeric, datetime and
# timedelta conversion from strings and numbers, and two low-level conversion
# helpers from the compiled library. Public contents are the benchmark classes
# ToNumeric, ToNumericDowncast, MaybeConvertNumeric, MaybeConvertObjects,
# ToDatetimeFromIntsFloats, ToDatetimeYYYYMMDD, ToDatetimeCacheSmallCount,
# ToDatetimeISO8601, ToDatetimeNONISO8601, ToDatetimeFormatQuarters,
# ToDatetimeFormat, ToDatetimeCache, ToTimedelta and ToTimedeltaErrors. Takes the
# compiled library handle (lib) from the shared helper module. The final line
# re-exports the common asv setup hook. Class bodies build large lists and arrays
# at import time (see ToNumericDowncast).
import numpy as np

from pandas import (
    Index,
    NaT,
    Series,
    date_range,
    to_datetime,
    to_numeric,
    to_timedelta,
)

from .pandas_vb_common import lib


class ToNumeric:
    # [CLASS DOC]
    # Benchmark of numeric conversion with errors coerced, for float, numeric
    # string and non-numeric string series. No parameters. State: float, numstr
    # and str series.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a float series of 10000 random normal values, its string form
        # (numstr) and a series of 10000 non-numeric strings (str).
        N = 10000
        self.float = Series(np.random.randn(N))
        self.numstr = self.float.astype("str")
        self.str = Series(Index([f"i-{i}" for i in range(N)], dtype=object))

    def time_from_float(self):
        # [FUNCTION DOC]
        # Times converting the float series, coercing failures to missing.
        to_numeric(self.float, errors="coerce")

    def time_from_numeric_str(self):
        # [FUNCTION DOC]
        # Times converting the numeric string series, coercing failures.
        to_numeric(self.numstr, errors="coerce")

    def time_from_str(self):
        # [FUNCTION DOC]
        # Times converting the non-numeric string series; every element fails and
        # is coerced to missing.
        to_numeric(self.str, errors="coerce")


class ToNumericDowncast:
    # [CLASS DOC]
    # Benchmark of numeric conversion with the optional downcast setting
    # (none, integer, signed, unsigned, float) on six input kinds. Parametrized
    # by input kind and downcast. The class attributes N (500000), N2 (half of N)
    # and the lookup table data_dict of ready-made inputs are built when the module
    # is imported (large lists and arrays, memory-heavy import). State: data.
    param_names = ["dtype", "downcast"]
    params = [
        [
            "string-float",
            "string-int",
            "string-nint",
            "datetime64",
            "int-list",
            "int32",
        ],
        [None, "integer", "signed", "unsigned", "float"],
    ]

    N = 500000
    N2 = N // 2

    data_dict = {
        "string-int": ["1"] * N2 + [2] * N2,
        "string-nint": ["-1"] * N2 + [2] * N2,
        "datetime64": np.repeat(
            np.array(["1970-01-01", "1970-01-02"], dtype="datetime64[D]"), N
        ),
        "string-float": ["1.1"] * N2 + [2] * N2,
        "int-list": [1] * N2 + [2] * N2,
        "int32": np.repeat(np.int32(1), N),
    }

    def setup(self, dtype, downcast):
        # [FUNCTION DOC]
        # Selects the prepared input of the given kind from the class lookup table
        # and stores it as data. Raises a key error for an unknown kind. The
        # downcast parameter is unused here.
        self.data = self.data_dict[dtype]

    def time_downcast(self, dtype, downcast):
        # [FUNCTION DOC]
        # Times numeric conversion of the data with the given downcast option.
        to_numeric(self.data, downcast=downcast)


class MaybeConvertNumeric:
    # [CLASS DOC]
    # Benchmark of the low-level numeric conversion helper on a large object array
    # of big unsigned values, half of them as strings. It depends only on the
    # compiled library and could live in the libs benchmark module. Data is built
    # once by the cache hook and passed to the timing method.
    # maybe_convert_numeric depends _exclusively_ on _libs, could
    #  go in benchmarks/libs.py

    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns an object array of one million elements: values
        # of two to the 63rd plus consecutive offsets as unsigned integers, with
        # every second element converted to its string form and the last element
        # replaced by minus one (forcing a mixed-sign case).
        N = 10**6
        arr = np.repeat([2**63], N) + np.arange(N).astype("uint64")
        data = arr.astype(object)
        data[1::2] = arr[1::2].astype(str)
        data[-1] = -1
        return data

    def time_convert(self, data):
        # [FUNCTION DOC]
        # Times the helper on the data with an empty set of extra missing-value
        # markers and numeric coercion disabled.
        lib.maybe_convert_numeric(data, set(), coerce_numeric=False)


class MaybeConvertObjects:
    # [CLASS DOC]
    # Benchmark of the low-level object-inference helper on an array that holds
    # integers and one not-a-time marker. It depends almost exclusively on the
    # compiled library but has some run-time imports from elsewhere. No
    # parameters. State: data.
    # maybe_convert_objects depends _almost_ exclusively on _libs, but
    #  does have some run-time imports from outside of _libs

    def setup(self):
        # [FUNCTION DOC]
        # Builds a list of the integers below 100000 with the first element
        # replaced by the not-a-time marker, converts it to an array (object dtype,
        # since the elements are mixed) and stores it as data.
        N = 10**5

        data = list(range(N))
        data[0] = NaT
        data = np.array(data)
        self.data = data

    def time_maybe_convert_objects(self):
        # [FUNCTION DOC]
        # Times the helper on the data array.
        lib.maybe_convert_objects(self.data)


class ToDatetimeFromIntsFloats:
    # [CLASS DOC]
    # Benchmark of converting integer, unsigned integer and float epoch values to
    # datetimes, in nanosecond and second units; the speeds of the three numeric
    # paths should be comparable. No parameters. State: ts_sec, ts_sec_uint,
    # ts_sec_float, ts_nanosec, ts_nanosec_uint, ts_nanosec_float.
    def setup(self):
        # [FUNCTION DOC]
        # Builds second-resolution epoch series covering one week (604800 values
        # from 1521080307) as int64, uint64 and float64, and nanosecond versions
        # (the second values multiplied by one million) in the same three types.
        self.ts_sec = Series(range(1521080307, 1521685107), dtype="int64")
        self.ts_sec_uint = Series(range(1521080307, 1521685107), dtype="uint64")
        self.ts_sec_float = self.ts_sec.astype("float64")

        self.ts_nanosec = 1_000_000 * self.ts_sec
        self.ts_nanosec_uint = 1_000_000 * self.ts_sec_uint
        self.ts_nanosec_float = self.ts_nanosec.astype("float64")

    # speed of int64, uint64 and float64 paths should be comparable

    def time_nanosec_int64(self):
        # [FUNCTION DOC]
        # Times datetime conversion of the int64 values in nanosecond unit.
        to_datetime(self.ts_nanosec, unit="ns")

    def time_nanosec_uint64(self):
        # [FUNCTION DOC]
        # Times datetime conversion of the uint64 values in nanosecond unit.
        to_datetime(self.ts_nanosec_uint, unit="ns")

    def time_nanosec_float64(self):
        # [FUNCTION DOC]
        # Times datetime conversion of the float64 values in nanosecond unit.
        to_datetime(self.ts_nanosec_float, unit="ns")

    def time_sec_uint64(self):
        # [FUNCTION DOC]
        # Times datetime conversion of the uint64 values in second unit.
        to_datetime(self.ts_sec_uint, unit="s")

    def time_sec_int64(self):
        # [FUNCTION DOC]
        # Times datetime conversion of the int64 values in second unit.
        to_datetime(self.ts_sec, unit="s")

    def time_sec_float64(self):
        # [FUNCTION DOC]
        # Times datetime conversion of the float64 values in second unit.
        to_datetime(self.ts_sec_float, unit="s")


class ToDatetimeYYYYMMDD:
    # [CLASS DOC]
    # Benchmark of parsing compact year-month-day strings with an explicit
    # format. No parameters. State: stringsD.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 10000 daily dates from 2000-01-01, formats them without
        # separators as a string series stringsD.
        rng = date_range(start="1/1/2000", periods=10000, freq="D")
        self.stringsD = Series(rng.strftime("%Y%m%d"))

    def time_format_YYYYMMDD(self):
        # [FUNCTION DOC]
        # Times parsing the strings with the compact format.
        to_datetime(self.stringsD, format="%Y%m%d")


class ToDatetimeCacheSmallCount:
    # [CLASS DOC]
    # Benchmark of date-string parsing with the conversion cache switched on or
    # off, for small and large numbers of unique strings. Parametrized by cache
    # flag and count. State: unique_date_strings.
    params = ([True, False], [50, 500, 5000, 100000])
    param_names = ["cache", "count"]

    def setup(self, cache, count):
        # [FUNCTION DOC]
        # Builds count consecutive daily dates from 1971-01-01 and formats them as
        # year-month-day strings in a list. The cache flag is unused here.
        rng = date_range(start="1/1/1971", periods=count)
        self.unique_date_strings = rng.strftime("%Y-%m-%d").tolist()

    def time_unique_date_strings(self, cache, count):
        # [FUNCTION DOC]
        # Times parsing the list of unique date strings with the cache option.
        to_datetime(self.unique_date_strings, cache=cache)


class ToDatetimeISO8601:
    # [CLASS DOC]
    # Benchmark of parsing ISO 8601 style timestamp strings in several layouts,
    # with and without an explicit format. No parameters. State: strings,
    # strings_nosep, strings_tz_space and strings_zero_tz.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 20000 hourly timestamps from 2000-01-01 and formats them four
        # ways: with separators (strings), with a compact date part (strings_nosep),
        # with a trailing space and minus-eight-hour offset (strings_tz_space), and
        # with a trailing Z for UTC (strings_zero_tz).
        rng = date_range(start="1/1/2000", periods=20000, freq="h")
        self.strings = rng.strftime("%Y-%m-%d %H:%M:%S").tolist()
        self.strings_nosep = rng.strftime("%Y%m%d %H:%M:%S").tolist()
        self.strings_tz_space = [
            x.strftime("%Y-%m-%d %H:%M:%S") + " -0800" for x in rng
        ]
        self.strings_zero_tz = [x.strftime("%Y-%m-%d %H:%M:%S") + "Z" for x in rng]

    def time_iso8601(self):
        # [FUNCTION DOC]
        # Times parsing the separated strings with format inference.
        to_datetime(self.strings)

    def time_iso8601_nosep(self):
        # [FUNCTION DOC]
        # Times parsing the compact-date strings with format inference.
        to_datetime(self.strings_nosep)

    def time_iso8601_format(self):
        # [FUNCTION DOC]
        # Times parsing the separated strings with an explicit format.
        to_datetime(self.strings, format="%Y-%m-%d %H:%M:%S")

    def time_iso8601_format_no_sep(self):
        # [FUNCTION DOC]
        # Times parsing the compact-date strings with an explicit format.
        to_datetime(self.strings_nosep, format="%Y%m%d %H:%M:%S")

    def time_iso8601_tz_spaceformat(self):
        # [FUNCTION DOC]
        # Times parsing the strings that end with a space and a numeric offset.
        to_datetime(self.strings_tz_space)

    def time_iso8601_infer_zero_tz_fromat(self):
        # [FUNCTION DOC]
        # Times parsing strings that end with the Z suffix with format inference
        # (issue 41047). The method name contains a misspelling of the word format
        # but is harmless.
        # GH 41047
        to_datetime(self.strings_zero_tz)


class ToDatetimeNONISO8601:
    # [CLASS DOC]
    # Benchmark of parsing non-ISO month-name timestamp strings with a numeric UTC
    # offset, with the same offset everywhere or two different offsets. No
    # parameters. State: same_offset and diff_offset.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a list of 10000 copies of one string (March 1, 2018 12:00:00 at
        # offset +0400) as same_offset, and a list with half of those and half with
        # offset +0500 as diff_offset.
        N = 10000
        half = N // 2
        ts_string_1 = "March 1, 2018 12:00:00+0400"
        ts_string_2 = "March 1, 2018 12:00:00+0500"
        self.same_offset = [ts_string_1] * N
        self.diff_offset = [ts_string_1] * half + [ts_string_2] * half

    def time_same_offset(self):
        # [FUNCTION DOC]
        # Times parsing the strings that share one offset.
        to_datetime(self.same_offset)

    def time_different_offset(self):
        # [FUNCTION DOC]
        # Times parsing the strings with two different offsets, converting to UTC.
        to_datetime(self.diff_offset, utc=True)


class ToDatetimeFormatQuarters:
    # [CLASS DOC]
    # Benchmark of inferring and parsing quarter-notation strings. No parameters.
    # State: s.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a series of 40000 strings cycling through four quarter notations
        # (2Q2005, 2Q05, 2005Q1, 05Q1) as s.
        self.s = Series(["2Q2005", "2Q05", "2005Q1", "05Q1"] * 10000)

    def time_infer_quarter(self):
        # [FUNCTION DOC]
        # Times parsing the series with format inference.
        to_datetime(self.s)


class ToDatetimeFormat:
    # [CLASS DOC]
    # Benchmark of parsing with explicit formats: day-month-year with month
    # abbreviations (exact and non-exact matching) and a format with a numeric
    # offset, same or different offsets, optionally converting to UTC. No
    # parameters. State: s, s2, same_offset, diff_offset.
    def setup(self):
        # [FUNCTION DOC]
        # With N equal to 100000: builds s as strings alternating a date (19MAY11)
        # and a date with a time suffix (19MAY11:00:00:00); s2 as s with the time
        # suffix stripped by a regular expression; same_offset as N copies of a
        # string with a minus-seven-hour colon offset; and diff_offset as N
        # strings cycling through ten offsets (minus zero to minus nine hours).
        N = 100000
        self.s = Series(["19MAY11", "19MAY11:00:00:00"] * N)
        self.s2 = self.s.str.replace(":\\S+$", "", regex=True)

        self.same_offset = ["10/11/2018 00:00:00.045-07:00"] * N
        self.diff_offset = [
            f"10/11/2018 00:00:00.045-0{offset}:00" for offset in range(10)
        ] * (N // 10)

    def time_exact(self):
        # [FUNCTION DOC]
        # Times parsing the cleaned strings with the day-month-abbreviation-year
        # format (exact matching).
        to_datetime(self.s2, format="%d%b%y")

    def time_no_exact(self):
        # [FUNCTION DOC]
        # Times parsing the original strings with the same format, allowing the
        # format to match only part of each string.
        to_datetime(self.s, format="%d%b%y", exact=False)

    def time_same_offset(self):
        # [FUNCTION DOC]
        # Times parsing the strings that share one offset with an explicit format.
        to_datetime(self.same_offset, format="%m/%d/%Y %H:%M:%S.%f%z")

    def time_same_offset_to_utc(self):
        # [FUNCTION DOC]
        # Times parsing the same-offset strings with an explicit format and
        # converting to UTC.
        to_datetime(self.same_offset, format="%m/%d/%Y %H:%M:%S.%f%z", utc=True)

    def time_different_offset_to_utc(self):
        # [FUNCTION DOC]
        # Times parsing the strings with ten different offsets with an explicit
        # format, converting to UTC.
        to_datetime(self.diff_offset, format="%m/%d/%Y %H:%M:%S.%f%z", utc=True)


class ToDatetimeCache:
    # [CLASS DOC]
    # Benchmark of parsing lists with many duplicate values, with the conversion
    # cache on or off. Parametrized by cache flag. State: unique_numeric_seconds,
    # dup_numeric_seconds, dup_string_dates and dup_string_with_tz.
    params = [True, False]
    param_names = ["cache"]

    def setup(self, cache):
        # [FUNCTION DOC]
        # Builds lists of 10000 entries: the integers 0 to 9999; the integer 1000
        # repeated; the string 2000-02-11 repeated; and the string
        # 2000-02-11 15:00:00-0800 repeated. The cache flag is unused here.
        N = 10000
        self.unique_numeric_seconds = list(range(N))
        self.dup_numeric_seconds = [1000] * N
        self.dup_string_dates = ["2000-02-11"] * N
        self.dup_string_with_tz = ["2000-02-11 15:00:00-0800"] * N

    def time_unique_seconds_and_unit(self, cache):
        # [FUNCTION DOC]
        # Times converting the distinct integers as seconds since the epoch.
        to_datetime(self.unique_numeric_seconds, unit="s", cache=cache)

    def time_dup_seconds_and_unit(self, cache):
        # [FUNCTION DOC]
        # Times converting the repeated integer as seconds since the epoch.
        to_datetime(self.dup_numeric_seconds, unit="s", cache=cache)

    def time_dup_string_dates(self, cache):
        # [FUNCTION DOC]
        # Times parsing the repeated date string with format inference.
        to_datetime(self.dup_string_dates, cache=cache)

    def time_dup_string_dates_and_format(self, cache):
        # [FUNCTION DOC]
        # Times parsing the repeated date string with an explicit format.
        to_datetime(self.dup_string_dates, format="%Y-%m-%d", cache=cache)

    def time_dup_string_tzoffset_dates(self, cache):
        # [FUNCTION DOC]
        # Times parsing the repeated string that carries a timezone offset.
        to_datetime(self.dup_string_with_tz, cache=cache)


class ToTimedelta:
    # [CLASS DOC]
    # Benchmark of converting integers and strings to timedeltas. No parameters.
    # State: ints, str_days and str_seconds.
    def setup(self):
        # [FUNCTION DOC]
        # Draws 10000 random integers below 60 as ints and builds two string
        # lists from them: a day-count form and a clock form with the integer as
        # zero-padded seconds.
        self.ints = np.random.randint(0, 60, size=10000)
        self.str_days = []
        self.str_seconds = []
        for i in self.ints:
            self.str_days.append(f"{i} days")
            self.str_seconds.append(f"00:00:{i:02d}")

    def time_convert_int(self):
        # [FUNCTION DOC]
        # Times converting the integers as a count of seconds.
        to_timedelta(self.ints, unit="s")

    def time_convert_string_days(self):
        # [FUNCTION DOC]
        # Times converting the day-count strings.
        to_timedelta(self.str_days)

    def time_convert_string_seconds(self):
        # [FUNCTION DOC]
        # Times converting the clock-form strings.
        to_timedelta(self.str_seconds)


class ToTimedeltaErrors:
    # [CLASS DOC]
    # Benchmark of timedelta conversion where one invalid string forces the error
    # handling path. No parameters. State: arr.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a list of 10000 day-count strings from random integers below 60
        # and replaces the last one with the invalid text apple. Stores as arr.
        ints = np.random.randint(0, 60, size=10000)
        self.arr = [f"{i} days" for i in ints]
        self.arr[-1] = "apple"

    def time_convert(self):
        # [FUNCTION DOC]
        # Times converting the list, coercing the invalid entry to missing.
        to_timedelta(self.arr, errors="coerce")


from .pandas_vb_common import setup  # noqa: F401 isort:skip
