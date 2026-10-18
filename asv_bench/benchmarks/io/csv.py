# [MODULE DOC]
# Benchmarks for comma-separated text writing and reading. Writing: frames of
# various shapes and dtypes, float formatting variants, multi-index writing with
# unused levels, datetime/period formatting and index layouts. Reading: datetime
# parsing and caching, uint64 handling, row skipping, thousands separators,
# comments, float precision, engine comparison (c, python, pyarrow), categorical
# conversion, memory growth with chunking, special date formats, memory-mapped
# UTF-8, index columns and low-memory parsing of very large input. Public
# contents are the write classes ToCSV, ToCSVFloatFormatVariants,
# ToCSVMultiIndexUnusedLevels, ToCSVDatetime, ToCSVDatetimeIndex, ToCSVPeriod,
# ToCSVPeriodIndex, ToCSVDatetimeBig, ToCSVIndexes and the read classes
# StringIORewind (a helper base), ReadCSVDInferDatetimeFormat,
# ReadCSVConcatDatetime, ReadCSVConcatDatetimeBadDateValue, ReadCSVSkipRows,
# ReadUint64Integers, ReadCSVThousands, ReadCSVComment, ReadCSVFloatPrecision,
# ReadCSVEngine, ReadCSVCategorical, ReadCSVParseDates, ReadCSVCachedParseDates,
# ReadCSVMemoryGrowth, ReadCSVParseSpecialDate, ReadCSVMemMapUTF8,
# ParseDateComparison, ReadCSVIndexCol, ReadCSVDatePyarrowEngine and
# ReadCSVCParserLowMemory. File-based classes derive from the shared file
# benchmark base class and use the file name __test__.csv in the working
# directory. Final line re-exports the common asv setup hook.
from io import (
    BytesIO,
    StringIO,
)
import random
import string

import numpy as np

from pandas import (
    Categorical,
    DataFrame,
    Index,
    concat,
    date_range,
    period_range,
    read_csv,
    to_datetime,
)

from ..pandas_vb_common import BaseIO


class ToCSV(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing a frame to a comma-separated file for a wide float
    # frame, a long numeric frame and a mixed-dtype frame. Inherits file cleanup
    # from the base class (the file is removed after each run). Parametrized by
    # kind. State: df.
    fname = "__test__.csv"
    params = ["wide", "long", "mixed"]
    param_names = ["kind"]

    def setup(self, kind):
        # [FUNCTION DOC]
        # Builds three frames: wide (3000 by 30 random normal), long (50000 rows,
        # four columns of consecutive numbers with float offsets) and mixed (5000
        # rows with a float column that has a block of missing values, an integer
        # column from truncated normals, a boolean column, a per-second datetime
        # column and a constant string column). Stores the one for kind as df.
        wide_frame = DataFrame(np.random.randn(3000, 30))
        long_frame = DataFrame(
            {
                "A": np.arange(50000),
                "B": np.arange(50000) + 1.0,
                "C": np.arange(50000) + 2.0,
                "D": np.arange(50000) + 3.0,
            }
        )
        mixed_frame = DataFrame(
            {
                "float": np.random.randn(5000),
                "int": np.random.randn(5000).astype(int),
                "bool": (np.arange(5000) % 2) == 0,
                "datetime": date_range("2001", freq="s", periods=5000),
                "object": ["foo"] * 5000,
            }
        )
        mixed_frame.loc[30:500, "float"] = np.nan
        data = {"wide": wide_frame, "long": long_frame, "mixed": mixed_frame}
        self.df = data[kind]

    def time_frame(self, kind):
        # [FUNCTION DOC]
        # Times writing the frame to the file (side effect: creates the file).
        self.df.to_csv(self.fname)


class ToCSVFloatFormatVariants(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing a 1000 by 1000 float frame with different float
    # formatting specifications. State: df (seeded generator).
    fname = "__test__.csv"

    def setup(self):
        # [FUNCTION DOC]
        # Builds a 1000 by 1000 frame of uniform random floats from a generator
        # with fixed seed 42, stored as df.
        self.df = DataFrame(np.random.default_rng(seed=42).random((1000, 1000)))

    def time_old_style_percent_format(self):
        # [FUNCTION DOC]
        # Times writing with a percent-style format of six decimals.
        self.df.to_csv(self.fname, float_format="%.6f")

    def time_new_style_brace_format(self):
        # [FUNCTION DOC]
        # Times writing with a brace-style format of six decimals.
        self.df.to_csv(self.fname, float_format="{:.6f}")

    def time_new_style_thousands_format(self):
        # [FUNCTION DOC]
        # Times writing with a brace-style format with thousands separators and two
        # decimals.
        self.df.to_csv(self.fname, float_format="{:,.2f}")

    def time_callable_format(self):
        # [FUNCTION DOC]
        # Times writing with a function that formats each float with six decimals.
        self.df.to_csv(self.fname, float_format=lambda x: f"{x:.6f}")


class ToCSVMultiIndexUnusedLevels(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing a two-level multi-indexed frame, its slice (which keeps
    # unused index levels) and a single-index slice. State: df, df_unused_levels
    # and df_single_index.
    fname = "__test__.csv"

    def setup(self):
        # [FUNCTION DOC]
        # Builds a frame of 100000 rows with a random column a and constant columns
        # b and c, indexed by a and b as df; df_unused_levels is the first 10000
        # rows of df (levels still hold all values); df_single_index is the first
        # 10000 rows of the frame indexed by a only.
        df = DataFrame({"a": np.random.randn(100_000), "b": 1, "c": 1})
        self.df = df.set_index(["a", "b"])
        self.df_unused_levels = self.df.iloc[:10_000]
        self.df_single_index = df.set_index(["a"]).iloc[:10_000]

    def time_full_frame(self):
        # [FUNCTION DOC]
        # Times writing the full multi-indexed frame.
        self.df.to_csv(self.fname)

    def time_sliced_frame(self):
        # [FUNCTION DOC]
        # Times writing the slice that retains unused levels.
        self.df_unused_levels.to_csv(self.fname)

    def time_single_index_frame(self):
        # [FUNCTION DOC]
        # Times writing the single-index slice.
        self.df_single_index.to_csv(self.fname)


class ToCSVDatetime(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing a daily datetime column and index with a custom date
    # format. State: data.
    fname = "__test__.csv"

    def setup(self):
        # [FUNCTION DOC]
        # Builds 1000 daily dates from 2000-01-01 and a frame with them as both the
        # single column and the index, stored as data.
        rng = date_range("1/1/2000", periods=1000)
        self.data = DataFrame(rng, index=rng)

    def time_frame_date_formatting(self):
        # [FUNCTION DOC]
        # Times writing with the compact year-month-day date format.
        self.data.to_csv(self.fname, date_format="%Y%m%d")


class ToCSVDatetimeIndex(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing a frame with a per-second datetime index, with and
    # without a date format. State: data.
    fname = "__test__.csv"

    def setup(self):
        # [FUNCTION DOC]
        # Builds 100000 per-second timestamps from 2000 as the index of a frame with
        # a constant column a; stores as data.
        rng = date_range("2000", periods=100_000, freq="s")
        self.data = DataFrame({"a": 1}, index=rng)

    def time_frame_date_formatting_index(self):
        # [FUNCTION DOC]
        # Times writing with an explicit date-time format.
        self.data.to_csv(self.fname, date_format="%Y-%m-%d %H:%M:%S")

    def time_frame_date_no_format_index(self):
        # [FUNCTION DOC]
        # Times writing with the default date formatting.
        self.data.to_csv(self.fname)


class ToCSVPeriod(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing a frame whose single column holds periods, for two
    # sizes and two frequencies (daily, hourly), with default, explicit-default and
    # custom date formats. Parametrized by nobs and freq. State: data and
    # default_fmt.
    fname = "__test__.csv"

    params = ([1000, 10000], ["D", "h"])
    param_names = ["nobs", "freq"]

    def setup(self, nobs, freq):
        # [FUNCTION DOC]
        # Builds a period range of nobs entries from 2000-01-01 with the frequency,
        # stores it as a one-column frame data, and sets default_fmt to the format
        # that matches the default period rendering for that frequency (date only
        # for daily; date and hour for hourly). Any other frequency leaves
        # default_fmt unset.
        rng = period_range(start="2000-01-01", periods=nobs, freq=freq)
        self.data = DataFrame(rng)
        if freq == "D":
            self.default_fmt = "%Y-%m-%d"
        elif freq == "h":
            self.default_fmt = "%Y-%m-%d %H:00"

    def time_frame_period_formatting_default(self, nobs, freq):
        # [FUNCTION DOC]
        # Times writing with default period formatting.
        self.data.to_csv(self.fname)

    def time_frame_period_formatting_default_explicit(self, nobs, freq):
        # [FUNCTION DOC]
        # Times writing with the explicit format equal to the default rendering.
        self.data.to_csv(self.fname, date_format=self.default_fmt)

    def time_frame_period_formatting(self, nobs, freq):
        # [FUNCTION DOC]
        # Times writing with a custom date format containing a marker string.
        # KNOWN ISSUE: the date format option is currently not applied to period
        # columns (issue 51621), so this timing is effectively identical to the
        # default formatting benchmark above and is expected to get slower when
        # that issue is fixed (the original comment asks to remove its note then).
        # Nb: `date_format` is not actually taken into account here today, so the
        # performance is currently identical to `time_frame_period_formatting_default`
        # above. This timer is therefore expected to degrade when GH#51621 is fixed.
        # (Remove this comment when GH#51621 is fixed.)
        self.data.to_csv(self.fname, date_format="%Y-%m-%d___%H:%M:%S")


class ToCSVPeriodIndex(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing a frame with a period index, for two sizes and two
    # frequencies, with custom, default and explicit-default date formats.
    # Parametrized by nobs and freq. State: data and default_fmt.
    fname = "__test__.csv"

    params = ([1000, 10000], ["D", "h"])
    param_names = ["nobs", "freq"]

    def setup(self, nobs, freq):
        # [FUNCTION DOC]
        # Builds a period range of nobs entries from 2000-01-01 with the frequency
        # as the index of a frame with a constant column a (data), and sets
        # default_fmt to the matching default rendering (date only for daily, date
        # and hour for hourly; otherwise unset).
        rng = period_range(start="2000-01-01", periods=nobs, freq=freq)
        self.data = DataFrame({"a": 1}, index=rng)
        if freq == "D":
            self.default_fmt = "%Y-%m-%d"
        elif freq == "h":
            self.default_fmt = "%Y-%m-%d %H:00"

    def time_frame_period_formatting_index(self, nobs, freq):
        # [FUNCTION DOC]
        # Times writing with a custom date format containing a marker string.
        self.data.to_csv(self.fname, date_format="%Y-%m-%d___%H:%M:%S")

    def time_frame_period_formatting_index_default(self, nobs, freq):
        # [FUNCTION DOC]
        # Times writing with default period index formatting.
        self.data.to_csv(self.fname)

    def time_frame_period_formatting_index_default_explicit(self, nobs, freq):
        # [FUNCTION DOC]
        # Times writing with the explicit format equal to the default rendering.
        self.data.to_csv(self.fname, date_format=self.default_fmt)


class ToCSVDatetimeBig(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing frames with repeated datetime and date values, for
    # three sizes, with a raised timeout of 1500 seconds. Parametrized by nobs.
    # State: data.
    fname = "__test__.csv"
    timeout = 1500
    params = [1000, 10000, 100000]
    param_names = ["nobs"]

    def setup(self, nobs):
        # [FUNCTION DOC]
        # Builds a frame of nobs rows: dt (the timestamp 2018-11-26 11:18:27
        # repeated), d (the date 2018-11-29 repeated) and r (one random uniform
        # number repeated). Stores as data.
        d = "2018-11-29"
        dt = "2018-11-26 11:18:27.0"
        self.data = DataFrame(
            {
                "dt": [np.datetime64(dt)] * nobs,
                "d": [np.datetime64(d)] * nobs,
                "r": [np.random.uniform()] * nobs,
            }
        )

    def time_frame(self, nobs):
        # [FUNCTION DOC]
        # Times writing the frame to the file.
        self.data.to_csv(self.fname)


class ToCSVIndexes(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing frames with a plain index, with a multi-index built
    # after truncating the frame, and with a head of a frame that was indexed
    # before truncation (which keeps unused levels). State: df_standard_index,
    # df_custom_index_then_head and df_head_then_custom_index.
    fname = "__test__.csv"

    @staticmethod
    def _create_df(rows, cols):
        # [FUNCTION DOC]
        # HELPER: frame factory for the setup method of this class. Static method.
        # Inputs: rows (count) and cols (number of data columns). Returns a frame
        # with three index-candidate columns (index1 random integers below rows;
        # index2 and index3 constant one) and cols data columns named col0 and up
        # holding random uniform numbers up to 100000.
        index_cols = {
            "index1": np.random.randint(0, rows, rows),
            "index2": np.full(rows, 1, dtype=int),
            "index3": np.full(rows, 1, dtype=int),
        }
        data_cols = {
            f"col{i}": np.random.uniform(0, 100000.0, rows) for i in range(cols)
        }
        df = DataFrame({**index_cols, **data_cols})
        return df

    def setup(self):
        # [FUNCTION DOC]
        # Builds three frames with 100000 rows and 5 data columns: a plain frame
        # (df_standard_index); one created ten times larger, indexed by the three
        # index columns and then truncated to the first 100000 rows
        # (df_custom_index_then_head, retaining unused index levels); and one
        # created ten times larger, truncated and then indexed
        # (df_head_then_custom_index).
        ROWS = 100000
        COLS = 5
        # For tests using .head(), create an initial dataframe with this many times
        # more rows
        HEAD_ROW_MULTIPLIER = 10

        self.df_standard_index = self._create_df(ROWS, COLS)

        self.df_custom_index_then_head = (
            self._create_df(ROWS * HEAD_ROW_MULTIPLIER, COLS)
            .set_index(["index1", "index2", "index3"])
            .head(ROWS)
        )

        self.df_head_then_custom_index = (
            self._create_df(ROWS * HEAD_ROW_MULTIPLIER, COLS)
            .head(ROWS)
            .set_index(["index1", "index2", "index3"])
        )

    def time_standard_index(self):
        # [FUNCTION DOC]
        # Times writing the plain-index frame.
        self.df_standard_index.to_csv(self.fname)

    def time_multiindex(self):
        # [FUNCTION DOC]
        # Times writing the frame truncated before being indexed (a clean
        # multi-index).
        self.df_head_then_custom_index.to_csv(self.fname)

    def time_head_of_multiindex(self):
        # [FUNCTION DOC]
        # Times writing the truncated multi-indexed frame whose index levels still
        # hold unused values.
        self.df_custom_index_then_head.to_csv(self.fname)


class StringIORewind:
    # [CLASS DOC]
    # HELPER: mixin base class for the read benchmarks in this module that read
    # from in-memory text buffers. It provides a method that rewinds a buffer so it
    # can be read again on each timing run. No state.
    def data(self, stringio_object):
        # [FUNCTION DOC]
        # HELPER: rewinds a buffer. Input: a seekable text or binary buffer.
        # Moves its read position to the start (side effect) and returns the same
        # buffer.
        stringio_object.seek(0)
        return stringio_object


class ReadCSVDInferDatetimeFormat(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading a one-column file of dates, parsing it as datetimes,
    # for no format, a custom format with fractional seconds, ISO 8601 style and a
    # compact year-month-day format. Parametrized by format name. State:
    # StringIO_input.
    params = [None, "custom", "iso8601", "ymd"]
    param_names = ["format"]

    def setup(self, format):
        # [FUNCTION DOC]
        # Builds 1000 daily dates from 2000-01-01, formats them with the pattern
        # chosen by the format name, joins them with newlines and stores them in a
        # text buffer as StringIO_input.
        rng = date_range("1/1/2000", periods=1000)
        formats = {
            None: None,
            "custom": "%m/%d/%Y %H:%M:%S.%f",
            "iso8601": "%Y-%m-%d %H:%M:%S",
            "ymd": "%Y%m%d",
        }
        dt_format = formats[format]
        self.StringIO_input = StringIO("\n".join(rng.strftime(dt_format).tolist()))

    def time_read_csv(self, format):
        # [FUNCTION DOC]
        # Times reading the rewound buffer as a headerless one-column file named
        # foo, parsing that column as dates. The format parameter is unused here.
        read_csv(
            self.data(self.StringIO_input),
            header=None,
            names=["foo"],
            parse_dates=["foo"],
        )


class ReadCSVConcatDatetime(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading 50000 ISO-formatted timestamp lines and parsing them as
    # dates. Class attribute iso8601 holds the format pattern. State:
    # StringIO_input.
    iso8601 = "%Y-%m-%d %H:%M:%S"

    def setup(self):
        # [FUNCTION DOC]
        # Builds 50000 per-second timestamps from 2000-01-01, formats them with
        # the class pattern, joins with newlines, and stores them in a text buffer.
        rng = date_range("1/1/2000", periods=50000, freq="s")
        self.StringIO_input = StringIO("\n".join(rng.strftime(self.iso8601).tolist()))

    def time_read_csv(self):
        # [FUNCTION DOC]
        # Times reading the rewound buffer as a headerless one-column file named
        # foo and parsing it as dates.
        read_csv(
            self.data(self.StringIO_input),
            header=None,
            names=["foo"],
            parse_dates=["foo"],
        )


class ReadCSVConcatDatetimeBadDateValue(StringIORewind):
    # [CLASS DOC]
    # Benchmark of parsing a date column that contains only invalid placeholder
    # values (the text nan, the text zero, or empty). Parametrized by the bad
    # value (given as a one-element tuple of options). State: StringIO_input.
    params = (["nan", "0", ""],)
    param_names = ["bad_date_value"]

    def setup(self, bad_date_value):
        # [FUNCTION DOC]
        # Builds a buffer of 50000 lines, each the bad value followed by a comma.
        self.StringIO_input = StringIO((f"{bad_date_value},\n") * 50000)

    def time_read_csv(self, bad_date_value):
        # [FUNCTION DOC]
        # Times reading the rewound buffer as a headerless two-column file (foo,
        # bar) and parsing foo as dates. The parameter is unused here.
        read_csv(
            self.data(self.StringIO_input),
            header=None,
            names=["foo", "bar"],
            parse_dates=["foo"],
        )


class ReadCSVSkipRows(BaseIO):
    # [CLASS DOC]
    # Benchmark of reading a 20000-row file while skipping no rows or the first
    # 10000, with three parser engines. Parametrized by skiprows and engine. The
    # file __test__.csv is written in setup and removed by the base class.
    fname = "__test__.csv"
    params = ([None, 10000], ["c", "python", "pyarrow"])
    param_names = ["skiprows", "engine"]

    def setup(self, skiprows, engine):
        # [FUNCTION DOC]
        # Writes a 20000-row file with a string index and columns float1, float2
        # (random normal), string1 (constant), bool1 (all true) and int1 (random
        # integers). Side effect: creates the file. Parameters are unused here.
        N = 20000
        index = Index([f"i-{i}" for i in range(N)], dtype=object)
        df = DataFrame(
            {
                "float1": np.random.randn(N),
                "float2": np.random.randn(N),
                "string1": ["foo"] * N,
                "bool1": [True] * N,
                "int1": np.random.randint(0, N, size=N),
            },
            index=index,
        )
        df.to_csv(self.fname)

    def time_skipprows(self, skiprows, engine):
        # [FUNCTION DOC]
        # Times reading the file with the row-skip option and engine.
        read_csv(self.fname, skiprows=skiprows, engine=engine)


class ReadUint64Integers(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading one-column files of unsigned 64-bit integers above the
    # signed range, with and without a negative value and with a missing-value
    # marker. State: na_values, data1 and data2.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 10000 consecutive integers starting at two to the 63rd. data1 is
        # a buffer of their text lines; data2 is the same with the entry at
        # position 500 replaced by minus one. na_values is a one-element list
        # holding the value two to the 63rd plus 500.
        self.na_values = [2**63 + 500]
        arr = np.arange(10000).astype("uint64") + 2**63
        self.data1 = StringIO("\n".join(arr.astype(str).tolist()))
        arr = arr.astype(object)
        arr[500] = -1
        self.data2 = StringIO("\n".join(arr.astype(str).tolist()))

    def time_read_uint64(self):
        # [FUNCTION DOC]
        # Times reading the all-unsigned buffer as a headerless column foo.
        read_csv(self.data(self.data1), header=None, names=["foo"])

    def time_read_uint64_neg_values(self):
        # [FUNCTION DOC]
        # Times reading the buffer that contains a negative value.
        read_csv(self.data(self.data2), header=None, names=["foo"])

    def time_read_uint64_na_values(self):
        # [FUNCTION DOC]
        # Times reading the all-unsigned buffer with an extra missing-value marker.
        read_csv(
            self.data(self.data1), header=None, names=["foo"], na_values=self.na_values
        )


class ReadCSVThousands(BaseIO):
    # [CLASS DOC]
    # Benchmark of reading numbers formatted with thousands separators, for comma
    # and pipe separators, with and without the thousands option, and two engines.
    # Parametrized by sep, thousands and engine. The file __test__.csv is written
    # in setup and removed by the base class.
    fname = "__test__.csv"
    params = ([",", "|"], [None, ","], ["c", "python"])
    param_names = ["sep", "thousands", "engine"]

    def setup(self, sep, thousands, engine):
        # [FUNCTION DOC]
        # Builds a 10000 by 8 frame of random normal numbers scaled by random
        # integers between 100 and 9999. If thousands is not none, formats every
        # cell as text with that thousands separator via a brace format built at
        # runtime. Writes the frame to the file with the separator (side effect).
        # The engine parameter is unused here.
        N = 10000
        K = 8
        data = np.random.randn(N, K) * np.random.randint(100, 10000, (N, K))
        df = DataFrame(data)
        if thousands is not None:
            fmt = f":{thousands}"
            fmt = "{" + fmt + "}"
            df = df.map(lambda x: fmt.format(x))
        df.to_csv(self.fname, sep=sep)

    def time_thousands(self, sep, thousands, engine):
        # [FUNCTION DOC]
        # Times reading the file with the separator, thousands option and engine.
        read_csv(self.fname, sep=sep, thousands=thousands, engine=engine)


class ReadCSVComment(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading lines that end with a comment, with the comment
    # character option, for two engines. Parametrized by engine. State:
    # StringIO_input.
    params = ["c", "python"]
    param_names = ["engine"]

    def setup(self, engine):
        # [FUNCTION DOC]
        # Builds a buffer of a header line and 100000 lines of three numbers
        # followed by a hash comment. The engine parameter is unused here.
        data = ["A,B,C"] + (["1,2,3 # comment"] * 100000)
        self.StringIO_input = StringIO("\n".join(data))

    def time_comment(self, engine):
        # [FUNCTION DOC]
        # Times reading the rewound buffer with the hash as comment character,
        # headerless, with column names a, b, c. The engine parameter is unused
        # here (the default engine is used).
        # KNOWN ISSUE: the engine parameter is never passed to the reader, so both
        # parameter values measure the same default engine.
        read_csv(
            self.data(self.StringIO_input), comment="#", header=None, names=list("abc")
        )


class ReadCSVFloatPrecision(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading 1000 rows of three long decimal numbers with different
    # separators, decimal characters and float precision modes. Parametrized by
    # sep, decimal and float_precision. State: StringIO_input.
    params = ([",", ";"], [".", "_"], [None, "high", "round_trip"])
    param_names = ["sep", "decimal", "float_precision"]

    def setup(self, sep, decimal, float_precision):
        # [FUNCTION DOC]
        # Generates 15 random 28-digit strings, builds a row template of three
        # numbers (zero, the decimal character, a placeholder) joined by the
        # separator, repeats the row five times, fills the placeholders with the 15
        # strings, and repeats the block 200 times (1000 lines of three values).
        # Stores the text in a buffer. The float_precision parameter is unused
        # here.
        floats = [
            "".join([random.choice(string.digits) for _ in range(28)])
            for _ in range(15)
        ]
        rows = sep.join([f"0{decimal}{{}}"] * 3) + "\n"
        data = rows * 5
        data = data.format(*floats) * 200  # 1000 x 3 strings csv
        self.StringIO_input = StringIO(data)

    def time_read_csv(self, sep, decimal, float_precision):
        # [FUNCTION DOC]
        # Times reading the buffer headerless with column names a, b, c, the
        # separator and the float precision mode.
        # KNOWN ISSUE: the decimal parameter is never passed to the reader, so the
        # underscore decimal character used in the generated data is not declared
        # to the parser as the decimal marker.
        read_csv(
            self.data(self.StringIO_input),
            sep=sep,
            header=None,
            names=list("abc"),
            float_precision=float_precision,
        )

    def time_read_csv_python_engine(self, sep, decimal, float_precision):
        # [FUNCTION DOC]
        # Times reading the buffer with the python engine, with the float precision
        # option fixed to none regardless of the parameter.
        # KNOWN ISSUE: the decimal and float_precision parameters are not used by
        # this method, so it repeats identical work across those parameter values.
        read_csv(
            self.data(self.StringIO_input),
            sep=sep,
            header=None,
            engine="python",
            float_precision=None,
            names=list("abc"),
        )


class ReadCSVEngine(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading a 100000-row, five-column integer file from a text
    # buffer and from a bytes buffer, for three engines, plus peak memory.
    # Parametrized by engine. State: StringIO_input and BytesIO_input.
    params = ["c", "python", "pyarrow"]
    param_names = ["engine"]

    def setup(self, engine):
        # [FUNCTION DOC]
        # Builds a header line A to E and 100000 rows of the numbers one to five, as
        # a text buffer, and a bytes buffer holding its UTF-8 encoding, to
        # simulate reading from a file. The engine parameter is unused here.
        data = ["A,B,C,D,E"] + (["1,2,3,4,5"] * 100000)
        self.StringIO_input = StringIO("\n".join(data))
        # simulate reading from file
        self.BytesIO_input = BytesIO(self.StringIO_input.read().encode("utf-8"))

    def time_read_stringcsv(self, engine):
        # [FUNCTION DOC]
        # Times reading the rewound text buffer with the engine.
        read_csv(self.data(self.StringIO_input), engine=engine)

    def time_read_bytescsv(self, engine):
        # [FUNCTION DOC]
        # Times reading the rewound bytes buffer with the engine.
        read_csv(self.data(self.BytesIO_input), engine=engine)

    def peakmem_read_csv(self, engine):
        # [FUNCTION DOC]
        # Peak-memory benchmark of reading the bytes buffer with the engine.
        read_csv(self.data(self.BytesIO_input), engine=engine)


class ReadCSVCategorical(BaseIO):
    # [CLASS DOC]
    # Benchmark of reading a file and converting columns to categoricals after
    # reading versus during reading, for two engines. Parametrized by engine. The
    # file __test__.csv is written in setup and removed by the base class.
    fname = "__test__.csv"
    params = ["c", "python"]
    param_names = ["engine"]

    def setup(self, engine):
        # [FUNCTION DOC]
        # Writes a file without index of 100000 rows and three columns a, b, c,
        # each a random choice among five long strings (side effect: creates the
        # file). The engine parameter is unused here.
        N = 100000
        group1 = ["aaaaaaaa", "bbbbbbb", "cccccccc", "dddddddd", "eeeeeeee"]
        df = DataFrame(np.random.choice(group1, (N, 3)), columns=list("abc"))
        df.to_csv(self.fname, index=False)

    def time_convert_post(self, engine):
        # [FUNCTION DOC]
        # Times reading the file with the engine and then converting each column to
        # a categorical.
        read_csv(self.fname, engine=engine).apply(Categorical)

    def time_convert_direct(self, engine):
        # [FUNCTION DOC]
        # Times reading the file with the engine converting all columns directly to
        # the categorical dtype.
        read_csv(self.fname, engine=engine, dtype="category")


class ReadCSVParseDates(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading a small file with a date column and several numeric
    # columns while parsing the second column as dates, for two engines.
    # Parametrized by engine. State: StringIO_input.
    params = ["c", "python"]
    param_names = ["engine"]

    def setup(self, engine):
        # [FUNCTION DOC]
        # Builds a five-line template of comma-separated lines (a station code and
        # date, then a time, a second time and numeric values) with placeholders,
        # fills each with the same station and date text, and stores it in a buffer.
        # Note the template lines are separated by blank lines and indented. The
        # engine parameter is unused here.
        data = """{},19:00:00,18:56:00,0.8100,2.8100,7.2000,0.0000,280.0000\n
                  {},20:00:00,19:56:00,0.0100,2.2100,7.2000,0.0000,260.0000\n
                  {},21:00:00,20:56:00,-0.5900,2.2100,5.7000,0.0000,280.0000\n
                  {},21:00:00,21:18:00,-0.9900,2.0100,3.6000,0.0000,270.0000\n
                  {},22:00:00,21:56:00,-0.5900,1.7100,5.1000,0.0000,290.0000\n
               """
        two_cols = ["KORD,19990127"] * 5
        data = data.format(*two_cols)
        self.StringIO_input = StringIO(data)

    def time_baseline(self, engine):
        # [FUNCTION DOC]
        # Times reading the rewound buffer headerless with the engine, comma
        # separator, nine column names (the digit characters zero to eight), and
        # parsing column index one as dates.
        read_csv(
            self.data(self.StringIO_input),
            engine=engine,
            sep=",",
            header=None,
            parse_dates=[1],
            names=list(string.digits[:9]),
        )


class ReadCSVCachedParseDates(StringIORewind):
    # [CLASS DOC]
    # Benchmark of date parsing with the date cache on or off, for two engines.
    # Parametrized by do_cache and engine. State: StringIO_input.
    params = ([True, False], ["c", "python"])
    param_names = ["do_cache", "engine"]

    def setup(self, do_cache, engine):
        # [FUNCTION DOC]
        # Builds a buffer of lines of the form 10/year for years 2000 to 2099,
        # repeated ten times (1000 lines). Parameters are unused here.
        data = ("\n".join([f"10/{year}" for year in range(2000, 2100)]) + "\n") * 10
        self.StringIO_input = StringIO(data)

    def time_read_csv_cached(self, do_cache, engine):
        # [FUNCTION DOC]
        # Times reading the rewound buffer headerless, parsing column zero as dates
        # with the cache option as given. A type error (older library versions that
        # lack the cache option) is silently ignored.
        try:
            read_csv(
                self.data(self.StringIO_input),
                engine=engine,
                header=None,
                parse_dates=[0],
                cache_dates=do_cache,
            )
        except TypeError:
            # cache_dates is a new keyword in 0.25
            pass


class ReadCSVMemoryGrowth(BaseIO):
    # [CLASS DOC]
    # Memory benchmark (issue 24805) of reading a file in small chunks, for two
    # engines. Class attributes: chunksize (20), num_rows (1000), fname. Parametrized
    # by engine. The file is written in setup and removed by the base class.
    chunksize = 20
    num_rows = 1000
    fname = "__test__.csv"
    params = ["c", "python"]
    param_names = ["engine"]

    def setup(self, engine):
        # [FUNCTION DOC]
        # Writes a file of num_rows lines, each holding its line number (side
        # effect: creates the file with UTF-8 encoding). The engine parameter is
        # unused here.
        with open(self.fname, "w", encoding="utf-8") as f:
            for i in range(self.num_rows):
                f.write(f"{i}\n")

    def mem_parser_chunks(self, engine):
        # [FUNCTION DOC]
        # Memory benchmark. Opens the file as a chunked reader with the chunk size
        # and engine and iterates all chunks, discarding them. See issue 24805.
        # see gh-24805.
        result = read_csv(self.fname, chunksize=self.chunksize, engine=engine)

        for _ in result:
            pass


class ReadCSVParseSpecialDate(StringIORewind):
    # [CLASS DOC]
    # Benchmark of parsing dates in unusual formats (month-year, month-day-year,
    # hour-minute) repeated 10000 times, for two engines. Parametrized by the
    # value kind and engine. The class attribute objects maps each kind to a block
    # of sample lines. State: StringIO_input.
    params = (["mY", "mdY", "hm"], ["c", "python"])
    param_names = ["value", "engine"]
    objects = {
        "mY": "01-2019\n10-2019\n02/2000\n",
        "mdY": "12/02/2010\n",
        "hm": "21:34\n",
    }

    def setup(self, value, engine):
        # [FUNCTION DOC]
        # Repeats the sample block for the kind 10000 times and stores it in a
        # buffer. The engine parameter is unused here.
        count_elem = 10000
        data = self.objects[value] * count_elem
        self.StringIO_input = StringIO(data)

    def time_read_special_date(self, value, engine):
        # [FUNCTION DOC]
        # Times reading the rewound buffer headerless with the engine as a single
        # column Date, parsing it as dates. The value parameter is unused here.
        read_csv(
            self.data(self.StringIO_input),
            engine=engine,
            sep=",",
            header=None,
            names=["Date"],
            parse_dates=["Date"],
        )


class ReadCSVMemMapUTF8:
    # [CLASS DOC]
    # Benchmark of reading a memory-mapped UTF-8 file containing characters across
    # the Unicode range, with five calls per measurement. No parameters. Does not
    # use the file-cleanup base class, so the file __test__.csv is left behind.
    fname = "__test__.csv"
    number = 5

    def setup(self):
        # [FUNCTION DOC]
        # Creates lines of 128 consecutive characters from the space character up
        # to the code point 0x10080, skipping chunks that cannot be encoded as
        # UTF-8 (such as surrogate code points). Builds a one-column frame of the
        # lines, stacks 100 copies, and writes it without header or index as a UTF-8
        # file (side effect: creates the file).
        lines = []
        line_length = 128
        start_char = " "
        end_char = "\U00010080"
        # This for loop creates a list of 128-char strings
        # consisting of consecutive Unicode chars
        for lnum in range(ord(start_char), ord(end_char), line_length):
            line = "".join([chr(c) for c in range(lnum, lnum + 0x80)]) + "\n"
            try:
                line.encode("utf-8")
            except UnicodeEncodeError:
                # Some 16-bit words are not valid Unicode chars and must be skipped
                continue
            lines.append(line)
        df = DataFrame(lines)
        df = concat([df for n in range(100)], ignore_index=True)
        df.to_csv(self.fname, index=False, header=False, encoding="utf-8")

    def time_read_memmapped_utf8(self):
        # [FUNCTION DOC]
        # Times reading the file headerless with memory mapping, UTF-8 encoding and
        # the c engine.
        read_csv(self.fname, header=None, memory_map=True, encoding="utf-8", engine="c")


class ParseDateComparison(StringIORewind):
    # [CLASS DOC]
    # Benchmark comparing day-first date parsing inside the reader versus a
    # separate conversion call, with the date cache off or on. Parametrized by
    # cache_dates. State: StringIO_input.
    params = ([False, True],)
    param_names = ["cache_dates"]

    def setup(self, cache_dates):
        # [FUNCTION DOC]
        # Builds a buffer of 10000 identical lines 12-02-2010. The cache_dates
        # parameter is unused here.
        count_elem = 10000
        data = "12-02-2010\n" * count_elem
        self.StringIO_input = StringIO(data)

    def time_read_csv_dayfirst(self, cache_dates):
        # [FUNCTION DOC]
        # Times reading the rewound buffer headerless as a Date column parsed as
        # dates with day first and the cache option as given. A type error (old
        # versions lacking the option) is silently ignored.
        try:
            read_csv(
                self.data(self.StringIO_input),
                sep=",",
                header=None,
                names=["Date"],
                parse_dates=["Date"],
                cache_dates=cache_dates,
                dayfirst=True,
            )
        except TypeError:
            # cache_dates is a new keyword in 0.25
            pass

    def time_to_datetime_dayfirst(self, cache_dates):
        # [FUNCTION DOC]
        # Times reading the buffer as strings in a column date, then converting that
        # column to datetimes with day first and the cache option (the read is part
        # of the timing).
        df = read_csv(
            self.data(self.StringIO_input), dtype={"date": str}, names=["date"]
        )
        to_datetime(df["date"], cache=cache_dates, dayfirst=True)

    def time_to_datetime_format_DD_MM_YYYY(self, cache_dates):
        # [FUNCTION DOC]
        # Times reading the buffer as strings, then converting with the explicit
        # day-month-year format and the cache option (the read is part of the
        # timing).
        df = read_csv(
            self.data(self.StringIO_input), dtype={"date": str}, names=["date"]
        )
        to_datetime(df["date"], cache=cache_dates, format="%d-%m-%Y")


class ReadCSVIndexCol(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading a two-column file while using the first column as the
    # index. No parameters. State: StringIO_input.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a buffer with a header line a,b followed by 100000 lines 1,2.
        count_elem = 100_000
        data = "a,b\n" + "1,2\n" * count_elem
        self.StringIO_input = StringIO(data)

    def time_read_csv_index_col(self):
        # [FUNCTION DOC]
        # Times reading the rewound buffer with column a as the index.
        read_csv(self.data(self.StringIO_input), index_col="a")


class ReadCSVDatePyarrowEngine(StringIORewind):
    # [CLASS DOC]
    # Benchmark of reading a one-column date file with the arrow engine and the
    # arrow-backed dtype backend while parsing dates. No parameters. State:
    # StringIO_input.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a buffer with a header line a followed by 100000 lines of the date
        # 2019-12-31.
        count_elem = 100_000
        data = "a\n" + "2019-12-31\n" * count_elem
        self.StringIO_input = StringIO(data)

    def time_read_csv_index_col(self):
        # [FUNCTION DOC]
        # Times reading the rewound buffer with the arrow engine, parsing column a
        # as dates, and using arrow-backed dtypes.
        # KNOWN ISSUE: the method name mentions an index column, but no index
        # column option is used; the name appears copied from the index-column
        # benchmark class above.
        read_csv(
            self.data(self.StringIO_input),
            parse_dates=["a"],
            engine="pyarrow",
            dtype_backend="pyarrow",
        )


class ReadCSVCParserLowMemory:
    # [CLASS DOC]
    # Peak-memory benchmark (issue 16798) of reading an input larger than two
    # gigabytes with the c engine and low-memory mode disabled. No parameters.
    # State: csv.
    # GH 16798
    def setup(self):
        # [FUNCTION DOC]
        # Builds a text buffer with a header line strings followed by 2100 lines of
        # one mebibyte each of the letter x (about 2.1 GiB of text in memory).
        self.csv = StringIO(
            "strings\n" + "\n".join(["x" * (1 << 20) for _ in range(2100)])
        )

    def peakmem_over_2gb_input(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of reading the buffer with the c engine and
        # low-memory chunking disabled. The buffer is not rewound, so repeated
        # calls on the same buffer would read nothing.
        read_csv(self.csv, engine="c", low_memory=False)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
