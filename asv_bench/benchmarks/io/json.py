# [MODULE DOC]
# Benchmarks for JSON reading, normalization and writing: reading files in various
# orientations and in line-delimited form (including chunked and row-limited
# reads), flattening nested records, and writing frames of many dtype mixes in each
# orientation (also wide frames, ISO date formatting, line-delimited output and peak
# memory of repeated small writes). Public contents are the benchmark classes
# ReadJSON, ReadJSONLines, NormalizeJSON, ToJSON, ToJSONWide, ToJSONISO,
# ToJSONLines and ToJSONMem. File-based classes derive from the shared
# file-benchmark base class and use temporary files in the working directory.
# Final line re-exports the common asv setup hook.
import sys

import numpy as np

from pandas import (
    DataFrame,
    Index,
    concat,
    date_range,
    json_normalize,
    read_json,
    timedelta_range,
)

from ..pandas_vb_common import BaseIO


class ReadJSON(BaseIO):
    # [CLASS DOC]
    # Benchmark of reading a JSON file of 100000 rows and five float columns in
    # several orientations, with an integer or datetime index. Parametrized by
    # orient and index kind. The file __test__.json is written in setup and removed
    # by the base class.
    fname = "__test__.json"
    params = (["split", "index", "records"], ["int", "datetime"])
    param_names = ["orient", "index"]

    def setup(self, orient, index):
        # [FUNCTION DOC]
        # Builds a frame of 100000 rows and columns float_0 to float_4 of random
        # normal numbers, indexed by integers or hourly timestamps from
        # 2000-01-01, and writes it to the file in the orientation (side effect:
        # creates the file).
        N = 100000
        indexes = {
            "int": np.arange(N),
            "datetime": date_range("20000101", periods=N, freq="h"),
        }
        df = DataFrame(
            np.random.randn(N, 5),
            columns=[f"float_{i}" for i in range(5)],
            index=indexes[index],
        )
        df.to_json(self.fname, orient=orient)

    def time_read_json(self, orient, index):
        # [FUNCTION DOC]
        # Times reading the file with the orientation. The index parameter is
        # unused here.
        read_json(self.fname, orient=orient)


class ReadJSONLines(BaseIO):
    # [CLASS DOC]
    # Benchmark of reading a line-delimited JSON file in full, in chunks and
    # limited to some rows, for time and peak memory. Parametrized by index kind
    # (integer, datetime). The file __test_lines__.json is written in setup and
    # removed by the base class.
    fname = "__test_lines__.json"
    params = ["int", "datetime"]
    param_names = ["index"]

    def setup(self, index):
        # [FUNCTION DOC]
        # Builds a frame of 100000 rows and five random float columns indexed by
        # integers or hourly timestamps, and writes it as one record per line (side
        # effect: creates the file).
        N = 100000
        indexes = {
            "int": np.arange(N),
            "datetime": date_range("20000101", periods=N, freq="h"),
        }
        df = DataFrame(
            np.random.randn(N, 5),
            columns=[f"float_{i}" for i in range(5)],
            index=indexes[index],
        )
        df.to_json(self.fname, orient="records", lines=True)

    def time_read_json_lines(self, index):
        # [FUNCTION DOC]
        # Times reading the whole line-delimited file.
        read_json(self.fname, orient="records", lines=True)

    def time_read_json_lines_concat(self, index):
        # [FUNCTION DOC]
        # Times reading the file in chunks of 25000 lines and concatenating the
        # chunks.
        concat(read_json(self.fname, orient="records", lines=True, chunksize=25000))

    def time_read_json_lines_nrows(self, index):
        # [FUNCTION DOC]
        # Times reading only the first 25000 lines.
        read_json(self.fname, orient="records", lines=True, nrows=25000)

    def peakmem_read_json_lines(self, index):
        # [FUNCTION DOC]
        # Peak-memory benchmark of reading the whole file.
        read_json(self.fname, orient="records", lines=True)

    def peakmem_read_json_lines_concat(self, index):
        # [FUNCTION DOC]
        # Peak-memory benchmark of chunked reading (25000 lines per chunk) with
        # concatenation.
        concat(read_json(self.fname, orient="records", lines=True, chunksize=25000))

    def peakmem_read_json_lines_nrows(self, index):
        # [FUNCTION DOC]
        # Peak-memory benchmark of reading only the first 15000 lines (the time
        # counterpart reads 25000).
        read_json(self.fname, orient="records", lines=True, nrows=15000)


class NormalizeJSON(BaseIO):
    # [CLASS DOC]
    # Benchmark of flattening a list of identical nested records.
    # KNOWN ISSUE: the class is parametrized by orient and frame names, but neither
    # parameter influences setup or the timing method, so all twenty-five
    # parameter combinations measure the same operation.
    # State: data.
    fname = "__test__.json"
    params = [
        ["split", "columns", "index", "values", "records"],
        ["df", "df_date_idx", "df_td_int_ts", "df_int_floats", "df_int_float_str"],
    ]
    param_names = ["orient", "frame"]

    def setup(self, orient, frame):
        # [FUNCTION DOC]
        # Builds one record with a mixed-type list, a doubly nested dictionary, a
        # nested dictionary with a list of mixed numbers, and a string; stores a
        # list of 10000 references to the same record as data. Parameters are
        # unused.
        data = {
            "hello": ["thisisatest", 999898, "mixed types"],
            "nest1": {"nest2": {"nest3": "nest3_value", "nest3_int": 3445}},
            "nest1_list": {"nest2": ["blah", 32423, 546456.876, 92030234]},
            "hello2": "string",
        }
        self.data = [data for i in range(10000)]

    def time_normalize_json(self, orient, frame):
        # [FUNCTION DOC]
        # Times flattening the list of records into a frame. Parameters are
        # unused.
        json_normalize(self.data)


class ToJSON(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing frames of many dtype mixes to a JSON file in each
    # orientation, for time and peak memory. Parametrized by orient and the frame
    # attribute name (df, df_date_idx, df_td_int_ts, df_int_floats,
    # df_int_float_str). The file __test__.json is written by the timing methods and
    # removed by the base class. State: df, df_date_idx, df_td_int_ts,
    # df_int_floats, df_int_float_str, df_longint_float_str.
    fname = "__test__.json"
    params = [
        ["split", "columns", "index", "values", "records"],
        ["df", "df_date_idx", "df_td_int_ts", "df_int_floats", "df_int_float_str"],
    ]
    param_names = ["orient", "frame"]

    def setup(self, orient, frame):
        # [FUNCTION DOC]
        # With 100000 rows, builds shared columns (hourly datetime index from
        # 2000-01-01; timedeltas and datetimes per second starting from one
        # nanosecond; random integers below 100 million; large integers made by
        # multiplying those by the maximum machine integer; random floats; distinct
        # strings), then creates the frames: df (five random columns, integer
        # index), df_date_idx (five random columns, hourly index), df_td_int_ts
        # (two timedelta, two integer, two datetime columns), df_int_floats (three
        # integer, three float columns), df_int_float_str (two integer, two float,
        # two string columns) and df_longint_float_str (two large-integer, two
        # float, two string columns). The last frame is not selectable by the frame
        # parameter in this class. Parameters are unused here.
        N = 10**5
        ncols = 5
        index = date_range("20000101", periods=N, freq="h")
        timedeltas = timedelta_range(start=1, periods=N, freq="s")
        datetimes = date_range(start=1, periods=N, freq="s")
        ints = np.random.randint(100000000, size=N)
        longints = sys.maxsize * np.random.randint(100000000, size=N)
        floats = np.random.randn(N)
        strings = Index([f"i-{i}" for i in range(N)], dtype=object)
        self.df = DataFrame(np.random.randn(N, ncols), index=np.arange(N))
        self.df_date_idx = DataFrame(np.random.randn(N, ncols), index=index)
        self.df_td_int_ts = DataFrame(
            {
                "td_1": timedeltas,
                "td_2": timedeltas,
                "int_1": ints,
                "int_2": ints,
                "ts_1": datetimes,
                "ts_2": datetimes,
            },
            index=index,
        )
        self.df_int_floats = DataFrame(
            {
                "int_1": ints,
                "int_2": ints,
                "int_3": ints,
                "float_1": floats,
                "float_2": floats,
                "float_3": floats,
            },
            index=index,
        )
        self.df_int_float_str = DataFrame(
            {
                "int_1": ints,
                "int_2": ints,
                "float_1": floats,
                "float_2": floats,
                "str_1": strings,
                "str_2": strings,
            },
            index=index,
        )

        self.df_longint_float_str = DataFrame(
            {
                "longint_1": longints,
                "longint_2": longints,
                "float_1": floats,
                "float_2": floats,
                "str_1": strings,
                "str_2": strings,
            },
            index=index,
        )

    def time_to_json(self, orient, frame):
        # [FUNCTION DOC]
        # Times writing the frame named by the frame parameter to the file in the
        # orientation.
        getattr(self, frame).to_json(self.fname, orient=orient)

    def peakmem_to_json(self, orient, frame):
        # [FUNCTION DOC]
        # Peak-memory benchmark of the same write.
        getattr(self, frame).to_json(self.fname, orient=orient)


class ToJSONWide(ToJSON):
    # [CLASS DOC]
    # Variant of the JSON write benchmark for a wide frame built by repeating the
    # first 100 rows of the chosen frame a thousand times side by side. Inherits the
    # parameters, file handling and the narrow-frame timing methods from the parent.
    # Additional state: df_wide.
    def setup(self, orient, frame):
        # [FUNCTION DOC]
        # Runs the parent setup, then copies the chosen frame, takes its first 100
        # rows, concatenates 1000 copies along the column axis with a fresh column
        # numbering, and stores the result as df_wide.
        super().setup(orient, frame)
        base_df = getattr(self, frame).copy()
        df_wide = concat([base_df.iloc[:100]] * 1000, ignore_index=True, axis=1)
        self.df_wide = df_wide

    def time_to_json_wide(self, orient, frame):
        # [FUNCTION DOC]
        # Times writing the wide frame to the file in the orientation.
        self.df_wide.to_json(self.fname, orient=orient)

    def peakmem_to_json_wide(self, orient, frame):
        # [FUNCTION DOC]
        # Peak-memory benchmark of writing the wide frame.
        self.df_wide.to_json(self.fname, orient=orient)


class ToJSONISO(BaseIO):
    # [CLASS DOC]
    # Benchmark of producing JSON text (not a file) with ISO 8601 date formatting
    # for a frame of timedelta and datetime columns. Parametrized by orient. State:
    # df. The inherited teardown tries to remove the file name, which is never
    # created here.
    fname = "__test__.json"
    params = [["split", "columns", "index", "values", "records"]]
    param_names = ["orient"]

    def setup(self, orient):
        # [FUNCTION DOC]
        # Builds a frame of 100000 rows with an hourly index from 2000-01-01 and
        # columns td_1, td_2 (per-second timedeltas) and ts_1, ts_2 (per-second
        # datetimes) as df. The orient parameter is unused here.
        N = 10**5
        index = date_range("20000101", periods=N, freq="h")
        timedeltas = timedelta_range(start=1, periods=N, freq="s")
        datetimes = date_range(start=1, periods=N, freq="s")
        self.df = DataFrame(
            {
                "td_1": timedeltas,
                "td_2": timedeltas,
                "ts_1": datetimes,
                "ts_2": datetimes,
            },
            index=index,
        )

    def time_iso_format(self, orient):
        # [FUNCTION DOC]
        # Times converting the frame to a JSON string in the orientation with ISO
        # date formatting.
        self.df.to_json(orient=orient, date_format="iso")


class ToJSONLines(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing frames of many dtype mixes as line-delimited JSON. No
    # parameters. The file __test__.json is written by the timing methods and removed
    # by the base class. State: the same six frames as the parameterized write
    # benchmark.
    fname = "__test__.json"

    def setup(self):
        # [FUNCTION DOC]
        # Builds the same shared columns and six frames as the parameterized write
        # benchmark's setup (df, df_date_idx, df_td_int_ts, df_int_floats,
        # df_int_float_str, df_longint_float_str); see that class for the layouts.
        N = 10**5
        ncols = 5
        index = date_range("20000101", periods=N, freq="h")
        timedeltas = timedelta_range(start=1, periods=N, freq="s")
        datetimes = date_range(start=1, periods=N, freq="s")
        ints = np.random.randint(100000000, size=N)
        longints = sys.maxsize * np.random.randint(100000000, size=N)
        floats = np.random.randn(N)
        strings = Index([f"i-{i}" for i in range(N)], dtype=object)
        self.df = DataFrame(np.random.randn(N, ncols), index=np.arange(N))
        self.df_date_idx = DataFrame(np.random.randn(N, ncols), index=index)
        self.df_td_int_ts = DataFrame(
            {
                "td_1": timedeltas,
                "td_2": timedeltas,
                "int_1": ints,
                "int_2": ints,
                "ts_1": datetimes,
                "ts_2": datetimes,
            },
            index=index,
        )
        self.df_int_floats = DataFrame(
            {
                "int_1": ints,
                "int_2": ints,
                "int_3": ints,
                "float_1": floats,
                "float_2": floats,
                "float_3": floats,
            },
            index=index,
        )
        self.df_int_float_str = DataFrame(
            {
                "int_1": ints,
                "int_2": ints,
                "float_1": floats,
                "float_2": floats,
                "str_1": strings,
                "str_2": strings,
            },
            index=index,
        )
        self.df_longint_float_str = DataFrame(
            {
                "longint_1": longints,
                "longint_2": longints,
                "float_1": floats,
                "float_2": floats,
                "str_1": strings,
                "str_2": strings,
            },
            index=index,
        )

    def time_floats_with_int_idex_lines(self):
        # [FUNCTION DOC]
        # Times writing the float frame with an integer index as line-delimited
        # records. (The method name misspells the word index.)
        self.df.to_json(self.fname, orient="records", lines=True)

    def time_floats_with_dt_index_lines(self):
        # [FUNCTION DOC]
        # Times writing the float frame with a datetime index as line-delimited
        # records.
        self.df_date_idx.to_json(self.fname, orient="records", lines=True)

    def time_delta_int_tstamp_lines(self):
        # [FUNCTION DOC]
        # Times writing the timedelta/integer/datetime frame as line-delimited
        # records.
        self.df_td_int_ts.to_json(self.fname, orient="records", lines=True)

    def time_float_int_lines(self):
        # [FUNCTION DOC]
        # Times writing the integer/float frame as line-delimited records.
        self.df_int_floats.to_json(self.fname, orient="records", lines=True)

    def time_float_int_str_lines(self):
        # [FUNCTION DOC]
        # Times writing the integer/float/string frame as line-delimited records.
        self.df_int_float_str.to_json(self.fname, orient="records", lines=True)

    def time_float_longint_str_lines(self):
        # [FUNCTION DOC]
        # Times writing the large-integer/float/string frame as line-delimited
        # records.
        self.df_longint_float_str.to_json(self.fname, orient="records", lines=True)


class ToJSONMem:
    # [CLASS DOC]
    # Peak-memory benchmark of repeatedly converting tiny frames to JSON text, to
    # detect memory leaks. Data is built once by the cache hook and passed to each
    # method as a mapping of frames. Does not use the file base class.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns a mapping with three tiny frames: int (a single
        # integer cell), float (the same as float) and datetime (eight values
        # indexed by minute timestamps from 2000-01-01).
        df = DataFrame([[1]])
        df2 = DataFrame(range(8), date_range("1/1/2000", periods=8, freq="min"))
        frames = {"int": df, "float": df.astype(float), "datetime": df2}

        return frames

    def peakmem_int(self, frames):
        # [FUNCTION DOC]
        # Converts the integer frame to JSON text 100000 times, discarding the
        # results.
        df = frames["int"]
        for _ in range(100_000):
            df.to_json()

    def peakmem_float(self, frames):
        # [FUNCTION DOC]
        # Converts the float frame to JSON text 100000 times, discarding the results.
        df = frames["float"]
        for _ in range(100_000):
            df.to_json()

    def peakmem_time(self, frames):
        # [FUNCTION DOC]
        # Converts the datetime-indexed frame to JSON text in the table
        # orientation 10000 times, discarding the results.
        df = frames["datetime"]
        for _ in range(10_000):
            df.to_json(orient="table")


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
