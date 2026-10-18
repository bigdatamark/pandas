"""
self-contained to write legacy storage pickle files

To use this script. Create an environment where you want
generate pickles, say its for 0.20.3, with your pandas clone
in ~/pandas

. activate pandas_0.20.3
cd ~/pandas/pandas

$ python -m tests.io.generate_legacy_storage_files \
    tests/io/data/legacy_pickle/0.20.3/ pickle

This script generates a storage file for the current arch, system,
and python version
  pandas version: 0.20.3
  output dir    : pandas/pandas/tests/io/data/legacy_pickle/0.20.3/
  storage format: pickle
created pickle file: 0.20.3_x86_64_darwin_3.5.2.pickle

The idea here is you are using the *current* version of the
generate_legacy_storage_files with an *older* version of pandas to
generate a pickle file. We will then check this file into a current
branch, and test using test_pickle.py. This will load the *older*
pickles and test versus the current data that is generated
(with main). These are then compared.

If we have cases where we changed the signature (e.g. we renamed
offset -> freq in Timestamp). Then we have to conditionally execute
in the generate_legacy_storage_files.py to make it
run under the older AND the newer version.

"""

# [MODULE DOC]
# Stand-alone script (not a test module) that writes legacy storage files
# (a pickle of many pandas objects, or HDF5 files of an all-dtypes frame) using
# whichever pandas version is installed, so the files can later be checked in
# and loaded by the current test suite. Public contents: the data builders
# create_pickle_data and create_dataframe_all_types, the naming helper
# platform_name, the writers write_legacy_pickles and write_legacy_hdf, and the
# command-line driver write_legacy_file, run when the file is executed as a
# program. Import-time side effects: it removes the first entry of the module
# search path (so the sibling json test directory is not mistaken for the
# standard json module) and then imports numpy and pandas, so it must be
# imported or run before anything depends on that entry.
# KNOWN ISSUE: the module docstring only documents the pickle storage type, but the driver also supports an hdf storage type that writes a fixed-format and a table-format file.

from datetime import timedelta
import os
import pickle
import platform as pl
import sys

# Remove script directory from path, otherwise Python will try to
# import the JSON test directory as the json module
sys.path.pop(0)

import numpy as np

import pandas
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    NaT,
    Period,
    RangeIndex,
    Series,
    Timestamp,
    bdate_range,
    date_range,
    interval_range,
    period_range,
    timedelta_range,
)
from pandas.arrays import SparseArray

from pandas.tseries.offsets import (
    FY5253,
    BusinessDay,
    BusinessHour,
    CustomBusinessDay,
    DateOffset,
    Day,
    Easter,
    Hour,
    LastWeekOfMonth,
    Minute,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    WeekOfMonth,
    YearBegin,
    YearEnd,
)


def _create_sp_series():
    # [FUNCTION DOC]
    # HELPER: builds the sparse series used by create_pickle_data. Takes no inputs.
    # Creates fifteen floating-point values counting up from zero, sets positions
    # seven through eleven and the last one to missing, wraps them in a block-kind
    # sparse array inside a series, names it bseries and returns it.
    nan = np.nan

    # nan-based
    arr = np.arange(15, dtype=np.float64)
    arr[7:12] = nan
    arr[-1:] = nan

    bseries = Series(SparseArray(arr, kind="block"))
    bseries.name = "bseries"
    return bseries


def _create_sp_tsseries():
    # [FUNCTION DOC]
    # HELPER: builds the sparse time-indexed series for create_pickle_data. Same
    # sparse values as the plain sparse series (fifteen floats, missing at seven
    # through eleven and the last), but the series has a business-day date index
    # starting 2011-01-01 with one entry per value, and is named btsseries. Returns
    # the series.
    nan = np.nan

    # nan-based
    arr = np.arange(15, dtype=np.float64)
    arr[7:12] = nan
    arr[-1:] = nan

    date_index = bdate_range("1/1/2011", periods=len(arr))
    bseries = Series(SparseArray(arr, kind="block"), index=date_index)
    bseries.name = "btsseries"
    return bseries


def _create_sp_frame():
    # [FUNCTION DOC]
    # HELPER: builds the sparse frame for create_pickle_data. Creates four columns
    # of ten values (two with missing runs at different positions, one int64
    # range, one with trailing missing values), indexes them by ten business days
    # starting 2011-01-01, converts every column to a sparse array and returns the
    # frame.
    nan = np.nan

    data = {
        "A": [nan, nan, nan, 0, 1, 2, 3, 4, 5, 6],
        "B": [0, 1, 2, nan, nan, nan, 3, 4, 5, 6],
        "C": np.arange(10).astype(np.int64),
        "D": [0, 1, 2, 3, 4, 5, nan, nan, nan, nan],
    }

    dates = bdate_range("1/1/2011", periods=10)
    return DataFrame(data, index=dates).apply(SparseArray)


def create_pickle_data(test: bool = True):
    # [FUNCTION DOC]
    # Builds the dictionary of sample pandas objects that is pickled for legacy
    # compatibility tests. Parameter test (default true): when true, two extra
    # timestamp entries (one with a frequency, one with timezone) are added because
    # they exist in legacy pickles from version 1.4 and earlier; write_legacy_pickles
    # passes false so newly generated files omit them. Returns a dictionary with
    # the keys series, frame, index, scalars, mi, sp_series, sp_frame, cat,
    # timestamp and offsets. Each maps names to sample objects: series of float,
    # int, mixed, datetime, timezone-aware, period, categorical, duplicate-index,
    # multi-index and string types; frames of float, int, mixed, multi-index,
    # duplicate columns, categorical, mixed time zones and string types; many index
    # types (integer, date, period, float, unsigned, timedelta, string, range,
    # interval); a two-level multi-index; scalar timestamp and period; categoricals
    # with small, medium and large category counts; timestamp variants including a
    # missing value; and one instance of many date offset types with assorted
    # arguments. No side effects.
    """create the pickle data"""
    data = {
        "A": [0.0, 1.0, 2.0, 3.0, np.nan],
        "B": [0, 1, 0, 1, 0],
        "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
        "D": date_range("1/1/2009", periods=5),
        "E": [0.0, 1, Timestamp("20100101"), "foo", 2.0],
    }

    scalars = {"timestamp": Timestamp("20130101"), "period": Period("2012", "M")}

    index = {
        "int": Index(np.arange(10)),
        "date": date_range("20130101", periods=10),
        "period": period_range("2013-01-01", freq="M", periods=10),
        "float": Index(np.arange(10, dtype=np.float64)),
        "uint": Index(np.arange(10, dtype=np.uint64)),
        "timedelta": timedelta_range("00:00:00", freq="30min", periods=10),
        "string": Index(["foo", "bar", "baz", "qux", "quux"], dtype="string"),
    }

    index["range"] = RangeIndex(10)

    index["interval"] = interval_range(0, periods=10)

    mi = {
        "reg2": MultiIndex.from_tuples(
            tuple(
                zip(
                    *[
                        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                        ["one", "two", "one", "two", "one", "two", "one", "two"],
                    ]
                )
            ),
            names=["first", "second"],
        )
    }

    series = {
        "float": Series(data["A"]),
        "int": Series(data["B"]),
        "mixed": Series(data["E"]),
        "ts": Series(
            np.arange(10).astype(np.int64), index=date_range("20130101", periods=10)
        ),
        "mi": Series(
            np.arange(5).astype(np.float64),
            index=MultiIndex.from_tuples(
                tuple(zip(*[[1, 1, 2, 2, 2], [3, 4, 3, 4, 5]])), names=["one", "two"]
            ),
        ),
        "dup": Series(np.arange(5).astype(np.float64), index=["A", "B", "C", "D", "A"]),
        "cat": Series(Categorical(["foo", "bar", "baz"])),
        "dt": Series(date_range("20130101", periods=5)),
        "dt_tz": Series(date_range("20130101", periods=5, tz="US/Eastern")),
        "period": Series([Period("2000Q1")] * 5),
        "string": Series(["foo", "bar", "baz", "qux", "quux"], dtype="string"),
    }

    mixed_dup_df = DataFrame(data)
    mixed_dup_df.columns = list("ABCDA")
    frame = {
        "float": DataFrame({"A": series["float"], "B": series["float"] + 1}),
        "int": DataFrame({"A": series["int"], "B": series["int"] + 1}),
        "mixed": DataFrame({k: data[k] for k in ["A", "B", "C", "D"]}),
        "mi": DataFrame(
            {"A": np.arange(5).astype(np.float64), "B": np.arange(5).astype(np.int64)},
            index=MultiIndex.from_tuples(
                tuple(
                    zip(
                        *[
                            ["bar", "bar", "baz", "baz", "baz"],
                            ["one", "two", "one", "two", "three"],
                        ]
                    )
                ),
                names=["first", "second"],
            ),
        ),
        "dup": DataFrame(
            np.arange(15).reshape(5, 3).astype(np.float64), columns=["A", "B", "A"]
        ),
        "cat_onecol": DataFrame({"A": Categorical(["foo", "bar"])}),
        "cat_and_float": DataFrame(
            {
                "A": Categorical(["foo", "bar", "baz"]),
                "B": np.arange(3).astype(np.int64),
            }
        ),
        "mixed_dup": mixed_dup_df,
        "dt_mixed_tzs": DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        ),
        "dt_mixed2_tzs": DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
                "C": Timestamp("20130603", tz="UTC"),
            },
            index=range(5),
        ),
        "string": DataFrame(
            {
                "A": Series(["foo", "bar", "baz", "qux", "quux"], dtype="string"),
                "B": Series(["one", "two", "one", "two", "three"], dtype="string"),
            }
        ),
    }

    cat = {
        "int8": Categorical(list("abcdefg")),
        "int16": Categorical(np.arange(1000)),
        "int32": Categorical(np.arange(10000)),
    }

    timestamp = {
        "normal": Timestamp("2011-01-01"),
        "nat": NaT,
        "tz": Timestamp("2011-01-01", tz="US/Eastern"),
    }
    if test:
        # kept because those are present in the legacy pickles (<= 1.4)
        timestamp["freq"] = Timestamp("2011-01-01")
        timestamp["both"] = Timestamp("2011-01-01", tz="Asia/Tokyo")

    off = {
        "DateOffset": DateOffset(years=1),
        "DateOffset_h_ns": DateOffset(hour=6, nanoseconds=5824),
        "BusinessDay": BusinessDay(offset=timedelta(seconds=9)),
        "BusinessHour": BusinessHour(normalize=True, n=6, end="15:14"),
        "CustomBusinessDay": CustomBusinessDay(weekmask="Mon Fri"),
        "SemiMonthBegin": SemiMonthBegin(day_of_month=9),
        "SemiMonthEnd": SemiMonthEnd(day_of_month=24),
        "MonthBegin": MonthBegin(1),
        "MonthEnd": MonthEnd(1),
        "QuarterBegin": QuarterBegin(1),
        "QuarterEnd": QuarterEnd(1),
        "Day": Day(1),
        "YearBegin": YearBegin(1),
        "YearEnd": YearEnd(1),
        "Week": Week(1),
        "Week_Tues": Week(2, normalize=False, weekday=1),
        "WeekOfMonth": WeekOfMonth(week=3, weekday=4),
        "LastWeekOfMonth": LastWeekOfMonth(n=1, weekday=3),
        "FY5253": FY5253(n=2, weekday=6, startingMonth=7, variation="last"),
        "Easter": Easter(),
        "Hour": Hour(1),
        "Minute": Minute(1),
    }

    return {
        "series": series,
        "frame": frame,
        "index": index,
        "scalars": scalars,
        "mi": mi,
        "sp_series": {"float": _create_sp_series(), "ts": _create_sp_tsseries()},
        "sp_frame": {"float": _create_sp_frame()},
        "cat": cat,
        "timestamp": timestamp,
        "offsets": off,
    }


def create_dataframe_all_types():
    # [FUNCTION DOC]
    # Builds a five-row frame with one column per supported dtype family for the
    # HDF legacy files: python-int, unsigned 64-bit, float, 32-bit float, bool,
    # datetime at four resolutions (ns, us, ms, s), tz-aware datetime at ns and us,
    # timedelta at four resolutions, and an integer categorical with one missing
    # value. The datetime and timedelta columns contain a missing value in the
    # second row. String and object columns are present only as commented-out
    # entries. Returns the frame.
    timestamps = Series(
        [
            Timestamp("2013-01-01"),
            NaT,
            Timestamp("2013-01-03"),
            Timestamp("2013-01-04"),
            Timestamp("2013-01-05"),
        ]
    )
    timedeltas = timestamps - timestamps[0]

    data = {
        # "string": Series(
        #     ["a", "b", "c", None, "e"], dtype=StringDtype(na_value=np.nan)
        # ),
        # "object": Series(["a", "b", "c", None, "e"], dtype=object),
        # "object_nan": Series(["a", "b", "c", np.nan, "e"], dtype=object),
        "int": list(range(1, 6)),
        "uint64": np.arange(3, 8).astype("uint64"),
        "float": [0.1, 0.2, 0.3, 0.4, np.nan],
        "float32": Series([0.1, 0.2, 0.3, 0.4, np.nan], dtype="float32"),
        "bool": [True, False, True, False, True],
        "datetime_ns": timestamps.dt.as_unit("ns"),
        "datetime_us": timestamps.dt.as_unit("us"),
        "datetime_ms": timestamps.dt.as_unit("ms"),
        "datetime_s": timestamps.dt.as_unit("s"),
        "datetimetz_ns": timestamps.dt.tz_localize("US/Eastern").dt.as_unit("ns"),
        "datetimetz_us": timestamps.dt.tz_localize("US/Eastern").dt.as_unit("us"),
        "timedelta_ns": timedeltas.dt.as_unit("ns"),
        "timedelta_us": timedeltas.dt.as_unit("us"),
        "timedelta_ms": timedeltas.dt.as_unit("ms"),
        "timedelta_s": timedeltas.dt.as_unit("s"),
        # "categorical": Categorical(
        #     Series(
        #         ["foo", "bar", "baz",np.nan,"foo"],dtype=StringDtype(na_value=np.nan)
        #     )
        # ),
        # "categorical_object": Categorical(
        #     Series(["foo", "bar", "baz", np.nan, "foo"], dtype=object)
        # ),
        "categorical_int": Categorical([1, 2, 3, np.nan, 1]),
    }
    return DataFrame(data)


def platform_name():
    # [FUNCTION DOC]
    # Returns a string identifying the current environment, made of the pandas
    # version, machine architecture, lowercase system name and Python version
    # joined by underscores. Used to name generated files.
    return "_".join(
        [
            str(pandas.__version__),
            str(pl.machine()),
            str(pl.system().lower()),
            str(pl.python_version()),
        ]
    )


def write_legacy_pickles(output_dir):
    # [FUNCTION DOC]
    # Writes the dictionary from create_pickle_data (called with test false) to a
    # file in output_dir (a directory path that must exist) named with the platform
    # name and a pickle extension, using the default pickle protocol. Prints a line
    # naming the created file. Returns nothing.
    pth = f"{platform_name()}.pickle"

    with open(os.path.join(output_dir, pth), "wb") as fh:
        pickle.dump(create_pickle_data(test=False), fh, pickle.DEFAULT_PROTOCOL)

    print(f"created pickle file: {pth}")


def write_legacy_hdf(output_dir, format):
    # [FUNCTION DOC]
    # Writes the all-dtypes frame to an HDF5 file in output_dir (existing directory).
    # Parameter format must be fixed or table; it goes into the file name together
    # with the pytables version and platform name. Imports tables lazily, so a
    # missing pytables raises an import error. For the fixed format the categorical
    # integer column is dropped because it is unsupported there; table format uses
    # compression level 9, fixed uses none. The frame is stored under the key
    # df_alltypes. Prints the created file name. Returns nothing.
    import tables

    pth = f"{platform_name()}_pytables-{tables.__version__}_{format}.h5"

    df = create_dataframe_all_types()
    if format == "fixed":
        # df = df.drop(columns=["categorical", "categorical_object", "categorical_int"])
        df = df.drop(columns=["categorical_int"])
    complevel = 9 if format == "table" else None
    df.to_hdf(
        os.path.join(output_dir, pth),
        key="df_alltypes",
        format=format,
        complevel=complevel,
    )

    print(f"created hdf file: {pth}")


def write_legacy_file():
    # [FUNCTION DOC]
    # Command-line driver. Inserts an empty entry at the front of the module search
    # path. Expects the argument vector to hold the output directory and the storage
    # type (pickle or hdf). If the argument count is outside the accepted range it
    # exits with a usage message. Prints a banner describing the pandas version,
    # output directory and storage format, creates the output directory (one level
    # only) if missing, then dispatches: pickle writes one pickle file, hdf writes a
    # fixed and then a table file, anything else exits with an error message. Side
    # effects: global search path mutation, console output, file creation, process
    # exit on bad input.
    # KNOWN ISSUE: the accepted argument-count check allows up to three user arguments while only two are ever read, so a stray extra argument is silently ignored; also the comment about forcing the current directory to be searched first contradicts the import-time removal of the first search path entry.
    # force our cwd to be the first searched
    sys.path.insert(0, "")

    if not 3 <= len(sys.argv) <= 4:
        sys.exit(
            "Specify output directory and storage type: generate_legacy_"
            "storage_files.py <output_dir> <storage_type> "
        )

    output_dir = str(sys.argv[1])
    storage_type = str(sys.argv[2])

    print(
        "This script generates a storage file for the current arch, system, "
        "and python version"
    )
    print(f"  pandas version: {pandas.__version__}")
    print(f"  output dir    : {output_dir}")
    print(f"  storage format: {storage_type}")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if storage_type == "pickle":
        write_legacy_pickles(output_dir=output_dir)
    elif storage_type == "hdf":
        write_legacy_hdf(output_dir=output_dir, format="fixed")
        write_legacy_hdf(output_dir=output_dir, format="table")
    else:
        sys.exit("storage_type must be one of {'pickle', 'hdf'}")


if __name__ == "__main__":
    write_legacy_file()
