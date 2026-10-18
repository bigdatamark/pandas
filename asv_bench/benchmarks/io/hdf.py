# [MODULE DOC]
# Benchmarks for the hierarchical data store: fixed and table formats, reading,
# writing, querying and representing a store, and reading/writing a single frame
# to a file with the one-call interface. Public contents are the benchmark classes
# HDFStoreDataFrame and HDF, both derived from the shared file-benchmark base
# class and writing a temporary file named __test__.h5 in the working directory.
# Final line re-exports the common asv setup hook.
import numpy as np

from pandas import (
    DataFrame,
    HDFStore,
    Index,
    date_range,
    read_hdf,
)

from ..pandas_vb_common import BaseIO


class HDFStoreDataFrame(BaseIO):
    # [CLASS DOC]
    # Benchmark of operations on an open store pre-populated with fixed-format and
    # table-format frames. No parameters. The store is opened in setup and closed in
    # teardown. State: df, df_mixed, df_wide, start_wide, stop_wide, df2, start,
    # stop, df_wide2, df_dc, fname and store. Timing methods that write mutate
    # the store, so it grows across repeated runs within one setup.
    def setup(self):
        # [FUNCTION DOC]
        # Builds sample frames of 25000 rows: df (two float columns, string index);
        # df_mixed (floats, a constant string, a boolean, an integer, string index);
        # df_wide (100 random columns); df2 (two float columns, daily dates from
        # 2000-01-01) with start/stop timestamps at positions 10000 and 15000;
        # df_wide2 (100 random columns with daily dates); df_dc (10 random columns
        # named C000 to C009). The wide frame bounds start_wide and stop_wide are
        # row labels 10000 and 15000. Opens a store on the file __test__.h5 (side
        # effect: creates the file) and stores df and df_mixed in fixed format, and
        # df2, df_mixed, df_wide and df_wide2 in table format under the names
        # fixed, fixed_mixed, table, table_mixed, table_wide and table_wide2.
        N = 25000
        index = Index([f"i-{i}" for i in range(N)], dtype=object)
        self.df = DataFrame(
            {"float1": np.random.randn(N), "float2": np.random.randn(N)}, index=index
        )
        self.df_mixed = DataFrame(
            {
                "float1": np.random.randn(N),
                "float2": np.random.randn(N),
                "string1": ["foo"] * N,
                "bool1": [True] * N,
                "int1": np.random.randint(0, N, size=N),
            },
            index=index,
        )
        self.df_wide = DataFrame(np.random.randn(N, 100))
        self.start_wide = self.df_wide.index[10000]
        self.stop_wide = self.df_wide.index[15000]
        self.df2 = DataFrame(
            {"float1": np.random.randn(N), "float2": np.random.randn(N)},
            index=date_range("1/1/2000", periods=N),
        )
        self.start = self.df2.index[10000]
        self.stop = self.df2.index[15000]
        self.df_wide2 = DataFrame(
            np.random.randn(N, 100), index=date_range("1/1/2000", periods=N)
        )
        self.df_dc = DataFrame(
            np.random.randn(N, 10), columns=[f"C{i:03d}" for i in range(10)]
        )

        self.fname = "__test__.h5"

        self.store = HDFStore(self.fname)
        self.store.put("fixed", self.df)
        self.store.put("fixed_mixed", self.df_mixed)
        self.store.append("table", self.df2)
        self.store.append("table_mixed", self.df_mixed)
        self.store.append("table_wide", self.df_wide)
        self.store.append("table_wide2", self.df_wide2)

    def teardown(self):
        # [FUNCTION DOC]
        # asv teardown hook. Closes the store, then removes the file (errors on
        # removal are ignored by the base class helper). Overrides the base class
        # teardown with a no-argument signature.
        self.store.close()
        self.remove(self.fname)

    def time_read_store(self):
        # [FUNCTION DOC]
        # Times reading the fixed-format frame from the store.
        self.store.get("fixed")

    def time_read_store_mixed(self):
        # [FUNCTION DOC]
        # Times reading the fixed-format mixed-dtype frame.
        self.store.get("fixed_mixed")

    def time_write_store(self):
        # [FUNCTION DOC]
        # Times writing the simple frame in fixed format under a new key
        # (replacing it on repeated calls).
        self.store.put("fixed_write", self.df)

    def time_write_store_mixed(self):
        # [FUNCTION DOC]
        # Times writing the mixed-dtype frame in fixed format under a new key.
        self.store.put("fixed_mixed_write", self.df_mixed)

    def time_read_store_table_mixed(self):
        # [FUNCTION DOC]
        # Times selecting all rows of the table-format mixed frame.
        self.store.select("table_mixed")

    def time_write_store_table_mixed(self):
        # [FUNCTION DOC]
        # Times appending the mixed-dtype frame to a table (grows the table on each
        # call).
        self.store.append("table_mixed_write", self.df_mixed)

    def time_read_store_table(self):
        # [FUNCTION DOC]
        # Times selecting all rows of the date-indexed table.
        self.store.select("table")

    def time_write_store_table(self):
        # [FUNCTION DOC]
        # Times appending the string-indexed frame to a table (grows it on each
        # call).
        self.store.append("table_write", self.df)

    def time_read_store_table_wide(self):
        # [FUNCTION DOC]
        # Times selecting all rows of the wide table.
        self.store.select("table_wide")

    def time_write_store_table_wide(self):
        # [FUNCTION DOC]
        # Times appending the wide frame to a table (grows it on each call).
        self.store.append("table_wide_write", self.df_wide)

    def time_write_store_table_dc(self):
        # [FUNCTION DOC]
        # Times appending the ten-column frame to a table with every column
        # indexed as a data column (grows it on each call).
        self.store.append("table_dc_write", self.df_dc, data_columns=True)

    def time_query_store_table_wide(self):
        # [FUNCTION DOC]
        # Times selecting from the wide table the rows whose index lies strictly
        # between the stored bounds; the bounds are looked up by name from the
        # instance in the caller's scope by the query expression.
        self.store.select(
            "table_wide", where="index > self.start_wide and index < self.stop_wide"
        )

    def time_query_store_table(self):
        # [FUNCTION DOC]
        # Times selecting from the date-indexed table the rows whose index lies
        # strictly between the stored start and stop timestamps.
        self.store.select("table", where="index > self.start and index < self.stop")

    def time_store_repr(self):
        # [FUNCTION DOC]
        # Times the official text representation of the store.
        repr(self.store)

    def time_store_str(self):
        # [FUNCTION DOC]
        # Times the informal text representation of the store.
        str(self.store)

    def time_store_info(self):
        # [FUNCTION DOC]
        # Times the store info method, which summarizes all stored objects.
        self.store.info()


class HDF(BaseIO):
    # [CLASS DOC]
    # Benchmark of reading and writing a mixed frame through the one-call file
    # interface, in table or fixed format, plus peak memory of reading.
    # Parametrized by format. State: fname, df and df1.
    params = ["table", "fixed"]
    param_names = ["format"]

    def setup(self, format):
        # [FUNCTION DOC]
        # Sets the file name __test__.h5 and builds a frame of 100000 rows (hourly
        # index from 2000-01-01, five random float columns and an object column of
        # distinct strings) as df and writes it to the file under key df in the
        # format (side effect: creates the file). Also makes df1 as a copy with
        # the index reset to a column and writes it under key df1 in the same file.
        self.fname = "__test__.h5"
        N = 100000
        C = 5
        self.df = DataFrame(
            np.random.randn(N, C),
            columns=[f"float{i}" for i in range(C)],
            index=date_range("20000101", periods=N, freq="h"),
        )
        self.df["object"] = Index([f"i-{i}" for i in range(N)], dtype=object)
        self.df.to_hdf(self.fname, key="df", format=format)

        # Numeric df
        self.df1 = self.df.copy()
        self.df1 = self.df1.reset_index()
        self.df1.to_hdf(self.fname, key="df1", format=format)

    def time_read_hdf(self, format):
        # [FUNCTION DOC]
        # Times reading key df from the file. The format parameter is unused here.
        read_hdf(self.fname, "df")

    def peakmem_read_hdf(self, format):
        # [FUNCTION DOC]
        # Peak-memory benchmark of reading key df from the file.
        read_hdf(self.fname, "df")

    def time_write_hdf(self, format):
        # [FUNCTION DOC]
        # Times writing the frame to the file under key df in the requested format
        # (replaces existing content for the key).
        self.df.to_hdf(self.fname, key="df", format=format)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
