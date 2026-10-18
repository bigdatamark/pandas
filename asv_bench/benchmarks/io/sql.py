# [MODULE DOC]
# Benchmarks for relational database input and output using an in-memory SQLite
# database, through either the SQLAlchemy engine or the standard-library SQLite
# connection. Covers writing frames, running queries, reading whole tables and
# reading single columns of each dtype. Public contents are the benchmark classes
# SQL, WriteSQLDtypes, ReadSQLTable and ReadSQLTableDtypes. Requires the optional
# SQLAlchemy package at import time (the engine factory is imported
# unconditionally). Connections are created in setup and never explicitly closed.
# Final line re-exports the common asv setup hook.
import sqlite3

import numpy as np
from sqlalchemy import create_engine

from pandas import (
    DataFrame,
    Index,
    date_range,
    read_sql_query,
    read_sql_table,
)


class SQL:
    # [CLASS DOC]
    # Benchmark of writing a mixed-dtype frame to a table and reading it back with
    # a query. Parametrized by connection kind (sqlalchemy, sqlite). State:
    # table_name, query_all, con and df.
    params = ["sqlalchemy", "sqlite"]
    param_names = ["connection"]

    def setup(self, connection):
        # [FUNCTION DOC]
        # Creates a fresh in-memory database connection of each kind, and uses the
        # requested one as con. Sets the table name test_type and a select-all
        # query. Builds a frame of 10000 rows (string index) with columns float,
        # float_with_nan (random normal, with rows 1000 to 2999 set to missing),
        # string (constant), bool (all true), int (random), datetime (per-second
        # from 2000-01-01), and derived date, time and datetime_string columns. Writes
        # it to the table, replacing any existing one (side effect on the database).
        N = 10000
        con = {
            "sqlalchemy": create_engine("sqlite:///:memory:"),
            "sqlite": sqlite3.connect(":memory:"),
        }
        self.table_name = "test_type"
        self.query_all = f"SELECT * FROM {self.table_name}"
        self.con = con[connection]
        self.df = DataFrame(
            {
                "float": np.random.randn(N),
                "float_with_nan": np.random.randn(N),
                "string": ["foo"] * N,
                "bool": [True] * N,
                "int": np.random.randint(0, N, size=N),
                "datetime": date_range("2000-01-01", periods=N, freq="s"),
            },
            index=Index([f"i-{i}" for i in range(N)], dtype=object),
        )
        self.df.iloc[1000:3000, 1] = np.nan
        self.df["date"] = self.df["datetime"].dt.date
        self.df["time"] = self.df["datetime"].dt.time
        self.df["datetime_string"] = self.df["datetime"].astype(str)
        self.df.to_sql(self.table_name, self.con, if_exists="replace")

    def time_to_sql_dataframe(self, connection):
        # [FUNCTION DOC]
        # Times writing the frame to a table named test1, replacing it if present.
        # The connection parameter is unused here.
        self.df.to_sql("test1", self.con, if_exists="replace")

    def time_read_sql_query(self, connection):
        # [FUNCTION DOC]
        # Times running the select-all query against the pre-populated table.
        read_sql_query(self.query_all, self.con)


class WriteSQLDtypes:
    # [CLASS DOC]
    # Benchmark of writing and reading a single column of each dtype kind.
    # Parametrized by connection kind and column name (float, float_with_nan,
    # string, bool, int, date, time, datetime). State: table_name, query_col, con
    # and df.
    params = (
        ["sqlalchemy", "sqlite"],
        [
            "float",
            "float_with_nan",
            "string",
            "bool",
            "int",
            "date",
            "time",
            "datetime",
        ],
    )
    param_names = ["connection", "dtype"]

    def setup(self, connection, dtype):
        # [FUNCTION DOC]
        # Same data and database setup as the general SQL benchmark (a 10000-row
        # mixed frame written to the table test_type of a fresh in-memory
        # database), and additionally builds a query selecting only the column
        # named by dtype from that table.
        N = 10000
        con = {
            "sqlalchemy": create_engine("sqlite:///:memory:"),
            "sqlite": sqlite3.connect(":memory:"),
        }
        self.table_name = "test_type"
        self.query_col = f"SELECT {dtype} FROM {self.table_name}"
        self.con = con[connection]
        self.df = DataFrame(
            {
                "float": np.random.randn(N),
                "float_with_nan": np.random.randn(N),
                "string": ["foo"] * N,
                "bool": [True] * N,
                "int": np.random.randint(0, N, size=N),
                "datetime": date_range("2000-01-01", periods=N, freq="s"),
            },
            index=Index([f"i-{i}" for i in range(N)], dtype=object),
        )
        self.df.iloc[1000:3000, 1] = np.nan
        self.df["date"] = self.df["datetime"].dt.date
        self.df["time"] = self.df["datetime"].dt.time
        self.df["datetime_string"] = self.df["datetime"].astype(str)
        self.df.to_sql(self.table_name, self.con, if_exists="replace")

    def time_to_sql_dataframe_column(self, connection, dtype):
        # [FUNCTION DOC]
        # Times writing only the selected column to a table named test1, replacing
        # it if present.
        self.df[[dtype]].to_sql("test1", self.con, if_exists="replace")

    def time_read_sql_query_select_column(self, connection, dtype):
        # [FUNCTION DOC]
        # Times running the single-column query against the pre-populated table.
        read_sql_query(self.query_col, self.con)


class ReadSQLTable:
    # [CLASS DOC]
    # Benchmark of reading a whole table, and a date-string column with date
    # parsing, using the SQLAlchemy engine. No parameters. State: table_name, con
    # and df.
    def setup(self):
        # [FUNCTION DOC]
        # Creates a fresh in-memory SQLAlchemy engine and writes the same 10000-row
        # mixed frame as the other classes in this module to a table named test,
        # replacing any existing one.
        N = 10000
        self.table_name = "test"
        self.con = create_engine("sqlite:///:memory:")
        self.df = DataFrame(
            {
                "float": np.random.randn(N),
                "float_with_nan": np.random.randn(N),
                "string": ["foo"] * N,
                "bool": [True] * N,
                "int": np.random.randint(0, N, size=N),
                "datetime": date_range("2000-01-01", periods=N, freq="s"),
            },
            index=Index([f"i-{i}" for i in range(N)], dtype=object),
        )
        self.df.iloc[1000:3000, 1] = np.nan
        self.df["date"] = self.df["datetime"].dt.date
        self.df["time"] = self.df["datetime"].dt.time
        self.df["datetime_string"] = self.df["datetime"].astype(str)
        self.df.to_sql(self.table_name, self.con, if_exists="replace")

    def time_read_sql_table_all(self):
        # [FUNCTION DOC]
        # Times reading the entire table.
        read_sql_table(self.table_name, self.con)

    def time_read_sql_table_parse_dates(self):
        # [FUNCTION DOC]
        # Times reading only the datetime_string column and parsing it as dates.
        read_sql_table(
            self.table_name,
            self.con,
            columns=["datetime_string"],
            parse_dates=["datetime_string"],
        )


class ReadSQLTableDtypes:
    # [CLASS DOC]
    # Benchmark of reading a single column of each dtype kind from a table with the
    # SQLAlchemy engine. Parametrized by column name. State: table_name, con and df.
    params = [
        "float",
        "float_with_nan",
        "string",
        "bool",
        "int",
        "date",
        "time",
        "datetime",
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Creates a fresh in-memory SQLAlchemy engine and writes the same 10000-row
        # mixed frame as the other classes in this module to a table named test,
        # replacing any existing one. The dtype parameter is unused here.
        N = 10000
        self.table_name = "test"
        self.con = create_engine("sqlite:///:memory:")
        self.df = DataFrame(
            {
                "float": np.random.randn(N),
                "float_with_nan": np.random.randn(N),
                "string": ["foo"] * N,
                "bool": [True] * N,
                "int": np.random.randint(0, N, size=N),
                "datetime": date_range("2000-01-01", periods=N, freq="s"),
            },
            index=Index([f"i-{i}" for i in range(N)], dtype=object),
        )
        self.df.iloc[1000:3000, 1] = np.nan
        self.df["date"] = self.df["datetime"].dt.date
        self.df["time"] = self.df["datetime"].dt.time
        self.df["datetime_string"] = self.df["datetime"].astype(str)
        self.df.to_sql(self.table_name, self.con, if_exists="replace")

    def time_read_sql_table_column(self, dtype):
        # [FUNCTION DOC]
        # Times reading only the column named by dtype from the table.
        read_sql_table(self.table_name, self.con, columns=[dtype])


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
