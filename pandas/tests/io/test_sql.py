# [MODULE DOC]
# Tests for the SQL io module (reading and writing frames with relational databases):
# the top-level read and write functions (read sql, read sql query, read sql table,
# the frame to-sql method), the sqlite and SQLAlchemy backed database classes, the
# generic database builder, schema generation and table checks. The same shared test
# bodies are run against many connection kinds through fixtures: SQLAlchemy engines and
# connections and connection strings for sqlite, MySQL (pymysql) and PostgreSQL
# (psycopg2), ADBC connections for sqlite and PostgreSQL, and the built-in sqlite3
# connection. Database-server variants need a live server and carry the database marker;
# sqlite variants run locally. The module is marked single-CPU and ignores a block manager
# deprecation warning. Module-level helper functions create and load the iris and types
# tables, list and drop tables and views, and count rows; module-level lists of fixture
# names group the connection kinds. Tests are in a flat set of functions grouped by area
# (read, write, dtype handling, dates, schema, transactions, ADBC specifics, sqlite
# built-in connection tests prefixed with xsqlite). There are no module-level test
# classes; the only classes are small helpers defined inside some test functions.
from __future__ import annotations

import contextlib
import csv
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas.compat import pa_version_under14p1
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    isna,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.util.version import Version

from pandas.io import sql
from pandas.io.sql import (
    SQLAlchemyEngine,
    SQLDatabase,
    SQLiteDatabase,
    get_engine,
    pandasSQL_builder,
    read_sql_query,
    read_sql_table,
)

if TYPE_CHECKING:
    import sqlalchemy


pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
    ),
    pytest.mark.single_cpu,
]


@pytest.fixture
def sql_strings():
    # [FUNCTION DOC]
    # Fixture returning a nested mapping from a query kind (parameterized read, named
    # parameters, a query with a literal percent sign) to a mapping from dialect (sqlite,
    # mysql, postgresql) to the query text in that dialect's parameter and quoting style, over the iris table.
    return {
        "read_parameters": {
            "sqlite": "SELECT * FROM iris WHERE Name=? AND SepalLength=?",
            "mysql": "SELECT * FROM iris WHERE `Name`=%s AND `SepalLength`=%s",
            "postgresql": 'SELECT * FROM iris WHERE "Name"=%s AND "SepalLength"=%s',
        },
        "read_named_parameters": {
            "sqlite": """
                SELECT * FROM iris WHERE Name=:name AND SepalLength=:length
                """,
            "mysql": """
                SELECT * FROM iris WHERE
                `Name`=%(name)s AND `SepalLength`=%(length)s
                """,
            "postgresql": """
                SELECT * FROM iris WHERE
                "Name"=%(name)s AND "SepalLength"=%(length)s
                """,
        },
        "read_no_parameters_with_percent": {
            "sqlite": "SELECT * FROM iris WHERE Name LIKE '%'",
            "mysql": "SELECT * FROM iris WHERE `Name` LIKE '%'",
            "postgresql": "SELECT * FROM iris WHERE \"Name\" LIKE '%'",
        },
    }


def iris_table_metadata():
    # [FUNCTION DOC]
    # HELPER: builds and returns a SQLAlchemy table definition for the iris table (four
    # double-precision float columns and a 200 character name column). Imports
    # SQLAlchemy lazily and uses a double type for SQLAlchemy 2 or later, otherwise float.
    import sqlalchemy
    from sqlalchemy import (
        Column,
        Double,
        Float,
        MetaData,
        String,
        Table,
    )

    dtype = Double if Version(sqlalchemy.__version__) >= Version("2.0.0") else Float
    metadata = MetaData()
    iris = Table(
        "iris",
        metadata,
        Column("SepalLength", dtype),
        Column("SepalWidth", dtype),
        Column("PetalLength", dtype),
        Column("PetalWidth", dtype),
        Column("Name", String(200)),
    )
    return iris


def create_and_load_iris_sqlite3(conn, iris_file: Path):
    # [FUNCTION DOC]
    # HELPER: creates the iris table in a built-in sqlite3 style connection (also used for ADBC sqlite) and loads rows from the iris csv file given by path.
    # The first csv row (header) is skipped and numeric fields are converted to floats because ADBC requires explicit types. Commits at the end.
    # KNOWN ISSUE: the records list is initialised empty and then immediately reassigned, so the first assignment is dead code.
    stmt = """CREATE TABLE iris (
            "SepalLength" REAL,
            "SepalWidth" REAL,
            "PetalLength" REAL,
            "PetalWidth" REAL,
            "Name" TEXT
        )"""

    cur = conn.cursor()
    cur.execute(stmt)
    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        stmt = "INSERT INTO iris VALUES(?, ?, ?, ?, ?)"
        # ADBC requires explicit types - no implicit str -> float conversion
        records = []
        records = [
            (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                row[4],
            )
            for row in reader
        ]

        cur.executemany(stmt, records)
    cur.close()

    conn.commit()


def create_and_load_iris_postgresql(conn, iris_file: Path):
    # [FUNCTION DOC]
    # HELPER: creates the iris table in a PostgreSQL ADBC connection using double precision and text columns, and loads rows from the iris csv path using positional dollar parameters; numeric fields are converted to floats. Commits at the end.
    stmt = """CREATE TABLE iris (
            "SepalLength" DOUBLE PRECISION,
            "SepalWidth" DOUBLE PRECISION,
            "PetalLength" DOUBLE PRECISION,
            "PetalWidth" DOUBLE PRECISION,
            "Name" TEXT
        )"""
    with conn.cursor() as cur:
        cur.execute(stmt)
        with iris_file.open(newline=None, encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            stmt = "INSERT INTO iris VALUES($1, $2, $3, $4, $5)"
            # ADBC requires explicit types - no implicit str -> float conversion
            records = [
                (
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    row[4],
                )
                for row in reader
            ]

            cur.executemany(stmt, records)

    conn.commit()


def create_and_load_iris(conn, iris_file: Path):
    # [FUNCTION DOC]
    # HELPER: creates the iris table on a SQLAlchemy engine or connection from the shared table definition and loads the rows from the csv file (header row used as keys). Inside a transaction it drops any existing iris table, creates it, and inserts all rows.
    from sqlalchemy import insert

    iris = iris_table_metadata()

    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        params = [dict(zip(header, row)) for row in reader]
        stmt = insert(iris).values(params)
        with conn.begin() as con:
            iris.drop(con, checkfirst=True)
            iris.create(bind=con)
            con.execute(stmt)


def create_and_load_iris_view(conn):
    # [FUNCTION DOC]
    # HELPER: creates a view of the iris table named iris view on the given connection. Handles three connection kinds: a built-in sqlite3 connection (executes via a cursor, no commit), an ADBC connection (executes via a cursor and commits) and otherwise a SQLAlchemy connection or engine (within a transaction).
    stmt = "CREATE VIEW iris_view AS SELECT * FROM iris"
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(stmt)
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(stmt)
            conn.commit()
        else:
            from sqlalchemy import text

            stmt = text(stmt)
            with conn.begin() as con:
                con.execute(stmt)


def types_table_metadata(dialect: str):
    # [FUNCTION DOC]
    # HELPER: builds a SQLAlchemy table definition for the types table for the given dialect name. For sqlite the date column is text and boolean columns are integers; for other dialects the date column is a datetime and the boolean columns are real booleans. Includes columns for text, date, integer-encoded dates, float, integer, boolean and nullable integer and boolean.
    from sqlalchemy import (
        TEXT,
        Boolean,
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        Table,
    )

    date_type = TEXT if dialect == "sqlite" else DateTime
    bool_type = Integer if dialect == "sqlite" else Boolean
    metadata = MetaData()
    types = Table(
        "types",
        metadata,
        Column("TextCol", TEXT),
        # error: Cannot infer type argument 1 of "Column"
        Column("DateCol", date_type),  # type: ignore[misc]
        Column("IntDateCol", Integer),
        Column("IntDateOnlyCol", Integer),
        Column("FloatCol", Float),
        Column("IntCol", Integer),
        # error: Cannot infer type argument 1 of "Column"
        Column("BoolCol", bool_type),  # type: ignore[misc]
        Column("IntColWithNull", Integer),
        # error: Cannot infer type argument 1 of "Column"
        Column("BoolColWithNull", bool_type),  # type: ignore[misc]
    )
    return types


def create_and_load_types_sqlite3(conn, types_data: list[dict]):
    # [FUNCTION DOC]
    # HELPER: creates the types table in a built-in sqlite3 (or ADBC sqlite) connection and bulk inserts the rows with positional parameters. For a built-in connection uses a plain cursor; otherwise a context-managed cursor and an explicit commit.
    # KNOWN ISSUE: the parameter is annotated as a list of mappings but callers pass sequences of tuples (positional values).
    stmt = """CREATE TABLE types (
                    "TextCol" TEXT,
                    "DateCol" TEXT,
                    "IntDateCol" INTEGER,
                    "IntDateOnlyCol" INTEGER,
                    "FloatCol" REAL,
                    "IntCol" INTEGER,
                    "BoolCol" INTEGER,
                    "IntColWithNull" INTEGER,
                    "BoolColWithNull" INTEGER
                )"""

    ins_stmt = """
                INSERT INTO types
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(stmt)
        cur.executemany(ins_stmt, types_data)
    else:
        with conn.cursor() as cur:
            cur.execute(stmt)
            cur.executemany(ins_stmt, types_data)

        conn.commit()


def create_and_load_types_postgresql(conn, types_data: list[dict]):
    # [FUNCTION DOC]
    # HELPER: creates the types table in a PostgreSQL ADBC connection with PostgreSQL column types and inserts the given rows (tuples) using numbered parameters, casting the date to timestamp; commits.
    with conn.cursor() as cur:
        stmt = """CREATE TABLE types (
                        "TextCol" TEXT,
                        "DateCol" TIMESTAMP,
                        "IntDateCol" INTEGER,
                        "IntDateOnlyCol" INTEGER,
                        "FloatCol" DOUBLE PRECISION,
                        "IntCol" INTEGER,
                        "BoolCol" BOOLEAN,
                        "IntColWithNull" INTEGER,
                        "BoolColWithNull" BOOLEAN
                    )"""
        cur.execute(stmt)

        stmt = """
                INSERT INTO types
                VALUES($1, $2::timestamp, $3, $4, $5, $6, $7, $8, $9)
                """

        cur.executemany(stmt, types_data)

    conn.commit()


def create_and_load_types(conn, types_data: list[dict], dialect: str):
    # [FUNCTION DOC]
    # HELPER: creates the types table on a SQLAlchemy engine or connection for the dialect name and inserts the rows. For an engine, opens a connection and transaction; otherwise uses the given connection. Drops any existing table first.
    from sqlalchemy import insert
    from sqlalchemy.engine import Engine

    types = types_table_metadata(dialect)

    stmt = insert(types).values(types_data)
    if isinstance(conn, Engine):
        with conn.connect() as conn:
            with conn.begin():
                types.drop(conn, checkfirst=True)
                types.create(bind=conn)
                conn.execute(stmt)
    else:
        with conn.begin():
            types.drop(conn, checkfirst=True)
            types.create(bind=conn)
            conn.execute(stmt)


def create_and_load_postgres_datetz(conn):
    # [FUNCTION DOC]
    # HELPER: creates a PostgreSQL table with a single timezone-aware datetime column and inserts two values given with different UTC offsets (winter and summer). Returns the expected column read back, as a series of microsecond-unit UTC datetimes named for the column.
    from sqlalchemy import (
        Column,
        DateTime,
        MetaData,
        Table,
        insert,
    )
    from sqlalchemy.engine import Engine

    metadata = MetaData()
    datetz = Table("datetz", metadata, Column("DateColWithTz", DateTime(timezone=True)))
    datetz_data = [
        {
            "DateColWithTz": "2000-01-01 00:00:00-08:00",
        },
        {
            "DateColWithTz": "2000-06-01 00:00:00-07:00",
        },
    ]
    stmt = insert(datetz).values(datetz_data)
    if isinstance(conn, Engine):
        with conn.connect() as conn:
            with conn.begin():
                datetz.drop(conn, checkfirst=True)
                datetz.create(bind=conn)
                conn.execute(stmt)
    else:
        with conn.begin():
            datetz.drop(conn, checkfirst=True)
            datetz.create(bind=conn)
            conn.execute(stmt)

    # "2000-01-01 00:00:00-08:00" should convert to
    # "2000-01-01 08:00:00"
    # "2000-06-01 00:00:00-07:00" should convert to
    # "2000-06-01 07:00:00"
    # GH 6415
    expected_data = [
        Timestamp("2000-01-01 08:00:00", tz="UTC"),
        Timestamp("2000-06-01 07:00:00", tz="UTC"),
    ]
    return Series(expected_data, name="DateColWithTz").astype("M8[us, UTC]")


def check_iris_frame(frame: DataFrame):
    # [FUNCTION DOC]
    # HELPER: asserts that a frame has the shape of the iris data: first column dtype is a floating type, the first row equals the known first iris record, and the shape is either the full 150 rows by 5 columns or the 8 by 5 test subset.
    pytype = frame.dtypes.iloc[0].type
    row = frame.iloc[0]
    assert issubclass(pytype, np.floating)
    tm.assert_series_equal(
        row, Series([5.1, 3.5, 1.4, 0.2, "Iris-setosa"], index=frame.columns, name=0)
    )
    assert frame.shape in ((150, 5), (8, 5))


def count_rows(conn, table_name: str):
    # [FUNCTION DOC]
    # HELPER: returns the number of rows in a table by a count query. Dispatches on the connection kind: built-in sqlite3, ADBC, a connection string (creates and disposes an engine), a SQLAlchemy engine or a SQLAlchemy connection.
    stmt = f"SELECT count(*) AS count_1 FROM {table_name}"
    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        return cur.execute(stmt).fetchone()[0]
    elif adbc and isinstance(conn, adbc.Connection):
        with conn.cursor() as cur:
            cur.execute(stmt)
            return cur.fetchone()[0]
    else:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine

        if isinstance(conn, str):
            try:
                engine = create_engine(conn)
                with engine.connect() as conn:
                    return conn.exec_driver_sql(stmt).scalar_one()
            finally:
                engine.dispose()
        elif isinstance(conn, Engine):
            with conn.connect() as conn:
                return conn.exec_driver_sql(stmt).scalar_one()
        else:
            return conn.exec_driver_sql(stmt).scalar_one()


@pytest.fixture
def iris_path(datapath):
    # [FUNCTION DOC]
    # Fixture returning the path object of the iris csv data file in the shared io test data directory.
    iris_path = datapath("io", "data", "csv", "iris.csv")
    return Path(iris_path)


@pytest.fixture
def types_data():
    # [FUNCTION DOC]
    # Fixture returning a list of two mappings (rows) for the types table, covering text, date string, integer-encoded date columns, float, integer, boolean and nullable integer and boolean values (the second row has missing values).
    return [
        {
            "TextCol": "first",
            "DateCol": "2000-01-03 00:00:00",
            "IntDateCol": 535852800,
            "IntDateOnlyCol": 20101010,
            "FloatCol": 10.10,
            "IntCol": 1,
            "BoolCol": False,
            "IntColWithNull": 1,
            "BoolColWithNull": False,
        },
        {
            "TextCol": "first",
            "DateCol": "2000-01-04 00:00:00",
            "IntDateCol": 1356998400,
            "IntDateOnlyCol": 20101212,
            "FloatCol": 10.10,
            "IntCol": 1,
            "BoolCol": False,
            "IntColWithNull": None,
            "BoolColWithNull": None,
        },
    ]


@pytest.fixture
def types_data_frame(types_data):
    # [FUNCTION DOC]
    # Fixture that builds a frame from the types rows with the expected pandas dtypes (strings, 64-bit integers, floats; booleans as integers; nullable columns as float).
    dtypes = {
        "TextCol": "str",
        "DateCol": "str",
        "IntDateCol": "int64",
        "IntDateOnlyCol": "int64",
        "FloatCol": "float",
        "IntCol": "int64",
        "BoolCol": "int64",
        "IntColWithNull": "float",
        "BoolColWithNull": "float",
    }
    df = DataFrame(types_data)
    return df[dtypes.keys()].astype(dtypes)


@pytest.fixture
def test_frame1():
    # [FUNCTION DOC]
    # Fixture returning a four-row frame with a string index column of dates and four float columns, used for write and round trip tests.
    columns = ["index", "A", "B", "C", "D"]
    data = [
        (
            "2000-01-03 00:00:00",
            0.980268513777,
            3.68573087906,
            -0.364216805298,
            -1.15973806169,
        ),
        (
            "2000-01-04 00:00:00",
            1.04791624281,
            -0.0412318367011,
            -0.16181208307,
            0.212549316967,
        ),
        (
            "2000-01-05 00:00:00",
            0.498580885705,
            0.731167677815,
            -0.537677223318,
            1.34627041952,
        ),
        (
            "2000-01-06 00:00:00",
            1.12020151869,
            1.56762092543,
            0.00364077397681,
            0.67525259227,
        ),
    ]
    return DataFrame(data, columns=columns)


@pytest.fixture
def test_frame3():
    # [FUNCTION DOC]
    # Fixture returning a four-row frame with a string index column of dates, an integer column (including the maximum 32-bit value) and a float column.
    columns = ["index", "A", "B"]
    data = [
        ("2000-01-03 00:00:00", 2**31 - 1, -1.987670),
        ("2000-01-04 00:00:00", -29, -0.0412318367011),
        ("2000-01-05 00:00:00", 20000, 0.731167677815),
        ("2000-01-06 00:00:00", -290867, 1.56762092543),
    ]
    return DataFrame(data, columns=columns)


def get_all_views(conn):
    # [FUNCTION DOC]
    # HELPER: returns the list of view names in the database behind the connection. Dispatches on built-in sqlite3 (queries the master table), ADBC (walks the catalog objects and collects entries of type view) or SQLAlchemy (inspector).
    # KNOWN ISSUE: in the ADBC branch, two lines index the catalog and schema dictionaries for names and discard the results; they have no effect.
    if isinstance(conn, sqlite3.Connection):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        return [view[0] for view in c.fetchall()]
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            results = []
            info = conn.adbc_get_objects().read_all().to_pylist()
            for catalog in info:
                catalog["catalog_name"]
                for schema in catalog["catalog_db_schemas"]:
                    schema["db_schema_name"]
                    for table in schema["db_schema_tables"]:
                        if table["table_type"] == "view":
                            view_name = table["table_name"]
                            results.append(view_name)

            return results
        else:
            from sqlalchemy import inspect

            return inspect(conn).get_view_names()


def get_all_tables(conn):
    # [FUNCTION DOC]
    # HELPER: returns the list of table names in the database behind the connection; same dispatch as listing views but collecting entries of type table.
    if isinstance(conn, sqlite3.Connection):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [table[0] for table in c.fetchall()]
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")

        if adbc and isinstance(conn, adbc.Connection):
            results = []
            info = conn.adbc_get_objects().read_all().to_pylist()
            for catalog in info:
                for schema in catalog["catalog_db_schemas"]:
                    for table in schema["db_schema_tables"]:
                        if table["table_type"] == "table":
                            table_name = table["table_name"]
                            results.append(table_name)

            return results
        else:
            from sqlalchemy import inspect

            return inspect(conn).get_table_names()


def drop_table(
    table_name: str,
    conn: sqlite3.Connection | sqlalchemy.engine.Engine | sqlalchemy.engine.Connection,
):
    # [FUNCTION DOC]
    # HELPER: drops the named table if it exists. Takes the table name and a built-in sqlite3 connection, SQLAlchemy engine or SQLAlchemy connection (or an ADBC connection). Built-in sqlite3 uses a quoted name and commits; ADBC uses a cursor; SQLAlchemy opens a transaction and uses the database wrapper drop-table method.
    if isinstance(conn, sqlite3.Connection):
        conn.execute(f"DROP TABLE IF EXISTS {sql._get_valid_sqlite_name(table_name)}")
        conn.commit()

    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        else:
            with conn.begin() as con:
                with sql.SQLDatabase(con) as db:
                    db.drop_table(table_name)


def drop_view(
    view_name: str,
    conn: sqlite3.Connection | sqlalchemy.engine.Engine | sqlalchemy.engine.Connection,
):
    # [FUNCTION DOC]
    # HELPER: drops the named view if it exists, with the same connection dispatch as dropping a table: built-in sqlite3 (commit), ADBC (cursor, no commit), and SQLAlchemy (quoted identifier through the dialect in a transaction).
    import sqlalchemy

    if isinstance(conn, sqlite3.Connection):
        conn.execute(f"DROP VIEW IF EXISTS {sql._get_valid_sqlite_name(view_name)}")
        conn.commit()
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(f'DROP VIEW IF EXISTS "{view_name}"')
        else:
            quoted_view = conn.engine.dialect.identifier_preparer.quote_identifier(
                view_name
            )
            stmt = sqlalchemy.text(f"DROP VIEW IF EXISTS {quoted_view}")
            with conn.begin() as con:
                con.execute(stmt)  # type: ignore[union-attr]


@pytest.fixture
def mysql_pymysql_engine():
    # [FUNCTION DOC]
    # Fixture (skips if SQLAlchemy or pymysql are missing) yielding a SQLAlchemy engine to a local MySQL server with multi-statement support and no connection pooling. After the test it drops all views and then all tables and disposes the engine.
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pymysql = pytest.importorskip("pymysql")
    engine = sqlalchemy.create_engine(
        "mysql+pymysql://root@localhost:3306/pandas",
        connect_args={"client_flag": pymysql.constants.CLIENT.MULTI_STATEMENTS},
        poolclass=sqlalchemy.pool.NullPool,
    )
    yield engine
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    engine.dispose()


@pytest.fixture
def mysql_pymysql_engine_iris(mysql_pymysql_engine, iris_path):
    # [FUNCTION DOC]
    # Fixture: the MySQL engine with the iris table and view created and loaded; returns the engine.
    create_and_load_iris(mysql_pymysql_engine, iris_path)
    create_and_load_iris_view(mysql_pymysql_engine)
    return mysql_pymysql_engine


@pytest.fixture
def mysql_pymysql_engine_types(mysql_pymysql_engine, types_data):
    # [FUNCTION DOC]
    # Fixture: the MySQL engine with the types table created and loaded; returns the engine.
    create_and_load_types(mysql_pymysql_engine, types_data, "mysql")
    return mysql_pymysql_engine


@pytest.fixture
def mysql_pymysql_conn(mysql_pymysql_engine):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the MySQL engine (closed afterwards).
    with mysql_pymysql_engine.connect() as conn:
        yield conn


@pytest.fixture
def mysql_pymysql_conn_iris(mysql_pymysql_engine_iris):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the MySQL engine with iris loaded.
    with mysql_pymysql_engine_iris.connect() as conn:
        yield conn


@pytest.fixture
def mysql_pymysql_conn_types(mysql_pymysql_engine_types):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the MySQL engine with the types table loaded.
    with mysql_pymysql_engine_types.connect() as conn:
        yield conn


@pytest.fixture
def postgresql_psycopg2_engine():
    # [FUNCTION DOC]
    # Fixture (skips if SQLAlchemy or psycopg2 are missing) yielding a SQLAlchemy engine for a local PostgreSQL server without pooling; afterwards drops all views and tables and disposes the engine.
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg2")
    engine = sqlalchemy.create_engine(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/pandas",
        poolclass=sqlalchemy.pool.NullPool,
    )
    yield engine
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    engine.dispose()


@pytest.fixture
def postgresql_psycopg2_engine_iris(postgresql_psycopg2_engine, iris_path):
    # [FUNCTION DOC]
    # Fixture: the PostgreSQL psycopg2 engine with the iris table and view created and loaded; returns the engine.
    create_and_load_iris(postgresql_psycopg2_engine, iris_path)
    create_and_load_iris_view(postgresql_psycopg2_engine)
    return postgresql_psycopg2_engine


@pytest.fixture
def postgresql_psycopg2_engine_types(postgresql_psycopg2_engine, types_data):
    # [FUNCTION DOC]
    # Fixture: the PostgreSQL psycopg2 engine with the types table loaded (using the postgres dialect); returns the engine.
    create_and_load_types(postgresql_psycopg2_engine, types_data, "postgres")
    return postgresql_psycopg2_engine


@pytest.fixture
def postgresql_psycopg2_conn(postgresql_psycopg2_engine):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the PostgreSQL psycopg2 engine.
    with postgresql_psycopg2_engine.connect() as conn:
        yield conn


@pytest.fixture
def postgresql_adbc_conn():
    # [FUNCTION DOC]
    # Fixture (skips without pyarrow and the ADBC PostgreSQL driver) yielding an ADBC connection to a local PostgreSQL server; after the test drops all views and tables and commits.
    pytest.importorskip("pyarrow")
    pytest.importorskip("adbc_driver_postgresql")
    from adbc_driver_postgresql import dbapi

    uri = "postgresql://postgres:postgres@localhost:5432/pandas"
    with dbapi.connect(uri) as conn:
        yield conn
        for view in get_all_views(conn):
            drop_view(view, conn)
        for tbl in get_all_tables(conn):
            drop_table(tbl, conn)
        conn.commit()


@pytest.fixture
def postgresql_adbc_iris(postgresql_adbc_conn, iris_path):
    # [FUNCTION DOC]
    # Fixture: on the PostgreSQL ADBC connection, creates and loads the iris table and the view only if their schemas are not found (a programming error triggers a rollback and the creation). Returns the connection.
    import adbc_driver_manager as mgr

    conn = postgresql_adbc_conn

    try:
        conn.adbc_get_table_schema("iris")
    except mgr.ProgrammingError:
        conn.rollback()
        create_and_load_iris_postgresql(conn, iris_path)
    try:
        conn.adbc_get_table_schema("iris_view")
    except mgr.ProgrammingError:  # note arrow-adbc issue 1022
        conn.rollback()
        create_and_load_iris_view(conn)
    return conn


@pytest.fixture
def postgresql_adbc_types(postgresql_adbc_conn, types_data):
    # [FUNCTION DOC]
    # Fixture: on the PostgreSQL ADBC connection, creates and loads the types table unless it already exists, converting the row mappings to tuples. Returns the connection.
    import adbc_driver_manager as mgr

    conn = postgresql_adbc_conn

    try:
        conn.adbc_get_table_schema("types")
    except mgr.ProgrammingError:
        conn.rollback()
        new_data = [tuple(entry.values()) for entry in types_data]

        create_and_load_types_postgresql(conn, new_data)

    return conn


@pytest.fixture
def postgresql_psycopg2_conn_iris(postgresql_psycopg2_engine_iris):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the PostgreSQL psycopg2 engine with iris loaded.
    with postgresql_psycopg2_engine_iris.connect() as conn:
        yield conn


@pytest.fixture
def postgresql_psycopg2_conn_types(postgresql_psycopg2_engine_types):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the PostgreSQL psycopg2 engine with types loaded.
    with postgresql_psycopg2_engine_types.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_str(temp_file):
    # [FUNCTION DOC]
    # Fixture (skips if SQLAlchemy is missing) returning a sqlite connection string pointing at the temporary file.
    pytest.importorskip("sqlalchemy")
    return f"sqlite:///{temp_file}"


@pytest.fixture
def sqlite_engine(sqlite_str):
    # [FUNCTION DOC]
    # Fixture yielding a SQLAlchemy engine for the sqlite temporary file (no pooling); afterwards drops all views and tables and disposes it.
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str, poolclass=sqlalchemy.pool.NullPool)
    yield engine
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    engine.dispose()


@pytest.fixture
def sqlite_conn(sqlite_engine):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the sqlite engine.
    with sqlite_engine.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_str_iris(sqlite_str, iris_path):
    # [FUNCTION DOC]
    # Fixture: creates and loads iris and its view via a temporary engine on the sqlite string (disposed afterwards) and returns the connection string.
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str)
    create_and_load_iris(engine, iris_path)
    create_and_load_iris_view(engine)
    engine.dispose()
    return sqlite_str


@pytest.fixture
def sqlite_engine_iris(sqlite_engine, iris_path):
    # [FUNCTION DOC]
    # Fixture: the sqlite engine with iris and its view created and loaded; returns the engine.
    create_and_load_iris(sqlite_engine, iris_path)
    create_and_load_iris_view(sqlite_engine)
    return sqlite_engine


@pytest.fixture
def sqlite_conn_iris(sqlite_engine_iris):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the sqlite engine with iris loaded.
    with sqlite_engine_iris.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_str_types(sqlite_str, types_data):
    # [FUNCTION DOC]
    # Fixture: creates and loads the types table through a temporary engine on the sqlite string and returns the connection string.
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str)
    create_and_load_types(engine, types_data, "sqlite")
    engine.dispose()
    return sqlite_str


@pytest.fixture
def sqlite_engine_types(sqlite_engine, types_data):
    # [FUNCTION DOC]
    # Fixture: the sqlite engine with the types table loaded; returns the engine.
    create_and_load_types(sqlite_engine, types_data, "sqlite")
    return sqlite_engine


@pytest.fixture
def sqlite_conn_types(sqlite_engine_types):
    # [FUNCTION DOC]
    # Fixture yielding a connection from the sqlite engine with types loaded.
    with sqlite_engine_types.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_adbc_conn(temp_file):
    # [FUNCTION DOC]
    # Fixture (skips without pyarrow and the ADBC sqlite driver) yielding an ADBC sqlite connection to the temporary file; afterwards drops views and tables and commits.
    pytest.importorskip("pyarrow")
    pytest.importorskip("adbc_driver_sqlite")
    from adbc_driver_sqlite import dbapi

    uri = f"file:{temp_file}"
    with dbapi.connect(uri) as conn:
        yield conn
        for view in get_all_views(conn):
            drop_view(view, conn)
        for tbl in get_all_tables(conn):
            drop_table(tbl, conn)
        conn.commit()


@pytest.fixture
def sqlite_adbc_iris(sqlite_adbc_conn, iris_path):
    # [FUNCTION DOC]
    # Fixture: on the sqlite ADBC connection creates and loads iris and its view unless they already exist (rolling back on the programming error); returns the connection.
    import adbc_driver_manager as mgr

    conn = sqlite_adbc_conn
    try:
        conn.adbc_get_table_schema("iris")
    except mgr.ProgrammingError:
        conn.rollback()
        create_and_load_iris_sqlite3(conn, iris_path)
    try:
        conn.adbc_get_table_schema("iris_view")
    except mgr.ProgrammingError:
        conn.rollback()
        create_and_load_iris_view(conn)
    return conn


@pytest.fixture
def sqlite_adbc_types(sqlite_adbc_conn, types_data):
    # [FUNCTION DOC]
    # Fixture: on the sqlite ADBC connection creates and loads the types table unless it exists, first converting boolean values to integers (mutating the row mappings) and the rows to tuples; commits. Returns the connection.
    # KNOWN ISSUE: the shared row mappings from the types data fixture are mutated in place.
    import adbc_driver_manager as mgr

    conn = sqlite_adbc_conn
    try:
        conn.adbc_get_table_schema("types")
    except mgr.ProgrammingError:
        conn.rollback()
        new_data = []
        for entry in types_data:
            entry["BoolCol"] = int(entry["BoolCol"])
            if entry["BoolColWithNull"] is not None:
                entry["BoolColWithNull"] = int(entry["BoolColWithNull"])
            new_data.append(tuple(entry.values()))

        create_and_load_types_sqlite3(conn, new_data)
        conn.commit()

    return conn


@pytest.fixture
def sqlite_buildin():
    # [FUNCTION DOC]
    # Fixture yielding a built-in sqlite3 in-memory connection, wrapped so the connection is closed and the transaction context managed on exit.
    with contextlib.closing(sqlite3.connect(":memory:")) as closing_conn:
        with closing_conn as conn:
            yield conn


@pytest.fixture
def sqlite_buildin_iris(sqlite_buildin, iris_path):
    # [FUNCTION DOC]
    # Fixture: built-in sqlite3 connection with iris and its view loaded; returns it.
    create_and_load_iris_sqlite3(sqlite_buildin, iris_path)
    create_and_load_iris_view(sqlite_buildin)
    return sqlite_buildin


@pytest.fixture
def sqlite_buildin_types(sqlite_buildin, types_data):
    # [FUNCTION DOC]
    # Fixture: built-in sqlite3 connection with types loaded from tuples built from the row mappings; returns it.
    types_data = [tuple(entry.values()) for entry in types_data]
    create_and_load_types_sqlite3(sqlite_buildin, types_data)
    return sqlite_buildin


mysql_connectable = [
    pytest.param("mysql_pymysql_engine", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn", marks=pytest.mark.db),
]

mysql_connectable_iris = [
    pytest.param("mysql_pymysql_engine_iris", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn_iris", marks=pytest.mark.db),
]

mysql_connectable_types = [
    pytest.param("mysql_pymysql_engine_types", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn_types", marks=pytest.mark.db),
]

postgresql_connectable = [
    pytest.param("postgresql_psycopg2_engine", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn", marks=pytest.mark.db),
]

postgresql_connectable_iris = [
    pytest.param("postgresql_psycopg2_engine_iris", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn_iris", marks=pytest.mark.db),
]

postgresql_connectable_types = [
    pytest.param("postgresql_psycopg2_engine_types", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn_types", marks=pytest.mark.db),
]

sqlite_connectable = [
    "sqlite_engine",
    "sqlite_conn",
    "sqlite_str",
]

sqlite_connectable_iris = [
    "sqlite_engine_iris",
    "sqlite_conn_iris",
    "sqlite_str_iris",
]

sqlite_connectable_types = [
    "sqlite_engine_types",
    "sqlite_conn_types",
    "sqlite_str_types",
]

sqlalchemy_connectable = mysql_connectable + postgresql_connectable + sqlite_connectable

sqlalchemy_connectable_iris = (
    mysql_connectable_iris + postgresql_connectable_iris + sqlite_connectable_iris
)

sqlalchemy_connectable_types = (
    mysql_connectable_types + postgresql_connectable_types + sqlite_connectable_types
)

adbc_connectable = [
    "sqlite_adbc_conn",
    pytest.param("postgresql_adbc_conn", marks=pytest.mark.db),
]

adbc_connectable_iris = [
    pytest.param("postgresql_adbc_iris", marks=pytest.mark.db),
    "sqlite_adbc_iris",
]

adbc_connectable_types = [
    pytest.param("postgresql_adbc_types", marks=pytest.mark.db),
    "sqlite_adbc_types",
]


all_connectable = [*sqlalchemy_connectable, "sqlite_buildin", *adbc_connectable]

all_connectable_iris = [
    *sqlalchemy_connectable_iris,
    "sqlite_buildin_iris",
    *adbc_connectable_iris,
]

all_connectable_types = [
    *sqlalchemy_connectable_types,
    "sqlite_buildin_types",
    *adbc_connectable_types,
]


@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql(conn, test_frame1, request):
    # [FUNCTION DOC]
    # Parametrized over every connection kind (SQLAlchemy engines/connections/strings, built-in sqlite3, ADBC). Regression smoke test: writes the first sample frame to a table, appending, without the index; passes if no error is raised.
    # GH 51086 if conn is sqlite_engine
    conn = request.getfixturevalue(conn)
    test_frame1.to_sql(name="test", con=conn, if_exists="append", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_empty(conn, test_frame1, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writes an empty (zero-row) slice of the sample frame, appending, without the index. For the PostgreSQL ADBC connection when string dtype inference is off and the driver is older than 1.11, the test is marked as an expected failure since the driver cannot insert an index of null type. Passes if no error is raised.
    if conn == "postgresql_adbc_conn" and not using_string_dtype():
        adbc_pg = pytest.importorskip("adbc_driver_postgresql")
        if Version(adbc_pg.__version__) < Version("1.11"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        "postgres ADBC driver < 1.11 cannot insert index with null type"
                    ),
                )
            )

    # GH 51086 if conn is sqlite_engine
    conn = request.getfixturevalue(conn)
    empty_df = test_frame1.iloc[:0]
    empty_df.to_sql(name="test", con=conn, if_exists="append", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Requires pyarrow. Writes a frame with Arrow-backed integer, datetime, date, duration and string columns, replacing any existing table. ADBC sqlite drops the duration column first. Expected warnings: ADBC with old pyarrow expects a deprecation warning about sparse checks; non-ADBC connections expect a user warning about the timedelta column; otherwise none.
    # GH 52046
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "int": pd.array([1], dtype="int8[pyarrow]"),
            "datetime": pd.array(
                [datetime(2023, 1, 1)], dtype="timestamp[ns][pyarrow]"
            ),
            "date": pd.array([date(2023, 1, 1)], dtype="date32[day][pyarrow]"),
            "timedelta": pd.array([timedelta(1)], dtype="duration[ns][pyarrow]"),
            "string": pd.array(["a"], dtype="string[pyarrow]"),
        }
    )

    if "adbc" in conn:
        if conn == "sqlite_adbc_conn":
            df = df.drop(columns=["timedelta"])
        if pa_version_under14p1:
            exp_warning = DeprecationWarning
            msg = "is_sparse is deprecated"
        else:
            exp_warning = None
            msg = ""
    else:
        exp_warning = UserWarning
        msg = "the 'timedelta'"

    conn = request.getfixturevalue(conn)
    with tm.assert_produces_warning(exp_warning, match=msg, check_stacklevel=False):
        df.to_sql(name="test_arrow", con=conn, if_exists="replace", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes_missing(conn, request, nulls_fixture):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds and the null-value fixture. Requires pyarrow; skipped for the decimal NaN null. Writes an Arrow timestamp column containing one real value and one null-like value, replacing the table; passes if no error is raised.
    # GH 52046
    pytest.importorskip("pyarrow")
    if isinstance(nulls_fixture, Decimal):
        pytest.skip(
            # GH#61773
            reason="Decimal('NaN') not supported in constructor for timestamp dtype"
        )

    df = DataFrame(
        {
            "datetime": pd.array(
                [datetime(2023, 1, 1), nulls_fixture], dtype="timestamp[ns][pyarrow]"
            ),
        }
    )
    conn = request.getfixturevalue(conn)
    df.to_sql(name="test_arrow", con=conn, if_exists="replace", index=False)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("method", [None, "multi"])
def test_to_sql(conn, method, test_frame1, request):
    # [FUNCTION DOC]
    # Parametrized over connection kinds and insertion method (default or multi-row). Writes the sample frame with a database builder in a transaction and checks the table exists and the row count equals the frame length. The multi-row method with ADBC is a strict expected failure because the method option is not supported there.
    if method == "multi" and "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'method' not implemented for ADBC drivers", strict=True
            )
        )

    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=method)
        assert pandasSQL.has_table("test_frame")
    assert count_rows(conn, "test_frame") == len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize(
    "mode, num_row_coef", [("replace", 1), ("append", 2), ("delete_rows", 1)]
)
def test_to_sql_exist(conn, mode, num_row_coef, test_frame1, request):
    # [FUNCTION DOC]
    # Parametrized over connection kinds and existing-table mode (replace, append, delete rows). Writes the sample frame (failing if exists) and then again with the mode; checks the table exists and the row count is the length times 1 (replace, delete rows) or 2 (append).
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists=mode)
        assert pandasSQL.has_table("test_frame")
    assert count_rows(conn, "test_frame") == num_row_coef * len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_exist_fail(conn, test_frame1, request):
    # [FUNCTION DOC]
    # Parametrized over connection kinds. After writing a table, a second write with the fail-if-exists mode raises a value error saying the table already exists.
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        assert pandasSQL.has_table("test_frame")

        msg = "Table 'test_frame' already exists"
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Reads the full iris table via the query function and the general read function and checks the iris frame shape and values; a query that returns no rows still has the five columns.
    conn = request.getfixturevalue(conn)
    iris_frame = read_sql_query("SELECT * FROM iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris where 0=1", conn)
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_chunksize(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Reads the iris table with chunk size seven using both read functions, concatenates the chunks, and checks the frame; an empty result concatenates to a frame with five columns. ADBC is a strict expected failure since chunk size is not supported.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'chunksize' not implemented for ADBC drivers",
                strict=True,
            )
        )
    conn = request.getfixturevalue(conn)
    iris_frame = concat(read_sql_query("SELECT * FROM iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris where 0=1", conn, chunksize=7))
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_query_expression_with_parameter(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy iris connection kinds. Builds a SQLAlchemy select over the reflected iris table (using an engine created from the string when needed) and reads it with the query function giving a parameters mapping; the result is the full iris frame. Disposes the temporary engine when a connection string was used.
    # KNOWN ISSUE: the ADBC expected-failure branch can never run because the connection kinds list contains no ADBC entries, and its reason text talks about chunk size which this test does not use.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'chunksize' not implemented for ADBC drivers",
                strict=True,
            )
        )
    conn = request.getfixturevalue(conn)
    from sqlalchemy import (
        MetaData,
        Table,
        create_engine,
        select,
    )

    metadata = MetaData()
    autoload_con = create_engine(conn) if isinstance(conn, str) else conn
    iris = Table("iris", metadata, autoload_with=autoload_con)
    iris_frame = read_sql_query(
        select(iris), conn, params={"name": "Iris-setosa", "length": 5.1}
    )
    check_iris_frame(iris_frame)
    if isinstance(conn, str):
        autoload_con.dispose()


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_string_with_parameter(conn, request, sql_strings):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Picks the parameterized query for the dialect found in the fixture name (raising a key error when none matches), then reads with a sequence of parameters (name and length); the result is the iris frame. ADBC is a strict expected failure.
    # KNOWN ISSUE: the expected-failure reason text says chunk size is unsupported although the test exercises query parameters.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'chunksize' not implemented for ADBC drivers",
                strict=True,
            )
        )

    for db, query in sql_strings["read_parameters"].items():
        if db in conn:
            break
    else:
        raise KeyError(f"No part of {conn} found in sql_strings['read_parameters']")
    conn = request.getfixturevalue(conn)
    iris_frame = read_sql_query(query, conn, params=("Iris-setosa", 5.1))
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy iris connection kinds. Reads the iris table by name with the table reader and the general reader; both give the iris frame (regression for connection strings with sqlite).
    # GH 51015 if conn = sqlite_iris_str
    conn = request.getfixturevalue(conn)
    iris_frame = read_sql_table("iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("iris", conn)
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table_chunksize(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy iris connection kinds. Reads the iris table by name in chunks of seven with both read functions and checks the concatenated frame.
    # KNOWN ISSUE: the ADBC expected-failure branch is unreachable because the connection kinds list has no ADBC entries.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    conn = request.getfixturevalue(conn)
    iris_frame = concat(read_sql_table("iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("iris", conn, chunksize=7))
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_to_sql_callable(conn, test_frame1, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Writes the sample frame through a user-supplied callable insertion method that records its invocation, converts the iterator rows to mappings and executes a bulk insert. Checks the callable was called exactly once, the table exists and all rows were inserted.
    conn = request.getfixturevalue(conn)

    check = []  # used to double check function below is really being used

    def sample(pd_table, conn, keys, data_iter):
        # [FUNCTION DOC]
        # HELPER: custom insertion method used by the callable-method test. Receives the pandas table object, the connection, column names and a row iterator; appends a marker to the enclosing list to show it ran, builds one mapping per row from keys and values and runs a bulk insert on the table. Returns nothing.
        check.append(1)
        data = [dict(zip(keys, row)) for row in data_iter]
        conn.execute(pd_table.table.insert(), data)

    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=sample)
        assert pandasSQL.has_table("test_frame")
    assert check == [1]
    assert count_rows(conn, "test_frame") == len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable_types)
def test_default_type_conversion(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all types connection kinds. Reads the types table (expected failure for the built-in sqlite3 connection since it lacks the table reader) and checks the default dtypes: floating for float, integer for integer, boolean for PostgreSQL else integer, integer-with-missing as float, and boolean-with-missing as object for PostgreSQL else float.
    conn_name = conn
    if conn_name == "sqlite_buildin_types":
        request.applymarker(
            pytest.mark.xfail(
                reason="sqlite_buildin connection does not implement read_sql_table"
            )
        )

    conn = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn)

    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)

    # MySQL/sqlite has no real BOOL type
    if "postgresql" in conn_name:
        assert issubclass(df.BoolCol.dtype.type, np.bool_)
    else:
        assert issubclass(df.BoolCol.dtype.type, np.integer)

    # Int column with NA values stays as float
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)

    # Bool column with NA = int column with NA values => becomes float
    if "postgresql" in conn_name:
        assert issubclass(df.BoolColWithNull.dtype.type, object)
    else:
        assert issubclass(df.BoolColWithNull.dtype.type, np.floating)


@pytest.mark.parametrize("conn", mysql_connectable)
def test_read_procedure(conn, request):
    # [FUNCTION DOC]
    # Parametrized over MySQL connections. Regression for stored procedures: writes a small frame, creates a procedure that selects from it (via a transaction on the engine or connection), then reads the result of calling it with both the query function and the general read function; both equal the frame.
    conn = request.getfixturevalue(conn)

    # GH 7324
    # Although it is more an api test, it is added to the
    # mysql tests as sqlite does not have stored procedures
    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df.to_sql(name="test_frame", con=conn, index=False)

    proc = """DROP PROCEDURE IF EXISTS get_testdb;

    CREATE PROCEDURE get_testdb ()

    BEGIN
        SELECT * FROM test_frame;
    END"""
    proc = text(proc)
    if isinstance(conn, Engine):
        with conn.connect() as engine_conn:
            with engine_conn.begin():
                engine_conn.execute(proc)
    else:
        with conn.begin():
            conn.execute(proc)

    res1 = sql.read_sql_query("CALL get_testdb();", conn)
    tm.assert_frame_equal(df, res1)

    # test delegation to read_sql_query
    res2 = sql.read_sql("CALL get_testdb();", conn)
    tm.assert_frame_equal(df, res2)


@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("expected_count", [2, "Success!"])
def test_copy_from_callable_insertion_method(conn, expected_count, request):
    # [FUNCTION DOC]
    # Parametrized over PostgreSQL connections and the value returned by the custom method (2 or a string). Regression test using the documented copy-based insertion method: writes a frame through a callable that streams csv rows into the table with a bulk copy and returns the given value; the return value of the write must equal it and the table read back must equal the frame.
    # KNOWN ISSUE: the branch handling a none expected count can never run because the parametrization only uses a number and a string.
    # GH 8953
    # Example in io.rst found under _io.sql.method
    # not available in sqlite, mysql
    def psql_insert_copy(table, conn, keys, data_iter):
        # [FUNCTION DOC]
        # HELPER: insertion method for the copy test. Takes the pandas table, the connection, column names and a row iterator; gets the underlying DBAPI connection, writes the rows to an in-memory csv buffer, builds a copy statement with the quoted column list and the schema-qualified table name, executes it through the driver's copy-expert call, and returns the expected count taken from the enclosing test parameter.
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ", ".join([f'"{k}"' for k in keys])
            if table.schema:
                table_name = f"{table.schema}.{table.name}"
            else:
                table_name = table.name

            sql_query = f"COPY {table_name} ({columns}) FROM STDIN WITH CSV"
            cur.copy_expert(sql=sql_query, file=s_buf)
        return expected_count

    conn = request.getfixturevalue(conn)
    expected = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    result_count = expected.to_sql(
        name="test_frame", con=conn, index=False, method=psql_insert_copy
    )
    # GH 46891
    if expected_count is None:
        assert result_count is None
    else:
        assert result_count == expected_count
    result = sql.read_sql_table("test_frame", conn)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", postgresql_connectable)
def test_insertion_method_on_conflict_do_nothing(conn, request):
    # [FUNCTION DOC]
    # Parametrized over PostgreSQL connections. Regression test of the documented example for conflict handling: creates a table with a primary key, writes a row, then writes a conflicting row using a custom method that inserts with on-conflict-do-nothing. The table must be unchanged (equal to the first frame) and the reported number of inserted rows is zero. Cleans up by dropping the table.
    # GH 15988: Example in to_sql docstring
    conn = request.getfixturevalue(conn)

    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def insert_on_conflict(table, conn, keys, data_iter):
        # [FUNCTION DOC]
        # HELPER: insertion method for the on-conflict test. Builds mappings from keys and rows, creates a PostgreSQL insert with an on-conflict-do-nothing clause on the key column, executes it and returns the row count.
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = (
            insert(table.table)
            .values(data)
            .on_conflict_do_nothing(index_elements=["a"])
        )
        result = conn.execute(stmt)
        return result.rowcount

    create_sql = text(
        """
    CREATE TABLE test_insert_conflict (
        a  integer PRIMARY KEY,
        b  numeric,
        c  text
    );
    """
    )
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with conn.begin():
            conn.execute(create_sql)

    expected = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    expected.to_sql(
        name="test_insert_conflict", con=conn, if_exists="append", index=False
    )

    df_insert = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    inserted = df_insert.to_sql(
        name="test_insert_conflict",
        con=conn,
        index=False,
        if_exists="append",
        method=insert_on_conflict,
    )
    result = sql.read_sql_table("test_insert_conflict", conn)
    tm.assert_frame_equal(result, expected)
    assert inserted == 0

    # Cleanup
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")


@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_on_public_schema(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writes and reads a frame using the public schema explicitly; expected failure for sqlite and MySQL since the schema is PostgreSQL specific. The re-read frame equals the written one.
    if "sqlite" in conn or "mysql" in conn:
        request.applymarker(
            pytest.mark.xfail(
                reason="test for public schema only specific to postgresql"
            )
        )

    conn = request.getfixturevalue(conn)

    test_data = DataFrame([[1, 2.1, "a"], [2, 3.1, "b"]], columns=list("abc"))
    test_data.to_sql(
        name="test_public_schema",
        con=conn,
        if_exists="append",
        index=False,
        schema="public",
    )

    df_out = sql.read_sql_table("test_public_schema", conn, schema="public")
    tm.assert_frame_equal(test_data, df_out)


@pytest.mark.parametrize("conn", mysql_connectable)
def test_insertion_method_on_conflict_update(conn, request):
    # [FUNCTION DOC]
    # Parametrized over MySQL connections. Regression test of the documented example: creates a table with a primary key, writes a row, then writes a conflicting row with a custom method using on-duplicate-key-update; the table equals the second frame and the reported row count is two (MySQL counts an update as two). Cleans up by dropping the table.
    # GH 14553: Example in to_sql docstring
    conn = request.getfixturevalue(conn)

    from sqlalchemy.dialects.mysql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def insert_on_conflict(table, conn, keys, data_iter):
        # [FUNCTION DOC]
        # HELPER: insertion method for the MySQL conflict test. Builds mappings from keys and rows, creates a MySQL insert with an on-duplicate-key-update clause updating the second and third columns from the inserted values, executes it and returns the row count.
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data)
        stmt = stmt.on_duplicate_key_update(b=stmt.inserted.b, c=stmt.inserted.c)
        result = conn.execute(stmt)
        return result.rowcount

    create_sql = text(
        """
    CREATE TABLE test_insert_conflict (
        a INT PRIMARY KEY,
        b FLOAT,
        c VARCHAR(10)
    );
    """
    )
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with conn.begin():
            conn.execute(create_sql)

    df = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    df.to_sql(name="test_insert_conflict", con=conn, if_exists="append", index=False)

    expected = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    inserted = expected.to_sql(
        name="test_insert_conflict",
        con=conn,
        index=False,
        if_exists="append",
        method=insert_on_conflict,
    )
    result = sql.read_sql_table("test_insert_conflict", conn)
    tm.assert_frame_equal(result, expected)
    assert inserted == 2

    # Cleanup
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")


@pytest.mark.parametrize("conn", postgresql_connectable)
def test_read_view_postgres(conn, request):
    # [FUNCTION DOC]
    # Parametrized over PostgreSQL connections. Regression test: creates a table, inserts a row and creates a view over it with randomly named objects, then reads the view with the table reader.
    # GH 52969
    conn = request.getfixturevalue(conn)

    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    table_name = f"group_{uuid.uuid4().hex}"
    view_name = f"group_view_{uuid.uuid4().hex}"

    sql_stmt = text(
        f"""
    CREATE TABLE {table_name} (
        group_id INTEGER,
        name TEXT
    );
    INSERT INTO {table_name} VALUES
        (1, 'name');
    CREATE VIEW {view_name}
    AS
    SELECT * FROM {table_name};
    """
    )
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(sql_stmt)
    else:
        with conn.begin():
            conn.execute(sql_stmt)
    result = read_sql_table(view_name, conn)
    expected = DataFrame({"group_id": [1], "name": "name"})
    tm.assert_frame_equal(result, expected)


def test_read_view_sqlite(sqlite_buildin):
    # [FUNCTION DOC]
    # Using a built-in sqlite3 connection, creates a table, inserts a row and creates a view, then reads the view with a query; result equals the expected one-row frame. Regression for views.
    # GH 52969
    create_table = """
CREATE TABLE groups (
   group_id INTEGER,
   name TEXT
);
"""
    insert_into = """
INSERT INTO groups VALUES
    (1, 'name');
"""
    create_view = """
CREATE VIEW group_view
AS
SELECT * FROM groups;
"""
    sqlite_buildin.execute(create_table)
    sqlite_buildin.execute(insert_into)
    sqlite_buildin.execute(create_view)
    result = pd.read_sql("SELECT * FROM group_view", sqlite_buildin)
    expected = DataFrame({"group_id": [1], "name": "name"})
    tm.assert_frame_equal(result, expected)


def flavor(conn_name):
    # [FUNCTION DOC]
    # HELPER: maps a connection fixture name to the dialect name (postgresql, sqlite or mysql) by substring, checked in that order; raises a value error for an unsupported name.
    if "postgresql" in conn_name:
        return "postgresql"
    elif "sqlite" in conn_name:
        return "sqlite"
    elif "mysql" in conn_name:
        return "mysql"

    raise ValueError(f"unsupported connection: {conn_name}")


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_parameter(conn, request, sql_strings):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Reads the parameterized iris query for the dialect through a database builder inside a transaction with a sequence of parameters, and checks the iris frame. ADBC is a strict expected failure (parameters unsupported).
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'params' not implemented for ADBC drivers",
                strict=True,
            )
        )
    conn_name = conn
    conn = request.getfixturevalue(conn)
    query = sql_strings["read_parameters"][flavor(conn_name)]
    params = ("Iris-setosa", 5.1)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=params)
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_named_parameter(conn, request, sql_strings):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Same as the previous test but with a named-parameter query and a mapping of parameters. ADBC is a strict expected failure.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'params' not implemented for ADBC drivers",
                strict=True,
            )
        )

    conn_name = conn
    conn = request.getfixturevalue(conn)
    query = sql_strings["read_named_parameters"][flavor(conn_name)]
    params = {"name": "Iris-setosa", "length": 5.1}
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=params)
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_no_parameter_with_percent(conn, request, sql_strings):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Reads a query containing a literal percent sign with no parameters; expected failure (marked broken) for MySQL and non-ADBC PostgreSQL connections. Result is the iris frame.
    if "mysql" in conn or ("postgresql" in conn and "adbc" not in conn):
        request.applymarker(pytest.mark.xfail(reason="broken test"))

    conn_name = conn
    conn = request.getfixturevalue(conn)

    query = sql_strings["read_no_parameters_with_percent"][flavor(conn_name)]
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=None)
    check_iris_frame(iris_frame)


# -----------------------------------------------------------------------------
# -- Testing the public API


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_api_read_sql_view(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Public API: reads the iris view with the query function and checks the iris frame.
    conn = request.getfixturevalue(conn)
    iris_frame = sql.read_sql_query("SELECT * FROM iris_view", conn)
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_api_read_sql_with_chunksize_no_result(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Reading a query that returns no rows with a chunk size gives chunks whose concatenation equals the non-chunked result. ADBC is an expected failure (chunk size unsupported).
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    conn = request.getfixturevalue(conn)
    query = 'SELECT * FROM iris_view WHERE "SepalLength" < 0.0'
    with_batch = sql.read_sql_query(query, conn, chunksize=5)
    without_batch = sql.read_sql_query(query, conn)
    tm.assert_frame_equal(concat(with_batch), without_batch)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Public API: drops the test table if it exists, writes the sample frame with the public function and checks the table exists.
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame1", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame1")

    sql.to_sql(test_frame1, "test_frame1", conn)
    assert sql.has_table("test_frame1", conn)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_fail(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Public API: after dropping any existing table, writes with fail-if-exists, then a second write raises a value error that the table already exists.
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame2", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame2")

    sql.to_sql(test_frame1, "test_frame2", conn, if_exists="fail")
    assert sql.has_table("test_frame2", conn)

    msg = "Table 'test_frame2' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(test_frame1, "test_frame2", conn, if_exists="fail")


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_replace(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Public API: writes with fail-if-exists then again with replace; the table exists and has the same number of rows as the frame.
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame3", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame3")

    sql.to_sql(test_frame1, "test_frame3", conn, if_exists="fail")
    # Add to table again
    sql.to_sql(test_frame1, "test_frame3", conn, if_exists="replace")
    assert sql.has_table("test_frame3", conn)

    num_entries = len(test_frame1)
    num_rows = count_rows(conn, "test_frame3")

    assert num_rows == num_entries


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_append(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Public API: the write function returns the number of rows written (four), first with fail-if-exists and then with append; the table ends up with twice the number of rows.
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame4", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame4")

    assert sql.to_sql(test_frame1, "test_frame4", conn, if_exists="fail") == 4

    # Add to table again
    assert sql.to_sql(test_frame1, "test_frame4", conn, if_exists="append") == 4
    assert sql.has_table("test_frame4", conn)

    num_entries = 2 * len(test_frame1)
    num_rows = count_rows(conn, "test_frame4")

    assert num_rows == num_entries


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_type_mapping(conn, request, test_frame3):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writes the integer-and-float sample frame without the index and reads it back with a plain query; the frames are equal (types mapped without change).
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame5", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame5")

    sql.to_sql(test_frame3, "test_frame5", conn, index=False)
    result = sql.read_sql("SELECT * FROM test_frame5", conn)

    tm.assert_frame_equal(test_frame3, result)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_series(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writing a series (named, 64-bit integers) without index and reading back gives a frame equal to the series converted to a frame.
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_series", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_series")

    s = Series(np.arange(5, dtype="int64"), name="series")
    sql.to_sql(s, "test_series", conn, index=False)
    s2 = sql.read_sql_query("SELECT * FROM test_series", conn)
    tm.assert_frame_equal(s.to_frame(), s2)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_roundtrip(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writes the sample frame with its index and reads it back with a query; the extra index column (named level_0, or a generated name for ADBC) is dropped before comparing with the original.
    # KNOWN ISSUE: the code labels the dropping of the index column as a hack; the expected index column name depends on the backend.
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")

    sql.to_sql(test_frame1, "test_frame_roundtrip", con=conn)
    result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=conn)

    # HACK!
    if "adbc" in conn_name:
        result = result.drop(columns="__index_level_0__")
    else:
        result = result.drop(columns="level_0")
    tm.assert_frame_equal(result, test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_roundtrip_chunksize(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writes the sample frame in chunks of two rows, without the index, and reads it back; equal to the original. ADBC is an expected failure (chunk size unsupported).
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")

    sql.to_sql(
        test_frame1,
        "test_frame_roundtrip",
        con=conn,
        index=False,
        chunksize=2,
    )
    result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=conn)
    tm.assert_frame_equal(result, test_frame1)


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_api_execute_sql(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Through a database builder, executes a select over the iris table, fetches one row and closes the result; the first row equals the known first iris record.
    # drop_sql = "DROP TABLE IF EXISTS test"  # should already be done
    conn = request.getfixturevalue(conn)
    with sql.pandasSQL_builder(conn) as pandas_sql:
        iris_results = pandas_sql.execute("SELECT * FROM iris")
        row = iris_results.fetchone()
        iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]


@pytest.mark.parametrize("conn", all_connectable_types)
def test_api_date_parsing(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all types connection kinds. Exercises date parsing in the query reader: no parsing leaves the date column non-datetime (except MySQL and PostgreSQL, which return datetimes natively); parsing by name, by a name with format mapping, integer epoch seconds column by name and with the seconds unit, and an integer date-only column with a year-month-day format each give the expected timestamps.
    conn_name = conn
    conn = request.getfixturevalue(conn)
    # Test date parsing in read_sql
    # No Parsing
    df = sql.read_sql_query("SELECT * FROM types", conn)
    if not ("mysql" in conn_name or "postgres" in conn_name):
        assert not issubclass(df.DateCol.dtype.type, np.datetime64)

    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates=["DateCol"])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [
        Timestamp(2000, 1, 3, 0, 0, 0),
        Timestamp(2000, 1, 4, 0, 0, 0),
    ]

    df = sql.read_sql_query(
        "SELECT * FROM types",
        conn,
        parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"},
    )
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [
        Timestamp(2000, 1, 3, 0, 0, 0),
        Timestamp(2000, 1, 4, 0, 0, 0),
    ]

    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates=["IntDateCol"])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [
        Timestamp(1986, 12, 25, 0, 0, 0),
        Timestamp(2013, 1, 1, 0, 0, 0),
    ]

    df = sql.read_sql_query(
        "SELECT * FROM types", conn, parse_dates={"IntDateCol": "s"}
    )
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [
        Timestamp(1986, 12, 25, 0, 0, 0),
        Timestamp(2013, 1, 1, 0, 0, 0),
    ]

    df = sql.read_sql_query(
        "SELECT * FROM types",
        conn,
        parse_dates={"IntDateOnlyCol": "%Y%m%d"},
    )
    assert issubclass(df.IntDateOnlyCol.dtype.type, np.datetime64)
    assert df.IntDateOnlyCol.tolist() == [
        Timestamp("2010-10-10"),
        Timestamp("2010-12-12"),
    ]


@pytest.mark.parametrize("conn", all_connectable_types)
@pytest.mark.parametrize("error", ["raise", "coerce"])
@pytest.mark.parametrize(
    "read_sql, text, mode",
    [
        (sql.read_sql, "SELECT * FROM types", ("sqlalchemy", "fallback")),
        (sql.read_sql, "types", ("sqlalchemy")),
        (
            sql.read_sql_query,
            "SELECT * FROM types",
            ("sqlalchemy", "fallback"),
        ),
        (sql.read_sql_table, "types", ("sqlalchemy")),
    ],
)
def test_api_custom_dateparsing_error(
    conn, request, read_sql, text, mode, error, types_data_frame
):
    # [FUNCTION DOC]
    # Parametrized over all types connection kinds, error handling mode (raise or coerce), and four reader variants (general reader with a query or a table name, the query reader, and the table reader, with the allowed modes). Reads with date parsing given as a mapping with an errors option; compares the result with the expected types frame whose date column is microsecond-unit datetime (PostgreSQL boolean columns converted to integer and float; the PostgreSQL ADBC columns narrowed to 32-bit integers). The combination of a table name with the built-in sqlite3 connection is an expected failure.
    # KNOWN ISSUE: the mode parameter is parametrized but not used; also the PostgreSQL branch has a to-do about cleaning up the shared types frame fixture.
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if text == "types" and conn_name == "sqlite_buildin_types":
        request.applymarker(
            pytest.mark.xfail(reason="failing combination of arguments")
        )

    expected = types_data_frame.astype({"DateCol": "datetime64[us]"})

    result = read_sql(
        text,
        con=conn,
        parse_dates={
            "DateCol": {"errors": error},
        },
    )
    if "postgres" in conn_name:
        # TODO: clean up types_data_frame fixture
        result["BoolCol"] = result["BoolCol"].astype(int)
        result["BoolColWithNull"] = result["BoolColWithNull"].astype(float)

    if conn_name == "postgresql_adbc_types":
        expected = expected.astype(
            {
                "IntDateCol": "int32",
                "IntDateOnlyCol": "int32",
                "IntCol": "int32",
            }
        )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable_types)
def test_api_date_and_index(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all types connection kinds. When the same column is both parsed as a date and used as the index column, the index is datetime and the other parsed column is datetime.
    # Test case where same column appears in parse_date and index_col
    conn = request.getfixturevalue(conn)
    df = sql.read_sql_query(
        "SELECT * FROM types",
        conn,
        index_col="DateCol",
        parse_dates=["DateCol", "IntDateCol"],
    )

    assert issubclass(df.index.dtype.type, np.datetime64)
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_timedelta(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writing a timedelta column returns two rows written, with a user warning (a deprecation warning for ADBC with old pyarrow, none otherwise); the sqlite ADBC driver is an expected failure. Reading back gives 64-bit integers (nanoseconds) for most backends, or DateOffset objects for PostgreSQL ADBC where the interval type is read as month-day-nano.
    # see #6921
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_timedelta", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_timedelta")

    df = to_timedelta(Series(["00:00:01", "00:00:03"], name="foo")).to_frame()

    if conn_name == "sqlite_adbc_conn":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="sqlite ADBC driver doesn't implement timedelta",
            )
        )

    if "adbc" in conn_name:
        if pa_version_under14p1:
            exp_warning = DeprecationWarning
        else:
            exp_warning = None
    else:
        exp_warning = UserWarning

    with tm.assert_produces_warning(exp_warning, check_stacklevel=False):
        result_count = df.to_sql(name="test_timedelta", con=conn)
    assert result_count == 2
    result = sql.read_sql_query("SELECT * FROM test_timedelta", conn)

    if conn_name == "postgresql_adbc_conn":
        # TODO: Postgres stores an INTERVAL, which ADBC reads as a Month-Day-Nano
        # Interval; the default pandas type mapper maps this to a DateOffset
        # but maybe we should try and restore the timedelta here?
        expected = Series(
            [
                pd.DateOffset(months=0, days=0, microseconds=1000000, nanoseconds=0),
                pd.DateOffset(months=0, days=0, microseconds=3000000, nanoseconds=0),
            ],
            name="foo",
        )
    else:
        expected = df["foo"].astype("int64")
    tm.assert_series_equal(result["foo"], expected)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_complex_raises(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writing complex numbers raises a value error (message about unsupported datatypes, differing for ADBC).
    # KNOWN ISSUE: the assertion inside the raises block is never evaluated because the write raises first.
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame({"a": [1 + 1j, 2j]})

    if "adbc" in conn_name:
        msg = "datatypes not supported"
    else:
        msg = "Complex datatypes not supported"
    with pytest.raises(ValueError, match=msg):
        assert df.to_sql("test_complex", con=conn) is None


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize(
    "index_name,index_label,expected",
    [
        # no index name, defaults to 'index'
        (None, None, "index"),
        # specifying index_label
        (None, "other_label", "other_label"),
        # using the index name
        ("index_name", None, "index_name"),
        # has index name, but specifying index_label
        ("index_name", "other_label", "other_label"),
        # index name is integer
        (0, None, "0"),
        # index name is None but index label is integer
        (None, 0, "0"),
    ],
)
def test_api_to_sql_index_label(conn, request, index_name, index_label, expected):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds and six combinations of index name and index label (including integers). Writes a frame with the given index name and label and checks the first column name read back equals the expected: the label if given, else the index name, else the default index name; integers are converted to strings. ADBC is an expected failure (index label unsupported).
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC")
        )
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_index_label", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")

    temp_frame = DataFrame({"col1": range(4)})
    temp_frame.index.name = index_name
    query = "SELECT * FROM test_index_label"
    sql.to_sql(temp_frame, "test_index_label", conn, index_label=index_label)
    frame = sql.read_sql_query(query, conn)
    assert frame.columns[0] == expected


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_index_label_multiindex(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writes a frame with a two-level index: default names are level zero and level one; an index label list overrides; the index level names are used when set; a list label overrides the names; a string label for two levels raises a value error that the length must match the number of levels. Each write returns four rows. MySQL is a non-strict expected failure (text keys) and ADBC an expected failure.
    conn_name = conn
    if "mysql" in conn_name:
        request.applymarker(
            pytest.mark.xfail(
                reason="MySQL can fail using TEXT without length as key", strict=False
            )
        )
    elif "adbc" in conn_name:
        request.node.add_marker(
            pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC")
        )

    conn = request.getfixturevalue(conn)
    if sql.has_table("test_index_label", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")

    expected_row_count = 4
    temp_frame = DataFrame(
        {"col1": range(4)},
        index=MultiIndex.from_product([("A0", "A1"), ("B0", "B1")]),
    )

    # no index name, defaults to 'level_0' and 'level_1'
    result = sql.to_sql(temp_frame, "test_index_label", conn)
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[0] == "level_0"
    assert frame.columns[1] == "level_1"

    # specifying index_label
    result = sql.to_sql(
        temp_frame,
        "test_index_label",
        conn,
        if_exists="replace",
        index_label=["A", "B"],
    )
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["A", "B"]

    # using the index name
    temp_frame.index.names = ["A", "B"]
    result = sql.to_sql(temp_frame, "test_index_label", conn, if_exists="replace")
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["A", "B"]

    # has index name, but specifying index_label
    result = sql.to_sql(
        temp_frame,
        "test_index_label",
        conn,
        if_exists="replace",
        index_label=["C", "D"],
    )
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["C", "D"]

    msg = "Length of 'index_label' should match number of levels, which is 2"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(
            temp_frame,
            "test_index_label",
            conn,
            if_exists="replace",
            index_label="C",
        )


@pytest.mark.parametrize("conn", all_connectable)
def test_api_multiindex_roundtrip(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writes a frame indexed by two columns and reads it back using those columns as the index; frames are equal including the index type.
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_multiindex_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_multiindex_roundtrip")

    df = DataFrame.from_records(
        [(1, 2.1, "line1"), (2, 1.5, "line2")],
        columns=["A", "B", "C"],
        index=["A", "B"],
    )

    df.to_sql(name="test_multiindex_roundtrip", con=conn)
    result = sql.read_sql_query(
        "SELECT * FROM test_multiindex_roundtrip", conn, index_col=["A", "B"]
    )
    tm.assert_frame_equal(df, result, check_index_type=True)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize(
    "dtype",
    [
        None,
        int,
        float,
        {"A": int, "B": float},
    ],
)
def test_api_dtype_argument(conn, request, dtype):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds and dtype argument values (none, integer, float, per-column mapping). Regression: writes a two-column float frame (two rows written) and reads back with the dtype argument; the result equals the original cast to that dtype. Quotes column names for PostgreSQL.
    # GH10285 Add dtype argument to read_sql_query
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_dtype_argument", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_dtype_argument")

    df = DataFrame([[1.2, 3.4], [5.6, 7.8]], columns=["A", "B"])
    assert df.to_sql(name="test_dtype_argument", con=conn) == 2

    expected = df.astype(dtype)

    if "postgres" in conn_name:
        query = 'SELECT "A", "B" FROM test_dtype_argument'
    else:
        query = "SELECT A, B FROM test_dtype_argument"
    result = sql.read_sql_query(query, con=conn, dtype=dtype)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_integer_col_names(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Smoke test: writing a frame with integer column labels replaces the table without error.
    conn = request.getfixturevalue(conn)
    df = DataFrame([[1, 2], [3, 4]], columns=[0, 1])
    sql.to_sql(df, "test_frame_integer_col_names", conn, if_exists="replace")


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. The schema generator returns a create statement for the sample frame. ADBC is a strict expected failure.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'get_schema' not implemented for ADBC drivers",
                strict=True,
            )
        )
    conn = request.getfixturevalue(conn)
    create_sql = sql.get_schema(test_frame1, "test", con=conn)
    assert "CREATE" in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_with_schema(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Regression: the schema generator with a schema name produces a statement prefixed with the schema-qualified create-table phrase. ADBC is a strict expected failure.
    # GH28486
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'get_schema' not implemented for ADBC drivers",
                strict=True,
            )
        )
    conn = request.getfixturevalue(conn)
    create_sql = sql.get_schema(test_frame1, "test", con=conn, schema="pypi")
    assert "CREATE TABLE pypi." in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_dtypes(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. The schema generator accepts a per-column type override mapping (a SQLAlchemy integer type, or the string INTEGER for the built-in sqlite3 connection): the generated create statement contains both the create keyword and the integer type name. ADBC is a strict expected failure.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'get_schema' not implemented for ADBC drivers",
                strict=True,
            )
        )
    conn_name = conn
    conn = request.getfixturevalue(conn)
    float_frame = DataFrame({"a": [1.1, 1.2], "b": [2.1, 2.2]})

    if conn_name == "sqlite_buildin":
        dtype = "INTEGER"
    else:
        from sqlalchemy import Integer

        dtype = Integer
    create_sql = sql.get_schema(float_frame, "test", con=conn, dtype={"b": dtype})
    assert "CREATE" in create_sql
    assert "INTEGER" in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_keys(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. The schema generator with a key column (one name, or a list of names) adds a named primary key constraint with the key columns, quoted with backticks for MySQL and double quotes otherwise. ADBC is a strict expected failure.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'get_schema' not implemented for ADBC drivers",
                strict=True,
            )
        )
    conn_name = conn
    conn = request.getfixturevalue(conn)
    frame = DataFrame({"Col1": [1.1, 1.2], "Col2": [2.1, 2.2]})
    create_sql = sql.get_schema(frame, "test", con=conn, keys="Col1")

    if "mysql" in conn_name:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`Col1`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1")'
    assert constraint_sentence in create_sql

    # multiple columns as key (GH10385)
    create_sql = sql.get_schema(test_frame1, "test", con=conn, keys=["A", "B"])
    if "mysql" in conn_name:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`A`, `B`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("A", "B")'
    assert constraint_sentence in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_chunksize_read(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writes 22 random rows and reads them in chunks of five (sizes five, five, five, five, two) using the query reader, and concatenated equals the unchunked read; chunked reading with the table reader gives the same, except the built-in sqlite3 connection which raises a not-implemented error with an empty message. ADBC is an expected failure.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_chunksize", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_chunksize")

    df = DataFrame(
        np.random.default_rng(2).standard_normal((22, 5)), columns=list("abcde")
    )
    df.to_sql(name="test_chunksize", con=conn, index=False)

    # reading the query in one time
    res1 = sql.read_sql_query("select * from test_chunksize", conn)

    # reading the query in chunks with read_sql_query
    res2 = DataFrame()
    i = 0
    sizes = [5, 5, 5, 5, 2]

    for chunk in sql.read_sql_query("select * from test_chunksize", conn, chunksize=5):
        res2 = concat([res2, chunk], ignore_index=True)
        assert len(chunk) == sizes[i]
        i += 1

    tm.assert_frame_equal(res1, res2)

    # reading the query in chunks with read_sql_query
    if conn_name == "sqlite_buildin":
        with pytest.raises(NotImplementedError, match="^$"):
            sql.read_sql_table("test_chunksize", conn, chunksize=5)
    else:
        res3 = DataFrame()
        i = 0
        sizes = [5, 5, 5, 5, 2]

        for chunk in sql.read_sql_table("test_chunksize", conn, chunksize=5):
            res3 = concat([res3, chunk], ignore_index=True)
            assert len(chunk) == sizes[i]
            i += 1

        tm.assert_frame_equal(res1, res3)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_categorical(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Regression: a categorical column is written as a plain (dense) column and read back as strings equal to the original; the PostgreSQL ADBC driver below version 0.9 is a strict expected failure.
    if conn == "postgresql_adbc_conn":
        adbc = import_optional_dependency("adbc_driver_postgresql", errors="ignore")
        if adbc is not None and Version(adbc.__version__) < Version("0.9.0"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="categorical dtype not implemented for ADBC postgres driver",
                    strict=True,
                )
            )
    # GH8624
    # test that categorical gets written correctly as dense column
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_categorical", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_categorical")

    df = DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_name": ["John P. Doe", "Jane Dove", "John P. Doe"],
        }
    )
    df2 = df.copy()
    df2["person_name"] = df2["person_name"].astype("category")

    df2.to_sql(name="test_categorical", con=conn, index=False)
    res = sql.read_sql_query("SELECT * FROM test_categorical", conn)

    tm.assert_frame_equal(res, df)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_unicode_column_name(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Regression smoke test: a column name with a non-ascii character can be written.
    # GH 11431
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_unicode", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_unicode")

    df = DataFrame([[1, 2], [3, 4]], columns=["\xe9", "b"])
    df.to_sql(name="test_unicode", con=conn, index=False)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_escaped_table_name(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Regression: a table name containing hyphens is written and read back with a query that quotes the name (double quotes for PostgreSQL, backticks otherwise).
    # GH 13206
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table("d1187b08-4943-4c8d-a7f6", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("d1187b08-4943-4c8d-a7f6")

    df = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
    df.to_sql(name="d1187b08-4943-4c8d-a7f6", con=conn, index=False)

    if "postgres" in conn_name:
        query = 'SELECT * FROM "d1187b08-4943-4c8d-a7f6"'
    else:
        query = "SELECT * FROM `d1187b08-4943-4c8d-a7f6`"
    res = sql.read_sql_query(query, conn)

    tm.assert_frame_equal(res, df)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_table_name_quoted(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds (skipped for the built-in sqlite3). Regression: a table name containing a space is written, read by the table reader, and rewritten with delete-rows mode, exercising identifier quoting in the write, read and delete paths of the ADBC database class.
    # GH#65065 ADBCDatabase did not escape SQL identifiers in
    # delete_rows, read_table, and to_sql (DROP TABLE path)
    conn_name = conn
    if conn_name == "sqlite_buildin":
        pytest.skip("sqlite_buildin connection does not implement read_sql_table")

    conn = request.getfixturevalue(conn)

    tricky_name = "my table"
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # to_sql: adbc_ingest handles quoting internally — should pass
    df.to_sql(tricky_name, conn, index=False, if_exists="replace")
    # read_table: was interpolating name raw into f-string
    result = read_sql_table(tricky_name, conn)
    tm.assert_frame_equal(result, df)
    # delete_rows: was interpolating name raw into DELETE FROM f-string
    df.to_sql(tricky_name, conn, index=False, if_exists="delete_rows")
    result = read_sql_table(tricky_name, conn)
    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_read_sql_duplicate_columns(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Regression: reading a query with duplicate output column names preserves both columns in the frame; for ADBC this works only with pyarrow 16 or newer on sqlite or PostgreSQL, else it is a strict expected failure.
    # GH#53117
    if "adbc" in conn:
        pa = pytest.importorskip("pyarrow")
        if not (
            Version(pa.__version__) >= Version("16.0")
            and conn in ["sqlite_adbc_conn", "postgresql_adbc_conn"]
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="pyarrow->pandas throws ValueError", strict=True
                )
            )
    conn = request.getfixturevalue(conn)
    if sql.has_table("test_table", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_table")

    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": 1})
    df.to_sql(name="test_table", con=conn, index=False)

    result = pd.read_sql("SELECT a, b, a +1 as a, c FROM test_table", conn)
    expected = DataFrame(
        [[1, 0.1, 2, 1], [2, 0.2, 3, 1], [3, 0.3, 4, 1]],
        columns=["a", "b", "a", "c"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
def test_read_table_columns(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. The table reader with a column list returns only those columns in that order; the built-in sqlite3 connection is an expected failure.
    # test columns argument in read_table
    conn_name = conn
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))

    conn = request.getfixturevalue(conn)
    sql.to_sql(test_frame1, "test_frame", conn)

    cols = ["A", "B"]

    result = sql.read_sql_table("test_frame", conn, columns=cols)
    assert result.columns.tolist() == cols


@pytest.mark.parametrize("conn", all_connectable)
def test_read_table_index_col(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. The table reader's index column argument: a single name or a list sets the index names, and combining it with a column list returns only those columns; the built-in sqlite3 connection is an expected failure.
    # test columns argument in read_table
    conn_name = conn
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))

    conn = request.getfixturevalue(conn)
    sql.to_sql(test_frame1, "test_frame", conn)

    result = sql.read_sql_table("test_frame", conn, index_col="index")
    assert result.index.names == ["index"]

    result = sql.read_sql_table("test_frame", conn, index_col=["A", "B"])
    assert result.index.names == ["A", "B"]

    result = sql.read_sql_table(
        "test_frame", conn, index_col=["A", "B"], columns=["C", "D"]
    )
    assert result.index.names == ["A", "B"]
    assert result.columns.tolist() == ["C", "D"]


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_delegate(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. The general reader delegates to the query reader for query text and to the table reader for a bare table name; results equal. The built-in sqlite3 connection is an expected failure.
    if conn == "sqlite_buildin_iris":
        request.applymarker(
            pytest.mark.xfail(
                reason="sqlite_buildin connection does not implement read_sql_table"
            )
        )

    conn = request.getfixturevalue(conn)
    iris_frame1 = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2 = sql.read_sql("SELECT * FROM iris", conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)

    iris_frame1 = sql.read_sql_table("iris", conn)
    iris_frame2 = sql.read_sql("iris", conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)


def test_not_reflect_all_tables(sqlite_conn):
    # [FUNCTION DOC]
    # Creates a table with an unknown column type and another valid table on the sqlite connection, then reads the valid table with both readers and expects no warning (an unreflectable table does not break reflection).
    # KNOWN ISSUE: the engine-handling branch rebinds the connection variable but cannot be reached because the fixture provides a connection, not an engine.
    conn = sqlite_conn
    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    # create invalid table
    query_list = [
        text("CREATE TABLE invalid (x INTEGER, y UNKNOWN);"),
        text("CREATE TABLE other_table (x INTEGER, y INTEGER);"),
    ]

    for query in query_list:
        if isinstance(conn, Engine):
            with conn.connect() as conn:
                with conn.begin():
                    conn.execute(query)
        else:
            with conn.begin():
                conn.execute(query)

    with tm.assert_produces_warning(None):
        sql.read_sql_table("other_table", conn)
        sql.read_sql_query("SELECT * FROM other_table", conn)


@pytest.mark.parametrize("conn", all_connectable)
def test_warning_case_insensitive_table_name(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Calling the case-sensitivity check on a table name that does not exist as written emits a user warning about case sensitivity; a normal write of a mixed-case name emits no warning. Built-in sqlite3 and ADBC are expected failures (no warning raised).
    conn_name = conn
    if conn_name == "sqlite_buildin" or "adbc" in conn_name:
        request.applymarker(pytest.mark.xfail(reason="Does not raise warning"))

    conn = request.getfixturevalue(conn)
    # see gh-7815
    with tm.assert_produces_warning(
        UserWarning,
        match=(
            r"The provided table name 'TABLE1' is not found exactly as such in "
            r"the database after writing the table, possibly due to case "
            r"sensitivity issues. Consider using lower case table names."
        ),
    ):
        with sql.SQLDatabase(conn) as db:
            db.check_case_sensitive("TABLE1", "")

    # Test that the warning is certainly NOT triggered in a normal case.
    with tm.assert_produces_warning(None):
        test_frame1.to_sql(name="CaseSensitive", con=conn)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_sqlalchemy_type_mapping(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. A timezone-aware datetime column maps to the SQLAlchemy TIMESTAMP type in the table definition.
    conn = request.getfixturevalue(conn)
    from sqlalchemy import TIMESTAMP

    # Test Timestamp objects (no datetime64 because of timezone) (GH9085)
    df = DataFrame(
        {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
    )
    with sql.SQLDatabase(conn) as db:
        table = sql.SQLTable("test_type", db, frame=df)
        # GH 9086: TIMESTAMP is the suggested type for datetimes with timezones
        assert isinstance(table.table.c["time"].type, TIMESTAMP)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
@pytest.mark.parametrize(
    "integer, expected",
    [
        ("int8", "SMALLINT"),
        ("Int8", "SMALLINT"),
        ("uint8", "SMALLINT"),
        ("UInt8", "SMALLINT"),
        ("int16", "SMALLINT"),
        ("Int16", "SMALLINT"),
        ("uint16", "INTEGER"),
        ("UInt16", "INTEGER"),
        ("int32", "INTEGER"),
        ("Int32", "INTEGER"),
        ("uint32", "BIGINT"),
        ("UInt32", "BIGINT"),
        ("int64", "BIGINT"),
        ("Int64", "BIGINT"),
        (int, "BIGINT" if np.dtype(int).name == "int64" else "INTEGER"),
    ],
)
def test_sqlalchemy_integer_mapping(conn, request, integer, expected):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds and integer dtypes (signed, unsigned, nullable, built-in integer). Regression: each integer dtype maps to the smallest SQL integer type that holds it (small, regular or big integer); the built-in integer maps by platform width.
    # GH35076 Map pandas integer to optimal SQLAlchemy integer type
    conn = request.getfixturevalue(conn)
    df = DataFrame([0, 1], columns=["a"], dtype=integer)
    with sql.SQLDatabase(conn) as db:
        table = sql.SQLTable("test_type", db, frame=df)

        result = str(table.table.c.a.type)
    assert result == expected


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
@pytest.mark.parametrize("integer", ["uint64", "UInt64"])
def test_sqlalchemy_integer_overload_mapping(conn, request, integer):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds and the unsigned 64-bit dtype names. Creating a table definition from such a column raises a value error that unsigned 64 bit integers are not supported.
    conn = request.getfixturevalue(conn)
    # GH35076 Map pandas integer to optimal SQLAlchemy integer type
    df = DataFrame([0, 1], columns=["a"], dtype=integer)
    with sql.SQLDatabase(conn) as db:
        with pytest.raises(
            ValueError, match="Unsigned 64 bit integer datatype is not supported"
        ):
            sql.SQLTable("test_type", db, frame=df)


def test_database_uri_string(temp_file, request, test_frame1):
    # [FUNCTION DOC]
    # Requires SQLAlchemy. Writes and reads through the public functions using a database URI string for a temporary sqlite file (general, table and query readers); every result equals the original.
    pytest.importorskip("sqlalchemy")
    # Test read_sql and .to_sql method with a database URI (GH10654)
    # db_uri = 'sqlite:///:memory:' # raises
    # sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) near
    # "iris": syntax error [SQL: 'iris']
    name = str(temp_file)
    db_uri = "sqlite:///" + name
    table = "iris"
    test_frame1.to_sql(name=table, con=db_uri, if_exists="replace", index=False)
    test_frame2 = sql.read_sql(table, db_uri)
    test_frame3 = sql.read_sql_table(table, db_uri)
    query = "SELECT * FROM iris"
    test_frame4 = sql.read_sql_query(query, db_uri)
    tm.assert_frame_equal(test_frame1, test_frame2)
    tm.assert_frame_equal(test_frame1, test_frame3)
    tm.assert_frame_equal(test_frame1, test_frame4)


@td.skip_if_installed("pg8000")
def test_pg8000_sqlalchemy_passthrough_error(request):
    # [FUNCTION DOC]
    # Runs only when the pg8000 driver is not installed. Requires SQLAlchemy. Reading with a connection string for that driver propagates the engine-creation import error naming the driver.
    pytest.importorskip("sqlalchemy")
    # using driver that will not be installed on CI to trigger error
    # in sqlalchemy.create_engine -> test passing of this error to user
    db_uri = "postgresql+pg8000://user:pass@host/dbname"
    with pytest.raises(ImportError, match="pg8000"):
        sql.read_sql("select * from table", db_uri)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_query_by_text_obj(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy iris connection kinds. Reads with a text clause object using a named bind parameter (quoted column for PostgreSQL); only rows of the requested species are returned.
    # WIP : GH10846
    conn_name = conn
    conn = request.getfixturevalue(conn)
    from sqlalchemy import text

    if "postgres" in conn_name:
        name_text = text('select * from iris where "Name"=:name')
    else:
        name_text = text("select * from iris where name=:name")
    iris_df = sql.read_sql(name_text, conn, params={"name": "Iris-versicolor"})
    all_names = set(iris_df["Name"])
    assert all_names == {"Iris-versicolor"}


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_query_by_select_obj(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy iris connection kinds. Reads with a select statement filtered by a bound parameter; only rows of the requested species are returned.
    conn = request.getfixturevalue(conn)
    # WIP : GH10846
    from sqlalchemy import (
        bindparam,
        select,
    )

    iris = iris_table_metadata()
    name_select = select(iris).where(iris.c.Name == bindparam("name"))
    iris_df = sql.read_sql(name_select, conn, params={"name": "Iris-setosa"})
    all_names = set(iris_df["Name"])
    assert all_names == {"Iris-setosa"}


@pytest.mark.parametrize("conn", all_connectable)
def test_column_with_percentage(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Regression: a column name containing a percent sign is written and read with the table reader. The built-in sqlite3 connection is an expected failure.
    # GH 37157
    conn_name = conn
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))

    conn = request.getfixturevalue(conn)
    df = DataFrame({"A": [0, 1, 2], "%_variation": [3, 4, 5]})
    df.to_sql(name="test_column_percentage", con=conn, index=False)

    res = sql.read_sql_table("test_column_percentage", conn)

    tm.assert_frame_equal(res, df)


def test_sql_open_close(temp_file, test_frame3):
    # [FUNCTION DOC]
    # Writes via a built-in sqlite3 connection that is then closed (the write returns four rows), then reads via a new connection to the same file; the result equals the original.
    # Test if the IO in the database still work if the connection closed
    # between the writing and reading (as in many real situations).

    with contextlib.closing(sqlite3.connect(temp_file)) as conn:
        assert sql.to_sql(test_frame3, "test_frame3_legacy", conn, index=False) == 4

    with contextlib.closing(sqlite3.connect(temp_file)) as conn:
        result = sql.read_sql_query("SELECT * FROM test_frame3_legacy;", conn)

    tm.assert_frame_equal(test_frame3, result)


@td.skip_if_installed("sqlalchemy")
def test_con_string_import_error():
    # [FUNCTION DOC]
    # Runs only when SQLAlchemy is not installed. Reading with a connection URI string raises an import error saying a URI string was used without SQLAlchemy.
    conn = "mysql://root@localhost/pandas"
    msg = "Using URI string without sqlalchemy installed"
    with pytest.raises(ImportError, match=msg):
        sql.read_sql("SELECT * FROM iris", conn)


@td.skip_if_installed("sqlalchemy")
def test_con_unknown_dbapi2_class_does_not_error_without_sql_alchemy_installed():
    # [FUNCTION DOC]
    # Runs only when SQLAlchemy is not installed. With a minimal wrapper object around a built-in sqlite3 connection (not recognised as a sqlite3 connection class), reading a trivial query emits a user warning that only SQLAlchemy is supported, but does not raise.
    class MockSqliteConnection:
        # [CLASS DOC]
        # HELPER: minimal stand-in for a sqlite connection used only by the test above. Wraps a real built-in sqlite3 connection created from the constructor arguments, forwards unknown attribute lookups to it and provides a close method that closes it; it is deliberately not a subclass of the sqlite connection class.
        def __init__(self, *args, **kwargs) -> None:
            # [FUNCTION DOC]
            # HELPER: constructor. Creates the wrapped sqlite3 connection from all positional and keyword arguments and stores it as the connection attribute.
            self.conn = sqlite3.Connection(*args, **kwargs)

        def __getattr__(self, name):
            # [FUNCTION DOC]
            # HELPER: attribute fallback; returns the attribute of the wrapped connection with the given name (any missing attribute is delegated).
            return getattr(self.conn, name)

        def close(self):
            # [FUNCTION DOC]
            # HELPER: closes the wrapped connection.
            self.conn.close()

    with contextlib.closing(MockSqliteConnection(":memory:")) as conn:
        with tm.assert_produces_warning(UserWarning, match="only supports SQLAlchemy"):
            sql.read_sql("SELECT 1", conn)


def test_sqlite_read_sql_delegate(sqlite_buildin_iris):
    # [FUNCTION DOC]
    # With a built-in sqlite3 iris connection, the general reader delegates a query to the query reader (equal results); passing a bare table name raises a database error whose message reports an execution failure with a syntax error near the table name.
    conn = sqlite_buildin_iris
    iris_frame1 = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2 = sql.read_sql("SELECT * FROM iris", conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)

    msg = "Execution failed on sql 'iris': near \"iris\": syntax error"
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql("iris", conn)


def test_get_schema2(test_frame1):
    # [FUNCTION DOC]
    # The schema generator works without a connection argument (kept for backward compatibility) and returns a create statement.
    # without providing a connection object (available for backwards comp)
    create_sql = sql.get_schema(test_frame1, "test")
    assert "CREATE" in create_sql


def test_sqlite_type_mapping(sqlite_buildin):
    # [FUNCTION DOC]
    # Creates a sqlite table definition (not through SQLAlchemy) for a timezone-aware datetime column and verifies in the generated schema text that the column type is timestamp.
    # Test Timestamp objects (no datetime64 because of timezone) (GH9085)
    conn = sqlite_buildin
    df = DataFrame(
        {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
    )
    db = sql.SQLiteDatabase(conn)
    table = sql.SQLiteTable("test_type", db, frame=df)
    schema = table.sql_schema()
    for col in schema.split("\n"):
        if col.split()[0].strip('"') == "time":
            assert col.split()[1] == "TIMESTAMP"


# -----------------------------------------------------------------------------
# -- Database flavor specific tests


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_create_table(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds (skipped for the connection string, which cannot be inspected). Writes a frame (four rows returned) through a database object in a transaction, verifies with the inspector that the table exists, and then drops it.
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    conn = request.getfixturevalue(conn)

    from sqlalchemy import inspect

    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4

    insp = inspect(conn)
    assert insp.has_table("temp_frame")

    # Cleanup
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("temp_frame")


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_drop_table(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds (skipped for the connection string). Creates a table and checks it exists via the inspector, drops it in another transaction, clears the inspector cache when supported, and verifies the table no longer exists.
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    conn = request.getfixturevalue(conn)

    from sqlalchemy import inspect

    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4

        insp = inspect(conn)
        assert insp.has_table("temp_frame")

        with pandasSQL.run_transaction():
            pandasSQL.drop_table("temp_frame")
        try:
            insp.clear_cache()  # needed with SQLAlchemy 2.0, unavailable prior
        except AttributeError:
            pass
        assert not insp.has_table("temp_frame")


@pytest.mark.parametrize("conn_name", all_connectable)
def test_delete_rows_success(conn_name, test_frame1, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. After writing a table, deleting all rows returns nothing; the row count is zero and the table still exists.
    table_name = "temp_delete_rows_frame"
    conn = request.getfixturevalue(conn_name)

    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(test_frame1, table_name) == test_frame1.shape[0]

        with pandasSQL.run_transaction():
            assert pandasSQL.delete_rows(table_name) is None

        assert count_rows(conn, table_name) == 0
        assert pandasSQL.has_table(table_name)


@pytest.mark.parametrize("conn_name", all_connectable)
def test_delete_rows_is_atomic(conn_name, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Requires SQLAlchemy. Creates a table with a uniqueness constraint and original rows (32-bit integers), then writes replacement rows with duplicate values in delete-rows mode: a database error is raised and the delete is rolled back, so the original data remains.
    sqlalchemy = pytest.importorskip("sqlalchemy")

    table_name = "temp_delete_rows_atomic_frame"
    table_stmt = f"CREATE TABLE {table_name} (a INTEGER, b INTEGER UNIQUE NOT NULL)"

    if conn_name != "sqlite_buildin" and "adbc" not in conn_name:
        table_stmt = sqlalchemy.text(table_stmt)

    # setting dtype is mandatory for adbc related tests
    original_df = DataFrame({"a": [1, 2], "b": [3, 4]}, dtype="int32")
    replacing_df = DataFrame({"a": [5, 6, 7], "b": [8, 8, 8]}, dtype="int32")

    conn = request.getfixturevalue(conn_name)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as cur:
            cur.execute(table_stmt)

        with pandasSQL.run_transaction():
            pandasSQL.to_sql(original_df, table_name, if_exists="append", index=False)

        # inserting duplicated values in a UNIQUE constraint column
        with pytest.raises(pd.errors.DatabaseError):
            with pandasSQL.run_transaction():
                pandasSQL.to_sql(
                    replacing_df, table_name, if_exists="delete_rows", index=False
                )

        # failed "delete_rows" is rolled back preserving original data
        with pandasSQL.run_transaction():
            result_df = pandasSQL.read_query(
                f"SELECT * FROM {table_name}", dtype="int32"
            )
            tm.assert_frame_equal(result_df, original_df)


@pytest.mark.parametrize("conn", all_connectable)
def test_roundtrip(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds (skipped for the connection string). Writes the sample frame (four rows) and reads it with a query through a database object in a transaction; after renaming the generated index column for ADBC and setting it as the index (name cleared), the frame equals the original.
    # KNOWN ISSUE: a commented-out line about converting the index to integers remains in the body.
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    conn_name = conn
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(test_frame1, "test_frame_roundtrip") == 4
            result = pandasSQL.read_query("SELECT * FROM test_frame_roundtrip")

    if "adbc" in conn_name:
        result = result.rename(columns={"__index_level_0__": "level_0"})
    result.set_index("level_0", inplace=True)
    # result.index.astype(int)

    result.index.name = None

    tm.assert_frame_equal(result, test_frame1)


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_execute_sql(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all iris connection kinds. Through a database object inside a transaction, executes a select on iris, fetches the first row and closes the result; the row equals the known first iris record.
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_results = pandasSQL.execute("SELECT * FROM iris")
            row = iris_results.fetchone()
            iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_sqlalchemy_read_table(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy iris connection kinds. The table reader returns the full iris frame.
    conn = request.getfixturevalue(conn)
    iris_frame = sql.read_sql_table("iris", con=conn)
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_sqlalchemy_read_table_columns(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy iris connection kinds. Selecting the same column name twice yields column labels where the duplicate gets a double-underscore numeric suffix.
    conn = request.getfixturevalue(conn)
    iris_frame = sql.read_sql_table(
        "iris", con=conn, columns=["SepalLength", "SepalLength"]
    )
    tm.assert_index_equal(iris_frame.columns, Index(["SepalLength", "SepalLength__1"]))


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_table_absent_raises(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy iris connection kinds. Reading a table that does not exist raises a value error that the table was not found.
    conn = request.getfixturevalue(conn)
    msg = "Table this_doesnt_exist not found"
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("this_doesnt_exist", con=conn)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_types)
def test_sqlalchemy_default_type_conversion(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy types connection kinds (skipped for the connection string). Reads the types table with the table reader and checks default dtypes: float, integer, boolean (true boolean), integer-with-missing as float and boolean-with-missing as object. Expected failure for MySQL and sqlite as booleans are not inferred.
    conn_name = conn
    if conn_name == "sqlite_str":
        pytest.skip("types tables not created in sqlite_str fixture")
    elif "mysql" in conn_name or "sqlite" in conn_name:
        request.applymarker(
            pytest.mark.xfail(reason="boolean dtype not inferred properly")
        )

    conn = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn)

    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    assert issubclass(df.BoolCol.dtype.type, np.bool_)

    # Int column with NA values stays as float
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)
    # Bool column with NA values becomes object
    assert issubclass(df.BoolColWithNull.dtype.type, object)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_bigint(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Regression: a 64-bit integer near 2 to the 62nd power is written as a big integer column (one row returned) and read back equal.
    # int64 should be converted to BigInteger, GH7433
    conn = request.getfixturevalue(conn)
    df = DataFrame(data={"i64": [2**62]})
    assert df.to_sql(name="test_bigint", con=conn, index=False) == 1
    result = sql.read_sql_table("test_bigint", conn)

    tm.assert_frame_equal(df, result)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_types)
def test_default_date_load(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy types connection kinds. Reading the types table gives a datetime dtype for the date column by default (skipped for the connection string; expected failure for sqlite).
    conn_name = conn
    if conn_name == "sqlite_str":
        pytest.skip("types tables not created in sqlite_str fixture")
    elif "sqlite" in conn_name:
        request.applymarker(
            pytest.mark.xfail(reason="sqlite does not read date properly")
        )

    conn = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn)

    assert issubclass(df.DateCol.dtype.type, np.datetime64)


@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("parse_dates", [None, ["DateColWithTz"]])
def test_datetime_with_timezone_query(conn, request, parse_dates):
    # [FUNCTION DOC]
    # Parametrized over PostgreSQL connections and date parsing (none or the column). Edge case: a timestamp-with-time-zone column read by a query is converted to UTC datetimes (microsecond unit) and equals the expected series from the helper.
    # edge case that converts postgresql datetime with time zone types
    # to datetime64[ns,psycopg2.tz.FixedOffsetTimezone..], which is ok
    # but should be more natural, so coerce to datetime64[ns] for now
    conn = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn)

    # GH11216
    df = read_sql_query("select * from datetz", conn, parse_dates=parse_dates)
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)


@pytest.mark.parametrize("conn", postgresql_connectable)
def test_datetime_with_timezone_query_chunksize(conn, request):
    # [FUNCTION DOC]
    # Parametrized over PostgreSQL connections. Same as the previous test but reading in chunks of one row, concatenating them.
    conn = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn)

    df = concat(
        list(read_sql_query("select * from datetz", conn, chunksize=1)),
        ignore_index=True,
    )
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)


@pytest.mark.parametrize("conn", postgresql_connectable)
def test_datetime_with_timezone_table(conn, request):
    # [FUNCTION DOC]
    # Parametrized over PostgreSQL connections. Reading the time zone table with the table reader gives the expected one-column frame in UTC.
    conn = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn)
    result = sql.read_sql_table("datetz", conn)

    exp_frame = expected.to_frame()
    tm.assert_frame_equal(result, exp_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime_with_timezone_roundtrip(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Writes a timezone-aware column (three values returned) and reads it back with the table reader and with a query. PostgreSQL returns UTC-aware values; other databases return naive values (converted from the original local time). For sqlite the query reader returns strings that are parsed.
    conn_name = conn
    conn = request.getfixturevalue(conn)
    # GH 9086
    # Write datetimetz data to a db and read it back
    # For dbs that support timestamps with timezones, should get back UTC
    # otherwise naive data should be returned
    expected = DataFrame(
        {"A": date_range("2013-01-01 09:00:00", periods=3, tz="US/Pacific", unit="us")}
    )
    assert expected.to_sql(name="test_datetime_tz", con=conn, index=False) == 3

    if "postgresql" in conn_name:
        # SQLAlchemy "timezones" (i.e. offsets) are coerced to UTC
        expected["A"] = expected["A"].dt.tz_convert("UTC")
    else:
        # Otherwise, timestamps are returned as local, naive
        expected["A"] = expected["A"].dt.tz_localize(None)

    result = sql.read_sql_table("test_datetime_tz", conn)
    tm.assert_frame_equal(result, expected)

    result = sql.read_sql_query("SELECT * FROM test_datetime_tz", conn)
    if "sqlite" in conn_name:
        # read_sql_query does not return datetime type like read_sql_table
        assert isinstance(result.loc[0, "A"], str)
        result["A"] = to_datetime(result["A"]).dt.as_unit("us")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_out_of_bounds_datetime(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Regression: a date far in the future (year 9999) is written and read back as a microsecond-unit datetime.
    # GH 26761
    conn = request.getfixturevalue(conn)
    data = DataFrame({"date": datetime(9999, 1, 1)}, index=[0])
    assert data.to_sql(name="test_datetime_obb", con=conn, index=False) == 1
    result = sql.read_sql_table("test_datetime_obb", conn)
    expected = DataFrame(
        np.array([datetime(9999, 1, 1)], dtype="M8[us]"), columns=["date"]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_naive_datetimeindex_roundtrip(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Regression: a naive datetime index with an index label is written (five rows) and read back with the label as index; it is not converted to UTC and the frames are equal ignoring names.
    # GH 23510
    # Ensure that a naive DatetimeIndex isn't converted to UTC
    conn = request.getfixturevalue(conn)
    dates = date_range("2018-01-01", periods=5, freq="6h", unit="us")._with_freq(None)
    expected = DataFrame({"nums": range(5)}, index=dates)
    assert expected.to_sql(name="foo_table", con=conn, index_label="info_date") == 5
    result = sql.read_sql_table("foo_table", conn, index_col="info_date")
    # result index with gain a name from a set_index operation; expected
    tm.assert_frame_equal(result, expected, check_names=False)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_types)
def test_date_parsing(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy types connection kinds. Table reader date parsing: by default the date column is object for sqlite and datetime otherwise; listing the column, giving a format mapping, a format option dict, and for the epoch-seconds integer column the column name, a unit string and a unit option dict all produce datetime dtypes.
    # No Parsing
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn)
    expected_type = object if "sqlite" in conn_name else np.datetime64
    assert issubclass(df.DateCol.dtype.type, expected_type)

    df = sql.read_sql_table("types", conn, parse_dates=["DateCol"])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)

    df = sql.read_sql_table("types", conn, parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)

    df = sql.read_sql_table(
        "types",
        conn,
        parse_dates={"DateCol": {"format": "%Y-%m-%d %H:%M:%S"}},
    )
    assert issubclass(df.DateCol.dtype.type, np.datetime64)

    df = sql.read_sql_table("types", conn, parse_dates=["IntDateCol"])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

    df = sql.read_sql_table("types", conn, parse_dates={"IntDateCol": "s"})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

    df = sql.read_sql_table("types", conn, parse_dates={"IntDateCol": {"unit": "s"}})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Writes a frame with a datetime column and a float column with the index (three rows); the table reader returns the datetime in microsecond unit; the query reader returns strings for sqlite which are parsed, else datetimes.
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame(
        {"A": date_range("2013-01-01 09:00:00", periods=3), "B": np.arange(3.0)}
    )
    assert df.to_sql(name="test_datetime", con=conn) == 3

    # with read_table -> type information from schema used
    result = sql.read_sql_table("test_datetime", conn)
    result = result.drop("index", axis=1)

    expected = df[:]
    expected["A"] = expected["A"].astype("M8[us]")
    tm.assert_frame_equal(result, expected)

    # with read_sql -> no type information -> sqlite has no native
    result = sql.read_sql_query("SELECT * FROM test_datetime", conn)
    result = result.drop("index", axis=1)
    if "sqlite" in conn_name:
        assert isinstance(result.loc[0, "A"], str)
        result["A"] = to_datetime(result["A"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime_NaT(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Same as the previous test but with a missing datetime and without the index; reading back preserves the missing value as NaT, with sqlite strings parsed with errors coerced.
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame(
        {"A": date_range("2013-01-01 09:00:00", periods=3), "B": np.arange(3.0)}
    )
    df.loc[1, "A"] = np.nan
    assert df.to_sql(name="test_datetime", con=conn, index=False) == 3

    # with read_table -> type information from schema used
    result = sql.read_sql_table("test_datetime", conn)
    expected = df[:]
    expected["A"] = expected["A"].astype("M8[us]")
    tm.assert_frame_equal(result, expected)

    # with read_sql -> no type information -> sqlite has no native
    result = sql.read_sql_query("SELECT * FROM test_datetime", conn)
    if "sqlite" in conn_name:
        assert isinstance(result.loc[0, "A"], str)
        result["A"] = to_datetime(result["A"], errors="coerce")

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime_date(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Writes date objects (two rows) and reads back datetimes equal to the converted series.
    # test support for datetime.date
    conn = request.getfixturevalue(conn)
    df = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    res = read_sql_table("test_date", conn)
    result = res["a"]
    expected = to_datetime(df["a"])
    # comes back as datetime64
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime_time(conn, request, sqlite_buildin):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds, also using the built-in sqlite3 connection fixture. Regression test: time objects round trip with the table reader; the fallback sqlite path stores them as formatted strings and a registered adapter does not affect the SQLAlchemy path (which returns times, or strings on query for sqlite).
    # test support for datetime.time
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame([time(9, 0, 0), time(9, 1, 30)], columns=["a"])
    assert df.to_sql(name="test_time", con=conn, index=False) == 2
    res = read_sql_table("test_time", conn)
    tm.assert_frame_equal(res, df)

    # GH8341
    # first, use the fallback to have the sqlite adapter put in place
    sqlite_conn = sqlite_buildin
    assert sql.to_sql(df, "test_time2", sqlite_conn, index=False) == 2
    res = sql.read_sql_query("SELECT * FROM test_time2", sqlite_conn)
    ref = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
    tm.assert_frame_equal(ref, res)  # check if adapter is in place
    # then test if sqlalchemy is unaffected by the sqlite adapter
    assert sql.to_sql(df, "test_time3", conn, index=False) == 2
    if "sqlite" in conn_name:
        res = sql.read_sql_query("SELECT * FROM test_time3", conn)
        ref = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
        tm.assert_frame_equal(ref, res)
    res = sql.read_sql_table("test_time3", conn)
    tm.assert_frame_equal(df, res)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_mixed_dtype_insert(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Regression: a frame with a 32-bit integer and a 32-bit float column is written (one row) and read back exactly equal (dtype not checked).
    # see GH6509
    conn = request.getfixturevalue(conn)
    s1 = Series(2**25 + 1, dtype=np.int32)
    s2 = Series(0.0, dtype=np.float32)
    df = DataFrame({"s1": s1, "s2": s2})

    # write and read again
    assert df.to_sql(name="test_read_write", con=conn, index=False) == 1
    df2 = sql.read_sql_table("test_read_write", conn)

    tm.assert_frame_equal(df, df2, check_dtype=False, check_exact=True)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_nan_numeric(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. A float column with a missing value round-trips with both the table reader and the query reader.
    # NaNs in numeric float column
    conn = request.getfixturevalue(conn)
    df = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3

    # with read_table
    result = sql.read_sql_table("test_nan", conn)
    tm.assert_frame_equal(result, df)

    # with read_sql
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)
    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_nan_fullcolumn(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. A float column that is entirely missing round-trips with the table reader; with the query reader the column comes back as an object column of none.
    # full NaN column (numeric float column)
    conn = request.getfixturevalue(conn)
    df = DataFrame({"A": [0, 1, 2], "B": [np.nan, np.nan, np.nan]})
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3

    # with read_table
    result = sql.read_sql_table("test_nan", conn)
    tm.assert_frame_equal(result, df)

    # with read_sql -> not type info from table -> stays None
    df["B"] = df["B"].astype("object")
    df["B"] = None
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)
    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_nan_string(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. A string column with a missing value round-trips with both readers, the missing value coming back as none.
    # NaNs in string column
    conn = request.getfixturevalue(conn)
    df = DataFrame({"A": [0, 1, 2], "B": ["a", "b", np.nan]})
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3

    # NaNs are coming back as None
    df.loc[2, "B"] = None

    # with read_table
    result = sql.read_sql_table("test_nan", conn)
    tm.assert_frame_equal(result, df)

    # with read_sql
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)
    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_save_index(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writing a frame whose index is a column named A (two rows) creates exactly one index in the database on that column; checked via the sqlite master table and index info for the sqlite connections or the inspector otherwise. ADBC is a strict expected failure since no index is created.
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="ADBC implementation does not create index", strict=True
            )
        )
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame.from_records(
        [(1, 2.1, "line1"), (2, 1.5, "line2")], columns=["A", "B", "C"], index=["A"]
    )

    tbl_name = "test_to_sql_saves_index"
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(df, tbl_name) == 2

    if conn_name in {"sqlite_buildin", "sqlite_str"}:
        ixs = sql.read_sql_query(
            "SELECT * FROM sqlite_master WHERE type = 'index' "
            f"AND tbl_name = '{tbl_name}'",
            conn,
        )
        ix_cols = []
        for ix_name in ixs.name:
            ix_info = sql.read_sql_query(f"PRAGMA index_info({ix_name})", conn)
            ix_cols.append(ix_info.name.tolist())
    else:
        from sqlalchemy import inspect

        insp = inspect(conn)

        ixs = insp.get_indexes(tbl_name)
        ix_cols = [i["column_names"] for i in ixs]

    assert ix_cols == [["A"]]


@pytest.mark.parametrize("conn", all_connectable)
def test_transactions(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Smoke test: executing a create-table statement inside a database transaction block (statement wrapped as a text clause for SQLAlchemy kinds) does not raise.
    conn_name = conn
    conn = request.getfixturevalue(conn)

    stmt = "CREATE TABLE test_trans (A INT, B TEXT)"
    if conn_name != "sqlite_buildin" and "adbc" not in conn_name:
        from sqlalchemy import text

        stmt = text(stmt)

    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            trans.execute(stmt)


@pytest.mark.parametrize("conn", all_connectable)
def test_transaction_rollback(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Creates a table in a transaction, then raises a local exception inside a second transaction after inserting a row; reading afterwards shows no rows (rolled back). A committed insert in a third transaction is visible (one row).
    conn_name = conn
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            stmt = "CREATE TABLE test_trans (A INT, B TEXT)"
            if "adbc" in conn_name or isinstance(pandasSQL, SQLiteDatabase):
                trans.execute(stmt)
            else:
                from sqlalchemy import text

                stmt = text(stmt)
                trans.execute(stmt)

        class DummyException(Exception):
            # [CLASS DOC]
            # HELPER: exception type defined only to be raised inside a transaction in the rollback test, so it can be distinguished and swallowed. No state or behavior.
            pass

        # Make sure when transaction is rolled back, no rows get inserted
        ins_sql = "INSERT INTO test_trans (A,B) VALUES (1, 'blah')"
        if isinstance(pandasSQL, SQLDatabase):
            from sqlalchemy import text

            ins_sql = text(ins_sql)
        try:
            with pandasSQL.run_transaction() as trans:
                trans.execute(ins_sql)
                raise DummyException("error")
        except DummyException:
            # ignore raised exception
            pass
        with pandasSQL.run_transaction():
            res = pandasSQL.read_query("SELECT * FROM test_trans")
        assert len(res) == 0

        # Make sure when transaction is committed, rows do get inserted
        with pandasSQL.run_transaction() as trans:
            trans.execute(ins_sql)
            res2 = pandasSQL.read_query("SELECT * FROM test_trans")
        assert len(res2) == 1


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_get_schema_create_table(conn, request, test_frame3):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds (expected failure for the connection string). Generates a create statement from a frame with the schema generator, executes it (opening a connection for an engine), then reads the empty table with the table reader and compares it to the zero-row frame (index type not checked).
    # Use a dataframe without a bool column, since MySQL converts bool to
    # TINYINT (which read_sql_table returns as an int and causes a dtype
    # mismatch)
    if conn == "sqlite_str":
        request.applymarker(
            pytest.mark.xfail(reason="test does not support sqlite_str fixture")
        )

    conn = request.getfixturevalue(conn)

    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    tbl = "test_get_schema_create_table"
    create_sql = sql.get_schema(test_frame3, tbl, con=conn)
    blank_test_df = test_frame3.iloc[:0]

    create_sql = text(create_sql)
    if isinstance(conn, Engine):
        with conn.connect() as newcon:
            with newcon.begin():
                newcon.execute(create_sql)
    else:
        conn.execute(create_sql)
    returned_df = sql.read_sql_table(tbl, conn)
    tm.assert_frame_equal(returned_df, blank_test_df, check_index_type=False)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_dtype(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds (skipped for the connection string). Writing with a per-column type mapping to TEXT gives a TEXT column in the reflected metadata; a plain string type (not a SQLAlchemy type) raises a value error; a string type with length ten yields length ten; a single type applies to all columns.
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    conn = request.getfixturevalue(conn)

    from sqlalchemy import (
        TEXT,
        String,
    )
    from sqlalchemy.schema import MetaData

    cols = ["A", "B"]
    data = [(0.8, True), (0.9, None)]
    df = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": TEXT}) == 2
    meta = MetaData()
    meta.reflect(bind=conn)
    sqltype = meta.tables["dtype_test2"].columns["B"].type
    assert isinstance(sqltype, TEXT)
    msg = "The type of B is not a SQLAlchemy type"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": str})

    # GH9083
    assert df.to_sql(name="dtype_test3", con=conn, dtype={"B": String(10)}) == 2
    meta.reflect(bind=conn)
    sqltype = meta.tables["dtype_test3"].columns["B"].type
    assert isinstance(sqltype, String)
    assert sqltype.length == 10

    # single dtype
    assert df.to_sql(name="single_dtype_test", con=conn, dtype=TEXT) == 2
    meta.reflect(bind=conn)
    sqltypea = meta.tables["single_dtype_test"].columns["A"].type
    sqltypeb = meta.tables["single_dtype_test"].columns["B"].type
    assert isinstance(sqltypea, TEXT)
    assert isinstance(sqltypeb, TEXT)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_notna_dtype(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds (skipped for the connection string). Writes a frame of boolean, date, object-integer and float columns each with a missing value; the reflected column types are boolean (integer for MySQL), datetime, integer and float, showing that types are inferred from the non-missing values.
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    conn_name = conn
    conn = request.getfixturevalue(conn)

    from sqlalchemy import (
        Boolean,
        DateTime,
        Float,
        Integer,
    )
    from sqlalchemy.schema import MetaData

    cols = {
        "Bool": Series([True, None]),
        "Date": Series([datetime(2012, 5, 1), None]),
        "Int": Series([1, None], dtype="object"),
        "Float": Series([1.1, None]),
    }
    df = DataFrame(cols)

    tbl = "notna_dtype_test"
    assert df.to_sql(name=tbl, con=conn) == 2
    _ = sql.read_sql_table(tbl, conn)
    meta = MetaData()
    meta.reflect(bind=conn)
    my_type = Integer if "mysql" in conn_name else Boolean
    col_dict = meta.tables[tbl].columns
    assert isinstance(col_dict["Bool"].type, my_type)
    assert isinstance(col_dict["Date"].type, DateTime)
    assert isinstance(col_dict["Int"].type, Integer)
    assert isinstance(col_dict["Float"].type, Float)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_double_precision(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds (skipped for the connection string). Writes 32- and 64-bit floats, one 64-bit column forced to a 23-bit precision float, and 32/64-bit integers; verifies 64-bit floats keep 14-decimal precision, the forced column has the same SQL type as the 32-bit float, and integer types are integer and big integer.
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    conn = request.getfixturevalue(conn)

    from sqlalchemy import (
        BigInteger,
        Float,
        Integer,
    )
    from sqlalchemy.schema import MetaData

    V = 1.23456789101112131415

    df = DataFrame(
        {
            "f32": Series([V], dtype="float32"),
            "f64": Series([V], dtype="float64"),
            "f64_as_f32": Series([V], dtype="float64"),
            "i32": Series([5], dtype="int32"),
            "i64": Series([5], dtype="int64"),
        }
    )

    assert (
        df.to_sql(
            name="test_dtypes",
            con=conn,
            index=False,
            if_exists="replace",
            dtype={"f64_as_f32": Float(precision=23)},
        )
        == 1
    )
    res = sql.read_sql_table("test_dtypes", conn)

    # check precision of float64
    assert np.round(df["f64"].iloc[0], 14) == np.round(res["f64"].iloc[0], 14)

    # check sql types
    meta = MetaData()
    meta.reflect(bind=conn)
    col_dict = meta.tables["test_dtypes"].columns
    assert str(col_dict["f32"].type) == str(col_dict["f64_as_f32"].type)
    assert isinstance(col_dict["f32"].type, Float)
    assert isinstance(col_dict["f64"].type, Float)
    assert isinstance(col_dict["i32"].type, Integer)
    assert isinstance(col_dict["i64"].type, BigInteger)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_connectable_issue_example(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. Reproduces a reported issue: reading a table through a connection obtained from an engine (or the connection itself) and appending the result to the same table within a transaction must work. Defines nested helper functions for selecting, appending, wrapping in a transaction and the entry point, then runs the entry point after creating the table with three rows.
    conn = request.getfixturevalue(conn)

    # This tests the example raised in issue
    # https://github.com/pandas-dev/pandas/issues/10104
    from sqlalchemy.engine import Engine

    def test_select(connection):
        # [FUNCTION DOC]
        # HELPER: nested helper of the connectable-issue test. Reads the whole test table through the given connection with the query reader and returns the frame.
        query = "SELECT test_foo_data FROM test_foo_data"
        return sql.read_sql_query(query, con=connection)

    def test_append(connection, data):
        # [FUNCTION DOC]
        # HELPER: nested helper of the connectable-issue test. Appends the given frame to the test table through the given connection; returns nothing.
        data.to_sql(name="test_foo_data", con=connection, if_exists="append")

    def test_connectable(conn):
        # [FUNCTION DOC]
        # HELPER: nested helper of the connectable-issue test. Reads the table and appends the result back using the given connection.
        # https://github.com/sqlalchemy/sqlalchemy/commit/
        # 00b5c10846e800304caa86549ab9da373b42fa5d#r48323973
        foo_data = test_select(conn)
        test_append(conn, foo_data)

    def main(connectable):
        # [FUNCTION DOC]
        # HELPER: nested helper of the connectable-issue test. If given an engine, opens a connection and a transaction and calls the connection helper within it; otherwise calls the helper directly.
        # KNOWN ISSUE: these nested helper functions are named like tests but are not collected since they are nested.
        if isinstance(connectable, Engine):
            with connectable.connect() as conn:
                with conn.begin():
                    test_connectable(conn)
        else:
            test_connectable(connectable)

    assert (
        DataFrame({"test_foo_data": [0, 1, 2]}).to_sql(name="test_foo_data", con=conn)
        == 3
    )
    main(conn)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
@pytest.mark.parametrize(
    "input",
    [{"foo": [np.inf]}, {"foo": [-np.inf]}, {"foo": [-np.inf], "infe0": ["bar"]}],
)
def test_to_sql_with_negative_npinf(conn, request, input):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds and three frames with infinity. Regression: writing negative or positive infinity. For MySQL, a database error with an execution-failed message is expected; for others the write returns one row and reading back equals the original.
    # GH 34431

    df = DataFrame(input)
    conn_name = conn
    conn = request.getfixturevalue(conn)

    if "mysql" in conn_name:
        # GH 36465
        # The input {"foo": [-np.inf], "infe0": ["bar"]} does not raise any error
        # for pymysql version >= 0.10
        msg = "Execution failed on sql"
        with pytest.raises(pd.errors.DatabaseError, match=msg):
            df.to_sql(name="foobar", con=conn, index=False)
    else:
        assert df.to_sql(name="foobar", con=conn, index=False) == 1
        res = sql.read_sql_table("foobar", conn)
        tm.assert_equal(df, res)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_temporary_table(conn, request):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds (skipped for the connection string). Regression: defines an ORM mapped class for a temporary table, creates the table inside an ORM session on the connection, adds a row and reads it back with the query reader using a select on the mapped column, within the same session connection; the result equals the expected one-row frame.
    if conn == "sqlite_str":
        pytest.skip("test does not work with str connection")

    conn = request.getfixturevalue(conn)

    from sqlalchemy import (
        Column,
        Integer,
        Unicode,
        select,
    )
    from sqlalchemy.orm import (
        Session,
        declarative_base,
    )

    test_data = "Hello, World!"
    expected = DataFrame({"spam": [test_data]})
    Base = declarative_base()

    class Temporary(Base):
        # [CLASS DOC]
        # HELPER: ORM mapped class for a temporary table with an integer primary key and a required unicode column; created inside the temporary table test. Declared with a prefix making the table temporary.
        __tablename__ = "temp_test"
        __table_args__ = {"prefixes": ["TEMPORARY"]}
        id = Column(Integer, primary_key=True)
        spam = Column(Unicode(30), nullable=False)

    with Session(conn) as session:
        with session.begin():
            conn = session.connection()
            Temporary.__table__.create(conn)
            session.add(Temporary(spam=test_data))
            session.flush()
            df = sql.read_sql_query(sql=select(Temporary.spam), con=conn)
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("conn", all_connectable)
def test_invalid_engine(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Passing an unknown engine name to the write raises a value error listing the allowed engines; the built-in sqlite3 and ADBC kinds are expected failures because they do not validate the engine.
    if conn == "sqlite_buildin" or "adbc" in conn:
        request.applymarker(
            pytest.mark.xfail(
                reason="SQLiteDatabase/ADBCDatabase does not raise for bad engine"
            )
        )

    conn = request.getfixturevalue(conn)
    msg = "engine must be one of 'auto', 'sqlalchemy'"
    with pandasSQL_builder(conn) as pandasSQL:
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame1", engine="bad_engine")


@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_with_sql_engine(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Writing with the automatic engine choice works: four rows written, table exists and row count equals the frame length.
    """`to_sql` with the `engine` param"""
    # mostly copied from this class's `_to_sql()` method
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(test_frame1, "test_frame1", engine="auto") == 4
            assert pandasSQL.has_table("test_frame1")

    num_entries = len(test_frame1)
    num_rows = count_rows(conn, "test_frame1")
    assert num_rows == num_entries


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_options_sqlalchemy(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over SQLAlchemy connection kinds. With the global io SQL engine option set to SQLAlchemy, a write succeeds (four rows) and the row count matches.
    # use the set option
    conn = request.getfixturevalue(conn)
    with pd.option_context("io.sql.engine", "sqlalchemy"):
        with pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, "test_frame1") == 4
                assert pandasSQL.has_table("test_frame1")

        num_entries = len(test_frame1)
        num_rows = count_rows(conn, "test_frame1")
        assert num_rows == num_entries


@pytest.mark.parametrize("conn", all_connectable)
def test_options_auto(conn, request, test_frame1):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Same with the engine option set to automatic.
    # use the set option
    conn = request.getfixturevalue(conn)
    with pd.option_context("io.sql.engine", "auto"):
        with pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, "test_frame1") == 4
                assert pandasSQL.has_table("test_frame1")

        num_entries = len(test_frame1)
        num_rows = count_rows(conn, "test_frame1")
        assert num_rows == num_entries


def test_options_get_engine():
    # [FUNCTION DOC]
    # Requires SQLAlchemy. The engine factory returns the SQLAlchemy engine implementation both when asked explicitly and when asked for automatic, with the global option set to either value.
    pytest.importorskip("sqlalchemy")
    assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)

    with pd.option_context("io.sql.engine", "sqlalchemy"):
        assert isinstance(get_engine("auto"), SQLAlchemyEngine)
        assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)

    with pd.option_context("io.sql.engine", "auto"):
        assert isinstance(get_engine("auto"), SQLAlchemyEngine)
        assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def test_read_sql_dtype_backend(
    conn,
    request,
    string_storage,
    func,
    dtype_backend,
    dtype_backend_data,
    dtype_backend_expected,
):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds, the two read functions, the string storage setting, the dtype backend and shared fixtures for the data and the expected frame. Writes the data frame and reads with the dtype backend, comparing with the expected frame built by the expected-frame fixture; then repeats with a chunk size of three and compares every chunk. ADBC is marked as expected failure before the chunked part.
    # KNOWN ISSUE: for ADBC the expected-failure marker is applied after the non-chunked comparison; the marker applies to the whole test, even though only the chunked part is unsupported.
    # GH#50048
    conn_name = conn
    conn = request.getfixturevalue(conn)
    table = "test"
    df = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")

    with pd.option_context("mode.string_storage", string_storage):
        result = getattr(pd, func)(
            f"Select * from {table}", conn, dtype_backend=dtype_backend
        )
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)

    tm.assert_frame_equal(result, expected)

    if "adbc" in conn_name:
        # adbc does not support chunksize argument
        request.applymarker(
            pytest.mark.xfail(reason="adbc does not support chunksize argument")
        )

    with pd.option_context("mode.string_storage", string_storage):
        iterator = getattr(pd, func)(
            f"Select * from {table}",
            con=conn,
            dtype_backend=dtype_backend,
            chunksize=3,
        )
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table"])
def test_read_sql_dtype_backend_table(
    conn,
    request,
    string_storage,
    func,
    dtype_backend,
    dtype_backend_data,
    dtype_backend_expected,
):
    # [FUNCTION DOC]
    # Parametrized like the previous test but for the general reader and the table reader with a table name. Non-ADBC sqlite kinds are expected failures. Writes data, reads with the backend and compares; for ADBC returns before the chunked part, otherwise also reads in chunks of three and compares each chunk.
    if "sqlite" in conn and "adbc" not in conn:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "SQLite actually returns proper boolean values via "
                    "read_sql_table, but before pytest refactor was skipped"
                )
            )
        )
    # GH#50048
    conn_name = conn
    conn = request.getfixturevalue(conn)
    table = "test"
    df = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")

    with pd.option_context("mode.string_storage", string_storage):
        result = getattr(pd, func)(table, conn, dtype_backend=dtype_backend)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    tm.assert_frame_equal(result, expected)

    if "adbc" in conn_name:
        # adbc does not support chunksize argument
        return

    with pd.option_context("mode.string_storage", string_storage):
        iterator = getattr(pd, func)(
            table,
            conn,
            dtype_backend=dtype_backend,
            chunksize=3,
        )
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table", "read_sql_query"])
def test_read_sql_invalid_dtype_backend_table(conn, request, func, dtype_backend_data):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds and the three read functions. An invalid dtype backend name raises a value error listing the two allowed backends.
    conn = request.getfixturevalue(conn)
    table = "test"
    df = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")

    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        getattr(pd, func)(table, conn, dtype_backend="numpy")


@pytest.fixture
def dtype_backend_data() -> DataFrame:
    # [FUNCTION DOC]
    # Fixture returning a frame of eight columns: nullable integers (with and without a missing value), nullable floats (with and without), booleans (with and without a missing value), and strings (with and without a missing value).
    return DataFrame(
        {
            "a": Series([1, pd.NA, 3], dtype="Int64"),
            "b": Series([1, 2, 3], dtype="Int64"),
            "c": Series([1.5, pd.NA, 2.5], dtype="Float64"),
            "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
            "e": [True, False, None],
            "f": [True, False, True],
            "g": ["a", "b", "c"],
            "h": ["a", "b", None],
        }
    )


@pytest.fixture
def dtype_backend_expected():
    # [FUNCTION DOC]
    # Fixture returning a function used by tests to compute the expected frame after reading the data with a dtype backend. Takes no arguments itself; the returned function takes the string storage, dtype backend and connection name.
    def func(string_storage, dtype_backend, conn_name) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: returned by the expected-frame fixture. Builds the expected frame for the shared data: nullable integer, float, boolean and string columns (string dtype from the string storage setting or the Arrow string type); for the pyarrow backend converts every column to an Arrow-backed extension array; for MySQL and sqlite connections the boolean columns become integer columns of the respective nullable or Arrow type. Returns the frame.
        string_dtype: pd.StringDtype | pd.ArrowDtype
        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)

        df = DataFrame(
            {
                "a": Series([1, pd.NA, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, pd.NA, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": Series([True, False, pd.NA], dtype="boolean"),
                "f": Series([True, False, True], dtype="boolean"),
                "g": Series(["a", "b", "c"], dtype=string_dtype),
                "h": Series(["a", "b", None], dtype=string_dtype),
            }
        )
        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")

            from pandas.arrays import ArrowExtensionArray

            df = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(df[col], from_pandas=True))
                    for col in df.columns
                }
            )

        if "mysql" in conn_name or "sqlite" in conn_name:
            if dtype_backend == "numpy_nullable":
                df = df.astype({"e": "Int64", "f": "Int64"})
            else:
                df = df.astype({"e": "int64[pyarrow]", "f": "int64[pyarrow]"})

        return df

    return func


@pytest.mark.parametrize("conn", all_connectable)
def test_chunksize_empty_dtypes(conn, request):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds. Regression: an empty frame with integer and object columns is written and read back in chunks of one row with those dtypes; every chunk (if any) equals the empty expected frame. ADBC is an expected failure.
    # GH#50245
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    conn = request.getfixturevalue(conn)
    dtypes = {"a": "int64", "b": "object"}
    df = DataFrame(columns=["a", "b"]).astype(dtypes)
    expected = df.copy()
    df.to_sql(name="test", con=conn, index=False, if_exists="replace")

    for result in read_sql_query(
        "SELECT * FROM test",
        conn,
        dtype=dtypes,
        chunksize=1,
    ):
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("dtype_backend", [lib.no_default, "numpy_nullable"])
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def test_read_sql_dtype(conn, request, func, dtype_backend):
    # [FUNCTION DOC]
    # Parametrized over all connection kinds, both read functions and the dtype backend (default or nullable). Regression: a per-column dtype argument converts the first column to 64-bit float, and the other column keeps its integer type (nullable integer if the nullable backend is requested).
    # GH#50797
    conn = request.getfixturevalue(conn)
    table = "test"
    df = DataFrame({"a": [1, 2, 3], "b": 5})
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")

    result = getattr(pd, func)(
        f"Select * from {table}",
        conn,
        dtype={"a": np.float64},
        dtype_backend=dtype_backend,
    )
    expected = DataFrame(
        {
            "a": Series([1, 2, 3], dtype=np.float64),
            "b": Series(
                [5, 5, 5],
                dtype="int64" if not dtype_backend == "numpy_nullable" else "Int64",
            ),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_bigint_warning(sqlite_engine):
    # [FUNCTION DOC]
    # Regression: reading a table with a big integer column through the sqlite engine emits no warning.
    conn = sqlite_engine
    # test no warning for BIGINT (to support int64) is raised (GH7433)
    df = DataFrame({"a": [1, 2]}, dtype="int64")
    assert df.to_sql(name="test_bigintwarning", con=conn, index=False) == 2

    with tm.assert_produces_warning(None):
        sql.read_sql_table("test_bigintwarning", conn)


def test_valueerror_exception(sqlite_engine):
    # [FUNCTION DOC]
    # Writing with an empty table name raises a value error saying the table name is empty.
    conn = sqlite_engine
    df = DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with pytest.raises(ValueError, match="Empty table name specified"):
        df.to_sql(name="", con=conn, if_exists="replace", index=False)


def test_row_object_is_named_tuple(sqlite_engine):
    # [FUNCTION DOC]
    # Regression test for the named tuple check: builds an ORM class and session on the sqlite engine, writes a frame, queries the mapped columns and constructs a frame from the query's row objects; the column labels equal the mapped column names.
    conn = sqlite_engine
    # GH 40682
    # Test for the is_named_tuple() function
    # Placed here due to its usage of sqlalchemy

    from sqlalchemy import (
        Column,
        Integer,
        String,
    )
    from sqlalchemy.orm import (
        declarative_base,
        sessionmaker,
    )

    BaseModel = declarative_base()

    class Test(BaseModel):
        # [CLASS DOC]
        # HELPER: ORM mapped class with an integer primary key and a 50 character string column, bound to the table named for the test frame; used only to produce row objects with named fields for the named tuple test.
        __tablename__ = "test_frame"
        id = Column(Integer, primary_key=True)
        string_column = Column(String(50))

    with conn.begin():
        BaseModel.metadata.create_all(conn)
    Session = sessionmaker(bind=conn)
    with Session() as session:
        df = DataFrame({"id": [0, 1], "string_column": ["hello", "world"]})
        assert (
            df.to_sql(name="test_frame", con=conn, index=False, if_exists="replace")
            == 2
        )
        session.commit()
        test_query = session.query(Test.id, Test.string_column)
        df = DataFrame(test_query)

    assert list(df.columns) == ["id", "string_column"]


def test_read_sql_string_inference(sqlite_engine):
    # [FUNCTION DOC]
    # Regression test: with the global string-inference option enabled, reading a table with a string column returns the string dtype (with missing value being not-a-number) for both the column and the column index.
    conn = sqlite_engine
    # GH#54430
    table = "test"
    df = DataFrame({"a": ["x", "y"]})
    df.to_sql(table, con=conn, index=False, if_exists="replace")

    with pd.option_context("future.infer_string", True):
        result = read_sql_table(table, conn)

    dtype = pd.StringDtype(na_value=np.nan)
    expected = DataFrame(
        {"a": ["x", "y"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
    )

    tm.assert_frame_equal(result, expected)


def test_roundtripping_datetimes(sqlite_engine):
    # [FUNCTION DOC]
    # Regression: a nanosecond-unit datetime written through the sqlite engine and read with a plain query comes back as the string with microsecond precision.
    conn = sqlite_engine
    # GH#54877
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql("test", conn, if_exists="replace", index=False)
    result = pd.read_sql("select * from test", conn).iloc[0, 0]
    assert result == "2020-12-31 12:00:00.000000"


@pytest.fixture
def sqlite_builtin_detect_types():
    # [FUNCTION DOC]
    # Fixture yielding an in-memory built-in sqlite3 connection configured to parse declared column types into native Python objects, properly closed after use.
    with contextlib.closing(
        sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
    ) as closing_conn:
        with closing_conn as conn:
            yield conn


def test_roundtripping_datetimes_detect_types(sqlite_builtin_detect_types):
    # [FUNCTION DOC]
    # Regression: with a declared-type parsing sqlite connection, a nanosecond-unit datetime written and read by query comes back as a timestamp equal to the original.
    # https://github.com/pandas-dev/pandas/issues/55554
    conn = sqlite_builtin_detect_types
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql("test", conn, if_exists="replace", index=False)
    result = pd.read_sql("select * from test", conn).iloc[0, 0]
    assert result == Timestamp("2020-12-31 12:00:00.000000")


@pytest.mark.db
def test_psycopg2_schema_support(postgresql_psycopg2_engine):
    # [FUNCTION DOC]
    # Database-marked test for PostgreSQL. Creates a schema, writes frames to the default, an explicit public schema and a second schema (each returning two rows), reads each back using the table reader with and without the schema argument (reading the second table under the wrong schema raises a value error that the table was not found). Then recreates the schema and checks the replace and append existence modes produce a doubled frame.
    conn = postgresql_psycopg2_engine

    # only test this for postgresql (schema's not supported in
    # mysql/sqlite)
    df = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})

    # create a schema
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")

    # write dataframe to different schema's
    assert df.to_sql(name="test_schema_public", con=conn, index=False) == 2
    assert (
        df.to_sql(
            name="test_schema_public_explicit",
            con=conn,
            index=False,
            schema="public",
        )
        == 2
    )
    assert (
        df.to_sql(name="test_schema_other", con=conn, index=False, schema="other") == 2
    )

    # read dataframes back in
    res1 = sql.read_sql_table("test_schema_public", conn)
    tm.assert_frame_equal(df, res1)
    res2 = sql.read_sql_table("test_schema_public_explicit", conn)
    tm.assert_frame_equal(df, res2)
    res3 = sql.read_sql_table("test_schema_public_explicit", conn, schema="public")
    tm.assert_frame_equal(df, res3)
    res4 = sql.read_sql_table("test_schema_other", conn, schema="other")
    tm.assert_frame_equal(df, res4)
    msg = "Table test_schema_other not found"
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("test_schema_other", conn, schema="public")

    # different if_exists options

    # create a schema
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")

    # write dataframe with different if_exists options
    assert (
        df.to_sql(name="test_schema_other", con=conn, schema="other", index=False) == 2
    )
    df.to_sql(
        name="test_schema_other",
        con=conn,
        schema="other",
        index=False,
        if_exists="replace",
    )
    assert (
        df.to_sql(
            name="test_schema_other",
            con=conn,
            schema="other",
            index=False,
            if_exists="append",
        )
        == 2
    )
    res = sql.read_sql_table("test_schema_other", conn, schema="other")
    tm.assert_frame_equal(concat([df, df], ignore_index=True), res)


@pytest.mark.db
def test_self_join_date_columns(postgresql_psycopg2_engine):
    # [FUNCTION DOC]
    # Database-marked test for PostgreSQL. Regression: a table with a timezone-aware timestamp column is self-joined through a query so the same column name appears twice; result is a frame with duplicate names holding UTC datetimes in microsecond unit. Cleans up the table.
    # GH 44421
    conn = postgresql_psycopg2_engine
    from sqlalchemy.sql import text

    create_table = text(
        """
    CREATE TABLE person
    (
        id serial constraint person_pkey primary key,
        created_dt timestamp with time zone
    );

    INSERT INTO person
        VALUES (1, '2021-01-01T00:00:00Z');
    """
    )
    with conn.connect() as con:
        with con.begin():
            con.execute(create_table)

    sql_query = (
        'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
    )
    result = pd.read_sql(sql_query, conn)
    expected = DataFrame(
        [[1, Timestamp("2021", tz="UTC")] * 2], columns=["id", "created_dt"] * 2
    )
    expected["created_dt"] = expected["created_dt"].astype("M8[us, UTC]")
    tm.assert_frame_equal(result, expected)

    # Cleanup
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("person")


def test_create_and_drop_table(sqlite_engine):
    # [FUNCTION DOC]
    # With the sqlite engine, writes a table (four rows) and verifies it exists, drops it in another transaction and verifies it no longer exists.
    conn = sqlite_engine
    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(temp_frame, "drop_test_frame") == 4

        assert pandasSQL.has_table("drop_test_frame")

        with pandasSQL.run_transaction():
            pandasSQL.drop_table("drop_test_frame")

        assert not pandasSQL.has_table("drop_test_frame")


def test_sqlite_datetime_date(sqlite_buildin):
    # [FUNCTION DOC]
    # With a built-in sqlite3 connection, date objects are written (two rows) and read back as strings.
    conn = sqlite_buildin
    df = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    res = read_sql_query("SELECT * FROM test_date", conn)
    # comes back as strings
    tm.assert_frame_equal(res, df.astype(str))


@pytest.mark.parametrize("tz_aware", [False, True])
def test_sqlite_datetime_time(tz_aware, sqlite_buildin):
    # [FUNCTION DOC]
    # Parametrized over timezone-naive or timezone-aware time values. Regression: time values (taken from a timezone-aware datetime range in the aware case) are written to built-in sqlite3 (two rows) and read back as formatted time strings with microseconds.
    conn = sqlite_buildin
    # test support for datetime.time, GH #8341
    if not tz_aware:
        tz_times = [time(9, 0, 0), time(9, 1, 30)]
    else:
        tz_dt = date_range("2013-01-01 09:00:00", periods=2, tz="US/Pacific")
        tz_times = Series(tz_dt.to_pydatetime()).map(lambda dt: dt.timetz())

    df = DataFrame(tz_times, columns=["a"])

    assert df.to_sql(name="test_time", con=conn, index=False) == 2
    res = read_sql_query("SELECT * FROM test_time", conn)
    # comes back as strings
    expected = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
    tm.assert_frame_equal(res, expected)


def get_sqlite_column_type(conn, table, column):
    # [FUNCTION DOC]
    # HELPER: returns the declared type string of a column in a sqlite table by reading its table information; raises a value error if the column is not found.
    recs = conn.execute(f"PRAGMA table_info({table})")
    for cid, name, ctype, not_null, default, pk in recs:
        if name == column:
            return ctype
    raise ValueError(f"Table {table}, column {column} not found")


def test_sqlite_test_dtype(sqlite_buildin):
    # [FUNCTION DOC]
    # With a built-in sqlite3 connection, writing a frame with booleans declares the boolean column as integer, and a per-column dtype mapping with a plain string type (STRING) sets the declared column type to it.
    conn = sqlite_buildin
    cols = ["A", "B"]
    data = [(0.8, True), (0.9, None)]
    df = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": "STRING"}) == 2

    # sqlite stores Boolean values as INTEGER
    assert get_sqlite_column_type(conn, "dtype_test", "B") == "INTEGER"

    assert get_sqlite_column_type(conn, "dtype_test2", "B") == "STRING"
    msg = r"B \(<class 'bool'>\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": bool})

    # single dtype
    assert df.to_sql(name="single_dtype_test", con=conn, dtype="STRING") == 2
    assert get_sqlite_column_type(conn, "single_dtype_test", "A") == "STRING"
    assert get_sqlite_column_type(conn, "single_dtype_test", "B") == "STRING"


def test_sqlite_notna_dtype(sqlite_buildin):
    # [FUNCTION DOC]
    # With a built-in sqlite3 connection, writing columns of booleans, dates, object-integers and floats each with a missing value gives declared column types integer, timestamp, integer and real.
    conn = sqlite_buildin
    cols = {
        "Bool": Series([True, None]),
        "Date": Series([datetime(2012, 5, 1), None]),
        "Int": Series([1, None], dtype="object"),
        "Float": Series([1.1, None]),
    }
    df = DataFrame(cols)

    tbl = "notna_dtype_test"
    assert df.to_sql(name=tbl, con=conn) == 2

    assert get_sqlite_column_type(conn, tbl, "Bool") == "INTEGER"
    assert get_sqlite_column_type(conn, tbl, "Date") == "TIMESTAMP"
    assert get_sqlite_column_type(conn, tbl, "Int") == "INTEGER"
    assert get_sqlite_column_type(conn, tbl, "Float") == "REAL"


def test_sqlite_illegal_names(sqlite_buildin):
    # [FUNCTION DOC]
    # With a built-in sqlite3 connection, an empty table name raises a value error; unusual table names (brackets, quotes, backticks, dots, digits first, non-ascii) and columns named with them can all be written (two rows each).
    # KNOWN ISSUE: the results of the table-exists checks after writing are discarded, so nothing asserts the tables actually exist.
    # For sqlite, these should work fine
    conn = sqlite_buildin
    df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])

    msg = "Empty table or column name specified"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="", con=conn)

    for ndx, weird_name in enumerate(
        [
            "test_weird_name]",
            "test_weird_name[",
            "test_weird_name`",
            'test_weird_name"',
            "test_weird_name'",
            "_b.test_weird_name_01-30",
            '"_b.test_weird_name_01-30"',
            "99beginswithnumber",
            "12345",
            "\xe9",
        ]
    ):
        assert df.to_sql(name=weird_name, con=conn) == 2
        sql.table_exists(weird_name, conn)

        df2 = DataFrame([[1, 2], [3, 4]], columns=["a", weird_name])
        c_tbl = f"test_weird_col_name{ndx:d}"
        assert df2.to_sql(name=c_tbl, con=conn) == 2
        sql.table_exists(c_tbl, conn)


def format_query(sql, *args):
    # [FUNCTION DOC]
    # HELPER: formats a query template with literal values for the old-style string interpolation. Takes the template and values; for each value that is a float and missing replaces it with none, picks a literal formatter by the value's exact type (quoted for strings, datetimes, bytes and booleans; fixed decimals for floats; integer for ints; the word NULL for none) and returns the template filled with the formatted values. Raises a key error for an unsupported value type.
    _formatters = {
        datetime: "'{}'".format,
        str: "'{}'".format,
        np.str_: "'{}'".format,
        bytes: "'{}'".format,
        float: "{:.8f}".format,
        int: "{:d}".format,
        type(None): lambda x: "NULL",
        np.float64: "{:.10f}".format,
        bool: "'{!s}'".format,
    }
    processed_args = []
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None

        formatter = _formatters[type(arg)]
        processed_args.append(formatter(arg))

    return sql % tuple(processed_args)


def tquery(query, con=None):
    # [FUNCTION DOC]
    # HELPER: replacement for a removed legacy function. Runs the query through a database object built from the connection, fetches all rows, and returns them as a list (or none if the result is none).
    """Replace removed sql.tquery function"""
    with sql.pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)


def test_xsqlite_basic(sqlite_buildin):
    # [FUNCTION DOC]
    # With a built-in sqlite3 connection: a random frame written without the index (ten rows) is read back equal after restoring the index; then a frame with an extra text column and an integer column used as the index column round-trips with that name.
    # KNOWN ISSUE: contains a hack replacing the index of the result, and an assigned expected variable that is overwritten before use.
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    assert sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 10
    result = sql.read_sql("select * from test_table", sqlite_buildin)

    # HACK! Change this once indexes are handled properly.
    result.index = frame.index

    expected = frame
    tm.assert_frame_equal(result, frame)

    frame["txt"] = ["a"] * len(frame)
    frame2 = frame.copy()
    new_idx = Index(np.arange(len(frame2)), dtype=np.int64) + 10
    frame2["Idx"] = new_idx.copy()
    assert sql.to_sql(frame2, name="test_table2", con=sqlite_buildin, index=False) == 10
    result = sql.read_sql("select * from test_table2", sqlite_buildin, index_col="Idx")
    expected = frame.copy()
    expected.index = new_idx
    expected.index.name = "Idx"
    tm.assert_frame_equal(expected, result)


def test_xsqlite_write_row_by_row(sqlite_buildin):
    # [FUNCTION DOC]
    # Creates a table from the schema generator, then inserts the frame one row at a time using formatted literal queries (one value is missing), commits, and compares the reread frame with a relative tolerance.
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    frame.iloc[0, 0] = np.nan
    create_sql = sql.get_schema(frame, "test")
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)

    ins = "INSERT INTO test VALUES (%s, %s, %s, %s)"
    for _, row in frame.iterrows():
        fmt_sql = format_query(ins, *row)
        tquery(fmt_sql, con=sqlite_buildin)

    sqlite_buildin.commit()

    result = sql.read_sql("select * from test", con=sqlite_buildin)
    result.index = frame.index
    tm.assert_frame_equal(result, frame, rtol=1e-3)


def test_xsqlite_execute(sqlite_buildin):
    # [FUNCTION DOC]
    # Creates a table from the schema generator, inserts the first row through the database object's parameterized execute method, commits, and verifies the table contains that single row.
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    create_sql = sql.get_schema(frame, "test")
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    ins = "INSERT INTO test VALUES (?, ?, ?, ?)"

    row = frame.iloc[0]
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute(ins, tuple(row))
    sqlite_buildin.commit()

    result = sql.read_sql("select * from test", sqlite_buildin)
    result.index = frame.index[:1]
    tm.assert_frame_equal(result, frame[:1])


def test_xsqlite_schema(sqlite_buildin):
    # [FUNCTION DOC]
    # The schema generated for a frame with a datetime index column named A uses the datetime type for it; with key columns, the statement contains a primary key clause for the two columns, and executes successfully on the connection.
    # KNOWN ISSUE: the line list computed after the second schema call is never used.
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    create_sql = sql.get_schema(frame, "test")
    lines = create_sql.splitlines()
    for line in lines:
        tokens = line.split(" ")
        if len(tokens) == 2 and tokens[0] == "A":
            assert tokens[1] == "DATETIME"

    create_sql = sql.get_schema(frame, "test", keys=["A", "B"])
    lines = create_sql.splitlines()
    assert 'PRIMARY KEY ("A", "B")' in create_sql
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)


def test_xsqlite_execute_fail(sqlite_buildin):
    # [FUNCTION DOC]
    # Creates a table with a composite primary key, inserts two rows, and verifies that inserting a duplicate key raises a database error with an execution failed message.
    create_sql = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)

    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 1.234)")
        pandas_sql.execute("INSERT INTO test VALUES('foo', 'baz', 2.567)")

        with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
            pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 7)")


def test_xsqlite_execute_closed_connection():
    # [FUNCTION DOC]
    # Creates a table and inserts a row inside a database object, closes the connection, then runs a query on the closed connection through the helper, which raises a sqlite programming error that the database is closed.
    create_sql = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        cur = conn.cursor()
        cur.execute(create_sql)

        with sql.pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute("INSERT INTO test VALUES('foo', 'bar', 1.234)")

    msg = "Cannot operate on a closed database."
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        tquery("select * from test", con=conn)


def test_xsqlite_keyword_as_column_names(sqlite_buildin):
    # [FUNCTION DOC]
    # Writing a column named with an SQL keyword (From) works: five rows written.
    df = DataFrame({"From": np.ones(5)})
    assert sql.to_sql(df, con=sqlite_buildin, name="testkeywords", index=False) == 5


def test_xsqlite_onecolumn_of_integer(sqlite_buildin):
    # [FUNCTION DOC]
    # Regression: a one-column integer frame is written (two rows), summing the column through plain SQL gives three, and the table reads back equal to the frame.
    # GH 3628
    # a column_of_integers dataframe should transfer well to sql

    mono_df = DataFrame([1, 2], columns=["c0"])
    assert sql.to_sql(mono_df, con=sqlite_buildin, name="mono_df", index=False) == 2
    # computing the sum via sql
    con_x = sqlite_buildin
    the_sum = sum(my_c0[0] for my_c0 in con_x.execute("select * from mono_df"))
    # it should not fail, and gives 3 ( Issue #3628 )
    assert the_sum == 3

    result = sql.read_sql("select * from mono_df", con_x)
    tm.assert_frame_equal(result, mono_df)


def test_xsqlite_if_exists(sqlite_buildin):
    # [FUNCTION DOC]
    # With a built-in sqlite3 connection, an invalid existence mode raises a value error; fail mode raises on an existing table; replace mode replaces the table contents with the new rows (three rows returned); append mode keeps and extends the rows. Compares the stored rows with the helper query between steps and drops the table between phases.
    df_if_exists_1 = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2 = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name = "table_if_exists"
    sql_select = f"SELECT * FROM {table_name}"

    msg = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(
            frame=df_if_exists_1,
            con=sqlite_buildin,
            name=table_name,
            if_exists="notvalidvalue",
        )
    drop_table(table_name, sqlite_buildin)

    # test if_exists='fail'
    sql.to_sql(
        frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail"
    )
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(
            frame=df_if_exists_1,
            con=sqlite_buildin,
            name=table_name,
            if_exists="fail",
        )
    # test if_exists='replace'
    sql.to_sql(
        frame=df_if_exists_1,
        con=sqlite_buildin,
        name=table_name,
        if_exists="replace",
        index=False,
    )
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert (
        sql.to_sql(
            frame=df_if_exists_2,
            con=sqlite_buildin,
            name=table_name,
            if_exists="replace",
            index=False,
        )
        == 3
    )
    assert tquery(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
    drop_table(table_name, sqlite_buildin)

    # test if_exists='append'
    assert (
        sql.to_sql(
            frame=df_if_exists_1,
            con=sqlite_buildin,
            name=table_name,
            if_exists="fail",
            index=False,
        )
        == 2
    )
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert (
        sql.to_sql(
            frame=df_if_exists_2,
            con=sqlite_buildin,
            name=table_name,
            if_exists="append",
            index=False,
        )
        == 3
    )
    assert tquery(sql_select, con=sqlite_buildin) == [
        (1, "A"),
        (2, "B"),
        (3, "C"),
        (4, "D"),
        (5, "E"),
    ]
    drop_table(table_name, sqlite_buildin)
