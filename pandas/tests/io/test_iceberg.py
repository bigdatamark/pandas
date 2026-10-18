"""
Tests for the Apache Iceberg format.

Tests in this file use a simple Iceberg catalog based on SQLite, with the same
data used for Parquet tests (``pandas/tests/io/data/parquet/simple.parquet``).
"""

# [MODULE DOC]
# Tests for Apache Iceberg reading and writing, using a SQLite-backed Iceberg catalog
# built per test with the same data as the simple parquet test file. Requires pyiceberg
# and pyarrow (module skipped otherwise); marked single-cpu. Imports the Iceberg reader
# from the pandas I/O package.

import collections
import importlib
import pathlib

import pytest

import pandas as pd
import pandas._testing as tm

from pandas.io.iceberg import read_iceberg

pytestmark = pytest.mark.single_cpu

pyiceberg = pytest.importorskip("pyiceberg")
pyiceberg_catalog = pytest.importorskip("pyiceberg.catalog")
pq = pytest.importorskip("pyarrow.parquet")

Catalog = collections.namedtuple("Catalog", ["name", "uri", "warehouse"])


@pytest.fixture
def catalog(request, tmp_path):
    # [FUNCTION DOC]
    # HELPER: fixture creating a temporary SQL (SQLite) Iceberg catalog.
    # Optional indirect parameter gives a catalog name (default name when absent).
    # Creates the catalog in the temp directory with the file warehouse, a namespace
    # named ns and a table ns.my_table holding the simple parquet data (columns A: 1,2,3
    # and B: foo x3), because the catalog stores absolute file paths and cannot be
    # checked in. When a name is given, writes a catalog configuration file in the
    # user's home directory and reloads the catalog module to pick it up, and deletes
    # the file on teardown. Yields a named tuple of name (or default), uri and
    # warehouse.
    # the catalog stores the full path of data files, so the catalog needs to be
    # created dynamically, and not saved in pandas/tests/io/data as other formats
    uri = f"sqlite:///{tmp_path}/catalog.sqlite"
    warehouse = f"file://{tmp_path}"
    catalog_name = request.param if hasattr(request, "param") else None
    catalog = pyiceberg_catalog.load_catalog(
        catalog_name or "default",
        type="sql",
        uri=uri,
        warehouse=warehouse,
    )
    catalog.create_namespace("ns")

    df = pq.read_table(
        pathlib.Path(__file__).parent / "data" / "parquet" / "simple.parquet"
    )
    table = catalog.create_table("ns.my_table", schema=df.schema)
    table.append(df)

    if catalog_name is not None:
        config_path = pathlib.Path.home() / ".pyiceberg.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"""\
catalog:
  {catalog_name}:
    type: sql
    uri: {uri}
    warehouse: {warehouse}""")

        importlib.reload(pyiceberg_catalog)  # needed to reload the config file

    yield Catalog(name=catalog_name or "default", uri=uri, warehouse=warehouse)

    if catalog_name is not None:
        config_path.unlink()


class TestIceberg:
    # [CLASS DOC]
    # Tests of reading (with filters, case sensitivity, limit, catalog name) and writing
    # (new table, append, overwrite) Iceberg tables via the read function and the frame
    # writer method.
    def test_read(self, catalog):
        # [FUNCTION DOC]
        # Reading ns.my_table with catalog properties giving the uri returns the
        # expected two-column frame.
        expected = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": ["foo", "foo", "foo"],
            }
        )
        result = read_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("catalog", ["default", "pandas_tests"], indirect=True)
    def test_read_by_catalog_name(self, catalog):
        # [FUNCTION DOC]
        # Parametrized (indirectly) over catalog names default and pandas_tests. Reading
        # by catalog name via configuration returns the expected frame.
        expected = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": ["foo", "foo", "foo"],
            }
        )
        result = read_iceberg(
            "ns.my_table",
            catalog_name=catalog.name,
        )
        tm.assert_frame_equal(result, expected)

    def test_read_with_row_filter(self, catalog):
        # [FUNCTION DOC]
        # A row filter keeping rows where A is greater than 1 returns the last two rows.
        expected = pd.DataFrame(
            {
                "A": [2, 3],
                "B": ["foo", "foo"],
            }
        )
        result = read_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
            row_filter="A > 1",
        )
        tm.assert_frame_equal(result, expected)

    def test_read_with_case_sensitive(self, catalog):
        # [FUNCTION DOC]
        # Selecting column a with case-insensitive matching returns column A; with case-
        # sensitive matching it raises a value error starting with Could not find
        # column.
        expected = pd.DataFrame(
            {
                "A": [1, 2, 3],
            }
        )
        result = read_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
            columns=["a"],
            case_sensitive=False,
        )
        tm.assert_frame_equal(result, expected)

        with pytest.raises(ValueError, match="^Could not find column"):
            read_iceberg(
                "ns.my_table",
                catalog_properties={"uri": catalog.uri},
                columns=["a"],
                case_sensitive=True,
            )

    def test_read_with_limit(self, catalog):
        # [FUNCTION DOC]
        # A limit of 2 returns the first two rows.
        expected = pd.DataFrame(
            {
                "A": [1, 2],
                "B": ["foo", "foo"],
            }
        )
        result = read_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
            limit=2,
        )
        tm.assert_frame_equal(result, expected)

    def test_write(self, catalog):
        # [FUNCTION DOC]
        # Writes a frame as a new table in the catalog (explicit location of the
        # warehouse) and reads it back; equal to the written frame.
        df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": ["foo", "foo", "foo"],
            }
        )
        df.to_iceberg(
            "ns.new_table",
            catalog_properties={"uri": catalog.uri},
            location=catalog.warehouse,
        )
        result = read_iceberg(
            "ns.new_table",
            catalog_properties={"uri": catalog.uri},
        )
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("catalog", ["default", "pandas_tests"], indirect=True)
    def test_write_by_catalog_name(self, catalog):
        # [FUNCTION DOC]
        # Parametrized over default and pandas_tests catalog names. Writes by catalog
        # name (no properties, no location) and reads back; equal to the frame.
        df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": ["foo", "foo", "foo"],
            }
        )
        df.to_iceberg(
            "ns.new_table",
            catalog_name=catalog.name,
        )
        result = read_iceberg(
            "ns.new_table",
            catalog_name=catalog.name,
        )
        tm.assert_frame_equal(result, df)

    def test_write_existing_table_with_append_true(self, catalog):
        # [FUNCTION DOC]
        # Appending to an existing table adds the new rows after the original; result
        # equals the concatenation with a fresh index.
        original = read_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
        )
        new = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": ["foo", "foo", "foo"],
            }
        )
        expected = pd.concat([original, new], ignore_index=True)
        new.to_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
            location=catalog.warehouse,
            append=True,
        )
        result = read_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
        )
        tm.assert_frame_equal(result, expected)

    def test_write_existing_table_with_append_false(self, catalog):
        # [FUNCTION DOC]
        # Writing to an existing table with append disabled replaces its contents;
        # reading gives only the new frame.
        df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": ["foo", "foo", "foo"],
            }
        )
        df.to_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
            location=catalog.warehouse,
            append=False,
        )
        result = read_iceberg(
            "ns.my_table",
            catalog_properties={"uri": catalog.uri},
        )
        tm.assert_frame_equal(result, df)
