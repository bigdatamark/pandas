# [MODULE DOC]
# Tests for HDFStore key listing: the keys method and iteration over a store, native (non-pandas) PyTables nodes via the include argument, argument validation, and that soft links are ignored. Skipped if tables is missing; module marked single_cpu.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    HDFStore,
    Index,
    Series,
    date_range,
)

tables = pytest.importorskip("tables")

pytestmark = [pytest.mark.single_cpu]


def test_keys(temp_hdfstore):
    # [FUNCTION DOC]
    # Stores two series and a frame under keys a, b, c; asserts the store length is three and that both the keys method and iteration produce the set of slash-prefixed keys.
    temp_hdfstore["a"] = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    temp_hdfstore["b"] = Series(
        range(10), dtype="float64", index=[f"i_{i}" for i in range(10)]
    )
    temp_hdfstore["c"] = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    assert len(temp_hdfstore) == 3
    expected = {"/a", "/b", "/c"}
    assert set(temp_hdfstore.keys()) == expected
    assert set(temp_hdfstore) == expected


def test_non_pandas_keys(temp_h5_path):
    # [FUNCTION DOC]
    # Builds a file directly with PyTables containing a group with three tables, each described by a locally defined description class. Opening it as a store, asserts that keys restricted to native nodes lists the three paths, keys restricted to pandas objects is empty, and each native table can be fetched as a frame with one column.
    class Table1(tables.IsDescription):
        # [CLASS DOC]
        # HELPER: PyTables row description with one 32-bit float column, used to create the first native table.
        value1 = tables.Float32Col()

    class Table2(tables.IsDescription):
        # [CLASS DOC]
        # HELPER: PyTables row description with one 32-bit float column named value2, used to create the second native table.
        value2 = tables.Float32Col()

    class Table3(tables.IsDescription):
        # [CLASS DOC]
        # HELPER: PyTables row description with one 32-bit float column named value3, used to create the third native table.
        value3 = tables.Float32Col()

    with tables.open_file(temp_h5_path, mode="w") as h5file:
        group = h5file.create_group("/", "group")
        h5file.create_table(group, "table1", Table1, "Table 1")
        h5file.create_table(group, "table2", Table2, "Table 2")
        h5file.create_table(group, "table3", Table3, "Table 3")
    with HDFStore(temp_h5_path) as store:
        assert len(store.keys(include="native")) == 3
        expected = {"/group/table1", "/group/table2", "/group/table3"}
        assert set(store.keys(include="native")) == expected
        assert set(store.keys(include="pandas")) == set()
        for name in expected:
            df = store.get(name)
            assert len(df.columns) == 1


def test_keys_illegal_include_keyword_value(temp_hdfstore):
    # [FUNCTION DOC]
    # Calling keys with an include value other than pandas or native raises a value error stating the allowed values and the illegal one.
    with pytest.raises(
        ValueError,
        match="`include` should be either 'pandas' or 'native' but is 'illegal'",
    ):
        temp_hdfstore.keys(include="illegal")


def test_keys_ignore_hdf_softlink(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 20523. After storing a frame, creates a soft link at the file root through the PyTables handle and asserts the keys listing still contains only the single real key.
    # GH 20523
    # Puts a softlink into HDF file and rereads
    df = DataFrame({"A": range(5), "B": range(5)})
    temp_hdfstore.put("df", df)

    assert temp_hdfstore.keys() == ["/df"]

    temp_hdfstore._handle.create_soft_link(temp_hdfstore._handle.root, "symlink", "df")

    # Should ignore the softlink
    assert temp_hdfstore.keys() == ["/df"]
