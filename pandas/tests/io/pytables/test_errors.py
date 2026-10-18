# [MODULE DOC]
# Tests for error handling and validation messages in the HDF5 IO layer: unsupported selections on fixed format, incompatible index dtypes on append, unimplemented table column dtypes, invalid query terms and variable references, mismatched column-name types, invalid compression library, unsupported multi-index extension dtypes, invalid or closed files and generic buffers, and the private name-adjustment helper's version validation.
# Marked single_cpu. Uses temp_hdfstore and temp_h5_path fixtures.
import datetime
from io import BytesIO
import re
import uuid

import numpy as np
import pytest

from pandas import (
    CategoricalIndex,
    DataFrame,
    HDFStore,
    Index,
    MultiIndex,
    date_range,
    read_hdf,
)

from pandas.io.pytables import (
    Term,
    _maybe_adjust_name,
)

pytestmark = [pytest.mark.single_cpu]


def test_pass_spec_to_storer(temp_hdfstore):
    # [FUNCTION DOC]
    # Stores a frame in fixed format. Asserts type errors when selecting with a column specification or a where specification, each saying the fixed-format store must be selected in its entirety.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    temp_hdfstore.put("df", df)
    msg = (
        "cannot pass a column specification when reading a Fixed format "
        "store. this store must be selected in its entirety"
    )
    with pytest.raises(TypeError, match=msg):
        temp_hdfstore.select("df", columns=["A"])
    msg = (
        "cannot pass a where specification when reading from a Fixed "
        "format store. this store must be selected in its entirety"
    )
    with pytest.raises(TypeError, match=msg):
        temp_hdfstore.select("df", where=["columns=A"])


def test_table_index_incompatible_dtypes(temp_hdfstore):
    # [FUNCTION DOC]
    # Puts an integer-indexed frame in table format then appends a datetime-indexed frame; asserts a type error stating the incompatible kind in the index column (integer versus datetime).
    df1 = DataFrame({"a": [1, 2, 3]})
    df2 = DataFrame(
        {"a": [4, 5, 6]}, index=date_range("1/1/2000", periods=3, unit="ns")
    )

    temp_hdfstore.put("frame", df1, format="table")
    msg = re.escape("incompatible kind in col [integer - datetime64[ns]]")
    with pytest.raises(TypeError, match=msg):
        temp_hdfstore.put("frame", df2, format="table", append=True)


def test_unimplemented_dtypes_table_columns(temp_hdfstore):
    # [FUNCTION DOC]
    # Loops over a list of unsupported column types (currently just a date object) and asserts that appending a frame with such a column raises a type error saying it is not implemented as a table column.
    dtypes = [("date", datetime.date(2001, 1, 2))]

    # currently not supported dtypes ####
    for n, f in dtypes:
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df[n] = f
        msg = re.escape(f"[{n}] is not implemented as a table column")
        with pytest.raises(TypeError, match=msg):
            temp_hdfstore.append(f"df1_{n}", df)


def test_unimplemented_dtypes_table_columns2(temp_hdfstore):
    # [FUNCTION DOC]
    # A frame with string columns and a date column placed in the object block (after consolidation); asserts a type error either that the column cannot be serialized because its contents are dates rather than strings, or that date is not implemented as a table column (either message accepted).
    # frame
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    df["obj1"] = "foo"
    df["obj2"] = "bar"
    df["datetime1"] = datetime.date(2001, 1, 2)
    df = df._consolidate()

    # this fails because we have a date in the object block......
    msg = "|".join(
        [
            re.escape(
                "Cannot serialize the column [datetime1]\nbecause its data "
                "contents are not [string] but [date] object dtype"
            ),
            re.escape("[date] is not implemented as a table column"),
        ]
    )
    with pytest.raises(TypeError, match=msg):
        temp_hdfstore.append("df_unimplemented", df)


def test_invalid_terms(temp_hdfstore):
    # [FUNCTION DOC]
    # Puts a table-format frame. Asserts a type error when constructing a query Term with no argument (missing where), a value error for an expression that is not a valid condition (an index element reference), and a syntax error for a dangling comparison.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    df["string"] = "foo"
    df.loc[df.index[0:4], "string"] = "bar"

    temp_hdfstore.put("df", df, format="table")

    # some invalid terms
    msg = re.escape("__init__() missing 1 required positional argument: 'where'")
    with pytest.raises(TypeError, match=msg):
        Term()

    # more invalid
    msg = re.escape(
        "cannot process expression [df.index[3]], "
        "[2000-01-06 00:00:00] is not a valid condition"
    )
    with pytest.raises(ValueError, match=msg):
        temp_hdfstore.select("df", "df.index[3]")

    msg = "invalid syntax"
    with pytest.raises(SyntaxError, match=msg):
        temp_hdfstore.select("df", "index>")


def test_invalid_terms_from_docs(temp_h5_path):
    # [FUNCTION DOC]
    # Documentation examples of valid queries (combined index and columns condition, and a data-column or-condition) on a table with all columns as data columns; passes if no exception is raised (no assertions).
    # from the docs
    dfq = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=list("ABCD"),
        index=date_range("20130101", periods=10, unit="ns"),
    )
    dfq.to_hdf(temp_h5_path, key="dfq", format="table", data_columns=True)

    # check ok
    read_hdf(
        temp_h5_path, "dfq", where="index>Timestamp('20130104') & columns=['A', 'B']"
    )
    read_hdf(temp_h5_path, "dfq", where="A>0 or C>0")


def test_invalid_terms_reference(temp_h5_path):
    # [FUNCTION DOC]
    # Table written without data columns; asserts a value error explaining that a where expression contains an invalid variable reference, because only index and columns are defined references.
    # catch the invalid reference
    dfq = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=list("ABCD"),
        index=date_range("20130101", periods=10, unit="ns"),
    )
    dfq.to_hdf(temp_h5_path, key="dfq", format="table")

    msg = (
        r"The passed where expression: A>0 or C>0\n\s*"
        r"contains an invalid variable reference\n\s*"
        r"all of the variable references must be a reference to\n\s*"
        r"an axis \(e.g. 'index' or 'columns'\), or a data_column\n\s*"
        r"The currently defined references are: index,columns\n"
    )
    with pytest.raises(ValueError, match=msg):
        read_hdf(temp_h5_path, "dfq", where="A>0 or C>0")


def test_append_with_diff_col_name_types_raises_value_error(temp_hdfstore):
    # [FUNCTION DOC]
    # Appends a frame with integer column labels, then asserts that appending frames whose column labels are strings, integer tuples, mixed tuples or tuples containing a type raises a value error that the structure cannot match the existing table.
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
    df2 = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
    df3 = DataFrame({(1, 2): np.random.default_rng(2).standard_normal(10)})
    df4 = DataFrame({("1", 2): np.random.default_rng(2).standard_normal(10)})
    df5 = DataFrame({("1", 2, object): np.random.default_rng(2).standard_normal(10)})

    name = "df_diff_valerror"
    temp_hdfstore.append(name, df)

    for d in (df2, df3, df4, df5):
        msg = re.escape(
            "cannot match existing table structure for [0] on appending data"
        )
        with pytest.raises(ValueError, match=msg):
            temp_hdfstore.append(name, d)


def test_invalid_complib(temp_h5_path):
    # [FUNCTION DOC]
    # Writing with an unknown compression library name raises a value error saying only the supported libraries are allowed.
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    msg = r"complib only supports \[.*\] compression."
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(temp_h5_path, key="df", complib="foolib")


@pytest.mark.parametrize(
    "idx",
    [
        date_range("2019", freq="D", periods=3, tz="UTC", unit="ns"),
        CategoricalIndex(list("abc")),
    ],
)
def test_to_hdf_multiindex_extension_dtype(idx, temp_h5_path):
    # [FUNCTION DOC]
    # GH 7775. Parametrized over a UTC datetime index and a categorical index used as both levels of a multi-index; saving raises a not-implemented error saying saving such a multi-index is unsupported.
    # GH 7775
    mi = MultiIndex.from_arrays([idx, idx])
    df = DataFrame(0, index=mi, columns=["a"])
    with pytest.raises(NotImplementedError, match="Saving a MultiIndex"):
        df.to_hdf(temp_h5_path, key="df")


def test_unsuppored_hdf_file_error(datapath):
    # [FUNCTION DOC]
    # GH 9539. Reading the bundled legacy dataset that is not a pandas table raises a value error that datasets are incompatible with pandas types, not a table, or none found.
    # GH 9539
    data_path = datapath("io", "data", "legacy_hdf/incompatible_dataset.h5")
    message = (
        r"Dataset\(s\) incompatible with Pandas data types, "
        "not table, or no datasets found in HDF5 file."
    )

    with pytest.raises(ValueError, match=message):
        read_hdf(data_path)


def test_read_hdf_errors(temp_h5_path):
    # [FUNCTION DOC]
    # Reading a nonexistent file name raises an OS error saying the file does not exist. Reading from a store object that has been closed raises an OS error that the store must be open for reading.
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    msg = r"File [\S]* does not exist"
    with pytest.raises(OSError, match=msg):
        read_hdf(f"{uuid.uuid4()}.h5", "key")

    df.to_hdf(temp_h5_path, key="df")
    store = HDFStore(temp_h5_path, mode="r")
    store.close()

    msg = "The HDFStore must be open for reading."
    with pytest.raises(OSError, match=msg):
        read_hdf(store, "df")


def test_read_hdf_generic_buffer_errors():
    # [FUNCTION DOC]
    # Reading from an in-memory bytes buffer raises a not-implemented error about generic buffers.
    msg = "Support for generic buffers has not been implemented."
    with pytest.raises(NotImplementedError, match=msg):
        read_hdf(BytesIO(b""), "df")


@pytest.mark.parametrize("bad_version", [(1, 2), (1,), [], "12", "123"])
def test_maybe_adjust_name_bad_version_raises(bad_version):
    # [FUNCTION DOC]
    # Parametrized over malformed version arguments (too short sequences, empty list, strings). The private name-adjustment function raises a value error saying the version must be a sequence of three integers.
    msg = "Version is incorrect, expected sequence of 3 integers"
    with pytest.raises(ValueError, match=msg):
        _maybe_adjust_name("values_block_0", version=bad_version)
