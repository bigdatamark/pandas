"""test parquet compat"""

# [MODULE DOC]
# Tests for Parquet I/O through the generic read and write functions with the pyarrow
# and fastparquet engines, engine selection, round trips for many dtypes and index
# shapes, partitioning, storage/filesystem arguments, S3 and URL access, and cross-
# engine compatibility. Import-time side effects: tries to import pyarrow and
# fastparquet, setting module flags for their availability and, for fastparquet,
# version-comparison flags used by expected-failure marks. Module-wide marks ignore two
# deprecation warnings (data attribute, block manager).

import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
    pa_version_under15p0,
    pa_version_under17p0,
    pa_version_under19p0,
    pa_version_under20p0,
)

import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version

from pandas.io.parquet import (
    FastParquetImpl,
    PyArrowImpl,
    get_engine,
    read_parquet,
    to_parquet,
)

try:
    import pyarrow

    _HAVE_PYARROW = True
except ImportError:
    _HAVE_PYARROW = False

try:
    import fastparquet

    _HAVE_FASTPARQUET = True
    _fp_version_lt_2025 = Version(fastparquet.__version__) < Version("2025.12.0")
    _fp_version_lt_2026_5 = Version(fastparquet.__version__) < Version("2026.5.0")
except ImportError:
    _HAVE_FASTPARQUET = False
    _fp_version_lt_2025 = False
    _fp_version_lt_2026_5 = False


pytestmark = [
    pytest.mark.filterwarnings("ignore:DataFrame._data is deprecated:FutureWarning"),
    pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
    ),
]


# setup engines & skips
@pytest.fixture(
    params=[
        pytest.param(
            "fastparquet",
            marks=[
                pytest.mark.skipif(
                    not _HAVE_FASTPARQUET,
                    reason="fastparquet is not installed",
                ),
                pytest.mark.xfail(
                    using_string_dtype(),
                    reason="TODO(infer_string) fastparquet",
                    strict=False,
                ),
            ],
        ),
        pytest.param(
            "pyarrow",
            marks=pytest.mark.skipif(
                not _HAVE_PYARROW, reason="pyarrow is not installed"
            ),
        ),
    ]
)
def engine(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture over the engine names fastparquet and pyarrow. Each
    # param is skipped when its library is missing; fastparquet is marked as a non-
    # strict expected failure under the future string dtype mode. Returns the engine
    # name.
    return request.param


@pytest.fixture
def pa():
    # [FUNCTION DOC]
    # HELPER: fixture returning the string pyarrow; skips the test when pyarrow is not
    # installed.
    if not _HAVE_PYARROW:
        pytest.skip("pyarrow is not installed")
    return "pyarrow"


@pytest.fixture
def fp(request):
    # [FUNCTION DOC]
    # HELPER: fixture returning the string fastparquet; skips when fastparquet is
    # missing and, under the future string dtype mode, applies a non-strict expected-
    # failure mark to the requesting test.
    if not _HAVE_FASTPARQUET:
        pytest.skip("fastparquet is not installed")
    if using_string_dtype():
        request.applymarker(
            pytest.mark.xfail(reason="TODO(infer_string) fastparquet", strict=False)
        )
    return "fastparquet"


@pytest.fixture
def df_compat():
    # [FUNCTION DOC]
    # HELPER: fixture returning a small frame: integer column A with 1,2,3 and string
    # column B filled with foo, with a plain index for column labels.
    return pd.DataFrame({"A": [1, 2, 3], "B": "foo"}, columns=pd.Index(["A", "B"]))


@pytest.fixture
def df_cross_compat():
    # [FUNCTION DOC]
    # HELPER: fixture returning a frame with string, integer, float, bool and daily
    # datetime columns; types that fail across engines (unsigned small ints, tz-aware
    # and nanosecond-frequency datetimes) are left commented out.
    df = pd.DataFrame(
        {
            "a": list("abc"),
            "b": list(range(1, 4)),
            # 'c': np.arange(3, 6).astype('u1'),
            "d": np.arange(4.0, 7.0, dtype="float64"),
            "e": [True, False, True],
            "f": pd.date_range("20130101", periods=3),
            # 'g': pd.date_range('20130101', periods=3,
            #                    tz='US/Eastern'),
            # 'h': pd.date_range('20130101', periods=3, freq='ns')
        }
    )
    return df


@pytest.fixture
def df_full():
    # [FUNCTION DOC]
    # HELPER: fixture returning a frame covering many dtypes: strings (also with NaN and
    # None), bytes, unicode strings, ints, small unsigned ints, floats (also with NaN),
    # bools, nanosecond datetimes and datetimes with NaT.
    return pd.DataFrame(
        {
            "string": list("abc"),
            "string_with_nan": ["a", np.nan, "c"],
            "string_with_none": ["a", None, "c"],
            "bytes": [b"foo", b"bar", b"baz"],
            "unicode": ["foo", "bar", "baz"],
            "int": list(range(1, 4)),
            "uint": np.arange(3, 6).astype("u1"),
            "float": np.arange(4.0, 7.0, dtype="float64"),
            "float_with_nan": [2.0, np.nan, 3.0],
            "bool": [True, False, True],
            "datetime": pd.date_range("20130101", periods=3, unit="ns"),
            "datetime_with_nat": [
                pd.Timestamp("20130101"),
                pd.NaT,
                pd.Timestamp("20130103"),
            ],
        }
    )


@pytest.fixture(
    params=[
        datetime.datetime.now(datetime.UTC),
        datetime.datetime.now(datetime.timezone.min),
        datetime.datetime.now(datetime.timezone.max),
        datetime.datetime.strptime("2019-01-04T16:41:24+0200", "%Y-%m-%dT%H:%M:%S%z"),
        datetime.datetime.strptime("2019-01-04T16:41:24+0215", "%Y-%m-%dT%H:%M:%S%z"),
        datetime.datetime.strptime("2019-01-04T16:41:24-0200", "%Y-%m-%dT%H:%M:%S%z"),
        datetime.datetime.strptime("2019-01-04T16:41:24-0215", "%Y-%m-%dT%H:%M:%S%z"),
    ]
)
def timezone_aware_date_list(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture of seven timezone-aware datetime values: now in UTC,
    # now with the minimum and maximum timezone offsets, and four fixed dates with
    # offsets of plus/minus 2:00 and 2:15. Returns one value. Evaluated at import so the
    # now-based values are fixed at collection.
    return request.param


def check_round_trip(
    df,
    temp_file,
    engine=None,
    path=None,
    write_kwargs=None,
    read_kwargs=None,
    expected=None,
    check_names=True,
    check_like=False,
    check_dtype=True,
    repeat=2,
):
    # [FUNCTION DOC]
    # HELPER: writes a frame to Parquet and reads it back, asserting equality.
    # Takes the frame, a pathlib path (temp_file; raises a value error if it is not a
    # path object), optional engine name, optional explicit path (default is the temp
    # path), optional write and read keyword dictionaries (writing defaults to no
    # compression), an optional expected frame (defaults to the input), the check_names,
    # check_like and check_dtype flags forwarded to the frame comparison, and repeat
    # (number of write/read cycles, default 2).
    # When an engine is given it is injected into both keyword dictionaries (mutating
    # the caller's dictionaries). Each cycle writes, reads, sets the string_with_nan
    # entry of the expected frame's second row to None when that column exists (mutating
    # the expected frame), and asserts frame equality.
    # KNOWN ISSUE: the docstring describes check_names as a closed set of column names,
    # but it is a boolean flag passed to the equality assertion; the docstring also
    # omits temp_file and check_dtype and does not give the default of repeat.
    # KNOWN ISSUE: both branches of the final path check call the comparison
    # identically, differing only in whether the path is first defaulted to the temp
    # file.
    """Verify parquet serializer and deserializer produce the same results.

    Performs a pandas to disk and disk to pandas round trip,
    then compares the 2 resulting DataFrames to verify equality.

    Parameters
    ----------
    df: Dataframe
    engine: str, optional
        'pyarrow' or 'fastparquet'
    path: str, optional
    write_kwargs: dict of str:str, optional
    read_kwargs: dict of str:str, optional
    expected: DataFrame, optional
        Expected deserialization result, otherwise will be equal to `df`
    check_names: list of str, optional
        Closed set of column names to be compared
    check_like: bool, optional
        If True, ignore the order of index & columns.
    repeat: int, optional
        How many times to repeat the test
    """
    if not isinstance(temp_file, pathlib.Path):
        raise ValueError("temp_file must be a pathlib.Path")
    write_kwargs = write_kwargs or {"compression": None}
    read_kwargs = read_kwargs or {}

    if expected is None:
        expected = df

    if engine:
        write_kwargs["engine"] = engine
        read_kwargs["engine"] = engine

    def compare(repeat):
        # [FUNCTION DOC]
        # HELPER: inner function of check_round_trip that, repeat times, writes the
        # frame to the path with the write keywords, reads with the read keywords,
        # patches the expected frame for the NaN string column, and asserts frame
        # equality using the enclosing flags. Takes the repeat count; returns nothing.
        for _ in range(repeat):
            df.to_parquet(path, **write_kwargs)
            actual = read_parquet(path, **read_kwargs)

            if "string_with_nan" in expected:
                expected.loc[1, "string_with_nan"] = None
            tm.assert_frame_equal(
                actual,
                expected,
                check_names=check_names,
                check_like=check_like,
                check_dtype=check_dtype,
            )

    if path is None:
        path = temp_file
        compare(repeat)
    else:
        compare(repeat)


def check_partition_names(path, expected):
    # [FUNCTION DOC]
    # HELPER: asserts the partition column names of a dataset on disk. Takes the dataset
    # path and the expected list of names; opens it as a hive-partitioned pyarrow
    # dataset and asserts the partitioning schema names equal the expected list.
    """Check partitions of a parquet file are as expected.

    Parameters
    ----------
    path: str
        Path of the dataset.
    expected: iterable of str
        Expected partition names.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, partitioning="hive")
    assert dataset.partitioning.schema.names == expected


def test_invalid_engine(df_compat, temp_file):
    # [FUNCTION DOC]
    # Calling the round-trip helper with an unknown engine name must raise a value error
    # stating the engine must be one of pyarrow or fastparquet.
    msg = "engine must be one of 'pyarrow', 'fastparquet'"
    with pytest.raises(ValueError, match=msg):
        check_round_trip(df_compat, temp_file, "foo", "bar")


def test_options_py(df_compat, pa, using_infer_string, temp_file):
    # [FUNCTION DOC]
    # Setting the default parquet engine option to pyarrow makes a round trip without an
    # explicit engine work; when the infer-string mode is on (and pyarrow is new enough)
    # the column labels are converted to the string dtype first.
    # use the set option
    if using_infer_string and not pa_version_under19p0:
        df_compat.columns = df_compat.columns.astype("str")

    with pd.option_context("io.parquet.engine", "pyarrow"):
        check_round_trip(df_compat, temp_file)


def test_options_fp(df_compat, fp, temp_file):
    # [FUNCTION DOC]
    # Same as above with the default engine option set to fastparquet.
    # use the set option

    with pd.option_context("io.parquet.engine", "fastparquet"):
        check_round_trip(df_compat, temp_file)


def test_options_auto(df_compat, fp, pa, temp_file):
    # [FUNCTION DOC]
    # With both engines installed, the engine option auto allows a round trip.
    # use the set option

    with pd.option_context("io.parquet.engine", "auto"):
        check_round_trip(df_compat, temp_file)


def test_options_get_engine(fp, pa):
    # [FUNCTION DOC]
    # With both engines installed, asking for the engine implementation returns the
    # pyarrow implementation class or the fastparquet implementation class for explicit
    # names regardless of the option, and for auto returns pyarrow unless the option is
    # fastparquet, in which case it returns fastparquet.
    assert isinstance(get_engine("pyarrow"), PyArrowImpl)
    assert isinstance(get_engine("fastparquet"), FastParquetImpl)

    with pd.option_context("io.parquet.engine", "pyarrow"):
        assert isinstance(get_engine("auto"), PyArrowImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)

    with pd.option_context("io.parquet.engine", "fastparquet"):
        assert isinstance(get_engine("auto"), FastParquetImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)

    with pd.option_context("io.parquet.engine", "auto"):
        assert isinstance(get_engine("auto"), PyArrowImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)


def test_get_engine_auto_error_message():
    # [FUNCTION DOC]
    # Determines whether pyarrow and fastparquet are installed and whether their
    # versions meet the minimum. Only if neither engine is usable it checks that asking
    # for the auto engine raises an import error: with a message about the required
    # minimum version if that library is installed too old, otherwise the unable-to-
    # find-engine message for pyarrow; and then the same for fastparquet (version
    # message or an install hint). Otherwise the test does nothing.
    # Expect different error messages from get_engine(engine="auto")
    # if engines aren't installed vs. are installed but bad version
    from pandas.compat._optional import VERSIONS

    # Do we have engines installed, but a bad version of them?
    pa_min_ver = VERSIONS.get("pyarrow")
    fp_min_ver = VERSIONS.get("fastparquet")
    have_pa_bad_version = (
        False
        if not _HAVE_PYARROW
        else Version(pyarrow.__version__) < Version(pa_min_ver)
    )
    have_fp_bad_version = (
        False
        if not _HAVE_FASTPARQUET
        else Version(fastparquet.__version__) < Version(fp_min_ver)
    )
    # Do we have usable engines installed?
    have_usable_pa = _HAVE_PYARROW and not have_pa_bad_version
    have_usable_fp = _HAVE_FASTPARQUET and not have_fp_bad_version

    if not have_usable_pa and not have_usable_fp:
        # No usable engines found.
        if have_pa_bad_version:
            match = f"Pandas requires version .{pa_min_ver}. or newer of .pyarrow."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")
        else:
            match = "Unable to find a usable engine; tried using: 'pyarrow'"
            with pytest.raises(ImportError, match=match):
                get_engine("auto")

        if have_fp_bad_version:
            match = f"Pandas requires version .{fp_min_ver}. or newer of .fastparquet."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")
        else:
            match = "Use pip or conda to install the fastparquet package"
            with pytest.raises(ImportError, match=match):
                get_engine("auto")


def test_cross_engine_pa_fp(df_cross_compat, pa, fp, temp_file):
    # [FUNCTION DOC]
    # Writes with pyarrow and reads with fastparquet, fully and with a column subset;
    # both equal the expected frame.
    # cross-compat with differing reading/writing engines

    df = df_cross_compat
    df.to_parquet(temp_file, engine=pa, compression=None)

    result = read_parquet(temp_file, engine=fp)
    tm.assert_frame_equal(result, df)

    result = read_parquet(temp_file, engine=fp, columns=["a", "d"])
    tm.assert_frame_equal(result, df[["a", "d"]])


@pytest.mark.xfail(
    using_string_dtype()
    and _HAVE_PYARROW
    and not _fp_version_lt_2025
    and _fp_version_lt_2026_5,
    reason="fastparquet >= 2025.12.0, < 2026.5.0 can't write ArrowStringArray",
)
def test_cross_engine_fp_pa(df_cross_compat, pa, fp, temp_file):
    # [FUNCTION DOC]
    # Expected to fail when the infer-string mode is on, pyarrow is present, and
    # fastparquet is in the range from 2025.12.0 up to (not including) 2026.5.0 (cannot
    # write the Arrow string array). Writes with fastparquet and reads with pyarrow,
    # fully and with a column subset.
    # cross-compat with differing reading/writing engines
    df = df_cross_compat

    df.to_parquet(temp_file, engine=fp, compression=None)

    result = read_parquet(temp_file, engine=pa)
    tm.assert_frame_equal(result, df)

    result = read_parquet(temp_file, engine=pa, columns=["a", "d"])
    tm.assert_frame_equal(result, df[["a", "d"]])


class Base:
    # [CLASS DOC]
    # HELPER: base class for engine-specific test classes providing the error-assertion
    # helpers below.
    def check_error_on_write(self, df, engine, exc, err_msg, temp_file_path):
        # [FUNCTION DOC]
        # HELPER: asserts that writing a frame with the given engine to the given path
        # (no compression) raises the given exception with a message matching the regex.
        # check that we are raising the exception on writing
        with pytest.raises(exc, match=err_msg):
            to_parquet(df, temp_file_path, engine, compression=None)

    def check_external_error_on_write(self, df, engine, exc, temp_file_path):
        # [FUNCTION DOC]
        # HELPER: asserts that writing the frame with the given engine to the path
        # raises an exception from an external library of the given class.
        # check that an external library is raising the exception on writing
        with tm.external_error_raised(exc):
            to_parquet(df, temp_file_path, engine, compression=None)


class TestBasic(Base):
    # [CLASS DOC]
    # Engine-parametrized tests of common parquet behaviors: non-frame errors, columns,
    # compression, filters, index handling, multi-level columns, dtype backend and
    # reading from URL.
    def test_error(self, engine, temp_file):
        # [FUNCTION DOC]
        # Writing a series, int, string, timestamp or array must raise a value error
        # stating that parquet writing only supports data frames.
        for obj in [
            pd.Series([1, 2, 3]),
            1,
            "foo",
            pd.Timestamp("20130101"),
            np.array([1, 2, 3]),
        ]:
            msg = "to_parquet only supports IO with DataFrames"
            self.check_error_on_write(obj, engine, ValueError, msg, temp_file)

    def test_columns_dtypes(self, engine, temp_file):
        # [FUNCTION DOC]
        # A frame with string column labels round-trips.
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})

        # unicode
        df.columns = ["foo", "bar"]
        check_round_trip(df, temp_file, engine)

    @pytest.mark.parametrize("compression", [None, "gzip", "snappy", "brotli"])
    def test_compression(self, engine, compression, temp_file):
        # [FUNCTION DOC]
        # Parametrized over no compression, gzip, snappy, brotli. Round trip with that
        # compression.
        df = pd.DataFrame({"A": [1, 2, 3]})
        check_round_trip(
            df, temp_file, engine, write_kwargs={"compression": compression}
        )

    def test_read_columns(self, engine, temp_file):
        # [FUNCTION DOC]
        # GH 18154. Reading with a column subset returns just that column.
        # GH18154
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})

        expected = pd.DataFrame({"string": list("abc")})
        check_round_trip(
            df,
            temp_file,
            engine,
            expected=expected,
            read_kwargs={"columns": ["string"]},
        )

    def test_read_filters(self, engine, tmp_path):
        # [FUNCTION DOC]
        # Writes a frame partitioned by the part column into a directory, then reads
        # with a filter selecting partition a and only column int; result is the two
        # matching rows. A single cycle only.
        df = pd.DataFrame(
            {
                "int": list(range(4)),
                "part": list("aabb"),
            }
        )

        expected = pd.DataFrame({"int": [0, 1]})
        check_round_trip(
            df,
            tmp_path,
            engine,
            expected=expected,
            write_kwargs={"partition_cols": ["part"]},
            read_kwargs={"filters": [("part", "==", "a")], "columns": ["int"]},
            repeat=1,
        )

    def test_write_index(self, temp_file):
        # [FUNCTION DOC]
        # Needs pyarrow. Round-trips with the default index, then non-default indexes
        # (ints, datetimes with frequency removed, strings, other ints), then a named
        # index.
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"A": [1, 2, 3]})
        check_round_trip(df, temp_file, "pyarrow")

        indexes = [
            [2, 3, 4],
            pd.date_range("20130101", periods=3, unit="ns"),
            list("abc"),
            [1, 3, 4],
        ]
        # non-default index
        for index in indexes:
            df.index = index
            if isinstance(index, pd.DatetimeIndex):
                df.index = df.index._with_freq(None)  # freq doesn't round-trip
            check_round_trip(df, temp_file, "pyarrow")

        # index with meta-data
        df.index = [0, 1, 2]
        df.index.name = "foo"
        check_round_trip(df, temp_file, "pyarrow")

    def test_write_multiindex(self, pa, temp_file):
        # [FUNCTION DOC]
        # Needs pyarrow (the comment says fastparquet does not support it). A row multi-
        # index round-trips.
        # Not supported in fastparquet as of 0.1.3 or older pyarrow version
        engine = pa

        df = pd.DataFrame({"A": [1, 2, 3]})
        index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
        df.index = index
        check_round_trip(df, temp_file, engine)

    def test_multiindex_with_columns(self, pa, temp_file):
        # [FUNCTION DOC]
        # Needs pyarrow. Frame with a two-level index (with and without level names)
        # round-trips, also with a column subset read.
        engine = pa
        dates = pd.date_range("01-Jan-2018", "01-Dec-2018", freq="MS", unit="ns")
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((2 * len(dates), 3)),
            columns=list("ABC"),
        )
        index1 = pd.MultiIndex.from_product(
            [["Level1", "Level2"], dates], names=["level", "date"]
        )
        index2 = index1.copy(names=None)
        for index in [index1, index2]:
            df.index = index

            check_round_trip(df, temp_file, engine)
            check_round_trip(
                df,
                temp_file,
                engine,
                read_kwargs={"columns": ["A", "B"]},
                expected=df[["A", "B"]],
            )

    def test_write_ignoring_index(self, engine, temp_file):
        # [FUNCTION DOC]
        # ENH 20768. Writing with the index flag off omits the index in the file:
        # reading gives a default integer index, for a default index, a custom string
        # index and a multi-index.
        # ENH 20768
        # Ensure index=False omits the index from the written Parquet file.
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["q", "r", "s"]})

        write_kwargs = {"compression": None, "index": False}

        # Because we're dropping the index, we expect the loaded dataframe to
        # have the default integer index.
        expected = df.reset_index(drop=True)

        check_round_trip(
            df, temp_file, engine, write_kwargs=write_kwargs, expected=expected
        )

        # Ignore custom index
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": ["q", "r", "s"]}, index=["zyx", "wvu", "tsr"]
        )

        check_round_trip(
            df, temp_file, engine, write_kwargs=write_kwargs, expected=expected
        )

        # Ignore multi-indexes as well.
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        df = pd.DataFrame(
            {"one": list(range(8)), "two": [-i for i in range(8)]}, index=arrays
        )

        expected = df.reset_index(drop=True)
        check_round_trip(
            df, temp_file, engine, write_kwargs=write_kwargs, expected=expected
        )

    def test_write_column_multiindex(self, engine, temp_file):
        # [FUNCTION DOC]
        # Multi-level columns with non-string labels: fastparquet raises a type error
        # that column names must be strings; pyarrow round-trips.
        # Not able to write column multi-indexes with non-string column names.
        mi_columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)), columns=mi_columns
        )

        if engine == "fastparquet":
            self.check_error_on_write(
                df, engine, TypeError, "Column name must be a string", temp_file
            )
        elif engine == "pyarrow":
            check_round_trip(df, temp_file, engine)

    def test_write_column_multiindex_nonstring(self, engine, temp_file):
        # [FUNCTION DOC]
        # GH 34777. Multi-level columns with an integer level: fastparquet raises a
        # value error about the column name; pyarrow round-trips.
        # GH #34777

        # Not able to write column multi-indexes with non-string column names
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            [1, 2, 1, 2, 1, 2, 1, 2],
        ]
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 8)), columns=arrays
        )
        df.columns.names = ["Level1", "Level2"]
        if engine == "fastparquet":
            self.check_error_on_write(df, engine, ValueError, "Column name", temp_file)
        elif engine == "pyarrow":
            check_round_trip(df, temp_file, engine)

    def test_write_column_multiindex_string(self, pa, temp_file):
        # [FUNCTION DOC]
        # GH 34777. Needs pyarrow. Multi-level string columns with named levels round-
        # trip.
        # GH #34777
        # Not supported in fastparquet as of 0.1.3
        engine = pa

        # Write column multi-indexes with string column names
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 8)), columns=arrays
        )
        df.columns.names = ["ColLevel1", "ColLevel2"]

        check_round_trip(df, temp_file, engine)

    def test_write_column_index_string(self, pa, temp_file):
        # [FUNCTION DOC]
        # GH 34777. Needs pyarrow. A named string column index round-trips.
        # GH #34777
        # Not supported in fastparquet as of 0.1.3
        engine = pa

        # Write column indexes with string column names
        arrays = ["bar", "baz", "foo", "qux"]
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)), columns=arrays
        )
        df.columns.name = "StringCol"

        check_round_trip(df, temp_file, engine)

    def test_write_column_index_nonstring(self, engine, temp_file):
        # [FUNCTION DOC]
        # GH 34777. Integer column labels with a named columns index: fastparquet raises
        # a type error about column names; pyarrow round-trips.
        # GH #34777

        # Write column indexes with string column names
        arrays = [1, 2, 3, 4]
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)), columns=arrays
        )
        df.columns.name = "NonStringCol"
        if engine == "fastparquet":
            self.check_error_on_write(
                df, engine, TypeError, "Column name must be a string", temp_file
            )
        else:
            check_round_trip(df, temp_file, engine)

    def test_dtype_backend(self, engine, request, temp_file):
        # [FUNCTION DOC]
        # Writes a table via pyarrow with nulls and several types, reads with the
        # default and the nullable-numpy backends; the default read turns the nullable
        # int column into float64; the nullable read equals a frame of nullable dtypes
        # (Int64, UInt8, string, boolean, Float32, Float64). For fastparquet the test is
        # an expected failure and the string column is dropped from the comparison.
        pq = pytest.importorskip("pyarrow.parquet")

        if engine == "fastparquet":
            # We are manually disabling fastparquet's
            # nullable dtype support pending discussion
            mark = pytest.mark.xfail(
                reason="Fastparquet nullable dtype support is disabled"
            )
            request.applymarker(mark)

        table = pyarrow.table(
            {
                "a": pyarrow.array([1, 2, 3, None], "int64"),
                "b": pyarrow.array([1, 2, 3, None], "uint8"),
                "c": pyarrow.array(["a", "b", "c", None]),
                "d": pyarrow.array([True, False, True, None]),
                # Test that nullable dtypes used even in absence of nulls
                "e": pyarrow.array([1, 2, 3, 4], "int64"),
                # GH 45694
                "f": pyarrow.array([1.0, 2.0, 3.0, None], "float32"),
                "g": pyarrow.array([1.0, 2.0, 3.0, None], "float64"),
            }
        )
        # write manually with pyarrow to write integers
        pq.write_table(table, temp_file)
        result1 = read_parquet(temp_file, engine=engine)
        result2 = read_parquet(temp_file, engine=engine, dtype_backend="numpy_nullable")

        assert result1["a"].dtype == np.dtype("float64")
        expected = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3, None], dtype="Int64"),
                "b": pd.array([1, 2, 3, None], dtype="UInt8"),
                "c": pd.array(["a", "b", "c", None], dtype="string"),
                "d": pd.array([True, False, True, None], dtype="boolean"),
                "e": pd.array([1, 2, 3, 4], dtype="Int64"),
                "f": pd.array([1.0, 2.0, 3.0, None], dtype="Float32"),
                "g": pd.array([1.0, 2.0, 3.0, None], dtype="Float64"),
            }
        )
        if engine == "fastparquet":
            # Fastparquet doesn't support string columns yet
            # Only int and boolean
            result2 = result2.drop("c", axis=1)
            expected = expected.drop("c", axis=1)
        tm.assert_frame_equal(result2, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            "Int64",
            "UInt8",
            "boolean",
            "object",
            "datetime64[ns, UTC]",
            "float",
            "period[D]",
            "Float64",
            "string",
        ],
    )
    def test_read_empty_array(self, pa, dtype, temp_file):
        # [FUNCTION DOC]
        # GH 41241. Parametrized over nine dtype names. Writes an empty column of that
        # dtype with pyarrow and reads with the nullable backend; the plain float case
        # is expected to come back as Float64, others equal the input.
        # GH #41241
        df = pd.DataFrame(
            {
                "value": pd.array([], dtype=dtype),
            }
        )
        pytest.importorskip("pyarrow", "11.0.0")
        # GH 45694
        expected = None
        if dtype == "float":
            expected = pd.DataFrame(
                {
                    "value": pd.array([], dtype="Float64"),
                }
            )
        check_round_trip(
            df,
            temp_file,
            pa,
            read_kwargs={"dtype_backend": "numpy_nullable"},
            expected=expected,
        )

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_parquet_read_from_url(self, httpserver, datapath, df_compat, engine):
        # [FUNCTION DOC]
        # Network, single-cpu. Serves a simple.parquet sample file from a local server
        # and reads it by URL; equals the compat frame, with column labels as object
        # when pyarrow is old.
        # KNOWN ISSUE: the skip condition compares the engine to auto, but the fixture
        # only ever supplies fastparquet or pyarrow, so the condition is always true.
        if engine != "auto":
            pytest.importorskip(engine)
        with open(datapath("io", "data", "parquet", "simple.parquet"), mode="rb") as f:
            httpserver.serve_content(content=f.read())
            df = read_parquet(httpserver.url, engine=engine)

        expected = df_compat
        if pa_version_under19p0:
            expected.columns = expected.columns.astype(object)
        tm.assert_frame_equal(df, expected)


class TestParquetPyArrow(Base):
    # [CLASS DOC]
    # Tests specific to the pyarrow engine (fixtures request pa). Inherits the base
    # error helpers.
    def test_basic(self, pa, df_full, temp_file):
        # [FUNCTION DOC]
        # Adds a tz-aware datetime column and a bool column with None to the full frame
        # and round-trips with pyarrow.
        df = df_full
        pytest.importorskip("pyarrow", "11.0.0")

        # additional supported types for pyarrow
        dti = pd.date_range("20130101", periods=3, tz="Europe/Brussels")
        dti = dti._with_freq(None)  # freq doesn't round-trip
        df["datetime_tz"] = dti
        df["bool_with_none"] = [True, None, True]

        check_round_trip(df, temp_file, pa)

    def test_basic_subset_columns(self, pa, df_full, temp_file):
        # [FUNCTION DOC]
        # GH 18628. Round trip reading only two columns.
        # GH18628

        df = df_full
        # additional supported types for pyarrow
        df["datetime_tz"] = pd.date_range("20130101", periods=3, tz="Europe/Brussels")

        check_round_trip(
            df,
            temp_file,
            pa,
            expected=df[["string", "int"]],
            read_kwargs={"columns": ["string", "int"]},
        )

    def test_to_bytes_without_path_or_buf_provided(self, pa, df_full):
        # [FUNCTION DOC]
        # GH 37105. Writing without a destination returns a bytes object; reading it
        # from a buffer gives the frame with the NaN-string patched to None and the NaT
        # datetime column in microsecond unit.
        # GH 37105
        buf_bytes = df_full.to_parquet(engine=pa)
        assert isinstance(buf_bytes, bytes)

        buf_stream = BytesIO(buf_bytes)
        res = read_parquet(buf_stream)

        expected = df_full.copy()
        expected.loc[1, "string_with_nan"] = None
        expected["datetime_with_nat"] = expected["datetime_with_nat"].astype("M8[us]")
        tm.assert_frame_equal(res, expected)

    def test_duplicate_columns(self, pa, temp_file):
        # [FUNCTION DOC]
        # Writing a frame with duplicate column labels must raise a value error saying
        # duplicate column names were found.
        # not currently able to handle duplicate columns
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list("aaa")).copy()
        self.check_error_on_write(
            df, pa, ValueError, "Duplicate column names found", temp_file
        )

    def test_timedelta(self, pa, temp_file):
        # [FUNCTION DOC]
        # Timedelta column round-trips.
        df = pd.DataFrame({"a": pd.timedelta_range("1 day", periods=3)})
        check_round_trip(df, temp_file, pa)

    def test_unsupported(self, pa, temp_file):
        # [FUNCTION DOC]
        # Mixed object column cannot be written: expects a pyarrow exception.
        # mixed python objects
        df = pd.DataFrame({"a": ["a", 1, 2.0]})
        # pyarrow 0.11 raises ArrowTypeError
        # older pyarrows raise ArrowInvalid
        self.check_external_error_on_write(df, pa, pyarrow.ArrowException, temp_file)

    def test_unsupported_float16(self, pa, temp_file):
        # [FUNCTION DOC]
        # GH 44847, 44914. For pyarrow older than 15 writing float16 raises a pyarrow
        # exception; newer versions round trip.
        # #44847, #44914
        # Not able to write float 16 column using pyarrow.
        data = np.arange(2, 10, dtype=np.float16)
        df = pd.DataFrame(data=data, columns=["fp16"])
        if pa_version_under15p0:
            self.check_external_error_on_write(
                df, pa, pyarrow.ArrowException, temp_file
            )
        else:
            check_round_trip(df, temp_file, pa)

    @pytest.mark.xfail(
        is_platform_windows(),
        reason=(
            "PyArrow does not cleanup of partial files dumps when unsupported "
            "dtypes are passed to_parquet function in windows"
        ),
    )
    @pytest.mark.skipif(not pa_version_under15p0, reason="float16 works on 15")
    @pytest.mark.parametrize("path_type", [str, pathlib.Path])
    def test_unsupported_float16_cleanup(self, pa, path_type, temp_file):
        # [FUNCTION DOC]
        # GH 44847, 44914. Expected failure on Windows; runs only for pyarrow older than
        # 15. Parametrized over string and path object paths. Writing a float16 column
        # raises a pyarrow error and the partial file must be removed.
        # #44847, #44914
        # Not able to write float 16 column using pyarrow.
        # Tests cleanup by pyarrow in case of an error
        data = np.arange(2, 10, dtype=np.float16)
        df = pd.DataFrame(data=data, columns=["fp16"])

        path = path_type(temp_file)
        with tm.external_error_raised(pyarrow.ArrowException):
            df.to_parquet(path=path, engine=pa)
        assert not os.path.isfile(path)

    def test_categorical(self, pa, temp_file):
        # [FUNCTION DOC]
        # Categorical columns (with nulls, an unobserved category and an ordered one)
        # round-trip.
        # supported in >= 0.7.0
        df = pd.DataFrame(
            {
                "a": pd.Categorical(list("abcdef")),
                # test for null, out-of-order values, and unobserved category
                "b": pd.Categorical(
                    ["bar", "foo", "foo", "bar", None, "bar"],
                    dtype=pd.CategoricalDtype(["foo", "bar", "baz"]),
                ),
                # test for ordered flag
                "c": pd.Categorical(
                    [None, "b", "c", None, "c", "b"],
                    categories=["b", "c", "d"],
                    ordered=True,
                ),
            }
        )

        check_round_trip(df, temp_file, pa)

    @pytest.mark.single_cpu
    def test_s3_roundtrip_explicit_fs(
        self, df_compat, s3_bucket_public, s3so, pa, temp_file
    ):
        # [FUNCTION DOC]
        # Single-cpu. Needs s3fs. Round trip to a mock S3 bucket passing an explicit
        # file-system object for both write and read.
        s3fs = pytest.importorskip("s3fs")
        s3 = s3fs.S3FileSystem(**s3so)
        kw = {"filesystem": s3}
        check_round_trip(
            df_compat,
            temp_file,
            pa,
            path=f"{s3_bucket_public.name}/pyarrow.parquet",
            read_kwargs=kw,
            write_kwargs=kw,
        )

    @pytest.mark.single_cpu
    def test_s3_roundtrip(self, df_compat, s3_bucket_public, s3so, pa, temp_file):
        # [FUNCTION DOC]
        # Single-cpu. GH 19134. Round trip to the mock bucket using an s3 URL with
        # storage options.
        # GH #19134
        s3so = {"storage_options": s3so}
        check_round_trip(
            df_compat,
            temp_file,
            pa,
            path=f"s3://{s3_bucket_public.name}/pyarrow.parquet",
            read_kwargs=s3so,
            write_kwargs=s3so,
        )

    @pytest.mark.single_cpu
    @pytest.mark.parametrize("partition_col", [["A"], []])
    def test_s3_roundtrip_for_dir(
        self, df_compat, s3_bucket_public, pa, partition_col, s3so, temp_file
    ):
        # [FUNCTION DOC]
        # Single-cpu. Parametrized over partition columns [A] or none. GH 26388, 35791.
        # Writes a partitioned dataset directory to the mock bucket and reads it back;
        # when partitioned the expected column A becomes categorical of 32-bit ints.
        # Column order ignored; a single cycle.
        pytest.importorskip("s3fs")
        # GH #26388
        expected_df = df_compat.copy()

        # GH #35791
        if partition_col:
            expected_df = expected_df.astype(dict.fromkeys(partition_col, np.int32))
            partition_col_type = "category"

            expected_df[partition_col] = expected_df[partition_col].astype(
                partition_col_type
            )

        check_round_trip(
            df_compat,
            temp_file,
            pa,
            expected=expected_df,
            path=f"s3://{s3_bucket_public.name}/parquet_dir",
            read_kwargs={"storage_options": s3so},
            write_kwargs={
                "partition_cols": partition_col,
                "compression": None,
                "storage_options": s3so,
            },
            check_like=True,
            repeat=1,
        )

    def test_read_file_like_obj_support(self, df_compat, using_infer_string):
        # [FUNCTION DOC]
        # Writes to an in-memory buffer and reads from the same buffer; frame equal
        # (with string column labels when infer-string mode is on and pyarrow is new).
        pytest.importorskip("pyarrow")
        buffer = BytesIO()
        df_compat.to_parquet(buffer)
        df_from_buf = read_parquet(buffer)
        if using_infer_string and not pa_version_under19p0:
            df_compat.columns = df_compat.columns.astype("str")
        tm.assert_frame_equal(df_compat, df_from_buf)

    def test_expand_user(self, df_compat, monkeypatch):
        # [FUNCTION DOC]
        # With the home environment variables pointed at a fake user, reading and
        # writing a path starting with a tilde raise an OS error whose message contains
        # the fake user name, proving home expansion.
        pytest.importorskip("pyarrow")
        monkeypatch.setenv("HOME", "TestingUser")
        monkeypatch.setenv("USERPROFILE", "TestingUser")
        with pytest.raises(OSError, match=r".*TestingUser.*"):
            read_parquet("~/file.parquet")
        with pytest.raises(OSError, match=r".*TestingUser.*"):
            df_compat.to_parquet("~/file.parquet")

    def test_partition_cols_supported(self, tmp_path, pa, df_full):
        # [FUNCTION DOC]
        # GH 23283. Writing partitioned by the bool and int columns creates a hive
        # dataset with those partition names; reading the directory gives the same
        # shape.
        # GH #23283
        partition_cols = ["bool", "int"]
        df = df_full
        df.to_parquet(tmp_path, partition_cols=partition_cols, compression=None)
        check_partition_names(tmp_path, partition_cols)
        assert read_parquet(tmp_path).shape == df.shape

    def test_partition_cols_string(self, tmp_path, pa, df_full):
        # [FUNCTION DOC]
        # GH 27117. Passing a single string as partition_cols works like a one-element
        # list.
        # GH #27117
        partition_cols = "bool"
        partition_cols_list = [partition_cols]
        df = df_full
        df.to_parquet(tmp_path, partition_cols=partition_cols, compression=None)
        check_partition_names(tmp_path, partition_cols_list)
        assert read_parquet(tmp_path).shape == df.shape

    @pytest.mark.parametrize(
        "path_type", [str, lambda x: x], ids=["string", "pathlib.Path"]
    )
    def test_partition_cols_pathlib(self, tmp_path, pa, df_compat, path_type):
        # [FUNCTION DOC]
        # GH 35902. Parametrized over a string path and a path object. Writing a
        # partitioned dataset and reading gives the same shape.
        # GH 35902

        partition_cols = "B"
        partition_cols_list = [partition_cols]
        df = df_compat

        path = path_type(tmp_path)
        df.to_parquet(path, partition_cols=partition_cols_list)
        assert read_parquet(path).shape == df.shape

    def test_empty_dataframe(self, pa, temp_file):
        # [FUNCTION DOC]
        # GH 27339. An empty frame round-trips.
        # GH #27339
        df = pd.DataFrame(index=[], columns=[])
        check_round_trip(df, temp_file, pa)

    def test_write_with_schema(self, pa, temp_file):
        # [FUNCTION DOC]
        # Passing an explicit schema that declares the column as boolean converts the
        # data on write; the read equals the data cast to bool.
        import pyarrow

        df = pd.DataFrame({"x": [0, 1]})
        schema = pyarrow.schema([pyarrow.field("x", type=pyarrow.bool_())])
        out_df = df.astype(bool)
        check_round_trip(
            df, temp_file, pa, write_kwargs={"schema": schema}, expected=out_df
        )

    def test_additional_extension_arrays(self, pa, using_infer_string, temp_file):
        # [FUNCTION DOC]
        # Nullable Int64 and UInt32 and string-dtype columns round-trip through the
        # Arrow array protocol (string column becomes plain string when infer-string and
        # old pyarrow); a second frame with missing nullable ints also round-trips.
        # test additional ExtensionArrays that are supported through the
        # __arrow_array__ protocol
        pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype="Int64"),
                "b": pd.Series([1, 2, 3], dtype="UInt32"),
                "c": pd.Series(["a", None, "c"], dtype="string"),
            }
        )
        if using_infer_string and pa_version_under19p0:
            check_round_trip(df, temp_file, pa, expected=df.astype({"c": "str"}))
        else:
            check_round_trip(df, temp_file, pa)

        df = pd.DataFrame({"a": pd.Series([1, 2, 3, None], dtype="Int64")})
        check_round_trip(df, temp_file, pa)

    def test_pyarrow_backed_string_array(
        self, pa, string_storage, using_infer_string, temp_file
    ):
        # [FUNCTION DOC]
        # A string column of pyarrow-backed string dtype round-trips; the expected dtype
        # follows the string storage option and infer-string mode (with string column
        # labels in that mode).
        # test ArrowStringArray supported through the __arrow_array__ protocol
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"a": pd.Series(["a", None, "c"], dtype="string[pyarrow]")})
        with pd.option_context("string_storage", string_storage):
            if using_infer_string:
                if pa_version_under19p0:
                    expected = df.astype("str")
                else:
                    expected = df.astype(f"string[{string_storage}]")
                expected.columns = expected.columns.astype("str")
            else:
                expected = df.astype(f"string[{string_storage}]")
            check_round_trip(df, temp_file, pa, expected=expected)

    def test_additional_extension_types(self, pa, temp_file):
        # [FUNCTION DOC]
        # Interval, period and datetime-interval columns round-trip (GH 45881 for the
        # datetime interval subtype).
        # test additional ExtensionArrays that are supported through the
        # __arrow_array__ protocol + by defining a custom ExtensionType
        pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "c": pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)]),
                "d": pd.period_range("2012-01-01", periods=3, freq="D"),
                # GH-45881 issue with interval with datetime64[ns] subtype
                "e": pd.IntervalIndex.from_breaks(
                    pd.date_range("2012-01-01", periods=4, freq="D")
                ),
            }
        )
        check_round_trip(df, temp_file, pa)

    def test_timestamp_nanoseconds(self, pa, temp_file):
        # [FUNCTION DOC]
        # Writing nanosecond timestamps with format version 2.6 works without error and
        # round-trips.
        # with version 2.6, pyarrow defaults to writing the nanoseconds, so
        # this should work without error, even for pyarrow < 13
        ver = "2.6"
        df = pd.DataFrame({"a": pd.date_range("2017-01-01", freq="1ns", periods=10)})
        check_round_trip(df, temp_file, pa, write_kwargs={"version": ver})

    def test_timezone_aware_index(self, pa, timezone_aware_date_list, temp_file):
        # [FUNCTION DOC]
        # Parametrized over the seven tz-aware values. A frame with tz-aware index and a
        # column of those values round-trips, ignoring dtype differences.
        idx = 5 * [timezone_aware_date_list]
        df = pd.DataFrame(index=idx, data={"index_as_col": idx})
        check_round_trip(df, temp_file, pa, check_dtype=False)

    def test_filter_row_groups(self, pa, temp_file):
        # [FUNCTION DOC]
        # Reading with a row filter on column a equal to zero returns one row.
        # https://github.com/pandas-dev/pandas/issues/26551
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"a": list(range(3))})
        df.to_parquet(temp_file, engine=pa)
        result = read_parquet(temp_file, pa, filters=[("a", "==", 0)])
        assert len(result) == 1

    @pytest.mark.filterwarnings("ignore:make_block is deprecated:DeprecationWarning")
    def test_read_dtype_backend_pyarrow_config(self, pa, df_full, temp_file):
        # [FUNCTION DOC]
        # Round-trips the full frame plus tz-aware and boolean-with-None columns reading
        # with the pyarrow dtype backend; the expected frame is the Arrow table
        # converted with Arrow dtypes, with the NaT datetime column cast to microsecond
        # timestamp.
        import pyarrow

        df = df_full

        # additional supported types for pyarrow
        dti = pd.date_range("20130101", periods=3, tz="Europe/Brussels", unit="ns")
        dti = dti._with_freq(None)  # freq doesn't round-trip
        df["datetime_tz"] = dti
        df["bool_with_none"] = [True, None, True]

        pa_table = pyarrow.Table.from_pandas(df)
        expected = pa_table.to_pandas(types_mapper=pd.ArrowDtype)
        expected["datetime_with_nat"] = expected["datetime_with_nat"].astype(
            "timestamp[us][pyarrow]"
        )

        check_round_trip(
            df,
            temp_file,
            engine=pa,
            read_kwargs={"dtype_backend": "pyarrow"},
            expected=expected,
        )

    def test_read_dtype_backend_pyarrow_config_index(self, pa, temp_file):
        # [FUNCTION DOC]
        # A frame with Arrow int64 dtype and named index reads back with an Arrow int
        # index under the pyarrow backend.
        df = pd.DataFrame(
            {"a": [1, 2]}, index=pd.Index([3, 4], name="test"), dtype="int64[pyarrow]"
        )
        expected = df.copy()

        expected.index = expected.index.astype("int64[pyarrow]")
        check_round_trip(
            df,
            temp_file,
            engine=pa,
            read_kwargs={"dtype_backend": "pyarrow"},
            expected=expected,
        )

    @pytest.mark.parametrize(
        "columns",
        [
            [0, 1],
            pytest.param(
                [b"foo", b"bar"],
                marks=pytest.mark.xfail(
                    pa_version_under20p0,
                    raises=NotImplementedError,
                    reason="https://github.com/apache/arrow/pull/44171",
                ),
            ),
            pytest.param(
                [
                    datetime.datetime(2011, 1, 1, 0, 0),
                    datetime.datetime(2011, 1, 1, 1, 1),
                ],
                marks=pytest.mark.xfail(
                    pa_version_under17p0,
                    reason="pa.pandas_compat passes 'datetime64' to .astype",
                ),
            ),
        ],
    )
    def test_columns_dtypes_not_invalid(self, pa, columns, temp_file):
        # [FUNCTION DOC]
        # Parametrized over integer labels, byte-string labels (expected failure for
        # pyarrow older than 20) and datetime labels (expected failure for pyarrow older
        # than 17). Non-string column labels round-trip with pyarrow.
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})

        df.columns = columns
        check_round_trip(df, temp_file, pa)

    def test_empty_columns(self, pa, temp_file):
        # [FUNCTION DOC]
        # GH 52034. A frame with no columns but a named string index round-trips.
        # GH 52034
        df = pd.DataFrame(index=pd.Index(["a", "b", "c"], name="custom name"))
        check_round_trip(df, temp_file, pa)

    def test_df_attrs_persistence(self, temp_file, pa):
        # [FUNCTION DOC]
        # Frame attrs dictionary persists across write and read.
        df = pd.DataFrame(data={1: [1]})
        df.attrs = {"test_attribute": 1}
        df.to_parquet(temp_file, engine=pa)
        new_df = read_parquet(temp_file, engine=pa)
        assert new_df.attrs == df.attrs

    def test_string_inference(self, temp_file, pa, using_infer_string):
        # [FUNCTION DOC]
        # GH 54431. Reads with the future infer-string option on; values, index and
        # column labels get the NaN-variant string dtype (column labels may be object
        # for old pyarrow when the global option is off).
        # GH#54431
        df = pd.DataFrame(data={"a": ["x", "y"]}, index=["a", "b"])
        df.to_parquet(temp_file, engine=pa)
        with pd.option_context("future.infer_string", True):
            result = read_parquet(temp_file, engine=pa)
        dtype = pd.StringDtype(na_value=np.nan)
        expected = pd.DataFrame(
            data={"a": ["x", "y"]},
            dtype=dtype,
            index=pd.Index(["a", "b"], dtype=dtype),
            columns=pd.Index(
                ["a"],
                dtype=(
                    object if pa_version_under19p0 and not using_infer_string else dtype
                ),
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_roundtrip_decimal(self, temp_file, pa):
        # [FUNCTION DOC]
        # GH 54768. Writes a string-dtype column holding a decimal value with an
        # explicit decimal128 schema; reading gives a string "123" for old pyarrow or an
        # object column holding the decimal for new pyarrow.
        # KNOWN ISSUE: the local pyarrow import shadows the pa fixture argument with the
        # module object, so the fixture value is discarded.
        # GH#54768
        import pyarrow as pa

        df = pd.DataFrame({"a": [Decimal("123.00")]}, dtype="string[pyarrow]")
        df.to_parquet(temp_file, schema=pa.schema([("a", pa.decimal128(5))]))
        result = read_parquet(temp_file)
        if pa_version_under19p0:
            expected = pd.DataFrame({"a": ["123"]}, dtype="string")
        else:
            expected = pd.DataFrame({"a": [Decimal("123.00")]}, dtype="object")
        tm.assert_frame_equal(result, expected)

    def test_infer_string_large_string_type(self, temp_file, pa):
        # [FUNCTION DOC]
        # GH 54798. A large-string column written with pyarrow is read under infer-
        # string mode as the NaN-variant string dtype.
        # KNOWN ISSUE: the local pyarrow import shadows the pa fixture argument with the
        # module object.
        # GH#54798
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"a": pa.array([None, "b", "c"], pa.large_string())})
        pq.write_table(table, temp_file)

        with pd.option_context("future.infer_string", True):
            result = read_parquet(temp_file)
        expected = pd.DataFrame(
            data={"a": [None, "b", "c"]},
            dtype=pd.StringDtype(na_value=np.nan),
            columns=pd.Index(["a"], dtype=pd.StringDtype(na_value=np.nan)),
        )
        tm.assert_frame_equal(result, expected)

    # NOTE: this test is not run by default, because it requires a lot of memory (>5GB)
    # @pytest.mark.slow
    # def test_string_column_above_2GB(self, tmp_path, pa):
    #     # https://github.com/pandas-dev/pandas/issues/55606
    #     # above 2GB of string data
    #     v1 = b"x" * 100000000
    #     v2 = b"x" * 147483646
    #     df = pd.DataFrame({"strings": [v1] * 20 + [v2] + ["x"] * 20}, dtype="string")
    #     df.to_parquet(tmp_path / "test.parquet")
    #     result = read_parquet(tmp_path / "test.parquet")
    #     assert result["strings"].dtype == "string"
    # FIXME: don't leave commented-out

    def test_non_nanosecond_timestamps(self, temp_file):
        # [FUNCTION DOC]
        # GH 49236. Needs pyarrow 13. A microsecond timestamp from the year 1600 reads
        # back as microsecond datetime values.
        # GH#49236
        pa = pytest.importorskip("pyarrow", "13.0.0")
        pq = pytest.importorskip("pyarrow.parquet")

        arr = pa.array([datetime.datetime(1600, 1, 1)], type=pa.timestamp("us"))
        table = pa.table([arr], names=["timestamp"])
        pq.write_table(table, temp_file)
        result = read_parquet(temp_file)
        expected = pd.DataFrame(
            data={"timestamp": [datetime.datetime(1600, 1, 1)]},
            dtype="datetime64[us]",
        )
        tm.assert_frame_equal(result, expected)

    def test_maps_as_pydicts(self, pa, temp_file):
        # [FUNCTION DOC]
        # Needs pyarrow 13. A map column written with an explicit schema and read with
        # the strict maps-as-dicts option round-trips as dictionaries.
        pyarrow = pytest.importorskip("pyarrow", "13.0.0")

        schema = pyarrow.schema(
            [("foo", pyarrow.map_(pyarrow.string(), pyarrow.int64()))]
        )
        df = pd.DataFrame([{"foo": {"A": 1}}, {"foo": {"B": 2}}])
        check_round_trip(
            df,
            temp_file,
            pa,
            write_kwargs={"schema": schema},
            read_kwargs={"to_pandas_kwargs": {"maps_as_pydicts": "strict"}},
        )


class TestParquetFastParquet(Base):
    # [CLASS DOC]
    # Tests specific to the fastparquet engine (and some engine-agnostic file-
    # handle/filesystem tests placed here). Inherits the base error helpers.
    def test_basic(self, fp, df_full, request, temp_file):
        # [FUNCTION DOC]
        # Needs pytz. Full frame plus tz-aware (Eastern) datetimes and timedeltas round-
        # trip with fastparquet. The request argument is unused.
        pytz = pytest.importorskip("pytz")

        tz = pytz.timezone("US/Eastern")
        df = df_full

        dti = pd.date_range("20130101", periods=3, tz=tz)
        dti = dti._with_freq(None)  # freq doesn't round-trip
        df["datetime_tz"] = dti
        df["timedelta"] = pd.timedelta_range("1 day", periods=3)
        check_round_trip(df, temp_file, fp)

    def test_columns_dtypes_invalid(self, fp, temp_file):
        # [FUNCTION DOC]
        # Non-string column labels (numbers, bytes, datetimes) must make fastparquet
        # raise a type error saying column names must be strings.
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})

        err = TypeError
        msg = "Column name must be a string"

        # numeric
        df.columns = [0, 1]
        self.check_error_on_write(df, fp, err, msg, temp_file)

        # bytes
        df.columns = [b"foo", b"bar"]
        self.check_error_on_write(df, fp, err, msg, temp_file)

        # python object
        df.columns = [
            datetime.datetime(2011, 1, 1, 0, 0),
            datetime.datetime(2011, 1, 1, 1, 1),
        ]
        self.check_error_on_write(df, fp, err, msg, temp_file)

    def test_duplicate_columns(self, fp, temp_file):
        # [FUNCTION DOC]
        # Duplicate column labels must raise a value error saying a dataset with
        # duplicate column names cannot be created.
        # not currently able to handle duplicate columns
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list("aaa")).copy()
        msg = "Cannot create parquet dataset with duplicate column names"
        self.check_error_on_write(df, fp, ValueError, msg, temp_file)

    def test_bool_with_none(self, fp, request, temp_file):
        # [FUNCTION DOC]
        # Bool column with None round-trips as float (comparison ignores dtype because
        # fastparquet 0.7.1 produces float64 rather than float16). The request argument
        # is unused.
        df = pd.DataFrame({"a": [True, None, False]})
        expected = pd.DataFrame({"a": [1.0, np.nan, 0.0]}, dtype="float16")
        # Fastparquet bug in 0.7.1 makes it so that this dtype becomes
        # float64
        check_round_trip(df, temp_file, fp, expected=expected, check_dtype=False)

    def test_unsupported(self, fp, temp_file):
        # [FUNCTION DOC]
        # Period dtype raises a value error from fastparquet (message unchecked); a
        # mixed-type object column raises a value error saying it cannot infer the
        # object conversion type.
        # period
        df = pd.DataFrame({"a": pd.period_range("2013", freq="M", periods=3)})
        # error from fastparquet -> don't check exact error message
        self.check_error_on_write(df, fp, ValueError, None, temp_file)

        # mixed
        df = pd.DataFrame({"a": ["a", 1, 2.0]})
        msg = "Can't infer object conversion type"
        self.check_error_on_write(df, fp, ValueError, msg, temp_file)

    @pytest.mark.xfail(
        using_string_dtype()
        and _HAVE_PYARROW
        and not _fp_version_lt_2025
        and _fp_version_lt_2026_5,
        reason="fastparquet >= 2025.12.0, < 2026.5.0 can't write ArrowStringArray",
    )
    def test_categorical(self, fp, temp_file):
        # [FUNCTION DOC]
        # Expected failure in the same fastparquet-version/infer-string condition as the
        # cross-engine test. A categorical column round-trips with fastparquet.
        df = pd.DataFrame({"a": pd.Categorical(list("abc"))})
        check_round_trip(df, temp_file, fp)

    def test_filter_row_groups(self, fp, temp_file):
        # [FUNCTION DOC]
        # Writes with one row per group and reads with a filter on column a equal to
        # zero; one row returned.
        d = {"a": list(range(3))}
        df = pd.DataFrame(d)
        df.to_parquet(temp_file, engine=fp, compression=None, row_group_offsets=1)
        result = read_parquet(temp_file, fp, filters=[("a", "==", 0)])
        assert len(result) == 1

    @pytest.mark.single_cpu
    def test_s3_roundtrip(self, df_compat, s3_bucket_public, s3so, fp, temp_file):
        # [FUNCTION DOC]
        # Single-cpu. GH 19134. Round trip to the mock bucket with storage options using
        # fastparquet.
        # GH #19134
        check_round_trip(
            df_compat,
            temp_file,
            fp,
            path=f"s3://{s3_bucket_public.name}/fastparquet.parquet",
            read_kwargs={"storage_options": s3so},
            write_kwargs={"compression": None, "storage_options": s3so},
        )

    @pytest.mark.xfail(
        using_string_dtype()
        and _HAVE_PYARROW
        and not _fp_version_lt_2025
        and _fp_version_lt_2026_5,
        reason="fastparquet >= 2025.12.0, < 2026.5.0 can't write ArrowStringArray",
    )
    def test_partition_cols_supported(self, tmp_path, fp, df_full):
        # [FUNCTION DOC]
        # Expected failure under the fastparquet-version/infer-string condition. GH
        # 23283. Writes partitioned by two columns; the directory exists and fastparquet
        # sees two categorical partition columns.
        # GH #23283
        partition_cols = ["bool", "int"]
        df = df_full
        df.to_parquet(
            tmp_path,
            engine="fastparquet",
            partition_cols=partition_cols,
            compression=None,
        )
        assert os.path.exists(tmp_path)
        import fastparquet

        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 2

    @pytest.mark.xfail(
        using_string_dtype()
        and _HAVE_PYARROW
        and not _fp_version_lt_2025
        and _fp_version_lt_2026_5,
        reason="fastparquet >= 2025.12.0, < 2026.5.0 can't write ArrowStringArray",
    )
    def test_partition_cols_string(self, tmp_path, fp, df_full):
        # [FUNCTION DOC]
        # Same expected-failure condition. GH 27117. A single string partition_cols
        # value gives one partition column.
        # GH #27117
        partition_cols = "bool"
        df = df_full
        df.to_parquet(
            tmp_path,
            engine="fastparquet",
            partition_cols=partition_cols,
            compression=None,
        )
        assert os.path.exists(tmp_path)
        import fastparquet

        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 1

    @pytest.mark.xfail(
        using_string_dtype()
        and _HAVE_PYARROW
        and not _fp_version_lt_2025
        and _fp_version_lt_2026_5,
        reason="fastparquet >= 2025.12.0, < 2026.5.0 can't write ArrowStringArray",
    )
    def test_partition_on_supported(self, tmp_path, fp, df_full):
        # [FUNCTION DOC]
        # Same expected-failure condition. GH 23283. The fastparquet-specific
        # partition_on argument gives two partition columns.
        # GH #23283
        partition_cols = ["bool", "int"]
        df = df_full
        df.to_parquet(
            tmp_path,
            engine="fastparquet",
            compression=None,
            partition_on=partition_cols,
        )
        assert os.path.exists(tmp_path)
        import fastparquet

        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 2

    def test_error_on_using_partition_cols_and_partition_on(
        self, tmp_path, fp, df_full
    ):
        # [FUNCTION DOC]
        # GH 23283. Supplying both partition_on and partition_cols must raise a value
        # error telling to use partition_cols.
        # GH #23283
        partition_cols = ["bool", "int"]
        df = df_full
        msg = (
            "Cannot use both partition_on and partition_cols. Use partition_cols for "
            "partitioning data"
        )
        with pytest.raises(ValueError, match=msg):
            df.to_parquet(
                tmp_path,
                engine="fastparquet",
                compression=None,
                partition_on=partition_cols,
                partition_cols=partition_cols,
            )

    def test_empty_dataframe(self, fp, temp_file):
        # [FUNCTION DOC]
        # GH 27339. A completely empty frame round-trips with fastparquet.
        # GH #27339
        df = pd.DataFrame()
        expected = df.copy()
        check_round_trip(df, temp_file, fp, expected=expected)

    def test_timezone_aware_index(
        self, fp, timezone_aware_date_list, request, temp_file
    ):
        # [FUNCTION DOC]
        # Parametrized over the seven tz-aware values. Round trip with fastparquet where
        # the expected frame's index name becomes index.
        idx = 5 * [timezone_aware_date_list]

        df = pd.DataFrame(index=idx, data={"index_as_col": idx})

        expected = df.copy()
        expected.index.name = "index"
        check_round_trip(df, temp_file, fp, expected=expected)

    def test_close_file_handle_on_read_error(self, temp_file):
        # [FUNCTION DOC]
        # Writes garbage bytes to a file; reading with fastparquet raises an external
        # error; then the file must be deletable (on Windows deletion fails if the
        # handle leaked).
        pathlib.Path(temp_file).write_bytes(b"breakit")
        with tm.external_error_raised(Exception):  # Not important which exception
            read_parquet(temp_file, engine="fastparquet")
        # The next line raises an error on Windows if the file is still open
        pathlib.Path(temp_file).unlink(missing_ok=False)

    def test_bytes_file_name(self, engine, temp_file):
        # [FUNCTION DOC]
        # GH 48944. Parametrized over both engines. Writing to an open binary file
        # object and reading back by file name yields the same frame.
        # GH#48944
        df = pd.DataFrame(data={"A": [0, 1], "B": [1, 0]})
        with open(temp_file, "wb") as f:
            df.to_parquet(f)

        result = read_parquet(temp_file, engine=engine)
        tm.assert_frame_equal(result, df)

    def test_filesystem_notimplemented(self, temp_file):
        # [FUNCTION DOC]
        # Needs fastparquet. Passing a filesystem argument for write or read with the
        # fastparquet engine must raise a not-implemented error that the filesystem is
        # not implemented.
        pytest.importorskip("fastparquet")
        df = pd.DataFrame(data={"A": [0, 1], "B": [1, 0]})
        with pytest.raises(NotImplementedError, match="filesystem is not implemented"):
            df.to_parquet(temp_file, engine="fastparquet", filesystem="foo")

        pathlib.Path(temp_file).write_bytes(b"foo")
        with pytest.raises(NotImplementedError, match="filesystem is not implemented"):
            read_parquet(temp_file, engine="fastparquet", filesystem="foo")

    def test_invalid_filesystem(self, temp_file):
        # [FUNCTION DOC]
        # Needs pyarrow. A non-filesystem value for filesystem must raise a value error
        # saying it must be a pyarrow or fsspec file system, for write and read.
        pytest.importorskip("pyarrow")
        df = pd.DataFrame(data={"A": [0, 1], "B": [1, 0]})

        with pytest.raises(
            ValueError, match="filesystem must be a pyarrow or fsspec FileSystem"
        ):
            df.to_parquet(temp_file, engine="pyarrow", filesystem="foo")

        pathlib.Path(temp_file).write_bytes(b"foo")
        with pytest.raises(
            ValueError, match="filesystem must be a pyarrow or fsspec FileSystem"
        ):
            read_parquet(temp_file, engine="pyarrow", filesystem="foo")

    def test_unsupported_pa_filesystem_storage_options(self, temp_file):
        # [FUNCTION DOC]
        # Needs pyarrow filesystem module. Providing both a pyarrow file system and
        # storage options must raise a not-implemented error for write and read.
        pa_fs = pytest.importorskip("pyarrow.fs")
        df = pd.DataFrame(data={"A": [0, 1], "B": [1, 0]})

        with pytest.raises(
            NotImplementedError,
            match="storage_options not supported with a pyarrow FileSystem.",
        ):
            df.to_parquet(
                temp_file,
                engine="pyarrow",
                filesystem=pa_fs.LocalFileSystem(),
                storage_options={"foo": "bar"},
            )

        pathlib.Path(temp_file).write_bytes(b"foo")
        with pytest.raises(
            NotImplementedError,
            match="storage_options not supported with a pyarrow FileSystem.",
        ):
            read_parquet(
                temp_file,
                engine="pyarrow",
                filesystem=pa_fs.LocalFileSystem(),
                storage_options={"foo": "bar"},
            )

    def test_invalid_dtype_backend(self, engine, temp_file):
        # [FUNCTION DOC]
        # Parametrized by the engine fixture. Reading with an invalid dtype backend must
        # raise a value error stating only the nullable-numpy and pyarrow backends are
        # allowed.
        # KNOWN ISSUE: the engine fixture parameter is accepted but never used; the
        # write and read calls use the default engine, so the parametrization only
        # repeats the same test.
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        df = pd.DataFrame({"int": list(range(1, 4))})
        df.to_parquet(temp_file)
        with pytest.raises(ValueError, match=msg):
            read_parquet(temp_file, dtype_backend="numpy")
