"""
Tests multithreading behaviour for reading and
parsing files for each parser defined in parsers.py
"""

# [MODULE DOC]
# Parser tests of reading CSV data from multiple threads: reading many in-memory buffers
# concurrently and reading disjoint row ranges of one file concurrently and
# concatenating them. All tests are marked single cpu and slow (module-level marks).
# Contains a helper that builds a frame using a pool of worker threads.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.

from contextlib import ExitStack
from io import BytesIO
from multiprocessing.pool import ThreadPool

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.util.version import Version

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")

# We'll probably always skip these for pyarrow
# Maybe we'll add our own tests for pyarrow too
pytestmark = [
    pytest.mark.single_cpu,
    pytest.mark.slow,
]


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_multi_thread_string_io_read_csv(all_parsers, request):
    # [FUNCTION DOC]
    # Creates ten in-memory binary buffers each holding 100 rows of three identical
    # integers, reads them all with the parser through a pool of eight threads, and
    # verifies every result frame equals the first. For pyarrow with a library version
    # below 16 the test is marked expected to fail. Ignores the block manager
    # deprecation warning.
    # see gh-11786
    parser = all_parsers
    if parser.engine == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        if Version(pa.__version__) < Version("16.0"):
            request.applymarker(
                pytest.mark.xfail(reason="# ValueError: Found non-unique column index")
            )
    max_row_range = 100
    num_files = 10

    bytes_to_df = (
        "\n".join([f"{i:d},{i:d},{i:d}" for i in range(max_row_range)]).encode()
        for _ in range(num_files)
    )

    # Read all files in many threads.
    with ExitStack() as stack:
        files = [stack.enter_context(BytesIO(b)) for b in bytes_to_df]

        pool = stack.enter_context(ThreadPool(8))

        results = pool.map(parser.read_csv, files)
        first_result = results[0]

        for result in results:
            tm.assert_frame_equal(first_result, result)


def _generate_multi_thread_dataframe(parser, path, num_rows, num_tasks):
    # [FUNCTION DOC]
    # HELPER: for the multi-thread path test. Parameters: a parser wrapper, a file path,
    # the total number of rows, and the number of tasks (threads). Splits the rows into
    # equal consecutive ranges, reads each range in its own thread (the first range with
    # a header and date column by name, later ranges by skipping rows with no header and
    # the date column by position), sets every later chunk's columns to the first
    # chunk's columns, concatenates them in order and returns the combined frame.
    """
    Generate a DataFrame via multi-thread.

    Parameters
    ----------
    parser : BaseParser
        The parser object to use for reading the data.
    path : str
        The location of the CSV file to read.
    num_rows : int
        The number of rows to read per task.
    num_tasks : int
        The number of tasks to use for reading this DataFrame.

    Returns
    -------
    df : DataFrame
    """

    def reader(arg):
        # [FUNCTION DOC]
        # HELPER: worker for the thread pool. Takes a pair of start row and number of
        # rows. For a start of zero reads with the first column as index, the header at
        # row zero, the row limit and the date column parsed by name; otherwise reads
        # the next block by skipping the header plus start rows with no header, the same
        # index column, the row limit, and the date column parsed by its position.
        # Returns the resulting frame.
        """
        Create a reader for part of the CSV.

        Parameters
        ----------
        arg : tuple
            A tuple of the following:

            * start : int
                The starting row to start for parsing CSV
            * nrows : int
                The number of rows to read.

        Returns
        -------
        df : DataFrame
        """
        start, nrows = arg

        if not start:
            return parser.read_csv(
                path, index_col=0, header=0, nrows=nrows, parse_dates=["date"]
            )

        return parser.read_csv(
            path,
            index_col=0,
            header=None,
            skiprows=int(start) + 1,
            nrows=nrows,
            parse_dates=[9],
        )

    tasks = [
        (num_rows * i // num_tasks, num_rows // num_tasks) for i in range(num_tasks)
    ]

    with ThreadPool(processes=num_tasks) as pool:
        results = pool.map(reader, tasks)

    header = results[0].columns

    for r in results[1:]:
        r.columns = header

    final_dataframe = pd.concat(results)
    return final_dataframe


@xfail_pyarrow  # ValueError: The 'nrows' option is not supported
def test_multi_thread_path_multipart_read_csv(tmp_path, all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (row limit unsupported). Writes a 48-row frame
    # of random floats, strings, dates and integers to a temporary CSV, reads it in four
    # concurrent row ranges through the helper, and verifies the combined result equals
    # the original frame with the date column in microsecond unit.
    # Side effect: writes a file in the temporary directory.
    # see gh-11786
    num_tasks = 4
    num_rows = 48

    parser = all_parsers
    file_name = "__thread_pool_reader__.csv"
    df = DataFrame(
        {
            "a": np.random.default_rng(2).random(num_rows),
            "b": np.random.default_rng(2).random(num_rows),
            "c": np.random.default_rng(2).random(num_rows),
            "d": np.random.default_rng(2).random(num_rows),
            "e": np.random.default_rng(2).random(num_rows),
            "foo": ["foo"] * num_rows,
            "bar": ["bar"] * num_rows,
            "baz": ["baz"] * num_rows,
            "date": pd.date_range("20000101 09:00:00", periods=num_rows, freq="s"),
            "int": np.arange(num_rows, dtype="int64"),
        }
    )

    path = tmp_path / file_name
    df.to_csv(path)

    result = _generate_multi_thread_dataframe(parser, path, num_rows, num_tasks)

    expected = df[:]
    expected["date"] = expected["date"].astype("M8[us]")
    tm.assert_frame_equal(result, expected)
