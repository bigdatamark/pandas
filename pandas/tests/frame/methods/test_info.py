# [MODULE DOC]
# Tests for DataFrame info (the text summary written to a buffer) and the related memory_usage method: layout, verbose and column-limit selection, count display, memory estimates (including deep and qualified lower-bound markers), and special index types.
# Most tests capture output in an in-memory text buffer. Some depend on platform (pypy, 32-bit, arm), optional libraries (pyarrow, numba) and the infer-string option.

from io import StringIO
import re
from string import ascii_uppercase
import sys
import textwrap

import numpy as np
import pytest

from pandas.compat import (
    HAS_PYARROW,
    IS64,
    PYPY,
    is_platform_arm,
)

from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
    option_context,
)
import pandas._testing as tm
from pandas.util.version import Version


@pytest.fixture
def duplicate_columns_frame():
    # [FUNCTION DOC]
    # HELPER: fixture returning a frame of 1500 rows of random numbers with duplicate column labels (two a and two b), used by the info smoke test.
    """Dataframe with duplicate column names."""
    return DataFrame(
        np.random.default_rng(2).standard_normal((1500, 4)),
        columns=["a", "a", "b", "b"],
    )


def test_info_empty():
    # [FUNCTION DOC]
    # GH #45494. Verifies info on an empty frame writes the class line, a range index of zero entries, and the empty-frame line, exactly.
    # GH #45494
    df = DataFrame()
    buf = StringIO()
    df.info(buf=buf)
    result = buf.getvalue()
    expected = textwrap.dedent(
        """\
        <class 'pandas.DataFrame'>
        RangeIndex: 0 entries
        Empty DataFrame\n"""
    )
    assert result == expected


def test_info_categorical_column_smoke_test():
    # [FUNCTION DOC]
    # Smoke test with an integer column and a categorical column of 2500 rows, then a filtered subset. Verifies only that info runs without error on categorical data.
    n = 2500
    df = DataFrame({"int64": np.random.default_rng(2).integers(100, size=n, dtype=int)})
    df["category"] = Series(
        np.array(list("abcdefghij")).take(
            np.random.default_rng(2).integers(0, 10, size=n, dtype=int)
        )
    ).astype("category")
    df.isna()
    buf = StringIO()
    df.info(buf=buf)

    df2 = df[df["category"] == "d"]
    buf = StringIO()
    df2.info(buf=buf)


@pytest.mark.parametrize(
    "fixture_func_name",
    [
        "int_frame",
        "float_frame",
        "datetime_frame",
        "duplicate_columns_frame",
        "float_string_frame",
    ],
)
def test_info_smoke_test(fixture_func_name, request):
    # [FUNCTION DOC]
    # Parametrized over fixture names (int, float, datetime, duplicate-columns and mixed string frames), fetched via the request object. Verifies info writes more than ten lines and also runs in non-verbose mode.
    frame = request.getfixturevalue(fixture_func_name)
    buf = StringIO()
    frame.info(buf=buf)
    result = buf.getvalue().splitlines()
    assert len(result) > 10

    buf = StringIO()
    frame.info(buf=buf, verbose=False)


def test_info_smoke_test2(float_frame):
    # [FUNCTION DOC]
    # Smoke test (described in its comment as nearly useless): info in non-verbose mode on one- and two-column frames and on an empty frame, writing into one shared buffer. Nothing is asserted.
    # pretty useless test, used to be mixed into the repr tests
    buf = StringIO()
    float_frame.reindex(columns=["A"]).info(verbose=False, buf=buf)
    float_frame.reindex(columns=["A", "B"]).info(verbose=False, buf=buf)

    # no columns or index
    DataFrame().info(buf=buf)


@pytest.mark.parametrize(
    "num_columns, max_info_columns, verbose",
    [
        (10, 100, True),
        (10, 11, True),
        (10, 10, True),
        (10, 9, False),
        (10, 1, False),
    ],
)
def test_info_default_verbose_selection(num_columns, max_info_columns, verbose):
    # [FUNCTION DOC]
    # Parametrized over column count, the max_info_columns option value and the expected verbose flag. Verifies that the default output with the option set equals the explicit output with the expected verbose choice: the detailed listing is used only when the number of columns does not exceed the option.
    frame = DataFrame(np.random.default_rng(2).standard_normal((5, num_columns)))
    with option_context("display.max_info_columns", max_info_columns):
        io_default = StringIO()
        frame.info(buf=io_default)
        result = io_default.getvalue()

        io_explicit = StringIO()
        frame.info(buf=io_explicit, verbose=verbose)
        expected = io_explicit.getvalue()

        assert result == expected


def test_info_verbose_check_header_separator_body():
    # [FUNCTION DOC]
    # Frame with 1001 columns. Verifies the verbose output contains the expected header and separator for the layout without counts-width issues, and that each column line starts with its zero-based number following the header lines.
    buf = StringIO()
    size = 1001
    start = 5
    frame = DataFrame(np.random.default_rng(2).standard_normal((3, size)))
    frame.info(verbose=True, buf=buf)

    res = buf.getvalue()
    header = " #     Column  Dtype  \n---    ------  -----  "
    assert header in res

    frame.info(verbose=True, buf=buf)
    buf.seek(0)
    lines = buf.readlines()
    assert len(lines) > 0

    for i, line in enumerate(lines):
        if start <= i < start + size:
            line_nr = f" {i - start} "
            assert line.startswith(line_nr)


@pytest.mark.parametrize(
    "size, header_exp, separator_exp, first_line_exp, last_line_exp",
    [
        (
            4,
            " #   Column  Non-Null Count  Dtype  ",
            "---  ------  --------------  -----  ",
            " 0   0       3 non-null      float64",
            " 3   3       3 non-null      float64",
        ),
        (
            11,
            " #   Column  Non-Null Count  Dtype  ",
            "---  ------  --------------  -----  ",
            " 0   0       3 non-null      float64",
            " 10  10      3 non-null      float64",
        ),
        (
            101,
            " #    Column  Non-Null Count  Dtype  ",
            "---   ------  --------------  -----  ",
            " 0    0       3 non-null      float64",
            " 100  100     3 non-null      float64",
        ),
        (
            1001,
            " #     Column  Non-Null Count  Dtype  ",
            "---    ------  --------------  -----  ",
            " 0     0       3 non-null      float64",
            " 1000  1000    3 non-null      float64",
        ),
        (
            10001,
            " #      Column  Non-Null Count  Dtype  ",
            "---     ------  --------------  -----  ",
            " 0      0       3 non-null      float64",
            " 10000  10000   3 non-null      float64",
        ),
    ],
)
def test_info_verbose_with_counts_spacing(
    size, header_exp, separator_exp, first_line_exp, last_line_exp
):
    # [FUNCTION DOC]
    # Parametrized over frame width (4 to 10001 columns) with expected header, separator, first and last table lines. Verifies the column table is aligned and the first and last row text widths adapt to the number of digits, with counts shown, using only the table lines between the header summary and the dtype and memory summary.
    """Test header column, spacer, first line and last line in verbose mode."""
    frame = DataFrame(np.random.default_rng(2).standard_normal((3, size)))
    with StringIO() as buf:
        frame.info(verbose=True, show_counts=True, buf=buf)
        all_lines = buf.getvalue().splitlines()
    # Here table would contain only header, separator and table lines
    # dframe repr, index summary, memory usage and dtypes are excluded
    table = all_lines[3:-2]
    header, separator, first_line, *rest, last_line = table
    assert header == header_exp
    assert separator == separator_exp
    assert first_line == first_line_exp
    assert last_line == last_line_exp


def test_info_memory():
    # [FUNCTION DOC]
    # Issue 21056. Verifies the exact info text of a one-column frame, including the memory usage in bytes matching the frame memory usage sum (no plus sign, as the index is a range).
    # https://github.com/pandas-dev/pandas/issues/21056
    df = DataFrame({"a": Series([1, 2], dtype="i8")})
    buf = StringIO()
    df.info(buf=buf)
    result = buf.getvalue()
    bytes = float(df.memory_usage().sum())
    expected = textwrap.dedent(
        f"""\
    <class 'pandas.DataFrame'>
    RangeIndex: 2 entries, 0 to 1
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       2 non-null      int64
    dtypes: int64(1)
    memory usage: {bytes} bytes
    """
    )
    assert result == expected


def test_info_wide():
    # [FUNCTION DOC]
    # Frame with 101 columns. Verifies the max_cols argument set to 101 gives the long detailed listing (more than 100 lines), and the display option for the maximum info columns set to 101 gives identical output.
    io = StringIO()
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 101)))
    df.info(buf=io)

    io = StringIO()
    df.info(buf=io, max_cols=101)
    result = io.getvalue()
    assert len(result.splitlines()) > 100

    expected = result
    with option_context("display.max_info_columns", 101):
        io = StringIO()
        df.info(buf=io)
        result = io.getvalue()
        assert result == expected


def test_info_duplicate_columns_shows_correct_dtypes():
    # [FUNCTION DOC]
    # GH11761. Verifies info lists duplicate column labels separately with their own dtypes (integer, then float) at fixed lines.
    # GH11761
    io = StringIO()
    frame = DataFrame([[1, 2.0]], columns=["a", "a"])
    frame.info(buf=io)
    lines = io.getvalue().splitlines(True)
    assert " 0   a       1 non-null      int64  \n" == lines[5]
    assert " 1   a       1 non-null      float64\n" == lines[6]


def test_info_shows_column_dtypes():
    # [FUNCTION DOC]
    # Verifies the dtype names of columns made from a range of dtypes (int, float, datetime, timedelta, complex, object, bool) appear in the verbose table, with the expected header.
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    data = {}
    n = 10
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    df = DataFrame(data)
    buf = StringIO()
    df.info(buf=buf)
    res = buf.getvalue()
    header = (
        " #   Column  Non-Null Count  Dtype          \n"
        "---  ------  --------------  -----          "
    )
    assert header in res
    for i, dtype in enumerate(dtypes):
        name = f" {i:d}   {i:d}       {n:d} non-null     {dtype}"
        assert name in res


def test_info_max_cols():
    # [FUNCTION DOC]
    # Frame with five columns. With the info columns option set to four or five, verifies the output has 5 lines when summarizing (verbose false, or none while columns exceed the option) and 12 lines when full output is produced (verbose true, or none when the option is not exceeded); an explicit max_cols argument overrides the option in the same way. Counts lines in the output.
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    for len_, verbose in [(5, None), (5, False), (12, True)]:
        # For verbose always      ^ setting  ^ summarize ^ full output
        with option_context("max_info_columns", 4):
            buf = StringIO()
            df.info(buf=buf, verbose=verbose)
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

    for len_, verbose in [(12, None), (5, False), (12, True)]:
        # max_cols not exceeded
        with option_context("max_info_columns", 5):
            buf = StringIO()
            df.info(buf=buf, verbose=verbose)
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

    for len_, max_cols in [(12, 5), (5, 4)]:
        # setting truncates
        with option_context("max_info_columns", 4):
            buf = StringIO()
            df.info(buf=buf, max_cols=max_cols)
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

        # setting wouldn't truncate
        with option_context("max_info_columns", 5):
            buf = StringIO()
            df.info(buf=buf, max_cols=max_cols)
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_


def test_info_memory_usage():
    # [FUNCTION DOC]
    # Large test of memory-usage reporting in info and the frame memory_usage method. Verifies: the last line shows memory usage when requested and not when disabled; a lower-bound marker (plus sign) appears when object columns make the estimate shallow, and not when object columns are excluded or when deep mode is used; object index also produces the marker; the total memory usage equals columns times rows times 8 bytes plus the index size; the memory_usage length equals the column count plus one for the index; deep equals shallow for numeric data; and calls on a multi-index frame produce positive sizes.
    # The statements that merely access index byte counts or call memory_usage without checking the result only check that they do not fail.
    # Ensure memory usage is displayed, when asserted, on the last line
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    data = {}
    n = 10
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    df = DataFrame(data)
    buf = StringIO()

    # display memory usage case
    df.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()
    assert "memory usage: " in res[-1]

    # do not display memory usage case
    df.info(buf=buf, memory_usage=False)
    res = buf.getvalue().splitlines()
    assert "memory usage: " not in res[-1]

    df.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()

    # memory usage is a lower bound, so print it as XYZ+ MB
    assert re.match(r"memory usage: [^+]+\+", res[-1])

    df.iloc[:, :5].info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()

    # excluded column with object dtype, so estimate is accurate
    assert not re.match(r"memory usage: [^+]+\+", res[-1])

    # Test a DataFrame with duplicate columns
    dtypes = ["int64", "int64", "int64", "float64"]
    data = {}
    n = 100
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    df = DataFrame(data)
    df.columns = dtypes

    df_with_object_index = DataFrame({"a": [1]}, index=Index(["foo"], dtype=object))
    df_with_object_index.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()
    assert re.match(r"memory usage: [^+]+\+", res[-1])

    df_with_object_index.info(buf=buf, memory_usage="deep")
    res = buf.getvalue().splitlines()
    assert re.match(r"memory usage: [^+]+$", res[-1])

    # Ensure df size is as expected
    # (cols * rows * bytes) + index size
    df_size = df.memory_usage().sum()
    exp_size = len(dtypes) * n * 8 + df.index.nbytes
    assert df_size == exp_size

    # Ensure number of cols in memory_usage is the same as df
    size_df = np.size(df.columns.values) + 1  # index=True; default
    assert size_df == np.size(df.memory_usage())

    # assert deep works only on object
    assert df.memory_usage().sum() == df.memory_usage(deep=True).sum()

    # test for validity
    DataFrame(1, index=["a"], columns=["A"]).memory_usage(index=True)
    DataFrame(1, index=["a"], columns=["A"]).index.nbytes
    df = DataFrame(
        data=1, index=MultiIndex.from_product([["a"], range(1000)]), columns=["A"]
    )
    df.index.nbytes
    df.memory_usage(index=True)
    df.index.values.nbytes

    mem = df.memory_usage(deep=True).sum()
    assert mem > 0


@pytest.mark.skipif(PYPY, reason="on PyPy deep=True doesn't change result")
def test_info_memory_usage_deep_not_pypy():
    # [FUNCTION DOC]
    # Skipped on pypy. Verifies deep memory usage is larger than shallow for a frame with an object index and for an object column.
    df_with_object_index = DataFrame({"a": [1]}, index=Index(["foo"], dtype=object))
    assert (
        df_with_object_index.memory_usage(index=True, deep=True).sum()
        > df_with_object_index.memory_usage(index=True).sum()
    )

    df_object = DataFrame({"a": Series(["a"], dtype=object)})
    assert df_object.memory_usage(deep=True).sum() > df_object.memory_usage().sum()


@pytest.mark.xfail(not PYPY, reason="on PyPy deep=True does not change result")
def test_info_memory_usage_deep_pypy():
    # [FUNCTION DOC]
    # Expected to fail unless running on pypy. Verifies on pypy that deep memory usage equals shallow memory usage for object columns and an object index.
    df_with_object_index = DataFrame({"a": [1]}, index=Index(["foo"], dtype=object))
    assert (
        df_with_object_index.memory_usage(index=True, deep=True).sum()
        == df_with_object_index.memory_usage(index=True).sum()
    )

    df_object = DataFrame({"a": Series(["a"], dtype=object)})
    assert df_object.memory_usage(deep=True).sum() == df_object.memory_usage().sum()


@pytest.mark.skipif(PYPY, reason="PyPy getsizeof() fails by design")
def test_usage_via_getsizeof():
    # [FUNCTION DOC]
    # Skipped on pypy. Verifies the system size-of function on a frame is close (within 100 bytes) to the deep memory usage sum, which it uses plus garbage-collector overhead.
    df = DataFrame(
        data=1, index=MultiIndex.from_product([["a"], range(1000)]), columns=["A"]
    )
    mem = df.memory_usage(deep=True).sum()
    # sys.getsizeof will call the .memory_usage with
    # deep=True, and add on some GC overhead
    diff = mem - sys.getsizeof(df)
    assert abs(diff) < 100


def test_info_memory_usage_qualified(using_infer_string):
    # [FUNCTION DOC]
    # Uses the infer-string fixture. Verifies the lower-bound plus sign appears in info output for an object index, for a string index (unless inferred strings with pyarrow are active, where it is exact), and for a multi-index with a string level (same exception); it does not appear for integer index or an integer multi-index.
    buf = StringIO()
    df = DataFrame(1, columns=list("ab"), index=[1, 2, 3])
    df.info(buf=buf)
    assert "+" not in buf.getvalue()

    buf = StringIO()
    df = DataFrame(1, columns=list("ab"), index=Index(list("ABC"), dtype=object))
    df.info(buf=buf)
    assert "+" in buf.getvalue()

    buf = StringIO()
    df = DataFrame(1, columns=list("ab"), index=Index(list("ABC"), dtype="str"))
    df.info(buf=buf)
    if using_infer_string and HAS_PYARROW:
        assert "+" not in buf.getvalue()
    else:
        assert "+" in buf.getvalue()

    buf = StringIO()
    df = DataFrame(
        1, columns=list("ab"), index=MultiIndex.from_product([range(3), range(3)])
    )
    df.info(buf=buf)
    assert "+" not in buf.getvalue()

    buf = StringIO()
    df = DataFrame(
        1, columns=list("ab"), index=MultiIndex.from_product([range(3), ["foo", "bar"]])
    )
    df.info(buf=buf)
    if using_infer_string and HAS_PYARROW:
        assert "+" not in buf.getvalue()
    else:
        assert "+" in buf.getvalue()


def test_info_memory_usage_bug_on_multiindex():
    # [FUNCTION DOC]
    # GH 14308. Verifies memory usage introspection on a multi-index frame does not materialize the values: the unstacked frame has equal values size but less deep memory usage than the stacked frame, and the difference is under 2000 bytes.
    # GH 14308
    # memory usage introspection should not materialize .values

    def memory_usage(f):
        # [FUNCTION DOC]
        # HELPER: returns the deep memory usage total of a frame; used for the multi-index comparison in the enclosing test.
        return f.memory_usage(deep=True).sum()

    N = 100
    M = len(ascii_uppercase)
    index = MultiIndex.from_product(
        [list(ascii_uppercase), date_range("20160101", periods=N)],
        names=["id", "date"],
    )
    df = DataFrame(
        {"value": np.random.default_rng(2).standard_normal(N * M)}, index=index
    )

    unstacked = df.unstack("id")
    assert df.values.nbytes == unstacked.values.nbytes
    assert memory_usage(df) > memory_usage(unstacked)

    # high upper bound
    assert memory_usage(unstacked) - memory_usage(df) < 2000


def test_info_categorical():
    # [FUNCTION DOC]
    # GH14298. Smoke test: info on a frame with a categorical index and categorical columns runs without error.
    # GH14298
    idx = CategoricalIndex(["a", "b"])
    df = DataFrame(np.zeros((2, 2)), index=idx, columns=idx)

    buf = StringIO()
    df.info(buf=buf)


@pytest.mark.xfail(not IS64, reason="GH 36579: fail on 32-bit system")
def test_info_int_columns(using_infer_string):
    # [FUNCTION DOC]
    # GH#37245. Expected to fail on 32-bit platforms. Verifies the exact info text for a frame with integer column labels and string index, where the memory line is 50.0 (exact) with inferred strings and pyarrow, otherwise 48.0 with a plus sign.
    # GH#37245
    df = DataFrame({1: [1, 2], 2: [2, 3]}, index=["A", "B"])
    buf = StringIO()
    df.info(show_counts=True, buf=buf)
    result = buf.getvalue()
    expected = textwrap.dedent(
        f"""\
        <class 'pandas.DataFrame'>
        Index: 2 entries, A to B
        Data columns (total 2 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   1       2 non-null      int64
         1   2       2 non-null      int64
        dtypes: int64(2)
        memory usage: {"50.0" if using_infer_string and HAS_PYARROW else "48.0+"} bytes
        """
    )
    assert result == expected


def test_memory_usage_empty_no_warning(using_infer_string):
    # [FUNCTION DOC]
    # GH#50066. Verifies memory_usage on a frame with only an index (no columns) emits no warning and returns a series with the single Index entry, sized 18 with inferred strings and pyarrow, otherwise 16 (8 on 32-bit).
    # GH#50066
    df = DataFrame(index=["a", "b"])
    with tm.assert_produces_warning(None):
        result = df.memory_usage()
    if using_infer_string and HAS_PYARROW:
        value = 18
    else:
        value = 16 if IS64 else 8
    expected = Series(value, index=["Index"])
    tm.assert_series_equal(result, expected)


@pytest.mark.single_cpu
def test_info_compute_numba():
    # [FUNCTION DOC]
    # GH#51922. Single-cpu test, skipped without numba and on arm with a specific numba version. Verifies info output is identical with the numba compute option enabled and disabled.
    # GH#51922
    numba = pytest.importorskip("numba")
    if Version(numba.__version__) == Version("0.61") and is_platform_arm():
        pytest.skip(f"Segfaults on ARM platforms with numba {numba.__version__}")
    df = DataFrame([[1, 2], [3, 4]])

    with option_context("compute.use_numba", True):
        buf = StringIO()
        df.info(buf=buf)
        result = buf.getvalue()

    buf = StringIO()
    df.info(buf=buf)
    expected = buf.getvalue()
    assert result == expected


@pytest.mark.parametrize(
    "row, columns, show_counts, result",
    [
        [20, 20, None, True],
        [20, 20, True, True],
        [20, 20, False, False],
        [5, 5, None, False],
        [5, 5, True, False],
        [5, 5, False, False],
    ],
)
def test_info_show_counts(row, columns, show_counts, result):
    # [FUNCTION DOC]
    # Parametrized over max_info_rows, max_info_columns, show_counts and the expected presence of non-null counts. Frame of 10 by 10 with one missing value. Verifies counts are shown when show_counts is true, or when it is not set and the frame is within both option limits; they are not shown when it is false or when it is unset and the limits are exceeded.
    # Explicit cast to float to avoid implicit cast when setting nan
    df = DataFrame(1, columns=range(10), index=range(10)).astype({1: "float"})
    df.iloc[1, 1] = np.nan

    with option_context(
        "display.max_info_rows", row, "display.max_info_columns", columns
    ):
        with StringIO() as buf:
            df.info(buf=buf, show_counts=show_counts)
            assert ("non-null" in buf.getvalue()) is result
