"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# [MODULE DOC]
# Parser tests (run against every engine through the parser fixture) for chunked
# reading: chunk size with an index column, invalid chunk sizes, combining chunk size
# with a row limit, fetching chunks explicitly, consistency of chunk dtypes, the
# mixed-type warning, and short trailing blocks.
# The pyarrow engine does not support the chunk size or row limit options, so most tests
# expect a value error naming the unsupported option for that engine and return early.

from io import StringIO

import numpy as np
import pytest

from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning

from pandas import (
    DataFrame,
    concat,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.mark.parametrize("index_col", [0, "index"])
def test_read_chunksize_with_index(all_parsers, index_col):
    # [FUNCTION DOC]
    # Reads a six-row CSV with an index column in chunks of two (index given by position
    # or by name) and verifies the three chunks equal the matching row slices of the
    # expected indexed frame. For pyarrow, verifies the unsupported chunk size error.
    # KNOWN ISSUE: the index_col parameter is accepted but never used; the reads always
    # use position zero as the index column, so the by-name variant is not exercised.
    parser = all_parsers
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    expected = DataFrame(
        [
            ["foo", 2, 3, 4, 5],
            ["bar", 7, 8, 9, 10],
            ["baz", 12, 13, 14, 15],
            ["qux", 12, 13, 14, 15],
            ["foo2", 12, 13, 14, 15],
            ["bar2", 12, 13, 14, 15],
        ],
        columns=["index", "A", "B", "C", "D"],
    )
    expected = expected.set_index("index")

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
                list(reader)
        return

    with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
        chunks = list(reader)
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


@pytest.mark.parametrize("chunksize", [1.3, "foo", 0])
def test_read_chunksize_bad(all_parsers, chunksize):
    # [FUNCTION DOC]
    # Verifies invalid chunk sizes (a float, a string, zero) raise a value error saying
    # chunksize must be an integer of at least one; for pyarrow the unsupported-option
    # message is expected instead.
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    msg = r"'chunksize' must be an integer >=1"
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"

    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), chunksize=chunksize) as _:
            pass


@pytest.mark.parametrize("chunksize", [2, 8])
def test_read_chunksize_and_nrows(all_parsers, chunksize):
    # [FUNCTION DOC]
    # Verifies that reading with both a row limit of five and a chunk size (two or
    # eight) gives, after concatenating the chunks, the same frame as reading with the
    # row limit alone. For pyarrow, verifies the unsupported row-limit error. Regression
    # for a reported issue.
    # see gh-15755
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0, "nrows": 5}

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return

    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=chunksize, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), expected)


def test_read_chunksize_and_nrows_changing_size(all_parsers):
    # [FUNCTION DOC]
    # Verifies explicit chunk fetching with a row limit of five: fetching sizes two and
    # four returns the first two and next three rows, and a further fetch raises a
    # stop-iteration error with an empty message. For pyarrow, verifies the unsupported
    # row-limit error.
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0, "nrows": 5}

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return

    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=8, **kwargs) as reader:
        tm.assert_frame_equal(reader.get_chunk(size=2), expected.iloc[:2])
        tm.assert_frame_equal(reader.get_chunk(size=4), expected.iloc[2:5])

        with pytest.raises(StopIteration, match="^$"):
            reader.get_chunk(size=3)


def test_get_chunk_passed_chunksize(all_parsers):
    # [FUNCTION DOC]
    # Verifies that fetching a chunk without a size uses the chunk size passed at open
    # (two), returning the first two rows. For pyarrow, verifies the unsupported chunk
    # size error.
    parser = all_parsers
    data = """A,B,C
1,2,3
4,5,6
7,8,9
1,2,3"""

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), chunksize=2) as reader:
                reader.get_chunk()
        return

    with parser.read_csv(StringIO(data), chunksize=2) as reader:
        result = reader.get_chunk()

    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{}, {"index_col": 0}])
def test_read_chunksize_compat(all_parsers, kwargs):
    # [FUNCTION DOC]
    # Verifies concatenating the chunks of a chunked read (size two) equals the
    # unchunked read, with and without an index column. For pyarrow, verifies the
    # unsupported chunk size error.
    # see gh-12185
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
                concat(reader)
        return

    with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
        via_reader = concat(reader)
    tm.assert_frame_equal(via_reader, result)


def test_read_chunksize_jagged_names(all_parsers):
    # [FUNCTION DOC]
    # Verifies reading rows of unequal widths (seven rows of one field and one row of
    # ten) with ten explicit column names in chunks of four pads missing fields with NaN
    # and yields the expected combined frame. For pyarrow, verifies the unsupported
    # chunk size error.
    # see gh-23509
    parser = all_parsers
    data = "\n".join(["0"] * 7 + [",".join(["0"] * 10)])

    expected = DataFrame([[0] + [np.nan] * 9] * 7 + [[0] * 10])

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(
                StringIO(data), names=range(10), chunksize=4
            ) as reader:
                concat(reader)
        return

    with parser.read_csv(StringIO(data), names=range(10), chunksize=4) as reader:
        result = concat(reader)
    tm.assert_frame_equal(result, expected)


def test_chunk_begins_with_newline_whitespace(all_parsers):
    # [FUNCTION DOC]
    # Verifies that data starting with a blank line followed by a line with leading
    # whitespace is read with no header as a single column with the leading space
    # preserved and the blank line skipped.
    # see gh-10022
    parser = all_parsers
    data = "\n hello\nworld\n"

    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([" hello", "world"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
def test_chunks_have_consistent_numerical_type(all_parsers, monkeypatch):
    # [FUNCTION DOC]
    # Marked slow. Builds a one-column input of integers followed by float-formatted
    # values and more integers, patches the parser's internal buffer-size heuristic to a
    # small value so the data is parsed in several internal chunks, and verifies the
    # result is coerced to a float column of numpy floats without any warning.
    # mainly an issue with the C parser
    heuristic = 2**3
    parser = all_parsers
    integers = [str(i) for i in range(heuristic - 1)]
    data = "a\n" + "\n".join([*integers, "1.0", "2.0", *integers])

    # Coercions should work without warnings.
    with monkeypatch.context() as m:
        m.setattr(libparsers, "DEFAULT_BUFFER_HEURISTIC", heuristic)
        result = parser.read_csv(StringIO(data))

    assert type(result.a[0]) is np.float64
    assert result.a.dtype == float


def test_warn_if_chunks_have_mismatched_type(
    all_parsers, using_infer_string, monkeypatch
):
    # [FUNCTION DOC]
    # Verifies that when internal chunks of a column have different types that cannot be
    # coerced numerically, the C engine with low memory mode emits a dtype warning
    # saying the column has mixed types, and the column is object dtype. Other engines
    # emit no warning; the column is a string dtype when string inference is enabled,
    # otherwise object.
    # Patches the buffer heuristic for the non-pyarrow engines and uses a smaller data
    # size than the heuristic in the warning case.
    warning_type = None
    parser = all_parsers
    heuristic = 2**3
    size = 10

    # see gh-3866: if chunks are different types and can't
    # be coerced using numerical types, then issue warning.
    if parser.engine == "c" and parser.low_memory:
        warning_type = DtypeWarning
        # Use a size to hit warning path dictated by DEFAULT_BUFFER_HEURISTIC
        # monkeypatched below
        size = heuristic - 1

    integers = [str(i) for i in range(size)]
    data = "a\n" + "\n".join([*integers, "a", "b", *integers])

    buf = StringIO(data)

    if parser.engine == "pyarrow":
        df = parser.read_csv(
            buf,
        )
    else:
        with monkeypatch.context() as m:
            m.setattr(libparsers, "DEFAULT_BUFFER_HEURISTIC", heuristic)
            df = parser.read_csv_check_warnings(
                warning_type,
                r"Columns \(0: a\) have mixed types. "
                "Specify dtype option on import or set low_memory=False.",
                buf,
            )
    if parser.engine == "c" and parser.low_memory:
        assert df.a.dtype == object
    elif using_infer_string:
        assert df.a.dtype == "str"
    else:
        assert df.a.dtype == object


@pytest.mark.parametrize("iterator", [True, False])
def test_empty_with_nrows_chunksize(all_parsers, iterator):
    # [FUNCTION DOC]
    # Verifies that a header-only input read with a row limit, or iterated with a chunk
    # size, gives an empty frame with the two header columns. For pyarrow, verifies the
    # unsupported-option error for either option.
    # see gh-9535
    parser = all_parsers
    expected = DataFrame(columns=["foo", "bar"])

    nrows = 10
    data = StringIO("foo,bar\n")

    if parser.engine == "pyarrow":
        msg = (
            "The '(nrows|chunksize)' option is not supported with the 'pyarrow' engine"
        )
        with pytest.raises(ValueError, match=msg):
            if iterator:
                with parser.read_csv(data, chunksize=nrows) as reader:
                    next(iter(reader))
            else:
                parser.read_csv(data, nrows=nrows)
        return

    if iterator:
        with parser.read_csv(data, chunksize=nrows) as reader:
            result = next(iter(reader))
    else:
        result = parser.read_csv(data, nrows=nrows)

    tm.assert_frame_equal(result, expected)


def test_chunksize_with_usecols_second_block_shorter(all_parsers):
    # [FUNCTION DOC]
    # Verifies chunked reading with no header, two names, a column selection of the
    # first two columns and chunk size two over data whose last row is shorter: the
    # chunks equal the expected two-row frame and a one-row frame with index two. For
    # pyarrow, verifies the unsupported chunk size error.
    # GH#21211
    parser = all_parsers
    data = """1,2,3,4
5,6,7,8
9,10,11
"""

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                names=["a", "b"],
                chunksize=2,
                usecols=[0, 1],
                header=None,
            )
        return

    result_chunks = parser.read_csv(
        StringIO(data),
        names=["a", "b"],
        chunksize=2,
        usecols=[0, 1],
        header=None,
    )

    expected_frames = [
        DataFrame({"a": [1, 5], "b": [2, 6]}),
        DataFrame({"a": [9], "b": [10]}, index=[2]),
    ]

    for i, result in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])


def test_chunksize_second_block_shorter(all_parsers):
    # [FUNCTION DOC]
    # Verifies chunked reading with chunk size two over data whose last row has one
    # fewer field: the second chunk fills the missing field with NaN. For pyarrow,
    # verifies the unsupported chunk size error.
    # GH#21211
    parser = all_parsers
    data = """a,b,c,d
1,2,3,4
5,6,7,8
9,10,11
"""

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), chunksize=2)
        return

    result_chunks = parser.read_csv(StringIO(data), chunksize=2)

    expected_frames = [
        DataFrame({"a": [1, 5], "b": [2, 6], "c": [3, 7], "d": [4, 8]}),
        DataFrame({"a": [9], "b": [10], "c": [11], "d": [np.nan]}, index=[2]),
    ]

    for i, result in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])
