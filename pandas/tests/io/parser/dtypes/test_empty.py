"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests of dtype handling when the input has a header but no data rows: string
# dtype for all columns, per-column dtypes, dtypes for index columns, duplicate and
# mangled column names with dtypes, and a table of dtype specifications for an empty
# frame.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# Uses the marker alias that skips tests for the pyarrow engine.

from io import StringIO

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm

skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_dtype_all_columns_empty(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies header-only input read with string dtype gives an
    # empty frame with columns A and B of string dtype.
    # see gh-12048
    parser = all_parsers
    result = parser.read_csv(StringIO("A,B"), dtype=str)

    expected = DataFrame({"A": [], "B": []}, dtype=str)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_pass_dtype(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies header-only input with an unsigned byte dtype for
    # one column gives an empty frame where that column has the requested dtype and the
    # other defaults to object.
    parser = all_parsers

    data = "one,two"
    result = parser.read_csv(StringIO(data), dtype={"one": "u1"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "two": np.empty(0, dtype=object)},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_index_pass_dtype(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies header-only input with the first column as index and
    # dtypes given by name for the index and by position for the other column yields an
    # empty frame with an unsigned byte index named one and a float32 column.
    parser = all_parsers

    data = "one,two"
    result = parser.read_csv(
        StringIO(data), index_col=["one"], dtype={"one": "u1", 1: "f"}
    )

    expected = DataFrame(
        {"two": np.empty(0, dtype="f")}, index=Index([], dtype="u1", name="one")
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_multi_index_pass_dtype(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Same as the previous with two index columns: the index levels
    # get unsigned byte and float64 dtypes and the remaining column is object.
    parser = all_parsers

    data = "one,two,three"
    result = parser.read_csv(
        StringIO(data), index_col=["one", "two"], dtype={"one": "u1", 1: "f8"}
    )

    exp_idx = MultiIndex.from_arrays(
        [np.empty(0, dtype="u1"), np.empty(0, dtype=np.float64)],
        names=["one", "two"],
    )
    expected = DataFrame({"three": np.empty(0, dtype=object)}, index=exp_idx)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_mangled_column_pass_dtype_by_names(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that for a duplicated header name the second column
    # is renamed with a numeric suffix and dtypes given by the original and mangled
    # names are applied.
    parser = all_parsers

    data = "one,one"
    result = parser.read_csv(StringIO(data), dtype={"one": "u1", "one.1": "f"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "one.1": np.empty(0, dtype="f")},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_mangled_column_pass_dtype_by_indexes(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies dtypes given by position are applied to the mangled
    # duplicate columns of a header-only input.
    parser = all_parsers

    data = "one,one"
    result = parser.read_csv(StringIO(data), dtype={0: "u1", 1: "f"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "one.1": np.empty(0, dtype="f")},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_dup_column_pass_dtype_by_indexes(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies, via a reference frame built by concatenating two
    # empty typed series, that position-based dtypes are applied to duplicate header
    # columns (regression for an older issue).
    # see gh-9424
    parser = all_parsers
    expected = concat(
        [Series([], name="one", dtype="u1"), Series([], name="one.1", dtype="f")],
        axis=1,
    )

    data = "one,one"
    result = parser.read_csv(StringIO(data), dtype={0: "u1", 1: "f"})
    tm.assert_frame_equal(result, expected)


def test_empty_with_dup_column_pass_dtype_by_indexes_raises(all_parsers):
    # [FUNCTION DOC]
    # Verifies that passing duplicate names explicitly with position-based dtypes for
    # empty input raises a value error about duplicate names.
    # KNOWN ISSUE: the reference frame built at the start (concatenated empty series
    # with an object-cast index) is never used by any assertion; it is dead code left
    # over from the sibling test.
    # see gh-9424
    parser = all_parsers
    expected = concat(
        [Series([], name="one", dtype="u1"), Series([], name="one.1", dtype="f")],
        axis=1,
    )
    expected.index = expected.index.astype(object)

    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(""), names=["one", "one"], dtype={0: "u1", 1: "f"})


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.float64, DataFrame(columns=["a", "b"], dtype=np.float64)),
        (
            "category",
            DataFrame({"a": Categorical([]), "b": Categorical([])}),
        ),
        (
            {"a": "category", "b": "category"},
            DataFrame({"a": Categorical([]), "b": Categorical([])}),
        ),
        ("datetime64[ns]", DataFrame(columns=["a", "b"], dtype="datetime64[ns]")),
        (
            "timedelta64[ns]",
            DataFrame(
                {
                    "a": Series([], dtype="timedelta64[ns]"),
                    "b": Series([], dtype="timedelta64[ns]"),
                },
            ),
        ),
        (
            {"a": np.int64, "b": np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),
        (
            {0: np.int64, 1: np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),
        (
            {"a": np.int64, 1: np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),
    ],
)
@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_dtype(all_parsers, dtype, expected):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Parametrized table of dtype specifications (a single numeric
    # type, category as string or mapping, datetime and timedelta types, and mappings by
    # name, by position or mixed) applied to header-only input; verifies the empty frame
    # has the requested dtype for each column.
    # see gh-14712
    parser = all_parsers
    data = "a,b"

    result = parser.read_csv(StringIO(data), header=0, dtype=dtype)
    tm.assert_frame_equal(result, expected)
