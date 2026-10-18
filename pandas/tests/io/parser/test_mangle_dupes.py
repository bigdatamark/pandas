"""
Tests that duplicate columns are handled appropriately when parsed by the
CSV engine. In general, the expected result is that they are either thoroughly
de-duplicated (if mangling requested) or ignored otherwise.
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of handling duplicated column names:
# de-duplication by appending numeric suffixes, collisions with names that already have
# suffixes, behavior with explicit names, and placeholder names for unnamed columns.
# Defines a marker alias for expected failures with the pyarrow engine.

from io import StringIO

import pytest

from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_basic(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies repeated header names get dot-number
    # suffixes in order of appearance (second a becomes a.1; third b becomes b.2).
    parser = all_parsers

    data = "a,a,b,b,b\n1,2,3,4,5"
    result = parser.read_csv(StringIO(data), sep=",")

    expected = DataFrame([[1, 2, 3, 4, 5]], columns=["a", "a.1", "b", "b.1", "b.2"])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_basic_names(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a duplicate a in the header becomes
    # a.1 when names are not given.
    # See gh-7160
    parser = all_parsers

    data = "a,b,a\n0,1,2\n3,4,5"
    expected = DataFrame([[0, 1, 2], [3, 4, 5]], columns=["a", "b", "a.1"])

    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_basic_names_raise(all_parsers):
    # [FUNCTION DOC]
    # Verifies explicitly passing duplicate names raises a value error saying there are
    # duplicate names.
    # See gh-7160
    parser = all_parsers

    data = "0,1,2\n3,4,5"
    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=["a", "b", "a"])


@xfail_pyarrow  # ValueError: Found non-unique column index
@pytest.mark.parametrize(
    "data,expected",
    [
        ("a,a,a.1\n1,2,3", DataFrame([[1, 2, 3]], columns=["a", "a.2", "a.1"])),
        (
            "a,a,a.1,a.1.1,a.1.1.1,a.1.1.1.1\n1,2,3,4,5,6",
            DataFrame(
                [[1, 2, 3, 4, 5, 6]],
                columns=["a", "a.2", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1"],
            ),
        ),
        (
            "a,a,a.3,a.1,a.2,a,a\n1,2,3,4,5,6,7",
            DataFrame(
                [[1, 2, 3, 4, 5, 6, 7]],
                columns=["a", "a.4", "a.3", "a.1", "a.2", "a.5", "a.6"],
            ),
        ),
    ],
)
def test_thorough_mangle_columns(all_parsers, data, expected):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies mangling avoids collisions with
    # header entries that already look mangled (a second a becomes a.2 because a.1
    # exists; deeper suffix chains and later duplicates get the next free suffix).
    # see gh-17060
    parser = all_parsers

    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,names,expected",
    [
        (
            "a,b,b\n1,2,3",
            ["a.1", "a.1", "a.1.1"],
            DataFrame(
                [["a", "b", "b"], ["1", "2", "3"]], columns=["a.1", "a.1.1", "a.1.1.1"]
            ),
        ),
        (
            "a,b,c,d,e,f\n1,2,3,4,5,6",
            ["a", "a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1"],
            DataFrame(
                [["a", "b", "c", "d", "e", "f"], ["1", "2", "3", "4", "5", "6"]],
                columns=["a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1", "a.1.1.1.1.1"],
            ),
        ),
        (
            "a,b,c,d,e,f,g\n1,2,3,4,5,6,7",
            ["a", "a", "a.3", "a.1", "a.2", "a", "a"],
            DataFrame(
                [
                    ["a", "b", "c", "d", "e", "f", "g"],
                    ["1", "2", "3", "4", "5", "6", "7"],
                ],
                columns=["a", "a.1", "a.3", "a.1.1", "a.2", "a.2.1", "a.3.1"],
            ),
        ),
    ],
)
def test_thorough_mangle_names(all_parsers, data, names, expected):
    # [FUNCTION DOC]
    # Verifies that explicit names with duplicates, including ones resembling suffixed
    # names, raise a value error about duplicate names.
    # KNOWN ISSUE: the expected frame parameter is never used; the test only checks
    # that the error is raised, so the expected mangled columns in the parametrization
    # are dead data.
    # see gh-17095
    parser = all_parsers

    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=names)


@xfail_pyarrow  # AssertionError: DataFrame.columns are different
def test_mangled_unnamed_placeholders(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Repeatedly writes a frame to CSV text (with
    # its index as an unnamed column) and reads it back three times, verifying each
    # round adds another unnamed index column whose placeholder name gains one more
    # numeric suffix level, so placeholders are mangled consistently.
    # xref gh-13017
    orig_key = "0"
    parser = all_parsers

    orig_value = [1, 2, 3]
    df = DataFrame({orig_key: orig_value})

    # This test recursively updates `df`.
    for i in range(3):
        expected = DataFrame(columns=Index([], dtype="str"))

        for j in range(i + 1):
            col_name = "Unnamed: 0" + f".{1 * j}" * min(j, 1)
            expected.insert(loc=0, column=col_name, value=[0, 1, 2])

        expected[orig_key] = orig_value
        df = parser.read_csv(StringIO(df.to_csv()))

        tm.assert_frame_equal(df, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_mangle_dupe_cols_already_exists(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies mangling of repeated header names
    # when some mangled forms are already present (a.1, a.3, a.1.1), giving unique names
    # with the next free numbers.
    # GH#14704
    parser = all_parsers

    data = "a,a,a.1,a,a.3,a.1,a.1.1\n1,2,3,4,5,6,7"
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6, 7]],
        columns=["a", "a.2", "a.1", "a.4", "a.3", "a.1.2", "a.1.1"],
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_mangle_dupe_cols_already_exists_unnamed_col(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies empty header entries and entries
    # already named as placeholders are mangled without collisions.
    # GH#14704
    parser = all_parsers

    data = ",Unnamed: 0,,Unnamed: 2\n1,2,3,4"
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [[1, 2, 3, 4]],
        columns=["Unnamed: 0.1", "Unnamed: 0", "Unnamed: 2.1", "Unnamed: 2"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("usecol, engine", [([0, 1, 1], "python"), ([0, 1, 1], "c")])
def test_mangle_cols_names(all_parsers, usecol, engine):
    # [FUNCTION DOC]
    # Verifies explicit duplicate names combined with a column selection that repeats an
    # index raise a value error about duplicate names.
    # KNOWN ISSUE: the engine value is passed on as a keyword argument but the parser
    # fixture's wrapper always overrides the engine with its own, and both
    # parametrizations differ only in that ignored value.
    # GH 11823
    parser = all_parsers
    data = "1,2,3"
    names = ["A", "A", "B"]
    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=names, usecols=usecol, engine=engine)
