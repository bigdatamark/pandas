"""
Tests column conversion functionality during parsing
for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of the converters option: validation
# of the argument type, converters keyed by name or position, interplay with numeric
# inference, locale-style decimal formats, missing values, index columns and multi-level
# headers.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# The pyarrow engine does not support converters, so tests expect a value error naming
# the option for that engine.

from io import StringIO

from dateutil.parser import parse
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


def test_converters_type_must_be_dict(all_parsers):
    # [FUNCTION DOC]
    # Verifies that passing a non-mapping (an integer) as converters raises a type error
    # starting with the words Type converters (pyarrow: unsupported converters option
    # value error).
    parser = all_parsers
    data = """index,A,B,C,D
foo,2,3,4,5
"""
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters=0)
        return
    with pytest.raises(TypeError, match="Type converters.+"):
        parser.read_csv(StringIO(data), converters=0)


@pytest.mark.parametrize("column", [3, "D"])
@pytest.mark.parametrize(
    "converter",
    [parse, lambda x: int(x.split("/")[2])],  # Produce integer.
)
def test_converters(all_parsers, column, converter):
    # [FUNCTION DOC]
    # Verifies converters given for the date column by position or name, using either a
    # date-string parser or a function extracting an integer from the date text, produce
    # the same result as mapping the converter over the unconverted column. For pyarrow,
    # verifies the unsupported converters error.
    parser = all_parsers
    data = """A,B,C,D
a,1,2,01/01/2009
b,3,4,01/02/2009
c,4,5,01/03/2009
"""
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters={column: converter})
        return

    result = parser.read_csv(StringIO(data), converters={column: converter})

    expected = parser.read_csv(StringIO(data))
    expected["D"] = expected["D"].map(converter)

    tm.assert_frame_equal(result, expected)


def test_converters_no_implicit_conv(all_parsers):
    # [FUNCTION DOC]
    # Verifies a converter on the first column (stripping whitespace) leaves the values
    # as strings with leading zeros preserved rather than letting numeric inference run
    # on the converted values, while other columns are inferred. For pyarrow, verifies
    # the unsupported converters error.
    # see gh-2184
    parser = all_parsers
    data = """000102,1.2,A\n001245,2,B"""

    converters = {0: lambda x: x.strip()}

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=None, converters=converters)
        return

    result = parser.read_csv(StringIO(data), header=None, converters=converters)

    # Column 0 should not be casted to numeric and should remain as object.
    expected = DataFrame([["000102", 1.2, "A"], ["001245", 2, "B"]])
    tm.assert_frame_equal(result, expected)


def test_converters_euro_decimal_format(all_parsers):
    # [FUNCTION DOC]
    # Verifies converters that replace a decimal comma with a dot and convert to float,
    # applied to three numeric columns of semicolon-separated data, give the expected
    # float and text columns. For pyarrow, verifies the unsupported converters error.
    # see gh-583
    converters = {}
    parser = all_parsers

    data = """Id;Number1;Number2;Text1;Text2;Number3
1;1521,1541;187101,9543;ABC;poi;4,7387
2;121,12;14897,76;DEF;uyt;0,3773
3;878,158;108013,434;GHI;rez;2,7356"""
    converters["Number1"] = converters["Number2"] = converters["Number3"] = lambda x: (
        float(x.replace(",", "."))
    )

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=";", converters=converters)
        return

    result = parser.read_csv(StringIO(data), sep=";", converters=converters)
    expected = DataFrame(
        [
            [1, 1521.1541, 187101.9543, "ABC", "poi", 4.7387],
            [2, 121.12, 14897.76, "DEF", "uyt", 0.3773],
            [3, 878.158, 108013.434, "GHI", "rez", 2.7356],
        ],
        columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
    )
    tm.assert_frame_equal(result, expected)


def test_converters_corner_with_nans(all_parsers):
    # [FUNCTION DOC]
    # Verifies converters on a score column (ranges averaged, blanks as NaN) and a days
    # column (plus suffix incremented, blanks as NaN), combined with empty-string and
    # None missing-value markers, treat empty fields as missing, and that two equivalent
    # days converters give identical frames. For pyarrow, verifies the unsupported
    # converters error for each.
    # KNOWN ISSUE: the second days converter is an exact duplicate of the first, so
    # comparing the two results does not exercise any different behavior (the sentinel
    # variant presumably once returned a sentinel value).
    parser = all_parsers
    data = """id,score,days
1,2,12
2,2-5,
3,,14+
4,6-12,2"""

    # Example converters.
    def convert_days(x):
        # [FUNCTION DOC]
        # HELPER: converter for the days column. Takes a string, strips whitespace;
        # returns NaN for an empty value; for a value ending with a plus sign returns
        # the integer before it plus one; otherwise returns the integer value.
        x = x.strip()

        if not x:
            return np.nan

        is_plus = x.endswith("+")

        if is_plus:
            x = int(x[:-1]) + 1
        else:
            x = int(x)

        return x

    def convert_days_sentinel(x):
        # [FUNCTION DOC]
        # HELPER: behaves exactly the same as the other days converter (strip, NaN for
        # empty, plus suffix adds one, otherwise integer).
        x = x.strip()

        if not x:
            return np.nan

        is_plus = x.endswith("+")

        if is_plus:
            x = int(x[:-1]) + 1
        else:
            x = int(x)

        return x

    def convert_score(x):
        # [FUNCTION DOC]
        # HELPER: converter for the score column. Takes a string, strips whitespace;
        # returns NaN for empty; for a range with a dash after the first character
        # returns the average of the two integers; otherwise returns the value as a
        # float.
        x = x.strip()

        if not x:
            return np.nan

        if x.find("-") > 0:
            val_min, val_max = map(int, x.split("-"))
            val = 0.5 * (val_min + val_max)
        else:
            val = float(x)

        return val

    results = []

    for day_converter in [convert_days, convert_days_sentinel]:
        if parser.engine == "pyarrow":
            msg = "The 'converters' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(
                    StringIO(data),
                    converters={"score": convert_score, "days": day_converter},
                    na_values=["", None],
                )
            continue

        result = parser.read_csv(
            StringIO(data),
            converters={"score": convert_score, "days": day_converter},
            na_values=["", None],
        )
        assert pd.isna(result["days"][1])
        results.append(result)

    if parser.engine != "pyarrow":
        tm.assert_frame_equal(results[0], results[1])


@pytest.mark.parametrize("conv_f", [lambda x: x, str])
def test_converter_index_col_bug(all_parsers, conv_f):
    # [FUNCTION DOC]
    # Verifies that a converter (identity or string conversion) on the column used as
    # index leaves the index values as strings named A, regardless of numeric inference.
    # For pyarrow, verifies the unsupported converters error.
    # see gh-1835 , GH#40589
    parser = all_parsers
    data = "A;B\n1;2\n3;4"

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), sep=";", index_col="A", converters={"A": conv_f}
            )
        return

    rs = parser.read_csv(
        StringIO(data), sep=";", index_col="A", converters={"A": conv_f}
    )

    xp = DataFrame({"B": [2, 4]}, index=Index(["1", "3"], name="A"))
    tm.assert_frame_equal(rs, xp)


def test_converter_identity_object(all_parsers):
    # [FUNCTION DOC]
    # Verifies an identity converter on a numeric-looking column leaves the values as
    # strings (object column) while the other column is inferred as integer. For
    # pyarrow, verifies the unsupported converters error.
    # GH#40589
    parser = all_parsers
    data = "A,B\n1,2\n3,4"

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters={"A": lambda x: x})
        return

    rs = parser.read_csv(StringIO(data), converters={"A": lambda x: x})

    xp = DataFrame({"A": ["1", "3"], "B": [2, 4]})
    tm.assert_frame_equal(rs, xp)


def test_converter_multi_index(all_parsers):
    # [FUNCTION DOC]
    # Verifies converters keyed by tuples with a two-row header: numpy dtype
    # constructors used as converters give int32, int32 and float32 columns. For
    # pyarrow, verifies the unsupported converters error.
    # GH 42446
    parser = all_parsers
    data = "A,B,B\nX,Y,Z\n1,2,3"

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                header=list(range(2)),
                converters={
                    ("A", "X"): np.int32,
                    ("B", "Y"): np.int32,
                    ("B", "Z"): np.float32,
                },
            )
        return

    result = parser.read_csv(
        StringIO(data),
        header=list(range(2)),
        converters={
            ("A", "X"): np.int32,
            ("B", "Y"): np.int32,
            ("B", "Z"): np.float32,
        },
    )

    expected = DataFrame(
        {
            ("A", "X"): np.int32([1]),
            ("B", "Y"): np.int32([2]),
            ("B", "Z"): np.float32([3]),
        }
    )

    tm.assert_frame_equal(result, expected)
