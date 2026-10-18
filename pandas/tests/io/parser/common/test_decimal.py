"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# [MODULE DOC]
# Parser tests for the thousands separator and decimal point options, including
# European-style number formats.

from io import StringIO

import pytest

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.mark.parametrize(
    "data,thousands,decimal",
    [
        (
            """A|B|C
1|2,334.01|5
10|13|10.
""",
            ",",
            ".",
        ),
        (
            """A|B|C
1|2.334,01|5
10|13|10,
""",
            ".",
            ",",
        ),
    ],
)
def test_1000_sep_with_decimal(all_parsers, data, thousands, decimal):
    # [FUNCTION DOC]
    # Verifies that pipe-separated data with a thousands separator and decimal marker
    # (comma-thousands with dot-decimal, and dot-thousands with comma-decimal) parses
    # into the expected numeric columns, including a trailing decimal mark treated as a
    # whole number. For pyarrow, verifies a value error that the thousands option is
    # unsupported.
    parser = all_parsers
    expected = DataFrame({"A": [1, 10], "B": [2334.01, 13], "C": [5, 10.0]})

    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), sep="|", thousands=thousands, decimal=decimal
            )
        return

    result = parser.read_csv(
        StringIO(data), sep="|", thousands=thousands, decimal=decimal
    )
    tm.assert_frame_equal(result, expected)


def test_euro_decimal_format(all_parsers):
    # [FUNCTION DOC]
    # Verifies that semicolon-separated data using a comma as decimal marker is read
    # with that decimal option into integer, float and string columns as expected.
    parser = all_parsers
    data = """Id;Number1;Number2;Text1;Text2;Number3
1;1521,1541;187101,9543;ABC;poi;4,738797819
2;121,12;14897,76;DEF;uyt;0,377320872
3;878,158;108013,434;GHI;rez;2,735694704"""

    result = parser.read_csv(StringIO(data), sep=";", decimal=",")
    expected = DataFrame(
        [
            [1, 1521.1541, 187101.9543, "ABC", "poi", 4.738797819],
            [2, 121.12, 14897.76, "DEF", "uyt", 0.377320872],
            [3, 878.158, 108013.434, "GHI", "rez", 2.735694704],
        ],
        columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
    )
    tm.assert_frame_equal(result, expected)
