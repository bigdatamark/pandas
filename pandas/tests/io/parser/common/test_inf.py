"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# [MODULE DOC]
# Parser tests for parsing infinity text in numeric columns, with and without the
# missing-value filter. Defines a marker alias for tests expected to fail for the
# pyarrow engine.

from io import StringIO

import pytest

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@xfail_pyarrow  # AssertionError: DataFrame.index are different
@pytest.mark.parametrize("na_filter", [True, False])
def test_inf_parsing(all_parsers, na_filter):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that spellings of infinity in
    # different cases and signs (inf, Inf, INF, mixed, with plus or minus) are all
    # parsed as float infinities alternating positive and negative, with the first
    # column as index, whether or not the missing-value filter is on.
    parser = all_parsers
    data = """\
,A
a,inf
b,-inf
c,+Inf
d,-Inf
e,INF
f,-INF
g,+INf
h,-INf
i,inF
j,-inF"""
    expected = DataFrame(
        {"A": [float("inf"), float("-inf")] * 5},
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # AssertionError: DataFrame.index are different
@pytest.mark.parametrize("na_filter", [True, False])
def test_infinity_parsing(all_parsers, na_filter):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that the long spelling of infinity,
    # with no sign, minus and plus, is parsed to positive and negative infinity,
    # regardless of the missing-value filter setting.
    parser = all_parsers
    data = """\
,A
a,Infinity
b,-Infinity
c,+Infinity
"""
    expected = DataFrame(
        {"A": [float("infinity"), float("-infinity"), float("+infinity")]},
        index=["a", "b", "c"],
    )
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)
