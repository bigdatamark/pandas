# [MODULE DOC]
# Tests for the internal extract_array helper in pandas.core.construction applied
# to a RangeIndex. No import-time side effects.

from pandas import Index
import pandas._testing as tm
from pandas.core.construction import extract_array


def test_extract_array_rangeindex():
    # [FUNCTION DOC]
    # Verifies extract_array on a RangeIndex returns the numpy values when the
    # range-extraction flag is true (whatever the numpy-extraction flag), and returns
    # the index itself when the range-extraction flag is false.
    ri = Index(range(5))

    expected = ri._values
    res = extract_array(ri, extract_numpy=True, extract_range=True)
    tm.assert_numpy_array_equal(res, expected)
    res = extract_array(ri, extract_numpy=False, extract_range=True)
    tm.assert_numpy_array_equal(res, expected)

    res = extract_array(ri, extract_numpy=True, extract_range=False)
    tm.assert_index_equal(res, ri)
    res = extract_array(ri, extract_numpy=False, extract_range=False)
    tm.assert_index_equal(res, ri)
