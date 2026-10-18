# [MODULE DOC]
# Test of the string representation of IntervalArray. No import-time side effects.

from pandas.core.arrays import IntervalArray


def test_repr():
    # [FUNCTION DOC]
    # Repr of an interval array from tuples shows the class header, the intervals in half-open notation, and a line with length and interval dtype (int64, right-closed).
    # GH#25022
    arr = IntervalArray.from_tuples([(0, 1), (1, 2)])
    result = repr(arr)
    expected = (
        "<IntervalArray>\n[(0, 1], (1, 2]]\nLength: 2, dtype: interval[int64, right]"
    )
    assert result == expected
