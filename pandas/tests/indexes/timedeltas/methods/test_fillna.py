# [MODULE DOC]
# Tests for filling missing values in a timedelta index.

from pandas import (
    Index,
    NaT,
    Timedelta,
    TimedeltaIndex,
)
import pandas._testing as tm


class TestFillNA:
    # [CLASS DOC]
    # Test class for fillna on timedelta indexes.
    def test_fillna_timedelta(self):
        # [FUNCTION DOC]
        # Filling the missing value of a three-element index with a timedelta gives the filled timedelta index. Filling with a string gives an object index containing the string in place of the missing value.
        # KNOWN ISSUE: the second step computes an expected index for a three-hour fill but never calls the equality assertion (the fill result is discarded), so that case is not verified.
        # GH#11343
        idx = TimedeltaIndex(["1 day", NaT, "3 day"])

        exp = TimedeltaIndex(["1 day", "2 day", "3 day"])
        tm.assert_index_equal(idx.fillna(Timedelta("2 day")), exp)

        exp = TimedeltaIndex(["1 day", "3 hour", "3 day"])
        idx.fillna(Timedelta("3 hour"))

        exp = Index([Timedelta("1 day"), "x", Timedelta("3 day")], dtype=object)
        tm.assert_index_equal(idx.fillna("x"), exp)
