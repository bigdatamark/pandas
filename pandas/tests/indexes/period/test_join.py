# [MODULE DOC]
# Tests for joining PeriodIndex objects: the private outer indexer, join types,
# self joins, mixed-type joins and mismatched frequencies.
import numpy as np

from pandas import (
    DataFrame,
    Index,
    PeriodIndex,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestJoin:
    # [CLASS DOC]
    # Groups join tests.
    def test_join_outer_indexer(self):
        # [FUNCTION DOC]
        # The private outer indexer on an index with itself returns the values and two
        # identity position arrays.
        pi = period_range("1/1/2000", "1/20/2000", freq="D")

        result = pi._outer_indexer(pi)
        tm.assert_extension_array_equal(result[0], pi._values)
        tm.assert_numpy_array_equal(result[1], np.arange(len(pi), dtype=np.intp))
        tm.assert_numpy_array_equal(result[2], np.arange(len(pi), dtype=np.intp))

    def test_joins(self, join_type):
        # [FUNCTION DOC]
        # Parametrized over the join-type fixture. Joining an index with a truncated
        # version of itself returns a PeriodIndex of unchanged frequency.
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        joined = index.join(index[:-5], how=join_type)

        assert isinstance(joined, PeriodIndex)
        assert joined.freq == index.freq

    def test_join_self(self, join_type):
        # [FUNCTION DOC]
        # Parametrized over the join-type fixture. Joining an index with itself returns
        # the very same object.
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        res = index.join(index, how=join_type)
        assert index is res

    def test_join_does_not_recur(self):
        # [FUNCTION DOC]
        # Joining a date index with a period index of columns (outer) yields an object
        # index of all four labels without infinite recursion.
        df = DataFrame(
            np.ones((3, 2)),
            index=date_range("2020-01-01", periods=3),
            columns=period_range("2020-01-01", periods=2),
        )
        ser = df.iloc[:2, 0]

        res = ser.index.join(df.columns, how="outer")
        expected = Index(
            [ser.index[0], ser.index[1], df.columns[0], df.columns[1]], object
        )
        tm.assert_index_equal(res, expected)

    def test_join_mismatched_freq_raises(self):
        # [FUNCTION DOC]
        # Joining indexes of different frequencies no longer raises; it falls back to
        # joining as object dtype and gives the same result as joining the object
        # versions.
        # KNOWN ISSUE: the name says it raises, but the comment states the raising behavior was changed and the test asserts a successful fallback.
        # pre-GH#55782 this raises IncompatibleFrequency
        index = period_range("1/1/2000", "1/20/2000", freq="D")
        index3 = period_range("1/1/2000", "1/20/2000", freq="2D")
        result = index.join(index3)
        expected = index.astype(object).join(index3.astype(object))
        tm.assert_index_equal(result, expected)
