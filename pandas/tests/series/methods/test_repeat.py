# [MODULE DOC]
# Tests for Series.repeat and NumPy's repeat on a Series. No import-time side
# effects.

import numpy as np
import pytest

from pandas import (
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestRepeat:
    # [CLASS DOC]
    # Groups the repeat tests. Stateless.
    def test_repeat(self):
        # [FUNCTION DOC]
        # Repeating by a scalar count repeats each value and its label; repeating by a
        # list of counts repeats each element that many times.
        ser = Series(np.random.default_rng(2).standard_normal(3), index=["a", "b", "c"])

        reps = ser.repeat(5)
        exp = Series(ser.values.repeat(5), index=ser.index.values.repeat(5))
        tm.assert_series_equal(reps, exp)

        to_rep = [2, 3, 4]
        reps = ser.repeat(to_rep)
        exp = Series(ser.values.repeat(to_rep), index=ser.index.values.repeat(to_rep))
        tm.assert_series_equal(reps, exp)

    def test_numpy_repeat(self):
        # [FUNCTION DOC]
        # NumPy's repeat function applied to a Series gives the equivalent Series and
        # keeps the name; passing an axis raises a value error saying the axis parameter
        # is not supported.
        ser = Series(np.arange(3), name="x")
        expected = Series(
            ser.values.repeat(2), name="x", index=ser.index.values.repeat(2)
        )
        tm.assert_series_equal(np.repeat(ser, 2), expected)

        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.repeat(ser, 2, axis=0)

    def test_repeat_with_multiindex(self):
        # [FUNCTION DOC]
        # GH#9361 (fixed by GH#7891): repeating a Series with a MultiIndex three times
        # gives a result of three times the length.
        # GH#9361, fixed by  GH#7891
        m_idx = MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6), (7, 8)])
        data = ["a", "b", "c", "d"]
        m_df = Series(data, index=m_idx)
        assert m_df.repeat(3).shape == (3 * len(data),)
