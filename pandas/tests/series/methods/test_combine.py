# [MODULE DOC]
# Tests for Series.combine, which merges two Series (or a Series and a scalar)
# by a binary function. No import-time side effects.

from pandas import (
    NA,
    Series,
)
import pandas._testing as tm


class TestCombine:
    # [CLASS DOC]
    # Groups the tests of Series.combine. Stateless.
    def test_combine_scalar(self):
        # [FUNCTION DOC]
        # Regression test (GH#21248): combining a Series with a scalar applies the
        # function with the scalar as second argument for every element, for both an
        # addition and a minimum function.
        # GH#21248
        ser = Series([i * 10 for i in range(5)])
        result = ser.combine(3, lambda x, y: x + y)
        expected = Series([i * 10 + 3 for i in range(5)])
        tm.assert_series_equal(result, expected)

        result = ser.combine(22, lambda x, y: min(x, y))
        expected = Series([min(i * 10, 22) for i in range(5)])
        tm.assert_series_equal(result, expected)

    def test_combine_series(self):
        # [FUNCTION DOC]
        # Regression test (GH#31899): combining two Int8 nullable Series with addition
        # keeps the Int8 dtype and propagates NA; 91 plus 91 wraps around in 8-bit
        # arithmetic to -74.
        # GH#31899
        s1 = Series([91, NA, 94], dtype="Int8")
        s2 = Series([91, NA, 11], dtype="Int8")
        result = s1.combine(s2, lambda x, y: x + y)
        expected = Series([-74, NA, 105], dtype="Int8")  # dtype should be preserved
        tm.assert_series_equal(result, expected)
