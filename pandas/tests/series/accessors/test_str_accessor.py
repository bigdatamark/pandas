# [MODULE DOC]
# Tests for the string accessor of Series: delegation of strip methods, error on
# non-string data, and updating after in-place modification.
import pytest

from pandas import Series
import pandas._testing as tm


class TestStrAccessor:
    # [CLASS DOC]
    # Groups the Series string accessor tests. Stateless.
    def test_str_attribute(self):
        # [FUNCTION DOC]
        # Regression test (GH 9068). For the strip, right-strip and left-strip methods,
        # calling the accessor class's method with the accessor equals the python str
        # method applied to each value. An integer series raises an attribute error
        # saying the accessor needs string values and naming the integer inference.
        # GH#9068
        methods = ["strip", "rstrip", "lstrip"]
        ser = Series([" jack", "jill ", " jesse ", "frank"])
        for method in methods:
            expected = Series([getattr(str, method)(x) for x in ser.values])
            tm.assert_series_equal(getattr(Series.str, method)(ser.str), expected)

        # str accessor only valid with string values
        ser = Series(range(5))
        msg = "Can only use .str accessor with string values, not integer"
        with pytest.raises(AttributeError, match=msg):
            ser.str.repeat(2)

    def test_str_accessor_updates_on_inplace(self):
        # [FUNCTION DOC]
        # Verifies the string accessor reflects an in-place drop: after dropping one
        # element in place (returns nothing) lowercasing yields two elements.
        ser = Series(list("abc"))
        return_value = ser.drop([0], inplace=True)
        assert return_value is None
        assert len(ser.str.lower()) == 2
