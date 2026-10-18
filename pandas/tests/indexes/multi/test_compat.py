# [MODULE DOC]
# Tests for numeric and logical operation compatibility of MultiIndex (all
# unsupported, raising type errors), cache behavior of the value array when
# levels or codes are replaced, and boxing of categorical values.
# Uses the shared idx fixture. Pure tests module.

import numpy as np
import pytest

import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm


def test_numeric_compat(idx):
    # [FUNCTION DOC]
    # Verifies multiplication, true division and floor division of a MultiIndex by a
    # number (and reflected forms) raise a type error naming the operator.
    with pytest.raises(TypeError, match="cannot perform __mul__"):
        idx * 1

    with pytest.raises(TypeError, match="cannot perform __rmul__"):
        1 * idx

    div_err = "cannot perform __truediv__"
    with pytest.raises(TypeError, match=div_err):
        idx / 1

    div_err = div_err.replace(" __", " __r")
    with pytest.raises(TypeError, match=div_err):
        1 / idx

    with pytest.raises(TypeError, match="cannot perform __floordiv__"):
        idx // 1

    with pytest.raises(TypeError, match="cannot perform __rfloordiv__"):
        1 // idx


@pytest.mark.parametrize("method", ["all", "any", "__invert__"])
def test_logical_compat(idx, method):
    # [FUNCTION DOC]
    # Parametrized over all, any and invert. Verifies each raises a type error stating
    # the method cannot be performed on this index.
    msg = f"cannot perform {method}"

    with pytest.raises(TypeError, match=msg):
        getattr(idx, method)()


def test_inplace_mutation_resets_values():
    # [FUNCTION DOC]
    # Verifies the value-array cache lifecycle: construction does not populate the
    # cached values; accessing the values caches a numpy array; replacing levels
    # returns new values but leaves the original cache intact; replacing codes returns
    # a new index with an empty cache that fills on access, with the expected object
    # array of tuples; and the original cache is unchanged.
    levels = [["a", "b", "c"], [4]]
    levels2 = [[1, 2, 3], ["a"]]
    codes = [[0, 1, 0, 2, 2, 0], [0, 0, 0, 0, 0, 0]]

    mi1 = MultiIndex(levels=levels, codes=codes)
    mi2 = MultiIndex(levels=levels2, codes=codes)

    # instantiating MultiIndex should not access/cache _.values
    assert "_values" not in mi1._cache
    assert "_values" not in mi2._cache

    vals = mi1.values.copy()
    vals2 = mi2.values.copy()

    # accessing .values should cache ._values
    assert mi1._values is mi1._cache["_values"]
    assert mi1.values is mi1._cache["_values"]
    assert isinstance(mi1._cache["_values"], np.ndarray)

    # Make sure level setting works
    new_vals = mi1.set_levels(levels2).values
    tm.assert_almost_equal(vals2, new_vals)

    #  Doesn't drop _values from _cache [implementation detail]
    tm.assert_almost_equal(mi1._cache["_values"], vals)

    # ...and values is still same too
    tm.assert_almost_equal(mi1.values, vals)

    # Make sure label setting works too
    codes2 = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    exp_values = np.empty((6,), dtype=object)
    exp_values[:] = [(1, "a")] * 6

    # Must be 1d array of tuples
    assert exp_values.shape == (6,)

    new_mi = mi2.set_codes(codes2)
    assert "_values" not in new_mi._cache
    new_values = new_mi.values
    assert "_values" in new_mi._cache

    # Shouldn't change cache
    tm.assert_almost_equal(mi2._cache["_values"], vals2)

    # Should have correct values
    tm.assert_almost_equal(exp_values, new_values)


def test_boxable_categorical_values():
    # [FUNCTION DOC]
    # Verifies the values of a product multi-index with a categorical datetime level
    # are a numpy object array of tuples with Timestamp objects (not categories),
    # and that a frame with categorical and object columns converts to a values array
    # with datetimes unboxed.
    cat = pd.Categorical(pd.date_range("2012-01-01", periods=3, freq="h"))
    result = MultiIndex.from_product([["a", "b", "c"], cat]).values
    expected = pd.Series(
        [
            ("a", pd.Timestamp("2012-01-01 00:00:00")),
            ("a", pd.Timestamp("2012-01-01 01:00:00")),
            ("a", pd.Timestamp("2012-01-01 02:00:00")),
            ("b", pd.Timestamp("2012-01-01 00:00:00")),
            ("b", pd.Timestamp("2012-01-01 01:00:00")),
            ("b", pd.Timestamp("2012-01-01 02:00:00")),
            ("c", pd.Timestamp("2012-01-01 00:00:00")),
            ("c", pd.Timestamp("2012-01-01 01:00:00")),
            ("c", pd.Timestamp("2012-01-01 02:00:00")),
        ]
    ).values
    tm.assert_numpy_array_equal(result, expected)
    result = pd.DataFrame({"a": ["a", "b", "c"], "b": cat, "c": np.array(cat)}).values
    expected = pd.DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": [
                pd.Timestamp("2012-01-01 00:00:00"),
                pd.Timestamp("2012-01-01 01:00:00"),
                pd.Timestamp("2012-01-01 02:00:00"),
            ],
            "c": [
                pd.Timestamp("2012-01-01 00:00:00"),
                pd.Timestamp("2012-01-01 01:00:00"),
                pd.Timestamp("2012-01-01 02:00:00"),
            ],
        }
    ).values
    tm.assert_numpy_array_equal(result, expected)
