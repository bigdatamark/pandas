# [MODULE DOC]
# Tests for the Series method converting to a mapping, checking the mapping type used and the Python types of returned values.
# No import-time side effects.

import collections

import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


class TestSeriesToDict:
    # [CLASS DOC]
    # Groups tests of converting a Series to a dictionary-like object. Stateless.
    @pytest.mark.parametrize(
        "mapping", (dict, collections.defaultdict(list), collections.OrderedDict)
    )
    def test_to_dict(self, mapping, datetime_series):
        # [FUNCTION DOC]
        # Verifies the conversion honors the target mapping type parameter (plain dict, default-dict instance, ordered dict; issue 16122): rebuilding a Series from the result equals the datetime series with index frequency cleared.
        # Also checks that converting into a counter type gives the same Series as building from a counter of the series' item pairs.
        # GH#16122
        result = Series(datetime_series.to_dict(into=mapping), name="ts")
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)

        from_method = Series(datetime_series.to_dict(into=collections.Counter))
        from_constructor = Series(collections.Counter(datetime_series.items()))
        tm.assert_series_equal(from_method, from_constructor)

    @pytest.mark.parametrize(
        "input",
        (
            {"a": np.int64(64), "b": 10},
            {"a": np.int64(64), "b": 10, "c": "ABC"},
            {"a": np.uint64(64), "b": 10, "c": "ABC"},
        ),
    )
    def test_to_dict_return_types(self, input):
        # [FUNCTION DOC]
        # Verifies that integer values (NumPy signed and unsigned 64-bit and Python ints), even when mixed with strings, are returned as built-in Python integers (issue 25969). Parametrized over three input mappings; asserts on keys "a" and "b".
        # GH25969

        d = Series(input).to_dict()
        assert isinstance(d["a"], int)
        assert isinstance(d["b"], int)
