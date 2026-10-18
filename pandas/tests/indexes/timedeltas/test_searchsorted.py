# [MODULE DOC]
# Tests for searchsorted on the timedelta index and its underlying array.

import numpy as np
import pytest

from pandas import (
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm


class TestSearchSorted:
    # [CLASS DOC]
    # Test class for searchsorted.
    def test_searchsorted_different_argument_classes(self, listlike_box):
        # [FUNCTION DOC]
        # Uses the listlike_box fixture. Searching an index for its own elements wrapped in different containers returns the positions zero to length minus one (and the same for the underlying array).
        idx = TimedeltaIndex(["1 day", "2 days", "3 days"])
        result = idx.searchsorted(listlike_box(idx))
        expected = np.arange(len(idx), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = idx._data.searchsorted(listlike_box(idx))
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
    )
    def test_searchsorted_invalid_argument_dtype(self, arg):
        # [FUNCTION DOC]
        # Parametrized over integers, strings and tz-aware timestamps. Searching with them raises a type error: value should be a timedelta, NaT or array of those.
        idx = TimedeltaIndex(["1 day", "2 days", "3 days"])
        msg = "value should be a 'Timedelta', 'NaT', or array of those. Got"
        with pytest.raises(TypeError, match=msg):
            idx.searchsorted(arg)
