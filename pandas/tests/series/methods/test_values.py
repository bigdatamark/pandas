# [MODULE DOC]
# Tests for the values attribute of a Series, for extension data and for a datetime series.
# No import-time side effects.

import numpy as np
import pytest

from pandas import (
    IntervalIndex,
    Series,
    period_range,
)
import pandas._testing as tm


class TestValues:
    # [CLASS DOC]
    # Groups tests of the values attribute of a Series. Stateless.
    @pytest.mark.parametrize(
        "data",
        [
            period_range("2000", periods=4),
            IntervalIndex.from_breaks([1, 2, 3, 4]),
        ],
    )
    def test_values_object_extension_dtypes(self, data):
        # [FUNCTION DOC]
        # Verifies that the values attribute of a Series built from a period index or an interval index is an object-dtype NumPy array of the scalar elements (issue 23995). Parametrized over the two index types.
        # https://github.com/pandas-dev/pandas/issues/23995
        result = Series(data).values
        expected = np.array(data.astype(object))
        tm.assert_numpy_array_equal(result, expected)

    def test_values(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that the values of the datetime-indexed fixture series are almost equal to the plain list of its elements, ignoring dtype.
        tm.assert_almost_equal(
            datetime_series.values, list(datetime_series), check_dtype=False
        )
