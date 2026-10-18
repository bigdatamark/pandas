"""
Includes test for last_valid_index.
"""

# [MODULE DOC]
# Tests for first_valid_index and last_valid_index on DataFrame and Series: results for empty objects, all-missing data, duplicate index labels, and leading or trailing missing values.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)


class TestFirstValidIndex:
    # [CLASS DOC]
    # Groups the first_valid_index and last_valid_index tests.
    def test_first_valid_index_single_nan(self, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over frame_or_series. A single missing value (and an empty slice of it) must make both methods return None rather than raise (GH#9752); only first_valid_index is checked.
        # GH#9752 Series/DataFrame should both return None, not raise
        obj = frame_or_series([np.nan])

        assert obj.first_valid_index() is None
        assert obj.iloc[:0].first_valid_index() is None

    @pytest.mark.parametrize(
        "empty", [DataFrame(), Series(dtype=object), Series([], index=[], dtype=object)]
    )
    def test_first_valid_index_empty(self, empty):
        # [FUNCTION DOC]
        # Parametrized over empty objects: an empty frame and empty object-dtype series. Verifies both last_valid_index and first_valid_index return None (GH#12800).
        # GH#12800
        assert empty.last_valid_index() is None
        assert empty.first_valid_index() is None

    @pytest.mark.parametrize(
        "data,idx,expected_first,expected_last",
        [
            ({"A": [1, 2, 3]}, [1, 1, 2], 1, 2),
            ({"A": [1, 2, 3]}, [1, 2, 2], 1, 2),
            ({"A": [1, 2, 3, 4]}, ["d", "d", "d", "d"], "d", "d"),
            ({"A": [1, np.nan, 3]}, [1, 1, 2], 1, 2),
            ({"A": [np.nan, np.nan, 3]}, [1, 1, 2], 2, 2),
            ({"A": [1, np.nan, 3]}, [1, 2, 2], 1, 2),
        ],
    )
    def test_first_last_valid_frame(self, data, idx, expected_first, expected_last):
        # [FUNCTION DOC]
        # Parametrized over data, index labels, and expected first and last labels (GH#21441), including duplicate index labels and missing values.
        # Verifies a frame returns the expected first and last valid labels, with duplicates resolving to the label values listed.
        # GH#21441
        df = DataFrame(data, index=idx)
        assert expected_first == df.first_valid_index()
        assert expected_last == df.last_valid_index()

    @pytest.mark.parametrize(
        "index",
        [Index([str(i) for i in range(20)]), date_range("2020-01-01", periods=20)],
    )
    def test_first_last_valid(self, index):
        # [FUNCTION DOC]
        # Parametrized over a string index and a datetime index of length 20. A single column has its first five and last five entries missing.
        # Verifies the frame and the series both report the label at position 5 as first valid and position -6 (sixth from the end) as last valid.
        mat = np.random.default_rng(2).standard_normal(len(index))
        mat[:5] = np.nan
        mat[-5:] = np.nan

        frame = DataFrame({"foo": mat}, index=index)
        assert frame.first_valid_index() == frame.index[5]
        assert frame.last_valid_index() == frame.index[-6]

        ser = frame["foo"]
        assert ser.first_valid_index() == frame.index[5]
        assert ser.last_valid_index() == frame.index[-6]

    @pytest.mark.parametrize(
        "index",
        [Index([str(i) for i in range(10)]), date_range("2020-01-01", periods=10)],
    )
    def test_first_last_valid_all_nan(self, index):
        # [FUNCTION DOC]
        # Parametrized over a string index and a datetime index of length 10, with every value missing (GH#17400).
        # Verifies first and last valid index are None for both the frame and its column series.
        # GH#17400: no valid entries
        frame = DataFrame(np.nan, columns=["foo"], index=index)

        assert frame.last_valid_index() is None
        assert frame.first_valid_index() is None

        ser = frame["foo"]
        assert ser.first_valid_index() is None
        assert ser.last_valid_index() is None
