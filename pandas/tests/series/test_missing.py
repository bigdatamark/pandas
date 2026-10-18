# [MODULE DOC]
# Tests of missing-data handling in Series: categorical missing codes, timedelta missing values and boolean assignment, a known-failing chained comparison, dropping missing values, and the non-caching of the has-missing property for Series.
# No import-time side effects.

from datetime import timedelta

import numpy as np
import pytest

from pandas._libs import iNaT

import pandas as pd
from pandas import (
    Categorical,
    Index,
    NaT,
    Series,
    isna,
)
import pandas._testing as tm


class TestSeriesMissingData:
    # [CLASS DOC]
    # Groups tests of missing data handling in Series. Stateless.
    def test_categorical_nan_handling(self):
        # [FUNCTION DOC]
        # Verifies that a categorical Series with one NaN has categories a and b only and that its integer codes mark the missing entry with minus one (int8 codes).
        # NaNs are represented as -1 in labels
        s = Series(Categorical(["a", "b", np.nan, "a"]))
        tm.assert_index_equal(s.cat.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(
            s.values.codes, np.array([0, 1, -1, 0], dtype=np.int8)
        )

    def test_timedelta64_nan(self):
        # [FUNCTION DOC]
        # Verifies missing-value handling in a ten-element timedelta Series. Setting NaN or NaT marks the element missing with the internal integer missing-value sentinel as its stored value, and restoring the original makes it non-missing. Assigning the raw sentinel integer raises a type error (invalid value; issue 16674) because it is treated as an integer. Assigning NaN through a boolean mask selecting values strictly between day three and day seven (issue 2899) sets exactly three elements missing.
        td = Series([timedelta(days=i) for i in range(10)])

        # nan ops on timedeltas
        td1 = td.copy()
        td1[0] = np.nan
        assert isna(td1[0])
        assert td1[0]._value == iNaT
        td1[0] = td[0]
        assert not isna(td1[0])

        # GH#16674 iNaT is treated as an integer when given by the user
        with pytest.raises(TypeError, match="Invalid value"):
            td1[1] = iNaT

        td1[2] = NaT
        assert isna(td1[2])
        assert td1[2]._value == iNaT
        td1[2] = td[2]
        assert not isna(td1[2])

        # boolean setting
        # GH#2899 boolean setting
        td3 = np.timedelta64(timedelta(days=3))
        td7 = np.timedelta64(timedelta(days=7))
        td[(td > td3) & (td < td7)] = np.nan
        assert isna(td).sum() == 3

    @pytest.mark.xfail(
        reason="Chained inequality raises when trying to define 'selector'"
    )
    def test_logical_range_select(self, datetime_series):
        # [FUNCTION DOC]
        # Expected-to-fail test (marked xfail because chained comparison raises when evaluating the selector, a NumPy limitation): compares a chained inequality on the datetime series to the combination of two separate comparisons.
        # NumPy limitation =(
        # https://github.com/pandas-dev/pandas/commit/9030dc021f07c76809848925cb34828f6c8484f3

        selector = -0.5 <= datetime_series <= 0.5
        expected = (datetime_series >= -0.5) & (datetime_series <= 0.5)
        tm.assert_series_equal(selector, expected)

    def test_valid(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that dropping missing values from a series (with every other value set to NaN and index frequency cleared) gives a result of length equal to the non-missing count and equal to both positional slicing of the odd positions and boolean selection with the not-missing mask.
        ts = datetime_series.copy()
        ts.index = ts.index._with_freq(None)
        ts[::2] = np.nan

        result = ts.dropna()
        assert len(result) == ts.count()
        tm.assert_series_equal(result, ts[1::2])
        tm.assert_series_equal(result, ts[pd.notna(ts)])


def test_hasnans_uncached_for_series():
    # [FUNCTION DOC]
    # Verifies (issue 19700) that the has-missing property is cached for an Index (a cache entry appears after first access) but not for a Series made from it: the Series has no cache attribute, and after setting the last element to NaN the property reports true.
    # GH#19700
    # set float64 dtype to avoid upcast when setting nan
    idx = Index([0, 1], dtype="float64")
    assert idx.hasnans is False
    assert "hasnans" in idx._cache
    ser = idx.to_series()
    assert ser.hasnans is False
    assert not hasattr(ser, "_cache")
    ser.iloc[-1] = np.nan
    assert ser.hasnans is True
