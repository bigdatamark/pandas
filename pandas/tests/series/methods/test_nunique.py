# [MODULE DOC]
# Tests for Series.nunique (count of distinct non-missing values). No import-time
# side effects.

import numpy as np

from pandas import (
    Categorical,
    Series,
)


def test_nunique():
    # [FUNCTION DOC]
    # Ten distinct random values, ten copies of 5000 and NaN for the rest give eleven
    # distinct values (NaN excluded); example from the basics documentation.
    # basics.rst doc example
    series = Series(np.random.default_rng(2).standard_normal(500))
    series[20:500] = np.nan
    series[10:20] = 5000
    result = series.nunique()
    assert result == 11


def test_nunique_categorical():
    # [FUNCTION DOC]
    # GH#18051: an empty categorical and a categorical of only NaN both have zero
    # distinct values.
    # GH#18051
    ser = Series(Categorical([]))
    assert ser.nunique() == 0

    ser = Series(Categorical([np.nan]))
    assert ser.nunique() == 0
