# [MODULE DOC]
# Tests for retrieving level values from a MultiIndex: boxing of datetimes,
# names, categorical levels, handling of missing values, period levels, and
# preservation of frequency.
# Uses the shared idx fixture. Pure tests module.

import numpy as np

import pandas as pd
from pandas import (
    CategoricalIndex,
    Index,
    MultiIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestGetLevelValues:
    # [CLASS DOC]
    # Groups level-value retrieval tests.
    def test_get_level_values_box_datetime64(self):
        # [FUNCTION DOC]
        # Verifies the values of a datetime level are returned as Timestamp objects.
        dates = date_range("1/1/2000", periods=4)
        levels = [dates, [0, 1]]
        codes = [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]]

        index = MultiIndex(levels=levels, codes=codes)

        assert isinstance(index.get_level_values(0)[0], Timestamp)


def test_get_level_values(idx):
    # [FUNCTION DOC]
    # Verifies retrieval by position and by name returns an index with the level's
    # name, and (GH 10460) categorical levels come back as categorical indexes with
    # the expanded values.
    result = idx.get_level_values(0)
    expected = Index(["foo", "foo", "bar", "baz", "qux", "qux"], name="first")
    tm.assert_index_equal(result, expected)
    assert result.name == "first"

    result = idx.get_level_values("first")
    expected = idx.get_level_values(0)
    tm.assert_index_equal(result, expected)

    # GH 10460
    index = MultiIndex(
        levels=[CategoricalIndex(["A", "B"]), CategoricalIndex([1, 2, 3])],
        codes=[np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 2, 0, 1, 2])],
    )

    exp = CategoricalIndex(["A", "A", "A", "B", "B", "B"])
    tm.assert_index_equal(index.get_level_values(0), exp)
    exp = CategoricalIndex([1, 2, 3, 1, 2, 3])
    tm.assert_index_equal(index.get_level_values(1), exp)


def test_get_level_values_all_na():
    # [FUNCTION DOC]
    # Verifies (GH#17924) a level made entirely of missing values comes back as a
    # float index of missing values, and a mixed level stays an object index.
    # GH#17924 when level entirely consists of nan
    arrays = [[np.nan, np.nan, np.nan], ["a", np.nan, 1]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(0)
    expected = Index([np.nan, np.nan, np.nan], dtype=np.float64)
    tm.assert_index_equal(result, expected)

    result = index.get_level_values(1)
    expected = Index(["a", np.nan, 1], dtype=object)
    tm.assert_index_equal(result, expected)


def test_get_level_values_int_with_na():
    # [FUNCTION DOC]
    # Verifies (GH#17924) an integer level with missing values returns a float index,
    # also when all but one are missing.
    # GH#17924
    arrays = [["a", "b", "b"], [1, np.nan, 2]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = Index([1, np.nan, 2])
    tm.assert_index_equal(result, expected)

    arrays = [["a", "b", "b"], [np.nan, np.nan, 2]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = Index([np.nan, np.nan, 2])
    tm.assert_index_equal(result, expected)


def test_get_level_values_na():
    # [FUNCTION DOC]
    # Verifies retrieval of levels with missing values: all-missing gives a float index;
    # mixed object values stay; a datetime level with NaT gives a datetime index;
    # empty arrays give an empty object index.
    arrays = [[np.nan, np.nan, np.nan], ["a", np.nan, 1]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(0)
    expected = Index([np.nan, np.nan, np.nan])
    tm.assert_index_equal(result, expected)

    result = index.get_level_values(1)
    expected = Index(["a", np.nan, 1])
    tm.assert_index_equal(result, expected)

    arrays = [["a", "b", "b"], pd.DatetimeIndex([0, 1, pd.NaT])]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = pd.DatetimeIndex([0, 1, pd.NaT])
    tm.assert_index_equal(result, expected)

    arrays = [[], []]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(0)
    expected = Index([], dtype=object)
    tm.assert_index_equal(result, expected)


def test_get_level_values_when_periods():
    # [FUNCTION DOC]
    # Verifies (GH33131) rebuilding a multi-index from the private level-value getter
    # on a period level yields monotonic increasing levels.
    # Note: the in-source comment says this can be removed when the period index engine is removed.
    # GH33131. See also discussion in GH32669.
    # This test can probably be removed when PeriodIndex._engine is removed.
    from pandas import (
        Period,
        PeriodIndex,
    )

    idx = MultiIndex.from_arrays(
        [PeriodIndex([Period("2019Q1"), Period("2019Q2")], name="b")]
    )
    idx2 = MultiIndex.from_arrays(
        [idx._get_level_values(level) for level in range(idx.nlevels)]
    )
    assert all(x.is_monotonic_increasing for x in idx2.levels)


def test_values_loses_freq_of_underlying_index():
    # [FUNCTION DOC]
    # Verifies (GH#49054) accessing the values of a multi-index does not remove the
    # frequency of the underlying datetime level, and the level is unchanged.
    # GH#49054
    idx = pd.DatetimeIndex(date_range("20200101", periods=3, freq="BME"))
    expected = idx.copy(deep=True)
    idx2 = Index([1, 2, 3])
    midx = MultiIndex(levels=[idx, idx2], codes=[[0, 1, 2], [0, 1, 2]])
    midx.values
    assert idx.freq is not None
    tm.assert_index_equal(idx, expected)


def test_get_level_values_gets_frequency_correctly():
    # [FUNCTION DOC]
    # Verifies (GH#57949, GH#58327) the retrieved datetime level of a product
    # multi-index keeps the year-start frequency.
    # GH#57949 GH#58327
    datetime_index = date_range(start=pd.to_datetime("1/1/2018"), periods=4, freq="YS")
    other_index = ["A"]
    multi_index = MultiIndex.from_product([datetime_index, other_index])

    assert multi_index.get_level_values(0).freq == datetime_index.freq
