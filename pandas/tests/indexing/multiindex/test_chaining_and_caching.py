# [MODULE DOC]
# Tests for chained assignment detection and index-cache behavior when the object is labelled by a multi-level index: chained in-place fill, setting through a chained expression, caching of indexers when the index exceeds the engine size cutoff, and which cache entries survive renaming levels.

import numpy as np

from pandas._libs import index as libindex

from pandas import (
    DataFrame,
    MultiIndex,
    RangeIndex,
    Series,
)
import pandas._testing as tm


def test_detect_chained_assignment():
    # [FUNCTION DOC]
    # Builds a frame with two-level columns (part, side) and one missing value. An in-place fill on a chained selection (select first-level then second-level then fill in place) raises the chained-assignment error rather than silently modifying a copy.
    # Inplace ops, originally from:
    # https://stackoverflow.com/questions/20508968/series-fillna-in-a-multiindex-dataframe-does-not-fill-is-this-a-bug
    a = [12, 23]
    b = [123, None]
    c = [1234, 2345]
    d = [12345, 23456]
    tuples = [("eyes", "left"), ("eyes", "right"), ("ears", "left"), ("ears", "right")]
    events = {
        ("eyes", "left"): a,
        ("eyes", "right"): b,
        ("ears", "left"): c,
        ("ears", "right"): d,
    }
    multiind = MultiIndex.from_tuples(tuples, names=["part", "side"])
    zed = DataFrame(events, index=["a", "b"], columns=multiind)

    with tm.raises_chained_assignment_error():
        zed["eyes"]["right"].fillna(value=555, inplace=True)


def test_cache_updating():
    # [FUNCTION DOC]
    # Builds a random three-column frame with a two-level integer row index. Setting through a chained selection (row label then column then positional item) raises the chained-assignment error; afterwards the targeted cell still equals the value in an untouched copy of the original frame. Then a direct label-based assignment on the tuple key and column works, giving the new value.
    # KNOWN ISSUE: the source comment above the chained assignment says it "actually works, since everything is a view", but the code asserts the chained-assignment error is raised and that the value is unchanged, so the comment is stale.
    # 5216
    # make sure that we don't try to set a dead cache
    a = np.random.default_rng(2).random((10, 3))
    df = DataFrame(a, columns=["x", "y", "z"])
    df_original = df.copy()
    tuples = [(i, j) for i in range(5) for j in range(2)]
    index = MultiIndex.from_tuples(tuples)
    df.index = index

    # setting via chained assignment
    # but actually works, since everything is a view

    with tm.raises_chained_assignment_error():
        df.loc[0]["z"].iloc[0] = 1.0

    assert df.loc[(0, 0), "z"] == df_original.loc[0, "z"]

    # correct setting
    df.loc[(0, 0), "z"] = 2
    result = df.loc[(0, 0), "z"]
    assert result == 2


def test_indexer_caching(monkeypatch):
    # [FUNCTION DOC]
    # Patches the engine size cutoff down to 20. A series of zeros on a two-level index of that length: setting values through a boolean condition on the series itself sets all to one; the result equals a series of ones with the same index. Checks nothing breaks with cached indexers on large multi-level indexes.
    # GH5727
    # make sure that indexers are in the _internal_names_set
    size_cutoff = 20
    with monkeypatch.context():
        monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
        index = MultiIndex.from_arrays([np.arange(size_cutoff), np.arange(size_cutoff)])
        s = Series(np.zeros(size_cutoff), index=index)

        # setitem
        s[s == 0] = 1
    expected = Series(np.ones(size_cutoff), index=index)
    tm.assert_series_equal(s, expected)


def test_set_names_only_clears_level_cache():
    # [FUNCTION DOC]
    # After touching dtypes, monotonicity, engine and levels of a two-level index, the cache holds exactly those four keys. Renaming the levels removes only the levels entry (the others remain); fetching levels again gives range indexes carrying the new names.
    mi = MultiIndex.from_arrays([range(4), range(4)], names=["a", "b"])
    mi.dtypes
    mi.is_monotonic_increasing
    mi._engine
    mi.levels
    old_cache_keys = sorted(mi._cache.keys())
    assert old_cache_keys == ["_engine", "dtypes", "is_monotonic_increasing", "levels"]
    mi.names = ["A", "B"]
    new_cache_keys = sorted(mi._cache.keys())
    assert new_cache_keys == ["_engine", "dtypes", "is_monotonic_increasing"]
    new_levels = mi.levels
    tm.assert_index_equal(new_levels[0], RangeIndex(4, name="A"))
    tm.assert_index_equal(new_levels[1], RangeIndex(4, name="B"))
