# [MODULE DOC]
# Tests that Series subclasses (the shared testing subclass, and a local subclass defined here) propagate their type through indexing, conversion to frame, unstack, repr, as-of lookup, explode, equality and construction from a dictionary.
# Defines a module-wide mark that ignores the block-manager deprecation warnings. No other import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)


class TestSeriesSubclassing:
    # [CLASS DOC]
    # Groups tests of behavior of Series subclasses using the testing helper subclasses for Series and DataFrame. Stateless.
    @pytest.mark.parametrize(
        "idx_method, indexer, exp_data, exp_idx",
        [
            ["loc", ["a", "b"], [1, 2], "ab"],
            ["iloc", [2, 3], [3, 4], "cd"],
        ],
    )
    def test_indexing_sliced(self, idx_method, indexer, exp_data, exp_idx):
        # [FUNCTION DOC]
        # Verifies that label-based and position-based indexing with a list returns a subclass instance with the expected values and index. Parametrized over the indexer name, the indexer, expected data and expected index characters.
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list("abcd"))
        res = getattr(s, idx_method)[indexer]
        exp = tm.SubclassedSeries(exp_data, index=list(exp_idx))
        tm.assert_series_equal(res, exp)

    def test_to_frame(self):
        # [FUNCTION DOC]
        # Verifies that converting a named subclassed Series to a frame yields the subclassed frame type with the series name as column and the same index.
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list("abcd"), name="xxx")
        res = s.to_frame()
        exp = tm.SubclassedDataFrame({"xxx": [1, 2, 3, 4]}, index=list("abcd"))
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self):
        # [FUNCTION DOC]
        # Verifies (issue 15564) that unstacking a subclassed Series with a two-level index gives the subclassed frame with columns x and y.
        # GH 15564
        s = tm.SubclassedSeries([1, 2, 3, 4], index=[list("aabb"), list("xyxy")])

        res = s.unstack()
        exp = tm.SubclassedDataFrame({"x": [1, 3], "y": [2, 4]}, index=["a", "b"])

        tm.assert_frame_equal(res, exp)

    def test_subclass_empty_repr(self):
        # [FUNCTION DOC]
        # Verifies that the representation of an empty subclassed Series contains the subclass name.
        sub_series = tm.SubclassedSeries()
        assert "SubclassedSeries" in repr(sub_series)

    def test_asof(self):
        # [FUNCTION DOC]
        # Verifies that the as-of lookup on a subclassed Series of NaN values with a datetime index, evaluated at the last two index labels, returns the subclass type.
        N = 3
        rng = pd.date_range("1/1/1990", periods=N, freq="53s")
        s = tm.SubclassedSeries({"A": [np.nan, np.nan, np.nan]}, index=rng)

        result = s.asof(rng[-2:])
        assert isinstance(result, tm.SubclassedSeries)

    def test_explode(self):
        # [FUNCTION DOC]
        # Verifies that exploding a subclassed Series with list-like, scalar and empty elements returns the subclass type.
        s = tm.SubclassedSeries([[1, 2, 3], "foo", [], [3, 4]])
        result = s.explode()
        assert isinstance(result, tm.SubclassedSeries)

    def test_equals(self):
        # [FUNCTION DOC]
        # Verifies that equality testing between a plain Series and a subclassed Series is true in both directions (pull request 34402).
        # https://github.com/pandas-dev/pandas/pull/34402
        # allow subclass in both directions
        s1 = pd.Series([1, 2, 3])
        s2 = tm.SubclassedSeries([1, 2, 3])
        assert s1.equals(s2)
        assert s2.equals(s1)


class SubclassedSeries(pd.Series):
    # [CLASS DOC]
    # HELPER: local Series subclass whose constructor hook returns a plain Series when the instance's name is the literal string test and otherwise returns this subclass; it exercises constructors that read instance state.
    # KNOWN ISSUE: this local class has the same name as the shared testing subclass used by the test class above, but the test class uses the testing module's class, so the two are easily confused.
    @property
    def _constructor(self):
        # [FUNCTION DOC]
        # HELPER: property returning a factory function used by the library to build new Series of the same kind.
        def _new(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: factory taking arbitrary positional and keyword arguments; consults the enclosing instance's name and returns either a plain Series or this subclass built from the arguments.
            # some constructor logic that accesses the Series' name
            if self.name == "test":
                return pd.Series(*args, **kwargs)
            return SubclassedSeries(*args, **kwargs)

        return _new


def test_constructor_from_dict():
    # [FUNCTION DOC]
    # Verifies (issue 52445) that constructing the local subclass from a dictionary returns an instance of the subclass even though the constructor hook reads the instance name.
    # https://github.com/pandas-dev/pandas/issues/52445
    result = SubclassedSeries({"a": 1, "b": 2, "c": 3})
    assert isinstance(result, SubclassedSeries)
