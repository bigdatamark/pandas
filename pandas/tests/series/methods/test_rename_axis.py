# [MODULE DOC]
# Tests for Series.rename_axis (renaming index level names). Uses the
# datetime_series fixture. No import-time side effects.

import pytest

from pandas import (
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestSeriesRenameAxis:
    # [CLASS DOC]
    # Groups the tests of Series.rename_axis. Stateless.
    def test_rename_axis_mapper(self):
        # [FUNCTION DOC]
        # GH 19978: on a two-level MultiIndex, a dict renames the named level, a
        # function applies to level names, and a list sets all names. An unexpected
        # keyword (columns) raises a type error.
        # GH 19978
        mi = MultiIndex.from_product([["a", "b", "c"], [1, 2]], names=["ll", "nn"])
        ser = Series(list(range(len(mi))), index=mi)

        result = ser.rename_axis(index={"ll": "foo"})
        assert result.index.names == ["foo", "nn"]

        result = ser.rename_axis(index=str.upper, axis=0)
        assert result.index.names == ["LL", "NN"]

        result = ser.rename_axis(index=["foo", "goo"])
        assert result.index.names == ["foo", "goo"]

        with pytest.raises(TypeError, match="unexpected"):
            ser.rename_axis(columns="wrong")

    def test_rename_axis_inplace(self, datetime_series):
        # [FUNCTION DOC]
        # GH 15704: renaming in place returns None and gives the same result as the
        # non-in-place version.
        # GH 15704
        expected = datetime_series.rename_axis("foo")
        result = datetime_series
        no_return = result.rename_axis("foo", inplace=True)

        assert no_return is None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("kwargs", [{"mapper": None}, {"index": None}, {}])
    def test_rename_axis_none(self, kwargs):
        # [FUNCTION DOC]
        # GH 25034, parametrized over mapper none, index none, and no arguments: passing
        # None as mapper or index removes the index name; passing nothing leaves the
        # index unchanged.
        # GH 25034
        index = Index(list("abc"), name="foo")
        ser = Series([1, 2, 3], index=index)

        result = ser.rename_axis(**kwargs)
        expected_index = index.rename(None) if kwargs else index
        expected = Series([1, 2, 3], index=expected_index)
        tm.assert_series_equal(result, expected)
