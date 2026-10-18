# [MODULE DOC]
# Tests for DatetimeIndex.map: string results, silent fallthrough, using bound methods, and
# tuple results producing a MultiIndex.
import pytest

from pandas import (
    DatetimeIndex,
    Index,
    MultiIndex,
    Period,
    date_range,
)
import pandas._testing as tm


class TestMap:
    # [CLASS DOC]
    # Groups map tests for DatetimeIndex; no state.
    def test_map(self):
        # [FUNCTION DOC]
        # Verifies mapping a date-formatting function over a daily range gives a string Index of the
        # formatted dates.
        rng = date_range("1/1/2000", periods=10)

        f = lambda x: x.strftime("%Y%m%d")
        result = rng.map(f)
        exp = Index([f(x) for x in rng])
        tm.assert_index_equal(result, exp)

    def test_map_fallthrough(self, capsys):
        # [FUNCTION DOC]
        # Regression test: mapping a function that returns Period objects over a business-day range
        # produces nothing on standard error (no warnings about silently ignored errors).
        # GH#22067, check we don't get warnings about silently ignored errors
        dti = date_range("2017-01-01", "2018-01-01", freq="B")

        dti.map(lambda x: Period(year=x.year, month=x.month, freq="M"))

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_map_bug_1677(self):
        # [FUNCTION DOC]
        # Regression test: mapping the index's own asof method over a one-element index works and gives
        # an Index of the asof result.
        index = DatetimeIndex(["2012-04-25 09:30:00.393000"])
        f = index.asof

        result = index.map(f)
        expected = Index([f(index[0])])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("name", [None, "name"])
    def test_index_map(self, name):
        # [FUNCTION DOC]
        # Parametrized over name (none or "name"). Verifies that mapping to (year, month) tuples over
        # a month-end range gives a MultiIndex product of year 2018 and months one to six with both level
        # names equal to the original name.
        # see GH#20990
        count = 6
        index = date_range("2018-01-01", periods=count, freq="ME", name=name).map(
            lambda x: (x.year, x.month)
        )
        exp_index = MultiIndex.from_product(((2018,), range(1, 7)), names=[name, name])
        tm.assert_index_equal(index, exp_index)
