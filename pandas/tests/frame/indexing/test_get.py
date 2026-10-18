# [MODULE DOC]
# Tests for the DataFrame get method (key lookup with default). Uses the float_frame
# fixture. No import-time side effects.
import pytest

from pandas import DataFrame
import pandas._testing as tm


class TestGet:
    # [CLASS DOC]
    # Test class for the get method.
    def test_get(self, float_frame):
        # [FUNCTION DOC]
        # Verifies get returns the column Series for an existing label, returns none for a missing label,
        # and returns the supplied default for a missing label when one is given.
        b = float_frame.get("B")
        tm.assert_series_equal(b, float_frame["B"])

        assert float_frame.get("foo") is None
        tm.assert_series_equal(
            float_frame.get("foo", float_frame["B"]), float_frame["B"]
        )

    @pytest.mark.parametrize(
        "columns, index",
        [
            [None, None],
            [list("AB"), None],
            [list("AB"), range(3)],
        ],
    )
    def test_get_none(self, columns, index):
        # [FUNCTION DOC]
        # Regression test for gh-5652. Parametrized over combinations of columns and index (none,
        # two columns, two columns plus a three-long range). Verifies get with a key of none returns
        # none for such empty frames rather than raising.
        # see gh-5652
        assert DataFrame(columns=columns, index=index).get(None) is None
