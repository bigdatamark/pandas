# [MODULE DOC]
# Test for the text representation of the Interval scalar.

from pandas import Interval


def test_interval_repr():
    # [FUNCTION DOC]
    # Checks the detailed representation names the class, both endpoints and the closed side, while the string form
    # uses a parenthesis for an open side and a bracket for a closed side. Verified for right-closed and left-closed intervals.
    interval = Interval(0, 1)
    assert repr(interval) == "Interval(0, 1, closed='right')"
    assert str(interval) == "(0, 1]"

    interval_left = Interval(0, 1, closed="left")
    assert repr(interval_left) == "Interval(0, 1, closed='left')"
    assert str(interval_left) == "[0, 1)"
