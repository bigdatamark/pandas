"""Test cases for DataFrame.plot"""

# [MODULE DOC]
# Tests that grouped box plots on a frame honor the shared-axis keywords:
# checks which tick labels remain visible for the sharey and sharex options.
# Skipped entirely if matplotlib is missing (import-or-skip at module level).
# No other import-time side effects.

import pytest

from pandas import DataFrame
from pandas.tests.plotting.common import _check_visible

pytest.importorskip("matplotlib")


class TestDataFramePlotsGroupby:
    # [CLASS DOC]
    # Test class for grouped frame plotting, with two private visibility assertion
    # helpers and the share-axis tests (GH 20968).
    def _assert_ytickslabels_visibility(self, axes, expected):
        # [FUNCTION DOC]
        # HELPER: asserts y tick label visibility. Parameters: axes is a sequence of
        # axes; expected is a sequence of booleans of the same length (a length
        # mismatch raises an error). Asserts each axes' y tick labels have the expected
        # visibility.
        for ax, exp in zip(axes, expected, strict=True):
            _check_visible(ax.get_yticklabels(), visible=exp)

    def _assert_xtickslabels_visibility(self, axes, expected):
        # [FUNCTION DOC]
        # HELPER: asserts x tick label visibility. Parameters: axes is a sequence of
        # axes; expected is a sequence of booleans of the same length (a length
        # mismatch raises an error). Asserts each axes' x tick labels have the expected
        # visibility.
        for ax, exp in zip(axes, expected, strict=True):
            _check_visible(ax.get_xticklabels(), visible=exp)

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            # behavior without keyword
            ({}, [True, False, True, False]),
            # set sharey=True should be identical
            ({"sharey": True}, [True, False, True, False]),
            # sharey=False, all yticklabels should be visible
            ({"sharey": False}, [True, True, True, True]),
        ],
    )
    def test_groupby_boxplot_sharey(self, kwargs, expected):
        # [FUNCTION DOC]
        # Parametrized over (keywords, expected y-label visibility). Box plots of the
        # grouped frame (grouping on column c giving four axes in a grid): by default
        # or with sharey true only the left column shows y labels; with sharey false
        # all axes show them.
        # https://github.com/pandas-dev/pandas/issues/20968
        # sharey can now be switched check whether the right
        # pair of axes is turned on or off
        df = DataFrame(
            {
                "a": [-1.43, -0.15, -3.70, -1.43, -0.14],
                "b": [0.56, 0.84, 0.29, 0.56, 0.85],
                "c": [0, 1, 2, 3, 1],
            },
            index=[0, 1, 2, 3, 4],
        )
        axes = df.groupby("c").boxplot(**kwargs)
        self._assert_ytickslabels_visibility(axes, expected)

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            # behavior without keyword
            ({}, [True, True, True, True]),
            # set sharex=False should be identical
            ({"sharex": False}, [True, True, True, True]),
            # sharex=True, xticklabels should be visible
            # only for bottom plots
            ({"sharex": True}, [False, False, True, True]),
        ],
    )
    def test_groupby_boxplot_sharex(self, kwargs, expected):
        # [FUNCTION DOC]
        # Parametrized over (keywords, expected x-label visibility). Box plots of the
        # grouped frame: by default or with sharex false all x labels are visible; with
        # sharex true only the bottom row shows x labels.
        # https://github.com/pandas-dev/pandas/issues/20968
        # sharex can now be switched check whether the right
        # pair of axes is turned on or off

        df = DataFrame(
            {
                "a": [-1.43, -0.15, -3.70, -1.43, -0.14],
                "b": [0.56, 0.84, 0.29, 0.56, 0.85],
                "c": [0, 1, 2, 3, 1],
            },
            index=[0, 1, 2, 3, 4],
        )
        axes = df.groupby("c").boxplot(**kwargs)
        self._assert_xtickslabels_visibility(axes, expected)
