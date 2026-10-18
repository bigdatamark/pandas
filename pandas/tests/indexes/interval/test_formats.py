# [MODULE DOC]
# Tests for text rendering of IntervalIndex: repr within a Series or DataFrame,
# float formatting, the CSV value formatter and aware-timestamp repr.
# Uses the unit and using_infer_string fixtures. Pure tests module.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


class TestIntervalIndexRendering:
    # [CLASS DOC]
    # Groups rendering tests for IntervalIndex.
    # TODO: this is a test for DataFrame/Series, not IntervalIndex
    @pytest.mark.parametrize(
        "constructor,expected",
        [
            (
                Series,
                ("(0.0, 1.0]    a\nNaN           b\n(2.0, 3.0]    c\ndtype: object"),
            ),
            (DataFrame, ("            0\n(0.0, 1.0]  a\nNaN         b\n(2.0, 3.0]  c")),
        ],
    )
    def test_repr_missing(self, constructor, expected, using_infer_string, request):
        # [FUNCTION DOC]
        # Note: the in-source TODO says this is really a test for DataFrame and Series
        # rather than for the index. Parametrized over a Series or DataFrame and the
        # expected text. Verifies (GH 25984) the repr of an object with an interval
        # index containing a missing label shows the missing label as NaN. Expected to
        # fail for the Series case when string inference is enabled.
        # GH 25984
        if using_infer_string and constructor is Series:
            request.applymarker(pytest.mark.xfail(reason="repr different"))
        index = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        obj = constructor(list("abc"), index=index)
        result = repr(obj)
        assert result == expected

    def test_repr_floats(self):
        # [FUNCTION DOC]
        # Verifies (GH 32553) the string form of a Series indexed by float intervals
        # shows the full float precision of the bounds.
        # GH 32553

        markers = Series(
            [1, 2],
            index=IntervalIndex(
                [
                    Interval(left, right)
                    for left, right in zip(
                        Index([329.973, 345.137], dtype="float64"),
                        Index([345.137, 360.191], dtype="float64"),
                    )
                ]
            ),
        )
        result = str(markers)
        expected = "(329.973, 345.137]    1\n(345.137, 360.191]    2\ndtype: int64"
        assert result == expected

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    @pytest.mark.parametrize(
        "tuples, closed, expected_data",
        [
            ([(0, 1), (1, 2), (2, 3)], "left", ["[0, 1)", "[1, 2)", "[2, 3)"]),
            (
                [(0.5, 1.0), np.nan, (2.0, 3.0)],
                "right",
                ["(0.5, 1.0]", "NaN", "(2.0, 3.0]"],
            ),
            (
                [
                    (Timestamp("20180101"), Timestamp("20180102")),
                    np.nan,
                    ((Timestamp("20180102"), Timestamp("20180103"))),
                ],
                "both",
                [
                    "[2018-01-01 00:00:00, 2018-01-02 00:00:00]",
                    "NaN",
                    "[2018-01-02 00:00:00, 2018-01-03 00:00:00]",
                ],
            ),
            (
                [
                    (Timedelta("0 days"), Timedelta("1 days")),
                    (Timedelta("1 days"), Timedelta("2 days")),
                    np.nan,
                ],
                "neither",
                [
                    "(0 days 00:00:00, 1 days 00:00:00)",
                    "(1 days 00:00:00, 2 days 00:00:00)",
                    "NaN",
                ],
            ),
        ],
    )
    def test_get_values_for_csv(self, tuples, closed, expected_data):
        # [FUNCTION DOC]
        # Parametrized over tuples, closed side and expected strings (integers, floats
        # with a missing entry, timestamps, timedeltas). Ignores invalid-cast runtime
        # warnings. Verifies (GH 28210) the private CSV-value formatter gives an array
        # of interval strings with the missing entry rendered as the NaN text.
        # GH 28210
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        result = index._get_values_for_csv(na_rep="NaN")
        expected = np.array(expected_data)
        tm.assert_numpy_array_equal(result, expected)

    def test_timestamp_with_timezone(self, unit):
        # [FUNCTION DOC]
        # Parametrized over unit. Verifies (GH 55035) the repr of an interval index of
        # UTC-aware timestamps includes the offsets and the dtype with unit and zone.
        # GH 55035
        left = DatetimeIndex(["2020-01-01"], dtype=f"M8[{unit}, UTC]")
        right = DatetimeIndex(["2020-01-02"], dtype=f"M8[{unit}, UTC]")
        index = IntervalIndex.from_arrays(left, right)
        result = repr(index)
        expected = (
            "IntervalIndex([(2020-01-01 00:00:00+00:00, 2020-01-02 00:00:00+00:00]], "
            f"dtype='interval[datetime64[{unit}, UTC], right]')"
        )
        assert result == expected
