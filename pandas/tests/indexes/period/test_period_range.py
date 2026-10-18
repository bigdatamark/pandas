# [MODULE DOC]
# Tests for the period_range function: argument validation, construction from
# strings and Period objects for several frequencies, frequency mismatch checks
# and rejected or deprecated frequency aliases.
import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas import (
    NaT,
    Period,
    PeriodIndex,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestPeriodRangeKeywords:
    # [CLASS DOC]
    # Groups tests of required keyword combinations.
    def test_required_arguments(self):
        # [FUNCTION DOC]
        # Passing start, end and a positional frequency but no periods raises a value
        # error that exactly two of start, end and periods must be given.
        msg = (
            "Of the three parameters: start, end, and periods, exactly two "
            "must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range("2011-1-1", "2012-1-1", "B")

    def test_required_arguments2(self):
        # [FUNCTION DOC]
        # Only a start Period raises the same value error.
        start = Period("02-Apr-2005", "D")
        msg = (
            "Of the three parameters: start, end, and periods, exactly two "
            "must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range(start=start)

    def test_required_arguments3(self):
        # [FUNCTION DOC]
        # Parametrized by nothing; checks that only start, only end, only periods and no
        # arguments each raise the same value error.
        # not enough params
        msg = (
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1")

        with pytest.raises(ValueError, match=msg):
            period_range(end="2017Q1")

        with pytest.raises(ValueError, match=msg):
            period_range(periods=5)

        with pytest.raises(ValueError, match=msg):
            period_range()

    def test_required_arguments_too_many(self):
        # [FUNCTION DOC]
        # Giving start, end and periods together raises the same value error.
        msg = (
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1", end="2018Q1", periods=8, freq="Q")

    def test_start_end_non_nat(self):
        # [FUNCTION DOC]
        # A missing-value start or end (with or without freq) raises a value error that
        # start and end must not be NaT.
        # start/end NaT
        msg = "start and end must not be NaT"
        with pytest.raises(ValueError, match=msg):
            period_range(start=NaT, end="2018Q1")
        with pytest.raises(ValueError, match=msg):
            period_range(start=NaT, end="2018Q1", freq="Q")

        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1", end=NaT)
        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1", end=NaT, freq="Q")

    def test_periods_requires_integer(self):
        # [FUNCTION DOC]
        # A non-integer periods raises a type error.
        # invalid periods param
        msg = "periods must be an integer, got foo"
        with pytest.raises(TypeError, match=msg):
            period_range(start="2017Q1", periods="foo")


class TestPeriodRange:
    # [CLASS DOC]
    # Groups tests of normal construction.
    @pytest.mark.parametrize(
        "freq_offset, freq_period",
        [
            ("D", "D"),
            ("W", "W"),
            ("QE", "Q"),
            ("YE", "Y"),
        ],
    )
    def test_construction_from_string(self, freq_offset, freq_period):
        # [FUNCTION DOC]
        # Parametrized over day, week, quarter and year frequencies. A range built from
        # string start/end, start and periods, or end and periods equals the equivalent
        # date range converted to periods; zero periods and reversed endpoints give an
        # empty index; name is kept.
        # non-empty
        expected = date_range(
            start="2017-01-01", periods=5, freq=freq_offset, name="foo"
        ).to_period()
        start, end = str(expected[0]), str(expected[-1])

        result = period_range(start=start, end=end, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=start, periods=5, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=5, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        # empty
        expected = PeriodIndex([], freq=freq_period, name="foo")

        result = period_range(start=start, periods=0, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=0, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=end, end=start, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

    def test_construction_from_string_monthly(self):
        # [FUNCTION DOC]
        # Same checks for monthly frequency (kept separate because the date-range alias
        # differs).
        # KNOWN ISSUE: this is a copy of the parametrized test with monthly frequency hard-coded; it adds no distinct behavior.
        # non-empty
        expected = date_range(
            start="2017-01-01", periods=5, freq="ME", name="foo"
        ).to_period()
        start, end = str(expected[0]), str(expected[-1])

        result = period_range(start=start, end=end, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=start, periods=5, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=5, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # empty
        expected = PeriodIndex([], freq="M", name="foo")

        result = period_range(start=start, periods=0, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=0, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=end, end=start, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

    def test_construction_from_period(self):
        # [FUNCTION DOC]
        # Period endpoints with a differing target frequency are up- and down-sampled;
        # comparing an index with its own values gives all true (issue 21793); empty
        # results for zero periods and reversed endpoints with weekly frequency.
        # upsampling
        start, end = Period("2017Q1", freq="Q"), Period("2018Q1", freq="Q")
        expected = date_range(
            start="2017-03-31", end="2018-03-31", freq="ME", name="foo"
        ).to_period()
        result = period_range(start=start, end=end, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # downsampling
        start = Period("2017-1", freq="M")
        end = Period("2019-12", freq="M")
        expected = date_range(
            start="2017-01-31", end="2019-12-31", freq="QE", name="foo"
        ).to_period()
        result = period_range(start=start, end=end, freq="Q", name="foo")
        tm.assert_index_equal(result, expected)

        # test for issue # 21793
        start = Period("2017Q1", freq="Q")
        end = Period("2018Q1", freq="Q")
        idx = period_range(start=start, end=end, freq="Q", name="foo")
        result = idx == idx.values
        expected = np.array([True, True, True, True, True])
        tm.assert_numpy_array_equal(result, expected)

        # empty
        expected = PeriodIndex([], freq="W", name="foo")

        result = period_range(start=start, periods=0, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=0, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=end, end=start, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

    def test_mismatched_start_end_freq_raises(self):
        # [FUNCTION DOC]
        # A start and end Period of different frequencies raise a value error that they
        # must have the same freq; matching frequencies work. Business-day deprecation
        # warnings are expected.
        depr_msg = "Period with BDay freq is deprecated"

        end_w = Period("2006-12-31", "1W")
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            start_b = Period("02-Apr-2005", "B")
            end_b = Period("2005-05-01", "B")

        msg = "start and end must have same freq"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                period_range(start=start_b, end=end_w)

        # without mismatch we are OK
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            period_range(start=start_b, end=end_b)


class TestPeriodRangeDisallowedFreqs:
    # [CLASS DOC]
    # Groups tests of rejected or deprecated frequency aliases.
    def test_constructor_U(self):
        # [FUNCTION DOC]
        # An unknown frequency alias raises a value error naming it.
        # KNOWN ISSUE: the comment says an undefined-period alias (U) was used, but the code checks the alias X.
        # U was used as undefined period
        with pytest.raises(ValueError, match="Invalid frequency: X"):
            period_range("2007-1-1", periods=500, freq="X")

    @pytest.mark.parametrize("freq_depr", ["2MIN", "2US", "2NS"])
    def test_uppercase_freq_deprecated_from_time_series(self, freq_depr):
        # [FUNCTION DOC]
        # Parametrized over uppercase minute, microsecond and nanosecond multiples.
        # Emits a Pandas4 warning recommending the lowercase alias (GH#52536, GH#54939).
        # GH#52536, GH#54939
        msg = (
            f"'{freq_depr[1:]}' is deprecated and will be removed in a "
            f"future version, please use '{freq_depr.lower()[1:]}' instead."
        )

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            period_range("2020-01-01 00:00:00 00:00", periods=2, freq=freq_depr)

    @pytest.mark.parametrize("freq", ["2m", "2q-sep", "2y", "2H", "2S"])
    def test_incorrect_case_freq_from_time_series_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over wrongly-cased aliases. Raises a value error naming the
        # frequency (GH#52536, GH#54939, GH#59143).
        # GH#52536, GH#54939, GH#59143
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            period_range(freq=freq, start="1/1/2001", end="12/1/2009")

    @pytest.mark.parametrize("freq", ["2A", "2a", "2A-AUG", "2A-aug"])
    def test_A_raises_from_time_series(self, freq):
        # [FUNCTION DOC]
        # Parametrized over legacy yearly aliases with or without anchor. Raises a value
        # error naming the frequency.
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            period_range(freq=freq, start="1/1/2001", end="12/1/2009")

    @pytest.mark.parametrize("freq", ["2w"])
    def test_lowercase_freq_from_time_series_deprecated(self, freq):
        # [FUNCTION DOC]
        # Parametrized over a lowercase week alias. Emits a Pandas4 warning recommending
        # uppercase (GH#52536, GH#54939).
        # GH#52536, GH#54939
        msg = (
            f"'{freq[1:]}' is deprecated and will be removed in a "
            f"future version, please use '{freq.upper()[1:]}' instead."
        )

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            period_range(freq=freq, start="1/1/2001", end="12/1/2009")
