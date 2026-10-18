# [MODULE DOC]
# Tests for constructing PeriodIndex (and period_range): rejected frequency aliases,
# ordinals, fields, array-likes, dtypes, missing values, multiples, combined
# frequencies, range lengths, internal constructors and shallow copies, and
# casting periods into Series.
import numpy as np
import pytest

from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.dtypes import PeriodDtype

from pandas import (
    Index,
    NaT,
    Period,
    PeriodIndex,
    Series,
    date_range,
    offsets,
    period_range,
)
import pandas._testing as tm
from pandas.core.arrays import PeriodArray


class TestPeriodIndexDisallowedFreqs:
    # [CLASS DOC]
    # Groups tests of frequency aliases that are rejected or deprecated when building
    # period indexes.
    @pytest.mark.parametrize(
        "freq,freq_depr",
        [
            ("2M", "2ME"),
            ("2Q-MAR", "2QE-MAR"),
            ("2Y-FEB", "2YE-FEB"),
            ("2M", "2me"),
            ("2Q-MAR", "2qe-MAR"),
            ("2Y-FEB", "2yE-feb"),
        ],
    )
    def test_period_index_offsets_frequency_error_message(self, freq, freq_depr):
        # [FUNCTION DOC]
        # Parametrized over valid frequency and its invalid end-style alias (various
        # case). Both PeriodIndex and period_range raise a value error saying the alias
        # is an invalid frequency (GH#52064).
        # GH#52064
        msg = f"Invalid frequency: {freq_depr}"

        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2020-01-01", "2020-01-02"], freq=freq_depr)

        with pytest.raises(ValueError, match=msg):
            period_range(start="2020-01-01", end="2020-01-02", freq=freq_depr)

    @pytest.mark.parametrize(
        "freq",
        ["2SME", "2sme", "2BYE", "2Bye", "2CBME"],
    )
    def test_period_index_frequency_invalid_freq(self, freq):
        # [FUNCTION DOC]
        # Parametrized over unsupported business/semi-month aliases. Both constructors
        # raise a value error naming the frequency (GH#9586).
        # GH#9586
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            period_range("2020-01", "2020-05", freq=freq)
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2020-01", "2020-05"], freq=freq)

    @pytest.mark.parametrize("freq", ["2BQE-SEP", "2BYE-MAR", "2BME"])
    def test_period_index_from_datetime_index_invalid_freq(self, freq):
        # [FUNCTION DOC]
        # Parametrized over business-offset aliases. Converting a date range with such a
        # frequency to periods raises a value error (GH#56899).
        # GH#56899
        msg = f"Invalid frequency: {freq}"

        rng = date_range("01-Jan-2012", periods=8, freq=freq)
        with pytest.raises(ValueError, match=msg):
            rng.to_period()

    @pytest.mark.parametrize("freq_depr", ["2T", "1l", "2U", "n"])
    def test_period_index_T_L_U_N_raises(self, freq_depr):
        # [FUNCTION DOC]
        # Parametrized over the removed minute/millisecond/microsecond/nanosecond
        # aliases. Both constructors raise a value error naming the frequency (GH#9586).
        # GH#9586
        msg = f"Invalid frequency: {freq_depr}"

        with pytest.raises(ValueError, match=msg):
            period_range("2020-01", "2020-05", freq=freq_depr)
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2020-01", "2020-05"], freq=freq_depr)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings("ignore:Period with BDay freq:FutureWarning")
    @pytest.mark.parametrize(
        "freq,freq_depr",
        [("2W", "2w"), ("2W-FRI", "2w-fri"), ("2D", "2d"), ("2B", "2b")],
    )
    def test_period_index_depr_lowercase_frequency(self, freq, freq_depr):
        # [FUNCTION DOC]
        # Parametrized over lowercase week, day and business-day aliases. Using them
        # emits the Pandas4 deprecation warning (message names the alias without its
        # leading multiple) and still produces the same index as the uppercase alias,
        # both for the constructor and period_range (GH#58998).
        # GH#58998
        msg = (
            f"'{freq_depr[1:]}' is deprecated and will be removed in a future version."
        )

        with tm.assert_produces_warning(
            Pandas4Warning, match=msg, raise_on_extra_warnings=False
        ):
            result = PeriodIndex(["2020-01-01", "2020-01-02"], freq=freq_depr)

        expected = PeriodIndex(["2020-01-01", "2020-01-02"], freq=freq)
        tm.assert_index_equal(result, expected)

        with tm.assert_produces_warning(
            Pandas4Warning, match=msg, raise_on_extra_warnings=False
        ):
            result = period_range(start="2020-01-01", end="2020-01-02", freq=freq_depr)

        expected = period_range(start="2020-01-01", end="2020-01-02", freq=freq)
        tm.assert_index_equal(result, expected)


class TestPeriodIndex:
    # [CLASS DOC]
    # Groups the main PeriodIndex construction tests.
    def test_from_ordinals(self):
        # [FUNCTION DOC]
        # Builds periods from ordinals, then verifies from_ordinals accepts a list or an
        # array with the same result.
        # KNOWN ISSUE: the two scalar Period constructions at the beginning are not asserted on; they only check construction does not fail.
        Period(ordinal=-1000, freq="Y")
        Period(ordinal=0, freq="Y")

        idx1 = PeriodIndex.from_ordinals(ordinals=[-1, 0, 1], freq="Y")
        idx2 = PeriodIndex.from_ordinals(ordinals=np.array([-1, 0, 1]), freq="Y")
        tm.assert_index_equal(idx1, idx2)

    def test_construction_base_constructor(self):
        # [FUNCTION DOC]
        # The generic Index constructor returns a PeriodIndex for periods with a shared
        # frequency (with NaT or NaN) and an object index when frequencies are mixed (GH
        # 13664).
        # GH 13664
        arr = [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")]
        tm.assert_index_equal(Index(arr), PeriodIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), PeriodIndex(np.array(arr)))

        arr = [np.nan, NaT, Period("2011-03", freq="M")]
        tm.assert_index_equal(Index(arr), PeriodIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), PeriodIndex(np.array(arr)))

        arr = [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="D")]
        tm.assert_index_equal(Index(arr), Index(arr, dtype=object))

        tm.assert_index_equal(Index(np.array(arr)), Index(np.array(arr), dtype=object))

    def test_base_constructor_with_period_dtype(self):
        # [FUNCTION DOC]
        # The generic Index constructor with a period dtype equals the PeriodIndex
        # constructor.
        dtype = PeriodDtype("D")
        values = ["2011-01-01", "2012-03-04", "2014-05-01"]
        result = Index(values, dtype=dtype)

        expected = PeriodIndex(values, dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "values_constructor", [list, np.array, PeriodIndex, PeriodArray._from_sequence]
    )
    def test_index_object_dtype(self, values_constructor):
        # [FUNCTION DOC]
        # Parametrized over list, array, PeriodIndex and PeriodArray inputs. With object
        # dtype the result is a plain Index holding the Period objects.
        # Index(periods, dtype=object) is an Index (not a PeriodIndex)
        periods = [
            Period("2011-01", freq="M"),
            NaT,
            Period("2011-03", freq="M"),
        ]
        values = values_constructor(periods)
        result = Index(values, dtype=object)

        assert type(result) is Index
        tm.assert_numpy_array_equal(result.values, np.array(values))

    def test_constructor_use_start_freq(self):
        # [FUNCTION DOC]
        # period_range with a Period start and no freq takes its business-day frequency;
        # deprecation warnings are expected (GH #1118).
        # GH #1118
        msg1 = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg1):
            p = Period("4/2/2012", freq="B")
        msg2 = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            expected = period_range(start="4/2/2012", periods=10, freq="B")

        with tm.assert_produces_warning(FutureWarning, match=msg2):
            index = period_range(start=p, periods=10)
        tm.assert_index_equal(index, expected)

    def test_constructor_field_arrays(self):
        # [FUNCTION DOC]
        # from_fields with year/quarter arrays builds the expected quarterly range, same
        # ordinals with a multiple frequency, and infers the frequency when omitted;
        # mismatched lengths raise a value error; year and month fields work (GH #1264).
        # GH #1264

        years = np.arange(1990, 2010).repeat(4)[2:-2]
        quarters = np.tile(np.arange(1, 5), 20)[2:-2]

        index = PeriodIndex.from_fields(year=years, quarter=quarters, freq="Q-DEC")
        expected = period_range("1990Q3", "2009Q2", freq="Q-DEC")
        tm.assert_index_equal(index, expected)

        index2 = PeriodIndex.from_fields(year=years, quarter=quarters, freq="2Q-DEC")
        tm.assert_numpy_array_equal(index.asi8, index2.asi8)

        index = PeriodIndex.from_fields(year=years, quarter=quarters)
        tm.assert_index_equal(index, expected)

        years = [2007, 2007, 2007]
        months = [1, 2]

        msg = "Mismatched Period array lengths"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex.from_fields(year=years, month=months, freq="M")
        with pytest.raises(ValueError, match=msg):
            PeriodIndex.from_fields(year=years, month=months, freq="2M")

        years = [2007, 2007, 2007]
        months = [1, 2, 3]
        idx = PeriodIndex.from_fields(year=years, month=months, freq="M")
        exp = period_range("2007-01", periods=3, freq="M")
        tm.assert_index_equal(idx, exp)

    def test_constructor_nano(self):
        # [FUNCTION DOC]
        # A range with nanosecond frequency between two ordinal periods has the four
        # expected entries.
        idx = period_range(
            start=Period(ordinal=1, freq="ns"),
            end=Period(ordinal=4, freq="ns"),
            freq="ns",
        )
        exp = PeriodIndex(
            [
                Period(ordinal=1, freq="ns"),
                Period(ordinal=2, freq="ns"),
                Period(ordinal=3, freq="ns"),
                Period(ordinal=4, freq="ns"),
            ],
            freq="ns",
        )
        tm.assert_index_equal(idx, exp)

    def test_constructor_arrays_negative_year(self):
        # [FUNCTION DOC]
        # Year and quarter attributes of an index built from fields round-trip.
        years = np.arange(1960, 2000, dtype=np.int64).repeat(4)
        quarters = np.tile(np.array([1, 2, 3, 4], dtype=np.int64), 40)
        pindex = PeriodIndex.from_fields(year=years, quarter=quarters)

        tm.assert_index_equal(pindex.year, Index(years))
        tm.assert_index_equal(pindex.quarter, Index(quarters))

    def test_constructor_invalid_quarters(self):
        # [FUNCTION DOC]
        # A quarter outside one to four raises a value error.
        msg = "Quarter must be 1 <= q <= 4"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex.from_fields(
                year=range(2000, 2004), quarter=list(range(4)), freq="Q-DEC"
            )

    def test_period_range_fractional_period(self):
        # [FUNCTION DOC]
        # A fractional periods argument raises a type error requiring an integer.
        msg = "periods must be an integer, got 10.5"
        with pytest.raises(TypeError, match=msg):
            period_range("2007-01", periods=10.5, freq="M")

    def test_constructor_with_without_freq(self):
        # [FUNCTION DOC]
        # period_range takes the frequency from the start period when not given
        # (GH53687).
        # GH53687
        start = Period("2002-01-01 00:00", freq="30min")
        exp = period_range(start=start, periods=5, freq=start.freq)
        result = period_range(start=start, periods=5)
        tm.assert_index_equal(exp, result)

    def test_constructor_fromarraylike(self):
        # [FUNCTION DOC]
        # Construction from period arrays or lists infers the frequency; integer
        # ordinals without freq raise a value error; a scalar Period raises a type
        # error; iterators, existing indexes, same and compatible freq arguments,
        # offsets and multiples (converted via asfreq) are checked.
        idx = period_range("2007-01", periods=20, freq="M")

        # values is an array of Period, thus can retrieve freq
        tm.assert_index_equal(PeriodIndex(idx.values), idx)
        tm.assert_index_equal(PeriodIndex(list(idx.values)), idx)

        msg = "freq not specified and cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(idx.asi8)
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(list(idx.asi8))

        msg = "'Period' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            PeriodIndex(data=Period("2007", freq="Y"))

        result = PeriodIndex(iter(idx))
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx)
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx, freq="M")
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx, freq=offsets.MonthEnd())
        tm.assert_index_equal(result, idx)
        assert result.freq == "ME"

        result = PeriodIndex(idx, freq="2M")
        tm.assert_index_equal(result, idx.asfreq("2M"))
        assert result.freq == "2ME"

        result = PeriodIndex(idx, freq=offsets.MonthEnd(2))
        tm.assert_index_equal(result, idx.asfreq("2M"))
        assert result.freq == "2ME"

        result = PeriodIndex(idx, freq="D")
        exp = idx.asfreq("D", "e")
        tm.assert_index_equal(result, exp)

    def test_constructor_datetime64arr(self):
        # [FUNCTION DOC]
        # Datetime arrays of different resolution (microsecond versus nanosecond)
        # produce the same daily periods.
        vals = np.arange(100000, 100000 + 10000, 100, dtype=np.int64)
        vals = vals.view(np.dtype("M8[us]"))

        pi = PeriodIndex(vals, freq="D")

        expected = PeriodIndex(vals.astype("M8[ns]"), freq="D")
        tm.assert_index_equal(pi, expected)

    @pytest.mark.parametrize("box", [None, "series", "index"])
    def test_constructor_datetime64arr_ok(self, box):
        # [FUNCTION DOC]
        # Parametrized over raw array, Series and Index containers of month-end dates.
        # Construction with a daily freq gives the expected periods (GH 23438).
        # https://github.com/pandas-dev/pandas/issues/23438
        data = date_range("2017", periods=4, freq="ME")
        if box is None:
            data = data._values
        elif box == "series":
            data = Series(data)

        result = PeriodIndex(data, freq="D")
        expected = PeriodIndex(
            ["2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30"], freq="D"
        )
        tm.assert_index_equal(result, expected)

    def test_constructor_dtype(self):
        # [FUNCTION DOC]
        # Period dtype strings (monthly, 3-day multiple) are honored and set the freq;
        # an existing index is converted when freq/dtype differs; passing a conflicting
        # freq and dtype raises an incompatible-frequency error.
        # KNOWN ISSUE: the first comment says the dtype with a timezone should localize, which has nothing to do with the code that follows (period dtypes).
        # passing a dtype with a tz should localize
        idx = PeriodIndex(["2013-01", "2013-03"], dtype="period[M]")
        exp = PeriodIndex(["2013-01", "2013-03"], freq="M")
        tm.assert_index_equal(idx, exp)
        assert idx.dtype == "period[M]"

        idx = PeriodIndex(["2013-01-05", "2013-03-05"], dtype="period[3D]")
        exp = PeriodIndex(["2013-01-05", "2013-03-05"], freq="3D")
        tm.assert_index_equal(idx, exp)
        assert idx.dtype == "period[3D]"

        # if we already have a freq and its not the same, then asfreq
        # (not changed)
        idx = PeriodIndex(["2013-01-01", "2013-01-02"], freq="D")

        res = PeriodIndex(idx, dtype="period[M]")
        exp = PeriodIndex(["2013-01", "2013-01"], freq="M")
        tm.assert_index_equal(res, exp)
        assert res.dtype == "period[M]"

        res = PeriodIndex(idx, freq="M")
        tm.assert_index_equal(res, exp)
        assert res.dtype == "period[M]"

        msg = "specified freq and dtype are different"
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(["2011-01"], freq="M", dtype="period[D]")

    def test_constructor_empty(self):
        # [FUNCTION DOC]
        # An empty monthly index is valid with month-end frequency; an empty index
        # without freq raises a value error.
        idx = PeriodIndex([], freq="M")
        assert isinstance(idx, PeriodIndex)
        assert len(idx) == 0
        assert idx.freq == "ME"

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex([])

    def test_constructor_pi_nat(self):
        # [FUNCTION DOC]
        # Periods mixed with NaT (lists and arrays, NaT first or not) construct with
        # inferred frequency, strings plus NaT with explicit freq work, and all-NaT
        # input without freq raises a value error.
        idx = PeriodIndex(
            [Period("2011-01", freq="M"), NaT, Period("2011-01", freq="M")]
        )
        exp = PeriodIndex(["2011-01", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            np.array([Period("2011-01", freq="M"), NaT, Period("2011-01", freq="M")])
        )
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            [NaT, NaT, Period("2011-01", freq="M"), Period("2011-01", freq="M")]
        )
        exp = PeriodIndex(["NaT", "NaT", "2011-01", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            np.array(
                [NaT, NaT, Period("2011-01", freq="M"), Period("2011-01", freq="M")]
            )
        )
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex([NaT, NaT, "2011-01", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex([NaT, NaT])

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(np.array([NaT, NaT]))

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(["NaT", "NaT"])

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(np.array(["NaT", "NaT"]))

    def test_constructor_incompat_freq(self):
        # [FUNCTION DOC]
        # A mix of monthly and daily periods raises an incompatible-frequency error,
        # including when the first element is NaT, for lists and arrays.
        msg = "Input has different freq=D from PeriodIndex\\(freq=M\\)"

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex([Period("2011-01", freq="M"), NaT, Period("2011-01", freq="D")])

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(
                np.array(
                    [Period("2011-01", freq="M"), NaT, Period("2011-01", freq="D")]
                )
            )

        # first element is NaT
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex([NaT, Period("2011-01", freq="M"), Period("2011-01", freq="D")])

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(
                np.array(
                    [NaT, Period("2011-01", freq="M"), Period("2011-01", freq="D")]
                )
            )

    def test_constructor_mixed(self):
        # [FUNCTION DOC]
        # Strings, NaT and Period objects combined infer frequency from the Period (or
        # the first parsed element).
        idx = PeriodIndex(["2011-01", NaT, Period("2011-01", freq="M")])
        exp = PeriodIndex(["2011-01", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(["NaT", NaT, Period("2011-01", freq="M")])
        exp = PeriodIndex(["NaT", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex([Period("2011-01-01", freq="D"), NaT, "2012-01-01"])
        exp = PeriodIndex(["2011-01-01", "NaT", "2012-01-01"], freq="D")
        tm.assert_index_equal(idx, exp)

    @pytest.mark.parametrize("floats", [[1.1, 2.1], np.array([1.1, 2.1])])
    def test_constructor_floats(self, floats):
        # [FUNCTION DOC]
        # Parametrized over list and array. Floating point input raises a type error.
        msg = "PeriodIndex does not allow floating point in construction"
        with pytest.raises(TypeError, match=msg):
            PeriodIndex(floats)

    def test_constructor_year_and_quarter(self):
        # [FUNCTION DOC]
        # from_fields with year and quarter Series equals building from quarter strings.
        year = Series([2001, 2002, 2003])
        quarter = year - 2000
        idx = PeriodIndex.from_fields(year=year, quarter=quarter)
        strs = [f"{t[0]:d}Q{t[1]:d}" for t in zip(quarter, year)]
        lops = list(map(Period, strs))
        p = PeriodIndex(lops)
        tm.assert_index_equal(p, idx)

    def test_constructor_freq_mult(self):
        # [FUNCTION DOC]
        # Multiples in frequency produce spaced ranges (monthly, 3-day, 4-hourly with
        # end anchor); non-positive multiples raise a value error about span (GH #7811).
        # GH #7811
        pidx = period_range(start="2014-01", freq="2M", periods=4)
        expected = PeriodIndex(["2014-01", "2014-03", "2014-05", "2014-07"], freq="2M")
        tm.assert_index_equal(pidx, expected)

        pidx = period_range(start="2014-01-02", end="2014-01-15", freq="3D")
        expected = PeriodIndex(
            ["2014-01-02", "2014-01-05", "2014-01-08", "2014-01-11", "2014-01-14"],
            freq="3D",
        )
        tm.assert_index_equal(pidx, expected)

        pidx = period_range(end="2014-01-01 17:00", freq="4h", periods=3)
        expected = PeriodIndex(
            ["2014-01-01 09:00", "2014-01-01 13:00", "2014-01-01 17:00"], freq="4h"
        )
        tm.assert_index_equal(pidx, expected)

        msg = "Frequency must be positive, because it represents span: -1M"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2011-01"], freq="-1M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2011-01"], freq="0M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            period_range("2011-01", periods=3, freq="0M")

    @pytest.mark.parametrize(
        "freq_offset, freq_period",
        [
            ("YE", "Y"),
            ("ME", "M"),
            ("D", "D"),
            ("min", "min"),
            ("s", "s"),
        ],
    )
    @pytest.mark.parametrize("mult", [1, 2, 3, 4, 5])
    def test_constructor_freq_mult_dti_compat(self, mult, freq_offset, freq_period):
        # [FUNCTION DOC]
        # Parametrized over multiple 1-5 and pairs of datetime/period frequency aliases.
        # period_range equals a date range converted to periods.
        freqstr_offset = str(mult) + freq_offset
        freqstr_period = str(mult) + freq_period
        pidx = period_range(start="2014-04-01", freq=freqstr_period, periods=10)
        expected = date_range(
            start="2014-04-01", freq=freqstr_offset, periods=10
        ).to_period(freqstr_period)
        tm.assert_index_equal(pidx, expected)

    @pytest.mark.parametrize("mult", [1, 2, 3, 4, 5])
    def test_constructor_freq_mult_dti_compat_month(self, mult):
        # [FUNCTION DOC]
        # Parametrized over multiples. Same compatibility check for months.
        pidx = period_range(start="2014-04-01", freq=f"{mult}M", periods=10)
        expected = date_range(
            start="2014-04-01", freq=f"{mult}ME", periods=10
        ).to_period(f"{mult}M")
        tm.assert_index_equal(pidx, expected)

    def test_constructor_freq_combined(self):
        # [FUNCTION DOC]
        # Combined frequencies like day-plus-hour build ranges with a 25-hour frequency.
        # KNOWN ISSUE: in the first loop the constructed index and expected index are never compared; only the second loop asserts equality.
        for freq in ["1D1h", "1h1D"]:
            pidx = PeriodIndex(["2016-01-01", "2016-01-02"], freq=freq)
            expected = PeriodIndex(["2016-01-01 00:00", "2016-01-02 00:00"], freq="25h")
        for freq in ["1D1h", "1h1D"]:
            pidx = period_range(start="2016-01-01", periods=2, freq=freq)
            expected = PeriodIndex(["2016-01-01 00:00", "2016-01-02 01:00"], freq="25h")
            tm.assert_index_equal(pidx, expected)

    def test_period_range_length(self):
        # [FUNCTION DOC]
        # Checks the number of periods for ranges from yearly to secondly, including
        # business days (with deprecation warning) and
        # anchor-from-end/start-with-periods ranges; weekly alias warning with
        # deprecation.
        pi = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 9

        pi = period_range(freq="Q", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 4 * 9

        pi = period_range(freq="M", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 12 * 9

        pi = period_range(freq="D", start="1/1/2001", end="12/31/2009")
        assert len(pi) == 365 * 9 + 2

        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            pi = period_range(freq="B", start="1/1/2001", end="12/31/2009")
        assert len(pi) == 261 * 9

        pi = period_range(freq="h", start="1/1/2001", end="12/31/2001 23:00")
        assert len(pi) == 365 * 24

        pi = period_range(freq="Min", start="1/1/2001", end="1/1/2001 23:59")
        assert len(pi) == 24 * 60

        pi = period_range(freq="s", start="1/1/2001", end="1/1/2001 23:59:59")
        assert len(pi) == 24 * 60 * 60

        with tm.assert_produces_warning(FutureWarning, match=msg):
            start = Period("02-Apr-2005", "B")
            i1 = period_range(start=start, periods=20)
        assert len(i1) == 20
        assert i1.freq == start.freq
        assert i1[0] == start

        end_intv = Period("2006-12-31", "W")
        i1 = period_range(end=end_intv, periods=10)
        assert len(i1) == 10
        assert i1.freq == end_intv.freq
        assert i1[-1] == end_intv

        msg = "'w' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            end_intv = Period("2006-12-31", "1w")
        i2 = period_range(end=end_intv, periods=10)
        assert len(i1) == len(i2)
        assert (i1 == i2).all()
        assert i1.freq == i2.freq

    def test_infer_freq_from_first_element(self):
        # [FUNCTION DOC]
        # With business-day periods, range and index construction infer the frequency
        # from the first element; deprecation warnings expected.
        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            start = Period("02-Apr-2005", "B")
            end_intv = Period("2005-05-01", "B")
            period_range(start=start, end=end_intv)

            # infer freq from first element
            i2 = PeriodIndex([end_intv, Period("2005-05-05", "B")])
        assert len(i2) == 2
        assert i2[0] == end_intv

        with tm.assert_produces_warning(FutureWarning, match=msg):
            i2 = PeriodIndex(np.array([end_intv, Period("2005-05-05", "B")]))
        assert len(i2) == 2
        assert i2[0] == end_intv

    def test_mixed_freq_raises(self):
        # [FUNCTION DOC]
        # Mixing business and weekly periods raises an incompatible-frequency error for
        # list and array inputs, while emitting a deprecation warning.
        # Mixed freq should fail
        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            end_intv = Period("2005-05-01", "B")

        vals = [end_intv, Period("2006-12-31", "W")]
        msg = r"Input has different freq=W-SUN from PeriodIndex\(freq=B\)"
        depr_msg = r"PeriodDtype\[B\] is deprecated"
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                PeriodIndex(vals)
        vals = np.array(vals)
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                PeriodIndex(vals)

    @pytest.mark.parametrize(
        "freq", ["M", "Q", "Y", "D", "B", "min", "s", "ms", "us", "ns", "h"]
    )
    @pytest.mark.filterwarnings(
        r"ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_recreate_from_data(self, freq):
        # [FUNCTION DOC]
        # Parametrized over many frequencies. Rebuilding from a one-element range's
        # values and frequency gives the equal index (business-day deprecation warnings
        # ignored).
        org = period_range(start="2001/04/01", freq=freq, periods=1)
        idx = PeriodIndex(org.values, freq=freq)
        tm.assert_index_equal(idx, org)

    def test_map_with_string_constructor(self):
        # [FUNCTION DOC]
        # Mapping the string constructor returns a plain Index of strings equal to the
        # expected strings.
        raw = [2005, 2007, 2009]
        index = PeriodIndex(raw, freq="Y")

        expected = Index([str(num) for num in raw])
        res = index.map(str)

        # should return an Index
        assert isinstance(res, Index)

        # preserve element types
        assert all(isinstance(resi, str) for resi in res)

        # lastly, values should compare equal
        tm.assert_index_equal(res, expected)


class TestSimpleNew:
    # [CLASS DOC]
    # Groups tests of the private _simple_new constructor.
    def test_constructor_simple_new(self):
        # [FUNCTION DOC]
        # Passing an index raises an assertion error; passing the backing array or an
        # integer array with dtype works; an integer Index instead of an array raises an
        # assertion error.
        idx = period_range("2007-01", name="p", periods=2, freq="M")

        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name="p")

        result = idx._simple_new(idx._data, name="p")
        tm.assert_index_equal(result, idx)

        msg = "Should be numpy array of type i8"
        with pytest.raises(AssertionError, match=msg):
            # Need ndarray, not int64 Index
            type(idx._data)._simple_new(Index(idx.asi8), dtype=idx.dtype)

        arr = type(idx._data)._simple_new(idx.asi8, dtype=idx.dtype)
        result = idx._simple_new(arr, name="p")
        tm.assert_index_equal(result, idx)

    def test_constructor_simple_new_empty(self):
        # [FUNCTION DOC]
        # Same behavior on an empty index (GH13079).
        # GH13079
        idx = PeriodIndex([], freq="M", name="p")
        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name="p")

        result = idx._simple_new(idx._data, name="p")
        tm.assert_index_equal(result, idx)

    @pytest.mark.parametrize("floats", [[1.1, 2.1], np.array([1.1, 2.1])])
    def test_period_index_simple_new_disallows_floats(self, floats):
        # [FUNCTION DOC]
        # Parametrized over list and array of floats. Raises an assertion error.
        with pytest.raises(AssertionError, match="<class "):
            PeriodIndex._simple_new(floats)


class TestShallowCopy:
    # [CLASS DOC]
    # Groups tests of the private shallow-copy helpers.
    def test_shallow_copy_empty(self):
        # [FUNCTION DOC]
        # Viewing an empty index returns an equal index (GH#13067).
        # GH#13067
        idx = PeriodIndex([], freq="M")
        result = idx._view()
        expected = idx

        tm.assert_index_equal(result, expected)

    def test_shallow_copy_disallow_i8(self):
        # [FUNCTION DOC]
        # Shallow copy from an integer array raises an assertion error (GH#24391).
        # GH#24391
        pi = period_range("2018-01-01", periods=3, freq="2D")
        with pytest.raises(AssertionError, match="ndarray"):
            pi._shallow_copy(pi.asi8)

    def test_shallow_copy_requires_disallow_period_index(self):
        # [FUNCTION DOC]
        # Shallow copy from a PeriodIndex raises an assertion error.
        pi = period_range("2018-01-01", periods=3, freq="2D")
        with pytest.raises(AssertionError, match="PeriodIndex"):
            pi._shallow_copy(pi)


class TestSeriesPeriod:
    # [CLASS DOC]
    # Groups tests of Series construction from period data.
    def test_constructor_cant_cast_period(self):
        # [FUNCTION DOC]
        # Building a Series from a PeriodIndex with float dtype raises a type error.
        msg = "Cannot cast PeriodIndex to dtype float64"
        with pytest.raises(TypeError, match=msg):
            Series(period_range("2000-01-01", periods=10, freq="D"), dtype=float)

    def test_constructor_cast_object(self):
        # [FUNCTION DOC]
        # Casting a PeriodIndex to the same period dtype in a Series gives an equal
        # Series.
        pi = period_range("1/1/2000", periods=10)
        ser = Series(pi, dtype=PeriodDtype("D"))
        exp = Series(pi)
        tm.assert_series_equal(ser, exp)
