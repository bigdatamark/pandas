# [MODULE DOC]
# Tests for Series item assignment (set item), organized in classes by key kind and
# dtype: datetime64 values, scalar and slice indexers, boolean masks, copy and view
# semantics, callables, categorical, integer/float/boolean/string upcasting, extension
# arrays, and many regression cases for sets with expansion. As of version 3.0 integer
# keys are labels, assignments that cannot be held losslessly raise a type error instead
# of upcasting silently, and copy on write is in effect.
import contextlib
from datetime import (
    date,
    datetime,
)
from decimal import Decimal

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError

from pandas.core.dtypes.common import is_list_like

from pandas import (
    NA,
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    Series,
    StringDtype,
    Timedelta,
    Timestamp,
    array,
    concat,
    date_range,
    interval_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import BDay


class TestSetitemDT64Values:
    # [CLASS DOC]
    # Groups setitem tests on datetime64 values. Stateless.
    def test_setitem_none_nan(self):
        # [FUNCTION DOC]
        # Verifies assigning None or NaN (scalar or slice) into a datetime series gives
        # NaT.
        series = Series(date_range("1/1/2000", periods=10, unit="ns"))
        series[3] = None
        assert series[3] is NaT

        series[3:5] = None
        assert series[4] is NaT

        series[5] = np.nan
        assert series[5] is NaT

        series[5:7] = np.nan
        assert series[6] is NaT

    def test_setitem_multiindex_empty_slice(self):
        # [FUNCTION DOC]
        # Regression test for GH 35878. Assigning to loc with an empty list on a
        # multi-index series changes nothing.
        # https://github.com/pandas-dev/pandas/issues/35878
        idx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        result = Series([1, 2], index=idx)
        expected = result.copy()
        result.loc[[]] = 0
        tm.assert_series_equal(result, expected)

    def test_setitem_with_string_index(self):
        # [FUNCTION DOC]
        # Regression test (GH 23451). On an object series with a label named Date,
        # assigning today's date stores the date object and it can be read back by
        # attribute and by label.
        # GH#23451
        # Set object dtype to avoid upcast when setting date.today()
        ser = Series([1, 2, 3], index=["Date", "b", "other"], dtype=object)
        ser["Date"] = date.today()
        assert ser.Date == date.today()
        assert ser["Date"] == date.today()

    def test_setitem_tuple_with_datetimetz_values(self):
        # [FUNCTION DOC]
        # Regression test (GH 20441). A tz-aware datetime series indexed by tuple
        # labels: setting by a tuple label to NaN equals setting the first position to
        # NaN.
        # GH#20441
        arr = date_range("2017", periods=4, tz="US/Eastern", unit="ns")
        index = [(0, 1), (0, 2), (0, 3), (0, 4)]
        result = Series(arr, index=index)
        expected = result.copy()
        result[(0, 1)] = np.nan
        expected.iloc[0] = np.nan
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "UTC", "Asia/Tokyo"])
    def test_setitem_with_tz(self, tz, indexer_sli):
        # [FUNCTION DOC]
        # Parametrized over timezones and the indexer fixture (loc and direct).
        # Assigning a tz-aware Timestamp to one position and a series of Timestamps to
        # two positions keeps the tz-aware dtype and gives the expected values.
        orig = Series(date_range("2016-01-01", freq="h", periods=3, tz=tz, unit="ns"))
        assert orig.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2016-01-01 02:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        # scalar
        ser = orig.copy()
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        tm.assert_series_equal(ser, exp)

        # vector
        vals = Series(
            [Timestamp("2011-01-01", tz=tz), Timestamp("2012-01-01", tz=tz)],
            index=[1, 2],
            dtype=orig.dtype,
        )
        assert vals.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2012-01-01 00:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        ser = orig.copy()
        indexer_sli(ser)[[1, 2]] = vals
        tm.assert_series_equal(ser, exp)

    def test_setitem_with_tz_dst(self, indexer_sli):
        # [FUNCTION DOC]
        # Regression test (GH 14146). Same as above for US/Eastern around the fall-back
        # transition: scalar and vector assignments keep the dtype and give the expected
        # wall times and offsets.
        # GH#14146 trouble setting values near DST boundary
        tz = "US/Eastern"
        orig = Series(date_range("2016-11-06", freq="h", periods=3, tz=tz, unit="ns"))
        assert orig.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-11-06 00:00-04:00", tz=tz),
                Timestamp("2011-01-01 00:00-05:00", tz=tz),
                Timestamp("2016-11-06 01:00-05:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        # scalar
        ser = orig.copy()
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        tm.assert_series_equal(ser, exp)

        # vector
        vals = Series(
            [Timestamp("2011-01-01", tz=tz), Timestamp("2012-01-01", tz=tz)],
            index=[1, 2],
            dtype=orig.dtype,
        )
        assert vals.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-11-06 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2012-01-01 00:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        ser = orig.copy()
        indexer_sli(ser)[[1, 2]] = vals
        tm.assert_series_equal(ser, exp)

    def test_object_series_setitem_dt64array_exact_match(self):
        # [FUNCTION DOC]
        # Verifies that a numpy datetime64 array assigned positionally to an object
        # series with a boolean list key is stored as numpy datetimes (not cast to
        # integers by numpy, issue 12550), with the expected object result.
        # make sure the dt64 isn't cast by numpy to integers
        # https://github.com/numpy/numpy/issues/12550

        ser = Series({"X": np.nan}, dtype=object)

        indexer = [True]

        # "exact_match" -> size of array being set matches size of ser
        value = np.array([4], dtype="M8[ns]")

        ser.iloc[indexer] = value

        expected = Series([value[0]], index=["X"], dtype=object)
        assert all(isinstance(x, np.datetime64) for x in expected.values)

        tm.assert_series_equal(ser, expected)


class TestSetitemScalarIndexer:
    # [CLASS DOC]
    # Groups setitem tests with scalar indexers. Stateless.
    def test_setitem_negative_out_of_bounds(self):
        # [FUNCTION DOC]
        # Verifies, as of version 3.0, integer keys are labels so a negative
        # out-of-range key enlarges the series with that integer label.
        # As of 3.0, int keys are treated as labels, so this becomes
        #  setitem-with-expansion
        ser = Series(["a"] * 10, index=["a"] * 10)
        ser[-11] = "foo"
        exp = Series(["a"] * 10 + ["foo"], index=["a"] * 10 + [-11])
        tm.assert_series_equal(ser, exp)

    @pytest.mark.parametrize("indexer", [tm.loc, tm.at])
    @pytest.mark.parametrize("ser_index", [0, 1])
    def test_setitem_series_object_dtype(self, indexer, ser_index):
        # [FUNCTION DOC]
        # Regression test (GH 38303). Parametrized over the loc and at indexers and
        # index label of the assigned series. Assigning a series to a scalar position in
        # an object series stores the series as an element.
        # GH#38303
        ser = Series([0, 0], dtype="object")
        idxr = indexer(ser)
        idxr[0] = Series([42], index=[ser_index])
        expected = Series([Series([42], index=[ser_index]), 0], dtype="object")
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("index, exp_value", [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index, exp_value):
        # [FUNCTION DOC]
        # Regression test (GH 38303). Parametrized over matching or non-matching labels.
        # Assigning a one-element series to a scalar position in a numeric series uses
        # the element aligned by label: value 42 or NaN.
        # GH#38303
        ser = Series([0, 0])
        ser.loc[0] = Series([42], index=[index])
        expected = Series([exp_value, 0])
        tm.assert_series_equal(ser, expected)


class TestSetitemSlices:
    # [CLASS DOC]
    # Groups setitem tests with slice indexers. Stateless.
    def test_setitem_slice_float_raises(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies a float slice on a datetime index raises a type error saying slice
        # indexing with floats is not possible, naming the bound.
        msg = (
            "cannot do slice indexing on DatetimeIndex with these indexers "
            r"\[{key}\] of type float"
        )
        with pytest.raises(TypeError, match=msg.format(key=r"4\.0")):
            datetime_series[4.0:10.0] = 0

        with pytest.raises(TypeError, match=msg.format(key=r"4\.5")):
            datetime_series[4.5:10.0] = 0

    def test_setitem_slice(self):
        # [FUNCTION DOC]
        # Verifies out-of-range negative slice bounds clamp: assigning from before the
        # start sets everything; assigning up to before the start sets nothing.
        ser = Series(range(10), index=list(range(10)))
        ser[-12:] = 0
        assert (ser == 0).all()

        ser[:-12] = 5
        assert (ser == 0).all()

    def test_setitem_slice_integers(self):
        # [FUNCTION DOC]
        # Verifies integer slice assignment is positional on an even-integer index and
        # only affects the selected positions.
        ser = Series(
            np.random.default_rng(2).standard_normal(8),
            index=[2, 4, 6, 8, 10, 12, 14, 16],
        )

        ser[:4] = 0
        assert (ser[:4] == 0).all()
        assert not (ser[4:] == 0).any()

    def test_setitem_slicestep(self):
        # [FUNCTION DOC]
        # Verifies a stepped slice assignment on a float series sets every other
        # element.
        # caught this bug when writing tests
        series = Series(
            np.arange(20, dtype=np.float64), index=np.arange(20, dtype=np.int64)
        )

        series[::2] = 0
        assert (series[::2] == 0).all()

    def test_setitem_multiindex_slice(self, indexer_sli):
        # [FUNCTION DOC]
        # Regression test (GH 8856). With the indexer fixture, assigning to a stepped
        # slice of a multi-index series sets every fourth element.
        # GH 8856
        mi = MultiIndex.from_product(([0, 1], list("abcde")))
        result = Series(np.arange(10, dtype=np.int64), mi)
        indexer_sli(result)[::4] = 100
        expected = Series([100, 1, 2, 3, 100, 5, 6, 7, 100, 9], mi)
        tm.assert_series_equal(result, expected)


class TestSetitemBooleanMask:
    # [CLASS DOC]
    # Groups setitem tests with boolean mask indexers. Stateless.
    def test_setitem_mask_cast(self):
        # [FUNCTION DOC]
        # Regression test (GH 2746). Assigning an aligned one-element integer series
        # with a boolean-list mask changes only the selected element and keeps the
        # dtype.
        # GH#2746
        # need to upcast
        ser = Series([1, 2], index=[1, 2], dtype="int64")
        ser[[True, False]] = Series([0], index=[1], dtype="int64")
        expected = Series([0, 2], index=[1, 2], dtype="int64")

        tm.assert_series_equal(ser, expected)

    def test_setitem_mask_align_and_promote(self):
        # [FUNCTION DOC]
        # Regression test (GH 8387). Assigning a string-converted series into a float
        # series through a boolean mask raises a type error (invalid value) instead of
        # promoting.
        # KNOWN ISSUE: the name says the dtype is promoted but the behavior is now to
        # raise an error.
        # GH#8387: test that changing types does not break alignment
        ts = Series(
            np.random.default_rng(2).standard_normal(100), index=np.arange(100, 0, -1)
        ).round(5)
        mask = ts > 0
        left = ts.copy()
        right = ts[mask].copy().map(str)
        with pytest.raises(TypeError, match="Invalid value"):
            left[mask] = right

    def test_setitem_mask_promote_strs(self):
        # [FUNCTION DOC]
        # Assigning strings into an integer series through a mask raises a type error
        # (invalid value) instead of promoting the dtype.
        # KNOWN ISSUE: the name says promote but the behavior is to raise.
        ser = Series([0, 1, 2, 0])
        mask = ser > 0
        ser2 = ser[mask].map(str)
        with pytest.raises(TypeError, match="Invalid value"):
            ser[mask] = ser2

    def test_setitem_mask_promote(self):
        # [FUNCTION DOC]
        # Verifies assigning the selected subset of a mixed object series back with the
        # mask leaves the series unchanged.
        ser = Series([0, "foo", "bar", 0])
        mask = Series([False, True, True, False])
        ser2 = ser[mask]
        ser[mask] = ser2

        expected = Series([0, "foo", "bar", 0])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean(self, string_series):
        # [FUNCTION DOC]
        # Verifies boolean-mask assignment from a similar-indexed series and from a
        # shorter series that needs alignment, comparing only the masked part.
        mask = string_series > string_series.median()

        # similar indexed series
        result = string_series.copy()
        result[mask] = string_series * 2
        expected = string_series * 2
        tm.assert_series_equal(result[mask], expected[mask])

        # needs alignment
        result = string_series.copy()
        result[mask] = (string_series * 2)[0:5]
        expected = (string_series * 2)[0:5].reindex_like(string_series)
        expected[-mask] = string_series[mask]
        tm.assert_series_equal(result[mask], expected[mask])

    def test_setitem_boolean_corner(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies a mask built from a shifted series (not aligned) raises an indexing
        # error about an unalignable boolean series for direct set and loc set.
        ts = datetime_series
        mask_shifted = ts.shift(1, freq=BDay()) > ts.median()

        msg = (
            r"Unalignable boolean Series provided as indexer \(index of "
            r"the boolean Series and of the indexed object do not match"
        )
        with pytest.raises(IndexingError, match=msg):
            ts[mask_shifted] = 1

        with pytest.raises(IndexingError, match=msg):
            ts.loc[mask_shifted] = 1

    def test_setitem_boolean_different_order(self, string_series):
        # [FUNCTION DOC]
        # Verifies a mask computed from a sorted copy (different order) is aligned,
        # giving the same assignment as the mask on the original order.
        ordered = string_series.sort_values()

        copy = string_series.copy()
        copy[ordered > 0] = 0

        expected = string_series.copy()
        expected[expected > 0] = 0

        tm.assert_series_equal(copy, expected)

    @pytest.mark.parametrize("func", [list, np.array, Series])
    def test_setitem_boolean_python_list(self, func):
        # [FUNCTION DOC]
        # Regression test (GH 19406). Parametrized over list, array and series masks.
        # Assigning a list of two strings through the mask into an object series fills
        # the true positions.
        # GH19406
        ser = Series([None, "b", None])
        mask = func([True, False, True])
        ser[mask] = ["a", "c"]
        expected = Series(["a", "b", "c"])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean_nullable_int_types(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Regression test (GH 26468). Parametrized over nullable numeric dtypes.
        # Assigning an extension series through a boolean mask on direct set, loc and
        # with a filtered series gives the expected series.
        # GH: 26468
        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        ser[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)
        expected = Series([5, 6, 2, 3], dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)

        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        ser.loc[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)

        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        loc_ser = Series(range(4), dtype=any_numeric_ea_dtype)
        ser.loc[ser > 6] = loc_ser.loc[loc_ser > 1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self):
        # [FUNCTION DOC]
        # Regression test (GH 30567). A list mask with five true values and a range of
        # five values assigns in order, keeping None elsewhere and object dtype.
        # GH#30567
        ser = Series([None] * 10)
        mask = [False] * 3 + [True] * 5 + [False] * 2
        ser[mask] = range(5)
        result = ser
        expected = Series([None] * 3 + list(range(5)) + [None] * 2, dtype=object)
        tm.assert_series_equal(result, expected)

    def test_setitem_nan_with_bool(self):
        # [FUNCTION DOC]
        # Regression test (GH 13034). Assigning NaN into a boolean series raises a type
        # error (invalid value).
        # GH 13034
        result = Series([True, False, True])
        with pytest.raises(TypeError, match="Invalid value"):
            result[0] = np.nan

    def test_setitem_mask_smallint_upcast(self):
        # [FUNCTION DOC]
        # Verifies assigning values not representable in int8 raises a type error
        # (invalid value) for set and in-place mask; but where with a series replacement
        # gives a default-integer result.
        orig = Series([1, 2, 3], dtype="int8")
        alt = np.array([999, 1000, 1001], dtype=np.int64)

        mask = np.array([True, False, True])

        ser = orig.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            ser[mask] = Series(alt)

        with pytest.raises(TypeError, match="Invalid value"):
            ser.mask(mask, alt, inplace=True)

        res = ser.where(~mask, Series(alt))
        expected = Series([999, 2, 1001])
        tm.assert_series_equal(res, expected)

    def test_setitem_mask_smallint_no_upcast(self):
        # [FUNCTION DOC]
        # Like the previous test but the selected values fit in uint8: set and in-place
        # mask keep the narrow dtype; where yields wider values compared without the
        # dtype check.
        # KNOWN ISSUE: a TODO in the code notes that where unnecessarily upcasts to a
        # wider integer type; that is why the dtype is not compared for the where
        # result.
        # like test_setitem_mask_smallint_upcast, but while we can't hold 'alt',
        #  we *can* hold alt[mask] without casting
        orig = Series([1, 2, 3], dtype="uint8")
        alt = Series([245, 1000, 246], dtype=np.int64)

        mask = np.array([True, False, True])

        ser = orig.copy()
        ser[mask] = alt
        expected = Series([245, 2, 246], dtype="uint8")
        tm.assert_series_equal(ser, expected)

        ser2 = orig.copy()
        ser2.mask(mask, alt, inplace=True)
        tm.assert_series_equal(ser2, expected)

        # TODO: ser.where(~mask, alt) unnecessarily upcasts to int64
        ser3 = orig.copy()
        res = ser3.where(~mask, alt)
        tm.assert_series_equal(res, expected, check_dtype=False)


class TestSetitemViewCopySemantics:
    # [CLASS DOC]
    # Groups setitem tests of copy and view semantics. Stateless.
    def test_setitem_invalidates_datetime_index_freq(self):
        # [FUNCTION DOC]
        # Regression test (GH 24096). Modifying a tz-aware series built from a datetime
        # index in place does not change the index and drops the frequency from the
        # series values; the series shares memory with the index until modified, after
        # which it does not.
        # GH#24096 altering a datetime64tz Series inplace invalidates the
        #  `freq` attribute on the underlying DatetimeIndex

        dti = date_range("20130101", periods=3, tz="US/Eastern")
        ts = dti[1]
        ser = Series(dti)
        assert ser._values is not dti
        assert ser._values._ndarray.base is dti._data._ndarray.base
        assert dti.freq == "D"
        ser.iloc[1] = NaT
        assert ser._values.freq is None

        # check that the DatetimeIndex was not altered in place
        assert ser._values is not dti
        assert ser._values._ndarray.base is not dti._data._ndarray.base
        assert dti[1] == ts
        assert dti.freq == "D"

    def test_dt64tz_setitem_does_not_mutate_dti(self):
        # [FUNCTION DOC]
        # Regression tests (GH 21907, GH 24096). A series built from a tz-aware index
        # shares the underlying array with it at first; assigning NaT to every third
        # element does not mutate the index.
        # GH#21907, GH#24096
        dti = date_range("2016-01-01", periods=10, tz="US/Pacific")
        ts = dti[0]
        ser = Series(dti)
        assert ser._values is not dti
        assert ser._values._ndarray.base is dti._data._ndarray.base
        assert ser._mgr.blocks[0].values._ndarray.base is dti._data._ndarray.base

        assert ser._mgr.blocks[0].values is not dti

        ser[::3] = NaT
        assert ser[0] is NaT
        assert dti[0] == ts


class TestSetitemCallable:
    # [CLASS DOC]
    # Groups setitem tests with callable keys. Stateless.
    def test_setitem_callable_key(self):
        # [FUNCTION DOC]
        # Regression test (GH 12533). A function key is evaluated with the series and
        # the result used as label for assignment.
        # GH#12533
        ser = Series([1, 2, 3, 4], index=list("ABCD"))
        ser[lambda x: "A"] = -1

        expected = Series([-1, 2, 3, 4], index=list("ABCD"))
        tm.assert_series_equal(ser, expected)

    def test_setitem_callable_other(self):
        # [FUNCTION DOC]
        # Regression test (GH 13299). A callable stored as a value (the key is a boolean
        # mask) is assigned as an object, not called, in an object series.
        # GH#13299
        inc = lambda x: x + 1

        # set object dtype to avoid upcast when setting inc
        ser = Series([1, 2, -1, 4], dtype=object)
        ser[ser < 0] = inc

        expected = Series([1, 2, inc, 4])
        tm.assert_series_equal(ser, expected)


class TestSetitemWithExpansion:
    # [CLASS DOC]
    # Groups setitem tests that enlarge the series by assigning to a new label.
    # Stateless.
    def test_setitem_empty_series(self, using_infer_string):
        # [FUNCTION DOC]
        # Regression test (GH 10193). Assigning a Timestamp key into an empty object
        # series gives a one-element series indexed by an object index (or a datetime
        # index when the infer-string option is on).
        # GH#10193
        key = Timestamp("2012-01-01")
        series = Series(dtype=object)
        series[key] = 47
        expected = Series(
            47, index=[key] if using_infer_string else Index([key], dtype=object)
        )
        tm.assert_series_equal(series, expected)

    def test_setitem_empty_series_datetimeindex_preserves_freq(self):
        # [FUNCTION DOC]
        # Regression test (GH 33573). Assigning a new timestamp key into an empty series
        # with a daily-frequency datetime index keeps the index frequency.
        # GH#33573 our index should retain its freq
        dti = DatetimeIndex([], freq="D", dtype="M8[ns]")
        series = Series([], index=dti, dtype=object)
        key = Timestamp("2012-01-01")
        series[key] = 47
        expected = Series(47, DatetimeIndex([key], freq="D").as_unit("ns"))
        tm.assert_series_equal(series, expected)
        assert series.index.freq == expected.index.freq

    def test_setitem_empty_series_timestamp_preserves_dtype(self):
        # [FUNCTION DOC]
        # Regression test (GH 21881). Setting a number and then a Timestamp in an
        # initially empty object series keeps the Timestamp value equal to the original.
        # GH 21881
        timestamp = Timestamp(1412526600000000000)
        series = Series([timestamp], index=["timestamp"], dtype=object)
        expected = series["timestamp"]

        series = Series([], dtype=object)
        series["anything"] = 300.0
        series["timestamp"] = timestamp
        result = series["timestamp"]
        assert result == expected

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta("9 days"),
            Timedelta("9 days").to_timedelta64(),
            Timedelta("9 days").to_pytimedelta(),
        ],
    )
    def test_append_timedelta_does_not_cast(self, td, using_infer_string, request):
        # [FUNCTION DOC]
        # Regression test (GH 22717). Parametrized over Timedelta, numpy and python
        # timedeltas. Appending a timedelta by direct set or loc to an object series
        # stores a Timedelta in an object series and does not cast it to integer;
        # expected failure for non-Timedelta inputs when the infer-string option is on
        # (GH 56010).
        # GH#22717 inserting a Timedelta should _not_ cast to int64
        if using_infer_string and not isinstance(td, Timedelta):
            # TODO: GH#56010
            request.applymarker(pytest.mark.xfail(reason="inferred as string"))

        expected = Series(["x", td], index=[0, "td"], dtype=object)

        ser = Series(["x"])
        ser["td"] = td
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser["td"], Timedelta)

        ser = Series(["x"])
        ser.loc["td"] = Timedelta("9 days")
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser["td"], Timedelta)

    def test_setitem_with_expansion_type_promotion(self):
        # [FUNCTION DOC]
        # Regression test (GH 12599). Successive assignments of a Timestamp, float and
        # string to new labels in an empty object series keep the original objects.
        # GH#12599
        ser = Series(dtype=object)
        ser["a"] = Timestamp("2016-01-01")
        ser["b"] = 3.0
        ser["c"] = "foo"
        expected = Series([Timestamp("2016-01-01"), 3.0, "foo"], index=["a", "b", "c"])
        tm.assert_series_equal(ser, expected)

    def test_setitem_not_contained(self, string_series):
        # [FUNCTION DOC]
        # Verifies assigning to a label not in a string-indexed series appends it, equal
        # to concatenating a one-element series.
        # set item that's not contained
        ser = string_series.copy()
        assert "foobar" not in ser.index
        ser["foobar"] = 1

        app = Series([1], index=["foobar"], name="series")
        expected = concat([string_series, app])
        tm.assert_series_equal(ser, expected)

    def test_setitem_keep_precision(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Regression test (GH 32346). Parametrized over nullable numeric dtypes.
        # Enlarging with an integer label and value keeps the extension dtype.
        # GH#32346
        ser = Series([1, 2], dtype=any_numeric_ea_dtype)
        ser[2] = 10
        expected = Series([1, 2, 10], dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "na, target_na, dtype, target_dtype, indexer, raises",
        [
            (NA, NA, "Int64", "Int64", 1, False),
            (NA, NA, "Int64", "Int64", 2, False),
            (NA, np.nan, "int64", "float64", 1, False),
            (NA, np.nan, "int64", "float64", 2, False),
            (NaT, NaT, "int64", "object", 1, True),
            (NaT, NaT, "int64", "object", 2, False),
            (np.nan, NA, "Int64", "Int64", 1, False),
            (np.nan, NA, "Int64", "Int64", 2, False),
            (np.nan, NA, "Float64", "Float64", 1, False),
            (np.nan, NA, "Float64", "Float64", 2, False),
            (np.nan, np.nan, "int64", "float64", 1, False),
            (np.nan, np.nan, "int64", "float64", 2, False),
        ],
    )
    def test_setitem_enlarge_with_na(
        self, na, target_na, dtype, target_dtype, indexer, raises
    ):
        # [FUNCTION DOC]
        # Regression test (GH 32346). Parametrized over missing value kinds, target
        # missing value, dtypes, indexer label (existing or new) and whether an error is
        # expected. Assigning NA, NaN or NaT to an integer or nullable series either
        # converts the dtype (integers to float) or keeps the nullable dtype and stores
        # NA; assigning NaT to an existing position of an integer series raises an
        # invalid value type error.
        # GH#32346
        ser = Series([1, 2], dtype=dtype)
        if raises:
            with pytest.raises(TypeError, match="Invalid value"):
                ser[indexer] = na
        else:
            ser[indexer] = na
            expected_values = [1, target_na] if indexer == 1 else [1, 2, target_na]
            expected = Series(expected_values, dtype=target_dtype)
            tm.assert_series_equal(ser, expected)

    def test_setitem_enlargement_object_none(self, nulls_fixture, using_infer_string):
        # [FUNCTION DOC]
        # Regression test (GH 48665). Enlarging a string series with each null fixture
        # value gives an object (or string, with the infer-string option on, except for
        # Decimal) series; the stored object is identical to the null fixture when not
        # inferring strings.
        # KNOWN ISSUE: in the infer-string branch the check is an identity expression
        # statement with no assertion, so it verifies nothing.
        # GH#48665
        ser = Series(["a", "b"])
        ser[3] = nulls_fixture
        dtype = (
            "str"
            if using_infer_string and not isinstance(nulls_fixture, Decimal)
            else object
        )
        expected = Series(["a", "b", nulls_fixture], index=[0, 1, 3], dtype=dtype)
        tm.assert_series_equal(ser, expected)
        if using_infer_string:
            ser[3] is np.nan
        else:
            assert ser[3] is nulls_fixture


def test_setitem_scalar_into_readonly_backing_data():
    # [FUNCTION DOC]
    # Regression test (GH 14359). Setting a scalar into a series backed by a read-only
    # array raises a value error saying the destination is read-only for every position,
    # and the array is unchanged.
    # GH#14359: test that you cannot mutate a read only buffer

    array = np.zeros(5)
    array.flags.writeable = False  # make the array immutable
    series = Series(array, copy=False)

    for n in series.index:
        msg = "assignment destination is read-only"
        with pytest.raises(ValueError, match=msg):
            series[n] = 1

        assert array[n] == 0


def test_setitem_slice_into_readonly_backing_data():
    # [FUNCTION DOC]
    # Regression test (GH 14359). Setting a slice into a series backed by a read-only
    # array raises a value error saying the destination is read-only, and the array is
    # unchanged.
    # GH#14359: test that you cannot mutate a read only buffer

    array = np.zeros(5)
    array.flags.writeable = False  # make the array immutable
    series = Series(array, copy=False)

    msg = "assignment destination is read-only"
    with pytest.raises(ValueError, match=msg):
        series[1:3] = 1

    assert not array.any()


def test_setitem_categorical_assigning_ops():
    # [FUNCTION DOC]
    # Verifies assignment into a categorical series by full slice, single position,
    # boolean-from-index mask, boolean list and label (after changing the index) with a
    # value that is a category gives the expected categorical with the same categories.
    orig = Series(Categorical(["b", "b"], categories=["a", "b"]))
    ser = orig.copy()
    ser[:] = "a"
    exp = Series(Categorical(["a", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[1] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[ser.index > 0] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[[False, True]] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser.index = ["x", "y"]
    ser["y"] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]), index=["x", "y"])
    tm.assert_series_equal(ser, exp)


def test_setitem_nan_into_categorical():
    # [FUNCTION DOC]
    # Verifies assigning NaN into a categorical series sets a missing element while
    # keeping the categories.
    # ensure that one can set something to np.nan
    ser = Series(Categorical([1, 2, 3]))
    exp = Series(Categorical([1, np.nan, 3], categories=[1, 2, 3]))
    ser[1] = np.nan
    tm.assert_series_equal(ser, exp)


class TestSetitemCasting:
    # [CLASS DOC]
    # Groups setitem tests where dtype casting is involved. Stateless.
    @pytest.mark.parametrize("unique", [True, False])
    @pytest.mark.parametrize("val", [3, 3.0, "3"], ids=type)
    def test_setitem_non_bool_into_bool(self, val, indexer_sli, unique):
        # [FUNCTION DOC]
        # Parametrized over uniqueness of the index, value types (integer, float, string
        # equal to three) and the loc or iloc indexer fixture. Assigning a non-boolean
        # value into a boolean series raises a type error (invalid value).
        # dont cast these 3-like values to bool
        ser = Series([True, False])
        if not unique:
            ser.index = [1, 1]

        with pytest.raises(TypeError, match="Invalid value"):
            indexer_sli(ser)[1] = val

    def test_setitem_boolean_array_into_npbool(self):
        # [FUNCTION DOC]
        # Regression test (GH 45462). Assigning a nullable boolean array without missing
        # values into a numpy boolean series is done in place (same backing array), but
        # assigning one with a missing value raises a type error (invalid value).
        # GH#45462
        ser = Series([True, False, True])
        values = ser._values
        arr = array([True, False, None])

        ser[:2] = arr[:2]  # no NAs -> can set inplace
        assert ser._values is values

        with pytest.raises(TypeError, match="Invalid value"):
            ser[1:] = arr[1:]  # has an NA -> cast to boolean dtype


class SetitemCastingEquivalents:
    # [CLASS DOC]
    # HELPER: base class for tests that check several equivalent ways of setting a value
    # in a series (setitem through the loc, iloc, at, iat indexers, with integer, slice,
    # list, array, generator and boolean-mask keys, and where, index where and index
    # putmask). Subclasses provide the fixtures: obj (starting series), key, expected
    # (result), val (value), raises (whether an invalid value error is expected). The
    # series is assumed to have the default range index and setting does not expand it.
    """
    Check each of several methods that _should_ be equivalent to `obj[key] = val`

    We assume that
        - obj.index is the default Index(range(len(obj)))
        - the setitem does not expand the obj
    """

    @pytest.fixture
    def is_inplace(self, obj, expected):
        # [FUNCTION DOC]
        # HELPER: fixture returning whether the setting is expected to happen in place:
        # true when the expected result dtype equals the original dtype.
        """
        Whether we expect the setting to be in-place or not.
        """
        return expected.dtype == obj.dtype

    def check_indexer(self, obj, key, expected, val, indexer, is_inplace):
        # [FUNCTION DOC]
        # HELPER: sets the value through the given indexer function on a copy of the
        # series and asserts the result equals the expected series, then checks in-place
        # semantics. Inputs: obj, key, expected, val, indexer (function wrapping the
        # series), is_inplace (bool or none).
        orig = obj
        obj = obj.copy()
        arr = obj._values

        indexer(obj)[key] = val
        tm.assert_series_equal(obj, expected)

        self._check_inplace(is_inplace, orig, arr, obj)

    def _check_inplace(self, is_inplace, orig, arr, obj):
        # [FUNCTION DOC]
        # HELPER: checks in-place behavior. Inputs: is_inplace (none to skip checking),
        # orig (original series), arr (array before the set), obj (series after). If in
        # place, the underlying array object (or its numpy data for datetime-likes) must
        # be the same; if not, the original array must be unchanged.
        if is_inplace is None:
            # We are not (yet) checking whether setting is inplace or not
            pass
        elif is_inplace:
            if arr.dtype.kind in ["m", "M"]:
                # We may not have the same DTA/TDA, but will have the same
                #  underlying data
                assert arr._ndarray is obj._values._ndarray
            else:
                assert obj._values is arr
        else:
            # otherwise original array should be unchanged
            tm.assert_equal(arr, orig._values)

    def test_int_key(self, obj, key, expected, raises, val, indexer_sli, is_inplace):
        # [FUNCTION DOC]
        # Runs the equivalent key forms for an integer key (skipped when the key is not
        # an integer): the given indexer (and at or iat when applicable), a range, a
        # one-element slice (not for loc), a list, an array and a generator. If an error
        # is expected, each must raise a type error (invalid value).
        if not isinstance(key, int):
            pytest.skip("Not relevant for int key")
        if raises:
            ctx = pytest.raises(TypeError, match="Invalid value")
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)

        if indexer_sli is tm.loc:
            with ctx:
                self.check_indexer(obj, key, expected, val, tm.at, is_inplace)
        elif indexer_sli is tm.iloc:
            with ctx:
                self.check_indexer(obj, key, expected, val, tm.iat, is_inplace)

        rng = range(key, key + 1)
        with ctx:
            self.check_indexer(obj, rng, expected, val, indexer_sli, is_inplace)

        if indexer_sli is not tm.loc:
            # Note: no .loc because that handles slice edges differently
            slc = slice(key, key + 1)
            with ctx:
                self.check_indexer(obj, slc, expected, val, indexer_sli, is_inplace)

        ilkey = [key]
        with ctx:
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)

        indkey = np.array(ilkey)
        with ctx:
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)

        genkey = (x for x in [key])
        with ctx:
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_slice_key(self, obj, key, expected, raises, val, indexer_sli, is_inplace):
        # [FUNCTION DOC]
        # Runs the equivalent key forms for a slice key (skipped otherwise): the slice
        # itself (not for loc), the list of covered positions, an array and a generator.
        # If an error is expected, each raises an invalid value type error.
        if not isinstance(key, slice):
            pytest.skip("Not relevant for slice key")
        if raises:
            ctx = pytest.raises(TypeError, match="Invalid value")
        else:
            ctx = contextlib.nullcontext()

        if indexer_sli is not tm.loc:
            # Note: no .loc because that handles slice edges differently
            with ctx:
                self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)

        ilkey = list(range(len(obj)))[key]
        with ctx:
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)

        indkey = np.array(ilkey)
        with ctx:
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)

        genkey = (x for x in indkey)
        with ctx:
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_mask_key(self, obj, key, expected, raises, val, indexer_sli):
        # [FUNCTION DOC]
        # Sets through a boolean mask built from the key. If the value is list-like and
        # shorter than the number of selected positions, an index error about the mask
        # not matching is expected; if an error is expected, a type error is raised;
        # otherwise it succeeds (result not checked).
        # setitem with boolean mask
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True

        obj = obj.copy()

        if is_list_like(val) and len(val) < mask.sum():
            msg = "boolean index did not match indexed array along dimension"
            with pytest.raises(IndexError, match=msg):
                indexer_sli(obj)[mask] = val
            return

        if raises:
            with pytest.raises(TypeError, match="Invalid value"):
                indexer_sli(obj)[mask] = val
        else:
            indexer_sli(obj)[mask] = val

    def test_series_where(self, obj, key, expected, raises, val, is_inplace):
        # [FUNCTION DOC]
        # Checks the where method with the inverted mask and value. If the value is
        # list-like and shorter than the series, a value error about broadcasting is
        # expected. Otherwise the result equals the expected series (after converting
        # missing values to NA or None for object results) and in-place semantics are
        # checked on the original.
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True

        if is_list_like(val) and len(val) < len(obj):
            # Series.where is not valid here
            msg = "operands could not be broadcast together with shapes"
            with pytest.raises(ValueError, match=msg):
                obj.where(~mask, val)
            return

        orig = obj
        obj = obj.copy()
        arr = obj._values

        res = obj.where(~mask, val)

        if val is NA and res.dtype == object:
            expected = expected.fillna(NA)
        elif val is None and res.dtype == object:
            assert expected.dtype == object
            expected = expected.copy()
            expected[expected.isna()] = None
        tm.assert_series_equal(res, expected)

        self._check_inplace(is_inplace, orig, arr, obj)

    def test_index_where(self, obj, key, expected, raises, val):
        # [FUNCTION DOC]
        # Checks the index where method with the inverted mask and value equals an index
        # of the expected values with the expected dtype.
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True

        res = Index(obj).where(~mask, val)
        expected_idx = Index(expected, dtype=expected.dtype)
        tm.assert_index_equal(res, expected_idx)

    def test_index_putmask(self, obj, key, expected, raises, val):
        # [FUNCTION DOC]
        # Checks the index put mask method with the mask and value equals an index of
        # the expected values with the expected dtype.
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True

        res = Index(obj).putmask(mask, val)
        tm.assert_index_equal(res, Index(expected, dtype=expected.dtype))


@pytest.mark.parametrize(
    "obj,expected,key,raises",
    [
        pytest.param(
            # GH#45568 setting a valid NA value into IntervalDtype[int] should
            #  cast to IntervalDtype[float]
            Series(interval_range(1, 5)),
            Series(
                [Interval(1, 2), np.nan, Interval(3, 4), Interval(4, 5)],
                dtype="interval[float64]",
            ),
            1,
            True,
            id="interval_int_na_value",
        ),
        pytest.param(
            # these induce dtype changes
            Series([2, 3, 4, 5, 6, 7, 8, 9, 10]),
            Series([np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]),
            slice(None, None, 2),
            False,
            id="int_series_slice_key_step",
        ),
        pytest.param(
            Series([True, True, False, False]),
            Series([np.nan, True, np.nan, False], dtype=object),
            slice(None, None, 2),
            True,
            id="bool_series_slice_key_step",
        ),
        pytest.param(
            # these induce dtype changes
            Series(np.arange(10)),
            Series([np.nan, np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9]),
            slice(None, 5),
            False,
            id="int_series_slice_key",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([1, 2, 3]),
            Series([np.nan, 2, 3]),
            0,
            False,
            id="int_series_int_key",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([False]),
            Series([np.nan], dtype=object),
            # TODO: maybe go to float64 since we are changing the _whole_ Series?
            0,
            True,
            id="bool_series_int_key_change_all",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([False, True]),
            Series([np.nan, True], dtype=object),
            0,
            True,
            id="bool_series_int_key",
        ),
    ],
)
class TestSetitemCastingEquivalents(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Concrete runner of the shared equivalence checks for missing-value assignments,
    # parametrized (by a class-level decorator) over several starting series, expected
    # results, keys and whether an error is expected: an interval series, integer series
    # with a stepped or plain slice or integer key (dtype changes to float), and boolean
    # series with slice or integer key (dtype becomes object, expected to raise). The
    # class supplies the value fixture.
    # KNOWN ISSUE: a comment on the first parameter set says assigning a valid NA into
    # an integer interval series should cast to a float interval series, but the
    # parameter marks the case as raising an error instead.
    @pytest.fixture(params=[np.nan, np.float64("NaN"), None, NA])
    def val(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over NaN (python float and numpy float64), none
        # and the NA singleton; the values that should be valid missing markers for all
        # dtypes.
        """
        NA values that should generally be valid_na for *all* dtypes.

        Include both python float NaN and np.float64; only np.float64 has a
        `dtype` attribute.
        """
        return request.param


class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Verifies a numpy timedelta64 value set into complex, integer or float series is
    # not treated as a number: an error (invalid value) is expected. Provides all
    # fixtures for the shared checks (key zero).
    # timedelta64 should not be treated as integers when setting into
    #  numeric Series

    @pytest.fixture
    def val(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a four-nanosecond numpy timedelta64.
        td = np.timedelta64(4, "ns")
        return td
        # TODO: could also try np.full((1,), td)

    @pytest.fixture(params=[complex, int, float])
    def dtype(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over complex, int and float numpy conversion
        # types.
        return request.param

    @pytest.fixture
    def obj(self, dtype):
        # [FUNCTION DOC]
        # HELPER: fixture returning a series of range values converted to the dtype
        # fixture.
        arr = np.arange(5).astype(dtype)
        ser = Series(arr)
        return ser

    @pytest.fixture
    def expected(self, dtype):
        # [FUNCTION DOC]
        # HELPER: fixture returning the series as it would be with the first element
        # replaced by the timedelta, as object dtype.
        arr = np.arange(5).astype(dtype)
        ser = Series(arr)
        ser = ser.astype(object)
        ser.iloc[0] = np.timedelta64(4, "ns")
        return ser

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning key zero.
        return 0

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that an error is expected (true).
        return True


class TestSetitemDT64IntoInt(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Regression tests (GH 39619): a datetime64 or timedelta64 numpy scalar (alone or
    # boxed in a list or array) set into an integer series through a slice is not cast
    # to integers; an error (invalid value) is expected. Provides the fixtures for the
    # shared checks.
    # GH#39619 dont cast dt64 to int when doing this setitem

    @pytest.fixture(params=["M8[ns]", "m8[ns]"])
    def dtype(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over nanosecond datetime64 and timedelta64 dtype
        # strings.
        return request.param

    @pytest.fixture
    def scalar(self, dtype):
        # [FUNCTION DOC]
        # HELPER: fixture returning a nanosecond datetime64 scalar, or, for the
        # timedelta dtype, the zero timedelta computed by subtracting it from itself.
        val = np.datetime64("2021-01-18 13:25:00", "ns")
        if dtype == "m8[ns]":
            val = val - val
        return val

    @pytest.fixture
    def expected(self, scalar):
        # [FUNCTION DOC]
        # HELPER: fixture returning an object series of two copies of the scalar
        # followed by the original value three, and checks the first element has the
        # scalar's type.
        expected = Series([scalar, scalar, 3], dtype=object)
        assert isinstance(expected[0], type(scalar))
        return expected

    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning an integer series of one, two, three.
        return Series([1, 2, 3])

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a slice excluding the last element.
        return slice(None, -1)

    @pytest.fixture(params=[None, list, np.array])
    def val(self, scalar, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over none, list or array: returns the scalar
        # itself or a two-element container of it.
        box = request.param
        if box is None:
            return scalar
        return box([scalar, scalar])

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that an error is expected (true).
        return True


class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Verifies setting compatible missing values (none or NaN) at a position or slice of
    # a period series gives NaT while keeping the period dtype; no error is expected.
    # Provides fixtures for the shared checks.
    # Setting compatible NA values into Series with PeriodDtype

    @pytest.fixture
    def expected(self, key):
        # [FUNCTION DOC]
        # HELPER: fixture building the expected series by writing the NaT integer into
        # the integer view of a copy of the period series at the key, and asserting NaT
        # is there.
        exp = Series(period_range("2000-01-01", periods=10, freq="D"))
        exp._values.view("i8")[key] = NaT._value
        assert exp[key] is NaT or all(x is NaT for x in exp[key])
        return exp

    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a daily period series of ten entries.
        return Series(period_range("2000-01-01", periods=10, freq="D"))

    @pytest.fixture(params=[3, slice(3, 5)])
    def key(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over position three or the slice three to five.
        return request.param

    @pytest.fixture(params=[None, np.nan])
    def val(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over none or NaN.
        return request.param

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that no error is expected.
        return False


class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Verifies (GH 18586) that setting missing-like values into datetime64 or
    # timedelta64 series either stays in the dtype (for none, NaN, NaT and the matching
    # typed NaT) or an error is expected when it would otherwise cast to object (the
    # mismatched typed NaT, or anything but NaT for tz-aware series).
    # some nat-like values should be cast to datetime64/timedelta64 when
    #  inserting into a datetime64/timedelta64 series.  Others should coerce
    #  to object and retain their dtypes.
    # GH#18586 for td64 and boolean mask case

    @pytest.fixture(
        params=["m8[ns]", "M8[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Central]"]
    )
    def dtype(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over four datetime-like dtypes (timedelta, naive,
        # UTC and Central-aware).
        return request.param

    @pytest.fixture
    def obj(self, dtype):
        # [FUNCTION DOC]
        # HELPER: fixture building a three-element series of the dtype from the integer
        # view of a daily range.
        i8vals = date_range("2016-01-01", periods=3).asi8
        idx = Index(i8vals, dtype=dtype)
        assert idx.dtype == dtype
        return Series(idx)

    @pytest.fixture(
        params=[
            None,
            np.nan,
            NaT,
            np.timedelta64("NaT", "ns"),
            np.datetime64("NaT", "ns"),
        ]
    )
    def val(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over none, NaN, NaT and the timedelta and
        # datetime typed NaT.
        return request.param

    @pytest.fixture
    def is_inplace(self, val, obj):
        # [FUNCTION DOC]
        # HELPER: fixture deciding whether setting is in place: true when the value is
        # NaT, none or NaN, or the value's dtype equals the series dtype.
        # td64   -> cast to object iff val is datetime64("NaT")
        # dt64   -> cast to object iff val is timedelta64("NaT")
        # dt64tz -> cast to object with anything _but_ NaT
        return val is NaT or val is None or val is np.nan or obj.dtype == val.dtype

    @pytest.fixture
    def expected(self, obj, val, is_inplace):
        # [FUNCTION DOC]
        # HELPER: fixture building the expected series with the value first, followed by
        # the rest of the original; its dtype is the original when in place, otherwise
        # object.
        dtype = obj.dtype if is_inplace else object
        expected = Series([val, *list(obj[1:])], dtype=dtype)
        return expected

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning key zero.
        return 0

    @pytest.fixture
    def raises(self, is_inplace):
        # [FUNCTION DOC]
        # HELPER: fixture returning that an error is expected exactly when the set is
        # not in place.
        return False if is_inplace else True


class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Regression tests (GH 24024). Setting a Timestamp with a different timezone into a
    # tz-aware series converts the value to the series' timezone and keeps the dtype
    # (before 2.0 it cast to object); no error is expected. Provides fixtures for the
    # shared checks.
    # GH#24024
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a two-element Central-timezone series.
        return Series(date_range("2000", periods=2, tz="US/Central"))

    @pytest.fixture
    def val(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a Timestamp in the Eastern timezone.
        return Timestamp("2000", tz="US/Eastern")

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning key zero.
        return 0

    @pytest.fixture
    def expected(self, obj, val):
        # [FUNCTION DOC]
        # HELPER: fixture building the expected series with the first value converted to
        # Central and the second unchanged, keeping the dtype.
        # pre-2.0 this would cast to object, in 2.0 we cast the val to
        #  the target tz
        expected = Series(
            [
                val.tz_convert("US/Central"),
                Timestamp("2000-01-02 00:00:00-06:00", tz="US/Central"),
            ],
            dtype=obj.dtype,
        )
        return expected

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that no error is expected.
        return False


@pytest.mark.parametrize(
    "obj,expected",
    [
        # For numeric series, we should coerce to NaN.
        (Series([1, 2, 3]), Series([np.nan, 2, 3])),
        (Series([1.0, 2.0, 3.0]), Series([np.nan, 2.0, 3.0])),
        # For datetime series, we should coerce to NaT.
        (
            Series([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]),
            Series([NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)]),
        ),
        # For objects, we should preserve the None value.
        (Series(["foo", "bar", "baz"]), Series([None, "bar", "baz"])),
    ],
)
class TestSeriesNoneCoercion(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Verifies assigning none at position zero coerces to the right missing marker: NaN
    # for integer and float series, NaT for datetime and none retained for object
    # series; parametrized by a class-level decorator.
    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning key zero.
        return 0

    @pytest.fixture
    def val(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning none.
        return None

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that no error is expected.
        return False


class TestSetitemFloatIntervalWithIntIntervalValues(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Regression tests (GH 44201) for setting a float interval into an integer interval
    # series: the assignment raises an invalid value type error. Provides fixtures for
    # the shared checks (the expected result is a float interval series).
    # GH#44201 Cast to shared IntervalDtype rather than object

    def test_setitem_example(self):
        # [FUNCTION DOC]
        # Documents the case the class targets: setting a float-boundary interval into
        # an integer-interval series raises a type error (invalid value).
        # Just a case here to make obvious what this test class is aimed at
        idx = IntervalIndex.from_breaks(range(4))
        obj = Series(idx)
        val = Interval(0.5, 1.5)

        with pytest.raises(TypeError, match="Invalid value"):
            obj[0] = val

    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning an integer-boundary interval series with breaks zero
        # to three.
        """
        Fixture to create a Series [(0, 1], (1, 2], (2, 3]]
        """
        idx = IntervalIndex.from_breaks(range(4))
        return Series(idx)

    @pytest.fixture
    def val(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the interval from 0.5 to 1.5.
        """
        Fixture to get an interval (0.5, 1.5]
        """
        return Interval(0.5, 1.5)

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning key zero.
        """
        Fixture to get a key 0
        """
        return 0

    @pytest.fixture
    def expected(self, obj, val):
        # [FUNCTION DOC]
        # HELPER: fixture building the expected float-interval series with the value
        # first followed by the remaining original intervals.
        """
        Fixture to get a Series [(0.5, 1.5], (1.0, 2.0], (2.0, 3.0]]
        """
        data = [val, *list(obj[1:])]
        idx = IntervalIndex(data, dtype="Interval[float64]")
        return Series(idx)

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that an error is expected (true).
        """
        Fixture to enable raising pytest exceptions
        """
        return True


class TestSetitemRangeIntoIntegerSeries(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Regression tests (GH 44261): setting a small range of integers into any integer
    # numpy dtype series over a slice does not need to upcast. Provides fixtures for the
    # shared checks.
    # GH#44261 Setting a range with sufficiently-small integers into
    #  small-itemsize integer dtypes should not need to upcast

    @pytest.fixture
    def obj(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # HELPER: fixture returning a range series of five elements in the dtype
        # fixture.
        dtype = np.dtype(any_int_numpy_dtype)
        ser = Series(range(5), dtype=dtype)
        return ser

    @pytest.fixture
    def val(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a range from 2 to 4.
        return range(2, 4)

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the slice from zero to two.
        return slice(0, 2)

    @pytest.fixture
    def expected(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # HELPER: fixture returning the expected series with the range written over the
        # first two elements, in the same integer dtype.
        dtype = np.dtype(any_int_numpy_dtype)
        exp = Series([2, 3, 2, 3, 4], dtype=dtype)
        return exp

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that no error is expected.
        return False


@pytest.mark.parametrize(
    "val, raises",
    [
        (np.array([2.0, 3.0]), False),
        (np.array([2.5, 3.5]), True),
        (
            np.array([2**65, 2**65 + 1], dtype=np.float64),
            True,
        ),  # all ints, but can't cast
    ],
)
class TestSetitemFloatNDarrayIntoIntegerSeries(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Parametrized (by a class-level decorator) over float arrays: whole-valued floats
    # can be set into an integer series without changing dtype; non-integral floats and
    # whole-valued floats too large for the integer type raise an invalid-value error.
    # Provides fixtures for the shared checks; the error expectation comes from the
    # parametrization.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a five-element int64 range series.
        return Series(range(5), dtype=np.int64)

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the slice from zero to two.
        return slice(0, 2)

    @pytest.fixture
    def expected(self, val):
        # [FUNCTION DOC]
        # HELPER: fixture building the expected series from a range array with the first
        # two elements replaced by the value, using integer dtype when the first value
        # is 2 and float64 otherwise (dependent on the hard-coded parameter cases).
        if val[0] == 2:
            # NB: this condition is based on currently-hardcoded "val" cases
            dtype = np.int64
        else:
            dtype = np.float64
        res_values = np.array(range(5), dtype=dtype)
        res_values[:2] = val
        return Series(res_values)


@pytest.mark.parametrize("val", [512, np.int16(512)])
class TestSetitemIntoIntegerSeriesNeedsUpcast(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Parametrized over 512 as a python int and numpy int16: setting this into an int8
    # series would need to widen, so an invalid value error is expected. Provides
    # fixtures for the shared checks.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning an int8 series of one, two, three.
        return Series([1, 2, 3], dtype=np.int8)

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning key one.
        return 1

    @pytest.fixture
    def expected(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the expected int16 series with 512 in the middle.
        return Series([1, 512, 3], dtype=np.int16)

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that an error is expected (true).
        return True


@pytest.mark.parametrize("val", [2**33 + 1.0, 2**33 + 1.1, 2**62])
class TestSmallIntegerSetitemUpcast(SetitemCastingEquivalents):
    # [CLASS DOC]
    # Parametrized over large values (2 to the 33rd plus one as float, plus one point
    # one, and 2 to the 62nd): setting these into an int32 series raises an invalid
    # value error. Provides fixtures for the shared checks (expected dtype is float when
    # the value has a fractional part, else int64).
    # https://github.com/pandas-dev/pandas/issues/39584#issuecomment-941212124
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a 32-bit integer series of one, two, three.
        return Series([1, 2, 3], dtype="i4")

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning key zero.
        return 0

    @pytest.fixture
    def expected(self, val):
        # [FUNCTION DOC]
        # HELPER: fixture building the expected series with the value first, as float64
        # when the value has a fractional part and int64 otherwise.
        if val % 1 != 0:
            dtype = "f8"
        else:
            dtype = "i8"
        return Series([val, 2, 3], dtype=dtype)

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that an error is expected (true).
        return True


class CoercionTest(SetitemCastingEquivalents):
    # [CLASS DOC]
    # HELPER: base class for the coercion tests ported from the older coercion test
    # module. Provides the key fixture (position one) and an expected fixture that
    # replaces that element of a list of the original values and builds a series with
    # the parametrized expected dtype. Subclasses supply obj, val, exp_dtype and raises.
    # Tests ported from tests.indexing.test_coercion

    @pytest.fixture
    def key(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning key one.
        return 1

    @pytest.fixture
    def expected(self, obj, key, val, exp_dtype):
        # [FUNCTION DOC]
        # HELPER: fixture building the expected series: list of the original values with
        # the key element replaced by the value, constructed with the expected dtype.
        vals = list(obj)
        vals[key] = val
        return Series(vals, dtype=exp_dtype)


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [(np.int32(1), np.int8, None), (np.int16(2**9), np.int16, True)],
)
class TestCoercionInt8(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for an int8 series: an int32 one fits (error flag is none) and an
    # int16 value of 512 is too big for int8 and expects an invalid-value error.
    # previously test_setitem_series_int8 in tests.indexing.test_coercion
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning an int8 series of one to four.
        return Series([1, 2, 3, 4], dtype=np.int8)


@pytest.mark.parametrize("val", [1, 1.1, 1 + 1j, True])
@pytest.mark.parametrize("exp_dtype", [object])
class TestCoercionObject(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for an object series: any of an integer, float, complex or boolean
    # value is stored as an object without errors.
    # previously test_setitem_series_object in tests.indexing.test_coercion
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning an object series of four strings.
        return Series(["a", "b", "c", "d"], dtype=object)

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that no error is expected.
        return False


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (1, object, True),
        ("e", StringDtype(na_value=np.nan), False),
    ],
)
class TestCoercionString(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for a string-dtype series: setting an integer expects an error;
    # setting another string succeeds and keeps the string dtype.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a four-string series with the string dtype that uses
        # NaN for missing.
        return Series(["a", "b", "c", "d"], dtype=StringDtype(na_value=np.nan))


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (1, np.complex128, False),
        (1.1, np.complex128, False),
        (1 + 1j, np.complex128, False),
        (True, object, True),
    ],
)
class TestCoercionComplex(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for a complex series: integer, float and complex values are held as
    # complex without error; a boolean expects an error (object).
    # previously test_setitem_series_complex128 in tests.indexing.test_coercion
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a complex series of four values.
        return Series([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (1, object, True),
        ("3", object, True),
        (3, object, True),
        (1.1, object, True),
        (1 + 1j, object, True),
        (True, bool, False),
    ],
)
class TestCoercionBool(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for a boolean series: only a boolean is accepted; integers,
    # strings, floats and complex raise an invalid value error.
    # previously test_setitem_series_bool in tests.indexing.test_coercion
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a boolean series of four values.
        return Series([True, False, True, False], dtype=bool)


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (1, np.int64, False),
        (1.1, np.float64, True),
        (1 + 1j, np.complex128, True),
        (True, object, True),
    ],
)
class TestCoercionInt64(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for an int64 series: an integer is held; a float, complex or
    # boolean raises an invalid-value error.
    # previously test_setitem_series_int64 in tests.indexing.test_coercion
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning an integer series of one to four.
        return Series([1, 2, 3, 4])


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (1, np.float64, False),
        (1.1, np.float64, False),
        (1 + 1j, np.complex128, True),
        (True, object, True),
    ],
)
class TestCoercionFloat64(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for a float64 series: integer and float values are held; complex or
    # boolean raise an invalid-value error.
    # previously test_setitem_series_float64 in tests.indexing.test_coercion
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a float64 series of four values.
        return Series([1.1, 2.2, 3.3, 4.4])


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (1, np.float32, False),
        pytest.param(
            1.1,
            np.float32,
            False,
            marks=pytest.mark.xfail(
                not np_version_gt2,
                reason="np.float32(1.1) ends up as 1.100000023841858, so "
                "np_can_hold_element raises and we cast to float64",
            ),
        ),
        (1 + 1j, np.complex128, True),
        (True, object, True),
        (np.uint8(2), np.float32, False),
        (np.uint32(2), np.float32, False),
        # float32 cannot hold np.iinfo(np.uint32).max exactly
        # (closest it can hold is 4294967300.0 which off by 5.0), so
        # we cast to float64
        (np.uint32(np.iinfo(np.uint32).max), np.float64, True),
        (np.uint64(2), np.float32, False),
        (np.int64(2), np.float32, False),
    ],
)
class TestCoercionFloat32(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for a float32 series: integers and several numpy integer scalars
    # representable exactly in float32 are held; one float case (one point one) is
    # expected to fail on numpy older than version 2 because the float32 value is not
    # exact; a uint32 maximum cannot be held exactly in float32, complex and boolean
    # also raise.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a float32 series of four values.
        return Series([1.1, 2.2, 3.3, 4.4], dtype=np.float32)

    def test_slice_key(self, obj, key, expected, raises, val, indexer_sli, is_inplace):
        # [FUNCTION DOC]
        # Runs the shared slice-key check then raises an assertion error when the value
        # is a python float, to prevent the expected-failure mark from silently passing
        # for this test (the slice check short-circuits).
        super().test_slice_key(obj, key, expected, raises, val, indexer_sli, is_inplace)

        if isinstance(val, float):
            # the xfail would xpass bc test_slice_key short-circuits
            raise AssertionError("xfail not relevant for this test.")


@pytest.mark.parametrize(
    "exp_dtype",
    [
        "M8[ms]",
        "M8[ms, UTC]",
        "m8[ms]",
    ],
)
class TestCoercionDatetime64HigherReso(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for datetime-like series with millisecond resolution (naive,
    # UTC-aware and timedelta) when the value has a different resolution: an error
    # (invalid value) is expected.
    @pytest.fixture
    def obj(self, exp_dtype):
        # [FUNCTION DOC]
        # HELPER: fixture building a four-element daily range at second resolution; for
        # the timedelta case subtracting the epoch, for the UTC case localizing to UTC.
        idx = date_range("2011-01-01", freq="D", periods=4, unit="s")
        if exp_dtype == "m8[ms]":
            idx = idx - Timestamp("1970-01-01").as_unit("s")
            assert idx.dtype == "m8[s]"
        elif exp_dtype == "M8[ms, UTC]":
            idx = idx.tz_localize("UTC")
        return Series(idx)

    @pytest.fixture
    def val(self, exp_dtype):
        # [FUNCTION DOC]
        # HELPER: fixture returning a millisecond-resolution Timestamp (or a timedelta
        # from the epoch, or a UTC localized one depending on the dtype parameter).
        ts = Timestamp("2011-01-02 03:04:05.678").as_unit("ms")
        if exp_dtype == "m8[ms]":
            return ts - Timestamp("1970-01-01").as_unit("s")
        elif exp_dtype == "M8[ms, UTC]":
            return ts.tz_localize("UTC")
        return ts

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that an error is expected (true).
        return True


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (Timestamp("2012-01-01"), "datetime64[ns]", False),
        (1, object, True),
        ("x", object, True),
    ],
)
class TestCoercionDatetime64(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for a nanosecond datetime64 series: a Timestamp is accepted (dtype
    # datetime64); an integer or string raises an invalid-value error.
    # previously test_setitem_series_datetime64 in tests.indexing.test_coercion

    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a four-element daily nanosecond datetime series.
        return Series(date_range("2011-01-01", freq="D", periods=4, unit="ns"))

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that no error is expected.
        # KNOWN ISSUE: this fixture is dead code because the class-level parametrization
        # supplies an argument with the same name, which takes precedence over a fixture.
        return False


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]", False),
        # pre-2.0, a mis-matched tz would end up casting to object
        (Timestamp("2012-01-01", tz="US/Pacific"), "datetime64[ns, US/Eastern]", False),
        (Timestamp("2012-01-01"), object, True),
        (1, object, True),
    ],
)
class TestCoercionDatetime64TZ(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for a tz-aware series: a Timestamp with the same or a different
    # timezone is held (converted to the series timezone); a naive Timestamp or an
    # integer raises an invalid-value error (object expected).
    # previously test_setitem_series_datetime64tz in tests.indexing.test_coercion
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a four-element daily US/Eastern datetime series.
        tz = "US/Eastern"
        return Series(date_range("2011-01-01", freq="D", periods=4, tz=tz, unit="ns"))

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that no error is expected.
        # KNOWN ISSUE: this fixture is dead code because the class-level parametrization
        # supplies an argument with the same name, which takes precedence over a fixture.
        return False


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (Timedelta("12 day"), "timedelta64[us]", False),
        (1, object, True),
        ("x", object, True),
    ],
)
class TestCoercionTimedelta64(CoercionTest):
    # [CLASS DOC]
    # Coercion tests for a timedelta series: a Timedelta is held (microsecond
    # resolution); an integer or string expects an error.
    # previously test_setitem_series_timedelta64 in tests.indexing.test_coercion
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a four-element timedelta range series starting at
        # one day.
        return Series(timedelta_range("1 day", periods=4))

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that no error is expected.
        # KNOWN ISSUE: this fixture is dead code because the class-level parametrization
        # supplies an argument with the same name, which takes precedence over a fixture.
        return False


@pytest.mark.parametrize(
    "val", ["foo", Period("2016", freq="Y"), Interval(1, 2, closed="both")]
)
@pytest.mark.parametrize("exp_dtype", [object])
class TestPeriodIntervalCoercion(CoercionTest):
    # [CLASS DOC]
    # Regression tests (GH 45768): a string, a Period or an Interval set into a period
    # series or an interval series is stored by casting to object (an error is expected
    # by the base checks).
    # GH#45768
    @pytest.fixture(
        params=[
            period_range("2016-01-01", periods=3, freq="D"),
            interval_range(1, 5),
        ]
    )
    def obj(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over a daily period series and an integer
        # interval series.
        return Series(request.param)

    @pytest.fixture
    def raises(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning that an error is expected (true).
        return True


def test_20643():
    # [FUNCTION DOC]
    # Regression test for GH 20643 (closed by GH 45121). Setting a non-integral float
    # into an integer series or frame element, via at, loc, direct set, iat, iloc,
    # raises a type error (invalid value).
    # closed by GH#45121
    orig = Series([0, 1, 2], index=["a", "b", "c"])

    ser = orig.copy()
    with pytest.raises(TypeError, match="Invalid value"):
        ser.at["b"] = 2.7

    with pytest.raises(TypeError, match="Invalid value"):
        ser.loc["b"] = 2.7

    with pytest.raises(TypeError, match="Invalid value"):
        ser["b"] = 2.7

    ser = orig.copy()
    with pytest.raises(TypeError, match="Invalid value"):
        ser.iat[1] = 2.7

    with pytest.raises(TypeError, match="Invalid value"):
        ser.iloc[1] = 2.7

    orig_df = orig.to_frame("A")

    df = orig_df.copy()
    with pytest.raises(TypeError, match="Invalid value"):
        df.at["b", "A"] = 2.7

    with pytest.raises(TypeError, match="Invalid value"):
        df.loc["b", "A"] = 2.7

    with pytest.raises(TypeError, match="Invalid value"):
        df.iloc[1, 0] = 2.7

    with pytest.raises(TypeError, match="Invalid value"):
        df.iat[1, 0] = 2.7


def test_20643_comment():
    # [FUNCTION DOC]
    # Regression test (comment on GH 20643). Setting none through positional scalar
    # setters gives NaN in an integer series (cast to float).
    # https://github.com/pandas-dev/pandas/issues/20643#issuecomment-431244590
    # fixed sometime prior to GH#45121
    orig = Series([0, 1, 2], index=["a", "b", "c"])
    expected = Series([np.nan, 1, 2], index=["a", "b", "c"])

    ser = orig.copy()
    ser.iat[0] = None
    tm.assert_series_equal(ser, expected)

    ser = orig.copy()
    ser.iloc[0] = None
    tm.assert_series_equal(ser, expected)


def test_15413():
    # [FUNCTION DOC]
    # Regression test for GH 15413 (fixed by GH 45121). Augmented assignment adding a
    # non-integral float to integer series elements through a mask, direct, loc, iloc,
    # iat and at raises a type error (invalid value).
    # fixed by GH#45121
    ser = Series([1, 2, 3])

    with pytest.raises(TypeError, match="Invalid value"):
        ser[ser == 2] += 0.5

    with pytest.raises(TypeError, match="Invalid value"):
        ser[1] += 0.5

    with pytest.raises(TypeError, match="Invalid value"):
        ser.loc[1] += 0.5

    with pytest.raises(TypeError, match="Invalid value"):
        ser.iloc[1] += 0.5

    with pytest.raises(TypeError, match="Invalid value"):
        ser.iat[1] += 0.5

    with pytest.raises(TypeError, match="Invalid value"):
        ser.at[1] += 0.5


def test_32878_int_itemsize():
    # [FUNCTION DOC]
    # Regression test for GH 32878 (fixed by GH 45121). Setting the maximum int64 into a
    # 32-bit integer series raises a type error (invalid value).
    # Fixed by GH#45121
    arr = np.arange(5).astype("i4")
    ser = Series(arr)
    val = np.int64(np.iinfo(np.int64).max)
    with pytest.raises(TypeError, match="Invalid value"):
        ser[0] = val


def test_32878_complex_itemsize():
    # [FUNCTION DOC]
    # Regression test for GH 32878. Setting the maximum finite float, as a double
    # complex, into a complex64 series raises a type error (invalid value; before it was
    # coerced to infinity).
    arr = np.arange(5).astype("c8")
    ser = Series(arr)
    val = np.finfo(np.float64).max
    val = val.astype("c16")

    # GH#32878 used to coerce val to inf+0.000000e+00j
    with pytest.raises(TypeError, match="Invalid value"):
        ser[0] = val


def test_37692(indexer_al):
    # [FUNCTION DOC]
    # Regression test for GH 37692. With the at and iat indexer fixture, setting a
    # string into an integer series raises a type error (invalid value).
    # GH#37692
    ser = Series([1, 2, 3], index=["a", "b", "c"])
    with pytest.raises(TypeError, match="Invalid value"):
        indexer_al(ser)["b"] = "test"


def test_setitem_bool_int_float_consistency(indexer_sli):
    # [FUNCTION DOC]
    # Regression test (GH 21513). With the loc and iloc fixture: setting True into float
    # or integer series raises an invalid value error, setting a number into a boolean
    # series raises one; setting a whole-valued float into an integer series, or an
    # integer into a float series, is lossless and keeps the dtype (the integer series
    # case is asserted; the float series case has no assertion).
    # KNOWN ISSUE: the leading comment says bool-with-int and bool-with-float upcast to
    # object, but the test expects an error to be raised in those cases.
    # GH#21513
    # bool-with-int and bool-with-float both upcast to object
    #  int-with-float and float-with-int are both non-casting so long
    #  as the setitem can be done losslessly
    for dtype in [np.float64, np.int64]:
        ser = Series(0, index=range(3), dtype=dtype)
        with pytest.raises(TypeError, match="Invalid value"):
            indexer_sli(ser)[0] = True

        ser = Series(0, index=range(3), dtype=bool)
        with pytest.raises(TypeError, match="Invalid value"):
            ser[0] = dtype(1)

    # 1.0 can be held losslessly, so no casting
    ser = Series(0, index=range(3), dtype=np.int64)
    indexer_sli(ser)[0] = np.float64(1.0)
    assert ser.dtype == np.int64

    # 1 can be held losslessly, so no casting
    ser = Series(0, index=range(3), dtype=np.float64)
    indexer_sli(ser)[0] = np.int64(1)


def test_setitem_positional_with_casting():
    # [FUNCTION DOC]
    # Regression test (GH 45070). As of version 3.0 an integer key on a string-indexed
    # series is a label, so assigning a string enlarges the series with the integer
    # label and object dtype.
    # GH#45070 case where in __setitem__ we get a KeyError, then when
    #  we fallback we *also* get a ValueError if we try to set inplace.
    # As of 3.0 we always treat int keys as labels, so this becomes
    #  setitem-with-expansion
    ser = Series([1, 2, 3], index=["a", "b", "c"])

    ser[0] = "X"
    expected = Series([1, 2, 3, "X"], index=["a", "b", "c", 0], dtype=object)
    tm.assert_series_equal(ser, expected)


def test_setitem_positional_float_into_int_coerces():
    # [FUNCTION DOC]
    # Verifies assigning a float to an integer label not present in a string-indexed
    # integer series enlarges the series (as of 3.0), keeping the float value rather
    # than truncating.
    # Case where we hit a KeyError and then trying to set in-place incorrectly
    #  casts a float to an int;
    # As of 3.0 we always treat int keys as labels, so this becomes
    #  setitem-with-expansion
    ser = Series([1, 2, 3], index=["a", "b", "c"])

    ser[0] = 1.5
    expected = Series([1, 2, 3, 1.5], index=["a", "b", "c", 0])
    tm.assert_series_equal(ser, expected)


def test_setitem_int_not_positional():
    # [FUNCTION DOC]
    # Regression test (GH 42215, enforced in 2.0). On a float index an existing
    # whole-number key sets that element; a missing integer key enlarges instead of
    # falling back to position, for a float index, an interval index (index becomes
    # object) and a multi-index (equals the loc result).
    # GH#42215 deprecated falling back to positional on __setitem__ with an
    #  int not contained in the index; enforced in 2.0
    ser = Series([1, 2, 3, 4], index=[1.1, 2.1, 3.0, 4.1])
    assert not ser.index._should_fallback_to_positional
    # assert not ser.index.astype(object)._should_fallback_to_positional

    # 3.0 is in our index, so post-enforcement behavior is unchanged
    ser[3] = 10
    expected = Series([1, 2, 10, 4], index=ser.index)
    tm.assert_series_equal(ser, expected)

    # pre-enforcement `ser[5] = 5` raised IndexError
    ser[5] = 5
    expected = Series([1, 2, 10, 4, 5], index=[1.1, 2.1, 3.0, 4.1, 5.0])
    tm.assert_series_equal(ser, expected)

    ii = IntervalIndex.from_breaks(range(10))[::2]
    ser2 = Series(range(len(ii)), index=ii)
    exp_index = ii.astype(object).append(Index([4]))
    expected2 = Series([0, 1, 2, 3, 4, 9], index=exp_index)
    # pre-enforcement `ser2[4] = 9` interpreted 4 as positional
    ser2[4] = 9
    tm.assert_series_equal(ser2, expected2)

    mi = MultiIndex.from_product([ser.index, ["A", "B"]])
    ser3 = Series(range(len(mi)), index=mi)
    expected3 = ser3.copy()
    expected3.loc[4] = 99
    # pre-enforcement `ser3[4] = 99` interpreted 4 as positional
    ser3[4] = 99
    tm.assert_series_equal(ser3, expected3)


def test_setitem_with_bool_indexer():
    # [FUNCTION DOC]
    # Regression test (GH 42530). Setting through a boolean list on a popped column
    # series and with loc on the frame work and give the expected values.
    # GH#42530

    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.pop("b").copy()
    result[[True, False, False]] = 9
    expected = Series(data=[9, 5, 6], name="b")
    tm.assert_series_equal(result, expected)

    df.loc[[True, False, False], "a"] = 10
    expected = DataFrame({"a": [10, 2, 3]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("size", range(2, 6))
@pytest.mark.parametrize(
    "mask", [[True, False, False, False, False], [True, False], [False]]
)
@pytest.mark.parametrize(
    "item", [2.0, np.nan, np.finfo(float).max, np.finfo(float).min]
)
@pytest.mark.parametrize("box", [np.array, list, tuple])
def test_setitem_bool_indexer_dont_broadcast_length1_values(size, mask, item, box):
    # [FUNCTION DOC]
    # Regression test (GH 44265). Parametrized over series size, mask pattern, float
    # item and the container type. A one-element container assigned through a boolean
    # mask is only allowed when exactly one position is selected (then equals setting
    # the plain item); otherwise a value error about the list-like length is raised.
    # GH#44265
    # see also tests.series.indexing.test_where.test_broadcast

    selection = np.resize(mask, size)

    data = np.arange(size, dtype=float)

    ser = Series(data)

    if selection.sum() != 1:
        msg = (
            "cannot set using a list-like indexer with a different "
            "length than the value"
        )
        with pytest.raises(ValueError, match=msg):
            # GH#44265
            ser[selection] = box([item])
    else:
        # In this corner case setting is equivalent to setting with the unboxed
        #  item
        ser[selection] = box([item])

        expected = Series(np.arange(size, dtype=float))
        expected[selection] = item
        tm.assert_series_equal(ser, expected)


def test_setitem_empty_mask_dont_upcast_dt64():
    # [FUNCTION DOC]
    # Verifies assigning a string through an all-false mask (set and in-place mask)
    # leaves a datetime series unchanged and does not upcast.
    dti = date_range("2016-01-01", periods=3)
    ser = Series(dti)
    orig = ser.copy()
    mask = np.zeros(3, dtype=bool)

    ser[mask] = "foo"
    assert ser.dtype == dti.dtype  # no-op -> dont upcast
    tm.assert_series_equal(ser, orig)

    ser.mask(mask, "foo", inplace=True)
    assert ser.dtype == dti.dtype  # no-op -> dont upcast
    tm.assert_series_equal(ser, orig)


def test_setitem_bool_dtype_with_boolean_indexer():
    # [FUNCTION DOC]
    # Regression test (GH 57338). Setting through a boolean list from another boolean
    # series with the same mask keeps the boolean dtype and writes the values.
    # GH 57338
    s1 = Series([True, True, True], dtype=bool)
    s2 = Series([False, False, False], dtype=bool)
    condition = [False, True, False]
    s1[condition] = s2[condition]
    expected = Series([True, False, True], dtype=bool)
    tm.assert_series_equal(s1, expected)
