# [MODULE DOC]
# Tests for Series.reindex: identity and sub-selection, NaN and datetime
# labels, fill methods and tolerance, dtype changes when labels are introduced,
# categorical, tz-aware, period, MultiIndex level and fill_value handling,
# argument validation. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    NA,
    Categorical,
    Float64Dtype,
    Index,
    MultiIndex,
    NaT,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm


def test_reindex(datetime_series, string_series):
    # [FUNCTION DOC]
    # Reindex to the same index shares memory with and is identical to the original
    # index. Subselecting labels returns matching values; labels absent from the
    # Series give NaN; noncontiguous indexes work; reindex with no arguments returns
    # a new object.
    identity = string_series.reindex(string_series.index)

    assert tm.shares_memory(string_series.index, identity.index)

    assert identity.index.is_(string_series.index)
    assert identity.index.identical(string_series.index)

    subIndex = string_series.index[10:20]
    subSeries = string_series.reindex(subIndex)

    for idx, val in subSeries.items():
        assert val == string_series[idx]

    subIndex2 = datetime_series.index[10:20]
    subTS = datetime_series.reindex(subIndex2)

    for idx, val in subTS.items():
        assert val == datetime_series[idx]
    stuffSeries = datetime_series.reindex(subIndex)

    assert np.isnan(stuffSeries).all()

    # This is extremely important for the Cython code to not screw up
    nonContigIndex = datetime_series.index[::2]
    subNonContig = datetime_series.reindex(nonContigIndex)
    for idx, val in subNonContig.items():
        assert val == datetime_series[idx]

    # return a copy the same index here
    result = datetime_series.reindex()
    assert result is not datetime_series


def test_reindex_nan():
    # [FUNCTION DOC]
    # Series with a NaN label: reindex by a list including NaN picks the matching
    # entries, as iloc does; after converting the index to object dtype the result
    # index dtype becomes float (index type check relaxed).
    ts = Series([2, 3, 5, 7], index=[1, 4, np.nan, 8])

    i, j = [np.nan, 1, np.nan, 8, 4, np.nan], [2, 0, 2, 3, 1, 2]
    tm.assert_series_equal(ts.reindex(i), ts.iloc[j])

    ts.index = ts.index.astype("object")

    # reindex coerces index.dtype to float, loc/iloc doesn't
    tm.assert_series_equal(ts.reindex(i), ts.iloc[j], check_index_type=False)


def test_reindex_series_add_nat():
    # [FUNCTION DOC]
    # Reindexing a datetime Series to a larger integer range keeps the datetime64
    # dtype with NaT in the five new rows.
    rng = date_range("1/1/2000 00:00:00", periods=10, freq="10s")
    series = Series(rng)

    result = series.reindex(range(15))
    assert np.issubdtype(result.dtype, np.dtype("M8[ns]"))

    mask = result.isna()
    assert mask[-5:].all()
    assert not mask[:-5].any()


def test_reindex_with_datetimes():
    # [FUNCTION DOC]
    # Reindexing by a list of dates gives the sliced Series (index frequency cleared);
    # selecting with the list gives the same.
    rng = date_range("1/1/2000", periods=20)
    ts = Series(np.random.default_rng(2).standard_normal(20), index=rng)

    result = ts.reindex(list(ts.index[5:10]))
    expected = ts[5:10]
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)

    result = ts[list(ts.index[5:10])]
    tm.assert_series_equal(result, expected)


def test_reindex_corner(datetime_series):
    # [FUNCTION DOC]
    # Reindexing an empty Series with the pad method works; reindexing by a plain
    # list equals the Series (after the index frequency is cleared); an unknown
    # fill method raises a value error listing the allowed methods.
    # KNOWN ISSUE: the first line carries a stale comment ("don't forget to fix
    # this") and the empty Series is reindexed twice, with the second result
    # unused.
    # (don't forget to fix this) I think it's fixed
    empty = Series(index=[])
    empty.reindex(datetime_series.index, method="pad")  # it works

    # corner case: pad empty series
    reindexed = empty.reindex(datetime_series.index, method="pad")

    # pass non-Index
    reindexed = datetime_series.reindex(list(datetime_series.index))
    datetime_series.index = datetime_series.index._with_freq(None)
    tm.assert_series_equal(datetime_series, reindexed)

    # bad fill method
    ts = datetime_series[::2]
    msg = (
        r"Invalid fill method\. Expecting pad \(ffill\), backfill "
        r"\(bfill\) or nearest\. Got foo"
    )
    with pytest.raises(ValueError, match=msg):
        ts.reindex(datetime_series.index, method="foo")


def test_reindex_pad():
    # [FUNCTION DOC]
    # Forward fill via pad and ffill gives the same result: the values repeated
    # across gaps of a stride-two subset of an integer Series.
    s = Series(np.arange(10), dtype="int64")
    s2 = s[::2]

    reindexed = s2.reindex(s.index, method="pad")
    reindexed2 = s2.reindex(s.index, method="ffill")
    tm.assert_series_equal(reindexed, reindexed2)

    expected = Series([0, 0, 2, 2, 4, 4, 6, 6, 8, 8])
    tm.assert_series_equal(reindexed, expected)


def test_reindex_pad2():
    # [FUNCTION DOC]
    # GH4604: reindexing a labelled Series to partly unknown labels then forward
    # filling gives floats; reindexing with the ffill method on the same Series
    # fills from the last preceding label in the sorted original order.
    # KNOWN ISSUE: the first reindex-and-ffill check is repeated identically.
    # GH4604
    s = Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    new_index = ["a", "g", "c", "f"]
    expected = Series([1, 1, 3, 3.0], index=new_index)

    # this changes dtype because the ffill happens after
    result = s.reindex(new_index).ffill()
    tm.assert_series_equal(result, expected)

    result = s.reindex(new_index).ffill()
    tm.assert_series_equal(result, expected)

    expected = Series([1, 5, 3, 5], index=new_index)
    result = s.reindex(new_index, method="ffill")
    tm.assert_series_equal(result, expected)


def test_reindex_inference():
    # [FUNCTION DOC]
    # Reindexing a boolean Series to a label set including a missing label then
    # forward filling infers object dtype.
    # inference of new dtype
    s = Series([True, False, False, True], index=list("abcd"))
    new_index = "agc"
    result = s.reindex(list(new_index)).ffill()
    expected = Series([True, True, False], index=list(new_index), dtype=object)
    tm.assert_series_equal(result, expected)


def test_reindex_downcasting():
    # [FUNCTION DOC]
    # GH4618: shifting an all-False boolean Series and back-filling gives object
    # dtype (no downcast back to bool).
    # GH4618 shifted series downcasting
    s = Series(False, index=range(5))
    result = s.shift(1).bfill()
    expected = Series(False, index=range(5), dtype=object)
    tm.assert_series_equal(result, expected)


def test_reindex_nearest():
    # [FUNCTION DOC]
    # Integer Series reindexed to fractional labels with the nearest method picks
    # the nearest position; a numeric tolerance leaves NaN for labels too far; a list
    # tolerance applies per label.
    s = Series(np.arange(10, dtype="int64"))
    target = [0.1, 0.9, 1.5, 2.0]
    result = s.reindex(target, method="nearest")
    expected = Series(np.around(target).astype("int64"), target)
    tm.assert_series_equal(expected, result)

    result = s.reindex(target, method="nearest", tolerance=0.2)
    expected = Series([0, 1, np.nan, 2], target)
    tm.assert_series_equal(expected, result)

    result = s.reindex(target, method="nearest", tolerance=[0.3, 0.01, 0.4, 3])
    expected = Series([0, np.nan, np.nan, 2], target)
    tm.assert_series_equal(expected, result)


def test_reindex_int(datetime_series):
    # [FUNCTION DOC]
    # Reindexing an integer Series to labels that introduce missing entries gives
    # float64; reindexing to a subset keeps integer dtype.
    ts = datetime_series[::2]
    int_ts = Series(np.zeros(len(ts), dtype=int), index=ts.index)

    # this should work fine
    reindexed_int = int_ts.reindex(datetime_series.index)

    # if NaNs introduced
    assert reindexed_int.dtype == np.float64

    # NO NaNs introduced
    reindexed_int = int_ts.reindex(int_ts.index[::2])
    assert reindexed_int.dtype == np.dtype(int)


def test_reindex_bool(datetime_series):
    # [FUNCTION DOC]
    # Boolean Series reindexed to introduce missing entries becomes object dtype;
    # reindexed to a subset it stays bool.
    # A series other than float, int, string, or object
    ts = datetime_series[::2]
    bool_ts = Series(np.zeros(len(ts), dtype=bool), index=ts.index)

    # this should work fine
    reindexed_bool = bool_ts.reindex(datetime_series.index)

    # if NaNs introduced
    assert reindexed_bool.dtype == np.object_

    # NO NaNs introduced
    reindexed_bool = bool_ts.reindex(bool_ts.index[::2])
    assert reindexed_bool.dtype == np.bool_


def test_reindex_bool_pad(datetime_series):
    # [FUNCTION DOC]
    # Reindexing a boolean Series with the pad method leaves the leading
    # rows (before the first label) missing.
    # KNOWN ISSUE: a stale "fail" comment at the top does not match the assertion
    # behavior.
    # fail
    ts = datetime_series[5:]
    bool_ts = Series(np.zeros(len(ts), dtype=bool), index=ts.index)
    filled_bool = bool_ts.reindex(datetime_series.index, method="pad")
    assert isna(filled_bool[:5]).all()


def test_reindex_categorical():
    # [FUNCTION DOC]
    # A categorical Series reindexed to non-matching labels is all NaN but keeps its
    # categories; partial overlap gives the matching categories; NaN fills other labels.
    index = date_range("20000101", periods=3)

    # reindexing to an invalid Categorical
    s = Series(["a", "b", "c"], dtype="category")
    result = s.reindex(index)
    expected = Series(
        Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"])
    )
    expected.index = index
    tm.assert_series_equal(result, expected)

    # partial reindexing
    expected = Series(
        Categorical(values=["b", "c"], categories=["a", "b", "c"]), index=range(1, 3)
    )
    result = s.reindex([1, 2])
    tm.assert_series_equal(result, expected)

    expected = Series(
        Categorical(values=["c", np.nan], categories=["a", "b", "c"]), index=range(2, 4)
    )
    result = s.reindex([2, 3])
    tm.assert_series_equal(result, expected)


def test_reindex_astype_order_consistency():
    # [FUNCTION DOC]
    # GH#17444: casting to categorical and then string after reindexing equals
    # doing the reindex after the categorical cast.
    # GH#17444
    ser = Series([1, 2, 3], index=[2, 0, 1])
    new_index = [0, 1, 2]
    temp_dtype = "category"
    new_dtype = str
    result = ser.reindex(new_index).astype(temp_dtype).astype(new_dtype)
    expected = ser.astype(temp_dtype).reindex(new_index).astype(new_dtype)
    tm.assert_series_equal(result, expected)


def test_reindex_fill_value():
    # [FUNCTION DOC]
    # Fill values on reindex for float, int, object and bool Series: missing labels
    # give NaN (and dtype promotion for int and bool) by default; an explicit fill
    # value of the right kind keeps the dtype.
    # -----------------------------------------------------------
    # floats
    floats = Series([1.0, 2.0, 3.0])
    result = floats.reindex([1, 2, 3])
    expected = Series([2.0, 3.0, np.nan], index=range(1, 4))
    tm.assert_series_equal(result, expected)

    result = floats.reindex([1, 2, 3], fill_value=0)
    expected = Series([2.0, 3.0, 0], index=range(1, 4))
    tm.assert_series_equal(result, expected)

    # -----------------------------------------------------------
    # ints
    ints = Series([1, 2, 3])

    result = ints.reindex([1, 2, 3])
    expected = Series([2.0, 3.0, np.nan], index=range(1, 4))
    tm.assert_series_equal(result, expected)

    # don't upcast
    result = ints.reindex([1, 2, 3], fill_value=0)
    expected = Series([2, 3, 0], index=range(1, 4))
    assert issubclass(result.dtype.type, np.integer)
    tm.assert_series_equal(result, expected)

    # -----------------------------------------------------------
    # objects
    objects = Series([1, 2, 3], dtype=object)

    result = objects.reindex([1, 2, 3])
    expected = Series([2, 3, np.nan], index=range(1, 4), dtype=object)
    tm.assert_series_equal(result, expected)

    result = objects.reindex([1, 2, 3], fill_value="foo")
    expected = Series([2, 3, "foo"], index=range(1, 4), dtype=object)
    tm.assert_series_equal(result, expected)

    # ------------------------------------------------------------
    # bools
    bools = Series([True, False, True])

    result = bools.reindex([1, 2, 3])
    expected = Series([False, True, np.nan], index=range(1, 4), dtype=object)
    tm.assert_series_equal(result, expected)

    result = bools.reindex([1, 2, 3], fill_value=False)
    expected = Series([False, True, False], index=range(1, 4))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize("fill_value", ["string", 0, Timedelta(0)])
def test_reindex_fill_value_datetimelike_upcast(dtype, fill_value):
    # [FUNCTION DOC]
    # GitHub issue 42921, parametrized over datetime and timedelta dtype and fill
    # values (string, 0, a timedelta or timestamp not compatible with the dtype):
    # reindexing an NaT Series with an incompatible fill value gives object dtype with
    # the NaT and the fill value.
    # https://github.com/pandas-dev/pandas/issues/42921
    if dtype == "timedelta64[ns]" and fill_value == Timedelta(0):
        # use the scalar that is not compatible with the dtype for this test
        fill_value = Timestamp(0)

    ser = Series([NaT], dtype=dtype)

    result = ser.reindex([0, 1], fill_value=fill_value)
    expected = Series([NaT, fill_value], index=range(2), dtype=object)
    tm.assert_series_equal(result, expected)


def test_reindex_datetimeindexes_tz_naive_and_aware():
    # [FUNCTION DOC]
    # GH 8306: reindexing a tz-aware index to a naive one using the ffill method
    # raises a type error saying the dtypes cannot be compared.
    # GH 8306
    idx = date_range("20131101", tz="America/Chicago", periods=7, unit="ns")
    newidx = date_range("20131103", periods=10, freq="h", unit="ns")
    s = Series(range(7), index=idx)
    msg = (
        r"Cannot compare dtypes datetime64\[ns, America/Chicago\] "
        r"and datetime64\[ns\]"
    )
    with pytest.raises(TypeError, match=msg):
        s.reindex(newidx, method="ffill")


def test_reindex_empty_series_tz_dtype():
    # [FUNCTION DOC]
    # GH 20869: reindexing an empty tz-aware datetime Series gives NaT values with
    # the same dtype.
    # GH 20869
    result = Series(dtype="datetime64[ns, UTC]").reindex([0, 1])
    expected = Series([NaT] * 2, dtype="datetime64[ns, UTC]")
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "p_values, o_values, values, expected_values",
    [
        (
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC"), "All"],
            [1.0, 1.0],
            [1.0, 1.0, np.nan],
        ),
        (
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [1.0, 1.0],
            [1.0, 1.0],
        ),
    ],
)
def test_reindex_periodindex_with_object(p_values, o_values, values, expected_values):
    # [FUNCTION DOC]
    # GH#28337, parametrized over period values and object-index values: reindexing
    # a period-indexed Series by an object index containing periods matches the
    # periods and gives NaN for non-period labels like "All".
    # GH#28337
    period_index = PeriodIndex(p_values)
    object_index = Index(o_values)

    ser = Series(values, index=period_index)
    result = ser.reindex(object_index)
    expected = Series(expected_values, index=object_index)
    tm.assert_series_equal(result, expected)


def test_reindex_too_many_args():
    # [FUNCTION DOC]
    # GH 40980: passing two positional arguments raises a type error stating that
    # reindex takes from one to two positional arguments.
    # GH 40980
    ser = Series([1, 2])
    msg = r"reindex\(\) takes from 1 to 2 positional arguments but 3 were given"
    with pytest.raises(TypeError, match=msg):
        ser.reindex([2, 3], False)


def test_reindex_double_index():
    # [FUNCTION DOC]
    # GH 40980: giving the index both positionally and as keyword raises a type
    # error about multiple values for the argument.
    # GH 40980
    ser = Series([1, 2])
    msg = r"reindex\(\) got multiple values for argument 'index'"
    with pytest.raises(TypeError, match=msg):
        ser.reindex([2, 3], index=[3, 4])


def test_reindex_no_posargs():
    # [FUNCTION DOC]
    # GH 40980: reindex using only the keyword index argument works.
    # GH 40980
    ser = Series([1, 2])
    result = ser.reindex(index=[1, 0])
    expected = Series([2, 1], index=[1, 0])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("values", [[["a"], ["x"]], [[], []]])
def test_reindex_empty_with_level(values):
    # [FUNCTION DOC]
    # GH41170, parametrized over a non-empty and an empty MultiIndex: reindexing to a
    # new label on level 0 gives an empty object Series with a MultiIndex of that
    # level label.
    # GH41170
    ser = Series(
        range(len(values[0])), index=MultiIndex.from_arrays(values), dtype="object"
    )
    result = ser.reindex(np.array(["b"]), level=0)
    expected = Series(
        index=MultiIndex(levels=[["b"], values[1]], codes=[[], []]), dtype="object"
    )
    tm.assert_series_equal(result, expected)


def test_reindex_missing_category():
    # [FUNCTION DOC]
    # GH#18185: filling with a value that is not a category raises a type error saying
    # a new category cannot be set.
    # GH#18185
    ser = Series([1, 2, 3, 1], dtype="category")
    msg = r"Cannot setitem on a Categorical with a new category \(-1\)"
    with pytest.raises(TypeError, match=msg):
        ser.reindex([1, 2, 3, 4, 5], fill_value=-1)


def test_reindexing_with_float64_NA_log():
    # [FUNCTION DOC]
    # GH 47055: reindexing a masked Float64 Series fills with missing values in the
    # underlying data (as NaN), and a NumPy log on it does not warn and gives NA.
    # GH 47055
    s = Series([1.0, NA], dtype=Float64Dtype())
    s_reindex = s.reindex(range(3))
    result = s_reindex.values._data
    expected = np.array([1, np.nan, np.nan])
    tm.assert_numpy_array_equal(result, expected)
    with tm.assert_produces_warning(None):
        result_log = np.log(s_reindex)
        expected_log = Series([0, NA, NA], dtype=Float64Dtype())
        tm.assert_series_equal(result_log, expected_log)


@pytest.mark.parametrize("dtype", ["timedelta64", "datetime64"])
def test_reindex_expand_nonnano_nat(dtype):
    # [FUNCTION DOC]
    # GH 53497, parametrized over timedelta and datetime: reindexing a second-resolution
    # array to a longer range extends with NaT while keeping the second resolution.
    # GH 53497
    ser = Series(np.array([1], dtype=f"{dtype}[s]"))
    result = ser.reindex(RangeIndex(2))
    expected = Series(
        np.array([1, getattr(np, dtype)("nat", "s")], dtype=f"{dtype}[s]")
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "name, expected_match_level_a",
    [
        ("a", True),
        (None, False),
        ("x", False),
    ],
)
def test_reindex_multiindex_automatic_level(name, expected_match_level_a):
    # [FUNCTION DOC]
    # Parametrized over index name and whether it matches level a: reindexing a
    # Series with a single named index to a three-level MultiIndex target aligns
    # automatically on the matching level (named "a") and broadcasts values; for
    # other names (None or "x") all values are NaN.
    series = Series([26.73, 24.255], index=Index([81, 82], name=name))
    target = MultiIndex.from_product(
        [[81, 82], [np.nan], ["2018-06-01", "2018-07-01"]], names=["a", "b", "c"]
    )

    result = series.reindex(target)

    if expected_match_level_a:
        expected = Series(
            data=[26.73, 26.73, 24.255, 24.255],
            index=MultiIndex.from_product(
                [[81, 82], [np.nan], ["2018-06-01", "2018-07-01"]],
                names=["a", "b", "c"],
            ),
            dtype=series.dtype,
        )
    else:
        expected = Series(
            data=[np.nan] * 4,
            index=MultiIndex.from_product(
                [[81, 82], [np.nan], ["2018-06-01", "2018-07-01"]],
                names=["a", "b", "c"],
            ),
            dtype=series.dtype,
        )

    tm.assert_series_equal(result, expected)
