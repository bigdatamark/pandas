"""test get/set & misc"""
# [MODULE DOC]
# Miscellaneous Series indexing tests: missing keys, label and positional access, boxing
# of datetime-like scalars, ambiguous integer keys, setting and slicing, negative-step
# slices, special index key types (tuples, frozensets), boolean indexers with missing
# values, enlargement, rejected dict and set indexers, and value validation on item
# assignment for bool, int and float dtypes.

from datetime import timedelta
import re

import numpy as np
import pytest

from pandas.errors import IndexingError

from pandas import (
    NA,
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    isna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


def test_basic_indexing():
    # [FUNCTION DOC]
    # Verifies looking up a missing integer or string key in a series with duplicated
    # string labels raises a key error showing the key, before and after sorting the
    # index.
    s = Series(
        np.random.default_rng(2).standard_normal(5), index=["a", "b", "a", "a", "b"]
    )

    with pytest.raises(KeyError, match="^5$"):
        s[5]

    with pytest.raises(KeyError, match=r"^'c'$"):
        s["c"]

    s = s.sort_index()

    with pytest.raises(KeyError, match="^5$"):
        s[5]


def test_getitem_numeric_should_not_fallback_to_positional(any_numeric_dtype):
    # [FUNCTION DOC]
    # Regression test (GH 51053). Parametrized over numeric dtypes. For an index with a
    # duplicated label one, getting label 1 returns the two matching elements by label
    # (not by position) with the same index dtype.
    # GH51053
    dtype = any_numeric_dtype
    idx = Index([1, 0, 1], dtype=dtype)
    ser = Series(range(3), index=idx)
    result = ser[1]
    expected = Series([0, 2], index=Index([1, 1], dtype=dtype))
    tm.assert_series_equal(result, expected, check_exact=True)


def test_setitem_numeric_should_not_fallback_to_positional(any_numeric_dtype):
    # [FUNCTION DOC]
    # Regression test (GH 51053). Parametrized over numeric dtypes. Setting label 1 on
    # an index with a duplicated label one changes the two matching entries, not the
    # position-one entry.
    # GH51053
    dtype = any_numeric_dtype
    idx = Index([1, 0, 1], dtype=dtype)
    ser = Series(range(3), index=idx)
    ser[1] = 10
    expected = Series([10, 1, 10], index=idx)
    tm.assert_series_equal(ser, expected, check_exact=True)


def test_basic_getitem_with_labels(datetime_series):
    # [FUNCTION DOC]
    # Verifies indexing by an index subset equals reindexing, and a label slice equals
    # the label slice through the loc indexer (fixture series).
    indices = datetime_series.index[[5, 10, 15]]

    result = datetime_series[indices]
    expected = datetime_series.reindex(indices)
    tm.assert_series_equal(result, expected)

    result = datetime_series[indices[0] : indices[2]]
    expected = datetime_series.loc[indices[0] : indices[2]]
    tm.assert_series_equal(result, expected)


def test_basic_getitem_dt64tz_values():
    # [FUNCTION DOC]
    # Regression test (GH 12089). For a tz-aware datetime series with string labels,
    # label access through loc, positional iloc and direct indexing all return the same
    # Timestamp.
    # GH12089
    # with tz for values
    ser = Series(
        date_range("2011-01-01", periods=3, tz="US/Eastern"), index=["a", "b", "c"]
    )
    expected = Timestamp("2011-01-01", tz="US/Eastern")
    result = ser.loc["a"]
    assert result == expected
    result = ser.iloc[0]
    assert result == expected
    result = ser["a"]
    assert result == expected


def test_getitem_setitem_ellipsis():
    # [FUNCTION DOC]
    # Verifies indexing a random series with the ellipsis returns the whole series.
    # KNOWN ISSUE: the name mentions setting too, but only the get is tested.
    s = Series(np.random.default_rng(2).standard_normal(10))

    result = s[...]
    tm.assert_series_equal(result, s)


@pytest.mark.parametrize(
    "result_1, duplicate_item, expected_1",
    [
        [
            {1: 12, 2: [1, 2, 2, 3]},
            {1: 313},
            Series({1: 12}, dtype=object),
        ],
        [
            {1: [1, 2, 3], 2: [1, 2, 2, 3]},
            {1: [1, 2, 3]},
            Series({1: [1, 2, 3]}),
        ],
    ],
)
def test_getitem_with_duplicates_indices(result_1, duplicate_item, expected_1):
    # [FUNCTION DOC]
    # Regression test (GH 17610). Parametrized over two pairs of dict data. Two series
    # are concatenated using the private append helper (to create duplicate labels) and
    # compared with the equivalent expected construction; label one returns the matching
    # element and label two returns the unchanged value from the first series.
    # GH 17610
    result_1 = Series(result_1)
    duplicate_item = Series(duplicate_item)
    result = result_1._append_internal(duplicate_item)
    expected = expected_1._append_internal(duplicate_item)
    tm.assert_series_equal(result[1], expected)
    assert result[2] == result_1[2]


def test_getitem_setitem_integers():
    # [FUNCTION DOC]
    # Verifies positional and label access agree on a string-labelled series and that
    # positional assignment is visible by label.
    # caused bug without test
    s = Series([1, 2, 3], ["a", "b", "c"])

    assert s.iloc[0] == s["a"]
    s.iloc[0] = 5
    tm.assert_almost_equal(s["a"], 5)


def test_series_box_timestamp():
    # [FUNCTION DOC]
    # Verifies a datetime series returns Timestamp objects from direct indexing, the
    # scalar accessors at and iat, and loc and iloc, for default and datetime indexes.
    rng = date_range("20090415", "20090519", freq="B")
    ser = Series(rng)
    assert isinstance(ser[0], Timestamp)
    assert isinstance(ser.at[1], Timestamp)
    assert isinstance(ser.iat[2], Timestamp)
    assert isinstance(ser.loc[3], Timestamp)
    assert isinstance(ser.iloc[4], Timestamp)

    ser = Series(rng, index=rng)
    assert isinstance(ser[rng[0]], Timestamp)
    assert isinstance(ser.at[rng[1]], Timestamp)
    assert isinstance(ser.iat[2], Timestamp)
    assert isinstance(ser.loc[rng[3]], Timestamp)
    assert isinstance(ser.iloc[4], Timestamp)


def test_series_box_timedelta():
    # [FUNCTION DOC]
    # Verifies a timedelta series returns Timedelta objects from direct indexing, at,
    # iat, loc and iloc.
    rng = timedelta_range("1 day 1 s", periods=5, freq="h")
    ser = Series(rng)
    assert isinstance(ser[0], Timedelta)
    assert isinstance(ser.at[1], Timedelta)
    assert isinstance(ser.iat[2], Timedelta)
    assert isinstance(ser.loc[3], Timedelta)
    assert isinstance(ser.iloc[4], Timedelta)


def test_getitem_ambiguous_keyerror(indexer_sl):
    # [FUNCTION DOC]
    # Uses the indexer fixture (loc and direct getitem variants). Looking up label one
    # in an even-integer index raises a key error without falling back to position.
    ser = Series(range(10), index=list(range(0, 20, 2)))
    with pytest.raises(KeyError, match=r"^1$"):
        indexer_sl(ser)[1]


def test_getitem_dups_with_missing(indexer_sl):
    # [FUNCTION DOC]
    # Regression test (GH 4246). For a duplicate-labelled series with the indexer
    # fixture, a list including a missing label raises a key error naming the missing
    # label.
    # breaks reindex, so need to use .loc internally
    # GH 4246
    ser = Series([1, 2, 3, 4], ["foo", "bar", "foo", "bah"])
    with pytest.raises(KeyError, match=re.escape("['bam'] not in index")):
        indexer_sl(ser)[["foo", "bar", "bah", "bam"]]


def test_setitem_ambiguous_keyerror(indexer_sl):
    # [FUNCTION DOC]
    # Uses the indexer fixture. Setting a label that is absent from an even-integer
    # index appends it, equal to concatenating a one-element series.
    s = Series(range(10), index=list(range(0, 20, 2)))

    # equivalent of an append
    s2 = s.copy()
    indexer_sl(s2)[1] = 5
    expected = concat([s, Series([5], index=[1])])
    tm.assert_series_equal(s2, expected)


def test_setitem(datetime_series):
    # [FUNCTION DOC]
    # Verifies setting values to NaN through a label, an integer-position list, a
    # position and a boolean mask works and changes the values (fixture datetime
    # series).
    datetime_series[datetime_series.index[5]] = np.nan
    datetime_series.iloc[[1, 2, 17]] = np.nan
    datetime_series.iloc[6] = np.nan
    assert np.isnan(datetime_series.iloc[6])
    assert np.isnan(datetime_series.iloc[2])
    datetime_series[np.isnan(datetime_series)] = 5
    assert not np.isnan(datetime_series.iloc[2])


def test_setslice(datetime_series):
    # [FUNCTION DOC]
    # Verifies slicing a fixture series gives an index of matching length and unique
    # labels.
    # KNOWN ISSUE: the name suggests testing slice assignment, but only the slice's
    # index is examined.
    sl = datetime_series[5:20]
    assert len(sl) == len(sl.index)
    assert sl.index.is_unique is True


def test_basic_getitem_setitem_corner(datetime_series):
    # [FUNCTION DOC]
    # Verifies invalid tuple keys (slice with scalar) raise a key error saying a tuple
    # key is not found and not a multi-index, for get and set; a one-element list
    # holding a slice raises a value error about single-item lists (GH 31299); a
    # one-element tuple holding a slice works; nested unhashable lists raise a type
    # error for get and set.
    # invalid tuples, e.g. td.ts[:, None] vs. td.ts[:, 2]
    msg = "key of type tuple not found and not a MultiIndex"
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2]
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2] = 2

    # weird lists. [slice(0, 5)] raises but not two slices
    msg = "Indexing with a single-item list"
    with pytest.raises(ValueError, match=msg):
        # GH#31299
        datetime_series[[slice(None, 5)]]

    # but we're OK with a single-element tuple
    result = datetime_series[(slice(None, 5),)]
    expected = datetime_series[:5]
    tm.assert_series_equal(result, expected)

    # OK
    msg = r"unhashable type(: 'slice')?"
    with pytest.raises(TypeError, match=msg):
        datetime_series[[5, [None, None]]]
    with pytest.raises(TypeError, match=msg):
        datetime_series[[5, [None, None]]] = 2


def test_slice(string_series, object_series):
    # [FUNCTION DOC]
    # Verifies slicing: bounds, lengths, labels and the numpy values of slices match;
    # modifying a slice does not change the parent (copy on write).
    original = string_series.copy()
    numSlice = string_series[10:20]
    numSliceEnd = string_series[-10:]
    objSlice = object_series[10:20]

    assert string_series.index[9] not in numSlice.index
    assert object_series.index[9] not in objSlice.index

    assert len(numSlice) == len(numSlice.index)
    assert string_series[numSlice.index[0]] == numSlice[numSlice.index[0]]

    assert numSlice.index[1] == string_series.index[11]
    tm.assert_numpy_array_equal(np.array(numSliceEnd), np.array(string_series)[-10:])

    # Test return view.
    sl = string_series[10:20]
    sl[:] = 0

    # Doesn't modify parent (CoW)
    tm.assert_series_equal(string_series, original)


def test_timedelta_assignment():
    # [FUNCTION DOC]
    # Regression test (GH 8209). Setting a timedelta on an empty object series with loc
    # gives a microsecond-resolution timedelta series with that label; reindexing adds
    # NaT, and setting the new label fills it.
    # GH 8209
    s = Series([], dtype=object)
    s.loc["B"] = timedelta(1)
    expected = Series(Timedelta("1 days"), dtype="timedelta64[us]", index=["B"])
    tm.assert_series_equal(s, expected)

    s = s.reindex(s.index.insert(0, "A"))
    expected = Series(
        [np.nan, Timedelta("1 days")], dtype="timedelta64[us]", index=["A", "B"]
    )
    tm.assert_series_equal(s, expected)

    s.loc["A"] = timedelta(1)
    expected = Series(Timedelta("1 days"), dtype="timedelta64[us]", index=["A", "B"])
    tm.assert_series_equal(s, expected)


def test_underlying_data_conversion():
    # [FUNCTION DOC]
    # Regression test (GH 4080). Updating a frame column by a series with a multi-level
    # key goes through chained assignment, which raises the chained-assignment error and
    # leaves the frame unchanged.
    # KNOWN ISSUE: a bare statement that only evaluates the frame has no effect.
    # GH 4080
    df = DataFrame({c: [1, 2, 3] for c in ["a", "b", "c"]})
    return_value = df.set_index(["a", "b", "c"], inplace=True)
    assert return_value is None
    s = Series([1], index=[(2, 2, 2)])
    df["val"] = 0
    df_original = df.copy()
    df

    with tm.raises_chained_assignment_error():
        df["val"].update(s)
    expected = df_original
    tm.assert_frame_equal(df, expected)


def test_preserve_refs(datetime_series):
    # [FUNCTION DOC]
    # Verifies modifying a series obtained from a list-like selection does not change
    # the original (fixture series).
    seq = datetime_series.iloc[[5, 10, 15]]
    seq.iloc[1] = np.nan
    assert not np.isnan(datetime_series.iloc[10])


def test_multilevel_preserve_name(lexsorted_two_level_string_multiindex, indexer_sl):
    # [FUNCTION DOC]
    # Uses the multi-index and indexer fixtures. Selecting the top-level key "foo" from
    # a two-level indexed series named sth keeps the name.
    index = lexsorted_two_level_string_multiindex
    ser = Series(
        np.random.default_rng(2).standard_normal(len(index)), index=index, name="sth"
    )

    result = indexer_sl(ser)["foo"]
    assert result.name == ser.name


# miscellaneous methods


@pytest.mark.parametrize(
    "index",
    [
        date_range("2014-01-01", periods=20, freq="MS"),
        period_range("2014-01", periods=20, freq="M"),
        timedelta_range("0", periods=20, freq="h"),
    ],
)
def test_slice_with_negative_step(index):
    # [FUNCTION DOC]
    # Parametrized over datetime, period and timedelta indexes. Slices with negative
    # steps using string or index-object bounds give the same result as the equivalent
    # positional slices, for start-only, stop-only and both bounds; reversed bounds give
    # an empty slice. Uses the testing helper for slice equivalence.
    keystr1 = str(index[9])
    keystr2 = str(index[13])

    ser = Series(np.arange(20), index)
    SLC = IndexSlice

    for key in [keystr1, index[9]]:
        tm.assert_indexing_slices_equivalent(ser, SLC[key::-1], SLC[9::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[:key:-1], SLC[:8:-1])

        for key2 in [keystr2, index[13]]:
            tm.assert_indexing_slices_equivalent(ser, SLC[key2:key:-1], SLC[13:8:-1])
            tm.assert_indexing_slices_equivalent(ser, SLC[key:key2:-1], SLC[0:0:-1])


def test_tuple_index():
    # [FUNCTION DOC]
    # Regression test (GH 35534). A series indexed by one-element tuples can be read and
    # set by tuple keys.
    # GH 35534 - Selecting values when a Series has an Index of tuples
    s = Series([1, 2], index=[("a",), ("b",)])
    assert s[("a",)] == 1
    assert s[("b",)] == 2
    s[("b",)] = 3
    assert s[("b",)] == 3


def test_frozenset_index():
    # [FUNCTION DOC]
    # Regression test (GH 35747). A series indexed by frozensets can be read and set by
    # frozenset keys.
    # GH35747 - Selecting values when a Series has an Index of frozenset
    idx0, idx1 = frozenset("a"), frozenset("b")
    s = Series([1, 2], index=[idx0, idx1])
    assert s[idx0] == 1
    assert s[idx1] == 2
    s[idx1] = 3
    assert s[idx1] == 3


def test_loc_setitem_all_false_indexer():
    # [FUNCTION DOC]
    # Regression test (GH 45778). Setting with a series on an all-false boolean mask
    # through loc changes nothing.
    # GH#45778
    ser = Series([1, 2], index=["a", "b"])
    expected = ser.copy()
    rhs = Series([6, 7], index=["a", "b"])
    ser.loc[ser > 100] = rhs
    tm.assert_series_equal(ser, expected)


def test_loc_boolean_indexer_non_matching_index():
    # [FUNCTION DOC]
    # Regression test (GH 46551). A nullable boolean indexer with a missing value and a
    # false value, with a different length from the series, selects nothing and gives an
    # empty integer series.
    # GH#46551
    ser = Series([1])
    result = ser.loc[Series([NA, False], dtype="boolean")]
    expected = Series([], dtype="int64")
    tm.assert_series_equal(result, expected)


def test_loc_boolean_indexer_miss_matching_index():
    # [FUNCTION DOC]
    # Regression test (GH 46551). A nullable boolean indexer with an index that does not
    # align raises an indexing error saying it is unalignable.
    # GH#46551
    ser = Series([1])
    indexer = Series([NA, False], dtype="boolean", index=[1, 2])
    with pytest.raises(IndexingError, match="Unalignable"):
        ser.loc[indexer]


def test_loc_setitem_nested_data_enlargement():
    # [FUNCTION DOC]
    # Regression test (GH 48614). Enlarging a series by loc with a frame as a value
    # stores the frame as an object element.
    # GH#48614
    df = DataFrame({"a": [1]})
    ser = Series({"label": df})
    ser.loc["new_label"] = df
    expected = Series({"label": df, "new_label": df})
    tm.assert_series_equal(ser, expected)


def test_loc_ea_numeric_index_oob_slice_end():
    # [FUNCTION DOC]
    # Regression test (GH 50161). Slicing with loc to an end beyond the range on a
    # nullable integer index returns the existing tail.
    # GH#50161
    ser = Series(1, index=Index([0, 1, 2], dtype="Int64"))
    result = ser.loc[2:3]
    expected = Series(1, index=Index([2], dtype="Int64"))
    tm.assert_series_equal(result, expected)


def test_getitem_bool_int_key():
    # [FUNCTION DOC]
    # Regression test (GH 48653). For a series indexed by booleans, looking up integer
    # zero with loc raises a key error.
    # GH#48653
    ser = Series({True: 1, False: 0})
    with pytest.raises(KeyError, match="0"):
        ser.loc[0]


@pytest.mark.parametrize("val", [{}, {"b": "x"}])
@pytest.mark.parametrize("indexer", [[], [False, False], slice(0, -1), np.array([])])
def test_setitem_empty_indexer(indexer, val):
    # [FUNCTION DOC]
    # Regression test (GH 45981). Parametrized over an extra column or none and over
    # empty indexers (empty list, all-false list, slice excluding the last row, empty
    # array). Assigning through loc on a frame with such an indexer does not change the
    # frame.
    # GH#45981
    df = DataFrame({"a": [1, 2], **val})
    expected = df.copy()
    df.loc[indexer] = 1.5
    tm.assert_frame_equal(df, expected)


def test_loc_non_monotonic_index_with_a_missing_label():
    # [FUNCTION DOC]
    # Verifies a loc slice on a non-monotonic index where the start label is missing
    # raises a key error saying the left slice bound cannot be found for a non-monotonic
    # index with a missing label.
    msg = "Cannot get left slice bound for non-monotonic index with a missing label 4"
    ser = Series([3, 6, 7, 6], index=[3, 8, 7, 6])
    with pytest.raises(KeyError, match=msg):
        ser.loc[4:7]


class TestDeprecatedIndexers:
    # [CLASS DOC]
    # Groups tests of dict and set indexers, which are rejected (formerly deprecated,
    # enforced in version 2.0). Stateless.
    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_getitem_dict_and_set_deprecated(self, key):
        # [FUNCTION DOC]
        # Regression test (GH 42825). Parametrized over a set and a dict. Indexing a
        # series with such a key through loc raises a type error saying it is not
        # supported as an indexer.
        # KNOWN ISSUE: the name says deprecated but the behavior is an error since 2.0.
        # GH#42825 enforced in 2.0
        ser = Series([1, 2])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key]

    @pytest.mark.parametrize("key", [{1}, {1: 1}, ({1}, 2), ({1: 1}, 2)])
    def test_getitem_dict_and_set_deprecated_multiindex(self, key):
        # [FUNCTION DOC]
        # Regression test (GH 42825). Same as the previous test on a multi-index series,
        # including keys that are tuples holding a set or a dict.
        # KNOWN ISSUE: the name says deprecated but the behavior is an error since 2.0.
        # GH#42825 enforced in 2.0
        ser = Series([1, 2], index=MultiIndex.from_tuples([(1, 2), (3, 4)]))
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key]

    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_setitem_dict_and_set_disallowed(self, key):
        # [FUNCTION DOC]
        # Regression test (GH 42825). Setting with a set or dict key through loc raises
        # a type error saying it is not supported as an indexer.
        # GH#42825 enforced in 2.0
        ser = Series([1, 2])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key] = 1

    @pytest.mark.parametrize("key", [{1}, {1: 1}, ({1}, 2), ({1: 1}, 2)])
    def test_setitem_dict_and_set_disallowed_multiindex(self, key):
        # [FUNCTION DOC]
        # Regression test (GH 42825). Same as the previous test on a multi-index series,
        # including tuples holding a set or dict.
        # GH#42825 enforced in 2.0
        ser = Series([1, 2], index=MultiIndex.from_tuples([(1, 2), (3, 4)]))
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key] = 1


class TestSetitemValidation:
    # [CLASS DOC]
    # Groups tests of value validation in item assignment (adapted from the masked array
    # indexing tests). Defines class-level lists of invalid scalars and indexers.
    # Stateless.
    # This is adapted from pandas/tests/arrays/masked/test_indexing.py
    def _check_setitem_invalid(self, ser, invalid, indexer):
        # [FUNCTION DOC]
        # HELPER: asserts that assigning an invalid value raises a type error with an
        # invalid value message through direct setitem, positional setting, label
        # setting and whole-series slice assignment. Inputs: ser (series), invalid (the
        # value), indexer.
        # KNOWN ISSUE: inside each raising block the statement that restores the copy
        # after the failing assignment is unreachable because the assignment raises
        # first, so the restore never happens.
        orig_ser = ser.copy()

        with pytest.raises(TypeError, match="Invalid value"):
            ser[indexer] = invalid
            ser = orig_ser.copy()

        with pytest.raises(TypeError, match="Invalid value"):
            ser.iloc[indexer] = invalid
            ser = orig_ser.copy()

        with pytest.raises(TypeError, match="Invalid value"):
            ser.loc[indexer] = invalid
            ser = orig_ser.copy()

        with pytest.raises(TypeError, match="Invalid value"):
            ser[:] = invalid

    def _check_setitem_valid(self, ser, value, indexer):
        # [FUNCTION DOC]
        # HELPER: performs the same four assignments as the invalid-check helper but
        # expects success; it asserts nothing about the result, only that no error is
        # raised. Inputs: ser, value, indexer.
        orig_ser = ser.copy()

        ser[indexer] = value
        ser = orig_ser.copy()

        ser.iloc[indexer] = value
        ser = orig_ser.copy()

        ser.loc[indexer] = value
        ser = orig_ser.copy()

        ser[:] = value

    _invalid_scalars = [
        1 + 2j,
        "True",
        "1",
        "1.0",
        NaT,
        np.datetime64("NaT", "ns"),
        np.timedelta64("NaT", "ns"),
    ]
    _indexers = [0, [0], slice(0, 1), [True, False, False], slice(None, None, None)]

    @pytest.mark.parametrize(
        "invalid", [*_invalid_scalars, 1, 1.0, np.int64(1), np.float64(1)]
    )
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_bool(self, invalid, indexer):
        # [FUNCTION DOC]
        # Parametrized over invalid scalars (complex, strings, NaT variants) plus
        # integers and floats and over the indexers. A boolean series rejects the
        # assignment with a type error.
        ser = Series([True, False, False], dtype="bool")
        self._check_setitem_invalid(ser, invalid, indexer)

    @pytest.mark.parametrize("invalid", [*_invalid_scalars, True, 1.5, np.float64(1.5)])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_int(self, invalid, any_int_numpy_dtype, indexer):
        # [FUNCTION DOC]
        # Parametrized over invalid scalars plus True and non-integral floats, any
        # integer numpy dtype and the indexers. Missing values other than datetime-like
        # missing values are accepted (valid helper); everything else raises the invalid
        # value type error.
        ser = Series([1, 2, 3], dtype=any_int_numpy_dtype)
        if isna(invalid) and invalid is not NaT and not np.isnat(invalid):
            self._check_setitem_valid(ser, invalid, indexer)
        else:
            self._check_setitem_invalid(ser, invalid, indexer)

    @pytest.mark.parametrize("invalid", [*_invalid_scalars, True])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_float(self, invalid, float_numpy_dtype, indexer):
        # [FUNCTION DOC]
        # Parametrized over invalid scalars plus True, float dtype fixture and indexers.
        # A float series that contains a missing value rejects all of them with a type
        # error.
        ser = Series([1, 2, None], dtype=float_numpy_dtype)
        self._check_setitem_invalid(ser, invalid, indexer)
