# [MODULE DOC]
# General tests of the base Index class: construction (from arrays, series, datetime-like indexes, empty inputs), dtype and view handling, equality and identity, mapping (functions, dictionaries, series, default dictionaries), appending, dropping labels, membership (isin), boolean comparison, level lookup, slicing properties, joins, string accessor, take, reindex, and many more. Most tests live in the main test class that provides a fixture for a simple string index; later classes and functions cover names, bool/int dtype quirks, numeric index behavior, helper functions in the index api module (combining indexes, ensuring index) and error cases. Parametrizes heavily over the shared index fixture, which supplies many index types by name.

from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re

import numpy as np
import pytest

from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import (
    is_any_real_numeric_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    IntervalIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    _get_combined_index,
    ensure_index,
    ensure_index_from_sequences,
)
from pandas.testing import assert_series_equal


class TestIndex:
    # [CLASS DOC]
    # Main test class for the base Index. Provides a fixture with a five-element string index.
    @pytest.fixture
    def simple_index(self) -> Index:
        # [FUNCTION DOC]
        # HELPER: fixture returning an index of the five letters a through e.
        return Index(list("abcde"))

    def test_can_hold_identifiers(self, simple_index):
        # [FUNCTION DOC]
        # The private identifier check returns true for a string index and its first element.
        index = simple_index
        key = index[0]
        assert index._can_hold_identifiers_and_holds_name(key) is True

    @pytest.mark.parametrize("index", ["datetime"], indirect=True)
    def test_new_axis(self, index):
        # [FUNCTION DOC]
        # Run only for the datetime index. Multi-dimensional indexing (adding a new axis with None and a slice) raises a value error saying multi-dimensional indexing is no longer supported.
        # TODO: a bunch of scattered tests check this deprecation is enforced.
        #  de-duplicate/centralize them.
        with pytest.raises(ValueError, match="Multi-dimensional indexing"):
            # GH#30588 multi-dimensional indexing deprecated
            index[None, :]

    def test_constructor_regular(self, index):
        # [FUNCTION DOC]
        # Uses the shared index fixture. An index contains all of its own elements.
        tm.assert_contains_all(index, index)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_constructor_casting(self, index):
        # [FUNCTION DOC]
        # Run only for the string index. Converting to a numpy array and constructing an index from it gives an equal index containing all the same items.
        # casting
        arr = np.array(index)
        new_index = Index(arr)
        tm.assert_contains_all(arr, new_index)
        tm.assert_index_equal(index, new_index)

    def test_constructor_copy(self, using_infer_string):
        # [FUNCTION DOC]
        # Constructing a named string index from a numpy array with copy enabled keeps the name and class; the stored values equal the array (or an extension string array when string inference is on). Mutating the source array afterwards does not alter the index.
        index = Index(list("abc"), name="name")
        arr = np.array(index)
        new_index = Index(arr, copy=True, name="name")
        assert isinstance(new_index, Index)
        assert new_index.name == "name"
        if using_infer_string:
            tm.assert_extension_array_equal(
                new_index.values, pd.array(arr, dtype="str")
            )
        else:
            tm.assert_numpy_array_equal(arr, new_index.values)
        arr[0] = "SOMEBIGLONGSTRING"
        assert new_index[0] != "SOMEBIGLONGSTRING"

    @pytest.mark.parametrize("cast_as_obj", [True, False])
    @pytest.mark.parametrize(
        "index",
        [
            date_range(
                "2015-01-01 10:00",
                freq="D",
                periods=3,
                tz="US/Eastern",
                name="Green Eggs & Ham",
            ),  # DTI with tz
            date_range("2015-01-01 10:00", freq="D", periods=3),  # DTI no tz
            timedelta_range("1 days", freq="D", periods=3),  # td
            period_range("2015-01-01", freq="D", periods=3),  # period
        ],
    )
    def test_constructor_from_index_dtlike(self, cast_as_obj, index):
        # [FUNCTION DOC]
        # Parametrized over cast_as_obj and datetime-like indexes (tz-aware, naive, timedelta, period). With cast_as_obj, building an index from the object-cast version gives object dtype; for datetime indexes adding 50 nanoseconds first must not raise and must preserve nanoseconds when forced to object dtype. Otherwise constructing from the index returns an equal index.
        if cast_as_obj:
            result = Index(index.astype(object))
            assert result.dtype == np.dtype(object)
            if isinstance(index, DatetimeIndex):
                # GH#23524 check that Index(dti, dtype=object) does not
                #  incorrectly raise ValueError, and that nanoseconds are not
                #  dropped
                index += pd.Timedelta(nanoseconds=50)
                result = Index(index, dtype=object)
                assert result.dtype == np.object_
                assert list(result) == list(index)
        else:
            result = Index(index)

            tm.assert_index_equal(result, index)

    @pytest.mark.parametrize(
        "index,has_tz",
        [
            (
                date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern"),
                True,
            ),  # datetimetz
            (timedelta_range("1 days", freq="D", periods=3), False),  # td
            (period_range("2015-01-01", freq="D", periods=3), False),  # period
        ],
    )
    def test_constructor_from_series_dtlike(self, index, has_tz):
        # [FUNCTION DOC]
        # Parametrized over tz-aware datetime, timedelta and period indexes with a has_tz flag. Index built from a series of the values equals the original index and keeps the timezone when applicable.
        result = Index(Series(index))
        tm.assert_index_equal(result, index)

        if has_tz:
            assert result.tz == index.tz

    def test_constructor_from_series_freq(self):
        # [FUNCTION DOC]
        # Building a datetime index from a series of month-start dates with an explicit month-start frequency equals the index built directly from the date strings with that frequency.
        # GH 6273
        # create from a series, passing a freq
        dts = ["1-1-1990", "2-1-1990", "3-1-1990", "4-1-1990", "5-1-1990"]
        expected = DatetimeIndex(dts, freq="MS")

        s = Series(pd.to_datetime(dts))
        result = DatetimeIndex(s, freq="MS")

        tm.assert_index_equal(result, expected)

    def test_constructor_from_frame_series_freq(self, using_infer_string):
        # [FUNCTION DOC]
        # Takes a frame column of date strings; datetime index built from it with a frequency equals an index built from the strings and named after the column. The column dtype is object (or str with string inference) and equals a series of the strings. When string inference is off, inferring the frequency of the column gives month-start.
        # GH 6273
        # create from a series, passing a freq
        dts = ["1-1-1990", "2-1-1990", "3-1-1990", "4-1-1990", "5-1-1990"]
        expected = DatetimeIndex(dts, freq="MS")

        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        df["date"] = dts
        result = DatetimeIndex(df["date"], freq="MS")
        dtype = object if not using_infer_string else "str"
        assert df["date"].dtype == dtype
        expected.name = "date"
        tm.assert_index_equal(result, expected)

        expected = Series(dts, name="date")
        tm.assert_series_equal(df["date"], expected)

        # GH 6274
        # infer freq of same
        if not using_infer_string:
            # Doesn't work with arrow strings
            freq = pd.infer_freq(df["date"])
            assert freq == "MS"

    def test_constructor_int_dtype_nan(self):
        # [FUNCTION DOC]
        # Constructing from a list containing NaN with dtype given as float gives a float64 index equal to the one built with explicit float64.
        # see gh-15187
        data = [np.nan]
        expected = Index(data, dtype=np.float64)
        result = Index(data, dtype="float")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "klass,dtype,na_val",
        [
            (Index, np.float64, np.nan),
            (DatetimeIndex, "datetime64[s]", pd.NaT),
        ],
    )
    def test_index_ctor_infer_nan_nat(self, klass, dtype, na_val):
        # [FUNCTION DOC]
        # Parametrized over (class, expected dtype, missing value) for float NaN and datetime NaT. An index of two missing values, from a list or a numpy array, equals the class-specific construction and has the expected dtype (float64 or second-resolution datetime).
        # GH 13467
        na_list = [na_val, na_val]
        expected = klass(na_list)
        assert expected.dtype == dtype

        result = Index(na_list)
        tm.assert_index_equal(result, expected)

        result = Index(np.array(na_list))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "vals,dtype",
        [
            ([1, 2, 3, 4, 5], "int"),
            ([1.1, np.nan, 2.2, 3.0], "float"),
            (["A", "B", "C", np.nan], "obj"),
        ],
    )
    def test_constructor_simple_new(self, vals, dtype):
        # [FUNCTION DOC]
        # Parametrized over integer, float-with-NaN, and object-with-NaN values. The private simple-new constructor, given the values and a name, returns an index equal to the normal constructor result.
        index = Index(vals, name=dtype)
        result = index._simple_new(index.values, dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    def test_constructor_dtypes_datetime(self, tz_naive_fixture, attr, klass):
        # [FUNCTION DOC]
        # Parametrized over attribute (values or asi8) and class (Index or datetime index), plus the timezone fixture. Starting from a nanosecond datetime range localized to the timezone, builds objects from the numpy datetimes or integer epoch values, either as arrays or lists, passing a timezone or dtype. Verifies: integer input is treated as wall-time then localized; Index rejects a tz keyword with a type error; converting naive to tz-aware dtype via cast raises a type error for a non-null timezone ("Cannot use .astype to convert from timezone-naive dtype"); otherwise results equal the expected localized index.
        # Test constructing with a datetimetz dtype
        # .values produces numpy datetimes, so these are considered naive
        # .asi8 produces integers, so these are considered epoch timestamps
        # ^the above will be true in a later version. Right now we `.view`
        # the i8 values as NS_DTYPE, effectively treating them as wall times.
        index = date_range("2011-01-01", periods=5, unit="ns")
        arg = getattr(index, attr)
        index = index.tz_localize(tz_naive_fixture)
        dtype = index.dtype

        # As of 2.0 astype raises on dt64.astype(dt64tz)
        err = tz_naive_fixture is not None
        msg = "Cannot use .astype to convert from timezone-naive dtype to"

        if attr == "asi8":
            result = DatetimeIndex(arg).tz_localize(tz_naive_fixture)
            tm.assert_index_equal(result, index)
        elif klass is Index:
            with pytest.raises(TypeError, match="unexpected keyword"):
                klass(arg, tz=tz_naive_fixture)
        else:
            result = klass(arg, tz=tz_naive_fixture)
            tm.assert_index_equal(result, index)

        if attr == "asi8":
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(arg).astype(dtype)
            else:
                result = DatetimeIndex(arg).astype(dtype)
                tm.assert_index_equal(result, index)
        else:
            result = klass(arg, dtype=dtype)
            tm.assert_index_equal(result, index)

        if attr == "asi8":
            result = DatetimeIndex(list(arg)).tz_localize(tz_naive_fixture)
            tm.assert_index_equal(result, index)
        elif klass is Index:
            with pytest.raises(TypeError, match="unexpected keyword"):
                klass(arg, tz=tz_naive_fixture)
        else:
            result = klass(list(arg), tz=tz_naive_fixture)
            tm.assert_index_equal(result, index)

        if attr == "asi8":
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(list(arg)).astype(dtype)
            else:
                result = DatetimeIndex(list(arg)).astype(dtype)
                tm.assert_index_equal(result, index)
        else:
            result = klass(list(arg), dtype=dtype)
            tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, TimedeltaIndex])
    def test_constructor_dtypes_timedelta(self, attr, klass):
        # [FUNCTION DOC]
        # Parametrized over attribute (values or asi8) and class (Index or timedelta index). From a nanosecond timedelta range with frequency cleared, constructing from the raw values array or list with that dtype gives back an equal index.
        index = timedelta_range("1 days", periods=5, unit="ns")
        index = index._with_freq(None)  # won't be preserved by constructors
        dtype = index.dtype

        values = getattr(index, attr)

        result = klass(values, dtype=dtype)
        tm.assert_index_equal(result, index)

        result = klass(list(values), dtype=dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("value", [[], iter([]), (_ for _ in [])])
    @pytest.mark.parametrize(
        "klass",
        [
            Index,
            CategoricalIndex,
            DatetimeIndex,
            TimedeltaIndex,
        ],
    )
    def test_constructor_empty(self, value, klass):
        # [FUNCTION DOC]
        # Parametrized over empty inputs (list, iterator, generator) and classes (Index, categorical, datetime, timedelta). The result is an instance of the class and has length zero.
        empty = klass(value)
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize(
        "empty,klass",
        [
            (PeriodIndex([], freq="D"), PeriodIndex),
            (PeriodIndex(iter([]), freq="D"), PeriodIndex),
            (PeriodIndex((_ for _ in []), freq="D"), PeriodIndex),
            (RangeIndex(step=1), RangeIndex),
            (MultiIndex(levels=[[1, 2], ["blue", "red"]], codes=[[], []]), MultiIndex),
        ],
    )
    def test_constructor_empty_special(self, empty, klass):
        # [FUNCTION DOC]
        # Parametrized over several already-built empty indexes (period from list/iterator/generator, range, multi-level). Each is an instance of its class with zero length.
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize(
        "index",
        [
            "datetime",
            "float64",
            "float32",
            "int64",
            "int32",
            "period",
            "range",
            "repeats",
            "timedelta",
            "tuples",
            "uint64",
            "uint32",
        ],
        indirect=True,
    )
    def test_view_with_args(self, index):
        # [FUNCTION DOC]
        # Parametrized over numeric, datetime, period, range, repeats, timedelta and tuple indexes. Viewing as 64-bit integers must not raise.
        index.view("i8")

    @pytest.mark.parametrize(
        "index",
        [
            "string",
            pytest.param("categorical", marks=pytest.mark.xfail(reason="gh-25464")),
            "bool-object",
            "bool-dtype",
            "empty",
        ],
        indirect=True,
    )
    def test_view_with_args_object_array_raises(self, index):
        # [FUNCTION DOC]
        # Parametrized over string, categorical (expected fail), bool-object, bool-dtype and empty indexes. Viewing as int64 raises a value error for boolean dtype (larger dtype message), otherwise a type error about not being able to change the data-type for object/string/reference arrays.
        if index.dtype == bool:
            msg = "When changing to a larger dtype"
            with pytest.raises(ValueError, match=msg):
                index.view("i8")
        else:
            msg = (
                r"Cannot change data-type for array of references\.|"
                r"Cannot change data-type for object array\.|"
                r"Cannot change data-type for array of strings\.|"
            )
            with pytest.raises(TypeError, match=msg):
                index.view("i8")

    @pytest.mark.parametrize(
        "index",
        ["int64", "int32", "range"],
        indirect=True,
    )
    def test_astype(self, index):
        # [FUNCTION DOC]
        # Parametrized over int64, int32 and range indexes. Casting to int64 works (the cast can be used for lookup of label 5) and preserves a name set on the source.
        casted = index.astype("i8")

        # it works!
        casted.get_loc(5)

        # pass on name
        index.name = "foobar"
        casted = index.astype("i8")
        assert casted.name == "foobar"

    def test_equals_object(self):
        # [FUNCTION DOC]
        # Two separately built string indexes with the same labels are equal.
        # same
        assert Index(["a", "b", "c"]).equals(Index(["a", "b", "c"]))

    @pytest.mark.parametrize(
        "comp", [Index(["a", "b"]), Index(["a", "b", "d"]), ["a", "b", "c"]]
    )
    def test_not_equals_object(self, comp):
        # [FUNCTION DOC]
        # Parametrized over a shorter index, a longer index and a plain list. A three-label string index is not equal to any of them.
        assert not Index(["a", "b", "c"]).equals(comp)

    def test_identical(self):
        # [FUNCTION DOC]
        # Identity requires equal names and types: equal values with different names are equal but not identical; after naming both the same they are identical; an index of tuples (which becomes multi-level) is not identical to one built from the same tuples without converting to multi-level.
        # index
        i1 = Index(["a", "b", "c"])
        i2 = Index(["a", "b", "c"])

        assert i1.identical(i2)

        i1 = i1.rename("foo")
        assert i1.equals(i2)
        assert not i1.identical(i2)

        i2 = i2.rename("foo")
        assert i1.identical(i2)

        i3 = Index([("a", "a"), ("a", "b"), ("b", "a")])
        i4 = Index([("a", "a"), ("a", "b"), ("b", "a")], tupleize_cols=False)
        assert not i3.identical(i4)

    def test_is_(self):
        # [FUNCTION DOC]
        # Checks the identity-of-underlying-data method: an index is itself and any chained views of itself; not another equal index, a copy (deep or shallow), a slice, or a numpy array. A view with a changed name is still "is" the original in both directions, but an index built from the raw values array is not, nor are two indexes built from the same array without copying.
        ind = Index(range(10))
        assert ind.is_(ind)
        assert ind.is_(ind.view().view().view().view())
        assert not ind.is_(Index(range(10)))
        assert not ind.is_(ind.copy())
        assert not ind.is_(ind.copy(deep=False))
        assert not ind.is_(ind[:])
        assert not ind.is_(np.array(range(10)))

        # quasi-implementation dependent
        assert ind.is_(ind.view())
        ind2 = ind.view()
        ind2.name = "bob"
        assert ind.is_(ind2)
        assert ind2.is_(ind)
        # doesn't matter if Indices are *actually* views of underlying data,
        assert not ind.is_(Index(ind.values))
        arr = np.array(range(1, 11))
        ind1 = Index(arr, copy=False)
        ind2 = Index(arr, copy=False)
        assert not ind1.is_(ind2)

    def test_asof_numeric_vs_bool_raises(self):
        # [FUNCTION DOC]
        # A numeric index and an object-bool index: asking for the as-of value at a boolean scalar raises a type error stating the dtypes cannot be compared; asking with an index argument in either direction raises an invalid-index error containing the offending index text.
        left = Index([1, 2, 3])
        right = Index([True, False], dtype=object)

        msg = "Cannot compare dtypes int64 and bool"
        with pytest.raises(TypeError, match=msg):
            left.asof(right[0])
        # TODO: should right.asof(left[0]) also raise?

        with pytest.raises(InvalidIndexError, match=re.escape(str(right))):
            left.asof(right)

        with pytest.raises(InvalidIndexError, match=re.escape(str(left))):
            right.asof(left)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_booleanindex(self, index):
        # [FUNCTION DOC]
        # Run only for the string index. Indexing with a boolean numpy mask (and with that mask as a list) keeps labels in order such that each label's location in the result equals its position.
        bool_index = np.ones(len(index), dtype=bool)
        bool_index[5:30:2] = False

        sub_index = index[bool_index]

        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i

        sub_index = index[list(bool_index)]
        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i

    def test_fancy(self, simple_index):
        # [FUNCTION DOC]
        # Indexing with a list of integer positions: each element of the result is found at its own location in the result.
        index = simple_index
        sl = index[[1, 2, 3]]
        for i in sl:
            assert i == sl[sl.get_loc(i)]

    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    @pytest.mark.parametrize("dtype", [int, np.bool_])
    def test_empty_fancy(self, index, dtype, request, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized over string and numeric indexes and over empty arrays of integer or boolean dtype. Indexing with an empty list returns an index identical to an empty one of the same dtype; with an empty integer array same result; with an empty boolean array raises a value error about the boolean indexer length. Expected failure for boolean dtype with string inference and string dtype (numpy behavior).
        if dtype is np.bool_ and using_infer_string and index.dtype == "string":
            request.applymarker(pytest.mark.xfail(reason="numpy behavior is buggy"))
        empty_arr = np.array([], dtype=dtype)
        empty_index = type(index)([], dtype=index.dtype)

        assert index[[]].identical(empty_index)
        if dtype == np.bool_:
            with pytest.raises(ValueError, match="length of the boolean indexer"):
                assert index[empty_arr].identical(empty_index)
        else:
            assert index[empty_arr].identical(empty_index)

    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    def test_empty_fancy_raises(self, index):
        # [FUNCTION DOC]
        # Parametrized over string and numeric indexes. An empty list indexes to an empty identical index, but an empty float array as indexer raises an index error saying arrays used as indices must be of integer or boolean type. Datetime indexes are excluded.
        # DatetimeIndex is excluded, because it overrides getitem and should
        # be tested separately.
        empty_farr = np.array([], dtype=np.float64)
        empty_index = type(index)([], dtype=index.dtype)

        assert index[[]].identical(empty_index)
        # np.ndarray only accepts ndarray of int & bool dtypes, so should Index
        msg = r"arrays used as indices must be of integer"
        with pytest.raises(IndexError, match=msg):
            index[empty_farr]

    def test_union_dt_as_obj(self, simple_index):
        # [FUNCTION DOC]
        # Union of a string index with a daily datetime range gives the concatenation as an object index; union with itself returns itself; each result contains all elements of the contributing inputs.
        # TODO: Replace with fixturesult
        index = simple_index
        date_index = date_range("2019-01-01", periods=10)
        first_cat = index.union(date_index)
        second_cat = index.union(index)

        appended = Index(np.append(index, date_index.astype("O")))

        tm.assert_index_equal(first_cat, appended)
        tm.assert_index_equal(second_cat, index)
        tm.assert_contains_all(index, first_cat)
        tm.assert_contains_all(index, second_cat)
        tm.assert_contains_all(date_index, first_cat)

    def test_map_with_tuples(self):
        # [FUNCTION DOC]
        # Mapping a function returning one-element tuples gives an index of tuples; mapping to two-element tuples gives a multi-level index.
        # GH 12766

        # Test that returning a single tuple from an Index
        #   returns an Index.
        index = Index(np.arange(3), dtype=np.int64)
        result = index.map(lambda x: (x,))
        expected = Index([(i,) for i in index])
        tm.assert_index_equal(result, expected)

        # Test that returning a tuple from a map of a single index
        #   returns a MultiIndex object.
        result = index.map(lambda x: (x, x == 1))
        expected = MultiIndex.from_tuples([(i, i == 1) for i in index])
        tm.assert_index_equal(result, expected)

    def test_map_with_tuples_mi(self):
        # [FUNCTION DOC]
        # Mapping a multi-level index with a function selecting the first element of each tuple returns a flat index of those first-level values.
        # Test that returning a single object from a MultiIndex
        #   returns an Index.
        first_level = ["foo", "bar", "baz"]
        multi_index = MultiIndex.from_tuples(zip(first_level, [1, 2, 3]))
        reduced_index = multi_index.map(lambda x: x[0])
        tm.assert_index_equal(reduced_index, Index(first_level))

    @pytest.mark.parametrize(
        "index",
        [
            date_range("2020-01-01", freq="D", periods=10),
            period_range("2020-01-01", freq="D", periods=10),
            timedelta_range("1 day", periods=10),
        ],
    )
    def test_map_tseries_indices_return_index(self, index):
        # [FUNCTION DOC]
        # Parametrized over datetime, period and timedelta indexes. Mapping every element to the constant one gives an integer index of ten ones.
        expected = Index([1] * 10)
        result = index.map(lambda x: 1)
        tm.assert_index_equal(expected, result)

    def test_map_tseries_indices_accsr_return_index(self):
        # [FUNCTION DOC]
        # Mapping an hourly named datetime index to the hour-of-day gives an int64 index of zero through 23 keeping the name (exact type).
        date_index = DatetimeIndex(
            date_range("2020-01-01", periods=24, freq="h"), name="hourly"
        )
        result = date_index.map(lambda x: x.hour)
        expected = Index(np.arange(24, dtype="int64"), name="hourly")
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    def test_map_dictlike_simple(self, mapper):
        # [FUNCTION DOC]
        # Parametrized over the mapper: a dictionary built from (index, value) pairs, or a series. Mapping an integer index with it gives an index of the three strings.
        # GH 12756
        expected = Index(["foo", "bar", "baz"])
        index = Index(np.arange(3), dtype=np.int64)
        result = index.map(mapper(expected.values, index))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_map_dictlike(self, index, mapper, request, using_infer_string):
        # [FUNCTION DOC]
        # Uses the shared index fixture, parametrized over mapper builders (dictionary or series). Skips categorical (tested elsewhere) and non-unique indexes; expected failure for python-storage string dtype without string inference (dtype not retained); ignores a period business-frequency deprecation warning. Maps to descending integers; expectation: empty index for empty input, the same constructor and dtype for numeric, the same dtype for nullable extension-backed base indexes, otherwise a plain integer index.
        # GH 12756
        if isinstance(index, CategoricalIndex):
            pytest.skip("Tested in test_categorical")
        elif not index.is_unique:
            pytest.skip("Cannot map duplicated index")
        if (
            not using_infer_string
            and isinstance(index.dtype, pd.StringDtype)
            and index.dtype.storage == "python"
        ):
            mark = pytest.mark.xfail(reason="map does not retain dtype")
            request.applymarker(mark)

        rng = np.arange(len(index), 0, -1, dtype=np.int64)

        if index.empty:
            # to match proper result coercion for uints
            expected = Index([])
        elif is_numeric_dtype(index.dtype):
            expected = index._constructor(rng, dtype=index.dtype)
        elif type(index) is Index and index.dtype != object:
            # i.e. EA-backed, for now just Nullable
            expected = Index(rng, dtype=index.dtype)
        else:
            expected = Index(rng)

        result = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [Series(["foo", 2.0, "baz"], index=[0, 2, -1]), {0: "foo", 2: 2.0, -1: "baz"}],
    )
    def test_map_with_non_function_missing_values(self, mapper):
        # [FUNCTION DOC]
        # Parametrized over a series mapper and an equivalent dictionary mapper with only some keys. Labels absent from the mapping become NaN; the result is mixed values.
        # GH 12756
        expected = Index([2.0, np.nan, "foo"])
        result = Index([2, 1, 0]).map(mapper)

        tm.assert_index_equal(expected, result)

    def test_map_na_exclusion(self):
        # [FUNCTION DOC]
        # Mapping with ignoring of missing values leaves NaN entries alone and applies the function elsewhere (doubling floats).
        index = Index([1.5, np.nan, 3, np.nan, 5])

        result = index.map(lambda x: x * 2, na_action="ignore")
        expected = index * 2
        tm.assert_index_equal(result, expected)

    def test_map_defaultdict(self):
        # [FUNCTION DOC]
        # Mapping with a default dictionary uses its default factory value for absent labels and the stored value for present labels.
        index = Index([1, 2, 3])
        default_dict = defaultdict(lambda: "blank")
        default_dict[1] = "stuff"
        result = index.map(default_dict)
        expected = Index(["stuff", "blank", "blank"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("name,expected", [("foo", "foo"), ("bar", None)])
    def test_append_empty_preserve_name(self, name, expected):
        # [FUNCTION DOC]
        # Parametrized over the right name and expected resulting name. Appending a named non-empty index to an empty named index keeps the name only when both names match.
        left = Index([], name="foo")
        right = Index([1, 2, 3], name=name)

        result = left.append(right)
        assert result.name == expected

    @pytest.mark.parametrize(
        "index, expected",
        [
            ("string", False),
            ("bool-object", False),
            ("bool-dtype", False),
            ("categorical", False),
            ("int64", True),
            ("int32", True),
            ("uint64", True),
            ("uint32", True),
            ("datetime", False),
            ("float64", True),
            ("float32", True),
        ],
        indirect=["index"],
    )
    def test_is_numeric(self, index, expected):
        # [FUNCTION DOC]
        # Parametrized over named fixtures and expected booleans. The "any real numeric dtype" predicate is true only for integer and float indexes (not strings, bools, categoricals or datetimes).
        assert is_any_real_numeric_dtype(index) is expected

    @pytest.mark.parametrize(
        "index, expected",
        [
            ("string", True),
            ("bool-object", True),
            ("bool-dtype", False),
            ("categorical", False),
            ("int64", False),
            ("int32", False),
            ("uint64", False),
            ("uint32", False),
            ("datetime", False),
            ("float64", False),
            ("float32", False),
        ],
        indirect=["index"],
    )
    def test_is_object(self, index, expected, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized likewise. The object-dtype predicate is true for string and bool-object fixtures, except that string indexes are not object with string inference on.
        if using_infer_string and index.dtype == "string" and expected:
            expected = False
        assert is_object_dtype(index) is expected

    def test_summary(self, index):
        # [FUNCTION DOC]
        # The private summary method runs on any index without raising.
        index._summary()

    def test_logical_compat(self, all_boolean_reductions, simple_index):
        # [FUNCTION DOC]
        # Uses the all_boolean_reductions fixture (all or any). The reduction on the index equals the reduction on its values array and its boolean result equals that on the corresponding series.
        index = simple_index
        left = getattr(index, all_boolean_reductions)()
        assert left == getattr(index.values, all_boolean_reductions)()
        right = getattr(index.to_series(), all_boolean_reductions)()
        # left might not match right exactly in e.g. string cases where the
        # because we use np.any/all instead of .any/all
        assert bool(left) == bool(right)

    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    def test_drop_by_str_label(self, index):
        # [FUNCTION DOC]
        # Dropping a set of labels from a string or numeric index removes them; dropping one scalar label removes only the first element.
        n = len(index)
        drop = index[list(range(5, 10))]
        dropped = index.drop(drop)

        expected = index[list(range(5)) + list(range(10, n))]
        tm.assert_index_equal(dropped, expected)

        dropped = index.drop(index[0])
        expected = index[1:]
        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    @pytest.mark.parametrize("keys", [["foo", "bar"], ["1", "bar"]])
    def test_drop_by_str_label_raises_missing_keys(self, index, keys):
        # [FUNCTION DOC]
        # Parametrized over key lists with labels not found. Dropping raises a key error with a message saying the labels were not found in axis.
        with pytest.raises(KeyError, match=".* not found in axis"):
            index.drop(keys)

    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    def test_drop_by_str_label_errors_ignore(self, index):
        # [FUNCTION DOC]
        # With error handling set to ignore, dropping a mix of present and absent labels removes only the present; dropping only absent labels returns the whole index.
        n = len(index)
        drop = index[list(range(5, 10))]
        mixed = [*drop.tolist(), "foo"]
        dropped = index.drop(mixed, errors="ignore")

        expected = index[list(range(5)) + list(range(10, n))]
        tm.assert_index_equal(dropped, expected)

        dropped = index.drop(["foo", "bar"], errors="ignore")
        expected = index[list(range(n))]
        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_loc(self):
        # [FUNCTION DOC]
        # Dropping the label one from a three-element integer index leaves the other two.
        # TODO: Parametrize numeric and str tests after self.strIndex fixture
        index = Index([1, 2, 3])
        dropped = index.drop(1)
        expected = Index([2, 3])

        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_raises_missing_keys(self):
        # [FUNCTION DOC]
        # Dropping a list that includes an absent label raises a key error naming that label as not found in axis.
        index = Index([1, 2, 3])
        with pytest.raises(KeyError, match=re.escape("[4] not found in axis")):
            index.drop([3, 4])

    @pytest.mark.parametrize(
        "key,expected", [(4, Index([1, 2, 3])), ([3, 4, 5], Index([1, 2]))]
    )
    def test_drop_by_numeric_label_errors_ignore(self, key, expected):
        # [FUNCTION DOC]
        # Parametrized over a scalar or list key and expected result. With ignore, absent labels are skipped.
        index = Index([1, 2, 3])
        dropped = index.drop(key, errors="ignore")

        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize(
        "values",
        [["a", "b", ("c", "d")], ["a", ("c", "d"), "b"], [("c", "d"), "a", "b"]],
    )
    @pytest.mark.parametrize("to_drop", [[("c", "d"), "a"], ["a", ("c", "d")]])
    def test_drop_tuple(self, values, to_drop):
        # [FUNCTION DOC]
        # Parametrized over orderings of an object index containing a tuple, and over drop lists containing the tuple and a string. Dropping both leaves only the third element; dropping one then the other (as scalar or one-element list) works; dropping the second again after it was already removed raises a key error with the tuple in the message.
        # GH 18304
        index = Index(values)
        expected = Index(["b"], dtype=object)

        result = index.drop(to_drop)
        tm.assert_index_equal(result, expected)

        removed = index.drop(to_drop[0])
        for drop_me in to_drop[1], [to_drop[1]]:
            result = removed.drop(drop_me)
            tm.assert_index_equal(result, expected)

        removed = index.drop(to_drop[1])
        msg = rf"\"\[{re.escape(to_drop[1].__repr__())}\] not found in axis\""
        for drop_me in to_drop[1], [to_drop[1]]:
            with pytest.raises(KeyError, match=msg):
                removed.drop(drop_me)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_drop_with_duplicates_in_index(self, index):
        # [FUNCTION DOC]
        # Uses the shared index fixture. Skips empty and multi-level indexes and interval indexes on 32-bit platforms. Repeats each unique value twice, then dropping the first value removes both copies. Ignores a period deprecation warning.
        # GH38051
        if len(index) == 0 or isinstance(index, MultiIndex):
            pytest.skip("Test doesn't make sense for empty MultiIndex")
        if isinstance(index, IntervalIndex) and not IS64:
            pytest.skip("Cannot test IntervalIndex with int64 dtype on 32 bit platform")
        index = index.unique().repeat(2)
        expected = index[2:]
        result = index.drop(index[0])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "attr",
        [
            "is_monotonic_increasing",
            "is_monotonic_decreasing",
            "_is_strictly_monotonic_increasing",
            "_is_strictly_monotonic_decreasing",
        ],
    )
    def test_is_monotonic_incomparable(self, attr):
        # [FUNCTION DOC]
        # Parametrized over the four monotonic attributes. An index mixing an integer and a datetime (not comparable) reports false for each.
        index = Index([5, datetime.now(), 7])
        assert not getattr(index, attr)

    @pytest.mark.parametrize("values", [["foo", "bar", "quux"], {"foo", "bar", "quux"}])
    @pytest.mark.parametrize(
        "index,expected",
        [
            (["qux", "baz", "foo", "bar"], [False, False, True, True]),
            ([], []),  # empty
        ],
    )
    def test_isin(self, values, index, expected):
        # [FUNCTION DOC]
        # Parametrized over values as a list or a set and over (index values, expected mask) including an empty index. isin returns a boolean array flagging membership.
        index = Index(index)
        result = index.isin(values)
        expected = np.array(expected, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_nan_common_object(
        self, nulls_fixture, nulls_fixture2, using_infer_string
    ):
        # [FUNCTION DOC]
        # Uses the nulls_fixture pair (cartesian product) and infer-string flag. An object index of a string and a null: when both nulls are floating NaN, or are the same object, the null is found; with string inference on a string-dtype index the null is also found; otherwise different null kinds do not match and nothing is found.
        # Test cartesian product of null fixtures and ensure that we don't
        # mangle the various types (save a corner case with PyPy)
        idx = Index(["a", nulls_fixture])

        # all nans are the same
        if (
            isinstance(nulls_fixture, float)
            and isinstance(nulls_fixture2, float)
            and math.isnan(nulls_fixture)
            and math.isnan(nulls_fixture2)
        ):
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        elif nulls_fixture is nulls_fixture2:  # should preserve NA type
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        elif using_infer_string and idx.dtype == "string":
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        else:
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, False]),
            )

    def test_isin_nan_common_float64(self, nulls_fixture, float_numpy_dtype):
        # [FUNCTION DOC]
        # Uses nulls_fixture and a float numpy dtype fixture. For NaT or NA, constructing a float index containing them raises a type error (cannot convert to float), and a float index containing NaN does not report them as members. Otherwise an index of one and the null reports NaN as member (second position) but not NaT.
        dtype = float_numpy_dtype

        if nulls_fixture is pd.NaT or nulls_fixture is pd.NA:
            # Check 1) that we cannot construct a float64 Index with this value
            #  and 2) that with a NaN we do not have .isin(nulls_fixture)
            msg = (
                r"float\(\) argument must be a string or a (real )?number, "
                f"not {type(nulls_fixture).__name__!r}"
            )
            with pytest.raises(TypeError, match=msg):
                Index([1.0, nulls_fixture], dtype=dtype)

            idx = Index([1.0, np.nan], dtype=dtype)
            assert not idx.isin([nulls_fixture]).any()
            return

        idx = Index([1.0, nulls_fixture], dtype=dtype)
        res = idx.isin([np.nan])
        tm.assert_numpy_array_equal(res, np.array([False, True]))

        # we cannot compare NaT with NaN
        res = idx.isin([pd.NaT])
        tm.assert_numpy_array_equal(res, np.array([False, False]))

    @pytest.mark.parametrize("level", [0, -1])
    @pytest.mark.parametrize(
        "index",
        [
            ["qux", "baz", "foo", "bar"],
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        ],
    )
    def test_isin_level_kwarg(self, level, index):
        # [FUNCTION DOC]
        # Parametrized over level (zero or minus one) and string or float indexes. isin with a level argument works for a single-level index, including after naming the index and passing its name as level.
        index = Index(index)
        values = [*index.tolist()[-2:], "nonexisting"]

        expected = np.array([False, False, True, True])
        tm.assert_numpy_array_equal(expected, index.isin(values, level=level))

        index.name = "foobar"
        tm.assert_numpy_array_equal(expected, index.isin(values, level="foobar"))

    def test_isin_level_kwarg_bad_level_raises(self, index):
        # [FUNCTION DOC]
        # Uses the shared index fixture. Levels outside the valid range (ten, number of levels, minus number of levels minus one) raise an index error saying too many levels.
        for level in [10, index.nlevels, -(index.nlevels + 1)]:
            with pytest.raises(IndexError, match="Too many levels"):
                index.isin([], level=level)

    @pytest.mark.parametrize("label", [1.0, "foobar", "xyzzy", np.nan])
    def test_isin_level_kwarg_bad_label_raises(self, label, index):
        # [FUNCTION DOC]
        # Parametrized over bad level labels. For multi-level indexes (renamed in the first two levels), an unknown level label raises a key error "Level ... not found"; for single-level indexes named "foo", it raises a key error saying the requested level does not match the index name.
        if isinstance(index, MultiIndex):
            index = index.rename(["foo", "bar", *index.names[2:]])
            msg = f"'Level {label} not found'"
        else:
            index = index.rename("foo")
            msg = rf"Requested level \({label}\) does not match index name \(foo\)"
        with pytest.raises(KeyError, match=msg):
            index.isin([], level=label)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # [FUNCTION DOC]
        # Parametrized over an empty list, empty object series, and empty numpy array. isin with an empty collection returns all false.
        # see gh-16991
        index = Index(["a", "b"])
        expected = np.array([False, False])

        result = index.isin(empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_isin_string_null(self, string_dtype_no_object):
        # [FUNCTION DOC]
        # Uses the non-object string dtype fixture. isin with a list holding None returns all false for a string index without nulls.
        # GH#55821
        index = Index(["a", "b"], dtype=string_dtype_no_object)
        result = index.isin([None])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "values",
        [
            [1, 2, 3, 4],
            [1.0, 2.0, 3.0, 4.0],
            [True, True, True, True],
            ["foo", "bar", "baz", "qux"],
            date_range("2018-01-01", freq="D", periods=4),
        ],
    )
    def test_boolean_cmp(self, values):
        # [FUNCTION DOC]
        # Parametrized over integer, float, boolean, string and datetime values. Comparing an index to the same values by equality gives an all-true boolean numpy array.
        index = Index(values)
        result = index == values
        expected = np.array([True, True, True, True], dtype=bool)

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize("name,level", [(None, 0), ("a", "a")])
    def test_get_level_values(self, index, name, level):
        # [FUNCTION DOC]
        # Run for the string index and parametrized over (name, level) pairs. The level-values method on a flat index returns an equal index, with the name set when given.
        expected = index.copy()
        if name:
            expected.name = name

        result = expected.get_level_values(level)
        tm.assert_index_equal(result, expected)

    def test_slice_keep_name(self):
        # [FUNCTION DOC]
        # A slice keeps the name of the string index.
        index = Index(["a", "b"], name="asdf")
        assert index.name == index[1:].name

    def test_slice_is_unique(self):
        # [FUNCTION DOC]
        # Index with duplicates is not unique, but a copy of a slice that excludes the duplicates is.
        # GH 57911
        index = Index([1, 1, 2, 3, 4])
        assert not index.is_unique
        filtered_index = index[2:].copy()
        assert filtered_index.is_unique

    def test_slice_is_montonic(self):
        # [FUNCTION DOC]
        # Checks monotonic-decreasing and increasing flags on slices of a non-strictly increasing index with a repeated end value: the trailing duplicate pair is both decreasing and increasing; larger slices are increasing only.
        """Test that is_monotonic_decreasing is correct on slices."""
        # GH 57911
        index = Index([1, 2, 3, 3])
        assert not index.is_monotonic_decreasing

        filtered_index = index[2:].copy()
        assert filtered_index.is_monotonic_decreasing
        assert filtered_index.is_monotonic_increasing

        filtered_index = index[1:].copy()
        assert not filtered_index.is_monotonic_decreasing
        assert filtered_index.is_monotonic_increasing

        filtered_index = index[:].copy()
        assert not filtered_index.is_monotonic_decreasing
        assert filtered_index.is_monotonic_increasing

    @pytest.mark.parametrize(
        "index",
        [
            "string",
            "datetime",
            "int64",
            "int32",
            "uint64",
            "uint32",
            "float64",
            "float32",
        ],
        indirect=True,
    )
    def test_join_self(self, index, join_type):
        # [FUNCTION DOC]
        # Run for string, datetime and numeric indexes with the join_type fixture. Joining an index with itself returns the index (sorted for outer joins).
        result = index.join(index, how=join_type)
        expected = index
        if join_type == "outer":
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("method", ["strip", "rstrip", "lstrip"])
    def test_str_attribute(self, method):
        # [FUNCTION DOC]
        # Parametrized over strip, rstrip, lstrip. The string accessor method applied to an index of padded names equals the plain string method applied element-wise.
        # GH9068
        index = Index([" jack", "jill ", " jesse ", "frank"])
        expected = Index([getattr(str, method)(x) for x in index.values])

        result = getattr(index.str, method)()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            Index(range(5)),
            date_range("2020-01-01", periods=10),
            MultiIndex.from_tuples([("foo", "1"), ("bar", "3")]),
            period_range(start="2000", end="2010", freq="Y"),
        ],
    )
    def test_str_attribute_raises(self, index):
        # [FUNCTION DOC]
        # Parametrized over non-string indexes (integer, datetime, multi-level, period). Using the string accessor raises an attribute error saying the accessor can only be used with string values.
        with pytest.raises(AttributeError, match="only use .str accessor"):
            index.str.repeat(2)

    @pytest.mark.parametrize(
        "expand,expected",
        [
            (None, Index([["a", "b", "c"], ["d", "e"], ["f"]])),
            (False, Index([["a", "b", "c"], ["d", "e"], ["f"]])),
            (
                True,
                MultiIndex.from_tuples(
                    [("a", "b", "c"), ("d", "e", np.nan), ("f", np.nan, np.nan)]
                ),
            ),
        ],
    )
    def test_str_split(self, expand, expected):
        # [FUNCTION DOC]
        # Parametrized over expand (none, false, true) and expected results. Splitting space-separated strings gives an index of lists when not expanding, or a multi-level index padded with NaN when expanding.
        index = Index(["a b c", "d e", "f"])
        if expand is not None:
            result = index.str.split(expand=expand)
        else:
            result = index.str.split()

        tm.assert_index_equal(result, expected)

    def test_str_bool_return(self):
        # [FUNCTION DOC]
        # A boolean string method (starts-with) on an index returns a numpy boolean array rather than a boolean index.
        # test boolean case, should return np.array instead of boolean Index
        index = Index(["a1", "a2", "b1", "b2"])
        result = index.str.startswith("a")
        expected = np.array([True, True, False, False])

        tm.assert_numpy_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_str_bool_series_indexing(self):
        # [FUNCTION DOC]
        # A series indexed by strings can be filtered with the starts-with mask from the string accessor of its index, yielding the matching rows.
        index = Index(["a1", "a2", "b1", "b2"])
        s = Series(range(4), index=index)

        result = s[s.index.str.startswith("a")]
        expected = Series(range(2), index=["a1", "a2"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "index,expected", [(list("abcd"), True), (range(4), False)]
    )
    def test_tab_completion(self, index, expected):
        # [FUNCTION DOC]
        # Parametrized over a string index and a range. The attribute listing contains the string accessor name only for the string index.
        # GH 9910
        index = Index(index)
        result = "str" in dir(index)
        assert result == expected

    def test_indexing_doesnt_change_class(self):
        # [FUNCTION DOC]
        # Slicing or list-indexing a mixed int/string object index yields object-dtype indexes that are identical to the expected, not reinterpreted as numeric.
        index = Index([1, 2, 3, "a", "b", "c"])

        assert index[1:3].identical(Index([2, 3], dtype=np.object_))
        assert index[[0, 1]].identical(Index([1, 2], dtype=np.object_))

    def test_outer_join_sort(self):
        # [FUNCTION DOC]
        # Outer-joining an integer index with a datetime index emits a runtime warning about unsupported comparison, and equals the union of the object-cast versions (which also warns).
        left_index = Index(np.random.default_rng(2).permutation(15))
        right_index = date_range("2020-01-01", periods=10)

        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            result = left_index.join(right_index, how="outer")

        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            expected = left_index.astype(object).union(right_index.astype(object))

        tm.assert_index_equal(result, expected)

    def test_take_fill_value(self):
        # [FUNCTION DOC]
        # Take with negative positions: default wraps; with fill enabled the minus-one position gives a missing value (NaN); with fill disabled even with fill_value flag it wraps; the name is preserved.
        # GH 12631
        index = Index(list("ABC"), name="xxx")
        result = index.take(np.array([1, 0, -1]))
        expected = Index(list("BAC"), name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        result = index.take(np.array([1, 0, -1]), fill_value=True)
        expected = Index(["B", "A", np.nan], name="xxx")
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = index.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index(["B", "A", "C"], name="xxx")
        tm.assert_index_equal(result, expected)

    def test_take_fill_value_none_raises(self):
        # [FUNCTION DOC]
        # With fill enabled, positions below minus one raise a value error stating that all indices must be at least minus one.
        index = Index(list("ABC"), name="xxx")
        msg = (
            "When allow_fill=True and fill_value is not None, all indices must be >= -1"
        )

        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -5]), fill_value=True)

    def test_take_bad_bounds_raises(self):
        # [FUNCTION DOC]
        # Taking a position out of bounds raises an index error saying out of bounds.
        index = Index(list("ABC"), name="xxx")
        with pytest.raises(IndexError, match="out of bounds"):
            index.take(np.array([1, -5]))

    @pytest.mark.parametrize("name", [None, "foobar"])
    @pytest.mark.parametrize(
        "labels",
        [
            [],
            np.array([]),
            ["A", "B", "C"],
            ["C", "B", "A"],
            np.array(["A", "B", "C"]),
            np.array(["C", "B", "A"]),
            # Must preserve name even if dtype changes
            date_range("20130101", periods=3).values,
            date_range("20130101", periods=3).tolist(),
        ],
    )
    def test_reindex_preserves_name_if_target_is_list_or_ndarray(self, name, labels):
        # [FUNCTION DOC]
        # Parametrized over name (none or a string) and target labels as lists, arrays, or datetimes. Reindexing preserves the source index's name even when the dtype changes.
        # GH6552
        index = Index([0, 1, 2])
        index.name = name
        assert index.reindex(labels)[0].name == name

    @pytest.mark.parametrize("labels", [[], np.array([]), np.array([], dtype=np.int64)])
    def test_reindex_preserves_type_if_target_is_empty_list_or_array(self, labels):
        # [FUNCTION DOC]
        # Reindexing a string index to an empty list or array keeps the dtype type.
        # GH7774
        index = Index(list("abc"))
        assert index.reindex(labels)[0].dtype.type == index.dtype.type

    def test_reindex_doesnt_preserve_type_if_target_is_empty_index(self):
        # [FUNCTION DOC]
        # Reindexing a string index to an empty datetime index gives a datetime64-typed result.
        # GH7774
        index = Index(list("abc"))
        labels = DatetimeIndex([])
        dtype = np.datetime64
        assert index.reindex(labels)[0].dtype.type == dtype

    def test_reindex_doesnt_preserve_type_if_target_is_empty_index_numeric(
        self, any_real_numpy_dtype
    ):
        # [FUNCTION DOC]
        # Parametrized over real numpy dtypes. Reindexing to an empty index of that numeric dtype gives that dtype.
        # GH7774
        dtype = any_real_numpy_dtype
        index = Index(list("abc"))
        labels = Index([], dtype=dtype)
        assert index.reindex(labels)[0].dtype == dtype

    def test_reindex_no_type_preserve_target_empty_mi(self):
        # [FUNCTION DOC]
        # Reindexing to an empty two-level multi-level index returns a multi-level result whose levels have the target's integer and float dtypes.
        index = Index(list("abc"))
        result = index.reindex(
            MultiIndex([Index([], np.int64), Index([], np.float64)], [[], []])
        )[0]
        assert result.levels[0].dtype.type == np.int64
        assert result.levels[1].dtype.type == np.float64

    def test_reindex_ignoring_level(self):
        # [FUNCTION DOC]
        # Reindexing with a level argument naming the (only) level works and returns the target index.
        # GH#35132
        idx = Index([1, 2, 3], name="x")
        idx2 = Index([1, 2, 3, 4], name="x")
        expected = Index([1, 2, 3, 4], name="x")
        result, _ = idx.reindex(idx2, level="x")
        tm.assert_index_equal(result, expected)

    def test_groupby(self):
        # [FUNCTION DOC]
        # Grouping a five-element integer index by an array of group keys returns a dictionary mapping each key to the sub-index of its members.
        index = Index(range(5))
        result = index.groupby(np.array([1, 1, 2, 2, 2]))
        expected = {1: Index([0, 1]), 2: Index([2, 3, 4])}

        tm.assert_dict_equal(result, expected)

    @pytest.mark.parametrize(
        "mi,expected",
        [
            (MultiIndex.from_tuples([(1, 2), (4, 5)]), np.array([True, True])),
            (MultiIndex.from_tuples([(1, 2), (4, 6)]), np.array([True, False])),
        ],
    )
    def test_equals_op_multiindex(self, mi, expected):
        # [FUNCTION DOC]
        # Parametrized over multi-level indexes (equal and one differing in an element) with the expected boolean array. Element-wise equality of a frame's two-level index with the multi-level index gives the expected mask.
        # GH9785
        # test comparisons of multiindex
        df = DataFrame(
            [3, 6],
            columns=["c"],
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        result = df.index == mi
        tm.assert_numpy_array_equal(result, expected)

    def test_equals_op_multiindex_identify(self):
        # [FUNCTION DOC]
        # Element-wise equality of a multi-level index with itself gives all true.
        df = DataFrame(
            [3, 6],
            columns=["c"],
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        result = df.index == df.index
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)]),
            Index(["foo", "bar", "baz"]),
        ],
    )
    def test_equals_op_mismatched_multiindex_raises(self, index):
        # [FUNCTION DOC]
        # Parametrized over a longer multi-level and a longer plain string index. Element-wise equality with a different length raises a value error with the message "Lengths must match".
        df = DataFrame(
            [3, 6],
            columns=["c"],
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        with pytest.raises(ValueError, match="Lengths must match"):
            df.index == index

    def test_equals_op_index_vs_mi_same_length(self, using_infer_string):
        # [FUNCTION DOC]
        # Equality comparison of a three-element multi-level index to a same-length string index gives all false. (The infer-string fixture is accepted but unused.)
        mi = MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)])
        index = Index(["foo", "bar", "baz"])

        result = mi == index
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt_conv, arg",
        [
            (pd.to_datetime, ["2000-01-01", "2000-01-02"]),
            (pd.to_timedelta, ["01:02:03", "01:02:04"]),
        ],
    )
    def test_dt_conversion_preserves_name(self, dt_conv, arg):
        # [FUNCTION DOC]
        # Parametrized over datetime and timedelta converters with sample strings. Converting a named string index keeps the name.
        # GH 10875
        index = Index(arg, name="label")
        assert index.name == dt_conv(index).name

    def test_cached_properties_not_settable(self):
        # [FUNCTION DOC]
        # Assigning to a cached property (uniqueness flag) of an index raises an attribute error ("Can't set attribute").
        index = Index([1, 2, 3])
        with pytest.raises(AttributeError, match="Can't set attribute"):
            index.is_unique = False

    def test_tab_complete_warning(self, ip):
        # [FUNCTION DOC]
        # Uses the interactive shell fixture; skipped without a recent enough IPython. Creates an index in the shell and requests tab completions for it, asserting no warnings are emitted (extra warnings tolerated).
        # https://github.com/pandas-dev/pandas/issues/16409
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; idx = pd.Index([1, 2])"
        ip.run_cell(code)

        # GH 31324 newer jedi version raises Deprecation warning;
        #  appears resolved 2021-02-02
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):
                list(ip.Completer.completions("idx.", 4))

    def test_contains_method_removed(self, index):
        # [FUNCTION DOC]
        # Uses the shared index fixture. The contains method exists only on interval indexes (called with 1 to make sure it works); for all others calling it raises an attribute error naming the class.
        # GH#30103 method removed for all types except IntervalIndex
        if isinstance(index, IntervalIndex):
            index.contains(1)
        else:
            msg = f"'{type(index).__name__}' object has no attribute 'contains'"
            with pytest.raises(AttributeError, match=msg):
                index.contains(1)

    def test_sortlevel(self):
        # [FUNCTION DOC]
        # Sorting a flat integer index by level: invalid ascending values (string, two-element list, list of string) raise errors with specific messages; valid forms (one-element list of true, true, false) return the sorted index (ascending or descending) as the first element of the result.
        index = Index([5, 4, 3, 2, 1])
        with pytest.raises(Exception, match="ascending must be a single bool value or"):
            index.sortlevel(ascending="True")

        with pytest.raises(
            Exception, match="ascending must be a list of bool values of length 1"
        ):
            index.sortlevel(ascending=[True, True])

        with pytest.raises(Exception, match="ascending must be a bool value"):
            index.sortlevel(ascending=["True"])

        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=[True])
        tm.assert_index_equal(result[0], expected)

        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=True)
        tm.assert_index_equal(result[0], expected)

        expected = Index([5, 4, 3, 2, 1])
        result = index.sortlevel(ascending=False)
        tm.assert_index_equal(result[0], expected)

    def test_sortlevel_na_position(self):
        # [FUNCTION DOC]
        # With the NaN position option set to first, a numeric index containing NaN sorts with NaN first.
        # GH#51612
        idx = Index([1, np.nan])
        result = idx.sortlevel(na_position="first")[0]
        expected = Index([np.nan, 1])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, expected_results",
        [
            (1, [np.nan, 10, 10, 10, 10]),
            (2, [np.nan, np.nan, 20, 20, 20]),
            (3, [np.nan, np.nan, np.nan, 30, 30]),
        ],
    )
    def test_index_diff(self, periods, expected_results):
        # [FUNCTION DOC]
        # Parametrized over periods one to three and expected values. Difference of an arithmetic index with that lag gives NaN for leading entries and constant differences afterwards (float index).
        # GH#19708
        idx = Index([10, 20, 30, 40, 50])
        result = idx.diff(periods)
        expected = Index(expected_results)

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "decimals, expected_results",
        [
            (0, [1.0, 2.0, 3.0]),
            (1, [1.2, 2.3, 3.5]),
            (2, [1.23, 2.35, 3.46]),
        ],
    )
    def test_index_round(self, decimals, expected_results):
        # [FUNCTION DOC]
        # Parametrized over decimals zero to two. Rounding a float index gives the expected rounded float index.
        # GH#19708
        idx = Index([1.234, 2.345, 3.456])
        result = idx.round(decimals)
        expected = Index(expected_results)

        tm.assert_index_equal(result, expected)


class TestMixedIntIndex:
    # [CLASS DOC]
    # Tests on an index mixing integers and strings (not orderable). Provides a fixture for such an index.
    # Mostly the tests from common.py for which the results differ
    # in py2 and py3 because ints and strings are uncomparable in py3
    # (GH 13514)
    @pytest.fixture
    def simple_index(self) -> Index:
        # [FUNCTION DOC]
        # HELPER: fixture returning an object index of mixed integers and letters.
        return Index([0, "a", 1, "b", 2, "c"])

    def test_argsort(self, simple_index):
        # [FUNCTION DOC]
        # Computing the sort order raises a type error about unsupported comparison between strings and integers.
        index = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            index.argsort()

    def test_numpy_argsort(self, simple_index):
        # [FUNCTION DOC]
        # The numpy argsort function applied to the index raises the same comparison type error.
        index = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            np.argsort(index)

    def test_copy_name(self, simple_index):
        # [FUNCTION DOC]
        # Constructing an index from another with a name and copy, then from that without copying, gives a new object that is equal and keeps the name; arithmetic between series using those indexes yields a result index with that name.
        # Check that "name" argument passed at initialization is honoured
        # GH12309
        index = simple_index

        first = type(index)(index, copy=True, name="mario")
        second = type(first)(first, copy=False)

        # Even though "copy=False", we want a new object.
        assert first is not second
        tm.assert_index_equal(first, second)

        assert first.name == "mario"
        assert second.name == "mario"

        s1 = Series(2, index=first)
        s2 = Series(3, index=second[:-1])

        s3 = s1 * s2

        assert s3.index.name == "mario"

    def test_copy_name2(self):
        # [FUNCTION DOC]
        # Copying keeps the name; copying with a new name gives that name without changing the original.
        # Check that adding a "name" parameter to the copy is honored
        # GH14302
        index = Index([1, 2], name="MyName")
        index1 = index.copy()

        tm.assert_index_equal(index, index1)

        index2 = index.copy(name="NewName")
        tm.assert_index_equal(index, index2, check_names=False)
        assert index.name == "MyName"
        assert index2.name == "NewName"

    def test_unique_na(self):
        # [FUNCTION DOC]
        # Unique values of a float index with NaN keep first-occurrence order, keep one NaN, and preserve the name.
        idx = Index([2, np.nan, 2, 1], name="my_index")
        expected = Index([2, np.nan, 1], name="my_index")
        result = idx.unique()
        tm.assert_index_equal(result, expected)

    def test_logical_compat(self, simple_index):
        # [FUNCTION DOC]
        # The all and any reductions agree with those of the values array.
        index = simple_index
        assert index.all() == index.values.all()
        assert index.any() == index.values.any()

    @pytest.mark.parametrize("how", ["any", "all"])
    @pytest.mark.parametrize("dtype", [None, object, "category"])
    @pytest.mark.parametrize(
        "vals,expected",
        [
            ([1, 2, 3], [1, 2, 3]),
            ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            ([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 3.0]),
            (["A", "B", "C"], ["A", "B", "C"]),
            (["A", np.nan, "B", "C"], ["A", "B", "C"]),
        ],
    )
    def test_dropna(self, how, dtype, vals, expected):
        # [FUNCTION DOC]
        # Parametrized over how (any or all), dtype (default, object, category) and values with and without missing. dropna removes missing entries and leaves other entries in order, keeping dtype.
        # GH 6194
        index = Index(vals, dtype=dtype)
        result = index.dropna(how=how)
        expected = Index(expected, dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("how", ["any", "all"])
    @pytest.mark.parametrize(
        "index,expected",
        [
            (
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),
            ),
            (
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03", pd.NaT]),
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),
            ),
            (
                TimedeltaIndex(["1 days", "2 days", "3 days"]),
                TimedeltaIndex(["1 days", "2 days", "3 days"]),
            ),
            (
                TimedeltaIndex([pd.NaT, "1 days", "2 days", "3 days", pd.NaT]),
                TimedeltaIndex(["1 days", "2 days", "3 days"]),
            ),
            (
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),
            ),
            (
                PeriodIndex(["2012-02", "2012-04", "NaT", "2012-05"], freq="M"),
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),
            ),
        ],
    )
    def test_dropna_dt_like(self, how, index, expected):
        # [FUNCTION DOC]
        # Parametrized over how and datetime, timedelta, period indexes with and without NaT. dropna removes missing values.
        result = index.dropna(how=how)
        tm.assert_index_equal(result, expected)

    def test_dropna_invalid_how_raises(self):
        # [FUNCTION DOC]
        # An unrecognized how option raises a value error saying the option is invalid.
        msg = "invalid how option: xxx"
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3]).dropna(how="xxx")

    @pytest.mark.parametrize(
        "index",
        [
            Index([np.nan]),
            Index([np.nan, 1]),
            Index([1, 2, np.nan]),
            Index(["a", "b", np.nan]),
            pd.to_datetime(["NaT"]),
            pd.to_datetime(["NaT", "2000-01-01"]),
            pd.to_datetime(["2000-01-01", "NaT", "2000-01-02"]),
            pd.to_timedelta(["1 day", "NaT"]),
        ],
    )
    def test_is_monotonic_na(self, index):
        # [FUNCTION DOC]
        # Parametrized over indexes containing a missing value. All four monotonic flags (increasing, decreasing, strictly both) are false.
        assert index.is_monotonic_increasing is False
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is False

    @pytest.mark.parametrize("dtype", ["f8", "m8[ns]", "M8[us]"])
    @pytest.mark.parametrize("unique_first", [True, False])
    def test_is_monotonic_unique_na(self, dtype, unique_first):
        # [FUNCTION DOC]
        # Parametrized over dtype (float, timedelta, datetime) and whether uniqueness is evaluated first. An index of a missing value and two equal values is not unique and not monotonic in either direction; checks both evaluation orders give the same results (cache order independence).
        # GH 55755
        index = Index([None, 1, 1], dtype=dtype)
        if unique_first:
            assert index.is_unique is False
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
        else:
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
            assert index.is_unique is False

    def test_int_name_format(self, frame_or_series):
        # [FUNCTION DOC]
        # Uses the frame_or_series fixture. A name of integer zero appears in the text representation of a series or frame with that index.
        index = Index(["a", "b", "c"], name=0)
        result = frame_or_series(list(range(3)), index=index)
        assert "0" in repr(result)

    def test_str_to_bytes_raises(self):
        # [FUNCTION DOC]
        # Converting an index of strings to bytes raises a type error saying the string object cannot be interpreted as an integer.
        # GH 26447
        index = Index([str(x) for x in range(10)])
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(index)

    @pytest.mark.filterwarnings("ignore:elementwise comparison failed:FutureWarning")
    def test_index_with_tuple_bool(self):
        # [FUNCTION DOC]
        # Ignores a numpy element-wise-comparison future warning. Comparing an index of tuples to a tuple gives a boolean array marking elements equal to that tuple.
        # GH34123
        # TODO: also this op right now produces FutureWarning from numpy
        #  https://github.com/numpy/numpy/issues/11521
        idx = Index([("a", "b"), ("b", "c"), ("c", "a")])
        result = idx == ("c", "a")
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)


class TestIndexUtils:
    # [CLASS DOC]
    # Tests of module-level helper functions for building indexes.
    @pytest.mark.parametrize(
        "data, names, expected",
        [
            ([[1, 2, 4]], None, Index([1, 2, 4])),
            ([[1, 2, 4]], ["name"], Index([1, 2, 4], name="name")),
            ([[1, 2, 3]], None, RangeIndex(1, 4)),
            ([[1, 2, 3]], ["name"], RangeIndex(1, 4, name="name")),
            (
                [["a", "a"], ["c", "d"]],
                None,
                MultiIndex([["a"], ["c", "d"]], [[0, 0], [0, 1]]),
            ),
            (
                [["a", "a"], ["c", "d"]],
                ["L1", "L2"],
                MultiIndex([["a"], ["c", "d"]], [[0, 0], [0, 1]], names=["L1", "L2"]),
            ),
        ],
    )
    def test_ensure_index_from_sequences(self, data, names, expected):
        # [FUNCTION DOC]
        # Parametrized over (data sequences, names, expected index). One sequence becomes a flat index (a range when consecutive integers) with optional name; several sequences become a multi-level index with optional names. Exact type match.
        result = ensure_index_from_sequences(data, names)
        tm.assert_index_equal(result, expected, exact=True)

    def test_ensure_index_mixed_closed_intervals(self):
        # [FUNCTION DOC]
        # Converting a list of intervals with different closed sides to an index gives an object-dtype index, not an interval index.
        # GH27172
        intervals = [
            pd.Interval(0, 1, closed="left"),
            pd.Interval(1, 2, closed="right"),
            pd.Interval(2, 3, closed="neither"),
            pd.Interval(3, 4, closed="both"),
        ]
        result = ensure_index(intervals)
        expected = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)

    def test_ensure_index_uint64(self):
        # [FUNCTION DOC]
        # A list with zero and the maximum unsigned 64-bit value is converted to an unsigned 64-bit index (not float), preserving the values exactly.
        # with both 0 and a large-uint64, np.array will infer to float64
        #  https://github.com/numpy/numpy/issues/19146
        #  but a more accurate choice would be uint64
        values = [0, np.iinfo(np.uint64).max]

        result = ensure_index(values)
        assert list(result) == values

        expected = Index(values, dtype="uint64")
        tm.assert_index_equal(result, expected)

    def test_get_combined_index(self):
        # [FUNCTION DOC]
        # Combining an empty list of indexes returns an empty range index.
        result = _get_combined_index([])
        expected = RangeIndex(0)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "opname",
    [
        "eq",
        "ne",
        "le",
        "lt",
        "ge",
        "gt",
        "add",
        "radd",
        "sub",
        "rsub",
        "mul",
        "rmul",
        "truediv",
        "rtruediv",
        "floordiv",
        "rfloordiv",
        "pow",
        "rpow",
        "mod",
        "divmod",
    ],
)
def test_generated_op_names(opname, index):
    # [FUNCTION DOC]
    # Parametrized over comparison and arithmetic operator names (including reflected, divmod, mod). Uses the shared index fixture; the dunder method fetched from the index has a name attribute equal to the dunder name.
    opname = f"__{opname}__"
    method = getattr(index, opname)
    assert method.__name__ == opname


@pytest.mark.parametrize(
    "klass",
    [
        partial(CategoricalIndex, data=[1]),
        partial(DatetimeIndex, data=["2020-01-01"]),
        partial(PeriodIndex, data=["2020-01-01"]),
        partial(TimedeltaIndex, data=["1 day"]),
        partial(RangeIndex, start=range(1)),
        partial(IntervalIndex, data=[pd.Interval(0, 1)]),
        partial(Index, data=["a"], dtype=object),
        partial(MultiIndex, levels=[1], codes=[0]),
    ],
)
def test_index_subclass_constructor_wrong_kwargs(klass):
    # [FUNCTION DOC]
    # Parametrized over partial constructors of the base index and its subclasses (categorical, datetime, period, timedelta, range, interval, multi-level). Passing an unknown keyword raises a type error about an unexpected keyword argument.
    # GH #19348
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        klass(foo="bar")


def test_deprecated_fastpath():
    # [FUNCTION DOC]
    # The removed fast-path constructor keyword raises a type error ("unexpected keyword argument") for object, integer, range and categorical constructors.
    msg = "[Uu]nexpected keyword argument"
    with pytest.raises(TypeError, match=msg):
        Index(np.array(["a", "b"], dtype=object), name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        Index(np.array([1, 2, 3], dtype="int64"), name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        RangeIndex(0, 5, 2, name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        CategoricalIndex(["a", "b", "c"], name="test", fastpath=True)


def test_shape_of_invalid_index():
    # [FUNCTION DOC]
    # Multi-dimensional indexing (all rows, new axis) of an index raises a value error about multi-dimensional indexing.
    # Pre-2.0, it was possible to create "invalid" index objects backed by
    # a multi-dimensional array (see https://github.com/pandas-dev/pandas/issues/27125
    # about this). However, as long as this is not solved in general,this test ensures
    # that the returned shape is consistent with this underlying array for
    # compat with matplotlib (see https://github.com/pandas-dev/pandas/issues/27775)
    idx = Index([0, 1, 2, 3])
    with pytest.raises(ValueError, match="Multi-dimensional indexing"):
        # GH#30588 multi-dimensional indexing deprecated
        idx[:, None]


@pytest.mark.parametrize("dtype", [None, np.int64, np.uint64, np.float64])
def test_validate_1d_input(dtype):
    # [FUNCTION DOC]
    # Parametrized over dtype (default, int64, uint64, float64). Building an index from a 3-dimensional array or a 2-dimensional frame raises a value error requiring 1-dimensional data; assigning a 2-dimensional array as a series' index raises the same.
    # GH#27125 check that we do not have >1-dimensional input
    msg = "Index data must be 1-dimensional"

    arr = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError, match=msg):
        Index(arr, dtype=dtype)

    df = DataFrame(arr.reshape(4, 2))
    with pytest.raises(ValueError, match=msg):
        Index(df, dtype=dtype)

    # GH#13601 trying to assign a multi-dimensional array to an index is not allowed
    ser = Series(0, range(4))
    with pytest.raises(ValueError, match=msg):
        ser.index = np.array([[2, 3]] * 4, dtype=dtype)


@pytest.mark.parametrize(
    "klass, extra_kwargs",
    [
        [Index, {}],
        *[[lambda x: Index(x, dtype=dtyp), {}] for dtyp in tm.ALL_REAL_NUMPY_DTYPES],
        [DatetimeIndex, {}],
        [TimedeltaIndex, {}],
        [PeriodIndex, {"freq": "Y"}],
    ],
)
def test_construct_from_memoryview(klass, extra_kwargs):
    # [FUNCTION DOC]
    # Parametrized over index classes (base, one per real numpy dtype, datetime, timedelta, and period with a yearly frequency). Constructing from a memory view of an integer array equals constructing from a list of the same values (exact type).
    # GH 13120
    result = klass(memoryview(np.arange(2000, 2005)), **extra_kwargs)
    expected = klass(list(range(2000, 2005)), **extra_kwargs)
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize("op", [operator.lt, operator.gt])
def test_nan_comparison_same_object(op):
    # [FUNCTION DOC]
    # Parametrized over less-than and greater-than. Comparing an index holding NaN with itself, or with a copy, gives false.
    # GH#47105
    idx = Index([np.nan])
    expected = np.array([False])

    result = op(idx, idx)
    tm.assert_numpy_array_equal(result, expected)

    result = op(idx, idx.copy())
    tm.assert_numpy_array_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_is_monotonic_pyarrow_list_type():
    # [FUNCTION DOC]
    # Skipped without pyarrow. An index of Arrow list values is neither monotonic increasing nor decreasing.
    # GH 57333
    import pyarrow as pa

    idx = Index([[1], [2, 3]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    assert not idx.is_monotonic_increasing
    assert not idx.is_monotonic_decreasing


def test_index_equals_different_string_dtype(string_dtype_no_object):
    # [FUNCTION DOC]
    # Uses the string dtype fixture. A default string index and a same-labelled index in a specific string dtype compare equal in both directions.
    # GH 61099
    idx_obj = Index(["a", "b", "c"])
    idx_str = Index(["a", "b", "c"], dtype=string_dtype_no_object)

    assert idx_obj.equals(idx_str)
    assert idx_str.equals(idx_obj)


def test_index_comparison_different_string_dtype(string_dtype_no_object):
    # [FUNCTION DOC]
    # Series with the same labels but a default string index versus one with a specific string dtype index: comparing them gives an all-true boolean series, with the index taking the dtype of the left operand.
    # GH 61099
    idx = Index(["a", "b", "c"])
    s_obj = Series([1, 2, 3], index=idx)
    s_str = Series([4, 5, 6], index=idx.astype(string_dtype_no_object))

    expected = Series([True, True, True], index=["a", "b", "c"])
    result = s_obj < s_str
    assert_series_equal(result, expected)

    result = s_str > s_obj
    expected.index = idx.astype(string_dtype_no_object)
    assert_series_equal(result, expected)
