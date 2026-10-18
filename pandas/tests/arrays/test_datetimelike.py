# [MODULE DOC]
# Shared tests of the datetime-like extension arrays (DatetimeArray, TimedeltaArray
# and PeriodArray). A base test class (SharedTests) holds behavior common to all
# three and is specialized by one subclass per array type, which also add
# type-specific tests (numpy array protocol, conversions, field properties,
# strftime, to_period / to_timestamp, take with fill values). Module level tests
# cover NaT casting on setitem, to_numpy corner cases, searchsorted with
# list-likes, and construction from numpy-backed arrays. Most array behavior is
# already tested via the Index classes; these tests check the arrays match their
# index counterparts. Defines the fixtures freqstr, period_index, datetime_index
# and timedelta_index. No import-time side effects.

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from pandas._libs import (
    NaT,
    Timestamp,
)
from pandas._libs.tslibs import to_offset
from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
from pandas import (
    DatetimeIndex,
    Period,
    PeriodIndex,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)


# TODO: more freq variants
@pytest.fixture(params=["D", "B", "W", "ME", "QE", "YE"])
def freqstr(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over frequency strings (D, B, W, ME, QE, YE).
    # Returns the current frequency string; used by the index and array fixtures.
    """Fixture returning parametrized frequency in string format."""
    return request.param


@pytest.fixture
def period_index(freqstr):
    # [FUNCTION DOC]
    # HELPER: fixture. Builds a PeriodIndex of 100 periods starting at 2000-01-01
    # using the frequency from the freqstr fixture (converted via a period dtype so
    # the period-appropriate alias is used). A warning about business-day periods is
    # suppressed during construction. Returns the index.
    """
    A fixture to provide PeriodIndex objects with different frequencies.

    Most PeriodArray behavior is already tested in PeriodIndex tests,
    so here we just test that the PeriodArray behavior matches
    the PeriodIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates
    with warnings.catch_warnings():
        # suppress deprecation of Period[B]
        warnings.filterwarnings(
            "ignore", message="Period with BDay freq", category=FutureWarning
        )
        freqstr = PeriodDtype(to_offset(freqstr))._freqstr
        pi = pd.period_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    return pi


@pytest.fixture
def datetime_index(freqstr):
    # [FUNCTION DOC]
    # HELPER: fixture. Builds a DatetimeIndex of 100 nanosecond-unit timestamps
    # starting at 2000-01-01 with the frequency from the freqstr fixture and returns it.
    """
    A fixture to provide DatetimeIndex objects with different frequencies.

    Most DatetimeArray behavior is already tested in DatetimeIndex tests,
    so here we just test that the DatetimeArray behavior matches
    the DatetimeIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates, timezones
    dti = pd.date_range(
        start=Timestamp("2000-01-01"), periods=100, freq=freqstr, unit="ns"
    )
    return dti


@pytest.fixture
def timedelta_index():
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a small TimedeltaIndex of one day, three hours and NaT.
    # KNOWN ISSUE: the docstring speaks of different frequencies, but the index is a
    # single fixed set of three values without a frequency.
    """
    A fixture to provide TimedeltaIndex objects with different frequencies.
     Most TimedeltaArray behavior is already tested in TimedeltaIndex tests,
    so here we just test that the TimedeltaArray behavior matches
    the TimedeltaIndex behavior.
    """
    # TODO: flesh this out
    return TimedeltaIndex(["1 Day", "3 Hours", "NaT"])


class SharedTests:
    # [CLASS DOC]
    # Base class of tests shared by the three datetime-like array types. Subclasses
    # must define the class attributes index_cls, array_cls, scalar_type and
    # example_dtype, and may override the arr1d fixture. The tests compare array
    # behavior with the corresponding index behavior or with numpy conventions.
    index_cls: type[DatetimeIndex | PeriodIndex | TimedeltaIndex]

    @pytest.fixture
    def arr1d(self):
        # [FUNCTION DOC]
        # HELPER: fixture. Returns a 1-d array of ten daily values starting at epoch,
        # as a PeriodArray with daily frequency for the period subclass or otherwise the
        # array underlying an index of the subclass index type with daily frequency.
        # Subclasses override it with their own parametrization.
        """Fixture returning DatetimeArray with daily frequency."""
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, freq="D")
        else:
            arr = self.index_cls(data, freq="D")._data
        return arr

    def test_compare_len1_raises(self, arr1d):
        # [FUNCTION DOC]
        # Verifies comparing an array with a length-one array (which numpy would
        # broadcast) raises a value error ("Lengths must match"), for the array and,
        # GH#23078, for the index.
        # make sure we raise when comparing with different lengths, specific
        #  to the case where one has length-1, which numpy would broadcast
        arr = arr1d
        idx = self.index_cls(arr)

        with pytest.raises(ValueError, match="Lengths must match"):
            arr == arr[:1]

        # test the index classes while we're at it, GH#23078
        with pytest.raises(ValueError, match="Lengths must match"):
            idx <= idx[[0]]

    @pytest.mark.parametrize(
        "result",
        [
            pd.date_range("2020", periods=3),
            pd.date_range("2020", periods=3, tz="UTC"),
            pd.timedelta_range("0 days", periods=3),
            pd.period_range("2020Q1", periods=3, freq="Q"),
        ],
    )
    def test_compare_with_Categorical(self, result):
        # [FUNCTION DOC]
        # Parametrized over a tz-naive and tz-aware date range, a timedelta range and a
        # period range. Verifies equality with the Categorical of the same values is true
        # everywhere and inequality is false everywhere.
        expected = pd.Categorical(result)
        assert all(result == expected)
        assert not any(result != expected)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_compare_categorical_dtype(self, arr1d, as_index, reverse, ordered):
        # [FUNCTION DOC]
        # Parametrized over reverse, as_index and the ordered fixture. Compares the array
        # with a Categorical (or categorical index) built from it, in both operand
        # orders: equality gives all true and inequality all false; the order comparisons
        # (strictly less, less-equal, greater, greater-equal) are checked only for the array
        # on the left and a Categorical (not an index), since otherwise a type error arises.
        # KNOWN ISSUE: a TODO notes the behavior should arguably be the same
        # regardless of operand order and box.
        other = pd.Categorical(arr1d, ordered=ordered)
        if as_index:
            other = pd.CategoricalIndex(other)

        left, right = arr1d, other
        if reverse:
            left, right = right, left

        ones = np.ones(arr1d.shape, dtype=bool)
        zeros = ~ones

        result = left == right
        tm.assert_numpy_array_equal(result, ones)

        result = left != right
        tm.assert_numpy_array_equal(result, zeros)

        if not reverse and not as_index:
            # Otherwise Categorical raises TypeError bc it is not ordered
            # TODO: we should probably get the same behavior regardless?
            result = left < right
            tm.assert_numpy_array_equal(result, zeros)

            result = left <= right
            tm.assert_numpy_array_equal(result, ones)

            result = left > right
            tm.assert_numpy_array_equal(result, zeros)

            result = left >= right
            tm.assert_numpy_array_equal(result, ones)

    def test_take(self):
        # [FUNCTION DOC]
        # Builds a shuffled array and an index sharing its data and verifies take with
        # list and array positions equals the index take.
        data = np.arange(100, dtype="i8") * 24 * 3600 * 10**9
        np.random.default_rng(2).shuffle(data)

        if self.array_cls is PeriodArray:
            arr = PeriodArray(data, dtype="period[D]")
        else:
            arr = self.index_cls(data)._data
        idx = self.index_cls._simple_new(arr)

        takers = [1, 4, 94]
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

        takers = np.array([1, 4, 94])
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

    @pytest.mark.parametrize("fill_value", [2, 2.0, Timestamp(2021, 1, 1, 12).time])
    def test_take_fill_raises(self, fill_value, arr1d):
        # [FUNCTION DOC]
        # Parametrized over invalid fill values (integer, float, bound method of a time).
        # Verifies take with allow_fill raises a type error saying the value should be the
        # scalar type or NaT.
        # KNOWN ISSUE: the third parameter is the bound time method of a timestamp,
        # not a time object (the method is not called).
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([0, 1], allow_fill=True, fill_value=fill_value)

    def test_take_fill(self, arr1d):
        # [FUNCTION DOC]
        # Verifies take with allow_fill and fill_value None, NaN, or NaT puts NaT in the
        # filled position.
        arr = arr1d

        result = arr.take([-1, 1], allow_fill=True, fill_value=None)
        assert result[0] is NaT

        result = arr.take([-1, 1], allow_fill=True, fill_value=np.nan)
        assert result[0] is NaT

        result = arr.take([-1, 1], allow_fill=True, fill_value=NaT)
        assert result[0] is NaT

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_take_fill_str(self, arr1d):
        # [FUNCTION DOC]
        # Verifies take accepts a string fill value that parses to the array scalar type
        # (equal to taking the element), and a string that cannot be parsed raises a type
        # error naming the expected scalar type or NaT. Business-day period deprecation
        # warnings are ignored.
        # Cast str fill_value matching other fill_value-taking methods
        result = arr1d.take([-1, 1], allow_fill=True, fill_value=str(arr1d[-1]))
        expected = arr1d[[-1, 1]]
        tm.assert_equal(result, expected)

        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([-1, 1], allow_fill=True, fill_value="foo")

    def test_concat_same_type(self, arr1d):
        # [FUNCTION DOC]
        # Verifies the private concatenation of same-type arrays equals the index built
        # from the concatenated object values.
        # KNOWN ISSUE: the index with an inserted NaT built at the start is never used,
        # and the array is assigned twice.
        arr = arr1d
        idx = self.index_cls(arr)
        idx = idx.insert(0, NaT)
        arr = arr1d

        result = arr._concat_same_type([arr[:-1], arr[1:], arr])
        arr2 = arr.astype(object)
        expected = self.index_cls(np.concatenate([arr2[:-1], arr2[1:], arr2]))

        tm.assert_index_equal(self.index_cls(result), expected)

    def test_unbox_scalar(self, arr1d):
        # [FUNCTION DOC]
        # Verifies the private scalar unboxing returns an integer of the underlying
        # numpy scalar type for a scalar and for NaT, and raises a value error
        # ("'value' should be a <scalar type>") for a string.
        result = arr1d._unbox_scalar(arr1d[0])
        expected = arr1d._ndarray.dtype.type
        assert isinstance(result, expected)

        result = arr1d._unbox_scalar(NaT)
        assert isinstance(result, expected)

        msg = f"'value' should be a {self.scalar_type.__name__}."
        with pytest.raises(ValueError, match=msg):
            arr1d._unbox_scalar("foo")

    def test_check_compatible_with(self, arr1d):
        # [FUNCTION DOC]
        # Verifies the private compatibility check accepts an element, a slice array
        # and NaT without raising.
        arr1d._check_compatible_with(arr1d[0])
        arr1d._check_compatible_with(arr1d[:1])
        arr1d._check_compatible_with(NaT)

    def test_scalar_from_string(self, arr1d):
        # [FUNCTION DOC]
        # Verifies parsing the string form of the first element via the private
        # scalar-from-string method gives the element.
        result = arr1d._scalar_from_string(str(arr1d[0]))
        assert result == arr1d[0]

    def test_reduce_invalid(self, arr1d):
        # [FUNCTION DOC]
        # Verifies the private reduce with an unknown method name raises a type error
        # ("does not support operation 'not a method'").
        msg = "does not support operation 'not a method'"
        with pytest.raises(TypeError, match=msg):
            arr1d._reduce("not a method")

    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_method_doesnt_change_orig(self, method):
        # [FUNCTION DOC]
        # Parametrized over method (pad, backfill). Sets one element to NaT, calls the
        # private pad-or-backfill, verifies the filled element equals the neighbor and
        # that the original array still holds NaT (not mutated).
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            dtype = "M8[ns]" if self.array_cls is DatetimeArray else "m8[ns]"
            arr = self.array_cls._from_sequence(data, dtype=np.dtype(dtype))
        arr[4] = NaT

        fill_value = arr[3] if method == "pad" else arr[5]

        result = arr._pad_or_backfill(method=method)
        assert result[4] == fill_value

        # check that the original was not changed
        assert arr[4] is NaT

    def test_searchsorted(self):
        # [FUNCTION DOC]
        # Verifies searchsorted for a scalar (left and right side), for an array of the same
        # type, and that NaT is placed at the end (position equal to length), matching
        # numpy convention (GH#29884).
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            dtype = "M8[ns]" if self.array_cls is DatetimeArray else "m8[ns]"
            arr = self.array_cls._from_sequence(data, dtype=np.dtype(dtype))

        # scalar
        result = arr.searchsorted(arr[1])
        assert result == 1

        result = arr.searchsorted(arr[2], side="right")
        assert result == 3

        # own-type
        result = arr.searchsorted(arr[1:3])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = arr.searchsorted(arr[1:3], side="right")
        expected = np.array([2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        # GH#29884 match numpy convention on whether NaT goes
        #  at the end or the beginning
        result = arr.searchsorted(NaT)
        assert result == 10

    @pytest.mark.parametrize("box", [None, "index", "series"])
    def test_searchsorted_castable_strings(
        self, arr1d, box, string_storage, using_infer_string
    ):
        # [FUNCTION DOC]
        # Parametrized over box (none, index, series), uses string_storage and
        # using_infer_string fixtures. Verifies searchsorted accepts strings parseable as
        # the scalar type, scalar and list-like, on the array, the index and the Series,
        # returning expected positions. Verifies unparseable strings raise a type error
        # saying the value should be the scalar type, NaT or an array of those (the message
        # names the str or string type depending on the infer-string mode and storage).
        arr = arr1d
        if box is None:
            pass
        elif box == "index":
            # Test the equivalent Index.searchsorted method while we're here
            arr = self.index_cls(arr)
        else:
            # Test the equivalent Series.searchsorted method while we're here
            arr = pd.Series(arr)

        # scalar
        result = arr.searchsorted(str(arr[1]))
        assert result == 1

        result = arr.searchsorted(str(arr[2]), side="right")
        assert result == 3

        result = arr.searchsorted([str(x) for x in arr[1:3]])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                "or array of those. Got 'str' instead."
            ),
        ):
            arr.searchsorted("foo")

        msg = re.escape(
            f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
            "or array of those. Got str array instead."
        )
        if not using_infer_string:
            msg = msg.replace("str", "string")
        with pd.option_context("string_storage", string_storage):
            with pytest.raises(
                TypeError,
                match=msg,
            ):
                arr.searchsorted([str(arr[1]), "baz"])

    def test_getitem_near_implementation_bounds(self):
        # [FUNCTION DOC]
        # Verifies indexing arrays, indexes and Series holding values near the lower
        # bound of representable values (just above NaT) does not raise an out-of-bounds
        # error.
        # We only check tz-naive for DTA bc the bounds are slightly different
        #  for other tzs
        i8vals = np.asarray([NaT._value + n for n in range(1, 5)], dtype="i8")
        if self.array_cls is PeriodArray:
            arr = self.array_cls(i8vals, dtype="period[ns]")
        else:
            arr = self.index_cls(i8vals, freq="ns")._data
        arr[0]  # should not raise OutOfBoundsDatetime

        index = pd.Index(arr)
        index[0]  # should not raise OutOfBoundsDatetime

        ser = pd.Series(arr)
        ser[0]  # should not raise OutOfBoundsDatetime

    def test_getitem_2d(self, arr1d):
        # [FUNCTION DOC]
        # Verifies indexing a 1-d array with a new axis gives a 2-d array of the same type,
        # 2-d slicing and indexing works, and scalar lookup on the 2-d array equals the
        # corresponding 1-d element.
        # 2d slicing on a 1D array
        expected = type(arr1d)._simple_new(
            arr1d._ndarray[:, np.newaxis], dtype=arr1d.dtype
        )
        result = arr1d[:, np.newaxis]
        tm.assert_equal(result, expected)

        # Lookup on a 2D array
        arr2d = expected
        expected = type(arr2d)._simple_new(arr2d._ndarray[:3, 0], dtype=arr2d.dtype)
        result = arr2d[:3, 0]
        tm.assert_equal(result, expected)

        # Scalar lookup
        result = arr2d[-1, 0]
        expected = arr1d[-1]
        assert result == expected

    def test_iter_2d(self, arr1d):
        # [FUNCTION DOC]
        # Verifies iterating a 2-d array yields 1-d arrays of the same class and dtype,
        # one per row (three for a three-row array).
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)
        result = list(arr2d)
        assert len(result) == 3
        for x in result:
            assert isinstance(x, type(arr1d))
            assert x.ndim == 1
            assert x.dtype == arr1d.dtype

    def test_repr_2d(self, arr1d):
        # [FUNCTION DOC]
        # Verifies the repr of a 2-d array: class name, bracketed rows each listing the
        # element text (timedelta arrays use the base repr of the scalar), shape and dtype line.
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)

        result = repr(arr2d)

        if isinstance(arr2d, TimedeltaArray):
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]._repr_base()}'],\n"
                f"['{arr1d[1]._repr_base()}'],\n"
                f"['{arr1d[2]._repr_base()}']\n"
                "]\n"
                f"Shape: (3, 1), dtype: {arr1d.dtype}"
            )
        else:
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]}'],\n"
                f"['{arr1d[1]}'],\n"
                f"['{arr1d[2]}']\n"
                "]\n"
                f"Shape: (3, 1), dtype: {arr1d.dtype}"
            )

        assert result == expected

    def test_setitem(self):
        # [FUNCTION DOC]
        # Verifies setitem with an element and with a slice by comparing the integer
        # representation with the expected numpy modification.
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.index_cls(data, freq="D")._data

        arr[0] = arr[1]
        expected = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        expected[0] = expected[1]

        tm.assert_numpy_array_equal(arr.asi8, expected)

        arr[:2] = arr[-2:]
        expected[:2] = expected[-2:]
        tm.assert_numpy_array_equal(arr.asi8, expected)

    def test_setitem_list_of_nats(self, arr1d):
        # [FUNCTION DOC]
        # Regression test (GH#63420). Verifies assigning a list of NaT values over the
        # whole array makes all elements missing.
        # GH#63420
        arr1d[:] = [NaT] * len(arr1d)
        assert arr1d.isna().all()

    @pytest.mark.parametrize(
        "box",
        [
            pd.Index,
            pd.Series,
            np.array,
            list,
            NumpyExtensionArray,
        ],
    )
    def test_setitem_object_dtype(self, box, arr1d):
        # [FUNCTION DOC]
        # Parametrized over value container (Index, Series, numpy array, list,
        # NumpyExtensionArray). Assigns the reversed array as object-dtype values over a
        # slice and verifies the array equals the reversed array (frequency cleared when the
        # type has one).
        expected = arr1d.copy()[::-1]
        if expected.dtype.kind in ["m", "M"]:
            expected = expected._with_freq(None)

        vals = expected
        if box is list:
            vals = list(vals)
        elif box is np.array:
            # if we do np.array(x).astype(object) then dt64 and td64 cast to ints
            vals = np.array(vals.astype(object))
        elif box is NumpyExtensionArray:
            vals = box(np.asarray(vals, dtype=object))
        else:
            vals = box(vals).astype(object)

        arr1d[:] = vals

        tm.assert_equal(arr1d, expected)

    def test_setitem_strs(self, arr1d):
        # [FUNCTION DOC]
        # Verifies setitem parses strings in both list-like and scalar assignment,
        # giving the same result as assigning the parsed elements.
        # Check that we parse strs in both scalar and listlike

        # Setting list-like of strs
        expected = arr1d.copy()
        expected[[0, 1]] = arr1d[-2:]

        result = arr1d.copy()
        result[:2] = [str(x) for x in arr1d[-2:]]
        tm.assert_equal(result, expected)

        # Same thing but now for just a scalar str
        expected = arr1d.copy()
        expected[0] = arr1d[-1]

        result = arr1d.copy()
        result[0] = str(arr1d[-1])
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("as_index", [True, False])
    def test_setitem_categorical(self, arr1d, as_index):
        # [FUNCTION DOC]
        # Parametrized over as_index. Assigning a Categorical (or categorical index) of
        # the same values reversed gives the reversed array (frequency cleared for
        # non-period types).
        expected = arr1d.copy()[::-1]
        if not isinstance(expected, PeriodArray):
            expected = expected._with_freq(None)

        cat = pd.Categorical(arr1d)
        if as_index:
            cat = pd.CategoricalIndex(cat)

        arr1d[:] = cat[::-1]

        tm.assert_equal(arr1d, expected)

    def test_setitem_raises(self, arr1d):
        # [FUNCTION DOC]
        # Verifies setitem errors: out of bounds position raises an index error; assigning
        # a generic object raises a type error; assigning a mismatched length list-like or
        # slice raises value errors (GH#36339).
        arr = arr1d[:10]
        val = arr[0]

        with pytest.raises(IndexError, match="index 12 is out of bounds"):
            arr[12] = val

        with pytest.raises(TypeError, match="value should be a.* 'object'"):
            arr[0] = object()

        msg = "cannot set using a list-like indexer with a different length"
        with pytest.raises(ValueError, match=msg):
            # GH#36339
            arr[[]] = [arr[1]]

        msg = "cannot set using a slice indexer with a different length than"
        with pytest.raises(ValueError, match=msg):
            # GH#36339
            arr[1:1] = arr[:3]

    @pytest.mark.parametrize("box", [list, np.array, pd.Index, pd.Series])
    def test_setitem_numeric_raises(self, arr1d, box):
        # [FUNCTION DOC]
        # Parametrized over container (list, numpy array, Index, Series). Verifies
        # assigning integers or floats raises a type error (no casting of numbers into
        # the datetime-like dtype).
        # We dont case e.g. int64 to our own dtype for setitem

        msg = (
            f"value should be a '{arr1d._scalar_type.__name__}', "
            "'NaT', or array of those. Got"
        )
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0, 1])

        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0.0, 1.0])

    def test_inplace_arithmetic(self):
        # [FUNCTION DOC]
        # Regression test (GH#24115). Verifies in-place add and subtract of a one day
        # timedelta produce the same results as the non-inplace operation, applied to
        # the array itself.
        # GH#24115 check that iadd and isub are actually in-place
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.index_cls(data, freq="D")._data

        expected = arr + pd.Timedelta(days=1)
        arr += pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)

        expected = arr - pd.Timedelta(days=1)
        arr -= pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)

    def test_shift_fill_int_deprecated(self, arr1d):
        # [FUNCTION DOC]
        # Verifies shift with an integer fill value raises a type error (deprecation from
        # GH#31971 enforced in version 2.0).
        # GH#31971, enforced in 2.0
        with pytest.raises(TypeError, match="value should be a"):
            arr1d.shift(1, fill_value=1)

    def test_median(self, arr1d):
        # [FUNCTION DOC]
        # Verifies median: for an odd-length array equals the middle element with the
        # same type; with the middle set to NaT the median skipping NaT equals the mean of
        # the neighbors (the middle element for periods), the non-skipping median is NaT;
        # empty input gives NaT; the 2-d case gives the same with axis none, axis zero
        # (one-element array) and axis one (the original shape).
        arr = arr1d
        if len(arr) % 2 == 0:
            # make it easier to define `expected`
            arr = arr[:-1]

        expected = arr[len(arr) // 2]

        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected

        arr[len(arr) // 2] = NaT
        if not isinstance(expected, Period):
            expected = arr[len(arr) // 2 - 1 : len(arr) // 2 + 2].mean()

        assert arr.median(skipna=False) is NaT

        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected

        assert arr[:0].median() is NaT
        assert arr[:0].median(skipna=False) is NaT

        # 2d Case
        arr2 = arr.reshape(-1, 1)

        result = arr2.median(axis=None)
        assert type(result) is type(expected)
        assert result == expected

        assert arr2.median(axis=None, skipna=False) is NaT

        result = arr2.median(axis=0)
        expected2 = type(arr)._from_sequence([expected], dtype=arr.dtype)
        tm.assert_equal(result, expected2)

        result = arr2.median(axis=0, skipna=False)
        expected2 = type(arr)._from_sequence([NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected2)

        result = arr2.median(axis=1)
        tm.assert_equal(result, arr)

        result = arr2.median(axis=1, skipna=False)
        tm.assert_equal(result, arr)

    def test_from_integer_array(self):
        # [FUNCTION DOC]
        # Verifies constructing from a nullable Int64 array gives the same result as
        # constructing from the corresponding int64 numpy array (using the period
        # constructor for period arrays and the sequence constructor otherwise).
        arr = np.array([1, 2, 3], dtype=np.int64)
        data = pd.array(arr, dtype="Int64")
        if self.array_cls is PeriodArray:
            expected = self.array_cls(arr, dtype=self.example_dtype)
            result = self.array_cls(data, dtype=self.example_dtype)
        else:
            expected = self.array_cls._from_sequence(arr, dtype=self.example_dtype)
            result = self.array_cls._from_sequence(data, dtype=self.example_dtype)

        tm.assert_extension_array_equal(result, expected)


class TestDatetimeArray(SharedTests):
    # [CLASS DOC]
    # Tests for DatetimeArray: inherits the shared tests, sets index_cls to
    # DatetimeIndex, array_cls DatetimeArray, scalar type Timestamp and example
    # dtype M8[ns].
    index_cls = DatetimeIndex
    array_cls = DatetimeArray
    scalar_type = Timestamp
    example_dtype = "M8[ns]"

    @pytest.fixture
    def arr1d(self, tz_naive_fixture, freqstr):
        # [FUNCTION DOC]
        # HELPER: fixture. Returns the array of five timestamps starting 2016-01-01
        # 01:01:00 with nanosecond unit, frequency from freqstr and timezone from the
        # tz_naive_fixture fixture.
        """
        Fixture returning DatetimeArray with parametrized frequency and
        timezones
        """
        tz = tz_naive_fixture
        dti = pd.date_range(
            "2016-01-01 01:01:00", periods=5, freq=freqstr, tz=tz, unit="ns"
        )
        dta = dti._data
        return dta

    def test_round(self, arr1d):
        # [FUNCTION DOC]
        # Regression test (GH#24064). Verifies rounding to two minutes on the
        # index and on the array gives the same result (values one minute before the
        # original; frequency removed).
        # GH#24064
        dti = self.index_cls(arr1d)

        result = dti.round(freq="2min")
        expected = dti - pd.Timedelta(minutes=1)
        expected = expected._with_freq(None)
        tm.assert_index_equal(result, expected)

        dta = dti._data
        result = dta.round(freq="2min")
        expected = expected._data._with_freq(None)
        tm.assert_datetime_array_equal(result, expected)

    def test_array_interface(self, datetime_index):
        # [FUNCTION DOC]
        # Verifies the numpy array protocol for tz-naive datetime data: asarray and
        # array with copy false return the very same underlying numpy array, as does
        # requesting datetime64 nanoseconds (an extra copy is expected before numpy 2,
        # TODO GH 57739); object dtype gives boxed timestamps; integer dtype always
        # copies; float64 and str results equal numpy casting of the default result.
        arr = datetime_index._data
        copy_false = None if np_version_gt2 else False

        # default asarray gives the same underlying data (for tz naive)
        result = np.asarray(arr)
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)

        # specifying M8[ns] gives the same result as default
        result = np.asarray(arr, dtype="datetime64[ns]")
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="datetime64[ns]", copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="datetime64[ns]")
        if not np_version_gt2:
            # TODO: GH 57739
            assert result is not expected
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to other dtype always copies
        result = np.asarray(arr, dtype="int64")
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)

        # other dtypes handled by numpy
        for dtype in ["float64", str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_array_object_dtype(self, arr1d):
        # [FUNCTION DOC]
        # Regression test (GH#23524). Verifies converting to numpy with object dtype
        # gives timestamps (equal to the list of index elements) for both array and index.
        # GH#23524
        arr = arr1d
        dti = self.index_cls(arr1d)

        expected = np.array(list(dti))

        result = np.array(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # also test the DatetimeIndex method while we're at it
        result = np.array(dti, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_array_tz(self, arr1d):
        # [FUNCTION DOC]
        # Regression test (GH#23524). Verifies converting a (possibly tz-aware) array to
        # numpy datetime64 nanosecond equals the UTC values, and that with copy false the
        # result shares its base with the stored values (no copy).
        # GH#23524
        arr = arr1d
        dti = self.index_cls(arr1d, copy=False)
        copy_false = None if np_version_gt2 else False

        expected = dti.asi8.view("M8[ns]")
        result = np.array(arr, dtype="M8[ns]")
        tm.assert_numpy_array_equal(result, expected)

        result = np.array(arr, dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(result, expected)

        # check that we are not making copies when setting copy=copy_false
        result = np.array(arr, dtype="M8[ns]", copy=copy_false)
        assert result.base is expected.base
        assert result.base is not None
        result = np.array(arr, dtype="datetime64[ns]", copy=copy_false)
        assert result.base is expected.base
        assert result.base is not None

    def test_array_i8_dtype(self, arr1d):
        # [FUNCTION DOC]
        # Verifies converting to numpy int64 gives the integer representation, and that
        # integer conversion always copies even when copy false is passed (the result has no base).
        arr = arr1d
        dti = self.index_cls(arr1d)
        copy_false = None if np_version_gt2 else False

        expected = dti.asi8
        result = np.array(arr, dtype="i8")
        tm.assert_numpy_array_equal(result, expected)

        result = np.array(arr, dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        # check that we are still making copies when setting copy=copy_false
        result = np.array(arr, dtype="i8", copy=copy_false)
        assert result.base is not expected.base
        assert result.base is None

    def test_from_array_keeps_base(self):
        # [FUNCTION DOC]
        # Verifies construction from a numpy datetime64 array reuses the same
        # underlying array, and an empty slice keeps the base.
        # Ensure that DatetimeArray._ndarray.base isn't lost.
        arr = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        dta = DatetimeArray._from_sequence(arr, dtype=arr.dtype)

        assert dta._ndarray is arr
        dta = DatetimeArray._from_sequence(arr[:0], dtype=arr.dtype)
        assert dta._ndarray.base is arr

    def test_from_dti(self, arr1d):
        # [FUNCTION DOC]
        # Verifies an index built from the array has the same elements and that the
        # generic Index constructor on an array returns a DatetimeIndex with equal elements.
        arr = arr1d
        dti = self.index_cls(arr1d)
        assert list(dti) == list(arr)

        # Check that Index.__new__ knows what to do with DatetimeArray
        dti2 = pd.Index(arr)
        assert isinstance(dti2, DatetimeIndex)
        assert list(dti2) == list(arr)

    def test_astype_object(self, arr1d):
        # [FUNCTION DOC]
        # Verifies casting to object gives a numpy object array equal to the index elements.
        arr = arr1d
        dti = self.index_cls(arr1d)

        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(dti)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_to_period(self, datetime_index, freqstr):
        # [FUNCTION DOC]
        # Verifies to_period with a frequency gives a PeriodArray equal to that of the
        # corresponding index. Ignores the business-day period deprecation warning.
        dti = datetime_index
        arr = dti._data

        freqstr = PeriodDtype(to_offset(freqstr))._freqstr
        expected = dti.to_period(freq=freqstr)
        result = arr.to_period(freq=freqstr)
        assert isinstance(result, PeriodArray)

        tm.assert_equal(result, expected._data)

    def test_to_period_2d(self, arr1d):
        # [FUNCTION DOC]
        # Verifies to_period works on a 2-d array, equal to reshaping the 1-d result, and
        # emits a user warning about dropping timezone information for tz-aware arrays.
        arr2d = arr1d.reshape(1, -1)

        warn = None if arr1d.tz is None else UserWarning
        with tm.assert_produces_warning(warn, match="will drop timezone information"):
            result = arr2d.to_period("D")
            expected = arr1d.to_period("D").reshape(1, -1)
        tm.assert_period_array_equal(result, expected)

    @pytest.mark.parametrize("propname", DatetimeArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        # [FUNCTION DOC]
        # Parametrized over the boolean properties of the array (just the leap year
        # flag). Verifies the property equals that of the index and that the frequencies agree.
        # in this case _bool_ops is just `is_leap_year`
        dti = self.index_cls(arr1d)
        arr = arr1d
        assert dti.freq == arr.freq

        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("propname", DatetimeArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        # [FUNCTION DOC]
        # Parametrized over the integer field properties of the array. Verifies each
        # equals the same property of the index.
        dti = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, arr1d, fixed_now_ts):
        # [FUNCTION DOC]
        # Verifies valid fills (a timestamp localized to the array timezone), and that
        # invalid fills raise type errors: a timedelta, a period, a timestamp with
        # mismatched tz-awareness (compare error), a raw integer NaT value, and a
        # timedelta64 NaT. For tz-aware arrays, a timestamp in another timezone is
        # accepted and converted (GH#37356).
        arr = arr1d
        dti = self.index_cls(arr1d)

        now = fixed_now_ts.tz_localize(dti.tz)
        result = arr.take([-1, 1], allow_fill=True, fill_value=now)
        assert result[0] == now

        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # fill_value Timedelta invalid
            arr.take([-1, 1], allow_fill=True, fill_value=now - now)

        with pytest.raises(TypeError, match=msg):
            # fill_value Period invalid
            arr.take([-1, 1], allow_fill=True, fill_value=Period("2014Q1"))

        tz = None if dti.tz is not None else "US/Eastern"
        now = fixed_now_ts.tz_localize(tz)
        msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            # Timestamp with mismatched tz-awareness
            arr.take([-1, 1], allow_fill=True, fill_value=now)

        value = NaT._value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # require NaT, not iNaT, as it could be confused with an integer
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        value = np.timedelta64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have an NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        if arr.tz is not None:
            # GH#37356
            # Assuming here that arr1d fixture does not include Australia/Melbourne
            value = fixed_now_ts.tz_localize("Australia/Melbourne")
            result = arr.take([-1, 1], allow_fill=True, fill_value=value)

            expected = arr.take(
                [-1, 1],
                allow_fill=True,
                fill_value=value.tz_convert(arr.dtype.tz),
            )
            tm.assert_equal(result, expected)

    def test_concat_same_type_invalid(self, arr1d):
        # [FUNCTION DOC]
        # Verifies concatenating arrays of different timezone awareness raises a value
        # error ("to_concat must have the same ...").
        # different timezones
        arr = arr1d

        if arr.tz is None:
            other = arr.tz_localize("UTC")
        else:
            other = arr.tz_localize(None)

        with pytest.raises(ValueError, match="to_concat must have the same"):
            arr._concat_same_type([arr, other])

    def test_concat_same_type_different_freq(self, unit):
        # [FUNCTION DOC]
        # Uses the unit fixture. Verifies concatenating tz-aware arrays with different
        # frequencies works and gives the expected timestamps and unit.
        # we *can* concatenate DTI with different freqs.
        a = pd.date_range("2000", periods=2, freq="D", tz="US/Central", unit=unit)._data
        b = pd.date_range("2000", periods=2, freq="h", tz="US/Central", unit=unit)._data
        result = DatetimeArray._concat_same_type([a, b])
        expected = (
            pd.to_datetime(
                [
                    "2000-01-01 00:00:00",
                    "2000-01-02 00:00:00",
                    "2000-01-01 00:00:00",
                    "2000-01-01 01:00:00",
                ]
            )
            .tz_localize("US/Central")
            .as_unit(unit)
            ._data
        )

        tm.assert_datetime_array_equal(result, expected)

    def test_strftime(self, arr1d, using_infer_string):
        # [FUNCTION DOC]
        # Verifies strftime with a year-month-name format equals the formatted element
        # strings; the expected values are string dtype when infer-string mode is on.
        arr = arr1d

        result = arr.strftime("%Y %b")
        expected = np.array([ts.strftime("%Y %b") for ts in arr], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)

    def test_strftime_nat(self, using_infer_string):
        # [FUNCTION DOC]
        # Regression test (GH 29578). Verifies strftime on an array containing NaT gives
        # a missing value for the NaT position.
        # GH 29578
        arr = DatetimeIndex(["2019-01-01", NaT])._data

        result = arr.strftime("%Y-%m-%d")
        expected = np.array(["2019-01-01", np.nan], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)


class TestTimedeltaArray(SharedTests):
    # [CLASS DOC]
    # Tests for TimedeltaArray: inherits the shared tests with index_cls
    # TimedeltaIndex, array_cls TimedeltaArray, scalar type Timedelta, example dtype
    # m8[ns].
    index_cls = TimedeltaIndex
    array_cls = TimedeltaArray
    scalar_type = pd.Timedelta
    example_dtype = "m8[ns]"

    def test_from_tdi(self):
        # [FUNCTION DOC]
        # Verifies an index and the generic Index constructor interoperate with a
        # TimedeltaArray: elements equal and the Index constructor returns a TimedeltaIndex.
        tdi = TimedeltaIndex(["1 Day", "3 Hours"])
        arr = tdi._data
        assert list(arr) == list(tdi)

        # Check that Index.__new__ knows what to do with TimedeltaArray
        tdi2 = pd.Index(arr)
        assert isinstance(tdi2, TimedeltaIndex)
        assert list(tdi2) == list(arr)

    def test_astype_object(self):
        # [FUNCTION DOC]
        # Verifies casting to object gives a numpy object array equal to the index elements.
        tdi = TimedeltaIndex(["1 Day", "3 Hours"])
        arr = tdi._data
        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(tdi)

    def test_to_pytimedelta(self, timedelta_index):
        # [FUNCTION DOC]
        # Verifies to_pytimedelta of the array equals the index method result.
        tdi = timedelta_index
        arr = tdi._data

        expected = tdi.to_pytimedelta()
        result = arr.to_pytimedelta()

        tm.assert_numpy_array_equal(result, expected)

    def test_total_seconds(self, timedelta_index):
        # [FUNCTION DOC]
        # Verifies total_seconds of the array equals the values of the index result.
        tdi = timedelta_index
        arr = tdi._data

        expected = tdi.total_seconds()
        result = arr.total_seconds()

        tm.assert_numpy_array_equal(result, expected.values)

    @pytest.mark.parametrize("propname", TimedeltaArray._field_ops)
    def test_int_properties(self, timedelta_index, propname):
        # [FUNCTION DOC]
        # Parametrized over the array integer field properties. Verifies each equals
        # the index property.
        tdi = timedelta_index
        arr = tdi._data

        result = getattr(arr, propname)
        expected = np.array(getattr(tdi, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, timedelta_index):
        # [FUNCTION DOC]
        # Verifies the numpy array protocol for timedelta data (same checks as the
        # datetime variant, with the timedelta64 microsecond unit): the default conversion
        # returns the underlying array without copying, object conversion gives
        # timedeltas, integer conversion copies, float64 and str follow numpy.
        arr = timedelta_index._data
        copy_false = None if np_version_gt2 else False

        # default asarray gives the same underlying data
        result = np.asarray(arr)
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)

        # specifying m8[us] gives the same result as default
        result = np.asarray(arr, dtype="timedelta64[us]")
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="timedelta64[us]", copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="timedelta64[us]")
        if not np_version_gt2:
            # TODO: GH 57739
            assert result is not expected
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to other dtype always copies
        result = np.asarray(arr, dtype="int64")
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)

        # other dtypes handled by numpy
        for dtype in ["float64", str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, timedelta_index, fixed_now_ts):
        # [FUNCTION DOC]
        # Verifies take accepts a timedelta fill value and raises type errors for a
        # timestamp, a period, and a datetime64 NaT.
        tdi = timedelta_index
        arr = tdi._data

        td1 = pd.Timedelta(days=1)
        result = arr.take([-1, 1], allow_fill=True, fill_value=td1)
        assert result[0] == td1

        value = fixed_now_ts
        msg = f"value should be a '{arr._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # fill_value Timestamp invalid
            arr.take([0, 1], allow_fill=True, fill_value=value)

        value = fixed_now_ts.to_period("D")
        with pytest.raises(TypeError, match=msg):
            # fill_value Period invalid
            arr.take([0, 1], allow_fill=True, fill_value=value)

        value = np.datetime64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have an NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)


@pytest.mark.filterwarnings(r"ignore:Period with BDay freq is deprecated:FutureWarning")
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
class TestPeriodArray(SharedTests):
    # [CLASS DOC]
    # Tests for PeriodArray: inherits the shared tests with index_cls PeriodIndex,
    # array_cls PeriodArray, scalar type Period, and example dtype taken from a weekly
    # empty PeriodIndex. Business-day period deprecation warnings are ignored for the class.
    index_cls = PeriodIndex
    array_cls = PeriodArray
    scalar_type = Period
    example_dtype = PeriodIndex([], freq="W").dtype

    @pytest.fixture
    def arr1d(self, period_index):
        # [FUNCTION DOC]
        # HELPER: fixture. Returns the PeriodArray underlying the period_index fixture.
        # KNOWN ISSUE: the docstring says it returns a DatetimeArray, but the value
        # is a PeriodArray.
        """
        Fixture returning DatetimeArray from parametrized PeriodIndex objects
        """
        return period_index._data

    def test_from_pi(self, arr1d):
        # [FUNCTION DOC]
        # Verifies an index built from the array has the same elements and that the
        # generic Index constructor on the array returns a PeriodIndex.
        pi = self.index_cls(arr1d)
        arr = arr1d
        assert list(arr) == list(pi)

        # Check that Index.__new__ knows what to do with PeriodArray
        pi2 = pd.Index(arr)
        assert isinstance(pi2, PeriodIndex)
        assert list(pi2) == list(arr)

    def test_astype_object(self, arr1d):
        # [FUNCTION DOC]
        # Verifies casting to object gives a numpy object array equal to the index elements.
        pi = self.index_cls(arr1d)
        arr = arr1d
        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(pi)

    def test_take_fill_valid(self, arr1d):
        # [FUNCTION DOC]
        # Verifies take rejects (type error) a raw integer NaT value and a timedelta64 NaT
        # as fill values.
        arr = arr1d

        value = NaT._value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # require NaT, not iNaT, as it could be confused with an integer
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        value = np.timedelta64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have an NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)

    @pytest.mark.parametrize("how", ["S", "E"])
    def test_to_timestamp(self, how, arr1d):
        # [FUNCTION DOC]
        # Parametrized over how (S start, E end). Verifies to_timestamp on the array
        # equals the index result and is a DatetimeArray.
        pi = self.index_cls(arr1d)
        arr = arr1d

        expected = DatetimeIndex(pi.to_timestamp(how=how))._data
        result = arr.to_timestamp(how=how)
        assert isinstance(result, DatetimeArray)

        tm.assert_equal(result, expected)

    def test_to_timestamp_roundtrip_bday(self):
        # [FUNCTION DOC]
        # Verifies a business-day datetime array converted to periods and back keeps
        # frequency B (and 2B for every second element, and for an explicit 2B period
        # frequency back to B), with values equal to the original as microsecond unit.
        # Case where infer_freq inside would choose "D" instead of "B"
        dta = pd.date_range("2021-10-18", periods=3, freq="B", unit="ns")._data
        parr = dta.to_period()
        result = parr.to_timestamp()
        assert result.freq == "B"
        tm.assert_extension_array_equal(result, dta.as_unit("us"))

        dta2 = dta[::2]
        parr2 = dta2.to_period()
        result2 = parr2.to_timestamp()
        assert result2.freq == "2B"
        tm.assert_extension_array_equal(result2, dta2.as_unit("us"))

        parr3 = dta.to_period("2B")
        result3 = parr3.to_timestamp()
        assert result3.freq == "B"
        tm.assert_extension_array_equal(result3, dta.as_unit("us"))

    def test_to_timestamp_out_of_bounds(self):
        # [FUNCTION DOC]
        # Regression test (GH#19643). Verifies converting year 1500 periods to timestamps
        # does not overflow silently; the first value is year 1500.
        # GH#19643 previously overflowed silently
        pi = pd.period_range("1500", freq="Y", periods=3)
        pi.to_timestamp()
        dta = pi._data.to_timestamp()
        assert dta[0] == Timestamp(1500, 1, 1)

    @pytest.mark.parametrize("propname", PeriodArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        # [FUNCTION DOC]
        # Parametrized over the period boolean properties (leap year flag). Verifies
        # equality with the index property.
        # in this case _bool_ops is just `is_leap_year`
        pi = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("propname", PeriodArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        # [FUNCTION DOC]
        # Parametrized over the period integer field properties. Verifies equality with
        # the index property.
        pi = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))

        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, arr1d):
        # [FUNCTION DOC]
        # Verifies the numpy array protocol for periods: default and object give boxed
        # Period objects, integer dtype gives the ordinal values (sharing memory between
        # calls without copy; copy true produces independent copies), float64 raises a
        # type error, and bytes strings work as in numpy.
        arr = arr1d

        # default asarray gives objects
        result = np.asarray(arr)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype (same as default)
        result = np.asarray(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to int64 gives the underlying representation
        result = np.asarray(arr, dtype="int64")
        tm.assert_numpy_array_equal(result, arr.asi8)

        result2 = np.asarray(arr, dtype="int64")
        assert np.may_share_memory(result, result2)

        result_copy1 = np.array(arr, dtype="int64", copy=True)
        result_copy2 = np.array(arr, dtype="int64", copy=True)
        assert not np.may_share_memory(result_copy1, result_copy2)

        # to other dtypes
        msg = r"float\(\) argument must be a string or a( real)? number, not 'Period'"
        with pytest.raises(TypeError, match=msg):
            np.asarray(arr, dtype="float64")

        result = np.asarray(arr, dtype="S20")
        expected = np.asarray(arr).astype("S20")
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime(self, arr1d, using_infer_string):
        # [FUNCTION DOC]
        # Verifies strftime with a year format equals the formatted element strings; string
        # dtype when infer-string mode is on.
        arr = arr1d

        result = arr.strftime("%Y")
        expected = np.array([per.strftime("%Y") for per in arr], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)

    def test_strftime_nat(self, using_infer_string):
        # [FUNCTION DOC]
        # Regression test (GH 29578). Verifies strftime on a period array with a
        # NaT gives a missing value for that position.
        # GH 29578
        arr = PeriodArray(PeriodIndex(["2019-01-01", NaT], dtype="period[D]"))

        result = arr.strftime("%Y-%m-%d")
        expected = np.array(["2019-01-01", np.nan], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "arr,casting_nats",
    [
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (NaT, np.timedelta64("NaT", "ns")),
        ),
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (NaT, np.datetime64("NaT", "ns")),
        ),
        (pd.period_range("2000-01-01", periods=3, freq="D")._data, (NaT,)),
    ],
    ids=lambda x: type(x).__name__,
)
def test_casting_nat_setitem_array(arr, casting_nats):
    # [FUNCTION DOC]
    # Parametrized over timedelta / datetime / period arrays and the NaT-like values
    # that can be cast. Verifies that setting the first element to each NaT-like
    # value (NaT and the matching numpy NaT) gives an array equal to one built with NaT.
    expected = type(arr)._from_sequence([NaT, arr[1], arr[2]], dtype=arr.dtype)

    for nat in casting_nats:
        arr = arr.copy()
        arr[0] = nat
        tm.assert_equal(arr, expected)


@pytest.mark.parametrize(
    "arr,non_casting_nats",
    [
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (np.datetime64("NaT", "ns"), NaT._value),
        ),
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (np.timedelta64("NaT", "ns"), NaT._value),
        ),
        (
            pd.period_range("2000-01-01", periods=3, freq="D")._data,
            (np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns"), NaT._value),
        ),
    ],
    ids=lambda x: type(x).__name__,
)
def test_invalid_nat_setitem_array(arr, non_casting_nats):
    # [FUNCTION DOC]
    # Parametrized over the three array kinds and NaT-like values of the wrong type
    # (a different numpy NaT or the raw integer NaT value). Verifies setting the first
    # element raises a type error with a message naming the expected scalar type.
    msg = (
        "value should be a '(Timestamp|Timedelta|Period)', 'NaT', or array of those. "
        "Got '(timedelta64|datetime64|int)' instead."
    )

    for nat in non_casting_nats:
        with pytest.raises(TypeError, match=msg):
            arr[0] = nat


@pytest.mark.parametrize(
    "arr",
    [
        pd.date_range("2000", periods=4)._values,
        pd.timedelta_range("2000", periods=4)._values,
    ],
)
def test_to_numpy_extra(arr):
    # [FUNCTION DOC]
    # Parametrized over datetime and timedelta arrays. Sets one element to NaT and
    # verifies to_numpy: default gives NaN at that position, integer dtype gives the
    # minimum 64-bit integer, integer with na_value zero gives zero, and
    # na_value set to a numpy scalar of another element fills the NaT; the array is
    # not changed.
    arr[0] = NaT
    original = arr.copy()

    result = arr.to_numpy()
    assert np.isnan(result[0])

    result = arr.to_numpy(dtype="int64")
    assert result[0] == -9223372036854775808

    result = arr.to_numpy(dtype="int64", na_value=0)
    assert result[0] == 0

    result = arr.to_numpy(na_value=arr[1].to_numpy())
    assert result[0] == result[1]

    result = arr.to_numpy(na_value=arr[1].to_numpy(copy=False))
    assert result[0] == result[1]

    tm.assert_equal(arr, original)


@pytest.mark.parametrize(
    "arr",
    [
        pd.date_range("2000", periods=4)._values,
        pd.timedelta_range("2000", periods=4)._values,
    ],
)
def test_to_numpy_extra_readonly(arr):
    # [FUNCTION DOC]
    # Parametrized over datetime and timedelta arrays. Sets the private read-only flag
    # and verifies to_numpy with object dtype and with int64 dtype each give a writeable
    # result; the array is unchanged.
    arr[0] = NaT
    original = arr.copy()
    arr._readonly = True

    result = arr.to_numpy(dtype=object)
    assert result.flags.writeable

    # numpy does not do zero-copy conversion from M8 to i8
    result = arr.to_numpy(dtype="int64")
    assert result.flags.writeable

    tm.assert_equal(arr, original)


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "klass",
    [
        list,
        np.array,
        pd.array,
        pd.Series,
        pd.Index,
        pd.Categorical,
        pd.CategoricalIndex,
    ],
)
def test_searchsorted_datetimelike_with_listlike(values, klass, as_index):
    # [FUNCTION DOC]
    # Regression test (GitHub 32762). Parametrized over the container of the search
    # values (list, numpy array, array, Series, Index, Categorical, categorical index),
    # the values (datetime, timedelta, period) and whether the searched object is the index or the
    # underlying array. Verifies searchsorted of the values with themselves gives
    # positions zero and one.
    # https://github.com/pandas-dev/pandas/issues/32762
    if not as_index:
        values = values._data

    result = values.searchsorted(klass(values))
    expected = np.array([0, 1], dtype=result.dtype)

    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
)
def test_searchsorted_datetimelike_with_listlike_invalid_dtype(values, arg):
    # [FUNCTION DOC]
    # Regression test (GitHub 32762). Parametrized over datetime, timedelta and period
    # values and invalid search arguments (integers, strings, tz-aware timestamps).
    # Verifies searchsorted raises a type error.
    # KNOWN ISSUE: the expected message pattern uses square brackets, which form a
    # character class rather than an alternation, so the match is looser than the
    # author likely intended.
    # https://github.com/pandas-dev/pandas/issues/32762
    msg = "[Unexpected type|Cannot compare]"
    with pytest.raises(TypeError, match=msg):
        values.searchsorted(arg)


@pytest.mark.parametrize("klass", [list, tuple, np.array, pd.Series])
def test_period_index_construction_from_strings(klass):
    # [FUNCTION DOC]
    # Regression test (GitHub 26109). Parametrized over input container (list,
    # tuple, array, Series). Verifies building a quarterly PeriodIndex from the
    # container of strings equals the one built from Period objects.
    # https://github.com/pandas-dev/pandas/issues/26109
    strings = ["2020Q1", "2020Q2"] * 2
    data = klass(strings)
    result = PeriodIndex(data, freq="Q")
    expected = PeriodIndex([Period(s) for s in strings])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
def test_from_pandas_array(dtype):
    # [FUNCTION DOC]
    # Regression test (GH#24615). Parametrized over datetime64 and timedelta64 nanosecond
    # dtype. Verifies wrapping the numpy values in a NumpyExtensionArray and passing it to
    # the array sequence constructor, to to_datetime or to_timedelta and to the
    # index constructor gives the same results as passing the raw numpy array.
    # GH#24615
    data = np.array([1, 2, 3], dtype=dtype)
    arr = NumpyExtensionArray(data)

    cls = {"M8[ns]": DatetimeArray, "m8[ns]": TimedeltaArray}[dtype]

    result = cls._from_sequence(arr, dtype=dtype)
    expected = cls._from_sequence(data, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    func = {"M8[ns]": pd.to_datetime, "m8[ns]": pd.to_timedelta}[dtype]
    result = func(arr).array
    expected = func(data).array
    tm.assert_equal(result, expected)

    # Let's check the Indexes while we're here
    idx_cls = {"M8[ns]": DatetimeIndex, "m8[ns]": TimedeltaIndex}[dtype]
    result = idx_cls(arr)
    expected = idx_cls(data)
    tm.assert_index_equal(result, expected)
