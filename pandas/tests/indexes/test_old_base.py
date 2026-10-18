# [MODULE DOC]
# Legacy base tests for Index behavior that were historically shared by many index subclasses: construction errors, name inheritance, numeric operator rejection, logical reductions, representation, copy semantics, memory usage, argsort, repeat, where, insert, delete, equality, missing-value methods, joins, mapping, casting to categorical, uniqueness, caching behavior, groupby, append, and inversion. Two classes: one with a fixture spanning range, float, unsigned, signed integer, string, mixed, period, timedelta, datetime and interval indexes, and one restricted to numeric indexes. Also uses the shared index fixture for tests that should run on every index type.

from __future__ import annotations

from datetime import datetime
import weakref

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas.compat import PY315
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    DatetimeTZDtype,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    StringDtype,
    TimedeltaIndex,
    isna,
    period_range,
)
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import BaseMaskedArray


class TestBase:
    # [CLASS DOC]
    # General legacy tests for many index types; provides a parametrized fixture.
    @pytest.fixture(
        params=[
            RangeIndex(start=0, stop=20, step=2),
            Index(np.arange(5, dtype=np.float64)),
            Index(np.arange(5, dtype=np.float32)),
            Index(np.arange(5, dtype=np.uint64)),
            Index(range(0, 20, 2), dtype=np.int64),
            Index(range(0, 20, 2), dtype=np.int32),
            Index(range(0, 20, 2), dtype=np.int16),
            Index(range(0, 20, 2), dtype=np.int8),
            Index(list("abcde")),
            Index([0, "a", 1, "b", 2, "c"]),
            period_range("20130101", periods=5, freq="D"),
            TimedeltaIndex(
                [
                    "0 days 01:00:00",
                    "1 days 01:00:00",
                    "2 days 01:00:00",
                    "3 days 01:00:00",
                    "4 days 01:00:00",
                ],
                dtype="timedelta64[ns]",
                freq="D",
            ),
            DatetimeIndex(
                ["2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05"],
                dtype="datetime64[ns]",
                freq="D",
            ),
            IntervalIndex.from_breaks(range(11), closed="right"),
        ]
    )
    def simple_index(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over a range index, float64/float32/uint64 and signed integer indexes (64 to 8 bit), a string index, a mixed integer/string index, a daily period range, a timedelta index and a datetime index (each with daily frequency), and an interval index with right-closed breaks from zero to ten.
        return request.param

    def test_pickle_compat_construction(self, simple_index):
        # [FUNCTION DOC]
        # Skips range indexes. Constructing the index class with no data argument raises a type error whose message is one of several variants (collection required, missing positional data argument, not iterable).
        # need an object to create with
        if isinstance(simple_index, RangeIndex):
            pytest.skip("RangeIndex() is a valid constructor")
        msg = "|".join(
            [
                r"Index\(\.\.\.\) must be called with a collection of some "
                r"kind, None was passed",
                r"DatetimeIndex\(\) must be called with a collection of some "
                r"kind, None was passed",
                r"TimedeltaIndex\(\) must be called with a collection of some "
                r"kind, None was passed",
                r"__new__\(\) missing 1 required positional argument: 'data'",
                r"__new__\(\) takes at least 2 arguments \(1 given\)",
                r"'NoneType' object is not iterable",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            type(simple_index)()

    def test_shift(self, simple_index):
        # [FUNCTION DOC]
        # Skips datetime, timedelta and period indexes. Shifting any other index (with one or two arguments) raises a not-implemented error saying the method is only implemented for datetime, period and timedelta indexes and naming the received type.
        # GH8083 test the base class for shift
        if isinstance(simple_index, (DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested in test_ops/test_arithmetic")
        idx = simple_index
        msg = (
            f"This method is only implemented for DatetimeIndex, PeriodIndex and "
            f"TimedeltaIndex; Got type {type(idx).__name__}"
        )
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1)
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1, 2)

    def test_constructor_name_unhashable(self, simple_index):
        # [FUNCTION DOC]
        # Constructing the same class from an index with an unhashable name (empty list) raises a type error saying the index name must be hashable.
        # GH#29069 check that name is hashable
        # See also same-named test in tests.series.test_constructors
        idx = simple_index
        with pytest.raises(TypeError, match="Index.name must be a hashable type"):
            type(idx)(idx, name=[])

    def test_create_index_existing_name(self, simple_index):
        # [FUNCTION DOC]
        # An index built from another inherits its name; an explicit name overrides it. For multi-level indexes (names set to foo and bar), the result is checked against an index of tuples with those names, or with explicit new names.
        # GH11193, when an existing index is passed, and a new name is not
        # specified, the new index should inherit the previous object name
        expected = simple_index.copy()
        if not isinstance(expected, MultiIndex):
            expected.name = "foo"
            result = Index(expected)
            tm.assert_index_equal(result, expected)

            result = Index(expected, name="bar")
            expected.name = "bar"
            tm.assert_index_equal(result, expected)
        else:
            expected.names = ["foo", "bar"]
            result = Index(expected)
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["foo", "bar"],
                ),
            )

            result = Index(expected, names=["A", "B"])
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["A", "B"],
                ),
            )

    def test_numeric_compat(self, simple_index):
        # [FUNCTION DOC]
        # Asserts the index is not multi-level; skips the plain base class, numeric dtypes and timedelta indexes. For remaining types (period, datetime, interval, string-like), multiplication, true division and floor division by an integer, on either side, each raise a type error whose message names unsupported operand types or "cannot perform" the operation with this index type.
        idx = simple_index
        # Check that this doesn't cover MultiIndex case, if/when it does,
        #  we can remove multi.test_compat.test_numeric_compat
        assert not isinstance(idx, MultiIndex)
        if type(idx) is Index:
            pytest.skip("Not applicable for Index")
        if is_numeric_dtype(simple_index.dtype) or isinstance(
            simple_index, TimedeltaIndex
        ):
            pytest.skip("Tested elsewhere.")

        typ = type(idx._data).__name__
        cls = type(idx).__name__
        lmsg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: '{typ}' and 'int'",
                "cannot perform (__mul__|__truediv__|__floordiv__) with "
                f"this index type: ({cls}|{typ})",
            ]
        )
        with pytest.raises(TypeError, match=lmsg):
            idx * 1
        rmsg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: 'int' and '{typ}'",
                "cannot perform (__rmul__|__rtruediv__|__rfloordiv__) with "
                f"this index type: ({cls}|{typ})",
            ]
        )
        with pytest.raises(TypeError, match=rmsg):
            1 * idx

        div_err = lmsg.replace("*", "/")
        with pytest.raises(TypeError, match=div_err):
            idx / 1
        div_err = rmsg.replace("*", "/")
        with pytest.raises(TypeError, match=div_err):
            1 / idx

        floordiv_err = lmsg.replace("*", "//")
        with pytest.raises(TypeError, match=floordiv_err):
            idx // 1
        floordiv_err = rmsg.replace("*", "//")
        with pytest.raises(TypeError, match=floordiv_err):
            1 // idx

    def test_logical_compat(self, simple_index):
        # [FUNCTION DOC]
        # Skips object and string dtypes. For integer, unsigned, float, complex, bool and timedelta dtype kinds all and any equal the same reductions on the values and on the series. Other dtypes raise a type error saying the dtype does not support the operation.
        if simple_index.dtype in (object, "string"):
            pytest.skip("Tested elsewhere.")
        idx = simple_index
        if idx.dtype.kind in "iufcbm":
            assert idx.all() == idx._values.all()
            assert idx.all() == idx.to_series().all()
            assert idx.any() == idx._values.any()
            assert idx.any() == idx.to_series().any()
        else:
            msg = "does not support operation '(any|all)'"
            with pytest.raises(TypeError, match=msg):
                idx.all()
            with pytest.raises(TypeError, match=msg):
                idx.any()

    def test_repr_roundtrip(self, simple_index):
        # [FUNCTION DOC]
        # Skips interval indexes (no valid repr). Evaluating the text representation gives an equal index.
        if isinstance(simple_index, IntervalIndex):
            pytest.skip(f"Not a valid repr for {type(simple_index).__name__}")
        idx = simple_index
        tm.assert_index_equal(eval(repr(idx)), idx)

    def test_repr_max_seq_item_setting(self, simple_index):
        # [FUNCTION DOC]
        # Skips interval indexes. With the display option for maximum sequence items unset, the text of a fifty-fold repeated index contains no ellipsis.
        # GH10182
        if isinstance(simple_index, IntervalIndex):
            pytest.skip(f"Not a valid repr for {type(simple_index).__name__}")
        idx = simple_index
        idx = idx.repeat(50)
        with pd.option_context("display.max_seq_items", None):
            repr(idx)
            assert "..." not in str(idx)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_ensure_copied_data(self, index):
        # [FUNCTION DOC]
        # Uses the shared index fixture; ignores a period deprecation warning. Skips range, multi-level, categorical. Checks the copy argument of each constructor: building from the values with copy true equals the original (after adjusting timezone and removing frequency for datetime-like). Then: period indexes built from ordinals share memory with the original; interval indexes pass; extension-array-backed base indexes built without copy share memory (masked array data and mask, python-storage string ndarray, or Arrow-backed memory sharing) and raise not-implemented for other extension dtypes; everything else shares the values array when copy is false.
        # Check the "copy" argument of each Index.__new__ is honoured
        # GH12309
        init_kwargs = {}
        if isinstance(index, PeriodIndex):
            # Needs "freq" specification:
            init_kwargs["freq"] = index.freq
        elif isinstance(index, (RangeIndex, MultiIndex, CategoricalIndex)):
            pytest.skip(
                "RangeIndex cannot be initialized from data, "
                "MultiIndex and CategoricalIndex are tested separately"
            )
        elif index.dtype == object and index.inferred_type in ["boolean", "string"]:
            init_kwargs["dtype"] = index.dtype

        index_type = type(index)
        result = index_type(index.values, copy=True, **init_kwargs)
        if isinstance(index.dtype, DatetimeTZDtype):
            result = result.tz_localize("UTC").tz_convert(index.tz)
        if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
            index = index._with_freq(None)

        tm.assert_index_equal(index, result)

        if isinstance(index, PeriodIndex):
            # .values an object array of Period, thus copied
            result = index_type.from_ordinals(ordinals=index.asi8, **init_kwargs)
            tm.assert_numpy_array_equal(index.asi8, result.asi8, check_same="same")
        elif isinstance(index, IntervalIndex):
            # checked in test_interval.py
            pass
        elif type(index) is Index and not isinstance(index.dtype, np.dtype):
            result = index_type(index.values, copy=False, **init_kwargs)
            tm.assert_index_equal(result, index)

            if isinstance(index._values, BaseMaskedArray):
                assert np.shares_memory(index._values._data, result._values._data)
                tm.assert_numpy_array_equal(
                    index._values._data, result._values._data, check_same="same"
                )
                assert np.shares_memory(index._values._mask, result._values._mask)
                tm.assert_numpy_array_equal(
                    index._values._mask, result._values._mask, check_same="same"
                )
            elif (
                isinstance(index.dtype, StringDtype) and index.dtype.storage == "python"
            ):
                assert np.shares_memory(index._values._ndarray, result._values._ndarray)
                tm.assert_numpy_array_equal(
                    index._values._ndarray, result._values._ndarray, check_same="same"
                )
            elif (
                isinstance(index.dtype, StringDtype)
                and index.dtype.storage == "pyarrow"
            ):
                assert tm.shares_memory(result._values, index._values)
            else:
                raise NotImplementedError(index.dtype)
        else:
            result = index_type(index.values, copy=False, **init_kwargs)
            tm.assert_numpy_array_equal(index.values, result.values, check_same="same")

    def test_memory_usage(self, index):
        # [FUNCTION DOC]
        # Uses the shared index fixture. After clearing the engine mapping, an empty index reports zero memory. Otherwise after a location lookup (which builds the engine) the memory use grows, except for range, interval and extension-array indexes which have no hash table. For object-inferred indexes the deep measurement exceeds the shallow one.
        index._engine.clear_mapping()
        result = index.memory_usage()
        if index.empty:
            # we report 0 for no-length
            assert result == 0
            return

        # non-zero length
        index.get_loc(index[0])
        result2 = index.memory_usage()
        result3 = index.memory_usage(deep=True)

        # RangeIndex, IntervalIndex
        # don't have engines
        # Index[EA] has engine but it does not have a Hashtable .mapping
        if not isinstance(index, (RangeIndex, IntervalIndex)) and not (
            type(index) is Index and not isinstance(index.dtype, np.dtype)
        ):
            assert result2 > result

        if index.inferred_type == "object":
            assert result3 > result2

    def test_memory_usage_doesnt_trigger_engine(self, index):
        # [FUNCTION DOC]
        # Clearing the cache and requesting memory usage must not create the engine; after explicitly creating the engine the reported usage is unchanged; usage is zero for empty indexes and positive otherwise.
        index._cache.clear()
        assert "_engine" not in index._cache

        res_without_engine = index.memory_usage()
        assert "_engine" not in index._cache

        # explicitly load and cache the engine
        _ = index._engine
        assert "_engine" in index._cache

        res_with_engine = index.memory_usage()

        # the empty engine doesn't affect the result even when initialized with values,
        # because engine.sizeof() doesn't consider the content of engine.values
        assert res_with_engine == res_without_engine

        if len(index) == 0:
            assert res_without_engine == 0
            assert res_with_engine == 0
        else:
            assert res_without_engine > 0
            assert res_with_engine > 0

    def test_argsort(self, index):
        # [FUNCTION DOC]
        # Skips categorical indexes. The index sort order equals numpy's argsort of the array form (dtype of the result not compared).
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"{type(self).__name__} separately tested")

        result = index.argsort()
        expected = np.array(index).argsort()
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_numpy_argsort(self, index):
        # [FUNCTION DOC]
        # The numpy function applied to an index equals the index method, including with the merge-sort kind. For categorical and range indexes only, the axis and order parameters raise value errors saying they are not supported.
        result = np.argsort(index)
        expected = index.argsort()
        tm.assert_numpy_array_equal(result, expected)

        result = np.argsort(index, kind="mergesort")
        expected = index.argsort(kind="mergesort")
        tm.assert_numpy_array_equal(result, expected)

        # these are the only two types that perform
        # pandas compatibility input validation - the
        # rest already perform separate (or no) such
        # validation via their 'values' attribute as
        # defined in pandas.core.indexes/base.py - they
        # cannot be changed at the moment due to
        # backwards compatibility concerns
        if isinstance(index, (CategoricalIndex, RangeIndex)):
            msg = "the 'axis' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, axis=1)

            msg = "the 'order' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, order=("a", "b"))

    def test_repeat(self, simple_index):
        # [FUNCTION DOC]
        # Repeating each element twice gives an index built from repeating the values, with name preserved; repeating by an array of counts (zero up to length minus one) likewise.
        rep = 2
        idx = simple_index.copy()
        new_index_cls = idx._constructor
        expected = new_index_cls(idx.values.repeat(rep), name=idx.name)
        tm.assert_index_equal(idx.repeat(rep), expected)

        idx = simple_index
        rep = np.arange(len(idx))
        expected = new_index_cls(idx.values.repeat(rep), name=idx.name)
        tm.assert_index_equal(idx.repeat(rep), expected)

    def test_numpy_repeat(self, simple_index):
        # [FUNCTION DOC]
        # The numpy repeat function equals the method; passing an axis raises a value error that the parameter is not supported.
        rep = 2
        idx = simple_index
        expected = idx.repeat(rep)
        tm.assert_index_equal(np.repeat(idx, rep), expected)

        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.repeat(idx, rep, axis=0)

    def test_where(self, listlike_box, simple_index):
        # [FUNCTION DOC]
        # Uses the listlike_box fixture. Skips interval, period and numeric indexes. Datetime-like indexes first have the frequency cleared (where does not preserve it). A condition that is all true returns the same index; a condition with the first element false replaces it with the index's missing-value marker, keeping dtype.
        if isinstance(simple_index, (IntervalIndex, PeriodIndex)) or is_numeric_dtype(
            simple_index.dtype
        ):
            pytest.skip("Tested elsewhere.")
        klass = listlike_box

        idx = simple_index
        if isinstance(idx, (DatetimeIndex, TimedeltaIndex)):
            # where does not preserve freq
            idx = idx._with_freq(None)

        cond = [True] * len(idx)
        result = idx.where(klass(cond))
        expected = idx
        tm.assert_index_equal(result, expected)

        cond = [False] + [True] * len(idx[1:])
        expected = Index([idx._na_value, *idx[1:].tolist()], dtype=idx.dtype)
        result = idx.where(klass(cond))
        tm.assert_index_equal(result, expected)

    def test_insert_base(self, index):
        # [FUNCTION DOC]
        # Skips empty indexes. Inserting the first element at the start of the slice between positions one and four gives the first four elements.
        # GH#51363
        trimmed = index[1:4]

        if not len(index):
            pytest.skip("Not applicable for empty index")

        result = trimmed.insert(0, index[0])
        assert index[0:4].equals(result)

    def test_insert_out_of_bounds(self, index, using_infer_string):
        # [FUNCTION DOC]
        # Uses the shared index fixture and infer-string flag. Inserting at a non-integer position (0.5) raises a type error for non-empty indexes (message about slice indices, differing by Python version and string inference) or an index error for empty ones (message about index out of bounds with size 0); with string inference some dtypes use the "loc must be an integer between" message. Inserting at positions one past the end or one before the beginning raises an index error.
        # TypeError/IndexError matches what np.insert raises in these cases

        if len(index) > 0:
            err = TypeError
        else:
            err = IndexError
        if len(index) == 0:
            # 0 vs 0.5 in error message varies with numpy version
            msg = "index (0|0.5) is out of bounds for axis 0 with size 0"
        elif PY315:
            msg = "slice indices must be integers or have an __index__ method"
        else:
            msg = "slice indices must be integers or None or have an __index__ method"

        if using_infer_string:
            if str(index.dtype) in {"str", "string", "category"}:
                msg = "loc must be an integer between"
            elif index.dtype == "object" and len(index) == 0:
                msg = "loc must be an integer between"
                err = TypeError

        with pytest.raises(err, match=msg):
            index.insert(0.5, "foo")

        msg = "|".join(
            [
                r"index -?\d+ is out of bounds for axis 0 with size \d+",
                "loc must be an integer between",
            ]
        )
        with pytest.raises(IndexError, match=msg):
            index.insert(len(index) + 1, 1)

        with pytest.raises(IndexError, match=msg):
            index.insert(-len(index) - 1, 1)

    def test_delete_base(self, index):
        # [FUNCTION DOC]
        # Skips empty and range indexes. Deleting position zero and minus one gives the expected slices and preserves the name; deleting the position equal to the length raises an index error stating it is out of bounds.
        if not len(index):
            pytest.skip("Not applicable for empty index")

        if isinstance(index, RangeIndex):
            # tested in class
            pytest.skip(f"{type(self).__name__} tested elsewhere")

        expected = index[1:]
        result = index.delete(0)
        assert result.equals(expected)
        assert result.name == expected.name

        expected = index[:-1]
        result = index.delete(-1)
        assert result.equals(expected)
        assert result.name == expected.name

        length = len(index)
        msg = f"index {length} is out of bounds for axis 0 with size {length}"
        with pytest.raises(IndexError, match=msg):
            index.delete(length)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_equals(self, index):
        # [FUNCTION DOC]
        # Skips interval indexes. An index equals itself and a copy, and its object-cast version (not for extension-backed base indexes); does not equal a list, numpy array or series of the same values; the object-dtype version built from it equals it in both directions (except range and extension-backed).
        if isinstance(index, IntervalIndex):
            pytest.skip(f"{type(index).__name__} tested elsewhere")

        is_ea_idx = type(index) is Index and not isinstance(index.dtype, np.dtype)

        assert index.equals(index)
        assert index.equals(index.copy())
        if not is_ea_idx:
            # doesn't hold for e.g. IntegerDtype
            assert index.equals(index.astype(object))

        assert not index.equals(list(index))
        assert not index.equals(np.array(index))

        # Cannot pass in non-int64 dtype to RangeIndex
        if not isinstance(index, RangeIndex) and not is_ea_idx:
            same_values = Index(index, dtype=object)
            assert index.equals(same_values)
            assert same_values.equals(index)

        if index.nlevels == 1:
            # do not test MultiIndex
            assert not index.equals(Series(index))

    def test_equals_op(self, simple_index):
        # [FUNCTION DOC]
        # Compares an index with indexes, arrays and series of different lengths: length mismatch raises a value error ("Lengths must match" or broadcast failure), equal-length comparisons produce boolean arrays or series; length-one operands raise; series with differing labels raise "Can only compare identically-labeled Series objects"; and (excluding multi-level) comparison with a scalar (the second-to-last item, assumed unique) broadcasts.
        # GH9947, GH10637
        index_a = simple_index

        n = len(index_a)
        index_b = index_a[0:-1]
        index_c = index_a[0:-1].append(index_a[-2:-1])
        index_d = index_a[0:1]

        msg = "Lengths must match|could not be broadcast"
        with pytest.raises(ValueError, match=msg):
            index_a == index_b
        expected1 = np.array([True] * n)
        expected2 = np.array([True] * (n - 1) + [False])
        tm.assert_numpy_array_equal(index_a == index_a, expected1)
        tm.assert_numpy_array_equal(index_a == index_c, expected2)

        # test comparisons with numpy arrays
        array_a = np.array(index_a)
        array_b = np.array(index_a[0:-1])
        array_c = np.array(index_a[0:-1].append(index_a[-2:-1]))
        array_d = np.array(index_a[0:1])
        with pytest.raises(ValueError, match=msg):
            index_a == array_b
        tm.assert_numpy_array_equal(index_a == array_a, expected1)
        tm.assert_numpy_array_equal(index_a == array_c, expected2)

        # test comparisons with Series
        series_a = Series(array_a)
        series_b = Series(array_b)
        series_c = Series(array_c)
        series_d = Series(array_d)
        with pytest.raises(ValueError, match=msg):
            index_a == series_b

        tm.assert_series_equal(index_a == series_a, Series(expected1))
        tm.assert_series_equal(index_a == series_c, Series(expected2))

        # cases where length is 1 for one of them
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == index_d
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == series_d
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == array_d
        msg = "Can only compare identically-labeled Series objects"
        with pytest.raises(ValueError, match=msg):
            series_a == series_d
        with pytest.raises(ValueError, match="Lengths must match"):
            series_a == array_d

        # comparing with a scalar should broadcast; note that we are excluding
        # MultiIndex because in this case each item in the index is a tuple of
        # length 2, and therefore is considered an array of length 2 in the
        # comparison instead of a scalar
        if not isinstance(index_a, MultiIndex):
            expected3 = np.array([False] * (len(index_a) - 2) + [True, False])
            # assuming the 2nd to last item is unique in the data
            item = index_a[-2]
            tm.assert_numpy_array_equal(index_a == item, expected3)
            tm.assert_series_equal(series_a == item, Series(expected3))

    def test_fillna(self, index):
        # [FUNCTION DOC]
        # Uses the shared index fixture. Skips empty, boolean and integer-dtype indexes. Multi-level indexes raise a not-implemented error. Others: filling with a scalar returns an equal but distinct index; a list-like value raises a type error ("value must be a scalar"); and after setting the second value to NaN in the values, rebuilding gives an index whose NaN mask marks only that position and reports NaNs present.
        # GH 11343
        if len(index) == 0:
            pytest.skip("Not relevant for empty index")
        elif index.dtype == bool:
            pytest.skip(f"{index.dtype} cannot hold NAs")
        elif isinstance(index, Index) and is_integer_dtype(index.dtype):
            pytest.skip(f"Not relevant for Index with {index.dtype}")
        elif isinstance(index, MultiIndex):
            idx = index.copy(deep=True)
            msg = "fillna is not defined for MultiIndex"
            with pytest.raises(NotImplementedError, match=msg):
                idx.fillna(idx[0])
        else:
            idx = index.copy(deep=True)
            result = idx.fillna(idx[0])
            tm.assert_index_equal(result, idx)
            assert result is not idx

            msg = "'value' must be a scalar, passed: "
            with pytest.raises(TypeError, match=msg):
                idx.fillna([idx[0]])

            idx = index.copy(deep=True)
            values = idx._values

            values[1] = np.nan

            idx = type(index)(values)

            expected = np.array([False] * len(idx), dtype=bool)
            expected[1] = True
            tm.assert_numpy_array_equal(idx._isnan, expected)
            assert idx.hasnans is True

    def test_nulls(self, index):
        # [FUNCTION DOC]
        # Smoke test of missing-value methods on the shared index fixture. Empty index gives an empty boolean array; multi-level raises a not-implemented error; indexes without NaNs give all-false is-missing and all-true not-missing; others agree with the module-level missing-value function and its negation.
        # this is really a smoke test for the methods
        # as these are adequately tested for function elsewhere
        if len(index) == 0:
            tm.assert_numpy_array_equal(index.isna(), np.array([], dtype=bool))
        elif isinstance(index, MultiIndex):
            idx = index.copy()
            msg = "isna is not defined for MultiIndex"
            with pytest.raises(NotImplementedError, match=msg):
                idx.isna()
        elif not index.hasnans:
            tm.assert_numpy_array_equal(index.isna(), np.zeros(len(index), dtype=bool))
            tm.assert_numpy_array_equal(index.notna(), np.ones(len(index), dtype=bool))
        else:
            result = isna(index)
            tm.assert_numpy_array_equal(index.isna(), result)
            tm.assert_numpy_array_equal(index.notna(), ~result)

    def test_empty(self, simple_index):
        # [FUNCTION DOC]
        # The empty flag is false for the fixture and true for a zero-length slice.
        # GH 15270
        idx = simple_index
        assert not idx.empty
        assert idx[:0].empty

    def test_join_self_unique(self, join_type, simple_index):
        # [FUNCTION DOC]
        # For unique indexes, joining with itself under each join type (join_type fixture) returns the index (sorted with the safe sort helper for outer joins).
        idx = simple_index
        if idx.is_unique:
            joined = idx.join(idx, how=join_type)
            expected = simple_index
            if join_type == "outer":
                expected = algos.safe_sort(expected)
            tm.assert_index_equal(joined, expected)

    def test_map(self, simple_index):
        # [FUNCTION DOC]
        # Skips timedelta and period indexes. Mapping with the identity function returns an equivalent index (range indexes may be equivalent to int64).
        # callable
        if isinstance(simple_index, (TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested elsewhere.")
        idx = simple_index

        result = idx.map(lambda x: x)
        # RangeIndex are equivalent to the similar Index with int64 dtype
        tm.assert_index_equal(result, idx, exact="equiv")

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_map_dictlike(self, mapper, simple_index, request):
        # [FUNCTION DOC]
        # Parametrized over dictionary and series mappers. Ignores a period deprecation warning. Skips datetime-like indexes. Mapping through an identity mapper returns an equivalent index; mapping through a mapper from an index-sized all-NaN expectation gives an all-NaN index (float dtype for float indexes).
        idx = simple_index
        if isinstance(idx, (DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested elsewhere.")

        identity = mapper(idx.values, idx)

        result = idx.map(identity)
        # RangeIndex are equivalent to the similar Index with int64 dtype
        tm.assert_index_equal(result, idx, exact="equiv")

        # empty mappable
        dtype = None
        if idx.dtype.kind == "f":
            dtype = idx.dtype

        expected = Index([np.nan] * len(idx), dtype=dtype)
        result = idx.map(mapper(expected, idx))
        tm.assert_index_equal(result, expected)

    def test_map_str(self, simple_index):
        # [FUNCTION DOC]
        # Skips categorical. Mapping with the string conversion function gives an index of the string forms.
        # GH 31202
        if isinstance(simple_index, CategoricalIndex):
            pytest.skip("See test_map.py")
        idx = simple_index
        result = idx.map(str)
        expected = Index([str(x) for x in idx])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_astype_category(self, copy, name, ordered, simple_index):
        # [FUNCTION DOC]
        # Parametrized over copy, name (none or foo), ordered. Casting to a default categorical dtype equals a categorical index (exact). Casting to categories that omit the last unique value emits the Pandas 4 warning about constructing a categorical with a dtype and values containing other values, and equals the categorical index constructed the same way. For unordered only, casting to the string category also equals the default categorical index.
        # GH 18630
        idx = simple_index
        if name:
            idx = idx.rename(name)

        # standard categories
        dtype = CategoricalDtype(ordered=ordered)
        result = idx.astype(dtype, copy=copy)
        expected = CategoricalIndex(idx, name=name, ordered=ordered)
        tm.assert_index_equal(result, expected, exact=True)

        # non-standard categories
        dtype = CategoricalDtype(idx.unique().tolist()[:-1], ordered)
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = idx.astype(dtype, copy=copy)
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            expected = CategoricalIndex(idx, name=name, dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

        if ordered is False:
            # dtype='category' defaults to ordered=False, so only test once
            result = idx.astype("category", copy=copy)
            expected = CategoricalIndex(idx, name=name)
            tm.assert_index_equal(result, expected, exact=True)

    def test_is_unique(self, simple_index):
        # [FUNCTION DOC]
        # Drops duplicates to get a unique index (unique); an empty slice is unique; inserting a duplicate of the first element makes it non-unique; inserting one NaN keeps it unique; inserting a second NaN makes it non-unique.
        # initialize a unique index
        index = simple_index.drop_duplicates()
        assert index.is_unique is True

        # empty index should be unique
        index_empty = index[:0]
        assert index_empty.is_unique is True

        # test basic dupes
        index_dup = index.insert(0, index[0])
        assert index_dup.is_unique is False

        # single NA should be unique
        index_na = index.insert(0, np.nan)
        assert index_na.is_unique is True

        # multiple NA should not be unique
        index_na_dup = index_na.insert(0, np.nan)
        assert index_na_dup.is_unique is False

    @pytest.mark.arm_slow
    def test_engine_reference_cycle(self, simple_index):
        # [FUNCTION DOC]
        # Marked slow on ARM. Accessing the engine of a copied index and then deleting the index lets the index be garbage collected (the weak reference is cleared), proving there is no reference cycle through the engine.
        # GH27585
        index = simple_index.copy()
        ref = weakref.ref(index)
        index._engine
        del index
        assert ref() is None

    def test_getitem_2d_deprecated(self, simple_index):
        # [FUNCTION DOC]
        # Skips interval indexes. Indexing with a slice and a new axis raises a value or index error about multi-dimensional indexing; indexing with a boolean scalar (true or false) raises the same, except for range indexes which raise an index error that only integers and slices are allowed.
        # GH#30588, GH#31479
        if isinstance(simple_index, IntervalIndex):
            pytest.skip("Tested elsewhere")
        idx = simple_index
        msg = "Multi-dimensional indexing|too many|only"
        with pytest.raises((ValueError, IndexError), match=msg):
            idx[:, None]

        if not isinstance(idx, RangeIndex):
            # GH#44051 RangeIndex already raised pre-2.0 with a different message
            with pytest.raises((ValueError, IndexError), match=msg):
                idx[True]
            with pytest.raises((ValueError, IndexError), match=msg):
                idx[False]
        else:
            msg = "only integers, slices"
            with pytest.raises(IndexError, match=msg):
                idx[True]
            with pytest.raises(IndexError, match=msg):
                idx[False]

    def test_copy_shares_cache(self, simple_index):
        # [FUNCTION DOC]
        # After a location lookup populates the cache, a copy has the very same cache dictionary object.
        # GH32898, GH36840
        idx = simple_index
        idx.get_loc(idx[0])  # populates the _cache.
        copy = idx.copy()

        assert copy._cache is idx._cache

    def test_shallow_copy_shares_cache(self, simple_index):
        # [FUNCTION DOC]
        # After populating the cache, the private view shares the cache dictionary; the private shallow copy built from the data gets a new empty cache.
        # GH32669, GH36840
        idx = simple_index
        idx.get_loc(idx[0])  # populates the _cache.
        shallow_copy = idx._view()

        assert shallow_copy._cache is idx._cache

        shallow_copy = idx._shallow_copy(idx._data)
        assert shallow_copy._cache is not idx._cache
        assert shallow_copy._cache == {}

    def test_index_groupby(self, simple_index):
        # [FUNCTION DOC]
        # Grouping the first five elements by a float array containing NaN gives a dictionary keyed by the float keys with member indexes (NaN group omitted); grouping by datetime64 values (with a missing value) gives timestamp keys.
        idx = simple_index[:5]
        to_groupby = np.array([1, 2, np.nan, 2, 1])
        tm.assert_dict_equal(
            idx.groupby(to_groupby), {1.0: idx[[0, 4]], 2.0: idx[[1, 3]]}
        )

        to_groupby = DatetimeIndex(
            [
                datetime(2011, 11, 1),
                datetime(2011, 12, 1),
                pd.NaT,
                datetime(2011, 12, 1),
                datetime(2011, 11, 1),
            ],
            tz="UTC",
        ).values

        ex_keys = [Timestamp("2011-11-01"), Timestamp("2011-12-01")]
        expected = {ex_keys[0]: idx[[0, 4]], ex_keys[1]: idx[[1, 3]]}
        tm.assert_dict_equal(idx.groupby(to_groupby), expected)

    def test_append_preserves_dtype(self, simple_index):
        # [FUNCTION DOC]
        # Appending an index to itself keeps its dtype (including float32); both halves equal the original with exact value comparison, and the whole equals taking positions twice.
        # In particular Index with dtype float32
        index = simple_index
        N = len(index)

        result = index.append(index)
        assert result.dtype == index.dtype

        tm.assert_index_equal(result[:N], index, exact=False, check_exact=True)
        tm.assert_index_equal(result[N:], index, exact=False, check_exact=True)

        alt = index.take(list(range(N)) * 2)
        tm.assert_index_equal(result, alt, check_exact=True)

    def test_inv(self, simple_index, using_infer_string):
        # [FUNCTION DOC]
        # Uses the infer-string flag fixture (unused). For integer kinds bitwise inversion gives an index of inverted values (matches the series result); otherwise raises a type error (ufunc message for floats, "bad operand" or string-dtype message otherwise), also for series.
        idx = simple_index

        if idx.dtype.kind in ["i", "u"]:
            res = ~idx
            expected = Index(~idx.values, name=idx.name)
            tm.assert_index_equal(res, expected)

            # check that we are matching Series behavior
            res2 = ~Series(idx)
            tm.assert_series_equal(res2, Series(expected))
        else:
            if idx.dtype.kind == "f":
                msg = "ufunc 'invert' not supported for the input types"
            else:
                msg = "bad operand|__invert__ is not supported for string dtype"
            with pytest.raises(TypeError, match=msg):
                ~idx

            # check that we get the same behavior with Series
            with pytest.raises(TypeError, match=msg):
                ~Series(idx)


class TestNumericBase:
    # [CLASS DOC]
    # Legacy tests for numeric indexes only; provides a fixture over the numeric subset (range, float64, float32, uint64, signed integers).
    @pytest.fixture(
        params=[
            RangeIndex(start=0, stop=20, step=2),
            Index(np.arange(5, dtype=np.float64)),
            Index(np.arange(5, dtype=np.float32)),
            Index(np.arange(5, dtype=np.uint64)),
            Index(range(0, 20, 2), dtype=np.int64),
            Index(range(0, 20, 2), dtype=np.int32),
            Index(range(0, 20, 2), dtype=np.int16),
            Index(range(0, 20, 2), dtype=np.int8),
        ]
    )
    def simple_index(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over a range index, float64 and float32 indexes, a uint64 index and int64/32/16/8 indexes.
        return request.param

    def test_constructor_unwraps_index(self, simple_index):
        # [FUNCTION DOC]
        # Skips range indexes. Constructing the same class from an index of the same dtype gives an index whose internal array equals the values.
        if isinstance(simple_index, RangeIndex):
            pytest.skip("Tested elsewhere.")
        index_cls = type(simple_index)
        dtype = simple_index.dtype

        idx = Index([1, 2], dtype=dtype)
        result = index_cls(idx)
        expected = np.array([1, 2], dtype=idx.dtype)
        tm.assert_numpy_array_equal(result._data, expected)

    def test_can_hold_identifiers(self, simple_index):
        # [FUNCTION DOC]
        # The private identifier check is false for numeric indexes.
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_view(self, simple_index):
        # [FUNCTION DOC]
        # Skips range indexes. An empty named index: default view keeps name; view with its own dtype equals a reconstructed index (exact); viewing as the index class raises a type error about changing data type.
        if isinstance(simple_index, RangeIndex):
            pytest.skip("Tested elsewhere.")
        index_cls = type(simple_index)
        dtype = simple_index.dtype

        idx = index_cls([], dtype=dtype, name="Foo")
        idx_view = idx.view()
        assert idx_view.name == "Foo"

        idx_view = idx.view(dtype)
        tm.assert_index_equal(idx, index_cls(idx_view, name="Foo"), exact=True)

        msg = (
            "Cannot change data-type for array of references.|"
            "Cannot change data-type for object array.|"
        )
        with pytest.raises(TypeError, match=msg):
            # GH#55709
            idx.view(index_cls)

    def test_insert_non_na(self, simple_index):
        # [FUNCTION DOC]
        # Inserting an element the index can already hold keeps dtype and type (exact).
        # GH#43921 inserting an element that we know we can hold should
        #  not change dtype or type (except for RangeIndex)
        index = simple_index

        result = index.insert(0, index[0])

        expected = Index([index[0], *list(index)], dtype=index.dtype)
        tm.assert_index_equal(result, expected, exact=True)

    def test_insert_na(self, nulls_fixture, simple_index):
        # [FUNCTION DOC]
        # Uses the nulls fixture. Inserting a missing value at position one: NaT yields an object index; other nulls yield an index with NaN (float dtype preserved for float indexes). Exact comparison.
        # GH 18295 (test missing)
        index = simple_index
        na_val = nulls_fixture

        if na_val is pd.NaT:
            expected = Index([index[0], pd.NaT, *list(index[1:])], dtype=object)
        else:
            expected = Index([index[0], np.nan, *list(index[1:])])
            # GH#43921 we preserve float dtype
            if index.dtype.kind == "f":
                expected = Index(expected, dtype=index.dtype)

        result = index.insert(1, na_val)
        tm.assert_index_equal(result, expected, exact=True)

    def test_arithmetic_explicit_conversions(self, simple_index):
        # [FUNCTION DOC]
        # Multiplying an integer index of five elements by a float scalar (either order) gives a float64 index; subtracting a float zero array from it and from a zero array gives the expected positive and negated float64 indexes.
        # GH 8608
        # add/sub are overridden explicitly for Float/Int Index
        index_cls = type(simple_index)
        if index_cls is RangeIndex:
            idx = RangeIndex(5)
        else:
            idx = index_cls(np.arange(5, dtype="int64"))

        # float conversions
        arr = np.arange(5, dtype="int64") * 3.2
        expected = Index(arr, dtype=np.float64)
        fidx = idx * 3.2
        tm.assert_index_equal(fidx, expected)
        fidx = 3.2 * idx
        tm.assert_index_equal(fidx, expected)

        # interops with numpy arrays
        expected = Index(arr, dtype=np.float64)
        a = np.zeros(5, dtype="float64")
        result = fidx - a
        tm.assert_index_equal(result, expected)

        expected = Index(-arr, dtype=np.float64)
        a = np.zeros(5, dtype="float64")
        result = a - fidx
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
    def test_astype_to_complex(self, complex_dtype, simple_index):
        # [FUNCTION DOC]
        # Parametrized over complex64 and complex128. Casting returns a plain Index with that complex dtype.
        result = simple_index.astype(complex_dtype)

        assert type(result) is Index and result.dtype == complex_dtype

    def test_cast_string(self, simple_index):
        # [FUNCTION DOC]
        # Skips range indexes. Constructing from numeric strings with the index's dtype equals constructing from the numbers.
        if isinstance(simple_index, RangeIndex):
            pytest.skip("casting of strings not relevant for RangeIndex")
        result = type(simple_index)(["0", "1", "2"], dtype=simple_index.dtype)
        expected = type(simple_index)([0, 1, 2], dtype=simple_index.dtype)
        tm.assert_index_equal(result, expected)
