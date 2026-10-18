# [MODULE DOC]
# Tests for the core algorithms module and hash-table layer: factorize, unique, nunique, isin, value counts, duplicated, rank, mode, diff, and union with duplicates, plus direct tests of the low-level hash tables. Organized by class per algorithm. Uses dtype and index fixtures from the test configuration, including the writable-array fixture that runs tests on read-only and writable inputs, the sort and time-unit fixtures, and the null-value fixtures.
from datetime import datetime
import struct

import numpy as np
import pytest

from pandas._libs import (
    algos as libalgos,
    hashtable as ht,
)

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_complex_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
)

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    PeriodIndex,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)
import pandas.core.common as com


class TestFactorize:
    # [CLASS DOC]
    # Tests for factorize (encoding values as integer codes plus the array of unique values) through the module function, the Series and Index methods, and the low-level factorizer objects.
    def test_factorize_complex(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies factorizing a complex array returns codes in order of appearance and the unique complex values.
        # GH#17927
        array = np.array([1, 2, 2 + 1j], dtype=complex)
        labels, uniques = algos.factorize(array)

        expected_labels = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, expected_labels)

        expected_uniques = np.array([(1 + 0j), (2 + 0j), (2 + 1j)], dtype=complex)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_factorize(self, index_or_series_obj, sort):
        # [FUNCTION DOC]
        # Uses the index-or-series fixture covering many object kinds and the sort fixture. Verifies the codes and uniques returned by the factorize method: the uniques equal the container of unique values (a MultiIndex built from tuples for a MultiIndex, with float16 promoted to float32 and boolean uniques of an object index kept as object), sorted if requested, and the codes are the position of each element in that uniques list; index equality is exact.
        obj = index_or_series_obj
        result_codes, result_uniques = obj.factorize(sort=sort)

        constructor = Index
        if isinstance(obj, MultiIndex):
            constructor = MultiIndex.from_tuples
        expected_arr = obj.unique()
        if expected_arr.dtype == np.float16:
            expected_arr = expected_arr.astype(np.float32)
        expected_uniques = constructor(expected_arr)
        if (
            isinstance(obj, Index)
            and expected_uniques.dtype == bool
            and obj.dtype == object
        ):
            expected_uniques = expected_uniques.astype(object)

        if sort:
            expected_uniques = expected_uniques.sort_values()

        # construct an integer ndarray so that
        # `expected_uniques.take(expected_codes)` is equal to `obj`
        expected_uniques_list = list(expected_uniques)
        expected_codes = [expected_uniques_list.index(val) for val in obj]
        expected_codes = np.asarray(expected_codes, dtype=np.intp)

        tm.assert_numpy_array_equal(result_codes, expected_codes)
        tm.assert_index_equal(result_uniques, expected_uniques, exact=True)

    def test_series_factorize_use_na_sentinel_false(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies that with the missing-value sentinel disabled, NaN becomes its own category: codes zero, one, zero, two and uniques one, two, NaN.
        # GH#35667
        values = np.array([1, 2, 1, np.nan])
        ser = Series(values)
        codes, uniques = ser.factorize(use_na_sentinel=False)

        expected_codes = np.array([0, 1, 0, 2], dtype=np.intp)
        expected_uniques = Index([1.0, 2.0, np.nan])

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_index_equal(uniques, expected_uniques)

    def test_basic(self):
        # [FUNCTION DOC]
        # Verifies factorize on an object array of letters and on integer and float arrays in reverse order, without and with sorting: codes by order of appearance (or sorted rank) and uniques in appearance (or sorted) order.
        items = np.array(["a", "b", "b", "a", "a", "c", "c", "c"], dtype=object)
        codes, uniques = algos.factorize(items)
        tm.assert_numpy_array_equal(uniques, np.array(["a", "b", "c"], dtype=object))

        codes, uniques = algos.factorize(items, sort=True)
        exp = np.array([0, 1, 1, 0, 0, 2, 2, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array(["a", "b", "c"], dtype=object)
        tm.assert_numpy_array_equal(uniques, exp)

        arr = np.arange(5, dtype=np.intp)[::-1]

        codes, uniques = algos.factorize(arr)
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([4, 3, 2, 1, 0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        codes, uniques = algos.factorize(arr, sort=True)
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([0, 1, 2, 3, 4], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        arr = np.arange(5.0)[::-1]

        codes, uniques = algos.factorize(arr)
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        codes, uniques = algos.factorize(arr, sort=True)
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

    def test_mixed(self):
        # [FUNCTION DOC]
        # Documentation example. Verifies factorizing a Series of strings, a float, infinity and NaN gives -1 for NaN and an Index of the unique non-null values in appearance order, or sorted (numbers before strings) with sort.
        # doc example reshaping.rst
        x = Series(["A", "A", np.nan, "B", 3.14, np.inf])
        codes, uniques = algos.factorize(x)

        exp = np.array([0, 0, -1, 1, 2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = Index(["A", "B", 3.14, np.inf])
        tm.assert_index_equal(uniques, exp)

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([2, 2, -1, 3, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = Index([3.14, np.inf, "A", "B"])
        tm.assert_index_equal(uniques, exp)

    def test_factorize_datetime64(self):
        # [FUNCTION DOC]
        # Verifies factorizing a Series of two timestamps returns codes and a DatetimeIndex of uniques in appearance order, or sorted order with the sort option.
        # M8
        v1 = Timestamp("20130101 09:00:00.00004")
        v2 = Timestamp("20130101")
        x = Series([v1, v1, v1, v2, v2, v1])
        codes, uniques = algos.factorize(x)

        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = DatetimeIndex([v1, v2])
        tm.assert_index_equal(uniques, exp)

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([1, 1, 1, 0, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = DatetimeIndex([v2, v1])
        tm.assert_index_equal(uniques, exp)

    def test_factorize_period(self):
        # [FUNCTION DOC]
        # Verifies factorizing a Series of periods returns a PeriodIndex of uniques; sorting does not change the order here as the values are already ordered by appearance.
        # period
        v1 = Period("201302", freq="M")
        v2 = Period("201303", freq="M")
        x = Series([v1, v1, v1, v2, v2, v1])

        # periods are not 'sorted' as they are converted back into an index
        codes, uniques = algos.factorize(x)
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, PeriodIndex([v1, v2]))

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, PeriodIndex([v1, v2]))

    def test_factorize_timedelta(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies factorizing timedeltas returns codes and a timedelta index of uniques in appearance or sorted order.
        # GH 5986
        v1 = to_timedelta("1 day 1 min")
        v2 = to_timedelta("1 day")
        x = Series([v1, v2, v1, v1, v2, v2, v1])
        codes, uniques = algos.factorize(x)
        exp = np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, to_timedelta([v1, v2]))

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, to_timedelta([v2, v1]))

    def test_factorize_nan(self):
        # [FUNCTION DOC]
        # Verifies the low-level object factorizer maps NaN to the sentinel it was given (minus one, and twenty — an index outside the reverse indexer), returning codes with that sentinel, and that NaN detection of the key array matches the sentinel positions.
        # nan should map to na_sentinel, not reverse_indexer[na_sentinel]
        # rizer.factorize should not raise an exception if na_sentinel indexes
        # outside of reverse_indexer
        key = np.array([1, 2, 1, np.nan], dtype="O")
        rizer = ht.ObjectFactorizer(len(key))
        for na_sentinel in (-1, 20):
            ids = rizer.factorize(key, na_sentinel=na_sentinel)
            expected = np.array([0, 1, 0, na_sentinel], dtype=np.intp)
            assert len(set(key)) == len(set(expected))
            tm.assert_numpy_array_equal(pd.isna(key), expected == na_sentinel)
            tm.assert_numpy_array_equal(ids, expected)

    def test_factorizer_with_mask(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies the 64-bit integer factorizer with a mask gives the sentinel for masked entries and records only unmasked uniques.
        # GH#49549
        data = np.array([1, 2, 3, 1, 1, 0], dtype="int64")
        mask = np.array([False, False, False, False, False, True])
        rizer = ht.Int64Factorizer(len(data))
        result = rizer.factorize(data, mask=mask)
        expected = np.array([0, 1, 2, 0, 0, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        expected_uniques = np.array([1, 2, 3], dtype="int64")
        tm.assert_numpy_array_equal(rizer.uniques.to_array(), expected_uniques)

    def test_factorizer_object_with_nan(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies the object factorizer treats NaN as missing (sentinel code) and the uniques exclude it.
        # GH#49549
        data = np.array([1, 2, 3, 1, np.nan])
        rizer = ht.ObjectFactorizer(len(data))
        result = rizer.factorize(data.astype(object))
        expected = np.array([0, 1, 2, 0, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        expected_uniques = np.array([1, 2, 3], dtype=object)
        tm.assert_numpy_array_equal(rizer.uniques.to_array(), expected_uniques)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                [(1, 1), (1, 2), (0, 0), (1, 2), "nonsense"],
                [0, 1, 2, 1, 3],
                [(1, 1), (1, 2), (0, 0), "nonsense"],
            ),
            (
                [(1, 1), (1, 2), (0, 0), (1, 2), (1, 2, 3)],
                [0, 1, 2, 1, 3],
                [(1, 1), (1, 2), (0, 0), (1, 2, 3)],
            ),
            ([(1, 1), (1, 2), (0, 0), (1, 2)], [0, 1, 2, 1], [(1, 1), (1, 2), (0, 0)]),
        ],
    )
    def test_factorize_tuple_list(self, data, expected_codes, expected_uniques):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over object arrays of tuples mixed with a string or with different-length tuples. Verifies the codes and uniques arrays treat tuples as single values.
        # GH9454
        data = com.asarray_tuplesafe(data, dtype=object)
        codes, uniques = pd.factorize(data)

        tm.assert_numpy_array_equal(codes, np.array(expected_codes, dtype=np.intp))

        expected_uniques_array = com.asarray_tuplesafe(expected_uniques, dtype=object)
        tm.assert_numpy_array_equal(uniques, expected_uniques_array)

    def test_complex_sorting(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue (checking for no crash). Verifies sorting factorization of an object array of complex numbers raises a type error saying ordering comparison is not supported.
        # gh 12666 - check no segfault
        x17 = np.array([complex(i) for i in range(17)], dtype=object)

        msg = "'[<>]' not supported between instances of .*"
        with pytest.raises(TypeError, match=msg):
            algos.factorize(x17[::-1], sort=True)

    def test_numeric_dtype_factorize(self, any_real_numpy_dtype):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses the real NumPy dtype fixture. Verifies codes and uniques of a small array keep the dtype.
        # GH41132
        dtype = any_real_numpy_dtype
        data = np.array([1, 2, 2, 1], dtype=dtype)
        expected_codes = np.array([0, 1, 1, 0], dtype=np.intp)
        expected_uniques = np.array([1, 2], dtype=dtype)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_float64_factorize(self, writable):
        # [FUNCTION DOC]
        # Uses the writable fixture. Verifies factorizing floats of widely differing scale, in a writable or read-only array.
        data = np.array([1.0, 1e8, 1.0, 1e-8, 1e8, 1.0], dtype=np.float64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        expected_uniques = np.array([1.0, 1e8, 1e-8], dtype=np.float64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_uint64_factorize(self, writable):
        # [FUNCTION DOC]
        # Uses the writable fixture. Verifies factorizing unsigned 64-bit values including the largest value.
        data = np.array([2**64 - 1, 1, 2**64 - 1], dtype=np.uint64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        expected_uniques = np.array([2**64 - 1, 1], dtype=np.uint64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_int64_factorize(self, writable):
        # [FUNCTION DOC]
        # Uses the writable fixture. Verifies factorizing signed 64-bit values at the extremes.
        data = np.array([2**63 - 1, -(2**63), 2**63 - 1], dtype=np.int64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        expected_uniques = np.array([2**63 - 1, -(2**63)], dtype=np.int64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_string_factorize(self, writable):
        # [FUNCTION DOC]
        # Uses the writable fixture. Verifies factorizing an object array of strings.
        data = np.array(["a", "c", "a", "b", "c"], dtype=object)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0, 2, 1], dtype=np.intp)
        expected_uniques = np.array(["a", "c", "b"], dtype=object)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_object_factorize(self, writable):
        # [FUNCTION DOC]
        # Uses the writable fixture. Verifies None, NaN and NaT in an object array all get the missing code and are excluded from the uniques.
        data = np.array(["a", "c", None, np.nan, "a", "b", NaT, "c"], dtype=object)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, -1, -1, 0, 2, -1, 1], dtype=np.intp)
        expected_uniques = np.array(["a", "c", "b"], dtype=object)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_datetime64_factorize(self, writable):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses the writable fixture. Verifies a possibly read-only datetime64 array can be factorized.
        # GH35650 Verify whether read-only datetime64 array can be factorized
        data = np.array([np.datetime64("2020-01-01T00:00:00.000")], dtype="M8[ns]")
        data.setflags(write=writable)
        expected_codes = np.array([0], dtype=np.intp)
        expected_uniques = np.array(
            ["2020-01-01T00:00:00.000000000"], dtype="datetime64[ns]"
        )

        codes, uniques = pd.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_factorize_rangeindex(self, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. Verifies factorizing an increasing RangeIndex through the function and the method returns the positions as codes and the range itself as the uniques, regardless of sorting.
        # increasing -> sort doesn't matter
        ri = pd.RangeIndex.from_range(range(10))
        expected = np.arange(10, dtype=np.intp), ri

        result = algos.factorize(ri, sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

        result = ri.factorize(sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

    def test_factorize_rangeindex_decreasing(self, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. Verifies factorizing a decreasing RangeIndex gives positions and the reversed range, or with sorting the reversed codes and the increasing range.
        # decreasing -> sort matters
        ri = pd.RangeIndex.from_range(range(10))
        expected = np.arange(10, dtype=np.intp), ri

        ri2 = ri[::-1]
        expected = expected[0], ri2
        if sort:
            expected = expected[0][::-1], expected[1][::-1]

        result = algos.factorize(ri2, sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

        result = ri2.factorize(sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

    def test_deprecate_order(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies passing the removed order keyword raises a type error about an unexpected keyword, and factorizing without it emits no warning.
        # KNOWN ISSUE: the comments and name describe a warning for a deprecated keyword, but the test checks for a type error because the keyword has been removed.
        # gh 19727 - check warning is raised for deprecated keyword, order.
        # Test not valid once order keyword is removed.
        data = np.array([2**63, 1, 2**63], dtype=np.uint64)
        with pytest.raises(TypeError, match="got an unexpected keyword"):
            algos.factorize(data, order=True)
        with tm.assert_produces_warning(False):
            algos.factorize(data)

    @pytest.mark.parametrize(
        "data",
        [
            np.array([0, 1, 0], dtype="u8"),
            np.array([-(2**63), 1, -(2**63)], dtype="i8"),
            np.array(["__nan__", "foo", "__nan__"], dtype="object"),
        ],
    )
    def test_parametrized_factorize_na_value_default(self, data):
        # [FUNCTION DOC]
        # Parametrized over unsigned, signed and object arrays containing the type's internal default missing marker. Verifies that value is treated as an ordinary value (not missing) when no missing value is specified.
        # arrays that include the NA default for that type, but isn't used.
        codes, uniques = algos.factorize(data)
        expected_uniques = data[[0, 1]]
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize(
        "data, na_value",
        [
            (np.array([0, 1, 0, 2], dtype="u8"), 0),
            (np.array([1, 0, 1, 2], dtype="u8"), 1),
            (np.array([-(2**63), 1, -(2**63), 0], dtype="i8"), -(2**63)),
            (np.array([1, -(2**63), 1, 0], dtype="i8"), 1),
            (np.array(["a", "", "a", "b"], dtype=object), "a"),
            (np.array([(), ("a", 1), (), ("a", 2)], dtype=object), ()),
            (np.array([("a", 1), (), ("a", 1), ("a", 2)], dtype=object), ("a", 1)),
        ],
    )
    def test_parametrized_factorize_na_value(self, data, na_value):
        # [FUNCTION DOC]
        # Parametrized over arrays and a value to be treated as missing (including an empty tuple and tuples in object arrays). Verifies the low-level array factorize gives the sentinel for entries equal to that value and excludes them from the uniques.
        codes, uniques = algos.factorize_array(data, na_value=na_value)
        expected_uniques = data[[1, 3]]
        expected_codes = np.array([-1, 0, -1, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize(
        "data, uniques",
        [
            (
                np.array(["b", "a", None, "b"], dtype=object),
                np.array(["b", "a"], dtype=object),
            ),
            (
                pd.array([2, 1, pd.NA, 2], dtype="Int64"),
                pd.array([2, 1], dtype="Int64"),
            ),
        ],
        ids=["numpy_array", "extension_array"],
    )
    def test_factorize_use_na_sentinel(self, sort, data, uniques):
        # [FUNCTION DOC]
        # Parametrized over the sort fixture and a NumPy object array and a nullable integer array with a missing element. Verifies codes with the sentinel enabled (minus one for missing), and compares uniques.
        # KNOWN ISSUE: the parametrized uniques argument is overwritten by the uniques returned from the call before it is used to build the expected value, so the comparison of uniques is circular (the result is compared with itself or its own sorted version).
        codes, uniques = algos.factorize(data, sort=sort, use_na_sentinel=True)
        if sort:
            expected_codes = np.array([1, 0, -1, 1], dtype=np.intp)
            expected_uniques = algos.safe_sort(uniques)
        else:
            expected_codes = np.array([0, 1, -1, 0], dtype=np.intp)
            expected_uniques = uniques
        tm.assert_numpy_array_equal(codes, expected_codes)
        if isinstance(data, np.ndarray):
            tm.assert_numpy_array_equal(uniques, expected_uniques)
        else:
            tm.assert_extension_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                ["a", None, "b", "a"],
                np.array([0, 1, 2, 0], dtype=np.dtype("intp")),
                np.array(["a", np.nan, "b"], dtype=object),
            ),
            (
                ["a", np.nan, "b", "a"],
                np.array([0, 1, 2, 0], dtype=np.dtype("intp")),
                np.array(["a", np.nan, "b"], dtype=object),
            ),
        ],
    )
    def test_object_factorize_use_na_sentinel_false(
        self, data, expected_codes, expected_uniques
    ):
        # [FUNCTION DOC]
        # Parametrized over lists with None and with NaN. Verifies that with the sentinel disabled, the missing value gets its own code and appears in the uniques as NaN (strict NaN checking).
        codes, uniques = algos.factorize(
            np.array(data, dtype=object), use_na_sentinel=False
        )

        tm.assert_numpy_array_equal(uniques, expected_uniques, strict_nan=True)
        tm.assert_numpy_array_equal(codes, expected_codes, strict_nan=True)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                np.array([1, None, 1, 2], dtype=object),
                np.array([0, 1, 0, 2], dtype=np.dtype("intp")),
                np.array([1, np.nan, 2], dtype="O"),
            ),
            (
                np.array([1, np.nan, 1, 2], dtype=np.float64),
                np.array([0, 1, 0, 2], dtype=np.dtype("intp")),
                np.array([1, np.nan, 2], dtype=np.float64),
            ),
        ],
    )
    def test_int_factorize_use_na_sentinel_false(
        self, data, expected_codes, expected_uniques
    ):
        # [FUNCTION DOC]
        # Parametrized over an object array with None and a float array with NaN. Verifies the missing value gets its own code and appears in the uniques.
        codes, uniques = algos.factorize(data, use_na_sentinel=False)

        tm.assert_numpy_array_equal(uniques, expected_uniques, strict_nan=True)
        tm.assert_numpy_array_equal(codes, expected_codes, strict_nan=True)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                Index(Categorical(["a", "a", "b"])),
                np.array([0, 0, 1], dtype=np.intp),
                CategoricalIndex(["a", "b"], categories=["a", "b"], dtype="category"),
            ),
            (
                Series(Categorical(["a", "a", "b"])),
                np.array([0, 0, 1], dtype=np.intp),
                CategoricalIndex(["a", "b"], categories=["a", "b"], dtype="category"),
            ),
            (
                Series(DatetimeIndex(["2017", "2017"], tz="US/Eastern")),
                np.array([0, 0], dtype=np.intp),
                DatetimeIndex(["2017"], tz="US/Eastern"),
            ),
        ],
    )
    def test_factorize_mixed_values(self, data, expected_codes, expected_uniques):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over a categorical Index, a categorical Series, and a timezone-aware Series. Verifies codes and that uniques are the matching index type with categories or timezone preserved.
        # GH 19721
        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_index_equal(uniques, expected_uniques)

    def test_factorize_interval_non_nano(self, unit):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses the time unit fixture. Verifies factorizing an interval index with datetime, timedelta, and timezone-aware datetime endpoints in a non-nanosecond unit keeps the unit in the categories' dtype.
        # GH#56099
        left = DatetimeIndex(["2016-01-01", np.nan, "2015-10-11"]).as_unit(unit)
        right = DatetimeIndex(["2016-01-02", np.nan, "2015-10-15"]).as_unit(unit)
        idx = IntervalIndex.from_arrays(left, right)
        codes, cats = idx.factorize()
        assert cats.dtype == f"interval[datetime64[{unit}], right]"

        ts = Timestamp(0).as_unit(unit)
        idx2 = IntervalIndex.from_arrays(left - ts, right - ts)
        codes2, cats2 = idx2.factorize()
        assert cats2.dtype == f"interval[timedelta64[{unit}], right]"

        idx3 = IntervalIndex.from_arrays(
            left.tz_localize("US/Pacific"), right.tz_localize("US/Pacific")
        )
        codes3, cats3 = idx3.factorize()
        assert cats3.dtype == f"interval[datetime64[{unit}, US/Pacific], right]"


class TestUnique:
    # [CLASS DOC]
    # Tests for the unique function and related behavior of the Series and Index methods.
    def test_ints(self):
        # [FUNCTION DOC]
        # Verifies unique of random integers returns a NumPy array.
        arr = np.random.default_rng(2).integers(0, 100, size=50)

        result = algos.unique(arr)
        assert isinstance(result, np.ndarray)

    def test_objects(self):
        # [FUNCTION DOC]
        # Verifies unique of random objects returns a NumPy array.
        arr = np.random.default_rng(2).integers(0, 100, size=50).astype("O")

        result = algos.unique(arr)
        assert isinstance(result, np.ndarray)

    def test_object_refcount_bug(self):
        # [FUNCTION DOC]
        # Regression test for a reference-counting bug: repeatedly calls unique on an object array a thousand times to check nothing crashes; there are no assertions.
        lst = np.array(["A", "B", "C", "D", "E"], dtype=object)
        for i in range(1000):
            len(algos.unique(lst))

    def test_index_returned(self, index):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses the index fixture. Verifies unique of an Index with every element doubled returns an Index of the same dtype with the first-occurrence order (a MultiIndex with names for MultiIndex; for timezone-aware data the expected index is normalized), exact equality.
        # GH#57043
        index = index.repeat(2)
        result = algos.unique(index)

        # dict.fromkeys preserves the order
        unique_values = list(dict.fromkeys(index.values))
        if isinstance(index, MultiIndex):
            expected = MultiIndex.from_tuples(unique_values, names=index.names)
        else:
            expected = Index(unique_values, dtype=index.dtype)
            if isinstance(index.dtype, DatetimeTZDtype):
                expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)

    def test_factorize_multiindex_empty(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies factorizing an empty MultiIndex returns empty integer codes and an empty MultiIndex.
        # GH#57517
        mi = MultiIndex.from_product(
            [Index([], name="a", dtype=object), Index([], name="i", dtype="f4")]
        )
        codes, uniques = mi.factorize()
        exp_codes = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp_codes)
        tm.assert_index_equal(uniques, mi[:0])

    def test_dtype_preservation(self, any_numpy_dtype):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses the any-NumPy-dtype fixture. Builds data by dtype kind (integers, floats, complex, boolean, object, strings), takes the unique values of a Series, and verifies equal to the expected array of the same dtype (object for string dtypes); datetime-like results are checked to be datetime or timedelta arrays and converted before comparison.
        # GH 15442
        if any_numpy_dtype in (tm.BYTES_DTYPES + tm.STRING_DTYPES):
            data = [1, 2, 2]
            uniques = [1, 2]
        elif is_integer_dtype(any_numpy_dtype):
            data = [1, 2, 2]
            uniques = [1, 2]
        elif is_float_dtype(any_numpy_dtype):
            data = [1, 2, 2]
            uniques = [1.0, 2.0]
        elif is_complex_dtype(any_numpy_dtype):
            data = [complex(1, 0), complex(2, 0), complex(2, 0)]
            uniques = [complex(1, 0), complex(2, 0)]
        elif is_bool_dtype(any_numpy_dtype):
            data = [True, True, False]
            uniques = [True, False]
        elif is_object_dtype(any_numpy_dtype):
            data = ["A", "B", "B"]
            uniques = ["A", "B"]
        else:
            # datetime64[ns]/M8[ns]/timedelta64[ns]/m8[ns] tested elsewhere
            data = [1, 2, 2]
            uniques = [1, 2]

        result = Series(data, dtype=any_numpy_dtype).unique()
        expected = np.array(uniques, dtype=any_numpy_dtype)

        if any_numpy_dtype in tm.STRING_DTYPES:
            expected = expected.astype(object)

        if expected.dtype.kind in ["m", "M"]:
            # We get TimedeltaArray/DatetimeArray
            assert isinstance(result, (DatetimeArray, TimedeltaArray))
            result = np.array(result)
        tm.assert_numpy_array_equal(result, expected)

    def test_datetime64_dtype_array_returned(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies unique of a DatetimeIndex returns a DatetimeIndex, and of a Series or raw array returns a datetime64 NumPy array.
        # GH 9431
        dt_arr = np.array(
            [
                "2015-01-03T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
            ],
            dtype="M8[ns]",
        )

        dt_index = to_datetime(
            [
                "2015-01-03T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
            ]
        )
        result = algos.unique(dt_index)
        expected = to_datetime(dt_arr)
        tm.assert_index_equal(result, expected, exact=True)

        s = Series(dt_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, dt_arr)
        assert result.dtype == dt_arr.dtype

        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, dt_arr)
        assert result.dtype == dt_arr.dtype

    def test_datetime_non_ns(self):
        # [FUNCTION DOC]
        # Verifies unique of a datetime64 array in seconds unit keeps that unit.
        a = np.array(["2000", "2000", "2001"], dtype="datetime64[s]")
        result = pd.unique(a)
        expected = np.array(["2000", "2001"], dtype="datetime64[s]")
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta_non_ns(self):
        # [FUNCTION DOC]
        # Verifies unique of a timedelta64 array in seconds unit keeps that unit.
        a = np.array(["2000", "2000", "2001"], dtype="timedelta64[s]")
        result = pd.unique(a)
        expected = np.array([2000, 2001], dtype="timedelta64[s]")
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta64_dtype_array_returned(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies unique of a timedelta index returns a timedelta index; of a Series or raw array a timedelta64 NumPy array.
        # GH 9431
        td_arr = np.array([31200, 45678, 10000], dtype="m8[ns]")

        td_index = to_timedelta([31200, 45678, 31200, 10000, 45678])
        result = algos.unique(td_index)
        expected = to_timedelta(td_arr)
        tm.assert_index_equal(result, expected)
        assert result.dtype == expected.dtype

        s = Series(td_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, td_arr)
        assert result.dtype == td_arr.dtype

        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, td_arr)
        assert result.dtype == td_arr.dtype

    def test_uint64_overflow(self):
        # [FUNCTION DOC]
        # Verifies unique of unsigned values at and above two to the sixty-third power.
        s = Series([1, 2, 2**63, 2**63], dtype=np.uint64)
        exp = np.array([1, 2, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(algos.unique(s), exp)

    def test_nan_in_object_array(self):
        # [FUNCTION DOC]
        # Verifies unique of an object array keeps a single NaN in place.
        duplicated_items = ["a", np.nan, "c", "c"]
        result = pd.unique(np.array(duplicated_items, dtype=object))
        expected = np.array(["a", np.nan, "c"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_categorical(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies unique on unordered categoricals returns values in order of appearance, ordered categoricals in category order, for the method, the function and the Series; and an index of categories returns a categorical index.
        # we are expecting to return in the order
        # of appearance
        expected = Categorical(list("bac"))

        # we are expecting to return in the order
        # of the categories
        expected_o = Categorical(list("bac"), categories=list("abc"), ordered=True)

        # GH 15939
        c = Categorical(list("baabc"))
        result = c.unique()
        tm.assert_categorical_equal(result, expected)

        result = algos.unique(c)
        tm.assert_categorical_equal(result, expected)

        c = Categorical(list("baabc"), ordered=True)
        result = c.unique()
        tm.assert_categorical_equal(result, expected_o)

        result = algos.unique(c)
        tm.assert_categorical_equal(result, expected_o)

        # Series of categorical dtype
        s = Series(Categorical(list("baabc")), name="foo")
        result = s.unique()
        tm.assert_categorical_equal(result, expected)

        result = pd.unique(s)
        tm.assert_categorical_equal(result, expected)

        # CI -> return CI
        ci = CategoricalIndex(Categorical(list("baabc"), categories=list("abc")))
        expected = CategoricalIndex(expected)
        result = ci.unique()
        tm.assert_index_equal(result, expected)

        result = pd.unique(ci)
        tm.assert_index_equal(result, expected)

    def test_datetime64tz_aware(self, unit):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses the time unit fixture. Verifies unique of timezone-aware data returns the first element as a datetime array (Series/function) or index (Index/function).
        # GH 15939

        dti = Index(
            [
                Timestamp("20160101", tz="US/Eastern"),
                Timestamp("20160101", tz="US/Eastern"),
            ]
        ).as_unit(unit)
        ser = Series(dti)

        result = ser.unique()
        expected = dti[:1]._data
        tm.assert_extension_array_equal(result, expected)

        result = dti.unique()
        expected = dti[:1]
        tm.assert_index_equal(result, expected)

        result = pd.unique(ser)
        expected = dti[:1]._data
        tm.assert_extension_array_equal(result, expected)

        result = pd.unique(dti)
        expected = dti[:1]
        tm.assert_index_equal(result, expected)

    def test_order_of_appearance(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; documentation examples. Verifies unique returns values in order of appearance for integers, objects and categoricals.
        # 9346
        # light testing of guarantee of order of appearance
        # these also are the doc-examples
        result = pd.unique(Series([2, 1, 3, 3]))
        tm.assert_numpy_array_equal(result, np.array([2, 1, 3], dtype="int64"))

        result = pd.unique(Series([2] + [1] * 5))
        tm.assert_numpy_array_equal(result, np.array([2, 1], dtype="int64"))

        data = np.array(["a", "a", "b", "c"], dtype=object)
        result = pd.unique(data)
        expected = np.array(["a", "b", "c"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        result = pd.unique(Series(Categorical(list("aabc"))))
        expected = Categorical(list("abc"))
        tm.assert_categorical_equal(result, expected)

    def test_order_of_appearance_dt64(self, unit):
        # [FUNCTION DOC]
        # Uses the time unit fixture. Verifies unique of a datetime Series returns a datetime64 NumPy array of the unit.
        ser = Series([Timestamp("20160101"), Timestamp("20160101")]).dt.as_unit(unit)
        result = pd.unique(ser)
        expected = np.array(["2016-01-01T00:00:00.000000000"], dtype=f"M8[{unit}]")
        tm.assert_numpy_array_equal(result, expected)

    def test_order_of_appearance_dt64tz(self, unit):
        # [FUNCTION DOC]
        # Uses the time unit fixture. Verifies unique of a timezone-aware index returns a DatetimeIndex with that dtype and no frequency.
        dti = DatetimeIndex(
            [
                Timestamp("20160101", tz="US/Eastern"),
                Timestamp("20160101", tz="US/Eastern"),
            ]
        ).as_unit(unit)
        result = pd.unique(dti)
        expected = DatetimeIndex(
            ["2016-01-01 00:00:00"], dtype=f"datetime64[{unit}, US/Eastern]", freq=None
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "arg ,expected",
        [
            (("1", "1", "2"), np.array(["1", "2"], dtype=object)),
            (("foo",), np.array(["foo"], dtype=object)),
        ],
    )
    def test_tuple_with_strings(self, arg, expected):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over tuples of strings. Verifies unique treats a tuple of strings as an array of strings after tuple-safe array conversion.
        # see GH 17108
        arg = com.asarray_tuplesafe(arg, dtype=object)
        result = pd.unique(arg)
        tm.assert_numpy_array_equal(result, expected)

    def test_obj_none_preservation(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies None in an object array is preserved by unique.
        # GH 20866
        arr = np.array(["foo", None], dtype=object)
        result = pd.unique(arr)
        expected = np.array(["foo", None], dtype=object)

        tm.assert_numpy_array_equal(result, expected, strict_nan=True)

    def test_signed_zero(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies negative and positive zero are considered equal, keeping the first.
        # GH 21866
        a = np.array([-0.0, 0.0])
        result = pd.unique(a)
        expected = np.array([-0.0])  # 0.0 and -0.0 are equivalent
        tm.assert_numpy_array_equal(result, expected)

    def test_different_nans(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Builds two NaN values with different bit patterns and verifies unique treats them as equivalent, returning a single NaN.
        # GH 21866
        # create different nans from bit-patterns:
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        a = np.array([NAN1, NAN2])  # NAN1 and NAN2 are equivalent
        result = pd.unique(a)
        expected = np.array([np.nan])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("el_type", [np.float64, object])
    def test_first_nan_kept(self, el_type):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over float and object arrays. Verifies unique of two NaNs with different bit patterns keeps the first, by checking the bit pattern.
        # GH 22295
        # create different nans from bit-patterns:
        bits_for_nan1 = 0xFFF8000000000001
        bits_for_nan2 = 0x7FF8000000000001
        NAN1 = struct.unpack("d", struct.pack("=Q", bits_for_nan1))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", bits_for_nan2))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        a = np.array([NAN1, NAN2], dtype=el_type)
        result = pd.unique(a)
        assert result.size == 1
        # use bit patterns to identify which nan was kept:
        result_nan_bits = struct.unpack("=Q", struct.pack("d", result[0]))[0]
        assert result_nan_bits == bits_for_nan1

    def test_do_not_mangle_na_values(self, unique_nulls_fixture, unique_nulls_fixture2):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses two null-value fixtures; returns early if they are the same object. Verifies unique of an object array with two different missing-value objects keeps both and leaves the input elements unchanged (identity).
        # GH 22295
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # skip it, values not unique
        a = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
        result = pd.unique(a)
        assert result.size == 2
        assert a[0] is unique_nulls_fixture
        assert a[1] is unique_nulls_fixture2

    def test_unique_masked(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses the numeric extension dtype fixture. Verifies unique of a masked array with NA returns a masked array of the unique values in order.
        # GH#48019
        ser = Series([1, pd.NA, 2] * 3, dtype=any_numeric_ea_dtype)
        result = pd.unique(ser)
        expected = pd.array([1, pd.NA, 2], dtype=any_numeric_ea_dtype)
        tm.assert_extension_array_equal(result, expected)

    def test_unique_NumpyExtensionArray(self):
        # [FUNCTION DOC]
        # Verifies unique of a NumPy-backed extension array of complex numbers returns the same kind of array with unique values.
        arr_complex = pd.array(
            [1 + 1j, 2, 3]
        )  # NumpyEADtype('complex128') => NumpyExtensionArray
        result = pd.unique(arr_complex)
        expected = pd.array([1 + 1j, 2 + 0j, 3 + 0j])
        tm.assert_extension_array_equal(result, expected)


def test_nunique_ints(index_or_series_or_array):
    # [FUNCTION DOC]
    # Module-level test, regression for a GitHub issue. Uses the index-or-series-or-array fixture. Verifies the integer-only distinct count helper equals the length of unique values.
    # GH#36327
    values = index_or_series_or_array(np.random.default_rng(2).integers(0, 20, 30))
    result = algos.nunique_ints(values)
    expected = len(algos.unique(values))
    assert result == expected


class TestIsin:
    # [CLASS DOC]
    # Tests for the isin function (element-wise membership).
    def test_invalid(self):
        # [FUNCTION DOC]
        # Verifies a non list-like first or second argument raises a type error saying only list-like objects are allowed.
        msg = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a `int`"
        )
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, 1)
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, [1])
        with pytest.raises(TypeError, match=msg):
            algos.isin([1], 1)

    def test_basic(self):
        # [FUNCTION DOC]
        # Verifies membership for integer arrays and Series with list, Series and set values, and for object arrays; a non-matching type gives all false.
        result = algos.isin(np.array([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), Series([1]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), {1})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        arg = np.array(["a", "b"], dtype=object)
        result = algos.isin(arg, ["a"])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series(arg), Series(["a"]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series(arg), {"a"})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arg, [1])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_i8(self):
        # [FUNCTION DOC]
        # Verifies membership of datetime64 and timedelta64 arrays against a list, array and set of their elements.
        arr = date_range("20130101", periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        arr = timedelta_range("1 day", periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype1", ["m8[ns]", "M8[ns]", "M8[ns, UTC]", "period[D]"])
    @pytest.mark.parametrize("dtype", ["i8", "f8", "u8"])
    def test_isin_datetimelike_values_numeric_comps(self, dtype, dtype1):
        # [FUNCTION DOC]
        # Parametrized over datetime-like dtypes (timedelta, datetime, timezone-aware datetime, period) and numeric comparison dtypes (signed, float, unsigned). Verifies that comparing numeric values with datetime-like values gives all false.
        # Anything but object and we get all-False shortcut

        dta = date_range("2013-01-01", periods=3)._values
        arr = Series(dta.view("i8")).array.view(dtype1)

        comps = arr.view("i8").astype(dtype)

        result = algos.isin(comps, arr)
        expected = np.zeros(comps.shape, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_large(self):
        # [FUNCTION DOC]
        # Verifies membership on a two-million-element datetime array finds the two included values.
        s = date_range("20000101", periods=2000000, freq="s").values
        result = algos.isin(s, s[0:2])
        expected = np.zeros(len(s), dtype=bool)
        expected[0] = True
        expected[1] = True
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]", "M8[ns, UTC]", "period[D]"])
    def test_isin_datetimelike_all_nat(self, dtype):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over datetime-like dtypes. Verifies a NaT in the values matches NaT elements, via a list containing only NaT.
        # GH#56427
        dta = date_range("2013-01-01", periods=3)._values
        arr = Series(dta.view("i8")).array.view(dtype)

        arr[0] = NaT
        result = algos.isin(arr, [NaT])
        expected = np.array([True, False, False], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]", "M8[ns, UTC]"])
    def test_isin_datetimelike_strings_returns_false(self, dtype):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over datetime-like dtypes. Verifies string representations of the elements, as a list or as a string array, do not match: all false.
        # GH#53111
        dta = date_range("2013-01-01", periods=3)._values
        arr = Series(dta.view("i8")).array.view(dtype)

        vals = [str(x) for x in arr]
        res = algos.isin(arr, vals)
        assert not res.any()

        vals2 = np.array(vals, dtype=str)
        res2 = algos.isin(arr, vals2)
        assert not res2.any()

    def test_isin_dt64tz_with_nat(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies membership of a timezone-aware array containing NaT with a list containing only NaT matches the NaT element (the all-NaT values used to be inferred as timezone-naive).
        # the all-NaT values used to get inferred to tznaive, which was evaluated
        #  as non-matching GH#56427
        dti = date_range("2016-01-01", periods=3, tz="UTC")
        ser = Series(dti)
        ser[0] = NaT

        res = algos.isin(ser._values, [NaT])
        exp = np.array([True, False, False], dtype=bool)
        tm.assert_numpy_array_equal(res, exp)

    def test_categorical_from_codes(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies membership between two Series of categoricals built from codes and shared categories.
        # GH 16639
        vals = np.array([0, 1, 2, 0])
        cats = ["a", "b", "c"]
        Sd = Series(Categorical([1]).from_codes(vals, cats))
        St = Series(Categorical([1]).from_codes(np.array([0, 1]), cats))
        expected = np.array([True, True, False, True])
        result = algos.isin(Sd, St)
        tm.assert_numpy_array_equal(expected, result)

    def test_categorical_isin(self):
        # [FUNCTION DOC]
        # Verifies membership between categoricals built from codes.
        vals = np.array([0, 1, 2, 0])
        cats = ["a", "b", "c"]
        cat = Categorical([1]).from_codes(vals, cats)
        other = Categorical([1]).from_codes(np.array([0, 1]), cats)

        expected = np.array([True, True, False, True])
        result = algos.isin(cat, other)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies NaN in an object array is found in a list containing NaN, following Python membership semantics.
        # GH 22160
        # nan is special, because from " a is b" doesn't follow "a == b"
        # at least, isin() should follow python's "np.nan in [nan] == True"
        # casting to -> np.float64 -> another float-object somewhere on
        # the way could lead jeopardize this behavior
        comps = np.array([np.nan], dtype=object)  # could be casted to float64
        values = [np.nan]
        expected = np.array([True])
        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in_large(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies membership on a million-element float array with one NaN against NaN and one is all true.
        # https://github.com/pandas-dev/pandas/issues/22205
        s = np.tile(1.0, 1_000_001)
        s[0] = np.nan
        result = algos.isin(s, np.array([np.nan, 1]))
        expected = np.ones(len(s), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_same_nan_is_in_large_series(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies the Series method on a million-element float array against NaN and one is all true.
        # https://github.com/pandas-dev/pandas/issues/22205
        s = np.tile(1.0, 1_000_001)
        series = Series(s)
        s[0] = np.nan
        result = series.isin(np.array([np.nan, 1]))
        expected = Series(np.ones(len(s), dtype=bool))
        tm.assert_series_equal(result, expected)

    def test_same_object_is_in(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Defines a class whose equality is always false but which hashes constantly. Verifies membership falls back to identity: the same object is found, a different instance is not.
        # GH 22160
        # there could be special treatment for nans
        # the user however could define a custom class
        # with similar behavior, then we at least should
        # fall back to usual python's behavior: "a in [a] == True"
        class LikeNan:
            # [CLASS DOC]
            # HELPER: local class imitating NaN: equality always false, constant hash. Used by the membership identity test.
            def __eq__(self, other) -> bool:
                # [FUNCTION DOC]
                # HELPER: always returns false.
                return False

            def __hash__(self):
                # [FUNCTION DOC]
                # HELPER: always returns zero.
                return 0

        a, b = LikeNan(), LikeNan()

        arg = np.array([a], dtype=object)

        # same object -> True
        tm.assert_numpy_array_equal(algos.isin(arg, [a]), np.array([True]))
        # different objects -> False
        tm.assert_numpy_array_equal(algos.isin(arg, [b]), np.array([False]))

    def test_different_nans(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies two distinct NaN objects are treated as equal for membership, in a Python list, an object array and a float array.
        # GH 22160
        # all nans are handled as equivalent

        comps = [float("nan")]
        values = [float("nan")]
        assert comps[0] is not values[0]  # different nan-objects

        # as list of python-objects:
        result = algos.isin(np.array(comps), values)
        tm.assert_numpy_array_equal(np.array([True]), result)

        # as object-array:
        result = algos.isin(
            np.asarray(comps, dtype=object), np.asarray(values, dtype=object)
        )
        tm.assert_numpy_array_equal(np.array([True]), result)

        # as float64-array:
        result = algos.isin(
            np.asarray(comps, dtype=np.float64), np.asarray(values, dtype=np.float64)
        )
        tm.assert_numpy_array_equal(np.array([True]), result)

    def test_no_cast(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies the integer forty-two is not matched against the string "42" (no casting to strings).
        # GH 22160
        # ensure 42 is not casted to a string
        comps = np.array(["ss", 42], dtype=object)
        values = ["42"]
        expected = np.array([False, False])

        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_empty(self, empty):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over an empty list, an empty Series and an empty array. Verifies membership against empty values is all false.
        # see gh-16991
        vals = Index(["a", "b"])
        expected = np.array([False, False])

        result = algos.isin(vals, empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nan_objects(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies in an object array that only a real float NaN matches a NaN value; the string "nan" and a complex NaN do not.
        # GH 22119
        comps = np.array(["nan", np.nan * 1j, float("nan")], dtype=object)
        vals = np.array([float("nan")], dtype=object)
        expected = np.array([False, False, True])
        result = algos.isin(comps, vals)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nans_as_float64(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Builds NaNs with different bit patterns (which would land in different hash buckets) and verifies either matches both in the float array.
        # GH 21866
        # create different nans from bit-patterns,
        # these nans will land in different buckets in the hash-table
        # if no special care is taken
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2

        # check that NAN1 and NAN2 are equivalent:
        arr = np.array([NAN1, NAN2], dtype=np.float64)
        lookup1 = np.array([NAN1], dtype=np.float64)
        result = algos.isin(arr, lookup1)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        lookup2 = np.array([NAN2], dtype=np.float64)
        result = algos.isin(arr, lookup2)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_int_df_string_search(self):
        # [FUNCTION DOC]
        # Verifies a frame of integers does not match the string "1" (docstring: int one is not equal to string one).
        """Comparing df with int`s (1,2) with a string at isin() ("1")
        -> should not match values because int 1 is not equal str 1"""
        df = DataFrame({"values": [1, 2]})
        result = df.isin(["1"])
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_nan_df_string_search(self):
        # [FUNCTION DOC]
        # Verifies a frame with NaN does not match the string "NaN".
        """Comparing df with nan value (np.nan,2) with a string at isin() ("NaN")
        -> should not match values because np.nan is not equal str NaN"""
        df = DataFrame({"values": [np.nan, 2]})
        result = df.isin(np.array(["NaN"], dtype=object))
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_float_df_string_search(self):
        # [FUNCTION DOC]
        # Verifies a frame of floats does not match the string of a float's text.
        """Comparing df with floats (1.4245,2.32441) with a string at isin() ("1.4245")
        -> should not match values because float 1.4245 is not equal str 1.4245"""
        df = DataFrame({"values": [1.4245, 2.32441]})
        result = df.isin(np.array(["1.4245"], dtype=object))
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_unsigned_dtype(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies membership of a large unsigned 64-bit value against a nearby larger value is false (no precision loss through floats).
        # GH#46485
        ser = Series([1378774140726870442], dtype=np.uint64)
        result = ser.isin([1378774140726870528])
        expected = Series(False)
        tm.assert_series_equal(result, expected)


class TestValueCounts:
    # [CLASS DOC]
    # Tests for value counts through the internal function and the Series method.
    def test_value_counts(self):
        # [FUNCTION DOC]
        # Verifies value counts of a binned (cut) array produces counts per interval with an ordered categorical interval index, compared sorted by index.
        arr = np.random.default_rng(1234).standard_normal(4)
        factor = cut(arr, 4)

        # assert isinstance(factor, n)
        result = algos.value_counts_internal(factor)
        breaks = [-1.606, -1.018, -0.431, 0.155, 0.741]
        index = IntervalIndex.from_breaks(breaks).astype(CategoricalDtype(ordered=True))
        expected = Series([1, 0, 2, 1], index=index, name="count")
        tm.assert_series_equal(result.sort_index(), expected.sort_index())

    def test_value_counts_bins(self):
        # [FUNCTION DOC]
        # Verifies the bins argument: one bin covers the full range with a slightly extended left edge; two bins unsorted give two intervals with equal counts.
        s = [1, 2, 3, 4]
        result = algos.value_counts_internal(s, bins=1)
        expected = Series(
            [4], index=IntervalIndex.from_tuples([(0.996, 4.0)]), name="count"
        )
        tm.assert_series_equal(result, expected)

        result = algos.value_counts_internal(s, bins=2, sort=False)
        expected = Series(
            [2, 2],
            index=IntervalIndex.from_tuples([(0.996, 2.5), (2.5, 4.0)]),
            name="count",
        )
        tm.assert_series_equal(result, expected)

    def test_value_counts_dtypes(self):
        # [FUNCTION DOC]
        # Verifies integer one and float one count together; with bins one, the same; an object Series with one, one-point-zero and the string "1" has two groups; and the bins argument with non-numeric data raises a type error saying it only works with numeric data.
        result = algos.value_counts_internal(np.array([1, 1.0]))
        assert len(result) == 1

        result = algos.value_counts_internal(np.array([1, 1.0]), bins=1)
        assert len(result) == 1

        result = algos.value_counts_internal(Series([1, 1.0, "1"]))  # object
        assert len(result) == 2

        msg = "bins argument only works with numeric data"
        with pytest.raises(TypeError, match=msg):
            algos.value_counts_internal(np.array(["1", 1], dtype=object), bins=1)

    def test_value_counts_nat(self):
        # [FUNCTION DOC]
        # Verifies NaT is dropped by default and counted when dropna is false, for timedeltas and datetimes, and the exact result with one remaining entry.
        td = Series([np.timedelta64(10000, "ns"), NaT], dtype="timedelta64[ns]")
        dt = to_datetime(["NaT", "2014-01-01"])

        for ser in [td, dt]:
            vc = algos.value_counts_internal(ser)
            vc_with_na = algos.value_counts_internal(ser, dropna=False)
            assert len(vc) == 1
            assert len(vc_with_na) == 2

        exp_dt = Series({Timestamp("2014-01-01 00:00:00"): 1}, name="count")
        result_dt = algos.value_counts_internal(dt)
        tm.assert_series_equal(result_dt, exp_dt)

        exp_td = Series([1], index=[np.timedelta64(10000, "ns")], name="count")
        result_td = algos.value_counts_internal(td)
        tm.assert_series_equal(result_td, exp_td)

    @pytest.mark.parametrize("dtype", [object, "M8[us]"])
    def test_value_counts_datetime_outofbounds(self, dtype):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over object and microsecond-datetime dtypes. Verifies counts of dates outside the nanosecond range, ordered by count descending, with a matching index dtype.
        # GH 13663
        ser = Series(
            [
                datetime(3000, 1, 1),
                datetime(5000, 1, 1),
                datetime(5000, 1, 1),
                datetime(6000, 1, 1),
                datetime(3000, 1, 1),
                datetime(3000, 1, 1),
            ],
            dtype=dtype,
        )

        res = ser.value_counts()

        exp_index = Index(
            [datetime(3000, 1, 1), datetime(5000, 1, 1), datetime(6000, 1, 1)],
            dtype=dtype,
        )
        exp = Series([3, 2, 1], index=exp_index, name="count")
        tm.assert_series_equal(res, exp)

    def test_categorical(self):
        # [FUNCTION DOC]
        # Verifies value counts of a categorical Series returns a categorical index in sorted-by-count order, and preserves orderedness for an ordered categorical.
        s = Series(Categorical(list("aaabbc")))
        result = s.value_counts()
        expected = Series(
            [3, 2, 1], index=CategoricalIndex(["a", "b", "c"]), name="count"
        )

        tm.assert_series_equal(result, expected, check_index_type=True)

        # preserve order?
        s = s.cat.as_ordered()
        result = s.value_counts()
        expected.index = expected.index.as_ordered()
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_categorical_nans(self):
        # [FUNCTION DOC]
        # Verifies value counts of a categorical with a NaN: dropped by default, and counted last when dropna is false, for unordered and for ordered categories in a custom order.
        s = Series(Categorical(list("aaaaabbbcc")))  # 4,3,2,1 (nan)
        s.iloc[1] = np.nan
        result = s.value_counts()
        expected = Series(
            [4, 3, 2],
            index=CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c"]),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)
        result = s.value_counts(dropna=False)
        expected = Series(
            [4, 3, 2, 1], index=CategoricalIndex(["a", "b", "c", np.nan]), name="count"
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

        # out of order
        s = Series(
            Categorical(list("aaaaabbbcc"), ordered=True, categories=["b", "a", "c"])
        )
        s.iloc[1] = np.nan
        result = s.value_counts()
        expected = Series(
            [4, 3, 2],
            index=CategoricalIndex(
                ["a", "b", "c"],
                categories=["b", "a", "c"],
                ordered=True,
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

        result = s.value_counts(dropna=False)
        expected = Series(
            [4, 3, 2, 1],
            index=CategoricalIndex(
                ["a", "b", "c", np.nan], categories=["b", "a", "c"], ordered=True
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_categorical_zeroes(self):
        # [FUNCTION DOC]
        # Verifies unused categories are kept with a count of zero in the result.
        # keep the `d` category with 0
        s = Series(Categorical(list("bbbaac"), categories=list("abcd"), ordered=True))
        result = s.value_counts()
        expected = Series(
            [3, 2, 1, 0],
            index=Categorical(
                ["b", "a", "c", "d"], categories=list("abcd"), ordered=True
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_value_counts_dropna(self):
        # [FUNCTION DOC]
        # Reference to a GitHub issue comment. Verifies counts with dropna true and false for booleans, booleans with None, floats, and floats with None, including the placement of the missing key and the index dtype.
        # https://github.com/pandas-dev/pandas/issues/9443#issuecomment-73719328

        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=True),
            Series([2, 1], index=[True, False], name="count"),
        )
        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=False),
            Series([2, 1], index=[True, False], name="count"),
        )

        tm.assert_series_equal(
            Series([True] * 3 + [False] * 2 + [None] * 5).value_counts(dropna=True),
            Series([3, 2], index=Index([True, False], dtype=object), name="count"),
        )
        tm.assert_series_equal(
            Series([True] * 5 + [False] * 3 + [None] * 2).value_counts(dropna=False),
            Series([5, 3, 2], index=[True, False, None], name="count"),
        )
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0]).value_counts(dropna=True),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0]).value_counts(dropna=False),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )

        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0, None]).value_counts(dropna=True),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )

        result = Series([10.3, 10.3, 5.0, 5.0, 5.0, None]).value_counts(dropna=False)
        expected = Series([3, 2, 1], index=[5.0, 10.3, None], name="count")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", (np.float64, object, "M8[ns]"))
    def test_value_counts_normalized(self, dtype):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over float, object and datetime dtypes. Verifies normalized counts with a missing value both dropped (0.6 and 0.4) and counted (0.5, 0.3, 0.2) and named proportion.
        # GH12558
        s = Series([1] * 2 + [2] * 3 + [np.nan] * 5)
        s_typed = s.astype(dtype)
        result = s_typed.value_counts(normalize=True, dropna=False)
        expected = Series(
            [0.5, 0.3, 0.2],
            index=Series([np.nan, 2.0, 1.0], dtype=dtype),
            name="proportion",
        )
        tm.assert_series_equal(result, expected)

        result = s_typed.value_counts(normalize=True, dropna=True)
        expected = Series(
            [0.6, 0.4], index=Series([2.0, 1.0], dtype=dtype), name="proportion"
        )
        tm.assert_series_equal(result, expected)

    def test_value_counts_uint64(self):
        # [FUNCTION DOC]
        # Verifies value counts for a single large unsigned value and for an object array holding minus one and a large unsigned value.
        arr = np.array([2**63], dtype=np.uint64)
        expected = Series([1], index=[2**63], name="count")
        result = algos.value_counts_internal(arr)

        tm.assert_series_equal(result, expected)

        arr = np.array([-1, 2**63], dtype=object)
        expected = Series([1, 1], index=[-1, 2**63], name="count")
        result = algos.value_counts_internal(arr)

        tm.assert_series_equal(result, expected)

    def test_value_counts_series(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies value counts with three bins on a float Series with a NaN gives interval counts of two, two and one.
        # GH#54857
        values = np.array([3, 1, 2, 3, 4, np.nan])
        result = Series(values).value_counts(bins=3)
        expected = Series(
            [2, 2, 1],
            index=IntervalIndex.from_tuples(
                [(0.996, 2.0), (2.0, 3.0), (3.0, 4.0)], dtype="interval[float64, right]"
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected)

    def test_value_counts_stability(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies the sorted internal value counts equal a stable descending sort of the unsorted counts, and differ from a quicksort result (an assertion error is expected for the latter comparison).
        # GH 63155
        arr = np.random.default_rng(2).integers(0, 32, 64)
        result = algos.value_counts_internal(arr, sort=True)

        value_counts = Series(arr).value_counts(sort=False)
        expected = value_counts.sort_values(ascending=False, kind="stable")
        tm.assert_series_equal(result, expected)

        unstable_sorted = value_counts.sort_values(ascending=False, kind="quicksort")
        with pytest.raises(AssertionError):
            tm.assert_series_equal(result, unstable_sorted)


class TestDuplicated:
    # [CLASS DOC]
    # Tests for the duplicated function (boolean mask of repeated entries).
    def test_duplicated_with_nas(self):
        # [FUNCTION DOC]
        # Verifies duplicated on an object array with NaNs for the keep options first (default), last and false, and on an object array of tuples containing NaN.
        keys = np.array([0, 1, np.nan, 0, 2, np.nan], dtype=object)

        result = algos.duplicated(keys)
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="first")
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="last")
        expected = np.array([True, False, True, False, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep=False)
        expected = np.array([True, False, True, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        keys = np.empty(8, dtype=object)
        for i, t in enumerate(
            zip([0, 0, np.nan, np.nan] * 2, [0, np.nan, 0, np.nan] * 2, strict=True)
        ):
            keys[i] = t

        result = algos.duplicated(keys)
        falses = [False] * 4
        trues = [True] * 4
        expected = np.array(falses + trues)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="last")
        expected = np.array(trues + falses)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep=False)
        expected = np.array(trues + trues)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "case",
        [
            np.array([1, 2, 1, 5, 3, 2, 4, 1, 5, 6]),
            np.array([1.1, 2.2, 1.1, np.nan, 3.3, 2.2, 4.4, 1.1, np.nan, 6.6]),
            np.array(
                [
                    1 + 1j,
                    2 + 2j,
                    1 + 1j,
                    5 + 5j,
                    3 + 3j,
                    2 + 2j,
                    4 + 4j,
                    1 + 1j,
                    5 + 5j,
                    6 + 6j,
                ]
            ),
            np.array(["a", "b", "a", "e", "c", "b", "d", "a", "e", "f"], dtype=object),
            np.array([1, 2**63, 1, 3**5, 10, 2**63, 39, 1, 3**5, 7], dtype=np.uint64),
        ],
    )
    def test_numeric_object_likes(self, case):
        # [FUNCTION DOC]
        # Parametrized over integer, float with NaN, complex, object and unsigned arrays of ten elements. Verifies the duplicated mask for keep first, last and false from the function, from Index and categorical Index methods and from Series and categorical Series methods, against the shared expected masks.
        exp_first = np.array(
            [False, False, True, False, False, True, False, True, True, False]
        )
        exp_last = np.array(
            [True, True, True, True, False, False, False, False, False, False]
        )
        exp_false = exp_first | exp_last

        res_first = algos.duplicated(case, keep="first")
        tm.assert_numpy_array_equal(res_first, exp_first)

        res_last = algos.duplicated(case, keep="last")
        tm.assert_numpy_array_equal(res_last, exp_last)

        res_false = algos.duplicated(case, keep=False)
        tm.assert_numpy_array_equal(res_false, exp_false)

        # index
        for idx in [Index(case), Index(case, dtype="category")]:
            res_first = idx.duplicated(keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = idx.duplicated(keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = idx.duplicated(keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

        # series
        for s in [Series(case), Series(case, dtype="category")]:
            res_first = s.duplicated(keep="first")
            tm.assert_series_equal(res_first, Series(exp_first))

            res_last = s.duplicated(keep="last")
            tm.assert_series_equal(res_last, Series(exp_last))

            res_false = s.duplicated(keep=False)
            tm.assert_series_equal(res_false, Series(exp_false))

    def test_datetime_likes(self):
        # [FUNCTION DOC]
        # Verifies the same duplicated masks for arrays of timestamps, timezone-aware timestamps, periods, datetime64 values and timedeltas containing NaT, via the function, Index (default, categorical, object) and Series.
        dt = [
            "2011-01-01",
            "2011-01-02",
            "2011-01-01",
            "NaT",
            "2011-01-03",
            "2011-01-02",
            "2011-01-04",
            "2011-01-01",
            "NaT",
            "2011-01-06",
        ]
        td = [
            "1 days",
            "2 days",
            "1 days",
            "NaT",
            "3 days",
            "2 days",
            "4 days",
            "1 days",
            "NaT",
            "6 days",
        ]

        cases = [
            np.array([Timestamp(d) for d in dt]),
            np.array([Timestamp(d, tz="US/Eastern") for d in dt]),
            np.array([Period(d, freq="D") for d in dt]),
            np.array([np.datetime64(d, "ns") for d in dt]),
            np.array([Timedelta(d) for d in td]),
        ]

        exp_first = np.array(
            [False, False, True, False, False, True, False, True, True, False]
        )
        exp_last = np.array(
            [True, True, True, True, False, False, False, False, False, False]
        )
        exp_false = exp_first | exp_last

        for case in cases:
            res_first = algos.duplicated(case, keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = algos.duplicated(case, keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = algos.duplicated(case, keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

            # index
            for idx in [
                Index(case),
                Index(case, dtype="category"),
                Index(case, dtype=object),
            ]:
                res_first = idx.duplicated(keep="first")
                tm.assert_numpy_array_equal(res_first, exp_first)

                res_last = idx.duplicated(keep="last")
                tm.assert_numpy_array_equal(res_last, exp_last)

                res_false = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(res_false, exp_false)

            # series
            for s in [
                Series(case),
                Series(case, dtype="category"),
                Series(case, dtype=object),
            ]:
                res_first = s.duplicated(keep="first")
                tm.assert_series_equal(res_first, Series(exp_first))

                res_last = s.duplicated(keep="last")
                tm.assert_series_equal(res_last, Series(exp_last))

                res_false = s.duplicated(keep=False)
                tm.assert_series_equal(res_false, Series(exp_false))

    @pytest.mark.parametrize("case", [Index([1, 2, 3]), pd.RangeIndex(0, 3)])
    def test_unique_index(self, case):
        # [FUNCTION DOC]
        # Parametrized over an integer Index and a RangeIndex. Verifies uniqueness is true and the duplicated mask is all false.
        assert case.is_unique is True
        tm.assert_numpy_array_equal(case.duplicated(), np.array([False, False, False]))

    @pytest.mark.parametrize(
        "arr, uniques",
        [
            (
                [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)],
                [(0, 0), (0, 1), (1, 0), (1, 1)],
            ),
            (
                [("b", "c"), ("a", "b"), ("a", "b"), ("b", "c")],
                [("b", "c"), ("a", "b")],
            ),
            ([("a", 1), ("b", 2), ("a", 3), ("a", 1)], [("a", 1), ("b", 2), ("a", 3)]),
        ],
    )
    def test_unique_tuples(self, arr, uniques):
        # [FUNCTION DOC]
        # Regression test for GitHub issues. Parametrized over lists of tuples. Verifies unique of a plain list raises a type error stating the accepted input types, while an object array of the tuples gives the unique tuples in order.
        # https://github.com/pandas-dev/pandas/issues/16519
        expected = np.empty(len(uniques), dtype=object)
        expected[:] = uniques

        msg = (
            r"unique requires a Series, Index, ExtensionArray, np.ndarray "
            r"or NumpyExtensionArray got list"
        )
        with pytest.raises(TypeError, match=msg):
            # GH#52986
            pd.unique(arr)

        res = pd.unique(com.asarray_tuplesafe(arr, dtype=object))
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize(
        "array,expected",
        [
            (
                [1 + 1j, 0, 1, 1j, 1 + 2j, 1 + 2j],
                np.array([(1 + 1j), 0j, (1 + 0j), 1j, (1 + 2j)], dtype=complex),
            )
        ],
    )
    def test_unique_complex_numbers(self, array, expected):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies unique of a plain list raises a type error stating the accepted types, and of a complex array gives the unique complex values in order.
        # GH 17927
        msg = (
            r"unique requires a Series, Index, ExtensionArray, np.ndarray "
            r"or NumpyExtensionArray got list"
        )

        with pytest.raises(TypeError, match=msg):
            # GH#52986
            pd.unique(array)

        res = pd.unique(np.array(array))
        tm.assert_numpy_array_equal(res, expected)


class TestHashTable:
    # [CLASS DOC]
    # Tests of the low-level hash table classes.
    @pytest.mark.parametrize(
        "htable, data",
        [
            (
                ht.PyObjectHashTable,
                np.array([f"foo_{i}" for i in range(1000)], dtype=object),
            ),
            (
                ht.StringHashTable,
                np.array([f"foo_{i}" for i in range(1000)], dtype=object),
            ),
            (ht.Float64HashTable, np.arange(1000, dtype=np.float64)),
            (ht.Int64HashTable, np.arange(1000, dtype=np.int64)),
            (ht.UInt64HashTable, np.arange(1000, dtype=np.uint64)),
        ],
    )
    def test_hashtable_unique(self, htable, data, writable):
        # [FUNCTION DOC]
        # Parametrized over object, string, float, signed and unsigned hash tables with 1000 unique values, and the writable fixture. Inserts NaN (float) or several kinds of null (object) into the data, samples three times with replacement, and verifies the table's unique method returns the same as dropping duplicates (first occurrence kept), and with the inverse option returns an inverse so that indexing the uniques reconstructs the input.
        # output of maker has guaranteed unique elements
        s = Series(data, dtype=data.dtype)
        if htable == ht.Float64HashTable:
            # add NaN for float column
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # use different NaN types for object column
            s.loc[500:502] = [np.nan, None, NaT]

        # create duplicated selection
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        s_duplicated.values.setflags(write=writable)

        # drop_duplicates has own cython code (hash_table_func_helper.pxi)
        # and is tested separately; keeps first occurrence like ht.unique()
        expected_unique = s_duplicated.drop_duplicates(keep="first").values
        result_unique = htable().unique(s_duplicated.values)
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # test return_inverse=True
        # reconstruction can only succeed if the inverse is correct
        result_unique, result_inverse = htable().unique(
            s_duplicated.values, return_inverse=True
        )
        tm.assert_numpy_array_equal(result_unique, expected_unique)
        reconstr = result_unique[result_inverse]
        tm.assert_numpy_array_equal(reconstr, s_duplicated.values)

    @pytest.mark.parametrize(
        "htable, data",
        [
            (
                ht.PyObjectHashTable,
                np.array([f"foo_{i}" for i in range(1000)], dtype=object),
            ),
            (
                ht.StringHashTable,
                np.array([f"foo_{i}" for i in range(1000)], dtype=object),
            ),
            (ht.Float64HashTable, np.arange(1000, dtype=np.float64)),
            (ht.Int64HashTable, np.arange(1000, dtype=np.int64)),
            (ht.UInt64HashTable, np.arange(1000, dtype=np.uint64)),
        ],
    )
    def test_hashtable_factorize(self, htable, writable, data):
        # [FUNCTION DOC]
        # Same parametrization as the unique test. Verifies the table's factorize method returns uniques without missing values (equal to dropping nulls then duplicates) and an inverse that reconstructs the non-null input.
        # output of maker has guaranteed unique elements
        s = Series(data, dtype=data.dtype)
        if htable == ht.Float64HashTable:
            # add NaN for float column
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # use different NaN types for object column
            s.loc[500:502] = [np.nan, None, NaT]

        # create duplicated selection
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        s_duplicated.values.setflags(write=writable)
        na_mask = s_duplicated.isna().values

        result_unique, result_inverse = htable().factorize(s_duplicated.values)

        # drop_duplicates has own cython code (hash_table_func_helper.pxi)
        # and is tested separately; keeps first occurrence like ht.factorize()
        # since factorize removes all NaNs, we do the same here
        expected_unique = s_duplicated.dropna().drop_duplicates().values
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # reconstruction can only succeed if the inverse is correct. Since
        # factorize removes the NaNs, those have to be excluded here as well
        result_reconstruct = result_unique[result_inverse[~na_mask]]
        expected_reconstruct = s_duplicated.dropna().values
        tm.assert_numpy_array_equal(result_reconstruct, expected_reconstruct)


class TestRank:
    # [CLASS DOC]
    # Tests for the rank function.
    @pytest.mark.parametrize(
        "arr",
        [
            [np.nan, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 3, np.nan],
            [4.0, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 4.0, np.nan],
        ],
    )
    def test_scipy_compat(self, arr):
        # [FUNCTION DOC]
        # Parametrized over two float arrays with NaN. Skips without scipy. Verifies the low-level one-dimensional rank equals SciPy's average rank with non-finite entries set to NaN.
        sp_stats = pytest.importorskip("scipy.stats")

        arr = np.array(arr)

        mask = ~np.isfinite(arr)
        result = libalgos.rank_1d(arr)
        arr[mask] = np.inf
        exp = sp_stats.rankdata(arr)
        exp[mask] = np.nan
        tm.assert_almost_equal(result, exp)

    def test_basic(self, writable, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Uses the writable and integer dtype fixtures. Verifies ranking a two-element integer Series gives one and two as floats.
        exp = np.array([1, 2], dtype=np.float64)

        data = np.array([1, 100], dtype=any_int_numpy_dtype)
        data.setflags(write=writable)
        ser = Series(data)
        result = algos.rank(ser)
        tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize("dtype", [np.float64, np.uint64])
    def test_uint64_overflow(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over float and unsigned dtypes. Verifies ranking one and two to the sixty-third power gives ranks one and two.
        exp = np.array([1, 2], dtype=np.float64)

        s = Series([1, 2**63], dtype=dtype)
        tm.assert_numpy_array_equal(algos.rank(s), exp)

    @pytest.mark.parametrize("method", ["average", "min", "max"])
    def test_rank_tiny_values(self, method):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over average, min and max. Verifies ranking an object Series of values spanning tiny and huge floats plus NA gives the right ranks and NaN for NA.
        # GH62036: regression test for ranking with tiny float values
        exp = np.array([4.0, 1.0, 3.0, np.nan, 2.0], dtype=np.float64)
        s = Series(
            [5.4954145e29, -9.791984e-21, 9.3715776e-26, pd.NA, 1.8790257e-28],
            dtype="Float64",
        )
        s = s.astype(object)
        result = algos.rank(s, method=method)
        tm.assert_numpy_array_equal(result, exp)

    def test_too_many_ndims(self):
        # [FUNCTION DOC]
        # Verifies ranking a three-dimensional array raises a type error saying arrays with more than two dimensions are not supported.
        arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        msg = "Array with ndim > 2 are not supported"

        with pytest.raises(TypeError, match=msg):
            algos.rank(arr)

    @pytest.mark.single_cpu
    def test_pct_max_many_rows(self):
        # [FUNCTION DOC]
        # Marked single-CPU. Regression test for a GitHub issue. Verifies percentage ranks of arrays with more than sixteen million rows (one-dimensional and two-dimensional) have a maximum of exactly one.
        # GH 18271
        values = np.arange(2**24 + 1)
        result = algos.rank(values, pct=True).max()
        assert result == 1

        values = np.arange(2**25 + 2).reshape(2**24 + 1, 2)
        result = algos.rank(values, pct=True).max()
        assert result == 1


class TestMode:
    # [CLASS DOC]
    # Tests for the mode function and Series method.
    def test_no_mode(self):
        # [FUNCTION DOC]
        # Verifies the mode of an empty float array is an empty float array.
        exp = Series([], dtype=np.float64, index=Index([], dtype=int))
        result, _ = algos.mode(np.array([]))
        tm.assert_numpy_array_equal(result, exp.values)

    def test_mode_single(self, any_real_numpy_dtype):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Uses the real dtype fixture. Verifies the mode of a one-element and of a two-equal-element Series, via function and method.
        # GH 15714
        exp_single = [1]
        data_single = [1]

        exp_multi = [1]
        data_multi = [1, 1]

        ser = Series(data_single, dtype=any_real_numpy_dtype)
        exp = Series(exp_single, dtype=any_real_numpy_dtype)
        result, _ = algos.mode(ser.values)
        tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

        ser = Series(data_multi, dtype=any_real_numpy_dtype)
        exp = Series(exp_multi, dtype=any_real_numpy_dtype)
        result, _ = algos.mode(ser.values)
        tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_mode_obj_int(self):
        # [FUNCTION DOC]
        # Verifies the mode of an integer Series of one element and of an object Series of three distinct strings (all are modes).
        exp = Series([1], dtype=int)
        result, _ = algos.mode(exp.values)
        tm.assert_numpy_array_equal(result, exp.values)

        exp = Series(["a", "b", "c"], dtype=object)
        result, _ = algos.mode(exp.values)
        tm.assert_numpy_array_equal(result, exp.values)

    def test_number_mode(self, any_real_numpy_dtype):
        # [FUNCTION DOC]
        # Uses the real dtype fixture. Verifies a single mode and a tie of two modes (returned in sorted order), via function and method.
        exp_single = [1]
        data_single = [1] * 5 + [2] * 3

        exp_multi = [1, 3]
        data_multi = [1] * 5 + [2] * 3 + [3] * 5

        ser = Series(data_single, dtype=any_real_numpy_dtype)
        exp = Series(exp_single, dtype=any_real_numpy_dtype)
        result, _ = algos.mode(ser.values)
        tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

        ser = Series(data_multi, dtype=any_real_numpy_dtype)
        exp = Series(exp_multi, dtype=any_real_numpy_dtype)
        result, _ = algos.mode(ser.values)
        tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_strobj_mode(self):
        # [FUNCTION DOC]
        # Verifies the mode of a character-typed Series.
        exp = ["b"]
        data = ["a"] * 2 + ["b"] * 3

        ser = Series(data, dtype="c")
        exp = Series(exp, dtype="c")
        result, _ = algos.mode(ser.values)
        tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    @pytest.mark.parametrize("dt", [str, object])
    def test_strobj_multi_char(self, dt, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized over string and object dtypes, with the infer-string fixture. Verifies the mode of multi-character strings; under inferred strings the function returns an extension array, otherwise a NumPy array.
        exp = ["bar"]
        data = ["foo"] * 2 + ["bar"] * 3

        ser = Series(data, dtype=dt)
        exp = Series(exp, dtype=dt)
        result, _ = algos.mode(ser.values)
        if using_infer_string and dt is str:
            tm.assert_extension_array_equal(result, exp.values)
        else:
            tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_datelike_mode(self):
        # [FUNCTION DOC]
        # Verifies the mode of datetime Series with all values tied returns them sorted, and with a tie of two repeated values returns those two.
        exp = Series(["1900-05-03", "2011-01-03", "2013-01-02"], dtype="M8[ns]")
        ser = Series(["2011-01-03", "2013-01-02", "1900-05-03"], dtype="M8[ns]")
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        tm.assert_series_equal(ser.mode(), exp)

        exp = Series(["2011-01-03", "2013-01-02"], dtype="M8[ns]")
        ser = Series(
            ["2011-01-03", "2013-01-02", "1900-05-03", "2011-01-03", "2013-01-02"],
            dtype="M8[ns]",
        )
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_timedelta_mode(self):
        # [FUNCTION DOC]
        # Verifies the same for timedelta Series.
        exp = Series(["-1 days", "0 days", "1 days"], dtype="timedelta64[ns]")
        ser = Series(["1 days", "-1 days", "0 days"], dtype="timedelta64[ns]")
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        tm.assert_series_equal(ser.mode(), exp)

        exp = Series(["2 min", "1 day"], dtype="timedelta64[ns]")
        ser = Series(
            ["1 day", "1 day", "-1 day", "-1 day 2 min", "2 min", "2 min"],
            dtype="timedelta64[ns]",
        )
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_mixed_dtype(self):
        # [FUNCTION DOC]
        # Verifies the mode of a mixed integer and string Series is the repeated string.
        exp = Series(["foo"], dtype=object)
        ser = Series([1, "foo", "foo"])
        result, _ = algos.mode(ser.values)
        tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_uint64_overflow(self):
        # [FUNCTION DOC]
        # Verifies the mode with unsigned values at and above two to the sixty-third power, with a single mode and with a tie.
        exp = Series([2**63], dtype=np.uint64)
        ser = Series([1, 2**63, 2**63], dtype=np.uint64)
        result, _ = algos.mode(ser.values)
        tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

        exp = Series([1, 2**63], dtype=np.uint64)
        ser = Series([1, 2**63], dtype=np.uint64)
        result, _ = algos.mode(ser.values)
        tm.assert_numpy_array_equal(result, exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_categorical(self):
        # [FUNCTION DOC]
        # Verifies mode on categoricals: all tied, a single mode with mixed categories, and a tie that keeps the category set.
        c = Categorical([1, 2])
        exp = c
        res = Series(c).mode()._values
        tm.assert_categorical_equal(res, exp)

        c = Categorical([1, "a", "a"])
        exp = Categorical(["a"], categories=[1, "a"])
        res = Series(c).mode()._values
        tm.assert_categorical_equal(res, exp)

        c = Categorical([1, 1, 2, 3, 3])
        exp = Categorical([1, 3], categories=[1, 2, 3])
        res = Series(c).mode()._values
        tm.assert_categorical_equal(res, exp)

    def test_index(self):
        # [FUNCTION DOC]
        # Verifies the mode function on Index objects of integers (all), mixed (single), with a tie; and that a timedelta index raises an attribute error because the function does not unwrap that index type.
        idx = Index([1, 2, 3])
        exp = Series([1, 2, 3], dtype=np.int64)
        result, _ = algos.mode(idx)
        tm.assert_numpy_array_equal(result, exp.values)

        idx = Index([1, "a", "a"])
        exp = Series(["a"], dtype=object)
        result, _ = algos.mode(idx)
        tm.assert_numpy_array_equal(result, exp.values)

        idx = Index([1, 1, 2, 3, 3])
        exp = Series([1, 3], dtype=np.int64)
        result, _ = algos.mode(idx)
        tm.assert_numpy_array_equal(result, exp.values)

        idx = Index(
            ["1 day", "1 day", "-1 day", "-1 day 2 min", "2 min", "2 min"],
            dtype="timedelta64[ns]",
        )
        with pytest.raises(AttributeError, match="TimedeltaIndex"):
            # algos.mode expects Arraylike, does *not* unwrap TimedeltaIndex
            algos.mode(idx)

    def test_ser_mode_with_name(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies the Series mode keeps the Series name.
        # GH 46737
        ser = Series([1, 1, 3], name="foo")
        result = ser.mode()
        expected = Series([1], name="foo")
        tm.assert_series_equal(result, expected)


class TestDiff:
    # [CLASS DOC]
    # Tests for the low-level diff function.
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_diff_datetimelike_nat(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over datetime64 and timedelta64. Verifies differencing along either axis of a matrix with a NaT column yields timedelta differences, NaT where either operand is NaT (NaT minus NaT is NaT), and NaT in the first row.
        # NaT - NaT is NaT, not 0
        arr = np.arange(12).astype(np.int64).view(dtype).reshape(3, 4)
        arr[:, 2] = arr.dtype.type("NaT", "ns")
        result = algos.diff(arr, 1, axis=0)

        expected = np.ones(arr.shape, dtype="timedelta64[ns]") * 4
        expected[:, 2] = np.timedelta64("NaT", "ns")
        expected[0, :] = np.timedelta64("NaT", "ns")

        tm.assert_numpy_array_equal(result, expected)

        result = algos.diff(arr.T, 1, axis=1)
        tm.assert_numpy_array_equal(result, expected.T)

    def test_diff_ea_axis(self):
        # [FUNCTION DOC]
        # Verifies differencing a datetime extension array along axis one raises a value error saying it cannot diff on axis one.
        dta = date_range("2016-01-01", periods=3, tz="US/Pacific")._data

        msg = "cannot diff DatetimeArray on axis=1"
        with pytest.raises(ValueError, match=msg):
            algos.diff(dta, 1, axis=1)

    @pytest.mark.parametrize("dtype", ["int8", "int16"])
    def test_diff_low_precision_int(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over eight- and sixteen-bit integers. Verifies the diff result is a 32-bit float array with NaN first.
        arr = np.array([0, 1, 1, 0, 0], dtype=dtype)
        result = algos.diff(arr, 1)
        expected = np.array([np.nan, 1, 0, -1, 0], dtype="float32")
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("op", [np.array, pd.array])
def test_union_with_duplicates(op):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over NumPy and pandas array constructors. Verifies the union keeping duplicates of two arrays gives the expected ordering and multiplicities, with the right array type.
    # GH#36289
    lvals = op([3, 1, 3, 4])
    rvals = op([2, 3, 1, 1])
    expected = op([3, 3, 1, 1, 4, 2])
    if isinstance(expected, np.ndarray):
        result = algos.union_with_duplicates(lvals, rvals)
        tm.assert_numpy_array_equal(result, expected)
    else:
        result = algos.union_with_duplicates(lvals, rvals)
        tm.assert_extension_array_equal(result, expected)
