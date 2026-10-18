# [MODULE DOC]
# Tests for the low-level hash table classes and helper routines in the
# compiled hashtable library (aliased as ht): the typed hash tables (object,
# string, float, complex, signed and unsigned integer widths, intp), the
# vector classes used to collect uniques, and the module-level helpers for
# counting values, finding duplicates, membership, mode and unique label
# indices. Many tests are parametrized over every table type and its numpy
# dtype, and use the shared writable fixture to check that read-only input
# buffers are accepted. Also checks NaN handling (all NaN payloads and nested
# NaN-containing tuples hash and compare equal) and that memory allocated by
# the C hash tables is reported to the tracemalloc facility.
# No import-time side effects.

from collections import namedtuple
from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc

import numpy as np
import pytest

from pandas._libs import hashtable as ht

import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin


@contextmanager
def activated_tracemalloc() -> Generator[None, None, None]:
    # [FUNCTION DOC]
    # HELPER: context manager used by the memory-tracking tests. Starts memory
    # allocation tracing on entry, yields nothing, and always stops tracing on
    # exit (even when the body raises). Side effect: toggles global tracemalloc state.
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()


def get_allocated_khash_memory():
    # [FUNCTION DOC]
    # HELPER: used by the tracemalloc tests. Takes a snapshot of traced allocations,
    # keeps only traces in the memory domain used by the hash table C code, and
    # returns the total number of bytes currently allocated in that domain as an
    # integer. Requires tracing to be active.
    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces(
        (tracemalloc.DomainFilter(True, ht.get_hashtable_trace_domain()),)
    )
    return sum(x.size for x in snapshot.traces)


@pytest.mark.parametrize(
    "table_type, dtype",
    [
        (ht.PyObjectHashTable, np.object_),
        (ht.Complex128HashTable, np.complex128),
        (ht.Int64HashTable, np.int64),
        (ht.UInt64HashTable, np.uint64),
        (ht.Float64HashTable, np.float64),
        (ht.Complex64HashTable, np.complex64),
        (ht.Int32HashTable, np.int32),
        (ht.UInt32HashTable, np.uint32),
        (ht.Float32HashTable, np.float32),
        (ht.Int16HashTable, np.int16),
        (ht.UInt16HashTable, np.uint16),
        (ht.Int8HashTable, np.int8),
        (ht.UInt8HashTable, np.uint8),
        (ht.IntpHashTable, np.intp),
    ],
)
class TestHashTable:
    # [CLASS DOC]
    # Tests of the typed hash table classes, parametrized over the table class and
    # a matching numpy dtype (object, complex128/64, int and uint of all widths,
    # float64/32, intp). Each test receives table_type and dtype. Covers item
    # access, key location maps, lookups, unique extraction, memory accounting
    # and preallocation behavior.
    def test_get_set_contains_len(self, table_type, dtype):
        # [FUNCTION DOC]
        # Sets and gets items on a table with a size hint. Checks length and membership
        # start empty, grow when new keys are added, and do not grow when an existing
        # key is overwritten (the value is replaced). Getting a missing key raises a
        # key error mentioning the key.
        index = 5
        table = table_type(55)
        assert len(table) == 0
        assert index not in table

        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42

        table.set_item(index + 1, 41)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41

        table.set_item(index, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 41
        assert index + 2 not in table

        table.set_item(index + 1, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 21

        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    def test_get_set_contains_len_mask(self, table_type, dtype):
        # [FUNCTION DOC]
        # Same as the basic get/set test but for tables created with mask support
        # (skipped for the object table). Checks the NA entry: getting it before it is
        # set raises a key error mentioning NA; after setting it the table contains NA,
        # counts it in the length, and a later set replaces its value.
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        index = 5
        table = table_type(55, uses_mask=True)
        assert len(table) == 0
        assert index not in table

        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42
        with pytest.raises(KeyError, match="NA"):
            table.get_na()

        table.set_item(index + 1, 41)
        table.set_na(41)
        assert pd.NA in table
        assert index in table
        assert index + 1 in table
        assert len(table) == 3
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41
        assert table.get_na() == 41

        table.set_na(21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 3
        assert table.get_item(index + 1) == 41
        assert table.get_na() == 21
        assert index + 2 not in table

        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    def test_map_keys_to_values(self, table_type, dtype, writable):
        # [FUNCTION DOC]
        # Checks the keys-to-values mapping method, which only the 64-bit integer table
        # provides (for other table types the test does nothing). Maps 77 keys to
        # values offset by 77, with the key and value arrays optionally read-only, then
        # verifies each key returns its value.
        # only Int64HashTable has this method
        if table_type == ht.Int64HashTable:
            N = 77
            table = table_type()
            keys = np.arange(N).astype(dtype)
            vals = np.arange(N).astype(np.int64) + N
            keys.flags.writeable = writable
            vals.flags.writeable = writable
            table.map_keys_to_values(keys, vals)
            for i in range(N):
                assert table.get_item(keys[i]) == i + N

    def test_map_locations(self, table_type, dtype, writable):
        # [FUNCTION DOC]
        # Maps 8 keys (offset by 8, possibly read-only array) to their positions and
        # checks each key's stored item equals its position.
        N = 8
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        for i in range(N):
            assert table.get_item(keys[i]) == i

    def test_map_locations_mask(self, table_type, dtype, writable):
        # [FUNCTION DOC]
        # Mask variant of the key-location map (skipped for object). Uses 129 keys
        # (more than 128, required for GH#58924) with only the last entry masked as NA.
        # Non-masked keys map to their positions, looking up the masked key's value
        # raises a key error, and the NA item is the last position.
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        N = 129  # must be > 128 to test GH#58924
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        mask = np.concatenate([np.repeat(False, N - 1), [True]], axis=0)
        table.map_locations(keys, mask)
        for i in range(N - 1):
            assert table.get_item(keys[i]) == i

        with pytest.raises(KeyError, match=re.escape(str(keys[N - 1]))):
            table.get_item(keys[N - 1])

        assert table.get_na() == N - 1

    def test_lookup(self, table_type, dtype, writable):
        # [FUNCTION DOC]
        # Maps three keys to positions, then looks up the same keys. Expects the
        # lookup to return positions zero to two.
        N = 3
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        result = table.lookup(keys)
        expected = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

    def test_lookup_wrong(self, table_type, dtype):
        # [FUNCTION DOC]
        # Looks up keys that were never inserted (the table holds keys offset by N,
        # the lookup uses the unshifted keys). Expects every result to be minus one.
        # N is smaller for 8-bit dtypes so values fit.
        if dtype in (np.int8, np.uint8):
            N = 100
        else:
            N = 512
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        table.map_locations(keys)
        wrong_keys = np.arange(N).astype(dtype)
        result = table.lookup(wrong_keys)
        assert np.all(result == -1)

    def test_lookup_mask(self, table_type, dtype, writable):
        # [FUNCTION DOC]
        # Mask variant of lookup (skipped for object). With the middle key masked,
        # looking up all keys with the same mask returns positions zero to two; looking
        # up an unseen key unmasked returns minus one.
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        N = 3
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        mask = np.array([False, True, False])
        keys.flags.writeable = writable
        table.map_locations(keys, mask)
        result = table.lookup(keys, mask)
        expected = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

        result = table.lookup(np.array([1 + N]).astype(dtype), np.array([False]))
        tm.assert_numpy_array_equal(
            result.astype(np.int64), np.array([-1], dtype=np.int64)
        )

    def test_unique(self, table_type, dtype, writable):
        # [FUNCTION DOC]
        # Builds keys by repeating each of N distinct values five times (N is smaller
        # for 8-bit dtypes) and checks the unique method returns the distinct values
        # in first-seen order.
        if dtype in (np.int8, np.uint8):
            N = 88
        else:
            N = 1000
        table = table_type()
        expected = (np.arange(N) + N).astype(dtype)
        keys = np.repeat(expected, 5)
        keys.flags.writeable = writable
        unique = table.unique(keys)
        tm.assert_numpy_array_equal(unique, expected)

    def test_tracemalloc_works(self, table_type, dtype):
        # [FUNCTION DOC]
        # Checks memory accounting: inside tracing, after mapping 30000 keys (256 for
        # 8-bit dtypes), the bytes tracked in the hash table domain equal the table's
        # reported size, and after deleting the table the tracked bytes drop to zero.
        if dtype in (np.int8, np.uint8):
            N = 256
        else:
            N = 30000
        keys = np.arange(N).astype(dtype)
        with activated_tracemalloc():
            table = table_type()
            table.map_locations(keys)
            used = get_allocated_khash_memory()
            my_size = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_tracemalloc_for_empty(self, table_type, dtype):
        # [FUNCTION DOC]
        # Same memory accounting check as the previous test but for an empty table:
        # tracked bytes equal the reported size and are zero after deletion.
        with activated_tracemalloc():
            table = table_type()
            used = get_allocated_khash_memory()
            my_size = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_get_state(self, table_type, dtype):
        # [FUNCTION DOC]
        # Creates a table with size hint 1000 and checks the state report: size and
        # occupied count are zero and the report includes the bucket count and
        # upper bound entries.
        table = table_type(1000)
        state = table.get_state()
        assert state["size"] == 0
        assert state["n_occupied"] == 0
        assert "n_buckets" in state
        assert "upper_bound" in state

    @pytest.mark.parametrize("N", range(1, 110, 4))
    def test_no_reallocation(self, table_type, dtype, N):
        # [FUNCTION DOC]
        # Parametrized additionally over N (1 to 109 step 4). Checks that a table
        # preallocated for N keys does not need to grow its bucket count when N keys
        # are mapped, and that a default-created table reaches the same bucket count
        # after mapping the same keys.
        keys = np.arange(N).astype(dtype)
        preallocated_table = table_type(N)
        n_buckets_start = preallocated_table.get_state()["n_buckets"]
        preallocated_table.map_locations(keys)
        n_buckets_end = preallocated_table.get_state()["n_buckets"]
        # original number of buckets was enough:
        assert n_buckets_start == n_buckets_end
        # check with clean table (not too much preallocated)
        clean_table = table_type()
        clean_table.map_locations(keys)
        assert n_buckets_start == clean_table.get_state()["n_buckets"]


class TestHashTableUnsorted:
    # [CLASS DOC]
    # Assorted tests moved from the algorithms tests: string table typing, float
    # special values, vector resizing and large size hints.
    # TODO: moved from test_algos; may be redundancies with other tests
    def test_string_hashtable_set_item_signature(self):
        # [FUNCTION DOC]
        # Regression for GH#30419: the string table's set method must reject a
        # non-string key with a type error naming the key argument, and its get method
        # rejects a non-string key with a type error naming the value argument (the
        # message wording refers to the argument as 'val'). A normal string key works.
        # GH#30419 fix typing in StringHashTable.set_item to prevent segfault
        tbl = ht.StringHashTable()

        tbl.set_item("key", 1)
        assert tbl.get_item("key") == 1

        with pytest.raises(TypeError, match="'key' has incorrect type"):
            # key arg typed as string, not object
            tbl.set_item(4, 6)
        with pytest.raises(TypeError, match="'val' has incorrect type"):
            tbl.get_item(4)

    def test_lookup_nan(self, writable):
        # [FUNCTION DOC]
        # Regression for GH#21688: a 64-bit float table maps keys containing NaN,
        # possibly from a read-only array, and lookup returns every position.
        # GH#21688 ensure we can deal with readonly memory views
        xs = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
        xs.setflags(write=writable)
        m = ht.Float64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    def test_add_signed_zeros(self):
        # [FUNCTION DOC]
        # Regression for GH#21866: positive and negative zero must hash to the same
        # bucket, so inserting both leaves a single entry.
        # GH#21866 inconsistent hash-function for float64
        # default hash-function would lead to different hash-buckets
        # for 0.0 and -0.0 if there are more than 2^30 hash-buckets
        # but this would mean 16GB
        N = 4  # 12 * 10**8 would trigger the error, if you have enough memory
        m = ht.Float64HashTable(N)
        m.set_item(0.0, 0)
        m.set_item(-0.0, 0)
        assert len(m) == 1  # 0.0 and -0.0 are equivalent

    def test_add_different_nans(self):
        # [FUNCTION DOC]
        # Regression for GH#21866: two NaN values built from different bit patterns
        # must be treated as the same key, so inserting both leaves one entry.
        # GH#21866 inconsistent hash-function for float64
        # create different nans from bit-patterns:
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        # default hash function would lead to different hash-buckets
        # for NAN1 and NAN2 even if there are only 4 buckets:
        m = ht.Float64HashTable()
        m.set_item(NAN1, 0)
        m.set_item(NAN2, 0)
        assert len(m) == 1  # NAN1 and NAN2 are equivalent

    def test_lookup_overflow(self, writable):
        # [FUNCTION DOC]
        # Regression for GH#21688: an unsigned 64-bit table with a value above the
        # signed range, from a possibly read-only array, maps and looks up all
        # positions correctly.
        xs = np.array([1, 2, 2**63], dtype=np.uint64)
        # GH 21688 ensure we can deal with readonly memory views
        xs.setflags(write=writable)
        m = ht.UInt64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    @pytest.mark.parametrize("nvals", [0, 10])  # resizing to 0 is special case
    @pytest.mark.parametrize(
        "htable, uniques, dtype, safely_resizes",
        [
            (ht.PyObjectHashTable, ht.ObjectVector, "object", False),
            (ht.StringHashTable, ht.ObjectVector, "object", True),
            (ht.Float64HashTable, ht.Float64Vector, "float64", False),
            (ht.Int64HashTable, ht.Int64Vector, "int64", False),
            (ht.Int32HashTable, ht.Int32Vector, "int32", False),
            (ht.UInt64HashTable, ht.UInt64Vector, "uint64", False),
        ],
    )
    def test_vector_resize(
        self, writable, htable, uniques, dtype, safely_resizes, nvals
    ):
        # [FUNCTION DOC]
        # Checks memory safety of the vector classes used for collecting uniques
        # (GH#7157). Parametrized over writability, nvals (0 or 10), and several
        # (table, vector, dtype, safely_resizes) combinations. Gets labels for a prefix
        # of 1000 values, converts the vector to an array (which marks an external
        # view), then gets labels again for all values: for the string table with an
        # object vector this succeeds, otherwise it must raise a value error mentioning
        # an external reference. Converting again must not raise and the earlier array
        # shape is unchanged.
        # Test for memory errors after internal vector
        # reallocations (GH 7157)
        # Changed from using np.random.default_rng(2).rand to range
        # which could cause flaky CI failures when safely_resizes=False
        vals = np.array(range(1000), dtype=dtype)

        # GH 21688 ensures we can deal with read-only memory views
        vals.setflags(write=writable)

        # initialise instances; cannot initialise in parametrization,
        # as otherwise external views would be held on the array (which is
        # one of the things this test is checking)
        htable = htable()
        uniques = uniques()

        # get_labels may append to uniques
        htable.get_labels(vals[:nvals], uniques, 0, -1)
        # to_array() sets an external_view_exists flag on uniques.
        tmp = uniques.to_array()
        oldshape = tmp.shape

        # subsequent get_labels() calls can no longer append to it
        # (except for StringHashTables + ObjectVector)
        if safely_resizes:
            htable.get_labels(vals, uniques, 0, -1)
        else:
            with pytest.raises(ValueError, match="external reference.*"):
                htable.get_labels(vals, uniques, 0, -1)

        uniques.to_array()  # should not raise here
        assert tmp.shape == oldshape

    @pytest.mark.parametrize(
        "hashtable",
        [
            ht.PyObjectHashTable,
            ht.StringHashTable,
            ht.Float64HashTable,
            ht.Int64HashTable,
            ht.Int32HashTable,
            ht.UInt64HashTable,
        ],
    )
    def test_hashtable_large_sizehint(self, hashtable):
        # [FUNCTION DOC]
        # Smoke test for GH#22729: constructing each listed table type with a size hint
        # larger than the 32-bit unsigned maximum must not raise.
        # GH#22729 smoketest for not raising when passing a large size_hint
        size_hint = np.iinfo(np.uint32).max + 1
        hashtable(size_hint=size_hint)


class TestPyObjectHashTableWithNans:
    # [CLASS DOC]
    # Tests of the object hash table treating NaN-containing keys (floats,
    # complex numbers, tuples, named tuples) as equal even when they are distinct
    # objects.
    def test_nan_float(self):
        # [FUNCTION DOC]
        # Two distinct float NaN objects: setting one and getting with the other
        # returns the stored value.
        nan1 = float("nan")
        nan2 = float("nan")
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_complex_both(self):
        # [FUNCTION DOC]
        # Two distinct complex numbers with NaN real and imaginary parts are the same key.
        nan1 = complex(float("nan"), float("nan"))
        nan2 = complex(float("nan"), float("nan"))
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_complex_real(self):
        # [FUNCTION DOC]
        # Complex numbers with NaN real part and equal imaginary part match each other;
        # a complex number with a different imaginary part is a missing key and raises
        # a key error showing its representation.
        nan1 = complex(float("nan"), 1)
        nan2 = complex(float("nan"), 1)
        other = complex(float("nan"), 2)
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_complex_imag(self):
        # [FUNCTION DOC]
        # Complex numbers with NaN imaginary part and equal real part match each other;
        # one with a different real part raises a key error showing its representation.
        nan1 = complex(1, float("nan"))
        nan2 = complex(1, float("nan"))
        other = complex(2, float("nan"))
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_in_tuple(self):
        # [FUNCTION DOC]
        # Tuples containing a NaN match even though the NaN objects differ.
        nan1 = (float("nan"),)
        nan2 = (float("nan"),)
        assert nan1[0] is not nan2[0]
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_in_nested_tuple(self):
        # [FUNCTION DOC]
        # Nested tuples containing a NaN match; a different tuple raises a key error
        # showing its representation.
        nan1 = (1, (2, (float("nan"),)))
        nan2 = (1, (2, (float("nan"),)))
        other = (1, 2)
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_in_namedtuple(self):
        # [FUNCTION DOC]
        # Named tuples holding NaN match each other even with distinct NaN objects.
        T = namedtuple("T", ["x"])
        nan1 = T(float("nan"))
        nan2 = T(float("nan"))
        assert nan1.x is not nan2.x
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_in_nested_namedtuple(self):
        # [FUNCTION DOC]
        # Nested named tuples holding NaN match; a different named tuple raises a key
        # error showing its representation.
        T = namedtuple("T", ["x", "y"])
        nan1 = T(1, (2, (float("nan"),)))
        nan2 = T(1, (2, (float("nan"),)))
        other = T(1, 2)
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)


def test_hash_equal_tuple_with_nans():
    # [FUNCTION DOC]
    # Checks that the hash function for objects and the equality helper treat two
    # nested tuples containing NaN values as the same.
    a = (float("nan"), (float("nan"), float("nan")))
    b = (float("nan"), (float("nan"), float("nan")))
    assert ht.object_hash(a) == ht.object_hash(b)
    assert ht.objects_are_equal(a, b)


def test_hash_equal_namedtuple_with_nans():
    # [FUNCTION DOC]
    # Same as the previous test but for named tuples with nested NaN tuples.
    T = namedtuple("T", ["x", "y"])
    a = T(float("nan"), (float("nan"), float("nan")))
    b = T(float("nan"), (float("nan"), float("nan")))
    assert ht.object_hash(a) == ht.object_hash(b)
    assert ht.objects_are_equal(a, b)


def test_hash_equal_namedtuple_and_tuple():
    # [FUNCTION DOC]
    # Checks that a named tuple and a plain tuple with the same contents have the
    # same object hash and compare equal under the object equality helper.
    T = namedtuple("T", ["x", "y"])
    a = T(1, (2, 3))
    b = (1, (2, 3))
    assert ht.object_hash(a) == ht.object_hash(b)
    assert ht.objects_are_equal(a, b)


def test_get_labels_groupby_for_Int64(writable):
    # [FUNCTION DOC]
    # Takes the writable fixture. Calls the grouped-label method of the 64-bit
    # integer table on values containing negatives. Expects labels of first-seen
    # order with negative values labeled minus one, and uniques of the nonnegative
    # distinct values.
    table = ht.Int64HashTable()
    vals = np.array([1, 2, -1, 2, 1, -1], dtype=np.int64)
    vals.flags.writeable = writable
    arr, unique = table.get_labels_groupby(vals)
    expected_arr = np.array([0, 1, -1, 1, 0, -1], dtype=np.intp)
    expected_unique = np.array([1, 2], dtype=np.int64)
    tm.assert_numpy_array_equal(arr, expected_arr)
    tm.assert_numpy_array_equal(unique, expected_unique)


def test_tracemalloc_works_for_StringHashTable():
    # [FUNCTION DOC]
    # Memory accounting for the string table after mapping 1000 string keys:
    # tracked bytes equal reported size, and are zero after deletion.
    N = 1000
    keys = np.arange(N).astype(np.str_).astype(np.object_)
    with activated_tracemalloc():
        table = ht.StringHashTable()
        table.map_locations(keys)
        used = get_allocated_khash_memory()
        my_size = table.sizeof()
        assert used == my_size
        del table
        assert get_allocated_khash_memory() == 0


def test_tracemalloc_for_empty_StringHashTable():
    # [FUNCTION DOC]
    # Memory accounting for an empty string table: tracked bytes equal reported size
    # and zero after deletion.
    with activated_tracemalloc():
        table = ht.StringHashTable()
        used = get_allocated_khash_memory()
        my_size = table.sizeof()
        assert used == my_size
        del table
        assert get_allocated_khash_memory() == 0


@pytest.mark.parametrize("N", range(1, 110, 4))
def test_no_reallocation_StringHashTable(N):
    # [FUNCTION DOC]
    # Parametrized over N. A string table preallocated for N keys keeps its bucket
    # count when N keys are mapped, and equals the bucket count of a default table
    # after mapping the same keys.
    keys = np.arange(N).astype(np.str_).astype(np.object_)
    preallocated_table = ht.StringHashTable(N)
    n_buckets_start = preallocated_table.get_state()["n_buckets"]
    preallocated_table.map_locations(keys)
    n_buckets_end = preallocated_table.get_state()["n_buckets"]
    # original number of buckets was enough:
    assert n_buckets_start == n_buckets_end
    # check with clean table (not too much preallocated)
    clean_table = ht.StringHashTable()
    clean_table.map_locations(keys)
    assert n_buckets_start == clean_table.get_state()["n_buckets"]


@pytest.mark.parametrize(
    "table_type, dtype",
    [
        (ht.Float64HashTable, np.float64),
        (ht.Float32HashTable, np.float32),
        (ht.Complex128HashTable, np.complex128),
        (ht.Complex64HashTable, np.complex64),
    ],
)
class TestHashTableWithNans:
    # [CLASS DOC]
    # Tests of the float and complex hash tables for NaN keys, parametrized over
    # table class and dtype.
    def test_get_set_contains_len(self, table_type, dtype):
        # [FUNCTION DOC]
        # Setting a NaN key makes the table contain it with one entry; setting NaN again
        # replaces the value and keeps length one.
        index = float("nan")
        table = table_type()
        assert index not in table

        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42

        table.set_item(index, 41)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 41

    def test_map_locations(self, table_type, dtype):
        # [FUNCTION DOC]
        # Maps an array of ten NaN keys; the table has a single entry whose position is
        # the last index.
        N = 10
        table = table_type()
        keys = np.full(N, np.nan, dtype=dtype)
        table.map_locations(keys)
        assert len(table) == 1
        assert table.get_item(np.nan) == N - 1

    def test_unique(self, table_type, dtype):
        # [FUNCTION DOC]
        # The unique method over an array of 1020 NaN values returns exactly one value
        # and it is NaN.
        N = 1020
        table = table_type()
        keys = np.full(N, np.nan, dtype=dtype)
        unique = table.unique(keys)
        assert np.all(np.isnan(unique)) and len(unique) == 1


def test_unique_for_nan_objects_floats():
    # [FUNCTION DOC]
    # Unique on an object array of 50 distinct NaN float objects yields one value.
    table = ht.PyObjectHashTable()
    keys = np.array([float("nan") for i in range(50)], dtype=np.object_)
    unique = table.unique(keys)
    assert len(unique) == 1


def test_unique_for_nan_objects_complex():
    # [FUNCTION DOC]
    # Unique on an object array of 50 distinct complex numbers with NaN real part
    # yields one value.
    table = ht.PyObjectHashTable()
    keys = np.array([complex(float("nan"), 1.0) for i in range(50)], dtype=np.object_)
    unique = table.unique(keys)
    assert len(unique) == 1


def test_unique_for_nan_objects_tuple():
    # [FUNCTION DOC]
    # Unique on an object array of an integer followed by 50 equal tuples containing
    # a NaN yields two values.
    table = ht.PyObjectHashTable()
    keys = np.array(
        [1] + [(1.0, (float("nan"), 1.0)) for i in range(50)], dtype=np.object_
    )
    unique = table.unique(keys)
    assert len(unique) == 2


@pytest.mark.parametrize(
    "dtype",
    [
        np.object_,
        np.complex128,
        np.int64,
        np.uint64,
        np.float64,
        np.complex64,
        np.int32,
        np.uint32,
        np.float32,
        np.int16,
        np.uint16,
        np.int8,
        np.uint8,
        np.intp,
    ],
)
class TestHelpFunctions:
    # [CLASS DOC]
    # Tests of module-level hash helpers (count, duplicated, membership, mode),
    # parametrized over a dtype list that includes object, complex, integer and
    # float types.
    def test_value_count(self, dtype, writable):
        # [FUNCTION DOC]
        # Counts occurrences of 43 distinct values repeated five times (possibly
        # read-only). Expects keys equal to the distinct values (after sorting) and
        # every count equal to five.
        N = 43
        expected = (np.arange(N) + N).astype(dtype)
        values = np.repeat(expected, 5)
        values.flags.writeable = writable
        keys, counts, _ = ht.value_count(values, False)
        tm.assert_numpy_array_equal(np.sort(keys), expected)
        assert np.all(counts == 5)

    def test_value_count_mask(self, dtype):
        # [FUNCTION DOC]
        # Mask variant (skipped for object): with two of five identical values masked,
        # counting gives two keys and an NA counter of two.
        if dtype == np.object_:
            pytest.skip("mask not implemented for object dtype")
        values = np.array([1] * 5, dtype=dtype)
        mask = np.zeros((5,), dtype=np.bool_)
        mask[1] = True
        mask[4] = True
        keys, counts, na_counter = ht.value_count(values, False, mask=mask)
        assert len(keys) == 2
        assert na_counter == 2

    def test_value_count_stable(self, dtype, writable):
        # [FUNCTION DOC]
        # Regression for GH12679: counting distinct values keeps first-seen order, so
        # the keys equal the input and all counts are one.
        # GH12679
        values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        values.flags.writeable = writable
        keys, counts, _ = ht.value_count(values, False)
        tm.assert_numpy_array_equal(keys, values)
        assert np.all(counts == 1)

    def test_duplicated_first(self, dtype, writable):
        # [FUNCTION DOC]
        # The duplicated helper over values repeated five times marks the first
        # occurrence of each value as not duplicated and all others as duplicated.
        N = 100
        values = np.repeat(np.arange(N).astype(dtype), 5)
        values.flags.writeable = writable
        result = ht.duplicated(values)
        expected = np.ones_like(values, dtype=np.bool_)
        expected[::5] = False
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(self, dtype, writable):
        # [FUNCTION DOC]
        # Membership helper where all values are in the comparison set: result is all true.
        N = 127
        arr = np.arange(N).astype(dtype)
        values = np.arange(N).astype(dtype)
        arr.flags.writeable = writable
        values.flags.writeable = writable
        result = ht.ismember(arr, values)
        expected = np.ones_like(values, dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(self, dtype):
        # [FUNCTION DOC]
        # Membership helper where none are in the comparison set: result is all false.
        N = 17
        arr = np.arange(N).astype(dtype)
        values = (np.arange(N) + N).astype(dtype)
        result = ht.ismember(arr, values)
        expected = np.zeros_like(values, dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(self, dtype, writable):
        # [FUNCTION DOC]
        # Mode helper with one value forced to 42 among values repeated five times
        # (array length depends on dtype size): the first returned mode is 42.
        if dtype in (np.int8, np.uint8):
            N = 53
        else:
            N = 11111
        values = np.repeat(np.arange(N).astype(dtype), 5)
        values[0] = 42
        values.flags.writeable = writable
        result = ht.mode(values, False)[0]
        assert result == 42

    def test_mode_stable(self, dtype, writable):
        # [FUNCTION DOC]
        # Mode helper with all distinct values returns them in first-seen order.
        values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        values.flags.writeable = writable
        keys = ht.mode(values, False)[0]
        tm.assert_numpy_array_equal(keys, values)


def test_modes_with_nans():
    # [FUNCTION DOC]
    # Regression for GH42688: mode of an object array with different null
    # representations (NA, NaN, NaT, None repeated, plus True) keeps them distinct,
    # yielding as many modes as null kinds.
    # GH42688, nans aren't mangled
    nulls = [pd.NA, np.nan, pd.NaT, None]
    values = np.array([True] + nulls * 2, dtype=np.object_)
    modes = ht.mode(values, False)[0]
    assert modes.size == len(nulls)


def test_unique_label_indices_intp(writable):
    # [FUNCTION DOC]
    # Takes the writable fixture. The unique label indices helper over intp labels
    # returns the first position of each distinct label.
    keys = np.array([1, 2, 2, 2, 1, 3], dtype=np.intp)
    keys.flags.writeable = writable
    result = ht.unique_label_indices(keys)
    expected = np.array([0, 1, 5], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


def test_unique_label_indices():
    # [FUNCTION DOC]
    # Compares the unique label indices helper against numpy's unique first-index
    # result for random labels (fixed seed), then sets some labels to minus one and
    # checks the helper skips negative labels (dropping the first entry from numpy's
    # result). Dtype is not checked.
    a = np.random.default_rng(2).integers(1, 1 << 10, 1 << 15).astype(np.intp)

    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1]

    tm.assert_numpy_array_equal(left, right, check_dtype=False)

    a[np.random.default_rng(2).choice(len(a), 10)] = -1
    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1][1:]
    tm.assert_numpy_array_equal(left, right, check_dtype=False)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float64,
        np.float32,
        np.complex128,
        np.complex64,
    ],
)
class TestHelpFunctionsWithNans:
    # [CLASS DOC]
    # Tests of module-level helpers on NaN data for float and complex dtypes.
    def test_value_count(self, dtype):
        # [FUNCTION DOC]
        # Counting three NaN values: dropping NaN yields no keys; keeping NaN yields one
        # key with a count of three.
        values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        keys, counts, _ = ht.value_count(values, True)
        assert len(keys) == 0
        keys, counts, _ = ht.value_count(values, False)
        assert len(keys) == 1 and np.all(np.isnan(keys))
        assert counts[0] == 3

    def test_duplicated_first(self, dtype):
        # [FUNCTION DOC]
        # Duplicated on three NaN values: only the first is not a duplicate.
        values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        result = ht.duplicated(values)
        expected = np.array([False, True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(self, dtype):
        # [FUNCTION DOC]
        # Membership of NaN values against a NaN comparison set: all true.
        arr = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        values = np.array([np.nan, np.nan], dtype=dtype)
        result = ht.ismember(arr, values)
        expected = np.array([True, True, True], dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(self, dtype):
        # [FUNCTION DOC]
        # Membership of NaN values against a set without NaN: all false.
        arr = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        values = np.array([1], dtype=dtype)
        result = ht.ismember(arr, values)
        expected = np.array([False, False, False], dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(self, dtype):
        # [FUNCTION DOC]
        # Mode with NaN skipped returns 42; with NaN included returns NaN.
        values = np.array([42, np.nan, np.nan, np.nan], dtype=dtype)
        assert ht.mode(values, True)[0] == 42
        assert np.isnan(ht.mode(values, False)[0])


def test_ismember_tuple_with_nans():
    # [FUNCTION DOC]
    # Regression for GH-41836: the algorithms isin function matches tuples that
    # contain NaN, returning true for the matching tuple and false for the other.
    # GH-41836
    values = np.empty(2, dtype=object)
    values[:] = [("a", float("nan")), ("b", 1)]
    comps = [("a", float("nan"))]

    result = isin(values, comps)
    expected = np.array([True, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)


def test_float_complex_int_are_equal_as_objects():
    # [FUNCTION DOC]
    # Checks that with a comparison array of 129 integers, the object values string,
    # integer 5, float 5.0 and complex 5 are respectively not found, and the other
    # three found, i.e. numerically equal numbers match as objects.
    values = ["a", 5, 5.0, 5.0 + 0j]
    comps = list(range(129))
    result = isin(np.array(values, dtype=object), np.asarray(comps))
    expected = np.array([False, True, True, True], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)
