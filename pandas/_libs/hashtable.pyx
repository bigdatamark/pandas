# [MODULE DOC]
# Hash table module of pandas (pandas._libs.hashtable). Provides khash-backed
# typed hash tables mapping keys to integer positions (one class per numeric dtype
# plus Python-object and str tables), growable typed Vectors used to collect
# unique values, Factorizer classes that incrementally assign integer codes to
# values, and module-level algorithms (unique, factorize helpers, value_counts,
# duplicated, isin, mode, ismember, label indices) generated per dtype from the
# Tempita templates hashtable_class_helper.pxi.in and hashtable_func_helper.pxi.in
# which are included here. Missing-value semantics: NaN equals NaN in all tables;
# optional masks (True = missing) can replace NaN detection for masked arrays.
# Defines SIZE_HINT_LIMIT (about 1M buckets) capping preallocation, the initial
# vector capacity (128), and the intp-dependent aliases IntpHashTable and
# unique_label_indices (raises ValueError at import on unsupported intp sizes).
cimport cython
from cpython.ref cimport (
    Py_INCREF,
    PyObject,
)
from libc.stdlib cimport (
    free,
    malloc,
)
from libc.string cimport memcpy

import numpy as np

cimport numpy as cnp
from numpy cimport ndarray

cnp.import_array()


from pandas._libs.dtypes cimport numeric_object_t
from pandas._libs.khash cimport (
    KHASH_TRACE_DOMAIN,
    are_equivalent_float32_t,
    are_equivalent_float64_t,
    are_equivalent_khcomplex64_t,
    are_equivalent_khcomplex128_t,
    kh_needed_n_buckets,
    kh_python_hash_equal,
    kh_python_hash_func,
    khiter_t,
)
from pandas._libs.missing cimport (
    checknull,
    is_matching_na,
)


def get_hashtable_trace_domain():
    # [FUNCTION DOC]
    # Return the integer tracemalloc domain id (424242) under which khash memory
    # allocations are tracked, so tests/users can filter tracemalloc snapshots.
    # Inputs: none. Outputs: int. Exceptions: none. Side effects: none.
    return KHASH_TRACE_DOMAIN


def object_hash(obj):
    # [FUNCTION DOC]
    # Compute the hash pandas object hash tables use for obj (NaN-aware: all float
    # NaNs hash alike; tuples hashed element-wise recursively; 64-bit hashes folded
    # to 32 bits). Inputs: obj - any Python object. Outputs: unsigned 32-bit int.
    # Exceptions: none; unhashable objects silently hash to 0. Side effects: none.
    return kh_python_hash_func(obj)


def objects_are_equal(a, b):
    # [FUNCTION DOC]
    # Report whether pandas object hash tables would treat a and b as the same key
    # (identity, or NaN == NaN for exact floats/complexes, element-wise for tuples,
    # else Python ==). Inputs: a, b - any objects. Outputs: bool.
    # Exceptions: none; errors raised by == are swallowed and give False.
    # Side effects: none.
    return kh_python_hash_equal(a, b)


SIZE_HINT_LIMIT = (1 << 20) + 7


cdef Py_ssize_t _INIT_VEC_CAP = 128

include "hashtable_class_helper.pxi"
include "hashtable_func_helper.pxi"


# map derived hash-map types onto basic hash-map types:
if np.dtype(np.intp) == np.dtype(np.int64):
    IntpHashTable = Int64HashTable
    unique_label_indices = _unique_label_indices_int64
elif np.dtype(np.intp) == np.dtype(np.int32):
    IntpHashTable = Int32HashTable
    unique_label_indices = _unique_label_indices_int32
else:
    raise ValueError(np.dtype(np.intp))


cdef class Factorizer:
    # [CLASS DOC]
    # Abstract base for stateful factorizers: objects that repeatedly map batches of
    # values to integer codes, remembering codes assigned in earlier batches. count is
    # the number of distinct (non-missing) values seen so far (read-only attribute).
    # Subclasses hold a typed hash table and a uniques Vector.
    cdef readonly:
        Py_ssize_t count

    def __cinit__(self, size_hint: int, uses_mask: bool = False):
        # [FUNCTION DOC]
        # Initialise count to 0. Inputs: size_hint - expected number of uniques (unused
        # here, used by subclasses); uses_mask - whether masks will be supplied (unused
        # here). Outputs: none. Side effects: sets self.count.
        self.count = 0

    def get_count(self) -> int:
        # [FUNCTION DOC]
        # Return the number of distinct values factorized so far (self.count).
        # Inputs: none. Outputs: int. Side effects: none.
        return self.count

    def factorize(self, values, na_sentinel=-1, na_value=None, mask=None) -> np.ndarray:
        # [FUNCTION DOC]
        # Abstract: subclasses map values to integer codes. Always raises
        # NotImplementedError here. Inputs: values, na_sentinel, na_value, mask.
        raise NotImplementedError

    def hash_inner_join(self, values, mask=None):
        # [FUNCTION DOC]
        # Abstract: subclasses compute inner-join positions against the table. Always
        # raises NotImplementedError here. Inputs: values, mask.
        raise NotImplementedError


cdef class ObjectFactorizer(Factorizer):
    # [CLASS DOC]
    # Factorizer for object-dtype arrays. Attributes (public): table - a
    # PyObjectHashTable mapping each seen value to its code; uniques - ObjectVector of
    # the distinct values in order of first appearance.
    cdef public:
        PyObjectHashTable table
        ObjectVector uniques

    def __cinit__(self, size_hint: int, uses_mask: bool = False):
        # [FUNCTION DOC]
        # Create the PyObjectHashTable (presized by size_hint) and an empty ObjectVector
        # of uniques. Inputs: size_hint - int; uses_mask - ignored (masks unsupported).
        # Outputs: none. Side effects: allocates the hash table.
        self.table = PyObjectHashTable(size_hint)
        self.uniques = ObjectVector()

    def factorize(
        self, ndarray[object] values, na_sentinel=-1, na_value=None, mask=None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Assign codes to an object array, continuing numbering from previous calls.
        # If the current uniques vector has been exposed (to_array was called) it is
        # first copied into a fresh ObjectVector so that it can keep growing. Then the
        # table's get_labels is run with count_prior = self.count, which hashes each
        # value, gives new values the next code and appends them to uniques, and gives
        # missing values (checknull, or equal to na_value) the code na_sentinel.
        # Inputs: values - 1-D object ndarray; na_sentinel - int code for missing
        # values;
        # na_value - optional extra value treated as missing; mask - must be None.
        # Outputs: 1-D intp ndarray of codes, same length as values.
        # Exceptions: NotImplementedError if mask is given.
        # Side effects: updates self.table, self.uniques and self.count.
        """

        Returns
        -------
        np.ndarray[np.intp]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = ObjectFactorizer(3)
        >>> fac.factorize(np.array([1,2,np.nan], dtype='O'), na_sentinel=20)
        array([ 0,  1, 20])
        """
        cdef:
            ndarray[intp_t] labels

        if mask is not None:
            raise NotImplementedError("mask not supported for ObjectFactorizer.")

        if self.uniques.external_view_exists:
            uniques = ObjectVector()
            uniques.extend(self.uniques.to_array())
            self.uniques = uniques
        labels = self.table.get_labels(values, self.uniques,
                                       self.count, na_sentinel, na_value)
        self.count = len(self.uniques)
        return labels
