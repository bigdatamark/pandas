# [MODULE DOC]
# Cython declarations for the khash open-addressing hash tables used by pandas
# (C source: pandas/vendored/klib/khash_python.h, a modified klib khash). Every
# table has power-of-two bucket count, a 1-bit-per-bucket flags array marking
# empty buckets (this variant has no 'deleted' state), a keys array and (for
# maps) a parallel vals array.
# Lookup returns a bucket index ("iterator"), equal to n_buckets when absent.
# Memory is allocated through traced malloc/realloc/free registered with
# tracemalloc under KHASH_TRACE_DOMAIN. Declares: complex key structs and
# NaN-aware equality predicates; Python-object map (pymap) and set (pyset) using
# pandas hashing where NaN == NaN; C-string map (str), string map with a
# first-character prefilter (str_starts), string-to-object map (strbox); bucket
# sizing helper; and, via khash_for_primitive_helper.pxi, one table type per
# numeric dtype.
from cpython.object cimport PyObject
from numpy cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)


cdef extern from "pandas/vendored/klib/khash_python.h":
    # [FUNCTION DOC]
    # Integer id of the tracemalloc domain under which all khash allocations are
    # recorded (424242). Constant; no side effects.
    const int KHASH_TRACE_DOMAIN

    # [FUNCTION DOC]
    # Unsigned 32-bit integer type used for bucket counts and sizes.
    ctypedef uint32_t khuint_t
    # [FUNCTION DOC]
    # Bucket index type returned by get/put; equals n_buckets for 'not found'.
    ctypedef khuint_t khiter_t

    ctypedef struct khcomplex128_t:
        # [CLASS DOC]
        # 128-bit complex number key (two C doubles: real, imag) used by the
        # complex128 table.
        double real
        double imag

    # [FUNCTION DOC]
    # Equality of two complex128 keys: real parts equal and imaginary parts equal,
    # where each part compares equal if both are NaN (so NaN-containing values
    # match) and +0.0 equals -0.0.
    # Inputs: two complex values. Outputs: boolean. No side effects; nogil.
    bint are_equivalent_khcomplex128_t \
        "kh_complex_hash_equal" (khcomplex128_t a, khcomplex128_t b) nogil

    ctypedef struct khcomplex64_t:
        # [CLASS DOC]
        # 64-bit complex number key (two C floats: real, imag) used by the complex64
        # table.
        float real
        float imag

    # [FUNCTION DOC]
    # Equality of two complex64 keys; same semantics as the complex128 variant
    # (component-wise, NaN equals NaN). Outputs: boolean. nogil; no side effects.
    bint are_equivalent_khcomplex64_t \
        "kh_complex_hash_equal" (khcomplex64_t a, khcomplex64_t b) nogil

    # [FUNCTION DOC]
    # Equality of two float64 keys: true if numerically equal (so +0.0 == -0.0)
    # or if both are NaN. Outputs: boolean. nogil; no side effects.
    bint are_equivalent_float64_t \
        "kh_floats_hash_equal" (float64_t a, float64_t b) nogil

    # [FUNCTION DOC]
    # Equality of two float32 keys; same semantics as the float64 variant.
    # Outputs: boolean. nogil; no side effects.
    bint are_equivalent_float32_t \
        "kh_floats_hash_equal" (float32_t a, float32_t b) nogil

    # [FUNCTION DOC]
    # Hash a Python object the way pandas object tables do: floats and complexes
    # use a hash where every NaN hashes to 0; tuples (and subclasses) are hashed
    # element-wise with this same function (xxHash-style combine, like CPython);
    # other objects use the normal Python hash. On 64-bit builds the 64-bit hash is
    # folded to 32 bits by xoring high and low halves.
    # Inputs: any object. Outputs: unsigned 32-bit hash.
    # Exceptions: none; if the object is unhashable the error is cleared and 0 is
    # returned. Requires the GIL.
    uint32_t kh_python_hash_func(object key)
    # [FUNCTION DOC]
    # Equality used by object tables: identical objects are equal; same-type
    # exact floats/complexes compare equal when both are NaN; tuples compare
    # element-wise recursively with this rule; otherwise normal Python equality.
    # Inputs: two objects. Outputs: boolean.
    # Exceptions: none; a raising comparison is cleared and treated as unequal.
    # Requires the GIL.
    bint kh_python_hash_equal(object a, object b)

    ctypedef struct kh_pymap_t:
        # [CLASS DOC]
        # Hash map from Python object pointers to integer values (pymap). Fields as
        # for every khash table: n_buckets, size, n_occupied, upper_bound, flags,
        # keys (object pointers, borrowed - the table does not own references),
        # vals (integer values).
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        PyObject **keys
        size_t *vals

    # [FUNCTION DOC]
    # Allocate and return a new empty pymap hash table (zero buckets).
    # Inputs: none.
    # Outputs: pointer to the newly heap-allocated table.
    # Side effects: allocates memory via the khash allocator (traced in
    # KHASH_TRACE_DOMAIN).
    # Requires the GIL (hashing calls Python).
    kh_pymap_t* kh_init_pymap()
    # [FUNCTION DOC]
    # Free the pymap table and its bucket/flag/key/value arrays.
    # Inputs: table pointer (may be NULL, which is a no-op).
    # Outputs: none.
    # Side effects: frees memory; the pointer must not be used afterwards.
    # Requires the GIL (hashing calls Python).
    void kh_destroy_pymap(kh_pymap_t*)
    # [FUNCTION DOC]
    # Mark every bucket of the pymap table empty; keeps the allocation.
    # Inputs: table pointer.
    # Outputs: none.
    # Side effects: size and n_occupied reset to 0.
    # Requires the GIL (hashing calls Python).
    void kh_clear_pymap(kh_pymap_t*)
    # [FUNCTION DOC]
    # Look up a key (object pointer) in the pymap table.
    # Inputs: table pointer, key.
    # Outputs: bucket index of the key, or n_buckets if absent.
    # Side effects: none.
    # Requires the GIL (hashing calls Python).
    khuint_t kh_get_pymap(kh_pymap_t*, PyObject*)
    # [FUNCTION DOC]
    # Rehash the pymap table into the given number of buckets (rounded up
    # to a power of two); a no-op if that would not hold the current size.
    # Inputs: table pointer, requested bucket count.
    # Outputs: none.
    # Side effects: reallocates arrays and rehashes all entries.
    # Requires the GIL (hashing calls Python).
    void kh_resize_pymap(kh_pymap_t*, khuint_t)
    # [FUNCTION DOC]
    # Insert a key (object pointer) into the pymap table if absent
    # (value slot left uninitialised).
    # Inputs: table pointer, key, pointer to int status.
    # Outputs: bucket index of the key; status set to 0 if the key already existed,
    # 1 if newly inserted (2 is never produced: this khash has no deleted state).
    # Side effects: may grow/rehash the table.
    # Requires the GIL (hashing calls Python).
    khuint_t kh_put_pymap(kh_pymap_t*, PyObject*, int*)
    # [FUNCTION DOC]
    # Intended to delete the given bucket of the pymap table.
    # Inputs: table pointer, bucket index (ignored if n_buckets or empty).
    # Outputs: none.
    # Side effects: decrements size only.
    # KNOWN ISSUE: this khash variant has no deleted-bucket marker, so the bucket
    # stays occupied and still matches lookups while size is decremented; pandas
    # never calls this.
    # Requires the GIL (hashing calls Python).
    void kh_del_pymap(kh_pymap_t*, khuint_t)

    # [FUNCTION DOC]
    # Test whether a bucket of the pymap table holds a live key.
    # Inputs: table pointer, bucket index.
    # Outputs: true if occupied (flag not empty).
    # Side effects: none.
    # Requires the GIL (hashing calls Python).
    bint kh_exist_pymap(kh_pymap_t*, khiter_t)

    ctypedef struct kh_pyset_t:
        # [CLASS DOC]
        # Hash set of Python object pointers (pyset); same fields as pymap. Keys are
        # borrowed references.
        # KNOWN ISSUE: the C set type stores 1-byte (char) values, while this
        # declaration types vals as size_t pointers; harmless as long as vals is never
        # accessed for sets (it is not in pandas).
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        PyObject **keys
        size_t *vals

    # [FUNCTION DOC]
    # Allocate and return a new empty pyset hash table (zero buckets).
    # Inputs: none.
    # Outputs: pointer to the newly heap-allocated table.
    # Side effects: allocates memory via the khash allocator (traced in
    # KHASH_TRACE_DOMAIN).
    # Requires the GIL (hashing calls Python).
    kh_pyset_t* kh_init_pyset()
    # [FUNCTION DOC]
    # Free the pyset table and its bucket/flag/key/value arrays.
    # Inputs: table pointer (may be NULL, which is a no-op).
    # Outputs: none.
    # Side effects: frees memory; the pointer must not be used afterwards.
    # Requires the GIL (hashing calls Python).
    void kh_destroy_pyset(kh_pyset_t*)
    # [FUNCTION DOC]
    # Mark every bucket of the pyset table empty; keeps the allocation.
    # Inputs: table pointer.
    # Outputs: none.
    # Side effects: size and n_occupied reset to 0.
    # Requires the GIL (hashing calls Python).
    void kh_clear_pyset(kh_pyset_t*)
    # [FUNCTION DOC]
    # Look up a key (object pointer) in the pyset table.
    # Inputs: table pointer, key.
    # Outputs: bucket index of the key, or n_buckets if absent.
    # Side effects: none.
    # Requires the GIL (hashing calls Python).
    khuint_t kh_get_pyset(kh_pyset_t*, PyObject*)
    # [FUNCTION DOC]
    # Rehash the pyset table into the given number of buckets (rounded up
    # to a power of two); a no-op if that would not hold the current size.
    # Inputs: table pointer, requested bucket count.
    # Outputs: none.
    # Side effects: reallocates arrays and rehashes all entries.
    # Requires the GIL (hashing calls Python).
    void kh_resize_pyset(kh_pyset_t*, khuint_t)
    # [FUNCTION DOC]
    # Insert a key (object pointer) into the pyset table if absent
    # (no value).
    # Inputs: table pointer, key, pointer to int status.
    # Outputs: bucket index of the key; status set to 0 if the key already existed,
    # 1 if newly inserted (2 is never produced: this khash has no deleted state).
    # Side effects: may grow/rehash the table.
    # Requires the GIL (hashing calls Python).
    khuint_t kh_put_pyset(kh_pyset_t*, PyObject*, int*)
    # [FUNCTION DOC]
    # Intended to delete the given bucket of the pyset table.
    # Inputs: table pointer, bucket index (ignored if n_buckets or empty).
    # Outputs: none.
    # Side effects: decrements size only.
    # KNOWN ISSUE: this khash variant has no deleted-bucket marker, so the bucket
    # stays occupied and still matches lookups while size is decremented; pandas
    # never calls this.
    # Requires the GIL (hashing calls Python).
    void kh_del_pyset(kh_pyset_t*, khuint_t)

    # [FUNCTION DOC]
    # Test whether a bucket of the pyset table holds a live key.
    # Inputs: table pointer, bucket index.
    # Outputs: true if occupied (flag not empty).
    # Side effects: none.
    # Requires the GIL (hashing calls Python).
    bint kh_exist_pyset(kh_pyset_t*, khiter_t)

    # [FUNCTION DOC]
    # C string (char pointer) key type for the string tables.
    ctypedef char* kh_cstr_t

    ctypedef struct kh_str_t:
        # [CLASS DOC]
        # Hash map from NUL-terminated C strings to size_t values (str). Keys are
        # stored as pointers only; the caller must keep the string memory alive.
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        kh_cstr_t *keys
        size_t *vals

    # [FUNCTION DOC]
    # Allocate and return a new empty str hash table (zero buckets).
    # Inputs: none.
    # Outputs: pointer to the newly heap-allocated table.
    # Side effects: allocates memory via the khash allocator (traced in
    # KHASH_TRACE_DOMAIN).
    # Callable without the GIL (nogil).
    kh_str_t* kh_init_str() nogil
    # [FUNCTION DOC]
    # Free the str table and its bucket/flag/key/value arrays.
    # Inputs: table pointer (may be NULL, which is a no-op).
    # Outputs: none.
    # Side effects: frees memory; the pointer must not be used afterwards.
    # Callable without the GIL (nogil).
    void kh_destroy_str(kh_str_t*) nogil
    # [FUNCTION DOC]
    # Mark every bucket of the str table empty; keeps the allocation.
    # Inputs: table pointer.
    # Outputs: none.
    # Side effects: size and n_occupied reset to 0.
    # Callable without the GIL (nogil).
    void kh_clear_str(kh_str_t*) nogil
    # [FUNCTION DOC]
    # Look up a key (C string) in the str table.
    # Inputs: table pointer, key.
    # Outputs: bucket index of the key, or n_buckets if absent.
    # Side effects: none.
    # Callable without the GIL (nogil).
    khuint_t kh_get_str(kh_str_t*, kh_cstr_t) nogil
    # [FUNCTION DOC]
    # Rehash the str table into the given number of buckets (rounded up
    # to a power of two); a no-op if that would not hold the current size.
    # Inputs: table pointer, requested bucket count.
    # Outputs: none.
    # Side effects: reallocates arrays and rehashes all entries.
    # Callable without the GIL (nogil).
    void kh_resize_str(kh_str_t*, khuint_t) nogil
    # [FUNCTION DOC]
    # Insert a key (C string) into the str table if absent
    # (value slot left uninitialised).
    # Inputs: table pointer, key, pointer to int status.
    # Outputs: bucket index of the key; status set to 0 if the key already existed,
    # 1 if newly inserted (2 is never produced: this khash has no deleted state).
    # Side effects: may grow/rehash the table.
    # Callable without the GIL (nogil).
    khuint_t kh_put_str(kh_str_t*, kh_cstr_t, int*) nogil
    # [FUNCTION DOC]
    # Intended to delete the given bucket of the str table.
    # Inputs: table pointer, bucket index (ignored if n_buckets or empty).
    # Outputs: none.
    # Side effects: decrements size only.
    # KNOWN ISSUE: this khash variant has no deleted-bucket marker, so the bucket
    # stays occupied and still matches lookups while size is decremented; pandas
    # never calls this.
    # Callable without the GIL (nogil).
    void kh_del_str(kh_str_t*, khuint_t) nogil

    # [FUNCTION DOC]
    # Test whether a bucket of the str table holds a live key.
    # Inputs: table pointer, bucket index.
    # Outputs: true if occupied (flag not empty).
    # Side effects: none.
    # Callable without the GIL (nogil).
    bint kh_exist_str(kh_str_t*, khiter_t) nogil

    ctypedef struct kh_str_starts_t:
        # [CLASS DOC]
        # String set accelerator: wraps a str table plus a 256-entry array flagging
        # which first bytes occur among inserted keys, so lookups of strings whose
        # first byte never appears can return early without hashing.
        kh_str_t *table
        int starts[256]

    # [FUNCTION DOC]
    # Allocate a zero-initialised str_starts wrapper and its inner str table.
    # Outputs: pointer to the new structure. Side effects: traced allocation; nogil.
    kh_str_starts_t* kh_init_str_starts() nogil
    # [FUNCTION DOC]
    # Insert a C string into the wrapped str table; if newly inserted, mark its
    # first byte in the starts array.
    # Inputs: wrapper pointer, key, pointer to int status (0 existed, 1 new).
    # Outputs: bucket index. Side effects: may grow the table; nogil.
    khuint_t kh_put_str_starts_item(kh_str_starts_t* table, char* key,
                                    int* ret) nogil
    # [FUNCTION DOC]
    # Membership test: returns 1 if the first byte of key is flagged and either
    # the key is the empty string or found in the inner table; else 0.
    # Inputs: wrapper pointer, key. Outputs: 0 or 1 (not a bucket index!).
    # Side effects: none; nogil.
    khuint_t kh_get_str_starts_item(kh_str_starts_t* table, char* key) nogil
    # [FUNCTION DOC]
    # Destroy the inner str table and free the wrapper. Side effects: frees
    # memory; nogil.
    void kh_destroy_str_starts(kh_str_starts_t*) nogil
    # [FUNCTION DOC]
    # Resize the inner str table to the given bucket count. nogil.
    void kh_resize_str_starts(kh_str_starts_t*, khuint_t) nogil

    # sweep factorize

    ctypedef struct kh_strbox_t:
        # [CLASS DOC]
        # Hash map from C strings to Python object pointers (strbox); used by sweep-
        # style factorize code. Values are raw object pointers (no reference counting).
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        kh_cstr_t *keys
        PyObject **vals

    # [FUNCTION DOC]
    # Allocate and return a new empty strbox hash table (zero buckets).
    # Inputs: none.
    # Outputs: pointer to the newly heap-allocated table.
    # Side effects: allocates memory via the khash allocator (traced in
    # KHASH_TRACE_DOMAIN).
    # Callable without the GIL (nogil).
    kh_strbox_t* kh_init_strbox() nogil
    # [FUNCTION DOC]
    # Free the strbox table and its bucket/flag/key/value arrays.
    # Inputs: table pointer (may be NULL, which is a no-op).
    # Outputs: none.
    # Side effects: frees memory; the pointer must not be used afterwards.
    # Callable without the GIL (nogil).
    void kh_destroy_strbox(kh_strbox_t*) nogil
    # [FUNCTION DOC]
    # Mark every bucket of the strbox table empty; keeps the allocation.
    # Inputs: table pointer.
    # Outputs: none.
    # Side effects: size and n_occupied reset to 0.
    # Callable without the GIL (nogil).
    void kh_clear_strbox(kh_strbox_t*) nogil
    # [FUNCTION DOC]
    # Look up a key (C string) in the strbox table.
    # Inputs: table pointer, key.
    # Outputs: bucket index of the key, or n_buckets if absent.
    # Side effects: none.
    # Callable without the GIL (nogil).
    khuint_t kh_get_strbox(kh_strbox_t*, kh_cstr_t) nogil
    # [FUNCTION DOC]
    # Rehash the strbox table into the given number of buckets (rounded up
    # to a power of two); a no-op if that would not hold the current size.
    # Inputs: table pointer, requested bucket count.
    # Outputs: none.
    # Side effects: reallocates arrays and rehashes all entries.
    # Callable without the GIL (nogil).
    void kh_resize_strbox(kh_strbox_t*, khuint_t) nogil
    # [FUNCTION DOC]
    # Insert a key (C string) into the strbox table if absent
    # (value slot left uninitialised).
    # Inputs: table pointer, key, pointer to int status.
    # Outputs: bucket index of the key; status set to 0 if the key already existed,
    # 1 if newly inserted (2 is never produced: this khash has no deleted state).
    # Side effects: may grow/rehash the table.
    # Callable without the GIL (nogil).
    khuint_t kh_put_strbox(kh_strbox_t*, kh_cstr_t, int*) nogil
    # [FUNCTION DOC]
    # Intended to delete the given bucket of the strbox table.
    # Inputs: table pointer, bucket index (ignored if n_buckets or empty).
    # Outputs: none.
    # Side effects: decrements size only.
    # KNOWN ISSUE: this khash variant has no deleted-bucket marker, so the bucket
    # stays occupied and still matches lookups while size is decremented; pandas
    # never calls this.
    # Callable without the GIL (nogil).
    void kh_del_strbox(kh_strbox_t*, khuint_t) nogil

    # [FUNCTION DOC]
    # Test whether a bucket of the strbox table holds a live key.
    # Inputs: table pointer, bucket index.
    # Outputs: true if occupied (flag not empty).
    # Side effects: none.
    # Callable without the GIL (nogil).
    bint kh_exist_strbox(kh_strbox_t*, khiter_t) nogil

    # [FUNCTION DOC]
    # Number of buckets needed to hold n elements without exceeding the load
    # factor (0.77): n rounded up to a power of two, doubled if that power of two
    # would be over the upper bound.
    # Inputs: element count. Outputs: bucket count. No side effects; nogil.
    khuint_t kh_needed_n_buckets(khuint_t element_n) nogil


include "khash_for_primitive_helper.pxi"
