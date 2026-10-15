# [MODULE DOC]
# Cython declarations (shared prototypes) for pandas._libs.hashtable so other
# Cython modules (index engines, join, interval tree, etc.) can cimport the typed
# hash table classes, the Int64Vector growable array and its data struct. Each
# numeric <Name>HashTable wraps a khash table pointer (key -> position), an
# na_position slot (position recorded for the missing value when masks are used;
# -1 if none) and a uses_mask flag. Implementations live in
# hashtable_class_helper.pxi.in.
from numpy cimport (
    intp_t,
    ndarray,
)

from pandas._libs.khash cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    kh_complex64_t,
    kh_complex128_t,
    kh_float32_t,
    kh_float64_t,
    kh_int8_t,
    kh_int16_t,
    kh_int32_t,
    kh_int64_t,
    kh_pymap_t,
    kh_str_t,
    kh_uint8_t,
    kh_uint16_t,
    kh_uint32_t,
    kh_uint64_t,
    khcomplex64_t,
    khcomplex128_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

# prototypes for sharing

cdef class HashTable:
    # [CLASS DOC]
    # Empty common base class of all pandas hash tables (used for isinstance
    # checks and typing only; no state or methods).
    pass

cdef class UInt64HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from uint64 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash uint64 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_uint64_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, uint64_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, uint64_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class Int64HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from int64 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash int64 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_int64_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, int64_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, int64_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class UInt32HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from uint32 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash uint32 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_uint32_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, uint32_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, uint32_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class Int32HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from int32 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash int32 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_int32_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, int32_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, int32_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class UInt16HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from uint16 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash uint16 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_uint16_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, uint16_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, uint16_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class Int16HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from int16 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash int16 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_int16_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, int16_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, int16_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class UInt8HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from uint8 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash uint8 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_uint8_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, uint8_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, uint8_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class Int8HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from int8 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash int8 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_int8_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, int8_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, int8_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class Float64HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from float64 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash float64 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_float64_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, float64_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, float64_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class Float32HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from float32 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash float32 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_float32_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, float32_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, float32_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class Complex64HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from complex64 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash complex64 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_complex64_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, complex64_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, complex64_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class Complex128HashTable(HashTable):
    # [CLASS DOC]
    # Hash table from complex128 keys to integer positions (Py_ssize_t).
    # Fields: table - pointer to the khash complex128 map; na_position - position
    # stored for the missing value (only when uses_mask, -1 if unset); uses_mask -
    # whether missing values are identified by a mask instead of the key.
    cdef kh_complex128_t *table
    cdef Py_ssize_t na_position
    cdef bint uses_mask

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, complex128_t val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, complex128_t key, Py_ssize_t val)
    # [FUNCTION DOC]
    # Return the position recorded for the missing value.
    # Outputs: int. Exceptions: NotImplementedError if uses_mask is False;
    # KeyError('NA') if no missing value was recorded.
    cpdef get_na(self)
    # [FUNCTION DOC]
    # Record val as the position of the missing value.
    # Inputs: val - Py_ssize_t. Exceptions: NotImplementedError if uses_mask is
    # False. Side effects: sets na_position.
    cpdef set_na(self, Py_ssize_t val)

cdef class PyObjectHashTable(HashTable):
    # [CLASS DOC]
    # Hash table from arbitrary hashable Python objects to integer positions
    # (NaN-aware equality). Field: table - pointer to the khash pymap. Keys are
    # borrowed pointers (no references held): callers must keep them alive.
    cdef kh_pymap_t *table

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, object val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, object key, Py_ssize_t val)


cdef class StringHashTable(HashTable):
    # [CLASS DOC]
    # Hash table from Python str values (keyed by their UTF-8 bytes) to integer
    # positions. Field: table - pointer to the khash str map; the UTF-8 buffers are
    # owned by the str objects, which must outlive their use as keys.
    cdef kh_str_t *table

    # [FUNCTION DOC]
    # Return the position stored for key val.
    # Inputs: val - key of the table's type. Outputs: int position.
    # Exceptions: KeyError if val is absent. Side effects: none.
    cpdef get_item(self, str val)
    # [FUNCTION DOC]
    # Insert or overwrite key -> val in the table.
    # Inputs: key - key of the table's type; val - position (Py_ssize_t).
    # Outputs: None. Side effects: may grow the table.
    cpdef set_item(self, str key, Py_ssize_t val)

cdef struct Int64VectorData:
    # [CLASS DOC]
    # Raw storage of an Int64Vector: data - pointer to int64 buffer; size -
    # number of used elements; capacity - allocated elements.
    int64_t *data
    Py_ssize_t size, capacity

cdef class Vector:
    # [CLASS DOC]
    # Base class of growable typed arrays. Field: external_view_exists - set once
    # the buffer has been handed out as an ndarray (to_array); afterwards any
    # reallocation raises ValueError to avoid invalidating that view.
    cdef bint external_view_exists

cdef class Int64Vector(Vector):
    # [CLASS DOC]
    # Growable array of int64 values backed by a numpy array. Fields: data -
    # Int64VectorData (raw pointer, size, capacity); ao - the owning ndarray whose
    # buffer data points into.
    cdef Int64VectorData data
    cdef ndarray ao

    # [FUNCTION DOC]
    # HELPER: Reallocate the buffer to max(new_size, 128) elements, updating the
    # raw pointer. Side effects: reallocates; no check for external views here.
    cdef resize(self, Py_ssize_t new_size)
    # [FUNCTION DOC]
    # Shrink the buffer to size, mark external_view_exists and return the
    # underlying ndarray (no copy). Outputs: 1-D int64 ndarray.
    # Exceptions: ValueError if a view already exists and shrinking is needed.
    cpdef ndarray to_array(self)
    # [FUNCTION DOC]
    # Append x, growing capacity 4x when full.
    # Exceptions: ValueError if growth is needed while an external view exists;
    # because the method is noexcept the error is reported as unraisable instead of
    # propagating.
    # Side effects: mutates the buffer.
    cdef void append(self, int64_t x) noexcept
    # [FUNCTION DOC]
    # Append all elements of the 1-D int64 memoryview x (memcpy), growing to
    # exactly the needed size when necessary.
    # Exceptions: ValueError if growth is needed while an external view exists.
    cdef extend(self, int64_t[:] x)
