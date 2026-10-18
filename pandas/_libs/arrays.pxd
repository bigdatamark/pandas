# [MODULE DOC]
# Cython declarations for pandas._libs.arrays: exposes the NDArrayBacked extension type
# so other Cython modules can access its backing ndarray and dtype attributes at C speed
# and call its cpdef methods directly.

from numpy cimport ndarray


cdef class NDArrayBacked:
    # [CLASS DOC]
    # Base extension type for array-likes that store their data in a single numpy
    # ndarray (_ndarray) plus an associated dtype object (_dtype). Both attributes are
    # read-only from Python and set from Cython (constructor/_from_backing_data/
    # __setstate__). Subclassed by DatetimeArray, TimedeltaArray, Categorical, etc.
    cdef:
        readonly ndarray _ndarray
        readonly object _dtype

    # [FUNCTION DOC]
    # Declaration: create a new instance of the same type as self sharing self._dtype
    # but wrapping the given ndarray, bypassing __init__/validation.
    # Inputs: values - ndarray of the backing data. Outputs: new NDArrayBacked subclass
    # instance. Side effects: none on self.
    cpdef NDArrayBacked _from_backing_data(self, ndarray values)
    # [FUNCTION DOC]
    # Declaration: restore _ndarray/_dtype (and any extra __dict__ attrs) from a pickled
    # state; see arrays.pyx for accepted state layouts.
    # Inputs: state - tuple or dict from pickling. Outputs: None. Side effects: mutates
    # self.
    cpdef __setstate__(self, state)
