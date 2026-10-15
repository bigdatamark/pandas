# [MODULE DOC]
# Cython base class NDArrayBacked for pandas extension arrays whose data lives in one
# numpy ndarray (_ndarray) plus a dtype (_dtype): DatetimeArray, TimedeltaArray,
# PeriodArray, Categorical, NumpyExtensionArray etc. via NDArrayBackedExtensionArray.
# Implements construction, pickling and cheap shape-manipulating methods in C for speed;
# every shape method applies the numpy operation to _ndarray and rewraps the result with
# the same type and dtype.
"""
Cython implementations for internal ExtensionArrays.
"""
cimport cython

import numpy as np

cimport numpy as cnp
from cpython cimport PyErr_Clear
from numpy cimport ndarray

cnp.import_array()


@cython.freelist(16)
cdef class NDArrayBacked:
    # [CLASS DOC]
    # Extension type holding _ndarray (ndarray) and _dtype (any dtype object), both
    # declared readonly in arrays.pxd. Uses a freelist of 16 instances to speed up
    # allocation. Methods never validate that dtype matches the ndarray; callers must.
    """
    Implementing these methods in cython improves performance quite a bit.

    import pandas as pd

    from pandas._libs.arrays import NDArrayBacked as cls

    dti = pd.date_range("2016-01-01", periods=3)
    dta = dti._data
    arr = dta._ndarray

    obj = cls._simple_new(arr, arr.dtype)

    # for foo in [arr, dta, obj]: ...

    %timeit foo.copy()
    299 ns ± 30 ns per loop     # <-- arr underlying ndarray (for reference)
    530 ns ± 9.24 ns per loop   # <-- dta with cython NDArrayBacked
    1.66 µs ± 46.3 ns per loop  # <-- dta without cython NDArrayBacked
    328 ns ± 5.29 ns per loop   # <-- obj with NDArrayBacked.__cinit__
    371 ns ± 6.97 ns per loop   # <-- obj with NDArrayBacked._simple_new

    %timeit foo.T
    125 ns ± 6.27 ns per loop   # <-- arr underlying ndarray (for reference)
    226 ns ± 7.66 ns per loop   # <-- dta with cython NDArrayBacked
    911 ns ± 16.6 ns per loop   # <-- dta without cython NDArrayBacked
    215 ns ± 4.54 ns per loop   # <-- obj with NDArrayBacked._simple_new

    """
    # TODO: implement take in terms of cnp.PyArray_TakeFrom
    # TODO: implement concat_same_type in terms of cnp.PyArray_Concatenate

    # cdef:
    #    readonly ndarray _ndarray
    #    readonly object _dtype

    def __init__(self, ndarray values, object dtype):
        # [FUNCTION DOC]
        # Store values as _ndarray and dtype as _dtype without validation.
        # Inputs: values - ndarray; dtype - dtype object. Outputs: None.
        # Exceptions: TypeError if values is not an ndarray. Side effects: mutates self.
        self._ndarray = values
        self._dtype = dtype

    @classmethod
    def _simple_new(cls, ndarray values, object dtype):
        # [FUNCTION DOC]
        # Classmethod: allocate an instance of cls without calling __init__ and set
        # _ndarray/_dtype directly. Inputs: values - ndarray; dtype - dtype object.
        # Outputs: new cls instance. Exceptions: TypeError if values is not an ndarray.
        # Side effects: none.
        cdef:
            NDArrayBacked obj
        obj = NDArrayBacked.__new__(cls)
        obj._ndarray = values
        obj._dtype = dtype
        return obj

    cpdef NDArrayBacked _from_backing_data(self, ndarray values):
        # [FUNCTION DOC]
        # Allocate a new instance of type(self) (no __init__) with _ndarray = values and
        # _dtype = self._dtype. Instance attributes in __dict__ (e.g. freq) are NOT
        # copied. Inputs: values - ndarray, assumed to have a compatible dtype.
        # Outputs: new instance. Side effects: none.
        """
        Construct a new ExtensionArray `new_array` with `arr` as its _ndarray.

        The returned array has the same dtype as self.

        Caller is responsible for ensuring `values.dtype == self._ndarray.dtype`.

        This should round-trip:
            self == self._from_backing_data(self._ndarray)
        """
        # TODO: re-reuse simple_new if/when it can be cpdef
        cdef:
            NDArrayBacked obj
        obj = NDArrayBacked.__new__(type(self))
        obj._ndarray = values
        obj._dtype = self._dtype
        return obj

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Restore state from pickling. Accepted forms: a dict with "_data" or "_ndarray"
        # (the backing array) and "_dtype", any remaining keys being set as attributes;
        # a 3-tuple (data, dtype, attrs dict) where data/dtype may appear in either
        # order (swapped if the second element is an ndarray); a 2-tuple, treated as
        # (data, dtype, {}); a 1-tuple containing a dict, handled as the dict form.
        # Inputs: state - dict or tuple. Outputs: None.
        # Exceptions: ValueError if a dict has no data key; KeyError if a dict lacks
        # "_dtype"; NotImplementedError for any other shape/type.
        # Side effects: mutates self (and sets arbitrary attributes); pops keys from the
        # passed dict, mutating the caller's state dict.
        if isinstance(state, dict):
            if "_data" in state:
                data = state.pop("_data")
            elif "_ndarray" in state:
                data = state.pop("_ndarray")
            else:
                raise ValueError  # pragma: no cover
            self._ndarray = data
            self._dtype = state.pop("_dtype")

            for key, val in state.items():
                setattr(self, key, val)
        elif isinstance(state, tuple):
            if len(state) != 3:
                if len(state) == 1 and isinstance(state[0], dict):
                    self.__setstate__(state[0])
                    return
                elif len(state) == 2:
                    # GH#62820: Handle missing attrs dict during auto-unpickling
                    self.__setstate__((*state, {}))
                    return
                raise NotImplementedError(state)  # pragma: no cover

            data, dtype = state[:2]
            if isinstance(dtype, np.ndarray):
                dtype, data = data, dtype
            self._ndarray = data
            self._dtype = dtype

            if isinstance(state[2], dict):
                for key, val in state[2].items():
                    setattr(self, key, val)
            else:
                raise NotImplementedError(state)  # pragma: no cover
        else:
            raise NotImplementedError(state)  # pragma: no cover

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Return len(self._ndarray) (length of the first axis).
        # Exceptions: TypeError for a 0-d array. Side effects: none.
        return len(self._ndarray)

    @property
    def shape(self):
        # [FUNCTION DOC]
        # Property: shape tuple of the backing ndarray. Side effects: none.
        # object cast bc _ndarray.shape is npy_intp*
        return (<object>(self._ndarray)).shape

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Property: number of dimensions of the backing ndarray.
        return self._ndarray.ndim

    @property
    def size(self) -> int:
        # [FUNCTION DOC]
        # Property: number of elements of the backing ndarray.
        return self._ndarray.size

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: total bytes consumed by the backing ndarray's elements (itemsize *
        # size; object arrays count pointer size only).
        return cnp.PyArray_NBYTES(self._ndarray)

    def copy(self, order="C"):
        # [FUNCTION DOC]
        # Return a copy with the backing data copied in the requested memory layout.
        # Inputs: order - one of "C", "F", "A", "K" (numpy order converter).
        # Outputs: new instance of type(self) with self's dtype (via
        # _from_backing_data).
        # Exceptions: ValueError for an invalid order (the numpy error is cleared
        # first).
        # Side effects: allocates a new ndarray.
        cdef:
            cnp.NPY_ORDER order_code
            int success

        success = cnp.PyArray_OrderConverter(order, &order_code)
        if not success:
            # clear exception so that we don't get a SystemError
            PyErr_Clear()
            # same message used by numpy
            msg = f"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')"
            raise ValueError(msg)

        res_values = cnp.PyArray_NewCopy(self._ndarray, order_code)
        return self._from_backing_data(res_values)

    def delete(self, loc, axis=0):
        # [FUNCTION DOC]
        # Return a new array with entries at loc removed along axis (np.delete).
        # Inputs: loc - int, slice or array of positions; axis - int. Outputs: new
        # instance of type(self) with self's dtype (via _from_backing_data).
        # Exceptions: IndexError for out-of-bounds positions.
        res_values = np.delete(self._ndarray, loc, axis=axis)
        return self._from_backing_data(res_values)

    def swapaxes(self, axis1, axis2):
        # [FUNCTION DOC]
        # Return a view with axis1 and axis2 interchanged.
        # Inputs: axis1, axis2 - ints. Outputs: new instance of type(self) with self's
        # dtype (via _from_backing_data).
        # Exceptions: numpy AxisError for invalid axes.
        res_values = cnp.PyArray_SwapAxes(self._ndarray, axis1, axis2)
        return self._from_backing_data(res_values)

    # TODO: pass NPY_MAXDIMS equiv to axis=None?
    def repeat(self, repeats, axis: int | np.integer = 0):
        # [FUNCTION DOC]
        # Repeat elements along an axis (None treated as 0, so unlike numpy it never
        # flattens). Inputs: repeats - int or array of ints; axis - int or None.
        # Outputs: new instance of type(self) with self's dtype (via
        # _from_backing_data).
        # Exceptions: ValueError for negative or mismatched repeats.
        if axis is None:
            axis = 0
        res_values = cnp.PyArray_Repeat(self._ndarray, repeats, <int>axis)
        return self._from_backing_data(res_values)

    def reshape(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Return a reshaped view/copy using ndarray.reshape with the same arguments.
        # Inputs: shape args and order kwarg. Outputs: new instance of type(self) with
        # self's dtype (via _from_backing_data).
        # Exceptions: ValueError if sizes are incompatible.
        res_values = self._ndarray.reshape(*args, **kwargs)
        return self._from_backing_data(res_values)

    def ravel(self, order="C"):
        # [FUNCTION DOC]
        # Return a flattened array using ndarray.ravel(order).
        # Inputs: order - "C", "F", "A" or "K". Outputs: new instance of type(self) with
        # self's dtype (via _from_backing_data).
        # cnp.PyArray_OrderConverter(PyObject* obj, NPY_ORDER* order)
        # res_values = cnp.PyArray_Ravel(self._ndarray, order)
        res_values = self._ndarray.ravel(order)
        return self._from_backing_data(res_values)

    @property
    def T(self):
        # [FUNCTION DOC]
        # Property: transposed view (ndarray.T) rewrapped. Outputs: new instance of
        # type(self) with self's dtype (via _from_backing_data).
        res_values = self._ndarray.T
        return self._from_backing_data(res_values)

    def transpose(self, *axes):
        # [FUNCTION DOC]
        # Return transpose with optional axes permutation.
        # Inputs: axes - permutation ints. Outputs: new instance of type(self) with
        # self's dtype (via _from_backing_data).
        # Exceptions: ValueError for an invalid permutation.
        res_values = self._ndarray.transpose(*axes)
        return self._from_backing_data(res_values)

    @classmethod
    def _concat_same_type(cls, to_concat, axis=0):
        # [FUNCTION DOC]
        # Classmethod: concatenate the backing arrays of several instances along axis
        # and wrap with the type/dtype of the first one. Dtypes are assumed equal and
        # not checked.
        # Inputs: to_concat - non-empty sequence of NDArrayBacked; axis - int.
        # Outputs: new instance like to_concat[0].
        # Exceptions: ValueError from numpy for empty input or shape mismatch.
        # Side effects: allocates a new ndarray.
        # NB: We are assuming at this point that dtypes all match
        new_values = [obj._ndarray for obj in to_concat]
        new_arr = cnp.PyArray_Concatenate(new_values, axis)
        return to_concat[0]._from_backing_data(new_arr)
