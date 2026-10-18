# [MODULE DOC]
# Descriptor classes used throughout pandas: CachedProperty (exported as
# cache_readonly), a read-only property whose value is computed once per instance
# and stored in the instance's _cache dict, and AxisProperty, which exposes an
# axis (index/columns) of a Series/DataFrame by reading its block manager.
from cpython.dict cimport (
    PyDict_Contains,
    PyDict_GetItem,
    PyDict_SetItem,
)
from cython cimport Py_ssize_t


cdef class CachedProperty:
    # [CLASS DOC]
    # Non-data-like descriptor caching the result of a getter function per instance.
    # Attributes (read-only): fget - the wrapped function; name - its __name__, used as
    # the cache dict key; __doc__ - copied from fget. Exported as cache_readonly.

    cdef readonly:
        object fget, name, __doc__

    def __init__(self, fget):
        # [FUNCTION DOC]
        # Wrap a getter function.
        # Inputs: fget - callable taking the instance.
        # Outputs: none.
        # Side effects: stores fget, its __name__ and its docstring (None if missing).
        self.fget = fget
        self.name = fget.__name__
        self.__doc__ = getattr(fget, "__doc__", None)

    def __get__(self, obj, typ):
        # [FUNCTION DOC]
        # Descriptor get: return the cached value or compute and cache it.
        # Inputs: obj - instance or None when accessed on the class; typ - owner class.
        # Outputs: self when obj is None; otherwise the value. Looks up obj._cache
        #   (creates
        # an empty dict and assigns it to obj._cache if absent or None). If the
        #   attribute
        # cannot be assigned (AttributeError), returns the descriptor itself. If the
        #   name
        # is in the cache, returns the cached value, else calls fget(obj), stores the
        # result under name and returns it.
        # Exceptions: whatever fget raises.
        # Side effects: may create obj._cache and inserts into it.
        # KNOWN ISSUE: when obj._cache cannot be created the descriptor object itself is
        # returned instead of the computed value, which silently yields a wrong object.
        if obj is None:
            # accessed on the class, not the instance
            return self

        # Get the cache or set a default one if needed
        cache = getattr(obj, "_cache", None)
        if cache is None:
            try:
                cache = obj._cache = {}
            except (AttributeError):
                return self

        if PyDict_Contains(cache, self.name):
            # not necessary to Py_INCREF
            val = <object>PyDict_GetItem(cache, self.name)
        else:
            val = self.fget(obj)
            PyDict_SetItem(cache, self.name, val)
        return val

    def __set__(self, obj, value):
        # [FUNCTION DOC]
        # Descriptor set: always refuses assignment.
        # Inputs: obj - instance; value - ignored.
        # Outputs: none.
        # Exceptions: AttributeError ("Can't set attribute") always.
        raise AttributeError("Can't set attribute")


cache_readonly = CachedProperty


cdef class AxisProperty:
    # [CLASS DOC]
    # Descriptor exposing one axis of an NDFrame (e.g. DataFrame.index/columns).
    # Attributes (read-only): axis - integer position in obj._mgr.axes; __doc__ - doc.

    cdef readonly:
        Py_ssize_t axis
        object __doc__

    def __init__(self, axis=0, doc=""):
        # [FUNCTION DOC]
        # Initialize the axis descriptor.
        # Inputs: axis - integer axis number (default 0); doc - docstring (default "").
        # Outputs: none.
        # Side effects: stores axis and doc.
        self.axis = axis
        self.__doc__ = doc

    def __get__(self, obj, type):
        # [FUNCTION DOC]
        # Descriptor get: return the axis object of the instance.
        # Inputs: obj - NDFrame instance or None (class access); type - owner class.
        # Outputs: self if obj is None, else obj._mgr.axes[axis] (an Index).
        # Exceptions: AttributeError if obj has no _mgr; IndexError if axis out of
        #   range.
        cdef:
            list axes

        if obj is None:
            # Only instances have _mgr, not classes
            return self
        else:
            axes = obj._mgr.axes
        return axes[self.axis]

    def __set__(self, obj, value):
        # [FUNCTION DOC]
        # Descriptor set: replace the axis on the instance.
        # Inputs: obj - NDFrame instance; value - new axis labels.
        # Outputs: none.
        # Side effects: delegates to obj._set_axis(axis, value), mutating obj.
        obj._set_axis(self.axis, value)
