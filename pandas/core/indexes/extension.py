"""
Shared methods for Index subclasses backed by ExtensionArray.
"""

# [MODULE DOC]
# Shared machinery for Index subclasses whose data is stored in an extension
# array (datetime-like, period, interval indexes).
# Public contents: the class decorator that pins array attributes onto an index
# class, a helper that builds each delegating attribute, the ExtensionIndex base
# class and the NDArrayBackedExtensionIndex subclass. Depends on the base Index
# class and the generic frame check; the array types are only imported for type
# checking. No import-time side effects beyond defining a type variable.

from __future__ import annotations

from inspect import signature
from typing import (
    TYPE_CHECKING,
    TypeVar,
)

from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.generic import ABCDataFrame

from pandas.core.indexes.base import Index

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from pandas._typing import (
        ArrayLike,
        npt,
    )

    from pandas.core.arrays import IntervalArray
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

_ExtensionIndexT = TypeVar("_ExtensionIndexT", bound="ExtensionIndex")


def _inherit_from_data(
    name: str, delegate: type, cache: bool = False, wrap: bool = False
):
    # [FUNCTION DOC]
    # HELPER: builds the delegating attribute that the class decorator installs on an
    # index class. Parameters: name is the attribute name looked up on the delegate
    # class; delegate is the array class that owns it; cache says whether read-only
    # properties should become cached read-only attributes (default false); wrap says
    # whether results should be re-wrapped as an index (default false).
    # Returns one of: a cached read-only property (property on the delegate with the
    # cache flag set; this branch ignores the wrap flag and returns the raw array
    # attribute), a read/write property that forwards get and set to the underlying
    # data (wrapping results when requested), the plain attribute itself when it is
    # not callable, or a method that forwards to the delegate's function applied to
    # the index's data. Raises an attribute error if the delegate lacks the name.
    # Copies the name, docstring and (for methods) the call signature from the
    # delegate.
    """
    Make an alias for a method of the underlying ExtensionArray.

    Parameters
    ----------
    name : str
        Name of an attribute the class should inherit from its EA parent.
    delegate : class
    cache : bool, default False
        Whether to convert wrapped properties into cache_readonly
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.

    Returns
    -------
    attribute, method, property, or cache_readonly
    """
    attr = getattr(delegate, name)

    if isinstance(attr, property) or type(attr).__name__ == "getset_descriptor":
        # getset_descriptor i.e. property defined in cython class
        if cache:

            def cached(self):
                # [FUNCTION DOC]
                # HELPER: getter of the cached property; returns the named attribute of the
                # index's underlying data unchanged.
                return getattr(self._data, name)

            cached.__name__ = name
            cached.__doc__ = attr.__doc__
            method = cache_readonly(cached)

        else:

            def fget(self):
                # [FUNCTION DOC]
                # HELPER: getter of the forwarded property. Reads the named attribute from the
                # underlying data. When wrap is set, a result of the same array type becomes a
                # new index of the same class keeping the name, a data frame result gets this
                # index set as its index, and anything else is wrapped in a generic index with
                # the result's dtype and no copy. Without wrap the result is returned as is.
                result = getattr(self._data, name)
                if wrap:
                    if isinstance(result, type(self._data)):
                        return type(self)._simple_new(result, name=self.name)
                    elif isinstance(result, ABCDataFrame):
                        return result.set_index(self)
                    return Index(result, name=self.name, dtype=result.dtype, copy=False)
                return result

            def fset(self, value) -> None:
                # [FUNCTION DOC]
                # HELPER: setter of the forwarded property; assigns the value to the named
                # attribute of the underlying data. Returns nothing.
                setattr(self._data, name, value)

            fget.__name__ = name
            fget.__doc__ = attr.__doc__

            method = property(fget, fset)

    elif not callable(attr):
        # just a normal attribute, no wrapping
        method = attr

    else:
        # error: Incompatible redefinition (redefinition with type "Callable[[Any,
        # VarArg(Any), KwArg(Any)], Any]", original type "property")
        def method(self, *args, **kwargs):  # type: ignore[misc]
            # [FUNCTION DOC]
            # HELPER: forwarding method installed for callable attributes. Takes the usual
            # positional and keyword arguments and calls the delegate's function with the
            # index's underlying data in place of the instance. Raises a value error naming
            # the class if an inplace keyword is supplied (indexes are immutable). The result
            # is wrapped like the property getter does when wrap is set (same-type array to
            # index of this class with the name kept, data frame to indexed frame, other to
            # generic index with the result dtype); otherwise returned unchanged.
            if "inplace" in kwargs:
                raise ValueError(f"cannot use inplace with {type(self).__name__}")
            result = attr(self._data, *args, **kwargs)
            if wrap:
                if isinstance(result, type(self._data)):
                    return type(self)._simple_new(result, name=self.name)
                elif isinstance(result, ABCDataFrame):
                    return result.set_index(self)
                return Index(result, name=self.name, dtype=result.dtype, copy=False)
            return result

        method.__name__ = name
        method.__doc__ = attr.__doc__
        method.__signature__ = signature(attr)  # type: ignore[attr-defined]
    return method


def inherit_names(
    names: list[str], delegate: type, cache: bool = False, wrap: bool = False
) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]:
    # [FUNCTION DOC]
    # Class decorator factory. Parameters: names is the list of attribute names to
    # pin; delegate is the array class they come from; cache and wrap are passed to
    # the helper that builds each attribute (defaults false). Returns a decorator
    # that sets one generated attribute per name on the decorated class and returns
    # that same class. The decorator mutates the class it is applied to.
    """
    Class decorator to pin attributes from an ExtensionArray to an Index subclass.

    Parameters
    ----------
    names : List[str]
    delegate : class
    cache : bool, default False
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.
    """

    def wrapper(cls: type[_ExtensionIndexT]) -> type[_ExtensionIndexT]:
        # [FUNCTION DOC]
        # HELPER: the decorator returned by the class decorator factory. Sets a generated
        # delegating attribute for every requested name on cls (side effect: mutates
        # cls) and returns cls.
        for name in names:
            meth = _inherit_from_data(name, delegate, cache=cache, wrap=wrap)
            setattr(cls, name, meth)

        return cls

    return wrapper


class ExtensionIndex(Index):
    # [CLASS DOC]
    # Base class for indexes backed by an extension array. Inherits from the generic
    # Index. The _data attribute holds the array (interval array or an array backed
    # by a numpy array); size, length and dtype are inherited from the base class
    # because it already delegates to the data. Adds two internal members: value
    # validation for insertion and a cached missing-value mask.
    """
    Index subclass for indexes backed by ExtensionArray.
    """

    # The base class already passes through to _data:
    #  size, __len__, dtype

    _data: IntervalArray | NDArrayBackedExtensionArray

    # ---------------------------------------------------------------------

    def _validate_fill_value(self, value):
        # [FUNCTION DOC]
        # HELPER: converts a fill value so it can be inserted into the underlying array
        # by delegating to the array's set-item validation. Returns the converted value;
        # raises whatever the array raises (typically a type or value error) if the value
        # is not insertable.
        """
        Convert value to be insertable to underlying array.
        """
        return self._data._validate_setitem_value(value)

    @cache_readonly
    def _isnan(self) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # HELPER: cached boolean mask of missing entries, computed by the array's
        # missing-value check.
        # error: Incompatible return value type (got "ExtensionArray", expected
        # "ndarray")
        return self._data.isna()  # type: ignore[return-value]


class NDArrayBackedExtensionIndex(ExtensionIndex):
    # [CLASS DOC]
    # Extension index whose array is backed by a numpy ndarray (datetime, timedelta
    # and period arrays). Inherits from the extension index base. Only supplies the
    # hooks the index engine and join machinery need to access the raw backing
    # ndarray.
    """
    Index subclass for indexes backed by NDArrayBackedExtensionArray.
    """

    _data: NDArrayBackedExtensionArray

    def _get_engine_target(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: returns the raw backing ndarray of the underlying data, which the index
        # engine uses as its lookup target.
        return self._data._ndarray

    def _from_join_target(self, result: np.ndarray) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: converts a joined ndarray back to an array of the original extension
        # type. The parameter result must have the same dtype as the backing ndarray
        # (asserted; an assertion error is raised otherwise). Returns an extension array
        # rebuilt from the backing data.
        assert result.dtype == self._data._ndarray.dtype
        return self._data._from_backing_data(result)
