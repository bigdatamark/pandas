# [MODULE DOC]
# Defines the Flags class, exposed publicly as the flags attribute of Series
# and DataFrame. Flags describe properties of the pandas object itself (as
# opposed to dataset metadata stored in attrs). Currently the only supported
# flag is whether duplicate labels are allowed. Relates to the generic frame
# base class (which owns the instance and creates Flags) and to Index
# uniqueness checking. Import-time side effects: none beyond importing a
# decorator that sets the public module name on the class.
from __future__ import annotations

from typing import TYPE_CHECKING
import weakref

from pandas.util._decorators import set_module

if TYPE_CHECKING:
    from pandas.core.generic import NDFrame


@set_module("pandas")
class Flags:
    """
    Flags that apply to pandas objects.

    “Flags” differ from “metadata”. Flags reflect properties of the pandas
    object (the Series or DataFrame). Metadata refer to properties of the
    dataset, and should be stored in DataFrame.attrs.

    Parameters
    ----------
    obj : Series or DataFrame
        The object these flags are associated with.
    allows_duplicate_labels : bool, default True
        Whether to allow duplicate labels in this object. By default,
        duplicate labels are permitted. Setting this to ``False`` will
        cause an :class:`errors.DuplicateLabelError` to be raised when
        `index` (or columns for DataFrame) is not unique, or any
        subsequent operation on introduces duplicates.
        See :ref:`duplicates.disallow` for more.

        .. warning::

           This is an experimental feature. Currently, many methods fail to
           propagate the ``allows_duplicate_labels`` value. In future versions
           it is expected that every method taking or returning one or more
           DataFrame or Series objects will propagate ``allows_duplicate_labels``.

    See Also
    --------
    DataFrame.attrs : Dictionary of global attributes of this dataset.
    Series.attrs : Dictionary of global attributes of this dataset.

    Examples
    --------
    Attributes can be set in two ways:

    >>> df = pd.DataFrame()
    >>> df.flags
    <Flags(allows_duplicate_labels=True)>
    >>> df.flags.allows_duplicate_labels = False
    >>> df.flags
    <Flags(allows_duplicate_labels=False)>

    >>> df.flags["allows_duplicate_labels"] = True
    >>> df.flags
    <Flags(allows_duplicate_labels=True)>
    """

    # [CLASS DOC]
    # Per-object container of boolean flags. State: a private boolean for the
    # duplicate-label permission and a weak reference to the owning object (so
    # the flags never keep the object alive). The class attribute _keys lists
    # the recognized flag names and gates mapping-style access. Supports
    # attribute access, item get/set by flag name, equality (by flag values
    # only, False for other types) and a repr.
    _keys: set[str] = {"allows_duplicate_labels"}

    def __init__(self, obj: NDFrame, *, allows_duplicate_labels: bool) -> None:
        # [FUNCTION DOC]
        # Parameters: obj is the Series or DataFrame that owns the flags (held
        # by weak reference only); allows_duplicate_labels (keyword-only,
        # required) is the initial permission. No validation is performed here.
        self._allows_duplicate_labels = allows_duplicate_labels
        self._obj = weakref.ref(obj)

    @property
    def allows_duplicate_labels(self) -> bool:
        # [FUNCTION DOC]
        # Property getter. Returns the stored boolean permission for duplicate
        # labels. No side effects.
        """
        Whether this object allows duplicate labels.

        Setting ``allows_duplicate_labels=False`` ensures that the
        index (and columns of a DataFrame) are unique. Most methods
        that accept and return a Series or DataFrame will propagate
        the value of ``allows_duplicate_labels``.

        See :ref:`duplicates` for more.

        See Also
        --------
        DataFrame.attrs : Set global metadata on this object.
        DataFrame.set_flags : Set global flags on this object.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]}, index=["a", "a"])
        >>> df.flags.allows_duplicate_labels
        True
        >>> df.flags.allows_duplicate_labels = False
        Traceback (most recent call last):
            ...
        pandas.errors.DuplicateLabelError: Index has duplicates.
              positions
        label
        a        [0, 1]
        """
        return self._allows_duplicate_labels

    @allows_duplicate_labels.setter
    def allows_duplicate_labels(self, value: bool) -> None:
        # [FUNCTION DOC]
        # Property setter. The value is coerced to a boolean. Raises a value
        # error if the owning object has been garbage collected. When the new
        # value is false, every axis of the owner is asked to verify uniqueness
        # (which raises the duplicate-label error if duplicates exist) before
        # the flag is stored; when true, no check is made. The flag is only
        # updated if all checks pass.
        value = bool(value)
        obj = self._obj()
        if obj is None:
            raise ValueError("This flag's object has been deleted.")

        if not value:
            for ax in obj.axes:
                ax._maybe_check_unique()

        self._allows_duplicate_labels = value

    def __getitem__(self, key: str):
        # [FUNCTION DOC]
        # Mapping-style read of a flag. key is the flag name; raises a key
        # error if it is not a recognized flag, otherwise returns the
        # attribute value of that name.
        if key not in self._keys:
            raise KeyError(key)

        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        # [FUNCTION DOC]
        # Mapping-style write of a flag. key is the flag name; raises a value
        # error (not a key error) listing the valid names if it is unknown.
        # Otherwise assigns through the attribute, so the setter's validation
        # and coercion apply.
        if key not in self._keys:
            raise ValueError(f"Unknown flag {key}. Must be one of {self._keys}")
        setattr(self, key, value)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a string of the form angle-bracketed Flags with the single
        # flag name and its current value.
        return f"<Flags(allows_duplicate_labels={self.allows_duplicate_labels})>"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality: true only when other is an instance of the same type and
        # has the same duplicate-label permission; false for any other type.
        # The owning objects are not compared.
        if isinstance(other, type(self)):
            return self.allows_duplicate_labels == other.allows_duplicate_labels
        return False
