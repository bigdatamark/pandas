"""define generic base classes for pandas objects"""

# [MODULE DOC]
# Defines virtual abstract base classes (the ABC-prefixed names) used for
# isinstance checks against pandas objects without importing the concrete
# classes, which avoids circular imports. Public contents: a factory function
# that builds such a class, and one instance per pandas container kind (indexes
# of each flavor, Series, DataFrame, the generic frame base, and the array
# types). Each generated class decides membership by reading the "_typ" tag
# attribute on the inspected object and comparing it against a fixed set of
# accepted tag values. The concrete classes are only imported under type
# checking, and the exported names are cast to the corresponding concrete
# types purely for the benefit of static type checkers.
# Import-time side effects: creates the metaclass and class objects; no I/O.
# KNOWN ISSUE: the ABCCategorical, ABCDatetimeArray and ABCTimedeltaArray
# definitions pass a parenthesized plain string rather than a one-element
# tuple, so the membership test performed is a substring test on that string
# rather than an equality test against a tag.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Type,
    cast,
)

if TYPE_CHECKING:
    from pandas import (
        Categorical,
        CategoricalIndex,
        DataFrame,
        DatetimeIndex,
        Index,
        IntervalIndex,
        MultiIndex,
        PeriodIndex,
        RangeIndex,
        Series,
        TimedeltaIndex,
    )
    from pandas.core.arrays import (
        DatetimeArray,
        ExtensionArray,
        NumpyExtensionArray,
        PeriodArray,
        TimedeltaArray,
    )
    from pandas.core.generic import NDFrame


# define abstract base classes to enable isinstance type checking on our
# objects
def create_pandas_abc_type(name, attr, comp) -> type:
    # [FUNCTION DOC]
    # Factory that builds an abstract-base-class-like type for isinstance and
    # issubclass checks.
    # Parameters: name is the string name given to the new class; attr is the
    # attribute name read from the inspected object to obtain its type tag;
    # comp is a collection of accepted tag values.
    # Returns a new class (an instance of a freshly built metaclass whose
    # instance-check and subclass-check hooks are customized) with no bases.
    # An instance check succeeds when the object's tag is in comp and the
    # object is not itself a class. A subclass check raises a type error when
    # its argument is not a class, otherwise succeeds when the class's tag is
    # in comp. No global state is modified.
    def _check(inst) -> bool:
        # [FUNCTION DOC]
        # HELPER: shared tag test for the factory above. Returns whether the
        # attribute named by attr on inst is a member of comp.
        # KNOWN ISSUE: the fallback value used when the attribute is missing is
        # the literal string "_typ" rather than a sentinel; it only works
        # because that string is never a member of any comp collection (but
        # for the substring-based string collections it is checked by
        # substring containment).
        return getattr(inst, attr, "_typ") in comp

    # https://github.com/python/mypy/issues/1006
    # error: 'classmethod' used with a non-method
    @classmethod  # type: ignore[misc]
    def _instancecheck(cls, inst) -> bool:
        # [FUNCTION DOC]
        # HELPER: instance-check hook installed on the generated metaclass.
        # Returns true when the tag test passes and inst is not a class object.
        return _check(inst) and not isinstance(inst, type)

    @classmethod  # type: ignore[misc]
    def _subclasscheck(cls, inst) -> bool:
        # [FUNCTION DOC]
        # HELPER: subclass-check hook installed on the generated metaclass.
        # Raises a type error if inst is not a class (matching the default
        # behavior); otherwise returns the result of the tag test.
        # Raise instead of returning False
        # This is consistent with default __subclasscheck__ behavior
        if not isinstance(inst, type):
            raise TypeError("issubclass() arg 1 must be a class")

        return _check(inst)

    dct = {"__instancecheck__": _instancecheck, "__subclasscheck__": _subclasscheck}
    meta = type("ABCBase", (type,), dct)
    return meta(name, (), dct)


ABCRangeIndex = cast(
    "Type[RangeIndex]",
    create_pandas_abc_type("ABCRangeIndex", "_typ", ("rangeindex",)),
)
ABCMultiIndex = cast(
    "Type[MultiIndex]",
    create_pandas_abc_type("ABCMultiIndex", "_typ", ("multiindex",)),
)
ABCDatetimeIndex = cast(
    "Type[DatetimeIndex]",
    create_pandas_abc_type("ABCDatetimeIndex", "_typ", ("datetimeindex",)),
)
ABCTimedeltaIndex = cast(
    "Type[TimedeltaIndex]",
    create_pandas_abc_type("ABCTimedeltaIndex", "_typ", ("timedeltaindex",)),
)
ABCPeriodIndex = cast(
    "Type[PeriodIndex]",
    create_pandas_abc_type("ABCPeriodIndex", "_typ", ("periodindex",)),
)
ABCCategoricalIndex = cast(
    "Type[CategoricalIndex]",
    create_pandas_abc_type("ABCCategoricalIndex", "_typ", ("categoricalindex",)),
)
ABCIntervalIndex = cast(
    "Type[IntervalIndex]",
    create_pandas_abc_type("ABCIntervalIndex", "_typ", ("intervalindex",)),
)
ABCIndex = cast(
    "Type[Index]",
    create_pandas_abc_type(
        "ABCIndex",
        "_typ",
        {
            "index",
            "rangeindex",
            "multiindex",
            "datetimeindex",
            "timedeltaindex",
            "periodindex",
            "categoricalindex",
            "intervalindex",
        },
    ),
)


ABCNDFrame = cast(
    "Type[NDFrame]",
    create_pandas_abc_type("ABCNDFrame", "_typ", ("series", "dataframe")),
)
ABCSeries = cast(
    "Type[Series]",
    create_pandas_abc_type("ABCSeries", "_typ", ("series",)),
)
ABCDataFrame = cast(
    "Type[DataFrame]", create_pandas_abc_type("ABCDataFrame", "_typ", ("dataframe",))
)

ABCCategorical = cast(
    "Type[Categorical]",
    create_pandas_abc_type("ABCCategorical", "_typ", ("categorical")),
)
ABCDatetimeArray = cast(
    "Type[DatetimeArray]",
    create_pandas_abc_type("ABCDatetimeArray", "_typ", ("datetimearray")),
)
ABCTimedeltaArray = cast(
    "Type[TimedeltaArray]",
    create_pandas_abc_type("ABCTimedeltaArray", "_typ", ("timedeltaarray")),
)
ABCPeriodArray = cast(
    "Type[PeriodArray]",
    create_pandas_abc_type("ABCPeriodArray", "_typ", ("periodarray",)),
)
ABCExtensionArray = cast(
    "Type[ExtensionArray]",
    create_pandas_abc_type(
        "ABCExtensionArray",
        "_typ",
        # Note: IntervalArray and SparseArray are included bc they have _typ="extension"
        {"extension", "categorical", "periodarray", "datetimearray", "timedeltaarray"},
    ),
)
ABCNumpyExtensionArray = cast(
    "Type[NumpyExtensionArray]",
    create_pandas_abc_type("ABCNumpyExtensionArray", "_typ", ("npy_extension",)),
)
