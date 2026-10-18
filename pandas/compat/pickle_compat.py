"""
Pickle compatibility to pandas version 1.0
"""

# [MODULE DOC]
# HELPER: support for loading pickles written by older pandas versions.
# Public contents: _class_locations_map (mapping from old (module, name) pairs to
# new pairs for classes and functions that moved or were removed: the old
# block creation function, the renamed missing-time unpickle function, the
# removed numeric index classes which now map to the general index, and the
# sparse dtype location), the Unpickler class (a subclass of the pure-python
# unpickler from the pickle module, using a non-public class), the function
# loads and the context manager patch_pickle which temporarily substitutes it
# for the standard loads. Import-time side effects: builds the table and the
# dispatch-table copy inside the class body.
# KNOWN ISSUE: the module docstring says compatibility is to version 1.0, while
# the table also contains mappings for later removals (e.g. issue 50775).
from __future__ import annotations

import contextlib
import io
import pickle
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import BaseOffset

from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.internals import BlockManager

if TYPE_CHECKING:
    from collections.abc import Generator


# If classes are moved, provide compat here.
_class_locations_map = {
    # Re-routing unpickle block logic to go through _unpickle_block instead
    # for pandas <= 1.3.5
    ("pandas.core.internals.blocks", "new_block"): (
        "pandas._libs.internals",
        "_unpickle_block",
    ),
    # Avoid Cython's warning "contradiction to Python 'class private name' rules"
    ("pandas._libs.tslibs.nattype", "__nat_unpickle"): (
        "pandas._libs.tslibs.nattype",
        "_nat_unpickle",
    ),
    # 50775, remove Int64Index, UInt64Index & Float64Index from codebase
    ("pandas.core.indexes.numeric", "Int64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.indexes.numeric", "UInt64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.indexes.numeric", "Float64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.arrays.sparse.dtype", "SparseDtype"): (
        "pandas.core.dtypes.dtypes",
        "SparseDtype",
    ),
}


# our Unpickler sub-class to override methods and some dispatcher
# functions for compat and uses a non-public class of the pickle module.
class Unpickler(pickle._Unpickler):
    # [CLASS DOC]
    # Unpickler for old pandas pickles. Inherits from the pure-python unpickler
    # class of the standard library (a non-public class). Overrides class lookup
    # to apply the relocation table, and replaces the handlers for the reduce
    # and new-object opcodes in its own copy of the opcode dispatch table
    # (assigned in the class body, so the standard table is not modified) to
    # tolerate old serialization forms of offsets, period arrays, datetime and
    # timedelta arrays and block managers.
    def find_class(self, module: str, name: str) -> Any:
        # [FUNCTION DOC]
        # Resolves a class or function named in the pickle. Parameters: module
        # and name (strings). If the pair appears in the relocation table, the
        # mapped pair is used instead; then delegates to the standard lookup,
        # which imports the module and returns the attribute (and may raise the
        # usual import or attribute errors).
        key = (module, name)
        module, name = _class_locations_map.get(key, key)
        return super().find_class(module, name)

    dispatch = pickle._Unpickler.dispatch.copy()

    def load_reduce(self) -> None:
        # [FUNCTION DOC]
        # Handler for the reduce opcode: pops the argument tuple, takes the
        # callable on top of the stack and replaces it with the call result.
        # If the call raises a type error (an old pickle calling the wrong
        # constructor): when the first argument is an offset class, builds the
        # object by calling the class's own instance-creation method with the
        # arguments and stops; else if the first argument is a subclass of the
        # period array class, creates it through the array base class's creation
        # method; otherwise re-raises the type error.
        # KNOWN ISSUE: the period-array check does not first confirm that the first
        # argument is a class, so a non-class first argument makes the subclass test
        # itself raise a type error, hiding the original one. The local variable
        # holding the class in that branch is assigned but never used.
        stack = self.stack  # type: ignore[attr-defined]
        args = stack.pop()
        func = stack[-1]

        try:
            stack[-1] = func(*args)
        except TypeError:
            # If we have a deprecated function,
            # try to replace and try again.
            if args and isinstance(args[0], type) and issubclass(args[0], BaseOffset):
                # TypeError: object.__new__(Day) is not safe, use Day.__new__()
                cls = args[0]
                stack[-1] = cls.__new__(*args)
                return
            elif args and issubclass(args[0], PeriodArray):
                cls = args[0]
                stack[-1] = NDArrayBacked.__new__(*args)
                return
            raise

    dispatch[pickle.REDUCE[0]] = load_reduce  # type: ignore[assignment]

    def load_newobj(self) -> None:
        # [FUNCTION DOC]
        # Handler for the new-object opcode: pops the argument tuple and the class
        # and pushes a new instance made by the class's creation method. Special
        # cases for old pickles with empty arguments: a datetime array subclass or
        # timedelta array subclass is created from an empty array of the
        # nanosecond dtype and that dtype; the block manager class is created
        # with empty blocks, empty axes and a false verification flag. Otherwise
        # the class is called with the popped arguments.
        args = self.stack.pop()  # type: ignore[attr-defined]
        cls = self.stack.pop()  # type: ignore[attr-defined]

        # compat
        if issubclass(cls, DatetimeArray) and not args:
            arr = np.array([], dtype="M8[ns]")
            obj = cls.__new__(cls, arr, arr.dtype)
        elif issubclass(cls, TimedeltaArray) and not args:
            arr = np.array([], dtype="m8[ns]")
            obj = cls.__new__(cls, arr, arr.dtype)
        elif cls is BlockManager and not args:
            obj = cls.__new__(cls, (), [], False)
        else:
            obj = cls.__new__(cls, *args)
        self.append(obj)  # type: ignore[attr-defined]

    dispatch[pickle.NEWOBJ[0]] = load_newobj  # type: ignore[assignment]


def loads(
    bytes_object: bytes,
    *,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
) -> Any:
    # [FUNCTION DOC]
    # Deserializes a byte string using the compatibility unpickler. Parameters:
    # bytes_object (pickled data), and keyword-only fix_imports (default true),
    # encoding (default ASCII) and errors (default strict), passed to the
    # unpickler as for the standard function. Wraps the bytes in an in-memory
    # stream and returns the loaded object. Raises the usual unpickling errors.
    """
    Analogous to pickle._loads.
    """
    fd = io.BytesIO(bytes_object)
    return Unpickler(
        fd, fix_imports=fix_imports, encoding=encoding, errors=errors
    ).load()


@contextlib.contextmanager
def patch_pickle() -> Generator[None]:
    # [FUNCTION DOC]
    # Context manager (generator based) that replaces the standard pickle module's
    # loads function with the compatibility loader for the duration of the with
    # block and restores the original afterwards, also on error. Yields nothing.
    # Global side effect: changes an attribute of the standard module; not
    # thread safe.
    """
    Temporarily patch pickle to use our unpickler.
    """
    orig_loads = pickle.loads
    try:
        setattr(pickle, "loads", loads)
        yield
    finally:
        setattr(pickle, "loads", orig_loads)
