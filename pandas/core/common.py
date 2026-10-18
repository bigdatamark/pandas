"""
Misc tools for implementing data structures

Note: pandas.core.common is *not* part of the public API.
"""

# [MODULE DOC]
# Miscellaneous internal utilities shared across the core package: nested
# sequence flattening, None-counting helpers, boolean-indexer and slice
# predicates, array and label conversion helpers (asarray_tuplesafe,
# index_labels_to_array, convert_to_list_like), callable helpers (naming,
# conditional application, pipe, rename-function building), a random-state
# normaliser, a temporary attribute-setting context manager, a mapping
# standardiser used by the dictionary export methods, a table mapping numpy and
# builtin reduction functions to internal method names, and a stack-frame
# helper for chained-assignment detection.
# Depends on the dtypes package, the compiled lib module and the column
# expression class (imported at import time from the col module, which means
# the col module must not import this one at module level). No other
# import-time side effects.

from __future__ import annotations

import builtins
from collections import (
    abc,
    defaultdict,
)
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Sequence,
)
import contextlib
from functools import partial
import inspect
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    TypeVar,
    cast,
    overload,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
)
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCNumpyExtensionArray,
    ABCSeries,
)
from pandas.core.dtypes.inference import iterable_not_string

from pandas.core.col import Expression

if TYPE_CHECKING:
    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        NpDtype,
        P,
        RandomState,
        T,
    )

    from pandas import Index


def flatten(line):
    # [FUNCTION DOC]
    # Generator that lazily flattens an arbitrarily nested sequence. Input
    # line: an iterable. Yields each leaf element in order; an element that is
    # iterable but not a string is recursed into, strings (and non-iterables)
    # are yielded as-is.
    """
    Flatten an arbitrarily nested sequence.

    Parameters
    ----------
    line : sequence
        The non string sequence to flatten

    Notes
    -----
    This doesn't consider strings sequences.

    Returns
    -------
    flattened : generator
    """
    for element in line:
        if iterable_not_string(element):
            yield from flatten(element)
        else:
            yield element


def consensus_name_attr(objs):
    # [FUNCTION DOC]
    # Finds a shared name among objects. Input objs: non-empty indexable
    # sequence of objects with a name attribute. Returns the first object's name
    # if every other object's name compares equal to it; returns none as soon as
    # one differs or the comparison itself raises a value error (for example a
    # name that is an array). An empty input raises an index error.
    name = objs[0].name
    for obj in objs[1:]:
        try:
            if obj.name != name:
                name = None
                break
        except ValueError:
            name = None
            break
    return name


def is_bool_indexer(key: Any) -> bool:
    # [FUNCTION DOC]
    # Decides whether key is a boolean mask. Input key: any object. Series,
    # arrays, indexes (except multi-level indexes) and extension arrays count if
    # their dtype is boolean; an object-dtype one counts only if all entries are
    # booleans, and if it has missing values mixed with booleans a value error
    # is raised (other object contents just return False). Plain lists count
    # when non-empty and entirely booleans (subclasses are copied to a plain list
    # first). Anything else returns False.
    """
    Check whether `key` is a valid boolean indexer.

    Parameters
    ----------
    key : Any
        Only list-likes may be considered boolean indexers.
        All other types are not considered a boolean indexer.
        For array-like input, boolean ndarrays or ExtensionArrays
        with ``_is_boolean`` set are considered boolean indexers.

    Returns
    -------
    bool
        Whether `key` is a valid boolean indexer.

    Raises
    ------
    ValueError
        When the array is an object-dtype ndarray or ExtensionArray
        and contains missing values.

    See Also
    --------
    check_array_indexer : Check that `key` is a valid array to index,
        and convert to an ndarray.
    """
    if isinstance(
        key,
        (ABCSeries, np.ndarray, ABCIndex, ABCExtensionArray, ABCNumpyExtensionArray),
    ) and not isinstance(key, ABCMultiIndex):
        if key.dtype == np.object_:
            key_array = np.asarray(key)

            if not lib.is_bool_array(key_array):
                na_msg = "Cannot mask with non-boolean array containing NA / NaN values"
                if lib.is_bool_array(key_array, skipna=True):
                    # Don't raise on e.g. ["A", "B", np.nan], see
                    #  test_loc_getitem_list_of_labels_categoricalindex_with_na
                    raise ValueError(na_msg)
                return False
            return True
        elif is_bool_dtype(key.dtype):
            return True
    elif isinstance(key, list):
        # check if np.array(key).dtype would be bool
        if len(key) > 0:
            if type(key) is not list:
                # GH#42461 cython will raise TypeError if we pass a subclass
                key = list(key)
            return lib.is_bool_list(key)

    return False


def cast_scalar_indexer(val):
    # [FUNCTION DOC]
    # Rejects whole-number float scalars used as positional keys. Input val: a
    # scalar (caller guarantees). Raises an index error if val is a float with an
    # integral value; otherwise returns val unchanged (non-integral floats and
    # other scalars pass through).
    """
    Disallow indexing with a float key, even if that key is a round number.

    Parameters
    ----------
    val : scalar

    Returns
    -------
    outval : scalar
    """
    # assumes lib.is_scalar(val)
    if lib.is_float(val) and val.is_integer():
        raise IndexError(
            # GH#34193
            "Indexing with a float is no longer supported. Manually convert "
            "to an integer key instead."
        )
    return val


def not_none(*args):
    # [FUNCTION DOC]
    # Returns a lazy generator over the arguments that are not none, in order.
    """
    Returns a generator consisting of the arguments that are not None.
    """
    return (arg for arg in args if arg is not None)


def any_none(*args) -> bool:
    # [FUNCTION DOC]
    # Returns True if at least one argument is none.
    """
    Returns a boolean indicating if any argument is None.
    """
    return any(arg is None for arg in args)


def all_none(*args) -> bool:
    # [FUNCTION DOC]
    # Returns True if every argument is none (also true for no arguments).
    """
    Returns a boolean indicating if all arguments are None.
    """
    return all(arg is None for arg in args)


def any_not_none(*args) -> bool:
    # [FUNCTION DOC]
    # Returns True if at least one argument is not none.
    """
    Returns a boolean indicating if any argument is not None.
    """
    return any(arg is not None for arg in args)


def all_not_none(*args) -> bool:
    # [FUNCTION DOC]
    # Returns True if no argument is none (also true for no arguments).
    """
    Returns a boolean indicating if all arguments are not None.
    """
    return all(arg is not None for arg in args)


def count_not_none(*args) -> int:
    # [FUNCTION DOC]
    # Returns how many arguments are not none.
    """
    Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)


@overload
def asarray_tuplesafe(
    values: ArrayLike | list | tuple | zip, dtype: NpDtype | None = ...
) -> np.ndarray:
    # [FUNCTION DOC]
    # Typing overload stub (array return for common input types); see the
    # implementation below.
    # ExtensionArray can only be returned when values is an Index, all other iterables
    # will return np.ndarray. Unfortunately "all other" cannot be encoded in a type
    # signature, so instead we special-case some common types.
    ...


@overload
def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = ...) -> ArrayLike:
    # [FUNCTION DOC]
    # Typing overload stub (general iterables); see the implementation below.
    ...


def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = None) -> ArrayLike:
    # [FUNCTION DOC]
    # Converts an iterable to a one-dimensional array without turning tuples
    # into extra dimensions. Inputs: values, any iterable; dtype, optional numpy
    # dtype (default none: infer). Iterables that are not lists, tuples or
    # array-convertible are first materialised as a list. Index and Series inputs
    # return their underlying values directly (possibly an extension array,
    # ignoring dtype). A list with an object dtype request builds an object
    # array directly. Otherwise numpy conversion is tried; a value error (ragged
    # input) falls back to an object array built from the list. Fixed-width
    # string results are redone as object arrays. A two-dimensional result is
    # replaced by an object array of tuples, one per row. Returns the array.
    if not (isinstance(values, (list, tuple)) or hasattr(values, "__array__")):
        values = list(values)
    elif isinstance(values, ABCIndex):
        return values._values
    elif isinstance(values, ABCSeries):
        return values._values

    if isinstance(values, list) and dtype in [np.object_, object]:
        return construct_1d_object_array_from_listlike(values)

    try:
        result = np.asarray(values, dtype=dtype)
    except ValueError:
        # Using try/except since it's more performant than checking is_list_like
        # over each element
        # error: Argument 1 to "construct_1d_object_array_from_listlike"
        # has incompatible type "Iterable[Any]"; expected "Sized"
        return construct_1d_object_array_from_listlike(values)  # type: ignore[arg-type]

    if issubclass(result.dtype.type, str):
        result = np.asarray(values, dtype=object)

    if result.ndim == 2:
        # Avoid building an array of arrays:
        values = [tuple(x) for x in values]
        result = construct_1d_object_array_from_listlike(values)

    return result


def index_labels_to_array(
    labels: np.ndarray | Iterable, dtype: NpDtype | None = None
) -> np.ndarray:
    # [FUNCTION DOC]
    # Converts a label or labels into an array for index construction. Inputs:
    # labels, a single label, tuple, string, array or iterable; dtype, optional
    # dtype (default infer). Strings and tuples are treated as a single label
    # (wrapped in a list); non-list, non-array iterables are listed; a
    # non-iterable is wrapped as one label. Delegates to the tuple-safe array
    # converter. Returns the array.
    """
    Transform label or iterable of labels to array, for use in Index.

    Parameters
    ----------
    dtype : dtype
        If specified, use as dtype of the resulting array, otherwise infer.

    Returns
    -------
    array
    """
    if isinstance(labels, (str, tuple)):
        labels = [labels]

    if not isinstance(labels, (list, np.ndarray)):
        try:
            labels = list(labels)
        except TypeError:  # non-iterable
            labels = [labels]

    rlabels = asarray_tuplesafe(labels, dtype=dtype)

    return rlabels


def maybe_make_list(obj):
    # [FUNCTION DOC]
    # Wraps obj in a one-element list unless it is none, a tuple or a list, in
    # which case it is returned unchanged.
    if obj is not None and not isinstance(obj, (tuple, list)):
        return [obj]
    return obj


def maybe_iterable_to_list(obj: Iterable[T] | T) -> Collection[T] | T:
    # [FUNCTION DOC]
    # Materialises one-shot iterables. If obj is iterable but has no length
    # (for example a generator), returns a new list of its items (consuming it);
    # anything else is returned unchanged.
    """
    If obj is Iterable but not list-like, consume into list.
    """
    if isinstance(obj, abc.Iterable) and not isinstance(obj, abc.Sized):
        return list(obj)
    obj = cast(Collection, obj)
    return obj


def is_null_slice(obj) -> bool:
    # [FUNCTION DOC]
    # True when obj is a slice with no start, stop or step (selects everything).
    """
    We have a null slice.
    """
    return (
        isinstance(obj, slice)
        and obj.start is None
        and obj.stop is None
        and obj.step is None
    )


def is_empty_slice(obj) -> bool:
    # [FUNCTION DOC]
    # True when obj is a slice whose start and stop are both given and equal
    # (selects nothing); the step is ignored.
    """
    We have an empty slice, e.g. no values are selected.
    """
    return (
        isinstance(obj, slice)
        and obj.start is not None
        and obj.stop is not None
        and obj.start == obj.stop
    )


def is_true_slices(line: abc.Iterable) -> abc.Generator[bool, None, None]:
    # [FUNCTION DOC]
    # Generator over line yielding, per element, True if it is a slice that is
    # not a null slice, else False.
    """
    Find non-trivial slices in "line": yields a bool.
    """
    for k in line:
        yield isinstance(k, slice) and not is_null_slice(k)


# TODO: used only once in indexing; belongs elsewhere?
def is_full_slice(obj, line: int) -> bool:
    # [FUNCTION DOC]
    # True when obj is a slice starting at zero, stopping exactly at line (the
    # length) with no step. A null slice does not count.
    """
    We have a full length slice.
    """
    return (
        isinstance(obj, slice)
        and obj.start == 0
        and obj.stop == line
        and obj.step is None
    )


def get_callable_name(obj):
    # [FUNCTION DOC]
    # Best-effort display name for a callable. Returns its own name attribute if
    # present; for a partial application, the name of the wrapped function;
    # for other callables, the class name; returns none if obj is not callable.
    # typical case has name
    if hasattr(obj, "__name__"):
        return obj.__name__
    # some objects don't; could recurse
    if isinstance(obj, partial):
        return get_callable_name(obj.func)
    # fall back to class name
    if callable(obj):
        return type(obj).__name__
    # everything failed (probably because the argument
    # wasn't actually callable); we return None
    # instead of the empty string in this case to allow
    # distinguishing between no name and a name of ''
    return None


def apply_if_callable(maybe_callable, obj, **kwargs):
    # [FUNCTION DOC]
    # Evaluates a possibly callable argument against obj. A column expression
    # object is evaluated via its private evaluation method with obj and the
    # keyword arguments; any other callable is called with obj and the keyword
    # arguments; non-callables are returned unchanged.
    """
    Evaluate possibly callable input using obj and kwargs if it is callable,
    otherwise return as it is.

    Parameters
    ----------
    maybe_callable : possibly a callable
    obj : NDFrame
    **kwargs
    """
    if isinstance(maybe_callable, Expression):
        return maybe_callable._eval_expression(obj, **kwargs)
    elif callable(maybe_callable):
        return maybe_callable(obj, **kwargs)

    return maybe_callable


def standardize_mapping(into):
    # [FUNCTION DOC]
    # Normalises a mapping specification into a constructor. Input into: a
    # mapping class, a mapping instance, or an initialised default-dict. An
    # initialised default-dict yields a partial constructor preserving its
    # default factory. Other instances are replaced by their class. Raises a
    # type error if the class is not a mapping subclass, or if it is the bare
    # default-dict class (uninitialised). Returns the class or partial.
    """
    Helper function to standardize a supplied mapping.

    Parameters
    ----------
    into : instance or subclass of collections.abc.Mapping
        Must be a class, an initialized collections.defaultdict,
        or an instance of a collections.abc.Mapping subclass.

    Returns
    -------
    mapping : a collections.abc.Mapping subclass or other constructor
        a callable object that can accept an iterator to create
        the desired Mapping.

    See Also
    --------
    DataFrame.to_dict
    Series.to_dict
    """
    if not inspect.isclass(into):
        if isinstance(into, defaultdict):
            return partial(defaultdict, into.default_factory)
        into = type(into)
    if not issubclass(into, abc.Mapping):
        raise TypeError(f"unsupported type: {into}")
    if into == defaultdict:
        raise TypeError("to_dict() only accepts initialized defaultdicts")
    return into


@overload
def random_state(state: np.random.Generator) -> np.random.Generator:
    # [FUNCTION DOC]
    # Typing overload stub (generator in, same generator out); see the
    # implementation below.
    ...


@overload
def random_state(
    state: int | np.ndarray | np.random.BitGenerator | np.random.RandomState | None,
) -> np.random.RandomState:
    # [FUNCTION DOC]
    # Typing overload stub for non-generator inputs; see the implementation
    # below.
    # KNOWN ISSUE: the declared return type is a seeded state object, but for a
    # none input the implementation returns the global random module instead.
    ...


def random_state(state: RandomState | None = None):
    # [FUNCTION DOC]
    # Normalises a random-state argument. Input state (default none): an
    # integer, array or bit generator creates a new legacy random state seeded
    # with it; an existing legacy state or new-style generator is returned
    # unchanged; none returns numpy's global random module (shared state). Any
    # other type raises a value error.
    """
    Helper function for processing random_state arguments.

    Parameters
    ----------
    state : int, array-like, BitGenerator, Generator, np.random.RandomState, None.
        If receives an int, array-like, or BitGenerator, passes to
        np.random.RandomState() as seed.
        If receives an np.random RandomState or Generator, just returns that unchanged.
        If receives `None`, returns np.random.
        If receives anything else, raises an informative ValueError.

        Default None.

    Returns
    -------
    np.random.RandomState or np.random.Generator. If state is None, returns np.random

    """
    if is_integer(state) or isinstance(state, (np.ndarray, np.random.BitGenerator)):
        return np.random.RandomState(state)
    elif isinstance(state, np.random.RandomState):
        return state
    elif isinstance(state, np.random.Generator):
        return state
    elif state is None:
        return np.random
    else:
        raise ValueError(
            "random_state must be an integer, array-like, a BitGenerator, Generator, "
            "a numpy RandomState, or None"
        )


_T = TypeVar("_T")  # Secondary TypeVar for use in pipe's type hints


@overload
def pipe(
    obj: _T,
    func: Callable[Concatenate[_T, P], T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    # [FUNCTION DOC]
    # Typing overload stub (callable applied with the object first); see the
    # implementation below.
    ...


@overload
def pipe(
    obj: Any,
    func: tuple[Callable[..., T], str],
    *args: Any,
    **kwargs: Any,
) -> T:
    # [FUNCTION DOC]
    # Typing overload stub (pair of callable and keyword name); see the
    # implementation below.
    ...


def pipe(
    obj: _T,
    func: Callable[Concatenate[_T, P], T] | tuple[Callable[..., T], str],
    *args: Any,
    **kwargs: Any,
) -> T:
    # [FUNCTION DOC]
    # Applies func to obj. If func is a pair (callable, keyword name), calls the
    # callable with the extra positional arguments and the keyword arguments plus
    # obj passed under the given keyword name; raises a value error if that
    # keyword is already in the keyword arguments. Otherwise calls func with obj
    # first followed by the extra arguments. Returns the call's result. Note the
    # keyword argument dict is mutated locally by the pair form (no effect on the
    # caller).
    """
    Apply a function ``func`` to object ``obj`` either by passing obj as the
    first argument to the function or, in the case that the func is a tuple,
    interpret the first element of the tuple as a function and pass the obj to
    that function as a keyword argument whose key is the value of the second
    element of the tuple.

    Parameters
    ----------
    func : callable or tuple of (callable, str)
        Function to apply to this object or, alternatively, a
        ``(callable, data_keyword)`` tuple where ``data_keyword`` is a
        string indicating the keyword of ``callable`` that expects the
        object.
    *args : iterable, optional
        Positional arguments passed into ``func``.
    **kwargs : dict, optional
        A dictionary of keyword arguments passed into ``func``.

    Returns
    -------
    object : the return type of ``func``.
    """
    if isinstance(func, tuple):
        # Assigning to func_ so pyright understands that it's a callable
        func_, target = func
        if target in kwargs:
            msg = f"{target} is both the pipe target and a keyword argument"
            raise ValueError(msg)
        kwargs[target] = obj
        return func_(*args, **kwargs)
    else:
        return func(obj, *args, **kwargs)


def get_rename_function(mapper):
    # [FUNCTION DOC]
    # Builds a label-mapping function. Input mapper: a mapping, a Series, or a
    # callable. For a mapping or Series, returns a function giving the mapped
    # value for labels present and the label itself otherwise; any other mapper
    # is returned as-is (assumed callable).
    """
    Returns a function that will map names/labels, dependent if mapper
    is a dict, Series or just a function.
    """

    def f(x):
        # [FUNCTION DOC]
        # HELPER: the generated lookup function: returns mapper's value for x when
        # x is a key, else x unchanged.
        if x in mapper:
            return mapper[x]
        else:
            return x

    return f if isinstance(mapper, (abc.Mapping, ABCSeries)) else mapper


def convert_to_list_like(
    values: Hashable | Iterable | AnyArrayLike,
) -> list | AnyArrayLike:
    # [FUNCTION DOC]
    # Ensures list-like output. Lists, numpy arrays, indexes, Series and
    # extension arrays are returned unchanged; other non-string iterables are
    # consumed into a new list; anything else (scalars, strings) is wrapped in a
    # one-element list.
    """
    Convert list-like or scalar input to list-like. List, numpy and pandas array-like
    inputs are returned unmodified whereas others are converted to list.
    """
    if isinstance(values, (list, np.ndarray, ABCIndex, ABCSeries, ABCExtensionArray)):
        return values
    elif isinstance(values, abc.Iterable) and not isinstance(values, str):
        return list(values)

    return [values]


@contextlib.contextmanager
def temp_setattr(obj, attr: str, value, condition: bool = True) -> Generator[None]:
    # [FUNCTION DOC]
    # Context manager that temporarily replaces an attribute. Inputs: obj, the
    # target; attr, attribute name; value, temporary value; condition (default
    # true): when false, nothing is changed. When active, remembers the old value
    # (the attribute must exist, else an attribute error), sets the new one,
    # yields obj, and restores the old value on exit even if the body raises.
    # KNOWN ISSUE: the generator is annotated to yield nothing, but it actually
    # yields the target object.
    """
    Temporarily set attribute on an object.

    Parameters
    ----------
    obj : object
        Object whose attribute will be modified.
    attr : str
        Attribute to modify.
    value : Any
        Value to temporarily set attribute to.
    condition : bool, default True
        Whether to set the attribute. Provided in order to not have to
        conditionally use this context manager.

    Yields
    ------
    object : obj with modified attribute.
    """
    if condition:
        old_value = getattr(obj, attr)
        setattr(obj, attr, value)
    try:
        yield obj
    finally:
        if condition:
            setattr(obj, attr, old_value)


def require_length_match(data, index: Index) -> None:
    # [FUNCTION DOC]
    # Raises a value error stating both lengths if data and index differ in
    # length; returns nothing otherwise.
    """
    Check the length of data matches the length of the index.
    """
    if len(data) != len(index):
        raise ValueError(
            "Length of values "
            f"({len(data)}) "
            "does not match length of index "
            f"({len(index)})"
        )


_cython_table = {
    builtins.sum: "sum",
    builtins.max: "max",
    builtins.min: "min",
    np.all: "all",
    np.any: "any",
    np.sum: "sum",
    np.nansum: "sum",
    np.mean: "mean",
    np.nanmean: "mean",
    np.prod: "prod",
    np.nanprod: "prod",
    np.std: "std",
    np.nanstd: "std",
    np.var: "var",
    np.nanvar: "var",
    np.median: "median",
    np.nanmedian: "median",
    np.max: "max",
    np.nanmax: "max",
    np.min: "min",
    np.nanmin: "min",
    np.cumprod: "cumprod",
    np.nancumprod: "cumprod",
    np.cumsum: "cumsum",
    np.nancumsum: "cumsum",
}


def get_cython_func(arg: Callable) -> str | None:
    # [FUNCTION DOC]
    # Looks up a numpy or builtin reduction function in the table of known
    # functions and returns the matching internal method name (for example the
    # name for sum), or none when the function is not in the table.
    """
    if we define an internal function for this argument, return it
    """
    return _cython_table.get(arg)


def fill_missing_names(names: Sequence[Hashable | None]) -> list[Hashable]:
    # [FUNCTION DOC]
    # Returns a new list in which each none entry is replaced by the text
    # "level_" followed by its position; other entries are kept as-is.
    """
    If a name is missing then replace it by level_n, where n is the count

    Parameters
    ----------
    names : list-like
        list of column names or None values.

    Returns
    -------
    list
        list of column names with the None values replaced.
    """
    return [f"level_{i}" if name is None else name for i, name in enumerate(names)]


def is_local_in_caller_frame(obj):
    # [FUNCTION DOC]
    # HELPER: for chained-assignment detection. Looks at the frame two levels up
    # the call stack (the caller of the function that called this helper) and
    # returns True if any of its local variables is the very same object as obj
    # (identity test), else False. Relies on the fixed call depth.
    """
    Helper function used in detecting chained assignment.

    If the pandas object (DataFrame/Series) is a local variable
    in the caller's frame, it should not be a case of chained
    assignment or method call.

    For example:

    def test():
        df = pd.DataFrame(...)
        df["a"] = 1  # not chained assignment

    Inside ``df.__setitem__``, we call this function to check whether `df`
    (`self`) is a local variable in `test` frame (the frame calling setitem). If
    so, we know it is not a case of chained assignment (even when the refcount
    of `df` is below the threshold due to optimization of local variables).
    """
    frame = sys._getframe(2)
    for v in frame.f_locals.values():
        if v is obj:
            return True
    return False
