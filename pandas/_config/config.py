"""
The config module holds package-wide configurables and provides
a uniform API for working with them.

Overview
========

This module supports the following requirements:
- options are referenced using keys in dot.notation, e.g. "x.y.option - z".
- keys are case-insensitive.
- functions should accept partial/regex keys, when unambiguous.
- options can be registered by modules at import time.
- options can be registered at init-time (via core.config_init)
- options have a default value, and (optionally) a description and
  validation function associated with them.
- options can be deprecated, in which case referencing them
  should produce a warning.
- deprecated options can optionally be rerouted to a replacement
  so that accessing a deprecated option reroutes to a differently
  named option.
- options can be reset to their default value.
- all option can be reset to their default value at once.
- all options in a certain sub - namespace can be reset at once.
- the user can set / get / reset or ask for the description of an option.
- a developer can register and mark an option as deprecated.
- you can register a callback to be invoked when the option value
  is set or reset. Changing the stored value is considered misuse, but
  is not verboten.

Implementation
==============

- Data is stored using nested dictionaries, and should be accessed
  through the provided API.

- "Registered options" and "Deprecated options" have metadata associated
  with them, which are stored in auxiliary dictionaries keyed on the
  fully-qualified key, e.g. "x.y.z.option".

- the config_init module is imported by the package's __init__.py file.
  placing any register_option() calls there will ensure those options
  are available as soon as pandas is loaded. If you use register_option
  in a module, it will only be available after that module is imported,
  which you should be aware of.

- `config_prefix` is a context_manager (for use with the `with` keyword)
  which can save developers some typing, see the docstring.

"""

# [MODULE DOC]
# The option registry and the API for reading and writing pandas options.
# Storage (module globals): _global_config (nested dictionaries of current
# values keyed by the parts of dotted option names), _registered_options (flat
# mapping from full lowercase key to registration metadata),
# _deprecated_options (flat mapping from key to deprecation metadata) and
# _reserved_keys (names that cannot be options; only the word all).
# Public contents: metadata tuple types DeprecatedOption and RegisteredOption;
# OptionError (both an attribute error and a key error; presents itself as
# belonging to the errors module); user functions get_option, set_option,
# describe_option, reset_option, the option_context context manager, and the
# attribute-style access object options (a DictWrapper over the global tree);
# developer functions register_option, deprecate_option, config_prefix and
# get_default_val; validator factories is_type_factory, is_instance_factory,
# is_one_of_factory and the validators is_nonnegative_int, is_callable and the
# ready-made is_int, is_bool, is_float, is_str and is_text.
# Option lookup accepts a regular expression that must match exactly one key
# (case-insensitive); an exact key is always preferred.
# Import-time side effects: creates the global options object and the empty
# registries, and sets the reported module of several user functions and of the
# options object to the pandas top-level name (done here because using the
# normal decorator would create a circular import).
from __future__ import annotations

from contextlib import contextmanager
import re
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    cast,
)
import warnings

from pandas._typing import F
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Sequence,
    )


class DeprecatedOption(NamedTuple):
    # [CLASS DOC]
    # Immutable record (named tuple) describing a deprecated option. Fields: key
    # (full option name), category (warning class to emit), msg (custom warning
    # text or none), rkey (name of the replacement option to reroute to, or
    # none) and removal_ver (version string in which it will be removed, or
    # none). Stored in the deprecation registry by deprecate_option.
    key: str
    category: type[Warning]
    msg: str | None
    rkey: str | None
    removal_ver: str | None


class RegisteredOption(NamedTuple):
    # [CLASS DOC]
    # Immutable record (named tuple) describing a registered option. Fields: key
    # (full lowercase option name), defval (default value), doc (description
    # text), validator (function raising a value error for illegal values, or
    # none) and cb (callback invoked with the full key after the value is set,
    # or none). Stored in the registration registry by register_option.
    key: str
    defval: Any
    doc: str
    validator: Callable[[object], Any] | None
    cb: Callable[[str], Any] | None


# holds deprecated option metadata
_deprecated_options: dict[str, DeprecatedOption] = {}

# holds registered option metadata
_registered_options: dict[str, RegisteredOption] = {}

# holds the current values for registered options
_global_config: dict[str, Any] = {}

# keys which have a special meaning
_reserved_keys: list[str] = ["all"]


class OptionError(AttributeError, KeyError):
    # [CLASS DOC]
    # Exception type raised for problems with options: unknown or ambiguous
    # keys, duplicate registration, invalid assignment through the options
    # object, and similar. Inherits from both the attribute error and key error
    # types so that attribute-style access and mapping-style checks both work
    # (backwards compatible with key error checks). Its reported module is
    # overridden to be the errors module of pandas for public display. No extra
    # state or methods.
    """
    Exception raised for pandas.options.

    Backwards compatible with KeyError checks.

    See Also
    --------
    options : Access and modify global pandas settings.

    Examples
    --------
    >>> pd.options.context
    Traceback (most recent call last):
    OptionError: No such option
    """

    __module__ = "pandas.errors"


#
# User API


def _get_single_key(pat: str) -> str:
    # [FUNCTION DOC]
    # HELPER: resolves a user supplied pattern to exactly one real option key.
    # Parameter pat is a string (exact name or regular expression). Selects
    # matching keys; if none match, first emits the deprecation warning for the
    # pattern itself if it names a deprecated option, then raises an option
    # error saying no such key. If more than one matches, raises an option error
    # saying the pattern matched multiple keys. For the single match, emits a
    # deprecation warning if that key is deprecated, then translates it to its
    # replacement key if one is defined, and returns the resulting key string.
    keys = _select_options(pat)
    if len(keys) == 0:
        _warn_if_deprecated(pat)
        raise OptionError(f"No such keys(s): {pat!r}")
    if len(keys) > 1:
        raise OptionError("Pattern matched multiple keys")
    key = keys[0]

    _warn_if_deprecated(key)

    key = _translate_key(key)

    return key


def get_option(pat: str) -> Any:
    # [FUNCTION DOC]
    # Returns the current value of one option. Parameter pat is an option name
    # or a regular expression matching exactly one registered option. Resolves
    # the key (which may emit a deprecation warning and may reroute to a
    # replacement option), then walks the nested value tree to read the value.
    # Raises an option error if no key or more than one key matches.
    # Note the reported module name is later set to pandas, and that the name
    # may be temporarily rebound by the prefix context manager.
    """
    Retrieve the value of the specified option.

    This method allows users to query the current value of a given option
    in the pandas configuration system. Options control various display,
    performance, and behavior-related settings within pandas.

    Parameters
    ----------
    pat : str
        Regexp which should match a single option.

        .. warning::

            Partial matches are supported for convenience, but unless you use the
            full option name (e.g. x.y.z.option_name), your code may break in future
            versions if new options with similar names are introduced.

    Returns
    -------
    Any
        The value of the option.

    Raises
    ------
    OptionError : if no such option exists

    See Also
    --------
    set_option : Set the value of the specified option or options.
    reset_option : Reset one or more options to their default value.
    describe_option : Print the description for one or more registered options.

    Notes
    -----
    For all available options, please view the :ref:`User Guide <options.available>`
    or use ``pandas.describe_option()``.

    Examples
    --------
    >>> pd.get_option("display.max_columns")  # doctest: +SKIP
    4
    """
    key = _get_single_key(pat)

    # walk the nested dict
    root, k = _get_root(key)
    return root[k]


def set_option(*args) -> None:
    # [FUNCTION DOC]
    # Sets one or more options. Takes either a single mapping of pattern to
    # value, or an even number of positional arguments alternating pattern and
    # value. A single mapping is flattened into the alternating form.
    # Raises a value error if there are no arguments or an odd number of them;
    # an option error from pattern resolution (none or ambiguous match).
    # For each pair, in order: resolves the pattern to a key (possibly with a
    # deprecation warning and rerouting), runs the option's validator (which can
    # raise a value error and leaves that option and later ones unchanged but
    # earlier pairs already applied), stores the value in the tree, and calls the
    # option's callback with the key if one is registered. Pairs are applied one
    # at a time, so the operation is not atomic. Mutates global configuration.
    # Calling with keyword arguments is a type error by the language.
    """
    Set the value of the specified option or options.

    This method allows fine-grained control over the behavior and display settings
    of pandas. Options affect various functionalities such as output formatting,
    display limits, and operational behavior. Settings can be modified at runtime
    without requiring changes to global configurations or environment variables.

    Parameters
    ----------
    *args : str | object | dict
        Arguments provided in pairs, which will be interpreted as (pattern, value),
        or as a single dictionary containing multiple option-value pairs.
        pattern: str
        Regexp which should match a single option
        value: object
        New value of option

        .. warning::

            Partial pattern matches are supported for convenience, but unless you
            use the full option name (e.g. x.y.z.option_name), your code may break in
            future versions if new options with similar names are introduced.

    Returns
    -------
    None
        No return value.

    Raises
    ------
    ValueError if odd numbers of non-keyword arguments are provided
    TypeError if keyword arguments are provided
    OptionError if no such option exists

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    reset_option : Reset one or more options to their default value.
    describe_option : Print the description for one or more registered options.
    option_context : Context manager to temporarily set options in a ``with``
        statement.

    Notes
    -----
    For all available options, please view the :ref:`User Guide <options.available>`
    or use ``pandas.describe_option()``.

    Examples
    --------
    Option-Value Pair Input:

    >>> pd.set_option("display.max_columns", 4)
    >>> df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> df
    0  1  ...  3   4
    0  1  2  ...  4   5
    1  6  7  ...  9  10
    [2 rows x 5 columns]
    >>> pd.reset_option("display.max_columns")

    Dictionary Input:

    >>> pd.set_option({"display.max_columns": 4, "display.precision": 1})
    >>> df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> df
    0  1  ...  3   4
    0  1  2  ...  4   5
    1  6  7  ...  9  10
    [2 rows x 5 columns]
    >>> pd.reset_option("display.max_columns")
    >>> pd.reset_option("display.precision")
    """
    # Handle dictionary input
    if len(args) == 1 and isinstance(args[0], dict):
        args = tuple(kv for item in args[0].items() for kv in item)

    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    for k, v in zip(args[::2], args[1::2], strict=True):
        key = _get_single_key(k)

        opt = _get_registered_option(key)
        if opt and opt.validator:
            opt.validator(v)

        # walk the nested dict
        root, k_root = _get_root(key)
        root[k_root] = v

        if opt.cb:
            opt.cb(key)


def describe_option(pat: str = "", _print_desc: bool = True) -> str | None:
    # [FUNCTION DOC]
    # Builds and shows the description of options matching a pattern.
    # Parameters: pat (string or regular expression; empty matches all options;
    # the word all also matches all) and _print_desc (when true, default,
    # prints the text and returns none; when false, returns the text).
    # Raises an option error if nothing matches. The text is the descriptions of
    # all matching keys in sorted order joined with newlines (see the private
    # description builder). Note it does not require a single match.
    """
    Print the description for one or more registered options.

    Call with no arguments to get a listing for all registered options.

    Parameters
    ----------
    pat : str, default ""
        String or string regexp pattern.
        Empty string will return all options.
        For regexp strings, all matching keys will have their description displayed.
    _print_desc : bool, default True
        If True (default) the description(s) will be printed to stdout.
        Otherwise, the description(s) will be returned as a string
        (for testing).

    Returns
    -------
    None
        If ``_print_desc=True``.
    str
        If the description(s) as a string if ``_print_desc=False``.

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    set_option : Set the value of the specified option or options.
    reset_option : Reset one or more options to their default value.

    Notes
    -----
    For all available options, please view the
    :ref:`User Guide <options.available>`.

    Examples
    --------
    >>> pd.describe_option("display.max_columns")  # doctest: +SKIP
    display.max_columns : int
        If max_cols is exceeded, switch to truncate view...
    """
    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError(f"No such keys(s) for {pat=}")

    s = "\n".join([_build_option_description(k) for k in keys])

    if _print_desc:
        print(s)
        return None
    return s


def reset_option(pat: str) -> None:
    # [FUNCTION DOC]
    # Resets matching options to their registered default values. Parameter pat
    # is a string or regular expression; the special word all matches every
    # option. Raises an option error if nothing matches. Raises a value error if
    # more than one key matches, the pattern is shorter than four characters
    # and is not the word all (a safety check against accidental mass resets).
    # Each matching key is reset by calling the setter with its default, so
    # validators and callbacks run and rerouting of deprecated keys applies.
    # Mutates global configuration.
    """
    Reset one or more options to their default value.

    This method resets the specified pandas option(s) back to their default
    values. It allows partial string matching for convenience, but users should
    exercise caution to avoid unintended resets due to changes in option names
    in future versions.

    Parameters
    ----------
    pat : str/regex
        If specified only options matching ``pat*`` will be reset.
        Pass ``"all"`` as argument to reset all options.

        .. warning::

            Partial matches are supported for convenience, but unless you
            use the full option name (e.g. x.y.z.option_name), your code may break
            in future versions if new options with similar names are introduced.

    Returns
    -------
    None
        No return value.

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    set_option : Set the value of the specified option or options.
    describe_option : Print the description for one or more registered options.

    Notes
    -----
    For all available options, please view the
    :ref:`User Guide <options.available>`.

    Examples
    --------
    >>> pd.reset_option("display.max_columns")  # doctest: +SKIP
    """
    keys = _select_options(pat)

    if len(keys) == 0:
        raise OptionError(f"No such keys(s) for {pat=}")

    if len(keys) > 1 and len(pat) < 4 and pat != "all":
        raise ValueError(
            "You must specify at least 4 characters when "
            "resetting multiple keys, use the special keyword "
            '"all" to reset all the options to their default value'
        )

    for k in keys:
        set_option(k, _registered_options[k].defval)


def get_default_val(pat: str):
    # [FUNCTION DOC]
    # Returns the registered default value of a single option. Parameter pat
    # is an option name or regular expression matching exactly one option;
    # resolution rules and errors are the same as for get_option. No docstring
    # exists for this developer function.
    key = _get_single_key(pat)
    return _get_registered_option(key).defval


class DictWrapper:
    # [CLASS DOC]
    # Attribute-style view over the nested option dictionary (used for the
    # options object). State: d (the wrapped dictionary, a subtree of the global
    # tree) and prefix (the dotted path of that subtree). Reading an attribute
    # returns either another wrapper for a subtree or the current option value
    # via the official getter; assigning to an attribute sets an existing leaf
    # option via the official setter. Directory listing shows the keys of the
    # subtree. Because it overrides attribute assignment, internal state is set
    # with the base object setter. Invariant: new options cannot be created and
    # subtrees cannot be overwritten through this class.
    """provide attribute-style access to a nested dict"""

    d: dict[str, Any]

    def __init__(self, d: dict[str, Any], prefix: str = "") -> None:
        # [FUNCTION DOC]
        # Stores the wrapped dictionary d and the dotted path prefix (default
        # empty string) using the base object attribute setter to bypass the
        # custom assignment behavior.
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "prefix", prefix)

    def __setattr__(self, key: str, val: Any) -> None:
        # [FUNCTION DOC]
        # Handles assignment of an attribute, which means setting an option.
        # Builds the full dotted name from the stored prefix and key. If the key
        # exists in the wrapped dictionary and its entry is not itself a
        # dictionary (a leaf option), delegates to the official setter with the
        # full name and value. Otherwise (unknown key, or a subtree) raises an
        # option error saying only existing options can be set.
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        # you can't set new keys
        # can you can't overwrite subtrees
        if key in self.d and not isinstance(self.d[key], dict):
            set_option(prefix, val)
        else:
            raise OptionError("You can only set the value of existing options")

    def __getattr__(self, key: str):
        # [FUNCTION DOC]
        # Handles attribute reads that are not found normally. Builds the full
        # dotted name from prefix and key. Looks the key up in the wrapped
        # dictionary: a missing key raises an option error (message says no such
        # option; chained from the key error). If the entry is a dictionary
        # returns a new wrapper over it with the extended prefix; otherwise
        # returns the value obtained through the official getter, so deprecation
        # warnings and rerouting apply.
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        try:
            v = object.__getattribute__(self, "d")[key]
        except KeyError as err:
            raise OptionError("No such option") from err
        if isinstance(v, dict):
            return DictWrapper(v, prefix)
        else:
            return get_option(prefix)

    def __dir__(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns the names available for tab completion: the keys of the wrapped
        # dictionary as a list (subtree names and option names at this level).
        return list(self.d.keys())


options = DictWrapper(_global_config)
# DictWrapper defines a custom setattr
object.__setattr__(options, "__module__", "pandas")

#
# Functions for use by pandas developers, in addition to User - api


@contextmanager
def option_context(*args) -> Generator[None]:
    # [FUNCTION DOC]
    # Context manager (generator based) that sets options for the duration of
    # the with block and restores their previous values afterwards, also when
    # the block raises. Takes a single mapping of pattern to value, or an even
    # number (at least two) of positional arguments alternating pattern and
    # value. Raises a value error for an empty or odd argument list (before
    # anything is changed). Yields nothing.
    # Flow: first records the current value of every pattern (so unknown or
    # ambiguous patterns raise an option error before any change), then sets
    # each in order; on exit sets every recorded value back in order.
    """
    Context manager to temporarily set options in a ``with`` statement.

    This method allows users to set one or more pandas options temporarily
    within a controlled block. The previous options' values are restored
    once the block is exited. This is useful when making temporary adjustments
    to pandas' behavior without affecting the global state.

    Parameters
    ----------
    *args : str | object | dict
        An even amount of arguments provided in pairs which will be
        interpreted as (pattern, value) pairs. Alternatively, a single
        dictionary of {pattern: value} may be provided.

    Returns
    -------
    None
        No return value.

    Yields
    ------
    None
        No yield value.

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    set_option : Set the value of the specified option.
    reset_option : Reset one or more options to their default value.
    describe_option : Print the description for one or more registered options.

    Notes
    -----
    For all available options, please view the :ref:`User Guide <options.available>`
    or use ``pandas.describe_option()``.

    Examples
    --------
    >>> from pandas import option_context
    >>> with option_context("display.max_rows", 10, "display.max_columns", 5):
    ...     pass
    >>> with option_context({"display.max_rows": 10, "display.max_columns": 5}):
    ...     pass
    """
    if len(args) == 1 and isinstance(args[0], dict):
        args = tuple(kv for item in args[0].items() for kv in item)

    if len(args) % 2 != 0 or len(args) < 2:
        raise ValueError(
            "Provide an even amount of arguments as "
            "option_context(pat, val, pat, val...)."
        )

    ops = tuple(zip(args[::2], args[1::2], strict=True))
    undo: tuple[tuple[Any, Any], ...] = ()
    try:
        undo = tuple((pat, get_option(pat)) for pat, val in ops)
        for pat, val in ops:
            set_option(pat, val)
        yield
    finally:
        for pat, val in undo:
            set_option(pat, val)


def register_option(
    key: str,
    defval: object,
    doc: str = "",
    validator: Callable[[object], Any] | None = None,
    cb: Callable[[str], Any] | None = None,
) -> None:
    # [FUNCTION DOC]
    # Registers a new option in the package-wide registry. Parameters: key
    # (dotted name, lowercased before use), defval (default and initial value),
    # doc (description text, default empty), validator (optional function that
    # raises a value error for illegal values) and cb (optional callback called
    # with the full key after the option is set or reset).
    # Raises an option error if the key is already registered, is a reserved key
    # (the word all), or if an existing option occupies a prefix of the path (a
    # leaf where a subtree is needed) or the final position holds a subtree. Raises
    # a value error if a path part is not a valid identifier or is a keyword,
    # and whatever the validator raises for the default value.
    # Effects: creates intermediate dictionaries in the global value tree,
    # stores the default at the leaf, and records the metadata in the
    # registration registry. Not thread safe.
    # KNOWN ISSUE: the existing docstring lists a validator-related error but
    # does not mention the option errors or identifier errors that are raised.
    """
    Register an option in the package-wide pandas config object

    Parameters
    ----------
    key : str
        Fully-qualified key, e.g. "x.y.option - z".
    defval : object
        Default value of the option.
    doc : str
        Description of the option.
    validator : Callable, optional
        Function of a single argument, should raise `ValueError` if
        called with a value which is not a legal value for the option.
    cb
        a function of a single argument "key", which is called
        immediately after an option value is set/reset. key is
        the full name of the option.

    Raises
    ------
    ValueError if `validator` is specified and `defval` is not a valid value.

    """
    import keyword
    import tokenize

    key = key.lower()

    if key in _registered_options:
        raise OptionError(f"Option '{key}' has already been registered")
    if key in _reserved_keys:
        raise OptionError(f"Option '{key}' is a reserved key")

    # the default value should be legal
    if validator:
        validator(defval)

    # walk the nested dict, creating dicts as needed along the path
    path = key.split(".")

    for k in path:
        if not re.match("^" + tokenize.Name + "$", k):
            raise ValueError(f"{k} is not a valid identifier")
        if keyword.iskeyword(k):
            raise ValueError(f"{k} is a python keyword")

    cursor = _global_config
    msg = "Path prefix to option '{option}' is already an option"

    for i, p in enumerate(path[:-1]):
        if not isinstance(cursor, dict):
            raise OptionError(msg.format(option=".".join(path[:i])))
        if p not in cursor:
            cursor[p] = {}
        cursor = cursor[p]

    if not isinstance(cursor, dict):
        raise OptionError(msg.format(option=".".join(path[:-1])))

    cursor[path[-1]] = defval  # initialize

    # save the option metadata
    _registered_options[key] = RegisteredOption(
        key=key, defval=defval, doc=doc, validator=validator, cb=cb
    )


def deprecate_option(
    key: str,
    category: type[Warning],
    msg: str | None = None,
    rkey: str | None = None,
    removal_ver: str | None = None,
) -> None:
    # [FUNCTION DOC]
    # Marks an option as deprecated so later accesses emit a warning.
    # Parameters: key (full option name, lowercased), category (warning class),
    # msg (optional custom message), rkey (optional replacement option; accesses
    # are rerouted to it for get, set and reset) and removal_ver (optional
    # version string used in the default message). Neither key nor rkey need
    # exist. Raises an option error if the key is already deprecated. Records a
    # deprecation metadata tuple in the deprecation registry (global state).
    """
    Mark option `key` as deprecated, if code attempts to access this option,
    a warning will be produced, using `msg` if given, or a default message
    if not.
    if `rkey` is given, any access to the key will be re-routed to `rkey`.

    Neither the existence of `key` nor that if `rkey` is checked. If they
    do not exist, any subsequence access will fail as usual, after the
    deprecation warning is given.

    Parameters
    ----------
    key : str
        Name of the option to be deprecated.
        must be a fully-qualified option name (e.g "x.y.z.rkey").
    category : Warning
        Warning class for the deprecation.
    msg : str, optional
        Warning message to output when the key is referenced.
        if no message is given a default message will be emitted.
    rkey : str, optional
        Name of an option to reroute access to.
        If specified, any referenced `key` will be
        re-routed to `rkey` including set/get/reset.
        rkey must be a fully-qualified option name (e.g "x.y.z.rkey").
        used by the default message if no `msg` is specified.
    removal_ver : str, optional
        Specifies the version in which this option will
        be removed. used by the default message if no `msg` is specified.

    Raises
    ------
    OptionError
        If the specified key has already been deprecated.
    """
    key = key.lower()

    if key in _deprecated_options:
        raise OptionError(f"Option '{key}' has already been defined as deprecated.")

    _deprecated_options[key] = DeprecatedOption(key, category, msg, rkey, removal_ver)


#
# functions internal to the module


def _select_options(pat: str) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: finds registered option keys for a pattern. Returns a one-item
    # list if the pattern is exactly a registered key (short-circuit); otherwise
    # the sorted list of all keys if the pattern is the word all; otherwise the
    # sorted keys that contain a match of the pattern taken as a regular
    # expression, ignoring case (a search, not a full match). May return an
    # empty list. Raises the regular expression error for invalid patterns.
    """
    returns a list of keys matching `pat`

    if pat=="all", returns all registered options
    """
    # short-circuit for exact key
    if pat in _registered_options:
        return [pat]

    # else look through all of them
    keys = sorted(_registered_options.keys())
    if pat == "all":  # reserved key
        return keys

    return [k for k in keys if re.search(pat, k, re.I)]


def _get_root(key: str) -> tuple[dict[str, Any], str]:
    # [FUNCTION DOC]
    # HELPER: locates the dictionary that directly holds an option's value.
    # Parameter key is a full dotted name. Walks the global value tree through
    # all path parts except the last. Returns the innermost dictionary and the
    # final path part, so the caller can read or write the value by that name.
    # Raises a key error if an intermediate part does not exist.
    path = key.split(".")
    cursor = _global_config
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _get_deprecated_option(key: str):
    # [FUNCTION DOC]
    # HELPER: looks up deprecation metadata. Parameter key is a full option
    # name. Returns the deprecation record if the key is in the deprecation
    # registry, otherwise none.
    """
    Retrieves the metadata for a deprecated option, if `key` is deprecated.

    Returns
    -------
    DeprecatedOption (namedtuple) if key is deprecated, None otherwise
    """
    try:
        d = _deprecated_options[key]
    except KeyError:
        return None
    else:
        return d


def _get_registered_option(key: str):
    # [FUNCTION DOC]
    # HELPER: looks up registration metadata. Parameter key is a full option
    # name. Returns the registration record, or none if the key is not
    # registered.
    # KNOWN ISSUE: the existing docstring says the record is returned "if key is
    # deprecated", but the code checks the registration registry, not the
    # deprecation registry.
    """
    Retrieves the option metadata if `key` is a registered option.

    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """
    return _registered_options.get(key)


def _translate_key(key: str) -> str:
    # [FUNCTION DOC]
    # HELPER: applies deprecation rerouting. Parameter key is a full option
    # name. If the key is deprecated and has a non-empty replacement key,
    # returns the replacement; otherwise returns the key unchanged.
    """
    if `key` is deprecated and a replacement key defined, will return the
    replacement key, otherwise returns `key` as-is
    """
    d = _get_deprecated_option(key)
    if d:
        return d.rkey or key
    else:
        return key


def _warn_if_deprecated(key: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: emits the deprecation warning for a key. Parameter key is a full
    # option name. If the key is not deprecated returns false without a
    # warning. Otherwise emits a warning of the recorded category, attributed
    # to the first stack frame outside the library: the custom message if one
    # was recorded; else the default message "<key> is deprecated", extended
    # with the removal version when known and either the replacement key to use
    # or advice to refrain from using it. Returns true when deprecated.
    """
    Checks if `key` is a deprecated option and if so, prints a warning.

    Returns
    -------
    bool - True if `key` is deprecated, False otherwise.
    """
    d = _get_deprecated_option(key)
    if d:
        if d.msg:
            warnings.warn(
                d.msg,
                d.category,
                stacklevel=find_stack_level(),
            )
        else:
            msg = f"'{key}' is deprecated"
            if d.removal_ver:
                msg += f" and will be removed in {d.removal_ver}"
            if d.rkey:
                msg += f", please use '{d.rkey}' instead."
            else:
                msg += ", please refrain from using it."

            warnings.warn(
                msg,
                d.category,
                stacklevel=find_stack_level(),
            )
        return True
    return False


def _build_option_description(k: str) -> str:
    # [FUNCTION DOC]
    # HELPER: for describe_option. Builds the text describing one registered
    # option. Parameter k is the full key. The text starts with the key and a
    # space, then the stripped option documentation (or a no-description
    # phrase if empty), then an indented line with the default and current
    # value (obtained via the getter with future and deprecation warnings
    # suppressed), and, if the option is deprecated, an indented note naming the
    # replacement key (empty if none). Returns the string; does not print.
    # KNOWN ISSUE: the existing docstring says the description is printed, but
    # the function only returns the string.
    # KNOWN ISSUE: the later truthiness test of the registration record is
    # dead logic, since the record was already dereferenced for its
    # documentation (a missing record would already have raised).
    """Builds a formatted description of a registered option and prints it"""
    o = _get_registered_option(k)
    d = _get_deprecated_option(k)

    s = f"{k} "

    if o.doc:
        s += "\n".join(o.doc.strip().split("\n"))
    else:
        s += "No description available."

    if o:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            s += f"\n    [default: {o.defval}] [currently: {get_option(k)}]"

    if d:
        rkey = d.rkey or ""
        s += "\n    (Deprecated"
        s += f", use `{rkey}` instead."
        s += ")"

    return s


# helpers


@contextmanager
def config_prefix(prefix: str) -> Generator[None]:
    # [FUNCTION DOC]
    # Context manager (generator based) that prefixes option names. Parameter
    # prefix is a dotted name part (e.g. a section). Inside the with block,
    # calls to the module-level setter, getter and registration functions made
    # through this module's global names have the prefix and a dot prepended to
    # their first argument. Implemented by temporarily rebinding those three
    # module globals to wrapper functions, and restoring the originals in a
    # finally step (so also on error). The reset function is not wrapped; the
    # reset function itself calls the setter with the key directly.
    # Yields nothing. Not thread safe; has no effect on code that imported the
    # functions by name into its own namespace.
    # KNOWN ISSUE: the example in the existing docstring passes the name of a
    # variable instead of a string to the setter and getter, which would fail.
    """
    contextmanager for multiple invocations of API with a common prefix

    supported API functions: (register / get / set )__option

    Warning: This is not thread - safe, and won't work properly if you import
    the API functions into your module using the "from x import y" construct.

    Example
    -------
    import pandas._config.config as cf
    with cf.config_prefix("display.font"):
        cf.register_option("color", "red")
        cf.register_option("size", " 5 pt")
        cf.set_option(size, " 6 pt")
        cf.get_option(size)
        ...

        etc'

    will register options "display.font.color", "display.font.size", set the
    value of "display.font.size"... and so on.
    """
    # Note: reset_option relies on set_option, and on key directly
    # it does not fit in to this monkey-patching scheme

    global register_option, get_option, set_option

    def wrap(func: F) -> F:
        # [FUNCTION DOC]
        # HELPER: for config_prefix. Takes a function whose first argument is an
        # option key and returns a new function with the same remaining
        # arguments that prepends the enclosing prefix and a dot to that key
        # before delegating. The return is cast to the original callable type for
        # type checkers.
        def inner(key: str, *args, **kwds):
            # [FUNCTION DOC]
            # HELPER: the prefixing wrapper built by wrap. Parameters: key
            # (option name) followed by arbitrary positional and keyword
            # arguments passed through. Calls the wrapped function with the
            # prefixed key and returns its result.
            pkey = f"{prefix}.{key}"
            return func(pkey, *args, **kwds)

        return cast(F, inner)

    _register_option = register_option
    _get_option = get_option
    _set_option = set_option
    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)
    try:
        yield
    finally:
        set_option = _set_option
        get_option = _get_option
        register_option = _register_option


# These factories and methods are handy for use as the validator
# arg in register_option


def is_type_factory(_type: type[Any]) -> Callable[[Any], None]:
    # [FUNCTION DOC]
    # Validator factory. Parameter _type is a type. Returns a validator function
    # taking one value that raises a value error (message naming the type) unless
    # the value's exact type equals _type. Subclasses do not pass (so for
    # example a boolean does not pass an integer check).
    """

    Parameters
    ----------
    `_type` - a type to be compared against (e.g. type(x) == `_type`)

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if type(x) is not equal to `_type`

    """

    def inner(x) -> None:
        # [FUNCTION DOC]
        # HELPER: the validator returned by the type factory. Raises a value
        # error if the exact type of x differs from the captured type. Returns
        # nothing.
        if type(x) != _type:
            raise ValueError(f"Value must have type '{_type}'")

    return inner


def is_instance_factory(_type: type | tuple[type, ...]) -> Callable[[Any], None]:
    # [FUNCTION DOC]
    # Validator factory. Parameter _type is a type or a tuple of types. Builds a
    # description for the error message (the types' names joined by a vertical
    # bar for a tuple, else the quoted type). Returns a validator function
    # taking one value that raises a value error unless the value is an instance
    # (subclasses allowed) of _type.
    """

    Parameters
    ----------
    `_type` - the type to be checked against

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if x is not an instance of `_type`

    """
    if isinstance(_type, tuple):
        type_repr = "|".join(map(str, _type))
    else:
        type_repr = f"'{_type}'"

    def inner(x) -> None:
        # [FUNCTION DOC]
        # HELPER: the validator returned by the instance factory. Raises a value
        # error naming the allowed types if x is not an instance of the captured
        # type(s). Returns nothing.
        if not isinstance(x, _type):
            raise ValueError(f"Value must be an instance of {type_repr}")

    return inner


def is_one_of_factory(legal_values: Sequence) -> Callable[[Any], None]:
    # [FUNCTION DOC]
    # Validator factory. Parameter legal_values is a sequence mixing literal
    # allowed values and callables (predicates). Splits it into the predicates
    # and the literal values. Returns a validator function taking one value that
    # passes if the value equals one of the literals or if any predicate returns
    # a truthy result for it; otherwise raises a value error listing the
    # literals joined by a vertical bar, plus a note about callables if any
    # exist. No docstring exists for this function.
    # KNOWN ISSUE: a legal value that is itself callable (for example a class)
    # is treated as a predicate and cannot be matched by equality.
    callables = [c for c in legal_values if callable(c)]
    legal_values = [c for c in legal_values if not callable(c)]

    def inner(x) -> None:
        # [FUNCTION DOC]
        # HELPER: the validator returned by the one-of factory. Accepts x if it
        # is among the literal values, else if any predicate accepts it; otherwise
        # raises a value error. Returns nothing.
        if x not in legal_values:
            if not any(c(x) for c in callables):
                uvals = [str(lval) for lval in legal_values]
                pp_values = "|".join(uvals)
                msg = f"Value must be one of {pp_values}"
                if len(callables):
                    msg += " or a callable"
                raise ValueError(msg)

    return inner


def is_nonnegative_int(value: object) -> None:
    # [FUNCTION DOC]
    # Validator. Parameter value is any object. Returns nothing (accepts) when
    # the value is none or an integer (including booleans, since they are an
    # integer subtype) that is zero or greater. Otherwise raises a value error
    # saying a nonnegative integer or none is required.
    # KNOWN ISSUE: the existing docstring says "positive int" and says the error
    # is raised when the value "is not None or is a negative integer"; the code
    # accepts zero and raises for non-integers and negative integers, so the
    # docstring wording is inaccurate.
    """
    Verify that value is None or a positive int.

    Parameters
    ----------
    value : None or int
            The `value` to be checked.

    Raises
    ------
    ValueError
        When the value is not None or is a negative integer
    """
    if value is None:
        return

    elif isinstance(value, int):
        if value >= 0:
            return

    msg = "Value must be a nonnegative integer or None"
    raise ValueError(msg)


# common type validators, for convenience
# usage: register_option(... , validator = is_int)
is_int = is_type_factory(int)
is_bool = is_type_factory(bool)
is_float = is_type_factory(float)
is_str = is_type_factory(str)
is_text = is_instance_factory((str, bytes))


def is_callable(obj: object) -> bool:
    # [FUNCTION DOC]
    # Validator. Parameter obj is any object. Returns true if it is callable;
    # otherwise raises a value error saying the value must be a callable.
    # (It never returns false.)
    """

    Parameters
    ----------
    `obj` - the object to be checked

    Returns
    -------
    validator - returns True if object is callable
        raises ValueError otherwise.

    """
    if not callable(obj):
        raise ValueError("Value must be a callable")
    return True


# import set_module here would cause circular import
get_option.__module__ = "pandas"
set_option.__module__ = "pandas"
describe_option.__module__ = "pandas"
reset_option.__module__ = "pandas"
option_context.__module__ = "pandas"
