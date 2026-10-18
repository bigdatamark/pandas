# [MODULE DOC]
# Docstring-manipulation and deprecation decorators used across pandas.
# Public contents (exported list): a function that wraps a callable so that
# calling it emits a deprecation warning, a decorator factory deprecating a
# keyword argument (optionally renaming it and mapping values), a decorator
# factory deprecating positional use of arguments, a helper producing the
# version-dependent beginning of such messages, a decorator concatenating and
# formatting docstring templates, two docstring decorator classes (one
# performing percent-style substitution, one appending text), and the cached
# read-only property descriptor re-exported from the compiled properties module.
# Also contains a private-purpose decorator that overrides the module
# attribute of a function or class (not in the exported list) and a small text
# indentation helper. Relies on the stack-level finder in the exceptions
# utility module. The Substitution and Appender classes derive from matplotlib's
# docstring module. No import-time side effects.
from __future__ import annotations

from functools import wraps
import inspect
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

from pandas._libs.properties import cache_readonly
from pandas._typing import (
    F,
    T,
)
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
    )

    from pandas.errors import PandasChangeWarning


def deprecate(
    klass: type[Warning],
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = None,
    stacklevel: int = 2,
    msg: str | None = None,
) -> Callable[[F], F]:
    # [FUNCTION DOC]
    # Builds and returns a wrapper around the callable alternative that emits a
    # warning and then delegates. Inputs: klass, the warning category; name, the
    # name of the deprecated function used in the message; alternative, the
    # callable to delegate to and whose name and docstring are copied; version, the
    # pandas version in which deprecation began (used in the docstring directive);
    # alt_name, optional replacement name for messages (defaults to the name of
    # alternative); stacklevel, default 2, passed to the warning call; msg, optional
    # custom warning text.
    # KNOWN ISSUE: the docstring claims the default message ends with a period and
    # says "Use ... instead", but the actual default message is lowercase "use",
    # joined by a comma, ending with a period: "<name> is deprecated, use <alt> instead."
    # When the alternative has a docstring, the wrapper's docstring is rebuilt from
    # the first summary line, followed by a deprecated directive naming the version
    # and a note (the custom message or a default pointing at the alternative), then
    # the remainder of the original docstring. Raises an assertion error when the
    # alternative docstring has fewer than three line breaks, starts text on the
    # first line, or lacks a blank line after a non-empty summary. Empty docstrings
    # are allowed (optimized interpreter mode). The wrapper's declared return type
    # is misleading; it actually returns whatever the alternative returns, and the
    # factory's return is not a decorator despite the declared type (type checking
    # is suppressed).
    """
    Return a new function that emits a deprecation warning on use.

    To use this method for a deprecated function, another function
    `alternative` with the same signature must exist. The deprecated
    function will emit a deprecation warning, and in the docstring
    it will contain the deprecation directive with the provided version
    so it can be detected for future removal.

    Parameters
    ----------
    klass : Warning
        The warning class to use.
    name : str
        Name of function to deprecate.
    alternative : func
        Function to use instead.
    version : str
        Version of pandas in which the method has been deprecated.
    alt_name : str, optional
        Name to use in preference of alternative.__name__.
    stacklevel : int, default 2
    msg : str
        The message to display in the warning.
        Default is '{name} is deprecated. Use {alt_name} instead.'
    """
    alt_name = alt_name or alternative.__name__
    warning_msg = msg or f"{name} is deprecated, use {alt_name} instead."

    @wraps(alternative)
    def wrapper(*args, **kwargs) -> Callable[..., Any]:
        # [FUNCTION DOC]
        # HELPER: inner wrapper of the deprecation function. Accepts any positional
        # and keyword arguments, emits the precomputed warning with the configured
        # category and stack level, then returns the result of calling the alternative.
        warnings.warn(warning_msg, klass, stacklevel=stacklevel)
        return alternative(*args, **kwargs)

    # adding deprecated directive to the docstring
    msg = msg or f"Use `{alt_name}` instead."
    doc_error_msg = (
        "deprecate needs a correctly formatted docstring in "
        "the target function (should have a one liner short "
        "summary, and opening quotes should be in their own "
        f"line). Found:\n{alternative.__doc__}"
    )

    # when python is running in optimized mode (i.e. `-OO`), docstrings are
    # removed, so we check that a docstring with correct formatting is used
    # but we allow empty docstrings
    if alternative.__doc__:
        if alternative.__doc__.count("\n") < 3:
            raise AssertionError(doc_error_msg)
        empty1, summary, empty2, doc_string = alternative.__doc__.split("\n", 3)
        if empty1 or (empty2 and not summary):
            raise AssertionError(doc_error_msg)
        wrapper.__doc__ = dedent(
            f"""
        {summary.strip()}

        .. deprecated:: {version}
            {msg}

        {dedent(doc_string)}"""
        )
    # error: Incompatible return value type (got "Callable[[VarArg(Any), KwArg(Any)],
    # Callable[...,Any]]", expected "Callable[[F], F]")
    return wrapper  # type: ignore[return-value]


def deprecate_kwarg(
    klass: type[Warning],
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = None,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    # [FUNCTION DOC]
    # Decorator factory deprecating a keyword argument. Inputs: klass, warning
    # category; old_arg_name, the deprecated keyword; new_arg_name, the replacement
    # keyword or none (meaning the keyword is deprecated with no replacement);
    # mapping, optional dictionary-like object or callable translating old values to
    # new values (values missing from a dictionary pass unchanged); stacklevel,
    # default 2. Returns the decorator. Raises a type error immediately at decoration
    # setup if mapping is neither dictionary-like nor callable. The decorated
    # function, when called, behaves as follows: the old keyword is removed from the
    # call arguments; if it was supplied with a value other than none, a warning is
    # issued; with no replacement the old keyword is put back and the function is
    # called; otherwise the (possibly mapped) value is passed under the new keyword,
    # and a type error is raised if the new keyword was also supplied with a value
    # other than none. Note that an old keyword explicitly passed as none is silently
    # dropped and never forwarded or warned about.
    """
    Decorator to deprecate a keyword argument of a function.

    Parameters
    ----------
    klass : Warning
        The warning class to use.
    old_arg_name : str
        Name of argument in function to deprecate.
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.
    stacklevel : int, default 2

    Examples
    --------
    The following deprecates 'cols', using 'columns' instead

    >>> @deprecate_kwarg(FutureWarning, old_arg_name="cols", new_arg_name="columns")
    ... def f(columns=""):
    ...     print(columns)
    >>> f(columns="should work ok")
    should work ok

    >>> f(cols="should raise warning")  # doctest: +SKIP
    FutureWarning: cols is deprecated, use columns instead
      warnings.warn(msg, FutureWarning)
    should raise warning

    >>> f(cols="should error", columns="can't pass do both")  # doctest: +SKIP
    TypeError: Can only specify 'cols' or 'columns', not both

    >>> @deprecate_kwarg(FutureWarning, "old", "new", {"yes": True, "no": False})
    ... def f(new=False):
    ...     print("yes!" if new else "no!")
    >>> f(old="yes")  # doctest: +SKIP
    FutureWarning: old='yes' is deprecated, use new=True instead
      warnings.warn(msg, FutureWarning)
    yes!

    To raise a warning that a keyword will be removed entirely in the future

    >>> @deprecate_kwarg(FutureWarning, old_arg_name="cols", new_arg_name=None)
    ... def f(cols="", another_param=""):
    ...     print(cols)
    >>> f(cols="should raise warning")  # doctest: +SKIP
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version. Please take steps to stop the use of 'cols'
    should raise warning
    >>> f(another_param="should not raise warning")  # doctest: +SKIP
    should not raise warning

    >>> f(cols="should raise warning", another_param="")  # doctest: +SKIP
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version. Please take steps to stop the use of 'cols'
    should raise warning
    """
    if mapping is not None and not hasattr(mapping, "get") and not callable(mapping):
        raise TypeError(
            "mapping from old to new argument values must be dict or callable!"
        )

    def _deprecate_kwarg(func: F) -> F:
        # [FUNCTION DOC]
        # HELPER: the actual decorator produced by the keyword deprecation factory.
        # Takes the function to decorate and returns a metadata-preserving wrapper
        # (cast to the same callable type) implementing the behavior described on the
        # factory.
        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable[..., Any]:
            # [FUNCTION DOC]
            # HELPER: wrapper applying the keyword deprecation logic to one call. Takes
            # arbitrary positional and keyword arguments, returns the wrapped function's
            # result. Hides itself from pytest tracebacks. Emits a warning of the configured
            # category and may raise a type error when both old and new keywords are given.
            __tracebackhide__ = True

            old_arg_value = kwargs.pop(old_arg_name, None)

            if old_arg_value is not None:
                if new_arg_name is None:
                    msg = (
                        f"the {old_arg_name!r} keyword is deprecated and "
                        "will be removed in a future version. Please take "
                        f"steps to stop the use of {old_arg_name!r}"
                    )
                    warnings.warn(msg, klass, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)

                elif mapping is not None:
                    if callable(mapping):
                        new_arg_value = mapping(old_arg_value)
                    else:
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    msg = (
                        f"the {old_arg_name}={old_arg_value!r} keyword is "
                        "deprecated, use "
                        f"{new_arg_name}={new_arg_value!r} instead."
                    )
                else:
                    new_arg_value = old_arg_value
                    msg = (
                        f"the {old_arg_name!r} keyword is deprecated, "
                        f"use {new_arg_name!r} instead."
                    )

                warnings.warn(msg, klass, stacklevel=stacklevel)
                if kwargs.get(new_arg_name) is not None:
                    msg = (
                        f"Can only specify {old_arg_name!r} "
                        f"or {new_arg_name!r}, not both."
                    )
                    raise TypeError(msg)
                kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return _deprecate_kwarg


def _format_argument_list(allow_args: list[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: formats the list of allowed positional argument names for the
    # non-keyword deprecation message. Input: allow_args, list of names. Returns an
    # empty string for an empty list (after dropping the instance-reference name),
    # a phrase beginning with a space and "except for the argument" followed by the
    # single quoted name for one entry, or "except for the arguments" followed by
    # comma-separated quoted names joined with "and" before the last for several.
    # Side effect: removes the instance-reference name from the passed list in place.
    # KNOWN ISSUE: the docstring examples and parameter name (allowed_args, and
    # described as possibly a tuple or integer) do not match the code, which takes a
    # list named allow_args; the example output also omits the leading space and
    # uses the plural for one name.
    """
    Convert the allow_args argument (either string or integer) of
    `deprecate_nonkeyword_arguments` function to a string describing
    it to be inserted into warning message.

    Parameters
    ----------
    allowed_args : list, tuple or int
        The `allowed_args` argument for `deprecate_nonkeyword_arguments`,
        but None value is not allowed.

    Returns
    -------
    str
        The substring describing the argument list in best way to be
        inserted to the warning message.

    Examples
    --------
    `format_argument_list([])` -> ''
    `format_argument_list(['a'])` -> "except for the arguments 'a'"
    `format_argument_list(['a', 'b'])` -> "except for the arguments 'a' and 'b'"
    `format_argument_list(['a', 'b', 'c'])` ->
        "except for the arguments 'a', 'b' and 'c'"
    """
    if "self" in allow_args:
        allow_args.remove("self")
    if not allow_args:
        return ""
    elif len(allow_args) == 1:
        return f" except for the argument '{allow_args[0]}'"
    else:
        last = allow_args[-1]
        args = ", ".join(["'" + x + "'" for x in allow_args[:-1]])
        return f" except for the arguments {args} and '{last}'"


def future_version_msg(version: str | None) -> str:
    # [FUNCTION DOC]
    # Returns the leading phrase of a future-change warning. Input: version, a
    # version string or none. Returns "In a future version of pandas" when none,
    # otherwise "Starting with pandas version" followed by the version.
    """Specify which version of pandas the deprecation will take place in."""
    if version is None:
        return "In a future version of pandas"
    else:
        return f"Starting with pandas version {version}"


def deprecate_nonkeyword_arguments(
    klass: type[PandasChangeWarning],
    allowed_args: list[str] | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    # [FUNCTION DOC]
    # Decorator factory warning when arguments are passed positionally beyond an
    # allowed set. Inputs: klass, a pandas warning category that must provide a
    # version query method (used for the message's lead phrase); allowed_args,
    # optional list of leading parameter names that may stay positional (when none,
    # all positional parameters without default values are allowed); name, optional
    # display name used in the message (default: the qualified name of the function).
    # Returns the decorator. The decorated function's reported signature is
    # rewritten so that all other positional parameters become keyword-only
    # (re-sorted by kind), while the function itself is unchanged. At call time, if
    # more positional arguments are passed than the number of allowed names, a
    # warning of category klass is emitted with the stack level found by the
    # exceptions helper, then the function is called normally.
    """
    Decorator to deprecate a use of non-keyword arguments of a function.

    Parameters
    ----------
    klass : Warning
        The warning class to use.
    allowed_args : list, optional
        In case of list, it must be the list of names of some
        first arguments of the decorated functions that are
        OK to be given as positional arguments. In case of None value,
        defaults to list of all arguments not having the
        default value.
    name : str, optional
        The specific name of the function to show in the warning
        message. If None, then the Qualified name of the function
        is used.
    """

    def decorate(func):
        # [FUNCTION DOC]
        # HELPER: decorator built by the non-keyword deprecation factory. Takes the
        # function, computes the allowed names, the new keyword-only signature and the
        # message template, and returns a wrapper with the replaced signature attribute.
        old_sig = inspect.signature(func)

        if allowed_args is not None:
            allow_args = allowed_args
        else:
            allow_args = [
                p.name
                for p in old_sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]

        new_params = [
            p.replace(kind=p.KEYWORD_ONLY)
            if (
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.name not in allow_args
            )
            else p
            for p in old_sig.parameters.values()
        ]
        new_params.sort(key=lambda p: p.kind)
        new_sig = old_sig.replace(parameters=new_params)

        num_allow_args = len(allow_args)
        msg = (
            f"{future_version_msg(klass.version())} all arguments of "
            f"{name or func.__qualname__}{{arguments}} will be keyword-only."
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: wrapper checking the count of positional arguments against the
            # allowed count, issuing the formatted warning when exceeded, then delegating
            # to the wrapped function and returning its result.
            if len(args) > num_allow_args:
                warnings.warn(
                    msg.format(arguments=_format_argument_list(allow_args)),
                    klass,
                    stacklevel=find_stack_level(),
                )
            return func(*args, **kwargs)

        # error: "Callable[[VarArg(Any), KwArg(Any)], Any]" has no
        # attribute "__signature__"
        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        return wrapper

    return decorate


def doc(*docstrings: None | str | Callable, **params: object) -> Callable[[F], F]:
    # [FUNCTION DOC]
    # Decorator factory that builds a docstring from templates. Inputs:
    # docstrings, any number of items each being none (skipped), a string template,
    # or a callable (or object) carrying a docstring; params, keyword values used to
    # format string templates. The decorated object's own dedented docstring comes
    # first, then each item in order; items that already carry a stored list of
    # components contribute those components instead; a non-string item is kept only
    # if it has a docstring. When params are given, string components are formatted
    # with them (so literal braces in templates must be doubled). Non-string
    # components are dedented docstring text. The pieces are joined without a
    # separator and assigned as the decorated object's docstring. A private
    # attribute holding the component list is attached to the object for later
    # reuse. Returns the object itself. Raises a key or index error from formatting
    # if a template field is missing.
    """
    A decorator to take docstring templates, concatenate them and perform string
    substitution on them.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # [FUNCTION DOC]
        # HELPER: decorator produced by the docstring-template factory; takes the
        # decorated object, assembles and applies the combined docstring, stores the
        # component list, and returns the same object.
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if docstring is None:
                continue
            if hasattr(docstring, "_docstring_components"):
                docstring_components.extend(
                    docstring._docstring_components  # pyright: ignore[reportAttributeAccessIssue]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        params_applied = [
            component.format(**params)
            if isinstance(component, str) and len(params) > 0
            else component
            for component in docstring_components
        ]

        decorated.__doc__ = "".join(
            [
                component
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in params_applied
            ]
        )

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator


# Substitution and Appender are derived from matplotlib.docstring (1.1.0)
# module https://matplotlib.org/users/license.html


class Substitution:
    # [CLASS DOC]
    # Callable decorator object performing percent-style substitution on the
    # docstring of the function it decorates. State: params, either the tuple of
    # positional arguments or the dictionary of keyword arguments given at
    # construction (mutually exclusive). Safe when docstrings are stripped (none
    # stays none). Derived from matplotlib's docstring helper.
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """

    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Stores the substitution values. Accepts positional or keyword values, never
        # both; raises an assertion error if both are given. With neither, params is an
        # empty dictionary.
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, func: F) -> F:
        # [FUNCTION DOC]
        # Applies the substitution to the docstring of func in place (leaving none or
        # empty docstrings untouched) and returns func. Raises the usual formatting
        # errors if the docstring has missing or mismatched fields.
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    def update(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Merges more values into params using dictionary update semantics (same
        # arguments as the dictionary update). Does nothing when params is positional.
        """
        Update self.params with supplied args.
        """
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)


class Appender:
    # [CLASS DOC]
    # Callable decorator object appending text to the docstring of the function it
    # decorates. State: addendum, the text to append (or none), and join, the
    # separator placed between the existing docstring and the addendum. Derived from
    # matplotlib's docstring helper. Tolerates missing docstrings.
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    addendum: str | None

    def __init__(self, addendum: str | None, join: str = "", indents: int = 0) -> None:
        # [FUNCTION DOC]
        # Stores the addendum and separator. Inputs: addendum, text or none; join,
        # default empty string; indents, default 0; when positive the addendum is passed
        # through the text indentation helper with that many levels.
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func: T) -> T:
        # [FUNCTION DOC]
        # Appends the addendum to the docstring of func in place and returns func. The
        # existing docstring (or empty text) and the addendum are joined with the join
        # separator and then dedented. Side effect: replaces a none addendum on the
        # decorator itself with an empty string.
        func.__doc__ = func.__doc__ if func.__doc__ else ""
        self.addendum = self.addendum if self.addendum else ""
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func


def indent(text: str | None, indents: int = 1) -> str:
    # [FUNCTION DOC]
    # Indents continuation lines of a text. Inputs: text, a string or none; indents,
    # default 1, number of four-space levels. Returns an empty string for none,
    # empty or non-string input; otherwise returns the text with every line break
    # followed by the requested indentation (the first line is not indented).
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\n"] + ["    "] * indents)
    return jointext.join(text.split("\n"))


__all__ = [
    "Appender",
    "Substitution",
    "cache_readonly",
    "deprecate",
    "deprecate_kwarg",
    "deprecate_nonkeyword_arguments",
    "doc",
    "future_version_msg",
]


def set_module(module) -> Callable[[F], F]:
    # [FUNCTION DOC]
    # Private decorator factory overriding the module attribute of the decorated
    # function or class. Input: module, the module name to assign (none means no
    # change). Returns the decorator.
    """Private decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module("pandas")
        def example():
            pass


        assert example.__module__ == "pandas"
    """

    def decorator(func: F) -> F:
        # [FUNCTION DOC]
        # HELPER: decorator returned by the module-override factory. For classes it first
        # records the original module in a private attribute (ignoring failures to set
        # it) so documentation link resolution can find the source, then assigns the new
        # module name and returns the object unchanged otherwise.
        if module is not None:
            if isinstance(func, type):
                # Store the original module for classes to ensure linkcode_resolve
                # can resolve the true source location after re-exporting
                try:
                    func._module_source = func.__module__  # type: ignore[attr-defined]
                except AttributeError:
                    pass

            func.__module__ = module
        return cast("F", func)  # type: ignore[redundant-cast]

    return decorator
