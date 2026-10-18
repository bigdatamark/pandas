# [MODULE DOC]
# Deferred column expressions. Public contents: the Expression class (a
# lazily evaluated computation on a data frame, built by operator overloading
# and attribute/call chaining, published in the typing namespace of the public
# API) and the col function (published in the top-level namespace) which
# creates an expression selecting a named column. Private helpers parse
# arguments that may themselves be expressions and format argument lists for
# the text representation.
# Expressions are consumed by common.apply_if_callable (the common module
# imports this one, so this module must not import it back at module level),
# so any API accepting a function of the frame also accepts an expression.
# A module-level table of operator method names to symbols exists for building
# text representations.
# KNOWN ISSUE: that table is not referenced anywhere in the code base; each
# operator method hard-codes its own symbol, so the table is dead data.
# Import-time side effects: none beyond imports.
from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)

from pandas.util._decorators import set_module

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Series,
    )


# Used only for generating the str repr of expressions.
_OP_SYMBOLS = {
    "__add__": "+",
    "__radd__": "+",
    "__sub__": "-",
    "__rsub__": "-",
    "__mul__": "*",
    "__rmul__": "*",
    "__truediv__": "/",
    "__rtruediv__": "/",
    "__floordiv__": "//",
    "__rfloordiv__": "//",
    "__mod__": "%",
    "__rmod__": "%",
    "__ge__": ">=",
    "__gt__": ">",
    "__le__": "<=",
    "__lt__": "<",
    "__eq__": "==",
    "__ne__": "!=",
    "__and__": "&",
    "__rand__": "&",
    "__or__": "|",
    "__ror__": "|",
    "__xor__": "^",
    "__rxor__": "^",
}


def _parse_args(df: DataFrame, *args: Any) -> tuple[Any, ...]:
    # [FUNCTION DOC]
    # HELPER: resolves positional arguments for deferred calls. Inputs: df, the
    # data frame to evaluate against; args, any values. Returns a tuple in which
    # every expression argument is replaced by its value evaluated on df and
    # all other arguments are kept as given.
    # Parse `args`, evaluating any expressions we encounter.
    return tuple(
        x._eval_expression(df) if isinstance(x, Expression) else x for x in args
    )


def _parse_kwargs(df: DataFrame, **kwargs: Any) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: keyword counterpart of the positional parser. Returns a new dict
    # with the same keys where expression values are replaced by their value
    # evaluated on df.
    # Parse `kwargs`, evaluating any expressions we encounter.
    return {
        key: val._eval_expression(df) if isinstance(val, Expression) else val
        for key, val in kwargs.items()
    }


def _pretty_print_args_kwargs(*args: Any, **kwargs: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: formats call arguments for expression text. Returns the
    # comma-and-space joined representations of the positional arguments,
    # followed by "name=representation" pairs for keyword arguments; empty
    # groups are omitted, so no arguments give an empty string.
    inputs_repr = ", ".join(repr(arg) for arg in args)
    kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())

    all_args = []
    if inputs_repr:
        all_args.append(inputs_repr)
    if kwargs_repr:
        all_args.append(kwargs_repr)

    return ", ".join(all_args)


@set_module("pandas.api.typing")
class Expression:
    # [CLASS DOC]
    # A deferred computation on a data frame. State: _func, a callable taking
    # the frame and returning the result; _repr_str, text describing the
    # computation; _needs_parentheses, whether the text must be parenthesised
    # when embedded in a larger expression (true for compound results).
    # Almost every operator, item access, attribute access and call on an
    # expression returns a new expression instead of computing; evaluation
    # happens only when the private evaluation method is given a frame.
    # Arguments that are themselves expressions are evaluated against the same
    # frame. Truth testing, iteration and copying are deliberately unsupported
    # and raise errors. Because equality is overloaded to build expressions,
    # instances are not meant to be compared or used as dict keys.
    """
    Class representing a deferred column.

    This is not meant to be instantiated directly. Instead, use :meth:`pandas.col`.
    """

    def __init__(
        self,
        func: Callable[[DataFrame], Any],
        repr_str: str,
        needs_parenthese: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the computation and its text. Inputs: func, callable from a
        # frame to a value; repr_str, text form; needs_parenthese (note the
        # spelling of this parameter, default false), whether the text needs
        # parentheses when nested, stored as the needs-parentheses flag.
        self._func = func
        self._repr_str = repr_str
        self._needs_parentheses = needs_parenthese

    def _eval_expression(self, df: DataFrame) -> Any:
        # [FUNCTION DOC]
        # Evaluates the expression: calls the stored computation with df and
        # returns its result.
        return self._func(df)

    def _with_op(
        self, op: str, other: Any, repr_str: str, needs_parentheses: bool = True
    ) -> Expression:
        # [FUNCTION DOC]
        # HELPER: shared builder for binary operators and item access. Inputs: op,
        # the name of the special method to invoke on the evaluated left value;
        # other, the right operand, evaluated against the frame first when it is
        # an expression and used as-is otherwise; repr_str, text for the new
        # expression; needs_parentheses (default true), flag for the new
        # expression. Returns a new expression that evaluates the left side,
        # looks up the named method and calls it with the right value.
        if isinstance(other, Expression):
            return Expression(
                lambda df: getattr(self._eval_expression(df), op)(
                    other._eval_expression(df)
                ),
                repr_str,
                needs_parenthese=needs_parentheses,
            )
        else:
            return Expression(
                lambda df: getattr(self._eval_expression(df), op)(other),
                repr_str,
                needs_parenthese=needs_parentheses,
            )

    def _maybe_wrap_parentheses(self, other: Any) -> tuple[str, str]:
        # [FUNCTION DOC]
        # HELPER: text preparation for binary operators. Returns a pair of
        # strings: the representation of this expression and of other, each
        # surrounded by parentheses if it is an expression flagged as needing
        # them (a non-expression other is never parenthesised and uses its
        # standard representation).
        if self._needs_parentheses:
            self_repr = f"({self!r})"
        else:
            self_repr = f"{self!r}"
        if isinstance(other, Expression) and other._needs_parentheses:
            other_repr = f"({other!r})"
        else:
            other_repr = f"{other!r}"
        return self_repr, other_repr

    # Binary ops
    def __add__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred addition. Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__add__", other, f"{self_repr} + {other_repr}")

    def __radd__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected addition (this expression is the right operand). Input
        # other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__radd__", other, f"{other_repr} + {self_repr}")

    def __sub__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred subtraction. Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__sub__", other, f"{self_repr} - {other_repr}")

    def __rsub__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected subtraction (this expression is the right operand). Input
        # other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rsub__", other, f"{other_repr} - {self_repr}")

    def __mul__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred multiplication. Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__mul__", other, f"{self_repr} * {other_repr}")

    def __rmul__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected multiplication (this expression is the right operand).
        # Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rmul__", other, f"{other_repr} * {self_repr}")

    def __matmul__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred matrix multiplication. Input other: right-hand value (or left-hand
        # value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__matmul__", other, f"{self_repr} @ {other_repr}")

    def __rmatmul__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected matrix multiplication (this expression is the right
        # operand). Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rmatmul__", other, f"{other_repr} @ {self_repr}")

    def __pow__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred exponentiation. Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__pow__", other, f"{self_repr} ** {other_repr}")

    def __rpow__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected exponentiation (this expression is the right operand).
        # Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rpow__", other, f"{other_repr} ** {self_repr}")

    def __truediv__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred true division. Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__truediv__", other, f"{self_repr} / {other_repr}")

    def __rtruediv__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected true division (this expression is the right operand). Input
        # other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rtruediv__", other, f"{other_repr} / {self_repr}")

    def __floordiv__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred floor division. Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__floordiv__", other, f"{self_repr} // {other_repr}")

    def __rfloordiv__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected floor division (this expression is the right operand).
        # Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rfloordiv__", other, f"{other_repr} // {self_repr}")

    def __ge__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred greater-than-or-equal comparison. Input other: right-hand value (or
        # left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__ge__", other, f"{self_repr} >= {other_repr}")

    def __gt__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred greater-than comparison. Input other: right-hand value (or left-hand
        # value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__gt__", other, f"{self_repr} > {other_repr}")

    def __le__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred less-than-or-equal comparison. Input other: right-hand value (or
        # left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__le__", other, f"{self_repr} <= {other_repr}")

    def __lt__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred less-than comparison. Input other: right-hand value (or left-hand
        # value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__lt__", other, f"{self_repr} < {other_repr}")

    def __eq__(self, other: object) -> Expression:  # type: ignore[override]
        # [FUNCTION DOC]
        # Deferred equality comparison. Input other: right-hand value (or left-hand
        # value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses. Because equality is overloaded, this
        # builds an expression rather than returning a boolean.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__eq__", other, f"{self_repr} == {other_repr}")

    def __ne__(self, other: object) -> Expression:  # type: ignore[override]
        # [FUNCTION DOC]
        # Deferred inequality comparison. Input other: right-hand value (or left-hand
        # value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses. Because equality is overloaded, this
        # builds an expression rather than returning a boolean.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__ne__", other, f"{self_repr} != {other_repr}")

    def __mod__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred remainder. Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__mod__", other, f"{self_repr} % {other_repr}")

    def __rmod__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected remainder (this expression is the right operand). Input
        # other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rmod__", other, f"{other_repr} % {self_repr}")

    # Logical ops
    def __and__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred bitwise/logical and. Input other: right-hand value (or left-hand
        # value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__and__", other, f"{self_repr} & {other_repr}")

    def __rand__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected bitwise/logical and (this expression is the right operand).
        # Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rand__", other, f"{other_repr} & {self_repr}")

    def __or__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred bitwise/logical or. Input other: right-hand value (or left-hand value
        # for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__or__", other, f"{self_repr} | {other_repr}")

    def __ror__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected bitwise/logical or (this expression is the right operand).
        # Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__ror__", other, f"{other_repr} | {self_repr}")

    def __xor__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred bitwise/logical exclusive or. Input other: right-hand value (or
        # left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is this expression's text, then the operator and
        # the other operand's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__xor__", other, f"{self_repr} ^ {other_repr}")

    def __rxor__(self, other: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred reflected bitwise/logical exclusive or (this expression is the right
        # operand). Input other: right-hand value (or left-hand value for
        # reflected forms), which may itself be an expression. Returns a new
        # expression whose evaluation invokes the matching special method of the
        # evaluated operands; its text is the other operand's text first, then the
        # operator and this expression's text (operands parenthesised when
        # needed), flagged as needing parentheses.
        self_repr, other_repr = self._maybe_wrap_parentheses(other)
        return self._with_op("__rxor__", other, f"{other_repr} ^ {self_repr}")

    def __invert__(self) -> Expression:
        # [FUNCTION DOC]
        # Deferred bitwise/logical inversion. Returns a new expression applying
        # inversion to the evaluated value, with text "~" followed by this
        # expression's text, flagged as needing parentheses.
        # KNOWN ISSUE: unlike negation and unary plus, the text is not wrapped in
        # parentheses when this expression is compound, so the printed form can
        # be ambiguous.
        return Expression(
            lambda df: ~self._eval_expression(df),
            f"~{self._repr_str}",
            needs_parenthese=True,
        )

    def __neg__(self) -> Expression:
        # [FUNCTION DOC]
        # Deferred arithmetic negation. Returns a new expression applying negation
        # to the evaluated value; text is a minus sign before this expression's
        # text, parenthesised when this expression needs it.
        if self._needs_parentheses:
            repr_str = f"-({self._repr_str})"
        else:
            repr_str = f"-{self._repr_str}"
        return Expression(
            lambda df: -self._eval_expression(df),
            repr_str,
            needs_parenthese=True,
        )

    def __pos__(self) -> Expression:
        # [FUNCTION DOC]
        # Deferred unary plus. Returns a new expression applying unary plus to the
        # evaluated value; text is a plus sign before this expression's text,
        # parenthesised when this expression needs it.
        if self._needs_parentheses:
            repr_str = f"+({self._repr_str})"
        else:
            repr_str = f"+{self._repr_str}"
        return Expression(
            lambda df: +self._eval_expression(df),
            repr_str,
            needs_parenthese=True,
        )

    def __abs__(self) -> Expression:
        # [FUNCTION DOC]
        # Deferred absolute value. Returns a new expression applying the builtin
        # absolute value to the evaluated value; text is "abs" with this
        # expression's text in parentheses.
        return Expression(
            lambda df: abs(self._eval_expression(df)),
            f"abs({self._repr_str})",
            needs_parenthese=True,
        )

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> Expression:
        # [FUNCTION DOC]
        # Numpy universal-function hook, so applying such a function to an
        # expression defers the call. Inputs: ufunc, the numpy function; method,
        # the ufunc method name (ignored: the function is always called
        # directly); inputs, positional operands (expressions allowed); kwargs,
        # keyword options. Returns a new expression that evaluates expression
        # arguments on the frame and calls the function; its text is the function
        # name with the formatted arguments.
        # KNOWN ISSUE: the inner function unpacks the keyword dict with a
        # positional splat when parsing, which passes the key names as extra
        # positional arguments to a function that accepts none, so any call with
        # keyword options fails with a type error at evaluation time.
        def func(df: DataFrame) -> Any:
            # [FUNCTION DOC]
            # HELPER: deferred evaluation. Resolves expression inputs against df
            # and calls the ufunc with the resolved inputs and keyword options.
            parsed_inputs = _parse_args(df, *inputs)
            parsed_kwargs = _parse_kwargs(df, *kwargs)
            return ufunc(*parsed_inputs, **parsed_kwargs)

        args_str = _pretty_print_args_kwargs(*inputs, **kwargs)
        repr_str = f"{ufunc.__name__}({args_str})"

        return Expression(func, repr_str)

    def __getitem__(self, item: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred item access. Input item: key, slice or expression. Returns a
        # new expression that evaluates this one and indexes the result with the
        # (evaluated) item; text is this expression's text followed by the item's
        # representation in square brackets.
        return self._with_op(
            "__getitem__", item, f"{self!r}[{item!r}]", needs_parentheses=True
        )

    def _call_with_func(self, func: Callable, **kwargs: Any) -> Expression:
        # [FUNCTION DOC]
        # HELPER: builds an expression that calls an arbitrary function with
        # keyword arguments only. Inputs: func, the callable (must have a name
        # attribute for the text); kwargs, keyword arguments that may include
        # expressions. Returns a new expression that evaluates expression
        # arguments and calls func; text is the function name with formatted
        # arguments. The receiver (this expression) is not used.
        def wrapped(df: DataFrame) -> Any:
            # [FUNCTION DOC]
            # HELPER: deferred evaluation: resolves keyword expressions against df
            # and calls func with them.
            parsed_kwargs = _parse_kwargs(df, **kwargs)
            return func(**parsed_kwargs)

        args_str = _pretty_print_args_kwargs(**kwargs)
        repr_str = func.__name__ + "(" + args_str + ")"

        return Expression(wrapped, repr_str)

    def __call__(self, *args: Any, **kwargs: Any) -> Expression:
        # [FUNCTION DOC]
        # Deferred call of the value this expression evaluates to (for example a
        # method obtained by attribute access). Inputs: args and kwargs, which
        # may contain expressions. Returns a new expression that evaluates this
        # one, resolves expression arguments, and calls the result; text is this
        # expression's text followed by the formatted arguments in parentheses.
        def func(df: DataFrame, *args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC]
            # HELPER: deferred evaluation: evaluates this expression on df,
            # resolves argument expressions, and calls the resulting callable.
            parsed_args = _parse_args(df, *args)
            parsed_kwargs = _parse_kwargs(df, **kwargs)
            return self._eval_expression(df)(*parsed_args, **parsed_kwargs)

        args_str = _pretty_print_args_kwargs(*args, **kwargs)
        repr_str = f"{self._repr_str}({args_str})"
        return Expression(lambda df: func(df, *args, **kwargs), repr_str)

    def __getattr__(self, name: str, /) -> Any:
        # [FUNCTION DOC]
        # Deferred attribute access, reached only when normal lookup fails.
        # Input name: attribute name. Returns a new expression that evaluates this
        # one and reads the named attribute of the result; text is this
        # expression's text (parenthesised if needed) followed by a dot and the
        # name. Used for accessors and methods such as string-accessor chains.
        repr_str = f"{self!r}"
        if self._needs_parentheses:
            repr_str = f"({repr_str})"
        repr_str += f".{name}"
        return Expression(lambda df: getattr(self._eval_expression(df), name), repr_str)

    def case_when(self, caselist: Sequence[tuple[Any, Any]]) -> Expression:
        # [FUNCTION DOC]
        # Deferred conditional replacement. Input caselist: sequence of
        # (condition, replacement) pairs, either of which may be an expression.
        # Returns a new expression that evaluates this expression to a series,
        # evaluates any expression conditions and replacements on the same frame,
        # and calls the series' case-when method with the evaluated pairs. Text is
        # this expression's text followed by an abbreviated case-when call.
        """
        Create an expression that evaluates :meth:`Series.case_when` in a DataFrame
        context.

        This is intended to enable patterns like::

            df.assign(result=pd.col("a").case_when([(pd.col("b") > 0, 1)]))

        where conditions/replacements may reference other columns via ``pd.col``.
        """

        def func(df: DataFrame) -> Any:
            # [FUNCTION DOC]
            # HELPER: deferred evaluation described by the enclosing method.
            ser = self._eval_expression(df)
            evaluated = []
            for condition, replacement in caselist:
                if isinstance(condition, Expression):
                    condition = condition._eval_expression(df)
                if isinstance(replacement, Expression):
                    replacement = replacement._eval_expression(df)
                evaluated.append((condition, replacement))
            return ser.case_when(evaluated)

        # Keep repr compact; caselist may be large.
        repr_str = f"{self!r}.case_when(...)"
        return Expression(func, repr_str)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the stored text, or a generic placeholder text if the stored
        # text is empty.
        return self._repr_str or "Expr(...)"

    # Unsupported ops
    def __bool__(self) -> NoReturn:
        # [FUNCTION DOC]
        # Always raises a type error: truth testing an expression is ambiguous.
        raise TypeError("boolean value of an expression is ambiguous")

    def __iter__(self) -> NoReturn:
        # [FUNCTION DOC]
        # Always raises a type error: expressions are not iterable.
        raise TypeError("Expression objects are not iterable")

    def __copy__(self) -> NoReturn:
        # [FUNCTION DOC]
        # Always raises a type error: expressions cannot be copied.
        raise TypeError("Expression objects are not copiable")

    def __deepcopy__(self, memo: dict[int, Any] | None) -> NoReturn:
        # [FUNCTION DOC]
        # Always raises a type error: expressions cannot be deep-copied (memo is
        # ignored).
        raise TypeError("Expression objects are not copiable")


@set_module("pandas")
def col(col_name: Hashable) -> Expression:
    # [FUNCTION DOC]
    # Public constructor for a deferred column. Input col_name: a hashable
    # column label. Raises a type error immediately if it is not hashable.
    # Returns an expression whose evaluation, given a frame, returns the column
    # of that name; if the label is not among the frame's columns, evaluation
    # raises a value error (not a key error) whose message includes the
    # frame's column list truncated to about ninety characters. The expression's
    # text is "col" with the label's representation in parentheses.
    """
    Generate deferred object representing a column of a DataFrame.

    Any place which accepts ``lambda df: df[col_name]``, such as
    :meth:`DataFrame.assign` or :meth:`DataFrame.loc`, can also accept
    ``pd.col(col_name)``.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    col_name : Hashable
        Column name.

    Returns
    -------
    `pandas.api.typing.Expression`
        A deferred object representing a column of a DataFrame.

    See Also
    --------
    DataFrame.query : Query columns of a dataframe using string expressions.

    Examples
    --------

    You can use `col` in `assign`.

    >>> df = pd.DataFrame({"name": ["beluga", "narwhal"], "speed": [100, 110]})
    >>> df.assign(name_titlecase=pd.col("name").str.title())
          name  speed name_titlecase
    0   beluga    100         Beluga
    1  narwhal    110        Narwhal

    You can also use it for filtering.

    >>> df.loc[pd.col("speed") > 105]
          name  speed
    1  narwhal    110
    """
    if not isinstance(col_name, Hashable):
        msg = f"Expected Hashable, got: {type(col_name)}"
        raise TypeError(msg)

    def func(df: DataFrame) -> Series:
        # [FUNCTION DOC]
        # HELPER: the deferred lookup: returns the column named col_name of df, or
        # raises a value error with a hint listing the frame's columns when the
        # name is absent.
        if col_name not in df.columns:
            columns_str = str(df.columns.tolist())
            max_len = 90
            if len(columns_str) > max_len:
                columns_str = columns_str[:max_len] + "...]"

            msg = (
                f"Column '{col_name}' not found in given DataFrame.\n\n"
                f"Hint: did you mean one of {columns_str} instead?"
            )
            raise ValueError(msg)
        return df[col_name]

    return Expression(func, f"col({col_name!r})")


__all__ = ["Expression", "col"]
