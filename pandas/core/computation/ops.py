"""
Operator classes for eval.
"""

# [MODULE DOC]
# Expression-tree node classes and operator tables for the expression
# evaluator. Public contents: Term (a named variable or literal leaf) and its
# Constant subclass; Op (generic operator node) with BinOp, UnaryOp and
# MathCall subclasses; FuncNode (wrapper for supported numpy math functions);
# the is_term predicate; symbol tuples for comparison, boolean, arithmetic and
# unary operators plus tables mapping symbols to functions; tuples of
# supported math and reduction function names; and the prefix tag marking
# local (at-sign) variables.
# Used by the expr parser (builds the tree), the engines (print and evaluate
# the tree) and the pytables module (subclasses these). Importing has no side
# effects beyond building the operator tables.

from __future__ import annotations

from datetime import datetime
from functools import partial
import operator
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np

from pandas._libs.tslibs import Timestamp

from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
)

import pandas.core.common as com
from pandas.core.computation.common import (
    ensure_decoded,
    result_type_many,
)
from pandas.core.computation.scope import DEFAULT_GLOBALS

from pandas.io.formats.printing import (
    pprint_thing,
    pprint_thing_encoded,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
    )

REDUCTIONS = ("sum", "prod", "min", "max")

_unary_math_ops = (
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "expm1",
    "log1p",
    "sqrt",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "arccosh",
    "arcsinh",
    "arctanh",
    "abs",
    "log10",
    "floor",
    "ceil",
)
_binary_math_ops = ("arctan2",)

MATHOPS = _unary_math_ops + _binary_math_ops


LOCAL_TAG = "__pd_eval_local_"


class Term:
    # [CLASS DOC]
    # Leaf node of an expression tree: a named variable that is resolved
    # against a scope when constructed. State: the name, the scope (env), the
    # side ("left" or "right" when used in a comparison, else none), an
    # encoding for byte strings, an is-local flag (name carries the local tag
    # or is in the default globals) and the resolved value.
    # Constructing with a non-string name yields the Constant subclass.
    # Key members: value/name/type properties, update (set value and register in
    # scope), evaluate (returns itself), calling returns the value.
    def __new__(cls, name, env, side=None, encoding=None):
        # [FUNCTION DOC]
        # HELPER: construction hook. Returns a new uninitialised instance of
        # the Constant class when name is not a string, otherwise of the class
        # being constructed. Other arguments are ignored here and consumed by
        # the initialiser.
        klass = Constant if not isinstance(name, str) else cls
        supr_new = super(Term, klass).__new__
        return supr_new(klass)

    is_local: bool

    def __init__(self, name, env, side=None, encoding=None) -> None:
        # [FUNCTION DOC]
        # Initialises the term. Inputs: name (string, or any literal for
        # constants), env (scope used to resolve it), side, encoding. Marks the
        # term local when its text starts with the local tag or is a default
        # global name, then resolves and stores the value (which may raise the
        # undefined-variable error or a not-implemented error for arrays with
        # more than two dimensions). The encoding is stored after resolution.
        # name is a str for Term, but may be something else for subclasses
        self._name = name
        self.env = env
        self.side = side
        tname = str(name)
        self.is_local = tname.startswith(LOCAL_TAG) or tname in DEFAULT_GLOBALS
        self._value = self._resolve_name()
        self.encoding = encoding

    @property
    def local_name(self) -> str:
        # [FUNCTION DOC]
        # Property: the name with the local tag removed (for non-local names, the
        # name itself).
        return self.name.replace(LOCAL_TAG, "")

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the pretty-printed name.
        return pprint_thing(self.name)

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Returns the current value; all arguments are ignored (they exist so
        # terms can be called like other tree nodes with a scope argument).
        return self.value

    def evaluate(self, *args, **kwargs) -> Term:
        # [FUNCTION DOC]
        # Pre-evaluation hook: a leaf needs no work, so returns itself; all
        # arguments are ignored.
        return self

    def _resolve_name(self):
        # [FUNCTION DOC]
        # HELPER: looks up the term's value in the scope. Uses the local name;
        # the local flag is cleared if the scope holds a class (type object)
        # under that name. Delegates to the scope's resolve routine, stores the
        # result through the update method (which registers it in the scope) and
        # returns it. Raises a not-implemented error if the result has an
        # integer ndim greater than two.
        local_name = str(self.local_name)
        is_local = self.is_local
        if local_name in self.env.scope and isinstance(
            self.env.scope[local_name], type
        ):
            is_local = False

        res = self.env.resolve(local_name, is_local=is_local)
        self.update(res)

        if hasattr(res, "ndim") and isinstance(res.ndim, int) and res.ndim > 2:
            raise NotImplementedError(
                "N-dimensional objects, where N > 2, are not supported with eval"
            )
        return res

    def update(self, value) -> None:
        # [FUNCTION DOC]
        # Sets the term's value. Input value: new value. For named (string)
        # terms, also asks the scope to register the value under the full name,
        # in the mapping holding the local name; then stores value.
        # KNOWN ISSUE: the docstring describes a lookup search order for local
        # variables, not what this method does (storing a value and updating the
        # scope).
        """
        search order for local (i.e., @variable) variables:

        scope, key_variable
        [('locals', 'local_name'),
         ('globals', 'local_name'),
         ('locals', 'key'),
         ('globals', 'key')]
        """
        key = self.name

        # if it's a variable name (otherwise a constant)
        if isinstance(key, str):
            self.env.swapkey(self.local_name, key, new_value=value)

        self.value = value

    @property
    def is_scalar(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the stored value is a scalar per the library's
        # scalar test.
        return is_scalar(self._value)

    @property
    def type(self):
        # [FUNCTION DOC]
        # Property: the value's dtype. Tries the dtype of the underlying values
        # attribute (can be slow for large mixed-type frames), then the value's
        # own dtype (arrays), then falls back to the Python type of the value
        # (scalars).
        try:
            # potentially very slow for large, mixed dtype frames
            return self._value.values.dtype
        except AttributeError:
            try:
                # ndarray
                return self._value.dtype
            except AttributeError:
                # scalar
                return type(self._value)

    return_type = type

    @property
    def raw(self) -> str:
        # [FUNCTION DOC]
        # Property: debug text giving the class name, name and type.
        return f"{type(self).__name__}(name={self.name!r}, type={self.type})"

    @property
    def is_datetime(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the term's type (unwrapping a dtype's scalar type
        # if present) is a datetime or numpy datetime type.
        try:
            t = self.type.type
        except AttributeError:
            t = self.type

        return issubclass(t, (datetime, np.datetime64))

    @property
    def value(self):
        # [FUNCTION DOC]
        # Property getter: the stored value.
        return self._value

    @value.setter
    def value(self, new_value) -> None:
        # [FUNCTION DOC]
        # Property setter: replaces the stored value without touching the scope.
        self._value = new_value

    @property
    def name(self):
        # [FUNCTION DOC]
        # Property: the term's name as given at construction.
        return self._name

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Property: number of dimensions of the stored value (raises an
        # attribute error for scalars).
        return self._value.ndim


class Constant(Term):
    # [CLASS DOC]
    # Literal leaf node (subclass of Term): its "name" is its own value, no
    # scope lookup occurs, and the printed form is the value's repr.
    def _resolve_name(self):
        # [FUNCTION DOC]
        # HELPER: a constant needs no lookup; returns the literal given as name.
        return self._name

    @property
    def name(self):
        # [FUNCTION DOC]
        # Property: the literal value itself (overrides the base meaning).
        return self.value

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the repr of the literal value.
        # in python 2 str() of float
        # can truncate shorter than repr()
        return repr(self.name)


_bool_op_map = {"not": "~", "and": "&", "or": "|"}


class Op:
    # [CLASS DOC]
    # Generic operator node of any arity. State: op, the operator symbol (the
    # words not, and, or are normalised to their symbolic forms), operands (an
    # iterable of Term or Op children) and encoding (for byte strings).
    # Iterating yields the operands. Provides result type and scalar/datetime
    # introspection used by the parser and subclasses.
    """
    Hold an operator of arbitrary arity.
    """

    op: str

    def __init__(self, op: str, operands: Iterable[Term | Op], encoding=None) -> None:
        # [FUNCTION DOC]
        # Stores the operator symbol (mapping the words not, and, or to the
        # symbols tilde, ampersand, pipe; other symbols as given), the operands
        # and the encoding.
        self.op = _bool_op_map.get(op, op)
        self.operands = operands
        self.encoding = encoding

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Returns an iterator over the operands.
        return iter(self.operands)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns infix text: each operand wrapped in parentheses and joined
        # with the operator symbol surrounded by spaces.
        """
        Print a generic n-ary operator and its operands using infix notation.
        """
        # recurse over the operands
        parened = (f"({pprint_thing(opr)})" for opr in self.operands)
        return pprint_thing(f" {self.op} ".join(parened))

    @property
    def return_type(self):
        # [FUNCTION DOC]
        # Property: result type of the operation. Comparison and boolean
        # operators give the numpy boolean type; otherwise the promoted common
        # type of the types of all leaf terms in the subtree.
        # clobber types to bool if the op is a boolean operator
        if self.op in (CMP_OPS_SYMS + BOOL_OPS_SYMS):
            return np.bool_
        return result_type_many(*(term.type for term in com.flatten(self)))

    @property
    def has_invalid_return_type(self) -> bool:
        # [FUNCTION DOC]
        # Property: flags an object-typed result built from non-object operand
        # types. Computed as the logical and of "return type is object" and the
        # set of operand types minus the object dtype.
        # KNOWN ISSUE: annotated to return a boolean, but when the result type is
        # object it returns the (possibly empty) set of remaining operand types
        # rather than a boolean.
        types = self.operand_types
        obj_dtype_set = frozenset([np.dtype("object")])
        return self.return_type == object and types - obj_dtype_set

    @property
    def operand_types(self):
        # [FUNCTION DOC]
        # Property: immutable set of the types of every leaf term in the subtree.
        return frozenset(term.type for term in com.flatten(self))

    @property
    def is_scalar(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when every direct operand is scalar.
        return all(operand.is_scalar for operand in self.operands)

    @property
    def is_datetime(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the operation's result type (unwrapping a dtype's
        # scalar type if present) is a datetime or numpy datetime type.
        try:
            t = self.return_type.type
        except AttributeError:
            t = self.return_type

        return issubclass(t, (datetime, np.datetime64))


def _in(x, y):
    # [FUNCTION DOC]
    # HELPER: implementation of the membership comparison operator. Inputs: x,
    # y. If x has a vectorised membership method, returns its result for y.
    # Otherwise, if x is list-like and y has that method, returns y's result
    # for x (reversed roles). Else falls back to plain Python membership.
    """
    Compute the vectorized membership of ``x in y`` if possible, otherwise
    use Python.
    """
    try:
        return x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return y.isin(x)
            except AttributeError:
                pass
        return x in y


def _not_in(x, y):
    # [FUNCTION DOC]
    # HELPER: implementation of the negated membership operator; same lookup
    # order as the positive version but returns the logical inversion of the
    # vectorised result, or plain non-membership as the fallback.
    """
    Compute the vectorized membership of ``x not in y`` if possible,
    otherwise use Python.
    """
    try:
        return ~x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return ~y.isin(x)
            except AttributeError:
                pass
        return x not in y


CMP_OPS_SYMS = (">", "<", ">=", "<=", "==", "!=", "in", "not in")
_cmp_ops_funcs = (
    operator.gt,
    operator.lt,
    operator.ge,
    operator.le,
    operator.eq,
    operator.ne,
    _in,
    _not_in,
)
_cmp_ops_dict = dict(zip(CMP_OPS_SYMS, _cmp_ops_funcs, strict=True))

BOOL_OPS_SYMS = ("&", "|", "and", "or")
_bool_ops_funcs = (operator.and_, operator.or_, operator.and_, operator.or_)
_bool_ops_dict = dict(zip(BOOL_OPS_SYMS, _bool_ops_funcs, strict=True))

ARITH_OPS_SYMS = ("+", "-", "*", "/", "**", "//", "%")
_arith_ops_funcs = (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.pow,
    operator.floordiv,
    operator.mod,
)
_arith_ops_dict = dict(zip(ARITH_OPS_SYMS, _arith_ops_funcs, strict=True))

_binary_ops_dict = {}

for d in (_cmp_ops_dict, _bool_ops_dict, _arith_ops_dict):
    _binary_ops_dict.update(d)


def is_term(obj) -> bool:
    # [FUNCTION DOC]
    # Returns True when obj is a Term instance (including constants).
    return isinstance(obj, Term)


class BinOp(Op):
    # [CLASS DOC]
    # Binary operator node (subclass of Op) with exactly two operands, stored
    # as lhs and rhs, and the Python function implementing the operator in func.
    # Supports comparison, boolean and arithmetic operators. Calling with a
    # scope evaluates directly in Python; the evaluate method pre-evaluates
    # sub-expressions for an engine.
    """
    Hold a binary operator and its operands.

    Parameters
    ----------
    op : str
    lhs : Term or Op
    rhs : Term or Op
    """

    def __init__(self, op: str, lhs, rhs) -> None:
        # [FUNCTION DOC]
        # Builds the node. Inputs: op, operator symbol; lhs and rhs, child
        # nodes. Rejects boolean operators with a scalar operand that are not
        # both boolean (not-implemented error), converts datetime comparisons
        # (in place on the children), then looks up the implementing function;
        # raises a value error listing valid operators if the symbol is unknown.
        super().__init__(op, (lhs, rhs))
        self.lhs = lhs
        self.rhs = rhs

        self._disallow_scalar_only_bool_ops()

        self.convert_values()

        try:
            self.func = _binary_ops_dict[op]
        except KeyError as err:
            # has to be made a list for python3
            keys = list(_binary_ops_dict.keys())
            raise ValueError(
                f"Invalid binary operator {op!r}, valid operators are {keys}"
            ) from err

    def __call__(self, env):
        # [FUNCTION DOC]
        # Evaluates the subtree directly in Python: calls both children with the
        # scope env and applies the operator function to the two results.
        # Returns that result.
        """
        Recursively evaluate an expression in Python space.

        Parameters
        ----------
        env : Scope

        Returns
        -------
        object
            The result of an evaluated expression.
        """
        # recurse over the left/right nodes
        left = self.lhs(env)
        right = self.rhs(env)

        return self.func(left, right)

    def evaluate(self, env, engine: str, parser, term_type, eval_in_python):
        # [FUNCTION DOC]
        # Pre-evaluates this node ahead of the engine. Inputs: env, scope;
        # engine and parser names; term_type, the class used to wrap the result;
        # eval_in_python, collection of operator symbols that must be evaluated
        # in Python. With the python engine the node is called directly.
        # Otherwise both children are pre-evaluated recursively; if the symbol is
        # in the Python-only collection the operator function is applied to the
        # children's values, else the full evaluator is invoked on this node
        # (local scope = env). The result is stored as a new temporary in env and
        # returned wrapped as a term of type term_type.
        """
        Evaluate a binary operation *before* being passed to the engine.

        Parameters
        ----------
        env : Scope
        engine : str
        parser : str
        term_type : type
        eval_in_python : list

        Returns
        -------
        term_type
            The "pre-evaluated" expression as an instance of ``term_type``
        """
        if engine == "python":
            res = self(env)
        else:
            # recurse over the left/right nodes

            left = self.lhs.evaluate(
                env,
                engine=engine,
                parser=parser,
                term_type=term_type,
                eval_in_python=eval_in_python,
            )

            right = self.rhs.evaluate(
                env,
                engine=engine,
                parser=parser,
                term_type=term_type,
                eval_in_python=eval_in_python,
            )

            # base cases
            if self.op in eval_in_python:
                res = self.func(left.value, right.value)
            else:
                from pandas.core.computation.eval import eval

                res = eval(self, local_dict=env, engine=engine, parser=parser)

        name = env.add_tmp(res)
        return term_type(name, env=env)

    def convert_values(self) -> None:
        # [FUNCTION DOC]
        # Normalises scalar operands compared against a datetime operand. If one
        # side is a datetime-typed term and the other a scalar term, the scalar
        # is converted to a timestamp (numbers are first turned into text,
        # bytes decoded); timezone-aware values are converted to UTC. The
        # converted value is written back to that term (mutation). Does nothing
        # for non-term operands.
        """
        Convert datetimes to a comparable value in an expression.
        """

        def stringify(value):
            # [FUNCTION DOC]
            # HELPER: converts value to text with the pretty printer, using the
            # node's encoding for byte output when one is set.
            encoder: Callable
            if self.encoding is not None:
                encoder = partial(pprint_thing_encoded, encoding=self.encoding)
            else:
                encoder = pprint_thing
            return encoder(value)

        lhs, rhs = self.lhs, self.rhs

        if is_term(lhs) and lhs.is_datetime and is_term(rhs) and rhs.is_scalar:
            v = rhs.value
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = Timestamp(ensure_decoded(v))
            if v.tz is not None:
                v = v.tz_convert("UTC")
            self.rhs.update(v)

        if is_term(rhs) and rhs.is_datetime and is_term(lhs) and lhs.is_scalar:
            v = lhs.value
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = Timestamp(ensure_decoded(v))
            if v.tz is not None:
                v = v.tz_convert("UTC")
            self.lhs.update(v)

    def _disallow_scalar_only_bool_ops(self) -> None:
        # [FUNCTION DOC]
        # HELPER: validation for the initialiser. Raises a not-implemented error
        # when the operator is a boolean one, at least one operand is scalar and
        # the two operand result types are not both boolean.
        rhs = self.rhs
        lhs = self.lhs

        # GH#24883 unwrap dtype if necessary to ensure we have a type object
        rhs_rt = rhs.return_type
        rhs_rt = getattr(rhs_rt, "type", rhs_rt)
        lhs_rt = lhs.return_type
        lhs_rt = getattr(lhs_rt, "type", lhs_rt)
        if (
            (lhs.is_scalar or rhs.is_scalar)
            and self.op in _bool_ops_dict
            and (
                not (
                    issubclass(rhs_rt, (bool, np.bool_))
                    and issubclass(lhs_rt, (bool, np.bool_))
                )
            )
        ):
            raise NotImplementedError("cannot evaluate scalar only bool ops")


UNARY_OPS_SYMS = ("+", "-", "~", "not")
_unary_ops_funcs = (operator.pos, operator.neg, operator.invert, operator.invert)
_unary_ops_dict = dict(zip(UNARY_OPS_SYMS, _unary_ops_funcs, strict=True))


class UnaryOp(Op):
    # [CLASS DOC]
    # Unary operator node (subclass of Op) with one operand and the implementing
    # function in func. Supported symbols: plus, minus, tilde and the word not
    # (both invert).
    """
    Hold a unary operator and its operands.

    Parameters
    ----------
    op : str
        The token used to represent the operator.
    operand : Term or Op
        The Term or Op operand to the operator.

    Raises
    ------
    ValueError
        * If no function associated with the passed operator token is found.
    """

    def __init__(self, op: Literal["+", "-", "~", "not"], operand) -> None:
        # [FUNCTION DOC]
        # Builds the node from an operator symbol and an operand; raises a value
        # error listing the valid symbols when the symbol is unsupported.
        super().__init__(op, (operand,))
        self.operand = operand

        try:
            self.func = _unary_ops_dict[op]
        except KeyError as err:
            raise ValueError(
                f"Invalid unary operator {op!r}, valid operators are {UNARY_OPS_SYMS}"
            ) from err

    def __call__(self, env) -> MathCall:
        # [FUNCTION DOC]
        # Evaluates the operand with the scope env and applies the unary function
        # to the result.
        # KNOWN ISSUE: the return annotation names the math-call node class, but
        # the method returns the computed value, not a node.
        operand = self.operand(env)
        # error: Cannot call function of unknown type
        return self.func(operand)  # type: ignore[operator]

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the operator symbol followed by the parenthesised operand text.
        return pprint_thing(f"{self.op}({self.operand})")

    @property
    def return_type(self) -> np.dtype:
        # [FUNCTION DOC]
        # Property: boolean dtype when the operand is boolean or is itself a
        # comparison or boolean operator node; otherwise the default integer
        # dtype.
        # KNOWN ISSUE: for non-boolean operands (for example negating a float
        # array) it still reports an integer dtype regardless of the operand's
        # actual type.
        operand = self.operand
        if operand.return_type == np.dtype("bool"):
            return np.dtype("bool")
        if isinstance(operand, Op) and (
            operand.op in _cmp_ops_dict or operand.op in _bool_ops_dict
        ):
            return np.dtype("bool")
        return np.dtype("int")


class MathCall(Op):
    # [CLASS DOC]
    # Operator node (subclass of Op) representing a call of a supported
    # numpy math function with any number of argument nodes; the operator symbol
    # is the function name and func holds the function wrapper.
    def __init__(self, func, args) -> None:
        # [FUNCTION DOC]
        # Builds the node from a function wrapper (supplying name and callable)
        # and the argument nodes.
        super().__init__(func.name, args)
        self.func = func

    def __call__(self, env):
        # [FUNCTION DOC]
        # Evaluates each argument with the scope env and calls the wrapped numpy
        # function on the results; returns that result.
        # error: "Op" not callable
        operands = [op(env) for op in self.operands]  # type: ignore[operator]
        return self.func.func(*operands)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the function name followed by the comma-joined argument texts in
        # parentheses.
        operands = map(str, self.operands)
        return pprint_thing(f"{self.op}({','.join(operands)})")


class FuncNode:
    # [CLASS DOC]
    # Wrapper for a supported math function: stores the function name and the
    # matching numpy function in func. Calling an instance builds a MathCall node.
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC]
        # Validates name against the supported math function names (raises a
        # value error naming the unsupported function) and looks up the numpy
        # function of that name.
        if name not in MATHOPS:
            raise ValueError(f'"{name}" is not a supported function')
        self.name = name
        self.func = getattr(np, name)

    def __call__(self, *args) -> MathCall:
        # [FUNCTION DOC]
        # Returns a MathCall node for this function with the given argument
        # nodes.
        return MathCall(self, args)
