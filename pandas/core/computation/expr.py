"""
:func:`~pandas.eval` parsers.
"""

# [MODULE DOC]
# Parser for evaluation expressions: turns expression text into a tree of
# term and operator nodes (defined in the ops module) by preprocessing the
# token stream and walking the resulting Python syntax tree.
# Public contents: BaseExprVisitor (the generic syntax-tree walker),
# PandasExprVisitor and PythonExprVisitor (the two registered parsers),
# the PARSERS registry mapping parser names to visitor classes, Expr (the
# object wrapping a parsed expression, used by the eval module and the engines),
# the disallow and add_ops class decorators, and token-level rewriting helpers.
# Import-time side effects: computes frozen sets of syntax-tree node names
# from the standard library, derives the supported and unsupported node sets
# from them (asserting no overlap), and decorates the visitor classes so that
# unsupported node visits raise not-implemented errors and operator visits
# are generated.

from __future__ import annotations

import ast
from functools import (
    partial,
    reduce,
)
from keyword import iskeyword
import tokenize
from typing import (
    TYPE_CHECKING,
    ClassVar,
    TypeVar,
)

import numpy as np

from pandas.errors import UndefinedVariableError

from pandas.core.dtypes.common import is_string_dtype

import pandas.core.common as com
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS,
    BOOL_OPS_SYMS,
    CMP_OPS_SYMS,
    LOCAL_TAG,
    MATHOPS,
    REDUCTIONS,
    UNARY_OPS_SYMS,
    BinOp,
    Constant,
    FuncNode,
    Op,
    Term,
    UnaryOp,
    is_term,
)
from pandas.core.computation.parsing import (
    clean_backtick_quoted_toks,
    tokenize_string,
)
from pandas.core.computation.scope import Scope

from pandas.io.formats import printing

if TYPE_CHECKING:
    from collections.abc import Callable


def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]:
    # [FUNCTION DOC]
    # HELPER: token rewriter. Input tok: pair (token number, token text).
    # Returns the pair with a single equals sign replaced by the double
    # equals comparison; every other token is returned unchanged.
    """
    Rewrite the assignment operator for PyTables expressions that use ``=``
    as a substitute for ``==``.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    return toknum, "==" if tokval == "=" else tokval


def _replace_booleans(tok: tuple[int, str]) -> tuple[int, str]:
    # [FUNCTION DOC]
    # HELPER: token rewriter. For operator tokens, the bitwise-and symbol
    # becomes the word "and" and the bitwise-or symbol becomes the word "or"
    # (as name tokens), giving boolean precedence; all other tokens are
    # returned unchanged.
    """
    Replace ``&`` with ``and`` and ``|`` with ``or`` so that bitwise
    precedence is changed to boolean precedence.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    if toknum == tokenize.OP:
        if tokval == "&":
            return tokenize.NAME, "and"
        elif tokval == "|":
            return tokenize.NAME, "or"
        return toknum, tokval
    return toknum, tokval


def _replace_locals(tok: tuple[int, str]) -> tuple[int, str]:
    # [FUNCTION DOC]
    # HELPER: token rewriter. An operator token that is the at-sign is
    # replaced by an operator token containing the local-variable tag prefix,
    # so that the following name becomes a tagged identifier after joining;
    # other tokens are returned unchanged.
    """
    Replace local variables with a syntactically valid name.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values

    Notes
    -----
    This is somewhat of a hack in that we rewrite a string such as ``'@a'`` as
    ``'__pd_eval_local_a'`` by telling the tokenizer that ``__pd_eval_local_``
    is a ``tokenize.OP`` and to replace the ``'@'`` symbol with it.
    """
    toknum, tokval = tok
    if toknum == tokenize.OP and tokval == "@":
        return tokenize.OP, LOCAL_TAG
    return toknum, tokval


def _compose2(f, g):
    # [FUNCTION DOC]
    # HELPER: function composition. Returns a callable accepting any
    # arguments that applies g to them first and then f to g's result.
    """
    Compose 2 callables.
    """
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def _compose(*funcs):
    # [FUNCTION DOC]
    # HELPER: composes two or more callables by folding with the pairwise
    # composer, so the last function listed is applied first. Asserts (raises
    # an assertion error) when fewer than two are given. Returns the composed
    # callable.
    """
    Compose 2 or more callables.
    """
    assert len(funcs) > 1, "At least 2 callables must be passed to compose"
    return reduce(_compose2, funcs)


def _preparse(
    source: str,
    f=_compose(
        _replace_locals, _replace_booleans, _rewrite_assign, clean_backtick_quoted_toks
    ),
) -> str:
    # [FUNCTION DOC]
    # HELPER: preprocesses expression source before parsing. Inputs: source,
    # text (backticks allowed); f, a callable mapping a (token number, token
    # text) pair to a pair of the same shape, defaulting to the composition of
    # the local-tag, boolean, assignment and backtick rewriters (the backtick
    # rewriter runs first). Asserts f is callable. Tokenizes the source, applies
    # f to each token and rejoins the tokens. Returns the new source text.
    # KNOWN ISSUE: the docstring lists only three rewriters as the default; the
    # real default also includes the backtick-quoted token cleaner.
    """
    Compose a collection of tokenization functions.

    Parameters
    ----------
    source : str
        A Python source code string
    f : callable
        This takes a tuple of (toknum, tokval) as its argument and returns a
        tuple with the same structure but possibly different elements. Defaults
        to the composition of ``_rewrite_assign``, ``_replace_booleans``, and
        ``_replace_locals``.

    Returns
    -------
    str
        Valid Python source code

    Notes
    -----
    The `f` parameter can be any callable that takes *and* returns input of the
    form ``(toknum, tokval)``, where ``toknum`` is one of the constants from
    the ``tokenize`` module and ``tokval`` is a string.
    """
    assert callable(f), "f must be callable"
    return tokenize.untokenize(
        f(x)
        for x in tokenize_string(source)  # pyright: ignore[reportArgumentType]
    )


def _is_type(t):
    # [FUNCTION DOC]
    # HELPER: factory. Input t: a type or tuple of types. Returns a predicate
    # that takes a term and reports whether its value is an instance of t.
    """
    Factory for a type checking function of type ``t`` or tuple of types.
    """
    return lambda x: isinstance(x.value, t)


_is_list = _is_type(list)
_is_str = _is_type(str)


# partition all AST nodes
_all_nodes = frozenset(
    node
    for node in (getattr(ast, name) for name in dir(ast))
    if isinstance(node, type) and issubclass(node, ast.AST)
)


def _filter_nodes(superclass, all_nodes=_all_nodes):
    # [FUNCTION DOC]
    # HELPER: Inputs: superclass, a syntax-tree base class; all_nodes, the
    # collection of all syntax-tree classes (default: the module-level set).
    # Returns a frozen set of the names of the classes that are subclasses of
    # superclass.
    """
    Filter out AST nodes that are subclasses of ``superclass``.
    """
    node_names = (node.__name__ for node in all_nodes if issubclass(node, superclass))
    return frozenset(node_names)


_all_node_names = frozenset(x.__name__ for x in _all_nodes)
_mod_nodes = _filter_nodes(ast.mod)
_stmt_nodes = _filter_nodes(ast.stmt)
_expr_nodes = _filter_nodes(ast.expr)
_expr_context_nodes = _filter_nodes(ast.expr_context)
_boolop_nodes = _filter_nodes(ast.boolop)
_operator_nodes = _filter_nodes(ast.operator)
_unary_op_nodes = _filter_nodes(ast.unaryop)
_cmp_op_nodes = _filter_nodes(ast.cmpop)
_comprehension_nodes = _filter_nodes(ast.comprehension)
_handler_nodes = _filter_nodes(ast.excepthandler)
_arguments_nodes = _filter_nodes(ast.arguments)
_keyword_nodes = _filter_nodes(ast.keyword)
_alias_nodes = _filter_nodes(ast.alias)


# nodes that we don't support directly but are needed for parsing
_hacked_nodes = frozenset(["Assign", "Module", "Expr"])


_unsupported_expr_nodes = frozenset(
    [
        "Yield",
        "GeneratorExp",
        "IfExp",
        "DictComp",
        "SetComp",
        "Repr",
        "Lambda",
        "Set",
        "AST",
        "Is",
        "IsNot",
    ]
)

# these nodes are low priority or won't ever be supported (e.g., AST)
_unsupported_nodes = (
    _stmt_nodes
    | _mod_nodes
    | _handler_nodes
    | _arguments_nodes
    | _keyword_nodes
    | _alias_nodes
    | _expr_context_nodes
    | _unsupported_expr_nodes
) - _hacked_nodes

# we're adding a different assignment in some cases to be equality comparison
# and we don't want `stmt` and friends in their so get only the class whose
# names are capitalized
_base_supported_nodes = (_all_node_names - _unsupported_nodes) | _hacked_nodes
intersection = _unsupported_nodes & _base_supported_nodes
_msg = f"cannot both support and not support {intersection}"
assert not intersection, _msg


def _node_not_implemented(node_name: str) -> Callable[..., None]:
    # [FUNCTION DOC]
    # HELPER: for the disallow decorator. Input node_name: syntax-tree node
    # class name. Returns a method that, when called with any arguments, raises
    # a not-implemented error saying nodes of that kind are not implemented.
    """
    Return a function that raises a NotImplementedError with a passed node name.
    """

    def f(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: the generated visit method; ignores its arguments and raises
        # a not-implemented error naming the node kind.
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")

    return f


# should be bound by BaseExprVisitor but that creates a circular dependency:
# _T is used in disallow, but disallow is used to define BaseExprVisitor
# https://github.com/microsoft/pyright/issues/2315
_T = TypeVar("_T")


def disallow(nodes: set[str]) -> Callable[[type[_T]], type[_T]]:
    # [FUNCTION DOC]
    # Class decorator factory. Input nodes: set of syntax-tree node class names
    # to forbid. Returns a decorator that, for each name, installs a visit
    # method which raises a not-implemented error, records the method names in
    # the class's tuple of unsupported node method names (reset to empty first,
    # then extended), and returns the class. Mutates the decorated class.
    """
    Decorator to disallow certain nodes from parsing. Raises a
    NotImplementedError instead.

    Returns
    -------
    callable
    """

    def disallowed(cls: type[_T]) -> type[_T]:
        # [FUNCTION DOC]
        # HELPER: the actual class decorator; see the factory description.
        # error: "Type[_T]" has no attribute "unsupported_nodes"
        cls.unsupported_nodes = ()  # type: ignore[attr-defined]
        for node in nodes:
            new_method = _node_not_implemented(node)
            name = f"visit_{node}"
            # error: "Type[_T]" has no attribute "unsupported_nodes"
            cls.unsupported_nodes += (name,)  # type: ignore[attr-defined]
            setattr(cls, name, new_method)
        return cls

    return disallowed


def _op_maker(op_class, op_symbol):
    # [FUNCTION DOC]
    # HELPER: for the operator-generating decorator. Inputs: op_class, an
    # operator node class; op_symbol, its operator symbol. Returns a method
    # (taking a node and extra arguments) whose result is a partially applied
    # constructor of op_class with the symbol already bound.
    """
    Return a function to create an op class with its symbol already passed.

    Returns
    -------
    callable
    """

    def f(self, node, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: the generated visit method for an operator node. The node
        # itself is ignored; returns a partial constructor with the symbol and
        # any extra arguments bound.
        """
        Return a partial function with an Op subclass with an operator already passed.

        Returns
        -------
        callable
        """
        return partial(op_class, op_symbol, *args, **kwargs)

    return f


_op_classes = {"binary": BinOp, "unary": UnaryOp}


def add_ops(op_classes):
    # [FUNCTION DOC]
    # Class decorator factory. Input op_classes: mapping from kind prefix
    # ("binary" or "unary") to the operator node class. Returns a decorator
    # which, for each kind, reads the class's operator symbols and symbol-to-
    # node-name map and installs a visit method for each node name that is not
    # none, built by the operator maker. Mutates the decorated class.
    """
    Decorator to add default implementation of ops.
    """

    def f(cls):
        # [FUNCTION DOC]
        # HELPER: the actual class decorator; see the factory description.
        for op_attr_name, op_class in op_classes.items():
            ops = getattr(cls, f"{op_attr_name}_ops")
            ops_map = getattr(cls, f"{op_attr_name}_op_nodes_map")
            for op in ops:
                op_node = ops_map[op]
                if op_node is not None:
                    made_op = _op_maker(op_class, op)
                    setattr(cls, f"visit_{op_node}", made_op)
        return cls

    return f


@disallow(_unsupported_nodes)
@add_ops(_op_classes)
class BaseExprVisitor(ast.NodeVisitor):
    # [CLASS DOC]
    # Syntax-tree walker (subclass of the standard node visitor) that converts
    # a parsed expression into nodes from the ops module. Constructor state:
    # env (scope), engine and parser names, preparser (callable converting
    # source text before parsing) and assigner (name assigned to, set when an
    # assignment is visited, else none).
    # Class attributes: the term and constant node classes used, tables of
    # binary and unary operator symbols with their syntax-tree node names, a map
    # rewriting equality and membership operators for string or list operands,
    # and the tuple of unsupported node visitor names filled by the disallow
    # decorator. Operator visit methods are generated by the add_ops decorator;
    # unsupported node visits raise not-implemented errors.
    """
    Custom ast walker. Parsers of other engines should subclass this class
    if necessary.

    Parameters
    ----------
    env : Scope
    engine : str
    parser : str
    preparser : callable
    """

    const_type: ClassVar[type[Term]] = Constant
    term_type: ClassVar[type[Term]] = Term

    binary_ops = CMP_OPS_SYMS + BOOL_OPS_SYMS + ARITH_OPS_SYMS
    binary_op_nodes = (
        "Gt",
        "Lt",
        "GtE",
        "LtE",
        "Eq",
        "NotEq",
        "In",
        "NotIn",
        "BitAnd",
        "BitOr",
        "And",
        "Or",
        "Add",
        "Sub",
        "Mult",
        "Div",
        "Pow",
        "FloorDiv",
        "Mod",
    )
    binary_op_nodes_map = dict(zip(binary_ops, binary_op_nodes, strict=True))

    unary_ops = UNARY_OPS_SYMS
    unary_op_nodes = "UAdd", "USub", "Invert", "Not"
    unary_op_nodes_map = dict(zip(unary_ops, unary_op_nodes, strict=True))

    rewrite_map = {
        ast.Eq: ast.In,
        ast.NotEq: ast.NotIn,
        ast.In: ast.In,
        ast.NotIn: ast.NotIn,
    }

    unsupported_nodes: tuple[str, ...]

    def __init__(self, env, engine, parser, preparser=_preparse) -> None:
        # [FUNCTION DOC]
        # Stores the scope, engine name, parser name and preparser, and sets the
        # assigner to none.
        self.env = env
        self.engine = engine
        self.parser = parser
        self.preparser = preparser
        self.assigner = None

    def visit(self, node, **kwargs):
        # [FUNCTION DOC]
        # Dispatching visit. Input node: either expression text or a syntax-tree
        # node; extra keyword arguments are forwarded to the specific visitor.
        # Text is first run through the preparser and parsed (with missing
        # location info filled); on a syntax error, if any whitespace-separated
        # word of the preprocessed text is a Python keyword, the error message is
        # replaced with a note that a keyword is not a valid identifier, and the
        # error is re-raised. Then dispatches to the method named after the node
        # class (attribute error if none exists) and returns its result.
        if isinstance(node, str):
            clean = self.preparser(node)
            try:
                node = ast.fix_missing_locations(ast.parse(clean))
            except SyntaxError as e:
                if any(iskeyword(x) for x in clean.split()):
                    e.msg = "Python keyword not valid identifier in numexpr query"
                raise e

        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method)
        return visitor(node, **kwargs)

    def visit_Module(self, node, **kwargs):
        # [FUNCTION DOC]
        # Visits the single statement of a module; raises a syntax error unless
        # the body holds exactly one statement. Returns that visit's result.
        if len(node.body) != 1:
            raise SyntaxError("only a single expression is allowed")
        expr = node.body[0]
        return self.visit(expr, **kwargs)

    def visit_Expr(self, node, **kwargs):
        # [FUNCTION DOC]
        # Visits the wrapped expression value and returns its result.
        return self.visit(node.value, **kwargs)

    def _rewrite_membership_op(self, node, left, right):
        # [FUNCTION DOC]
        # HELPER: for comparison handling. Inputs: node, the comparison or
        # binary-operation node; left and right, already visited operands. When
        # both operands are terms and the operator is equality, inequality,
        # membership or non-membership, and at least one operand holds a string
        # or list, the operator is replaced by the membership counterpart
        # (equality becomes membership, inequality becomes non-membership). Any
        # string-valued operand is wrapped into a new one-element list stored as
        # a temporary variable in the scope and replaced by a term for it.
        # Returns (operator node visit result, operator instance, left, right).
        # the kind of the operator (is actually an instance)
        op_instance = node.op
        op_type = type(op_instance)

        # must be two terms and the comparison operator must be ==/!=/in/not in
        if is_term(left) and is_term(right) and op_type in self.rewrite_map:
            left_list, right_list = map(_is_list, (left, right))
            left_str, right_str = map(_is_str, (left, right))

            # if there are any strings or lists in the expression
            if left_list or right_list or left_str or right_str:
                op_instance = self.rewrite_map[op_type]()

            # pop the string variable out of locals and replace it with a list
            # of one string, kind of a hack
            if right_str:
                name = self.env.add_tmp([right.value])
                right = self.term_type(name, self.env)

            if left_str:
                name = self.env.add_tmp([left.value])
                left = self.term_type(name, self.env)

        op = self.visit(op_instance)
        return op, op_instance, left, right

    def _maybe_transform_eq_ne(self, node, left=None, right=None):
        # [FUNCTION DOC]
        # HELPER: visits the operands if not supplied (left with side "left",
        # right with side "right"; for a boolean-operation node the caller
        # supplies both) and applies the membership rewrite. Returns the same
        # four values as the membership rewrite helper.
        if left is None:
            left = self.visit(node.left, side="left")
        if right is None:
            right = self.visit(node.right, side="right")
        op, op_class, left, right = self._rewrite_membership_op(node, left, right)
        return op, op_class, left, right

    def _maybe_downcast_constants(self, left, right):
        # [FUNCTION DOC]
        # HELPER: when one operand is a scalar and the other a non-scalar of
        # 32-bit float type, the scalar is converted to a 32-bit float stored as
        # a temporary and replaced by a new term, to avoid promoting the array
        # to 64 bits. Returns the possibly replaced (left, right).
        f32 = np.dtype(np.float32)
        if (
            left.is_scalar
            and hasattr(left, "value")
            and not right.is_scalar
            and right.return_type == f32
        ):
            # right is a float32 array, left is a scalar
            name = self.env.add_tmp(np.float32(left.value))
            left = self.term_type(name, self.env)
        if (
            right.is_scalar
            and hasattr(right, "value")
            and not left.is_scalar
            and left.return_type == f32
        ):
            # left is a float32 array, right is a scalar
            name = self.env.add_tmp(np.float32(right.value))
            right = self.term_type(name, self.env)

        return left, right

    def _maybe_eval(self, binop, eval_in_python):
        # [FUNCTION DOC]
        # HELPER: pre-evaluates a binary operation node via its evaluate method
        # with this visitor's scope, engine, parser and term class, treating the
        # operators in eval_in_python as Python-space. Returns the resulting term
        # (a temporary holding the result).
        # eval `in` and `not in` (for now) in "partial" python space
        # things that can be evaluated in "eval" space will be turned into
        # temporary variables. for example,
        # [1,2] in a + 2 * b
        # in that case a + 2 * b will be evaluated using numexpr, and the "in"
        # call will be evaluated using isin (in python space)
        return binop.evaluate(
            self.env, self.engine, self.parser, self.term_type, eval_in_python
        )

    def _maybe_evaluate_binop(
        self,
        op,
        op_class,
        lhs,
        rhs,
        eval_in_python=("in", "not in"),
        maybe_eval_in_python=("==", "!=", "<", ">", "<=", ">="),
    ):
        # [FUNCTION DOC]
        # HELPER: builds and possibly pre-evaluates a binary operation.
        # Inputs: op, the partially applied operator constructor; op_class
        # (unused); lhs and rhs, operand nodes; eval_in_python, operator symbols
        # always evaluated in Python (default membership and non-membership);
        # maybe_eval_in_python, comparison symbols evaluated in Python when an
        # operand is of object or string type.
        # Constructs the node; raises a type error if its result type is invalid
        # (object result from non-object operands). For non-table engines, if
        # the right operand is datetime-typed, or the left is datetime-typed and
        # the operator is a comparison, pre-evaluates in Python. Otherwise
        # membership operators are pre-evaluated in Python; for non-table
        # engines, operands of object or string type cause pre-evaluation with
        # the extended Python-space set. Otherwise returns the unevaluated node.
        # KNOWN ISSUE: because of operator precedence in the condition, the
        # datetime check on the right operand is not restricted to comparison
        # operators, unlike the left operand's check.
        res = op(lhs, rhs)

        if res.has_invalid_return_type:
            raise TypeError(
                f"unsupported operand type(s) for {res.op}: "
                f"'{lhs.type}' and '{rhs.type}'"
            )

        if self.engine != "pytables" and (
            (res.op in CMP_OPS_SYMS and getattr(lhs, "is_datetime", False))
            or getattr(rhs, "is_datetime", False)
        ):
            # all date ops must be done in python bc numexpr doesn't work
            # well with NaT
            return self._maybe_eval(res, self.binary_ops)

        if res.op in eval_in_python:
            # "in"/"not in" ops are always evaluated in python
            return self._maybe_eval(res, eval_in_python)
        elif self.engine != "pytables":
            if (
                getattr(lhs, "return_type", None) == object
                or is_string_dtype(getattr(lhs, "return_type", None))
                or getattr(rhs, "return_type", None) == object
                or is_string_dtype(getattr(rhs, "return_type", None))
            ):
                # evaluate "==" and "!=" in python if either of our operands
                # has an object or string return type
                return self._maybe_eval(res, eval_in_python + maybe_eval_in_python)
        return res

    def visit_BinOp(self, node, **kwargs):
        # [FUNCTION DOC]
        # Visits a binary operation: transforms equality/membership operands,
        # downcasts constants against 32-bit float arrays, and builds (and maybe
        # pre-evaluates) the operator node. Returns that node or term.
        op, op_class, left, right = self._maybe_transform_eq_ne(node)
        left, right = self._maybe_downcast_constants(left, right)
        return self._maybe_evaluate_binop(op, op_class, left, right)

    def visit_UnaryOp(self, node, **kwargs):
        # [FUNCTION DOC]
        # Visits a unary operation: visits the operator and operand and returns
        # the unary operator node built from them.
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return op(operand)

    def visit_Name(self, node, **kwargs) -> Term:
        # [FUNCTION DOC]
        # Returns a term for the identifier, resolved in the scope; extra keyword
        # arguments (such as side) are passed to the term. May raise the
        # undefined-variable error.
        return self.term_type(node.id, self.env, **kwargs)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_NameConstant(self, node, **kwargs) -> Term:
        # [FUNCTION DOC]
        # Legacy visitor for named constants (true, false, none): returns a
        # constant term for the node's value.
        # KNOWN ISSUE: legacy; the supported Python versions parse these as
        # generic constant nodes, so this method is dead code (marked for removal
        # in a comment).
        return self.const_type(node.value, self.env)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_Num(self, node, **kwargs) -> Term:
        # [FUNCTION DOC]
        # Legacy visitor for number nodes: returns a constant term for the
        # node's value.
        # KNOWN ISSUE: legacy; modern parsers emit generic constant nodes, so
        # this method is dead code.
        return self.const_type(node.value, self.env)

    def visit_Constant(self, node, **kwargs) -> Term:
        # [FUNCTION DOC]
        # Returns a constant term holding the literal value of the node.
        return self.const_type(node.value, self.env)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_Str(self, node, **kwargs) -> Term:
        # [FUNCTION DOC]
        # Legacy visitor for string nodes: stores the string as a scope
        # temporary and returns a term for it.
        # KNOWN ISSUE: legacy; modern parsers emit generic constant nodes, so
        # this method is dead code.
        name = self.env.add_tmp(node.s)
        return self.term_type(name, self.env)

    def visit_List(self, node, **kwargs) -> Term:
        # [FUNCTION DOC]
        # Visits a list (also used for tuples): visits and evaluates each element
        # with the scope, stores the resulting list as a scope temporary and
        # returns a term for it.
        name = self.env.add_tmp([self.visit(e)(self.env) for e in node.elts])
        return self.term_type(name, self.env)

    visit_Tuple = visit_List

    def visit_Index(self, node, **kwargs):
        # [FUNCTION DOC]
        # Visits the wrapped value of a subscript index node (legacy wrapper node
        # from older Python versions) and returns its result.
        """df.index[4]"""
        return self.visit(node.value)

    def visit_Subscript(self, node, **kwargs) -> Term:
        # [FUNCTION DOC]
        # Handles indexing. Visits the indexed value and the slice, fully
        # evaluates the slice expression with the full evaluator in the same
        # scope, engine and parser, then indexes the value (the term's own value,
        # or, when the value is an operator node, its evaluated result) by that
        # result. Stores the outcome as a scope temporary and returns a term.
        from pandas import eval as pd_eval

        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        result = pd_eval(
            slobj, local_dict=self.env, engine=self.engine, parser=self.parser
        )
        try:
            # a Term instance
            v = value.value[result]
        except AttributeError:
            # an Op instance
            lhs = pd_eval(
                value, local_dict=self.env, engine=self.engine, parser=self.parser
            )
            v = lhs[result]
        name = self.env.add_tmp(v)
        return self.term_type(name, env=self.env)

    def visit_Slice(self, node, **kwargs) -> slice:
        # [FUNCTION DOC]
        # Builds a slice object: each of lower, upper and step that is present
        # is visited and its value taken; missing parts stay none. Returns the
        # slice.
        """df.index[slice(4,6)]"""
        lower = node.lower
        if lower is not None:
            lower = self.visit(lower).value
        upper = node.upper
        if upper is not None:
            upper = self.visit(upper).value
        step = node.step
        if step is not None:
            step = self.visit(step).value

        return slice(lower, upper, step)

    def visit_Assign(self, node, **kwargs):
        # [FUNCTION DOC]
        # Handles a single assignment. Raises a syntax error unless there is
        # exactly one target and it is a plain name, and a value error if the
        # scope has no target object. Resolves the target name through the
        # normal name visit (falling back to the bare identifier when undefined),
        # records its name as the assigner (a syntax error if that is empty) and
        # returns the visited right-hand side.
        """
        support a single assignment node, like

        c = a + b

        set the assigner at the top level, must be a Name node which
        might or might not exist in the resolvers

        """
        if len(node.targets) != 1:
            raise SyntaxError("can only assign a single expression")
        if not isinstance(node.targets[0], ast.Name):
            raise SyntaxError("left hand side of an assignment must be a single name")
        if self.env.target is None:
            raise ValueError("cannot assign without a target object")

        try:
            assigner = self.visit(node.targets[0], **kwargs)
        except UndefinedVariableError:
            assigner = node.targets[0].id

        self.assigner = getattr(assigner, "name", assigner)
        if self.assigner is None:
            raise SyntaxError(
                "left hand side of an assignment must be a single resolvable name"
            )

        return self.visit(node.value, **kwargs)

    def visit_Attribute(self, node, **kwargs):
        # [FUNCTION DOC]
        # Handles attribute access in a load context (other contexts raise a
        # value error). Visits the owner, takes its value (or evaluates it with
        # the scope if it has no value attribute), reads the attribute, stores it
        # as a scope temporary and returns a term. If the attribute is missing
        # and the owner is a name equal to the attribute (for example a module
        # name shadowed in the scope), returns the resolved owner object itself;
        # otherwise re-raises the attribute error.
        attr = node.attr
        value = node.value

        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            # resolve the value
            visited_value = self.visit(value)
            if hasattr(visited_value, "value"):
                resolved = visited_value.value
            else:
                resolved = visited_value(self.env)
            try:
                v = getattr(resolved, attr)
                name = self.env.add_tmp(v)
                return self.term_type(name, self.env)
            except AttributeError:
                # something like datetime.datetime where scope is overridden
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved
                raise

        raise ValueError(f"Invalid Attribute context {type(ctx).__name__}")

    def visit_Call(self, node, side=None, **kwargs):
        # [FUNCTION DOC]
        # Handles function calls. Inputs: node; side (unused); extra keyword
        # arguments, which are overwritten/extended with call keywords.
        # Callee resolution: attribute callees (other than the call dunder) are
        # visited as attributes; non-name, non-attribute callees raise a type
        # error; names are resolved in the scope, falling back to the supported
        # math function wrapper when undefined (the original undefined-variable
        # error is re-raised if the name is not a supported function). A missing
        # callee raises a value error. Terms are unwrapped to their values.
        # Supported math function wrappers get their visited arguments and return
        # an operator node (keyword arguments raise a type error). Other callables
        # are called immediately with evaluated positional and keyword
        # arguments; the result is stored as a scope temporary and returned as a
        # term.
        # KNOWN ISSUE: the fallback block that catches a value error from the
        # function wrapper only re-raises it, so it has no effect.
        if isinstance(node.func, ast.Attribute) and node.func.attr != "__call__":
            res = self.visit_Attribute(node.func)
        elif not isinstance(node.func, ast.Name):
            raise TypeError("Only named functions are supported")
        else:
            try:
                res = self.visit(node.func)
            except UndefinedVariableError:
                # Check if this is a supported function name
                try:
                    res = FuncNode(node.func.id)
                except ValueError:
                    # Raise original error
                    raise

        if res is None:
            # error: "expr" has no attribute "id"
            raise ValueError(
                f"Invalid function call {node.func.id}"  # type: ignore[union-attr]
            )
        if hasattr(res, "value"):
            res = res.value

        if isinstance(res, FuncNode):
            new_args = [self.visit(arg) for arg in node.args]

            if node.keywords:
                raise TypeError(
                    f'Function "{res.name}" does not support keyword arguments'
                )

            return res(*new_args)

        else:
            new_args = [self.visit(arg)(self.env) for arg in node.args]

            for key in node.keywords:
                if not isinstance(key, ast.keyword):
                    # error: Item "Attribute" of "Attribute | Name" has no
                    # attribute "id"
                    raise ValueError(
                        f"keyword error in function call '{node.func.id}'"  # type: ignore[union-attr]
                    )

                if key.arg:
                    kwargs[key.arg] = self.visit(key.value)(self.env)

            name = self.env.add_tmp(res(*new_args, **kwargs))
            return self.term_type(name=name, env=self.env)

    def translate_In(self, op):
        # [FUNCTION DOC]
        # Hook that lets subclasses translate a membership operator node;
        # the default returns op unchanged.
        return op

    def visit_Compare(self, node, **kwargs):
        # [FUNCTION DOC]
        # Handles comparisons. A single comparison is rewritten as a binary
        # operation node and visited. A chained comparison is split into pairwise
        # comparisons (each right operand becomes the next left operand), each
        # visited, and the results are combined with logical and by visiting a
        # boolean-operation node.
        ops = node.ops
        comps = node.comparators

        # base case: we have something like a CMP b
        if len(comps) == 1:
            op = self.translate_In(ops[0])
            binop = ast.BinOp(op=op, left=node.left, right=comps[0])
            return self.visit(binop)

        # recursive case: we have a chained comparison, a CMP b CMP c, etc.
        left = node.left
        values = []
        for op, comp in zip(ops, comps, strict=True):
            new_node = self.visit(
                ast.Compare(comparators=[comp], left=left, ops=[self.translate_In(op)])
            )
            left = comp
            values.append(new_node)
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def _try_visit_binop(self, bop):
        # [FUNCTION DOC]
        # HELPER: returns bop unchanged if it is already an operator or term node,
        # otherwise visits it and returns the result.
        if isinstance(bop, (Op, Term)):
            return bop
        return self.visit(bop)

    def visit_BoolOp(self, node, **kwargs):
        # [FUNCTION DOC]
        # Handles logical and/or over any number of values by folding pairwise
        # from the left with the inner combiner; returns the combined node.
        def visitor(x, y):
            # [FUNCTION DOC]
            # HELPER: combines two operands (visiting if not yet nodes) into a
            # binary operation node using the enclosing boolean operator, with
            # the same transform and evaluation steps as binary operations.
            lhs = self._try_visit_binop(x)
            rhs = self._try_visit_binop(y)

            op, op_class, lhs, rhs = self._maybe_transform_eq_ne(node, lhs, rhs)
            return self._maybe_evaluate_binop(op, node.op, lhs, rhs)

        operands = node.values
        return reduce(visitor, operands)


_python_not_supported = frozenset(["Dict", "BoolOp", "In", "NotIn"])
_numexpr_supported_calls = frozenset(REDUCTIONS + MATHOPS)


@disallow(
    (_unsupported_nodes | _python_not_supported)
    - (_boolop_nodes | frozenset(["BoolOp", "Attribute", "In", "NotIn", "Tuple"]))
)
class PandasExprVisitor(BaseExprVisitor):
    # [CLASS DOC]
    # Registered parser named "pandas" (subclass of the base visitor). Differs
    # from standard Python semantics: its preparser turns bitwise and/or into
    # boolean and/or word tokens (changing precedence) and local at-sign
    # names into tagged names, but does not rewrite the single equals sign.
    # Unlike the python parser it allows boolean operators, attribute access,
    # membership and tuples, while other unsupported nodes are disallowed.
    def __init__(
        self,
        env,
        engine,
        parser,
        preparser=partial(
            _preparse,
            f=_compose(_replace_locals, _replace_booleans, clean_backtick_quoted_toks),
        ),
    ) -> None:
        # [FUNCTION DOC]
        # Initialises the base visitor; the default preparser is the generic
        # preprocessor with the local-tag, boolean and backtick rewriters
        # (no equals rewrite).
        super().__init__(env, engine, parser, preparser)


@disallow(_unsupported_nodes | _python_not_supported | frozenset(["Not"]))
class PythonExprVisitor(BaseExprVisitor):
    # [CLASS DOC]
    # Registered parser named "python" (subclass of the base visitor) that keeps
    # strict Python semantics: its default preparser returns the source text
    # unchanged. Disallows unsupported nodes, dictionaries, boolean-operation
    # nodes, membership operators and the logical-not node.
    def __init__(
        self, env, engine, parser, preparser=lambda source, f=None: source
    ) -> None:
        # [FUNCTION DOC]
        # Initialises the base visitor with the given preparser (default: an
        # identity function that accepts and ignores an optional second argument).
        super().__init__(env, engine, parser, preparser=preparser)


class Expr:
    # [CLASS DOC]
    # A parsed expression. State: expr (source text or operator object),
    # env (scope), engine and parser names, a private visitor instance of the
    # chosen parser class, and terms (the parsed node tree, built at
    # construction). Calling the object evaluates the tree in Python with the
    # scope; length is the length of the source; the printed form is the tree's
    # text; names gives the variable names used.
    # KNOWN ISSUE: the docstring says the default for level is 2, but the real
    # default for level is 0.
    """
    Object encapsulating an expression.

    Parameters
    ----------
    expr : str
    engine : str, optional, default 'numexpr'
    parser : str, optional, default 'pandas'
    env : Scope, optional, default None
    level : int, optional, default 2
    """

    env: Scope
    engine: str
    parser: str

    def __init__(
        self,
        expr,
        engine: str = "numexpr",
        parser: str = "pandas",
        env: Scope | None = None,
        level: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Inputs: expr, source text; engine, engine name (default numexpr);
        # parser, key into the registry of parsers (default pandas; an unknown
        # name raises a key error); env, optional scope, else a new one built
        # from level plus one; level, stack depth for that scope (default 0).
        # Creates the visitor and immediately parses, storing the node tree.
        self.expr = expr
        self.env = env or Scope(level=level + 1)
        self.engine = engine
        self.parser = parser
        self._visitor = PARSERS[parser](self.env, self.engine, self.parser)
        self.terms = self.parse()

    @property
    def assigner(self):
        # [FUNCTION DOC]
        # Property: the name assigned to by the expression, or none when there
        # was no assignment.
        return getattr(self._visitor, "assigner", None)

    def __call__(self):
        # [FUNCTION DOC]
        # Evaluates the parsed tree directly in Python using the stored scope and
        # returns the result.
        return self.terms(self.env)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the pretty-printed text of the parsed tree.
        return printing.pprint_thing(self.terms)

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the length of the original expression (text length for strings).
        return len(self.expr)

    def parse(self):
        # [FUNCTION DOC]
        # Runs the visitor over the stored expression and returns the resulting
        # node tree (a term or operator node).
        """
        Parse an expression.
        """
        return self._visitor.visit(self.expr)

    @property
    def names(self):
        # [FUNCTION DOC]
        # Property: frozen set of variable names appearing in the tree. A lone
        # term yields a set holding just its name; otherwise the names of all
        # leaf terms of the tree.
        """
        Get the names in an expression.
        """
        if is_term(self.terms):
            return frozenset([self.terms.name])
        return frozenset(term.name for term in com.flatten(self.terms))


PARSERS = {"python": PythonExprVisitor, "pandas": PandasExprVisitor}
