"""manage PyTables query interface via Expressions"""

# [MODULE DOC]
# Adapts the expression machinery to the HDF5 table store's where-clause
# queries. Public contents: PyTablesScope (scope that also knows the
# queryable columns), Term and Constant (leaf nodes resolved against the
# queryables), BinOp and its specialisations FilterBinOp / JointFilterBinOp
# (post-read filters) and ConditionBinOp / JointConditionBinOp (conditions
# pushed into the table read as a numexpr-style string), UnaryOp (negation),
# PyTablesExprVisitor (syntax-tree walker for this dialect), PyTablesExpr (a
# parsed where clause), TermValue (a typed value for building conditions) and
# the maybe_expression predicate.
# A where clause is parsed into a tree; evaluating a PyTablesExpr prunes the
# tree twice, once into a condition and once into a filter.
# Subclasses classes from the ops, scope and expr modules. Used by the table
# store code in the io package. No import-time side effects.

from __future__ import annotations

import ast
from decimal import (
    Decimal,
    InvalidOperation,
)
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Self,
    cast,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
)
from pandas.errors import UndefinedVariableError

from pandas.core.dtypes.common import is_list_like

import pandas.core.common as com
from pandas.core.computation import (
    expr,
    ops,
    scope as _scope,
)
from pandas.core.computation.common import ensure_decoded
from pandas.core.computation.expr import BaseExprVisitor
from pandas.core.computation.ops import is_term
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index

from pandas.io.formats.printing import (
    pprint_thing,
    pprint_thing_encoded,
)

if TYPE_CHECKING:
    from pandas._typing import (
        TimeUnit,
        npt,
    )


class PyTablesScope(_scope.Scope):
    # [CLASS DOC]
    # Scope (subclass of the generic evaluation scope) carrying an extra slot,
    # queryables: a dict mapping queryable column or axis names to their
    # description objects (carrying kind, metadata and similar attributes).
    __slots__ = ("queryables",)

    queryables: dict[str, Any]

    def __init__(
        self,
        level: int,
        global_dict=None,
        local_dict=None,
        queryables: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the scope one level deeper (to skip this frame) with the given
        # globals and locals; queryables defaults to a fresh empty dict when
        # none or empty is given.
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables = queryables or {}


class Term(ops.Term):
    # [CLASS DOC]
    # Leaf node for table queries (subclass of the generic term). Name
    # resolution differs: a left-side term must be a queryable name; other terms
    # are looked up in the scope and fall back to the literal name text when
    # undefined (so unquoted words act as strings). The value property is
    # read-only.
    env: PyTablesScope

    def __new__(cls, name, env, side=None, encoding=None):
        # [FUNCTION DOC]
        # HELPER: construction hook. Returns a new uninitialised instance of
        # the constant class of this module for non-string names, otherwise of
        # the class being constructed.
        if isinstance(name, str):
            klass = cls
        else:
            klass = Constant
        return object.__new__(klass)

    def __init__(self, name, env: PyTablesScope, side=None, encoding=None) -> None:
        # [FUNCTION DOC]
        # Forwards all arguments to the generic term initialiser (no extra
        # behaviour; exists to narrow the scope type).
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        # [FUNCTION DOC]
        # HELPER: resolves the term's value. For the left side, the name must be
        # a queryable (otherwise raises a name error) and the name text itself is
        # the value. For other sides, looks up the name as a non-local in the
        # scope, returning the name text when it is undefined. Unlike the base
        # version, it does not register the value in the scope.
        # must be a queryables
        if self.side == "left":
            # Note: The behavior of __new__ ensures that self.name is a str here
            if self.name not in self.env.queryables:
                raise NameError(f"name {self.name!r} is not defined")
            return self.name

        # resolve the rhs (and allow it to be None)
        try:
            return self.env.resolve(self.name, is_local=False)
        except UndefinedVariableError:
            return self.name

    # read-only property overwriting read/write property
    @property  # type: ignore[misc]
    def value(self):
        # [FUNCTION DOC]
        # Read-only property returning the resolved value (replaces the base
        # read/write property, so assignment raises an error).
        return self._value


class Constant(Term):
    # [CLASS DOC]
    # Literal leaf node for table queries (subclass of this module's term):
    # requires a table scope and uses the literal itself as its value.
    def __init__(self, name, env: PyTablesScope, side=None, encoding=None) -> None:
        # [FUNCTION DOC]
        # Asserts the scope is a table scope (assertion error otherwise), then
        # runs the term initialiser.
        assert isinstance(env, PyTablesScope), type(env)
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        # [FUNCTION DOC]
        # HELPER: a constant's value is the literal given as name.
        return self._name


class BinOp(ops.BinOp):
    # [CLASS DOC]
    # Binary operator node for table queries (subclass of the generic binary
    # operator). Extra state: queryables (the dict of queryable names to
    # descriptions), encoding, and condition (string form, none until built).
    # The class attribute _max_selectors (31) caps how many equality values may
    # be OR-ed into one condition before a post-read filter is used instead.
    # Provides the helpers shared by the filter and condition subclasses:
    # pruning a tree to a given specialised class, value conforming and
    # conversion to the column kind, and column metadata properties.
    _max_selectors = 31

    op: str
    queryables: dict[str, Any]
    condition: str | None

    def __init__(self, op: str, lhs, rhs, queryables: dict[str, Any], encoding) -> None:
        # [FUNCTION DOC]
        # Runs the generic binary-operator initialiser, then stores the
        # queryables and encoding and sets condition to none.
        super().__init__(op, lhs, rhs)
        self.queryables = queryables
        self.encoding = encoding
        self.condition = None

    def _disallow_scalar_only_bool_ops(self) -> None:
        # [FUNCTION DOC]
        # HELPER: overrides the base validation with a no-op, since scalar-only
        # boolean operations are allowed in table queries.
        pass

    def prune(self, klass):
        # [FUNCTION DOC]
        # Recursively reduces this subtree to a single specialised node (a
        # condition or filter node). Input klass: the target specialised class.
        # Terms contribute their values; operator children are pruned first. The
        # combiner returns the other side when one side is none; when both sides
        # are of the same specialised kind it builds the joint class; when only
        # one side is of the requested class it returns that side; otherwise it
        # builds a new node of klass from the operator and sides and evaluates it
        # (which may give none). Returns the resulting node or none.
        def pr(left, right):
            # [FUNCTION DOC]
            # HELPER: combines two pruned sides as described for the enclosing
            # method.
            """create and return a new specialized BinOp from myself"""
            if left is None:
                return right
            elif right is None:
                return left

            k = klass
            if isinstance(left, ConditionBinOp):
                if isinstance(right, ConditionBinOp):
                    k = JointConditionBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right

            elif isinstance(left, FilterBinOp):
                if isinstance(right, FilterBinOp):
                    k = JointFilterBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right

            return k(
                self.op, left, right, queryables=self.queryables, encoding=self.encoding
            ).evaluate()

        left, right = self.lhs, self.rhs

        if is_term(left) and is_term(right):
            res = pr(left.value, right.value)
        elif not is_term(left) and is_term(right):
            res = pr(left.prune(klass), right.value)
        elif is_term(left) and not is_term(right):
            res = pr(left.value, right.prune(klass))
        elif not (is_term(left) or is_term(right)):
            res = pr(left.prune(klass), right.prune(klass))

        return res

    def conform(self, rhs):
        # [FUNCTION DOC]
        # Normalises a right-hand value into a flat list-like. Non-list-like
        # inputs are wrapped in a one-element list; arrays are flattened to one
        # dimension. Returns the result.
        # KNOWN ISSUE: the docstring says it works in place, but it returns a new
        # (wrapped or flattened) object and does not mutate its argument.
        """inplace conform rhs"""
        if not is_list_like(rhs):
            rhs = [rhs]
        if isinstance(rhs, np.ndarray):
            rhs = rhs.ravel()
        return rhs

    @property
    def is_valid(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the left operand is a known queryable name.
        """return True if this is a valid field"""
        return self.lhs in self.queryables

    @property
    def is_in_table(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the left operand's queryable entry exists and is
        # not none, meaning it is an actual table column that can appear in a
        # condition.
        """
        return True if this is a valid column name for generation (e.g. an
        actual column in the table)
        """
        return self.queryables.get(self.lhs) is not None

    @property
    def kind(self):
        # [FUNCTION DOC]
        # Property: the kind attribute of the left operand's queryable entry,
        # or none if absent.
        """the kind of my field"""
        return getattr(self.queryables.get(self.lhs), "kind", None)

    @property
    def meta(self):
        # [FUNCTION DOC]
        # Property: the meta attribute of the left operand's queryable entry
        # (for example a category marker), or none.
        """the meta of my field"""
        return getattr(self.queryables.get(self.lhs), "meta", None)

    @property
    def metadata(self):
        # [FUNCTION DOC]
        # Property: the metadata attribute of the left operand's queryable
        # entry (for example category values), or none.
        """the metadata of my field"""
        return getattr(self.queryables.get(self.lhs), "metadata", None)

    def generate(self, v) -> str:
        # [FUNCTION DOC]
        # Builds the condition fragment for one value. Input v: a typed term
        # value. Returns text of the form: open paren, left operand, operator,
        # the value rendered with the stored encoding, close paren.
        """create and return the op string for this TermValue"""
        val = v.tostring(self.encoding)
        return f"({self.lhs} {self.op} {val})"

    def convert_value(self, conv_val) -> TermValue:
        # [FUNCTION DOC]
        # Converts a raw query value into a typed term value matching the
        # column kind (decoded if bytes). Input conv_val: raw value.
        # Datetime kinds: numbers are turned into text, parsed as a timestamp in
        # the unit given in square brackets in the kind (default nanoseconds), made
        # UTC if timezone-aware; the converted value is the integer representation.
        # Timedelta kinds: text parsed as a timedelta, numbers taken as seconds,
        # other values converted; stored as an integer in the kind's unit.
        # Category columns: converted value is the integer code of the first
        # matching category, or minus one if absent, with kind integer.
        # Integer kind: parsed as a decimal and rounded half-even to an integer
        # (a non-numeric value raises the float conversion error). Float kind:
        # converted to a float. Boolean kind: text is true unless it is a
        # false-like word (false, f, no, n, none, 0, empty brackets or braces,
        # empty), other values use truthiness. Any other kind with a text value:
        # kept as a string, quoted according to encoding settings.
        # Raises a type error for non-text values with other kinds.
        # KNOWN ISSUE: if the column has no kind (none), the branch testing the
        # kind prefix for timedelta fails with an attribute error rather than a
        # clear message.
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """

        def stringify(value):
            # [FUNCTION DOC]
            # HELPER: text for value using the node's encoding for encoded output
            # when set, else the plain pretty printer.
            if self.encoding is not None:
                return pprint_thing_encoded(value, encoding=self.encoding)
            return pprint_thing(value)

        kind = ensure_decoded(self.kind)
        meta = ensure_decoded(self.meta)
        if kind == "datetime" or (kind and kind.startswith("datetime64")):
            if isinstance(conv_val, (int, float)):
                conv_val = stringify(conv_val)
            conv_val = ensure_decoded(conv_val)
            unit: TimeUnit = "ns"
            if "[" in kind:
                unit = cast("TimeUnit", kind.split("[")[-1][:-1])
            conv_val = Timestamp(conv_val).as_unit(unit)
            if conv_val.tz is not None:
                conv_val = conv_val.tz_convert("UTC")
            return TermValue(conv_val, conv_val._value, kind)
        elif kind.startswith("timedelta"):
            unit = "ns"
            if "[" in kind:
                unit = cast("TimeUnit", kind.split("[")[-1][:-1])
            if isinstance(conv_val, str):
                conv_val = Timedelta(conv_val)
            elif lib.is_integer(conv_val) or lib.is_float(conv_val):
                conv_val = Timedelta(conv_val, unit="s")
            else:
                conv_val = Timedelta(conv_val)
            conv_val = conv_val.as_unit(unit)._value
            return TermValue(int(conv_val), conv_val, kind)

        elif meta == "category":
            metadata = extract_array(self.metadata, extract_numpy=True)
            result: npt.NDArray[np.intp] | np.intp | int
            if conv_val not in metadata:
                result = -1
            else:
                # Find the index of the first match of conv_val in metadata
                result = np.flatnonzero(metadata == conv_val)[0]
            return TermValue(result, result, "integer")
        elif kind == "integer":
            try:
                v_dec = Decimal(conv_val)
            except InvalidOperation:
                # GH 54186
                # convert v to float to raise float's ValueError
                float(conv_val)
            else:
                conv_val = int(v_dec.to_integral_exact(rounding="ROUND_HALF_EVEN"))
            return TermValue(conv_val, conv_val, kind)
        elif kind == "float":
            conv_val = float(conv_val)
            return TermValue(conv_val, conv_val, kind)
        elif kind == "bool":
            if isinstance(conv_val, str):
                conv_val = conv_val.strip().lower() not in [
                    "false",
                    "f",
                    "no",
                    "n",
                    "none",
                    "0",
                    "[]",
                    "{}",
                    "",
                ]
            else:
                conv_val = bool(conv_val)
            return TermValue(conv_val, conv_val, kind)
        elif isinstance(conv_val, str):
            # string quoting
            return TermValue(conv_val, stringify(conv_val), "string")
        else:
            raise TypeError(
                f"Cannot compare {conv_val} of type {type(conv_val)} to {kind} column"
            )

    def convert_values(self) -> None:
        # [FUNCTION DOC]
        # Overrides the datetime conversion step of the base class with a no-op;
        # conversion happens later per value.
        pass


class FilterBinOp(BinOp):
    # [CLASS DOC]
    # Binary node (subclass of the table binary operator) describing a filter
    # applied after reading, used for columns outside the table or when too many
    # equality values exist. State: filter, none until evaluated, else a triple
    # (column name, function applying (non-)membership to an axis and values,
    # index of values).
    filter: tuple[Any, Any, Index] | None = None

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns "Filter: Not Initialized" when there is no filter, otherwise
        # text showing the column and the operator function.
        # KNOWN ISSUE: the produced text opens two brackets but closes only one.
        if self.filter is None:
            return "Filter: Not Initialized"
        return pprint_thing(f"[Filter : [{self.filter[0]}] -> [{self.filter[1]}]")

    def invert(self) -> Self:
        # [FUNCTION DOC]
        # Flips the filter's membership sense (in becomes not-in and vice versa),
        # mutating the stored filter if one exists. Returns self.
        """invert the filter"""
        if self.filter is not None:
            self.filter = (
                self.filter[0],
                self.generate_filter_op(invert=True),
                self.filter[2],
            )
        return self

    def format(self):
        # [FUNCTION DOC]
        # Returns a one-element list containing the filter triple (possibly none).
        """return the actual filter format"""
        return [self.filter]

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self | None:  # type: ignore[override]
        # [FUNCTION DOC]
        # Decides whether this node becomes a filter. Raises a value error if the
        # left operand is not a queryable. Values are the conformed right operand.
        # For a table column: only equality/inequality with more values than the
        # selector cap yields a filter (returns self); anything else returns none
        # (a condition handles it). For a non-table name: equality/inequality
        # builds the filter and returns self; any other operator raises a type
        # error. Mutates the filter attribute.
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        rhs = self.conform(self.rhs)
        values = list(rhs)

        if self.is_in_table:
            # if too many values to create the expression, use a filter instead
            if self.op in ["==", "!="] and len(values) > self._max_selectors:
                filter_op = self.generate_filter_op()
                self.filter = (self.lhs, filter_op, Index(values))

                return self
            return None

        # equality conditions
        if self.op in ["==", "!="]:
            filter_op = self.generate_filter_op()
            self.filter = (self.lhs, filter_op, Index(values))

        else:
            raise TypeError(
                f"passing a filterable condition to a non-table indexer [{self}]"
            )

        return self

    def generate_filter_op(self, invert: bool = False):
        # [FUNCTION DOC]
        # Returns a function of (axis, values) that tests membership (or its
        # negation). Input invert (default false) flips the sense. Inequality
        # without invert, or equality with invert, give the negated test; others
        # give the plain membership test.
        if (self.op == "!=" and not invert) or (self.op == "==" and invert):
            return lambda axis, vals: ~axis.isin(vals)
        else:
            return lambda axis, vals: axis.isin(vals)


class JointFilterBinOp(FilterBinOp):
    # [CLASS DOC]
    # Filter node that combines two filter nodes; it cannot be reduced to a
    # single filter format.
    def format(self):
        # [FUNCTION DOC]
        # Always raises a not-implemented error: joint filters cannot be
        # collapsed into a single filter.
        raise NotImplementedError("unable to collapse Joint Filters")

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self:  # type: ignore[override]
        # [FUNCTION DOC]
        # Returns self unchanged (no per-value work for a joint node).
        return self


class ConditionBinOp(BinOp):
    # [CLASS DOC]
    # Binary node (subclass of the table binary operator) that builds a condition
    # string to be evaluated by the table read. State: condition (inherited),
    # none until evaluated.
    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns text showing the condition string.
        return pprint_thing(f"[Condition : [{self.condition}]]")

    def invert(self):
        # [FUNCTION DOC]
        # Always raises a not-implemented error: conditions cannot be inverted
        # for the numexpr path.
        # KNOWN ISSUE: the docstring says it inverts the condition, but the body
        # only raises (the inversion code is commented out).
        """invert the condition"""
        # if self.condition is not None:
        #    self.condition = "~(%s)" % self.condition
        # return self
        raise NotImplementedError(
            "cannot use an invert condition when passing to numexpr"
        )

    def format(self):
        # [FUNCTION DOC]
        # Returns the condition string (possibly none).
        """return the actual ne format"""
        return self.condition

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self | None:  # type: ignore[override]
        # [FUNCTION DOC]
        # Builds the condition string. Raises a value error if the left operand
        # is not a queryable. Returns none for names not in the table. Converts
        # each right-hand value to a typed value. For equality/inequality with at
        # most the selector cap of values, the condition is the OR of one fragment
        # per value (more values: returns none so a filter is used). For other
        # operators the condition is the fragment for the first value only.
        # Returns self when a condition was set. Mutates the condition attribute.
        # KNOWN ISSUE: for inequality the fragments are still joined with OR,
        # which does not express "not in" semantics (each fragment is a
        # not-equal test).
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        # convert values if we are in the table
        if not self.is_in_table:
            return None

        rhs = self.conform(self.rhs)
        values = [self.convert_value(v) for v in rhs]

        # equality conditions
        if self.op in ["==", "!="]:
            # too many values to create the expression?
            if len(values) <= self._max_selectors:
                vs = [self.generate(v) for v in values]
                self.condition = f"({' | '.join(vs)})"

            # use a filter after reading
            else:
                return None
        else:
            self.condition = self.generate(values[0])

        return self


class JointConditionBinOp(ConditionBinOp):
    # [CLASS DOC]
    # Condition node combining two condition nodes with a boolean operator.
    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self:  # type: ignore[override]
        # [FUNCTION DOC]
        # Sets the condition to the two sides' condition strings joined by the
        # operator, each in parentheses, and returns self.
        self.condition = f"({self.lhs.condition} {self.op} {self.rhs.condition})"
        return self


class UnaryOp(ops.UnaryOp):
    # [CLASS DOC]
    # Unary node for table queries (subclass of the generic unary operator);
    # only logical inversion is supported, implemented by pruning.
    def prune(self, klass):
        # [FUNCTION DOC]
        # Prunes the operand to the requested specialised class and inverts the
        # result. Input klass: condition or filter node class. Raises a
        # not-implemented error if the operator is not the invert symbol. Returns
        # the inverted operand when it exists and holds a condition (for the
        # condition class) or filter (for the filter class), else none. Note the
        # condition class's inversion itself raises a not-implemented error.
        if self.op != "~":
            raise NotImplementedError("UnaryOp only support invert type ops")

        operand = self.operand
        operand = operand.prune(klass)

        if operand is not None and (
            (issubclass(klass, ConditionBinOp) and operand.condition is not None)
            or (
                not issubclass(klass, ConditionBinOp)
                and issubclass(klass, FilterBinOp)
                and operand.filter is not None
            )
        ):
            return operand.invert()
        return None


class PyTablesExprVisitor(BaseExprVisitor):
    # [CLASS DOC]
    # Syntax-tree walker for table where-clauses (subclass of the base
    # visitor). Uses this module's term and constant classes. Binary operator
    # visits create partial constructors of this module's binary node with the
    # visitor's extra keyword arguments (queryables, encoding) bound. A single
    # equals sign is a comparison here, and membership is treated as equality.
    const_type: ClassVar[type[ops.Term]] = Constant
    term_type: ClassVar[type[Term]] = Term

    def __init__(self, env, engine, parser, **kwargs) -> None:
        # [FUNCTION DOC]
        # Initialises the base visitor (the default preparser, which the
        # expression class later supplements) and replaces each generated binary
        # operator visit method on the instance by a function returning a
        # partial constructor of the table binary node with the operator symbol
        # and the extra keyword arguments bound (the lambda created per operator
        # takes the syntax node and ignores it).
        super().__init__(env, engine, parser)
        for bin_op in self.binary_ops:
            bin_node = self.binary_op_nodes_map[bin_op]
            setattr(
                self,
                f"visit_{bin_node}",
                lambda node, bin_op=bin_op: partial(BinOp, bin_op, **kwargs),
            )

    def visit_UnaryOp(self, node, **kwargs) -> ops.Term | UnaryOp | None:
        # [FUNCTION DOC]
        # Logical not or invert: returns a table unary node (inversion) around the
        # visited operand. Unary minus: returns a constant with the negated value
        # of the visited operand. Unary plus: raises a not-implemented error.
        # Other operators return none.
        # KNOWN ISSUE: a source comment says the final return of none may be
        # unreachable, since the parser only produces the four unary operators.
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return UnaryOp("~", self.visit(node.operand))
        elif isinstance(node.op, ast.USub):
            return self.const_type(-self.visit(node.operand).value, self.env)
        elif isinstance(node.op, ast.UAdd):
            raise NotImplementedError("Unary addition not supported")
        # TODO: return None might never be reached
        return None

    def visit_Index(self, node, **kwargs):
        # [FUNCTION DOC]
        # Visits the wrapped value of an index node and returns its value
        # (unwrapped from the term).
        return self.visit(node.value).value

    def visit_Assign(self, node, **kwargs):
        # [FUNCTION DOC]
        # Treats an assignment as an equality comparison between the target and
        # the value (a single equals sign means equality in where-clauses);
        # visits that comparison and returns the result.
        cmpr = ast.Compare(
            ops=[ast.Eq()], left=node.targets[0], comparators=[node.value]
        )
        return self.visit(cmpr)

    def visit_Subscript(self, node, **kwargs) -> ops.Term:
        # [FUNCTION DOC]
        # Handles simple indexing only. Visits the indexed value and the slice,
        # unwraps terms to their values and returns a constant holding the indexed
        # result. A type error during indexing is re-raised as a value error
        # naming the value and index.
        # only allow simple subscripts

        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        try:
            value = value.value
        except AttributeError:
            pass

        if isinstance(slobj, Term):
            # In py39 np.ndarray lookups with Term containing int raise
            slobj = slobj.value

        try:
            return self.const_type(value[slobj], self.env)
        except TypeError as err:
            raise ValueError(f"cannot subscript {value!r} with {slobj!r}") from err

    def visit_Attribute(self, node, **kwargs):
        # [FUNCTION DOC]
        # Handles attribute access in a load context: visits the owner, unwraps
        # term values, and returns a term for the attribute value. If the
        # attribute is missing and the owner is a name equal to the attribute,
        # returns the resolved owner object.
        # KNOWN ISSUE: when the attribute lookup fails and the owner is not a
        # matching name, or the context is not load, control reaches a value
        # error complaining about the attribute context, which is misleading for
        # the load case (the original attribute error is lost).
        attr = node.attr
        value = node.value

        ctx = type(node.ctx)
        if ctx == ast.Load:
            # resolve the value
            resolved = self.visit(value)

            # try to get the value to see if we are another expression
            try:
                resolved = resolved.value
            except AttributeError:
                pass

            try:
                return self.term_type(getattr(resolved, attr), self.env)
            except AttributeError:
                # something like datetime.datetime where scope is overridden
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved

        raise ValueError(f"Invalid Attribute context {ctx.__name__}")

    def translate_In(self, op):
        # [FUNCTION DOC]
        # Returns an equality operator node when op is a membership operator
        # node, else op unchanged.
        return ast.Eq() if isinstance(op, ast.In) else op

    def _rewrite_membership_op(self, node, left, right):
        # [FUNCTION DOC]
        # HELPER: overrides the base rewrite with no rewriting: returns the visited
        # operator, the operator node, and the operands unchanged.
        return self.visit(node.op), node.op, left, right


def _validate_where(w):
    # [FUNCTION DOC]
    # HELPER: validates a where-clause argument. Accepts a table expression
    # object, a string, or any list-like; returns it unchanged. Raises a type
    # error for anything else.
    # KNOWN ISSUE: the docstring gives a mapping as an example of an invalid
    # type, but a mapping counts as list-like and is therefore accepted.
    """
    Validate that the where statement is of the right type.

    The type may either be String, Expr, or list-like of Exprs.

    Parameters
    ----------
    w : String term expression, Expr, or list-like of Exprs.

    Returns
    -------
    where : The original where clause if the check was successful.

    Raises
    ------
    TypeError : An invalid data type was passed in for w (e.g. dict).
    """
    if not (isinstance(w, (PyTablesExpr, str)) or is_list_like(w)):
        raise TypeError(
            "where must be passed as a string, PyTablesExpr, "
            "or list-like of PyTablesExpr"
        )

    return w


class PyTablesExpr(expr.Expr):
    # [CLASS DOC]
    # A parsed where-clause for the table store (subclass of the generic parsed
    # expression). State: encoding, condition and filter (none until
    # evaluated), terms (parsed tree, none when no queryables were provided),
    # the private visitor (none likewise), env (table scope with queryables) and
    # expr (the combined text). Evaluating produces a (condition, filter) pair.
    # KNOWN ISSUE: the class docstring claims a "Returns" value and that a missing
    # queryables map means the column is non-indexable; in fact a missing map just
    # skips parsing, leaving no terms.
    """
    Hold a pytables-like expression, comprised of possibly multiple 'terms'.

    Parameters
    ----------
    where : string term expression, PyTablesExpr, or list-like of PyTablesExprs
    queryables : a "kinds" map (dict of column name -> kind), or None if column
        is non-indexable
    encoding : an encoding that will encode the query terms

    Returns
    -------
    a PyTablesExpr object

    Examples
    --------
    'index>=date'
    "columns=['A', 'D']"
    'columns=A'
    'columns==A'
    "~(columns=['A','B'])"
    'index>df.index[3] & string="bar"'
    '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    "ts>=Timestamp('2012-02-01')"
    "major_axis>=20130101"
    """

    _visitor: PyTablesExprVisitor | None
    env: PyTablesScope
    expr: str

    def __init__(
        self,
        where,
        queryables: dict[str, Any] | None = None,
        encoding=None,
        scope_level: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Inputs: where, a string, another table expression, or a list-like of
        # them (validated; invalid types raise a type error); queryables,
        # optional dict of names to descriptions; encoding, optional byte-string
        # encoding; scope_level, stack depth for the scope (default 0).
        # Sets attributes, captures the variables of any nested expression as the
        # local dictionary, joins list items into one text with each item
        # parenthesised and combined by logical and, builds the table scope, and
        # if queryables were given and the text is a string, registers them in the
        # scope, builds the visitor and parses the where text.
        where = _validate_where(where)

        self.encoding = encoding
        self.condition = None
        self.filter = None
        self.terms = None
        self._visitor = None

        # capture the environment if needed
        local_dict: _scope.DeepChainMap[Any, Any] | None = None

        if isinstance(where, PyTablesExpr):
            local_dict = where.env.scope
            _where = where.expr

        elif is_list_like(where):
            where = list(where)
            for idx, w in enumerate(where):
                if isinstance(w, PyTablesExpr):
                    local_dict = w.env.scope
                else:
                    where[idx] = _validate_where(w)
            _where = " & ".join([f"({w})" for w in com.flatten(where)])
        else:
            # _validate_where ensures we otherwise have a string
            _where = where

        self.expr = _where
        self.env = PyTablesScope(scope_level + 1, local_dict=local_dict)

        if queryables is not None and isinstance(self.expr, str):
            self.env.queryables.update(queryables)
            self._visitor = PyTablesExprVisitor(
                self.env,
                queryables=queryables,
                parser="pytables",
                engine="pytables",
                encoding=encoding,
            )
            self.terms = self.parse()

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the pretty-printed parsed tree if there is one, else the text
        # of the expression.
        if self.terms is not None:
            return pprint_thing(self.terms)
        return pprint_thing(self.expr)

    def evaluate(self):
        # [FUNCTION DOC]
        # Prunes the parsed tree to a condition and to a filter, storing and
        # returning the pair (condition, filter); either may be none. Raises a
        # value error stating the expression is not a valid condition or filter
        # when pruning fails with an attribute error (for example when there are
        # no parsed terms).
        """create and return the numexpr condition and filter"""
        try:
            self.condition = self.terms.prune(ConditionBinOp)
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] "
                "is not a valid condition"
            ) from err
        try:
            self.filter = self.terms.prune(FilterBinOp)
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] "
                "is not a valid filter"
            ) from err

        return self.condition, self.filter


class TermValue:
    # [CLASS DOC]
    # Plain holder for one query value: value (original form), converted (form
    # used in the condition text) and kind (a string naming the column kind).
    """hold a term value the we use to construct a condition/filter"""

    def __init__(self, value, converted, kind: str) -> None:
        # [FUNCTION DOC]
        # Stores the three fields; asserts that kind is a string.
        assert isinstance(kind, str), kind
        self.value = value
        self.converted = converted
        self.kind = kind

    def tostring(self, encoding) -> str:
        # [FUNCTION DOC]
        # Renders the converted value as condition text. Input encoding: encoding
        # name or none. String kind: with an encoding, the converted text
        # unquoted; otherwise wrapped in double quotes. Float kind: the repr text
        # (round-trippable). Any other kind: the plain text of the value.
        """quote the string if not encoded else encode and return"""
        if self.kind == "string":
            if encoding is not None:
                return str(self.converted)
            return f'"{self.converted}"'
        elif self.kind == "float":
            # python 2 str(float) is not always
            # round-trippable so use repr()
            return repr(self.converted)
        return str(self.converted)


def maybe_expression(s) -> bool:
    # [FUNCTION DOC]
    # Loose test for whether s could be a table where-expression. Returns False
    # for non-strings; otherwise True when the string contains any binary
    # operator symbol, unary operator symbol or a single equals sign as a
    # substring.
    """loose checking if s is a pytables-acceptable expression"""
    if not isinstance(s, str):
        return False
    operations = PyTablesExprVisitor.binary_ops + PyTablesExprVisitor.unary_ops + ("=",)

    # make sure we have an op at least
    return any(op in s for op in operations)
