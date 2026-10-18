"""
Engine classes for :func:`~pandas.eval`
"""

# [MODULE DOC]
# Evaluation engines for the expression evaluator. Public contents: the
# abstract engine base class, the numexpr-backed engine, the pure-Python
# engine, and the ENGINES registry mapping engine names ("numexpr" and
# "python") to classes, which the eval module uses to pick an engine.
# Relies on the align module for operand alignment and result rebuilding and
# on the ops module for the set of names reserved by numexpr. No import-time
# side effects beyond importing those modules.

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from pandas.errors import NumExprClobberingError

from pandas.core.computation.align import (
    align_terms,
    reconstruct_object,
)
from pandas.core.computation.ops import (
    MATHOPS,
    REDUCTIONS,
)

from pandas.io.formats import printing

if TYPE_CHECKING:
    from pandas.core.computation.expr import Expr

_ne_builtins = frozenset(MATHOPS + REDUCTIONS)


def _check_ne_builtin_clash(expr: Expr) -> None:
    # [FUNCTION DOC]
    # HELPER: guard used by the numexpr engine before evaluation.
    # Input expr: a parsed expression exposing the set of names it references.
    # Raises the numexpr-clobbering error when any referenced name equals one of
    # numexpr's reserved math or reduction function names (message lists the
    # clashing names and the expression). Returns nothing otherwise.
    """
    Attempt to prevent foot-shooting in a helpful way.

    Parameters
    ----------
    expr : Expr
        Terms can contain
    """
    names = expr.names
    overlap = names & _ne_builtins

    if overlap:
        s = ", ".join([repr(x) for x in overlap])
        raise NumExprClobberingError(
            f'Variables in expression "{expr}" overlap with builtins: ({s})'
        )


class AbstractEngine(metaclass=abc.ABCMeta):
    # [CLASS DOC]
    # Abstract base for evaluation engines. Holds the parsed expression and
    # the alignment results (aligned axes, result container type and result
    # name; all unset until alignment runs). Subclasses must implement the
    # private evaluation step; the class attribute has_neg_frac tells callers
    # whether the engine supports negative fractional powers (false here).
    # Key methods: convert (expression to string), evaluate (align then
    # evaluate then rebuild the result).
    """Object serving as a base class for all engines."""

    has_neg_frac = False

    def __init__(self, expr) -> None:
        # [FUNCTION DOC]
        # Stores expr (the parsed expression object) and initialises the
        # aligned axes, result type and result name to unset.
        self.expr = expr
        self.aligned_axes = None
        self.result_type = None
        self.result_name = None

    def convert(self) -> str:
        # [FUNCTION DOC]
        # Returns the string form of the expression, produced by the pretty
        # printer. Subclasses may override to produce engine-specific text.
        """
        Convert an expression for evaluation.

        Defaults to return the expression as a string.
        """
        return printing.pprint_thing(self.expr)

    def evaluate(self) -> object:
        # [FUNCTION DOC]
        # Runs alignment (only if not already aligned), calls the subclass
        # evaluation step, and rebuilds the raw result into the aligned
        # container type using the expression's return type and the result
        # name. Returns the rebuilt result. Side effect: stores alignment
        # results on the engine and mutates the expression's terms.
        """
        Run the engine on the expression.

        This method performs alignment which is necessary no matter what engine
        is being used, thus its implementation is in the base class.

        Returns
        -------
        object
            The result of the passed expression.
        """
        if not self._is_aligned:
            self.result_type, self.aligned_axes, self.result_name = align_terms(
                self.expr.terms
            )

        # make sure no names in resolvers and locals/globals clash
        res = self._evaluate()
        return reconstruct_object(
            self.result_type,
            res,
            self.aligned_axes,
            self.expr.terms.return_type,
            self.result_name,
        )

    @property
    def _is_aligned(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property telling evaluate whether alignment already ran.
        # True when both aligned axes and result type are set, else False.
        return self.aligned_axes is not None and self.result_type is not None

    @abc.abstractmethod
    def _evaluate(self):
        # [FUNCTION DOC]
        # HELPER: abstract hook that subclasses implement to compute the raw
        # (unwrapped) result of the aligned expression.
        # KNOWN ISSUE: the docstring documents an env parameter, but the method
        # takes no such argument; the environment is read from the expression.
        """
        Return an evaluated expression.

        Parameters
        ----------
        env : Scope
            The local and global environment in which to evaluate an
            expression.

        Notes
        -----
        Must be implemented by subclasses.
        """


class NumExprEngine(AbstractEngine):
    # [CLASS DOC]
    # Engine that delegates evaluation to the optional numexpr library
    # (subclass of the abstract engine). Reports support for negative
    # fractional powers.
    """NumExpr engine class"""

    has_neg_frac = True

    def _evaluate(self):
        # [FUNCTION DOC]
        # HELPER: numexpr evaluation step. Imports numexpr lazily, converts the
        # expression to a string, checks that no referenced variable clashes
        # with numexpr builtins (raising the clobbering error if so) and
        # evaluates with the expression environment's full scope as the
        # variable dictionary. Returns the raw numexpr result.
        import numexpr as ne

        # convert the expression to a valid numexpr expression
        s = self.convert()

        env = self.expr.env
        scope = env.full_scope
        _check_ne_builtin_clash(self.expr)
        return ne.evaluate(s, local_dict=scope)


class PythonEngine(AbstractEngine):
    # [CLASS DOC]
    # Engine that evaluates by calling the expression tree directly in
    # Python, with no numexpr involvement; mostly used for testing. Does not
    # support negative fractional powers.
    """
    Evaluate an expression in Python space.

    Mostly for testing purposes.
    """

    has_neg_frac = False

    def evaluate(self):
        # [FUNCTION DOC]
        # Overrides the base method: skips alignment and result rebuilding and
        # simply returns the value of calling the expression.
        return self.expr()

    def _evaluate(self) -> None:
        # [FUNCTION DOC]
        # HELPER: required abstract hook, intentionally does nothing because
        # the evaluate override never calls it.
        pass


ENGINES: dict[str, type[AbstractEngine]] = {
    "numexpr": NumExprEngine,
    "python": PythonEngine,
}
