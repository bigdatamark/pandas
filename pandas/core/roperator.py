"""
Reversed Operations not available in the stdlib operator module.
Defining these instead of using lambdas allows us to reference them by name.
"""
# [MODULE DOC]
# Reflected (reversed-operand) arithmetic and logical operator functions.
# Each function takes (left, right) and applies the corresponding binary
# operation with the operands swapped, i.e. the right operand becomes the
# first operand of the operation. They are used by the arithmetic-method
# machinery of Series/DataFrame/arrays to implement the reflected dunder
# methods (add, subtract, multiply, divide, floor divide, modulo, divmod,
# power, and, or, xor) as named, referenceable functions rather than lambdas.
# Public contents: radd, rsub, rmul, rdiv, rtruediv, rfloordiv, rmod, rdivmod,
# rpow, rand_, ror_, rxor. Only the stdlib operator module is imported; there
# are no import-time side effects.

from __future__ import annotations

import operator


def radd(left, right):
    # [FUNCTION DOC]
    # Reflected addition. Takes two operands of any type supporting addition
    # and returns the sum with the right operand first and the left operand
    # second (matters for non-commutative additions such as string or list
    # concatenation). Errors from the underlying addition propagate.
    return right + left


def rsub(left, right):
    # [FUNCTION DOC]
    # Reflected subtraction. Returns the right operand minus the left operand.
    # Errors from the underlying subtraction propagate.
    return right - left


def rmul(left, right):
    # [FUNCTION DOC]
    # Reflected multiplication. Returns the right operand multiplied by the
    # left operand (operand order matters for sequence repetition). Errors from
    # the underlying multiplication propagate.
    return right * left


def rdiv(left, right):
    # [FUNCTION DOC]
    # Reflected true division. Returns the right operand divided by the left
    # operand. Behaves identically to rtruediv (kept as a separate name for the
    # legacy division alias). Division errors propagate.
    return right / left


def rtruediv(left, right):
    # [FUNCTION DOC]
    # Reflected true division. Returns the right operand divided by the left
    # operand using true (non floor) division. Division errors propagate.
    return right / left


def rfloordiv(left, right):
    # [FUNCTION DOC]
    # Reflected floor division. Returns the right operand floor-divided by the
    # left operand. Division errors propagate.
    return right // left


def rmod(left, right):
    # [FUNCTION DOC]
    # Reflected modulo. Returns the right operand modulo the left operand.
    # Raises a type error when the right operand is a string, because the
    # modulo operator on a string means string formatting; the message names
    # the type of the left operand and says it cannot perform the mod operation.
    # Otherwise computes the remainder with the right operand first.
    # check if right is a string as % is the string
    # formatting operation; this is a TypeError
    # otherwise perform the op
    if isinstance(right, str):
        typ = type(left).__name__
        raise TypeError(f"{typ} cannot perform the operation mod")

    return right % left


def rdivmod(left, right):
    # [FUNCTION DOC]
    # Reflected divmod. Returns the pair (quotient, remainder) of dividing the
    # right operand by the left operand, as produced by the builtin divmod.
    # Division errors propagate.
    return divmod(right, left)


def rpow(left, right):
    # [FUNCTION DOC]
    # Reflected exponentiation. Returns the right operand raised to the power
    # of the left operand. Errors from the underlying power operation propagate.
    return right**left


def rand_(left, right):
    # [FUNCTION DOC]
    # Reflected bitwise/logical and. Applies the stdlib and-operator with the
    # right operand first and the left operand second, and returns the result.
    return operator.and_(right, left)


def ror_(left, right):
    # [FUNCTION DOC]
    # Reflected bitwise/logical or. Applies the stdlib or-operator with the
    # right operand first and the left operand second, and returns the result.
    return operator.or_(right, left)


def rxor(left, right):
    # [FUNCTION DOC]
    # Reflected bitwise/logical exclusive or. Applies the stdlib xor-operator
    # with the right operand first and the left operand second, and returns the
    # result.
    return operator.xor(right, left)
