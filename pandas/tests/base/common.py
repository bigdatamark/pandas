# [MODULE DOC]
# HELPER: shared support module for the pandas.tests.base tests. Provides one
# predicate used to skip test cases involving missing values. No import-time
# side effects.

from typing import Any

from pandas import Index


def allow_na_ops(obj: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: used by the base tests to decide whether missing-value test cases apply
    # to an object. Input: obj, an Index or Series. Returns true when the object can
    # hold missing values and is not an Index with boolean inferred type; false
    # otherwise. Reads private attribute can_hold_na of the object; no side effects.
    """Whether to skip test cases including NaN"""
    is_bool_index = isinstance(obj, Index) and obj.inferred_type == "boolean"
    return not is_bool_index and obj._can_hold_na
