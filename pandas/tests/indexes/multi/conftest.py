# [MODULE DOC]
# Shared fixtures for the MultiIndex tests in this package. Defines one fixture,
# a general-purpose two-level index. Fixtures here are available to all test
# modules in the directory without importing.

import numpy as np
import pytest

from pandas import (
    Index,
    MultiIndex,
)


# Note: identical the "multi" entry in the top-level "index" fixture
@pytest.fixture
def idx():
    # [FUNCTION DOC]
    # HELPER: fixture returning a small named two-level MultiIndex with six entries:
    # first level values foo, bar, baz, qux; second level one, two; codes pairing the
    # levels as (foo,one), (foo,two), (bar,one), (baz,two), (qux,one), (qux,two);
    # level names first and second; integrity verification disabled. Identical to the
    # multi entry of the top-level index fixture (per the in-source note). A fresh
    # index is built per test.
    # a MultiIndex used to test the general functionality of the
    # general functionality of this object
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ["first", "second"]
    mi = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )
    return mi
