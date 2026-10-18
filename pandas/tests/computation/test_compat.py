# [MODULE DOC]
# Tests for compatibility of the expression evaluation machinery with the
# installed optional numexpr library and for basic evaluation on every engine /
# parser combination. No import-time side effects.

import pytest

from pandas.compat._optional import VERSIONS

import pandas as pd
from pandas.core.computation import expr
from pandas.core.computation.engines import ENGINES
from pandas.util.version import Version


def test_compat():
    # [FUNCTION DOC]
    # Skips when numexpr is not installed. Verifies the module flag recording whether
    # numexpr is usable is false when the installed numexpr is older than the minimum
    # version required by pandas, and true otherwise.
    # test we have compat with our version of numexpr

    from pandas.core.computation.check import NUMEXPR_INSTALLED

    ne = pytest.importorskip("numexpr")

    ver = ne.__version__
    if Version(ver) < Version(VERSIONS["numexpr"]):
        assert not NUMEXPR_INSTALLED
    else:
        assert NUMEXPR_INSTALLED


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("parser", expr.PARSERS)
def test_invalid_numexpr_version(engine, parser):
    # [FUNCTION DOC]
    # Parametrized over every engine and every parser. Skips the numexpr engine
    # when numexpr is not installed. Verifies evaluating the string expression adding
    # two local integer variables with the engine and parser gives 3.
    # KNOWN ISSUE: the name suggests it tests behavior for an invalid numexpr
    # version, but the body only checks that a simple addition evaluates correctly.
    if engine == "numexpr":
        pytest.importorskip("numexpr")
    a, b = 1, 2  # noqa: F841
    res = pd.eval("a + b", engine=engine, parser=parser)
    assert res == 3
