"""
Base test suite for extension arrays.

These tests are intended for third-party libraries to subclass to validate
that their extension arrays and dtypes satisfy the interface. Moving or
renaming the tests should not be done lightly.

Libraries are expected to implement a few pytest fixtures to provide data
for the tests. The fixtures may be located in either

* The same module as your test class.
* A ``conftest.py`` in the same directory as your test class.

The full list of fixtures may be found in the ``conftest.py`` next to this
file.

.. code-block:: python

   import pytest
   from pandas.tests.extension.base import BaseDtypeTests


   @pytest.fixture
   def dtype():
       return MyDtype()


   class TestMyDtype(BaseDtypeTests):
       pass


Your class ``TestDtype`` will inherit all the tests defined on
``BaseDtypeTests``. pytest's fixture discover will supply your ``dtype``
wherever the test requires it. You're free to implement additional tests.

"""
# [PACKAGE DOC]
# Entry point of the base extension-array test suite (see package documentation).
# Imports and re-exports every test mixin class from the sibling modules (accumulate, casting,
# constructors, dim2, dtype, getitem, groupby, index, interface, io, methods, missing, ops,
# printing, reduce, reshaping, setitem) and defines ExtensionTests, a convenience class
# combining all of them except the two-dimensional ndarray-backed test classes (the
# dim-2 compatibility mixin is included, the NDArrayBacked2D tests are not).
# Some imports carry a lint suppression because they are exported but not used here.
# No import-time side effects other than module imports.

from pandas.tests.extension.base.accumulate import BaseAccumulateTests
from pandas.tests.extension.base.casting import BaseCastingTests
from pandas.tests.extension.base.constructors import BaseConstructorsTests
from pandas.tests.extension.base.dim2 import (  # noqa: F401
    Dim2CompatTests,
    NDArrayBacked2DTests,
)
from pandas.tests.extension.base.dtype import BaseDtypeTests
from pandas.tests.extension.base.getitem import BaseGetitemTests
from pandas.tests.extension.base.groupby import BaseGroupbyTests
from pandas.tests.extension.base.index import BaseIndexTests
from pandas.tests.extension.base.interface import BaseInterfaceTests
from pandas.tests.extension.base.io import BaseParsingTests
from pandas.tests.extension.base.methods import BaseMethodsTests
from pandas.tests.extension.base.missing import BaseMissingTests
from pandas.tests.extension.base.ops import (  # noqa: F401
    BaseArithmeticOpsTests,
    BaseComparisonOpsTests,
    BaseOpsUtil,
    BaseUnaryOpsTests,
)
from pandas.tests.extension.base.printing import BasePrintingTests
from pandas.tests.extension.base.reduce import BaseReduceTests
from pandas.tests.extension.base.reshaping import BaseReshapingTests
from pandas.tests.extension.base.setitem import BaseSetitemTests


# One test class that you can inherit as an alternative to inheriting all the
# test classes above.
# Note this excludes Dim2CompatTests and NDArrayBacked2DTests.
class ExtensionTests(
    BaseAccumulateTests,
    BaseCastingTests,
    BaseConstructorsTests,
    BaseDtypeTests,
    BaseGetitemTests,
    BaseGroupbyTests,
    BaseIndexTests,
    BaseInterfaceTests,
    BaseParsingTests,
    BaseMethodsTests,
    BaseMissingTests,
    BaseArithmeticOpsTests,
    BaseComparisonOpsTests,
    BaseUnaryOpsTests,
    BasePrintingTests,
    BaseReduceTests,
    BaseReshapingTests,
    BaseSetitemTests,
    Dim2CompatTests,
):
    # [CLASS DOC]
    # Convenience test class inheriting all base test mixins (accumulate, casting, constructors,
    # dtype, getitem, groupby, index, interface, parsing, methods, missing, arithmetic ops,
    # comparison ops, unary ops, printing, reduce, reshaping, setitem and dim2 compatibility).
    # Subclassing it in a third-party test module runs the whole suite given the required fixtures.
    # Has no state or methods of its own. Excludes the NDArray-backed 2D tests.
    pass
