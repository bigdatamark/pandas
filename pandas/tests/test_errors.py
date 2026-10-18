# [MODULE DOC]
# Tests for the public errors and warnings module: that listed exception and
# warning classes are importable and instantiable, the out-of-bounds datetime
# error, the undefined-variable error message, the abstract-method error message,
# and the hierarchy of version-specific change warnings.
import warnings

import pytest

from pandas.errors import (
    AbstractMethodError,
    Pandas4Warning,
    Pandas5Warning,
    PandasChangeWarning,
    PandasDeprecationWarning,
    PandasPendingDeprecationWarning,
    UndefinedVariableError,
)

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "exc",
    [
        "AttributeConflictWarning",
        "CSSWarning",
        "CategoricalConversionWarning",
        "ClosedFileError",
        "DataError",
        "DatabaseError",
        "DtypeWarning",
        "EmptyDataError",
        "IncompatibilityWarning",
        "IndexingError",
        "InvalidColumnName",
        "InvalidComparison",
        "InvalidVersion",
        "LossySetitemError",
        "MergeError",
        "NoBufferPresent",
        "NumExprClobberingError",
        "NumbaUtilError",
        "OptionError",
        "OutOfBoundsDatetime",
        "ParserError",
        "ParserWarning",
        "PerformanceWarning",
        "PossibleDataLossError",
        "PossiblePrecisionLoss",
        "PyperclipException",
        "SpecificationError",
        "UnsortedIndexError",
        "UnsupportedFunctionCall",
        "ValueLabelTypeMismatch",
    ],
)
def test_exception_importable(exc):
    # [FUNCTION DOC]
    # Parametrized over the names of about thirty exception and warning classes.
    # Verifies each name is exposed by the errors module, is not None, and can be raised
    # with no arguments, producing an empty message.
    from pandas import errors

    err = getattr(errors, exc)
    assert err is not None

    # check that we can raise on them

    msg = "^$"

    with pytest.raises(err, match=msg):
        raise err()


def test_catch_oob():
    # [FUNCTION DOC]
    # Verifies converting a timestamp far in the past to nanosecond unit raises the
    # out-of-bounds datetime error with the expected overflow message.
    from pandas import errors

    msg = "Cannot cast 1500-01-01 00:00:00 to unit='ns' without overflow"
    with pytest.raises(errors.OutOfBoundsDatetime, match=msg):
        pd.Timestamp("15000101").as_unit("ns")


@pytest.mark.parametrize("is_local", [True, False])
def test_catch_undefined_variable_error(is_local):
    # [FUNCTION DOC]
    # Parametrized on a flag for local versus global scope. Verifies the undefined
    # variable error built from a name and the flag produces a message saying either
    # "local variable ... is not defined" or "name ... is not defined".
    variable_name = "x"
    if is_local:
        msg = f"local variable '{variable_name}' is not defined"
    else:
        msg = f"name '{variable_name}' is not defined"

    with pytest.raises(UndefinedVariableError, match=msg):
        raise UndefinedVariableError(variable_name, is_local)


class Foo:
    # [CLASS DOC]
    # HELPER: sample class used by the abstract-method error test; each member raises
    # the abstract-method error for a different kind of member (class method, property,
    # ordinary method).
    @classmethod
    def classmethod(cls):
        # [FUNCTION DOC]
        # HELPER: class method that raises the abstract-method error, reporting the kind as class method.
        raise AbstractMethodError(cls, methodtype="classmethod")

    @property
    def property(self):
        # [FUNCTION DOC]
        # HELPER: property that raises the abstract-method error, reporting the kind as property.
        raise AbstractMethodError(self, methodtype="property")

    def method(self):
        # [FUNCTION DOC]
        # HELPER: ordinary method that raises the abstract-method error with the default kind (method).
        raise AbstractMethodError(self)


def test_AbstractMethodError_classmethod():
    # [FUNCTION DOC]
    # Verifies the abstract-method error message names the member kind (class method,
    # property, method) and the concrete class name, for each of the three kinds in the
    # sample class.
    # KNOWN ISSUE: the test name mentions only the class method case but the body covers all three kinds.
    xpr = "This classmethod must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo.classmethod()

    xpr = "This property must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().property

    xpr = "This method must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().method()


@pytest.mark.parametrize(
    "warn_category, catch_category",
    [
        (Pandas4Warning, PandasChangeWarning),
        (Pandas4Warning, PandasDeprecationWarning),
        (Pandas5Warning, PandasChangeWarning),
        (Pandas5Warning, PandasPendingDeprecationWarning),
    ],
)
def test_pandas_warnings(warn_category, catch_category):
    # [FUNCTION DOC]
    # Parametrized over (emitted warning category, caught category) pairs. Verifies that
    # the major-version warning classes derive from the broader change, deprecation and
    # pending-deprecation categories, so that emitting one is recorded as the other.
    # https://github.com/pandas-dev/pandas/pull/61468
    with tm.assert_produces_warning(catch_category):
        warnings.warn("test", category=warn_category)


@pytest.mark.parametrize(
    "warn_category, filter_category",
    [
        (Pandas4Warning, PandasChangeWarning),
        (Pandas4Warning, PandasDeprecationWarning),
        (Pandas5Warning, PandasChangeWarning),
        (Pandas5Warning, PandasPendingDeprecationWarning),
    ],
)
def test_pandas_warnings_filter(warn_category, filter_category):
    # [FUNCTION DOC]
    # Parametrized over (emitted category, filtered category) pairs. Verifies a warning
    # filter that ignores a broader base category suppresses the derived version warning,
    # so no warning is produced.
    # https://github.com/pandas-dev/pandas/pull/61468
    # Ensure users can suppress warnings.
    with tm.assert_produces_warning(None), warnings.catch_warnings():
        warnings.filterwarnings(category=filter_category, action="ignore")
        warnings.warn("test", category=warn_category)
