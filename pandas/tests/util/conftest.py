# [MODULE DOC]
# Shared pytest fixtures for the testing-utility assertion tests in this directory: boolean and tolerance parameter fixtures that the assert-equal test modules use to run each check under multiple option settings. Fixtures only; no import-time side effects.

import pytest


@pytest.fixture(params=[True, False])
def check_dtype(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false. Supplies the option controlling whether dtypes must be identical when comparing two data structures.
    """
    Fixture returning `True` or `False`, determining whether to check
    if the `dtype` is identical or not, when comparing two data structures,
    e.g. `Series`, `SparseArray` or `DataFrame`.
    """
    return request.param


@pytest.fixture(params=[True, False])
def check_exact(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false. Supplies the option controlling whether floating point numbers are compared exactly.
    """
    Fixture returning `True` or `False`, determining whether to
    compare floating point numbers exactly or not.
    """
    return request.param


@pytest.fixture(params=[True, False])
def check_index_type(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false. Supplies the option controlling whether index types must be identical.
    """
    Fixture returning `True` or `False`, determining whether to check
    if the `Index` types are identical or not.
    """
    return request.param


@pytest.fixture(params=[0.5e-3, 0.5e-5])
def rtol(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the relative tolerances 0.5e-3 and 0.5e-5.
    """
    Fixture returning 0.5e-3 or 0.5e-5. Those values are used as relative tolerance.
    """
    return request.param


@pytest.fixture(params=[True, False])
def check_categorical(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false. Supplies the option controlling whether categoricals are compared exactly.
    """
    Fixture returning `True` or `False`, determining whether to
    compare internal `Categorical` exactly or not.
    """
    return request.param
