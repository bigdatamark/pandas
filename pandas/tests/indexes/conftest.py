# [MODULE DOC]
# Shared pytest fixtures for the tests under the indexes test package (parameter values for
# set operation sort arguments, frequency strings and list-like boxes). Fixtures are discovered
# automatically by pytest; nothing else is exported.
import numpy as np
import pytest

from pandas import (
    Series,
    array,
)


@pytest.fixture(params=[None, False])
def sort(request):
    # [FUNCTION DOC]
    # Fixture parametrized over none and False; returns the current parameter, a valid sort argument
    # for the Index set operations (intersection, union, and others).
    # KNOWN ISSUE: the fixture docstring says sort=True is not permitted in Index set operations,
    # but tests (for example the base-class sort validation test) call these operations with sort
    # True without error, so the statement is outdated.
    """
    Valid values for the 'sort' parameter used in the Index
    setops methods (intersection, union, etc.)

    Caution:
        Don't confuse this one with the "sort" fixture used
        for concat. That one has parameters [True, False].

        We can't combine them as sort=True is not permitted
        in the Index setops methods.
    """
    return request.param


@pytest.fixture(params=["D", "3D", "-3D", "h", "2h", "-2h", "min", "2min", "s", "-3s"])
def freq_sample(request):
    # [FUNCTION DOC]
    # Fixture parametrized over ten frequency strings (daily, multiples of days, hours, minutes,
    # seconds, including negative multiples); returns the current string for building date and
    # timedelta ranges.
    """
    Valid values for 'freq' parameter used to create date_range and
    timedelta_range..
    """
    return request.param


@pytest.fixture(params=[list, tuple, np.array, array, Series])
def listlike_box(request):
    # [FUNCTION DOC]
    # Fixture parametrized over list, tuple, numpy array constructor, the pandas array factory and
    # Series; returns the current container type, to be used to wrap an indexer for searchsorted.
    """
    Types that may be passed as the indexer to searchsorted.
    """
    return request.param
