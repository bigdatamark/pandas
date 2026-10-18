# [MODULE DOC]
# Shared pytest fixtures for the JSON tests in this directory.
# Defines a single fixture, orient, available to all sibling test modules.

import pytest


@pytest.fixture(params=["split", "records", "index", "columns", "values"])
def orient(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the five non-table JSON orientations: split,
    # records, index, columns and values. Takes the pytest request object and
    # returns the current parameter string. The table orientation is deliberately
    # excluded.
    """
    Fixture for orients excluding the table format.
    """
    return request.param
