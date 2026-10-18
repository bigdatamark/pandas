# [MODULE DOC]
# Shared pytest fixtures for the pytables test modules.
# Import-time side effects: skips the whole module if the tables library is missing, and limits the numexpr, blosc and general thread counts of tables to one so concurrent tests do not share files.
# Fixtures: a temporary HDF5 path and an open HDFStore on it.
import uuid

import pytest

from pandas.io.pytables import HDFStore

tables = pytest.importorskip("tables")
# set these parameters so we don't have file sharing
tables.parameters.MAX_NUMEXPR_THREADS = 1
tables.parameters.MAX_BLOSC_THREADS = 1
tables.parameters.MAX_THREADS = 1


@pytest.fixture
def temp_h5_path(tmp_path):
    # [FUNCTION DOC]
    # Fixture providing a path to a new, empty HDF5 file inside the pytest temporary directory, named with a random UUID and created via touching.
    # Returns the path object. Requires the tmp_path fixture.
    """Fixture for HDF5 path"""
    file_path = tmp_path / f"{uuid.uuid4()}.h5"
    file_path.touch()
    return file_path


@pytest.fixture
def temp_hdfstore(temp_h5_path):
    # [FUNCTION DOC]
    # Fixture that opens an HDFStore on the temporary HDF5 path in append mode, yields it to the test, and closes it on exit through the context manager. Depends on the temporary path fixture.
    with HDFStore(temp_h5_path, mode="a") as store:
        yield store
