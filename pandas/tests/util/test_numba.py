# [MODULE DOC]
# Test of the compute option for numba usage when numba is not installed.

import pytest

import pandas.util._test_decorators as td

from pandas import option_context


@td.skip_if_installed("numba")
def test_numba_not_installed_option_context():
    # [FUNCTION DOC]
    # Runs only when numba is not installed. Setting the option enabling numba computation inside an option context raises an import error whose message says the numba import failed.
    with pytest.raises(ImportError, match="`Import numba` failed"):
        with option_context("compute.use_numba", True):
            pass
