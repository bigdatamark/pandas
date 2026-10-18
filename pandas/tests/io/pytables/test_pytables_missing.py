# [MODULE DOC]
# Test that HDF writing gives a helpful import error when the optional tables dependency is not installed.
# Contains a single test, skipped whenever tables is installed.
import pytest

import pandas.util._test_decorators as td

import pandas as pd


@td.skip_if_installed("tables")
def test_pytables_raises(temp_h5_path):
    # [FUNCTION DOC]
    # Builds a small one-column frame and asserts that writing it to the temporary HDF5 path raises an import error whose message mentions tables.
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(ImportError, match="tables"):
        df.to_hdf(temp_h5_path, key="df")
