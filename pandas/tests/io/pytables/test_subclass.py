# [MODULE DOC]
# Tests that HDF5 round-trips work for user-defined subclasses of frame and series (GH 33748).
# Skipped at import if tables is missing. Contains one class with a frame test and a series test.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm

from pandas.io.pytables import (
    HDFStore,
    read_hdf,
)

pytest.importorskip("tables")


class TestHDFStoreSubclass:
    # [CLASS DOC]
    # Tests for storing subclassed DataFrame and Series objects with the HDF store.
    # GH 33748
    def test_supported_for_subclass_dataframe(self, temp_h5_path):
        # [FUNCTION DOC]
        # Writes a subclassed frame (integer pointer dtype) via the to_hdf method and again through HDFStore put, reads each back with read_hdf and asserts equality with a plain frame built from the same data.
        data = {"a": [1, 2], "b": [3, 4]}
        sdf = tm.SubclassedDataFrame(data, dtype=np.intp)

        expected = DataFrame(data, dtype=np.intp)

        sdf.to_hdf(temp_h5_path, key="df")
        result = read_hdf(temp_h5_path, "df")
        tm.assert_frame_equal(result, expected)

        with HDFStore(temp_h5_path) as store:
            store.put("df", sdf)
        result = read_hdf(temp_h5_path, "df")
        tm.assert_frame_equal(result, expected)

    def test_supported_for_subclass_series(self, temp_h5_path):
        # [FUNCTION DOC]
        # Same as the frame test but for a subclassed series: writes via to_hdf and via store put, reads back and asserts equality with a plain series.
        data = [1, 2, 3]
        sser = tm.SubclassedSeries(data, dtype=np.intp)

        expected = Series(data, dtype=np.intp)

        sser.to_hdf(temp_h5_path, key="ser")
        result = read_hdf(temp_h5_path, "ser")
        tm.assert_series_equal(result, expected)

        with HDFStore(temp_h5_path) as store:
            store.put("ser", sser)
        result = read_hdf(temp_h5_path, "ser")
        tm.assert_series_equal(result, expected)
