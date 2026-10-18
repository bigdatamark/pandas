# [MODULE DOC]
# Tests for converting DataFrame and Series to xarray objects. The whole module is
# skipped
# if xarray is not installed (checked at import); for xarray older than 2025.1.0 it also
# ignores warnings about non-nanosecond precision conversion through a module-wide
# marker.
# Uses the index_flat fixture for index type coverage.
import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.util.version import Version

xarray = pytest.importorskip("xarray")

if xarray is not None and Version(xarray.__version__) < Version("2025.1.0"):
    pytestmark = pytest.mark.filterwarnings(
        "ignore:Converting non-nanosecond precision:UserWarning"
    )


class TestDataFrameToXArray:
    # [CLASS DOC]
    # Groups DataFrame to xarray conversion tests.
    @pytest.fixture
    def df(self):
        # [FUNCTION DOC]
        # Fixture providing a four-row frame with string, integer, unsigned byte, float,
        # bool, categorical, naive datetime and tz-aware datetime columns.
        return DataFrame(
            {
                "a": list("abcd"),
                "b": list(range(1, 5)),
                "c": np.arange(3, 7).astype("u1"),
                "d": np.arange(4.0, 8.0, dtype="float64"),
                "e": [True, False, True, False],
                "f": Categorical(list("abcd")),
                "g": date_range("20130101", periods=4),
                "h": date_range("20130101", periods=4, tz="US/Eastern"),
            }
        )

    def test_to_xarray_index_types(self, index_flat, df, request):
        # [FUNCTION DOC]
        # Verifies converting a frame with each flat index type yields a dataset with
        # the index as the single coordinate and eight data variables,
        # and that converting back gives the original frame (column name lost). Skipped
        # for empty indexes and for xarray older than 2025.9.0.
        # The request fixture is unused. Uses the first four index entries, named foo,
        # with column name bar.
        index = index_flat
        # MultiIndex is tested in test_to_xarray_with_multiindex
        if len(index) == 0:
            pytest.skip("Test doesn't make sense for empty index")
        if Version(xarray.__version__) < Version("2025.9.0"):
            pytest.skip("Xarray bug https://github.com/pydata/xarray/issues/9661")

        df.index = index[:4]
        df.index.name = "foo"
        df.columns.name = "bar"
        result = df.to_xarray()
        assert result.sizes["foo"] == 4
        assert len(result.coords) == 1
        assert len(result.data_vars) == 8
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, xarray.Dataset)

        # idempotency
        # datetimes w/tz are preserved
        # column names are lost
        expected = df.copy()
        expected.columns.name = None
        tm.assert_frame_equal(result.to_dataframe(), expected)

    def test_to_xarray_empty(self, df):
        # [FUNCTION DOC]
        # Verifies converting an empty (zero-row) slice of the frame gives a dataset
        # whose dimension has size zero.
        df.index.name = "foo"
        result = df[0:0].to_xarray()
        assert result.sizes["foo"] == 0
        assert isinstance(result, xarray.Dataset)

    def test_to_xarray_with_multiindex(self, df, using_infer_string):
        # [FUNCTION DOC]
        # Verifies converting a frame with a two-level multi-index yields two
        # coordinates and eight variables, and
        # round-trips; the categorical column returns as object (or string under
        # inferred strings) and, for xarray older than 2025.1.0, datetime columns are
        # cast to nanosecond precision.
        # MultiIndex
        df.index = MultiIndex.from_product([["a"], range(4)], names=["one", "two"])
        result = df.to_xarray()
        assert result.sizes["one"] == 1
        assert result.sizes["two"] == 4
        assert len(result.coords) == 2
        assert len(result.data_vars) == 8
        tm.assert_almost_equal(list(result.coords.keys()), ["one", "two"])
        assert isinstance(result, xarray.Dataset)

        result = result.to_dataframe()
        expected = df.copy()
        expected["f"] = expected["f"].astype(
            object if not using_infer_string else "str"
        )
        if Version(xarray.__version__) < Version("2025.1.0"):
            expected["g"] = expected["g"].astype("M8[ns]")
            expected["h"] = expected["h"].astype("M8[ns, US/Eastern]")
        expected.columns.name = None
        tm.assert_frame_equal(result, expected)


class TestSeriesToXArray:
    # [CLASS DOC]
    # Groups Series to xarray conversion tests.
    def test_to_xarray_index_types(self, index_flat, request):
        # [FUNCTION DOC]
        # Verifies converting an integer series for each flat index type gives a data
        # array of the same length with the
        # index as sole coordinate and round-trips to the same series. The request
        # fixture is unused.
        # MultiIndex is tested in test_to_xarray_with_multiindex
        index = index_flat

        ser = Series(range(len(index)), index=index, dtype="int64")
        ser.index.name = "foo"
        result = ser.to_xarray()
        repr(result)
        assert len(result) == len(index)
        assert len(result.coords) == 1
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, xarray.DataArray)

        # idempotency
        tm.assert_series_equal(result.to_series(), ser)

    def test_to_xarray_empty(self):
        # [FUNCTION DOC]
        # Verifies converting an empty object series with a named index gives an empty
        # data array keeping one coordinate.
        ser = Series([], dtype=object)
        ser.index.name = "foo"
        result = ser.to_xarray()
        assert len(result) == 0
        assert len(result.coords) == 1
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, xarray.DataArray)

    def test_to_xarray_with_multiindex(self):
        # [FUNCTION DOC]
        # Verifies converting a series with a two-level multi-index gives a data array
        # with the two level coordinates; converting back reproduces the series.
        mi = MultiIndex.from_product([["a", "b"], range(3)], names=["one", "two"])
        ser = Series(range(6), dtype="int64", index=mi)
        result = ser.to_xarray()
        assert len(result) == 2
        tm.assert_almost_equal(list(result.coords.keys()), ["one", "two"])
        assert isinstance(result, xarray.DataArray)
        res = result.to_series()
        tm.assert_series_equal(res, ser)
