# [MODULE DOC]
# Tests for constructing DatetimeArray via its private sequence constructor, the sequence conversion path,
# and conversion from pyarrow timestamp arrays through the datetime tz dtype. Defines module constants of
# extreme and safe values used to parametrize the arrow tests. Uses the unit fixture from the shared conftest;
# the arrow tests skip when pyarrow is missing. No import-time side effects beyond defining constants.

import numpy as np
import pytest

from pandas._libs import iNaT

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestDatetimeArrayConstructor:
    # [CLASS DOC]
    # Groups tests of the datetime array sequence constructor. Stateless.
    def test_from_sequence_invalid_type(self):
        # [FUNCTION DOC]
        # A multi-index as input raises a type error saying a DatetimeArray cannot be created from it.
        mi = pd.MultiIndex.from_product([np.arange(5), np.arange(5)])
        with pytest.raises(TypeError, match="Cannot create a DatetimeArray"):
            DatetimeArray._from_sequence(mi, dtype="M8[ns]")

    @pytest.mark.parametrize(
        "meth",
        [
            DatetimeArray._from_sequence,
            pd.to_datetime,
            pd.DatetimeIndex,
        ],
    )
    def test_mixing_naive_tzaware_raises(self, meth):
        # [FUNCTION DOC]
        # Parametrized over the sequence constructor, to_datetime and the index constructor. An object array mixing
        # tz-naive and tz-aware timestamps, in either order, raises a value error (cannot mix, or tz-aware not
        # convertible unless utc).
        # GH#24569
        arr = np.array([pd.Timestamp("2000"), pd.Timestamp("2000", tz="CET")])

        msg = "|".join(
            [
                "Cannot mix tz-aware with tz-naive values",
                "Tz-aware datetime.datetime cannot be converted "
                "to datetime64 unless utc=True",
            ]
        )

        for obj in [arr, arr[::-1]]:
            # check that we raise regardless of whether naive is found
            #  before aware or vice-versa
            with pytest.raises(ValueError, match=msg):
                meth(obj)

    def test_from_pandas_array(self):
        # [FUNCTION DOC]
        # A nullable integer array of hourly nanosecond offsets converts to the same datetime array as an hourly date range (frequency inferred).
        arr = pd.array(np.arange(5, dtype=np.int64)) * 3600 * 10**9

        result = DatetimeArray._from_sequence(arr, dtype="M8[ns]")._with_freq("infer")

        expected = pd.date_range("1970-01-01", periods=5, freq="h", unit="ns")._data
        tm.assert_datetime_array_equal(result, expected)

    def test_bool_dtype_raises(self):
        # [FUNCTION DOC]
        # A boolean array raises a type error (cannot convert to datetime64 nanoseconds) from the array constructor, the index constructor and to_datetime.
        arr = np.array([1, 2, 3], dtype="bool")

        msg = r"dtype bool cannot be converted to datetime64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            DatetimeArray._from_sequence(arr, dtype="M8[ns]")

        with pytest.raises(TypeError, match=msg):
            pd.DatetimeIndex(arr)

        with pytest.raises(TypeError, match=msg):
            pd.to_datetime(arr)

    def test_copy(self):
        # [FUNCTION DOC]
        # With copy disabled the array wraps the same underlying numpy array; with copy enabled it holds a different one.
        data = np.array([1, 2, 3], dtype="M8[ns]")
        arr = DatetimeArray._from_sequence(data, dtype=data.dtype, copy=False)
        assert arr._ndarray is data

        arr = DatetimeArray._from_sequence(data, dtype=data.dtype, copy=True)
        assert arr._ndarray is not data

    def test_numpy_datetime_unit(self, unit):
        # [FUNCTION DOC]
        # Uses the unit fixture. A numpy datetime array keeps its unit in the resulting array and in its scalar elements.
        data = np.array([1, 2, 3], dtype=f"M8[{unit}]")
        arr = DatetimeArray._from_sequence(data)
        assert arr.unit == unit
        assert arr[0].unit == unit


class TestSequenceToDT64NS:
    # [CLASS DOC]
    # Groups tests of the sequence conversion path of the datetime array constructor. Stateless.
    def test_tz_dtype_mismatch_raises(self):
        # [FUNCTION DOC]
        # Constructing from a tz-aware array with a different timezone dtype raises a type error that the data is already tz-aware.
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(TypeError, match="data is already tz-aware"):
            DatetimeArray._from_sequence(arr, dtype=DatetimeTZDtype(tz="UTC"))

    def test_tz_dtype_matches(self):
        # [FUNCTION DOC]
        # Constructing from a tz-aware array with the same dtype returns an equal array.
        dtype = DatetimeTZDtype(tz="US/Central")
        arr = DatetimeArray._from_sequence(["2000"], dtype=dtype)
        result = DatetimeArray._from_sequence(arr, dtype=dtype)
        tm.assert_equal(arr, result)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_2d(self, order):
        # [FUNCTION DOC]
        # Parametrized over C and Fortran memory order. A 3 by 2 object array of tz-aware timestamps converts to a
        # two-dimensional array equal to converting the flattened data and reshaping.
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        arr = np.array(dti, dtype=object).reshape(3, 2)
        if order == "F":
            arr = arr.T

        res = DatetimeArray._from_sequence(arr, dtype=dti.dtype)
        expected = DatetimeArray._from_sequence(arr.ravel(), dtype=dti.dtype).reshape(
            arr.shape
        )
        tm.assert_datetime_array_equal(res, expected)


# ----------------------------------------------------------------------------
# Arrow interaction


EXTREME_VALUES = [0, 123456789, None, iNaT, 2**63 - 1, -(2**63) + 1]
FINE_TO_COARSE_SAFE = [123_000_000_000, None, -123_000_000_000]
COARSE_TO_FINE_SAFE = [123, None, -123]


@pytest.mark.parametrize(
    ("pa_unit", "pd_unit", "pa_tz", "pd_tz", "data"),
    [
        ("s", "s", "UTC", "UTC", EXTREME_VALUES),
        ("ms", "ms", "UTC", "Europe/Berlin", EXTREME_VALUES),
        ("us", "us", "US/Eastern", "UTC", EXTREME_VALUES),
        ("ns", "ns", "US/Central", "Asia/Kolkata", EXTREME_VALUES),
        ("ns", "s", "UTC", "UTC", FINE_TO_COARSE_SAFE),
        ("us", "ms", "UTC", "Europe/Berlin", FINE_TO_COARSE_SAFE),
        ("ms", "us", "US/Eastern", "UTC", COARSE_TO_FINE_SAFE),
        ("s", "ns", "US/Central", "Asia/Kolkata", COARSE_TO_FINE_SAFE),
    ],
)
def test_from_arrow_with_different_units_and_timezones_with(
    pa_unit, pd_unit, pa_tz, pd_tz, data
):
    # [FUNCTION DOC]
    # Parametrized over arrow unit, pandas unit, arrow tz, pandas tz and data (extreme values for the same unit;
    # safe values when changing between coarse and fine units). Converting a pyarrow timestamp array, also
    # chunked, through the tz dtype arrow hook gives the array expected from casting the values to the target
    # dtype. Skips without pyarrow.
    pa = pytest.importorskip("pyarrow")

    pa_type = pa.timestamp(pa_unit, tz=pa_tz)
    arr = pa.array(data, type=pa_type)
    dtype = DatetimeTZDtype(unit=pd_unit, tz=pd_tz)

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(data, dtype=f"M8[{pa_unit}, UTC]").astype(
        dtype, copy=False
    )
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    ("unit", "tz"),
    [
        ("s", "UTC"),
        ("ms", "Europe/Berlin"),
        ("us", "US/Eastern"),
        ("ns", "Asia/Kolkata"),
        ("ns", "UTC"),
    ],
)
def test_from_arrow_from_empty(unit, tz):
    # [FUNCTION DOC]
    # Parametrized over unit and timezone. An empty non-timestamp pyarrow array (and chunked version) converts
    # through the tz dtype hook to an empty tz-aware array of that unit. Skips without pyarrow.
    pa = pytest.importorskip("pyarrow")

    data = []
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit=unit, tz=tz)

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(
        np.array(data, dtype=f"datetime64[{unit}]"), dtype=np.dtype(f"M8[{unit}]")
    )
    expected = expected.tz_localize(tz=tz)
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


def test_from_arrow_from_integers():
    # [FUNCTION DOC]
    # Integer pyarrow data (including missing, extremes and the not-a-time sentinel) converts through the tz dtype
    # hook for nanosecond UTC as if it were nanosecond datetimes localized to UTC; chunked also. Skips without pyarrow.
    pa = pytest.importorskip("pyarrow")

    data = [0, 123456789, None, 2**63 - 1, iNaT, -123456789]
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit="ns", tz="UTC")

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(
        np.array(data, dtype="datetime64[ns]"), dtype=np.dtype("M8[ns]")
    )
    expected = expected.tz_localize("UTC")
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)
