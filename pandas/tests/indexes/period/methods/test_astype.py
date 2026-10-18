# [MODULE DOC]
# Tests for PeriodIndex.astype: invalid targets, object, integer, string, category,
# boolean fallback, unsigned errors and conversion to timezone-aware datetimes.
import numpy as np
import pytest

from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    NaT,
    Period,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestPeriodIndexAsType:
    # [CLASS DOC]
    # Groups astype tests on PeriodIndex.
    @pytest.mark.parametrize("dtype", [float, "timedelta64", "timedelta64[ns]"])
    def test_astype_raises(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over float and timedelta dtypes. Raises a type error saying the
        # PeriodIndex cannot be cast (GH#13149, GH#13209).
        # GH#13149, GH#13209
        idx = PeriodIndex(["2016-05-16", "NaT", NaT, np.nan], freq="D")
        msg = "Cannot cast PeriodIndex to dtype"
        with pytest.raises(TypeError, match=msg):
            idx.astype(dtype)

    def test_astype_conversion(self, using_infer_string):
        # [FUNCTION DOC]
        # Casting an index with missing values to object gives Period objects (missing
        # as Period NaT), to 64-bit integer gives ordinals with the minimum integer as
        # missing, to string gives text per element (missing as none under
        # infer-string); casting yearly periods to an integer string alias returns the
        # ordinals index with its name.
        # GH#13149, GH#13209
        idx = PeriodIndex(["2016-05-16", "NaT", NaT, np.nan], freq="D", name="idx")

        result = idx.astype(object)
        expected = Index(
            [Period("2016-05-16", freq="D")] + [Period(NaT, freq="D")] * 3,
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(np.int64)
        expected = Index(
            [16937] + [-9223372036854775808] * 3, dtype=np.int64, name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(str)
        if using_infer_string:
            expected = Index(
                [str(x) if x is not NaT else None for x in idx], name="idx", dtype="str"
            )
        else:
            expected = Index([str(x) for x in idx], name="idx", dtype=object)
        tm.assert_index_equal(result, expected)

        idx = period_range("1990", "2009", freq="Y", name="idx")
        result = idx.astype("i8")
        tm.assert_index_equal(result, Index(idx.asi8, name="idx"))
        tm.assert_numpy_array_equal(result.values, idx.asi8)

    def test_astype_uint(self):
        # [FUNCTION DOC]
        # Casting to 64- or 32-bit unsigned integers raises a type error advising to
        # cast to signed 64-bit first.
        arr = period_range("2000", periods=2, name="idx")

        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint64")
        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint32")

    def test_astype_object(self):
        # [FUNCTION DOC]
        # Empty, monthly with NaT and daily with NaT indexes cast to object give arrays
        # of Period objects; the matplotlib representation helper gives the same array.
        idx = PeriodIndex([], freq="M")

        exp = np.array([], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        idx = PeriodIndex(["2011-01", NaT], freq="M")

        exp = np.array([Period("2011-01", freq="M"), NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        exp = np.array([Period("2011-01-01", freq="D"), NaT], dtype=object)
        idx = PeriodIndex(["2011-01-01", NaT], freq="D")
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

    # TODO: de-duplicate this version (from test_ops) with the one above
    # (from test_period)
    def test_astype_object2(self):
        # [FUNCTION DOC]
        # Casting a monthly then a daily index with NaT to object gives an object Index
        # of periods with preserved name; NaT elements are the NaT singleton; tolist
        # matches.
        # KNOWN ISSUE: a TODO says this duplicates the earlier object test (from another test module).
        idx = period_range(start="2013-01-01", periods=4, freq="M", name="idx")
        expected_list = [
            Period("2013-01-31", freq="M"),
            Period("2013-02-28", freq="M"),
            Period("2013-03-31", freq="M"),
            Period("2013-04-30", freq="M"),
        ]
        expected = Index(expected_list, dtype=object, name="idx")
        result = idx.astype(object)
        assert isinstance(result, Index)
        assert result.dtype == object
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert idx.tolist() == expected_list

        idx = PeriodIndex(
            ["2013-01-01", "2013-01-02", "NaT", "2013-01-04"], freq="D", name="idx"
        )
        expected_list = [
            Period("2013-01-01", freq="D"),
            Period("2013-01-02", freq="D"),
            Period("NaT", freq="D"),
            Period("2013-01-04", freq="D"),
        ]
        expected = Index(expected_list, dtype=object, name="idx")
        result = idx.astype(object)
        assert isinstance(result, Index)
        assert result.dtype == object
        tm.assert_index_equal(result, expected)
        for i in [0, 1, 3]:
            assert result[i] == expected[i]
        assert result[2] is NaT
        assert result.name == expected.name

        result_list = idx.tolist()
        for i in [0, 1, 3]:
            assert result_list[i] == expected_list[i]
        assert result_list[2] is NaT

    def test_astype_category(self):
        # [FUNCTION DOC]
        # Casting to category yields a CategoricalIndex of periods with the name; the
        # underlying array cast gives the equal categorical.
        obj = period_range("2000", periods=2, name="idx")
        result = obj.astype("category")
        expected = CategoricalIndex(
            [Period("2000-01-01", freq="D"), Period("2000-01-02", freq="D")], name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = obj._data.astype("category")
        expected = expected.values
        tm.assert_categorical_equal(result, expected)

    def test_astype_array_fallback(self):
        # [FUNCTION DOC]
        # Casting to boolean falls back to array behavior: all true for non-missing
        # periods, on both the index and its backing array.
        obj = period_range("2000", periods=2, name="idx")
        result = obj.astype(bool)
        expected = Index(np.array([True, True]), name="idx")
        tm.assert_index_equal(result, expected)

        result = obj._data.astype(bool)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_period_astype_to_timestamp(self, unit):
        # [FUNCTION DOC]
        # Parametrized over datetime resolution unit. Casting to a timezone-aware
        # datetime dtype converts to start-of-period timestamps localized to the zone
        # with the same frequency (GH#55958).
        # GH#55958
        pi = PeriodIndex(["2011-01", "2011-02", "2011-03"], freq="M")

        exp = DatetimeIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01"], tz="US/Eastern"
        ).as_unit(unit)
        res = pi.astype(f"datetime64[{unit}, US/Eastern]")
        tm.assert_index_equal(res, exp)
        assert res.freq == exp.freq
