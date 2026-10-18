# [MODULE DOC]
# Tests for the method that fills missing values of one frame with values from another,
# aligned on index and columns (combine_first). Covers mixed dtypes, disjoint and
# overlapping frames, empty frames, datetime, tz-aware, timedelta, period, nullable and
# integer data, multi-level indexes with missing labels, non-unique columns and column
# order. Uses fixtures from conftest (float frame, unit, nulls, nullable string dtype)
# and the testing helpers module. No import-time side effects.

from datetime import datetime

import numpy as np
import pytest

from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestDataFrameCombineFirst:
    # [CLASS DOC]
    # Main group of combine_first tests; stateless pytest class.
    def test_combine_first_mixed(self):
        # [FUNCTION DOC]
        # Two frames with string and integer columns and disjoint indexes combine to the
        # stacked frame with union index.
        a = Series(["a", "b"], index=range(2))
        b = Series(range(2), index=range(2))
        f = DataFrame({"A": a, "B": b})

        a = Series(["a", "b"], index=range(5, 7))
        b = Series(range(2), index=range(5, 7))
        g = DataFrame({"A": a, "B": b})

        exp = DataFrame({"A": list("abab"), "B": [0, 1, 0, 1]}, index=[0, 1, 5, 6])
        combined = f.combine_first(g)
        tm.assert_frame_equal(combined, exp)

    def test_combine_first_disjoint(self, float_frame):
        # [FUNCTION DOC]
        # Splitting the float frame into head and tail and combining gives back the
        # original rows (reordered by the union index), same columns.
        head, tail = float_frame[:5], float_frame[5:]
        combined = head.combine_first(tail)
        reordered_frame = float_frame.reindex(combined.index)

        tm.assert_frame_equal(combined, reordered_frame)
        tm.assert_index_equal(combined.columns, float_frame.columns)
        tm.assert_series_equal(combined["A"], reordered_frame["A"])

        tm.assert_series_equal(combined["A"].reindex(head.index), head["A"])
        tm.assert_series_equal(combined["A"].reindex(tail.index), tail["A"])

    def test_combine_first_same_index(self, float_frame):
        # [FUNCTION DOC]
        # Frames over the same index with different columns: result takes the first
        # frame values where present and fills missing columns from the second.
        fcopy = float_frame.copy()
        fcopy["A"] = 1
        del fcopy["C"]

        fcopy2 = float_frame.copy()
        fcopy2["B"] = 0
        del fcopy2["D"]

        combined = fcopy.combine_first(fcopy2)

        assert (combined["A"] == 1).all()
        tm.assert_series_equal(combined["B"], fcopy["B"])
        tm.assert_series_equal(combined["C"], fcopy2["C"])
        tm.assert_series_equal(combined["D"], fcopy["D"])

    def test_combine_first_overlap(self, float_frame):
        # [FUNCTION DOC]
        # Values from the first frame take priority over the second in overlapping rows.
        combined = float_frame[:5].combine_first(float_frame[5:])
        reordered_frame = float_frame.reindex(combined.index)
        head, tail = reordered_frame[:10].copy(), reordered_frame.copy()
        head["A"] = 1
        combined = head.combine_first(tail)
        assert (combined["A"][:10] == 1).all()

    def test_combine_first_reverse_overlap(self, float_frame):
        # [FUNCTION DOC]
        # Same priority check with the roles reversed.
        combined = float_frame[:5].combine_first(float_frame[5:])
        reordered_frame = float_frame.reindex(combined.index)
        head, tail = reordered_frame[:10].copy(), reordered_frame

        tail.iloc[:10, tail.columns.get_loc("A")] = 0
        combined = tail.combine_first(head)
        assert (combined["A"][:10] == 0).all()

    def test_combine_first_with_empty(self, float_frame):
        # [FUNCTION DOC]
        # Combining with an empty frame returns the frame; combining an empty frame with
        # the float frame returns it sorted by index.
        comb = float_frame.combine_first(DataFrame())
        tm.assert_frame_equal(comb, float_frame)

        comb = DataFrame().combine_first(float_frame)
        tm.assert_frame_equal(comb, float_frame.sort_index())

    def test_combine_first_with_new_index(self, float_frame):
        # [FUNCTION DOC]
        # Combining with a frame having new row labels adds those labels to the result
        # index.
        comb = float_frame.combine_first(DataFrame(index=["faz", "boo"]))
        assert "faz" in comb.index

    def test_combine_first_column_union(self):
        # [FUNCTION DOC]
        # Combining with a frame that has an extra (empty) column includes that column
        # in the result.
        # GH#2525
        df = DataFrame({"a": [1]}, index=[datetime(2012, 1, 1)])
        df2 = DataFrame(columns=["b"])
        result = df.combine_first(df2)
        assert "b" in result

    def test_combine_first_mixed_bug(self):
        # [FUNCTION DOC]
        # Combining frames with partially overlapping indexes and columns of mixed dtype
        # gives five columns (union).
        idx = Index(["a", "b", "c", "e"])
        ser1 = Series([5.0, -9.0, 4.0, 100.0], index=idx)
        ser2 = Series(["a", "b", "c", "e"], index=idx)
        ser3 = Series([12, 4, 5, 97], index=idx)

        frame1 = DataFrame({"col0": ser1, "col2": ser2, "col3": ser3})

        idx = Index(["a", "b", "c", "f"])
        ser1 = Series([5.0, -9.0, 4.0, 100.0], index=idx)
        ser2 = Series(["a", "b", "c", "f"], index=idx)
        ser3 = Series([12, 4, 5, 97], index=idx)

        frame2 = DataFrame({"col1": ser1, "col2": ser2, "col5": ser3})

        combined = frame1.combine_first(frame2)
        assert len(combined.columns) == 5

    def test_combine_first_same_as_in_update(self):
        # [FUNCTION DOC]
        # Behaves like update: with an other frame covering one cell, missing values in
        # the first are filled; boolean columns are retained.
        # gh 3016 (same as in update)
        df = DataFrame(
            [[1.0, 2.0, False, True], [4.0, 5.0, True, False]],
            columns=["A", "B", "bool1", "bool2"],
        )

        other = DataFrame([[45, 45]], index=[0], columns=["A", "B"])
        result = df.combine_first(other)
        tm.assert_frame_equal(result, df)

        df.loc[0, "A"] = np.nan
        result = df.combine_first(other)
        df.loc[0, "A"] = 45
        tm.assert_frame_equal(result, df)

    def test_combine_first_doc_example(self):
        # [FUNCTION DOC]
        # Reproduces the documentation example: missing entries filled from the other
        # frame, union of index.
        # doc example
        df1 = DataFrame(
            {"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
        )

        df2 = DataFrame(
            {
                "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
                "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
            }
        )

        result = df1.combine_first(df2)
        expected = DataFrame({"A": [1, 2, 3, 5, 3, 7.0], "B": [np.nan, 2, 3, 4, 6, 8]})
        tm.assert_frame_equal(result, expected)

    def test_combine_first_return_obj_type_with_bools(self):
        # [FUNCTION DOC]
        # Boolean column stays bool in both call orders when combined frames have partly
        # overlapping rows.
        # GH3552

        df1 = DataFrame(
            [[np.nan, 3.0, True], [-4.6, np.nan, True], [np.nan, 7.0, False]]
        )
        df2 = DataFrame([[-42.6, np.nan, True], [-5.0, 1.6, False]], index=[1, 2])

        expected = Series([True, True, False], name=2, dtype=bool)

        result_12 = df1.combine_first(df2)[2]
        tm.assert_series_equal(result_12, expected)

        result_21 = df2.combine_first(df1)[2]
        tm.assert_series_equal(result_21, expected)

    @pytest.mark.parametrize(
        "data1, data2, data_expected",
        (
            (
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [pd.NaT, pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [pd.NaT, pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [datetime(2000, 1, 2), pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 2), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 2), pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
        ),
    )
    def test_combine_first_convert_datatime_correctly(
        self, data1, data2, data_expected
    ):
        # [FUNCTION DOC]
        # Parametrized over four datetime/NaT combinations (data1, data2, expected).
        # Missing datetimes in the first are filled from the second; datetime dtype is
        # retained.
        # GH 3593

        df1, df2 = DataFrame({"a": data1}), DataFrame({"a": data2})
        result = df1.combine_first(df2)
        expected = DataFrame({"a": data_expected})
        tm.assert_frame_equal(result, expected)

    def test_combine_first_align_nan(self):
        # [FUNCTION DOC]
        # Pins current behavior for combining with a frame of fewer columns and with an
        # empty first frame: datetime stays in second resolution when the first has
        # data; with an empty first frame the result column dtypes are float and int.
        # KNOWN ISSUE: comments in the test say the empty-left result for the datetime
        # column should be datetime and some columns should be int64 (TODO), yet the
        # assertions pin the current (float64) behavior, so the test encodes behavior it
        # describes as wrong.
        # GH 7509 (not fixed)
        ts = pd.Timestamp("2011-01-01").as_unit("s")
        dfa = DataFrame([[ts, 2]], columns=["a", "b"])
        dfb = DataFrame([[4], [5]], columns=["b"])
        assert dfa["a"].dtype == "datetime64[s]"
        assert dfa["b"].dtype == "int64"

        res = dfa.combine_first(dfb)
        exp = DataFrame(
            {"a": [ts, pd.NaT], "b": [2, 5]},
            columns=["a", "b"],
        )
        tm.assert_frame_equal(res, exp)
        assert res["a"].dtype == "datetime64[s]"
        # TODO: this must be int64
        assert res["b"].dtype == "int64"

        res = dfa.iloc[:0].combine_first(dfb)
        exp = DataFrame({"a": [np.nan, np.nan], "b": [4, 5]}, columns=["a", "b"])
        tm.assert_frame_equal(res, exp)
        # TODO: this must be datetime64
        assert res["a"].dtype == "float64"
        # TODO: this must be int64
        assert res["b"].dtype == "int64"

    def test_combine_first_timezone(self, unit):
        # [FUNCTION DOC]
        # Takes unit fixture. Combining tz-aware columns over different indexes keeps
        # the tz-aware dtype in the shared and new columns, with the unit preserved.
        # see gh-7630
        data1 = pd.to_datetime("20100101 01:01").tz_localize("UTC").as_unit(unit)
        df1 = DataFrame(
            columns=["UTCdatetime", "abc"],
            data=data1,
            index=pd.date_range("20140627", periods=1, unit=unit),
        )
        data2 = pd.to_datetime("20121212 12:12").tz_localize("UTC").as_unit(unit)
        df2 = DataFrame(
            columns=["UTCdatetime", "xyz"],
            data=data2,
            index=pd.date_range("20140628", periods=1, unit=unit),
        )
        res = df2[["UTCdatetime"]].combine_first(df1)
        exp = DataFrame(
            {
                "UTCdatetime": [
                    pd.Timestamp("2010-01-01 01:01", tz="UTC"),
                    pd.Timestamp("2012-12-12 12:12", tz="UTC"),
                ],
                "abc": [pd.Timestamp("2010-01-01 01:01:00", tz="UTC"), pd.NaT],
            },
            columns=["UTCdatetime", "abc"],
            index=pd.date_range("20140627", periods=2, freq="D", unit=unit),
            dtype=f"datetime64[{unit}, UTC]",
        )
        assert res["UTCdatetime"].dtype == f"datetime64[{unit}, UTC]"
        assert res["abc"].dtype == f"datetime64[{unit}, UTC]"

        tm.assert_frame_equal(res, exp)

    def test_combine_first_timezone2(self, unit):
        # [FUNCTION DOC]
        # Takes unit fixture. Combining two tz-aware UTC columns where the second only
        # covers part of the first returns the first unchanged with tz-aware dtype.
        # see gh-10567
        dts1 = pd.date_range("2015-01-01", "2015-01-05", tz="UTC", unit=unit)
        df1 = DataFrame({"DATE": dts1})
        dts2 = pd.date_range("2015-01-03", "2015-01-05", tz="UTC", unit=unit)
        df2 = DataFrame({"DATE": dts2})

        res = df1.combine_first(df2)
        tm.assert_frame_equal(res, df1)
        assert res["DATE"].dtype == f"datetime64[{unit}, UTC]"

    def test_combine_first_timezone3(self, unit):
        # [FUNCTION DOC]
        # Takes unit fixture. US/Eastern columns with NaT: missing entries filled from
        # the other, union of indexes, tz and unit preserved.
        dts1 = pd.DatetimeIndex(
            ["2011-01-01", "NaT", "2011-01-03", "2011-01-04"], tz="US/Eastern"
        ).as_unit(unit)
        df1 = DataFrame({"DATE": dts1}, index=[1, 3, 5, 7])
        dts2 = pd.DatetimeIndex(
            ["2012-01-01", "2012-01-02", "2012-01-03"], tz="US/Eastern"
        ).as_unit(unit)
        df2 = DataFrame({"DATE": dts2}, index=[2, 4, 5])

        res = df1.combine_first(df2)
        exp_dts = pd.DatetimeIndex(
            [
                "2011-01-01",
                "2012-01-01",
                "NaT",
                "2012-01-02",
                "2011-01-03",
                "2011-01-04",
            ],
            tz="US/Eastern",
        ).as_unit(unit)
        exp = DataFrame({"DATE": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        tm.assert_frame_equal(res, exp)

    def test_combine_first_timezone4(self, unit):
        # [FUNCTION DOC]
        # Takes unit fixture. When the first frame has no missing values, its tz-aware
        # dtype is kept even if the other is naive.
        # different tz
        dts1 = pd.date_range("2015-01-01", "2015-01-05", tz="US/Eastern", unit=unit)
        df1 = DataFrame({"DATE": dts1})
        dts2 = pd.date_range("2015-01-03", "2015-01-05", unit=unit)
        df2 = DataFrame({"DATE": dts2})

        # if df1 doesn't have NaN, keep its dtype
        res = df1.combine_first(df2)
        tm.assert_frame_equal(res, df1)
        assert res["DATE"].dtype == f"datetime64[{unit}, US/Eastern]"

    def test_combine_first_timezone5(self, unit):
        # [FUNCTION DOC]
        # Takes unit fixture. Mixed tz-aware and naive datetimes with overlapping rows
        # yield an object column of timestamps.
        dts1 = pd.date_range("2015-01-01", "2015-01-02", tz="US/Eastern", unit=unit)
        df1 = DataFrame({"DATE": dts1})
        dts2 = pd.date_range("2015-01-01", "2015-01-03", unit=unit)
        df2 = DataFrame({"DATE": dts2})

        res = df1.combine_first(df2)
        exp_dts = [
            pd.Timestamp("2015-01-01", tz="US/Eastern"),
            pd.Timestamp("2015-01-02", tz="US/Eastern"),
            pd.Timestamp("2015-01-03"),
        ]
        exp = DataFrame({"DATE": exp_dts})
        tm.assert_frame_equal(res, exp)
        assert res["DATE"].dtype == "object"

    def test_combine_first_timedelta(self):
        # [FUNCTION DOC]
        # Timedelta columns with missing values are filled from the other; the result
        # has microsecond-resolution timedelta dtype.
        data1 = pd.TimedeltaIndex(["1 day", "NaT", "3 day", "4day"])
        df1 = DataFrame({"TD": data1}, index=[1, 3, 5, 7])
        data2 = pd.TimedeltaIndex(["10 day", "11 day", "12 day"])
        df2 = DataFrame({"TD": data2}, index=[2, 4, 5])

        res = df1.combine_first(df2)
        exp_dts = pd.TimedeltaIndex(
            ["1 day", "10 day", "NaT", "11 day", "3 day", "4 day"]
        )
        exp = DataFrame({"TD": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        tm.assert_frame_equal(res, exp)
        assert res["TD"].dtype == "timedelta64[us]"

    def test_combine_first_period(self):
        # [FUNCTION DOC]
        # Monthly period columns combine keeping period dtype; if the other has daily
        # frequency, the result is an object column of period objects.
        data1 = pd.PeriodIndex(["2011-01", "NaT", "2011-03", "2011-04"], freq="M")
        df1 = DataFrame({"P": data1}, index=[1, 3, 5, 7])
        data2 = pd.PeriodIndex(["2012-01-01", "2012-02", "2012-03"], freq="M")
        df2 = DataFrame({"P": data2}, index=[2, 4, 5])

        res = df1.combine_first(df2)
        exp_dts = pd.PeriodIndex(
            ["2011-01", "2012-01", "NaT", "2012-02", "2011-03", "2011-04"], freq="M"
        )
        exp = DataFrame({"P": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        tm.assert_frame_equal(res, exp)
        assert res["P"].dtype == data1.dtype

        # different freq
        dts2 = pd.PeriodIndex(["2012-01-01", "2012-01-02", "2012-01-03"], freq="D")
        df2 = DataFrame({"P": dts2}, index=[2, 4, 5])

        res = df1.combine_first(df2)
        exp_dts = [
            pd.Period("2011-01", freq="M"),
            pd.Period("2012-01-01", freq="D"),
            pd.NaT,
            pd.Period("2012-01-02", freq="D"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-04", freq="M"),
        ]
        exp = DataFrame({"P": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        tm.assert_frame_equal(res, exp)
        assert res["P"].dtype == "object"

    def test_combine_first_int(self):
        # [FUNCTION DOC]
        # Combining integer frames with misaligned rows keeps integer dtype and the
        # first frame values take priority.
        # GH14687 - integer series that do no align exactly

        df1 = DataFrame({"a": [0, 1, 3, 5]}, dtype="int64")
        df2 = DataFrame({"a": [1, 4]}, dtype="int64")

        result_12 = df1.combine_first(df2)
        expected_12 = DataFrame({"a": [0, 1, 3, 5]})
        tm.assert_frame_equal(result_12, expected_12)

        result_21 = df2.combine_first(df1)
        expected_21 = DataFrame({"a": [1, 4, 3, 5]})
        tm.assert_frame_equal(result_21, expected_21)

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_combine_first_with_asymmetric_other(self, val):
        # [FUNCTION DOC]
        # Parametrized over int and float values. Combining with a frame having a
        # different column adds that column without changing the value column.
        # see gh-20699
        df1 = DataFrame({"isNum": [val]})
        df2 = DataFrame({"isBool": [True]})

        res = df1.combine_first(df2)
        exp = DataFrame({"isNum": [val], "isBool": [True]})

        tm.assert_frame_equal(res, exp)

    def test_combine_first_string_dtype_only_na(self, nullable_string_dtype):
        # [FUNCTION DOC]
        # Takes nullable string dtype fixture. Multi-level index built from columns that
        # include missing values: combined result equals the first frame.
        # GH: 37519
        df = DataFrame(
            {"a": ["962", "85"], "b": [pd.NA] * 2}, dtype=nullable_string_dtype
        )
        df2 = DataFrame({"a": ["85"], "b": [pd.NA]}, dtype=nullable_string_dtype)
        df.set_index(["a", "b"], inplace=True)
        df2.set_index(["a", "b"], inplace=True)
        result = df.combine_first(df2)
        expected = DataFrame(
            {"a": ["962", "85"], "b": [pd.NA] * 2}, dtype=nullable_string_dtype
        ).set_index(["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "wide_val, dtype",
        (
            (1666880195890293744, "UInt64"),
            (-1666880195890293744, "Int64"),
        ),
    )
    def test_combine_first_preserve_EA_precision(self, wide_val, dtype):
        # [FUNCTION DOC]
        # Parametrized over very large values and the matching unsigned or signed
        # nullable integer dtype. Large integers keep exact precision (no float round
        # trip).
        # GH#60128
        df1 = DataFrame({"A": [wide_val, 5]}, dtype=dtype)
        df2 = DataFrame({"A": [6, 7, wide_val]}, dtype=dtype)
        result = df1.combine_first(df2)
        expected = DataFrame({"A": [wide_val, 5, wide_val]}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_combine_first_non_unique_columns(self):
        # [FUNCTION DOC]
        # Other frame has duplicated column labels; the result has the same duplicated
        # labels, with the first frame values used where not missing and missing cells
        # filled positionally from the other.
        # GH#29135
        df1 = DataFrame([[1, np.nan], [3, 4]], columns=["P", "Q"], index=["A", "B"])
        df2 = DataFrame(
            [[5, 6, 7], [8, 9, np.nan]], columns=["P", "Q", "Q"], index=["A", "B"]
        )
        result = df1.combine_first(df2)
        expected = DataFrame(
            [[1, 6.0, 7.0], [3, 4.0, 4.0]], index=["A", "B"], columns=["P", "Q", "Q"]
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "scalar1, scalar2",
    [
        (datetime(2020, 1, 1), datetime(2020, 1, 2)),
        (pd.Period("2020-01-01", "D"), pd.Period("2020-01-02", "D")),
        (pd.Timedelta("89 days"), pd.Timedelta("60 min")),
        (pd.Interval(left=0, right=1), pd.Interval(left=2, right=3, closed="left")),
    ],
)
def test_combine_first_timestamp_bug(scalar1, scalar2, nulls_fixture):
    # [FUNCTION DOC]
    # Parametrized over scalar pairs (datetime, period, timedelta, interval) and takes
    # the nulls fixture. Combining a frame of missing values with an other holding
    # scalars: the shared column takes the scalar only if the common dtype is object, or
    # dtypes are equal, or both datetime kinds, else it stays missing; the result column
    # is cast to the common dtype.
    # GH28481
    na_value = nulls_fixture

    frame = DataFrame([[na_value, na_value]], columns=["a", "b"])
    other = DataFrame([[scalar1, scalar2]], columns=["b", "c"])

    common_dtype = find_common_type([frame.dtypes["b"], other.dtypes["b"]])

    if (
        is_dtype_equal(common_dtype, "object")
        or frame.dtypes["b"] == other.dtypes["b"]
        or frame.dtypes["b"].kind == frame.dtypes["b"].kind == "M"
    ):
        val = scalar1
    else:
        val = na_value

    result = frame.combine_first(other)

    expected = DataFrame([[na_value, val, scalar2]], columns=["a", "b", "c"])

    expected["b"] = expected["b"].astype(common_dtype)

    tm.assert_frame_equal(result, expected)


def test_combine_first_timestamp_bug_NaT():
    # [FUNCTION DOC]
    # Frame of NaT combined with a frame of datetimes: the shared column is filled with
    # datetimes and the new column is added.
    # GH28481
    frame = DataFrame([[pd.NaT, pd.NaT]], columns=["a", "b"])
    other = DataFrame(
        [[datetime(2020, 1, 1), datetime(2020, 1, 2)]], columns=["b", "c"]
    )

    result = frame.combine_first(other)
    expected = DataFrame(
        [[pd.NaT, datetime(2020, 1, 1), datetime(2020, 1, 2)]], columns=["a", "b", "c"]
    )

    tm.assert_frame_equal(result, expected)


def test_combine_first_with_nan_multiindex():
    # [FUNCTION DOC]
    # Multi-level indexes containing a missing label: result index is the sorted union
    # (missing label last) and values from both frames are placed correctly.
    # gh-36562

    mi1 = MultiIndex.from_arrays(
        [["b", "b", "c", "a", "b", np.nan], [1, 2, 3, 4, 5, 6]], names=["a", "b"]
    )
    df = DataFrame({"c": [1, 1, 1, 1, 1, 1]}, index=mi1)
    mi2 = MultiIndex.from_arrays(
        [["a", "b", "c", "a", "b", "d"], [1, 1, 1, 1, 1, 1]], names=["a", "b"]
    )
    s = Series([1, 2, 3, 4, 5, 6], index=mi2)
    res = df.combine_first(DataFrame({"d": s}))
    mi_expected = MultiIndex.from_arrays(
        [
            ["a", "a", "a", "b", "b", "b", "b", "c", "c", "d", np.nan],
            [1, 1, 4, 1, 1, 2, 5, 1, 3, 1, 6],
        ],
        names=["a", "b"],
    )
    expected = DataFrame(
        {
            "c": [np.nan, np.nan, 1, 1, 1, 1, 1, np.nan, 1, np.nan, 1],
            "d": [1.0, 4.0, np.nan, 2.0, 5.0, np.nan, np.nan, 3.0, np.nan, 6.0, np.nan],
        },
        index=mi_expected,
    )
    tm.assert_frame_equal(res, expected)


def test_combine_preserve_dtypes():
    # [FUNCTION DOC]
    # Combining frames with string, integer columns and disjoint row labels preserves
    # integer and object dtypes (missing filled as appropriate).
    # GH7509
    a_column = Series(["a", "b"], index=range(2))
    b_column = Series(range(2), index=range(2))
    df1 = DataFrame({"A": a_column, "B": b_column})

    c_column = Series(["a", "b"], index=range(5, 7))
    b_column = Series(range(-1, 1), index=range(5, 7))
    df2 = DataFrame({"B": b_column, "C": c_column})

    expected = DataFrame(
        {
            "A": ["a", "b", np.nan, np.nan],
            "B": [0, 1, -1, 0],
            "C": [np.nan, np.nan, "a", "b"],
        },
        index=[0, 1, 5, 6],
    )
    combined = df1.combine_first(df2)
    tm.assert_frame_equal(combined, expected)


def test_combine_first_duplicates_rows_for_nan_index_values():
    # [FUNCTION DOC]
    # Multi-level index with missing values must not duplicate rows when combining;
    # result has the union of rows once each.
    # GH39881
    df1 = DataFrame(
        {"x": [9, 10, 11]},
        index=MultiIndex.from_arrays([[1, 2, 3], [np.nan, 5, 6]], names=["a", "b"]),
    )

    df2 = DataFrame(
        {"y": [12, 13, 14]},
        index=MultiIndex.from_arrays([[1, 2, 4], [np.nan, 5, 7]], names=["a", "b"]),
    )

    expected = DataFrame(
        {
            "x": [9.0, 10.0, 11.0, np.nan],
            "y": [12.0, 13.0, np.nan, 14.0],
        },
        index=MultiIndex.from_arrays(
            [[1, 2, 3, 4], [np.nan, 5, 6, 7]], names=["a", "b"]
        ),
    )
    combined = df1.combine_first(df2)
    tm.assert_frame_equal(combined, expected)


def test_combine_first_int64_not_cast_to_float64():
    # [FUNCTION DOC]
    # Integer columns stay int64 when the other frame has an extra column and no value
    # is missing.
    # GH 28613
    df_1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df_2 = DataFrame({"A": [1, 20, 30], "B": [40, 50, 60], "C": [12, 34, 65]})
    result = df_1.combine_first(df_2)
    expected = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [12, 34, 65]})
    tm.assert_frame_equal(result, expected)


def test_midx_losing_dtype():
    # [FUNCTION DOC]
    # Multi-level index with all-missing second level keeps its structure after
    # combining; result values are as expected.
    # GH#49830
    midx = MultiIndex.from_arrays([[0, 0], [np.nan, np.nan]])
    midx2 = MultiIndex.from_arrays([[1, 1], [np.nan, np.nan]])
    df1 = DataFrame({"a": [None, 4]}, index=midx)
    df2 = DataFrame({"a": [3, 3]}, index=midx2)
    result = df1.combine_first(df2)
    expected_midx = MultiIndex.from_arrays(
        [[0, 0, 1, 1], [np.nan, np.nan, np.nan, np.nan]]
    )
    expected = DataFrame({"a": [np.nan, 4, 3, 3]}, index=expected_midx)
    tm.assert_frame_equal(result, expected)


def test_combine_first_empty_columns():
    # [FUNCTION DOC]
    # Combining two frames with no rows gives an empty frame whose columns are the union
    # of both column sets.
    left = DataFrame(columns=["a", "b"])
    right = DataFrame(columns=["a", "c"])
    result = left.combine_first(right)
    expected = DataFrame(columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)


def test_combine_first_preserve_column_order():
    # [FUNCTION DOC]
    # The result keeps the column order of the first frame and fills missing entries
    # from the other.
    # GH#60427
    df1 = DataFrame({"B": [1, 2, 3], "A": [4, None, 6]})
    df2 = DataFrame({"A": [5]}, index=[1])

    result = df1.combine_first(df2)
    expected = DataFrame({"B": [1, 2, 3], "A": [4.0, 5.0, 6.0]})
    tm.assert_frame_equal(result, expected)
