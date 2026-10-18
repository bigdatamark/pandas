# [MODULE DOC]
# Tests for filling missing values in DataFrame: fillna with scalars, dicts, series and frames; forward and backward fill (with limits and limit_area); in-place behavior; dtype handling for datetime, timezone, categorical and object data; and error cases.
# Imports a private helper from the frame test common module to check mixed float dtypes.

import numpy as np
import pytest

from pandas.errors import OutOfBoundsDatetime

from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float


class TestFillNA:
    # [CLASS DOC]
    # Groups the DataFrame fillna, ffill and bfill tests.
    def test_fillna_dict_inplace_nonunique_columns(self):
        # [FUNCTION DOC]
        # Frame with duplicate column labels. Verifies in-place fillna with a dict keyed by the repeated label fills all columns carrying that label (the datetime column gets a mixed result), and that the column which cannot be filled in place no longer shares memory with the original.
        df = DataFrame(
            {"A": [np.nan] * 3, "B": [NaT, Timestamp(1), NaT], "C": [np.nan, "foo", 2]}
        )
        df.columns = ["A", "A", "A"]
        orig = df[:]

        df.fillna({"A": 2}, inplace=True)
        # The first and third columns can be set inplace, while the second cannot.

        expected = DataFrame(
            {"A": [2.0] * 3, "B": [2, Timestamp(1), 2], "C": [2, "foo", 2]}
        )
        expected.columns = ["A", "A", "A"]
        tm.assert_frame_equal(df, expected)
        assert not tm.shares_memory(df.iloc[:, 1], orig.iloc[:, 1])

    def test_fillna_on_column_view(self):
        # [FUNCTION DOC]
        # GH#46149. Frame wrapping a numpy array without copying. Verifies in-place fillna on a single column via chained access triggers the chained assignment error and leaves the array untouched, and that the frame still has one block sharing memory with the array.
        # GH#46149 avoid unnecessary copies
        arr = np.full((40, 50), np.nan)
        df = DataFrame(arr, copy=False)

        with tm.raises_chained_assignment_error():
            df[0].fillna(-1, inplace=True)
        assert np.isnan(arr[:, 0]).all()

        # i.e. we didn't create a new 49-column block
        assert len(df._mgr.blocks) == 1
        assert np.shares_memory(df.values, arr)

    def test_fillna_datetime(self, datetime_frame):
        # [FUNCTION DOC]
        # Using datetime_frame with leading and trailing missing values in one column. Verifies fillna with zero fills the head, forward fill leaves leading missing values missing, and calling fillna with no value raises a type error about the missing required value argument.
        tf = datetime_frame
        tf.loc[tf.index[:5], "A"] = np.nan
        tf.loc[tf.index[-5:], "A"] = np.nan

        zero_filled = datetime_frame.fillna(0)
        assert (zero_filled.loc[zero_filled.index[:5], "A"] == 0).all()

        padded = datetime_frame.ffill()
        assert np.isnan(padded.loc[padded.index[:5], "A"]).all()

        msg = r"missing 1 required positional argument: 'value'"
        with pytest.raises(TypeError, match=msg):
            datetime_frame.fillna()

    def test_fillna_mixed_type(self, float_string_frame, using_infer_string):
        # [FUNCTION DOC]
        # Mixed float and string frame with missing values. Verifies forward fill fills numeric and string gaps from the previous value, and fillna with zero fills both numeric and string columns with zero.
        # KNOWN ISSUE: the using_infer_string fixture argument is accepted but never used in the test body.
        mf = float_string_frame
        mf.loc[mf.index[5:20], "foo"] = np.nan
        mf.loc[mf.index[-10:], "A"] = np.nan

        result = mf.ffill()
        assert (
            result.loc[result.index[-10:], "A"] == result.loc[result.index[-11], "A"]
        ).all()
        assert (result.loc[result.index[5:20], "foo"] == "bar").all()

        result = mf.fillna(value=0)
        assert (result.loc[result.index[-10:], "A"] == 0).all()
        assert (result.loc[result.index[5:20], "foo"] == 0).all()

    def test_fillna_mixed_float(self, mixed_float_frame):
        # [FUNCTION DOC]
        # Uses mixed_float_frame with float32 and float64 columns (no float16). Verifies the dtypes of each column are preserved after fillna with zero and after forward fill.
        # mixed numeric (but no float16)
        mf = mixed_float_frame.reindex(columns=["A", "B", "D"])
        mf.loc[mf.index[-10:], "A"] = np.nan
        result = mf.fillna(value=0)
        _check_mixed_float(result, dtype={"C": None})
        result = mf.ffill()
        _check_mixed_float(result, dtype={"C": None})

    def test_fillna_different_dtype(self):
        # [FUNCTION DOC]
        # GH#3386. All-missing float columns filled by a dict with a string. Verifies the filled column becomes object dtype, and the inplace variant returns the frame itself with the same contents.
        # with different dtype (GH#3386)
        df = DataFrame(
            [["a", "a", np.nan, "a"], ["b", "b", np.nan, "b"], ["c", "c", np.nan, "c"]]
        )

        result = df.fillna({2: "foo"})
        expected = DataFrame(
            [["a", "a", "foo", "a"], ["b", "b", "foo", "b"], ["c", "c", "foo", "c"]]
        )
        # column is originally float (all-NaN) -> filling with string gives object dtype
        expected[2] = expected[2].astype("object")
        tm.assert_frame_equal(result, expected)

        result = df.fillna({2: "foo"}, inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)

    def test_fillna_limit_and_value(self):
        # [FUNCTION DOC]
        # Verifies fillna with a limit fills only the first missing value in each column with the given value.
        # limit and value
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
        df.iloc[2:7, 0] = np.nan
        df.iloc[3:5, 2] = np.nan

        expected = df.copy()
        expected.iloc[2, 0] = 999
        expected.iloc[3, 2] = 999
        result = df.fillna(999, limit=1)
        tm.assert_frame_equal(result, expected)

    def test_fillna_datelike(self):
        # [FUNCTION DOC]
        # GH#6344. Verifies fillna with a dict whose value is a series fills a datetime column from the aligned values of another datetime column.
        # with datelike
        # GH#6344
        df = DataFrame(
            {
                "Date": [NaT, Timestamp("2014-1-1")],
                "Date2": [Timestamp("2013-1-1"), NaT],
            }
        )

        expected = df.copy()
        expected["Date"] = expected["Date"].fillna(df.loc[df.index[0], "Date2"])
        result = df.fillna(value={"Date": df["Date2"]})
        tm.assert_frame_equal(result, expected)

    def test_fillna_tzaware(self):
        # [FUNCTION DOC]
        # GH#15855. Verifies forward fill and backward fill of timezone-aware timestamps with missing values.
        # with timezone
        # GH#15855
        df = DataFrame({"A": [Timestamp("2012-11-11 00:00:00+01:00"), NaT]})
        exp = DataFrame(
            {
                "A": [
                    Timestamp("2012-11-11 00:00:00+01:00"),
                    Timestamp("2012-11-11 00:00:00+01:00"),
                ]
            }
        )
        res = df.ffill()
        tm.assert_frame_equal(res, exp)

        df = DataFrame({"A": [NaT, Timestamp("2012-11-11 00:00:00+01:00")]})
        exp = DataFrame(
            {
                "A": [
                    Timestamp("2012-11-11 00:00:00+01:00"),
                    Timestamp("2012-11-11 00:00:00+01:00"),
                ]
            }
        )
        res = df.bfill()
        tm.assert_frame_equal(res, exp)

    def test_fillna_tzaware_different_column(self):
        # [FUNCTION DOC]
        # GH#15522. Frame with a timezone-aware column and a float column with trailing missing values. Verifies forward fill fills the float column and preserves the timezone column.
        # with timezone in another column
        # GH#15522
        df = DataFrame(
            {
                "A": date_range("20130101", periods=4, tz="US/Eastern"),
                "B": [1, 2, np.nan, np.nan],
            }
        )
        result = df.ffill()
        expected = DataFrame(
            {
                "A": date_range("20130101", periods=4, tz="US/Eastern"),
                "B": [1.0, 2.0, 2.0, 2.0],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_na_actions_categorical(self):
        # [FUNCTION DOC]
        # Categorical column with a missing value. Verifies fillna with a dict value (an existing category) works; a value that is not a category raises a type error about setting a new category; forward fill works; dropna with subset or on all columns drops the expected rows; and fillna with a scalar takes missing values in categories into account.
        cat = Categorical([1, 2, 3, np.nan], categories=[1, 2, 3])
        vals = ["a", "b", np.nan, "d"]
        df = DataFrame({"cats": cat, "vals": vals})
        cat2 = Categorical([1, 2, 3, 3], categories=[1, 2, 3])
        vals2 = ["a", "b", "b", "d"]
        df_exp_fill = DataFrame({"cats": cat2, "vals": vals2})
        cat3 = Categorical([1, 2, 3], categories=[1, 2, 3])
        vals3 = ["a", "b", np.nan]
        df_exp_drop_cats = DataFrame({"cats": cat3, "vals": vals3})
        cat4 = Categorical([1, 2], categories=[1, 2, 3])
        vals4 = ["a", "b"]
        df_exp_drop_all = DataFrame({"cats": cat4, "vals": vals4})

        # fillna
        res = df.fillna(value={"cats": 3, "vals": "b"})
        tm.assert_frame_equal(res, df_exp_fill)

        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            df.fillna(value={"cats": 4, "vals": "c"})

        res = df.ffill()
        tm.assert_frame_equal(res, df_exp_fill)

        # dropna
        res = df.dropna(subset=["cats"])
        tm.assert_frame_equal(res, df_exp_drop_cats)

        res = df.dropna()
        tm.assert_frame_equal(res, df_exp_drop_all)

        # make sure that fillna takes missing values into account
        c = Categorical([np.nan, "b", np.nan], categories=["a", "b"])
        df = DataFrame({"cats": c, "vals": [1, 2, 3]})

        cat_exp = Categorical(["a", "b", "a"], categories=["a", "b"])
        df_exp = DataFrame({"cats": cat_exp, "vals": [1, 2, 3]})

        res = df.fillna("a")
        tm.assert_frame_equal(res, df_exp)

    def test_fillna_categorical_nan(self):
        # [FUNCTION DOC]
        # GH#14021. Verifies a missing value is always a valid filler for categorical data (including series fill with a median-like series), and that fillna with NaT on categoricals built from datetime, period and timedelta indexes returns the same frame.
        # GH#14021
        # np.nan should always be a valid filler
        cat = Categorical([np.nan, 2, np.nan])
        val = Categorical([np.nan, np.nan, np.nan])
        df = DataFrame({"cats": cat, "vals": val})

        # GH#32950 df.median() is poorly behaved because there is no
        #  Categorical.median
        median = Series({"cats": 2.0, "vals": np.nan})

        res = df.fillna(median)
        v_exp = [np.nan, np.nan, np.nan]
        df_exp = DataFrame({"cats": [2, 2, 2], "vals": v_exp}, dtype="category")
        tm.assert_frame_equal(res, df_exp)

        result = df.cats.fillna(np.nan)
        tm.assert_series_equal(result, df.cats)

        result = df.vals.fillna(np.nan)
        tm.assert_series_equal(result, df.vals)

        idx = DatetimeIndex(
            ["2011-01-01 09:00", "2016-01-01 23:45", "2011-01-01 09:00", NaT, NaT]
        )
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

        idx = PeriodIndex(["2011-01", "2011-01", "2011-01", NaT, NaT], freq="M")
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

        idx = TimedeltaIndex(["1 days", "2 days", "1 days", NaT, NaT])
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

    def test_fillna_with_categorical_series(self):
        # [FUNCTION DOC]
        # Issue 56329. Verifies filling a float column with a categorical series filler through a dict value fills with the filler's values at the missing positions.
        # https://github.com/pandas-dev/pandas/issues/56329
        df = DataFrame(
            {"cats": Categorical(["A", "B", "C"]), "ints": [1.0, 2.0, np.nan]}
        )

        filler = Series(Categorical([10.0, 20.0, 30.0]))
        result = df.fillna({"ints": filler})

        expected = DataFrame(
            {"cats": Categorical(["A", "B", "C"]), "ints": [1.0, 2.0, 30.0]}
        )
        tm.assert_frame_equal(result, expected)

    def test_fillna_no_downcast(self, frame_or_series):
        # [FUNCTION DOC]
        # GH#45603. Parametrized over frame and series. Object dtype data without missing values filled with an empty string keeps object dtype and is unchanged (no downcasting).
        # GH#45603 preserve object dtype
        obj = frame_or_series([1, 2, 3], dtype="object")
        result = obj.fillna("")
        tm.assert_equal(result, obj)

    @pytest.mark.parametrize("columns", [["A", "A", "B"], ["A", "A"]])
    def test_fillna_dictlike_value_duplicate_colnames(self, columns):
        # [FUNCTION DOC]
        # GH#43476. Parametrized over column lists with duplicate labels. Verifies a dict fill keyed by the repeated label fills all columns with that label, and no warning is emitted.
        # GH#43476
        df = DataFrame(np.nan, index=[0, 1], columns=columns)
        with tm.assert_produces_warning(None):
            result = df.fillna({"A": 0})

        expected = df.copy()
        expected["A"] = 0.0
        tm.assert_frame_equal(result, expected)

    def test_fillna_dtype_conversion(self):
        # [FUNCTION DOC]
        # Verifies fillna on an object frame that is all missing (and has no values) fills with the scalar giving object dtype, and that an all-missing float frame filled with a string becomes object dtype.
        # make sure that fillna on an empty frame works
        df = DataFrame(index=["A", "B", "C"], columns=[1, 2, 3, 4, 5])
        result = df.dtypes
        expected = Series([np.dtype("object")] * 5, index=[1, 2, 3, 4, 5])
        tm.assert_series_equal(result, expected)
        result = df.fillna(1)
        expected = DataFrame(
            1, index=["A", "B", "C"], columns=[1, 2, 3, 4, 5], dtype=object
        )
        tm.assert_frame_equal(result, expected)

        # empty block
        df = DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
        result = df.fillna("nan")
        expected = DataFrame("nan", dtype="object", index=range(3), columns=["A", "B"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("val", ["", 1, np.nan, 1.0])
    def test_fillna_dtype_conversion_equiv_replace(self, val):
        # [FUNCTION DOC]
        # Parametrized over fill values (empty string, int, nan, float). Verifies fillna gives the same result as replacing missing values with that value.
        df = DataFrame({"A": [1, np.nan], "B": [1.0, 2.0]})
        expected = df.replace(np.nan, val)
        result = df.fillna(val)
        tm.assert_frame_equal(result, expected)

    def test_fillna_datetime_columns(self):
        # [FUNCTION DOC]
        # GH#7095. Frames with datetime columns (one without missing values, one with a missing timestamp). Verifies fillna with a string fills missing entries in the other columns and in the datetime column with the string, giving object dtype where needed.
        # GH#7095
        df = DataFrame(
            {
                "A": [-1, -2, np.nan],
                "B": date_range("20130101", periods=3),
                "C": ["foo", "bar", None],
                "D": ["foo2", "bar2", None],
            },
            index=date_range("20130110", periods=3),
        )
        result = df.fillna("?")
        expected = DataFrame(
            {
                "A": [-1, -2, "?"],
                "B": date_range("20130101", periods=3),
                "C": ["foo", "bar", "?"],
                "D": ["foo2", "bar2", "?"],
            },
            index=date_range("20130110", periods=3),
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {
                "A": [-1, -2, np.nan],
                "B": [Timestamp("2013-01-01"), Timestamp("2013-01-02"), NaT],
                "C": ["foo", "bar", None],
                "D": ["foo2", "bar2", None],
            },
            index=date_range("20130110", periods=3),
        )
        result = df.fillna("?")
        expected = DataFrame(
            {
                "A": [-1, -2, "?"],
                "B": [Timestamp("2013-01-01"), Timestamp("2013-01-02"), "?"],
                "C": ["foo", "bar", "?"],
                "D": ["foo2", "bar2", "?"],
            },
            index=date_range("20130110", periods=3),
        )
        tm.assert_frame_equal(result, expected)

    def test_ffill(self, datetime_frame):
        # [FUNCTION DOC]
        # Verifies forward fill on datetime_frame with missing values in one column.
        # KNOWN ISSUE: the test compares the result of forward fill with another call of forward fill on the same frame, so it checks only determinism, not the filled values.
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        alt = datetime_frame.ffill()
        tm.assert_frame_equal(datetime_frame.ffill(), alt)

    def test_bfill(self, datetime_frame):
        # [FUNCTION DOC]
        # Verifies backward fill on datetime_frame with missing values in one column.
        # KNOWN ISSUE: the test compares the result of backward fill with another call of backward fill on the same frame, so it checks only determinism, not the filled values.
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        alt = datetime_frame.bfill()
        tm.assert_frame_equal(datetime_frame.bfill(), alt)

    def test_frame_pad_backfill_limit(self):
        # [FUNCTION DOC]
        # Verifies reindexing with the pad or backfill method and a limit fills at most the limit number of positions, equal to forward or backward filling followed by blanking the positions beyond the limit.
        index = np.arange(10)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=index)

        result = df[:2].reindex(index, method="pad", limit=5)

        expected = df[:2].reindex(index).ffill()
        expected.iloc[-3:] = np.nan
        tm.assert_frame_equal(result, expected)

        result = df[-2:].reindex(index, method="backfill", limit=5)

        expected = df[-2:].reindex(index).bfill()
        expected.iloc[:3] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_frame_fillna_limit(self):
        # [FUNCTION DOC]
        # Verifies forward fill and backward fill with a limit of five after reindexing fill at most five rows from each end.
        index = np.arange(10)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=index)

        result = df[:2].reindex(index)
        result = result.ffill(limit=5)

        expected = df[:2].reindex(index).ffill()
        expected.iloc[-3:] = np.nan
        tm.assert_frame_equal(result, expected)

        result = df[-2:].reindex(index)
        result = result.bfill(limit=5)

        expected = df[-2:].reindex(index).bfill()
        expected.iloc[:3] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_fillna_skip_certain_blocks(self):
        # [FUNCTION DOC]
        # Verifies fillna with nan on an integer-valued frame works (integer blocks are skipped). Smoke test only; the result is not compared.
        # don't try to fill boolean, int blocks

        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)).astype(int))

        # it works!
        df.fillna(np.nan)

    @pytest.mark.parametrize("type", [int, float])
    def test_fillna_positive_limit(self, type):
        # [FUNCTION DOC]
        # Parametrized over int and float data. Verifies a negative limit raises a value error saying limit must be greater than zero.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4))).astype(type)

        msg = "Limit must be greater than 0"
        with pytest.raises(ValueError, match=msg):
            df.fillna(0, limit=-5)

    @pytest.mark.parametrize("type", [int, float])
    def test_fillna_integer_limit(self, type):
        # [FUNCTION DOC]
        # Parametrized over int and float data. Verifies a non-integer limit raises a value error saying limit must be an integer.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4))).astype(type)

        msg = "Limit must be an integer"
        with pytest.raises(ValueError, match=msg):
            df.fillna(0, limit=0.5)

    def test_fillna_inplace(self):
        # [FUNCTION DOC]
        # Verifies inplace fillna (scalar, dict, and forward fill) returns the same frame object and gives the same data as the non-inplace results.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        df.loc[:4, 1] = np.nan
        df.loc[-4:, 3] = np.nan

        expected = df.fillna(value=0)
        assert expected is not df

        result = df.fillna(value=0, inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)

        result = df.fillna(value={0: 0}, inplace=True)
        assert result is df

        df.loc[:4, 1] = np.nan
        df.loc[-4:, 3] = np.nan
        expected = df.ffill()
        assert expected is not df

        df.ffill(inplace=True)
        tm.assert_frame_equal(df, expected)

    def test_fillna_dict_series(self):
        # [FUNCTION DOC]
        # Verifies fillna with a dict fills only the listed columns, extra keys in the dict are ignored, and a series filler behaves like the equivalent dict.
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            }
        )

        result = df.fillna({"a": 0, "b": 5})

        expected = df.copy()
        expected["a"] = expected["a"].fillna(0)
        expected["b"] = expected["b"].fillna(5)
        tm.assert_frame_equal(result, expected)

        # it works
        result = df.fillna({"a": 0, "b": 5, "d": 7})

        # Series treated same as dict
        result = df.fillna(df.max())
        expected = df.fillna(df.max().to_dict())
        tm.assert_frame_equal(result, expected)

    def test_fillna_dict_series_axis_1(self):
        # [FUNCTION DOC]
        # Verifies fillna along the columns axis using a series of row-wise maxima, in place, returns the frame itself and fills each missing value with its row maximum.
        # KNOWN ISSUE: the first non-inplace call result is immediately overwritten, so that line has no effect on the checks.
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            }
        )
        result = df.fillna(df.max(axis=1), axis=1)
        result = df.fillna(df.max(axis=1), axis=1, inplace=True)
        assert result is df
        expected = DataFrame(
            {
                "a": [1.0, 1.0, 2.0, 3.0, 4.0],
                "b": [1.0, 2.0, 3.0, 3.0, 4.0],
                "c": [1.0, 1.0, 2.0, 3.0, 4.0],
            }
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, expected)

    def test_fillna_dict_series_axis_1_mismatch_cols(self):
        # [FUNCTION DOC]
        # Verifies filling along the columns axis with a series of strings on a frame of mixed dtypes raises a value error that all columns must have the same dtype.
        df = DataFrame(
            {
                "a": ["abc", "def", np.nan, "ghi", "jkl"],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            }
        )
        with pytest.raises(ValueError, match="All columns must have the same dtype"):
            df.fillna(Series({"a": "abc", "b": "def", "c": "hij"}), axis=1)

    def test_fillna_dict_series_axis_1_value_mismatch_with_cols(self):
        # [FUNCTION DOC]
        # Verifies filling along the columns axis with a series of strings on a numeric frame raises a value error saying the value is not a suitable type to fill into the data.
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            }
        )
        with pytest.raises(ValueError, match=".* not a suitable type to fill into .*"):
            df.fillna(Series({"a": "abc", "b": "def", "c": "hij"}), axis=1)

    def test_fillna_dataframe(self):
        # [FUNCTION DOC]
        # GH#8377. Verifies fillna with a frame fills only positions whose index and column labels are shared with the filler frame; other positions remain missing.
        # GH#8377
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            },
            index=list("VWXYZ"),
        )

        # df2 may have different index and columns
        df2 = DataFrame(
            {
                "a": [np.nan, 10, 20, 30, 40],
                "b": [50, 60, 70, 80, 90],
                "foo": ["bar"] * 5,
            },
            index=list("VWXuZ"),
        )

        result = df.fillna(df2)

        # only those columns and indices which are shared get filled
        expected = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, 40],
                "b": [1, 2, 3, np.nan, 90],
                "c": [np.nan, 1, 2, 3, 4],
            },
            index=list("VWXYZ"),
        )

        tm.assert_frame_equal(result, expected)

    def test_fillna_columns(self):
        # [FUNCTION DOC]
        # Verifies forward fill along axis one equals the transposed forward fill, also after inserting an integer column (which is treated as float).
        arr = np.random.default_rng(2).standard_normal((10, 10))
        arr[:, ::2] = np.nan
        df = DataFrame(arr)

        result = df.ffill(axis=1)
        expected = df.T.ffill().T
        tm.assert_frame_equal(result, expected)

        df.insert(6, "foo", 5)
        result = df.ffill(axis=1)
        expected = df.astype(float).ffill(axis=1)
        tm.assert_frame_equal(result, expected)

    def test_fillna_invalid_value(self, float_frame):
        # [FUNCTION DOC]
        # Verifies fillna with a list or tuple raises a type error saying the value must be a scalar or dict, and filling a series with a frame raises a type error saying value must be a scalar, dict or series.
        # list
        msg = '"value" parameter must be a scalar or dict, but you passed a "{}"'
        with pytest.raises(TypeError, match=msg.format("list")):
            float_frame.fillna([1, 2])
        # tuple
        with pytest.raises(TypeError, match=msg.format("tuple")):
            float_frame.fillna((1, 2))
        # frame with series
        msg = (
            '"value" parameter must be a scalar, dict or Series, but you '
            'passed a "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            float_frame.iloc[:, 0].fillna(float_frame)

    def test_fillna_col_reordering(self):
        # [FUNCTION DOC]
        # Verifies forward fill keeps the original (unsorted) column order.
        cols = ["COL." + str(i) for i in range(5, 0, -1)]
        data = np.random.default_rng(2).random((20, 5))
        df = DataFrame(index=range(20), columns=cols, data=data)
        filled = df.ffill()
        assert df.columns.tolist() == filled.columns.tolist()

    def test_fill_empty(self, float_frame):
        # [FUNCTION DOC]
        # Verifies fillna on a frame with no columns returns an equal frame.
        df = float_frame.reindex(columns=[])
        result = df.fillna(value=0)
        tm.assert_frame_equal(result, df)

    def test_fillna_with_columns_and_limit(self):
        # [FUNCTION DOC]
        # GH40989. Verifies fillna along the columns axis with a scalar and a limit of one or two fills at most that many missing values per row, from the left.
        # GH40989
        df = DataFrame(
            [
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5],
                [np.nan, 3, np.nan, 4],
            ],
            columns=list("ABCD"),
        )
        result = df.fillna(axis=1, value=100, limit=1)
        result2 = df.fillna(axis=1, value=100, limit=2)

        expected = DataFrame(
            {
                "A": Series([100, 3, 100, 100], dtype="float64"),
                "B": [2, 4, np.nan, 3],
                "C": [np.nan, 100, np.nan, np.nan],
                "D": Series([0, 1, 5, 4], dtype="float64"),
            },
            index=[0, 1, 2, 3],
        )
        expected2 = DataFrame(
            {
                "A": Series([100, 3, 100, 100], dtype="float64"),
                "B": Series([2, 4, 100, 3], dtype="float64"),
                "C": [100, 100, np.nan, 100],
                "D": Series([0, 1, 5, 4], dtype="float64"),
            },
            index=[0, 1, 2, 3],
        )

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected2)

    def test_fillna_datetime_inplace(self):
        # [FUNCTION DOC]
        # GH#48863. Verifies in-place fillna with nan on datetime columns holding missing timestamps leaves the frame unchanged.
        # GH#48863
        df = DataFrame(
            {
                "date1": to_datetime(["2018-05-30", None]),
                "date2": to_datetime(["2018-09-30", None]),
            }
        )
        expected = df.copy()
        df.fillna(np.nan, inplace=True)
        tm.assert_frame_equal(df, expected)

    def test_fillna_inplace_with_columns_limit_and_value(self):
        # [FUNCTION DOC]
        # GH40989. Verifies in-place fillna along the columns axis with a value and limit returns the same frame object and gives the same data as the non-inplace result.
        # GH40989
        df = DataFrame(
            [
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5],
                [np.nan, 3, np.nan, 4],
            ],
            columns=list("ABCD"),
        )

        expected = df.fillna(axis=1, value=100, limit=1)
        assert expected is not df

        result = df.fillna(axis=1, value=100, limit=1, inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("val", [-1, {"x": -1, "y": -1}])
    def test_inplace_dict_update_view(self, val):
        # [FUNCTION DOC]
        # GH#47188. Parametrized over a scalar and a dict as value. Verifies in-place fillna updates the frame while a previously taken full slice view keeps the original data.
        # GH#47188
        df = DataFrame({"x": [np.nan, 2], "y": [np.nan, 2]})
        df_orig = df.copy()
        result_view = df[:]
        df.fillna(val, inplace=True)
        expected = DataFrame({"x": [-1, 2.0], "y": [-1.0, 2]})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(result_view, df_orig)

    def test_single_block_df_with_horizontal_axis(self):
        # [FUNCTION DOC]
        # GH 47713. Single-block float frame. Verifies fillna along the columns axis with limit one fills only the first missing value of each row.
        # GH 47713
        df = DataFrame(
            {
                "col1": [5, 0, np.nan, 10, np.nan],
                "col2": [7, np.nan, np.nan, 5, 3],
                "col3": [12, np.nan, 1, 2, 0],
                "col4": [np.nan, 1, 1, np.nan, 18],
            }
        )
        result = df.fillna(50, limit=1, axis=1)
        expected = DataFrame(
            [
                [5.0, 7.0, 12.0, 50.0],
                [0.0, 50.0, np.nan, 1.0],
                [50.0, np.nan, 1.0, 1.0],
                [10.0, 5.0, 2.0, 50.0],
                [50.0, 3.0, 0.0, 18.0],
            ],
            columns=["col1", "col2", "col3", "col4"],
        )
        tm.assert_frame_equal(result, expected)

    def test_fillna_with_multi_index_frame(self):
        # [FUNCTION DOC]
        # GH 47649. Frame with two-level column labels. Verifies a dict keyed by the first level fills all columns under it, and a more specific full key overrides the first-level value irrespective of dict order.
        # GH 47649
        pdf = DataFrame(
            {
                ("x", "a"): [np.nan, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, np.nan],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        expected = DataFrame(
            {
                ("x", "a"): [-1.0, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, -1.0],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        tm.assert_frame_equal(pdf.fillna({"x": -1}), expected)
        tm.assert_frame_equal(pdf.fillna({"x": -1, ("x", "b"): -2}), expected)

        expected = DataFrame(
            {
                ("x", "a"): [-1.0, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, -2.0],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        tm.assert_frame_equal(pdf.fillna({("x", "b"): -2, "x": -1}), expected)


def test_fillna_nonconsolidated_frame():
    # [FUNCTION DOC]
    # Issue 36495. Fills a non-consolidated frame (made by a pivot) with zero and verifies no missing values remain.
    # https://github.com/pandas-dev/pandas/issues/36495
    df = DataFrame(
        [
            [1, 1, 1, 1.0],
            [2, 2, 2, 2.0],
            [3, 3, 3, 3.0],
        ],
        columns=["i1", "i2", "i3", "f1"],
    )
    df_nonconsol = df.pivot(index="i1", columns="i2")
    result = df_nonconsol.fillna(0)
    assert result.isna().sum().sum() == 0


def test_fillna_nones_inplace():
    # [FUNCTION DOC]
    # GH 48480. Verifies in-place fillna of a dict on a frame of None values returns the same frame and fills with object dtype.
    # GH 48480
    df = DataFrame(
        [[None, None], [None, None]],
        columns=["A", "B"],
    )
    result = df.fillna(value={"A": 1, "B": 2}, inplace=True)
    assert result is df

    expected = DataFrame([[1, 2], [1, 2]], columns=["A", "B"], dtype=object)
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "data, expected_data, method, kwargs",
    (
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0],
            "ffill",
            {"limit_area": "outside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            range(5),
            range(5),
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside", "limit": 1},
        ),
    ),
)
def test_ffill_bfill_limit_area(data, expected_data, method, kwargs):
    # [FUNCTION DOC]
    # GH#56492. Parametrized over data, expected data, method (forward or backward fill) and keyword arguments (limit_area inside or outside, optional limit). Verifies fills are restricted to missing runs surrounded by valid values (inside) or to runs outside the valid values (outside), honoring the limit; all-missing and gapless data are unchanged.
    # GH#56492
    df = DataFrame(data)
    expected = DataFrame(expected_data)
    result = getattr(df, method)(**kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("test_frame", [True, False])
@pytest.mark.parametrize("dtype", ["float", "object"])
def test_fillna_with_none_object(test_frame, dtype):
    # [FUNCTION DOC]
    # GH#57723. Parametrized over frame versus series and float versus object dtype. Verifies fillna with value of none replaces missing values by none while keeping the dtype.
    # GH#57723
    obj = Series([1, np.nan, 3], dtype=dtype)
    if test_frame:
        obj = obj.to_frame()
    result = obj.fillna(value=None)
    expected = Series([1, None, 3], dtype=dtype)
    if test_frame:
        expected = expected.to_frame()
    tm.assert_equal(result, expected)


def test_fillna_out_of_bounds_datetime():
    # [FUNCTION DOC]
    # GH#61208. Verifies filling a nanosecond datetime column with a timestamp outside the nanosecond range raises an out-of-bounds datetime error with a message about overflow in nanosecond unit.
    # GH#61208
    df = DataFrame(
        {
            "datetime": date_range("1/1/2011", periods=3, freq="h", unit="ns"),
            "value": [1, 2, 3],
        }
    )
    df.iloc[0, 0] = None

    msg = "Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        df.fillna(Timestamp("0001-01-01"))
