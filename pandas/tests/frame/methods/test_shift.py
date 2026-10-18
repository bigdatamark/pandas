# [MODULE DOC]
# Tests for shift on DataFrame and Series: integer shifts along either axis, frequency-based shifts (offset,
# string, inferred, period and datetime indexes), fill values, dtype handling (integer, categorical, datetime with
# timezone, interval, boolean), multiple periods given as an iterable, and error cases. Many tests are
# parametrized with the frame-or-series fixture so they run for both container types.

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    NaT,
    Series,
    date_range,
    offsets,
)
import pandas._testing as tm


class TestDataFrameShift:
    # [CLASS DOC]
    # Test container for shift.
    def test_shift_axis1_with_valid_fill_value_one_array(self):
        # [FUNCTION DOC]
        # Shifting a float frame by one column with a fill value equals transposing, shifting rows and transposing back. A one-column nullable float frame
        # shifted along axis 1 becomes a column entirely of the fill value in the same extension dtype (the single-array path).
        # Case with axis=1 that does not go through the "len(arrays)>1" path
        #  in DataFrame.shift
        data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        res = df.shift(axis=1, periods=1, fill_value=12345)
        expected = df.T.shift(periods=1, fill_value=12345).T
        tm.assert_frame_equal(res, expected)

        # same but with a 1D ExtensionArray backing it
        df2 = df[[0]].astype("Float64")
        res2 = df2.shift(axis=1, periods=1, fill_value=12345)
        expected2 = DataFrame([12345] * 5, dtype="Float64")
        tm.assert_frame_equal(res2, expected2)

    def test_shift_disallow_freq_and_fill_value(self, frame_or_series):
        # [FUNCTION DOC]
        # Passing both a frequency and a fill value raises a value error stating they cannot be combined, along the rows and (for frames with a datetime column index) along axis 1.
        # Can't pass both!
        obj = frame_or_series(
            np.random.default_rng(2).standard_normal(5),
            index=date_range("1/1/2000", periods=5, freq="h"),
        )

        msg = "Passing a 'freq' together with a 'fill_value'"
        with pytest.raises(ValueError, match=msg):
            obj.shift(1, fill_value=1, freq="h")

        if frame_or_series is DataFrame:
            obj.columns = date_range("1/1/2000", periods=1, freq="h")
            with pytest.raises(ValueError, match=msg):
                obj.shift(1, axis=1, fill_value=1, freq="h")

    @pytest.mark.parametrize(
        "input_data, output_data",
        [(np.empty(shape=(0,)), []), (np.ones(shape=(2,)), [np.nan, 1.0])],
    )
    def test_shift_non_writable_array(self, input_data, output_data, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 21049). Parametrized over an empty and a two-element float array made read-only. Shifting by one works and yields float results (leading NaN); for frames an explicit single column is expected in the empty case.
        # GH21049 Verify whether non writable numpy array is shiftable
        input_data.setflags(write=False)

        result = frame_or_series(input_data).shift(1)
        if frame_or_series is not Series:
            # need to explicitly specify columns in the empty case
            expected = frame_or_series(
                output_data,
                index=range(len(output_data)),
                columns=range(1),
                dtype="float64",
            )
        else:
            expected = frame_or_series(output_data, dtype="float64")

        tm.assert_equal(result, expected)

    def test_shift_mismatched_freq(self, frame_or_series):
        # [FUNCTION DOC]
        # Shifting an hourly-indexed object by one period with a five-minute frequency moves the index by that offset; a four-hour frequency (a multiple of the base frequency, GH 1063) moves by four hours.
        ts = frame_or_series(
            np.random.default_rng(2).standard_normal(5),
            index=date_range("1/1/2000", periods=5, freq="h"),
        )

        result = ts.shift(1, freq="5min")
        exp_index = ts.index.shift(1, freq="5min")
        tm.assert_index_equal(result.index, exp_index)

        # GH#1063, multiple of same base
        result = ts.shift(1, freq="4h")
        exp_index = ts.index + offsets.Hour(4)
        tm.assert_index_equal(result.index, exp_index)

    @pytest.mark.parametrize(
        "obj",
        [
            Series([np.arange(5)]),
            date_range("1/1/2011", periods=24, freq="h"),
            Series(range(5), index=date_range("2017", periods=5)),
        ],
    )
    @pytest.mark.parametrize("shift_size", [0, 1, 2])
    def test_shift_always_copy(self, obj, shift_size, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 22397). Parametrized over a series, a datetime index and a date-indexed series, and shifts of 0, 1 and 2 (wrapped as frames when needed). The result is always a new object, even for a zero shift.
        # GH#22397
        if frame_or_series is not Series:
            obj = obj.to_frame()
        assert obj.shift(shift_size) is not obj

    def test_shift_object_non_scalar_fill(self):
        # [FUNCTION DOC]
        # A non-scalar fill value (empty list or array) raises a value error requiring a scalar, for series and frames, but is accepted for object dtype (an empty mapping is placed at the first position).
        # shift requires scalar fill_value except for object dtype
        ser = Series(range(3))
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            ser.shift(1, fill_value=[])

        df = ser.to_frame()
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            df.shift(1, fill_value=np.arange(3))

        obj_ser = ser.astype(object)
        result = obj_ser.shift(1, fill_value={})
        assert result[0] == {}

        obj_df = obj_ser.to_frame()
        result = obj_df.shift(1, fill_value={})
        assert result.iloc[0, 0] == {}

    def test_shift_int(self, datetime_frame, frame_or_series):
        # [FUNCTION DOC]
        # Shifting integer-typed data by one gives the same as shifting the float version (integer data upcasts to float with missing values).
        ts = tm.get_obj(datetime_frame, frame_or_series).astype(int)
        shifted = ts.shift(1)
        expected = ts.astype(float).shift(1)
        tm.assert_equal(shifted, expected)

    @pytest.mark.parametrize("dtype", ["int32", "int64"])
    def test_shift_32bit_take(self, frame_or_series, dtype):
        # [FUNCTION DOC]
        # Regression test (GH 8129). Parametrized over int32 and int64 dtypes. The shift amount is taken from an element of the data (value 1) and the result has missing first value and the remaining values moved down.
        # 32-bit taking
        # GH#8129
        index = date_range("2000-01-01", periods=5)
        arr = np.arange(5, dtype=dtype)
        s1 = frame_or_series(arr, index=index)
        p = arr[1]
        result = s1.shift(periods=p)
        expected = frame_or_series([np.nan, 0, 1, 2, 3], index=index)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("periods", [1, 2, 3, 4])
    def test_shift_preserve_freqstr(self, periods, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 21275). Parametrized over 1 to 4 periods. Shifting one period with a two-hour frequency (given positionally) moves the hourly index by two hours with data unchanged.
        # GH#21275
        obj = frame_or_series(
            range(periods),
            index=date_range("2016-1-1 00:00:00", periods=periods, freq="h"),
        )

        result = obj.shift(1, "2h")

        expected = frame_or_series(
            range(periods),
            index=date_range("2016-1-1 02:00:00", periods=periods, freq="h"),
        )
        tm.assert_equal(result, expected)

    def test_shift_dst(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 13926). Data of timezone-aware (US/Eastern, across a DST change) hourly timestamps: shifting by 0, 1 and -2 keeps the timezone-aware dtype, fills with NaT and equals the expected shifted values.
        # GH#13926
        dates = date_range(
            "2016-11-06", freq="h", periods=10, tz="US/Eastern", unit="ns"
        )
        obj = frame_or_series(dates)

        res = obj.shift(0)
        tm.assert_equal(res, obj)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

        res = obj.shift(1)
        exp_vals = [NaT, *dates.astype(object).values.tolist()[:9]]
        exp = frame_or_series(exp_vals)
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

        res = obj.shift(-2)
        exp_vals = [*dates.astype(object).values.tolist()[2:], NaT, NaT]
        exp = frame_or_series(exp_vals)
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

    @pytest.mark.parametrize("ex", [10, -10, 20, -20])
    def test_shift_dst_beyond(self, frame_or_series, ex):
        # [FUNCTION DOC]
        # Regression test (GH 13926). Parametrized over shifts of plus/minus 10 and 20 (at least the length of the data): all values become NaT with the timezone-aware dtype retained.
        # GH#13926
        dates = date_range(
            "2016-11-06", freq="h", periods=10, tz="US/Eastern", unit="ns"
        )
        obj = frame_or_series(dates)
        res = obj.shift(ex)
        exp = frame_or_series([NaT] * 10, dtype="datetime64[ns, US/Eastern]")
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

    def test_shift_by_zero(self, datetime_frame, frame_or_series):
        # [FUNCTION DOC]
        # Shifting by zero returns an object equal to the original.
        # shift by 0
        obj = tm.get_obj(datetime_frame, frame_or_series)
        unshifted = obj.shift(0)
        tm.assert_equal(unshifted, obj)

    def test_shift(self, datetime_frame):
        # [FUNCTION DOC]
        # Using the datetime fixture frame: shifting by 5 and -5 keeps the index and each column matches shifting the column alone; shifting forward then back restores all but the last five rows when missing values are dropped.
        # naive shift
        ser = datetime_frame["A"]

        shifted = datetime_frame.shift(5)
        tm.assert_index_equal(shifted.index, datetime_frame.index)

        shifted_ser = ser.shift(5)
        tm.assert_series_equal(shifted["A"], shifted_ser)

        shifted = datetime_frame.shift(-5)
        tm.assert_index_equal(shifted.index, datetime_frame.index)

        shifted_ser = ser.shift(-5)
        tm.assert_series_equal(shifted["A"], shifted_ser)

        unshifted = datetime_frame.shift(5).shift(-5)
        tm.assert_numpy_array_equal(
            unshifted.dropna().values, datetime_frame.values[:-5]
        )

        unshifted_ser = ser.shift(5).shift(-5)
        tm.assert_numpy_array_equal(unshifted_ser.dropna().values, ser.values[:-5])

    def test_shift_by_offset(self, datetime_frame, frame_or_series):
        # [FUNCTION DOC]
        # Shifting by 5 with a business day offset (as object or as the string B) keeps the length, and shifting back restores the original object; zero shift with an offset is the identity; the row at the original first date equals the row at the first date plus five business days in the shifted object.
        # shift by DateOffset
        obj = tm.get_obj(datetime_frame, frame_or_series)
        offset = offsets.BDay()

        shifted = obj.shift(5, freq=offset)
        assert len(shifted) == len(obj)
        unshifted = shifted.shift(-5, freq=offset)
        tm.assert_equal(unshifted, obj)

        shifted2 = obj.shift(5, freq="B")
        tm.assert_equal(shifted, shifted2)

        unshifted = obj.shift(0, freq=offset)
        tm.assert_equal(unshifted, obj)

        d = obj.index[0]
        shifted_d = d + offset * 5
        if frame_or_series is DataFrame:
            tm.assert_series_equal(obj.xs(d), shifted.xs(shifted_d), check_names=False)
        else:
            tm.assert_almost_equal(obj.at[d], shifted.at[shifted_d])

    def test_shift_with_periodindex(self, frame_or_series):
        # [FUNCTION DOC]
        # With a daily period index: plain shift and shift back keep the index and restore data except the last element; shifting with a daily frequency (string or offset, positional or keyword) gives equal results and round-trips; a weekly frequency raises a value error saying it does not match the period index frequency.
        # Shifting with PeriodIndex
        ps = DataFrame(
            np.arange(4, dtype=float), index=pd.period_range("2020-01-01", periods=4)
        )
        ps = tm.get_obj(ps, frame_or_series)

        shifted = ps.shift(1)
        unshifted = shifted.shift(-1)
        tm.assert_index_equal(shifted.index, ps.index)
        tm.assert_index_equal(unshifted.index, ps.index)
        if frame_or_series is DataFrame:
            tm.assert_numpy_array_equal(
                unshifted.iloc[:, 0].dropna().values, ps.iloc[:-1, 0].values
            )
        else:
            tm.assert_numpy_array_equal(unshifted.dropna().values, ps.values[:-1])

        shifted2 = ps.shift(1, "D")
        shifted3 = ps.shift(1, offsets.Day())
        tm.assert_equal(shifted2, shifted3)
        tm.assert_equal(ps, shifted2.shift(-1, "D"))

        msg = "does not match PeriodIndex freq"
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq="W")

        # legacy support
        shifted4 = ps.shift(1, freq="D")
        tm.assert_equal(shifted2, shifted4)

        shifted5 = ps.shift(1, freq=offsets.Day())
        tm.assert_equal(shifted5, shifted4)

    def test_shift_other_axis(self):
        # [FUNCTION DOC]
        # Regression test (GH 6371). Shifting by one along axis 1 puts a NaN column first and moves the remaining columns right, dropping the last.
        # shift other axis
        # GH#6371
        df = DataFrame(np.random.default_rng(2).random((10, 5)))
        expected = pd.concat(
            [DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:-1]],
            ignore_index=True,
            axis=1,
        )
        result = df.shift(1, axis=1)
        tm.assert_frame_equal(result, expected)

    def test_shift_named_axis(self):
        # [FUNCTION DOC]
        # Same as the other-axis test but with the axis given by name (columns).
        # shift named axis
        df = DataFrame(np.random.default_rng(2).random((10, 5)))
        expected = pd.concat(
            [DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:-1]],
            ignore_index=True,
            axis=1,
        )
        result = df.shift(1, axis="columns")
        tm.assert_frame_equal(result, expected)

    def test_shift_other_axis_with_freq(self, datetime_frame):
        # [FUNCTION DOC]
        # Regression test (GH 47039). On the transposed datetime frame, shifting columns by 5 with a business day offset along axis 1 and back restores the original.
        obj = datetime_frame.T
        offset = offsets.BDay()

        # GH#47039
        shifted = obj.shift(5, freq=offset, axis=1)
        assert len(shifted) == len(obj)
        unshifted = shifted.shift(-5, freq=offset, axis=1)
        tm.assert_equal(unshifted, obj)

    def test_shift_bool(self):
        # [FUNCTION DOC]
        # Shifting a boolean frame by one gives an object frame with NaN in the first row and the original booleans shifted down.
        df = DataFrame({"high": [True, False], "low": [False, False]})
        rs = df.shift(1)
        xp = DataFrame(
            np.array([[np.nan, np.nan], [True, False]], dtype=object),
            columns=["high", "low"],
        )
        tm.assert_frame_equal(rs, xp)

    def test_shift_categorical1(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 9416). Categorical data: shifting by 1 and back restores all but the last value; the shifted codes have -1 (missing) at the vacated positions and shifted codes otherwise, for shifts of 1 and -2; categories are unchanged.
        # GH#9416
        obj = frame_or_series(["a", "b", "c", "d"], dtype="category")

        rt = obj.shift(1).shift(-1)
        tm.assert_equal(obj.iloc[:-1], rt.dropna())

        def get_cat_values(ndframe):
            # [FUNCTION DOC]
            # HELPER: returns the categorical array stored in the first internal block of the given series or frame (a block-level accessor used because one-dimensional categoricals are not available for frames).
            # For Series we could just do ._values; for DataFrame
            #  we may be able to do this if we ever have 2D Categoricals
            return ndframe._mgr.blocks[0].values

        cat = get_cat_values(obj)

        sp1 = obj.shift(1)
        tm.assert_index_equal(obj.index, sp1.index)
        assert np.all(get_cat_values(sp1).codes[:1] == -1)
        assert np.all(cat.codes[:-1] == get_cat_values(sp1).codes[1:])

        sn2 = obj.shift(-2)
        tm.assert_index_equal(obj.index, sn2.index)
        assert np.all(get_cat_values(sn2).codes[-2:] == -1)
        assert np.all(cat.codes[2:] == get_cat_values(sn2).codes[:-2])

        tm.assert_index_equal(cat.categories, get_cat_values(sp1).categories)
        tm.assert_index_equal(cat.categories, get_cat_values(sn2).categories)

    def test_shift_categorical(self):
        # [FUNCTION DOC]
        # Regression test (GH 9416). A frame of two categorical columns shifted by one equals the frame of each column shifted individually.
        # GH#9416
        s1 = Series(["a", "b", "c"], dtype="category")
        s2 = Series(["A", "B", "C"], dtype="category")
        df = DataFrame({"one": s1, "two": s2})
        rs = df.shift(1)
        xp = DataFrame({"one": s1.shift(1), "two": s2.shift(1)})
        tm.assert_frame_equal(rs, xp)

    def test_shift_categorical_fill_value(self, frame_or_series):
        # [FUNCTION DOC]
        # Shifting categorical data with a fill value that is an existing category fills with it; a value that is not a category raises a type error saying a new category cannot be set.
        ts = frame_or_series(["a", "b", "c", "d"], dtype="category")
        res = ts.shift(1, fill_value="a")
        expected = frame_or_series(
            pd.Categorical(
                ["a", "a", "b", "c"], categories=["a", "b", "c", "d"], ordered=False
            )
        )
        tm.assert_equal(res, expected)

        # check for incorrect fill_value
        msg = r"Cannot setitem on a Categorical with a new category \(f\)"
        with pytest.raises(TypeError, match=msg):
            ts.shift(1, fill_value="f")

    def test_shift_fill_value(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 24128). Float data: shift of 1 and 2 with a fill of zero fills the leading values. Integer data keeps its dtype when filled with an integer zero and values match expectations.
        # GH#24128
        dti = date_range("1/1/2000", periods=5, freq="h")

        ts = frame_or_series([1.0, 2.0, 3.0, 4.0, 5.0], index=dti)
        exp = frame_or_series([0.0, 1.0, 2.0, 3.0, 4.0], index=dti)
        # check that fill value works
        result = ts.shift(1, fill_value=0.0)
        tm.assert_equal(result, exp)

        exp = frame_or_series([0.0, 0.0, 1.0, 2.0, 3.0], index=dti)
        result = ts.shift(2, fill_value=0.0)
        tm.assert_equal(result, exp)

        ts = frame_or_series([1, 2, 3])
        res = ts.shift(2, fill_value=0)
        assert tm.get_dtype(res) == tm.get_dtype(ts)

        # retain integer dtype
        obj = frame_or_series([1, 2, 3, 4, 5], index=dti)
        exp = frame_or_series([0, 1, 2, 3, 4], index=dti)
        result = obj.shift(1, fill_value=0)
        tm.assert_equal(result, exp)

        exp = frame_or_series([0, 0, 1, 2, 3], index=dti)
        result = obj.shift(2, fill_value=0)
        tm.assert_equal(result, exp)

    def test_shift_empty(self):
        # [FUNCTION DOC]
        # Regression test (GH 8019). Shifting a frame with an empty column returns a frame equal to the original.
        # Regression test for GH#8019
        df = DataFrame({"foo": []})
        rs = df.shift(-1)

        tm.assert_frame_equal(df, rs)

    def test_shift_duplicate_columns(self):
        # [FUNCTION DOC]
        # Regression test (GH 9092). Position-based shifting of column i by i plus one rows gives identical results with unique labels, all-identical labels and partly duplicated labels; the base case has 1 to 5 missing values per column.
        # GH#9092; verify that position-based shifting works
        # in the presence of duplicate columns
        column_lists = [list(range(5)), [1] * 5, [1, 1, 2, 2, 1]]
        data = np.random.default_rng(2).standard_normal((20, 5))

        shifted = []
        for columns in column_lists:
            df = DataFrame(data.copy(), columns=columns)
            for s in range(5):
                df.iloc[:, s] = df.iloc[:, s].shift(s + 1)
            df.columns = range(5)
            shifted.append(df)

        # sanity check the base case
        nulls = shifted[0].isna().sum()
        tm.assert_series_equal(nulls, Series(range(1, 6), dtype="int64"))

        # check all answers are the same
        tm.assert_frame_equal(shifted[0], shifted[1])
        tm.assert_frame_equal(shifted[0], shifted[2])

    def test_shift_axis1_multiple_blocks(self):
        # [FUNCTION DOC]
        # Regression test (GH 35488). A frame with two internal blocks shifted by plus/minus 2 along axis 1 equals taking the reordered columns with leading or trailing columns set to NaN and cast to float (explicit casting done through relabeling since labels are duplicated).
        # GH#35488
        df1 = DataFrame(np.random.default_rng(2).integers(1000, size=(5, 3)))
        df2 = DataFrame(np.random.default_rng(2).integers(1000, size=(5, 2)))
        df3 = pd.concat([df1, df2], axis=1)
        assert len(df3._mgr.blocks) == 2

        result = df3.shift(2, axis=1)

        expected = df3.take([-1, -1, 0, 1, 2], axis=1)
        # Explicit cast to float to avoid implicit cast when setting nan.
        # Column names aren't unique, so directly calling `expected.astype` won't work.
        expected = expected.pipe(
            lambda df: (
                df.set_axis(range(df.shape[1]), axis=1)
                .astype({0: "float", 1: "float"})
                .set_axis(df.columns, axis=1)
            )
        )
        expected.iloc[:, :2] = np.nan
        expected.columns = df3.columns

        tm.assert_frame_equal(result, expected)

        # Case with periods < 0
        # rebuild df3 because `take` call above consolidated
        df3 = pd.concat([df1, df2], axis=1)
        assert len(df3._mgr.blocks) == 2
        result = df3.shift(-2, axis=1)

        expected = df3.take([2, 3, 4, -1, -1], axis=1)
        # Explicit cast to float to avoid implicit cast when setting nan.
        # Column names aren't unique, so directly calling `expected.astype` won't work.
        expected = expected.pipe(
            lambda df: (
                df.set_axis(range(df.shape[1]), axis=1)
                .astype({3: "float", 4: "float"})
                .set_axis(df.columns, axis=1)
            )
        )
        expected.iloc[:, -2:] = np.nan
        expected.columns = df3.columns

        tm.assert_frame_equal(result, expected)

    def test_shift_axis1_multiple_blocks_with_int_fill(self):
        # [FUNCTION DOC]
        # Regression test (GH 42719). Same as multiple-block shifting but with an integer zero fill value: vacated columns become that integer value for shifts of 2 and -2, with two blocks present.
        # GH#42719
        rng = np.random.default_rng(2)
        df1 = DataFrame(rng.integers(1000, size=(5, 3), dtype=int))
        df2 = DataFrame(rng.integers(1000, size=(5, 2), dtype=int))
        df3 = pd.concat([df1.iloc[:4, 1:3], df2.iloc[:4, :]], axis=1)
        result = df3.shift(2, axis=1, fill_value=np.int_(0))
        assert len(df3._mgr.blocks) == 2

        expected = df3.take([-1, -1, 0, 1], axis=1)
        expected.iloc[:, :2] = np.int_(0)
        expected.columns = df3.columns

        tm.assert_frame_equal(result, expected)

        # Case with periods < 0
        df3 = pd.concat([df1.iloc[:4, 1:3], df2.iloc[:4, :]], axis=1)
        result = df3.shift(-2, axis=1, fill_value=np.int_(0))
        assert len(df3._mgr.blocks) == 2

        expected = df3.take([2, 3, -1, -1], axis=1)
        expected.iloc[:, -2:] = np.int_(0)
        expected.columns = df3.columns

        tm.assert_frame_equal(result, expected)

    def test_period_index_frame_shift_with_freq(self, frame_or_series):
        # [FUNCTION DOC]
        # With a daily period index, shifting with the inferred frequency by one and back restores the original; shifting with a day frequency (string or offset) and no periods equals the shift by one.
        ps = DataFrame(range(4), index=pd.period_range("2020-01-01", periods=4))
        ps = tm.get_obj(ps, frame_or_series)

        shifted = ps.shift(1, freq="infer")
        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_equal(unshifted, ps)

        shifted2 = ps.shift(freq="D")
        tm.assert_equal(shifted, shifted2)

        shifted3 = ps.shift(freq=offsets.Day())
        tm.assert_equal(shifted, shifted3)

    def test_datetime_frame_shift_with_freq(self, datetime_frame, frame_or_series):
        # [FUNCTION DOC]
        # For the datetime fixture: shifting with inferred frequency by one and back restores the original, and shifting with the index's own frequency equals the inferred shift. For an index with no frequency set (rebuilt from raw values) the inferred shift still works, the expected index has no frequency, and shifting back restores it.
        dtobj = tm.get_obj(datetime_frame, frame_or_series)
        shifted = dtobj.shift(1, freq="infer")
        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_equal(dtobj, unshifted)

        shifted2 = dtobj.shift(freq=dtobj.index.freq)
        tm.assert_equal(shifted, shifted2)

        inferred_ts = DataFrame(
            datetime_frame.values,
            Index(np.asarray(datetime_frame.index)),
            columns=datetime_frame.columns,
        )
        inferred_ts = tm.get_obj(inferred_ts, frame_or_series)
        shifted = inferred_ts.shift(1, freq="infer")
        expected = dtobj.shift(1, freq="infer")
        expected.index = expected.index._with_freq(None)
        tm.assert_equal(shifted, expected)

        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_equal(unshifted, inferred_ts)

    def test_period_index_frame_shift_with_freq_error(self, frame_or_series):
        # [FUNCTION DOC]
        # A monthly frequency on a daily period index raises a value error saying the given frequency does not match the index frequency.
        ps = DataFrame(range(4), index=pd.period_range("2020-01-01", periods=4))
        ps = tm.get_obj(ps, frame_or_series)
        msg = "Given freq M does not match PeriodIndex freq D"
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq="M")

    def test_datetime_frame_shift_with_freq_error(
        self, datetime_frame, frame_or_series
    ):
        # [FUNCTION DOC]
        # Shifting an unevenly spaced (no frequency) datetime selection with the inferred frequency raises a value error saying the frequency was not set and cannot be inferred.
        dtobj = tm.get_obj(datetime_frame, frame_or_series)
        no_freq = dtobj.iloc[[0, 5, 7]]
        msg = "Freq was not set in the index hence cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            no_freq.shift(freq="infer")

    def test_shift_dt64values_int_fill_deprecated(self):
        # [FUNCTION DOC]
        # Regression test (GH 31971). Filling datetime values with an integer raises a type error that the value should be a datetime-compatible type, for series and frames. For frames with two datetime columns shifted along axis 1 with an integer fill (consolidated or not) the result is an object frame with the fill in the vacated column.
        # KNOWN ISSUE: the test name says deprecated, but the first part now expects a type error rather than a deprecation warning.
        # GH#31971
        ser = Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])

        with pytest.raises(TypeError, match="value should be a"):
            ser.shift(1, fill_value=0)

        df = ser.to_frame()
        with pytest.raises(TypeError, match="value should be a"):
            df.shift(1, fill_value=0)

        # axis = 1
        df2 = DataFrame({"A": ser, "B": ser})
        df2._consolidate_inplace()

        result = df2.shift(1, axis=1, fill_value=0)
        expected = DataFrame({"A": [0, 0], "B": df2["A"]})
        tm.assert_frame_equal(result, expected)

        # same thing but not consolidated; pre-2.0 we got different behavior
        df3 = DataFrame({"A": ser})
        df3["B"] = ser
        assert len(df3._mgr.blocks) == 2
        result = df3.shift(1, axis=1, fill_value=0)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "as_cat",
        [
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
            False,
        ],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            date_range("2020-01-01", periods=2),
            date_range("2020-01-01", periods=2, tz="US/Pacific"),
            pd.period_range("2020-01-01", periods=2, freq="D"),
            pd.timedelta_range("2020 Days", periods=2, freq="D"),
            pd.interval_range(0, 3, periods=2),
            pytest.param(
                pd.array([1, 2], dtype="Int64"),
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
            pytest.param(
                pd.array([1, 2], dtype="Float32"),
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_shift_dt64values_axis1_invalid_fill(self, vals, as_cat):
        # [FUNCTION DOC]
        # Regression test (GH 44564). Parametrized over several extension value kinds (datetime, tz-aware datetime, period, timedelta, interval, nullable integer and float; the last two and the categorical versions marked as expected failures because of an element-compatibility check that wrongly always passes) and categorical or not. Shifting by -1 along axis 1 with a string fill converts the vacated column to the fill value (object), for single, consolidated two-column and non-consolidated two-column frames.
        # GH#44564
        ser = Series(vals)
        if as_cat:
            ser = ser.astype("category")

        df = DataFrame({"A": ser})
        result = df.shift(-1, axis=1, fill_value="foo")
        expected = DataFrame({"A": ["foo", "foo"]})
        tm.assert_frame_equal(result, expected)

        # same thing but multiple blocks
        df2 = DataFrame({"A": ser, "B": ser})
        df2._consolidate_inplace()

        result = df2.shift(-1, axis=1, fill_value="foo")
        expected = DataFrame({"A": df2["B"], "B": ["foo", "foo"]})
        tm.assert_frame_equal(result, expected)

        # same thing but not consolidated
        df3 = DataFrame({"A": ser})
        df3["B"] = ser
        assert len(df3._mgr.blocks) == 2
        result = df3.shift(-1, axis=1, fill_value="foo")
        tm.assert_frame_equal(result, expected)

    def test_shift_axis1_categorical_columns(self):
        # [FUNCTION DOC]
        # Regression test (GH 38434). A frame with categorical column labels shifted along axis 1 by default (1) and by 2 moves the columns right and fills vacated ones with NaN, keeping the categorical labels.
        # GH#38434
        ci = CategoricalIndex(["a", "b", "c"])
        df = DataFrame(
            {"a": [1, 3], "b": [2, 4], "c": [5, 6]}, index=ci[:-1], columns=ci
        )
        result = df.shift(axis=1)

        expected = DataFrame(
            {"a": [np.nan, np.nan], "b": [1, 3], "c": [2, 4]}, index=ci[:-1], columns=ci
        )
        tm.assert_frame_equal(result, expected)

        # periods != 1
        result = df.shift(2, axis=1)
        expected = DataFrame(
            {"a": [np.nan, np.nan], "b": [np.nan, np.nan], "c": [1, 3]},
            index=ci[:-1],
            columns=ci,
        )
        tm.assert_frame_equal(result, expected)

    def test_shift_axis1_many_periods(self):
        # [FUNCTION DOC]
        # Regression test (GH 44978). Shifting by more than the number of columns (6 or -6 on 3 columns) along axis 1 with fill None gives all NaN.
        # GH#44978 periods > len(columns)
        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        shifted = df.shift(6, axis=1, fill_value=None)

        expected = df * np.nan
        tm.assert_frame_equal(shifted, expected)

        shifted2 = df.shift(-6, axis=1, fill_value=None)
        tm.assert_frame_equal(shifted2, expected)

    def test_shift_with_offsets_freq(self):
        # [FUNCTION DOC]
        # Shifting a daily-indexed frame with a month-start frequency string moves every index entry to the next month start, so all entries equal the same date.
        df = DataFrame({"x": [1, 2, 3]}, index=date_range("2000", periods=3))
        shifted = df.shift(freq="1MS")
        expected = DataFrame(
            {"x": [1, 2, 3]},
            index=date_range(start="02/01/2000", end="02/01/2000", periods=3),
        )
        tm.assert_frame_equal(shifted, expected)

    def test_shift_with_iterable_basic_functionality(self):
        # [FUNCTION DOC]
        # Regression test (GH 44424). Passing a list of periods (0, 1, 2) returns a wide frame with columns named by original label and period (label underscore period) side by side in period order, with NaN-filled shifted values (float).
        # GH#44424
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        shifts = [0, 1, 2]

        df = DataFrame(data)
        shifted = df.shift(shifts)

        expected = DataFrame(
            {
                "a_0": [1, 2, 3],
                "b_0": [4, 5, 6],
                "a_1": [np.nan, 1.0, 2.0],
                "b_1": [np.nan, 4.0, 5.0],
                "a_2": [np.nan, np.nan, 1.0],
                "b_2": [np.nan, np.nan, 4.0],
            }
        )
        tm.assert_frame_equal(expected, shifted)

    def test_shift_with_iterable_series(self):
        # [FUNCTION DOC]
        # Regression test (GH 44424). Shifting a series by a list of periods gives the same as shifting a one-column frame by that list.
        # GH#44424
        data = {"a": [1, 2, 3]}
        shifts = [0, 1, 2]

        df = DataFrame(data)
        s = df["a"]
        tm.assert_frame_equal(s.shift(shifts), df.shift(shifts))

    def test_shift_with_iterable_freq_and_fill_value(self):
        # [FUNCTION DOC]
        # Regression test (GH 44424). A one-element list of periods gives the same as the integer shift (after renaming the string column names back to integers), for both fill value and frequency arguments.
        # GH#44424
        df = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=date_range("1/1/2000", periods=5, freq="h"),
        )

        tm.assert_frame_equal(
            # rename because shift with an iterable leads to str column names
            df.shift([1], fill_value=1).rename(columns=lambda x: int(x[0])),
            df.shift(1, fill_value=1),
        )

        tm.assert_frame_equal(
            df.shift([1], freq="h").rename(columns=lambda x: int(x[0])),
            df.shift(1, freq="h"),
        )

    def test_shift_with_iterable_check_other_arguments(self):
        # [FUNCTION DOC]
        # Regression test (GH 44424). Verifies the suffix option names columns with the suffix before the period. Error cases: a multi-shift on axis 1 raises a value error; a non-integer period raises a type error; an empty list raises a value error; a suffix with an integer period raises a value error.
        # GH#44424
        data = {"a": [1, 2], "b": [4, 5]}
        shifts = [0, 1]
        df = DataFrame(data)

        # test suffix
        shifted = df[["a"]].shift(shifts, suffix="_suffix")
        expected = DataFrame({"a_suffix_0": [1, 2], "a_suffix_1": [np.nan, 1.0]})
        tm.assert_frame_equal(shifted, expected)

        # check bad inputs when doing multiple shifts
        msg = "If `periods` contains multiple shifts, `axis` cannot be 1."
        with pytest.raises(ValueError, match=msg):
            df.shift(shifts, axis=1)

        msg = "Periods must be integer, but s is <class 'str'>."
        with pytest.raises(TypeError, match=msg):
            df.shift(["s"])

        msg = "If `periods` is an iterable, it cannot be empty."
        with pytest.raises(ValueError, match=msg):
            df.shift([])

        msg = "Cannot specify `suffix` if `periods` is an int."
        with pytest.raises(ValueError, match=msg):
            df.shift(1, suffix="fails")

    def test_shift_axis_one_empty(self):
        # [FUNCTION DOC]
        # Regression test (GH 57301). Shifting an empty frame along axis 1 returns an equal empty frame.
        # GH#57301
        df = DataFrame()
        result = df.shift(1, axis=1)
        tm.assert_frame_equal(result, df)

    def test_shift_with_offsets_freq_empty(self):
        # [FUNCTION DOC]
        # Regression test (GH 60102). A frame with no columns and a daily index shifted by a day offset has its index moved forward one day.
        # GH#60102
        dates = date_range("2020-01-01", periods=3, freq="D")
        offset = offsets.Day()
        shifted_dates = dates + offset
        df = DataFrame(index=dates)
        df_shifted = DataFrame(index=shifted_dates)
        result = df.shift(freq=offset)
        tm.assert_frame_equal(result, df_shifted)

    def test_series_shift_interval_preserves_closed(self):
        # [FUNCTION DOC]
        # Regression test (GH 60389). Shifting a series of right-closed intervals keeps the closed side on the remaining intervals with a leading NaN.
        # GH#60389
        ser = Series(
            [pd.Interval(1, 2, closed="right"), pd.Interval(2, 3, closed="right")]
        )
        result = ser.shift(1)
        expected = Series([np.nan, pd.Interval(1, 2, closed="right")])
        tm.assert_series_equal(result, expected)

    def test_shift_invalid_fill_value_deprecation(self):
        # [FUNCTION DOC]
        # Regression test (GH 53802). Filling integer or boolean data (frame or either column) with a string or with NaT emits the Pandas4 deprecation warning about shifting with a fill value that cannot be held without changing dtype.
        # GH#53802
        df = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [True, False, True],
            }
        )

        msg = "shifting with a fill value that cannot"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            df.shift(1, fill_value="foo")

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            df["a"].shift(1, fill_value="foo")
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            df["b"].shift(1, fill_value="foo")

        # An incompatible null value
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            df.shift(1, fill_value=NaT)
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            df["a"].shift(1, fill_value=NaT)
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            df["b"].shift(1, fill_value=NaT)

    def test_shift_dt_index_multiple_periods_unsorted(self):
        # [FUNCTION DOC]
        # Regression test for pull request 62843. A frame with an unsorted daily index shifted by a list of periods (1, 2) with a day frequency yields a frame with columns for each period and the union of the shifted index labels, with NaN where a label has no value from that shift.
        # https://github.com/pandas-dev/pandas/pull/62843
        values = date_range("1/1/2000", periods=4, freq="D")
        df = DataFrame({"a": [1, 2]}, index=[values[1], values[0]])
        result = df.shift(periods=[1, 2], freq="D")
        expected = DataFrame(
            {
                "a_1": [1.0, 2.0, np.nan],
                "a_2": [2.0, np.nan, 1.0],
            },
            index=[values[2], values[1], values[3]],
        )
        tm.assert_frame_equal(result, expected)
