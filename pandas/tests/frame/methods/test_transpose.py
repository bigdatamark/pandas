# [MODULE DOC]
# Tests for DataFrame.transpose and its T alias: dtype preservation and inference rules for timezone-aware,
# interval, unsigned, extension and mixed data, view behavior, and memory layout.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    Series,
    Timestamp,
    bdate_range,
    date_range,
    timedelta_range,
)
import pandas._testing as tm


class TestTranspose:
    # [CLASS DOC]
    # Test container for transpose.
    def test_transpose_td64_intervals(self):
        # [FUNCTION DOC]
        # Regression test (GH 44917). A one-column frame of timedelta intervals (with a missing value inserted) is transposed;
        # after relabeling columns with consecutive integers it equals a frame whose columns are each single element slices of the interval index.
        # GH#44917
        tdi = timedelta_range("0 Days", "3 Days")
        ii = IntervalIndex.from_breaks(tdi)
        ii = ii.insert(-1, np.nan)
        df = DataFrame(ii)

        result = df.T
        result.columns = Index(list(range(len(ii))))
        expected = DataFrame({i: ii[i : i + 1] for i in range(len(ii))})
        tm.assert_frame_equal(result, expected)

    def test_transpose_empty_preserves_datetimeindex(self):
        # [FUNCTION DOC]
        # Regression test (GH 41382). For an empty frame with a nanosecond datetime index, summing the transpose and summing along axis 1 both
        # give an empty datetime index with that dtype and no frequency.
        # GH#41382
        dti = DatetimeIndex([], dtype="M8[ns]")
        df = DataFrame(index=dti)

        expected = DatetimeIndex([], dtype="datetime64[ns]", freq=None)

        result1 = df.T.sum().index
        result2 = df.sum(axis=1).index

        tm.assert_index_equal(result1, expected)
        tm.assert_index_equal(result2, expected)

    def test_transpose_tzaware_1col_single_tz(self):
        # [FUNCTION DOC]
        # Regression test (GH 26825). A one-column UTC frame keeps the timezone-aware dtype in all columns after transposing.
        # GH#26825
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")

        df = DataFrame(dti)
        assert (df.dtypes == dti.dtype).all()
        res = df.T
        assert (res.dtypes == dti.dtype).all()

    def test_transpose_tzaware_2col_single_tz(self):
        # [FUNCTION DOC]
        # Regression test (GH 26825). Two columns sharing the same UTC dtype keep that dtype after transposing.
        # GH#26825
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")

        df3 = DataFrame({"A": dti, "B": dti})
        assert (df3.dtypes == dti.dtype).all()
        res3 = df3.T
        assert (res3.dtypes == dti.dtype).all()

    def test_transpose_tzaware_2col_mixed_tz(self):
        # [FUNCTION DOC]
        # Regression test (GH 26825). Two columns with different timezones (UTC and US/Pacific) transpose to object dtype columns; transposing
        # back equals the original frame cast to object.
        # GH#26825
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        dti2 = dti.tz_convert("US/Pacific")

        df4 = DataFrame({"A": dti, "B": dti2})
        assert (df4.dtypes == [dti.dtype, dti2.dtype]).all()
        assert (df4.T.dtypes == object).all()
        tm.assert_frame_equal(df4.T.T, df4.astype(object))

    @pytest.mark.parametrize("tz", [None, "America/New_York"])
    def test_transpose_preserves_dtindex_equality_with_dst(self, tz):
        # [FUNCTION DOC]
        # Regression test (GH 19970). Parametrized over naive and New York timezone. A frame indexed by a four-hourly range crossing a
        # daylight-saving change, transposed and compared with itself elementwise, gives an all-true frame indexed by the labels with the datetime columns.
        # GH#19970
        idx = date_range("20161101", "20161130", freq="4h", tz=tz)
        df = DataFrame({"a": range(len(idx)), "b": range(len(idx))}, index=idx)
        result = df.T == df.T
        expected = DataFrame(True, index=list("ab"), columns=idx)
        tm.assert_frame_equal(result, expected)

    def test_transpose_object_to_tzaware_mixed_tz(self):
        # [FUNCTION DOC]
        # Regression test (GH 26825). A frame built from two rows with different timezones has object dtype columns, which remain object after transposing.
        # GH#26825
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        dti2 = dti.tz_convert("US/Pacific")

        # mixed all-tzaware dtypes
        df2 = DataFrame([dti, dti2])
        assert (df2.dtypes == object).all()
        res2 = df2.T
        assert (res2.dtypes == object).all()

    def test_transpose_uint64(self):
        # [FUNCTION DOC]
        # Transposing an unsigned 64-bit frame with values above the signed range equals the frame built from the transposed raw values with the original column labels as index (no precision loss or dtype change).
        df = DataFrame(
            {"A": np.arange(3), "B": [2**63, 2**63 + 5, 2**63 + 10]},
            dtype=np.uint64,
        )
        result = df.T
        expected = DataFrame(df.values.T)
        expected.index = ["A", "B"]
        tm.assert_frame_equal(result, expected)

    def test_transpose_float(self, float_frame):
        # [FUNCTION DOC]
        # Every cell of the transposed float fixture frame equals the corresponding original cell (missing values matching missing).
        frame = float_frame
        dft = frame.T
        for idx, series in dft.items():
            for col, value in series.items():
                if np.isnan(value):
                    assert np.isnan(frame[col][idx])
                else:
                    assert value == frame[col][idx]

    def test_transpose_mixed(self):
        # [FUNCTION DOC]
        # A frame with float, string and datetime columns transposes to columns that all have object dtype.
        # mixed type
        mixed = DataFrame(
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
                "D": bdate_range("1/1/2009", periods=5),
            },
            index=Index(["a", "b", "c", "d", "e"], dtype=object),
        )

        mixed_T = mixed.T
        for col, s in mixed_T.items():
            assert s.dtype == np.object_

    def test_transpose_get_view(self, float_frame):
        # [FUNCTION DOC]
        # Assigns a value to a slice of columns of the transposed float frame and expects the original frame's values for the corresponding rows to be unchanged, i.e. the transpose does not write through.
        # KNOWN ISSUE: the name suggests the transpose yields a view, but the assertion verifies the opposite (no change to the source).
        dft = float_frame.T
        dft.iloc[:, 5:10] = 5
        assert (float_frame.values[5:10] != 5).all()

    def test_transpose_get_view_dt64tzget_view(self):
        # [FUNCTION DOC]
        # Transposes a single-block timezone-aware frame built from a two-dimensional datetime array. The result must also have a single block, and its
        # block's underlying array must share memory with the original's (the transpose is a view for this case).
        dti = date_range("2016-01-01", periods=6, tz="US/Pacific")
        arr = dti._data.reshape(3, 2)
        df = DataFrame(arr)
        assert df._mgr.nblocks == 1

        result = df.T
        assert result._mgr.nblocks == 1

        rtrip = result._mgr.blocks[0].values
        assert np.shares_memory(df._mgr.blocks[0].values._ndarray, rtrip._ndarray)

    def test_transpose_not_inferring_dt(self):
        # [FUNCTION DOC]
        # Regression test (GH 51546). Transposing an object-dtype frame of timestamps keeps object dtype and does not infer a datetime dtype.
        # GH#51546
        df = DataFrame(
            {
                "a": [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            },
            dtype=object,
        )
        result = df.T
        expected = DataFrame(
            [[Timestamp("2019-12-31"), Timestamp("2019-12-31")]],
            index=["a"],
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    def test_transpose_not_inferring_dt_mixed_blocks(self):
        # [FUNCTION DOC]
        # Regression test (GH 51546). Frame with an explicit object column and an inferred datetime column of the same timestamps transposes to an object-dtype frame of timestamps.
        # GH#51546
        df = DataFrame(
            {
                "a": Series(
                    [Timestamp("2019-12-31"), Timestamp("2019-12-31")], dtype=object
                ),
                "b": [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            }
        )
        result = df.T
        expected = DataFrame(
            [
                [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
                [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            ],
            index=["a", "b"],
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype1", ["Int64", "Float64"])
    @pytest.mark.parametrize("dtype2", ["Int64", "Float64"])
    def test_transpose(self, dtype1, dtype2):
        # [FUNCTION DOC]
        # Regression test (GH 57315). Parametrized over pairs of nullable integer or float dtypes for two columns. After transposing, every internal block's
        # data must be Fortran contiguous (using the masked array's data when both dtypes are equal, otherwise the numpy object array directly).
        # GH#57315 - transpose should have F contiguous blocks
        df = DataFrame(
            {
                "a": pd.array([1, 1, 2], dtype=dtype1),
                "b": pd.array([3, 4, 5], dtype=dtype2),
            }
        )
        result = df.T
        for blk in result._mgr.blocks:
            # When dtypes are unequal, we get NumPy object array
            data = blk.values._data if dtype1 == dtype2 else blk.values
            assert data.flags["F_CONTIGUOUS"]
