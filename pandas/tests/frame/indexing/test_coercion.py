"""
Tests for values coercion in setitem-like operations on DataFrame.

For the most part, these should be multi-column DataFrames, otherwise
we would share the tests with Series.
"""
# [MODULE DOC]
# Tests that setitem-like operations on multi-column DataFrames coerce (or refuse to coerce)
# values correctly: dtype preservation on multi-index column assignment, strict rejection of
# invalid values in scalar setters, and regression tests keyed by issue number. Uses the
# indexer_al fixture (parametrized to scalar at and iat style accessors). No import-time
# side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameSetitemCoercion:
    # [CLASS DOC]
    # Test class with a single multi-index column assignment test.
    @pytest.mark.parametrize("consolidate", [True, False])
    def test_loc_setitem_multiindex_columns(self, consolidate):
        # [FUNCTION DOC]
        # Regression test for GH#18415, parametrized by whether the frame is consolidated. Builds a
        # frame of 32-bit float zeros, concatenates it with itself under two keys (two blocks unless
        # consolidated), then assigns 32-bit float ones to a sub-block of columns by label with row
        # slices of varying extent. After each assignment all column dtypes must still be 32-bit float.
        # GH#18415 Setting values in a single column preserves dtype,
        #  while setting them in multiple columns did unwanted cast.

        # Note that A here has 2 blocks, below we do the same thing
        #  with a consolidated frame.
        A = DataFrame(np.zeros((6, 5), dtype=np.float32))
        A = pd.concat([A, A], axis=1, keys=[1, 2])
        if consolidate:
            A = A._consolidate()

        A.loc[2:3, (1, slice(2, 3))] = np.ones((2, 2), dtype=np.float32)
        assert (A.dtypes == np.float32).all()

        A.loc[0:5, (1, slice(2, 3))] = np.ones((6, 2), dtype=np.float32)

        assert (A.dtypes == np.float32).all()

        A.loc[:, (1, slice(2, 3))] = np.ones((6, 2), dtype=np.float32)
        assert (A.dtypes == np.float32).all()

        # TODO: i think this isn't about MultiIndex and could be done with iloc?


def test_37477():
    # [FUNCTION DOC]
    # Regression test for GH#37477 (fixed by GH#45121): on an integer frame, assigning a float with
    # a fractional part to an integer cell via each of label scalar, label location, integer scalar
    # and integer location accessors raises a type error with message about an invalid value.
    # fixed by GH#45121
    orig = DataFrame({"A": [1, 2, 3], "B": [3, 4, 5]})

    df = orig.copy()
    with pytest.raises(TypeError, match="Invalid value"):
        df.at[1, "B"] = 1.2

    with pytest.raises(TypeError, match="Invalid value"):
        df.loc[1, "B"] = 1.2

    with pytest.raises(TypeError, match="Invalid value"):
        df.iat[1, 1] = 1.2

    with pytest.raises(TypeError, match="Invalid value"):
        df.iloc[1, 1] = 1.2


def test_6942(indexer_al):
    # [FUNCTION DOC]
    # Regression test for GH#6942: sets a timestamp via the parametrized scalar accessor into an
    # empty-valued object frame with a single date row, assigns a boolean to another column, then
    # overwrites via the label accessor and verifies the first cell equals the second timestamp.
    # check that the .at __setitem__ after setting "Live" actually sets the data
    start = Timestamp("2014-04-01")
    t1 = Timestamp("2014-04-23 12:42:38.883082")
    t2 = Timestamp("2014-04-24 01:33:30.040039")

    dti = date_range(start, periods=1)
    orig = DataFrame(index=dti, columns=["timenow", "Live"])

    df = orig.copy()
    indexer_al(df)[start, "timenow"] = t1

    df["Live"] = True

    df.at[start, "timenow"] = t2
    assert df.iloc[0, 0] == t2


def test_26395(indexer_al):
    # [FUNCTION DOC]
    # Regression test for GH#26395 (the label-accessor case fixed by GH#45121, best guess): setting
    # an integer into an integer column through the parametrized accessor works and keeps 64-bit
    # integer dtype; setting a fractional float or a string raises a type error about invalid value.
    # .at case fixed by GH#45121 (best guess)
    df = DataFrame(index=["A", "B", "C"])
    df["D"] = 0

    indexer_al(df)["C", "D"] = 2
    expected = DataFrame({"D": [0, 0, 2]}, index=["A", "B", "C"], dtype=np.int64)
    tm.assert_frame_equal(df, expected)

    with pytest.raises(TypeError, match="Invalid value"):
        indexer_al(df)["C", "D"] = 44.5

    with pytest.raises(TypeError, match="Invalid value"):
        indexer_al(df)["C", "D"] = "hello"


@pytest.mark.xfail(reason="unwanted upcast")
def test_15231():
    # [FUNCTION DOC]
    # Expected-to-fail (unwanted upcast) regression test for GH#15231: appending a row by label
    # from an integer Series keeps all integer dtypes, and then appending a row with a missing
    # field should not upcast the column that has no missing values (expected dtypes integer, float).
    df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    df.loc[2] = Series({"a": 5, "b": 6})
    assert (df.dtypes == np.int64).all()

    df.loc[3] = Series({"a": 7})

    # df["a"] doesn't have any NaNs, should not have been cast
    exp_dtypes = Series([np.int64, np.float64], dtype=object, index=["a", "b"])
    tm.assert_series_equal(df.dtypes, exp_dtypes)


def test_iloc_setitem_unnecesssary_float_upcasting():
    # [FUNCTION DOC]
    # Regression test for GH#12255: assigning a column slice by position with values taken from
    # the same float32 column must not upcast, leaving the frame (float32, float32, string)
    # identical to the original copy.
    # GH#12255
    df = DataFrame(
        {
            0: np.array([1, 3], dtype=np.float32),
            1: np.array([2, 4], dtype=np.float32),
            2: ["a", "b"],
        }
    )
    orig = df.copy()

    values = df[0].values.reshape(2, 1)
    df.iloc[:, 0:1] = values

    tm.assert_frame_equal(df, orig)


@pytest.mark.xfail(reason="unwanted casting to dt64")
def test_12499():
    # [FUNCTION DOC]
    # Expected-to-fail (unwanted casting to datetime) regression test for GH#12499: a row with a
    # numeric and a UTC timestamp column is extended by assigning a row of NaN and NaT, both by
    # label and by label with a full column slice; expected the numeric column to be float with
    # NaN and the timestamp column datetime64 with UTC and NaT. A note in the source says a
    # different missing sentinel in the original report might matter.
    # KNOWN ISSUE: the data list is reassigned just before the second assignment but never used
    # afterwards (the second frame is built from the saved original), so that line is dead code.
    # TODO: OP in GH#12499 used np.datetim64("NaT") instead of pd.NaT,
    #  which has consequences for the expected df["two"] (though i think at
    #  the time it might not have because of a separate bug). See if it makes
    #  a difference which one we use here.
    ts = Timestamp("2016-03-01 03:13:22.98986", tz="UTC")

    data = [{"one": 0, "two": ts}]
    orig = DataFrame(data)
    df = orig.copy()
    df.loc[1] = [np.nan, NaT]

    expected = DataFrame(
        {"one": [0, np.nan], "two": Series([ts, NaT], dtype="datetime64[ns, UTC]")}
    )
    tm.assert_frame_equal(df, expected)

    data = [{"one": 0, "two": ts}]
    df = orig.copy()
    df.loc[1, :] = [np.nan, NaT]
    tm.assert_frame_equal(df, expected)


def test_20476():
    # [FUNCTION DOC]
    # Regression test for GH#20476: a two-level column frame filled with -1 has the first group
    # assigned from a frame with integer and float columns. Verifies the values and that the
    # resulting dtypes are two integer columns, one float, then three untouched integer columns.
    mi = MultiIndex.from_product([["A", "B"], ["a", "b", "c"]])
    df = DataFrame(-1, index=range(3), columns=mi)
    filler = DataFrame([[1, 2, 3.0]] * 3, index=range(3), columns=["a", "b", "c"])
    df["A"] = filler

    expected = DataFrame(
        {
            0: [1, 1, 1],
            1: [2, 2, 2],
            2: [3.0, 3.0, 3.0],
            3: [-1, -1, -1],
            4: [-1, -1, -1],
            5: [-1, -1, -1],
        }
    )
    expected.columns = mi
    exp_dtypes = Series(
        [np.dtype(np.int64)] * 2 + [np.dtype(np.float64)] + [np.dtype(np.int64)] * 3,
        index=mi,
    )
    tm.assert_series_equal(df.dtypes, exp_dtypes)
