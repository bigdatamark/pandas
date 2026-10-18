# [MODULE DOC]
# Tests for head and tail on DataFrame and Series, including default counts, zero, oversize and negative counts, and empty frames.

import numpy as np

from pandas import DataFrame
import pandas._testing as tm


def test_head_tail_generic(index, frame_or_series):
    # [FUNCTION DOC]
    # Parametrized (via the index and frame_or_series fixtures) over many index types and over frame and series (GH#5370). Builds random data sized to the index.
    # Verifies default head and tail return the first and last five rows; a count of zero gives an empty object; a count larger than the length returns everything; a negative count n returns all rows except the last or first n, matching the equivalent positive count.
    # GH#5370

    ndim = 2 if frame_or_series is DataFrame else 1
    shape = (len(index),) * ndim
    vals = np.random.default_rng(2).standard_normal(shape)
    obj = frame_or_series(vals, index=index)

    tm.assert_equal(obj.head(), obj.iloc[:5])
    tm.assert_equal(obj.tail(), obj.iloc[-5:])

    # 0-len
    tm.assert_equal(obj.head(0), obj.iloc[0:0])
    tm.assert_equal(obj.tail(0), obj.iloc[0:0])

    # bounded
    tm.assert_equal(obj.head(len(obj) + 1), obj)
    tm.assert_equal(obj.tail(len(obj) + 1), obj)

    # neg index
    tm.assert_equal(obj.head(-3), obj.head(len(index) - 3))
    tm.assert_equal(obj.tail(-3), obj.tail(len(index) - 3))


def test_head_tail(float_frame):
    # [FUNCTION DOC]
    # Using the float_frame fixture, verifies head and tail agree with positional slicing for default, zero, negative (-1) and one-row counts, both on the original index and on a float index.
    tm.assert_frame_equal(float_frame.head(), float_frame[:5])
    tm.assert_frame_equal(float_frame.tail(), float_frame[-5:])

    tm.assert_frame_equal(float_frame.head(0), float_frame[0:0])
    tm.assert_frame_equal(float_frame.tail(0), float_frame[0:0])

    tm.assert_frame_equal(float_frame.head(-1), float_frame[:-1])
    tm.assert_frame_equal(float_frame.tail(-1), float_frame[1:])
    tm.assert_frame_equal(float_frame.head(1), float_frame[:1])
    tm.assert_frame_equal(float_frame.tail(1), float_frame[-1:])
    # with a float index
    df = float_frame.copy()
    df.index = np.arange(len(float_frame)) + 0.1
    tm.assert_frame_equal(df.head(), df.iloc[:5])
    tm.assert_frame_equal(df.tail(), df.iloc[-5:])
    tm.assert_frame_equal(df.head(0), df[0:0])
    tm.assert_frame_equal(df.tail(0), df[0:0])
    tm.assert_frame_equal(df.head(-1), df.iloc[:-1])
    tm.assert_frame_equal(df.tail(-1), df.iloc[1:])


def test_head_tail_empty():
    # [FUNCTION DOC]
    # Verifies head and tail of an empty frame return a frame equal to the empty frame.
    # test empty dataframe
    empty_df = DataFrame()
    tm.assert_frame_equal(empty_df.tail(), empty_df)
    tm.assert_frame_equal(empty_df.head(), empty_df)
