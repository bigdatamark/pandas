"""
Tests for DataFrame.mask; tests DataFrame.where as a side-effect.
"""
# [MODULE DOC]
# Tests for DataFrame (and Series) mask, in which where is tested as a side effect. Covers
# equivalence with where, in-place use, callables, edge shapes and dtype handling. Uses the
# frame_or_series fixture in one test. No import-time side effects.

import numpy as np

from pandas import (
    NA,
    DataFrame,
    Float64Dtype,
    Series,
    StringDtype,
    Timedelta,
    isna,
)
import pandas._testing as tm


class TestDataFrameMask:
    # [CLASS DOC]
    # Test class for DataFrame mask.
    def test_mask(self):
        # [FUNCTION DOC]
        # Verifies mask with the inverted condition equals where with a missing-value replacement, and
        # mask with another frame equals where with the same other, for a random float frame and the
        # condition of being positive.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        cond = df > 0

        rs = df.where(cond, np.nan)
        tm.assert_frame_equal(rs, df.mask(df <= 0))
        tm.assert_frame_equal(rs, df.mask(~cond))

        other = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        rs = df.where(cond, other)
        tm.assert_frame_equal(rs, df.mask(df <= 0, other))
        tm.assert_frame_equal(rs, df.mask(~cond, other))

    def test_mask2(self):
        # [FUNCTION DOC]
        # Regression test for GH#21891: masking a one-column frame with a nested list of booleans
        # replaces masked positions with missing values.
        # see GH#21891
        df = DataFrame([1, 2])
        res = df.mask([[True], [False]])

        exp = DataFrame([np.nan, 2])
        tm.assert_frame_equal(res, exp)

    def test_mask_inplace(self):
        # [FUNCTION DOC]
        # Regression test for GH#8801: where with the in-place option returns the same object (identity)
        # and the frame equals where and the inverted mask, with and without a replacement frame.
        # GH#8801
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        cond = df > 0

        rdf = df.copy()

        result = rdf.where(cond, inplace=True)
        assert result is rdf
        tm.assert_frame_equal(rdf, df.where(cond))
        tm.assert_frame_equal(rdf, df.mask(~cond))

        rdf = df.copy()
        result = rdf.where(cond, -df, inplace=True)
        assert result is rdf
        tm.assert_frame_equal(rdf, df.where(cond, -df))
        tm.assert_frame_equal(rdf, df.mask(~cond, -df))

    def test_mask_edge_case_1xN_frame(self):
        # [FUNCTION DOC]
        # Regression test for GH#4071: masking a one-row two-column frame with a boolean frame of the
        # same shape works and replaces only the selected entry with a missing value.
        # GH#4071
        df = DataFrame([[1, 2]])
        res = df.mask(DataFrame([[True, False]]))
        expec = DataFrame([[np.nan, 2]])
        tm.assert_frame_equal(res, expec)

    def test_mask_callable(self):
        # [FUNCTION DOC]
        # Regression test for GH#12533: mask accepts callables for the condition and the replacement.
        # Verifies results for callables returning frames, one returning an array with a scalar-returning
        # replacement callable, and chained use on a shifted frame; each also equals the non-callable form.
        # GH#12533
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = df.mask(lambda x: x > 4, lambda x: x + 1)
        exp = DataFrame([[1, 2, 3], [4, 6, 7], [8, 9, 10]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, df.mask(df > 4, df + 1))

        # return ndarray and scalar
        result = df.mask(lambda x: (x % 2 == 0).values, lambda x: 99)
        exp = DataFrame([[1, 99, 3], [99, 5, 99], [7, 99, 9]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, df.mask(df % 2 == 0, 99))

        # chain
        result = (df + 2).mask(lambda x: x > 8, lambda x: x + 10)
        exp = DataFrame([[3, 4, 5], [6, 7, 8], [19, 20, 21]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, (df + 2).mask((df + 2) > 8, (df + 2) + 10))

    def test_mask_dtype_bool_conversion(self):
        # [FUNCTION DOC]
        # Regression test for GH#3733: masking a boolean frame that has missing positions with the null
        # mask equals doing the same on its object-dtype cast.
        # GH#3733
        df = DataFrame(data=np.random.default_rng(2).standard_normal((100, 50)))
        df = df.where(df > 0)  # create nans
        bools = df > 0
        mask = isna(df)
        expected = bools.astype(object).mask(mask)
        result = bools.mask(mask)
        tm.assert_frame_equal(result, expected)


def test_mask_stringdtype(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 40824: masks a string-dtype frame (or its single Series, per the
    # frame_or_series fixture) with a boolean Series aligned by label and a replacement object
    # aligned on the labels; masked entries take the replacement values and the missing value stays.
    # GH 40824
    obj = DataFrame(
        {"A": ["foo", "bar", "baz", NA]},
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    filtered_obj = DataFrame(
        {"A": ["this", "that"]}, index=["id2", "id3"], dtype=StringDtype()
    )
    expected = DataFrame(
        {"A": ["foo", "this", "that", NA]},
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    if frame_or_series is Series:
        obj = obj["A"]
        filtered_obj = filtered_obj["A"]
        expected = expected["A"]

    filter_ser = Series(
        [False, True, True, False],
        index=["id1", "id2", "id3", "id4"],
    )
    result = obj.mask(filter_ser, filtered_obj)

    tm.assert_equal(result, expected)


def test_mask_where_dtype_timedelta():
    # [FUNCTION DOC]
    # Regression test for issue 39548: mask with the not-missing condition on a timedelta frame
    # gives all missing timedeltas (second unit), and where with a threshold condition keeps only
    # values over two days.
    # https://github.com/pandas-dev/pandas/issues/39548
    df = DataFrame([Timedelta(i, unit="D") for i in range(5)])

    expected = DataFrame(np.full(5, np.nan, dtype="timedelta64[s]"))
    tm.assert_frame_equal(df.mask(df.notna()), expected)

    expected = DataFrame(
        [np.nan, np.nan, np.nan, Timedelta("3 day"), Timedelta("4 day")],
        dtype="m8[s]",
    )
    result = df.where(df > Timedelta(2, unit="D"))
    tm.assert_frame_equal(result, expected)


def test_mask_return_dtype():
    # [FUNCTION DOC]
    # Regression test for GH#50488: masking a masked-float Series with a not-missing condition and a
    # boolean other keeps the masked float dtype, converting the replacement to floats.
    # GH#50488
    ser = Series([0.0, 1.0, 2.0, 3.0], dtype=Float64Dtype())
    cond = ~ser.isna()
    other = Series([True, False, True, False])
    excepted = Series([1.0, 0.0, 1.0, 0.0], dtype=ser.dtype)
    result = ser.mask(cond, other)
    tm.assert_series_equal(result, excepted)


def test_mask_inplace_no_other():
    # [FUNCTION DOC]
    # Regression test for GH#51685: in-place mask with a boolean frame condition and no other value
    # sets masked entries to missing in a mixed float and string frame.
    # GH#51685
    df = DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    cond = DataFrame({"a": [True, False], "b": [False, True]})
    df.mask(cond, inplace=True)
    expected = DataFrame({"a": [np.nan, 2], "b": ["x", np.nan]})
    tm.assert_frame_equal(df, expected)
