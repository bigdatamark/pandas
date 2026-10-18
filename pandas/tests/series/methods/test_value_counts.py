# [MODULE DOC]
# Tests for counting distinct values in a Series for datetime, period, categorical, boolean, complex and masked integer data, with and without normalization and dropping of missing values. Most dtypes are tested elsewhere (the base tests); this module covers Series-specific behavior, including parity with the corresponding Index.
# Expected results are named count (or proportion when normalized) and indexed by the unique values with the series' name. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    Series,
)
import pandas._testing as tm


class TestSeriesValueCounts:
    # [CLASS DOC]
    # Groups tests of Series value counting. Stateless.
    def test_value_counts_datetime(self, unit):
        # [FUNCTION DOC]
        # Verifies counts for naive datetimes at the parametrized resolution unit: the result is ordered by decreasing count, indexed by a datetime index named like the series, and a datetime index built from the same values gives the same result. With normalization the counts become fractions of the total (three, two and one sixths) named proportion.
        # most dtypes are tested in tests/base
        values = [
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 10:00"),
            pd.Timestamp("2011-01-01 11:00"),
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 11:00"),
        ]

        exp_idx = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 11:00", "2011-01-01 10:00"],
            name="xxx",
        ).as_unit(unit)
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx").dt.as_unit(unit)
        tm.assert_series_equal(ser.value_counts(), exp)
        # check DatetimeIndex outputs the same result
        idx = pd.DatetimeIndex(values, name="xxx").as_unit(unit)
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_datetime_tz(self, unit):
        # [FUNCTION DOC]
        # Same as the naive datetime counts test but for timestamps in the US/Eastern zone, at the parametrized resolution unit: Series and datetime index give equal counts and proportions.
        values = [
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 10:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 11:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 11:00", tz="US/Eastern"),
        ]

        exp_idx = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 11:00", "2011-01-01 10:00"],
            tz="US/Eastern",
            name="xxx",
        ).as_unit(unit)
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx").dt.as_unit(unit)
        tm.assert_series_equal(ser.value_counts(), exp)
        idx = pd.DatetimeIndex(values, name="xxx").as_unit(unit)
        tm.assert_series_equal(idx.value_counts(), exp)

        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_period(self):
        # [FUNCTION DOC]
        # Verifies counts for monthly periods: the result is indexed by a period index in decreasing count order and equals the period index's own result, with proportions under normalization.
        # KNOWN ISSUE: an inline comment says datetime index but the object compared is a period index.
        values = [
            pd.Period("2011-01", freq="M"),
            pd.Period("2011-02", freq="M"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-01", freq="M"),
            pd.Period("2011-01", freq="M"),
            pd.Period("2011-03", freq="M"),
        ]

        exp_idx = pd.PeriodIndex(
            ["2011-01", "2011-03", "2011-02"], freq="M", name="xxx"
        )
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx")
        tm.assert_series_equal(ser.value_counts(), exp)
        # check DatetimeIndex outputs the same result
        idx = pd.PeriodIndex(values, name="xxx")
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_categorical_ordered(self):
        # [FUNCTION DOC]
        # Verifies counts for an ordered categorical: the result index is an ordered categorical index in order of decreasing count and the same as the categorical index result, including normalized proportions.
        # most dtypes are tested in tests/base
        values = Categorical([1, 2, 3, 1, 1, 3], ordered=True)

        exp_idx = CategoricalIndex(
            [1, 3, 2], categories=[1, 2, 3], ordered=True, name="xxx"
        )
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx")
        tm.assert_series_equal(ser.value_counts(), exp)
        # check CategoricalIndex outputs the same result
        idx = CategoricalIndex(values, name="xxx")
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_categorical_not_ordered(self):
        # [FUNCTION DOC]
        # Same as the ordered categorical test but for an unordered categorical: counts and proportions match between the Series and the categorical index.
        values = Categorical([1, 2, 3, 1, 1, 3], ordered=False)

        exp_idx = CategoricalIndex(
            [1, 3, 2], categories=[1, 2, 3], ordered=False, name="xxx"
        )
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx")
        tm.assert_series_equal(ser.value_counts(), exp)
        # check CategoricalIndex outputs the same result
        idx = CategoricalIndex(values, name="xxx")
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_categorical(self):
        # [FUNCTION DOC]
        # Verifies categorical counting (issue 12835): without sorting the result follows the category order including a zero count for an unused category; with sorting it is ordered by decreasing count with the unused category last. Also checks that an object-dtype Series is treated with the same name handling.
        # GH#12835
        cats = Categorical(list("abcccb"), categories=list("cabd"))
        ser = Series(cats, name="xxx")
        res = ser.value_counts(sort=False)

        exp_index = CategoricalIndex(
            list("cabd"), categories=cats.categories, name="xxx"
        )
        exp = Series([3, 1, 2, 0], name="count", index=exp_index)
        tm.assert_series_equal(res, exp)

        res = ser.value_counts(sort=True)

        exp_index = CategoricalIndex(
            list("cbad"), categories=cats.categories, name="xxx"
        )
        exp = Series([3, 2, 1, 0], name="count", index=exp_index)
        tm.assert_series_equal(res, exp)

        # check object dtype handles the Series.name as the same
        # (tested in tests/base)
        ser = Series(["a", "b", "c", "c", "c", "b"], name="xxx")
        res = ser.value_counts()
        exp = Series([3, 2, 1], name="count", index=Index(["c", "b", "a"], name="xxx"))
        tm.assert_series_equal(res, exp)

    def test_value_counts_categorical_with_nan(self):
        # [FUNCTION DOC]
        # Verifies categorical counting with missing values (issue 9443), for a Series built from a category dtype string and from a categorical object. Dropping missing values excludes them; keeping them counts them and, when sorting by count, puts the missing entry first; when not sorting, the missing entry is placed last since it is not a category.
        # KNOWN ISSUE: the sanity-check section calls the counting with dropna enabled twice with the same arguments and compares to the same expectation, so the second call is redundant.
        # see GH#9443

        # sanity check
        ser = Series(["a", "b", "a"], dtype="category")
        exp = Series([2, 1], index=CategoricalIndex(["a", "b"]), name="count")

        res = ser.value_counts(dropna=True)
        tm.assert_series_equal(res, exp)

        res = ser.value_counts(dropna=True)
        tm.assert_series_equal(res, exp)

        # same Series via two different constructions --> same behaviour
        series = [
            Series(["a", "b", None, "a", None, None], dtype="category"),
            Series(
                Categorical(["a", "b", None, "a", None, None], categories=["a", "b"])
            ),
        ]

        for ser in series:
            # None is a NaN value, so we exclude its count here
            exp = Series([2, 1], index=CategoricalIndex(["a", "b"]), name="count")
            res = ser.value_counts(dropna=True)
            tm.assert_series_equal(res, exp)

            # we don't exclude the count of None and sort by counts
            exp = Series(
                [3, 2, 1], index=CategoricalIndex([np.nan, "a", "b"]), name="count"
            )
            res = ser.value_counts(dropna=False)
            tm.assert_series_equal(res, exp)

            # When we aren't sorting by counts, and np.nan isn't a
            # category, it should be last.
            exp = Series(
                [2, 1, 3], index=CategoricalIndex(["a", "b", np.nan]), name="count"
            )
            res = ser.value_counts(dropna=False, sort=False)
            tm.assert_series_equal(res, exp)

    @pytest.mark.parametrize(
        "ser, dropna, exp",
        [
            (
                Series([False, True, True, pd.NA]),
                False,
                Series([2, 1, 1], index=[True, False, pd.NA], name="count"),
            ),
            (
                Series([False, True, True, pd.NA]),
                True,
                Series([2, 1], index=Index([True, False], dtype=object), name="count"),
            ),
            (
                Series(range(3), index=[True, False, np.nan]).index,
                False,
                Series([1, 1, 1], index=[True, False, np.nan], name="count"),
            ),
        ],
    )
    def test_value_counts_bool_with_nan(self, ser, dropna, exp):
        # [FUNCTION DOC]
        # Verifies counting for a nullable boolean Series with a missing value (issue 32146), with and without dropping missing values, plus an index of booleans and NaN. Parametrized over input, dropna and expected result; when missing values are dropped the index is object dtype holding true and false.
        # GH32146
        out = ser.value_counts(dropna=dropna)
        tm.assert_series_equal(out, exp)

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_value_counts_complex_numbers(self, dtype):
        # [FUNCTION DOC]
        # Verifies counting of complex values for both 128-bit and 64-bit complex dtypes (issue 17927): result is ordered by decreasing count and indexed by a complex index of the matching dtype.
        # GH 17927
        input_array = np.array([1 + 1j, 1 + 1j, 1, 3j, 3j, 3j], dtype=dtype)
        result = Series(input_array).value_counts()
        expected = Series(
            [3, 2, 1], index=Index([3j, 1 + 1j, 1], dtype=dtype), name="count"
        )
        tm.assert_series_equal(result, expected)

    def test_value_counts_masked(self):
        # [FUNCTION DOC]
        # Verifies counting of a masked nullable integer Series with missing values (issue 54984): with missing values kept, the missing value is counted and the result and index keep the nullable integer dtype; with them dropped only the non-missing values are counted.
        # GH#54984
        dtype = "Int64"
        ser = Series([1, 2, None, 2, None, 3], dtype=dtype)
        result = ser.value_counts(dropna=False)
        expected = Series(
            [2, 2, 1, 1],
            index=Index([2, None, 1, 3], dtype=dtype),
            dtype=dtype,
            name="count",
        )
        tm.assert_series_equal(result, expected)

        result = ser.value_counts(dropna=True)
        expected = Series(
            [2, 1, 1], index=Index([2, 1, 3], dtype=dtype), dtype=dtype, name="count"
        )
        tm.assert_series_equal(result, expected)
