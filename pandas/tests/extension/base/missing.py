# [MODULE DOC]
# Base tests for missing-value handling: isna/notna, dropna and the fillna family on arrays,
# series and frames. Public contents: BaseMissingTests mixin. Uses data, data_missing (two
# elements: missing first, valid second) and fillna_method fixtures.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class BaseMissingTests:
    # [CLASS DOC]
    # Mixin with missing-value tests for an extension array.
    def test_isna(self, data_missing):
        # [FUNCTION DOC]
        # The top-level missing check on data_missing gives a numpy array true then false; the
        # series method gives an equal boolean series; and an empty series (after dropping both rows)
        # gives an empty boolean series (regression for an upstream issue).
        expected = np.array([True, False])

        result = pd.isna(data_missing)
        tm.assert_numpy_array_equal(result, expected)

        result = pd.Series(data_missing).isna()
        expected = pd.Series(expected)
        tm.assert_series_equal(result, expected)

        # GH 21189
        result = pd.Series(data_missing).drop([0, 1]).isna()
        expected = pd.Series([], dtype=bool)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("na_func", ["isna", "notna"])
    def test_isna_returns_copy(self, data_missing, na_func):
        # [FUNCTION DOC]
        # Parametrized over isna and notna. Computes the mask of a series, overwrites the mask with
        # true everywhere, and asserts the original series is unchanged (the mask must be a fresh
        # array). A sparse mask is first converted to a writable numpy array (workaround for an upstream
        # issue).
        result = pd.Series(data_missing)
        expected = result.copy()
        mask = getattr(result, na_func)()
        if isinstance(mask.dtype, pd.SparseDtype):
            # TODO: GH 57739
            mask = np.array(mask)
            mask.flags.writeable = True

        mask[:] = True
        tm.assert_series_equal(result, expected)

    def test_dropna_array(self, data_missing):
        # [FUNCTION DOC]
        # Array dropna keeps only the second (valid) element.
        result = data_missing.dropna()
        expected = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_dropna_series(self, data_missing):
        # [FUNCTION DOC]
        # Series dropna keeps only the row with the valid element.
        ser = pd.Series(data_missing)
        result = ser.dropna()
        expected = ser.iloc[[1]]
        tm.assert_series_equal(result, expected)

    def test_dropna_frame(self, data_missing):
        # [FUNCTION DOC]
        # Frame dropna: default drops the row with missing; dropping columns removes the only column
        # (leaving an empty column index with the original two-row index); with an additional numeric
        # column whose missing value is in the other row, all rows are dropped.
        df = pd.DataFrame({"A": data_missing}, columns=pd.Index(["A"], dtype=object))

        # defaults
        result = df.dropna()
        expected = df.iloc[[1]]
        tm.assert_frame_equal(result, expected)

        # axis = 1
        result = df.dropna(axis="columns")
        expected = pd.DataFrame(index=pd.RangeIndex(2), columns=pd.Index([]))
        tm.assert_frame_equal(result, expected)

        # multiple
        df = pd.DataFrame({"A": data_missing, "B": [1, np.nan]})
        result = df.dropna()
        expected = df.iloc[:0]
        tm.assert_frame_equal(result, expected)

    def test_fillna_scalar(self, data_missing):
        # [FUNCTION DOC]
        # Fills missing values with the valid element at index one.
        # KNOWN ISSUE: both the result and the expected value are computed with the same call, so the
        # assertion compares a computation with itself and cannot fail.
        valid = data_missing[1]
        result = data_missing.fillna(valid)
        expected = data_missing.fillna(valid)
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_with_none(self, data_missing):
        # [FUNCTION DOC]
        # Filling with none leaves the array equal to the original (regression for an upstream
        # issue).
        # GH#57723
        result = data_missing.fillna(None)
        expected = data_missing
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_limit_pad(self, data_missing):
        # [FUNCTION DOC]
        # Forward fill with limit two on a series arranged valid, missing, missing, missing, valid
        # fills only the first two missing slots.
        arr = data_missing.take([1, 0, 0, 0, 1])
        result = pd.Series(arr).ffill(limit=2)
        expected = pd.Series(data_missing.take([1, 1, 1, 0, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "limit_area, input_ilocs, expected_ilocs",
        [
            ("outside", [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
            ("outside", [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),
            ("outside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),
            ("outside", [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),
            ("inside", [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
            ("inside", [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),
        ],
    )
    def test_ffill_limit_area(
        self, data_missing, limit_area, input_ilocs, expected_ilocs
    ):
        # [FUNCTION DOC]
        # Parametrized over the limit area (outside or inside) and input/expected positions into
        # data_missing (zero means missing, one valid). Forward fill with the limit area restricts
        # filling to gaps surrounded by valid values (inside) or to those outside the first and last
        # valid values (outside); the series must equal the expected arrangement.
        # GH#56616
        arr = data_missing.take(input_ilocs)
        result = pd.Series(arr).ffill(limit_area=limit_area)
        expected = pd.Series(data_missing.take(expected_ilocs))
        tm.assert_series_equal(result, expected)

    def test_fillna_limit_backfill(self, data_missing):
        # [FUNCTION DOC]
        # Backward fill with limit two on valid, missing, missing, missing, valid fills only the two
        # missing slots next to the final valid value.
        arr = data_missing.take([1, 0, 0, 0, 1])
        result = pd.Series(arr).bfill(limit=2)
        expected = pd.Series(data_missing.take([1, 0, 1, 1, 1]))
        tm.assert_series_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        # [FUNCTION DOC]
        # With the missing entries removed, filling with a valid value and the internal backward
        # pad-or-backfill must both return a new object (not the same one) equal to the input.
        data = data[~data.isna()]

        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

        result = data._pad_or_backfill(method="backfill")
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def test_fillna_readonly(self, data_missing):
        # [FUNCTION DOC]
        # Sets the private read-only flag on a copy of data_missing. Filling by default makes a
        # copy, works, and leaves the input unchanged. Filling with copy disabled must raise a value
        # error saying the array is read-only (for arrays that honor the copy argument), with the
        # input still unchanged.
        data = data_missing.copy()
        data._readonly = True

        # by default fillna(copy=True), then this works fine
        result = data.fillna(data_missing[1])
        assert result[0] == data_missing[1]
        tm.assert_extension_array_equal(data, data_missing)

        # but with copy=False, this raises for EAs that respect the copy keyword
        with pytest.raises(ValueError, match="Cannot modify read-only array"):
            data.fillna(data_missing[1], copy=False)
        tm.assert_extension_array_equal(data, data_missing)

    def test_fillna_series(self, data_missing):
        # [FUNCTION DOC]
        # Fill a series with a scalar gives both elements valid; filling with a series of the filled
        # values gives the same; filling with the series itself leaves the series unchanged.
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        result = ser.fillna(fill_value)
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )
        tm.assert_series_equal(result, expected)

        # Fill with a series
        result = ser.fillna(expected)
        tm.assert_series_equal(result, expected)

        # Fill with a series not affecting the missing values
        result = ser.fillna(ser)
        tm.assert_series_equal(result, ser)

    def test_fillna_series_method(self, data_missing, fillna_method):
        # [FUNCTION DOC]
        # Uses the fillna_method fixture (forward or backward fill). For forward fill the data is
        # reversed first so that the missing value comes after the valid one. Asserts the method fills
        # the missing slot with the valid element.
        fill_value = data_missing[1]

        if fillna_method == "ffill":
            data_missing = data_missing[::-1]

        result = getattr(pd.Series(data_missing), fillna_method)()
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )

        tm.assert_series_equal(result, expected)

    def test_fillna_frame(self, data_missing):
        # [FUNCTION DOC]
        # Frame with the extension column and an integer column: filling with the valid element fills
        # the extension column only; the integer column is unchanged.
        fill_value = data_missing[1]

        result = pd.DataFrame({"A": data_missing, "B": [1, 2]}).fillna(fill_value)

        expected = pd.DataFrame(
            {
                "A": data_missing._from_sequence(
                    [fill_value, fill_value], dtype=data_missing.dtype
                ),
                "B": [1, 2],
            }
        )

        tm.assert_frame_equal(result, expected)

    def test_fillna_fill_other(self, data):
        # [FUNCTION DOC]
        # Frame with the data and an all-NaN float column, filled with a mapping for the float column
        # only: the float column becomes zeros and the extension column is unchanged.
        result = pd.DataFrame({"A": data, "B": [np.nan] * len(data)}).fillna({"B": 0.0})

        expected = pd.DataFrame({"A": data, "B": [0.0] * len(result)})

        tm.assert_frame_equal(result, expected)
