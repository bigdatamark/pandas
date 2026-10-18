# [MODULE DOC]
# Tests for the scalar label accessor (the at attribute) on series and frames: assignment,
# expansion, duplicate axes and error behavior (must match the label accessor, with no
# positional fallback).
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas.errors import InvalidIndexError

from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm


def test_at_timezone():
    # https://github.com/pandas-dev/pandas/issues/33544
    # [FUNCTION DOC]
    # Issue 33544. Setting a timezone-aware datetime into a timezone-naive datetime column through
    # the scalar accessor raises a type error about an invalid value.
    result = DataFrame({"foo": [datetime(2000, 1, 1)]})
    with pytest.raises(TypeError, match="Invalid value"):
        result.at[0, "foo"] = datetime(2000, 1, 2, tzinfo=timezone.utc)


def test_selection_methods_of_assigned_col():
    # GH 29282
    # [FUNCTION DOC]
    # GH 29282. After assigning a column from a frame with a reversed index (aligned by label) and
    # setting one cell by label, the frame, scalar, column and column-list selections reflect the
    # aligned and updated values.
    df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = DataFrame(data={"c": [7, 8, 9]}, index=[2, 1, 0])
    df["c"] = df2["c"]
    df.at[1, "c"] = 11
    result = df
    expected = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [9, 11, 7]})
    tm.assert_frame_equal(result, expected)
    result = df.at[1, "c"]
    assert result == 11

    result = df["c"]
    expected = Series([9, 11, 7], name="c")
    tm.assert_series_equal(result, expected)

    result = df[["c"]]
    expected = DataFrame({"c": [9, 11, 7]})
    tm.assert_frame_equal(result, expected)


class TestAtSetItem:
    # [CLASS DOC]
    # Test collection for scalar setitem through the label accessor.
    def test_at_setitem_mixed_index_assignment(self):
        # GH#19860
        # [FUNCTION DOC]
        # GH 19860. For a series with mixed string and integer labels, assigning with a string and an
        # integer label updates the right element (checked via positional accessor).
        ser = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        ser.at["a"] = 11
        assert ser.iat[0] == 11
        ser.at[1] = 22
        assert ser.iat[3] == 22

    def test_at_setitem_categorical_missing(self):
        # [FUNCTION DOC]
        # Setting a valid category in a categorical frame of missing values fills only that cell and
        # keeps the categorical dtype.
        df = DataFrame(
            index=range(3), columns=range(3), dtype=CategoricalDtype(["foo", "bar"])
        )
        df.at[1, 1] = "foo"

        expected = DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, "foo", np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=CategoricalDtype(["foo", "bar"]),
        )

        tm.assert_frame_equal(df, expected)

    def test_at_setitem_multiindex(self):
        # [FUNCTION DOC]
        # Setting by first-level column label sets all columns beneath it in the first row.
        df = DataFrame(
            np.zeros((3, 2), dtype="int64"),
            columns=MultiIndex.from_tuples([("a", 0), ("a", 1)]),
        )
        df.at[0, "a"] = 10
        expected = DataFrame(
            [[10, 10], [0, 0], [0, 0]],
            columns=MultiIndex.from_tuples([("a", 0), ("a", 1)]),
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("row", (Timestamp("2019-01-01"), "2019-01-01"))
    def test_at_datetime_index(self, row):
        # Set float64 dtype to avoid upcast when setting .5
        # [FUNCTION DOC]
        # Parametrized on a timestamp or equivalent string row label. Setting a float in the float
        # column of a datetime-indexed frame updates the first row.
        df = DataFrame(
            data=[[1] * 2], index=DatetimeIndex(data=["2019-01-01", "2019-01-02"])
        ).astype({0: "float64"})
        expected = DataFrame(
            data=[[0.5, 1], [1.0, 1]],
            index=DatetimeIndex(data=["2019-01-01", "2019-01-02"]),
        )

        df.at[row, 0] = 0.5
        tm.assert_frame_equal(df, expected)


class TestAtSetItemWithExpansion:
    # [CLASS DOC]
    # Test collection for setitem that enlarges the object through the label accessor.
    def test_at_setitem_expansion_series_dt64tz_value(self, tz_naive_fixture):
        # GH#25506
        # [FUNCTION DOC]
        # GH 25506. Parametrized on the timezone fixture. Setting a new label on a series holding a
        # timestamp appends an element, keeping a consistent datetime dtype.
        ts = (
            Timestamp("2017-08-05 00:00:00+0100", tz=tz_naive_fixture)
            if tz_naive_fixture is not None
            else Timestamp("2017-08-05 00:00:00+0100")
        )
        result = Series(ts)
        result.at[1] = ts
        expected = Series([ts, ts])
        tm.assert_series_equal(result, expected)


class TestAtWithDuplicates:
    # [CLASS DOC]
    # Test collection for scalar access with duplicate axis labels.
    def test_at_with_duplicate_axes_requires_scalar_lookup(self):
        # GH#33041 check that falling back to loc doesn't allow non-scalar
        #  args to slip in

        # [FUNCTION DOC]
        # GH 33041. With duplicate columns, get and set with non-scalar keys (list, list column, slice)
        # all raise a value error stating that the call is invalid for scalar access.
        arr = np.random.default_rng(2).standard_normal(6).reshape(3, 2)
        df = DataFrame(arr, columns=["A", "A"])

        msg = "Invalid call for scalar access"
        with pytest.raises(ValueError, match=msg):
            df.at[[1, 2]]
        with pytest.raises(ValueError, match=msg):
            df.at[1, ["A"]]
        with pytest.raises(ValueError, match=msg):
            df.at[:, "A"]

        with pytest.raises(ValueError, match=msg):
            df.at[[1, 2]] = 1
        with pytest.raises(ValueError, match=msg):
            df.at[1, ["A"]] = 1
        with pytest.raises(ValueError, match=msg):
            df.at[:, "A"] = 1


class TestAtErrors:
    # TODO: De-duplicate/parametrize
    #  test_at_series_raises_key_error2, test_at_frame_raises_key_error2

    # [CLASS DOC]
    # Test collection for errors raised by scalar access.
    def test_at_series_raises_key_error(self, indexer_al):
        # GH#31724 .at should match .loc

        # [FUNCTION DOC]
        # GH 31724. Parametrized on the fixture providing both the label accessor variants. A label
        # lookup on an integer-indexed series returns the labelled element (not positional); an absent
        # string label raises a key error.
        ser = Series([1, 2, 3], index=[3, 2, 1])
        result = indexer_al(ser)[1]
        assert result == 3

        with pytest.raises(KeyError, match="a"):
            indexer_al(ser)["a"]

    def test_at_frame_raises_key_error(self, indexer_al):
        # GH#31724 .at should match .loc

        # [FUNCTION DOC]
        # GH 31724. Same for a frame: valid row and column pair works; an absent row or absent column
        # raises a key error.
        df = DataFrame({0: [1, 2, 3]}, index=[3, 2, 1])

        result = indexer_al(df)[1, 0]
        assert result == 3

        with pytest.raises(KeyError, match="a"):
            indexer_al(df)["a", 0]

        with pytest.raises(KeyError, match="a"):
            indexer_al(df)[1, "a"]

    def test_at_series_raises_key_error2(self, indexer_al):
        # at should not fallback
        # GH#7814
        # GH#31724 .at should match .loc
        # [FUNCTION DOC]
        # GH 7814, GH 31724. On a string-indexed series, string lookup works while an integer label 0
        # raises a key error (no positional fallback).
        ser = Series([1, 2, 3], index=list("abc"))
        result = indexer_al(ser)["a"]
        assert result == 1

        with pytest.raises(KeyError, match="^0$"):
            indexer_al(ser)[0]

    def test_at_frame_raises_key_error2(self, indexer_al):
        # GH#31724 .at should match .loc
        # [FUNCTION DOC]
        # GH 31724. On a string-indexed frame, an integer column key raises a key error with just the
        # integer in the message.
        df = DataFrame({"A": [1, 2, 3]}, index=list("abc"))
        result = indexer_al(df)["a", "A"]
        assert result == 1

        with pytest.raises(KeyError, match="^0$"):
            indexer_al(df)["a", 0]

    def test_at_frame_multiple_columns(self):
        # GH#48296 - at shouldn't modify multiple columns
        # [FUNCTION DOC]
        # GH 48296. Assigning a list to a single key with the label accessor raises the invalid-index
        # error saying only scalars may be assigned; no column is modified.
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        new_row = [6, 7]
        with pytest.raises(
            InvalidIndexError,
            match=f"You can only assign a scalar value not a \\{type(new_row)}",
        ):
            df.at[5] = new_row

    def test_at_getitem_mixed_index_no_fallback(self):
        # GH#19860
        # [FUNCTION DOC]
        # GH 19860. Integer labels not in a mixed index raise key errors and do not fall back to
        # position.
        ser = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        with pytest.raises(KeyError, match="^0$"):
            ser.at[0]
        with pytest.raises(KeyError, match="^4$"):
            ser.at[4]

    def test_at_categorical_integers(self):
        # CategoricalIndex with integer categories that don't happen to match
        #  the Categorical's codes
        # [FUNCTION DOC]
        # A categorical index with integer categories: integer keys that happen to equal codes but not
        # categories raise key errors, for the frame and its transpose.
        ci = CategoricalIndex([3, 4])

        arr = np.arange(4).reshape(2, 2)
        frame = DataFrame(arr, index=ci)

        for df in [frame, frame.T]:
            for key in [0, 1]:
                with pytest.raises(KeyError, match=str(key)):
                    df.at[key, key]

    def test_at_applied_for_rows(self):
        # GH#48729 .at should raise InvalidIndexError when assigning rows
        # [FUNCTION DOC]
        # GH 48729. Assigning a list with a single key raises the invalid-index error about assigning
        # only a scalar value.
        df = DataFrame(index=["a"], columns=["col1", "col2"])
        new_row = [123, 15]
        with pytest.raises(
            InvalidIndexError,
            match=f"You can only assign a scalar value not a \\{type(new_row)}",
        ):
            df.at["a"] = new_row
