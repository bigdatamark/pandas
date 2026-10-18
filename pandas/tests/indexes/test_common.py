"""
Collection of tests asserting things that should be true for
any index subclass except for MultiIndex. Makes use of the `index_flat`
fixture defined in pandas/conftest.py.
"""

# [MODULE DOC]
# Tests of behavior common to every flat (non multi-level) Index subclass, mostly using the flat-index fixture from the top-level test configuration. Covers frame conversion, level dropping, name handling (hashability, setting, copying), copy and deepcopy, unique and duplicate handling, monotonic searchsorted, astype name preservation, missing-value flags, sorting with missing values and a natural-order sort key, array-compatibility properties, comparison of read-only data, and a deprecation warning for joining with a series. A single class holds the fixture-based tests; module-level tests follow.

from copy import (
    copy,
    deepcopy,
)
import re

import numpy as np
import pytest

from pandas.compat import IS64
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_numeric_dtype,
)

import pandas as pd
from pandas import (
    CategoricalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
)
import pandas._testing as tm


class TestCommon:
    # [CLASS DOC]
    # Test class for behavior shared by all flat index types.
    @pytest.mark.parametrize("name", [None, "new_name"])
    def test_to_frame(self, name, index_flat):
        # [FUNCTION DOC]
        # Parametrized over the name (none or custom). Converting to a frame with one column: the frame's index is the same object as the index, there is exactly one column with the expected name (the index name or zero when unnamed); with index disabled the frame has a different index object.
        # see GH#15230, GH#22580
        idx = index_flat

        if name:
            idx_name = name
        else:
            idx_name = idx.name or 0

        df = idx.to_frame(name=idx_name)

        assert df.index is idx
        assert len(df.columns) == 1
        assert df.columns[0] == idx_name

        df = idx.to_frame(index=False, name=idx_name)
        assert df.index is not idx

    def test_droplevel(self, index_flat):
        # [FUNCTION DOC]
        # Dropping an empty list of levels returns an equal index. Dropping the only level (by name or name in a list; tuple names skipped for the bare form) raises a value error that at least one level must remain. Using an unknown level label (string or list) raises a key error saying the requested level does not match the index name (none).
        # GH 21115
        # MultiIndex is tested separately in test_multi.py
        index = index_flat

        assert index.droplevel([]).equals(index)

        for level in [index.name, [index.name]]:
            if isinstance(index.name, tuple) and level is index.name:
                # GH 21121 : droplevel with tuple name
                continue
            msg = (
                "Cannot remove 1 levels from an index with 1 levels: at least one "
                "level must be left."
            )
            with pytest.raises(ValueError, match=msg):
                index.droplevel(level)

        for level in "wrong", ["wrong"]:
            with pytest.raises(
                KeyError,
                match=r"'Requested level \(wrong\) does not match index name \(None\)'",
            ):
                index.droplevel(level)

    def test_constructor_non_hashable_name(self, index_flat):
        # [FUNCTION DOC]
        # Renaming or setting names with an unhashable value (nested list) raises a type error saying the name must be hashable.
        # GH 20527
        index = index_flat

        message = "Index.name must be a hashable type"
        renamed = [["1"]]

        # With .rename()
        with pytest.raises(TypeError, match=message):
            index.rename(name=renamed)

        # With .set_names()
        with pytest.raises(TypeError, match=message):
            index.set_names(names=renamed)

    def test_constructor_unwraps_index(self, index_flat):
        # [FUNCTION DOC]
        # Constructing an index of the same class from an index with the same dtype gives an index with equal underlying data.
        a = index_flat
        # Passing dtype is necessary for Index([True, False], dtype=object)
        #  case.
        b = type(a)(a, dtype=a.dtype)
        tm.assert_equal(a._data, b._data)

    def test_to_flat_index(self, index_flat):
        # [FUNCTION DOC]
        # The flat-index conversion of a flat index returns an equal index.
        # 22866
        index = index_flat

        result = index.to_flat_index()
        tm.assert_index_equal(result, index)

    def test_set_name_methods(self, index_flat):
        # [FUNCTION DOC]
        # Setting names with a one-element list returns a new index with that name and leaves the original; in-place rename returns nothing and updates name and names; passing a level to set_names on a flat index raises a value error ("Level must be None"); tuple names are accepted as is.
        # MultiIndex tested separately
        index = index_flat
        new_name = "This is the new name for this index"

        original_name = index.name
        new_ind = index.set_names([new_name])
        assert new_ind.name == new_name
        assert index.name == original_name
        res = index.rename(new_name, inplace=True)

        # should return None
        assert res is None
        assert index.name == new_name
        assert index.names == [new_name]
        with pytest.raises(ValueError, match="Level must be None"):
            index.set_names("a", level=0)

        # rename in place just leaves tuples and other containers alone
        name = ("A", "B")
        index.rename(name, inplace=True)
        assert index.name == name
        assert index.names == [name]

    @pytest.mark.xfail
    def test_set_names_single_label_no_level(self, index_flat):
        # [FUNCTION DOC]
        # Marked expected-to-fail. Intended behavior: setting names to a plain string (not list-like) with no level should raise a type error mentioning list-like even if the length would fit.
        with pytest.raises(TypeError, match="list-like"):
            # should still fail even if it would be the right length
            index_flat.set_names("a")

    def test_copy_and_deepcopy(self, index_flat):
        # [FUNCTION DOC]
        # Shallow and deep copies from the copy module are new objects equal to the original; copy with deep and a new name returns that name.
        index = index_flat

        for func in (copy, deepcopy):
            idx_copy = func(index)
            assert idx_copy is not index
            assert idx_copy.equals(index)

        new_copy = index.copy(deep=True, name="banana")
        assert new_copy.name == "banana"

    def test_copy_name(self, index_flat):
        # [FUNCTION DOC]
        # Building an index of the same class with a name and copy, then again without copy, yields a new object with equal values and the name "mario"; the index equals the first (names differ); arithmetic of two series built on these indexes keeps the name.
        # GH#12309: Check that the "name" argument
        # passed at initialization is honored.
        index = index_flat

        first = type(index)(index, copy=True, name="mario")
        second = type(first)(first, copy=False)

        # Even though "copy=False", we want a new object.
        assert first is not second
        tm.assert_index_equal(first, second)

        # Not using tm.assert_index_equal() since names differ.
        assert index.equals(first)

        assert first.name == "mario"
        assert second.name == "mario"

        # TODO: belongs in series arithmetic tests?
        s1 = pd.Series(2, index=first)
        s2 = pd.Series(3, index=second[:-1])
        # See GH#13365
        s3 = s1 * s2
        assert s3.index.name == "mario"

    def test_copy_name2(self, index_flat):
        # [FUNCTION DOC]
        # Copying with a name sets it; a list of two names raises a value error (new names must have length one); an unhashable name raises a type error naming the class.
        # GH#35592
        index = index_flat

        assert index.copy(name="mario").name == "mario"

        with pytest.raises(ValueError, match="Length of new names must be 1, got 2"):
            index.copy(name=["mario", "luigi"])

        msg = f"{type(index).__name__}.name must be a hashable type"
        with pytest.raises(TypeError, match=msg):
            index.copy(name=[["mario"]])

    def test_unique_level(self, index_flat):
        # [FUNCTION DOC]
        # unique with level zero, the index name, or none equals dropping duplicates. Level three raises an index error (too many levels). An unknown level label raises a key error about not matching the index name.
        # don't test a MultiIndex here (as its tested separated)
        index = index_flat

        # GH 17896
        expected = index.drop_duplicates()
        for level in [0, index.name, None]:
            result = index.unique(level=level)
            tm.assert_index_equal(result, expected)

        msg = "Too many levels: Index has only 1 level, not 4"
        with pytest.raises(IndexError, match=msg):
            index.unique(level=3)

        msg = (
            rf"Requested level \(wrong\) does not match index name "
            rf"\({re.escape(index.name.__repr__())}\)"
        )
        with pytest.raises(KeyError, match=msg):
            index.unique(level="wrong")

    def test_unique(self, index_flat):
        # [FUNCTION DOC]
        # Skips empty indexes. An index made by repeating the first element five times has unique equal to a one-element index (exact type except for range indexes). If the index can hold missing values: replace the first element of the repeated values with NaN, build via the private shallow copy, and check the dtype is unchanged and unique yields the two-element NaN-then-value index for both the repeated and the already unique inputs.
        # MultiIndex tested separately
        index = index_flat
        if not len(index):
            pytest.skip("Skip check for empty Index and MultiIndex")

        idx = index[[0] * 5]
        idx_unique = index[[0]]

        # We test against `idx_unique`, so first we make sure it's unique
        # and doesn't contain nans.
        assert idx_unique.is_unique is True
        try:
            assert idx_unique.hasnans is False
        except NotImplementedError:
            pass

        result = idx.unique()
        tm.assert_index_equal(
            result, idx_unique, exact=not isinstance(index, RangeIndex)
        )

        # nans:
        if not index._can_hold_na:
            pytest.skip("Skip na-check if index cannot hold na")

        vals = index._values.copy()[[0] * 5]
        vals[0] = np.nan

        vals_unique = vals[:2]
        idx_nan = index._shallow_copy(vals)
        idx_unique_nan = index._shallow_copy(vals_unique)
        assert idx_unique_nan.is_unique is True

        assert idx_nan.dtype == index.dtype
        assert idx_unique_nan.dtype == index.dtype

        expected = idx_unique_nan
        for pos, i in enumerate([idx_nan, idx_unique_nan]):
            result = i.unique()
            tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:Period with BDay freq:FutureWarning")
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_searchsorted_monotonic(self, index_flat, request):
        # [FUNCTION DOC]
        # Ignores period business-day warnings. Interval indexes are expected to fail (searching with an interval is unsupported). Skips empty indexes. Computes expected left insertion position zero and expected right position as the first position not equal to the first value (the full length if all equal). For monotonic indexes both the private monotonic search and the public search must match the expected positions on both sides; for non-monotonic indexes the private method raises a value error that the index must be monotonic.
        # GH17271
        index = index_flat
        # not implemented for tuple searches in MultiIndex
        # or Intervals searches in IntervalIndex
        if isinstance(index, pd.IntervalIndex):
            mark = pytest.mark.xfail(
                reason="IntervalIndex.searchsorted does not support Interval arg",
                raises=NotImplementedError,
            )
            request.applymarker(mark)

        # nothing to test if the index is empty
        if index.empty:
            pytest.skip("Skip check for empty Index")
        value = index[0]

        # determine the expected results (handle dupes for 'right')
        expected_left, expected_right = 0, (index == value).argmin()
        if expected_right == 0:
            # all values are the same, expected_right should be length
            expected_right = len(index)

        if index.is_monotonic_increasing or index.is_monotonic_decreasing:
            ssm_left = index._searchsorted_monotonic(value, side="left")
            assert expected_left == ssm_left

            ssm_right = index._searchsorted_monotonic(value, side="right")
            assert expected_right == ssm_right

            ss_left = index.searchsorted(value, side="left")
            assert expected_left == ss_left

            ss_right = index.searchsorted(value, side="right")
            assert expected_right == ss_right
        else:
            # non-monotonic should raise.
            msg = "index must be monotonic increasing or decreasing"
            with pytest.raises(ValueError, match=msg):
                index._searchsorted_monotonic(value, side="left")

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_drop_duplicates(self, index_flat, keep):
        # [FUNCTION DOC]
        # Uses the keep fixture (first, last, false). Skips range and empty indexes. Builds a unique index from the distinct values (with the index dtype for numeric ones), then a duplicated index by random sampling (seeded) of one and a half times its length. duplicated must match the series duplicated result and drop_duplicates must match the series drop_duplicates result. Ignores period warnings.
        # MultiIndex is tested separately
        index = index_flat
        if isinstance(index, RangeIndex):
            pytest.skip(
                "RangeIndex is tested in test_drop_duplicates_no_duplicates "
                "as it cannot hold duplicates"
            )
        if len(index) == 0:
            pytest.skip(
                "empty index is tested in test_drop_duplicates_no_duplicates "
                "as it cannot hold duplicates"
            )

        # make unique index
        holder = type(index)
        unique_values = list(set(index))
        dtype = index.dtype if is_numeric_dtype(index) else None
        unique_idx = holder(unique_values, dtype=dtype)

        # make duplicated index
        n = len(unique_idx)
        duplicated_selection = np.random.default_rng(2).choice(n, int(n * 1.5))
        idx = holder(unique_idx.values[duplicated_selection])

        # Series.duplicated is tested separately
        expected_duplicated = (
            pd.Series(duplicated_selection).duplicated(keep=keep).values
        )
        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected_duplicated)

        # Series.drop_duplicates is tested separately
        expected_dropped = holder(pd.Series(idx).drop_duplicates(keep=keep))
        tm.assert_index_equal(idx.drop_duplicates(keep=keep), expected_dropped)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_drop_duplicates_no_duplicates(self, index_flat):
        # [FUNCTION DOC]
        # A unique index (a range index is used as is) has no duplicates and drop_duplicates returns an equal but distinct object.
        # MultiIndex is tested separately
        index = index_flat

        # make unique index
        if isinstance(index, RangeIndex):
            # RangeIndex cannot have duplicates
            unique_idx = index
        else:
            holder = type(index)
            unique_values = list(set(index))
            dtype = index.dtype if is_numeric_dtype(index) else None
            unique_idx = holder(unique_values, dtype=dtype)

        # check on unique index
        expected_duplicated = np.array([False] * len(unique_idx), dtype="bool")
        tm.assert_numpy_array_equal(unique_idx.duplicated(), expected_duplicated)
        result_dropped = unique_idx.drop_duplicates()
        tm.assert_index_equal(result_dropped, unique_idx)
        # validate shallow copy
        assert result_dropped is not unique_idx

    def test_drop_duplicates_inplace(self, index):
        # [FUNCTION DOC]
        # Uses the any index fixture. Passing an in-place argument to drop_duplicates raises a type error about an unexpected keyword argument.
        msg = r"drop_duplicates\(\) got an unexpected keyword argument"
        with pytest.raises(TypeError, match=msg):
            index.drop_duplicates(inplace=True)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_has_duplicates(self, index_flat):
        # [FUNCTION DOC]
        # Skips empty and range indexes. An index made from five copies of the first element is not unique and reports duplicates.
        # MultiIndex tested separately in:
        #   tests/indexes/multi/test_unique_and_duplicates.
        index = index_flat
        holder = type(index)
        if not len(index) or isinstance(index, RangeIndex):
            # MultiIndex tested separately in:
            #   tests/indexes/multi/test_unique_and_duplicates.
            # RangeIndex is unique by definition.
            pytest.skip("Skip check for empty Index, MultiIndex, and RangeIndex")

        idx = holder([index[0]] * 5)
        assert idx.is_unique is False
        assert idx.has_duplicates is True

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "uint64", "float64", "category", "datetime64[ns]", "timedelta64[ns]"],
    )
    def test_astype_preserves_name(self, index, dtype):
        # [FUNCTION DOC]
        # Parametrized over target dtypes (int64, uint64, float64, category, datetime64, timedelta64). Names the index (all levels for multi-level) then converts; conversions that cannot be performed (value, type, not-implemented or system errors) end the test silently. A complex-warning is expected when casting complex to numeric real dtypes; extra warnings are checked only for the pyarrow string to category case. On success the result keeps the name(s).
        # https://github.com/pandas-dev/pandas/issues/32013
        if isinstance(index, MultiIndex):
            index.names = ["idx" + str(i) for i in range(index.nlevels)]
        else:
            index.name = "idx"

        warn = (
            np.exceptions.ComplexWarning
            if index.dtype.kind == "c" and dtype in ["float64", "int64", "uint64"]
            else None
        )

        is_pyarrow_str = str(index.dtype) == "string[pyarrow]" and dtype == "category"
        try:
            # Some of these conversions cannot succeed so we use a try / except
            with tm.assert_produces_warning(
                warn,
                raise_on_extra_warnings=is_pyarrow_str,
                check_stacklevel=False,
            ):
                result = index.astype(dtype)
        except (ValueError, TypeError, NotImplementedError, SystemError):
            return

        if isinstance(index, MultiIndex):
            assert result.names == index.names
        else:
            assert result.name == index.name

    def test_hasnans_isnans(self, index_flat):
        # [FUNCTION DOC]
        # For an index without missing values the NaN mask is all false and the NaN flag false. Then returns early for empty, integer-dtype and bool indexes; otherwise sets the second value to NaN in the values, rebuilds, and checks the mask marks only that position and the flag is true.
        # GH#11343, added tests for hasnans / isnans
        index = index_flat

        # cases in indices doesn't include NaN
        idx = index.copy(deep=True)
        expected = np.array([False] * len(idx), dtype=bool)
        tm.assert_numpy_array_equal(idx._isnan, expected)
        assert idx.hasnans is False

        idx = index.copy(deep=True)
        values = idx._values

        if len(index) == 0:
            return
        elif is_integer_dtype(index.dtype):
            return
        elif index.dtype == bool:
            # values[1] = np.nan below casts to True!
            return

        values[1] = np.nan

        idx = type(index)(values)

        expected = np.array([False] * len(idx), dtype=bool)
        expected[1] = True
        tm.assert_numpy_array_equal(idx._isnan, expected)
        assert idx.hasnans is True


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize("na_position", [None, "middle"])
def test_sort_values_invalid_na_position(index_with_missing, na_position):
    # [FUNCTION DOC]
    # Parametrized over invalid na_position values (none, "middle"). Sorting an index with missing values raises a value error saying the na_position is invalid.
    with pytest.raises(ValueError, match=f"invalid na_position: {na_position}"):
        index_with_missing.sort_values(na_position=na_position)


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_sort_values_with_missing(index_with_missing, na_position, request):
    # [FUNCTION DOC]
    # Parametrized over na_position (first or last). Categorical is allowed to fail non-strictly. Computes the expected result by sorting the non-missing values and placing the missing count of missing markers at the chosen end, constructing it with the same class and dtype, and compares with the result of sort_values.
    # GH 35584. Test that sort_values works with missing values,
    # sort non-missing and place missing according to na_position

    if isinstance(index_with_missing, CategoricalIndex):
        request.applymarker(
            pytest.mark.xfail(
                reason="missing value sorting order not well-defined", strict=False
            )
        )

    missing_count = np.sum(index_with_missing.isna())
    not_na_vals = index_with_missing[index_with_missing.notna()].values
    sorted_values = np.sort(not_na_vals)
    if na_position == "first":
        sorted_values = np.concatenate([[None] * missing_count, sorted_values])
    else:
        sorted_values = np.concatenate([sorted_values, [None] * missing_count])

    # Explicitly pass dtype needed for Index backed by EA e.g. IntegerArray
    expected = type(index_with_missing)(sorted_values, dtype=index_with_missing.dtype)

    result = index_with_missing.sort_values(na_position=na_position)
    tm.assert_index_equal(result, expected)


def test_sort_values_natsort_key():
    # [FUNCTION DOC]
    # Sorting an index of dotted version strings with a key function that converts each string to a tuple of integers yields natural numeric ordering (1.9, 1.10, 1.11, 2.0).
    # GH#56081
    def split_convert(s):
        # [FUNCTION DOC]
        # HELPER: converts a dotted version string to a tuple of integers for use by the sort key in the enclosing test.
        return tuple(int(x) for x in s.split("."))

    idx = pd.Index(["1.9", "2.0", "1.11", "1.10"])
    expected = pd.Index(["1.9", "1.10", "1.11", "2.0"])
    result = idx.sort_values(key=lambda x: tuple(map(split_convert, x)))
    tm.assert_index_equal(result, expected)


def test_ndarray_compat_properties(index):
    # [FUNCTION DOC]
    # Skips period indexes on 32-bit platforms. Transposed index equals the original; shape, number of dimensions and size match the values array; the byte count matches the values array except for range and multi-level indexes (not array-backed). Accessing the byte counts must not raise.
    if isinstance(index, PeriodIndex) and not IS64:
        pytest.skip("Overflow")
    idx = index
    assert idx.T.equals(idx)
    assert idx.transpose().equals(idx)

    values = idx.values

    assert idx.shape == values.shape
    assert idx.ndim == values.ndim
    assert idx.size == values.size

    if not isinstance(index, (RangeIndex, MultiIndex)):
        # These two are not backed by an ndarray
        assert idx.nbytes == values.nbytes

    # test for validity
    idx.nbytes
    idx.values.nbytes


def test_compare_read_only_array():
    # [FUNCTION DOC]
    # An index built from a read-only empty object array can be compared with a number and gives a boolean-dtype result (no write to the read-only buffer).
    # GH#57130
    arr = np.array([], dtype=object)
    arr.flags.writeable = False
    idx = pd.Index(arr)
    result = idx > 69
    assert result.dtype == bool


def test_to_frame_column_rangeindex():
    # [FUNCTION DOC]
    # Converting a one-element integer index to a frame gives columns that are a range index (exact type).
    idx = pd.Index([1])
    result = idx.to_frame().columns
    expected = RangeIndex(1)
    tm.assert_index_equal(result, expected, exact=True)


def test_to_frame_name_tuple_multiindex():
    # [FUNCTION DOC]
    # Converting with a tuple as name produces a frame whose columns are a two-level multi-level index built from the tuple parts.
    idx = pd.Index([1])
    result = idx.to_frame(name=(1, 2))
    expected = pd.DataFrame([1], columns=MultiIndex.from_arrays([[1], [2]]), index=idx)
    tm.assert_frame_equal(result, expected)


def test_join_series_deprecated():
    # [FUNCTION DOC]
    # Joining an index with a series (instead of an index) emits the Pandas 4 deprecation warning ("Passing ... to ... is deprecated").
    # GH#62897
    idx = pd.Index([1, 2])
    ser = pd.Series([1, 2, 2])
    with tm.assert_produces_warning(
        Pandas4Warning, match="Passing .* to .* is deprecated"
    ):
        idx.join(ser)
