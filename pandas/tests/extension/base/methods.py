# [MODULE DOC]
# Base tests for assorted Series, DataFrame and array methods on extension arrays: hashing,
# value counts, apply/map, sorting and ranking, argmin/argmax, duplicated, unique, factorize,
# fillna limits and copying, combine, shift, diff, searchsorted, where, repeat, delete, insert and
# equality. Public contents: BaseMethodsTests mixin. Many tests rely on the layout described for
# the sorting and grouping fixtures in the conftest: data_for_sorting is (B, C, A) with A < B < C,
# data_missing_for_sorting is (B, NA, A), data_for_grouping has eight values. Boolean-like dtypes
# get special expected values because they have only two distinct values.
import inspect
import operator

import numpy as np
import pytest

from pandas._typing import Dtype

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
)
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort


class BaseMethodsTests:
    # [CLASS DOC]
    # Mixin with method tests. Class attribute _combine_le_expected_dtype (a numpy-backed
    # boolean extension dtype by default) names the dtype of the result of combining with a
    # less-or-equal comparison; subclasses override it for other boolean-like result dtypes.
    # Helper methods: _construct_for_combine_add and _test_searchsorted_bool_dtypes.
    """Various Series and DataFrame methods."""

    def test_hash_pandas_object(self, data):
        # [FUNCTION DOC]
        # Asserts the array's private hashing method (with utf-8 encoding, the default hash key and no
        # categorization) returns an unsigned 64-bit integer array with the same shape as the data.
        # _hash_pandas_object should return a uint64 ndarray of the same length
        # as the data
        from pandas.core.util.hashing import _default_hash_key

        res = data._hash_pandas_object(
            encoding="utf-8", hash_key=_default_hash_key, categorize=False
        )
        assert res.dtype == np.uint64
        assert res.shape == data.shape

    def test_value_counts_default_dropna(self, data):
        # [FUNCTION DOC]
        # Skips if the array has no value counts method. Otherwise inspects its signature and asserts
        # the dropna parameter defaults to true.
        # make sure we have consistent default dropna kwarg
        if not hasattr(data, "value_counts"):
            pytest.skip(f"value_counts is not implemented for {type(data)}")
        sig = inspect.signature(data.value_counts)
        kwarg = sig.parameters["dropna"]
        assert kwarg.default is True

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        # [FUNCTION DOC]
        # Parametrized over dropna. Value counts (sorted by index) of the full data must equal value
        # counts of the data with missing entries removed when dropna is true, or of the data itself when
        # false.
        # KNOWN ISSUE: because the comparison for the non-dropna case counts the same data on both sides
        # the second case compares a computation with itself.
        if dropna:
            other = all_data[~all_data.isna()]
        else:
            other = all_data

        result = pd.Series(all_data).value_counts(dropna=dropna).sort_index()
        expected = pd.Series(other).value_counts(dropna=dropna).sort_index()

        tm.assert_series_equal(result, expected)

    def test_value_counts_with_normalize(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Takes the unique values, counts them normalized and
        # compares to a series named proportion with equal shares (for categorical data the shares are
        # zero except for observed categories). The expected dtype is adjusted with special cases: float64
        # for NaN-based string dtype, pyarrow double for pyarrow-backed dtypes, and the nullable Float64
        # when the dtype's missing value is the NA singleton. Compares with series equality.
        # GH 33172
        data = data.unique()
        values = np.array(data[~data.isna()])
        ser = pd.Series(data, dtype=data.dtype)

        result = ser.value_counts(normalize=True).sort_index()

        if not isinstance(data, pd.Categorical):
            expected = pd.Series(
                [1 / len(values)] * len(values), index=result.index, name="proportion"
            )
        else:
            expected = pd.Series(0.0, index=result.index, name="proportion")
            expected[result > 0] = 1 / len(values)

        if isinstance(data.dtype, pd.StringDtype) and data.dtype.na_value is np.nan:
            # TODO: avoid special-casing
            expected = expected.astype("float64")
        elif getattr(data.dtype, "storage", "") == "pyarrow" or isinstance(
            data.dtype, pd.ArrowDtype
        ):
            # TODO: avoid special-casing
            expected = expected.astype("double[pyarrow]")
        elif na_value_for_dtype(data.dtype) is pd.NA:
            # TODO(GH#44692): avoid special-casing
            expected = expected.astype("Float64")

        tm.assert_series_equal(result, expected)

    def test_count(self, data_missing):
        # [FUNCTION DOC]
        # Frame count along columns for data_missing (missing, valid) gives zero then one.
        df = pd.DataFrame({"A": data_missing})
        result = df.count(axis="columns")
        expected = pd.Series([0, 1])
        tm.assert_series_equal(result, expected)

    def test_series_count(self, data_missing):
        # [FUNCTION DOC]
        # Series count of data_missing is one (regression for an upstream issue).
        # GH#26835
        ser = pd.Series(data_missing)
        result = ser.count()
        expected = 1
        assert result == expected

    def test_apply_simple_series(self, data):
        # [FUNCTION DOC]
        # Applying the identity-getting builtin to a series returns a series (smoke test).
        result = pd.Series(data).apply(id)
        assert isinstance(result, pd.Series)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing, na_action):
        # [FUNCTION DOC]
        # Parametrized over na_action none and ignore. The array's map with an identity function must
        # return a numpy array equal to the numpy conversion of the array.
        result = data_missing.map(lambda x: x, na_action=na_action)
        expected = data_missing.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort(self, data_for_sorting):
        # [FUNCTION DOC]
        # Series argsort of data_for_sorting gives a series with positions two, zero, one as the platform
        # index integer type (the result is later used for take).
        result = pd.Series(data_for_sorting).argsort()
        # argsort result gets passed to take, so should be np.intp
        expected = pd.Series(np.array([2, 0, 1], dtype=np.intp))
        tm.assert_series_equal(result, expected)

    def test_argsort_missing_array(self, data_missing_for_sorting):
        # [FUNCTION DOC]
        # Array argsort of data_missing_for_sorting gives positions two, zero, one (missing sorted
        # last) as an index-integer numpy array.
        result = data_missing_for_sorting.argsort()
        # argsort result gets passed to take, so should be np.intp
        expected = np.array([2, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort_missing(self, data_missing_for_sorting):
        # [FUNCTION DOC]
        # Series version of the missing argsort: a series holding positions two, zero, one.
        result = pd.Series(data_missing_for_sorting).argsort()
        expected = pd.Series(np.array([2, 0, 1], dtype=np.intp))
        tm.assert_series_equal(result, expected)

    def test_argmin_argmax(self, data_for_sorting, data_missing_for_sorting, na_value):
        # [FUNCTION DOC]
        # Regression for an upstream issue. For data_for_sorting the argmax is 1 (0 for boolean-like
        # dtypes, per the fixture's documentation) and argmin 2. For a take with repeated values the
        # first occurrence wins (argmax 3, or 1 for boolean-like; argmin 0). With missing values skipped,
        # argmax is 0 and argmin is 2.
        # GH 24382
        is_bool = data_for_sorting.dtype._is_boolean

        exp_argmax = 1
        exp_argmax_repeated = 3
        if is_bool:
            # See data_for_sorting docstring
            exp_argmax = 0
            exp_argmax_repeated = 1

        # data_for_sorting -> [B, C, A] with A < B < C
        assert data_for_sorting.argmax() == exp_argmax
        assert data_for_sorting.argmin() == 2

        # with repeated values -> first occurrence
        data = data_for_sorting.take([2, 0, 0, 1, 1, 2])
        assert data.argmax() == exp_argmax_repeated
        assert data.argmin() == 0

        # with missing values
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        assert data_missing_for_sorting.argmax() == 0
        assert data_missing_for_sorting.argmin() == 2

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_empty_array(self, method, data):
        # [FUNCTION DOC]
        # Parametrized over argmax and argmin. On an empty slice the method must raise a value error
        # whose message begins with the phrase about attempting to get a result.
        # GH 24382
        err_msg = "attempt to get"
        with pytest.raises(ValueError, match=err_msg):
            getattr(data[:0], method)()

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_all_na(self, method, data, na_value):
        # [FUNCTION DOC]
        # Parametrized over argmax and argmin. For an array of two missing values (skipping missing by
        # default, equivalent to empty) the method must raise the same value error as for empty.
        # all missing with skipna=True is the same as empty
        err_msg = "attempt to get"
        data_na = type(data)._from_sequence([na_value, na_value], dtype=data.dtype)
        with pytest.raises(ValueError, match=err_msg):
            getattr(data_na, method)()

    @pytest.mark.parametrize(
        "op_name, skipna, expected",
        [
            ("idxmax", True, 0),
            ("idxmin", True, 2),
            ("argmax", True, 0),
            ("argmin", True, 2),
            ("idxmax", False, -1),
            ("idxmin", False, -1),
            ("argmax", False, -1),
            ("argmin", False, -1),
        ],
    )
    def test_argreduce_series(
        self, data_missing_for_sorting, op_name, skipna, expected
    ):
        # [FUNCTION DOC]
        # Parametrized over idxmax, idxmin, argmax, argmin with skipna true or false. For
        # data_missing_for_sorting: skipping missing gives 0 for the max variants and 2 for the min
        # variants; not skipping must raise a value error saying an NA value was encountered.
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        ser = pd.Series(data_missing_for_sorting)
        if expected == -1:
            with pytest.raises(ValueError, match="Encountered an NA value"):
                getattr(ser, op_name)(skipna=skipna)
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            tm.assert_almost_equal(result, expected)

    def test_argmax_argmin_no_skipna_notimplemented(self, data_missing_for_sorting):
        # [FUNCTION DOC]
        # With skipna false, argmin and argmax on data_missing_for_sorting each raise a value error
        # about encountering an NA value (regression for an upstream issue; the test name mentions
        # not-implemented but the expected error is the value error).
        # GH#38733
        data = data_missing_for_sorting

        with pytest.raises(ValueError, match="Encountered an NA value"):
            data.argmin(skipna=False)

        with pytest.raises(ValueError, match="Encountered an NA value"):
            data.argmax(skipna=False)

    @pytest.mark.parametrize(
        "na_position, expected",
        [
            ("last", np.array([2, 0, 1], dtype=np.dtype("intp"))),
            ("first", np.array([1, 2, 0], dtype=np.dtype("intp"))),
        ],
    )
    def test_nargsort(self, data_missing_for_sorting, na_position, expected):
        # [FUNCTION DOC]
        # Parametrized over na_position last or first. The internal null-aware argsort helper on
        # data_missing_for_sorting gives positions two, zero, one for last and one, two, zero for
        # first.
        # GH 25439
        result = nargsort(data_missing_for_sorting, na_position=na_position)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        # [FUNCTION DOC]
        # Parametrized over ascending and using the sort_by_key fixture. Sorts a series of
        # data_for_sorting; ascending expects positions two, zero, one. Descending must be stable: if
        # there are only two unique values (boolean-like) expects zero, one, two; otherwise one, zero,
        # two.
        ser = pd.Series(data_for_sorting)
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        expected = ser.iloc[[2, 0, 1]]
        if not ascending:
            # GH 35922. Expect stable sort
            if ser.nunique() == 2:
                expected = ser.iloc[[0, 1, 2]]
            else:
                expected = ser.iloc[[1, 0, 2]]

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_missing(
        self, data_missing_for_sorting, ascending, sort_by_key
    ):
        # [FUNCTION DOC]
        # Parametrized over ascending, using sort_by_key. For data_missing_for_sorting, ascending puts
        # the missing value last (two, zero, one); descending gives zero, two, one (missing still last).
        ser = pd.Series(data_missing_for_sorting)
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        if ascending:
            expected = ser.iloc[[2, 0, 1]]
        else:
            expected = ser.iloc[[0, 2, 1]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(self, data_for_sorting, ascending):
        # [FUNCTION DOC]
        # Parametrized over ascending. Sorts a frame by an integer column then the extension column
        # and expects the ascending arrangement with index two, zero, one.
        # KNOWN ISSUE: the ascending parameter is never used; the sort call does not receive it, so both
        # parametrized cases run the same ascending test.
        df = pd.DataFrame({"A": [1, 2, 1], "B": data_for_sorting})
        result = df.sort_values(["A", "B"])
        expected = pd.DataFrame(
            {"A": [1, 1, 2], "B": data_for_sorting.take([2, 0, 1])}, index=[2, 0, 1]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_rank(self, data_for_sorting, ascending):
        # [FUNCTION DOC]
        # Parametrized over ascending. Series rank of data_for_sorting must be float typed; after casting
        # to float64 it equals 2, 3, 1 ascending or 2, 1, 3 descending. For boolean-like dtypes, ties
        # give 2.5, 2.5, 1 ascending or 1.5, 1.5, 3 descending.
        ser = pd.Series(data_for_sorting)
        result = ser.rank(ascending=ascending)
        # result should be float, but exact dtype (numpy/nullable/pyarrow) depends
        # so here just assert it is float and normalize to float64 for comparison
        assert is_float_dtype(result.dtype)
        result = result.astype("float64")
        if is_bool_dtype(ser.dtype):
            expected = pd.Series([2.5, 2.5, 1.0] if ascending else [1.5, 1.5, 3.0])
        else:
            expected = pd.Series([2.0, 3.0, 1.0] if ascending else [2.0, 1.0, 3.0])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["average", "min"])
    def test_rank_method(self, data_for_sorting, method):
        # [FUNCTION DOC]
        # Parametrized over method average and min. Ranks a series arranged B, A, B. Average gives a
        # float result equal to 2.5, 1, 2.5; min gives 2, 1, 2 and the result dtype may be integer or
        # float (the code notes it is inconsistent across arrays).
        ser = pd.Series(data_for_sorting.take([0, 2, 0]))
        result = ser.rank(method=method)
        if method == "average":
            assert is_float_dtype(result.dtype)
            result = result.astype("float64")
            expected = pd.Series([2.5, 1.0, 2.5])
        else:
            # TODO the exact dtype here is inconsistent across EAs (should all be int?)
            assert is_integer_dtype(result.dtype) or is_float_dtype(result.dtype)
            expected = pd.Series([2, 1, 2], dtype=result.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("na_option", ["keep", "top", "bottom"])
    def test_rank_missing(self, data_missing_for_sorting, na_option):
        # [FUNCTION DOC]
        # Parametrized over na_option keep, top, bottom. Ranks data_missing_for_sorting; result must be
        # float typed; keep gives 2, NaN, 1; top gives 3, 1, 2; bottom gives 2, 3, 1.
        ser = pd.Series(data_missing_for_sorting)
        result = ser.rank(na_option=na_option)
        assert is_float_dtype(result.dtype)
        result = result.astype("float64")
        if na_option == "keep":
            expected = pd.Series([2.0, np.nan, 1.0])
        elif na_option == "top":
            expected = pd.Series([3.0, 1.0, 2.0])
        else:
            # na_option == "bottom"
            expected = pd.Series([2.0, 3.0, 1.0])

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("keep", ["first", "last", False])
    def test_duplicated(self, data, keep):
        # [FUNCTION DOC]
        # Parametrized over keep first, last and false. For an array arranged x, y, x, y the boolean
        # mask is false,false,true,true for first; true,true,false,false for last; all true for false.
        arr = data.take([0, 1, 0, 1])
        result = arr.duplicated(keep=keep)
        if keep == "first":
            expected = np.array([False, False, True, True])
        elif keep == "last":
            expected = np.array([True, True, False, False])
        else:
            expected = np.array([True, True, True, True])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        # [FUNCTION DOC]
        # Parametrized over input wrapping (series or plain array) and the unique function (method or
        # top-level function). Two copies of the same element deduplicate to a single element of the
        # array class equal to that element.
        duplicated = box(data._from_sequence([data[0], data[0]], dtype=data.dtype))

        result = method(duplicated)

        assert len(result) == 1
        assert isinstance(result, type(data))
        assert result[0] == duplicated[0]

    def test_factorize(self, data_for_grouping):
        # [FUNCTION DOC]
        # Factorizes data_for_grouping with the missing sentinel enabled. Expected codes: 0,0,-1,-1,1,1,0,2
        # (only two unique values for boolean-like dtypes: 0,0,-1,-1,1,1,0,0), with uniques taken from
        # positions 0,4,7 (0,4 for boolean-like).
        codes, uniques = pd.factorize(data_for_grouping, use_na_sentinel=True)

        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values
            expected_codes = np.array([0, 0, -1, -1, 1, 1, 0, 0], dtype=np.intp)
            expected_uniques = data_for_grouping.take([0, 4])
        else:
            expected_codes = np.array([0, 0, -1, -1, 1, 1, 0, 2], dtype=np.intp)
            expected_uniques = data_for_grouping.take([0, 4, 7])

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_extension_array_equal(uniques, expected_uniques)

    def test_factorize_equivalence(self, data_for_grouping):
        # [FUNCTION DOC]
        # Asserts the top-level factorize and the array method give equal codes and uniques; the
        # uniques contain no duplicates and have the array's dtype.
        codes_1, uniques_1 = pd.factorize(data_for_grouping, use_na_sentinel=True)
        codes_2, uniques_2 = data_for_grouping.factorize(use_na_sentinel=True)

        tm.assert_numpy_array_equal(codes_1, codes_2)
        tm.assert_extension_array_equal(uniques_1, uniques_2)
        assert len(uniques_1) == len(pd.unique(uniques_1))
        assert uniques_1.dtype == data_for_grouping.dtype

    def test_factorize_empty(self, data):
        # [FUNCTION DOC]
        # Factorizing an empty array returns an empty intp code array and an empty array of the same
        # type and dtype.
        codes, uniques = pd.factorize(data[:0])
        expected_codes = np.array([], dtype=np.intp)
        expected_uniques = type(data)._from_sequence([], dtype=data[:0].dtype)

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_extension_array_equal(uniques, expected_uniques)

    def test_fillna_limit_frame(self, data_missing):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Frame column arranged missing, valid, missing, valid: filling
        # with the valid value and limit one fills only the first missing entry.
        # GH#58001
        df = pd.DataFrame({"A": data_missing.take([0, 1, 0, 1])})
        expected = pd.DataFrame({"A": data_missing.take([1, 1, 0, 1])})
        result = df.fillna(value=data_missing[1], limit=1)
        tm.assert_frame_equal(result, expected)

    def test_fillna_limit_series(self, data_missing):
        # [FUNCTION DOC]
        # Series version of the limit test: same arrangement and same expected output.
        # GH#58001
        ser = pd.Series(data_missing.take([0, 1, 0, 1]))
        expected = pd.Series(data_missing.take([1, 1, 0, 1]))
        result = ser.fillna(value=data_missing[1], limit=1)
        tm.assert_series_equal(result, expected)

    def test_fillna_copy_frame(self, data_missing):
        # [FUNCTION DOC]
        # Fills a frame without missing values, then assigns into the result and asserts the original
        # frame is unchanged (the result must not share memory with the input).
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr})
        df_orig = df.copy()

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        result.iloc[0, 0] = filled_val

        tm.assert_frame_equal(df, df_orig)

    def test_fillna_copy_series(self, data_missing):
        # [FUNCTION DOC]
        # Series version of the copy test: a series with no copy of its array, filled, then mutated;
        # the original series must remain unchanged.
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr, copy=False)
        ser_orig = ser.copy()

        filled_val = ser[0]
        result = ser.fillna(filled_val)
        result.iloc[0] = filled_val

        tm.assert_series_equal(ser, ser_orig)

    def test_fillna_length_mismatch(self, data_missing):
        # [FUNCTION DOC]
        # Filling with an array whose length differs from the data must raise a value error saying the
        # length of value does not match (the dot in the pattern is a regex wildcard).
        msg = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg):
            data_missing.fillna(data_missing.take([1]))

    # Subclasses can override if we expect e.g Sparse[bool], boolean, pyarrow[bool]
    _combine_le_expected_dtype: Dtype = NumpyEADtype("bool")

    def test_combine_le(self, data_repeated):
        # [FUNCTION DOC]
        # Regression for an upstream issue. With two arrays from the data_repeated fixture, combining two
        # series with a less-or-equal function equals a series of the pointwise comparisons with the
        # class-level expected dtype; combining with a scalar (the first element) behaves the same.
        # GH 20825
        # Test that combine works when doing a <= (le) comparison
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            pd.array(
                [
                    a <= b
                    for (a, b) in zip(list(orig_data1), list(orig_data2), strict=True)
                ],
                dtype=self._combine_le_expected_dtype,
            )
        )
        tm.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            pd.array(
                [a <= val for a in list(orig_data1)],
                dtype=self._combine_le_expected_dtype,
            )
        )
        tm.assert_series_equal(result, expected)

    def _construct_for_combine_add(self, left, right):
        # [FUNCTION DOC]
        # HELPER: for test_combine_add; builds the expected array of pointwise sums. Inputs: left (an
        # array) and right (an array of the same type, or a scalar). Returns an array of left's dtype
        # whose elements are the sums of the pairs (elementwise with right) or of each element with the
        # scalar. May raise a type error if the scalars cannot be added.
        if isinstance(right, type(left)):
            return left._from_sequence(
                [a + b for (a, b) in zip(list(left), list(right), strict=True)],
                dtype=left.dtype,
            )
        else:
            return left._from_sequence(
                [a + right for a in list(left)],
                dtype=left.dtype,
            )

    def test_combine_add(self, data_repeated):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Combines two series with an addition function. First
        # computes the expected pointwise sums with the helper (overflow warnings suppressed); if
        # that raises a type error the series combine must also raise a type error and the test ends.
        # Otherwise the combined series must equal the expected sums, both for series-with-series and for
        # series-with-scalar combination.
        # GH 20825
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)

        # Check if the operation is supported pointwise for our scalars. If not,
        #  we will expect Series.combine to raise as well.
        try:
            with np.errstate(over="ignore"):
                arr = self._construct_for_combine_add(orig_data1, orig_data2)
        except TypeError:
            # If the operation is not supported pointwise for our scalars,
            #  then Series.combine should also raise
            with pytest.raises(TypeError):
                s1.combine(s2, lambda x1, x2: x1 + x2)
            return
        expected = pd.Series(arr)

        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        tm.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        arr = self._construct_for_combine_add(orig_data1, val)
        expected = pd.Series(arr)
        tm.assert_series_equal(result, expected)

    def test_combine_first(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Combine-first of a series of the first three elements with
        # a series of elements two to four (indexed two to four) equals the first five elements.
        # https://github.com/pandas-dev/pandas/issues/24147
        a = pd.Series(data[:3])
        b = pd.Series(data[2:5], index=[2, 3, 4])
        result = a.combine_first(b)
        expected = pd.Series(data[:5])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("frame", [True, False])
    @pytest.mark.parametrize(
        "periods, indices",
        [(-2, [2, 3, 4, -1, -1]), (0, [0, 1, 2, 3, 4]), (2, [-1, -1, 0, 1, 2])],
    )
    def test_container_shift(self, data, frame, periods, indices):
        # [FUNCTION DOC]
        # Parametrized over frame vs series and over periods -2, 0, 2 with the matching expected
        # fill positions. Shifts a series (or a two-column frame with an integer column) and compares with
        # the array taken at positions with missing fill at the vacated entries.
        # https://github.com/pandas-dev/pandas/issues/22386
        subset = data[:5]
        data = pd.Series(subset, name="A")
        expected = pd.Series(subset.take(indices, allow_fill=True), name="A")

        if frame:
            result = data.to_frame(name="A").assign(B=1).shift(periods)
            expected = pd.concat(
                [expected, pd.Series([1] * 5, name="B").shift(periods)], axis=1
            )
            compare = tm.assert_frame_equal
        else:
            result = data.shift(periods)
            compare = tm.assert_series_equal

        compare(result, expected)

    def test_shift_0_periods(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue: shifting by zero returns a copy. After shifting, mutates the
        # original array (changing the first element) and asserts the shifted result is unaffected.
        # Mutates the data fixture.
        # GH#33856 shifting with periods=0 should return a copy, not same obj
        result = data.shift(0)
        assert data[0] != data[1]  # otherwise below is invalid
        data[0] = data[1]
        assert result[0] != result[1]  # i.e. not the same object/view

    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(self, data, periods):
        # [FUNCTION DOC]
        # Parametrized over periods 1 and -2. Skips if the array does not support subtraction (exclusive
        # or for booleans). Series difference must equal the elementwise operation between the array and
        # its shift. For a frame with an additional float column the difference of the float column is
        # NaN first (period 1) or NaN last two (period -2), zeros otherwise.
        data = data[:5]
        if is_bool_dtype(data.dtype):
            op = operator.xor
        else:
            op = operator.sub
        try:
            # does this array implement ops?
            op(data, data)
        except Exception:
            pytest.skip(f"{type(data)} does not support diff")
        s = pd.Series(data)
        result = s.diff(periods)
        expected = pd.Series(op(data, data.shift(periods)))
        tm.assert_series_equal(result, expected)

        df = pd.DataFrame({"A": data, "B": [1.0] * 5})
        result = df.diff(periods)
        if periods == 1:
            b = [np.nan, 0, 0, 0, 0]
        else:
            b = [0, 0, 0, np.nan, np.nan]
        expected = pd.DataFrame({"A": expected, "B": b})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, indices",
        [[-4, [-1, -1]], [-1, [1, -1]], [0, [0, 1]], [1, [-1, 0]], [4, [-1, -1]]],
    )
    def test_shift_non_empty_array(self, data, periods, indices):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Parametrized over periods and the take positions with fill.
        # Shifting a two-element array equals taking the given positions with missing fill (for example
        # shift by four gives all missing).
        # https://github.com/pandas-dev/pandas/issues/23911
        subset = data[:2]
        result = subset.shift(periods)
        expected = subset.take(indices, allow_fill=True)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("periods", [-4, -1, 0, 1, 4])
    def test_shift_empty_array(self, data, periods):
        # [FUNCTION DOC]
        # Parametrized over several periods. Shifting an empty array returns an equal empty array.
        # https://github.com/pandas-dev/pandas/issues/23911
        empty = data[:0]
        result = empty.shift(periods)
        expected = empty
        tm.assert_extension_array_equal(result, expected)

    def test_shift_zero_copies(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue: shifting by zero returns a new object, and shifting an empty
        # slice also returns a new object.
        # KNOWN ISSUE: the second assertion compares the shifted empty slice against the original full
        # array object, which is trivially distinct, so it does not verify that the empty case copies.
        # GH#31502
        result = data.shift(0)
        assert result is not data

        result = data[:0].shift(2)
        assert result is not data

    def test_shift_fill_value(self, data):
        # [FUNCTION DOC]
        # Shifting the first four elements by one with the first element as fill value gives positions
        # 0,0,1,2; shifting by minus two gives 2,3,0,0.
        arr = data[:4]
        fill_value = data[0]
        result = arr.shift(1, fill_value=fill_value)
        expected = data.take([0, 0, 1, 2])
        tm.assert_extension_array_equal(result, expected)

        result = arr.shift(-2, fill_value=fill_value)
        expected = data.take([2, 3, 0, 0])
        tm.assert_extension_array_equal(result, expected)

    def test_not_hashable(self, data):
        # [FUNCTION DOC]
        # Hashing the array must raise a type error saying the type is unhashable (arrays are mutable).
        # We are in general mutable, so not hashable
        with pytest.raises(TypeError, match="unhashable type"):
            hash(data)

    def test_hash_pandas_object_works(self, data, as_frame):
        # [FUNCTION DOC]
        # Regression for an upstream issue. Using the as_frame fixture to wrap the series in a frame
        # or not, the pandas-level hashing helper applied twice gives equal results (deterministic).
        # https://github.com/pandas-dev/pandas/issues/23066
        data = pd.Series(data)
        if as_frame:
            data = data.to_frame()
        a = pd.util.hash_pandas_object(data)
        b = pd.util.hash_pandas_object(data)
        tm.assert_equal(a, b)

    def test_searchsorted(self, data_for_sorting, as_series):
        # [FUNCTION DOC]
        # Uses as_series fixture. For boolean-like dtypes delegates to the boolean helper and returns
        # its result. Otherwise sorts data_for_sorting into ascending order (a, b, c), optionally boxed in a
        # series, and asserts left and right insertion positions for each of the three values (0/1,
        # 1/2, 2/3), that an array of positions zero and two is found at zero and two (as intp), and that
        # a custom sorter argument is honored.
        if data_for_sorting.dtype._is_boolean:
            return self._test_searchsorted_bool_dtypes(data_for_sorting, as_series)

        b, c, a = data_for_sorting
        arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]

        if as_series:
            arr = pd.Series(arr)
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side="right") == 1

        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side="right") == 2

        assert arr.searchsorted(c) == 2
        assert arr.searchsorted(c, side="right") == 3

        result = arr.searchsorted(arr.take([0, 2]))
        expected = np.array([0, 2], dtype=np.intp)

        tm.assert_numpy_array_equal(result, expected)

        # sorter
        sorter = np.array([1, 2, 0])
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    def _test_searchsorted_bool_dtypes(self, data_for_sorting, as_series):
        # [FUNCTION DOC]
        # HELPER: boolean-like variant of the search test, called by test_searchsorted. Input:
        # data_for_sorting (only its dtype is used) and as_series. Builds a two-element array of True and
        # False, sorts it, and asserts insertion positions for each value on each side (0/1, 1/2), lookup
        # of both elements as an array (positions 0, 1) and use of a sorter argument.
        # We call this from test_searchsorted in cases where we have a
        #  boolean-like dtype. The non-bool test assumes we have more than 2
        #  unique values.
        dtype = data_for_sorting.dtype
        data_for_sorting = pd.array([True, False], dtype=dtype)
        b, a = data_for_sorting
        arr = type(data_for_sorting)._from_sequence([a, b], dtype=dtype)

        if as_series:
            arr = pd.Series(arr)
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side="right") == 1

        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side="right") == 2

        result = arr.searchsorted(arr.take([0, 1]))
        expected = np.array([0, 1], dtype=np.intp)

        tm.assert_numpy_array_equal(result, expected)

        # sorter
        sorter = np.array([1, 0])
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    def test_where_series(self, data, na_value, as_frame):
        # [FUNCTION DOC]
        # Uses as_frame and na_value. Builds an array arranged a,a,b,b, wrapping as series or one-column
        # frame. where with a mask true,true,false,false gives missing in the false positions; mask with
        # inverted condition applied in place gives the same. With an array other and a different mask,
        # where takes the other's values where the mask is false (expected a,b,b,b), again with in-place
        # mask. Requires the first two elements to differ.
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        orig = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
        ser = orig.copy()
        cond = np.array([True, True, False, False])

        if as_frame:
            ser = ser.to_frame(name="a")
            cond = cond.reshape(-1, 1)

        result = ser.where(cond)
        expected = pd.Series(
            cls._from_sequence([a, a, na_value, na_value], dtype=data.dtype)
        )

        if as_frame:
            expected = expected.to_frame(name="a")
        tm.assert_equal(result, expected)

        ser.mask(~cond, inplace=True)
        tm.assert_equal(ser, expected)

        # array other
        ser = orig.copy()
        if as_frame:
            ser = ser.to_frame(name="a")
        cond = np.array([True, False, True, True])
        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        if as_frame:
            other = pd.DataFrame({"a": other})
            cond = pd.DataFrame({"a": cond})
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        if as_frame:
            expected = expected.to_frame(name="a")
        tm.assert_equal(result, expected)

        ser.mask(~cond, other, inplace=True)
        tm.assert_equal(ser, expected)

    @pytest.mark.parametrize("repeats", [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data, repeats, as_series, use_numpy):
        # [FUNCTION DOC]
        # Parametrized over repeats 0, 1, 2 and a list of per-element counts; uses the as_series and
        # use_numpy fixtures. Repeats a three-element array (or series) with the method or numpy's
        # function and compares with the manually expanded list; for series the index is repeated the
        # same way.
        arr = type(data)._from_sequence(data[:3], dtype=data.dtype)
        if as_series:
            arr = pd.Series(arr)

        result = np.repeat(arr, repeats) if use_numpy else arr.repeat(repeats)

        repeats = [repeats] * 3 if isinstance(repeats, int) else repeats
        expected = [x for x, n in zip(arr, repeats, strict=True) for _ in range(n)]
        expected = type(data)._from_sequence(expected, dtype=data.dtype)
        if as_series:
            expected = pd.Series(expected, index=arr.index.repeat(repeats))

        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "repeats, kwargs, error, msg",
        [
            (2, {"axis": 1}, ValueError, "axis"),
            (-1, {}, ValueError, "negative"),
            ([1, 2], {}, ValueError, "shape"),
            (2, {"foo": "bar"}, TypeError, "'foo'"),
        ],
    )
    def test_repeat_raises(self, data, repeats, kwargs, error, msg, use_numpy):
        # [FUNCTION DOC]
        # Parametrized over error cases: an axis keyword raises a value error mentioning axis; a negative
        # count raises a value error mentioning negative; a counts list of the wrong length raises a value
        # error mentioning shape; an unknown keyword raises a type error naming it. Run via the method
        # or numpy depending on use_numpy.
        with pytest.raises(error, match=msg):
            if use_numpy:
                np.repeat(data, repeats, **kwargs)
            else:
                data.repeat(repeats, **kwargs)

    def test_delete(self, data):
        # [FUNCTION DOC]
        # Deleting position zero yields the data without the first element; deleting positions one and
        # three yields the concatenation of the remaining pieces.
        result = data.delete(0)
        expected = data[1:]
        tm.assert_extension_array_equal(result, expected)

        result = data.delete([1, 3])
        expected = data._concat_same_type([data[[0]], data[[2]], data[4:]])
        tm.assert_extension_array_equal(result, expected)

    def test_insert(self, data):
        # [FUNCTION DOC]
        # Inserting the first element back at position zero (and at the equivalent negative position)
        # recovers the data. Inserting the last element into the middle (position four of the array
        # without it) recovers the data with that element at position four and the rest shifted.
        # insert at the beginning
        result = data[1:].insert(0, data[0])
        tm.assert_extension_array_equal(result, data)

        result = data[1:].insert(-len(data[1:]), data[0])
        tm.assert_extension_array_equal(result, data)

        # insert at the middle
        result = data[:-1].insert(4, data[-1])

        taker = np.arange(len(data))
        taker[5:] = taker[4:-1]
        taker[4] = len(data) - 1
        expected = data.take(taker)
        tm.assert_extension_array_equal(result, expected)

    def test_insert_invalid(self, data, invalid_scalar):
        # [FUNCTION DOC]
        # Using the invalid_scalar fixture, inserting at the start, the middle and the end must raise a
        # type error or a value error.
        item = invalid_scalar

        with pytest.raises((TypeError, ValueError)):
            data.insert(0, item)

        with pytest.raises((TypeError, ValueError)):
            data.insert(4, item)

        with pytest.raises((TypeError, ValueError)):
            data.insert(len(data) - 1, item)

    def test_insert_invalid_loc(self, data):
        # [FUNCTION DOC]
        # Inserting one past the end or one before the start raises an index error; a non-integer
        # location (1.5) raises a type error, matching numpy's behavior.
        ub = len(data)

        with pytest.raises(IndexError):
            data.insert(ub + 1, data[0])

        with pytest.raises(IndexError):
            data.insert(-ub - 1, data[0])

        with pytest.raises(TypeError):
            # we expect TypeError here instead of IndexError to match np.insert
            data.insert(1.5, data[0])

    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        # [FUNCTION DOC]
        # Parametrized over box types (array, series, frame); uses as_series and na_value fixtures. For
        # the boxed data, an all-same-element version and an all-missing version, asserts the equality
        # method returns exactly true or false (a real Python bool). Equal to itself and its copy; not
        # equal to different data or missing data or a different length; empty slices are equal; not equal
        # to none or to a scalar element.
        # KNOWN ISSUE: the as_series fixture is requested but unused.
        data2 = type(data)._from_sequence([data[0]] * len(data), dtype=data.dtype)
        data_na = type(data)._from_sequence([na_value] * len(data), dtype=data.dtype)

        data = tm.box_expected(data, box, transpose=False)
        data2 = tm.box_expected(data2, box, transpose=False)
        data_na = tm.box_expected(data_na, box, transpose=False)

        # we are asserting with `is True/False` explicitly, to test that the
        # result is an actual Python bool, and not something "truthy"

        assert data.equals(data) is True
        assert data.equals(data.copy()) is True

        # unequal other data
        assert data.equals(data2) is False
        assert data.equals(data_na) is False

        # different length
        assert data[:2].equals(data[:3]) is False

        # empty are equal
        assert data[:0].equals(data[:0]) is True

        # other types
        assert data.equals(None) is False
        assert data[[0]].equals(data[0]) is False

    def test_equals_same_data_different_object(self, data):
        # [FUNCTION DOC]
        # Regression for an upstream issue: two series built from the same array compare equal.
        # https://github.com/pandas-dev/pandas/issues/34660
        assert pd.Series(data).equals(pd.Series(data))
