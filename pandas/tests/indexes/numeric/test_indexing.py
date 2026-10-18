# [MODULE DOC]
# Tests for indexing behavior of numeric Index objects (int64, uint64, float64 and
# masked or arrow-backed numeric extension types): get_loc, get_indexer with fill
# methods and tolerance, where, take, membership, slice_locs and slice bounds.
import numpy as np
import pytest

from pandas.errors import InvalidIndexError

from pandas import (
    NA,
    Index,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowExtensionArray,
    FloatingArray,
)


class TestGetLoc:
    # [CLASS DOC]
    # Groups get_loc tests for numeric indexes.
    def test_get_loc(self):
        # [FUNCTION DOC]
        # Position lookup of an existing integer label.
        index = Index([0, 1, 2])
        assert index.get_loc(1) == 1

    def test_get_loc_raises_bad_label(self):
        # [FUNCTION DOC]
        # A list-like label raises an invalid-index error showing the list.
        index = Index([0, 1, 2])
        with pytest.raises(InvalidIndexError, match=r"\[1, 2\]"):
            index.get_loc([1, 2])

    def test_get_loc_float64(self):
        # [FUNCTION DOC]
        # On a float index, strings, absent floats and booleans raise key errors showing
        # the key.
        idx = Index([0.0, 1.0, 2.0], dtype=np.float64)

        with pytest.raises(KeyError, match="^'foo'$"):
            idx.get_loc("foo")
        with pytest.raises(KeyError, match=r"^1\.5$"):
            idx.get_loc(1.5)
        with pytest.raises(KeyError, match="^True$"):
            idx.get_loc(True)
        with pytest.raises(KeyError, match="^False$"):
            idx.get_loc(False)

    def test_get_loc_na(self):
        # [FUNCTION DOC]
        # NaN lookups in float indexes: unique NaN gives its position; with repeated
        # NaN, slice_locs on NaN raises a key error saying the left bound is non-unique,
        # both when the NaN positions could be a slice and when they cannot.
        idx = Index([np.nan, 1, 2], dtype=np.float64)
        assert idx.get_loc(1) == 1
        assert idx.get_loc(np.nan) == 0

        idx = Index([np.nan, 1, np.nan], dtype=np.float64)
        assert idx.get_loc(1) == 1

        # representable by slice [0:2:2]
        msg = "'Cannot get left slice bound for non-unique label: nan'"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)
        # not representable by slice
        idx = Index([np.nan, 1, np.nan, np.nan], dtype=np.float64)
        assert idx.get_loc(1) == 1
        msg = "'Cannot get left slice bound for non-unique label: nan"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)

    def test_get_loc_missing_nan(self):
        # [FUNCTION DOC]
        # Absent label and NaN raise key errors; list-like raises an invalid-index error
        # (GH#8569).
        # GH#8569
        idx = Index([1, 2], dtype=np.float64)
        assert idx.get_loc(1) == 0
        with pytest.raises(KeyError, match=r"^3$"):
            idx.get_loc(3)
        with pytest.raises(KeyError, match="^nan$"):
            idx.get_loc(np.nan)
        with pytest.raises(InvalidIndexError, match=r"\[nan\]"):
            # listlike/non-hashable raises TypeError
            idx.get_loc([np.nan])

    @pytest.mark.parametrize("vals", [[1], [1.0], [Timestamp("2019-12-31")], ["test"]])
    def test_get_loc_float_index_nan_with_method(self, vals):
        # [FUNCTION DOC]
        # Parametrized over index value kinds. get_loc of NaN raises a key error
        # (GH#39382).
        # GH#39382
        idx = Index(vals)
        with pytest.raises(KeyError, match="nan"):
            idx.get_loc(np.nan)

    @pytest.mark.parametrize("dtype", ["f8", "i8", "u8"])
    def test_get_loc_numericindex_none_raises(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over float, signed and unsigned dtypes with a very large index.
        # get_loc of none raises a key error when the lookup goes through ordered
        # search.
        # case that goes through searchsorted and key is non-comparable to values
        arr = np.arange(10**7, dtype=dtype)
        idx = Index(arr)
        with pytest.raises(KeyError, match="None"):
            idx.get_loc(None)

    def test_get_loc_overflows(self):
        # [FUNCTION DOC]
        # A key beyond the 64-bit integer limit raises key errors both from the index
        # and from its engine.
        # unique but non-monotonic goes through IndexEngine.mapping.get_item
        idx = Index([0, 2, 1])

        val = np.iinfo(np.int64).max + 1

        with pytest.raises(KeyError, match=str(val)):
            idx.get_loc(val)
        with pytest.raises(KeyError, match=str(val)):
            idx._engine.get_loc(val)


class TestGetIndexer:
    # [CLASS DOC]
    # Groups get_indexer tests for numeric indexes.
    def test_get_indexer(self):
        # [FUNCTION DOC]
        # Exact-match indexer positions with -1 for absent labels.
        index1 = Index([1, 2, 3, 4, 5])
        index2 = Index([2, 4, 6])

        r1 = index1.get_indexer(index2)
        e1 = np.array([1, 3, -1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize(
        "expected,method",
        [
            ([-1, 0, 0, 1, 1], "pad"),
            ([-1, 0, 0, 1, 1], "ffill"),
            ([0, 0, 1, 1, 2], "backfill"),
            ([0, 0, 1, 1, 2], "bfill"),
        ],
    )
    def test_get_indexer_methods(self, reverse, expected, method):
        # [FUNCTION DOC]
        # Parametrized over reversal and the pad, ffill, backfill, bfill methods. Checks
        # fill positions, and that reversing the target reverses the result.
        index1 = Index([1, 2, 3, 4, 5])
        index2 = Index([2, 4, 6])
        expected = np.array(expected, dtype=np.intp)
        if reverse:
            index1 = index1[::-1]
            expected = expected[::-1]

        result = index2.get_indexer(index1, method=method)
        tm.assert_almost_equal(result, expected)

    def test_get_indexer_invalid(self):
        # [FUNCTION DOC]
        # A tolerance or limit without a fill method raises a value error (GH10411).
        # GH10411
        index = Index(np.arange(10))

        with pytest.raises(ValueError, match="tolerance argument"):
            index.get_indexer([1, 0], tolerance=1)

        with pytest.raises(ValueError, match="limit argument"):
            index.get_indexer([1, 0], limit=1)

    @pytest.mark.parametrize(
        "method, tolerance, indexer, expected",
        [
            ("pad", None, [0, 5, 9], [0, 5, 9]),
            ("backfill", None, [0, 5, 9], [0, 5, 9]),
            ("nearest", None, [0, 5, 9], [0, 5, 9]),
            ("pad", 0, [0, 5, 9], [0, 5, 9]),
            ("backfill", 0, [0, 5, 9], [0, 5, 9]),
            ("nearest", 0, [0, 5, 9], [0, 5, 9]),
            ("pad", None, [0.2, 1.8, 8.5], [0, 1, 8]),
            ("backfill", None, [0.2, 1.8, 8.5], [1, 2, 9]),
            ("nearest", None, [0.2, 1.8, 8.5], [0, 2, 9]),
            ("pad", 1, [0.2, 1.8, 8.5], [0, 1, 8]),
            ("backfill", 1, [0.2, 1.8, 8.5], [1, 2, 9]),
            ("nearest", 1, [0.2, 1.8, 8.5], [0, 2, 9]),
            ("pad", 0.2, [0.2, 1.8, 8.5], [0, -1, -1]),
            ("backfill", 0.2, [0.2, 1.8, 8.5], [-1, 2, -1]),
            ("nearest", 0.2, [0.2, 1.8, 8.5], [0, 2, -1]),
        ],
    )
    def test_get_indexer_nearest(self, method, tolerance, indexer, expected):
        # [FUNCTION DOC]
        # Parametrized over fill method, scalar tolerance, target values and expected
        # positions. Checks pad, backfill and nearest results for exact and fractional
        # targets and tolerances.
        index = Index(np.arange(10))

        actual = index.get_indexer(indexer, method=method, tolerance=tolerance)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize("listtype", [list, tuple, Series, np.array])
    @pytest.mark.parametrize(
        "tolerance, expected",
        [
            [[0.3, 0.3, 0.1], [0, 2, -1]],
            [[0.2, 0.1, 0.1], [0, -1, -1]],
            [[0.1, 0.5, 0.5], [-1, 2, 9]],
        ],
    )
    def test_get_indexer_nearest_listlike_tolerance(
        self, tolerance, expected, listtype
    ):
        # [FUNCTION DOC]
        # Parametrized over container type (list, tuple, Series, array) and tolerance
        # values. Nearest lookup with an element-wise tolerance gives expected
        # positions.
        index = Index(np.arange(10))

        actual = index.get_indexer(
            [0.2, 1.8, 8.5], method="nearest", tolerance=listtype(tolerance)
        )
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    def test_get_indexer_nearest_error(self):
        # [FUNCTION DOC]
        # Nearest with a limit raises a value error; a tolerance of different length
        # than the target raises a value error about size.
        index = Index(np.arange(10))
        with pytest.raises(ValueError, match="limit argument"):
            index.get_indexer([1, 0], method="nearest", limit=1)

        with pytest.raises(ValueError, match="tolerance size must match"):
            index.get_indexer([1, 0], method="nearest", tolerance=[1, 2, 3])

    @pytest.mark.parametrize(
        "method,expected",
        [("pad", [8, 7, 0]), ("backfill", [9, 8, 1]), ("nearest", [9, 7, 0])],
    )
    def test_get_indexer_nearest_decreasing(self, method, expected):
        # [FUNCTION DOC]
        # Parametrized over method and expected positions. On a decreasing index exact
        # targets map to expected positions and fractional targets follow
        # pad/backfill/nearest semantics.
        index = Index(np.arange(10))[::-1]

        actual = index.get_indexer([0, 5, 9], method=method)
        tm.assert_numpy_array_equal(actual, np.array([9, 4, 0], dtype=np.intp))

        actual = index.get_indexer([0.2, 1.8, 8.5], method=method)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize("idx_dtype", ["int64", "float64", "uint64", "range"])
    @pytest.mark.parametrize("method", ["get_indexer", "get_indexer_non_unique"])
    def test_get_indexer_numeric_index_boolean_target(self, method, idx_dtype):
        # [FUNCTION DOC]
        # Parametrized over index types (including a range) and both indexer methods. A
        # boolean target never matches numeric values: all -1, and for the non-unique
        # variant all targets reported as missing (GH 16877).
        # GH 16877

        if idx_dtype == "range":
            numeric_index = RangeIndex(4)
        else:
            numeric_index = Index(np.arange(4, dtype=idx_dtype))

        other = Index([True, False, True])

        result = getattr(numeric_index, method)(other)
        expected = np.array([-1, -1, -1], dtype=np.intp)
        if method == "get_indexer":
            tm.assert_numpy_array_equal(result, expected)
        else:
            missing = np.arange(3, dtype=np.intp)
            tm.assert_numpy_array_equal(result[0], expected)
            tm.assert_numpy_array_equal(result[1], missing)

    @pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
    def test_get_indexer_with_method_numeric_vs_bool(self, method):
        # [FUNCTION DOC]
        # Using a fill method between numeric and boolean indexes (either direction)
        # raises a type error about comparison.
        left = Index([1, 2, 3])
        right = Index([True, False])

        with pytest.raises(TypeError, match="Cannot compare"):
            left.get_indexer(right, method=method)

        with pytest.raises(TypeError, match="Cannot compare"):
            right.get_indexer(left, method=method)

    def test_get_indexer_numeric_vs_bool(self):
        # [FUNCTION DOC]
        # Without a fill method, numeric versus boolean gives all -1 in both directions
        # for both indexer methods.
        left = Index([1, 2, 3])
        right = Index([True, False])

        res = left.get_indexer(right)
        expected = -1 * np.ones(len(right), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = right.get_indexer(left)
        expected = -1 * np.ones(len(left), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = left.get_indexer_non_unique(right)[0]
        expected = -1 * np.ones(len(right), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = right.get_indexer_non_unique(left)[0]
        expected = -1 * np.ones(len(left), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

    def test_get_indexer_float64(self):
        # [FUNCTION DOC]
        # Float index indexer for exact match and for pad, backfill, nearest on
        # fractional targets.
        idx = Index([0.0, 1.0, 2.0], dtype=np.float64)
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        target = [-0.1, 0.5, 1.1]
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )

    def test_get_indexer_nan(self):
        # [FUNCTION DOC]
        # Indexer of NaN in a float index returns the NaN position (GH#7820).
        # GH#7820
        result = Index([1, 2, np.nan], dtype=np.float64).get_indexer([np.nan])
        expected = np.array([2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_int64(self):
        # [FUNCTION DOC]
        # Integer index indexer for exact, pad and backfill.
        index = Index(range(0, 20, 2), dtype=np.int64)
        target = Index(np.arange(10), dtype=np.int64)
        indexer = index.get_indexer(target)
        expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Index(np.arange(10), dtype=np.int64)
        indexer = index.get_indexer(target, method="pad")
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Index(np.arange(10), dtype=np.int64)
        indexer = index.get_indexer(target, method="backfill")
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_uint64(self):
        # [FUNCTION DOC]
        # Same for large unsigned values.
        index_large = Index(
            [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25],
            dtype=np.uint64,
        )
        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        indexer = index_large.get_indexer(target)
        expected = np.array([0, -1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        indexer = index_large.get_indexer(target, method="pad")
        expected = np.array([0, 0, 1, 2, 3, 4, 4, 4, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        indexer = index_large.get_indexer(target, method="backfill")
        expected = np.array([0, 1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    @pytest.mark.parametrize("val, val2", [(4, 5), (4, 4), (4, NA), (NA, NA)])
    def test_get_loc_masked(self, val, val2, any_numeric_ea_and_arrow_dtype):
        # [FUNCTION DOC]
        # Parametrized over trailing value pairs and numeric extension or arrow dtypes.
        # get_loc finds an existing value and raises a key error for an absent value
        # (GH#39133).
        # GH#39133
        idx = Index([1, 2, 3, val, val2], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.get_loc(2)
        assert result == 1

        with pytest.raises(KeyError, match="9"):
            idx.get_loc(9)

    def test_get_loc_masked_na(self, any_numeric_ea_and_arrow_dtype):
        # [FUNCTION DOC]
        # get_loc of NA returns the single position, a boolean mask for repeated NA, and
        # raises a key error when none exists (GH#39133).
        # GH#39133
        idx = Index([1, 2, NA], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.get_loc(NA)
        assert result == 2

        idx = Index([1, 2, NA, NA], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.get_loc(NA)
        tm.assert_numpy_array_equal(result, np.array([False, False, True, True]))

        idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
        with pytest.raises(KeyError, match="NA"):
            idx.get_loc(NA)

    def test_get_loc_masked_na_and_nan(self, using_nan_is_na):
        # [FUNCTION DOC]
        # Index of nullable floats with both a masked NA and a real NaN. Depending on
        # the nan-is-na option NA and NaN are treated as the same (mask result) or
        # separately (distinct positions, key errors when only the other is present)
        # (GH#39133).
        # GH#39133
        mask = np.array([False, False, True, False])
        if using_nan_is_na:
            mask[-1] = True

        idx = Index(FloatingArray(np.array([1, 2, 1, np.nan]), mask=mask))
        if using_nan_is_na:
            # NaN and NA are consistently treated as the same
            result = idx.get_loc(NA)
            expected = np.array([False, False, True, True])
            tm.assert_numpy_array_equal(result, expected)
            result = idx.get_loc(np.nan)
            tm.assert_numpy_array_equal(result, expected)
        else:
            result = idx.get_loc(NA)
            assert result == 2
            result = idx.get_loc(np.nan)
            assert result == 3

        idx = Index(
            FloatingArray(np.array([1, 2, 1.0]), mask=np.array([False, False, True]))
        )
        result = idx.get_loc(NA)
        assert result == 2
        if using_nan_is_na:
            result = idx.get_loc(np.nan)
            assert result == 2
        else:
            with pytest.raises(KeyError, match="nan"):
                idx.get_loc(np.nan)

        mask = np.array([False, False, False])
        if using_nan_is_na:
            mask[-1] = True
        idx = Index(FloatingArray(np.array([1, 2, np.nan]), mask=mask))
        result = idx.get_loc(np.nan)
        assert result == 2
        if using_nan_is_na:
            result = idx.get_loc(NA)
            assert result == 2
        else:
            with pytest.raises(KeyError, match="NA"):
                idx.get_loc(NA)

    @pytest.mark.parametrize("val", [4, 2])
    def test_get_indexer_masked_na(self, any_numeric_ea_and_arrow_dtype, val):
        # [FUNCTION DOC]
        # Parametrized over a trailing value. get_indexer_for finds masked NA position
        # among other labels (GH#39133).
        # GH#39133
        idx = Index([1, 2, NA, 3, val], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.get_indexer_for([1, NA, 5])
        expected = np.array([0, 2, -1])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("dtype", ["boolean", "bool[pyarrow]"])
    def test_get_indexer_masked_na_boolean(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over masked-boolean and pyarrow boolean (skipped if pyarrow is
        # unavailable). get_loc finds False and NA (GH#39133).
        # KNOWN ISSUE: named as an indexer test, but it calls the location lookup method.
        # GH#39133
        if dtype == "bool[pyarrow]":
            pytest.importorskip("pyarrow")
        idx = Index([True, False, NA], dtype=dtype)
        result = idx.get_loc(False)
        assert result == 1
        result = idx.get_loc(NA)
        assert result == 2

    def test_get_indexer_arrow_dictionary_target(self):
        # [FUNCTION DOC]
        # Skipped without pyarrow. A dictionary-encoded arrow target is matched by value
        # in both get_indexer and get_indexer_non_unique, with the missing position
        # reported by the latter.
        pa = pytest.importorskip("pyarrow")
        target = Index(
            ArrowExtensionArray(
                pa.array([1, 2], type=pa.dictionary(pa.int8(), pa.int8()))
            )
        )
        idx = Index([1])

        result = idx.get_indexer(target)
        expected = np.array([0, -1], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        result_1, result_2 = idx.get_indexer_non_unique(target)
        expected_1, expected_2 = (
            np.array([0, -1], dtype=np.int64),
            np.array([1], dtype=np.int64),
        )
        tm.assert_numpy_array_equal(result_1, expected_1)
        tm.assert_numpy_array_equal(result_2, expected_2)


class TestWhere:
    # [CLASS DOC]
    # Groups where tests for numeric indexes.
    @pytest.mark.parametrize(
        "index",
        [
            Index(np.arange(5, dtype="float64")),
            Index(range(0, 20, 2), dtype=np.int64),
            Index(np.arange(5, dtype="uint64")),
        ],
    )
    def test_where(self, listlike_box, index):
        # [FUNCTION DOC]
        # Parametrized over float, signed and unsigned indexes and the list-like
        # fixture. A condition with one false entry yields a float index with a missing
        # value there.
        # KNOWN ISSUE: the result for the all-true condition is computed but never asserted.
        cond = [True] * len(index)
        expected = index
        result = index.where(listlike_box(cond))

        cond = [False] + [True] * (len(index) - 1)
        expected = Index([index._na_value, *index[1:].tolist()], dtype=np.float64)
        result = index.where(listlike_box(cond))
        tm.assert_index_equal(result, expected)

    def test_where_uint64(self):
        # [FUNCTION DOC]
        # where and putmask on an unsigned index with a signed replacement keep the
        # unsigned dtype.
        idx = Index([0, 6, 2], dtype=np.uint64)
        mask = np.array([False, True, False])
        other = np.array([1], dtype=np.int64)

        expected = Index([1, 6, 1], dtype=np.uint64)

        result = idx.where(mask, other)
        tm.assert_index_equal(result, expected)

        result = idx.putmask(~mask, other)
        tm.assert_index_equal(result, expected)

    def test_where_infers_type_instead_of_trying_to_convert_string_to_float(self):
        # [FUNCTION DOC]
        # where with string replacement values falls back to a mixed object index rather
        # than failing (GH 32413).
        # GH 32413
        index = Index([1, np.nan])
        cond = index.notna()
        other = Index(["a", "b"], dtype="string")

        expected = Index([1.0, "b"])
        result = index.where(cond, other)

        tm.assert_index_equal(result, expected)


class TestTake:
    # [CLASS DOC]
    # Groups take tests for numeric indexes.
    @pytest.mark.parametrize("idx_dtype", [np.float64, np.int64, np.uint64])
    def test_take_preserve_name(self, idx_dtype):
        # [FUNCTION DOC]
        # Parametrized over dtypes; take keeps the name.
        index = Index([1, 2, 3, 4], dtype=idx_dtype, name="foo")
        taken = index.take([3, 0, 1])
        assert index.name == taken.name

    def test_take_fill_value_float64(self):
        # [FUNCTION DOC]
        # take with a negative position wraps, with fill enabled uses NaN, with fill
        # disabled wraps; positions below -1 with fill raise value errors; out-of-bounds
        # raises an index error (GH 12631).
        # GH 12631
        idx = Index([1.0, 2.0, 3.0], name="xxx", dtype=np.float64)
        result = idx.take(np.array([1, 0, -1]))
        expected = Index([2.0, 1.0, 3.0], dtype=np.float64, name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = Index([2.0, 1.0, np.nan], dtype=np.float64, name="xxx")
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index([2.0, 1.0, 3.0], dtype=np.float64, name="xxx")
        tm.assert_index_equal(result, expected)

        msg = (
            "When allow_fill=True and fill_value is not None, all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_take_fill_value_ints(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over signed and unsigned. Wrapping works; fill with value raises
        # a value error that integer indexes cannot hold NA; fill disabled wraps
        # (gh-12631).
        # see gh-12631
        idx = Index([1, 2, 3], dtype=dtype, name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = Index([2, 1, 3], dtype=dtype, name="xxx")
        tm.assert_index_equal(result, expected)

        name = type(idx).__name__
        msg = f"Unable to fill values because {name} cannot contain NA"

        # fill_value=True
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -1]), fill_value=True)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index([2, 1, 3], dtype=dtype, name="xxx")
        tm.assert_index_equal(result, expected)

        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))


class TestContains:
    # [CLASS DOC]
    # Groups membership tests for numeric indexes.
    @pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64])
    def test_contains_none(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtypes. none is not contained and no error is raised
        # (GH#35788).
        # GH#35788 should return False, not raise TypeError
        index = Index([0, 1, 2, 3, 4], dtype=dtype)
        assert None not in index

    def test_contains_float64_nans(self):
        # [FUNCTION DOC]
        # NaN is contained in a float index holding NaN.
        index = Index([1.0, 2.0, np.nan], dtype=np.float64)
        assert np.nan in index

    def test_contains_float64_not_nans(self):
        # [FUNCTION DOC]
        # An ordinary float is contained in an index that also holds NaN.
        index = Index([1.0, 2.0, np.nan], dtype=np.float64)
        assert 1.0 in index


class TestSliceLocs:
    # [CLASS DOC]
    # Groups slice_locs tests for numeric indexes.
    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over int and float. Start/end positions for several bounds,
        # including reversed index.
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
        n = len(index)

        assert index.slice_locs(start=2) == (2, n)
        assert index.slice_locs(start=3) == (3, n)
        assert index.slice_locs(3, 8) == (3, 6)
        assert index.slice_locs(5, 10) == (3, n)
        assert index.slice_locs(end=8) == (0, 6)
        assert index.slice_locs(end=9) == (0, 7)

        # reversed
        index2 = index[::-1]
        assert index2.slice_locs(8, 2) == (2, 6)
        assert index2.slice_locs(7, 3) == (2, 5)

    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs_float_locs(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over int and float. Fractional bounds, also reversed.
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
        n = len(index)
        assert index.slice_locs(5.0, 10.0) == (3, n)
        assert index.slice_locs(4.5, 10.5) == (3, 8)

        index2 = index[::-1]
        assert index2.slice_locs(8.5, 1.5) == (2, 6)
        assert index2.slice_locs(10.5, -1) == (0, n)

    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs_dup_numeric(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over int and float. Duplicated labels yield spans covering the
        # duplicates, also reversed.
        index = Index(np.array([10, 12, 12, 14], dtype=dtype))
        assert index.slice_locs(12, 12) == (1, 3)
        assert index.slice_locs(11, 13) == (1, 3)

        index2 = index[::-1]
        assert index2.slice_locs(12, 12) == (1, 3)
        assert index2.slice_locs(13, 11) == (1, 3)

    def test_slice_locs_na(self):
        # [FUNCTION DOC]
        # Bounds with NaN work when NaN values are first or in the middle of the index.
        index = Index([np.nan, 1, 2])
        assert index.slice_locs(1) == (1, 3)
        assert index.slice_locs(np.nan) == (0, 3)

        index = Index([0, np.nan, np.nan, 1, 2])
        assert index.slice_locs(np.nan) == (1, 5)

    def test_slice_locs_na_raises(self):
        # [FUNCTION DOC]
        # Missing label bounds on an index with NaN raise key errors about a
        # non-monotonic index with a missing label.
        index = Index([np.nan, 1, 2])
        msg = "non-monotonic index with a missing label 1.5"
        with pytest.raises(KeyError, match=msg):
            index.slice_locs(start=1.5)

        with pytest.raises(KeyError, match=msg):
            index.slice_locs(end=1.5)


class TestGetSliceBounds:
    # [CLASS DOC]
    # Groups get_slice_bound tests for numeric indexes.
    @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])
    def test_get_slice_bounds_within(self, side, expected):
        # [FUNCTION DOC]
        # Parametrized over side. A bound inside the range returns the left or right
        # insertion point.
        index = Index(range(6))
        result = index.get_slice_bound(4, side=side)
        assert result == expected

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("bound, expected", [(-1, 0), (10, 6)])
    def test_get_slice_bounds_outside(self, side, expected, bound):
        # [FUNCTION DOC]
        # Parametrized over side and out-of-range bounds. Bounds below return zero,
        # above return the length.
        index = Index(range(6))
        result = index.get_slice_bound(bound, side=side)
        assert result == expected
