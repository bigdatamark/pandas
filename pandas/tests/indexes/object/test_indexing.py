# [MODULE DOC]
# Tests for indexing methods of object-dtype Index: get_indexer (fill methods,
# missing values) and get_indexer_non_unique with null markers and numpy NaT values.
from decimal import Decimal

import numpy as np
import pytest

from pandas._libs.missing import is_matching_na

from pandas import Index
import pandas._testing as tm


class TestGetIndexer:
    # [CLASS DOC]
    # Groups get_indexer tests for object indexes.
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("pad", [-1, 0, 1, 1]),
            ("backfill", [0, 0, 1, -1]),
        ],
    )
    def test_get_indexer_strings(self, method, expected):
        # [FUNCTION DOC]
        # Parametrized over pad and backfill. A sorted string index gets the expected
        # positions for a target of strings, with -1 for those without a fill source.
        expected = np.array(expected, dtype=np.intp)
        index = Index(["b", "c"], dtype=object)
        actual = index.get_indexer(["a", "b", "c", "d"], method=method)

        tm.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_strings_raises(self):
        # [FUNCTION DOC]
        # The nearest method, or a tolerance, on a string index raises a type error
        # since distance cannot be computed on strings (message depends on string
        # backend).
        index = Index(["b", "c"], dtype=object)

        msg = "|".join(
            [
                "operation 'sub' not supported for dtype 'str'",
                r"unsupported operand type\(s\) for -: 'str' and 'str'",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            index.get_indexer(["a", "b", "c", "d"], method="nearest")

        with pytest.raises(TypeError, match=msg):
            index.get_indexer(["a", "b", "c", "d"], method="pad", tolerance=2)

        with pytest.raises(TypeError, match=msg):
            index.get_indexer(
                ["a", "b", "c", "d"], method="pad", tolerance=[2, 2, 2, 2]
            )

    def test_get_indexer_with_NA_values(
        self, unique_nulls_fixture, unique_nulls_fixture2
    ):
        # [FUNCTION DOC]
        # Parametrized over pairs of distinct null values from the unique-nulls
        # fixtures. Each null is matched at its own position and an unknown label gives
        # -1 (GH#22332); identical pairs are skipped by returning early.
        # GH#22332
        # check pairwise, that no pair of na values
        # is mangled
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # skip it, values are not unique
        arr = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
        index = Index(arr, dtype=object)
        result = index.get_indexer(
            Index(
                [unique_nulls_fixture, unique_nulls_fixture2, "Unknown"], dtype=object
            )
        )
        expected = np.array([0, 1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_infer_string_missing_values(self):
        # [FUNCTION DOC]
        # With object dtype, a target list containing None is kept as object so None is
        # matched in the index (GitHub 55834).
        # ensure the passed list is not cast to string but to object so that
        # the None value is matched in the index
        # https://github.com/pandas-dev/pandas/issues/55834
        idx = Index(["a", "b", None], dtype="object")
        result = idx.get_indexer([None, "x"])
        expected = np.array([2, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)


class TestGetIndexerNonUnique:
    # [CLASS DOC]
    # Groups get_indexer_non_unique tests for object indexes.
    def test_get_indexer_non_unique_nas(self, nulls_fixture):
        # [FUNCTION DOC]
        # Parametrized over the nulls fixture. Null lookup returns the single position
        # when the index has one null, all positions when there are several, and also
        # matches equal-but-not-identical NaN objects; no missing entries.
        # even though this isn't non-unique, this should still work
        index = Index(["a", "b", nulls_fixture], dtype=object)
        indexer, missing = index.get_indexer_non_unique([nulls_fixture])

        expected_indexer = np.array([2], dtype=np.intp)
        expected_missing = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)

        # actually non-unique
        index = Index(["a", nulls_fixture, "b", nulls_fixture], dtype=object)
        indexer, missing = index.get_indexer_non_unique([nulls_fixture])

        expected_indexer = np.array([1, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)

        # matching-but-not-identical nans
        if is_matching_na(nulls_fixture, float("NaN")):
            index = Index(["a", float("NaN"), "b", float("NaN")], dtype=object)
            match_but_not_identical = True
        elif is_matching_na(nulls_fixture, Decimal("NaN")):
            index = Index(["a", Decimal("NaN"), "b", Decimal("NaN")], dtype=object)
            match_but_not_identical = True
        else:
            match_but_not_identical = False

        if match_but_not_identical:
            indexer, missing = index.get_indexer_non_unique([nulls_fixture])

            expected_indexer = np.array([1, 3], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)

    @pytest.mark.filterwarnings("ignore:elementwise comp:DeprecationWarning")
    def test_get_indexer_non_unique_np_nats(self, np_nat_fixture, np_nat_fixture2):
        # [FUNCTION DOC]
        # Parametrized over pairs of numpy datetime/timedelta NaT fixtures. Matching NaT
        # kinds are all found; mismatching kinds only match their own kind; uncomparable
        # unit pairs return early. Ignores an element-wise comparison deprecation
        # warning.
        expected_missing = np.array([], dtype=np.intp)
        # matching-but-not-identical nats
        if is_matching_na(np_nat_fixture, np_nat_fixture2):
            # ensure nats are different objects
            index = Index(
                np.array(
                    ["2021-10-02", np_nat_fixture.copy(), np_nat_fixture2.copy()],
                    dtype=object,
                ),
                dtype=object,
            )
            # pass as index to prevent target from being casted to DatetimeIndex
            indexer, missing = index.get_indexer_non_unique(
                Index([np_nat_fixture], dtype=object)
            )
            expected_indexer = np.array([1, 2], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)
        # dt64nat vs td64nat
        else:
            try:
                np_nat_fixture == np_nat_fixture2
            except (TypeError, OverflowError):
                # Numpy will raise on uncomparable types, like
                # np.datetime64('NaT', 'Y') and np.datetime64('NaT', 'ps')
                # https://github.com/numpy/numpy/issues/22762
                return
            index = Index(
                np.array(
                    [
                        "2021-10-02",
                        np_nat_fixture,
                        np_nat_fixture2,
                        np_nat_fixture,
                        np_nat_fixture2,
                    ],
                    dtype=object,
                ),
                dtype=object,
            )
            # pass as index to prevent target from being casted to DatetimeIndex
            indexer, missing = index.get_indexer_non_unique(
                Index([np_nat_fixture], dtype=object)
            )
            expected_indexer = np.array([1, 3], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)
