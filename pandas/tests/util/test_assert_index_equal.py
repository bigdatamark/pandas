# [MODULE DOC]
# Tests for the testing-utility function that asserts two indexes are equal: level count, value, length, class (including range index equivalence), inferred type,
# approximate versus exact numeric comparison with tolerance, ordering control, names, categorical handling, missing values in extension indexes and incomparable categories.
# Uses the exact-comparison, tolerance and categorical-check fixtures from the local configuration.

import numpy as np
import pytest

from pandas import (
    NA,
    Categorical,
    CategoricalIndex,
    Index,
    MultiIndex,
    NaT,
    RangeIndex,
)
import pandas._testing as tm


def test_index_equal_levels_mismatch():
    # [FUNCTION DOC]
    # A flat index and a two-level multi-index compared with the inexact class check raise an assertion error saying the number of levels differs and showing both representations.
    msg = """Index are different

Index levels are different
\\[left\\]:  1, Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: 2, MultiIndex\\(\\[\\('A', 1\\),
            \\('A', 2\\),
            \\('B', 3\\),
            \\('B', 4\\)\\],
           \\)"""

    idx1 = Index([1, 2, 3])
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=False)


def test_index_equal_values_mismatch(check_exact):
    # [FUNCTION DOC]
    # Uses the exact-comparison fixture. Multi-indexes differing in one value of the second level raise an assertion error naming level 1 with 25 percent differing and both level listings.
    msg = """MultiIndex level \\[1\\] are different

MultiIndex level \\[1\\] values are different \\(25\\.0 %\\)
\\[left\\]:  Index\\(\\[2, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"""

    idx1 = MultiIndex.from_tuples([("A", 2), ("A", 2), ("B", 3), ("B", 4)])
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)


def test_index_equal_length_mismatch(check_exact):
    # [FUNCTION DOC]
    # Uses the exact-comparison fixture. Indexes of lengths 3 and 4 raise an assertion error saying the lengths differ with both representations.
    msg = """Index are different

Index length are different
\\[left\\]:  3, Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: 4, Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"""

    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3, 4])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)


@pytest.mark.parametrize("exact", [False, "equiv"])
def test_index_equal_class(exact):
    # [FUNCTION DOC]
    # Parametrized over exact being false or "equiv". A plain integer index and a range index of the same values compare equal.
    idx1 = Index([0, 1, 2])
    idx2 = RangeIndex(3)

    tm.assert_index_equal(idx1, idx2, exact=exact)


def test_int_float_index_equal_class_mismatch(check_exact):
    # [FUNCTION DOC]
    # Uses the exact-comparison fixture. With exact class checking, integer versus float indexes raise an assertion error saying the inferred type attribute differs (integer versus floating).
    msg = """Index are different

Attribute "inferred_type" are different
\\[left\\]:  integer
\\[right\\]: floating"""

    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3], dtype=np.float64)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)


def test_range_index_equal_class_mismatch(check_exact):
    # [FUNCTION DOC]
    # Uses the exact-comparison fixture. A plain integer index versus a range index with exact class checking raises an assertion error saying the classes differ.
    msg = """Index are different

Index classes are different
\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: """

    idx1 = Index([1, 2, 3])
    idx2 = RangeIndex(range(3))

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)


def test_index_equal_values_close(check_exact):
    # [FUNCTION DOC]
    # Uses the exact-comparison fixture. Float indexes differing by 1e-10: with exact comparison an assertion error shows 33.33 percent differing; otherwise they are considered equal.
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0000000001])

    if check_exact:
        msg = """Index are different

Index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)
\\[right\\]: Index\\(\\[1.0, 2.0, 3.0000000001\\], dtype='float64'\\)"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_exact=check_exact)
    else:
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)


def test_index_equal_values_less_close(check_exact, rtol):
    # [FUNCTION DOC]
    # Uses the exact-comparison and tolerance fixtures. Float indexes differing by 1e-4 in the last value: an assertion error appears when exact comparison is on or the relative tolerance is smaller than 0.5e-3; otherwise equal.
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0001])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    if check_exact or rtol < 0.5e-3:
        msg = """Index are different

Index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)
\\[right\\]: Index\\(\\[1.0, 2.0, 3.0001\\], dtype='float64'\\)"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, **kwargs)
    else:
        tm.assert_index_equal(idx1, idx2, **kwargs)


def test_index_equal_values_too_far(check_exact, rtol):
    # [FUNCTION DOC]
    # Uses the exact-comparison and tolerance fixtures. Integer indexes differing by one in the last value always raise an assertion error showing 33.33 percent differing.
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 4])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    msg = """Index are different

Index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 4\\], dtype='int64'\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)


@pytest.mark.parametrize("check_order", [True, False])
def test_index_equal_value_order_mismatch(check_exact, rtol, check_order):
    # [FUNCTION DOC]
    # Parametrized over the order check flag (plus the exact and tolerance fixtures). Reversed integer indexes raise an assertion error (66.67 percent differing) when the order check is on, and compare equal when off.
    idx1 = Index([1, 2, 3])
    idx2 = Index([3, 2, 1])

    msg = """Index are different

Index values are different \\(66\\.66667 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[3, 2, 1\\], dtype='int64'\\)"""

    if check_order:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(
                idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=True
            )
    else:
        tm.assert_index_equal(
            idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=False
        )


def test_index_equal_level_values_mismatch(check_exact, rtol):
    # [FUNCTION DOC]
    # Uses the exact-comparison and tolerance fixtures. Multi-indexes with a differing second-level value raise an assertion error naming level 1 with 25 percent differing.
    idx1 = MultiIndex.from_tuples([("A", 2), ("A", 2), ("B", 3), ("B", 4)])
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    msg = """MultiIndex level \\[1\\] are different

MultiIndex level \\[1\\] values are different \\(25\\.0 %\\)
\\[left\\]:  Index\\(\\[2, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)


@pytest.mark.parametrize(
    "name1,name2",
    [(None, "x"), ("x", "x"), (np.nan, np.nan), (NaT, NaT), (np.nan, NaT)],
)
def test_index_equal_names(name1, name2):
    # [FUNCTION DOC]
    # Parametrized over name pairs (none and a string, equal strings, NaN pair, NaT pair, NaN versus NaT). Equal or identical names pass; otherwise an assertion error says the names attribute differs, showing both names (strings in quotes).
    idx1 = Index([1, 2, 3], name=name1)
    idx2 = Index([1, 2, 3], name=name2)

    if name1 == name2 or name1 is name2:
        tm.assert_index_equal(idx1, idx2)
    else:
        name1 = "'x'" if name1 == "x" else name1
        name2 = "'x'" if name2 == "x" else name2
        msg = f"""Index are different

Attribute "names" are different
\\[left\\]:  \\[{name1}\\]
\\[right\\]: \\[{name2}\\]"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2)


def test_index_equal_category_mismatch(check_categorical, using_infer_string):
    # [FUNCTION DOC]
    # Uses the categorical-check fixture and infer-string fixture (expected category dtype name). Categorical indexes with different category sets raise an assertion error showing the two categorical dtypes when categorical checking is on, and pass when off.
    if using_infer_string:
        dtype = "str"
    else:
        dtype = "object"
    msg = f"""Index are different

Attribute "dtype" are different
\\[left\\]:  CategoricalDtype\\(categories=\\['a', 'b'\\], ordered=False, \
categories_dtype={dtype}\\)
\\[right\\]: CategoricalDtype\\(categories=\\['a', 'b', 'c'\\], \
ordered=False, categories_dtype={dtype}\\)"""

    idx1 = Index(Categorical(["a", "b"]))
    idx2 = Index(Categorical(["a", "b"], categories=["a", "b", "c"]))

    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
    else:
        tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)


@pytest.mark.parametrize("exact", [False, True])
def test_index_equal_range_categories(check_categorical, exact):
    # [FUNCTION DOC]
    # GH41263. Parametrized over exact flag, plus the categorical-check fixture. Categorical indexes built from a range versus a list of integers raise an assertion error about differing classes only when both categorical checking and exact checking are on; otherwise pass.
    # GH41263
    msg = """\
Index are different

Index classes are different
\\[left\\]:  RangeIndex\\(start=0, stop=10, step=1\\)
\\[right\\]: Index\\(\\[0, 1, 2, 3, 4, 5, 6, 7, 8, 9\\], dtype='int64'\\)"""

    rcat = CategoricalIndex(RangeIndex(10))
    icat = CategoricalIndex(list(range(10)))

    if check_categorical and exact:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(rcat, icat, check_categorical=True, exact=True)
    else:
        tm.assert_index_equal(
            rcat, icat, check_categorical=check_categorical, exact=exact
        )


def test_assert_index_equal_different_inferred_types():
    # [FUNCTION DOC]
    # GH#31884. An index of NA with a NumPy datetime NaT versus NA with the pandas NaT raises an assertion error saying the inferred type differs (mixed versus datetime).
    # GH#31884
    msg = """\
Index are different

Attribute "inferred_type" are different
\\[left\\]:  mixed
\\[right\\]: datetime"""

    idx1 = Index([NA, np.datetime64("nat", "ns")])
    idx2 = Index([NA, NaT])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2)


def test_assert_index_equal_different_names_check_order_false():
    # [FUNCTION DOC]
    # GH#47328. Indexes with the same values in different order but different names raise an assertion error that names differ even when the order check is off and name checking is on.
    # GH#47328
    idx1 = Index([1, 3], name="a")
    idx2 = Index([3, 1], name="b")
    with pytest.raises(AssertionError, match='"names" are different'):
        tm.assert_index_equal(idx1, idx2, check_order=False, check_names=True)


def test_assert_index_equal_mixed_dtype():
    # [FUNCTION DOC]
    # GH#39168. An index with mixed-type labels equals itself with the order check off.
    # GH#39168
    idx = Index(["foo", "bar", 42])
    tm.assert_index_equal(idx, idx, check_order=False)


def test_assert_index_equal_ea_dtype_order_false(any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # GH#47207. Parametrized over numeric extension dtypes; reversed-order indexes compare equal with the order check off.
    # GH#47207
    idx1 = Index([1, 3], dtype=any_numeric_ea_dtype)
    idx2 = Index([3, 1], dtype=any_numeric_ea_dtype)
    tm.assert_index_equal(idx1, idx2, check_order=False)


def test_assert_index_equal_object_ints_order_false():
    # [FUNCTION DOC]
    # GH#47207. Reversed-order object-dtype integer indexes compare equal with the order check off.
    # GH#47207
    idx1 = Index([1, 3], dtype="object")
    idx2 = Index([3, 1], dtype="object")
    tm.assert_index_equal(idx1, idx2, check_order=False)


@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize("check_names", [True, False])
def test_assert_ea_index_equal_non_matching_na(check_names, check_categorical):
    # [FUNCTION DOC]
    # GH#48608. Parametrized over name checking and categorical checking. A nullable integer index versus one with a missing value raises an assertion error showing 50 percent differing.
    # GH#48608
    idx1 = Index([1, 2], dtype="Int64")
    idx2 = Index([1, NA], dtype="Int64")
    with pytest.raises(AssertionError, match="50.0 %"):
        tm.assert_index_equal(
            idx1, idx2, check_names=check_names, check_categorical=check_categorical
        )


@pytest.mark.parametrize("check_categorical", [True, False])
def test_assert_multi_index_dtype_check_categorical(check_categorical):
    # [FUNCTION DOC]
    # GH#52126. Parametrized over categorical checking. Multi-indexes with unsigned versus signed integer categorical levels raise an assertion error naming level 0 when categorical checking is on, and pass when off.
    # GH#52126
    idx1 = MultiIndex.from_arrays([Categorical(np.array([1, 2], dtype=np.uint64))])
    idx2 = MultiIndex.from_arrays([Categorical(np.array([1, 2], dtype=np.int64))])
    if check_categorical:
        with pytest.raises(
            AssertionError, match=r"^MultiIndex level \[0\] are different"
        ):
            tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
    else:
        tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)


def test_assert_index_equal_categorical_incomparable_categories():
    # [FUNCTION DOC]
    # GH#61935. Categorical indexes whose categories differ in value raise an assertion error saying the types are not comparable (with categorical checking on and the inexact class check).
    # GH#61935
    left = Index([1, 2, 3], name="a", dtype="category")
    right = Index([1, 2, 6], name="a", dtype="category")
    with pytest.raises(AssertionError, match="types are not comparable"):
        tm.assert_index_equal(left, right, check_categorical=True, exact=False)
