# [MODULE DOC]
# Tests for MultiIndex construction: the class constructor (levels and codes),
# from_arrays, from_tuples, from_product and from_frame, with argument validation,
# empty inputs, names inference, datetime-like/categorical/extension inputs, sort order
# validation, and dtype inference consistency. Sections are separated by banner comments.
# Uses the shared idx fixture plus using_infer_string. Pure tests module.

from datetime import (
    date,
    datetime,
)
import itertools

import numpy as np
import pytest

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


def test_constructor_single_level():
    # [FUNCTION DOC]
    # Verifies a single-level constructor call gives a multi-index whose level equals an
    # index with the given name and whose names list has that name.
    result = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"]], codes=[[0, 1, 2, 3]], names=["first"]
    )
    assert isinstance(result, MultiIndex)
    expected = Index(["foo", "bar", "baz", "qux"], name="first")
    tm.assert_index_equal(result.levels[0], expected)
    assert result.names == ["first"]


def test_constructor_no_levels():
    # [FUNCTION DOC]
    # Verifies constructing with empty levels and codes raises a value error requiring a
    # non-zero number of levels, and giving only levels or only codes raises a type error
    # that both must be passed.
    msg = "non-zero number of levels/codes"
    with pytest.raises(ValueError, match=msg):
        MultiIndex(levels=[], codes=[])

    msg = "Must pass both levels and codes"
    with pytest.raises(TypeError, match=msg):
        MultiIndex(levels=[])
    with pytest.raises(TypeError, match=msg):
        MultiIndex(codes=[])


def test_constructor_nonhashable_names():
    # [FUNCTION DOC]
    # Verifies (GH 20527) unhashable names (lists) raise a type error through the
    # constructor, rename and set_names.
    # GH 20527
    levels = [[1, 2], ["one", "two"]]
    codes = [[0, 0, 1, 1], [0, 1, 0, 1]]
    names = (["foo"], ["bar"])
    msg = r"MultiIndex\.name must be a hashable type"
    with pytest.raises(TypeError, match=msg):
        MultiIndex(levels=levels, codes=codes, names=names)

    # With .rename()
    mi = MultiIndex(
        levels=[[1, 2], ["one", "two"]],
        codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        names=("foo", "bar"),
    )
    renamed = [["fooo"], ["barr"]]
    with pytest.raises(TypeError, match=msg):
        mi.rename(names=renamed)

    # With .set_names()
    with pytest.raises(TypeError, match=msg):
        mi.set_names(names=renamed)


def test_constructor_mismatched_codes_levels(idx):
    # [FUNCTION DOC]
    # Verifies validation errors: levels and codes of different length; a code beyond the level
    # size (message notes inconsistent state); unequal code lengths; and a code below minus one.
    # Checks the same validation through set_levels and set_codes, and that disabling integrity
    # verification when setting codes does not raise.
    codes = [np.array([1]), np.array([2]), np.array([3])]
    levels = ["a"]

    msg = "Length of levels and codes must be the same"
    with pytest.raises(ValueError, match=msg):
        MultiIndex(levels=levels, codes=codes)

    length_error = (
        r"On level 0, code max \(3\) >= length of level \(1\)\. "
        "NOTE: this index is in an inconsistent state"
    )
    label_error = r"Unequal code lengths: \[4, 2\]"
    code_value_error = r"On level 0, code value \(-2\) < -1"

    # important to check that it's looking at the right thing.
    with pytest.raises(ValueError, match=length_error):
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, 1, 2, 3], [0, 3, 4, 1]])

    with pytest.raises(ValueError, match=label_error):
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, 0, 0, 0], [0, 0]])

    # external API
    with pytest.raises(ValueError, match=length_error):
        idx.copy().set_levels([["a"], ["b"]])

    with pytest.raises(ValueError, match=label_error):
        idx.copy().set_codes([[0, 0, 0, 0], [0, 0]])

    # test set_codes with verify_integrity=False
    # the setting should not raise any value error
    idx.copy().set_codes(codes=[[0, 0, 0, 0], [0, 0]], verify_integrity=False)

    # code value smaller than -1
    with pytest.raises(ValueError, match=code_value_error):
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, -2], [0, 0]])


def test_na_levels():
    # [FUNCTION DOC]
    # Verifies (GH26408) that codes pointing at missing-value labels in the levels are
    # normalized to the missing code, in the constructor and in set_levels/set_codes.
    # GH26408
    # test if codes are re-assigned value -1 for levels
    # with missing values (NaN, NaT, None)
    result = MultiIndex(
        levels=[[np.nan, None, pd.NaT, 128, 2]], codes=[[0, -1, 1, 2, 3, 4]]
    )
    expected = MultiIndex(
        levels=[[np.nan, None, pd.NaT, 128, 2]], codes=[[-1, -1, -1, -1, 3, 4]]
    )
    tm.assert_index_equal(result, expected)

    result = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[0, -1, 1, 2, 3, 4]]
    )
    expected = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[-1, -1, 1, -1, 3, -1]]
    )
    tm.assert_index_equal(result, expected)

    # verify set_levels and set_codes
    result = MultiIndex(
        levels=[[1, 2, 3, 4, 5]], codes=[[0, -1, 1, 2, 3, 4]]
    ).set_levels([[np.nan, "s", pd.NaT, 128, None]])
    tm.assert_index_equal(result, expected)

    result = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[1, 2, 2, 2, 2, 2]]
    ).set_codes([[0, -1, 1, 2, 3, 4]])
    tm.assert_index_equal(result, expected)


def test_copy_in_constructor():
    # [FUNCTION DOC]
    # Verifies the copy option of the constructor: modifying the input code or level arrays
    # after construction does not change the index.
    levels = np.array(["a", "b", "c"])
    codes = np.array([1, 1, 2, 0, 0, 1, 1])
    val = codes[0]
    mi = MultiIndex(levels=[levels, levels], codes=[codes, codes], copy=True)
    assert mi.codes[0][0] == val
    codes[0] = 15
    assert mi.codes[0][0] == val
    val = levels[0]
    levels[0] = "PANDA"
    assert mi.levels[0][0] == val


# ----------------------------------------------------------------------------
# from_arrays
# ----------------------------------------------------------------------------
def test_from_arrays(idx):
    # [FUNCTION DOC]
    # Verifies from_arrays rebuilds the shared index from per-level value arrays with names,
    # and infers levels while dropping a missing time from a datetime level.
    arrays = [
        np.asarray(lev).take(level_codes)
        for lev, level_codes in zip(idx.levels, idx.codes, strict=True)
    ]

    # list of arrays as input
    result = MultiIndex.from_arrays(arrays, names=idx.names)
    tm.assert_index_equal(result, idx)

    # infer correctly
    result = MultiIndex.from_arrays([[pd.NaT, Timestamp("20130101")], ["a", "b"]])
    assert result.levels[0].equals(Index([Timestamp("20130101")]))
    assert result.levels[1].equals(Index(["a", "b"]))


def test_from_arrays_iterator(idx):
    # [FUNCTION DOC]
    # Verifies (GH 18434) from_arrays accepts an iterator of arrays and raises a type error
    # for a non-sequence input.
    # GH 18434
    arrays = [
        np.asarray(lev).take(level_codes)
        for lev, level_codes in zip(idx.levels, idx.codes, strict=True)
    ]

    # iterator as input
    result = MultiIndex.from_arrays(iter(arrays), names=idx.names)
    tm.assert_index_equal(result, idx)

    # invalid iterator input
    msg = "Input must be a list / sequence of array-likes."
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_arrays(0)


def test_from_arrays_tuples(idx):
    # [FUNCTION DOC]
    # Verifies from_arrays accepts a tuple of tuples.
    arrays = tuple(
        tuple(np.asarray(lev).take(level_codes))
        for lev, level_codes in zip(idx.levels, idx.codes, strict=True)
    )

    # tuple of tuples as input
    result = MultiIndex.from_arrays(arrays, names=idx.names)
    tm.assert_index_equal(result, idx)


@pytest.mark.parametrize(
    ("idx1", "idx2"),
    [
        (
            pd.period_range("2011-01-01", freq="D", periods=3),
            pd.period_range("2015-01-01", freq="h", periods=3),
        ),
        (
            date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern"),
            date_range("2015-01-01 10:00", freq="h", periods=3, tz="Asia/Tokyo"),
        ),
        (
            pd.timedelta_range("1 days", freq="D", periods=3),
            pd.timedelta_range("2 hours", freq="h", periods=3),
        ),
    ],
)
def test_from_arrays_index_series_period_datetimetz_and_timedelta(idx1, idx2):
    # [FUNCTION DOC]
    # Parametrized over pairs of period, aware datetime, or timedelta indexes. Verifies
    # from_arrays preserves each level's values for index or Series inputs and both give
    # equal results.
    result = MultiIndex.from_arrays([idx1, idx2])
    tm.assert_index_equal(result.get_level_values(0), idx1)
    tm.assert_index_equal(result.get_level_values(1), idx2)

    result2 = MultiIndex.from_arrays([Series(idx1), Series(idx2)])
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    tm.assert_index_equal(result2.get_level_values(1), idx2)

    tm.assert_index_equal(result, result2)


def test_from_arrays_index_datetimelike_mixed():
    # [FUNCTION DOC]
    # Verifies from_arrays with mixed datetime-like levels (aware, naive, timedelta, period)
    # preserves the level values, from indexes and Series alike.
    idx1 = date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern")
    idx2 = date_range("2015-01-01 10:00", freq="h", periods=3)
    idx3 = pd.timedelta_range("1 days", freq="D", periods=3)
    idx4 = pd.period_range("2011-01-01", freq="D", periods=3)

    result = MultiIndex.from_arrays([idx1, idx2, idx3, idx4])
    tm.assert_index_equal(result.get_level_values(0), idx1)
    tm.assert_index_equal(result.get_level_values(1), idx2)
    tm.assert_index_equal(result.get_level_values(2), idx3)
    tm.assert_index_equal(result.get_level_values(3), idx4)

    result2 = MultiIndex.from_arrays(
        [Series(idx1), Series(idx2), Series(idx3), Series(idx4)]
    )
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    tm.assert_index_equal(result2.get_level_values(1), idx2)
    tm.assert_index_equal(result2.get_level_values(2), idx3)
    tm.assert_index_equal(result2.get_level_values(3), idx4)

    tm.assert_index_equal(result, result2)


def test_from_arrays_index_series_categorical():
    # [FUNCTION DOC]
    # Verifies (GH13743) from_arrays with categorical indexes (ordered and unordered) preserves
    # categories and ordering for index, Series and categorical-array inputs.
    # GH13743
    idx1 = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=False)
    idx2 = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=True)

    result = MultiIndex.from_arrays([idx1, idx2])
    tm.assert_index_equal(result.get_level_values(0), idx1)
    tm.assert_index_equal(result.get_level_values(1), idx2)

    result2 = MultiIndex.from_arrays([Series(idx1), Series(idx2)])
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    tm.assert_index_equal(result2.get_level_values(1), idx2)

    result3 = MultiIndex.from_arrays([idx1.values, idx2.values])
    tm.assert_index_equal(result3.get_level_values(0), idx1)
    tm.assert_index_equal(result3.get_level_values(1), idx2)


def test_from_arrays_empty():
    # [FUNCTION DOC]
    # Verifies from_arrays with no arrays raises a value error; with one or several empty arrays
    # returns an empty multi-index with the given names equal to a constructor-built one.
    # 0 levels
    msg = "Must pass non-zero number of levels/codes"
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_arrays(arrays=[])

    # 1 level
    result = MultiIndex.from_arrays(arrays=[[]], names=["A"])
    assert isinstance(result, MultiIndex)
    expected = Index([], name="A")
    tm.assert_index_equal(result.levels[0], expected)
    assert result.names == ["A"]

    # N levels
    for N in [2, 3]:
        arrays = [[]] * N
        names = list("ABC")[:N]
        result = MultiIndex.from_arrays(arrays=arrays, names=names)
        expected = MultiIndex(levels=[[]] * N, codes=[[]] * N, names=names)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_sequence_of_arrays",
    [
        1,
        [1],
        [1, 2],
        [[1], 2],
        [1, [2]],
        "a",
        ["a"],
        ["a", "b"],
        [["a"], "b"],
        (1,),
        (1, 2),
        ([1], 2),
        (1, [2]),
        ("a",),
        ("a", "b"),
        (["a"], "b"),
        [(1,), 2],
        [1, (2,)],
        [("a",), "b"],
        ((1,), 2),
        (1, (2,)),
        (("a",), "b"),
    ],
)
def test_from_arrays_invalid_input(invalid_sequence_of_arrays):
    # [FUNCTION DOC]
    # Parametrized over many invalid scalar, string and mixed inputs. Verifies from_arrays
    # raises a type error saying input must be a list of array-likes.
    msg = "Input must be a list / sequence of array-likes"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_arrays(arrays=invalid_sequence_of_arrays)


@pytest.mark.parametrize(
    "idx1, idx2", [([1, 2, 3], ["a", "b"]), ([], ["a", "b"]), ([1, 2, 3], [])]
)
def test_from_arrays_different_lengths(idx1, idx2):
    # [FUNCTION DOC]
    # Parametrized over arrays of unequal length. Verifies (gh-13599) from_arrays raises a
    # value error that all arrays must be the same length.
    # see gh-13599
    msg = "^all arrays must be same length$"
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_arrays([idx1, idx2])


def test_from_arrays_respects_none_names():
    # [FUNCTION DOC]
    # Verifies (GH27292) passing names as none explicitly gives an unnamed index even when
    # the input Series are named.
    # GH27292
    a = Series([1, 2, 3], name="foo")
    b = Series(["a", "b", "c"], name="bar")

    result = MultiIndex.from_arrays([a, b], names=None)
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b", "c"]], codes=[[0, 1, 2], [0, 1, 2]], names=None
    )

    tm.assert_index_equal(result, expected)


# ----------------------------------------------------------------------------
# from_tuples
# ----------------------------------------------------------------------------
def test_from_tuples():
    # [FUNCTION DOC]
    # Verifies from_tuples raises a type error on an empty list (cannot infer number of
    # levels) and builds the expected index from a tuple of tuples.
    msg = "Cannot infer number of levels from empty list"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_tuples([])

    expected = MultiIndex(
        levels=[[1, 3], [2, 4]], codes=[[0, 1], [0, 1]], names=["a", "b"]
    )

    # input tuples
    result = MultiIndex.from_tuples(((1, 2), (3, 4)), names=["a", "b"])
    tm.assert_index_equal(result, expected)


def test_from_tuples_iterator():
    # [FUNCTION DOC]
    # Verifies (GH 18434) from_tuples accepts an iterator and raises a type error for a non-iterable.
    # GH 18434
    # input iterator for tuples
    expected = MultiIndex(
        levels=[[1, 3], [2, 4]], codes=[[0, 1], [0, 1]], names=["a", "b"]
    )

    result = MultiIndex.from_tuples(zip([1, 3], [2, 4], strict=True), names=["a", "b"])
    tm.assert_index_equal(result, expected)

    # input non-iterables
    msg = "Input must be a list / sequence of tuple-likes."
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_tuples(0)


def test_from_tuples_empty():
    # [FUNCTION DOC]
    # Verifies (GH 16777) from_tuples with an empty list and explicit names equals an empty
    # from_arrays result.
    # GH 16777
    result = MultiIndex.from_tuples([], names=["a", "b"])
    expected = MultiIndex.from_arrays(arrays=[[], []], names=["a", "b"])
    tm.assert_index_equal(result, expected)


def test_from_tuples_index_values(idx):
    # [FUNCTION DOC]
    # Verifies from_tuples of an existing multi-index has equal values.
    result = MultiIndex.from_tuples(idx)
    assert (result.values == idx.values).all()


def test_tuples_with_name_string():
    # [FUNCTION DOC]
    # Verifies (GH 15110, GH 14848) constructing a generic Index from tuples with a string name
    # raises a value error asking for list-like names.
    # GH 15110 and GH 14848

    li = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    msg = "Names should be list-like for a MultiIndex"
    with pytest.raises(ValueError, match=msg):
        Index(li, name="abc")
    with pytest.raises(ValueError, match=msg):
        Index(li, name="a")


def test_from_tuples_with_tuple_label():
    # [FUNCTION DOC]
    # Verifies (GH 15457) a tuple used as a label within a level is retained, so a frame
    # indexed by that multi-index equals one set up via a column index.
    # GH 15457
    expected = pd.DataFrame(
        [[2, 1, 2], [4, (1, 2), 3]], columns=["a", "b", "c"]
    ).set_index(["a", "b"])
    idx = MultiIndex.from_tuples([(2, 1), (4, (1, 2))], names=("a", "b"))
    result = pd.DataFrame([2, 3], columns=["c"], index=idx)
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "keys, expected",
    [
        ((("l1",), ("l1", "l2")), (("l1", np.nan), ("l1", "l2"))),
        ((("l1", "l2"), ("l1",)), (("l1", "l2"), ("l1", np.nan))),
    ],
)
def test_from_tuples_with_various_tuple_lengths(keys, expected):
    # [FUNCTION DOC]
    # Parametrized over key sets of different tuple lengths. Verifies (GH 60695) from_tuples
    # pads shorter tuples with missing values.
    # GH 60695
    idx = MultiIndex.from_tuples(keys)
    assert tuple(idx) == expected


# ----------------------------------------------------------------------------
# from_product
# ----------------------------------------------------------------------------
def test_from_product_empty_zero_levels():
    # [FUNCTION DOC]
    # Verifies from_product with no iterables raises a value error requiring levels.
    # 0 levels
    msg = "Must pass non-zero number of levels/codes"
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_product([])


def test_from_product_empty_one_level():
    # [FUNCTION DOC]
    # Verifies from_product with one empty iterable returns an index with that level and name.
    result = MultiIndex.from_product([[]], names=["A"])
    expected = Index([], name="A")
    tm.assert_index_equal(result.levels[0], expected)
    assert result.names == ["A"]


@pytest.mark.parametrize(
    "first, second", [([], []), (["foo", "bar", "baz"], []), ([], ["a", "b", "c"])]
)
def test_from_product_empty_two_levels(first, second):
    # [FUNCTION DOC]
    # Parametrized over pairs with at least one empty list. Verifies from_product equals the
    # constructor with empty codes.
    names = ["A", "B"]
    result = MultiIndex.from_product([first, second], names=names)
    expected = MultiIndex(levels=[first, second], codes=[[], []], names=names)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("N", list(range(4)))
def test_from_product_empty_three_levels(N):
    # [FUNCTION DOC]
    # Parametrized over sizes zero to three for the middle list. Verifies (GH12258) from_product
    # with empty first and third lists equals the constructor result.
    # GH12258
    names = ["A", "B", "C"]
    lvl2 = list(range(N))
    result = MultiIndex.from_product([[], lvl2, []], names=names)
    expected = MultiIndex(levels=[[], lvl2, []], codes=[[], [], []], names=names)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_input", [1, [1], [1, 2], [[1], 2], "a", ["a"], ["a", "b"], [["a"], "b"]]
)
def test_from_product_invalid_input(invalid_input):
    # [FUNCTION DOC]
    # Parametrized over invalid inputs. Verifies from_product raises a type error about
    # input being a list of iterables or list-like.
    msg = r"Input must be a list / sequence of iterables|Input must be list-like"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_product(iterables=invalid_input)


def test_from_product_datetimeindex():
    # [FUNCTION DOC]
    # Verifies from_product with a datetime level gives values equal to an object array of tuples
    # with Timestamp elements.
    dt_index = date_range("2000-01-01", periods=2)
    mi = MultiIndex.from_product([[1, 2], dt_index])
    etalon = construct_1d_object_array_from_listlike(
        [
            (1, Timestamp("2000-01-01")),
            (1, Timestamp("2000-01-02")),
            (2, Timestamp("2000-01-01")),
            (2, Timestamp("2000-01-02")),
        ]
    )
    tm.assert_numpy_array_equal(mi.values, etalon)


def test_from_product_rangeindex():
    # [FUNCTION DOC]
    # Verifies a range index passed as a level is kept as a range index in the internal levels.
    # RangeIndex is preserved by factorize, so preserved in levels
    rng = Index(range(5))
    other = ["a", "b"]
    mi = MultiIndex.from_product([rng, other])
    tm.assert_index_equal(mi._levels[0], rng, exact=True)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("f", [lambda x: x, lambda x: Series(x), lambda x: x.values])
def test_from_product_index_series_categorical(ordered, f):
    # [FUNCTION DOC]
    # Parametrized over ordered flag and input form (index, Series, array). Verifies (GH13743)
    # categorical values keep categories and order in the product's second level, repeated.
    # GH13743
    first = ["foo", "bar"]

    idx = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=ordered)
    expected = pd.CategoricalIndex(
        list("abcaab") + list("abcaab"), categories=list("bac"), ordered=ordered
    )

    result = MultiIndex.from_product([first, f(idx)])
    tm.assert_index_equal(result.get_level_values(1), expected)


def test_from_product():
    # [FUNCTION DOC]
    # Verifies from_product with two lists gives the Cartesian product, equal to the index
    # from the explicit tuples, with names.
    first = ["foo", "bar", "buz"]
    second = ["a", "b", "c"]
    names = ["first", "second"]
    result = MultiIndex.from_product([first, second], names=names)

    tuples = [
        ("foo", "a"),
        ("foo", "b"),
        ("foo", "c"),
        ("bar", "a"),
        ("bar", "b"),
        ("bar", "c"),
        ("buz", "a"),
        ("buz", "b"),
        ("buz", "c"),
    ]
    expected = MultiIndex.from_tuples(tuples, names=names)

    tm.assert_index_equal(result, expected)


def test_from_product_iterator():
    # [FUNCTION DOC]
    # Verifies (GH 18434) from_product accepts an iterator of iterables and raises a type
    # error for non-iterable input.
    # GH 18434
    first = ["foo", "bar", "buz"]
    second = ["a", "b", "c"]
    names = ["first", "second"]
    tuples = [
        ("foo", "a"),
        ("foo", "b"),
        ("foo", "c"),
        ("bar", "a"),
        ("bar", "b"),
        ("bar", "c"),
        ("buz", "a"),
        ("buz", "b"),
        ("buz", "c"),
    ]
    expected = MultiIndex.from_tuples(tuples, names=names)

    # iterator as input
    result = MultiIndex.from_product(iter([first, second]), names=names)
    tm.assert_index_equal(result, expected)

    # Invalid non-iterable input
    msg = "Input must be a list / sequence of iterables."
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_product(0)


@pytest.mark.parametrize(
    "a, b, expected_names",
    [
        (
            Series([1, 2, 3], name="foo"),
            Series(["a", "b"], name="bar"),
            ["foo", "bar"],
        ),
        (Series([1, 2, 3], name="foo"), ["a", "b"], ["foo", None]),
        ([1, 2, 3], ["a", "b"], None),
    ],
)
def test_from_product_infer_names(a, b, expected_names):
    # [FUNCTION DOC]
    # Parametrized over named Series and plain lists. Verifies (GH27292) names are inferred
    # from Series names (none for plain lists).
    # GH27292
    result = MultiIndex.from_product([a, b])
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b"]],
        codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        names=expected_names,
    )
    tm.assert_index_equal(result, expected)


def test_from_product_respects_none_names():
    # [FUNCTION DOC]
    # Verifies (GH27292) explicit none names override named Series inputs.
    # GH27292
    a = Series([1, 2, 3], name="foo")
    b = Series(["a", "b"], name="bar")

    result = MultiIndex.from_product([a, b], names=None)
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b"]],
        codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        names=None,
    )
    tm.assert_index_equal(result, expected)


def test_from_product_readonly():
    # [FUNCTION DOC]
    # Verifies (GH#15286) from_product accepts a read-only array.
    # GH#15286 passing read-only array to from_product
    a = np.array(range(3))
    b = ["a", "b"]
    expected = MultiIndex.from_product([a, b])

    a.setflags(write=False)
    result = MultiIndex.from_product([a, b])
    tm.assert_index_equal(result, expected)


def test_create_index_existing_name(idx):
    # [FUNCTION DOC]
    # Verifies (GH11193) building a generic Index from a multi-index inherits no name unless
    # given, and an explicit name is applied, the result being an object index of tuples.
    # GH11193, when an existing index is passed, and a new name is not
    # specified, the new index should inherit the previous object name
    index = idx
    index.names = ["foo", "bar"]
    result = Index(index)
    expected = Index(
        Index(
            [
                ("foo", "one"),
                ("foo", "two"),
                ("bar", "one"),
                ("baz", "two"),
                ("qux", "one"),
                ("qux", "two"),
            ],
            dtype="object",
        )
    )
    tm.assert_index_equal(result, expected)

    result = Index(index, name="A")
    expected = Index(
        Index(
            [
                ("foo", "one"),
                ("foo", "two"),
                ("bar", "one"),
                ("baz", "two"),
                ("qux", "one"),
                ("qux", "two"),
            ],
            dtype="object",
        ),
        name="A",
    )
    tm.assert_index_equal(result, expected)


# ----------------------------------------------------------------------------
# from_frame
# ----------------------------------------------------------------------------
def test_from_frame():
    # [FUNCTION DOC]
    # Verifies (GH 22420) from_frame builds a multi-index with frame columns as names.
    # GH 22420
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]], columns=["L1", "L2"]
    )
    expected = MultiIndex.from_tuples(
        [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")], names=["L1", "L2"]
    )
    result = MultiIndex.from_frame(df)
    tm.assert_index_equal(expected, result)


def test_from_frame_missing_values_multiIndex():
    # [FUNCTION DOC]
    # Requires pyarrow. Verifies (GH 39984) from_frame keeps extension-dtype columns with missing values.
    # GH 39984
    pa = pytest.importorskip("pyarrow")

    df = pd.DataFrame(
        {
            "a": Series([1, 2, None], dtype="Int64"),
            "b": pd.Float64Dtype().__from_arrow__(pa.array([0.2, None, None])),
        }
    )
    multi_indexed = MultiIndex.from_frame(df)
    expected = MultiIndex.from_arrays(
        [
            Series([1, 2, None], dtype="Int64"),
            pd.Float64Dtype().__from_arrow__(pa.array([0.2, None, None])),
        ],
        names=["a", "b"],
    )
    tm.assert_index_equal(multi_indexed, expected)


@pytest.mark.parametrize(
    "non_frame",
    [
        Series([1, 2, 3, 4]),
        [1, 2, 3, 4],
        [[1, 2], [3, 4], [5, 6]],
        Index([1, 2, 3, 4]),
        np.array([[1, 2], [3, 4], [5, 6]]),
        27,
    ],
)
def test_from_frame_error(non_frame):
    # [FUNCTION DOC]
    # Parametrized over non-frame inputs. Verifies (GH 22420) from_frame raises a type error
    # that the input must be a DataFrame.
    # GH 22420
    with pytest.raises(TypeError, match="Input must be a DataFrame"):
        MultiIndex.from_frame(non_frame)


def test_from_frame_dtype_fidelity():
    # [FUNCTION DOC]
    # Verifies (GH 22420) from_frame preserves level dtypes (aware datetime, integer,
    # categorical, string) and values.
    # GH 22420
    df = pd.DataFrame(
        {
            "dates": date_range("19910905", periods=6, tz="US/Eastern"),
            "a": [1, 1, 1, 2, 2, 2],
            "b": pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),
            "c": ["x", "x", "y", "z", "x", "y"],
        }
    )
    original_dtypes = df.dtypes.to_dict()

    expected_mi = MultiIndex.from_arrays(
        [
            date_range("19910905", periods=6, tz="US/Eastern"),
            [1, 1, 1, 2, 2, 2],
            pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),
            ["x", "x", "y", "z", "x", "y"],
        ],
        names=["dates", "a", "b", "c"],
    )
    mi = MultiIndex.from_frame(df)
    mi_dtypes = {name: mi.levels[i].dtype for i, name in enumerate(mi.names)}

    tm.assert_index_equal(expected_mi, mi)
    assert original_dtypes == mi_dtypes


@pytest.mark.parametrize(
    "names_in,names_out", [(None, [("L1", "x"), ("L2", "y")]), (["x", "y"], ["x", "y"])]
)
def test_from_frame_valid_names(names_in, names_out):
    # [FUNCTION DOC]
    # Parametrized over (names argument, expected names). Verifies (GH 22420) frame column
    # tuples become names by default and explicit names override them.
    # GH 22420
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]],
        columns=MultiIndex.from_tuples([("L1", "x"), ("L2", "y")]),
    )
    mi = MultiIndex.from_frame(df, names=names_in)
    assert mi.names == names_out


@pytest.mark.parametrize(
    "names,expected_error_msg",
    [
        ("bad_input", "Names should be list-like for a MultiIndex"),
        (["a", "b", "c"], "Length of names must match number of levels in MultiIndex"),
    ],
)
def test_from_frame_invalid_names(names, expected_error_msg):
    # [FUNCTION DOC]
    # Parametrized over invalid names (a string, a list of wrong length). Verifies (GH 22420)
    # from_frame raises a value error with specific messages.
    # GH 22420
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]],
        columns=MultiIndex.from_tuples([("L1", "x"), ("L2", "y")]),
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        MultiIndex.from_frame(df, names=names)


def test_index_equal_empty_iterable():
    # [FUNCTION DOC]
    # Verifies (#16844) an empty multi-index built via the constructor equals one built by
    # from_arrays from empty arrays.
    # #16844
    a = MultiIndex(levels=[[], []], codes=[[], []], names=["a", "b"])
    b = MultiIndex.from_arrays(arrays=[[], []], names=["a", "b"])
    tm.assert_index_equal(a, b)


def test_raise_invalid_sortorder():
    # [FUNCTION DOC]
    # Verifies (GH#28518) the constructor accepts a valid sort order and raises a value error
    # that mentions the sort order and actual lexsort depth when the data is not sorted as claimed.
    # Test that the MultiIndex constructor raise when an incorrect sortorder is given
    # GH#28518

    levels = [[0, 1], [0, 1, 2]]

    # Correct sortorder
    MultiIndex(
        levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], sortorder=2
    )

    with pytest.raises(ValueError, match=r".* sortorder 2 with lexsort_depth 1.*"):
        MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]], sortorder=2
        )

    with pytest.raises(ValueError, match=r".* sortorder 1 with lexsort_depth 0.*"):
        MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]], sortorder=1
        )


def test_datetimeindex():
    # [FUNCTION DOC]
    # Verifies datetime levels keep time zones, and (GH 7888) product of numpy, datetime and
    # Timestamp scalars gives datetime levels, but plain date objects do not.
    idx1 = pd.DatetimeIndex(
        ["2013-04-01 9:00", "2013-04-02 9:00", "2013-04-03 9:00"] * 2, tz="Asia/Tokyo"
    )
    idx2 = date_range("2010/01/01", periods=6, freq="ME", tz="US/Eastern")
    idx = MultiIndex.from_arrays([idx1, idx2])

    expected1 = pd.DatetimeIndex(
        ["2013-04-01 9:00", "2013-04-02 9:00", "2013-04-03 9:00"], tz="Asia/Tokyo"
    )

    tm.assert_index_equal(idx.levels[0], expected1)
    tm.assert_index_equal(idx.levels[1], idx2)

    # from datetime combos
    # GH 7888
    date1 = np.datetime64("today")
    date2 = datetime.today()
    date3 = Timestamp.today()

    for d1, d2 in itertools.product([date1, date2, date3], [date1, date2, date3]):
        index = MultiIndex.from_product([[d1], [d2]])
        assert isinstance(index.levels[0], pd.DatetimeIndex)
        assert isinstance(index.levels[1], pd.DatetimeIndex)

    # but NOT date objects, matching Index behavior
    date4 = date.today()
    index = MultiIndex.from_product([[date4], [date2]])
    assert not isinstance(index.levels[0], pd.DatetimeIndex)
    assert isinstance(index.levels[1], pd.DatetimeIndex)


def test_constructor_with_tz():
    # [FUNCTION DOC]
    # Verifies from_arrays of aware indexes and Series keeps names and levels.
    index = pd.DatetimeIndex(
        ["2013/01/01 09:00", "2013/01/02 09:00"], name="dt1", tz="US/Pacific"
    )
    columns = pd.DatetimeIndex(
        ["2014/01/01 09:00", "2014/01/02 09:00"], name="dt2", tz="Asia/Tokyo"
    )

    result = MultiIndex.from_arrays([index, columns])

    assert result.names == ["dt1", "dt2"]
    tm.assert_index_equal(result.levels[0], index)
    tm.assert_index_equal(result.levels[1], columns)

    result = MultiIndex.from_arrays([Series(index), Series(columns)])

    assert result.names == ["dt1", "dt2"]
    tm.assert_index_equal(result.levels[0], index)
    tm.assert_index_equal(result.levels[1], columns)


def test_multiindex_inference_consistency():
    # [FUNCTION DOC]
    # Verifies that level dtype inference for date objects is object dtype for the Index
    # constructor and for from_arrays, from_product and from_tuples, consistent with the base class.
    # check that inference behavior matches the base class

    v = date.today()

    arr = [v, v]

    idx = Index(arr)
    assert idx.dtype == object

    mi = MultiIndex.from_arrays([arr])
    lev = mi.levels[0]
    assert lev.dtype == object

    mi = MultiIndex.from_product([arr])
    lev = mi.levels[0]
    assert lev.dtype == object

    mi = MultiIndex.from_tuples([(x,) for x in arr])
    lev = mi.levels[0]
    assert lev.dtype == object


def test_dtype_representation(using_infer_string):
    # [FUNCTION DOC]
    # Verifies (GH#46900) the dtypes of a multi-index with tuple names are returned as a
    # Series with a multi-index of those names.
    # GH#46900
    pmidx = MultiIndex.from_arrays([[1], ["a"]], names=[("a", "b"), ("c", "d")])
    result = pmidx.dtypes
    exp = "object" if not using_infer_string else pd.StringDtype(na_value=np.nan)
    expected = Series(
        ["int64", exp],
        index=MultiIndex.from_tuples([("a", "b"), ("c", "d")]),
        dtype=object,
    )
    tm.assert_series_equal(result, expected)
