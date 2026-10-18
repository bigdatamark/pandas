# [MODULE DOC]
# Tests for Series.explode, which turns list-like elements into separate rows
# (repeating the index). Covers lists, tuples, sets, arrays, nested containers,
# empty and missing elements, MultiIndex, duplicated index, ignore_index,
# extension arrays, pyarrow list types and string dtype. Some tests need pyarrow.
# No import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_basic():
    # [FUNCTION DOC]
    # A Series of a list, NaN, an empty list and a tuple explodes to one row per
    # element; the NaN and the empty list each give a NaN row; the index repeats; the
    # result is object dtype and keeps the name.
    s = pd.Series([[0, 1, 2], np.nan, [], (3, 4)], index=list("abcd"), name="foo")
    result = s.explode()
    expected = pd.Series(
        [0, 1, 2, np.nan, np.nan, 3, 4], index=list("aaabcdd"), dtype=object, name="foo"
    )
    tm.assert_series_equal(result, expected)


def test_mixed_type():
    # [FUNCTION DOC]
    # Mixed elements (list, NaN, None, empty array, Series) explode with None kept
    # and NaN for empty containers; the index values repeat accordingly.
    s = pd.Series(
        [[0, 1, 2], np.nan, None, np.array([]), pd.Series(["a", "b"])], name="foo"
    )
    result = s.explode()
    expected = pd.Series(
        [0, 1, 2, np.nan, None, np.nan, "a", "b"],
        index=[0, 0, 0, 1, 2, 3, 4, 4],
        dtype=object,
        name="foo",
    )
    tm.assert_series_equal(result, expected)


def test_empty():
    # [FUNCTION DOC]
    # Exploding an empty object Series returns an equal empty Series.
    s = pd.Series(dtype=object)
    result = s.explode()
    expected = s.copy()
    tm.assert_series_equal(result, expected)


def test_nested_lists():
    # [FUNCTION DOC]
    # Only one level of nesting is exploded: a nested list stays as a list element
    # while scalar elements are kept.
    s = pd.Series([[[1, 2, 3]], [1, 2], 1])
    result = s.explode()
    expected = pd.Series([[1, 2, 3], 1, 2, 1], index=[0, 1, 1, 2])
    tm.assert_series_equal(result, expected)


def test_multi_index():
    # [FUNCTION DOC]
    # A Series with a two-level MultiIndex explodes with the full index tuples
    # repeated, keeping index names and Series name; object dtype.
    s = pd.Series(
        [[0, 1, 2], np.nan, [], (3, 4)],
        name="foo",
        index=pd.MultiIndex.from_product([list("ab"), range(2)], names=["foo", "bar"]),
    )
    result = s.explode()
    index = pd.MultiIndex.from_tuples(
        [("a", 0), ("a", 0), ("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 1)],
        names=["foo", "bar"],
    )
    expected = pd.Series(
        [0, 1, 2, np.nan, np.nan, 3, 4], index=index, dtype=object, name="foo"
    )
    tm.assert_series_equal(result, expected)


def test_large():
    # [FUNCTION DOC]
    # Intended to check a 256-element range in a single element.
    # KNOWN ISSUE: the first explode is assigned to the variable that is then
    # exploded again, so the test only checks that exploding an already exploded
    # Series is a no-op.
    s = pd.Series([range(256)]).explode()
    result = s.explode()
    tm.assert_series_equal(result, s)


def test_invert_array():
    # [FUNCTION DOC]
    # Applying a function that returns the extension array of each row of a tz-aware
    # datetime frame column gives a Series of arrays; exploding it recovers the
    # original tz-aware Series (with its name removed).
    df = pd.DataFrame({"a": pd.date_range("20190101", periods=3, tz="UTC")})

    listify = df.apply(lambda x: x.array, axis=1)
    result = listify.explode()
    tm.assert_series_equal(result, df["a"].rename())


@pytest.mark.parametrize(
    "data", [[1, 2, 3], pd.date_range("2019", periods=3, tz="UTC")]
)
def test_non_object_dtype(data):
    # [FUNCTION DOC]
    # Parametrized over integer data and tz-aware datetimes: a non-object Series is
    # returned unchanged.
    ser = pd.Series(data)
    result = ser.explode()
    tm.assert_series_equal(result, ser)


def test_typical_usecase():
    # [FUNCTION DOC]
    # Typical use: split a comma-separated string column into lists, explode and
    # join with another column; each original row yields three rows with the
    # repeated index.
    df = pd.DataFrame(
        [{"var1": "a,b,c", "var2": 1}, {"var1": "d,e,f", "var2": 2}],
        columns=["var1", "var2"],
    )
    exploded = df.var1.str.split(",").explode()
    result = df[["var2"]].join(exploded)
    expected = pd.DataFrame(
        {"var2": [1, 1, 1, 2, 2, 2], "var1": list("abcdef")},
        columns=["var2", "var1"],
        index=[0, 0, 0, 1, 1, 1],
    )
    tm.assert_frame_equal(result, expected)


def test_nested_EA():
    # [FUNCTION DOC]
    # A Series whose elements are tz-aware datetime arrays explodes into a single
    # tz-aware datetime Series with repeated index values.
    # a nested EA array
    s = pd.Series(
        [
            pd.date_range("20170101", periods=3, tz="UTC"),
            pd.date_range("20170104", periods=3, tz="UTC"),
        ]
    )
    result = s.explode()
    expected = pd.Series(
        pd.date_range("20170101", periods=6, tz="UTC"), index=[0, 0, 0, 1, 1, 1]
    )
    tm.assert_series_equal(result, expected)


def test_duplicate_index():
    # [FUNCTION DOC]
    # Regression test (GH 28005): with a duplicated index, the exploded result
    # repeats the index labels and has object dtype.
    # GH 28005
    s = pd.Series([[1, 2], [3, 4]], index=[0, 0])
    result = s.explode()
    expected = pd.Series([1, 2, 3, 4], index=[0, 0, 0, 0], dtype=object)
    tm.assert_series_equal(result, expected)


def test_ignore_index():
    # [FUNCTION DOC]
    # Regression test (GH 34932): the ignore_index option gives a fresh range index.
    # GH 34932
    s = pd.Series([[1, 2], [3, 4]])
    result = s.explode(ignore_index=True)
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3], dtype=object)
    tm.assert_series_equal(result, expected)


def test_explode_sets():
    # [FUNCTION DOC]
    # Exploding a Series holding a set gives each element as a row (sorted in the
    # test for determinism) repeating the index label.
    # https://github.com/pandas-dev/pandas/issues/35614
    s = pd.Series([{"a", "b", "c"}], index=[1])
    result = s.explode().sort_values()
    expected = pd.Series(["a", "b", "c"], index=[1, 1, 1])
    tm.assert_series_equal(result, expected)


def test_explode_scalars_can_ignore_index():
    # [FUNCTION DOC]
    # Scalars (non list-like) with ignore_index: the result values are unchanged with
    # a fresh range index.
    # https://github.com/pandas-dev/pandas/issues/40487
    s = pd.Series([1, 2, 3], index=["a", "b", "c"])
    result = s.explode(ignore_index=True)
    expected = pd.Series([1, 2, 3])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("list_type", ["list_", "large_list"])
def test_explode_pyarrow_list_type(ignore_index, list_type):
    # [FUNCTION DOC]
    # Requires pyarrow. Parametrized over ignore_index and list or large_list arrow
    # types (GH 53602, 61091). Exploding a pyarrow list Series with None elements, an
    # empty list and a missing list gives int64 pyarrow values, one missing for each
    # empty list or missing list, with the index repeated (or a fresh index when
    # ignored).
    # GH 53602, 61091
    pa = pytest.importorskip("pyarrow")

    data = [
        [None, None],
        [1],
        [],
        [2, 3],
        None,
    ]
    ser = pd.Series(data, dtype=pd.ArrowDtype(getattr(pa, list_type)(pa.int64())))
    result = ser.explode(ignore_index=ignore_index)
    expected = pd.Series(
        data=[None, None, 1, None, 2, 3, None],
        index=None if ignore_index else [0, 0, 1, 2, 3, 3, 4],
        dtype=pd.ArrowDtype(pa.int64()),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ignore_index", [True, False])
def test_explode_pyarrow_non_list_type(ignore_index):
    # [FUNCTION DOC]
    # Requires pyarrow, parametrized over ignore_index: exploding a pyarrow
    # (non-list) integer Series returns it unchanged with a range index.
    pa = pytest.importorskip("pyarrow")
    data = [1, 2, 3]
    ser = pd.Series(data, dtype=pd.ArrowDtype(pa.int64()))
    result = ser.explode(ignore_index=ignore_index)
    expected = pd.Series([1, 2, 3], dtype="int64[pyarrow]", index=[0, 1, 2])
    tm.assert_series_equal(result, expected)


def test_str_dtype():
    # [FUNCTION DOC]
    # GitHub PR 61623: exploding a string-dtype Series returns an equal Series that
    # is a different object.
    # https://github.com/pandas-dev/pandas/pull/61623
    ser = pd.Series(["x", "y"], dtype="str")
    result = ser.explode()
    assert result is not ser
    tm.assert_series_equal(result, ser)
