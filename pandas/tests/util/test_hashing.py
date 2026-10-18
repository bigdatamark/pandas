# [MODULE DOC]
# Tests for the row-hashing utilities: hashing arrays, tuples and pandas objects (series, data frames, indexes, multi-indexes) into unsigned 64-bit values. Covers determinism, ground-truth values, error handling for bad input, index inclusion, categorical consistency with values, hash keys, encodings, collision regressions, tuples inside frames and the explicit none hash key. Defines two fixtures: a set of series of various dtypes and an index-inclusion boolean.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
    hash_array,
    hash_pandas_object,
)


@pytest.fixture(
    params=[
        Series([1, 2, 3] * 3, dtype="int32"),
        Series([None, 2.5, 3.5] * 3, dtype="float32"),
        Series(["a", "b", "c"] * 3, dtype="category"),
        Series(["d", "e", "f"] * 3),
        Series([True, False, True] * 3),
        Series(pd.date_range("20130101", periods=9)),
        Series(pd.date_range("20130101", periods=9, tz="US/Eastern")),
        Series(timedelta_range("2000", periods=9)),
    ]
)
def series(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over eight series (int32, float32 with a missing value, categorical, strings, booleans, naive datetimes, timezone-aware datetimes, timedeltas), each of length nine.
    return request.param


@pytest.fixture(params=[True, False])
def index(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false; supplies the index-inclusion option of the hashing function. (Shadows the name of the common index fixtures within this module.)
    return request.param


def test_consistency():
    # [FUNCTION DOC]
    # Ground-truth test: hashing an index of three strings returns a series indexed by those strings whose values equal three hardcoded unsigned 64-bit integers, so any accidental change to the hashing algorithm is caught.
    # Check that our hash doesn't change because of a mistake
    # in the actual code; this is the ground truth.
    result = hash_pandas_object(Index(["foo", "bar", "baz"]))
    expected = Series(
        np.array(
            [3600424527151052760, 1374399572096150070, 477881037637427054],
            dtype="uint64",
        ),
        index=["foo", "bar", "baz"],
    )
    tm.assert_series_equal(result, expected)


def test_hash_array(series):
    # [FUNCTION DOC]
    # For each series fixture, hashing the underlying values twice gives equal arrays (determinism).
    arr = series.values
    tm.assert_numpy_array_equal(hash_array(arr), hash_array(arr))


@pytest.mark.parametrize("dtype", ["U", object])
def test_hash_array_mixed(dtype):
    # [FUNCTION DOC]
    # Parametrized over unicode string dtype and object dtype. A string array and an array mixing numbers and strings (made into the same strings) hash identically.
    result1 = hash_array(np.array(["3", "4", "All"]))
    result2 = hash_array(np.array([3, 4, "All"], dtype=dtype))

    tm.assert_numpy_array_equal(result1, result2)


@pytest.mark.parametrize("val", [5, "foo", pd.Timestamp("20130101")])
def test_hash_array_errors(val):
    # [FUNCTION DOC]
    # Parametrized over an integer, a string and a timestamp. Hashing a non-array raises a type error saying an array-like must be passed.
    msg = "must pass an ndarray-like"
    with pytest.raises(TypeError, match=msg):
        hash_array(val)


def test_hash_array_index_exception():
    # [FUNCTION DOC]
    # GH42003. Hashing a timezone-aware datetime index directly raises a type error telling the caller to use the object-hashing function instead (rather than an attribute error).
    # GH42003 TypeError instead of AttributeError
    obj = pd.DatetimeIndex(["2018-10-28 01:20:00"], tz="Europe/Berlin")

    msg = "Use hash_pandas_object instead"
    with pytest.raises(TypeError, match=msg):
        hash_array(obj)


def test_hash_tuples():
    # [FUNCTION DOC]
    # Hashing a list of tuples equals the hash values of a multi-index built from them. Hashing a single tuple (not a list of tuples) raises a type error about not being iterable or about the zip argument.
    tuples = [(1, "one"), (1, "two"), (2, "one")]
    result = hash_tuples(tuples)

    expected = hash_pandas_object(MultiIndex.from_tuples(tuples)).values
    tm.assert_numpy_array_equal(result, expected)

    # We only need to support MultiIndex and list-of-tuples
    msg = "|".join(["object is not iterable", "zip argument #1 must support iteration"])
    with pytest.raises(TypeError, match=msg):
        hash_tuples(tuples[0])


@pytest.mark.parametrize("val", [5, "foo", pd.Timestamp("20130101")])
def test_hash_tuples_err(val):
    # [FUNCTION DOC]
    # Parametrized over an integer, string and timestamp. Hashing raises a type error saying the argument must be convertible to a list of tuples.
    msg = "must be convertible to a list-of-tuples"
    with pytest.raises(TypeError, match=msg):
        hash_tuples(val)


def test_multiindex_unique():
    # [FUNCTION DOC]
    # A multi-index of four distinct integer pairs is unique and so is the series of its hashes (exactly true).
    mi = MultiIndex.from_tuples([(118, 472), (236, 118), (51, 204), (102, 51)])
    assert mi.is_unique is True

    result = hash_pandas_object(mi)
    assert result.is_unique is True


def test_multiindex_objects():
    # [FUNCTION DOC]
    # Builds a multi-index with unsorted levels and its monotonic-sorted version; asserts the two are equal and that indexes made from their values are equal.
    mi = MultiIndex(
        levels=[["b", "d", "a"], [1, 2, 3]],
        codes=[[0, 1, 0, 2], [2, 0, 0, 1]],
        names=["col1", "col2"],
    )
    recons = mi._sort_levels_monotonic()

    # These are equal.
    assert mi.equals(recons)
    assert Index(mi.values).equals(Index(recons.values))


@pytest.mark.parametrize(
    "obj",
    [
        Series([1, 2, 3]),
        Series([1.0, 1.5, 3.2]),
        Series([1.0, 1.5, np.nan]),
        Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]),
        Series(["a", "b", "c"]),
        Series(["a", np.nan, "c"]),
        Series(["a", None, "c"]),
        Series([True, False, True]),
        Series(dtype=object),
        DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]}),
        DataFrame(),
        DataFrame(np.full((10, 4), np.nan)),
        DataFrame(
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"], dtype=object),
                "D": pd.date_range("20130101", periods=5),
            }
        ),
        DataFrame(range(5), index=pd.date_range("2020-01-01", periods=5)),
        Series(range(5), index=pd.date_range("2020-01-01", periods=5)),
        Series(period_range("2020-01-01", periods=10, freq="D")),
        Series(pd.date_range("20130101", periods=3, tz="US/Eastern")),
    ],
)
def test_hash_pandas_object(obj, index):
    # [FUNCTION DOC]
    # Parametrized over seventeen objects (series, data frames, empty ones, with various index types) and the index-inclusion fixture. Hashing twice produces equal series.
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize(
    "obj",
    [
        Series([1, 2, 3]),
        Series([1.0, 1.5, 3.2]),
        Series([1.0, 1.5, np.nan]),
        Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]),
        Series(["a", "b", "c"]),
        Series(["a", np.nan, "c"]),
        Series(["a", None, "c"]),
        Series([True, False, True]),
        DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]}),
        DataFrame(np.full((10, 4), np.nan)),
        DataFrame(
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"], dtype=object),
                "D": pd.date_range("20130101", periods=5),
            }
        ),
        DataFrame(range(5), index=pd.date_range("2020-01-01", periods=5)),
        Series(range(5), index=pd.date_range("2020-01-01", periods=5)),
        Series(period_range("2020-01-01", periods=10, freq="D")),
        Series(pd.date_range("20130101", periods=3, tz="US/Eastern")),
    ],
)
def test_hash_pandas_object_diff_index_non_empty(obj):
    # [FUNCTION DOC]
    # Parametrized over the same objects without the empty ones. Hashing with the index included differs (not all equal) from hashing without it.
    a = hash_pandas_object(obj, index=True)
    b = hash_pandas_object(obj, index=False)
    assert not (a == b).all()


@pytest.mark.parametrize(
    "obj",
    [
        Index([1, 2, 3]),
        Index([True, False, True]),
        timedelta_range("1 day", periods=2),
        period_range("2020-01-01", freq="D", periods=2),
        MultiIndex.from_product(
            [range(5), ["foo", "bar", "baz"], pd.date_range("20130101", periods=2)]
        ),
        MultiIndex.from_product([pd.CategoricalIndex(list("aabc")), range(3)]),
    ],
)
def test_hash_pandas_index(obj, index):
    # [FUNCTION DOC]
    # Parametrized over indexes (integer, boolean, timedelta, period, two multi-indexes) and the index-inclusion fixture. Hashing twice gives equal series.
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


def test_hash_pandas_series(series, index):
    # [FUNCTION DOC]
    # Using the series and index fixtures, hashing twice gives equal series.
    a = hash_pandas_object(series, index=index)
    b = hash_pandas_object(series, index=index)
    tm.assert_series_equal(a, b)


def test_hash_pandas_series_diff_index(series):
    # [FUNCTION DOC]
    # For each fixture series, hash with and without the index differ (not all equal).
    a = hash_pandas_object(series, index=True)
    b = hash_pandas_object(series, index=False)
    assert not (a == b).all()


@pytest.mark.parametrize("klass", [Index, Series])
@pytest.mark.parametrize("dtype", ["float64", "object"])
def test_hash_pandas_empty_object(klass, dtype, index):
    # [FUNCTION DOC]
    # Parametrized over index and series classes, float and object dtypes and the index fixture. Hashing an empty object twice gives equal series (the index makes no difference because there is no data).
    # These are by-definition the same with
    # or without the index as the data is empty.
    obj = klass([], dtype=dtype)
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize(
    "s1",
    [
        ["a", "b", "c", "d"],
        [1000, 2000, 3000, 4000],
        pd.date_range(0, periods=4),
    ],
)
@pytest.mark.parametrize("categorize", [True, False])
def test_categorical_consistency(s1, categorize):
    # [FUNCTION DOC]
    # Regression test (gh-15143). Parametrized over three kinds of values (strings, integers, datetimes) and the categorize option. A plain series, its categorical version, and a categorical with reversed category order all hash identically: categoricals hash by value, not codes.
    # see gh-15143
    #
    # Check that categoricals hash consistent with their values,
    # not codes. This should work for categoricals of any dtype.
    s1 = Series(s1)
    s2 = s1.astype("category").cat.set_categories(s1)
    s3 = s2.cat.set_categories(list(reversed(s1)))

    # These should all hash identically.
    h1 = hash_pandas_object(s1, categorize=categorize)
    h2 = hash_pandas_object(s2, categorize=categorize)
    h3 = hash_pandas_object(s3, categorize=categorize)

    tm.assert_series_equal(h1, h2)
    tm.assert_series_equal(h1, h3)


def test_categorical_with_nan_consistency(unit):
    # [FUNCTION DOC]
    # Parametrized by the time unit fixture. Hashes a categorical (without categorizing) with a missing code and five datetime categories; then hashes a two-element categorical with the missing code and the first timestamp as its only category. Both result values must be among the hashes of the first.
    dti = pd.date_range("2012-01-01", periods=5, name="B", unit=unit)
    cat = pd.Categorical.from_codes([-1, 0, 1, 2, 3, 4], categories=dti)
    expected = hash_array(cat, categorize=False)

    ts = pd.Timestamp("2012-01-01").as_unit(unit)
    cat2 = pd.Categorical.from_codes([-1, 0], categories=[ts])
    result = hash_array(cat2, categorize=False)

    assert result[0] in expected
    assert result[1] in expected


def test_pandas_errors():
    # [FUNCTION DOC]
    # Hashing a scalar timestamp with the pandas-object function raises a type error saying the type is unexpected.
    msg = "Unexpected type for hashing"
    with pytest.raises(TypeError, match=msg):
        hash_pandas_object(pd.Timestamp("20130101"))


def test_hash_keys():
    # [FUNCTION DOC]
    # With two hash keys differing in the last two characters, the hashes of an object-dtype string series all differ.
    # Using different hash keys, should have
    # different hashes for the same data.
    #
    # This only matters for object dtypes.
    obj = Series(list("abc"))

    a = hash_pandas_object(obj, hash_key="9876543210123456")
    b = hash_pandas_object(obj, hash_key="9876543210123465")

    assert (a != b).all()


def test_df_hash_keys():
    # [FUNCTION DOC]
    # Issue 41404. Same as the hash keys test but for a data frame with an integer and a string column.
    # DataFrame version of the test_hash_keys.
    # https://github.com/pandas-dev/pandas/issues/41404
    obj = DataFrame({"x": np.arange(3), "y": list("abc")})

    a = hash_pandas_object(obj, hash_key="9876543210123456")
    b = hash_pandas_object(obj, hash_key="9876543210123465")

    assert (a != b).all()


def test_df_encoding():
    # [FUNCTION DOC]
    # Issue 41404 and pull request 42049. Hashing a frame with strings a, plus, c using utf8 versus utf7 encoding gives equal hashes for the first and third rows and a different hash for the row with the plus character (which utf7 encodes as plus-dash).
    # Check that DataFrame recognizes optional encoding.
    # https://github.com/pandas-dev/pandas/issues/41404
    # https://github.com/pandas-dev/pandas/pull/42049
    obj = DataFrame({"x": np.arange(3), "y": list("a+c")})

    a = hash_pandas_object(obj, encoding="utf8")
    b = hash_pandas_object(obj, encoding="utf7")

    # Note that the "+" is encoded as "+-" in utf-7.
    assert a[0] == b[0]
    assert a[1] != b[1]
    assert a[2] == b[2]


def test_invalid_key():
    # [FUNCTION DOC]
    # A hash key that is not a 16-character string raises a value error saying the key must be a 16-byte string.
    # This only matters for object dtypes.
    msg = "key should be a 16-byte string encoded"

    with pytest.raises(ValueError, match=msg):
        hash_pandas_object(Series(list("abc")), hash_key="foo")


def test_already_encoded(index):
    # [FUNCTION DOC]
    # Series of already encoded bytes hashed twice (with and without the index per the fixture) give equal series.
    # If already encoded, then ok.
    obj = Series(list("abc")).str.encode("utf8")
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


def test_alternate_encoding(index):
    # [FUNCTION DOC]
    # Series of strings hashed twice with the default encoding give equal series.
    # KNOWN ISSUE: despite the name, no alternate encoding is passed to the hashing function, so the test just repeats the default-encoding determinism check.
    obj = Series(list("abc"))
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize("l_exp", range(8))
@pytest.mark.parametrize("l_add", [0, 1])
def test_same_len_hash_collisions(l_exp, l_add):
    # [FUNCTION DOC]
    # Parametrized over exponent 0 to 7 and an add of 0 or 1. Builds an object array of string numbers whose length is two to the power of the exponent plus eight, plus the add; hashing with utf8 gives different hashes for the first two elements.
    length = 2 ** (l_exp + 8) + l_add
    idx = np.array([str(i) for i in range(length)], dtype=object)

    result = hash_array(idx, "utf8")
    assert not result[0] == result[1]


def test_hash_collisions():
    # [FUNCTION DOC]
    # Regression test for a known collision issue (14711). Two long strings hash to the hardcoded unsigned 64-bit values individually, and hashing both together gives the concatenation of those values.
    # Hash collisions are bad.
    #
    # https://github.com/pandas-dev/pandas/issues/14711#issuecomment-264885726
    hashes = [
        "Ingrid-9Z9fKIZmkO7i7Cn51Li34pJm44fgX6DYGBNj3VPlOH50m7HnBlPxfIwFMrcNJNMP6PSgLmwWnInciMWrCSAlLEvt7JkJl4IxiMrVbXSa8ZQoVaq5xoQPjltuJEfwdNlO6jo8qRRHvD8sBEBMQASrRa6TsdaPTPCBo3nwIBpE7YzzmyH0vMBhjQZLx1aCT7faSEx7PgFxQhHdKFWROcysamgy9iVj8DO2Fmwg1NNl93rIAqC3mdqfrCxrzfvIY8aJdzin2cHVzy3QUJxZgHvtUtOLxoqnUHsYbNTeq0xcLXpTZEZCxD4PGubIuCNf32c33M7HFsnjWSEjE2yVdWKhmSVodyF8hFYVmhYnMCztQnJrt3O8ZvVRXd5IKwlLexiSp4h888w7SzAIcKgc3g5XQJf6MlSMftDXm9lIsE1mJNiJEv6uY6pgvC3fUPhatlR5JPpVAHNSbSEE73MBzJrhCAbOLXQumyOXigZuPoME7QgJcBalliQol7YZ9",
        "Tim-b9MddTxOWW2AT1Py6vtVbZwGAmYCjbp89p8mxsiFoVX4FyDOF3wFiAkyQTUgwg9sVqVYOZo09Dh1AzhFHbgij52ylF0SEwgzjzHH8TGY8Lypart4p4onnDoDvVMBa0kdthVGKl6K0BDVGzyOXPXKpmnMF1H6rJzqHJ0HywfwS4XYpVwlAkoeNsiicHkJUFdUAhG229INzvIAiJuAHeJDUoyO4DCBqtoZ5TDend6TK7Y914yHlfH3g1WZu5LksKv68VQHJriWFYusW5e6ZZ6dKaMjTwEGuRgdT66iU5nqWTHRH8WSzpXoCFwGcTOwyuqPSe0fTe21DVtJn1FKj9F9nEnR9xOvJUO7E0piCIF4Ad9yAIDY4DBimpsTfKXCu1vdHpKYerzbndfuFe5AhfMduLYZJi5iAw8qKSwR5h86ttXV0Mc0QmXz8dsRvDgxjXSmupPxBggdlqUlC828hXiTPD7am0yETBV0F3bEtvPiNJfremszcV8NcqAoARMe",
    ]

    # These should be different.
    result1 = hash_array(np.asarray(hashes[0:1], dtype=object), "utf8")
    expected1 = np.array([14963968704024874985], dtype=np.uint64)
    tm.assert_numpy_array_equal(result1, expected1)

    result2 = hash_array(np.asarray(hashes[1:2], dtype=object), "utf8")
    expected2 = np.array([16428432627716348016], dtype=np.uint64)
    tm.assert_numpy_array_equal(result2, expected2)

    result = hash_array(np.asarray(hashes, dtype=object), "utf8")
    tm.assert_numpy_array_equal(result, np.concatenate([expected1, expected2], axis=0))


@pytest.mark.parametrize(
    "data, result_data",
    [
        [[tuple("1"), tuple("2")], [10345501319357378243, 8331063931016360761]],
        [[(1,), (2,)], [9408946347443669104, 3278256261030523334]],
    ],
)
def test_hash_with_tuple(data, result_data):
    # [FUNCTION DOC]
    # GH#28969. Parametrized over frames holding string tuples or integer tuples in a column. Hashing the frame gives a series of hardcoded unsigned 64-bit values (works around a numpy bug when converting an array containing tuples to strings).
    # GH#28969 array containing a tuple raises on call to arr.astype(str)
    #  apparently a numpy bug github.com/numpy/numpy/issues/9441

    df = DataFrame({"data": data})
    result = hash_pandas_object(df)
    expected = Series(result_data, dtype=np.uint64)
    tm.assert_series_equal(result, expected)


def test_hashable_tuple_args():
    # [FUNCTION DOC]
    # A frame whose column has tuples containing a list or a dictionary raises a type error that the list type is unhashable when hashed.
    # require that the elements of such tuples are themselves hashable

    df3 = DataFrame(
        {
            "data": [
                (
                    1,
                    [],
                ),
                (
                    2,
                    {},
                ),
            ]
        }
    )
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        hash_pandas_object(df3)


def test_hash_object_none_key():
    # [FUNCTION DOC]
    # Issue 30887. Hashing a series of two strings with a none hash key via the util namespace gives two hardcoded unsigned 64-bit values.
    # https://github.com/pandas-dev/pandas/issues/30887
    result = pd.util.hash_pandas_object(Series(["a", "b"]), hash_key=None)
    expected = Series([4578374827886788867, 17338122309987883691], dtype="uint64")
    tm.assert_series_equal(result, expected)
