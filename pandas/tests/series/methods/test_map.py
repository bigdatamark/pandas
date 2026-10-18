# [MODULE DOC]
# Tests for Series.map with callables, dicts, Series, defaultdicts, Counter,
# dict subclasses and abstract mappings as the mapping argument. Covers
# element boxing (Timestamp, Timedelta, Period), categorical/datetimetz/string
# dtypes, NA handling (na_action), type inference, extra keyword arguments,
# argument errors and execution engines. Imports the engine fixture from the
# apply tests' conftest to reuse it here. No import-time side effects.

from collections import (
    Counter,
    defaultdict,
)
from decimal import Decimal
import math

import numpy as np
import pytest

from pandas.errors import Pandas4Warning
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    bdate_range,
    date_range,
    isna,
    timedelta_range,
)
import pandas._testing as tm

# The fixture it's mostly used in pandas/tests/apply, so it's defined in that
# conftest, which is out of scope here. So we need to manually import
from pandas.tests.apply.conftest import engine  # noqa: F401


def test_series_map_box_timedelta():
    # [FUNCTION DOC]
    # Regression test (GH#11349): mapping a function that uses a timedelta method
    # over a timedelta Series works, so elements are boxed as timedelta objects. No
    # result is checked beyond the absence of an error.
    # GH#11349
    ser = Series(timedelta_range("1 day 1 s", periods=5, freq="h"))

    def f(x):
        # [FUNCTION DOC]
        # HELPER: callable passed to map in the boxing test; returns the total number of
        # seconds of the Timedelta argument.
        return x.total_seconds()

    ser.map(f)


def test_map_callable(datetime_series, engine):  # noqa: F811
    # [FUNCTION DOC]
    # With the engine fixture: mapping NumPy square root and the math exponential
    # function element-wise equals the vectorized NumPy function; mapping the
    # identity function over an empty named Series returns an equal but distinct
    # Series sharing the same index object and keeping dtype and name (GH 9322); and
    # an index-only float Series is unchanged.
    with np.errstate(all="ignore"):
        tm.assert_series_equal(
            datetime_series.map(np.sqrt, engine=engine), np.sqrt(datetime_series)
        )

    # map function element-wise
    tm.assert_series_equal(
        datetime_series.map(math.exp, engine=engine), np.exp(datetime_series)
    )

    # empty series
    s = Series(dtype=object, name="foo", index=Index([], name="bar"))
    rs = s.map(lambda x: x, engine=engine)
    tm.assert_series_equal(s, rs)

    # check all metadata (GH 9322)
    assert s is not rs
    assert s.index is rs.index
    assert s.dtype == rs.dtype
    assert s.name == rs.name

    # index but no data
    s = Series(index=[1, 2, 3], dtype=np.float64)
    rs = s.map(lambda x: x, engine=engine)
    tm.assert_series_equal(s, rs)


def test_map_same_length_inference_bug():
    # [FUNCTION DOC]
    # A function returning a two-tuple per element yields a Series of tuples (not
    # a frame), and splitting strings into tuples likewise; the inference does not
    # expand the tuple.
    # KNOWN ISSUE: the first Series assigned in the body is never used because it
    # is immediately reassigned.
    s = Series([1, 2])

    def f(x):
        # [FUNCTION DOC]
        # HELPER: callable for the inference test; returns a pair of the argument and the
        # argument plus one.
        return (x, x + 1)

    s = Series([1, 2, 3])
    result = s.map(f)
    expected = Series([(1, 2), (2, 3), (3, 4)])
    tm.assert_series_equal(result, expected)

    s = Series(["foo,bar"])
    result = s.map(lambda x: x.split(","))
    expected = Series([("foo", "bar")])
    tm.assert_series_equal(result, expected)


def test_series_map_box_timestamps():
    # [FUNCTION DOC]
    # Regression test (GH#2689, GH#2627): mapping a function over a datetime Series
    # passes Timestamp objects (hour, day and month attributes work).
    # GH#2689, GH#2627
    ser = Series(date_range("1/1/2000", periods=3))

    def func(x):
        # [FUNCTION DOC]
        # HELPER: callable returning a tuple of the hour, day and month of its Timestamp
        # argument.
        return (x.hour, x.day, x.month)

    result = ser.map(func)
    expected = Series([(0, 1, 1), (0, 2, 1), (0, 3, 1)])
    tm.assert_series_equal(result, expected)


def test_map_series_stringdtype(any_string_dtype, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over string dtypes (GH#40823). Mapping a Series of ids through a
    # Series keyed by ids returns the matching values; the unknown id gives a
    # missing value (NA or NaN depending on dtype); with the inferred-string option the
    # object dtype case yields the new string dtype.
    # map test on StringDType, GH#40823
    ser1 = Series(
        data=["cat", "dog", "rabbit"],
        index=["id1", "id2", "id3"],
        dtype=any_string_dtype,
    )
    ser2 = Series(["id3", "id2", "id1", "id7000"], dtype=any_string_dtype)
    result = ser2.map(ser1)

    item = pd.NA
    if ser2.dtype == object:
        item = np.nan

    expected = Series(data=["rabbit", "dog", "cat", item], dtype=any_string_dtype)
    if using_infer_string and any_string_dtype == "object":
        expected = expected.astype("str")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected_dtype",
    [(["1-1", "1-1", np.nan], "category"), (["1-1", "1-2", np.nan], "str")],
)
def test_map_categorical_with_nan_values(data, expected_dtype):
    # [FUNCTION DOC]
    # Parametrized over data and expected dtype. For a categorical with NaN, mapping
    # with na_action ignore keeps the NaN and applies the function (split off the
    # prefix) to others. If the mapped values are all the same the result stays
    # categorical; otherwise it is a string dtype (GH 20714, GH 24275).
    # GH 20714 bug fixed in: GH 24275
    def func(val):
        # [FUNCTION DOC]
        # HELPER: callable returning the text before the first hyphen in its argument.
        return val.split("-")[0]

    s = Series(data, dtype="category")

    result = s.map(func, na_action="ignore")
    expected = Series(["1", "1", np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_map_empty_integer_series():
    # [FUNCTION DOC]
    # Regression test (GH52384): mapping the identity over an empty integer Series
    # returns an equal Series.
    # GH52384
    s = Series([], dtype=int)
    result = s.map(lambda x: x)
    tm.assert_series_equal(result, s)


def test_map_empty_integer_series_with_datetime_index():
    # [FUNCTION DOC]
    # Regression test (GH 21245): same as above with an empty datetime index.
    # GH 21245
    s = Series([], index=date_range(start="2018-01-01", periods=0), dtype=int)
    result = s.map(lambda x: x)
    tm.assert_series_equal(result, s)


@pytest.mark.parametrize("func", [str, lambda x: str(x)])
def test_map_simple_str_callables_same_as_astype(
    string_series, func, using_infer_string
):
    # [FUNCTION DOC]
    # Parametrized over the string conversion type and a lambda doing the same.
    # Mapping a string conversion over a float Series equals casting to string
    # (or the new string dtype under the inferred-string option), i.e. the function
    # is evaluated element by element.
    # test that we are evaluating row-by-row first
    # before vectorized evaluation
    result = string_series.map(func)
    expected = string_series.astype(str if not using_infer_string else "str")
    tm.assert_series_equal(result, expected)


def test_list_raises(string_series):
    # [FUNCTION DOC]
    # Passing a list as the mapper raises a type error saying a list is not callable.
    with pytest.raises(TypeError, match="'list' object is not callable"):
        string_series.map([lambda x: x])


def test_map():
    # [FUNCTION DOC]
    # Mapping a Series of floats keyed by string labels through targets: each result
    # equals the source value for the matching key, whether the mapper is a Series
    # or a dict.
    data = {
        "A": [0.0, 1.0, 2.0, 3.0, 4.0],
        "B": [0.0, 1.0, 0.0, 1.0, 0.0],
        "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
        "D": bdate_range("1/1/2009", periods=5),
    }

    source = Series(data["B"], index=data["C"])
    target = Series(data["C"][:4], index=data["D"][:4])

    merged = target.map(source)

    for k, v in merged.items():
        assert v == source[target[k]]

    # input could be a dict
    merged = target.map(source.to_dict())

    for k, v in merged.items():
        assert v == source[target[k]]


def test_map_datetime(datetime_series):
    # [FUNCTION DOC]
    # Mapping a doubling function over the datetime_series fixture equals doubling
    # the Series.
    # function
    result = datetime_series.map(lambda x: x * 2)
    tm.assert_series_equal(result, datetime_series * 2)


def test_map_category():
    # [FUNCTION DOC]
    # Regression test (GH 10324): mapping integers through a categorical Series
    # gives a categorical result with NaN for unknown keys; through a plain string
    # Series gives the same values as non-categorical.
    # GH 10324
    a = Series([1, 2, 3, 4])
    b = Series(["even", "odd", "even", "odd"], dtype="category")
    c = Series(["even", "odd", "even", "odd"])

    exp = Series(["odd", "even", "odd", np.nan], dtype="category")
    tm.assert_series_equal(a.map(b), exp)
    exp = Series(["odd", "even", "odd", np.nan])
    tm.assert_series_equal(a.map(c), exp)


def test_map_category_numeric():
    # [FUNCTION DOC]
    # Mapping strings through Series keyed by a categorical index or a plain index
    # gives the same numbers with NaN for unknown labels.
    a = Series(["a", "b", "c", "d"])
    b = Series([1, 2, 3, 4], index=pd.CategoricalIndex(["b", "c", "d", "e"]))
    c = Series([1, 2, 3, 4], index=Index(["b", "c", "d", "e"]))

    exp = Series([np.nan, 1, 2, 3])
    tm.assert_series_equal(a.map(b), exp)
    exp = Series([np.nan, 1, 2, 3])
    tm.assert_series_equal(a.map(c), exp)


def test_map_category_string():
    # [FUNCTION DOC]
    # Mapping strings through a categorical Series (with categorical index) gives a
    # categorical result with all four categories, and through a plain string Series
    # gives plain strings; unknown labels become NaN.
    a = Series(["a", "b", "c", "d"])
    b = Series(
        ["B", "C", "D", "E"],
        dtype="category",
        index=pd.CategoricalIndex(["b", "c", "d", "e"]),
    )
    c = Series(["B", "C", "D", "E"], index=Index(["b", "c", "d", "e"]))

    exp = Series(
        pd.Categorical([np.nan, "B", "C", "D"], categories=["B", "C", "D", "E"])
    )
    tm.assert_series_equal(a.map(b), exp)
    exp = Series([np.nan, "B", "C", "D"])
    tm.assert_series_equal(a.map(c), exp)


def test_map_empty(request, index):
    # [FUNCTION DOC]
    # Uses the index fixture: mapping an empty dict over a Series built from the
    # index gives all NaN with the same index. A MultiIndex is an expected failure
    # because a Series cannot be built from it.
    if isinstance(index, MultiIndex):
        request.applymarker(
            pytest.mark.xfail(
                reason="Initializing a Series from a MultiIndex is not supported"
            )
        )

    s = Series(index)
    result = s.map({})

    expected = Series(np.nan, index=s.index)
    tm.assert_series_equal(result, expected)


def test_map_compat():
    # [FUNCTION DOC]
    # Related GH 8024: boolean values mapped through a dict of booleans to strings.
    # related GH 8024
    s = Series([True, True, False], index=[1, 2, 3])
    result = s.map({True: "foo", False: "bar"})
    expected = Series(["foo", "foo", "bar"], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)


def test_map_int():
    # [FUNCTION DOC]
    # Float values mapped through an integer-valued Series keyed by integers give
    # float results with NaN for values with no matching key.
    left = Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4})
    right = Series({1: 11, 2: 22, 3: 33})

    assert left.dtype == np.float64
    assert issubclass(right.dtype.type, np.integer)

    merged = left.map(right)
    assert merged.dtype == np.float64
    assert isna(merged["d"])
    assert not isna(merged["c"])


def test_map_type_inference():
    # [FUNCTION DOC]
    # Mapping a function that returns an array (a NumPy conditional) over an integer
    # Series still gives an integer dtype.
    s = Series(range(3))
    s2 = s.map(lambda x: np.where(x == 0, 0, 1))
    assert issubclass(s2.dtype.type, np.integer)


def test_map_decimal(string_series):
    # [FUNCTION DOC]
    # Mapping a function that returns Decimal objects gives object dtype with
    # Decimal elements.
    result = string_series.map(lambda x: Decimal(str(x)))
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)


def test_map_na_exclusion(engine):  # noqa: F811
    # [FUNCTION DOC]
    # With the engine fixture and na_action ignore, the doubling function is applied
    # to non-missing values and NaN stays NaN.
    s = Series([1.5, np.nan, 3, np.nan, 5])

    result = s.map(lambda x: x * 2, na_action="ignore", engine=engine)
    exp = s * 2
    tm.assert_series_equal(result, exp)


def test_map_dict_with_tuple_keys():
    # [FUNCTION DOC]
    # GH 18496: a dict with tuple keys passed to map is used as a plain mapping, so
    # tuple values in the Series are mapped to the labels (not interpreted as a
    # MultiIndex).
    """
    Due to new MultiIndex-ing behaviour in v0.14.0,
    dicts with tuple keys passed to map were being
    converted to a multi-index, preventing tuple values
    from being mapped properly.
    """
    # GH 18496
    df = DataFrame({"a": [(1,), (2,), (3, 4), (5, 6)]})
    label_mappings = {(1,): "A", (2,): "B", (3, 4): "A", (5, 6): "B"}

    df["labels"] = df["a"].map(label_mappings)
    df["expected_labels"] = Series(["A", "B", "A", "B"], index=df.index)
    # All labels should be filled now
    tm.assert_series_equal(df["labels"], df["expected_labels"], check_names=False)


def test_map_counter():
    # [FUNCTION DOC]
    # A Counter used as the mapper returns the stored count and zero for keys not
    # present (Counter default behavior).
    s = Series(["a", "b", "c"], index=[1, 2, 3])
    counter = Counter()
    counter["b"] = 5
    counter["c"] += 1
    result = s.map(counter)
    expected = Series([0, 5, 1], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)


def test_map_defaultdict():
    # [FUNCTION DOC]
    # A defaultdict used as the mapper uses its default factory for unknown keys
    # (key 1 maps to its stored value, others to "blank").
    s = Series([1, 2, 3], index=["a", "b", "c"])
    default_dict = defaultdict(lambda: "blank")
    default_dict[1] = "stuff"
    result = s.map(default_dict)
    expected = Series(["stuff", "blank", "blank"], index=["a", "b", "c"])
    tm.assert_series_equal(result, expected)


def test_map_dict_na_key():
    # [FUNCTION DOC]
    # Issue 17648: a dict containing NaN as a key maps NaN values in the Series.
    # https://github.com/pandas-dev/pandas/issues/17648
    # Checks that np.nan key is appropriately mapped
    s = Series([1, 2, np.nan])
    expected = Series(["a", "b", "c"])
    result = s.map({1: "a", 2: "b", np.nan: "c"})
    tm.assert_series_equal(result, expected)


def test_map_defaultdict_na_key(na_action):
    # [FUNCTION DOC]
    # GH 48813, parametrized over na_action (fixture): a defaultdict containing a NaN
    # key maps NaN to its value when na_action is none, and leaves NaN when ignore.
    # GH 48813
    s = Series([1, 2, np.nan])
    default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", np.nan: "c"})
    result = s.map(default_map, na_action=na_action)
    expected = Series({0: "a", 1: "b", 2: "c" if na_action is None else np.nan})
    tm.assert_series_equal(result, expected)


def test_map_defaultdict_missing_key(na_action):
    # [FUNCTION DOC]
    # GH 48813, parametrized over na_action: a defaultdict without a NaN key uses its
    # default factory for NaN when na_action is none, and leaves NaN when ignore.
    # GH 48813
    s = Series([1, 2, np.nan])
    default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", 3: "c"})
    result = s.map(default_map, na_action=na_action)
    expected = Series({0: "a", 1: "b", 2: "missing" if na_action is None else np.nan})
    tm.assert_series_equal(result, expected)


def test_map_defaultdict_unmutated(na_action):
    # [FUNCTION DOC]
    # GH 48813, parametrized over na_action: mapping with a defaultdict does not add
    # any keys to the defaultdict (stays equal to its copy).
    # GH 48813
    s = Series([1, 2, np.nan])
    default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", np.nan: "c"})
    expected_default_map = default_map.copy()
    s.map(default_map, na_action=na_action)
    assert default_map == expected_default_map


@pytest.mark.parametrize("arg_func", [dict, Series])
def test_map_dict_ignore_na(arg_func):
    # [FUNCTION DOC]
    # GH#47527, parametrized over dict or Series as the mapper: with na_action ignore
    # the NaN key in the mapper is not used; the NaN stays NaN and unmatched values
    # become NaN.
    # GH#47527
    mapping = arg_func({1: 10, np.nan: 42})
    ser = Series([1, np.nan, 2])
    result = ser.map(mapping, na_action="ignore")
    expected = Series([10, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_map_defaultdict_ignore_na():
    # [FUNCTION DOC]
    # GH#47527: a defaultdict(int) with a NaN key and no na_action: NaN is mapped via
    # its key, and missing values use the default zero.
    # GH#47527
    mapping = defaultdict(int, {1: 10, np.nan: 42})
    ser = Series([1, np.nan, 2])
    result = ser.map(mapping)
    expected = Series([10, 42, 0])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "na_action, expected",
    [(None, Series([10.0, 42.0, np.nan])), ("ignore", Series([10, np.nan, np.nan]))],
)
def test_map_categorical_na_ignore(na_action, expected):
    # [FUNCTION DOC]
    # GH#47527, parametrized over na_action and the expected result: mapping a
    # categorical Series (with NaN) through a dict with a NaN key. With no
    # na_action the NaN gets 42 and the result is float; with ignore the NaN stays
    # NaN.
    # GH#47527
    values = pd.Categorical([1, np.nan, 2], categories=[10, 1, 2])
    ser = Series(values)
    result = ser.map({1: 10, np.nan: 42}, na_action=na_action)
    tm.assert_series_equal(result, expected)


def test_map_dict_subclass_with_missing():
    # [FUNCTION DOC]
    # GH #15999: a dict subclass defining a missing-key hook gives its value for
    # unknown keys when used as the mapper.
    """
    Test Series.map with a dictionary subclass that defines __missing__,
    i.e. sets a default value (GH #15999).
    """

    class DictWithMissing(dict):
        # [CLASS DOC]
        # HELPER: local dict subclass whose missing-key hook returns the string "missing".
        def __missing__(self, key):
            # [FUNCTION DOC]
            # HELPER: missing-key hook; takes the absent key and returns the fixed string
            # "missing".
            return "missing"

    s = Series([1, 2, 3])
    dictionary = DictWithMissing({3: "three"})
    result = s.map(dictionary)
    expected = Series(["missing", "missing", "three"])
    tm.assert_series_equal(result, expected)


def test_map_dict_subclass_without_missing():
    # [FUNCTION DOC]
    # A plain dict subclass (no missing-key hook) behaves like a dict: unknown keys
    # map to NaN.
    class DictWithoutMissing(dict):
        # [CLASS DOC]
        # HELPER: local dict subclass that adds nothing.
        pass

    s = Series([1, 2, 3])
    dictionary = DictWithoutMissing({3: "three"})
    result = s.map(dictionary)
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_abc_mapping(non_dict_mapping_subclass):
    # [FUNCTION DOC]
    # GitHub issue 29733: a non-dict mapping (from the non_dict_mapping_subclass
    # fixture) can be used as the mapper; unknown keys give NaN.
    # https://github.com/pandas-dev/pandas/issues/29733
    # Check collections.abc.Mapping support as mapper for Series.map
    s = Series([1, 2, 3])
    not_a_dictionary = non_dict_mapping_subclass({3: "three"})
    result = s.map(not_a_dictionary)
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_abc_mapping_with_missing(non_dict_mapping_subclass):
    # [FUNCTION DOC]
    # GitHub issue 29733: a non-dict mapping subclass defining a missing-key hook
    # does not use it: the hook is a dict concept, so unknown keys still give NaN.
    # https://github.com/pandas-dev/pandas/issues/29733
    # Check collections.abc.Mapping support as mapper for Series.map
    class NonDictMappingWithMissing(non_dict_mapping_subclass):
        # [CLASS DOC]
        # HELPER: local subclass of the fixture's mapping type that adds a missing-key
        # hook returning "missing".
        def __missing__(self, key):
            # [FUNCTION DOC]
            # HELPER: missing-key hook returning the fixed string "missing" for any key.
            return "missing"

    s = Series([1, 2, 3])
    not_a_dictionary = NonDictMappingWithMissing({3: "three"})
    result = s.map(not_a_dictionary)
    # __missing__ is a dict concept, not a Mapping concept,
    # so it should not change the result!
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_box_dt64(unit):
    # [FUNCTION DOC]
    # Parametrized over datetime unit. The function receives Timestamp objects and
    # no timezone; the formatted text results confirm class name, day and tz.
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f"datetime64[{unit}]"
    # boxed value must be Timestamp instance
    res = ser.map(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    tm.assert_series_equal(res, exp)


def test_map_box_dt64tz(unit):
    # [FUNCTION DOC]
    # Parametrized over datetime unit. For a tz-aware Series the function receives
    # Timestamp objects carrying the US/Eastern timezone.
    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f"datetime64[{unit}, US/Eastern]"
    res = ser.map(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    tm.assert_series_equal(res, exp)


def test_map_box_td64(unit):
    # [FUNCTION DOC]
    # Parametrized over timedelta unit. The function receives Timedelta objects (class
    # name and day count are checked).
    # timedelta
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f"timedelta64[{unit}]"
    res = ser.map(lambda x: f"{type(x).__name__}_{x.days}")
    exp = Series(["Timedelta_1", "Timedelta_2"])
    tm.assert_series_equal(res, exp)


def test_map_box_period():
    # [FUNCTION DOC]
    # The function receives Period objects (class name and frequency string checked).
    # period
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    ser = Series(vals)
    assert ser.dtype == "Period[M]"
    res = ser.map(lambda x: f"{type(x).__name__}_{x.freqstr}")
    exp = Series(["Period_M", "Period_M"])
    tm.assert_series_equal(res, exp)


def test_map_categorical(na_action, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over na_action and inferred-string option. Mapping a function over
    # an ordered categorical returns a categorical with mapped categories (order
    # preserved); mapping to a constant string gives an object (or str) Series.
    # KNOWN ISSUE: the final assertion has a conditional expression whose
    # else branch is a non-empty string, so under the inferred-string option the
    # assertion is always true and verifies nothing.
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    s = Series(values, name="XX", index=list("abcdefg"))

    result = s.map(lambda x: x.lower(), na_action=na_action)
    exp_values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    exp = Series(exp_values, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp_values)

    result = s.map(lambda x: "A", na_action=na_action)
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    assert result.dtype == object if not using_infer_string else "str"


@pytest.mark.parametrize(
    "na_action, expected",
    (
        [None, Series(["A", "B", "nan"], name="XX")],
        [
            "ignore",
            Series(
                ["A", "B", np.nan],
                name="XX",
                dtype=pd.CategoricalDtype(list("DCBA"), True),
            ),
        ],
    ),
)
def test_map_categorical_na_action(na_action, expected):
    # [FUNCTION DOC]
    # Parametrized over na_action (none or ignore). Mapping the string conversion
    # over an ordered categorical with NaN: with no na_action the NaN becomes the
    # text "nan"; with ignore, NaN stays and the result is categorical with the
    # original categories.
    dtype = pd.CategoricalDtype(list("DCBA"), ordered=True)
    values = pd.Categorical([*list("AB"), np.nan], dtype=dtype)
    s = Series(values, name="XX")
    result = s.map(str, na_action=na_action)
    tm.assert_series_equal(result, expected)


def test_map_datetimetz():
    # [FUNCTION DOC]
    # A tz-aware (Asia/Tokyo) Series: mapping adding a day keeps the timezone;
    # mapping to the hour attribute gives int64; a function that raises unless it is
    # given a Timestamp shows that elements are boxed (not vectorized) and returns
    # the zone name.
    values = date_range("2011-01-01", "2011-01-02", freq="h").tz_localize("Asia/Tokyo")
    s = Series(values, name="XX")

    # keep tz
    result = s.map(lambda x: x + pd.offsets.Day())
    exp_values = date_range("2011-01-02", "2011-01-03", freq="h").tz_localize(
        "Asia/Tokyo"
    )
    exp = Series(exp_values, name="XX")
    tm.assert_series_equal(result, exp)

    result = s.map(lambda x: x.hour)
    exp = Series([*list(range(24)), 0], name="XX", dtype=np.int64)
    tm.assert_series_equal(result, exp)

    # not vectorized
    def f(x):
        # [FUNCTION DOC]
        # HELPER: callable requiring a Timestamp argument (raises a value error
        # otherwise) and returning the text of its timezone.
        if not isinstance(x, pd.Timestamp):
            raise ValueError
        return str(x.tz)

    result = s.map(f)
    exp = Series(["Asia/Tokyo"] * 25, name="XX")
    tm.assert_series_equal(result, exp)


@pytest.mark.parametrize(
    "vals,mapping,exp",
    [
        (list("abc"), {np.nan: "not NaN"}, [np.nan] * 3 + ["not NaN"]),
        (list("abc"), {"a": "a letter"}, ["a letter"] + [np.nan] * 3),
        (list(range(3)), {0: 42}, [42] + [np.nan] * 3),
    ],
)
def test_map_missing_mixed(vals, mapping, exp):
    # [FUNCTION DOC]
    # GH20495, parametrized over (values, mapping, expected): mapping dicts with
    # missing values in the Series; only matching keys (including a NaN key) are
    # mapped, other values become NaN.
    # GH20495
    s = Series([*vals, np.nan])
    result = s.map(mapping)
    exp = Series(exp)
    tm.assert_series_equal(result, exp)


def test_map_scalar_on_date_time_index_aware_series():
    # [FUNCTION DOC]
    # GH 25959: mapping a function on a tz-aware (UTC) DatetimeIndex wrapped in a
    # Series does not raise and gives integer ones.
    # GH 25959
    # Calling map on a localized time series should not cause an error
    series = Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10, tz="UTC"),
        name="ts",
    )
    result = Series(series.index).map(lambda x: 1)
    tm.assert_series_equal(result, Series(np.ones(len(series)), dtype="int64"))


def test_map_float_to_string_precision():
    # [FUNCTION DOC]
    # GH 13228: mapping the string conversion over a float keeps the full repr
    # precision (0.3333333333333333).
    # GH 13228
    ser = Series(1 / 3)
    result = ser.map(lambda val: str(val)).to_dict()
    expected = {0: "0.3333333333333333"}
    assert result == expected


def test_map_to_timedelta():
    # [FUNCTION DOC]
    # Mapping the timedelta parser over strings (valid, then with NaN and NaT)
    # matches parsing the whole list.
    list_of_valid_strings = ["00:00:01", "00:00:02"]
    a = pd.to_timedelta(list_of_valid_strings)
    b = Series(list_of_valid_strings).map(pd.to_timedelta)
    tm.assert_series_equal(Series(a), b)

    list_of_strings = ["00:00:01", np.nan, pd.NaT, pd.NaT]

    a = pd.to_timedelta(list_of_strings)
    ser = Series(list_of_strings)
    b = ser.map(pd.to_timedelta)
    tm.assert_series_equal(Series(a), b)


def test_map_type():
    # [FUNCTION DOC]
    # GH 46719: mapping the built-in type function over mixed objects returns the
    # classes.
    # GH 46719
    s = Series([3, "string", float], index=["a", "b", "c"])
    result = s.map(type)
    expected = Series([int, str, type], index=["a", "b", "c"])
    tm.assert_series_equal(result, expected)


def test_map_kwargs():
    # [FUNCTION DOC]
    # GH 59814: extra keyword arguments are forwarded to the function.
    # GH 59814
    result = Series([2, 4, 5]).map(lambda x, y: x + y, y=2)
    expected = Series([4, 6, 7])
    tm.assert_series_equal(result, expected)


def test_map_arg_as_kwarg():
    # [FUNCTION DOC]
    # Using the old keyword name for the first argument emits a Pandas4Warning that
    # the argument has been renamed to func.
    with tm.assert_produces_warning(
        Pandas4Warning, match="`arg` has been renamed to `func`"
    ):
        Series([1, 2]).map(arg={})


def test_map_func_and_arg():
    # [FUNCTION DOC]
    # When func is a callable, a keyword named arg is passed through to it as a normal
    # keyword argument.
    # `arg`is considered a normal kwarg that should be passed to the function
    result = Series([1, 2]).map(lambda _, arg: arg, arg=3)
    expected = Series([3, 3])
    tm.assert_series_equal(result, expected)


def test_map_no_func_or_arg():
    # [FUNCTION DOC]
    # Calling map without any func raises a value error that the func parameter is
    # required.
    with pytest.raises(ValueError, match="The `func` parameter is required"):
        Series([1, 2]).map()


def test_map_func_is_none():
    # [FUNCTION DOC]
    # Passing None as func raises a value error that the func parameter is required.
    with pytest.raises(ValueError, match="The `func` parameter is required"):
        Series([1, 2]).map(func=None)


@pytest.mark.parametrize("func", [{}, {1: 2}, Series([3, 4])])
def test_map_engine_no_function(func):
    # [FUNCTION DOC]
    # Parametrized over dict, dict and Series mappers: specifying an engine with a
    # non-callable mapper raises a value error that the engine argument can only be
    # specified with a callable.
    s = Series([1, 2])

    with pytest.raises(ValueError, match="engine argument can only be specified"):
        s.map(func, engine="something")


def test_map_engine_not_executor():
    # [FUNCTION DOC]
    # An unknown engine name raises a value error stating it is not a valid engine.
    s = Series([1, 2])

    with pytest.raises(ValueError, match="Not a valid engine: 'something'"):
        s.map(lambda x: x, engine="something")


@td.skip_if_no("pyarrow")
@pytest.mark.parametrize("as_td", [True, False])
def test_map_pyarrow_timestamp(as_td):
    # [FUNCTION DOC]
    # GH#61231, needs pyarrow; parametrized over timestamps or durations. Maps
    # pyarrow timestamp/duration Series values through a dict keyed by the elements.
    # The Series result is int64; the Index result is converted to pyarrow int64
    # (Index inference differs from Series).
    # GH#61231
    dti = date_range("2018-01-01 00:00:00", "2018-01-07 00:00:00")
    ser = Series(dti, dtype="timestamp[ns][pyarrow]", name="a")
    if as_td:
        # duration dtype
        ser = ser - ser[0]

    mapper = {date: i for i, date in enumerate(ser)}

    res_series = ser.map(mapper)
    expected = Series(range(len(ser)), name="a", dtype="int64")
    tm.assert_series_equal(res_series, expected)

    res_index = Index(ser).map(mapper)
    # For now (as of 2025-09-06) at least, we do inference on Index.map that
    #  we don't for Series.map
    expected_index = Index(expected).astype("int64[pyarrow]")
    tm.assert_index_equal(res_index, expected_index)
