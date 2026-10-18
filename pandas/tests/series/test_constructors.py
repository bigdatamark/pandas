# [MODULE DOC]
# Tests for the Series constructor: inference and explicit dtypes from lists, tuples, arrays, dicts, generators, scalars, other Series, Index, categoricals, masked arrays, datetimes/timedeltas/periods/intervals with units and time zones, extension and nullable dtypes, string and pyarrow-backed data, copying and view semantics, name and index handling, and error cases and deprecation warnings.
# Contains three test classes (general constructor behavior, index coercion, and internal block structure) followed by a few module-level tests. Imports the internal numpy block class and several array types from the internals for identity checks. No import-time side effects.

from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
    datetime,
    timedelta,
)

from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest

from pandas._libs import (
    iNaT,
    lib,
)
from pandas.compat import HAS_PYARROW
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
    IntCastingNaNError,
    Pandas4Warning,
)

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    DatetimeTZDtype,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    IntegerArray,
    IntervalArray,
    period_array,
)
from pandas.core.internals.blocks import NumpyBlock


class TestSeriesConstructors:
    # [CLASS DOC]
    # Groups the general Series constructor tests. Stateless; fixtures supply Series/Index boxes, string dtypes, extension scalars and the infer-string flag.
    def test_from_ints_with_non_nano_dt64_dtype(self, index_or_series):
        # [FUNCTION DOC]
        # Verifies that building a Series or Index (fixture) from integer values or an integer list with a non-nanosecond second-resolution datetime dtype interprets the integers as seconds since the epoch, giving the same result as casting an integer array to that dtype.
        values = np.arange(10)

        res = index_or_series(values, dtype="M8[s]")
        expected = index_or_series(values.astype("M8[s]"))
        tm.assert_equal(res, expected)

        res = index_or_series(list(values), dtype="M8[s]")
        tm.assert_equal(res, expected)

    def test_from_na_value_and_interval_of_datetime_dtype(self):
        # [FUNCTION DOC]
        # Verifies (issue 41805) that a Series of a single None with an interval-of-datetime dtype is all missing and has the dtype name with closed-right interval of nanosecond datetimes.
        # GH#41805
        ser = Series([None], dtype="interval[datetime64[ns]]")
        assert ser.isna().all()
        assert ser.dtype == "interval[datetime64[ns], right]"

    def test_infer_with_date_and_datetime(self):
        # [FUNCTION DOC]
        # Verifies (issue 49341) that a list holding a datetime and a date infers object dtype for both Series and Index, instead of being inferred as datetimes as before version 2.0.
        # GH#49341 pre-2.0 we inferred datetime-and-date to datetime64, which
        #  was inconsistent with Index behavior
        ts = Timestamp(2016, 1, 1)
        vals = [ts.to_pydatetime(), ts.date()]

        ser = Series(vals)
        expected = Series(vals, dtype=object)
        tm.assert_series_equal(ser, expected)

        idx = Index(vals)
        expected = Index(vals, dtype=object)
        tm.assert_index_equal(idx, expected)

    def test_unparsable_strings_with_dt64_dtype(self):
        # [FUNCTION DOC]
        # Verifies that a list or object array of strings that cannot be parsed, with datetime64 nanosecond dtype requested, raises a value error naming the unknown format and the string.
        # pre-2.0 these would be silently ignored and come back with object dtype
        vals = ["aa"]
        msg = "^Unknown datetime string format, unable to parse: aa$"
        with pytest.raises(ValueError, match=msg):
            Series(vals, dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            Series(np.array(vals, dtype=object), dtype="datetime64[ns]")

    def test_invalid_dtype_conversion_datetime_to_timedelta(self):
        # [FUNCTION DOC]
        # Verifies (issue 60728) that building a timedelta64 Series from a datetime64 Series (with a missing value) raises a type error saying the datetime array cannot be cast to the timedelta dtype.
        # GH#60728
        vals = Series([NaT, Timestamp(2025, 1, 1)], dtype="datetime64[ns]")
        msg = r"^Cannot cast DatetimeArray to dtype timedelta64\[ns\]$"
        with pytest.raises(TypeError, match=msg):
            Series(vals, dtype="timedelta64[ns]")

    @pytest.mark.parametrize(
        "constructor",
        [
            # NOTE: some overlap with test_constructor_empty but that test does not
            # test for None or an empty generator.
            # test_constructor_pass_none tests None but only with the index also
            # passed.
            (lambda idx: Series(index=idx)),
            (lambda idx: Series(None, index=idx)),
            (lambda idx: Series({}, index=idx)),
            (lambda idx: Series((), index=idx)),
            (lambda idx: Series([], index=idx)),
            (lambda idx: Series((_ for _ in []), index=idx)),
            (lambda idx: Series(data=None, index=idx)),
            (lambda idx: Series(data={}, index=idx)),
            (lambda idx: Series(data=(), index=idx)),
            (lambda idx: Series(data=[], index=idx)),
            (lambda idx: Series(data=(_ for _ in []), index=idx)),
        ],
    )
    @pytest.mark.parametrize("empty_index", [None, []])
    def test_empty_constructor(self, constructor, empty_index):
        # [FUNCTION DOC]
        # Verifies that many equivalent spellings of empty input (no data, none, empty dict, tuple, list, generator, positional or keyword) combined with no index or an empty list index all give an empty object-dtype Series equal to the one built with just the index (issue 49573 added the empty-index parameter). Parametrized over constructor spellings and empty index; the spellings are anonymous functions.
        # GH 49573 (addition of empty_index parameter)
        expected = Series(index=empty_index)
        result = constructor(empty_index)

        assert result.dtype == object
        assert len(result.index) == 0
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_invalid_dtype(self):
        # [FUNCTION DOC]
        # Verifies (issue 15520) that passing a class, a name that is not a known dtype, or a list type as dtype raises a type error saying the dtype is not understood.
        # GH15520
        msg = "not understood"
        invalid_list = [Timestamp, "Timestamp", list]
        for dtype in invalid_list:
            with pytest.raises(TypeError, match=msg):
                Series([], name="time", dtype=dtype)

    def test_invalid_compound_dtype(self):
        # [FUNCTION DOC]
        # Verifies (issue 13296) that building a Series from a structured (compound) array raises a value error telling the user to use a DataFrame.
        # GH#13296
        c_dtype = np.dtype([("a", "i8"), ("b", "f4")])
        cdt_arr = np.array([(1, 0.4), (256, -13)], dtype=c_dtype)

        with pytest.raises(ValueError, match="Use DataFrame instead"):
            Series(cdt_arr, index=["A", "B"])

    def test_scalar_conversion(self):
        # [FUNCTION DOC]
        # Verifies that building from a scalar gives a Series rather than a plain float.
        # Pass in scalar is disabled
        scalar = Series(0.5)
        assert not isinstance(scalar, float)

    def test_scalar_extension_dtype(self, ea_scalar_and_dtype):
        # [FUNCTION DOC]
        # Verifies (issue 28401) that broadcasting an extension-dtype scalar over a range index gives a Series with the matching extension dtype equal to the one built from a list of the scalar repeated. Uses a fixture providing scalar and dtype pairs.
        # GH 28401

        ea_scalar, ea_dtype = ea_scalar_and_dtype

        ser = Series(ea_scalar, index=range(3))
        expected = Series([ea_scalar] * 3, dtype=ea_dtype)

        assert ser.dtype == ea_dtype
        tm.assert_series_equal(ser, expected)

    def test_constructor(self, datetime_series, using_infer_string):
        # [FUNCTION DOC]
        # Verifies basic constructor behavior: the datetime fixture index is all dates; building from a Series reuses the same index object; a list with a string and NaN gives object dtype (the string dtype when infer-string is on) with NaN preserved; empty Series have non-date indexes; a two-dimensional array raises a value error stating that data must be one-dimensional (issue 35744); the name of the source Series is kept; and building from a multi-index raises a not-implemented error (issue 4187).
        # KNOWN ISSUE: the dtype assertion for the mixed Series combines a comparison with a conditional expression without parentheses, so when the infer-string flag is on it asserts a non-empty string constant and always passes.
        empty_series = Series()
        assert datetime_series.index._is_all_dates

        # Pass in Series
        derived = Series(datetime_series)
        assert derived.index._is_all_dates

        tm.assert_index_equal(derived.index, datetime_series.index)
        # Ensure new index is not created
        assert id(datetime_series.index) == id(derived.index)

        # Mixed type Series
        mixed = Series(["hello", np.nan], index=[0, 1])
        assert mixed.dtype == np.object_ if not using_infer_string else "str"
        assert np.isnan(mixed[1])

        assert not empty_series.index._is_all_dates
        assert not Series().index._is_all_dates

        # exception raised is of type ValueError GH35744
        with pytest.raises(
            ValueError,
            match=r"Data must be 1-dimensional, got ndarray of shape \(3, 3\) instead",
        ):
            Series(np.random.default_rng(2).standard_normal((3, 3)), index=np.arange(3))

        mixed.name = "Series"
        rs = Series(mixed).name
        xp = "Series"
        assert rs == xp

        # raise on MultiIndex GH4187
        m = MultiIndex.from_arrays([[1, 2], [3, 4]])
        msg = "initializing a Series from a MultiIndex is not supported"
        with pytest.raises(NotImplementedError, match=msg):
            Series(m)

    def test_constructor_index_ndim_gt_1_raises(self):
        # [FUNCTION DOC]
        # Verifies (issue 18579) that passing a DataFrame as the index raises a value error that the index data must be one-dimensional.
        # GH#18579
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=[3, 6, 9])
        with pytest.raises(ValueError, match="Index data must be 1-dimensional"):
            Series([1, 3, 2], index=df)

    @pytest.mark.parametrize("input_class", [list, dict, OrderedDict])
    def test_constructor_empty(self, input_class, using_infer_string):
        # [FUNCTION DOC]
        # Verifies empty construction from a list, dict or ordered dict (parametrized): no data equals empty container data for default, float64 and category dtype (issue 18515), index-only and index-with-NaN float equivalents for dict types, and (issue 19853) an empty string scalar with string dtype and index of three, using the string dtype name when infer-string is on.
        empty = Series()
        empty2 = Series(input_class())

        # these are Index() and RangeIndex() which don't compare type equal
        # but are just .equals
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        # With explicit dtype:
        empty = Series(dtype="float64")
        empty2 = Series(input_class(), dtype="float64")
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        # GH 18515 : with dtype=category:
        empty = Series(dtype="category")
        empty2 = Series(input_class(), dtype="category")
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        if input_class is not list:
            # With index:
            empty = Series(index=range(10))
            empty2 = Series(input_class(), index=range(10))
            tm.assert_series_equal(empty, empty2)

            # With index and dtype float64:
            empty = Series(np.nan, index=range(10))
            empty2 = Series(input_class(), index=range(10), dtype="float64")
            tm.assert_series_equal(empty, empty2)

            # GH 19853 : with empty string, index and dtype str
            empty = Series("", dtype=str, index=range(3))
            if using_infer_string:
                empty2 = Series("", index=range(3), dtype="str")
            else:
                empty2 = Series("", index=range(3))
            tm.assert_series_equal(empty, empty2)

    @pytest.mark.parametrize("input_arg", [np.nan, float("nan")])
    def test_constructor_nan(self, input_arg):
        # [FUNCTION DOC]
        # Verifies that a NaN scalar (NumPy or built-in) with an index equals an empty float64 Series with that index.
        empty = Series(dtype="float64", index=range(10))
        empty2 = Series(input_arg, index=range(10))

        tm.assert_series_equal(empty, empty2, check_index_type=False)

    @pytest.mark.parametrize(
        "dtype",
        ["f8", "i8", "M8[ns]", "m8[ns]", "category", "object", "datetime64[ns, UTC]"],
    )
    @pytest.mark.parametrize("index", [None, Index([])])
    def test_constructor_dtype_only(self, dtype, index):
        # [FUNCTION DOC]
        # Verifies (issue 20865) that giving only a dtype (float, int, datetime, timedelta, category, object, tz-aware datetime) with no index or an empty index yields an empty Series of that dtype.
        # GH-20865
        result = Series(dtype=dtype, index=index)
        assert result.dtype == dtype
        assert len(result) == 0

    def test_constructor_no_data_index_order(self):
        # [FUNCTION DOC]
        # Verifies that constructing with only an index keeps the index order given.
        result = Series(index=["b", "a", "c"])
        assert result.index.tolist() == ["b", "a", "c"]

    def test_constructor_no_data_string_type(self):
        # [FUNCTION DOC]
        # Verifies (issue 22477) that constructing with only an index and the str dtype yields missing values.
        # GH 22477
        result = Series(index=[1], dtype=str)
        assert np.isnan(result.iloc[0])

    @pytest.mark.parametrize("item", ["entry", "ѐ", 13])
    def test_constructor_string_element_string_type(self, item):
        # [FUNCTION DOC]
        # Verifies (issue 22477) that a scalar (string, non-ASCII string or integer) broadcast with the str dtype is converted to its text form.
        # GH 22477
        result = Series(item, index=[1], dtype=str)
        assert result.iloc[0] == str(item)

    def test_constructor_dtype_str_na_values(self, string_dtype):
        # [FUNCTION DOC]
        # Verifies (issue 21083) that with string dtype (fixture), None and NaN in the input are treated as missing and keep their identity: None stays None and NaN stays NaN.
        # https://github.com/pandas-dev/pandas/issues/21083
        ser = Series(["x", None], dtype=string_dtype)
        result = ser.isna()
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)
        assert ser.iloc[1] is None

        ser = Series(["x", np.nan], dtype=string_dtype)
        assert np.isnan(ser.iloc[1])

    def test_constructor_series(self):
        # [FUNCTION DOC]
        # Verifies that building from a Series with a differently ordered index reindexes it, equal to the sorted original.
        index1 = ["d", "b", "a", "c"]
        index2 = sorted(index1)
        s1 = Series([4, 7, -5, 3], index=index1)
        s2 = Series(s1, index=index2)

        tm.assert_series_equal(s2, s1.sort_index())

    def test_constructor_iterable(self):
        # [FUNCTION DOC]
        # Verifies (issue 21987) that an object that only defines iteration (a local class yielding ten integers) is accepted as data and gives an int64 Series.
        # GH 21987
        class Iter:
            # [CLASS DOC]
            # HELPER: local iterable class yielding the integers zero to nine, used to test construction from a generic iterable.
            def __iter__(self) -> Iterator:
                # [FUNCTION DOC]
                # HELPER: generator yielding each integer from zero up to nine.
                yield from range(10)

        expected = Series(list(range(10)), dtype="int64")
        result = Series(Iter(), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_sequence(self):
        # [FUNCTION DOC]
        # Verifies (issue 21987) construction from a range with int64 dtype equals construction from a list.
        # GH 21987
        expected = Series(list(range(10)), dtype="int64")
        result = Series(range(10), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_single_str(self):
        # [FUNCTION DOC]
        # Verifies (issue 21987) that a single string scalar gives a one-element Series.
        # GH 21987
        expected = Series(["abc"])
        result = Series("abc")
        tm.assert_series_equal(result, expected)

    def test_constructor_list_like(self):
        # [FUNCTION DOC]
        # Verifies that list, tuple and int64 array inputs all give an int64 Series with a given index, independent of platform.
        # make sure that we are coercing different
        # list-likes to standard dtypes and not
        # platform specific
        expected = Series([1, 2, 3], dtype="int64")
        for obj in [[1, 2, 3], (1, 2, 3), np.array([1, 2, 3], dtype="int64")]:
            result = Series(obj, index=[0, 1, 2])
            tm.assert_series_equal(result, expected)

    def test_constructor_boolean_index(self):
        # [FUNCTION DOC]
        # Verifies (issue 18579) that a boolean Series used as the index gives a boolean index.
        # GH#18579
        s1 = Series([1, 2, 3], index=[4, 5, 6])

        index = s1 == 2
        result = Series([1, 3, 2], index=index)
        expected = Series([1, 3, 2], index=[False, True, False])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float64"])
    def test_constructor_index_dtype(self, dtype):
        # [FUNCTION DOC]
        # Verifies (issue 17088) that a Series built from an Index with a requested dtype (bool, int32, int64, float64) has that dtype.
        # GH 17088

        s = Series(Index([0, 2, 4]), dtype=dtype)
        assert s.dtype == dtype

    @pytest.mark.parametrize(
        "input_vals",
        [
            [1, 2],
            ["1", "2"],
            list(date_range("1/1/2011", periods=2, freq="h")),
            list(date_range("1/1/2011", periods=2, freq="h", tz="US/Eastern")),
            [Interval(left=0, right=5)],
        ],
    )
    def test_constructor_list_str(self, input_vals, string_dtype):
        # [FUNCTION DOC]
        # Verifies (issue 16605) that building from integers, strings, datetimes, tz-aware datetimes or intervals with a string dtype (fixture) equals building without dtype and casting to the string dtype.
        # GH 16605
        # Ensure that data elements from a list are converted to strings
        # when dtype is str, 'str', or 'U'
        result = Series(input_vals, dtype=string_dtype)
        expected = Series(input_vals).astype(string_dtype)
        tm.assert_series_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype):
        # [FUNCTION DOC]
        # Verifies that a float list with NaN and a string dtype (fixture) gives text for the numbers and a preserved NaN, as an object Series.
        result = Series([1.0, 2.0, np.nan], dtype=string_dtype)
        expected = Series(["1.0", "2.0", np.nan], dtype=object)
        tm.assert_series_equal(result, expected)
        assert np.isnan(result[2])

    def test_constructor_generator(self):
        # [FUNCTION DOC]
        # Verifies that a generator gives the same Series as a range, also with a non-default index.
        gen = (i for i in range(10))

        result = Series(gen)
        exp = Series(range(10))
        tm.assert_series_equal(result, exp)

        # same but with non-default index
        gen = (i for i in range(10))
        result = Series(gen, index=range(10, 20))
        exp.index = range(10, 20)
        tm.assert_series_equal(result, exp)

    def test_constructor_map(self):
        # [FUNCTION DOC]
        # Verifies (issue 8909) the same generator behavior as the previous test.
        # KNOWN ISSUE: the name says map but the data is built with a generator expression, making this a duplicate of the generator test.
        # GH8909
        m = (x for x in range(10))

        result = Series(m)
        exp = Series(range(10))
        tm.assert_series_equal(result, exp)

        # same but with non-default index
        m = (x for x in range(10))
        result = Series(m, index=range(10, 20))
        exp.index = range(10, 20)
        tm.assert_series_equal(result, exp)

    def test_constructor_categorical(self):
        # [FUNCTION DOC]
        # Verifies that building a Series from a categorical with integer codes and a second positional argument of categories emits the future-warning for constructing a categorical with a dtype and mismatched values (message check), the Series values equal the categorical, and an integer categorical can be cast to int64 on construction.
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            cat = Categorical([0, 1, 2, 0, 1, 2], ["a", "b", "c"])
        res = Series(cat)
        tm.assert_categorical_equal(res.values, cat)

        # can cast to a new dtype
        result = Series(Categorical([1, 2, 3]), dtype="int64")
        expected = Series([1, 2, 3], dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_construct_from_categorical_with_dtype(self):
        # [FUNCTION DOC]
        # Verifies (issue 12574) building from a categorical with the category dtype name keeps a categorical dtype.
        # GH12574
        ser = Series(Categorical([1, 2, 3]), dtype="category")
        assert isinstance(ser.dtype, CategoricalDtype)

    def test_construct_intlist_values_category_dtype(self):
        # [FUNCTION DOC]
        # Verifies that integers with the category dtype name give a categorical dtype.
        ser = Series([1, 2, 3], dtype="category")
        assert isinstance(ser.dtype, CategoricalDtype)

    def test_constructor_categorical_with_coercion(self):
        # [FUNCTION DOC]
        # Verifies coercion of categoricals: a Series from a categorical has category dtype and matching length; the same categorical placed in a frame and selected by label or position equals it; a frame built from the Series (and a frame with several columns, including a scalar column) keep the dtype and each column's name.
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"])
        # test basic creation / coercion of categoricals
        s = Series(factor, name="A")
        assert s.dtype == "category"
        assert len(s) == len(factor)

        # in a frame
        df = DataFrame({"A": factor})
        result = df["A"]
        tm.assert_series_equal(result, s)
        result = df.iloc[:, 0]
        tm.assert_series_equal(result, s)
        assert len(df) == len(factor)

        df = DataFrame({"A": s})
        result = df["A"]
        tm.assert_series_equal(result, s)
        assert len(df) == len(factor)

        # multiples
        df = DataFrame({"A": s, "B": s, "C": 1})
        result1 = df["A"]
        result2 = df["B"]
        tm.assert_series_equal(result1, s)
        tm.assert_series_equal(result2, s, check_names=False)
        assert result2.name == "B"
        assert len(df) == len(factor)

    def test_constructor_categorical_with_coercion2(self):
        # [FUNCTION DOC]
        # Verifies (issue 8623) that after converting a frame column of names to categorical, getting the first element by attribute access, plain integer lookup, label and the row lookup all give the same value.
        # GH8623
        x = DataFrame(
            [[1, "John P. Doe"], [2, "Jane Dove"], [1, "John P. Doe"]],
            columns=["person_id", "person_name"],
        )
        x["person_name"] = Categorical(x.person_name)  # doing this breaks transform

        expected = x.iloc[0].person_name
        result = x.person_name.iloc[0]
        assert result == expected

        result = x.person_name[0]
        assert result == expected

        result = x.person_name.loc[0]
        assert result == expected

    def test_constructor_series_to_categorical(self):
        # [FUNCTION DOC]
        # Verifies (issue 16524) building a categorical Series from a string Series with the category dtype name.
        # see GH#16524: test conversion of Series to Categorical
        series = Series(["a", "b", "c"])

        result = Series(series, dtype="category")
        expected = Series(["a", "b", "c"], dtype="category")

        tm.assert_series_equal(result, expected)

    def test_constructor_categorical_dtype(self):
        # [FUNCTION DOC]
        # Verifies building with an explicit categorical dtype: an ordered dtype with extra category, an unordered one with reversed categories (ordered false), and (issue 19565) broadcast of a scalar to a two-element index with the categorical dtype.
        result = Series(
            ["a", "b"], dtype=CategoricalDtype(["a", "b", "c"], ordered=True)
        )
        assert isinstance(result.dtype, CategoricalDtype)
        tm.assert_index_equal(result.cat.categories, Index(["a", "b", "c"]))
        assert result.cat.ordered

        result = Series(["a", "b"], dtype=CategoricalDtype(["b", "a"]))
        assert isinstance(result.dtype, CategoricalDtype)
        tm.assert_index_equal(result.cat.categories, Index(["b", "a"]))
        assert result.cat.ordered is False

        # GH 19565 - Check broadcasting of scalar with Categorical dtype
        result = Series(
            "a", index=[0, 1], dtype=CategoricalDtype(["a", "b"], ordered=True)
        )
        expected = Series(
            ["a", "a"], index=[0, 1], dtype=CategoricalDtype(["a", "b"], ordered=True)
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_categorical_string(self):
        # [FUNCTION DOC]
        # Verifies (issue 26336) that giving the category dtype name for a categorical or categorical Series keeps the existing categorical dtype with its categories and order.
        # GH 26336: the string 'category' maintains existing CategoricalDtype
        cdt = CategoricalDtype(categories=list("dabc"), ordered=True)
        expected = Series(list("abcabc"), dtype=cdt)

        # Series(Categorical, dtype='category') keeps existing dtype
        cat = Categorical(list("abcabc"), dtype=cdt)
        result = Series(cat, dtype="category")
        tm.assert_series_equal(result, expected)

        # Series(Series[Categorical], dtype='category') keeps existing dtype
        result = Series(result, dtype="category")
        tm.assert_series_equal(result, expected)

    def test_categorical_sideeffects_free(self):
        # [FUNCTION DOC]
        # Verifies copy semantics when building from a categorical: with copying requested the Series is independent of the source (renaming categories and setting a value do not alter the categorical); with the default no-copy the Series shares the categorical object until categories are renamed, after which it is a different object and later assignments do not change the original. Compares via array conversions.
        # Passing a categorical to a Series and then changing values in either
        # the series or the categorical should not change the values in the
        # other one, IF you specify copy!
        cat = Categorical(["a", "b", "c", "a"])
        s = Series(cat, copy=True)
        assert s.cat is not cat
        s = s.cat.rename_categories([1, 2, 3])
        exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
        exp_cat = np.array(["a", "b", "c", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(s.__array__(), exp_s)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # setting
        s[0] = 2
        exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s2)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # however, copy is False by default
        # so this WILL change values
        cat = Categorical(["a", "b", "c", "a"])
        s = Series(cat, copy=False)
        assert s._values is cat
        s = s.cat.rename_categories([1, 2, 3])
        assert s._values is not cat
        exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s)

        s[0] = 2
        exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s2)

    def test_unordered_compare_equal(self):
        # [FUNCTION DOC]
        # Verifies that an unordered categorical dtype with values a, b and None equals a Series built from a categorical with a NaN value and the same categories.
        left = Series(["a", "b", None], dtype=CategoricalDtype(["a", "b"]))
        right = Series(Categorical(["a", "b", np.nan], categories=["a", "b"]))
        tm.assert_series_equal(left, right)

    def test_constructor_maskedarray(self):
        # [FUNCTION DOC]
        # Verifies construction from NumPy masked arrays of float, integer, boolean and nanosecond datetime kinds, starting fully masked and unmasking elements step by step, with and without an index. Masked float entries become NaN; masked integer entries give a float Series; masked boolean entries give an object Series (True, NaN, False); masked datetimes become NaT.
        # Once nothing remains masked, the integer and boolean cases are asserted to NOT equal plain integer or boolean Series: the comparison is expected to fail with a message about differing Series classes.
        # KNOWN ISSUE: a TODO (issue 56131) questions whether the comparison of the unmasked integer and boolean results should raise a class mismatch at all; the test currently pins that surprising behavior by requiring an assertion error.
        data = ma.masked_all((3,), dtype=float)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan])
        tm.assert_series_equal(result, expected)

        data[0] = 0.0
        data[2] = 2.0
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([0.0, np.nan, 2.0], index=index)
        tm.assert_series_equal(result, expected)

        data[1] = 1.0
        result = Series(data, index=index)
        expected = Series([0.0, 1.0, 2.0], index=index)
        tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype=int)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan], dtype=float)
        tm.assert_series_equal(result, expected)

        data[0] = 0
        data[2] = 2
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([0, np.nan, 2], index=index, dtype=float)
        tm.assert_series_equal(result, expected)

        data[1] = 1
        result = Series(data, index=index)
        expected = Series([0, 1, 2], index=index, dtype=int)
        with pytest.raises(AssertionError, match="Series classes are different"):
            # TODO should this be raising at all?
            # https://github.com/pandas-dev/pandas/issues/56131
            tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype=bool)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan], dtype=object)
        tm.assert_series_equal(result, expected)

        data[0] = True
        data[2] = False
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([True, np.nan, False], index=index, dtype=object)
        tm.assert_series_equal(result, expected)

        data[1] = True
        result = Series(data, index=index)
        expected = Series([True, True, False], index=index, dtype=bool)
        with pytest.raises(AssertionError, match="Series classes are different"):
            # TODO should this be raising at all?
            # https://github.com/pandas-dev/pandas/issues/56131
            tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype="M8[ns]")
        result = Series(data)
        expected = Series([iNaT, iNaT, iNaT], dtype="M8[ns]")
        tm.assert_series_equal(result, expected)

        data[0] = datetime(2001, 1, 1)
        data[2] = datetime(2001, 1, 3)
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series(
            [datetime(2001, 1, 1), iNaT, datetime(2001, 1, 3)],
            index=index,
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)

        data[1] = datetime(2001, 1, 2)
        result = Series(data, index=index)
        expected = Series(
            [datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 3)],
            index=index,
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_maskedarray_hardened(self):
        # [FUNCTION DOC]
        # Verifies (issue 24574) that a float masked array with a hard mask, fully masked, builds a Series of NaN.
        # Check numpy masked arrays with hard masks -- from GH24574
        data = ma.masked_all((3,), dtype=float).harden_mask()
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan])
        tm.assert_series_equal(result, expected)

    def test_series_ctor_plus_datetimeindex(self):
        # [FUNCTION DOC]
        # Verifies that constructing from a dict keyed by a datetime index along with that same index keeps the very same index object.
        rng = date_range("20090415", "20090519", freq="B")
        data = dict.fromkeys(rng, 1)

        result = Series(data, index=rng)
        assert result.index.is_(rng)

    def test_constructor_default_index(self):
        # [FUNCTION DOC]
        # Verifies that without an index the Series gets an exact range index of the data length.
        s = Series([0, 1, 2])
        tm.assert_index_equal(s.index, Index(range(3)), exact=True)

    @pytest.mark.parametrize(
        "input",
        [
            [1, 2, 3],
            (1, 2, 3),
            list(range(3)),
            Categorical(["a", "b", "a"]),
            (i for i in range(3)),
            (x for x in range(3)),
        ],
    )
    def test_constructor_index_mismatch(self, input):
        # [FUNCTION DOC]
        # Verifies (issue 19342) that lists, tuples, ranges, categoricals and generators of length three with an index of length four raise a value error stating the lengths of values and index differ. Parametrized over the input.
        # GH 19342
        # test that construction of a Series with an index of different length
        # raises an error
        msg = r"Length of values \(3\) does not match length of index \(4\)"
        with pytest.raises(ValueError, match=msg):
            Series(input, index=np.arange(4))

    def test_constructor_numpy_scalar(self):
        # [FUNCTION DOC]
        # Verifies (issue 19342) that a zero-dimensional NumPy array broadcasts to the index like a scalar with the requested int64 dtype.
        # GH 19342
        # construction with a numpy scalar
        # should not raise
        result = Series(np.array(100), index=np.arange(4), dtype="int64")
        expected = Series(100, index=np.arange(4), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_broadcast_list(self):
        # [FUNCTION DOC]
        # Verifies (issue 19342) that a one-element list with a three-element index raises a value error on length mismatch instead of broadcasting.
        # GH 19342
        # construction with single-element container and index
        # should raise
        msg = r"Length of values \(1\) does not match length of index \(3\)"
        with pytest.raises(ValueError, match=msg):
            Series(["foo"], index=["a", "b", "c"])

    def test_constructor_corner(self):
        # [FUNCTION DOC]
        # Smoke test: a Series whose two elements are frames, with an index of two labels, can be constructed.
        df = DataFrame(range(5), index=date_range("2020-01-01", periods=5))
        objs = [df, df]
        s = Series(objs, index=[0, 1])
        assert isinstance(s, Series)

    def test_constructor_sanitize(self):
        # [FUNCTION DOC]
        # Verifies that floats with whole values convert to the requested int64 dtype, and that a float array containing NaN with an integer dtype raises the integer-casting-of-NaN error (message about non-finite values) even with copy requested.
        s = Series(np.array([1.0, 1.0, 8.0]), dtype="i8")
        assert s.dtype == np.dtype("i8")

        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(np.array([1.0, 1.0, np.nan]), copy=True, dtype="i8")

    def test_constructor_copy(self):
        # [FUNCTION DOC]
        # Verifies (issue 15125) that building from a Series with copy and a dtype argument gives an equal independent copy for both list-based and array-based sources: changing the source afterward does not change the copy.
        # GH15125
        # test dtype parameter has no side effects on copy=True
        for data in [[1.0], np.array([1.0])]:
            x = Series(data)
            y = Series(x, copy=True, dtype=float)

            # copy=True maintains original data in Series
            tm.assert_series_equal(x, y)

            # changes to origin of copy does not affect the copy
            x[0] = 2.0
            assert not x.equals(y)
            assert x[0] == 2.0
            assert y[0] == 1.0

    @pytest.mark.parametrize(
        "index",
        [
            date_range("20170101", periods=3, tz="US/Eastern"),
            date_range("20170101", periods=3),
            timedelta_range("1 day", periods=3),
            period_range("2012Q1", periods=3, freq="Q"),
            Index(list("abc")),
            Index([1, 2, 3]),
            RangeIndex(0, 3),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_limit_copies(self, index):
        # [FUNCTION DOC]
        # Smoke test (issue 17449): building a Series from various index types (tz-aware and naive datetime, timedelta, period, string, integer and range) stores a copy: the first internal block's values are not the index object. Parametrized over the index types with ids by class name.
        # GH 17449
        # limit copies of input
        s = Series(index)

        # we make 1 copy; this is just a smoke test here
        assert s._mgr.blocks[0].values is not index

    def test_constructor_shallow_copy(self):
        # [FUNCTION DOC]
        # Verifies (issue 49523) that building from an existing Series without copying yields a new object with its own manager (shallow copy): replacing the index of the new Series does not change the original.
        # constructing a Series from Series with copy=False should still
        # give a "shallow" copy (share data, not attributes)
        # https://github.com/pandas-dev/pandas/issues/49523
        s = Series([1, 2, 3])
        s_orig = s.copy()
        s2 = Series(s)
        assert s2._mgr is not s._mgr
        # Overwriting index of s2 doesn't change s
        s2.index = ["a", "b", "c"]
        tm.assert_series_equal(s, s_orig)

    def test_constructor_pass_none(self):
        # [FUNCTION DOC]
        # Verifies that none data with an index gives float64 values (or object when requested), and (issue 7431) an index made from an array holding only None is inferred the same way as an Index with None.
        s = Series(None, index=range(5))
        assert s.dtype == np.float64

        s = Series(None, index=range(5), dtype=object)
        assert s.dtype == np.object_

        # GH 7431
        # inference on the index
        s = Series(index=np.array([None]))
        expected = Series(index=Index([None]))
        tm.assert_series_equal(s, expected)

    def test_constructor_pass_nan_nat(self):
        # [FUNCTION DOC]
        # Verifies (issue 13467) that lists or arrays of NaN give float64; those of NaT, or NaT mixed with NaN in either order, give second-resolution datetime64 dtype.
        # GH 13467
        exp = Series([np.nan, np.nan], dtype=np.float64)
        assert exp.dtype == np.float64
        tm.assert_series_equal(Series([np.nan, np.nan]), exp)
        tm.assert_series_equal(Series(np.array([np.nan, np.nan])), exp)

        exp = Series([NaT, NaT])
        assert exp.dtype == "datetime64[s]"
        tm.assert_series_equal(Series([NaT, NaT]), exp)
        tm.assert_series_equal(Series(np.array([NaT, NaT])), exp)

        tm.assert_series_equal(Series([NaT, np.nan]), exp)
        tm.assert_series_equal(Series(np.array([NaT, np.nan])), exp)

        tm.assert_series_equal(Series([np.nan, NaT]), exp)
        tm.assert_series_equal(Series(np.array([np.nan, NaT])), exp)

    def test_constructor_cast(self):
        # [FUNCTION DOC]
        # Verifies that non-numeric strings with float dtype raise a value error that a string could not be converted to float.
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            Series(["a", "b", "c"], dtype=float)

    def test_constructor_signed_int_overflow_raises(self):
        # [FUNCTION DOC]
        # Verifies (issue 41734) that integer values that do not fit in 8-bit signed or unsigned dtype raise an error instead of silently overflowing: an overflow error with the NumPy-2 message, or a value error with a message about values being too large on older NumPy.
        # GH#41734 disallow silent overflow, enforced in 2.0
        if np_version_gt2:
            msg = "The elements provided in the data cannot all be casted to the dtype"
            err = OverflowError
        else:
            msg = "Values are too large to be losslessly converted"
            err = ValueError
        with pytest.raises(err, match=msg):
            Series([1, 200, 923442], dtype="int8")

        with pytest.raises(err, match=msg):
            Series([1, 200, 923442], dtype="uint8")

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1], dtype=np.uint16),
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint64),
            [np.uint16(1)],
            [np.uint32(1)],
            [np.uint64(1)],
        ],
    )
    def test_constructor_numpy_uints(self, values):
        # [FUNCTION DOC]
        # Verifies (issue 47294) that Series built from unsigned 16/32/64-bit arrays or lists of NumPy unsigned scalars keep the element dtype and value when read back.
        # GH#47294
        value = values[0]
        result = Series(values)

        assert result[0].dtype == value.dtype
        assert result[0] == value

    def test_constructor_unsigned_dtype_overflow(self, any_unsigned_int_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies (issue 15832) that a negative value with each unsigned integer dtype (fixture) raises an overflow error (NumPy-2 message naming the dtype, or the older coercion message).
        # see gh-15832
        if np_version_gt2:
            msg = (
                f"The elements provided in the data cannot "
                f"all be casted to the dtype {any_unsigned_int_numpy_dtype}"
            )
        else:
            msg = "Trying to coerce negative values to unsigned integers"
        with pytest.raises(OverflowError, match=msg):
            Series([-1], dtype=any_unsigned_int_numpy_dtype)

    def test_constructor_floating_data_int_dtype(self, frame_or_series):
        # [FUNCTION DOC]
        # Verifies (issues 40110, 49599) that for Series and DataFrame (fixture) floating data with a non-integral value and an integer dtype raises a value error about coercing floats to integers (arrays and lists); with a NaN present it raises the integer-casting error for arrays and, for Series from a list, a value error about converting float NaN; losslessly integral floats convert to the integer dtype.
        # GH#40110
        arr = np.random.default_rng(2).standard_normal(2)

        # Long-standing behavior (for Series, new in 2.0 for DataFrame)
        #  has been to ignore the dtype on these;
        #  not clear if this is what we want long-term
        # expected = frame_or_series(arr)

        # GH#49599 as of 2.0 we raise instead of silently retaining float dtype
        msg = "Trying to coerce float values to integer"
        with pytest.raises(ValueError, match=msg):
            frame_or_series(arr, dtype="i8")

        with pytest.raises(ValueError, match=msg):
            frame_or_series(list(arr), dtype="i8")

        # pre-2.0, when we had NaNs, we silently ignored the integer dtype
        arr[0] = np.nan
        # expected = frame_or_series(arr)

        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            frame_or_series(arr, dtype="i8")

        exc = IntCastingNaNError
        if frame_or_series is Series:
            # TODO: try to align these
            exc = ValueError
            msg = "cannot convert float NaN to integer"
        with pytest.raises(exc, match=msg):
            # same behavior if we pass list instead of the ndarray
            frame_or_series(list(arr), dtype="i8")

        # float array that can be losslessly cast to integers
        arr = np.array([1.0, 2.0], dtype="float64")
        expected = frame_or_series(arr.astype("i8"))

        obj = frame_or_series(arr, dtype="i8")
        tm.assert_equal(obj, expected)

        obj = frame_or_series(list(arr), dtype="i8")
        tm.assert_equal(obj, expected)

    def test_constructor_coerce_float_fail(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies (issue 15832) that a list or array containing a non-integral float with any integer dtype (fixture) raises a value error about coercing floats to integers.
        # see gh-15832
        # Updated: make sure we treat this list the same as we would treat
        #  the equivalent ndarray
        # GH#49599 pre-2.0 we silently retained float dtype, in 2.0 we raise
        vals = [1, 2, 3.5]

        msg = "Trying to coerce float values to integer"
        with pytest.raises(ValueError, match=msg):
            Series(vals, dtype=any_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            Series(np.array(vals), dtype=any_int_numpy_dtype)

    def test_constructor_coerce_float_valid(self, float_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies that a mix of integers and a float with a float dtype (fixture) equals the default-inferred Series cast to that dtype.
        s = Series([1, 2, 3.5], dtype=float_numpy_dtype)
        expected = Series([1, 2, 3.5]).astype(float_numpy_dtype)
        tm.assert_series_equal(s, expected)

    def test_constructor_invalid_coerce_ints_with_float_nan(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies (issue 22585) that integers and NaN with an integer dtype raise a value error about converting float NaN for lists, and the integer-casting error for arrays.
        # GH 22585
        # Updated: make sure we treat this list the same as we would treat the
        # equivalent ndarray
        vals = [1, 2, np.nan]
        # pre-2.0 this would return with a float dtype, in 2.0 we raise

        msg = "cannot convert float NaN to integer"
        with pytest.raises(ValueError, match=msg):
            Series(vals, dtype=any_int_numpy_dtype)
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(np.array(vals), dtype=any_int_numpy_dtype)

    def test_constructor_dtype_no_cast(self):
        # [FUNCTION DOC]
        # Verifies (issue 1572) that building from a Series with the same dtype does not link the new Series to the old: assignment on the new one leaves the original unchanged.
        # see gh-1572
        s = Series([1, 2, 3])
        s2 = Series(s, dtype=np.int64)

        s2[1] = 5
        assert s[1] == 2

    def test_constructor_datelike_coercion(self):
        # [FUNCTION DOC]
        # Verifies (issue 9477) that a list with a timestamp and a non-date string with object dtype requested stays object with elements unchanged.
        # GH 9477
        # incorrectly inferring on dateimelike looking when object dtype is
        # specified
        s = Series([Timestamp("20130101"), "NOV"], dtype=object)
        assert s.iloc[0] == Timestamp("20130101")
        assert s.iloc[1] == "NOV"
        assert s.dtype == object

    def test_constructor_datelike_coercion2(self):
        # [FUNCTION DOC]
        # Verifies that selecting rows from a mixed frame (strings and datetimes, labels such as 3T19) returns an object-dtype row, not re-inferred as datetime.
        # the dtype was being reset on the slicing and re-inferred to datetime
        # even thought the blocks are mixed
        belly = "216 3T19".split()
        wing1 = "2T15 4H19".split()
        wing2 = "416 4T20".split()
        mat = pd.to_datetime("2016-01-22 2019-09-07".split())
        df = DataFrame({"wing1": wing1, "wing2": wing2, "mat": mat}, index=belly)

        result = df.loc["3T19"]
        assert result.dtype == object
        result = df.loc["216"]
        assert result.dtype == object

    def test_constructor_mixed_int_and_timestamp(self, frame_or_series):
        # [FUNCTION DOC]
        # Verifies, for Series and DataFrame (fixture), that a list mixing a nanosecond timestamp, an integer and the internal NaT integer with nanosecond datetime dtype equals the construction from timestamps and NaT.
        # specifically Timestamp with nanos, not datetimes
        objs = [Timestamp(9), 10, NaT._value]
        result = frame_or_series(objs, dtype="M8[ns]")

        expected = frame_or_series([Timestamp(9), Timestamp(10), NaT])
        tm.assert_equal(result, expected)

    def test_constructor_datetimes_with_nulls(self):
        # [FUNCTION DOC]
        # Verifies (issue 15869) that object arrays of None and a current datetime infer microsecond datetime dtype.
        # gh-15869
        for arr in [
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
        ]:
            result = Series(arr)
            assert result.dtype == "M8[us]"

    def test_constructor_dtype_datetime64(self):
        # [FUNCTION DOC]
        # Verifies datetime64 construction with missing sentinels: the internal NaT integer scalar with nanosecond datetime dtype and an index is all missing; without dtype it is not all missing (it is just an integer); NaN scalar with the dtype is all missing; a list with a datetime and either the integer sentinel or NaN gives a missing second element with the nanosecond dtype.
        s = Series(iNaT, dtype="M8[ns]", index=range(5))
        assert isna(s).all()

        # in theory this should be all nulls, but since
        # we are not specifying a dtype is ambiguous
        s = Series(iNaT, index=range(5))
        assert not isna(s).all()

        s = Series(np.nan, dtype="M8[ns]", index=range(5))
        assert isna(s).all()

        s = Series([datetime(2001, 1, 2, 0, 0), iNaT], dtype="M8[ns]")
        assert isna(s[1])
        assert s.dtype == "M8[ns]"

        s = Series([datetime(2001, 1, 2, 0, 0), np.nan], dtype="M8[ns]")
        assert isna(s[1])
        assert s.dtype == "M8[ns]"

    def test_constructor_dtype_datetime64_10(self):
        # [FUNCTION DOC]
        # Verifies (issues 3416, 3414) datetime64 construction from NumPy datetimes of three days: inferred microsecond dtype, retained after setting an element to NaN; constructing with millisecond dtype from dates (including via division of integer representation) equals one built from Python datetimes; a list with NaN and datetimes with nanosecond dtype gives NaT first.
        # GH3416
        pydates = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)]
        dates = [np.datetime64(x) for x in pydates]

        ser = Series(dates)
        assert ser.dtype == "M8[us]"

        ser.iloc[0] = np.nan
        assert ser.dtype == "M8[us]"

        # GH3414 related
        expected = Series(pydates, dtype="datetime64[ms]")

        result = Series(Series(dates).astype(np.int64) / 1000, dtype="M8[ms]")
        tm.assert_series_equal(result, expected)

        result = Series(dates, dtype="datetime64[ms]")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [NaT, datetime(2013, 1, 2), datetime(2013, 1, 3)], dtype="datetime64[ns]"
        )
        result = Series([np.nan, *dates[1:]], dtype="datetime64[ns]")
        tm.assert_series_equal(result, expected)

    def test_constructor_dtype_datetime64_11(self):
        # [FUNCTION DOC]
        # Verifies that a datetime Series of NumPy dates: casting to int64 works; casting to int32 raises a type error saying the conversion is not supported; and constructing with int64 dtype equals casting.
        pydates = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)]
        dates = [np.datetime64(x) for x in pydates]

        dts = Series(dates, dtype="datetime64[ns]")

        # valid astype
        dts.astype("int64")

        # invalid casting
        msg = r"Converting from datetime64\[ns\] to int32 is not supported"
        with pytest.raises(TypeError, match=msg):
            dts.astype("int32")

        # ints are ok
        # we test with np.int64 to get similar results on
        # windows / 32-bit platforms
        result = Series(dts, dtype=np.int64)
        expected = Series(dts.astype(np.int64))
        tm.assert_series_equal(result, expected)

    def test_constructor_dtype_datetime64_9(self):
        # [FUNCTION DOC]
        # Verifies that out-of-range-for-nanosecond datetimes (year 2 and year 3000) are retained as values in a Series.
        # invalid dates can be help as object
        result = Series([datetime(2, 1, 1)])
        assert result[0] == datetime(2, 1, 1, 0, 0)

        result = Series([datetime(3000, 1, 1)])
        assert result[0] == datetime(3000, 1, 1, 0, 0)

    def test_constructor_dtype_datetime64_8(self):
        # [FUNCTION DOC]
        # Verifies that a list mixing a timestamp and an integer is not coerced: each keeps its type and value.
        # don't mix types
        result = Series([Timestamp("20130101"), 1], index=["a", "b"])
        assert result["a"] == Timestamp("20130101")
        assert result["b"] == 1

    def test_constructor_dtype_datetime64_7(self):
        # [FUNCTION DOC]
        # Verifies (issues 6529, 13876) datetime64 construction at different units (second, day, millisecond, microsecond, nanosecond): the dtype matches the unit (day becomes second, the nearest supported) and values equal the cast expected; with object dtype requested values equal the object expected; an array of date objects stays object and unchanged.
        # GH6529
        # coerce datetime64 non-ns properly
        dates = date_range("01-Jan-2015", "01-Dec-2015", freq="ME")
        values2 = dates.view(np.ndarray).astype("datetime64[ns]")
        expected = Series(values2, index=dates)

        for unit in ["s", "D", "ms", "us", "ns"]:
            dtype = np.dtype(f"M8[{unit}]")
            values1 = dates.view(np.ndarray).astype(dtype)
            result = Series(values1, dates)
            if unit == "D":
                # for unit="D" we cast to nearest-supported reso, i.e. "s"
                dtype = np.dtype("M8[s]")
            assert result.dtype == dtype
            tm.assert_series_equal(result, expected.astype(dtype))

        # GH 13876
        # coerce to non-ns to object properly
        expected = Series(values2, index=dates, dtype=object)
        for dtype in ["s", "D", "ms", "us", "ns"]:
            values1 = dates.view(np.ndarray).astype(f"M8[{dtype}]")
            result = Series(values1, index=dates, dtype=object)
            tm.assert_series_equal(result, expected)

        # leave datetime.date alone
        dates2 = np.array([d.date() for d in dates.to_pydatetime()], dtype=object)
        series1 = Series(dates2, dates)
        tm.assert_numpy_array_equal(series1.values, dates2)
        assert series1.dtype == object

    def test_constructor_dtype_datetime64_6(self):
        # [FUNCTION DOC]
        # Verifies that lists mixing None or NaN, NaT and a datetime-like string infer object dtype (no parsing of strings since version 2.0), for four orderings.
        # as of 2.0, these no longer infer datetime64 based on the strings,
        #  matching the Index behavior

        ser = Series([None, NaT, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([np.nan, NaT, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([NaT, None, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([NaT, np.nan, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

    def test_constructor_dtype_datetime64_5(self):
        # [FUNCTION DOC]
        # Verifies (issue 8411) that Series built from naive, UTC and US/Eastern date ranges expose the matching timezone on elements.
        # tz-aware (UTC and other tz's)
        # GH 8411
        dr = date_range("20130101", periods=3)
        assert Series(dr).iloc[0].tz is None
        dr = date_range("20130101", periods=3, tz="UTC")
        assert str(Series(dr).iloc[0].tz) == "UTC"
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        assert str(Series(dr).iloc[0].tz) == "US/Eastern"

    def test_constructor_dtype_datetime64_4(self):
        # [FUNCTION DOC]
        # Verifies that integers mixed with NaT give object dtype, the NaT element stays the NaT singleton, and its text form includes NaT.
        # non-convertible
        ser = Series([1479596223000, -1479590, NaT])
        assert ser.dtype == "object"
        assert ser[2] is NaT
        assert "NaT" in str(ser)

    def test_constructor_dtype_datetime64_3(self):
        # [FUNCTION DOC]
        # Verifies that datetimes (including out-of-nanosecond-range year 2) with NaT infer microsecond datetime dtype and keep NaT as the singleton.
        # if we passed a NaT it remains
        ser = Series([datetime(2010, 1, 1), datetime(2, 1, 1), NaT])
        assert ser.dtype == "M8[us]"
        assert ser[2] is NaT
        assert "NaT" in str(ser)

    def test_constructor_dtype_datetime64_2(self):
        # [FUNCTION DOC]
        # Verifies that datetimes (including year 2) with NaN infer microsecond datetime dtype and NaN is converted to NaT.
        # if we passed a nan it remains
        ser = Series([datetime(2010, 1, 1), datetime(2, 1, 1), np.nan])
        assert ser.dtype == "M8[us]"
        assert ser[2] is NaT
        assert "NaT" in str(ser)

    def test_constructor_with_datetime_tz(self):
        # [FUNCTION DOC]
        # Verifies (issue 8260) a Series built from a nanosecond US/Eastern date range: dtype name and type are the zone-aware datetime dtype and appear in the text form; the values export is a naive nanosecond datetime64 NumPy array that, relocalized from UTC to the zone, equals the original range; scalar access by position gives the expected zone-aware timestamp; boolean mask selection and slicing work; concatenating two slices restores the Series; shifting shows NaT in the text; and a datetime index built from the Series with inferred frequency equals the range.
        # 8260
        # support datetime64 with tz

        dr = date_range("20130101", periods=3, tz="US/Eastern", unit="ns")
        s = Series(dr)
        assert s.dtype.name == "datetime64[ns, US/Eastern]"
        assert s.dtype == "datetime64[ns, US/Eastern]"
        assert isinstance(s.dtype, DatetimeTZDtype)
        assert "datetime64[ns, US/Eastern]" in str(s)

        # export
        result = s.values
        assert isinstance(result, np.ndarray)
        assert result.dtype == "datetime64[ns]"

        exp = DatetimeIndex(result)
        exp = exp.tz_localize("UTC").tz_convert(tz=s.dt.tz)
        tm.assert_index_equal(dr, exp)

        # indexing
        result = s.iloc[0]
        assert result == Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern")
        result = s[0]
        assert result == Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern")

        result = s[Series([True, True, False], index=s.index)]
        tm.assert_series_equal(result, s[0:2])

        result = s.iloc[0:1]
        tm.assert_series_equal(result, Series(dr[0:1]))

        # concat
        result = pd.concat([s.iloc[0:1], s.iloc[1:]])
        tm.assert_series_equal(result, s)

        # short str
        assert "datetime64[ns, US/Eastern]" in str(s)

        # formatting with NaT
        result = s.shift()
        assert "datetime64[ns, US/Eastern]" in str(result)
        assert "NaT" in str(result)

        result = DatetimeIndex(s, freq="infer")
        tm.assert_index_equal(result, dr)

    def test_constructor_with_datetime_tz5(self):
        # [FUNCTION DOC]
        # Verifies that a long (1000-element) zone-aware Series' text form still mentions the dtype name.
        # long str
        ser = Series(date_range("20130101", periods=1000, tz="US/Eastern", unit="ns"))
        assert "datetime64[ns, US/Eastern]" in str(ser)

    def test_constructor_with_datetime_tz4(self):
        # [FUNCTION DOC]
        # Verifies inference: a list of two second-resolution Pacific timestamps gives a second-resolution Pacific datetime dtype, and the library's type inference reports datetime64.
        # inference
        ser = Series(
            [
                Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific").as_unit("s"),
                Timestamp("2013-01-02 14:00:00-0800", tz="US/Pacific").as_unit("s"),
            ]
        )
        assert ser.dtype == "datetime64[s, US/Pacific]"
        assert lib.infer_dtype(ser, skipna=True) == "datetime64"

    def test_constructor_with_datetime_tz3(self):
        # [FUNCTION DOC]
        # Verifies that timestamps in two different time zones (Pacific and Eastern) give object dtype, and type inference reports datetime.
        ser = Series(
            [
                Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
                Timestamp("2013-01-02 14:00:00-0800", tz="US/Eastern"),
            ]
        )
        assert ser.dtype == "object"
        assert lib.infer_dtype(ser, skipna=True) == "datetime"

    def test_constructor_with_datetime_tz2(self):
        # [FUNCTION DOC]
        # Verifies that a scalar NaT broadcast over an index with a nanosecond Eastern datetime dtype equals a Series built from an all-NaT zone-aware datetime index.
        # with all NaT
        ser = Series(NaT, index=[0, 1], dtype="datetime64[ns, US/Eastern]")
        dti = DatetimeIndex(["NaT", "NaT"], tz="US/Eastern").as_unit("ns")
        expected = Series(dti)
        tm.assert_series_equal(ser, expected)

    def test_constructor_no_partial_datetime_casting(self):
        # [FUNCTION DOC]
        # Verifies (issue 40111) that a list mixing the string nan, a timestamp, offset-bearing datetime strings and None stays object, with every element retained as the identical original object (no partial datetime conversion).
        # GH#40111
        vals = [
            "nan",
            Timestamp("1990-01-01"),
            "2015-03-14T16:15:14.123-08:00",
            "2019-03-04T21:56:32.620-07:00",
            None,
        ]
        ser = Series(vals)
        assert all(ser[i] is vals[i] for i in range(len(vals)))

    @pytest.mark.parametrize("arr_dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("kind", ["M", "m"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_construction_to_datetimelike_unit(self, arr_dtype, kind, unit):
        # [FUNCTION DOC]
        # Verifies datetime-like unit handling when casting an integer or float Series of 1, 2, 3 to datetime or timedelta dtypes at many units (nanosecond to day), parametrized over array dtype, kind and unit: the cast equals a Series built from the cast array; units finer than days keep the unit, others become seconds (nearest supported).
        # KNOWN ISSUE: a TODO notes the originating issue (19223) was about casting rather than construction, so the test is misplaced in this module.
        # tests all units
        # gh-19223
        # TODO: GH#19223 was about .astype, doesn't belong here
        dtype = f"{kind}8[{unit}]"
        arr = np.array([1, 2, 3], dtype=arr_dtype)
        ser = Series(arr)
        result = ser.astype(dtype)

        expected = Series(arr.astype(dtype))

        if unit in ["ns", "us", "ms", "s"]:
            assert result.dtype == dtype
            assert expected.dtype == dtype
        else:
            # Otherwise we cast to nearest-supported unit, i.e. seconds
            assert result.dtype == f"{kind}8[s]"
            assert expected.dtype == f"{kind}8[s]"

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("arg", ["2013-01-01 00:00:00", NaT, np.nan, None])
    def test_constructor_with_naive_string_and_datetimetz_dtype(self, arg):
        # [FUNCTION DOC]
        # Verifies (issue 17415) that a naive datetime string, NaT, NaN or None with a zone-aware datetime dtype (CET) is interpreted as local wall time in that zone, equal to localizing a naive Series. Parametrized over the input value.
        # GH 17415: With naive string
        result = Series([arg], dtype="datetime64[ns, CET]")
        expected = Series([Timestamp(arg)], dtype="M8[ns]").dt.tz_localize("CET")
        tm.assert_series_equal(result, expected)

    def test_constructor_datetime64_bigendian(self):
        # [FUNCTION DOC]
        # Verifies (issue 30976) that a big-endian millisecond datetime64 array is converted to the native-endian millisecond dtype with the same value.
        # GH#30976
        ms = np.datetime64(1, "ms")
        arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")

        result = Series(arr)
        expected = Series([Timestamp(ms)]).astype("M8[ms]")
        assert expected.dtype == "M8[ms]"
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("interval_constructor", [IntervalIndex, IntervalArray])
    def test_construction_interval(self, interval_constructor):
        # [FUNCTION DOC]
        # Verifies that a Series built from an interval index or interval array (parametrized) of right-closed integer breaks has the closed-right int64 interval dtype and holds the same intervals.
        # construction from interval & array of intervals
        intervals = interval_constructor.from_breaks(np.arange(3), closed="right")
        result = Series(intervals)
        assert result.dtype == "interval[int64, right]"
        tm.assert_index_equal(Index(result.values), Index(intervals))

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_infer_interval(self, data_constructor):
        # [FUNCTION DOC]
        # Verifies (issue 23563) that a list or object array of intervals with the same closure plus None infers float64 right-closed interval dtype, equal to building from an interval array.
        # GH 23563: consistent closed results in interval dtype
        data = [Interval(0, 1), Interval(0, 2), None]
        result = Series(data_constructor(data))
        expected = Series(IntervalArray(data))
        assert result.dtype == "interval[float64, right]"
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_interval_mixed_closed(self, data_constructor):
        # [FUNCTION DOC]
        # Verifies (issue 23563) that intervals with different closures (both and neither) give object dtype and the elements are preserved in order.
        # GH 23563: mixed closed results in object dtype (not interval dtype)
        data = [Interval(0, 1, closed="both"), Interval(0, 2, closed="neither")]
        result = Series(data_constructor(data))
        assert result.dtype == object
        assert result.tolist() == data

    def test_construction_consistency(self):
        # [FUNCTION DOC]
        # Verifies (issue 14928) that zone-aware datetimes are not relocalized on construction: passing a Series with its own dtype, a UTC-converted Series with the zone dtype, and values converted via the library all equal the original; naive datetime64 values with a zone dtype are treated as wall times (as of version 2.0, issue 33401); the suggested alternatives (localize to UTC then convert, or pass the integer view with the dtype) produce no warnings and equal the original.
        # make sure that we are not re-localizing upon construction
        # GH 14928
        ser = Series(date_range("20130101", periods=3, tz="US/Eastern"))

        result = Series(ser, dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        result = Series(ser.dt.tz_convert("UTC"), dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        # Pre-2.0 dt64 values were treated as utc, which was inconsistent
        #  with DatetimeIndex, which treats them as wall times, see GH#33401
        result = Series(ser.values, dtype=ser.dtype)
        expected = Series(ser.values).dt.tz_localize(ser.dtype.tz)
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(None):
            # one suggested alternative to the deprecated (changed in 2.0) usage
            middle = Series(ser.values).dt.tz_localize("UTC")
            result = middle.dt.tz_convert(ser.dtype.tz)
        tm.assert_series_equal(result, ser)

        with tm.assert_produces_warning(None):
            # the other suggested alternative to the deprecated usage
            result = Series(ser.values.view("int64"), dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_infer_period(self, data_constructor):
        # [FUNCTION DOC]
        # Verifies that a list or object array of periods with the same frequency and None infers the daily period dtype, equal to the Series built from a period array.
        data = [Period("2000", "D"), Period("2001", "D"), None]
        result = Series(data_constructor(data))
        expected = Series(period_array(data))
        tm.assert_series_equal(result, expected)
        assert result.dtype == "Period[D]"

    @pytest.mark.xfail(reason="PeriodDtype Series not supported yet")
    def test_construct_from_ints_including_iNaT_scalar_period_dtype(self):
        # [FUNCTION DOC]
        # Expected-to-fail test (marked: period dtype Series not supported yet): a daily-period Series built from integers including the internal NaT sentinel; checks that the last value is missing and that assigning it elsewhere keeps the position missing.
        series = Series([0, 1000, 2000, pd._libs.iNaT], dtype="period[D]")

        val = series[3]
        assert isna(val)

        series[2] = val
        assert isna(series[2])

    def test_constructor_period_incompatible_frequency(self):
        # [FUNCTION DOC]
        # Verifies that periods of different frequencies (daily and yearly) give object dtype, retaining the elements.
        data = [Period("2000", "D"), Period("2001", "Y")]
        result = Series(data)
        assert result.dtype == object
        assert result.tolist() == data

    def test_constructor_periodindex(self):
        # [FUNCTION DOC]
        # Verifies (issue 7932) that a Series built from a daily period index has period dtype with daily frequency, and that the object-cast index gives object dtype.
        # KNOWN ISSUE: the comparison target is created from the object-cast index but is never compared to the Series, and the assertions only check each dtype separately.
        # GH7932
        # converting a PeriodIndex when put in a Series

        pi = period_range("20130101", periods=5, freq="D")
        s = Series(pi)
        assert s.dtype == "Period[D]"
        expected = Series(pi.astype(object))
        assert expected.dtype == object

    def test_constructor_dict(self):
        # [FUNCTION DOC]
        # Verifies construction from dictionaries: keys become the index (in insertion order), a given index selects and orders keys with NaN for absent ones (floats), and a period index of ten with two dict entries keyed by periods gives float values with NaN elsewhere.
        d = {"a": 0.0, "b": 1.0, "c": 2.0}

        result = Series(d)
        expected = Series(d, index=sorted(d.keys()))
        tm.assert_series_equal(result, expected)

        result = Series(d, index=["b", "c", "d", "a"])
        expected = Series([1, 2, np.nan, 0], index=["b", "c", "d", "a"])
        tm.assert_series_equal(result, expected)

        pidx = period_range("2020-01-01", periods=10, freq="D")
        d = {pidx[0]: 0, pidx[1]: 1}
        result = Series(d, index=pidx)
        expected = Series(np.nan, pidx, dtype=np.float64)
        expected.iloc[0] = 0
        expected.iloc[1] = 1
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_list_value_explicit_dtype(self):
        # [FUNCTION DOC]
        # Verifies (issue 18625) that a dict holding a list of lists with explicit object dtype and a one-label index equals the same without dtype.
        # GH 18625
        d = {"a": [[2], [3], [4]]}
        result = Series(d, index=["a"], dtype="object")
        expected = Series(d, index=["a"])
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_order(self):
        # [FUNCTION DOC]
        # Verifies (issue 19018) that dict construction keeps insertion order of keys.
        # GH19018
        # initialization ordering: by insertion order
        d = {"b": 1, "a": 0, "c": 2}
        result = Series(d)
        expected = Series([1, 0, 2], index=list("bac"))
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_extension(self, ea_scalar_and_dtype):
        # [FUNCTION DOC]
        # Verifies that a dict holding an extension-dtype scalar (fixture) with an index keeps the matching extension dtype and equals broadcasting the scalar with that dtype.
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        d = {"a": ea_scalar}
        result = Series(d, index=["a"])
        expected = Series(ea_scalar, index=["a"], dtype=ea_dtype)

        assert result.dtype == ea_dtype

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("value", [2, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value):
        # [FUNCTION DOC]
        # Verifies (issue 18480) dict construction with NaN-like keys (the parametrized value plus a float NaN): after sorting by value, the index contains the NaN-like labels in expected places; also for tuple keys, building a multi-index with NaN in a level.
        # GH 18480
        d = {1: "a", value: "b", float("nan"): "c", 4: "d"}
        result = Series(d).sort_values()
        expected = Series(["a", "b", "c", "d"], index=[1, value, np.nan, 4])
        tm.assert_series_equal(result, expected)

        # MultiIndex:
        d = {(1, 1): "a", (2, np.nan): "b", (3, value): "c"}
        result = Series(d).sort_values()
        expected = Series(
            ["a", "b", "c"], index=Index([(1, 1), (2, np.nan), (3, value)])
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_datetime64_index(self):
        # [FUNCTION DOC]
        # Verifies (issue 9456) that dictionaries keyed by NumPy datetimes, Python datetimes or timestamps give a datetime index, with the NumPy-datetime case at second resolution, and the others equal to the expected Series.
        # GH 9456

        dates_as_str = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]
        values = [42544017.198965244, 1234565, 40512335.181958228, -1]

        def create_data(constructor):
            # [FUNCTION DOC]
            # HELPER: nested function taking a key-constructor and returning a dictionary pairing each of the four date strings, converted with that constructor, to the four float values.
            return dict(zip((constructor(x) for x in dates_as_str), values))

        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        data_Timestamp = create_data(Timestamp)

        expected = Series(values, (Timestamp(x) for x in dates_as_str))

        result_datetime64 = Series(data_datetime64)
        result_datetime = Series(data_datetime)
        result_Timestamp = Series(data_Timestamp)

        tm.assert_series_equal(
            result_datetime64, expected.set_axis(expected.index.as_unit("s"))
        )
        tm.assert_series_equal(result_datetime, expected)
        tm.assert_series_equal(result_Timestamp, expected)

    def test_constructor_dict_tuple_indexer(self):
        # [FUNCTION DOC]
        # Verifies (issue 12948) that a dict with a tuple key containing None builds a multi-index with a NaN entry (code minus one), and one value.
        # GH 12948
        data = {(1, 1, None): -1.0}
        result = Series(data)
        expected = Series(
            -1.0, index=MultiIndex(levels=[[1], [1], [np.nan]], codes=[[0], [0], [-1]])
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_mapping(self, non_dict_mapping_subclass):
        # [FUNCTION DOC]
        # Verifies (issue 29788) that a mapping subclass that is not a dict (fixture) is accepted like a dict.
        # GH 29788
        ndm = non_dict_mapping_subclass({3: "three"})
        result = Series(ndm)
        expected = Series(["three"], index=[3])

        tm.assert_series_equal(result, expected)

    def test_constructor_list_of_tuples(self):
        # [FUNCTION DOC]
        # Verifies that a list of tuples builds a Series whose elements are those tuples.
        data = [(1, 1), (2, 2), (2, 3)]
        s = Series(data)
        assert list(s) == data

    def test_constructor_tuple_of_tuples(self):
        # [FUNCTION DOC]
        # Verifies that a tuple of tuples builds a Series whose elements are those tuples.
        data = ((1, 1), (2, 2), (2, 3))
        s = Series(data)
        assert tuple(s) == data

    @pytest.mark.parametrize(
        "data, expected_values, expected_index",
        [
            ({(1, 2): 3, (None, 5): 6}, [3, 6], [(1, 2), (None, 5)]),
            ({(1,): 3, (4, 5): 6}, [3, 6], [(1, None), (4, 5)]),
        ],
    )
    def test_constructor_dict_of_tuples(self, data, expected_values, expected_index):
        # [FUNCTION DOC]
        # Verifies (issue 60695) that a dict with tuple keys of equal or unequal length builds a multi-index, padding shorter tuples with a missing marker; sorted by value. Parametrized over data, expected values and expected index tuples.
        # GH 60695
        result = Series(data).sort_values()
        expected = Series(expected_values, index=MultiIndex.from_tuples(expected_index))
        tm.assert_series_equal(result, expected)

    # https://github.com/pandas-dev/pandas/issues/22698
    @pytest.mark.filterwarnings("ignore:elementwise comparison:FutureWarning")
    def test_fromDict(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies (issue 22698) construction from dicts: integer values give a sorted index; mixed integer, strings and datetime give object dtype; all-string-and-integer values give object (or string dtype when infer-string); a dict of numeric strings with float dtype gives float64. An elementwise-comparison future warning is filtered.
        # KNOWN ISSUE: the dtype assertion for the all-string case has unparenthesized conditional precedence, so with infer-string on it asserts a non-empty constant and always passes.
        data = {"a": 0, "b": 1, "c": 2, "d": 3}

        series = Series(data)
        tm.assert_is_sorted(series.index)

        data = {"a": 0, "b": "1", "c": "2", "d": datetime.now()}
        series = Series(data)
        assert series.dtype == np.object_

        data = {"a": 0, "b": "1", "c": "2", "d": "3"}
        series = Series(data)
        assert series.dtype == np.object_ if not using_infer_string else "str"

        data = {"a": "0", "b": "1"}
        series = Series(data, dtype=float)
        assert series.dtype == np.float64

    def test_fromValue(self, datetime_series, using_infer_string):
        # [FUNCTION DOC]
        # Verifies broadcasting scalars to the datetime fixture's index: NaN with float64 gives float64; a string gives object dtype (or string dtype when infer-string); a current datetime gives microsecond datetime64 dtype; and (issue 12336) an integer with category dtype equals casting the integer Series to category.
        # KNOWN ISSUE: the object-or-string dtype assertion has unparenthesized conditional precedence, so with infer-string on it asserts a non-empty constant and always passes.
        nans = Series(np.nan, index=datetime_series.index, dtype=np.float64)
        assert nans.dtype == np.float64
        assert len(nans) == len(datetime_series)

        strings = Series("foo", index=datetime_series.index)
        assert strings.dtype == np.object_ if not using_infer_string else "str"
        assert len(strings) == len(datetime_series)

        d = datetime.now()
        dates = Series(d, index=datetime_series.index)
        assert dates.dtype == "M8[us]"
        assert len(dates) == len(datetime_series)

        # GH12336
        # Test construction of categorical series from value
        categorical = Series(0, index=datetime_series.index, dtype="category")
        expected = Series(0, index=datetime_series.index).astype("category")
        assert categorical.dtype == "category"
        assert len(categorical) == len(datetime_series)
        tm.assert_series_equal(categorical, expected)

    def test_constructor_dtype_timedelta64(self):
        # [FUNCTION DOC]
        # Verifies timedelta64 construction and inference: lists of Python timedeltas infer microsecond dtype; mixing with NaT or NaN and nanosecond dtype gives nanosecond; NumPy nanosecond timedeltas with NaT or NaN infer nanosecond (issue 5689); with the integer NaT sentinel the result is object (integer not coerced); a second-resolution NumPy timedelta keeps second dtype; casting to int64 works and to int32 raises a type error saying the conversion is unsupported; a timedelta with an invalid string and a timedelta dtype raises a value error that conversion failed; the same list without dtype stays object; and lists of None/NaN/NaT with a duration string stay object (as of version 2.0).
        # basic
        td = Series([timedelta(days=i) for i in range(3)])
        assert td.dtype == "timedelta64[us]"

        td = Series([timedelta(days=1)])
        assert td.dtype == "timedelta64[us]"

        td = Series([timedelta(days=1), timedelta(days=2), np.timedelta64(1, "s")])

        assert td.dtype == "timedelta64[us]"

        # mixed with NaT
        td = Series([timedelta(days=1), NaT], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        td = Series([timedelta(days=1), np.nan], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        td = Series([np.timedelta64(300000000, "ns"), NaT], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        # improved inference
        # GH5689
        td = Series([np.timedelta64(300000000, "ns"), NaT])
        assert td.dtype == "timedelta64[ns]"

        # because iNaT is int, not coerced to timedelta
        td = Series([np.timedelta64(300000000, "ns"), iNaT])
        assert td.dtype == "object"

        td = Series([np.timedelta64(300000000, "ns"), np.nan])
        assert td.dtype == "timedelta64[ns]"

        td = Series([NaT, np.timedelta64(300000000, "ns")])
        assert td.dtype == "timedelta64[ns]"

        td = Series([np.timedelta64(1, "s")])
        assert td.dtype == "timedelta64[s]"

        # valid astype
        td.astype("int64")

        # invalid casting
        msg = r"Converting from timedelta64\[s\] to int32 is not supported"
        with pytest.raises(TypeError, match=msg):
            td.astype("int32")

        # this is an invalid casting
        msg = "|".join(
            [
                "Could not convert object to NumPy timedelta",
                "Could not convert 'foo' to NumPy timedelta",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            Series([timedelta(days=1), "foo"], dtype="m8[ns]")

        # leave as object here
        td = Series([timedelta(days=i) for i in range(3)] + ["foo"])
        assert td.dtype == "object"

        # as of 2.0, these no longer infer timedelta64 based on the strings,
        #  matching Index behavior
        ser = Series([None, NaT, "1 Day"])
        assert ser.dtype == object

        ser = Series([np.nan, NaT, "1 Day"])
        assert ser.dtype == object

        ser = Series([NaT, None, "1 Day"])
        assert ser.dtype == object

        ser = Series([NaT, np.nan, "1 Day"])
        assert ser.dtype == object

    # GH 16406
    def test_constructor_mixed_tz(self):
        # [FUNCTION DOC]
        # Verifies (issue 16406) that a naive and a tz-aware timestamp together give object dtype with both retained.
        s = Series([Timestamp("20130101"), Timestamp("20130101", tz="US/Eastern")])
        expected = Series(
            [Timestamp("20130101"), Timestamp("20130101", tz="US/Eastern")],
            dtype="object",
        )
        tm.assert_series_equal(s, expected)

    def test_NaT_scalar(self):
        # [FUNCTION DOC]
        # Verifies that a nanosecond datetime Series built from integers including the NaT sentinel has a missing last value, and assigning that value elsewhere keeps the position missing.
        series = Series([0, 1000, 2000, iNaT], dtype="M8[ns]")

        val = series[3]
        assert isna(val)

        series[2] = val
        assert isna(series[2])

    def test_NaT_cast(self):
        # [FUNCTION DOC]
        # Verifies (issue 10747) casting a NaN Series to nanosecond datetime gives NaT.
        # GH10747
        result = Series([np.nan]).astype("M8[ns]")
        expected = Series([NaT], dtype="M8[ns]")
        tm.assert_series_equal(result, expected)

    def test_constructor_name_hashable(self):
        # [FUNCTION DOC]
        # Verifies that hashable names (numbers, string, datetime, tuple, non-ASCII string) are stored as the name for list, array and dict inputs.
        for n in [777, 777.0, "name", datetime(2001, 11, 11), (1,), "\u05d0"]:
            for data in [[1, 2, 3], np.ones(3), {"a": 0, "b": 1}]:
                s = Series(data, name=n)
                assert s.name == n

    def test_constructor_name_unhashable(self):
        # [FUNCTION DOC]
        # Verifies that unhashable names (list, array, dict) raise a type error that the name must be hashable, for each combination with list, array and dict data.
        msg = r"Series\.name must be a hashable type"
        for n in [["name_list"], np.ones(2), {1: 2}]:
            for data in [["name_list"], np.ones(2), {1: 2}]:
                with pytest.raises(TypeError, match=msg):
                    Series(data, name=n)

    def test_auto_conversion(self):
        # [FUNCTION DOC]
        # Verifies that a list of nanosecond timestamps automatically gives nanosecond datetime64 dtype.
        series = Series(list(date_range("1/1/2000", periods=10, unit="ns")))
        assert series.dtype == "M8[ns]"

    def test_convert_non_ns(self):
        # [FUNCTION DOC]
        # Verifies that NumPy arrays of non-nanosecond timedelta (seconds), datetime in days (becomes seconds) and datetime in seconds keep (or convert to supported) resolution and equal the equivalent index-built Series.
        # convert from a numpy array of non-ns timedelta64
        arr = np.array([1, 2, 3], dtype="timedelta64[s]")
        ser = Series(arr)
        assert ser.dtype == arr.dtype

        tdi = timedelta_range("00:00:01", periods=3, freq="s").as_unit("s")
        expected = Series(tdi)
        assert expected.dtype == arr.dtype
        tm.assert_series_equal(ser, expected)

        # convert from a numpy array of non-ns datetime64
        arr = np.array(
            ["2013-01-01", "2013-01-02", "2013-01-03"], dtype="datetime64[D]"
        )
        ser = Series(arr)
        expected = Series(date_range("20130101", periods=3, freq="D"), dtype="M8[s]")
        assert expected.dtype == "M8[s]"
        tm.assert_series_equal(ser, expected)

        arr = np.array(
            ["2013-01-01 00:00:01", "2013-01-01 00:00:02", "2013-01-01 00:00:03"],
            dtype="datetime64[s]",
        )
        ser = Series(arr)
        expected = Series(
            date_range("20130101 00:00:01", periods=3, freq="s"), dtype="M8[s]"
        )
        assert expected.dtype == "M8[s]"
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "index",
        [
            date_range("1/1/2000", periods=10),
            timedelta_range("1 day", periods=10),
            period_range("2000-Q1", periods=10, freq="Q"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_cant_cast_datetimelike(self, index):
        # [FUNCTION DOC]
        # Verifies that a Series from a datetime, timedelta or period index with float dtype raises a type error "Cannot cast ... to" (parametrized over the three index types), while int64 dtype gives the same values as the integer cast of the index.
        # floats are not ok
        # strip Index to convert PeriodIndex -> Period
        # We don't care whether the error message says
        # PeriodIndex or PeriodArray
        msg = f"Cannot cast {type(index).__name__.rstrip('Index')}.*? to "

        with pytest.raises(TypeError, match=msg):
            Series(index, dtype=float)

        # ints are ok
        # we test with np.int64 to get similar results on
        # windows / 32-bit platforms
        result = Series(index, dtype=np.int64)
        expected = Series(index.astype(np.int64))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            date_range("1/1/2000", periods=10),
            timedelta_range("1 day", periods=10),
            period_range("2000-Q1", periods=10, freq="Q"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_cast_object(self, index):
        # [FUNCTION DOC]
        # Verifies that datetime, timedelta and period indexes with object dtype give a Series equal to the object-cast Series, also when passing an object Index or the object-cast index.
        s = Series(index, dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

        s = Series(Index(index, dtype=object), dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

        s = Series(index.astype(object), dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    def test_constructor_generic_timestamp_no_frequency(self, dtype, request):
        # [FUNCTION DOC]
        # Verifies (issues 15524, 15987) that the generic NumPy datetime64 or timedelta64 class (without unit) as dtype raises a value error saying the dtype has no unit. Conditionally marked expected-to-fail when the dtype name is not one of those two (issue 33890).
        # see gh-15524, gh-15987
        msg = "dtype has no unit. Please pass in"

        if np.dtype(dtype).name not in ["timedelta64", "datetime64"]:
            mark = pytest.mark.xfail(reason="GH#33890 Is assigned ns unit")
            request.applymarker(mark)

        with pytest.raises(ValueError, match=msg):
            Series([], dtype=dtype)

    @pytest.mark.parametrize("unit", ["ps", "as", "fs", "Y", "M", "W", "D", "h", "m"])
    @pytest.mark.parametrize("kind", ["m", "M"])
    def test_constructor_generic_timestamp_bad_frequency(self, kind, unit):
        # [FUNCTION DOC]
        # Verifies (issues 15524, 15987) that datetime or timedelta dtypes with unsupported units (picoseconds, attoseconds, femtoseconds, years, months, weeks, days, hours, minutes) raise a type error saying the dtype is unsupported and listing supported resolutions, both for a Series and for a one-cell DataFrame. Parametrized over unit and kind.
        # see gh-15524, gh-15987
        # as of 2.0 we raise on any non-supported unit rather than silently
        #  cast to nanos; previously we only raised for frequencies higher
        #  than ns
        dtype = f"{kind}8[{unit}]"

        msg = "dtype=.* is not supported. Supported resolutions are"
        with pytest.raises(TypeError, match=msg):
            Series([], dtype=dtype)

        with pytest.raises(TypeError, match=msg):
            # pre-2.0 the DataFrame cast raised but the Series case did not
            DataFrame([[0]], dtype=dtype)

    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    def test_constructor_range_dtype(self, dtype):
        # [FUNCTION DOC]
        # Verifies (issue 16804) that a Series from a range with no dtype, unsigned 8-bit or category equals the same values built from a list with the dtype (int64 by default). Parametrized over dtype.
        # GH 16804
        expected = Series([0, 1, 2, 3, 4], dtype=dtype or "int64")
        result = Series(range(5), dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_constructor_range_overflows(self):
        # [FUNCTION DOC]
        # Verifies (issue 30173) construction from ranges beyond int64: ranges starting at 2 to the 63rd (ascending or descending) give unsigned 64-bit dtype; a range going below the signed minimum, or starting at 2 to the 73rd, gives object dtype. In each case values equal those built from the list and the iteration matches the range.
        # GH#30173 range objects that overflow int64
        rng = range(2**63, 2**63 + 4)
        ser = Series(rng)
        expected = Series(list(rng))
        tm.assert_series_equal(ser, expected)
        assert list(ser) == list(rng)
        assert ser.dtype == np.uint64

        rng2 = range(2**63 + 4, 2**63, -1)
        ser2 = Series(rng2)
        expected2 = Series(list(rng2))
        tm.assert_series_equal(ser2, expected2)
        assert list(ser2) == list(rng2)
        assert ser2.dtype == np.uint64

        rng3 = range(-(2**63), -(2**63) - 4, -1)
        ser3 = Series(rng3)
        expected3 = Series(list(rng3))
        tm.assert_series_equal(ser3, expected3)
        assert list(ser3) == list(rng3)
        assert ser3.dtype == object

        rng4 = range(2**73, 2**73 + 4)
        ser4 = Series(rng4)
        expected4 = Series(list(rng4))
        tm.assert_series_equal(ser4, expected4)
        assert list(ser4) == list(rng4)
        assert ser4.dtype == object

    def test_constructor_tz_mixed_data(self):
        # [FUNCTION DOC]
        # Verifies (issue 13051) that a naive and a tz-aware timestamp together give object dtype.
        # GH 13051
        dt_list = [
            Timestamp("2016-05-01 02:03:37"),
            Timestamp("2016-04-30 19:03:37-0700", tz="US/Pacific"),
        ]
        result = Series(dt_list)
        expected = Series(dt_list, dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("pydt", [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
        # [FUNCTION DOC]
        # Verifies (issues 25843, 41555, 33401) that zone-aware data (a timestamp or a Python datetime, parametrized, in each zone from the aware-zone fixture) with a naive datetime64 dtype raises a value error telling the user to localize to none instead; this holds for a list, an object array and a dict; a scalar broadcast with an index raises a type error about unboxing aware to naive.
        # GH#25843, GH#41555, GH#33401
        tz = tz_aware_fixture
        ts = Timestamp("2019", tz=tz)
        if pydt:
            ts = ts.to_pydatetime()

        msg = (
            "Cannot convert timezone-aware data to timezone-naive dtype. "
            r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
        )
        with pytest.raises(ValueError, match=msg):
            Series([ts], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            Series(np.array([ts], dtype=object), dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            Series({0: ts}, dtype="datetime64[ns]")

        msg = "Cannot unbox tzaware Timestamp to tznaive dtype"
        with pytest.raises(TypeError, match=msg):
            Series(ts, index=[0], dtype="datetime64[ns]")

    def test_constructor_datetime64(self):
        # [FUNCTION DOC]
        # Verifies that a NumPy datetime array from a ten-second range gives a Series whose dtype is a sub-dtype of nanosecond datetime64.
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        dates = np.asarray(rng)

        series = Series(dates)
        assert np.issubdtype(series.dtype, np.dtype("M8[ns]"))

    def test_constructor_datetimelike_scalar_to_string_dtype(
        self, nullable_string_dtype
    ):
        # [FUNCTION DOC]
        # Verifies (pull request 33846) that a scalar string that looks like a datetime unit (M) is broadcast with a nullable string dtype (fixture) to the same Series as a list of that string.
        # https://github.com/pandas-dev/pandas/pull/33846
        result = Series("M", index=[1, 2, 3], dtype=nullable_string_dtype)
        expected = Series(["M", "M", "M"], index=[1, 2, 3], dtype=nullable_string_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("box", [lambda x: x, np.datetime64])
    def test_constructor_sparse_datetime64(self, box):
        # [FUNCTION DOC]
        # Verifies (issue 35762) that datetime values (as strings or NumPy datetimes, parametrized) with a sparse nanosecond datetime dtype equal the Series built from a sparse array of the same.
        # https://github.com/pandas-dev/pandas/issues/35762
        values = [box("2012-01-01"), box("2013-01-01")]
        dtype = pd.SparseDtype("datetime64[ns]")
        result = Series(values, dtype=dtype)
        arr = pd.arrays.SparseArray(values, dtype=dtype)
        expected = Series(arr)
        tm.assert_series_equal(result, expected)

    def test_construction_from_ordered_collection(self):
        # [FUNCTION DOC]
        # Verifies (issue 36044) that views of a dictionary's keys and values (ordered collections) are accepted as data.
        # https://github.com/pandas-dev/pandas/issues/36044
        result = Series({"a": 1, "b": 2}.keys())
        expected = Series(["a", "b"])
        tm.assert_series_equal(result, expected)

        result = Series({"a": 1, "b": 2}.values())
        expected = Series([1, 2])
        tm.assert_series_equal(result, expected)

    def test_construction_from_large_int_scalar_no_overflow(self):
        # [FUNCTION DOC]
        # Verifies (issue 36291) that a huge Python integer scalar with an index works the same with and without the index (no overflow).
        # https://github.com/pandas-dev/pandas/issues/36291
        n = 1_000_000_000_000_000_000_000
        result = Series(n, index=[0])
        expected = Series(n)
        tm.assert_series_equal(result, expected)

    def test_constructor_list_of_periods_infers_period_dtype(self):
        # [FUNCTION DOC]
        # Verifies that a list of period objects with the same frequency (from a range or built by hand) infers daily period dtype.
        series = Series(list(period_range("2000-01-01", periods=10, freq="D")))
        assert series.dtype == "Period[D]"

        series = Series(
            [Period("2011-01-01", freq="D"), Period("2011-02-01", freq="D")]
        )
        assert series.dtype == "Period[D]"

    def test_constructor_subclass_dict(self, dict_subclass):
        # [FUNCTION DOC]
        # Verifies that a dict subclass (fixture) with numeric keys and float values builds the same Series as a plain dict of its items.
        data = dict_subclass((x, 10.0 * x) for x in range(10))
        series = Series(data)
        expected = Series(dict(data.items()))
        tm.assert_series_equal(series, expected)

    def test_constructor_ordereddict(self):
        # [FUNCTION DOC]
        # Verifies (issue 3283) that an ordered dictionary of twelve random entries, and a subclass of it, builds a Series with keys as index in order and values as data.
        # GH3283
        data = OrderedDict(
            (f"col{i}", np.random.default_rng(2).random()) for i in range(12)
        )

        series = Series(data)
        expected = Series(list(data.values()), list(data.keys()))
        tm.assert_series_equal(series, expected)

        # Test with subclass
        class A(OrderedDict):
            # [CLASS DOC]
            # HELPER: trivial local subclass of the ordered dictionary type used to test dictionary subclass handling.
            pass

        series = Series(A(data))
        tm.assert_series_equal(series, expected)

    @pytest.mark.parametrize(
        "data, expected_index_multi",
        [
            ({("a", "a"): 0.0, ("b", "a"): 1.0, ("b", "c"): 2.0}, True),
            ({("a",): 0.0, ("a", "b"): 1.0}, True),
            ({"z": 111.0, ("a", "a"): 0.0, ("b", "a"): 1.0, ("b", "c"): 2.0}, False),
        ],
    )
    def test_constructor_dict_multiindex(self, data, expected_index_multi):
        # [FUNCTION DOC]
        # Verifies (issue 60695) dict construction with tuple keys: when all keys are tuples (even of unequal length) the index is a multi-index; when a non-tuple key is present the index is a plain object index of the keys. Parametrized over data and whether a multi-index is expected.
        # GH#60695
        result = Series(data)

        if expected_index_multi:
            expected = Series(
                list(data.values()),
                index=MultiIndex.from_tuples(list(data.keys())),
            )
            tm.assert_series_equal(result, expected)
        else:
            expected = Series(
                list(data.values()),
                index=Index(list(data.keys())),
            )
            tm.assert_series_equal(result, expected)

    def test_constructor_dict_multiindex_reindex_flat(self):
        # [FUNCTION DOC]
        # Verifies the corner case of reindexing with a mixed index (tuple keys plus a flat key containing NaN): rebuilding from the dict of all but the last element with the full index equals the original Series.
        # construction involves reindexing with a MultiIndex corner case
        data = {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2, "j": np.nan}
        expected = Series(data)

        result = Series(expected[:-1].to_dict(), index=expected.index)
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_timedelta_index(self):
        # [FUNCTION DOC]
        # Verifies (issue 12169) that a dict keyed by timedeltas with a timedelta index yields the data (not NaN), equal to the Series built from list data and the same index.
        # GH #12169 : Resample category data with timedelta index
        # construct Series from dict as data and TimedeltaIndex as index
        # will result NaN in result Series data
        expected = Series(
            data=["A", "B", "C"], index=pd.to_timedelta([0, 10, 20], unit="s")
        )

        result = Series(
            data={
                pd.to_timedelta(0, unit="s"): "A",
                pd.to_timedelta(10, unit="s"): "B",
                pd.to_timedelta(20, unit="s"): "C",
            },
            index=pd.to_timedelta([0, 10, 20], unit="s"),
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_infer_index_tz(self):
        # [FUNCTION DOC]
        # Verifies that an index of datetimes with a fixed UTC offset time zone object infers a datetime index with that timezone, and (issue 2443) that the text form of its first element does not raise.
        values = [188.5, 328.25]
        tzinfo = tzoffset(None, 7200)
        index = [
            datetime(2012, 5, 11, 11, tzinfo=tzinfo),
            datetime(2012, 5, 11, 12, tzinfo=tzinfo),
        ]
        series = Series(data=values, index=index)

        assert series.index.tz == tzinfo

        # it works! GH#2443
        repr(series.index[0])

    def test_constructor_with_pandas_dtype(self):
        # [FUNCTION DOC]
        # Verifies that data going through the two-dimensional to one-dimensional path (a list of one-element tuples) gives the same result when the NumPy extension dtype taken from the first Series is passed explicitly.
        # going through 2D->1D path
        vals = [(1,), (2,), (3,)]
        ser = Series(vals)
        dtype = ser.array.dtype  # NumpyEADtype
        ser2 = Series(vals, dtype=dtype)
        tm.assert_series_equal(ser, ser2)

    def test_constructor_int_dtype_missing_values(self):
        # [FUNCTION DOC]
        # Verifies (issue 43017) that an index with no data and int64 dtype is filled with NaN, giving float64.
        # GH#43017
        result = Series(index=[0], dtype="int64")
        expected = Series(np.nan, index=[0], dtype="float64")
        tm.assert_series_equal(result, expected)

    def test_constructor_bool_dtype_missing_values(self):
        # [FUNCTION DOC]
        # Verifies (issue 43018) that an index with no data and boolean dtype is filled with True.
        # GH#43018
        result = Series(index=[0], dtype="bool")
        expected = Series(True, index=[0], dtype="bool")
        tm.assert_series_equal(result, expected)

    def test_constructor_int64_dtype(self, any_int_dtype):
        # [FUNCTION DOC]
        # Verifies (issue 44923) that numeric strings convert to any integer dtype (fixture), equal to the numeric list with that dtype.
        # GH#44923
        result = Series(["0", "1", "2"], dtype=any_int_dtype)
        expected = Series([0, 1, 2], dtype=any_int_dtype)
        tm.assert_series_equal(result, expected)

    def test_constructor_raise_on_lossy_conversion_of_strings(self):
        # [FUNCTION DOC]
        # Verifies (issue 44923) that a numeric string too large for int8 raises an overflow error (NumPy 2 message) or a value error about lossless cast (older NumPy).
        # GH#44923
        if not np_version_gt2:
            raises = pytest.raises(
                ValueError, match="string values cannot be losslessly cast to int8"
            )
        else:
            raises = pytest.raises(
                OverflowError, match="The elements provided in the data"
            )
        with raises:
            Series(["128"], dtype="int8")

    def test_constructor_dtype_timedelta_alternative_construct(self):
        # [FUNCTION DOC]
        # Verifies (issue 35465) that integers with microsecond timedelta dtype equal a Series from converting with the microsecond unit.
        # GH#35465
        result = Series([1000000, 200000, 3000000], dtype="timedelta64[us]")
        expected = Series(pd.to_timedelta([1000000, 200000, 3000000], unit="us"))
        tm.assert_series_equal(result, expected)

    def test_constructor_dtype_timedelta_ns_s_astype_int64(self):
        # [FUNCTION DOC]
        # Verifies (issue 35465) that casting nanosecond and second timedelta Series built from the same integers to int64 gives equal results.
        # GH#35465
        result = Series([1000000, 200000, 3000000], dtype="timedelta64[ns]").astype(
            "int64"
        )
        expected = Series([1000000, 200000, 3000000], dtype="timedelta64[s]").astype(
            "int64"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning"
    )
    @pytest.mark.parametrize("func", [Series, DataFrame, Index, pd.array])
    def test_constructor_mismatched_null_nullable_dtype(
        self, func, any_numeric_ea_dtype
    ):
        # [FUNCTION DOC]
        # Verifies (issue 44514) that a missing marker of a mismatched kind (each NumPy datetime/timedelta NaT object and the pandas NaT) among floats raises a type error with one of several library-dependent messages when a nullable numeric dtype (fixture) is requested, for Series, DataFrame, Index and the array factory (parametrized). Elementwise comparison deprecation warnings are filtered.
        # GH#44514
        msg = "|".join(
            [
                "cannot safely cast non-equivalent object",
                r"int\(\) argument must be a string, a bytes-like object "
                "or a (real )?number",
                r"Cannot cast array data from dtype\('O'\) to dtype\('float64'\) "
                "according to the rule 'safe'",
                "object cannot be converted to a FloatingDtype",
                "'values' contains non-numeric NA",
            ]
        )

        for null in [*tm.NP_NAT_OBJECTS, NaT]:
            with pytest.raises(TypeError, match=msg):
                func([null, 1.0, 3.0], dtype=any_numeric_ea_dtype)

    def test_series_constructor_ea_int_from_bool(self):
        # [FUNCTION DOC]
        # Verifies (issue 42137) that booleans, with or without a missing value, convert to the nullable Int64 dtype as zero and one.
        # GH#42137
        result = Series([True, False, True, pd.NA], dtype="Int64")
        expected = Series([1, 0, 1, pd.NA], dtype="Int64")
        tm.assert_series_equal(result, expected)

        result = Series([True, False, True], dtype="Int64")
        expected = Series([1, 0, 1], dtype="Int64")
        tm.assert_series_equal(result, expected)

    def test_series_constructor_ea_int_from_string_bool(self):
        # [FUNCTION DOC]
        # Verifies (issue 42137) that the strings True and False with Int64 dtype raise a value error about an invalid literal.
        # GH#42137
        with pytest.raises(ValueError, match="invalid literal"):
            Series(["True", "False", "True", pd.NA], dtype="Int64")

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_series_constructor_overflow_uint_ea(self, val):
        # [FUNCTION DOC]
        # Verifies (issue 38798) that a value near the maximum of unsigned 64-bit and a small number (integer or float, parametrized) construct with nullable UInt64 dtype without overflow.
        # GH#38798
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, val], dtype="UInt64")
        expected = Series(np.array([max_val, 1], dtype="uint64"), dtype="UInt64")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_series_constructor_overflow_uint_ea_with_na(self, val):
        # [FUNCTION DOC]
        # Verifies (issue 38798) the same large-value construction with an additional missing value: the result equals an integer array with a mask marking the third element.
        # GH#38798
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, val, pd.NA], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([max_val, 1, 0], dtype="uint64"),
                np.array([0, 0, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_constructor_overflow_uint_with_nan(self):
        # [FUNCTION DOC]
        # Verifies (issue 38798) a large unsigned value plus a missing value with nullable UInt64 dtype equals an integer array with a mask marking the second element (the data placeholder under the mask is one).
        # GH#38798
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, pd.NA], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([max_val, 1], dtype="uint64"),
                np.array([0, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_constructor_ea_all_na(self):
        # [FUNCTION DOC]
        # Verifies (issue 38798) a Series of only missing values with nullable UInt64 dtype equals an integer array with a full mask.
        # GH#38798
        result = Series([pd.NA, pd.NA], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([1, 1], dtype="uint64"),
                np.array([1, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_from_index_dtype_equal_does_not_copy(self):
        # [FUNCTION DOC]
        # Verifies (issue 52008) that building from an Index with the same dtype does not share memory: assigning into the Series leaves the Index unchanged.
        # GH#52008
        idx = Index([1, 2, 3])
        expected = idx.copy(deep=True)
        ser = Series(idx, dtype="int64")
        ser.iloc[0] = 100
        tm.assert_index_equal(idx, expected)

    def test_series_string_inference(self):
        # [FUNCTION DOC]
        # Verifies (issue 54430) that with the future infer-string option on, a list of strings infers the NaN-variant string dtype (pyarrow storage when pyarrow is installed, otherwise python), while a mixed list of a string and an integer stays object.
        # GH#54430
        with pd.option_context("future.infer_string", True):
            ser = Series(["a", "b"])
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series(["a", "b"], dtype=dtype)
        tm.assert_series_equal(ser, expected)

        expected = Series(["a", 1], dtype="object")
        with pd.option_context("future.infer_string", True):
            ser = Series(["a", 1])
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("na_value", [None, np.nan, pd.NA])
    def test_series_string_with_na_inference(self, na_value):
        # [FUNCTION DOC]
        # Verifies (issue 54430) that with the infer-string option on, a string with a missing value (none, NaN or NA; parametrized) infers the NaN-variant string dtype with the missing value as None-equivalent.
        # GH#54430
        with pd.option_context("future.infer_string", True):
            ser = Series(["a", na_value])
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series(["a", None], dtype=dtype)
        tm.assert_series_equal(ser, expected)

    def test_series_string_inference_scalar(self):
        # [FUNCTION DOC]
        # Verifies (issue 54430) that with the infer-string option on, a scalar string broadcast over an index infers the NaN-variant string dtype.
        # GH#54430
        with pd.option_context("future.infer_string", True):
            ser = Series("a", index=[1])
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series("a", index=[1], dtype=dtype)
        tm.assert_series_equal(ser, expected)

    def test_series_string_inference_array_string_dtype(self):
        # [FUNCTION DOC]
        # Verifies (issue 54496) that with the infer-string option on, a NumPy fixed-width unicode array infers the NaN-variant string dtype.
        # GH#54496
        with pd.option_context("future.infer_string", True):
            ser = Series(np.array(["a", "b"]))
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series(["a", "b"], dtype=dtype)
        tm.assert_series_equal(ser, expected)

    def test_series_string_inference_storage_definition(self):
        # [FUNCTION DOC]
        # Verifies the dtype named string with infer-string on returns the NA-variant string dtype (pyarrow storage if available, else python), and the dtype named str returns the NaN-variant string dtype, as decided by the string dtype proposal (issue 54793).
        # https://github.com/pandas-dev/pandas/issues/54793
        # but after PDEP-14 (string dtype), it was decided to keep dtype="string"
        # returning the NA string dtype, so expected is changed from
        # "string[pyarrow_numpy]" to "string[python]"
        expected = Series(
            ["a", "b"], dtype="string[pyarrow]" if HAS_PYARROW else "string[python]"
        )
        with pd.option_context("future.infer_string", True):
            result = Series(["a", "b"], dtype="string")
        tm.assert_series_equal(result, expected)

        expected = Series(["a", "b"], dtype=pd.StringDtype(na_value=np.nan))
        with pd.option_context("future.infer_string", True):
            result = Series(["a", "b"], dtype="str")
        tm.assert_series_equal(result, expected)

    def test_series_constructor_infer_string_scalar(self):
        # [FUNCTION DOC]
        # Verifies (issue 55537) that an explicit python-storage string dtype with a scalar and infer-string on stays python-storage and equals the expected Series.
        # GH#55537
        with pd.option_context("future.infer_string", True):
            ser = Series("a", index=[1, 2], dtype="string[python]")
        expected = Series(["a", "a"], index=[1, 2], dtype="string[python]")
        tm.assert_series_equal(ser, expected)
        assert ser.dtype.storage == "python"

    def test_series_string_inference_na_first(self):
        # [FUNCTION DOC]
        # Verifies (issue 55655) that with infer-string on, a list starting with NA followed by a string infers the NaN-variant string dtype with the missing value first.
        # GH#55655
        with pd.option_context("future.infer_string", True):
            result = Series([pd.NA, "b"])
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series([None, "b"], dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("klass", [Series, Index])
    def test_inference_on_pandas_objects(self, klass):
        # [FUNCTION DOC]
        # Verifies (issue 56012) that building a Series from a pandas Series or Index of object dtype holding a timestamp keeps object dtype (no inference). Parametrized over Series and Index.
        # GH#56012
        obj = klass([Timestamp("2019-12-31")], dtype=object)
        # This doesn't do inference
        result = Series(obj)
        assert result.dtype == np.object_


class TestSeriesConstructorIndexCoercion:
    # [CLASS DOC]
    # Groups tests of how the Series constructor coerces its index argument. Stateless.
    def test_series_constructor_datetimelike_index_coercion(self):
        # [FUNCTION DOC]
        # Verifies (issues 39307, 23598) that an object-dtype index of datetimes is not silently cast to a datetime index as of version 2.0.
        idx = date_range("2020-01-01", periods=5)
        ser = Series(
            np.random.default_rng(2).standard_normal(len(idx)), idx.astype(object)
        )
        # as of 2.0, we no longer silently cast the object-dtype index
        #  to DatetimeIndex GH#39307, GH#23598
        assert not isinstance(ser.index, DatetimeIndex)

    @pytest.mark.parametrize("container", [None, np.array, Series, Index])
    @pytest.mark.parametrize("data", [1.0, range(4)])
    def test_series_constructor_infer_multiindex(self, container, data):
        # [FUNCTION DOC]
        # Verifies that a list of two label lists, or two NumPy arrays, Series or Index (parametrized container), passed as the index with a scalar or range of four values builds a multi-index.
        indexes = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]
        if container is not None:
            indexes = [container(ind) for ind in indexes]

        multi = Series(data, index=indexes)
        assert isinstance(multi.index, MultiIndex)

    # TODO: make this not cast to object in pandas 3.0
    @pytest.mark.skipif(
        not np_version_gt2, reason="StringDType only available in numpy 2 and above"
    )
    @pytest.mark.parametrize(
        "data",
        [
            ["a", "b", "c"],
            ["a", "b", np.nan],
        ],
    )
    def test_np_string_array_object_cast(self, data):
        # [FUNCTION DOC]
        # Verifies that a NumPy variable-length string dtype array (NumPy 2 and above only; skipped otherwise), without or with a NaN, builds an object-dtype Series. In the NaN case the last element reads back as the text nan (comparison semantics changed by issue 62522), while the preceding elements equal the input; without NaN all elements equal the input. A TODO states this should stop casting to object.
        from numpy.dtypes import StringDType

        arr = np.array(data, dtype=StringDType())
        res = Series(arr)
        assert res.dtype == np.object_

        if data[-1] is np.nan:
            # as of GH#62522 the comparison op for `res==data` casts data
            #  using sanitize_array, which casts to 'str' dtype, which does not
            #  consider string 'nan' to be equal to np.nan,
            #  (which apparently numpy does?  weird.)
            assert (res.iloc[:-1] == data[:-1]).all()
            assert res.iloc[-1] == "nan"
        else:
            assert (res == data).all()


class TestSeriesConstructorInternals:
    # [CLASS DOC]
    # Groups tests that inspect the internal block structure of Series built via the constructor. Stateless.
    def test_constructor_no_pandas_array(self):
        # [FUNCTION DOC]
        # Verifies that building a Series from another Series' array gives an equal Series whose first internal block is a numeric NumPy-backed block (not a wrapped extension array).
        ser = Series([1, 2, 3])
        result = Series(ser.array)
        tm.assert_series_equal(ser, result)
        assert isinstance(result._mgr.blocks[0], NumpyBlock)
        assert result._mgr.blocks[0].is_numeric

    def test_from_array(self):
        # [FUNCTION DOC]
        # Verifies that timedelta and datetime arrays built from strings with nanosecond dtype produce a Series whose internal block is not an extension block.
        result = Series(pd.array(["1h", "2h"], dtype="timedelta64[ns]"))
        assert result._mgr.blocks[0].is_extension is False

        result = Series(pd.array(["2015"], dtype="datetime64[ns]"))
        assert result._mgr.blocks[0].is_extension is False

    def test_from_list_dtype(self):
        # [FUNCTION DOC]
        # Verifies that lists of strings with nanosecond timedelta or datetime dtype produce a non-extension internal block.
        result = Series(["1h", "2h"], dtype="timedelta64[ns]")
        assert result._mgr.blocks[0].is_extension is False

        result = Series(["2015"], dtype="datetime64[ns]")
        assert result._mgr.blocks[0].is_extension is False


def test_constructor(rand_series_with_duplicate_datetimeindex):
    # [FUNCTION DOC]
    # Verifies the duplicate-index fixture returns a Series with a datetime index.
    dups = rand_series_with_duplicate_datetimeindex
    assert isinstance(dups, Series)
    assert isinstance(dups.index, DatetimeIndex)


@pytest.mark.parametrize(
    "input_dict,expected",
    [
        ({0: 0}, np.array([[0]], dtype=np.int64)),
        ({"a": "a"}, np.array([["a"]], dtype=object)),
        ({1: 1}, np.array([[1]], dtype=np.int64)),
    ],
)
def test_numpy_array(input_dict, expected):
    # [FUNCTION DOC]
    # Verifies that wrapping a one-element Series in a NumPy array yields a two-dimensional array of one row with the element, of int64 dtype for integer values and object for a string value. Parametrized over input dict and expected array.
    result = np.array([Series(input_dict)])
    tm.assert_numpy_array_equal(result, expected)


def test_index_ordered_dict_keys():
    # [FUNCTION DOC]
    # Verifies (issue 22077) that an index built from the keys view of an ordered dictionary of nested tuples (one with None) becomes a multi-index of tuple elements.
    # GH 22077

    param_index = OrderedDict(
        [
            ((("a", "b"), ("c", "d")), 1),
            ((("a", None), ("c", "d")), 2),
        ]
    )
    series = Series([1, 2], index=param_index.keys())
    expected = Series(
        [1, 2],
        index=MultiIndex.from_tuples(
            [(("a", "b"), ("c", "d")), (("a", None), ("c", "d"))]
        ),
    )
    tm.assert_series_equal(series, expected)


@pytest.mark.parametrize(
    "input_list",
    [
        [1, complex("nan"), 2],
        [1 + 1j, complex("nan"), 2 + 2j],
    ],
)
def test_series_with_complex_nan(input_list):
    # [FUNCTION DOC]
    # Verifies (issue 53627) that lists of complex numbers or of integers with a complex NaN give complex128 dtype and that rebuilding from the Series' array gives an equal Series.
    # GH#53627
    ser = Series(input_list)
    result = Series(ser.array)
    assert ser.dtype == "complex128"
    tm.assert_series_equal(ser, result)


def test_dict_keys_rangeindex():
    # [FUNCTION DOC]
    # Verifies that a dict with consecutive integer keys starting at zero gives an index equal to a range index in type as well as value.
    result = Series({0: 1, 1: 2})
    expected = Series([1, 2], index=RangeIndex(2))
    tm.assert_series_equal(result, expected, check_index_type=True)
