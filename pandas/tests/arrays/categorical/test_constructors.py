# [MODULE DOC]
# Tests for constructing Categorical arrays: the main constructor (values, categories, ordered, dtype),
# the class method that builds from integer codes, and the private helper that builds from inferred
# categories. Covers validation errors, dtype handling, preservation of index types and the
# Pandas4Warning raised when values are not in the supplied categories. Uses fixtures ordered,
# nulls_fixture, any_numeric_ea_dtype and using_infer_string from the shared conftest. No import-time
# side effects.

from datetime import (
    date,
    datetime,
)

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestCategoricalConstructors:
    # [CLASS DOC]
    # Groups all Categorical construction tests. Stateless.
    def test_categorical_from_cat_and_dtype_str_preserve_ordered(self):
        # [FUNCTION DOC]
        # Constructing from an ordered Categorical with the dtype given as the category alias string must keep the
        # ordered flag of the input.
        # GH#49309 we should preserve orderedness in `res`
        cat = Categorical([3, 1], categories=[3, 2, 1], ordered=True)

        res = Categorical(cat, dtype="category")
        assert res.dtype.ordered

    def test_categorical_disallows_scalar(self):
        # [FUNCTION DOC]
        # A scalar input must raise a type error saying the input must be list-like.
        # GH#38433
        with pytest.raises(TypeError, match="Categorical input must be list-like"):
            Categorical("A", categories=["A", "B"])

    def test_categorical_1d_only(self):
        # [FUNCTION DOC]
        # A two dimensional input must raise a not-implemented error stating more than one dimension is unsupported.
        # ndim > 1
        msg = "> 1 ndim Categorical are not supported at this time"
        with pytest.raises(NotImplementedError, match=msg):
            Categorical(np.array([list("abcd")]))

    def test_validate_ordered(self):
        # [FUNCTION DOC]
        # A non-boolean ordered argument (an integer array) must raise a type error, both in the constructor and in
        # the codes-based class constructor.
        # see gh-14058
        exp_msg = "'ordered' must either be 'True' or 'False'"
        exp_err = TypeError

        # This should be a boolean.
        ordered = np.array([0, 1, 2])

        with pytest.raises(exp_err, match=exp_msg):
            Categorical([1, 2, 3], ordered=ordered)

        with pytest.raises(exp_err, match=exp_msg):
            Categorical.from_codes(
                [0, 0, 1], categories=["a", "b", "c"], ordered=ordered
            )

    def test_constructor_empty(self):
        # [FUNCTION DOC]
        # Empty input gives empty categories (default object-like index), or the given categories as 64-bit
        # integer index when provided.
        # GH 17248
        c = Categorical([])
        expected = Index([])
        tm.assert_index_equal(c.categories, expected)

        c = Categorical([], categories=[1, 2, 3])
        expected = Index([1, 2, 3], dtype=np.int64)
        tm.assert_index_equal(c.categories, expected)

    def test_constructor_empty_boolean(self):
        # [FUNCTION DOC]
        # Empty input with boolean categories keeps both boolean categories (compared after sorting).
        # see gh-22702
        cat = Categorical([], categories=[True, False])
        categories = sorted(cat.categories.tolist())
        assert categories == [False, True]

    def test_constructor_tuples(self):
        # [FUNCTION DOC]
        # Object array of tuples is treated as scalar categories; categories are the unique tuples and ordered is false.
        values = np.array([(1,), (1, 2), (1,), (1, 2)], dtype=object)
        result = Categorical(values)
        expected = Index([(1,), (1, 2)], tupleize_cols=False)
        tm.assert_index_equal(result.categories, expected)
        assert result.ordered is False

    def test_constructor_tuples_datetimes(self):
        # [FUNCTION DOC]
        # Tuples containing timestamps become categories without being converted to a multi-level index. A
        # trailing tuple of a different length is added and sliced off so numpy does not reshape the array.
        # numpy will auto reshape when all of the tuples are the
        # same len, so add an extra one with 2 items and slice it off
        values = np.array(
            [
                (Timestamp("2010-01-01"),),
                (Timestamp("2010-01-02"),),
                (Timestamp("2010-01-01"),),
                (Timestamp("2010-01-02"),),
                ("a", "b"),
            ],
            dtype=object,
        )[:-1]
        result = Categorical(values)
        expected = Index(
            [(Timestamp("2010-01-01"),), (Timestamp("2010-01-02"),)],
            tupleize_cols=False,
        )
        tm.assert_index_equal(result.categories, expected)

    def test_constructor_unsortable(self):
        # [FUNCTION DOC]
        # Mixed-type values that cannot be sorted work when unordered, but raise a type error asking for explicit
        # categories when ordered is requested.
        # it works!
        arr = np.array([1, 2, 3, datetime.now()], dtype="O")
        factor = Categorical(arr, ordered=False)
        assert not factor.ordered

        # this however will raise as cannot be sorted
        msg = (
            "'values' is not ordered, please explicitly specify the "
            "categories order by passing in a categories argument."
        )
        with pytest.raises(TypeError, match=msg):
            Categorical(arr, ordered=True)

    def test_constructor_interval(self):
        # [FUNCTION DOC]
        # Constructing from a list of intervals equals constructing from an interval index and uses it as categories.
        result = Categorical(
            [Interval(1, 2), Interval(2, 3), Interval(3, 6)], ordered=True
        )
        ii = IntervalIndex([Interval(1, 2), Interval(2, 3), Interval(3, 6)])
        exp = Categorical(ii, ordered=True)
        tm.assert_categorical_equal(result, exp)
        tm.assert_index_equal(result.categories, ii)

    def test_constructor(self):
        # [FUNCTION DOC]
        # Large omnibus test: values with and without categories; duplicate categories raise a value error;
        # default is unordered; construction from a Categorical and from a Series (categorical or plain)
        # reproduces or recodes it; integer versus float category inference with missing values; single value
        # corner cases; and the Pandas4Warning when no value is in the categories.
        # KNOWN ISSUE: a trailing block of commented-out code about preserving the original integer type is left in the test body and has no effect.
        # Finally checks that an empty integer input with categories raises no warning.
        exp_arr = np.array(["a", "b", "c", "a", "b", "c"], dtype=np.object_)
        c1 = Categorical(exp_arr)
        tm.assert_numpy_array_equal(c1.__array__(), exp_arr)
        c2 = Categorical(exp_arr, categories=["a", "b", "c"])
        tm.assert_numpy_array_equal(c2.__array__(), exp_arr)
        c2 = Categorical(exp_arr, categories=["c", "b", "a"])
        tm.assert_numpy_array_equal(c2.__array__(), exp_arr)

        # categories must be unique
        msg = "Categorical categories must be unique"
        with pytest.raises(ValueError, match=msg):
            Categorical([1, 2], [1, 2, 2])

        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], ["a", "b", "b"])

        # The default should be unordered
        c1 = Categorical(["a", "b", "c", "a"])
        assert not c1.ordered

        # Categorical as input
        c1 = Categorical(["a", "b", "c", "a"])
        c2 = Categorical(c1)
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        c2 = Categorical(c1)
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "c", "b"])
        c2 = Categorical(c1)
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "c", "b"])
        c2 = Categorical(c1, categories=["a", "b", "c"])
        tm.assert_numpy_array_equal(c1.__array__(), c2.__array__())
        tm.assert_index_equal(c2.categories, Index(["a", "b", "c"]))

        # Series of dtype category
        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        c2 = Categorical(Series(c1))
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "c", "b"])
        c2 = Categorical(Series(c1))
        tm.assert_categorical_equal(c1, c2)

        # Series
        c1 = Categorical(["a", "b", "c", "a"])
        c2 = Categorical(Series(["a", "b", "c", "a"]))
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        c2 = Categorical(Series(["a", "b", "c", "a"]), categories=["a", "b", "c", "d"])
        tm.assert_categorical_equal(c1, c2)

        # This should result in integer categories, not float!
        cat = Categorical([1, 2, 3, np.nan], categories=[1, 2, 3])
        assert is_integer_dtype(cat.categories)

        # https://github.com/pandas-dev/pandas/issues/3678
        cat = Categorical([np.nan, 1, 2, 3])
        assert is_integer_dtype(cat.categories)

        # this should result in floats
        cat = Categorical([np.nan, 1, 2.0, 3])
        assert is_float_dtype(cat.categories)

        cat = Categorical([np.nan, 1.0, 2.0, 3.0])
        assert is_float_dtype(cat.categories)

        # This doesn't work -> this would probably need some kind of "remember
        # the original type" feature to try to cast the array interface result
        # to...

        # vals = np.asarray(cat[cat.notna()])
        # assert is_integer_dtype(vals)

        # corner cases
        cat = Categorical([1])
        assert len(cat.categories) == 1
        assert cat.categories[0] == 1
        assert len(cat.codes) == 1
        assert cat.codes[0] == 0

        cat = Categorical(["a"])
        assert len(cat.categories) == 1
        assert cat.categories[0] == "a"
        assert len(cat.codes) == 1
        assert cat.codes[0] == 0

        # two arrays
        #  - when the first is an integer dtype and the second is not
        #  - when the resulting codes are all -1/NaN
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            Categorical([0, 1, 2, 0, 1, 2], categories=["a", "b", "c"])

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            Categorical([0, 1, 2, 0, 1, 2], categories=[3, 4, 5])

        # the next one are from the old docs
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            Categorical([0, 1, 2, 0, 1, 2], [1, 2, 3])
            cat = Categorical([1, 2], categories=[1, 2, 3])

        # this is a legitimate constructor
        with tm.assert_produces_warning(None):
            Categorical(np.array([], dtype="int64"), categories=[3, 2, 1], ordered=True)

    def test_constructor_with_existing_categories(self):
        # [FUNCTION DOC]
        # Regression test: building from another Categorical or a Series of one with explicit categories must
        # recode the values instead of reusing the codes, matching construction from the raw values. Each
        # construction expects the Pandas4Warning about values not in the categories.
        # GH25318: constructing with pd.Series used to bogusly skip recoding
        # categories
        c0 = Categorical(["a", "b", "c", "a"])
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            c1 = Categorical(["a", "b", "c", "a"], categories=["b", "c"])

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            c2 = Categorical(c0, categories=c1.categories)
        tm.assert_categorical_equal(c1, c2)

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            c3 = Categorical(Series(c0), categories=c1.categories)
        tm.assert_categorical_equal(c1, c3)

    def test_constructor_not_sequence(self):
        # [FUNCTION DOC]
        # Passing a string as categories must raise a type error saying categories must be list-like.
        # https://github.com/pandas-dev/pandas/issues/16022
        msg = r"^Parameter 'categories' must be list-like, was"
        with pytest.raises(TypeError, match=msg):
            Categorical(["a", "b"], categories="a")

    def test_constructor_with_null(self):
        # [FUNCTION DOC]
        # Missing values (NaN, None, NaT) among the categories must raise a value error saying categories cannot
        # be null.
        # Cannot have NaN in categories
        msg = "Categorical categories cannot be null"
        with pytest.raises(ValueError, match=msg):
            Categorical([np.nan, "a", "b", "c"], categories=[np.nan, "a", "b", "c"])

        with pytest.raises(ValueError, match=msg):
            Categorical([None, "a", "b", "c"], categories=[None, "a", "b", "c"])

        with pytest.raises(ValueError, match=msg):
            Categorical(
                DatetimeIndex(["nat", "20160101"]),
                categories=[NaT, Timestamp("20160101")],
            )

    def test_constructor_with_index(self):
        # [FUNCTION DOC]
        # Categorical built from a categorical index equals its values, with or without casting to object plus explicit categories.
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        tm.assert_categorical_equal(ci.values, Categorical(ci))

        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        tm.assert_categorical_equal(
            ci.values, Categorical(ci.astype(object), categories=ci.categories)
        )

    def test_constructor_with_generator(self):
        # [FUNCTION DOC]
        # Generators and ranges are accepted for both values and categories and give equal results.
        # KNOWN ISSUE: the multi-index creation statement in the middle of the test has an unused result, so it asserts nothing (leftover dead code).
        # This was raising an Error in isna(single_val).any() because isna
        # returned a scalar for a generator

        exp = Categorical([0, 1, 2])
        cat = Categorical(x for x in [0, 1, 2])
        tm.assert_categorical_equal(cat, exp)
        cat = Categorical(range(3))
        tm.assert_categorical_equal(cat, exp)

        MultiIndex.from_product([range(5), ["a", "b", "c"]])

        # check that categories accept generators and sequences
        cat = Categorical([0, 1, 2], categories=(x for x in [0, 1, 2]))
        tm.assert_categorical_equal(cat, exp)
        cat = Categorical([0, 1, 2], categories=range(3))
        tm.assert_categorical_equal(cat, exp)

    def test_constructor_with_rangeindex(self):
        # [FUNCTION DOC]
        # A range index used as values or as categories is kept as a range index in the categories (exact index type check).
        # RangeIndex is preserved in Categories
        rng = Index(range(3))

        cat = Categorical(rng)
        tm.assert_index_equal(cat.categories, rng, exact=True)

        cat = Categorical([1, 2, 0], categories=rng)
        tm.assert_index_equal(cat.categories, rng, exact=True)

    @pytest.mark.parametrize(
        "dtl",
        [
            date_range("1995-01-01 00:00:00", periods=5, freq="s"),
            date_range("1995-01-01 00:00:00", periods=5, freq="s", tz="US/Eastern"),
            timedelta_range("1 day", periods=5, freq="s"),
        ],
    )
    def test_constructor_with_datetimelike(self, dtl):
        # [FUNCTION DOC]
        # Parametrized over tz-naive datetimes, tz-aware datetimes and timedeltas. Categories equal the index of the
        # same type built from the series (with frequency cleared), codes are sequential, and a missing last
        # value gets code minus one and appears as NaT in the representation.
        # see gh-12077
        # constructor with a datetimelike and NaT

        s = Series(dtl)
        c = Categorical(s)

        expected = type(dtl)(s)
        expected._data.freq = None

        tm.assert_index_equal(c.categories, expected)
        tm.assert_numpy_array_equal(c.codes, np.arange(5, dtype="int8"))

        # with NaT
        s2 = s.copy()
        s2.iloc[-1] = NaT
        c = Categorical(s2)

        expected = type(dtl)(s2.dropna())
        expected._data.freq = None

        tm.assert_index_equal(c.categories, expected)

        exp = np.array([0, 1, 2, 3, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(c.codes, exp)

        result = repr(c)
        assert "NaT" in result

    def test_constructor_from_index_series_datetimetz(self):
        # [FUNCTION DOC]
        # Categorical from a tz-aware datetime index or Series keeps the index as categories (frequency removed).
        idx = date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern")
        idx = idx._with_freq(None)  # freq not preserved in result.categories
        result = Categorical(idx)
        tm.assert_index_equal(result.categories, idx)

        result = Categorical(Series(idx))
        tm.assert_index_equal(result.categories, idx)

    def test_constructor_date_objects(self):
        # [FUNCTION DOC]
        # Plain date objects are not converted to timestamps: categories stay object dtype holding date instances.
        # we dont cast date objects to timestamps, matching Index constructor
        v = date.today()

        cat = Categorical([v, v])
        assert cat.categories.dtype == object
        assert type(cat.categories[0]) is date

    def test_constructor_from_index_series_timedelta(self):
        # [FUNCTION DOC]
        # Categorical from a timedelta index or Series keeps it as categories (frequency removed).
        idx = timedelta_range("1 days", freq="D", periods=3)
        idx = idx._with_freq(None)  # freq not preserved in result.categories
        result = Categorical(idx)
        tm.assert_index_equal(result.categories, idx)

        result = Categorical(Series(idx))
        tm.assert_index_equal(result.categories, idx)

    def test_constructor_from_index_series_period(self):
        # [FUNCTION DOC]
        # Categorical from a period index or Series keeps it as categories.
        idx = period_range("2015-01-01", freq="D", periods=3)
        result = Categorical(idx)
        tm.assert_index_equal(result.categories, idx)

        result = Categorical(Series(idx))
        tm.assert_index_equal(result.categories, idx)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1.0, 1.2, 1.8, np.nan]),
            np.array([1, 2, 3], dtype="int64"),
            ["a", "b", "c", np.nan],
            [pd.Period("2014-01"), pd.Period("2014-02"), NaT],
            [Timestamp("2014-01-01"), Timestamp("2014-01-02"), NaT],
            [
                Timestamp("2014-01-01", tz="US/Eastern"),
                Timestamp("2014-01-02", tz="US/Eastern"),
                NaT,
            ],
        ],
    )
    def test_constructor_invariant(self, values):
        # [FUNCTION DOC]
        # Parametrized over floats with NaN, integers, strings with NaN, periods, timestamps and tz-aware
        # timestamps with NaT. Constructing from an existing Categorical gives an equal Categorical.
        # GH 14190
        c = Categorical(values)
        c2 = Categorical(c)
        tm.assert_categorical_equal(c, c2)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_constructor_with_dtype(self, ordered):
        # [FUNCTION DOC]
        # Parametrized over ordered; the categorical dtype supplies categories and order, giving the same result as explicit arguments.
        categories = ["b", "a", "c"]
        dtype = CategoricalDtype(categories, ordered=ordered)
        result = Categorical(["a", "b", "a", "c"], dtype=dtype)
        expected = Categorical(
            ["a", "b", "a", "c"], categories=categories, ordered=ordered
        )
        tm.assert_categorical_equal(result, expected)
        assert result.ordered is ordered

    def test_constructor_dtype_and_others_raises(self):
        # [FUNCTION DOC]
        # Giving a dtype together with categories or ordered must raise a value error that they cannot be combined.
        dtype = CategoricalDtype(["a", "b"], ordered=True)
        msg = "Cannot specify `categories` or `ordered` together with `dtype`."
        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], categories=["a", "b"], dtype=dtype)

        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], ordered=True, dtype=dtype)

        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], ordered=False, dtype=dtype)

    @pytest.mark.parametrize("categories", [None, ["a", "b"], ["a", "c"]])
    def test_constructor_str_category(self, categories, ordered):
        # [FUNCTION DOC]
        # Parametrized over categories (none, matching, one missing value) and the ordered fixture. Passing the
        # category alias as dtype along with the other arguments equals omitting the dtype. The Pandas4Warning is
        # expected only when a value is not in the categories.
        warn = Pandas4Warning if categories == ["a", "c"] else None
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(warn, match=msg):
            result = Categorical(
                ["a", "b"], categories=categories, ordered=ordered, dtype="category"
            )
            expected = Categorical(["a", "b"], categories=categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_constructor_str_unknown(self):
        # [FUNCTION DOC]
        # An unknown dtype string must raise a value error.
        with pytest.raises(ValueError, match="Unknown dtype"):
            Categorical([1, 2], dtype="foo")

    def test_constructor_np_strs(self):
        # [FUNCTION DOC]
        # Categories made of numpy string scalars plus an integer must work (hashtable mapping must handle numpy
        # strings); mixed input avoids conversion to the string dtype.
        # GH#31499 Hashtable.map_locations needs to work on np.str_ objects
        #  We can't pass all-strings because the constructor would cast
        #  those to StringDtype post-PDEP14
        cat = Categorical(["1", "0", "1", 2], [np.str_("0"), np.str_("1"), 2])
        assert all(isinstance(x, (np.str_, int)) for x in cat.categories)

    def test_constructor_from_categorical_with_dtype(self):
        # [FUNCTION DOC]
        # Constructing from a Categorical with a dtype that has categories uses the dtype categories, so the
        # unlisted value becomes missing; the Pandas4Warning is expected.
        dtype = CategoricalDtype(["a", "b", "c"], ordered=True)
        values = Categorical(["a", "b", "d"])
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = Categorical(values, dtype=dtype)
        # We use dtype.categories, not values.categories
        expected = Categorical(
            ["a", "b", None], categories=["a", "b", "c"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

    def test_constructor_from_categorical_with_unknown_dtype(self):
        # [FUNCTION DOC]
        # A categorical dtype with no categories applies only its ordered flag: categories come from the input.
        dtype = CategoricalDtype(None, ordered=True)
        values = Categorical(["a", "b", "d"])
        result = Categorical(values, dtype=dtype)
        # We use values.categories, not dtype.categories
        expected = Categorical(
            ["a", "b", "d"], categories=["a", "b", "d"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

    def test_constructor_from_categorical_string(self):
        # [FUNCTION DOC]
        # Constructing from a Categorical with explicit categories and ordered, with and without the alias string
        # dtype, gives the same result and the Pandas4Warning.
        values = Categorical(["a", "b", "d"])
        # use categories, ordered
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = Categorical(
                values, categories=["a", "b", "c"], ordered=True, dtype="category"
            )
        expected = Categorical(
            ["a", "b", None], categories=["a", "b", "c"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

        # No string
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = Categorical(values, categories=["a", "b", "c"], ordered=True)
        tm.assert_categorical_equal(result, expected)

    def test_constructor_with_categorical_categories(self):
        # [FUNCTION DOC]
        # Categories passed as a Categorical or categorical index are accepted and equal passing plain categories.
        # GH17884
        expected = Categorical(["a", "b"], categories=["a", "b", "c"])

        result = Categorical(["a", "b"], categories=Categorical(["a", "b", "c"]))
        tm.assert_categorical_equal(result, expected)

        result = Categorical(["a", "b"], categories=CategoricalIndex(["a", "b", "c"]))
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("klass", [lambda x: np.array(x, dtype=object), list])
    def test_construction_with_null(self, klass, nulls_fixture):
        # [FUNCTION DOC]
        # Parametrized over object array and list inputs and the nulls fixture. Missing values of any kind become code
        # minus one while categories contain only the non-missing values.
        # https://github.com/pandas-dev/pandas/issues/31927
        values = klass(["a", nulls_fixture, "b"])
        result = Categorical(values)

        dtype = CategoricalDtype(["a", "b"])
        codes = [0, -1, 1]
        expected = Categorical.from_codes(codes=codes, dtype=dtype)

        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("validate", [True, False])
    def test_from_codes_nullable_int_categories(self, any_numeric_ea_dtype, validate):
        # [FUNCTION DOC]
        # Parametrized over validate flag and nullable numeric dtypes. Building from codes with categories of an extension
        # dtype keeps that dtype in the categories.
        # GH#39649
        cats = pd.array(range(5), dtype=any_numeric_ea_dtype)
        codes = np.random.default_rng(2).integers(5, size=3)
        dtype = CategoricalDtype(cats)
        arr = Categorical.from_codes(codes, dtype=dtype, validate=validate)
        assert arr.categories.dtype == cats.dtype
        tm.assert_index_equal(arr.categories, Index(cats))

    def test_from_codes_empty(self):
        # [FUNCTION DOC]
        # Empty codes with categories give an empty Categorical equal to the one built from empty values.
        cat = ["a", "b", "c"]
        result = Categorical.from_codes([], categories=cat)
        expected = Categorical([], categories=cat)

        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("validate", [True, False])
    def test_from_codes_validate(self, validate):
        # [FUNCTION DOC]
        # With validation on, out-of-range codes raise a value error; with validation off they are accepted without
        # error (caller responsibility).
        # GH53122
        dtype = CategoricalDtype(["a", "b"])
        if validate:
            with pytest.raises(ValueError, match="codes need to be between "):
                Categorical.from_codes([4, 5], dtype=dtype, validate=validate)
        else:
            # passes, though has incorrect codes, but that's the user responsibility
            Categorical.from_codes([4, 5], dtype=dtype, validate=validate)

    def test_from_codes_too_few_categories(self):
        # [FUNCTION DOC]
        # Codes greater than the last valid index raise a value error, both with categories and with dtype.
        dtype = CategoricalDtype(categories=[1, 2])
        msg = "codes need to be between "
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([1, 2], categories=dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([1, 2], dtype=dtype)

    def test_from_codes_non_int_codes(self):
        # [FUNCTION DOC]
        # String codes raise a value error that codes need to be array-like integers (both argument forms).
        dtype = CategoricalDtype(categories=[1, 2])
        msg = "codes need to be array-like integers"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(["a"], categories=dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(["a"], dtype=dtype)

    def test_from_codes_non_unique_categories(self):
        # [FUNCTION DOC]
        # Duplicate categories raise a value error.
        with pytest.raises(ValueError, match="Categorical categories must be unique"):
            Categorical.from_codes([0, 1, 2], categories=["a", "a", "b"])

    def test_from_codes_nan_cat_included(self):
        # [FUNCTION DOC]
        # A missing value in categories raises a value error that categories cannot be null.
        with pytest.raises(ValueError, match="Categorical categories cannot be null"):
            Categorical.from_codes([0, 1, 2], categories=["a", "b", np.nan])

    def test_from_codes_too_negative(self):
        # [FUNCTION DOC]
        # Codes below minus one raise a value error stating the valid range.
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        msg = r"codes need to be between -1 and len\(categories\)-1"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([-2, 1, 2], categories=dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([-2, 1, 2], dtype=dtype)

    def test_from_codes(self):
        # [FUNCTION DOC]
        # Valid codes with categories or dtype produce the expected unordered Categorical.
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        exp = Categorical(["a", "b", "c"], ordered=False)
        res = Categorical.from_codes([0, 1, 2], categories=dtype.categories)
        tm.assert_categorical_equal(exp, res)

        res = Categorical.from_codes([0, 1, 2], dtype=dtype)
        tm.assert_categorical_equal(exp, res)

    @pytest.mark.parametrize("klass", [Categorical, CategoricalIndex])
    def test_from_codes_with_categorical_categories(self, klass):
        # [FUNCTION DOC]
        # Parametrized over Categorical and categorical index passed as categories; accepted.
        # GH17884
        expected = Categorical(["a", "b"], categories=["a", "b", "c"])

        result = Categorical.from_codes([0, 1], categories=klass(["a", "b", "c"]))
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("klass", [Categorical, CategoricalIndex])
    def test_from_codes_with_non_unique_categorical_categories(self, klass):
        # [FUNCTION DOC]
        # Parametrized over Categorical and categorical index with duplicate categories; value error expected.
        with pytest.raises(ValueError, match="Categorical categories must be unique"):
            Categorical.from_codes([0, 1], klass(["a", "b", "a"]))

    def test_from_codes_with_nan_code(self):
        # [FUNCTION DOC]
        # Codes containing NaN (so floats) raise a value error that codes need to be array-like integers.
        # GH21767
        codes = [1, 2, np.nan]
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        with pytest.raises(ValueError, match="codes need to be array-like integers"):
            Categorical.from_codes(codes, categories=dtype.categories)
        with pytest.raises(ValueError, match="codes need to be array-like integers"):
            Categorical.from_codes(codes, dtype=dtype)

    @pytest.mark.parametrize("codes", [[1.0, 2.0, 0], [1.1, 2.0, 0]])
    def test_from_codes_with_float(self, codes):
        # [FUNCTION DOC]
        # Parametrized over integral-valued floats and non-integral floats. Float codes always raise a value error.
        # GH21767
        # float codes should raise even if values are equal to integers
        dtype = CategoricalDtype(categories=["a", "b", "c"])

        msg = "codes need to be array-like integers"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, dtype=dtype)

    def test_from_codes_with_dtype_raises(self):
        # [FUNCTION DOC]
        # Giving dtype together with categories or ordered raises a value error saying they cannot be specified together.
        msg = "Cannot specify"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(
                [0, 1], categories=["a", "b"], dtype=CategoricalDtype(["a", "b"])
            )

        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(
                [0, 1], ordered=True, dtype=CategoricalDtype(["a", "b"])
            )

    def test_from_codes_neither(self):
        # [FUNCTION DOC]
        # Giving neither categories nor dtype raises a value error mentioning that both were none.
        msg = "Both were None"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([0, 1])

    def test_from_codes_with_nullable_int(self):
        # [FUNCTION DOC]
        # Nullable integer codes without missing values are accepted and equal to plain integer codes.
        codes = pd.array([0, 1], dtype="Int64")
        categories = ["a", "b"]

        result = Categorical.from_codes(codes, categories=categories)
        expected = Categorical.from_codes(codes.to_numpy(int), categories=categories)

        tm.assert_categorical_equal(result, expected)

    def test_from_codes_with_nullable_int_na_raises(self):
        # [FUNCTION DOC]
        # Nullable integer codes containing NA raise a value error that codes cannot contain NA values.
        codes = pd.array([0, None], dtype="Int64")
        categories = ["a", "b"]

        msg = "codes cannot contain NA values"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, categories=categories)

    @pytest.mark.parametrize("dtype", [None, "category"])
    def test_from_inferred_categories(self, dtype):
        # [FUNCTION DOC]
        # Exercises the private from-inferred-categories constructor. Parametrized over dtype none and the alias
        # string. Given categories and codes, the result equals the codes-based constructor.
        cats = ["a", "b"]
        codes = np.array([0, 0, 1, 1], dtype="i8")
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        expected = Categorical.from_codes(codes, cats)
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, "category"])
    def test_from_inferred_categories_sorts(self, dtype):
        # [FUNCTION DOC]
        # Exercises the private from-inferred-categories constructor with unsorted categories: the result has
        # sorted categories and recoded codes. Parametrized over dtype none and the alias string.
        cats = ["b", "a"]
        codes = np.array([0, 1, 1, 1], dtype="i8")
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        expected = Categorical.from_codes([1, 0, 0, 0], ["a", "b"])
        tm.assert_categorical_equal(result, expected)

    def test_from_inferred_categories_dtype(self):
        # [FUNCTION DOC]
        # With a given ordered dtype, inferred categories not in the dtype become missing and a Pandas4Warning is
        # expected (stack level not checked).
        cats = ["a", "b", "d"]
        codes = np.array([0, 1, 0, 2], dtype="i8")
        dtype = CategoricalDtype(["c", "b", "a"], ordered=True)
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(
            Pandas4Warning, match=msg, check_stacklevel=False
        ):
            result = Categorical._from_inferred_categories(cats, codes, dtype)
        expected = Categorical(
            ["a", "b", "a", None], categories=["c", "b", "a"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

    def test_from_inferred_categories_coerces(self):
        # [FUNCTION DOC]
        # Inferred string categories are coerced to the dtype categories (integers); the unparsable one becomes
        # missing, with the Pandas4Warning expected.
        cats = ["1", "2", "bad"]
        codes = np.array([0, 0, 1, 2], dtype="i8")
        dtype = CategoricalDtype([1, 2])
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(
            Pandas4Warning, match=msg, check_stacklevel=False
        ):
            result = Categorical._from_inferred_categories(cats, codes, dtype)
        expected = Categorical([1, 1, 2, np.nan])
        tm.assert_categorical_equal(result, expected)

    def test_construction_with_ordered(self, ordered):
        # [FUNCTION DOC]
        # Uses the ordered fixture; the ordered attribute of the result equals the boolean of the argument.
        # GH 9347, 9190
        cat = Categorical([0, 1, 2], ordered=ordered)
        assert cat.ordered == bool(ordered)

    def test_constructor_imaginary(self):
        # [FUNCTION DOC]
        # Complex numbers work as categories and round-trip to numpy.
        values = [1, 2, 3 + 1j]
        c1 = Categorical(values)
        tm.assert_index_equal(c1.categories, Index(values))
        tm.assert_numpy_array_equal(np.array(c1), np.array(values))

    def test_constructor_string_and_tuples(self):
        # [FUNCTION DOC]
        # Mixed strings and tuples give categories as an index of those objects (tuples not expanded).
        # GH 21416
        c = Categorical(np.array(["c", ("a", "b"), ("b", "a"), "c"], dtype=object))
        expected_index = Index([("a", "b"), ("b", "a"), "c"])
        assert c.categories.equals(expected_index)

    def test_interval(self):
        # [FUNCTION DOC]
        # Categorical with interval categories: from index with and without categories, from lists, shuffled values,
        # values outside categories (missing with Pandas4Warning), and overlapping intervals. Checks codes
        # (8-bit) and categories.
        idx = pd.interval_range(0, 10, periods=10)
        cat = Categorical(idx, categories=idx)
        expected_codes = np.arange(10, dtype="int8")
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # infer categories
        cat = Categorical(idx)
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # list values
        cat = Categorical(list(idx))
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # list values, categories
        cat = Categorical(list(idx), categories=list(idx))
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # shuffled
        values = idx.take([1, 2, 0])
        cat = Categorical(values, categories=idx)
        tm.assert_numpy_array_equal(cat.codes, np.array([1, 2, 0], dtype="int8"))
        tm.assert_index_equal(cat.categories, idx)

        # extra
        values = pd.interval_range(8, 11, periods=3)
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            cat = Categorical(values, categories=idx)
        expected_codes = np.array([8, 9, -1], dtype="int8")
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # overlapping
        idx = IntervalIndex([Interval(0, 2), Interval(0, 1)])
        cat = Categorical(idx, categories=idx)
        expected_codes = np.array([0, 1], dtype="int8")
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

    def test_categorical_extension_array_nullable(self, nulls_fixture):
        # [FUNCTION DOC]
        # Using the nulls fixture, a string-dtype extension array of only missing values gives categories of the
        # same dtype and equals constructing from a Series of that dtype.
        # GH:
        arr = pd.array([nulls_fixture] * 2, dtype=pd.StringDtype())
        result = Categorical(arr)
        assert arr.dtype == result.categories.dtype
        expected = Categorical(Series([pd.NA, pd.NA], dtype=arr.dtype))
        tm.assert_categorical_equal(result, expected)

    def test_from_sequence_copy(self):
        # [FUNCTION DOC]
        # Private sequence constructor: with copy disabled the codes array is shared (same object); with copy
        # enabled memory is not shared.
        cat = Categorical(np.arange(5).repeat(2))
        result = Categorical._from_sequence(cat, dtype=cat.dtype, copy=False)

        # more generally, we'd be OK with a view
        assert result._codes is cat._codes

        result = Categorical._from_sequence(cat, dtype=cat.dtype, copy=True)

        assert not tm.shares_memory(result, cat)

    def test_constructor_datetime64_non_nano(self):
        # [FUNCTION DOC]
        # Non-nanosecond datetime values and categories can be used; all values compare equal to the input.
        categories = np.arange(10).view("M8[D]")
        values = categories[::2].copy()

        cat = Categorical(values, categories=categories)
        assert (cat == values).all()

    def test_constructor_preserves_freq(self):
        # [FUNCTION DOC]
        # The frequency of a datetime index passed as values survives in the categories.
        # GH33830 freq retention in categorical
        dti = date_range("2016-01-01", periods=5)

        expected = dti.freq

        cat = Categorical(dti)
        result = cat.categories.freq

        assert expected == result

    @pytest.mark.parametrize(
        "values, categories",
        [
            [range(5), None],
            [range(4), range(5)],
            [[0, 1, 2, 3], range(5)],
            [[], range(5)],
        ],
    )
    def test_range_values_preserves_rangeindex_categories(self, values, categories):
        # [FUNCTION DOC]
        # Parametrized over ranges, lists and empty values with range or no categories. Categories must be exactly a
        # range index of five integers.
        result = Categorical(values=values, categories=categories).categories
        expected = RangeIndex(range(5))
        tm.assert_index_equal(result, expected, exact=True)

    def test_categorical_preserve_object_dtype_from_pandas(self, using_infer_string):
        # [FUNCTION DOC]
        # Regression test: from an object Series or Index the categories keep object dtype; from a numpy array or list they
        # are inferred as the string dtype when inference of strings is enabled (fixture), else object.
        # GH#61778
        pylist = ["foo", "bar", "baz"]
        ser = Series(pylist, dtype="object")
        idx = Index(pylist, dtype="object")
        arr = np.array(pylist, dtype="object")

        cat_from_ser = Categorical(ser)
        cat_from_idx = Categorical(idx)
        cat_from_arr = Categorical(arr)
        cat_from_list = Categorical(pylist)

        # Series/Index with object dtype: infer string
        # dtype if all elements are strings
        assert cat_from_ser.categories.dtype == object
        assert cat_from_idx.categories.dtype == object

        if using_infer_string:
            # Numpy array or list: infer string dtype
            assert cat_from_arr.categories.dtype == "str"
            assert cat_from_list.categories.dtype == "str"
        else:
            assert cat_from_arr.categories.dtype == object
            assert cat_from_list.categories.dtype == object
