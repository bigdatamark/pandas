# [MODULE DOC]
# Tests for the categorical accessor of Series and related accessor delegation: name
# propagation, accessor construction and errors, delegation of categories, codes,
# ordering and category-editing methods, the datetime accessor on categorical series,
# and category renaming in frames.
import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties


class TestCatAccessor:
    # [CLASS DOC]
    # Groups the Series categorical accessor tests. Stateless.
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.cat.set_categories([1, 2, 3]),
            lambda x: x.cat.reorder_categories([2, 3, 1], ordered=True),
            lambda x: x.cat.rename_categories([1, 2, 3]),
            lambda x: x.cat.remove_unused_categories(),
            lambda x: x.cat.remove_categories([2]),
            lambda x: x.cat.add_categories([4]),
            lambda x: x.cat.as_ordered(),
            lambda x: x.cat.as_unordered(),
        ],
    )
    def test_getname_categorical_accessor(self, method):
        # [FUNCTION DOC]
        # Regression test (GH 17509). Parametrized over eight category-editing accessor
        # operations (set, reorder, rename, remove unused, remove, add, ordered,
        # unordered), each given as a function of the series. The result keeps the
        # series name A.
        # GH#17509
        ser = Series([1, 2, 3], name="A").astype("category")
        expected = "A"
        result = method(ser).name
        assert result == expected

    def test_cat_accessor(self):
        # [FUNCTION DOC]
        # Verifies the accessor on a categorical series with a missing value: categories
        # index is a and b, ordering is off, setting categories to b and a gives the
        # expected categorical, and after assigning a to all elements removing unused
        # categories leaves only a.
        ser = Series(Categorical(["a", "b", np.nan, "a"]))
        tm.assert_index_equal(ser.cat.categories, Index(["a", "b"]))
        assert not ser.cat.ordered

        exp = Categorical(["a", "b", np.nan, "a"], categories=["b", "a"])

        res = ser.cat.set_categories(["b", "a"])
        tm.assert_categorical_equal(res.values, exp)

        ser[:] = "a"
        ser = ser.cat.remove_unused_categories()
        tm.assert_index_equal(ser.cat.categories, Index(["a"]))

    def test_cat_accessor_api(self):
        # [FUNCTION DOC]
        # Regression test (GH 9322). The class attribute is the categorical accessor
        # class; an instance is created for a categorical series; for a non-categorical
        # series, accessing it raises an attribute error saying only the accessor can be
        # used with categorical data, and the attribute check reports it absent.
        # GH#9322

        assert Series.cat is CategoricalAccessor
        ser = Series(list("aabbcde")).astype("category")
        assert isinstance(ser.cat, CategoricalAccessor)

        invalid = Series([1])
        with pytest.raises(AttributeError, match="only use .cat accessor"):
            invalid.cat
        assert not hasattr(invalid, "cat")

    def test_cat_accessor_no_new_attributes(self):
        # [FUNCTION DOC]
        # Regression test for GH 10673. Setting a new attribute on the accessor object
        # raises an attribute error saying new attributes cannot be added.
        # https://github.com/pandas-dev/pandas/issues/10673
        cat = Series(list("aabbcde")).astype("category")
        with pytest.raises(AttributeError, match="You cannot add any new attribute"):
            cat.cat.xlabel = "a"

    def test_categorical_delegations(self):
        # [FUNCTION DOC]
        # Verifies delegation to the categorical. The accessor on non-categorical dtypes
        # (integers, strings, floats, timestamps), also when called, raises an attribute
        # error about the category dtype requirement. For an ordered categorical:
        # categories, renaming categories, codes (int8 series), ordered flag and
        # switching ordering behave right; setting categories reorders and values/array
        # conversion remain as object arrays; removing unused categories works; the set
        # categories method directly on the Series raises an attribute error (it exists
        # only on the accessor); renaming categories with a callable (GH 18862)
        # uppercases categories.
        # invalid accessor
        msg = r"Can only use \.cat accessor with a 'category' dtype"
        with pytest.raises(AttributeError, match=msg):
            Series([1, 2, 3]).cat
        with pytest.raises(AttributeError, match=msg):
            Series([1, 2, 3]).cat()
        with pytest.raises(AttributeError, match=msg):
            Series(["a", "b", "c"]).cat
        with pytest.raises(AttributeError, match=msg):
            Series(np.arange(5.0)).cat
        with pytest.raises(AttributeError, match=msg):
            Series([Timestamp("20130101")]).cat

        # Series should delegate calls to '.categories', '.codes', '.ordered'
        # and the methods '.set_categories()' 'drop_unused_categories()' to the
        # categorical
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        exp_categories = Index(["a", "b", "c"])
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        ser = ser.cat.rename_categories([1, 2, 3])
        exp_categories = Index([1, 2, 3])
        tm.assert_index_equal(ser.cat.categories, exp_categories)

        exp_codes = Series([0, 1, 2, 0], dtype="int8")
        tm.assert_series_equal(ser.cat.codes, exp_codes)

        assert ser.cat.ordered
        ser = ser.cat.as_unordered()
        assert not ser.cat.ordered

        ser = ser.cat.as_ordered()
        assert ser.cat.ordered

        # reorder
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        exp_categories = Index(["c", "b", "a"])
        exp_values = np.array(["a", "b", "c", "a"], dtype=np.object_)
        ser = ser.cat.set_categories(["c", "b", "a"])
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        tm.assert_numpy_array_equal(ser.values.__array__(), exp_values)
        tm.assert_numpy_array_equal(ser.__array__(), exp_values)

        # remove unused categories
        ser = Series(Categorical(["a", "b", "b", "a"], categories=["a", "b", "c"]))
        exp_categories = Index(["a", "b"])
        exp_values = np.array(["a", "b", "b", "a"], dtype=np.object_)
        ser = ser.cat.remove_unused_categories()
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        tm.assert_numpy_array_equal(ser.values.__array__(), exp_values)
        tm.assert_numpy_array_equal(ser.__array__(), exp_values)

        # This method is likely to be confused, so test that it raises an error
        # on wrong inputs:
        msg = "'Series' object has no attribute 'set_categories'"
        with pytest.raises(AttributeError, match=msg):
            ser.set_categories([4, 3, 2, 1])

        # right: ser.cat.set_categories([4,3,2,1])

        # GH#18862 (let Series.cat.rename_categories take callables)
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        result = ser.cat.rename_categories(lambda x: x.upper())
        expected = Series(
            Categorical(["A", "B", "C", "A"], categories=["A", "B", "C"], ordered=True)
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "idx",
        [
            date_range("1/1/2015", periods=5, unit="ns"),
            date_range("1/1/2015", periods=5, tz="MET", unit="ns"),
            period_range("1/1/2015", freq="D", periods=5),
            timedelta_range("1 days", "10 days"),
        ],
    )
    def test_dt_accessor_api_for_categorical(self, idx):
        # [FUNCTION DOC]
        # Regression test (GH 10661). Parametrized over four datetime-like indexes
        # (datetime with and without timezone, period, timedelta). The datetime accessor
        # on a categorical series is the properties accessor; every public method (with
        # default arguments, or special arguments for format, round, floor, ceil,
        # frequency, unit, localization or conversion) and every field attribute gives
        # results equal to the same call on the non-categorical series, with the
        # expected warnings (user warning when converting an aware index to period, a
        # Pandas4 warning for python timedelta conversion, GH 57463).
        # KNOWN ISSUE: the unit special case is written with parentheses around a single
        # string rather than a one-element tuple, so the argument list is the string
        # itself; it only works because the string has a single character.
        # https://github.com/pandas-dev/pandas/issues/10661

        ser = Series(idx)
        cat = ser.astype("category")

        # only testing field (like .day)
        # and bool (is_month_start)
        attr_names = type(ser._values)._datetimelike_ops

        assert isinstance(cat.dt, Properties)

        special_func_defs = [
            ("strftime", ("%Y-%m-%d",), {}),
            ("round", ("D",), {}),
            ("floor", ("D",), {}),
            ("ceil", ("D",), {}),
            ("asfreq", ("D",), {}),
            ("as_unit", ("s"), {}),
        ]
        if idx.dtype == "M8[ns]":
            # exclude dt64tz since that is already localized and would raise
            tup = ("tz_localize", ("UTC",), {})
            special_func_defs.append(tup)
        elif idx.dtype.kind == "M":
            # exclude dt64 since that is not localized so would raise
            tup = ("tz_convert", ("EST",), {})
            special_func_defs.append(tup)

        _special_func_names = [f[0] for f in special_func_defs]

        _ignore_names = ["components", "tz_localize", "tz_convert"]

        func_names = [
            fname
            for fname in dir(ser.dt)
            if not (
                fname.startswith("_")
                or fname in attr_names
                or fname in _special_func_names
                or fname in _ignore_names
            )
        ]

        func_defs = [(fname, (), {}) for fname in func_names]
        func_defs.extend(
            f_def for f_def in special_func_defs if f_def[0] in dir(ser.dt)
        )

        for func, args, kwargs in func_defs:
            warn_cls = []
            if func == "to_period" and getattr(idx, "tz", None) is not None:
                # dropping TZ
                warn_cls.append(UserWarning)
            elif func == "to_pytimedelta":
                # GH 57463
                warn_cls.append(Pandas4Warning)
            if warn_cls:
                warn_cls = tuple(warn_cls)
            else:
                warn_cls = None
            with tm.assert_produces_warning(warn_cls):
                res = getattr(cat.dt, func)(*args, **kwargs)
                exp = getattr(ser.dt, func)(*args, **kwargs)

            tm.assert_equal(res, exp)

        for attr in attr_names:
            res = getattr(cat.dt, attr)
            exp = getattr(ser.dt, attr)

            tm.assert_equal(res, exp)

    def test_dt_accessor_api_for_categorical_invalid(self):
        # [FUNCTION DOC]
        # Verifies the datetime accessor on a categorical series of integers raises an
        # attribute error about needing datetime-like values.
        # KNOWN ISSUE: the final assertion checks the absence of the string accessor
        # rather than the datetime accessor, which seems unrelated to what the test is
        # named for.
        invalid = Series([1, 2, 3]).astype("category")
        msg = "Can only use .dt accessor with datetimelike"

        with pytest.raises(AttributeError, match=msg):
            invalid.dt
        assert not hasattr(invalid, "str")

    def test_set_categories_setitem(self):
        # [FUNCTION DOC]
        # Regression test (GH 43334). Renaming categories of integer categorical columns
        # to labels and recreating them as categoricals with explicit categories keeps
        # the values as labels, without turning them into NaN.
        # GH#43334

        df = DataFrame({"Survived": [1, 0, 1], "Sex": [0, 1, 1]}, dtype="category")

        df["Survived"] = df["Survived"].cat.rename_categories(["No", "Yes"])
        df["Sex"] = df["Sex"].cat.rename_categories(["female", "male"])

        # values should not be coerced to NaN
        assert list(df["Sex"]) == ["female", "male", "male"]
        assert list(df["Survived"]) == ["Yes", "No", "Yes"]

        df["Sex"] = Categorical(df["Sex"], categories=["female", "male"], ordered=False)
        df["Survived"] = Categorical(
            df["Survived"], categories=["No", "Yes"], ordered=False
        )

        # values should not be coerced to NaN
        assert list(df["Sex"]) == ["female", "male", "male"]
        assert list(df["Survived"]) == ["Yes", "No", "Yes"]

    def test_categorical_of_booleans_is_boolean(self):
        # [FUNCTION DOC]
        # Regression test for GH 46313. The category dtype of a boolean categorical
        # column is the numpy boolean dtype object.
        # https://github.com/pandas-dev/pandas/issues/46313
        df = DataFrame(
            {"int_cat": [1, 2, 3], "bool_cat": [True, False, False]}, dtype="category"
        )
        value = df["bool_cat"].cat.categories.dtype
        expected = np.dtype(np.bool_)
        assert value is expected
