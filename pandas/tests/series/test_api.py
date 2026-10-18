# [MODULE DOC]
# Miscellaneous tests of the Series public API surface: tab completion (directory listing) of accessors and index labels, hashability, axis aliases, emptiness and size, attribute dictionary propagation, attribute-error behavior, and the numeric-only keyword across reduction and transform kernels.
# No import-time side effects.

import inspect
import pydoc

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestSeriesMisc:
    # [CLASS DOC]
    # Groups the miscellaneous Series API tests. Stateless.
    def test_tab_completion(self):
        # [FUNCTION DOC]
        # Verifies (issue 9910) that a Series of strings lists the string accessor, but not the datetime or categorical accessors, in its attribute directory.
        # GH 9910
        s = Series(list("abcd"))
        # Series of str values should have .str but not .dt/.cat in __dir__
        assert "str" in dir(s)
        assert "dt" not in dir(s)
        assert "cat" not in dir(s)

    def test_tab_completion_dt(self):
        # [FUNCTION DOC]
        # Verifies that a datetime Series lists the datetime accessor but neither the string nor the categorical accessor.
        # similarly for .dt
        s = Series(date_range("1/1/2015", periods=5))
        assert "dt" in dir(s)
        assert "str" not in dir(s)
        assert "cat" not in dir(s)

    def test_tab_completion_cat(self):
        # [FUNCTION DOC]
        # Verifies that a categorical Series of strings lists the categorical and string accessors but not the datetime accessor.
        # Similarly for .cat, but with the twist that str and dt should be
        # there if the categories are of that type first cat and str.
        s = Series(list("abbcd"), dtype="category")
        assert "cat" in dir(s)
        assert "str" in dir(s)  # as it is a string categorical
        assert "dt" not in dir(s)

    def test_tab_completion_cat_str(self):
        # [FUNCTION DOC]
        # Verifies that a categorical Series of datetimes lists the categorical and datetime accessors but not the string accessor.
        # similar to cat and str
        s = Series(date_range("1/1/2015", periods=5)).astype("category")
        assert "cat" in dir(s)
        assert "str" not in dir(s)
        assert "dt" in dir(s)  # as it is a datetime categorical

    def test_tab_completion_with_categorical(self):
        # [FUNCTION DOC]
        # Verifies that the public attribute names (not starting with an underscore) in the directory of the categorical accessor are exactly the expected eleven names: categories, codes, ordered and the category-manipulation methods.
        # test the tab completion display
        ok_for_cat = [
            "categories",
            "codes",
            "ordered",
            "set_categories",
            "add_categories",
            "remove_categories",
            "rename_categories",
            "reorder_categories",
            "remove_unused_categories",
            "as_ordered",
            "as_unordered",
        ]

        s = Series(list("aabbcde")).astype("category")
        results = sorted({r for r in s.cat.__dir__() if not r.startswith("_")})
        tm.assert_almost_equal(results, sorted(set(ok_for_cat)))

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("ab") * 5, dtype="category"),
            Index([str(i) for i in range(10)]),
            Index(["foo", "bar", "baz"] * 2),
            date_range("2020-01-01", periods=10),
            period_range("2020-01-01", periods=10, freq="D"),
            timedelta_range("1 day", periods=10),
            Index(np.arange(10), dtype=np.uint64),
            Index(np.arange(10), dtype=np.int64),
            Index(np.arange(10), dtype=np.float64),
            Index([True, False]),
            Index([f"a{i}" for i in range(101)]),
            pd.MultiIndex.from_tuples(zip("ABCD", "EFGH")),
            pd.MultiIndex.from_tuples(zip([0, 1, 2, 3], "EFGH")),
        ],
    )
    def test_index_tab_completion(self, index):
        # [FUNCTION DOC]
        # Verifies the attribute directory of an empty object Series built on various index types (categorical, string, datetime, period, timedelta, unsigned/signed/float numeric, boolean, a long string index, and multi-indexes). For the first hundred unique first-level labels, any label that is a string and a valid identifier must appear in the directory; for labels beyond one hundred, they must not appear.
        # dir contains string-like values of the Index.
        s = Series(index=index, dtype=object)
        dir_s = dir(s)
        for i, x in enumerate(s.index.unique(level=0)):
            if i < 100:
                assert not isinstance(x, str) or not x.isidentifier() or x in dir_s
            else:
                assert x not in dir_s

    @pytest.mark.parametrize("ser", [Series(dtype=object), Series([1])])
    def test_not_hashable(self, ser):
        # [FUNCTION DOC]
        # Verifies that hashing a Series, empty or not, raises a type error stating that Series is unhashable.
        msg = "unhashable type: 'Series'"
        with pytest.raises(TypeError, match=msg):
            hash(ser)

    def test_contains(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that every element of the datetime fixture series is contained in its index (a membership helper is applied from the index against the series).
        tm.assert_contains_all(datetime_series.index, datetime_series)

    def test_axis_alias(self):
        # [FUNCTION DOC]
        # Verifies that the axis alias for rows is treated as the index axis: dropping missing values with either name gives equal results, a sum along rows equals the expected total of 3, and the internal helpers map the rows alias to axis number zero and to the name index.
        s = Series([1, 2, np.nan])
        tm.assert_series_equal(s.dropna(axis="rows"), s.dropna(axis="index"))
        assert s.dropna().sum(axis="rows") == 3
        assert s._get_axis_number("rows") == 0
        assert s._get_axis_name("rows") == "index"

    def test_class_axis(self):
        # [FUNCTION DOC]
        # Verifies (issue 18147) that accessing the index descriptor on the Series class itself does not raise and that its documentation is non-empty.
        # https://github.com/pandas-dev/pandas/issues/18147
        # no exception and no empty docstring
        assert pydoc.getdoc(Series.index)

    def test_ndarray_compat(self):
        # [FUNCTION DOC]
        # Verifies NumPy-compatibility when applying a function over a frame's columns where the function selects the element at the label of the maximum: the result equals the column maximums of the 1000x3 random time-indexed frame.
        # test numpy compat with Series as sub-class of NDFrame
        tsdf = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 3)),
            columns=["A", "B", "C"],
            index=date_range("1/1/2000", periods=1000),
        )

        def f(x):
            # [FUNCTION DOC]
            # HELPER: nested function for the compatibility test; takes a Series and returns the value located at the label of its maximum.
            return x[x.idxmax()]

        result = tsdf.apply(f)
        expected = tsdf.max()
        tm.assert_series_equal(result, expected)

    def test_ndarray_compat_like_func(self):
        # [FUNCTION DOC]
        # Verifies that applying a NumPy function creating ones with the same shape to a random Series gives a float64 Series of ones with a default range index.
        # using an ndarray like function
        s = Series(np.random.default_rng(2).standard_normal(10))
        result = Series(np.ones_like(s))
        expected = Series(1, index=range(10), dtype="float64")
        tm.assert_series_equal(result, expected)

    def test_empty_method(self):
        # [FUNCTION DOC]
        # Verifies that an empty object Series reports being empty.
        s_empty = Series(dtype=object)
        assert s_empty.empty

    @pytest.mark.parametrize("dtype", ["int64", object])
    def test_empty_method_full_series(self, dtype):
        # [FUNCTION DOC]
        # Verifies that a Series with one index label and no values (filled with missing) is not reported as empty, for integer and object dtypes.
        full_series = Series(index=[1], dtype=dtype)
        assert not full_series.empty

    @pytest.mark.parametrize("dtype", [None, "Int64"])
    def test_integer_series_size(self, dtype):
        # [FUNCTION DOC]
        # Verifies that the size of a nine-element Series is nine for both default and nullable Int64 dtype (issue 25580).
        # GH 25580
        s = Series(range(9), dtype=dtype)
        assert s.size == 9

    def test_attrs(self):
        # [FUNCTION DOC]
        # Verifies that the attrs mapping starts empty and that an entry set on the Series is propagated to the result of an arithmetic operation.
        s = Series([0, 1], name="abc")
        assert s.attrs == {}
        s.attrs["version"] = 1
        result = s + 1
        assert result.attrs == {"version": 1}

    def test_inspect_getmembers(self):
        # [FUNCTION DOC]
        # Smoke test (issue 38782): inspecting the members of an empty Series does not raise.
        # GH38782
        ser = Series(dtype=object)
        inspect.getmembers(ser)

    def test_unknown_attribute(self):
        # [FUNCTION DOC]
        # Verifies (issue 9680) that accessing a nonexistent attribute on a Series with a timedelta index raises an attribute error naming the Series class and the attribute, and does not create an entry in the instance dictionary.
        # GH#9680
        tdi = timedelta_range(start=0, periods=10, freq="1s")
        ser = Series(np.random.default_rng(2).normal(size=10), index=tdi)
        assert "foo" not in ser.__dict__
        msg = "'Series' object has no attribute 'foo'"
        with pytest.raises(AttributeError, match=msg):
            ser.foo

    @pytest.mark.parametrize("op", ["year", "day", "second", "weekday"])
    def test_datetime_series_no_datelike_attrs(self, op, datetime_series):
        # [FUNCTION DOC]
        # Verifies (issue 7206) that datetime-property names (year, day, second, weekday) are not accessible directly on a datetime Series; each raises an attribute error. Parametrized over the property name.
        # GH#7206
        msg = f"'Series' object has no attribute '{op}'"
        with pytest.raises(AttributeError, match=msg):
            getattr(datetime_series, op)

    def test_series_datetimelike_attribute_access(self):
        # [FUNCTION DOC]
        # Verifies that when a Series has labels named year, month and day, attribute access returns the corresponding values.
        # attribute access should still work!
        ser = Series({"year": 2000, "month": 1, "day": 10})
        assert ser.year == 2000
        assert ser.month == 1
        assert ser.day == 10

    def test_series_datetimelike_attribute_access_invalid(self):
        # [FUNCTION DOC]
        # Verifies that, for a Series with year/month/day labels, accessing a missing label as an attribute (weekday) raises an attribute error.
        ser = Series({"year": 2000, "month": 1, "day": 10})
        msg = "'Series' object has no attribute 'weekday'"
        with pytest.raises(AttributeError, match=msg):
            ser.weekday

    @pytest.mark.parametrize(
        "kernel, has_numeric_only",
        [
            ("skew", True),
            ("var", True),
            ("all", False),
            ("prod", True),
            ("any", False),
            ("idxmin", False),
            ("quantile", False),
            ("idxmax", False),
            ("min", True),
            ("sem", True),
            ("mean", True),
            ("nunique", False),
            ("max", True),
            ("sum", True),
            ("count", False),
            ("median", True),
            ("std", True),
            ("rank", True),
            ("pct_change", False),
            ("cummax", False),
            ("shift", False),
            ("diff", False),
            ("cumsum", False),
            ("cummin", False),
            ("cumprod", False),
            ("fillna", False),
            ("ffill", False),
            ("bfill", False),
            ("sample", False),
            ("tail", False),
            ("take", False),
            ("head", False),
            ("cov", False),
            ("corr", False),
        ],
    )
    @pytest.mark.parametrize("dtype", [bool, int, float, object])
    def test_numeric_only(self, kernel, has_numeric_only, dtype):
        # [FUNCTION DOC]
        # Verifies the numeric-only keyword (issue 47500) across many kernels (reductions, cumulative ops, rank, shift, fill methods, sampling, covariance and correlation), crossed with Series dtypes bool, int, float and object. Arguments are chosen per kernel: the other series for correlation and covariance, a probability for quantile, a position list for take, and a boolean for fill.
        # Kernels flagged as lacking the keyword must raise a type error about an unexpected keyword or too many arguments. For kernels that support it with an object dtype, a type error is raised stating it does not allow the flag with non-numeric data. Otherwise the result with the flag equals the result without it, compared as a Series for transformers or as a scalar for reducers.
        # KNOWN ISSUE: the argument-selection chain has dead branches: kernel names for cross-frame correlation and for the nth-row selection never appear in the parametrization, and the fill-method branch is listed twice with the same condition so the second (the forward-fill argument) can never run. As a result, fill receives a boolean argument.
        # GH#47500
        ser = Series([0, 1, 1], dtype=dtype)
        if kernel == "corrwith":
            args = (ser,)
        elif kernel == "corr":
            args = (ser,)
        elif kernel == "cov":
            args = (ser,)
        elif kernel == "nth":
            args = (0,)
        elif kernel == "fillna":
            args = (True,)
        elif kernel == "fillna":
            args = ("ffill",)
        elif kernel == "take":
            args = ([0],)
        elif kernel == "quantile":
            args = (0.5,)
        else:
            args = ()
        method = getattr(ser, kernel)
        if not has_numeric_only:
            msg = (
                "(got an unexpected keyword argument 'numeric_only'"
                "|too many arguments passed in)"
            )
            with pytest.raises(TypeError, match=msg):
                method(*args, numeric_only=True)
        elif dtype is object:
            msg = f"Series.{kernel} does not allow numeric_only=True with non-numeric"
            with pytest.raises(TypeError, match=msg):
                method(*args, numeric_only=True)
        else:
            result = method(*args, numeric_only=True)
            expected = method(*args, numeric_only=False)
            if isinstance(expected, Series):
                # transformer
                tm.assert_series_equal(result, expected)
            else:
                # reducer
                assert result == expected
