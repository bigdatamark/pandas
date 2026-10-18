# [MODULE DOC]
# Tests for the base classes' behavior shared by Series, Index and DataFrame:
# the PandasDelegate accessor delegation machinery, the NoNewAttributesMixin
# freeze mechanism, and constructor dtype inference with out-of-bounds or
# non-nanosecond datetimes and dict key views. Defines the constructor fixture
# and two helper constructors. No import-time side effects.

from datetime import datetime
import sys

import numpy as np
import pytest

from pandas.compat import PYPY

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
    NoNewAttributesMixin,
    PandasObject,
)


def series_via_frame_from_dict(x, **kwargs):
    # [FUNCTION DOC]
    # HELPER: constructor adapter for the constructor fixture. Input: x (the data),
    # extra keyword arguments passed to the DataFrame constructor. Builds a one-column
    # DataFrame from a dict with column a and returns that column as a Series.
    return DataFrame({"a": x}, **kwargs)["a"]


def series_via_frame_from_scalar(x, **kwargs):
    # [FUNCTION DOC]
    # HELPER: constructor adapter for the constructor fixture. Input: x (the data),
    # extra keyword arguments passed to the DataFrame constructor. Builds a DataFrame
    # directly from x and returns its column labelled zero as a Series.
    return DataFrame(x, **kwargs)[0]


@pytest.fixture(
    params=[
        Series,
        series_via_frame_from_dict,
        series_via_frame_from_scalar,
        Index,
    ],
    ids=["Series", "DataFrame-dict", "DataFrame-array", "Index"],
)
def constructor(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over four ways to build a one-dimensional pandas
    # object: Series, a Series made through a DataFrame from a dict, a Series made
    # through a DataFrame from an array-like, and Index (ids Series, DataFrame-dict,
    # DataFrame-array, Index). Returns the constructor callable.
    return request.param


class TestPandasDelegate:
    # [CLASS DOC]
    # Tests the PandasDelegate mixin, which routes accessor attributes to a delegate
    # object. Defines two nested test helper classes, Delegator and Delegate.
    class Delegator:
        # [CLASS DOC]
        # HELPER: plain class acting as the delegate target in the tests. Declares the
        # property and method name lists used to register accessors (a property named prop
        # and a method named test_method) and implements a property with getter and setter.
        _properties = ["prop"]
        _methods = ["test_method"]

        def _set_prop(self, value):
            # [FUNCTION DOC]
            # HELPER: property setter for Delegator. Stores the given value in the prop attribute.
            self.prop = value

        def _get_prop(self):
            # [FUNCTION DOC]
            # HELPER: property getter for Delegator. Returns the stored prop attribute.
            return self.prop

        prop = property(_get_prop, _set_prop, doc="foo property")

        def test_method(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: no-op method of Delegator accepting any arguments; used as the
            # method accessor target.
            """a test method"""

    class Delegate(PandasDelegate, PandasObject):
        # [CLASS DOC]
        # HELPER: subclass of PandasDelegate and PandasObject that holds the wrapped object
        # in the obj attribute. It does not override the private delegate get, set or call
        # methods, so delegated access raises errors.
        def __init__(self, obj) -> None:
            # [FUNCTION DOC]
            # HELPER: initializer. Stores the given obj as the obj attribute.
            self.obj = obj

    def test_invalid_delegation(self):
        # [FUNCTION DOC]
        # Registers the property and the method of Delegator as delegate accessors
        # on Delegate (class-level side effect), then verifies reading the property raises
        # a type error ("You cannot access the property prop") and setting it raises a
        # type error ("The property prop cannot be set"), because the delegate hooks
        # are not overridden.
        # these show that in order for the delegation to work
        # the _delegate_* methods need to be overridden to not raise
        # a TypeError

        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator,
            accessors=self.Delegator._properties,
            typ="property",
        )
        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator, accessors=self.Delegator._methods, typ="method"
        )

        delegate = self.Delegate(self.Delegator())

        msg = "You cannot access the property prop"
        with pytest.raises(TypeError, match=msg):
            delegate.prop

        msg = "The property prop cannot be set"
        with pytest.raises(TypeError, match=msg):
            delegate.prop = 5

        msg = "You cannot access the property prop"
        with pytest.raises(TypeError, match=msg):
            delegate.prop

    @pytest.mark.skipif(PYPY, reason="not relevant for PyPy")
    def test_memory_usage(self):
        # [FUNCTION DOC]
        # Regression test (GH 12924). Skipped on PyPy. Verifies that the delegate object,
        # which does not implement memory usage, can be measured with the built-in size
        # function (falls back to the default sizeof).
        # Delegate does not implement memory_usage.
        # Check that we fall back to in-built `__sizeof__`
        # GH 12924
        delegate = self.Delegate(self.Delegator())
        sys.getsizeof(delegate)


class TestNoNewAttributesMixin:
    # [CLASS DOC]
    # Tests the NoNewAttributesMixin class.
    def test_mixin(self):
        # [FUNCTION DOC]
        # Defines a local subclass T of the mixin. Verifies a new instance has no frozen
        # marker, attributes can be added before freezing, and after the freeze method is
        # called adding a new attribute raises an attribute error ("You cannot add any
        # new attribute") and the attribute is not set.
        class T(NoNewAttributesMixin):
            # [CLASS DOC]
            # HELPER: empty local subclass of NoNewAttributesMixin used by the enclosing test.
            pass

        t = T()
        assert not hasattr(t, "__frozen")

        t.a = "test"
        assert t.a == "test"

        t._freeze()
        assert "__frozen" in dir(t)
        assert getattr(t, "__frozen")
        msg = "You cannot add any new attribute"
        with pytest.raises(AttributeError, match=msg):
            t.b = "test"

        assert not hasattr(t, "b")


class TestConstruction:
    # [CLASS DOC]
    # Groups tests of constructor dtype inference across Series, Index and DataFrame
    # (via the constructor fixture).
    # test certain constructor behaviours on dtype inference across Series,
    # Index and DataFrame

    @pytest.mark.parametrize(
        "a",
        [
            np.array(["2263-01-01"], dtype="datetime64[D]"),
            np.array([datetime(2263, 1, 1)], dtype=object),
            np.array([np.datetime64("2263-01-01", "D")], dtype=object),
            np.array(["2263-01-01"], dtype=object),
        ],
        ids=[
            "datetime64[D]",
            "object-datetime.datetime",
            "object-numpy-scalar",
            "object-string",
        ],
    )
    def test_constructor_datetime_outofbound(
        self, a, constructor, request, using_infer_string
    ):
        # [FUNCTION DOC]
        # Parametrized over four numpy inputs holding the year 2263 (outside nanosecond
        # bounds): datetime64 day, object arrays of Python datetimes, of numpy day scalars
        # and of strings (GH-26853, GH-26206). Without dtype: numpy-datetime inputs infer a
        # second-unit datetime dtype, Python datetime inputs a microsecond-unit dtype, and
        # strings stay object (string dtype if infer-string is on) with the original data
        # preserved. With an explicit nanosecond datetime dtype all cases raise an out of
        # bounds datetime error.
        # KNOWN ISSUE: the object-string branch repeats the constructor call that was
        # already made before the branching; the repeated call is redundant.
        # GH-26853 (+ bug GH-26206 out of bound non-ns unit)

        # No dtype specified (dtype inference)
        # datetime64[non-ns] raise error, other cases result in object dtype
        # and preserve original data
        result = constructor(a)
        if a.dtype.kind == "M" or isinstance(a[0], np.datetime64):
            # Can't fit in nanosecond bounds -> get the nearest supported unit
            assert result.dtype == "M8[s]"
        elif isinstance(a[0], datetime):
            assert result.dtype == "M8[us]", result.dtype
        else:
            result = constructor(a)
            if using_infer_string and "object-string" in request.node.callspec.id:
                assert result.dtype == "string"
            else:
                assert result.dtype == "object"
            tm.assert_numpy_array_equal(result.to_numpy(), a)

        # Explicit dtype specified
        # Forced conversion fails for all -> all cases raise error
        msg = "Out of bounds|Out of bounds .* present at position 0"
        with pytest.raises(pd.errors.OutOfBoundsDatetime, match=msg):
            constructor(a, dtype="datetime64[ns]")

    def test_constructor_datetime_nonns(self, constructor):
        # [FUNCTION DOC]
        # Verifies a microsecond datetime64 array and a DatetimeArray of the same unit
        # give equal results with that dtype preserved, also when the numpy array is read-only
        # (GitHub 34843).
        arr = np.array(["2020-01-01T00:00:00.000000"], dtype="datetime64[us]")
        dta = pd.core.arrays.DatetimeArray._simple_new(arr, dtype=arr.dtype)
        expected = constructor(dta)
        assert expected.dtype == arr.dtype

        result = constructor(arr)
        tm.assert_equal(result, expected)

        # https://github.com/pandas-dev/pandas/issues/34843
        arr.flags.writeable = False
        result = constructor(arr)
        tm.assert_equal(result, expected)

    def test_constructor_from_dict_keys(self, constructor, using_infer_string):
        # [FUNCTION DOC]
        # Regression test (GitHub 60343). Verifies building from a dict key view with the
        # str dtype gives object dtype, or str dtype when infer-string mode is on, and equals
        # the result from a list of the keys.
        # https://github.com/pandas-dev/pandas/issues/60343
        d = {"a": 1, "b": 2}
        result = constructor(d.keys(), dtype="str")
        if using_infer_string:
            assert result.dtype == "str"
        else:
            assert result.dtype == "object"
        expected = constructor(list(d.keys()), dtype="str")
        tm.assert_equal(result, expected)
