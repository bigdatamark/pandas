# [MODULE DOC]
# Tests for the accessor registration machinery (register accessor decorators for
# Series, DataFrame and Index) and the mixin that controls attribute listing. Uses a
# sample accessor class and a context manager that cleans up registered accessors,
# since registration mutates the global pandas classes.
from collections.abc import Generator
import contextlib
import weakref

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core import accessor


def test_dirname_mixin() -> None:
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Defines a local class using the directory-names
    # mixin with a class attribute, an annotation-only attribute and an instance attribute
    # set in the constructor. Verifies the public attribute listing contains only the
    # class attribute and the instance attribute (the annotation-only name is absent).
    # GH37173

    class X(accessor.DirNamesMixin):
        # [CLASS DOC]
        # HELPER: local class used by the directory-names mixin test.
        x = 1
        y: int

        def __init__(self) -> None:
            # [FUNCTION DOC]
            # HELPER: sets an instance attribute used to check it appears in the attribute listing.
            self.z = 3

    result = [attr_name for attr_name in dir(X()) if not attr_name.startswith("_")]

    assert result == ["x", "z"]


@contextlib.contextmanager
def ensure_removed(obj, attr) -> Generator[None, None, None]:
    # [FUNCTION DOC]
    # HELPER: context manager that yields to the test body and afterwards, even on error,
    # removes the named attribute from the given class (ignoring a missing attribute) and
    # discards the name from the class's registered-accessor set. Parameters: obj is the
    # class, attr is the accessor name. Returns a generator for use as a context manager.
    """Ensure that an attribute added to 'obj' during the test is
    removed when we're done
    """
    try:
        yield
    finally:
        try:
            delattr(obj, attr)
        except AttributeError:
            pass
        obj._accessors.discard(attr)


class MyAccessor:
    # [CLASS DOC]
    # HELPER: sample accessor class used by the registration tests. Stores the wrapped
    # object and a fixed string item; exposes a property and a method that both return it.
    def __init__(self, obj) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the wrapped object and sets the item string to a fixed value.
        self.obj = obj
        self.item = "item"

    @property
    def prop(self):
        # [FUNCTION DOC]
        # HELPER: property returning the stored item string.
        return self.item

    def method(self):
        # [FUNCTION DOC]
        # HELPER: method returning the stored item string.
        return self.item


@pytest.mark.parametrize(
    "obj, registrar",
    [
        (pd.Series, pd.api.extensions.register_series_accessor),
        (pd.DataFrame, pd.api.extensions.register_dataframe_accessor),
        (pd.Index, pd.api.extensions.register_index_accessor),
    ],
)
def test_register(obj, registrar):
    # [FUNCTION DOC]
    # Parametrized over (class, registration decorator) for Series, DataFrame and Index.
    # Verifies registering the sample accessor under a name adds exactly that one name to
    # the class's attribute listing, that the accessor property returns the item on an
    # empty instance, and that the name is recorded in the class's accessor set.
    with ensure_removed(obj, "mine"):
        before = set(dir(obj))
        registrar("mine")(MyAccessor)
        o = obj([]) if obj is not pd.Series else obj([], dtype=object)
        assert o.mine.prop == "item"
        after = set(dir(obj))
        assert (before ^ after) == {"mine"}
        assert "mine" in obj._accessors


def test_accessor_works():
    # [FUNCTION DOC]
    # Verifies a registered Series accessor receives the Series itself as its wrapped
    # object and that its property and method return the item string.
    with ensure_removed(pd.Series, "mine"):
        pd.api.extensions.register_series_accessor("mine")(MyAccessor)

        s = pd.Series([1, 2])
        assert s.mine.obj is s

        assert s.mine.prop == "item"
        assert s.mine.method() == "item"


def test_overwrite_warns():
    # [FUNCTION DOC]
    # Verifies that registering an accessor under a name already set on Series emits a
    # user warning naming the accessor class, the name and Series, and that the new
    # accessor still works afterwards.
    match = r".*MyAccessor.*fake.*Series.*"
    with tm.assert_produces_warning(UserWarning, match=match):
        with ensure_removed(pd.Series, "fake"):
            setattr(pd.Series, "fake", 123)
            pd.api.extensions.register_series_accessor("fake")(MyAccessor)
            s = pd.Series([1, 2])
            assert s.fake.prop == "item"


def test_raises_attribute_error():
    # [FUNCTION DOC]
    # Verifies that an accessor whose constructor raises an attribute error propagates
    # that error (with its message) when the accessor is accessed, rather than being
    # swallowed into a generic missing-attribute error.
    with ensure_removed(pd.Series, "bad"):

        @pd.api.extensions.register_series_accessor("bad")
        class Bad:
            # [CLASS DOC]
            # HELPER: accessor class whose constructor always raises an attribute error.
            def __init__(self, data) -> None:
                # [FUNCTION DOC]
                # HELPER: always raises an attribute error with a fixed message.
                raise AttributeError("whoops")

        with pytest.raises(AttributeError, match="whoops"):
            pd.Series([], dtype=object).bad


@pytest.mark.parametrize(
    "klass, registrar",
    [
        (pd.Series, pd.api.extensions.register_series_accessor),
        (pd.DataFrame, pd.api.extensions.register_dataframe_accessor),
        (pd.Index, pd.api.extensions.register_index_accessor),
    ],
)
def test_no_circular_reference(klass, registrar):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over class and registration
    # decorator. Verifies the accessor holds the object, and that after the object is
    # deleted its weak reference is dead, i.e. the accessor creates no reference cycle
    # keeping the object alive.
    # GH 41357
    with ensure_removed(klass, "access"):
        registrar("access")(MyAccessor)
        obj = klass([0])
        ref = weakref.ref(obj)
        assert obj.access.obj is obj
        del obj
        assert ref() is None
