# [MODULE DOC]
# Tests that subclasses of Categorical keep their type through the constructor, the codes-based
# constructor and map. Defines a trivial subclass used only here. No import-time side effects.

from pandas import Categorical
import pandas._testing as tm


class SubclassedCategorical(Categorical):
    # [CLASS DOC]
    # HELPER: empty subclass of Categorical used to verify that class methods and map return the subclass type. Adds no state or behavior.
    pass


class TestCategoricalSubclassing:
    # [CLASS DOC]
    # Groups the subclass-preservation tests. Stateless.
    def test_constructor(self):
        # [FUNCTION DOC]
        # Constructing the subclass gives an instance of it equal in content to a plain Categorical.
        sc = SubclassedCategorical(["a", "b", "c"])
        assert isinstance(sc, SubclassedCategorical)
        tm.assert_categorical_equal(sc, Categorical(["a", "b", "c"]))

    def test_from_codes(self):
        # [FUNCTION DOC]
        # The codes-based class constructor called on the subclass returns an instance of the subclass with the expected content.
        sc = SubclassedCategorical.from_codes([1, 0, 2], ["a", "b", "c"])
        assert isinstance(sc, SubclassedCategorical)
        exp = Categorical.from_codes([1, 0, 2], ["a", "b", "c"])
        tm.assert_categorical_equal(sc, exp)

    def test_map(self):
        # [FUNCTION DOC]
        # Mapping a subclass instance with an upper-casing function (missing-value action unset) returns the subclass with mapped categories.
        sc = SubclassedCategorical(["a", "b", "c"])
        res = sc.map(lambda x: x.upper(), na_action=None)
        assert isinstance(res, SubclassedCategorical)
        exp = Categorical(["A", "B", "C"])
        tm.assert_categorical_equal(res, exp)
