# [MODULE DOC]
# Tests for the abstract base classes in the generic dtypes module (named
# starting with ABC), which let code check pandas object types without
# importing the concrete classes: instance and subclass checks for sample
# objects, the class hierarchy, coverage of all defined ABCs, plus a test about
# attribute assignment warnings.

import re

import numpy as np
import pytest

from pandas.core.dtypes import generic as gt

import pandas as pd
import pandas._testing as tm


class TestABCClasses:
    # [CLASS DOC]
    # Test group with class-level fixtures built at import time: sample indexes
    # (multi, datetime, timedelta, period, categorical, range, interval), arrays
    # (categorical, sparse, datetime, timedelta, period, numpy extension), Series and
    # frames; a list of pairs of an ABC name and a matching sample instance; and a
    # mapping of parent ABC names to lists of subclass ABC names (index types,
    # Series/DataFrame under the N-dimensional frame ABC, and the extension array
    # ones). Tests iterate over these.
    tuples = [[1, 2, 2], ["red", "blue", "red"]]
    multi_index = pd.MultiIndex.from_arrays(tuples, names=("number", "color"))
    datetime_index = pd.to_datetime(["2000/1/1", "2010/1/1"])
    timedelta_index = pd.to_timedelta(np.arange(5), unit="s")
    period_index = pd.period_range("2000/1/1", "2010/1/1/", freq="M")
    categorical = pd.Categorical([1, 2, 3], categories=[2, 3, 1])
    categorical_df = pd.DataFrame({"values": [1, 2, 3]}, index=categorical)
    df = pd.DataFrame({"names": ["a", "b", "c"]}, index=multi_index)
    sparse_array = pd.arrays.SparseArray(np.random.default_rng(2).standard_normal(10))

    datetime_array = datetime_index.array
    timedelta_array = timedelta_index.array

    abc_pairs = [
        ("ABCMultiIndex", multi_index),
        ("ABCDatetimeIndex", datetime_index),
        ("ABCRangeIndex", pd.RangeIndex(3)),
        ("ABCTimedeltaIndex", timedelta_index),
        ("ABCIntervalIndex", pd.interval_range(start=0, end=3)),
        (
            "ABCPeriodArray",
            pd.arrays.PeriodArray([2000, 2001, 2002], dtype="period[D]"),
        ),
        ("ABCNumpyExtensionArray", pd.arrays.NumpyExtensionArray(np.array([0, 1, 2]))),
        ("ABCPeriodIndex", period_index),
        ("ABCCategoricalIndex", categorical_df.index),
        ("ABCSeries", pd.Series([1, 2, 3])),
        ("ABCDataFrame", df),
        ("ABCCategorical", categorical),
        ("ABCDatetimeArray", datetime_array),
        ("ABCTimedeltaArray", timedelta_array),
    ]

    @pytest.mark.parametrize("abctype1, inst", abc_pairs)
    @pytest.mark.parametrize("abctype2, _", abc_pairs)
    def test_abc_pairs_instance_check(self, abctype1, abctype2, inst, _):
        # [FUNCTION DOC]
        # Regression test for GitHub issues; parametrized over all pairs of ABC entries.
        # For the same ABC an instance is an instance of it, and its type object is not an
        # instance. For different ABCs the instance is not an instance of the other ABC.
        # GH 38588, 46719
        if abctype1 == abctype2:
            assert isinstance(inst, getattr(gt, abctype2))
            assert not isinstance(type(inst), getattr(gt, abctype2))
        else:
            assert not isinstance(inst, getattr(gt, abctype2))

    @pytest.mark.parametrize("abctype1, inst", abc_pairs)
    @pytest.mark.parametrize("abctype2, _", abc_pairs)
    def test_abc_pairs_subclass_check(self, abctype1, abctype2, inst, _):
        # [FUNCTION DOC]
        # Parametrized over all pairs of ABC entries. For the same ABC the instance's
        # type is a subclass of the ABC and checking the instance itself raises a type
        # error saying the first argument must be a class; for different ABCs the type is
        # not a subclass.
        # GH 38588, 46719
        if abctype1 == abctype2:
            assert issubclass(type(inst), getattr(gt, abctype2))

            with pytest.raises(
                TypeError, match=re.escape("issubclass() arg 1 must be a class")
            ):
                issubclass(inst, getattr(gt, abctype2))
        else:
            assert not issubclass(type(inst), getattr(gt, abctype2))

    abc_subclasses = {
        "ABCIndex": [
            abctype
            for abctype, _ in abc_pairs
            if "Index" in abctype and abctype != "ABCIndex"
        ],
        "ABCNDFrame": ["ABCSeries", "ABCDataFrame"],
        "ABCExtensionArray": [
            "ABCCategorical",
            "ABCDatetimeArray",
            "ABCPeriodArray",
            "ABCTimedeltaArray",
        ],
    }

    @pytest.mark.parametrize("parent, subs", abc_subclasses.items())
    @pytest.mark.parametrize("abctype, inst", abc_pairs)
    def test_abc_hierarchy(self, parent, subs, abctype, inst):
        # [FUNCTION DOC]
        # Parametrized over (parent ABC, list of sub ABC names) and each sample pair.
        # The instance is an instance of the parent ABC exactly when its ABC name is
        # listed among that parent's subclasses.
        # GH 38588
        if abctype in subs:
            assert isinstance(inst, getattr(gt, parent))
        else:
            assert not isinstance(inst, getattr(gt, parent))

    @pytest.mark.parametrize("abctype", [e for e in gt.__dict__ if e.startswith("ABC")])
    def test_abc_coverage(self, abctype):
        # [FUNCTION DOC]
        # Parametrized over all names defined in the generic module that start with ABC.
        # Each must appear either in the sample pairs or as a key of the hierarchy mapping, so
        # newly added ABCs get tests.
        # GH 38588
        assert (
            abctype in (e for e, _ in self.abc_pairs) or abctype in self.abc_subclasses
        )


def test_setattr_warnings():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue about attribute-style column assignment.
    # Adding a column via item assignment, modifying a column in place through
    # attribute access, and setting a new attribute on a column Series do not warn.
    # Assigning to a non-existent column name through attribute access emits a user
    # warning saying it does not allow columns to be created this way, though the
    # attribute is set.
    # GH7175 - GOTCHA: You can't use dot notation to add a column...
    d = {
        "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
        "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
    }
    df = pd.DataFrame(d)

    with tm.assert_produces_warning(None):
        #  successfully add new column
        #  this should not raise a warning
        df["three"] = df.two + 1
        assert df.three.sum() > df.two.sum()

    with tm.assert_produces_warning(None):
        #  successfully modify column in place
        #  this should not raise a warning
        df.one += 1
        assert df.one.iloc[0] == 2

    with tm.assert_produces_warning(None):
        #  successfully add an attribute to a series
        #  this should not raise a warning
        df.two.not_an_index = [1, 2]

    with tm.assert_produces_warning(UserWarning, match="doesn't allow columns"):
        #  warn when setting column to nonexistent name
        df.four = df.two + 2
        assert df.four.sum() > df.two.sum()
