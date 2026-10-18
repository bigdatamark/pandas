# [MODULE DOC]
# Tests for CategoricalIndex.map: preserving categorical-ness and orderedness, results that
# are plain Indexes, mapping with dict or Series, and NA handling by na_action.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    Index,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "data, categories",
    [
        (list("abcbca"), list("cab")),
        (pd.interval_range(0, 3).repeat(3), pd.interval_range(0, 3)),
    ],
    ids=["string", "interval"],
)
def test_map_str(data, categories, ordered):
    # [FUNCTION DOC]
    # Parametrized over string and interval data (ids given), using the ordered fixture. Verifies
    # mapping with the string conversion function gives a CategoricalIndex whose values and
    # categories are converted and whose orderedness is preserved.
    # GH 31202 - override base class since we want to maintain categorical/ordered
    index = CategoricalIndex(data, categories=categories, ordered=ordered)
    result = index.map(str)
    expected = CategoricalIndex(
        map(str, data), categories=map(str, categories), ordered=ordered
    )
    tm.assert_index_equal(result, expected)


def test_map():
    # [FUNCTION DOC]
    # Verifies mapping a function over a CategoricalIndex keeps categories (transformed in order),
    # orderedness and name; a function returning a constant gives a plain int64 Index with the name;
    # a function changing the category dtype (via dictionary lookup) gives the new categorical
    # (categories in order of first appearance of original categories); a Series mapper and a dict
    # mapper give the same result.
    ci = CategoricalIndex(list("ABABC"), categories=list("CBA"), ordered=True)
    result = ci.map(lambda x: x.lower())
    exp = CategoricalIndex(list("ababc"), categories=list("cba"), ordered=True)
    tm.assert_index_equal(result, exp)

    ci = CategoricalIndex(
        list("ABABC"), categories=list("BAC"), ordered=False, name="XXX"
    )
    result = ci.map(lambda x: x.lower())
    exp = CategoricalIndex(
        list("ababc"), categories=list("bac"), ordered=False, name="XXX"
    )
    tm.assert_index_equal(result, exp)

    # GH 12766: Return an index not an array
    tm.assert_index_equal(
        ci.map(lambda x: 1), Index(np.array([1] * 5, dtype=np.int64), name="XXX")
    )

    # change categories dtype
    ci = CategoricalIndex(list("ABABC"), categories=list("BAC"), ordered=False)

    def f(x):
        # [FUNCTION DOC]
        # HELPER: used by test_map; looks up the letter in a dictionary mapping A, B, C to 10, 20, 30
        # and returns the value (or none if absent).
        return {"A": 10, "B": 20, "C": 30}.get(x)

    result = ci.map(f)
    exp = CategoricalIndex([10, 20, 10, 20, 30], categories=[20, 10, 30], ordered=False)
    tm.assert_index_equal(result, exp)

    result = ci.map(Series([10, 20, 30], index=["A", "B", "C"]))
    tm.assert_index_equal(result, exp)

    result = ci.map({"A": 10, "B": 20, "C": 30})
    tm.assert_index_equal(result, exp)


def test_map_with_categorical_series():
    # [FUNCTION DOC]
    # Regression test: mapping an integer Index through a categorical Series gives a
    # CategoricalIndex with NaN for unmapped value, whereas an object Series gives a plain Index.
    # GH 12756
    a = Index([1, 2, 3, 4])
    b = Series(["even", "odd", "even", "odd"], dtype="category")
    c = Series(["even", "odd", "even", "odd"])

    exp = CategoricalIndex(["odd", "even", "odd", np.nan])
    tm.assert_index_equal(a.map(b), exp)
    exp = Index(["odd", "even", "odd", np.nan])
    tm.assert_index_equal(a.map(c), exp)


@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        ([1, 1, np.nan], pd.isna, CategoricalIndex([False, False, np.nan])),
        ([1, 2, np.nan], pd.isna, Index([False, False, np.nan])),
        ([1, 1, np.nan], {1: False}, CategoricalIndex([False, False, np.nan])),
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),
        (
            [1, 1, np.nan],
            Series([False, False]),
            CategoricalIndex([False, False, np.nan]),
        ),
        (
            [1, 2, np.nan],
            Series([False, False, False]),
            Index([False, False, np.nan]),
        ),
    ),
)
def test_map_with_nan_ignore(data, f, expected):  # GH 24241
    # [FUNCTION DOC]
    # Parametrized over (data, mapper, expected): a function that detects missing values, a dict,
    # and a Series, each with data having duplicates (stays categorical) or not (plain Index).
    # With the na_action set to ignore, NaN entries are left as NaN and not passed to the mapper.
    values = CategoricalIndex(data)
    result = values.map(f, na_action="ignore")
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        ([1, 1, np.nan], pd.isna, Index([False, False, True])),
        ([1, 2, np.nan], pd.isna, Index([False, False, True])),
        ([1, 1, np.nan], {1: False}, CategoricalIndex([False, False, np.nan])),
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),
        (
            [1, 1, np.nan],
            Series([False, False]),
            CategoricalIndex([False, False, np.nan]),
        ),
        (
            [1, 2, np.nan],
            Series([False, False, False]),
            Index([False, False, np.nan]),
        ),
    ),
)
def test_map_with_nan_none(data, f, expected):  # GH 24241
    # [FUNCTION DOC]
    # Same parametrization style but with na_action None: the missing-value detecting function is
    # applied to NaN too (giving True), while dict and Series mappers behave as before.
    values = CategoricalIndex(data)
    result = values.map(f, na_action=None)
    tm.assert_index_equal(result, expected)


def test_map_with_dict_or_series():
    # [FUNCTION DOC]
    # Verifies mapping a categorical index of mixed-type labels with a Series or dict mapper
    # gives a categorical index with the mapped values (category order may differ) and the same
    # name.
    orig_values = ["a", "B", 1, "a"]
    new_values = ["one", 2, 3.0, "one"]
    cur_index = CategoricalIndex(orig_values, name="XXX")
    expected = CategoricalIndex(new_values, name="XXX", categories=[3.0, 2, "one"])

    mapper = Series(new_values[:-1], index=orig_values[:-1])
    result = cur_index.map(mapper)
    # Order of categories in result can be different
    tm.assert_index_equal(result, expected)

    mapper = dict(zip(orig_values[:-1], new_values[:-1]))
    result = cur_index.map(mapper)
    # Order of categories in result can be different
    tm.assert_index_equal(result, expected)
