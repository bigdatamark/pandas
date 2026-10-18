# [MODULE DOC]
# Tests for Series.info (text summary written to a buffer): categorical data,
# MultiIndex, memory usage reporting (shallow, deep, with the "+" qualifier for
# estimated object memory), dtype listings, show_counts and argument errors.
# Uses the lexsorted_two_level_string_multiindex and using_infer_string fixtures
# and the PyPy and PyArrow compatibility flags. No import-time side effects
# besides evaluating the parametrization (which reads the string dtype config).

from io import StringIO
from string import ascii_uppercase
import textwrap

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import (
    HAS_PYARROW,
    PYPY,
)

from pandas import (
    CategoricalIndex,
    Index,
    MultiIndex,
    Series,
    date_range,
)


def test_info_categorical_column_just_works():
    # [FUNCTION DOC]
    # Smoke test: info works for a large categorical Series and for a filtered
    # subset of it; no output is checked.
    n = 2500
    data = np.array(list("abcdefghij")).take(
        np.random.default_rng(2).integers(0, 10, size=n, dtype=int)
    )
    s = Series(data).astype("category")
    s.isna()
    buf = StringIO()
    s.info(buf=buf)

    s2 = s[s == "d"]
    buf = StringIO()
    s2.info(buf=buf)


def test_info_categorical():
    # [FUNCTION DOC]
    # Smoke test (GH14298): info works for a Series with a categorical index.
    # GH14298
    idx = CategoricalIndex(["a", "b"])
    s = Series(np.zeros(2), index=idx)
    buf = StringIO()
    s.info(buf=buf)


@pytest.mark.parametrize("verbose", [True, False])
def test_info_series(
    lexsorted_two_level_string_multiindex, verbose, using_infer_string
):
    # [FUNCTION DOC]
    # Parametrized over verbose. Writes info for a named Series on a two-level string
    # MultiIndex to a buffer and compares the exact text: class line, MultiIndex range
    # line, optional (verbose only) name/count/dtype table, the dtypes line and the
    # memory usage line. The memory line has a "+" suffix unless the inferred
    # string option with pyarrow is active.
    index = lexsorted_two_level_string_multiindex
    ser = Series(range(len(index)), index=index, name="sth")
    buf = StringIO()
    ser.info(verbose=verbose, buf=buf)
    result = buf.getvalue()

    expected = textwrap.dedent(
        """\
        <class 'pandas.Series'>
        MultiIndex: 10 entries, ('foo', 'one') to ('qux', 'three')
        """
    )
    if verbose:
        expected += textwrap.dedent(
            """\
            Series name: sth
            Non-Null Count  Dtype
            --------------  -----
            10 non-null     int64
            """
        )
    qualifier = "" if using_infer_string and HAS_PYARROW else "+"
    expected += textwrap.dedent(
        f"""\
        dtypes: int64(1)
        memory usage: {ser.memory_usage()}.0{qualifier} bytes
        """
    )
    assert result == expected


def test_info_memory():
    # [FUNCTION DOC]
    # For a small integer Series, info output exactly matches the expected text with
    # a range index, a table with the non-null count and the memory size as a float.
    s = Series([1, 2], dtype="i8")
    buf = StringIO()
    s.info(buf=buf)
    result = buf.getvalue()
    memory_bytes = float(s.memory_usage())
    expected = textwrap.dedent(
        f"""\
    <class 'pandas.Series'>
    RangeIndex: 2 entries, 0 to 1
    Series name: None
    Non-Null Count  Dtype
    --------------  -----
    2 non-null      int64
    dtypes: int64(1)
    memory usage: {memory_bytes} bytes
    """
    )
    assert result == expected


def test_info_wide():
    # [FUNCTION DOC]
    # Passing max_cols to Series.info raises a value error saying that argument only
    # works for DataFrame.info.
    # KNOWN ISSUE: the test name and the 101-element data suggest it checks wide
    # output, but it only checks the max_cols error; the data length is
    # irrelevant.
    s = Series(np.random.default_rng(2).standard_normal(101))
    msg = "Argument `max_cols` can only be passed in DataFrame.info, not Series.info"
    with pytest.raises(ValueError, match=msg):
        s.info(max_cols=1)


def test_info_shows_dtypes():
    # [FUNCTION DOC]
    # For int64, float64, datetime64, timedelta64, complex128, object and bool data
    # (random 0/1 cast to each type), the info text contains the line "10 non-null"
    # with the dtype name.
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    n = 10
    for dtype in dtypes:
        s = Series(np.random.default_rng(2).integers(2, size=n).astype(dtype))
        buf = StringIO()
        s.info(buf=buf)
        res = buf.getvalue()
        name = f"{n:d} non-null     {dtype}"
        assert name in res


@pytest.mark.xfail(PYPY, reason="on PyPy deep=True doesn't change result")
def test_info_memory_usage_deep_not_pypy():
    # [FUNCTION DOC]
    # Expected to fail on PyPy. With a string-labelled index or object values, deep
    # memory usage is larger than shallow usage.
    s_with_object_index = Series({"a": [1]}, index=["foo"])
    assert s_with_object_index.memory_usage(
        index=True, deep=True
    ) > s_with_object_index.memory_usage(index=True)

    s_object = Series({"a": ["a"]})
    assert s_object.memory_usage(deep=True) > s_object.memory_usage()


@pytest.mark.xfail(not PYPY, reason="on PyPy deep=True does not change result")
def test_info_memory_usage_deep_pypy():
    # [FUNCTION DOC]
    # Expected to fail when not on PyPy. On PyPy deep memory usage equals shallow
    # usage, for the same Series as in the CPython test.
    s_with_object_index = Series({"a": [1]}, index=["foo"])
    assert s_with_object_index.memory_usage(
        index=True, deep=True
    ) == s_with_object_index.memory_usage(index=True)

    s_object = Series({"a": ["a"]})
    assert s_object.memory_usage(deep=True) == s_object.memory_usage()


@pytest.mark.parametrize(
    "index, plus",
    [
        ([1, 2, 3], False),
        (Index(list("ABC"), dtype="str"), not (using_string_dtype() and HAS_PYARROW)),
        (Index(list("ABC"), dtype=object), True),
        (MultiIndex.from_product([range(3), range(3)]), False),
        (
            MultiIndex.from_product([range(3), ["foo", "bar"]]),
            not (using_string_dtype() and HAS_PYARROW),
        ),
    ],
)
def test_info_memory_usage_qualified(index, plus):
    # [FUNCTION DOC]
    # Parametrized over (index, whether a "+" is expected): integer, string, object
    # and MultiIndex indexes. The info text contains a "+" in memory usage exactly
    # when the memory is an estimate (object-like index data), except that with the
    # string dtype option on and pyarrow installed the string cases are exact.
    buf = StringIO()
    series = Series(1, index=index)
    series.info(buf=buf)
    if plus:
        assert "+" in buf.getvalue()
    else:
        assert "+" not in buf.getvalue()


def test_info_memory_usage_bug_on_multiindex():
    # [FUNCTION DOC]
    # Regression test (GH 14308): memory usage of a MultiIndex Series does not
    # materialize the values. The unstacked frame has equal data bytes; the Series'
    # deep memory is larger than the frame's total, but by less than 2000 bytes.
    # GH 14308
    # memory usage introspection should not materialize .values
    N = 100
    M = len(ascii_uppercase)
    index = MultiIndex.from_product(
        [list(ascii_uppercase), date_range("20160101", periods=N)],
        names=["id", "date"],
    )
    s = Series(np.random.default_rng(2).standard_normal(N * M), index=index)

    unstacked = s.unstack("id")
    assert s.values.nbytes == unstacked.values.nbytes
    assert s.memory_usage(deep=True) > unstacked.memory_usage(deep=True).sum()

    # high upper bound
    diff = unstacked.memory_usage(deep=True).sum() - s.memory_usage(deep=True)
    assert diff < 2000


def test_info_show_counts_false():
    # [FUNCTION DOC]
    # With show_counts disabled, info prints a table with only the dtype column; the
    # output matches the expected text ignoring surrounding whitespace.
    s = Series([1])
    buf = StringIO()
    s.info(buf=buf, show_counts=False)
    result = buf.getvalue()
    memory_bytes = float(s.memory_usage())
    expected = textwrap.dedent(
        f"""\
    <class 'pandas.Series'>
    RangeIndex: 1 entries, 0 to 0
    Series name: None
    Dtype
    -----
    int64
    dtypes: int64(1)
    memory usage: {memory_bytes} bytes"""
    )
    assert result.strip() == expected.strip()
