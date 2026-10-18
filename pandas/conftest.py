"""
This file is very long and growing, but it was decided to not split it yet, as
it's still manageable (2020-03-17, ~1.1k LoC). See gh-31989

Instead of splitting it was decided to define sections here:
- Configuration / Settings
- Autouse fixtures
- Common arguments
- Missing values & co.
- Classes
- Indices
- Series'
- DataFrames
- Operators & Operations
- Data sets/files
- Time zones
- Dtypes
- Misc
"""

# [MODULE DOC]
# Root pytest configuration for the whole pandas test suite and doctests. It is
# loaded automatically by pytest and provides (1) pytest hooks (command line option
# registration, session start patch of the doctest finder, doctest warning
# filtering), (2) hypothesis configuration (a CI profile loaded at import and
# strategies registered for several date offset types), (3) autouse fixtures, and
# (4) a very large collection of shared fixtures grouped in sections: common
# arguments, missing values, classes, indices, Series, DataFrames, scalars,
# operators and operations, data files, time zones, dtypes and miscellaneous.
# Many fixtures are parametrized over lists defined in this module (for example the
# dictionary of simple indices, from which Series dictionaries are derived) and
# reuse helper values from the testing utilities.
# Import-time side effects: detects whether pyarrow is installed (module flag),
# optionally imports pytz, registers and loads a hypothesis settings profile,
# registers hypothesis type strategies for month, year and quarter offsets, and
# builds module-level dictionaries of index and Series objects used as fixture params.
from __future__ import annotations

from collections import abc
from datetime import (
    date,
    datetime,
    time,
    timedelta,
    timezone,
)
from decimal import Decimal
import gc
import operator
import os
from typing import (
    TYPE_CHECKING,
    Any,
)
import uuid

from dateutil.tz import (
    tzlocal,
    tzutc,
)
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest

from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td

from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
)

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Interval,
    IntervalIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterator,
    )

try:
    import pyarrow as pa
except ImportError:
    has_pyarrow = False
else:
    del pa
    has_pyarrow = True

pytz = import_optional_dependency("pytz", errors="ignore")


# ----------------------------------------------------------------
# Configuration / Settings
# ----------------------------------------------------------------
# pytest


def pytest_addoption(parser) -> None:
    # [FUNCTION DOC]
    # pytest hook that registers a command line option. Input: parser (pytest
    # argument parser). Adds the no-strict-data-files flag, which stores a false
    # value when given (so by default data files are strict) and is used to avoid
    # failing when a test is skipped for a missing data file. Returns nothing.
    parser.addoption(
        "--no-strict-data-files",
        action="store_false",
        help="Don't fail if a test is skipped for missing data file.",
    )


def pytest_sessionstart(session):
    # [FUNCTION DOC]
    # pytest hook run once at session start. Input: session (unused). Side effect
    # (global, process wide): replaces the module-matching check of the standard
    # library doctest finder with a version that accepts any method of a class
    # (detected by a dot in the qualified name) without comparing the method's
    # module, because pandas changes the module attribute of public classes but not
    # of their methods, which would otherwise make the doctests be skipped. Other
    # objects fall through to the original check.
    import doctest
    import inspect

    # https://github.com/pandas-dev/pandas/pull/62988
    # When we modify the __module__ of a class, the __module__ on the methods
    # of that class do not change. When these two disagree, doctests would not
    # typically run. We hack `DocTestFinder` to avoid this.
    orig = doctest.DocTestFinder._from_module  # type: ignore[attr-defined]

    def _from_module(self, module, object):
        # [FUNCTION DOC]
        # HELPER: replacement for the doctest finder's module membership check.
        # Inputs: self (finder), module and object. Returns true for plain functions
        # whose qualified name contains a dot (methods); otherwise returns what the
        # original check returns.
        # When . is in __qualname__, object is a method of a class.
        if inspect.isfunction(object) and "." in object.__qualname__:
            # We only get here when the class that the method is on is from the
            # appropriate module. So ignore checking the __module__ of the method
            # itself and run the doctest.
            return True
        return orig(self, module, object)

    doctest.DocTestFinder._from_module = _from_module  # type: ignore[attr-defined]


def ignore_doctest_warning(item: pytest.Item, path: str, message: str) -> None:
    # [FUNCTION DOC]
    # HELPER: for the doctest warning filtering hook. Inputs: item (test item), path
    # (suffix of the item's name to match) and message (warning text prefix to
    # ignore). If the item name ends with the path, adds a marker that ignores
    # warnings starting with the message (mutates the item). Returns nothing.
    """Ignore doctest warning.

    Parameters
    ----------
    item : pytest.Item
        pytest test item.
    path : str
        Module path to Python object, e.g. "pandas.DataFrame.append". A
        warning will be filtered when item.name ends with in given path. So it is
        sufficient to specify e.g. "DataFrame.append".
    message : str
        Message to be filtered.
    """
    if item.name.endswith(path):
        item.add_marker(pytest.mark.filterwarnings(f"ignore:{message}"))


def pytest_collection_modifyitems(items, config) -> None:
    # [FUNCTION DOC]
    # pytest hook called after test collection. Inputs: items (collected tests,
    # mutated in place) and config. If running doctests (module doctests or the
    # Cython doctest option is on), applies a built-in table of (name suffix,
    # warning message) pairs to every item through the warning ignore helper, so
    # known deprecation or numerical warnings emitted by documented examples do not
    # fail the run. Does nothing for ordinary test runs. Returns nothing.
    is_doctest = config.getoption("--doctest-modules") or config.getoption(
        "--doctest-cython", default=False
    )

    # Warnings from doctests that can be ignored; place reason in comment above.
    # Each entry specifies (path, message) - see the ignore_doctest_warning function
    ignored_doctest_warnings = [
        ("api.interchange.from_dataframe", "The DataFrame Interchange Protocol"),
        ("is_int64_dtype", "is_int64_dtype is deprecated"),
        ("is_interval_dtype", "is_interval_dtype is deprecated"),
        ("is_period_dtype", "is_period_dtype is deprecated"),
        ("is_datetime64tz_dtype", "is_datetime64tz_dtype is deprecated"),
        ("is_categorical_dtype", "is_categorical_dtype is deprecated"),
        ("is_sparse", "is_sparse is deprecated"),
        ("CategoricalDtype._from_values_or_dtype", "Constructing a Categorical"),
        ("DataFrame.__dataframe__", "The DataFrame Interchange Protocol"),
        ("DataFrameGroupBy.fillna", "DataFrameGroupBy.fillna is deprecated"),
        ("DataFrameGroupBy.corrwith", "DataFrameGroupBy.corrwith is deprecated"),
        ("NDFrame.replace", "Series.replace without 'value'"),
        ("NDFrame.clip", "Downcasting behavior in Series and DataFrame methods"),
        ("Series.idxmin", "The behavior of Series.idxmin"),
        ("Series.idxmax", "The behavior of Series.idxmax"),
        ("SeriesGroupBy.fillna", "SeriesGroupBy.fillna is deprecated"),
        ("SeriesGroupBy.idxmin", "The behavior of Series.idxmin"),
        ("SeriesGroupBy.idxmax", "The behavior of Series.idxmax"),
        ("to_pytimedelta", "The behavior of TimedeltaProperties.to_pytimedelta"),
        ("NDFrame.reindex_like", "keyword argument 'method' is deprecated"),
        # Docstring divides by zero to show behavior difference
        ("missing.mask_zero_div_zero", "divide by zero encountered"),
        (
            "pandas.core.generic.NDFrame.first",
            "first is deprecated and will be removed in a future version. "
            "Please create a mask and filter using `.loc` instead",
        ),
        (
            "Resampler.fillna",
            "DatetimeIndexResampler.fillna is deprecated",
        ),
        (
            "DataFrameGroupBy.fillna",
            "DataFrameGroupBy.fillna with 'method' is deprecated",
        ),
        ("read_parquet", "Passing a BlockManager to DataFrame is deprecated"),
    ]

    if is_doctest:
        for item in items:
            for path, message in ignored_doctest_warnings:
                ignore_doctest_warning(item, path, message)


# Similar to "ci" config in
# https://hypothesis.readthedocs.io/en/latest/reference/api.html#built-in-profiles
hypothesis.settings.register_profile(
    "pandas_ci",
    database=None,
    deadline=None,
    max_examples=15,
    suppress_health_check=(
        hypothesis.HealthCheck.too_slow,
        hypothesis.HealthCheck.differing_executors,
    ),
)
hypothesis.settings.load_profile("pandas_ci")

# Registering these strategies makes them globally available via st.from_type,
# which is use for offsets in tests/tseries/offsets/test_offsets_properties.py
for name in "MonthBegin MonthEnd BMonthBegin BMonthEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls, st.builds(cls, n=st.integers(-99, 99), normalize=st.booleans())
    )

for name in "YearBegin YearEnd BYearBegin BYearEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls,
        st.builds(
            cls,
            n=st.integers(-5, 5),
            normalize=st.booleans(),
            month=st.integers(min_value=1, max_value=12),
        ),
    )

for name in "QuarterBegin QuarterEnd BQuarterBegin BQuarterEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls,
        st.builds(
            cls,
            n=st.integers(-24, 24),
            normalize=st.booleans(),
            startingMonth=st.integers(min_value=1, max_value=12),
        ),
    )


# ----------------------------------------------------------------
# Autouse fixtures
# ----------------------------------------------------------------


# https://github.com/pytest-dev/pytest/issues/11873
# Would like to avoid autouse=True, but cannot as of pytest 8.0.0
@pytest.fixture(autouse=True)
def add_doctest_imports(doctest_namespace) -> None:
    # [FUNCTION DOC]
    # Autouse fixture. Input: doctest_namespace (pytest's mutable namespace used by
    # doctests). Adds the names np (numpy) and pd (pandas) to it so doctest
    # examples can use them without importing. Returns nothing.
    """
    Make `np` and `pd` names available for doctests.
    """
    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd


@pytest.fixture(autouse=True)
def configure_tests() -> None:
    # [FUNCTION DOC]
    # Autouse fixture run for every test. Sets the global pandas option for chained
    # assignment to raise an error, and never restores it (global state). Returns
    # nothing.
    """
    Configure settings for all tests and test modules.
    """
    pd.set_option("chained_assignment", "raise")


# ----------------------------------------------------------------
# Common arguments
# ----------------------------------------------------------------
@pytest.fixture(params=[0, 1, "index", "columns"], ids=lambda x: f"axis={x!r}")
def axis(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the axis spellings zero, one, index and columns
    # (test ids show the value). Returns the current parameter.
    """
    Fixture for returning the axis numbers of a DataFrame.
    """
    return request.param


@pytest.fixture(params=[True, False])
def observed(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false for the groupby observed keyword.
    # Returns the current parameter.
    """
    Pass in the observed keyword to groupby for [True, False]
    This indicates whether categoricals should return values for
    values which are not in the grouper [False / None], or only values which
    appear in the grouper [True]. [None] is supported for future compatibility
    if we decide to change the default (and would need to warn if this
    parameter is not passed).
    """
    return request.param


@pytest.fixture(params=[True, False, None])
def ordered(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true, false and none for the categorical ordered
    # argument. Returns the current parameter.
    """
    Boolean 'ordered' parameter for Categorical.
    """
    return request.param


@pytest.fixture(params=[True, False])
def dropna(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false for a dropna argument. Returns the
    # current parameter.
    """
    Boolean 'dropna' parameter.
    """
    return request.param


@pytest.fixture(params=[True, False])
def sort(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false for a sort argument. Returns the
    # current parameter.
    """
    Boolean 'sort' parameter.
    """
    return request.param


@pytest.fixture(params=[True, False])
def skipna(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false for a skipna argument. Returns the
    # current parameter.
    """
    Boolean 'skipna' parameter.
    """
    return request.param


@pytest.fixture(params=["first", "last", False])
def keep(request):
    # [FUNCTION DOC]
    # Parametrized fixture over first, last and false for the keep argument of
    # duplicate detection. Returns the current parameter.
    """
    Valid values for the 'keep' parameter used in
    .duplicated or .drop_duplicates
    """
    return request.param


@pytest.fixture(params=["both", "neither", "left", "right"])
def inclusive_endpoints_fixture(request):
    # [FUNCTION DOC]
    # Parametrized fixture over both, neither, left and right for interval
    # inclusiveness. Returns the current parameter.
    """
    Fixture for trying all interval 'inclusive' parameters.
    """
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def closed(request):
    # [FUNCTION DOC]
    # Parametrized fixture over left, right, both and neither for the interval
    # closed argument. Returns the current parameter.
    """
    Fixture for trying all interval closed parameters.
    """
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def other_closed(request):
    # [FUNCTION DOC]
    # Same parameter set as the closed fixture, provided as a second fixture so
    # tests can parametrize over all pairs of closed values. Returns the current
    # parameter.
    """
    Secondary closed fixture to allow parametrizing over all pairs of closed.
    """
    return request.param


@pytest.fixture(
    params=[
        None,
        "gzip",
        "bz2",
        "zip",
        "xz",
        "tar",
        pytest.param("zstd", marks=td.skip_if_no("zstandard")),
    ]
)
def compression(request):
    # [FUNCTION DOC]
    # Parametrized fixture over no compression and the gzip, bz2, zip, xz, tar and
    # zstd formats (the zstd case is skipped if the zstandard package is missing).
    # Returns the current parameter.
    """
    Fixture for trying common compression types in compression tests.
    """
    return request.param


@pytest.fixture(
    params=[
        "gzip",
        "bz2",
        "zip",
        "xz",
        "tar",
        pytest.param("zstd", marks=td.skip_if_no("zstandard")),
    ]
)
def compression_only(request):
    # [FUNCTION DOC]
    # Like the compression fixture but without the uncompressed case: gzip, bz2, zip,
    # xz, tar and zstd (skipped if zstandard is missing). Returns the current
    # parameter.
    """
    Fixture for trying common compression types in compression tests excluding
    uncompressed case.
    """
    return request.param


@pytest.fixture(params=[True, False])
def writable(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false describing whether an array should be
    # writable. Returns the current parameter.
    """
    Fixture that an array is writable.
    """
    return request.param


@pytest.fixture(params=["inner", "outer", "left", "right"])
def join_type(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the join kinds inner, outer, left and right. Returns
    # the current parameter.
    """
    Fixture for trying all types of join operations.
    """
    return request.param


@pytest.fixture(params=["nlargest", "nsmallest"])
def nselect_method(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the method names nlargest and nsmallest. Returns the
    # current parameter.
    """
    Fixture for trying all nselect methods.
    """
    return request.param


@pytest.fixture(params=[None, "ignore"])
def na_action(request):
    # [FUNCTION DOC]
    # Parametrized fixture over none and the string ignore for the na_action argument
    # of map. Returns the current parameter.
    """
    Fixture for 'na_action' argument in map.
    """
    return request.param


@pytest.fixture(params=[True, False])
def ascending(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false for an ascending argument. Returns the
    # current parameter.
    # KNOWN ISSUE: the docstring below says it is for the na_action argument of the
    # sorting and ranking methods, but the fixture is a boolean ascending flag.
    """
    Fixture for 'na_action' argument in sort_values/sort_index/rank.
    """
    return request.param


@pytest.fixture(params=["average", "min", "max", "first", "dense"])
def rank_method(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the rank tie-breaking methods average, min, max,
    # first and dense. Returns the current parameter.
    """
    Fixture for 'rank' argument in rank.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_index(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false for the groupby as_index argument.
    # Returns the current parameter.
    """
    Fixture for 'as_index' argument in groupby.
    """
    return request.param


@pytest.fixture(params=[True, False])
def cache(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false for the cache argument of datetime
    # conversion. Returns the current parameter.
    """
    Fixture for 'cache' argument in to_datetime.
    """
    return request.param


@pytest.fixture(params=[True, False])
def parallel(request):
    # [FUNCTION DOC]
    # Parametrized fixture over true and false for the parallel option of numba
    # compilation. Returns the current parameter.
    """
    Fixture for parallel keyword argument for numba.jit.
    """
    return request.param


# Can parameterize nogil & nopython over True | False, but limiting per
# https://github.com/pandas-dev/pandas/pull/41971#issuecomment-860607472


@pytest.fixture(params=[False])
def nogil(request):
    # [FUNCTION DOC]
    # Fixture with the single parameter false for the nogil option of numba
    # compilation (deliberately limited to reduce the test matrix). Returns the
    # parameter.
    """
    Fixture for nogil keyword argument for numba.jit.
    """
    return request.param


@pytest.fixture(params=[True])
def nopython(request):
    # [FUNCTION DOC]
    # Fixture with the single parameter true for the nopython option of numba
    # compilation (deliberately limited to reduce the test matrix). Returns the
    # parameter.
    """
    Fixture for nopython keyword argument for numba.jit.
    """
    return request.param


# ----------------------------------------------------------------
# Missing values & co.
# ----------------------------------------------------------------
@pytest.fixture(params=tm.NULL_OBJECTS, ids=lambda x: type(x).__name__)
def nulls_fixture(request):
    # [FUNCTION DOC]
    # Parametrized fixture over every null-like object in the testing utilities'
    # list (ids are the type names). Returns the current parameter. A module-level
    # alias with a trailing 2 allows cartesian products of two such fixtures.
    """
    Fixture for each null type in pandas.
    """
    return request.param


nulls_fixture2 = nulls_fixture  # Generate cartesian product of nulls_fixture


@pytest.fixture(params=[None, np.nan, pd.NaT])
def unique_nulls_fixture(request):
    # [FUNCTION DOC]
    # Parametrized fixture over none, NaN and NaT: each pandas null kind exactly
    # once. Returns the current parameter. A module-level alias with a trailing 2
    # allows cartesian products.
    """
    Fixture for each null type in pandas, each null type exactly once.
    """
    return request.param


# Generate cartesian product of unique_nulls_fixture:
unique_nulls_fixture2 = unique_nulls_fixture


@pytest.fixture(params=tm.NP_NAT_OBJECTS, ids=lambda x: type(x).__name__)
def np_nat_fixture(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the numpy not-a-time objects (datetime64 and
    # timedelta64 flavors; ids are the type names). Returns the current parameter.
    # A module-level alias with a trailing 2 allows cartesian products.
    """
    Fixture for each NaT type in numpy.
    """
    return request.param


# Generate cartesian product of np_nat_fixture:
np_nat_fixture2 = np_nat_fixture


# ----------------------------------------------------------------
# Classes
# ----------------------------------------------------------------


@pytest.fixture(params=[DataFrame, Series])
def frame_or_series(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the DataFrame and Series classes. Returns the current
    # class.
    """
    Fixture to parametrize over DataFrame and Series.
    """
    return request.param


@pytest.fixture(params=[Index, Series], ids=["index", "series"])
def index_or_series(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the Index and Series classes (ids index and
    # series). Returns the current class.
    """
    Fixture to parametrize over Index and Series, made necessary by a mypy
    bug, giving an error:

    List item 0 has incompatible type "Type[Series]"; expected "Type[PandasObject]"

    See GH#29725
    """
    return request.param


@pytest.fixture(params=[Index, Series, pd.array], ids=["index", "series", "array"])
def index_or_series_or_array(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the Index class, the Series class and the array
    # constructor function (ids index, series, array). Returns the current callable.
    """
    Fixture to parametrize over Index, Series, and ExtensionArray
    """
    return request.param


@pytest.fixture(params=[Index, Series, DataFrame, pd.array], ids=lambda x: x.__name__)
def box_with_array(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the Index, Series and DataFrame classes and the
    # array constructor function (ids are the callable names), used to test that
    # behavior is the same across containers. Returns the current callable. A
    # module-level alias with a trailing 2 allows cartesian products.
    """
    Fixture to test behavior for Index, Series, DataFrame, and pandas Array
    classes
    """
    return request.param


box_with_array2 = box_with_array


@pytest.fixture
def dict_subclass() -> type[dict]:
    # [FUNCTION DOC]
    # Fixture returning a freshly defined trivial subclass of dict (the class, not an
    # instance), used to test that dict subclasses are handled like dicts.
    """
    Fixture for a dictionary subclass.
    """

    class TestSubDict(dict):
        # [CLASS DOC]
        # HELPER: trivial dict subclass returned by the dict subclass fixture. It
        # adds no behavior beyond its constructor, which forwards to dict.

        def __init__(self, *args, **kwargs) -> None:
            # [FUNCTION DOC]
            # HELPER: forwards all arguments to the dict constructor.
            dict.__init__(self, *args, **kwargs)

    return TestSubDict


@pytest.fixture
def non_dict_mapping_subclass() -> type[abc.Mapping]:
    # [FUNCTION DOC]
    # Fixture returning a freshly defined read-only mapping class that is not a dict
    # subclass (the class, not an instance), used to test handling of generic
    # mappings.
    # KNOWN ISSUE: the docstring calls it a non-mapping dictionary subclass, but the
    # class is a Mapping that is not a dict subclass.
    """
    Fixture for a non-mapping dictionary subclass.
    """

    class TestNonDictMapping(abc.Mapping):
        # [CLASS DOC]
        # HELPER: minimal mapping that wraps another mapping in the attribute named
        # data and delegates item access, iteration and length to it. Subclasses the
        # abstract Mapping base class and is read only.

        def __init__(self, underlying_dict) -> None:
            # [FUNCTION DOC]
            # HELPER: stores the wrapped mapping.
            self._data = underlying_dict

        def __getitem__(self, key):
            # [FUNCTION DOC]
            # HELPER: returns the wrapped mapping's item for key (key error if
            # missing).
            return self._data.__getitem__(key)

        def __iter__(self) -> Iterator:
            # [FUNCTION DOC]
            # HELPER: returns an iterator over the wrapped mapping's keys.
            return self._data.__iter__()

        def __len__(self) -> int:
            # [FUNCTION DOC]
            # HELPER: returns the number of items of the wrapped mapping.
            return self._data.__len__()

    return TestNonDictMapping


# ----------------------------------------------------------------
# Indices
# ----------------------------------------------------------------
@pytest.fixture
def multiindex_year_month_day_dataframe_random_data():
    # [FUNCTION DOC]
    # Fixture building a frame with columns A to D of seeded random normal values for
    # the first 100 business days from 2000-01-01, grouped by year, month and day
    # and summed. The resulting three-level index has its levels converted to 64-bit
    # integers and named year, month and day. Returns the grouped frame.
    """
    DataFrame with 3 level MultiIndex (year, month, day) covering
    first 100 business days from 2000-01-01 with random data
    """
    tdf = DataFrame(
        np.random.default_rng(2).standard_normal((100, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=100, freq="B"),
    )
    ymd = tdf.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day]).sum()
    # use int64 Index, to make sure things work
    ymd.index = ymd.index.set_levels([lev.astype("i8") for lev in ymd.index.levels])
    ymd.index.set_names(["year", "month", "day"], inplace=True)
    return ymd


@pytest.fixture
def lexsorted_two_level_string_multiindex() -> MultiIndex:
    # [FUNCTION DOC]
    # Fixture returning a fixed ten-entry two-level MultiIndex of strings (first
    # level foo, bar, baz, qux; second level one, two, three), named first and second.
    # The codes are in lexicographic order (sorted by code, not by label).
    """
    2-level MultiIndex, lexsorted, with string names.
    """
    return MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )


@pytest.fixture
def multiindex_dataframe_random_data(
    lexsorted_two_level_string_multiindex,
) -> DataFrame:
    # [FUNCTION DOC]
    # Fixture depending on the two-level string MultiIndex fixture. Returns a 10 by 3
    # frame of seeded random normal values using that index as rows and columns A, B,
    # C whose column index is named exp.
    """DataFrame with 2 level MultiIndex with random data"""
    index = lexsorted_two_level_string_multiindex
    return DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        index=index,
        columns=Index(["A", "B", "C"], name="exp"),
    )


def _create_multiindex():
    # [FUNCTION DOC]
    # HELPER: builds the standard small MultiIndex used in the dictionary of test
    # indices: levels foo, bar, baz, qux and one, two with explicit codes giving six
    # entries, names first and second, without integrity verification. No inputs.
    # Returns the MultiIndex.
    """
    MultiIndex used to test the general functionality of this object
    """

    # See Also: tests.multi.conftest.idx
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ["first", "second"]
    return MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )


def _create_mi_with_dt64tz_level():
    # [FUNCTION DOC]
    # HELPER: builds a MultiIndex from the product of [1, 2], [a, b] and three daily
    # US/Eastern time zone-aware dates from 2013-01-01, with level names one, two and
    # three. Used in the dictionary of test indices. Returns the MultiIndex.
    """
    MultiIndex with a level that is a tzaware DatetimeIndex.
    """
    # GH#8367 round trip with pickle
    return MultiIndex.from_product(
        [[1, 2], ["a", "b"], date_range("20130101", periods=3, tz="US/Eastern")],
        names=["one", "two", "three"],
    )


indices_dict = {
    "object": Index([f"pandas_{i}" for i in range(10)], dtype=object),
    "string": Index([f"pandas_{i}" for i in range(10)], dtype="str"),
    "datetime": date_range("2020-01-01", periods=10),
    "datetime-tz": date_range("2020-01-01", periods=10, tz="US/Pacific"),
    "period": period_range("2020-01-01", periods=10, freq="D"),
    "timedelta": timedelta_range(start="1 day", periods=10, freq="D"),
    "range": RangeIndex(10),
    "int8": Index(np.arange(10), dtype="int8"),
    "int16": Index(np.arange(10), dtype="int16"),
    "int32": Index(np.arange(10), dtype="int32"),
    "int64": Index(np.arange(10), dtype="int64"),
    "uint8": Index(np.arange(10), dtype="uint8"),
    "uint16": Index(np.arange(10), dtype="uint16"),
    "uint32": Index(np.arange(10), dtype="uint32"),
    "uint64": Index(np.arange(10), dtype="uint64"),
    "float32": Index(np.arange(10), dtype="float32"),
    "float64": Index(np.arange(10), dtype="float64"),
    "bool-object": Index([True, False] * 5, dtype=object),
    "bool-dtype": Index([True, False] * 5, dtype=bool),
    "complex64": Index(
        np.arange(10, dtype="complex64") + 1.0j * np.arange(10, dtype="complex64")
    ),
    "complex128": Index(
        np.arange(10, dtype="complex128") + 1.0j * np.arange(10, dtype="complex128")
    ),
    "categorical": CategoricalIndex(list("abcd") * 2),
    "interval": IntervalIndex.from_breaks(np.linspace(0, 100, num=11)),
    "empty": Index([]),
    "tuples": MultiIndex.from_tuples(
        zip(["foo", "bar", "baz"], [1, 2, 3], strict=True)
    ),
    "mi-with-dt64tz-level": _create_mi_with_dt64tz_level(),
    "multi": _create_multiindex(),
    "repeats": Index([0, 0, 1, 1, 2, 2]),
    "nullable_int": Index(np.arange(10), dtype="Int64"),
    "nullable_uint": Index(np.arange(10), dtype="UInt16"),
    "nullable_float": Index(np.arange(10), dtype="Float32"),
    "nullable_bool": Index(np.arange(10).astype(bool), dtype="boolean"),
    "string-python": Index(
        pd.array([f"pandas_{i}" for i in range(10)], dtype="string[python]")
    ),
}
if has_pyarrow:
    idx = Index(pd.array([f"pandas_{i}" for i in range(10)], dtype="string[pyarrow]"))
    indices_dict["string-pyarrow"] = idx


@pytest.fixture(params=indices_dict.keys())
def index(request):
    # [FUNCTION DOC]
    # Parametrized fixture over every key of the module-level dictionary of simple
    # indices (object, string, datetime, period, range, numeric, nullable, interval,
    # categorical, MultiIndex variants and more; pyarrow strings only if installed).
    # Returns a shallow copy of the selected index so tests can mutate attributes
    # such as the name.
    """
    Fixture for many "simple" kinds of indices.

    These indices are unlikely to cover corner cases, e.g.
        - no names
        - no NaTs/NaNs
        - no values near implementation bounds
        - ...
    """
    # copy to avoid mutation, e.g. setting .name
    return indices_dict[request.param].copy(deep=False)


@pytest.fixture(
    params=[
        key for key, value in indices_dict.items() if not isinstance(value, MultiIndex)
    ]
)
def index_flat(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the keys of the dictionary of simple indices whose
    # values are not MultiIndex. Returns a shallow copy of the selected index.
    """
    index fixture, but excluding MultiIndex cases.
    """
    key = request.param
    return indices_dict[key].copy(deep=False)


@pytest.fixture(
    params=[
        key
        for key, value in indices_dict.items()
        if not (
            key.startswith(("int", "uint", "float"))
            or key in ["range", "empty", "repeats", "bool-dtype"]
        )
        and not isinstance(value, MultiIndex)
    ]
)
def index_with_missing(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the simple indices that can hold missing values
    # (excludes integer, unsigned, float, range, empty, repeats, plain boolean and
    # every MultiIndex entry).
    # Returns a new index equal to the original with the first and last element
    # replaced by a missing value. The original shared index is not modified (the
    # values are copied).
    # KNOWN ISSUE: the parameter filter removes every MultiIndex entry, so the branch
    # below that handles the tuples, multi and datetime-zone-level MultiIndex keys
    # can never run; it is dead code.
    """
    Fixture for indices with missing values.

    Integer-dtype and empty cases are excluded because they cannot hold missing
    values.

    MultiIndex is excluded because isna() is not defined for MultiIndex.
    """
    ind = indices_dict[request.param]
    if request.param in ["tuples", "mi-with-dt64tz-level", "multi"]:
        # For setting missing values in the top level of MultiIndex
        vals = ind.tolist()
        vals[0] = (None, *vals[0][1:])
        vals[-1] = (None, *vals[-1][1:])
        return MultiIndex.from_tuples(vals)
    else:
        vals = ind.values.copy()
        vals[0] = None
        vals[-1] = None
        return type(ind)(vals, copy=False)


# ----------------------------------------------------------------
# Series'
# ----------------------------------------------------------------
@pytest.fixture
def string_series() -> Series:
    # [FUNCTION DOC]
    # Fixture returning a 30-element float Series (multiples of 1.1) with a unique
    # string index i_0 to i_29, named series.
    """
    Fixture for Series of floats with Index of unique strings
    """
    return Series(
        np.arange(30, dtype=np.float64) * 1.1,
        index=Index([f"i_{i}" for i in range(30)]),
        name="series",
    )


@pytest.fixture
def object_series() -> Series:
    # [FUNCTION DOC]
    # Fixture returning a 30-element object-dtype Series of strings foo_0 to foo_29
    # with unique string index bar_0 to bar_29, named objects.
    """
    Fixture for Series of dtype object with Index of unique strings
    """
    data = [f"foo_{i}" for i in range(30)]
    index = Index([f"bar_{i}" for i in range(30)])
    return Series(data, index=index, name="objects", dtype=object)


@pytest.fixture
def datetime_series() -> Series:
    # [FUNCTION DOC]
    # Fixture returning a 30-element Series of seeded random normal floats indexed
    # by business days from 2000-01-01, named ts.
    """
    Fixture for Series of floats with DatetimeIndex
    """
    return Series(
        np.random.default_rng(2).standard_normal(30),
        index=date_range("2000-01-01", periods=30, freq="B"),
        name="ts",
    )


def _create_series(index):
    # [FUNCTION DOC]
    # HELPER: builds a Series for the module-level Series dictionary and a fixture.
    # Input: index (any index). Returns a Series named a with seeded random normal
    # values, one per index entry, without copying the data.
    """Helper for the _series dict"""
    size = len(index)
    data = np.random.default_rng(2).standard_normal(size)
    return Series(data, index=index, name="a", copy=False)


_series = {
    f"series-with-{index_id}-index": _create_series(index)
    for index_id, index in indices_dict.items()
}


@pytest.fixture
def series_with_simple_index(index) -> Series:
    # [FUNCTION DOC]
    # Fixture depending on the simple index fixture (so it is parametrized over all
    # simple indices). Returns a random Series built on that index with the helper.
    """
    Fixture for tests on series with changing types of indices.
    """
    return _create_series(index)


_narrow_series = {
    f"{dtype.__name__}-series": Series(
        range(30), index=[f"i-{i}" for i in range(30)], name="a", dtype=dtype
    )
    for dtype in tm.NARROW_NP_DTYPES
}


_index_or_series_objs = {**indices_dict, **_series, **_narrow_series}


@pytest.fixture(params=_index_or_series_objs.keys())
def index_or_series_obj(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the combined dictionary of simple indices, Series
    # with each simple index, and Series of narrow numpy dtypes. Returns a shallow
    # copy of the selected object so tests may mutate attributes such as the name.
    """
    Fixture for tests on indexes, series and series with a narrow dtype
    copy to avoid mutation, e.g. setting .name
    """
    return _index_or_series_objs[request.param].copy(deep=False)


_typ_objects_series = {
    f"{dtype.__name__}-series": Series(dtype) for dtype in tm.PYTHON_DATA_TYPES
}


_index_or_series_memory_objs = {
    **indices_dict,
    **_series,
    **_narrow_series,
    **_typ_objects_series,
}


@pytest.fixture(params=_index_or_series_memory_objs.keys())
def index_or_series_memory_obj(request):
    # [FUNCTION DOC]
    # Parametrized fixture like the index-or-series fixture but extended with empty
    # Series of each Python builtin data type. Returns a shallow copy of the
    # selected object.
    """
    Fixture for tests on indexes, series, series with a narrow dtype and
    series with empty objects type
    copy to avoid mutation, e.g. setting .name
    """
    return _index_or_series_memory_objs[request.param].copy(deep=False)


# ----------------------------------------------------------------
# DataFrames
# ----------------------------------------------------------------
@pytest.fixture
def int_frame() -> DataFrame:
    # [FUNCTION DOC]
    # Fixture returning a 30 by 4 frame of 64-bit integer ones with unique string row
    # labels foo_0 to foo_29 and columns A to D.
    """
    Fixture for DataFrame of ints with index of unique strings

    Columns are ['A', 'B', 'C', 'D']
    """
    return DataFrame(
        np.ones((30, 4), dtype=np.int64),
        index=Index([f"foo_{i}" for i in range(30)]),
        columns=Index(list("ABCD")),
    )


@pytest.fixture
def float_frame() -> DataFrame:
    # [FUNCTION DOC]
    # Fixture returning a 30 by 4 frame of seeded random normal floats with unique
    # string row labels foo_0 to foo_29 and columns A to D.
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    return DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)]),
        columns=Index(list("ABCD")),
    )


@pytest.fixture
def rand_series_with_duplicate_datetimeindex() -> Series:
    # [FUNCTION DOC]
    # Fixture returning a 10-element Series of seeded random normal values indexed by
    # a fixed list of dates in which 2000-01-02, 2000-01-03 and 2000-01-04 each
    # appear three times and 2000-01-05 once.
    """
    Fixture for Series with a DatetimeIndex that has duplicates.
    """
    dates = [
        datetime(2000, 1, 2),
        datetime(2000, 1, 2),
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
        datetime(2000, 1, 3),
        datetime(2000, 1, 3),
        datetime(2000, 1, 4),
        datetime(2000, 1, 4),
        datetime(2000, 1, 4),
        datetime(2000, 1, 5),
    ]

    return Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)


# ----------------------------------------------------------------
# Scalars
# ----------------------------------------------------------------
@pytest.fixture(
    params=[
        (Interval(left=0, right=5), IntervalDtype("int64", "right")),
        (Interval(left=0.1, right=0.5), IntervalDtype("float64", "right")),
        (Period("2012-01", freq="M"), "period[M]"),
        (Period("2012-02-01", freq="D"), "period[D]"),
        (
            Timestamp("2011-01-01", tz="US/Eastern").as_unit("s"),
            DatetimeTZDtype(unit="s", tz="US/Eastern"),
        ),
        (Timedelta(seconds=500), "timedelta64[us]"),
    ]
)
def ea_scalar_and_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over pairs of an extension-type scalar and its dtype (or
    # dtype string): an integer interval, a float interval, monthly and daily
    # periods, a time zone-aware timestamp in seconds resolution, and a timedelta
    # with microsecond dtype. Returns the current (scalar, dtype) pair.
    """
    Fixture that tests each scalar and datetime type.
    """
    return request.param


# ----------------------------------------------------------------
# Operators & Operations
# ----------------------------------------------------------------


@pytest.fixture(params=tm.arithmetic_dunder_methods)
def all_arithmetic_operators(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the dunder method names for common arithmetic
    # operations, taken from the testing utilities' list. Returns the current name.
    """
    Fixture for dunder names for common arithmetic operations.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        ops.radd,
        operator.sub,
        ops.rsub,
        operator.mul,
        ops.rmul,
        operator.truediv,
        ops.rtruediv,
        operator.floordiv,
        ops.rfloordiv,
        operator.mod,
        ops.rmod,
        operator.pow,
        ops.rpow,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.and_,
        ops.rand_,
        operator.xor,
        ops.rxor,
        operator.or_,
        ops.ror_,
    ]
)
def all_binary_operators(request):
    # [FUNCTION DOC]
    # Parametrized fixture over operator functions: arithmetic (add, subtract,
    # multiply, true and floor division, modulo, power), their reflected versions,
    # the six comparisons, and logical and, exclusive or, or with the reflected
    # logical forms. Returns the current function.
    """
    Fixture for operator and roperator arithmetic, comparison, and logical ops.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        ops.radd,
        operator.sub,
        ops.rsub,
        operator.mul,
        ops.rmul,
        operator.truediv,
        ops.rtruediv,
        operator.floordiv,
        ops.rfloordiv,
        operator.mod,
        ops.rmod,
        operator.pow,
        ops.rpow,
    ]
)
def all_arithmetic_functions(request):
    # [FUNCTION DOC]
    # Parametrized fixture over arithmetic operator functions (add, subtract,
    # multiply, true and floor division, modulo, power) and their reflected
    # versions. Returns the current function.
    # KNOWN ISSUE: the note below says divmod and rdivmod are included, but they are
    # not in the parameter list.
    """
    Fixture for operator and roperator arithmetic functions.

    Notes
    -----
    This includes divmod and rdivmod, whereas all_arithmetic_operators
    does not.
    """
    return request.param


_all_numeric_reductions = [
    "count",
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
    "sem",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    # [FUNCTION DOC]
    # Parametrized fixture over numeric reduction method names (count, sum, max, min,
    # mean, prod, std, var, median, kurt, skew, sem). Returns the current name.
    """
    Fixture for numeric reduction names.
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the boolean reduction names all and any. Returns the
    # current name.
    """
    Fixture for boolean reduction names.
    """
    return request.param


_all_reductions = _all_numeric_reductions + _all_boolean_reductions


@pytest.fixture(params=_all_reductions)
def all_reductions(request):
    # [FUNCTION DOC]
    # Parametrized fixture over all numeric and boolean reduction names. Returns the
    # current name.
    """
    Fixture for all (boolean + numeric) reduction names.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    ]
)
def comparison_op(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the six operator-module comparison functions (equal,
    # not equal, greater, greater or equal, less, less or equal). Returns the
    # current function.
    """
    Fixture for operator module comparison functions.
    """
    return request.param


@pytest.fixture(params=["__le__", "__lt__", "__ge__", "__gt__"])
def compare_operators_no_eq_ne(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the dunder names of the ordering comparisons (less
    # or equal, less, greater or equal, greater), excluding equality and
    # inequality. Returns the current name.
    """
    Fixture for dunder names for compare operations except == and !=

    * >=
    * >
    * <
    * <=
    """
    return request.param


@pytest.fixture(
    params=["__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"]
)
def all_logical_operators(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the dunder names of the logical operations and,
    # reflected and, or, reflected or, exclusive or, reflected exclusive or. Returns
    # the current name.
    """
    Fixture for dunder names for common logical operations

    * |
    * &
    * ^
    """
    return request.param


_all_numeric_accumulations = ["cumsum", "cumprod", "cummin", "cummax"]


@pytest.fixture(params=_all_numeric_accumulations)
def all_numeric_accumulations(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the accumulation method names cumsum, cumprod,
    # cummin and cummax. Returns the current name.
    """
    Fixture for numeric accumulation names
    """
    return request.param


# ----------------------------------------------------------------
# Data sets/files
# ----------------------------------------------------------------
@pytest.fixture
def strict_data_files(pytestconfig):
    # [FUNCTION DOC]
    # Fixture reading the command line option for non-strict data files from the
    # pytest configuration. Returns the option value: true by default, false when the
    # no-strict-data-files flag was given (the flag stores false).
    """
    Returns the configuration for the test setting `--no-strict-data-files`.
    """
    return pytestconfig.getoption("--no-strict-data-files")


@pytest.fixture
def datapath(strict_data_files: str) -> Callable[..., str]:
    # [FUNCTION DOC]
    # Fixture returning a function that resolves test data file paths. The returned
    # function takes path components, joins them under the tests directory next to
    # this file and returns the full path. If the path does not exist: when strict
    # (the default) a value error is raised mentioning the option; otherwise the
    # current test is skipped.
    # KNOWN ISSUE: the parameter annotation says the strictness flag is a string,
    # but it is the boolean option value; the docstring documents a parameter named
    # path, which belongs to the returned function rather than the fixture.
    """
    Get the path to a data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``pandas/tests/``

    Returns
    -------
    path including ``pandas/tests``.

    Raises
    ------
    ValueError
        If the path doesn't exist and the --no-strict-data-files option is not set.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "tests")

    def deco(*args):
        # [FUNCTION DOC]
        # HELPER: the path resolver returned by the data path fixture. Inputs: path
        # components relative to the tests directory. Returns the joined path if it
        # exists; raises a value error if missing and strict; skips the test if
        # missing and not strict.
        path = os.path.join(BASE_PATH, *args)
        if not os.path.exists(path):
            if strict_data_files:
                raise ValueError(
                    f"Could not find file {path} and --no-strict-data-files is not set."
                )
            pytest.skip(f"Could not find {path}.")
        return path

    return deco


# ----------------------------------------------------------------
# Time zones
# ----------------------------------------------------------------
TIMEZONES = [
    None,
    "UTC",
    "US/Eastern",
    "Asia/Tokyo",
    "dateutil/US/Pacific",
    "dateutil/Asia/Singapore",
    "+01:15",
    "-02:15",
    "UTC+01:15",
    "UTC-02:15",
    tzutc(),
    tzlocal(),
    timezone.utc,
    timezone(timedelta(hours=1)),
    timezone(timedelta(hours=-1), name="foo"),
]
if pytz is not None:
    TIMEZONES.extend(
        (
            pytz.FixedOffset(300),
            pytz.FixedOffset(0),
            pytz.FixedOffset(-300),
            pytz.timezone("US/Pacific"),
            pytz.timezone("UTC"),
        )
    )
TIMEZONE_IDS = [repr(i) for i in TIMEZONES]


@td.parametrize_fixture_doc(str(TIMEZONE_IDS))
@pytest.fixture(params=TIMEZONES, ids=TIMEZONE_IDS)
def tz_naive_fixture(request):
    # [FUNCTION DOC]
    # Parametrized fixture over time zone specifications including none (naive):
    # strings in several naming styles, fixed-offset strings, dateutil and standard
    # library tzinfo objects, and, if pytz is installed, pytz zones. The docstring is
    # a template filled with the list of ids by a decorator. Returns the current
    # time zone value.
    """
    Fixture for trying timezones including default (None): {0}
    """
    return request.param


@td.parametrize_fixture_doc(str(TIMEZONE_IDS[1:]))
@pytest.fixture(params=TIMEZONES[1:], ids=TIMEZONE_IDS[1:])
def tz_aware_fixture(request):
    # [FUNCTION DOC]
    # Same as the naive time zone fixture without the none entry, so every parameter
    # is an explicit time zone. Returns the current time zone value.
    """
    Fixture for trying explicit timezones: {0}
    """
    return request.param


_UTCS = ["utc", "dateutil/UTC", tzutc(), timezone.utc]

if pytz is not None:
    _UTCS.append(pytz.utc)


@pytest.fixture(params=_UTCS)
def utc_fixture(request):
    # [FUNCTION DOC]
    # Parametrized fixture over spellings of UTC: strings utc and dateutil/UTC, the
    # dateutil and standard library UTC objects and, if installed, pytz UTC. Returns
    # the current value. A module-level alias with a trailing 2 allows cartesian
    # products.
    """
    Fixture to provide variants of UTC timezone strings and tzinfo objects.
    """
    return request.param


utc_fixture2 = utc_fixture


@pytest.fixture(params=["s", "ms", "us", "ns"])
def unit(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the supported datetime64 resolutions: s, ms, us, ns.
    # Returns the current unit. A module-level alias with a trailing 2 allows
    # cartesian products.
    """
    datetime64 units we support.
    """
    return request.param


unit2 = unit


# ----------------------------------------------------------------
# Dtypes
# ----------------------------------------------------------------
@pytest.fixture(params=tm.STRING_DTYPES)
def string_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the string dtype specifications listed in the testing
    # utilities (the str class, the str name and the U code). Returns the current
    # value.
    """
    Parametrized fixture for string dtypes.

    * str
    * 'str'
    * 'U'
    """
    return request.param


@pytest.fixture(
    params=[
        ("python", pd.NA),
        pytest.param(("pyarrow", pd.NA), marks=td.skip_if_no("pyarrow")),
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
        ("python", np.nan),
    ],
    ids=[
        "string=string[python]",
        "string=string[pyarrow]",
        "string=str[pyarrow]",
        "string=str[python]",
    ],
)
def string_dtype_no_object(request):
    # [FUNCTION DOC]
    # Parametrized fixture over four StringDtype configurations: python storage with
    # the NA marker, pyarrow storage with the NA marker, pyarrow storage with NaN,
    # and python storage with NaN (pyarrow cases skipped if pyarrow is missing). The
    # dtype is constructed in the fixture body, not in the parameters, to avoid
    # importing pyarrow at collection time. Returns the StringDtype instance.
    """
    Parametrized fixture for string dtypes.
    * 'string[python]' (NA variant)
    * 'string[pyarrow]' (NA variant)
    * 'str' (NaN variant, with pyarrow)
    * 'str' (NaN variant, without pyarrow)
    """
    # need to instantiate the StringDtype here instead of in the params
    # to avoid importing pyarrow during test collection
    storage, na_value = request.param
    return pd.StringDtype(storage, na_value)


@pytest.fixture(
    params=[
        "string[python]",
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
    ]
)
def nullable_string_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the dtype strings string[python] and string[pyarrow]
    # (the latter skipped without pyarrow). Returns the current string.
    """
    Parametrized fixture for string dtypes.

    * 'string[python]'
    * 'string[pyarrow]'
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
        pytest.param(("pyarrow", pd.NA), marks=td.skip_if_no("pyarrow")),
    ]
)
def pyarrow_string_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over pyarrow-backed StringDtype with NaN and with the NA
    # marker (both skipped without pyarrow). Returns a StringDtype built from the
    # storage and missing-value marker pair.
    """
    Parametrized fixture for string dtypes backed by Pyarrow.

    * 'str[pyarrow]'
    * 'string[pyarrow]'
    """
    return pd.StringDtype(*request.param)


@pytest.fixture(
    params=[
        "python",
        pytest.param("pyarrow", marks=td.skip_if_no("pyarrow")),
    ]
)
def string_storage(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the string storage option values python and pyarrow
    # (pyarrow skipped if missing). Returns the current value. A module-level alias
    # with a trailing 2 allows cartesian products.
    """
    Parametrized fixture for pd.options.mode.string_storage.

    * 'python'
    * 'pyarrow'
    """
    return request.param


@pytest.fixture(
    params=[
        ("python", pd.NA),
        pytest.param(("pyarrow", pd.NA), marks=td.skip_if_no("pyarrow")),
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
        ("python", np.nan),
    ],
    ids=[
        "string=string[python]",
        "string=string[pyarrow]",
        "string=str[pyarrow]",
        "string=str[python]",
    ],
)
def string_dtype_arguments(request):
    # [FUNCTION DOC]
    # Parametrized fixture over (storage, missing-value marker) pairs for building a
    # StringDtype: python with NA, pyarrow with NA, pyarrow with NaN and python with
    # NaN (pyarrow cases skipped if missing). Returns the pair unchanged (not yet a
    # dtype). A module-level alias with a trailing 2 allows cartesian products.
    """
    Parametrized fixture for StringDtype storage and na_value.

    * 'python' + pd.NA
    * 'pyarrow' + pd.NA
    * 'pyarrow' + np.nan
    """
    return request.param


@pytest.fixture(
    params=[
        "numpy_nullable",
        pytest.param("pyarrow", marks=td.skip_if_no("pyarrow")),
    ]
)
def dtype_backend(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the dtype backend names numpy_nullable and pyarrow
    # (pyarrow skipped if missing). Returns the current name.
    # KNOWN ISSUE: the docstring below says this is for the string storage option
    # and lists python and pyarrow, but the parameters are dtype backend names.
    """
    Parametrized fixture for pd.options.mode.string_storage.

    * 'python'
    * 'pyarrow'
    """
    return request.param


# Alias so we can test with cartesian product of string_storage
string_storage2 = string_storage
string_dtype_arguments2 = string_dtype_arguments


@pytest.fixture(params=tm.BYTES_DTYPES)
def bytes_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the bytes dtype specifications in the testing
    # utilities (the bytes class and its name). Returns the current value.
    """
    Parametrized fixture for bytes dtypes.

    * bytes
    * 'bytes'
    """
    return request.param


@pytest.fixture(params=tm.OBJECT_DTYPES)
def object_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the object dtype specifications in the testing
    # utilities (the object class and its name). Returns the current value.
    """
    Parametrized fixture for object dtypes.

    * object
    * 'object'
    """
    return request.param


@pytest.fixture(
    params=[
        np.dtype("object"),
        ("python", pd.NA),
        pytest.param(("pyarrow", pd.NA), marks=td.skip_if_no("pyarrow")),
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
        ("python", np.nan),
    ],
    ids=[
        "string=object",
        "string=string[python]",
        "string=string[pyarrow]",
        "string=str[pyarrow]",
        "string=str[python]",
    ],
)
def any_string_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over five string-holding dtypes: numpy object, and the
    # four StringDtype configurations (python/NA, pyarrow/NA, pyarrow/NaN,
    # python/NaN; pyarrow cases skipped if missing). A numpy dtype parameter is
    # returned as is; the other parameters are (storage, marker) pairs converted to
    # a StringDtype in the body, to avoid importing pyarrow at collection time. A
    # module-level alias with a trailing 2 allows cartesian products.
    """
    Parametrized fixture for string dtypes.
    * 'object'
    * 'string[python]' (NA variant)
    * 'string[pyarrow]' (NA variant)
    * 'str' (NaN variant, with pyarrow)
    * 'str' (NaN variant, without pyarrow)
    """
    if isinstance(request.param, np.dtype):
        return request.param
    else:
        # need to instantiate the StringDtype here instead of in the params
        # to avoid importing pyarrow during test collection
        storage, na_value = request.param
        return pd.StringDtype(storage, na_value)


any_string_dtype2 = any_string_dtype


@pytest.fixture(params=tm.DATETIME64_DTYPES)
def datetime64_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the datetime64 dtype specifications in the testing
    # utilities (nanosecond-resolution names and codes). Returns the current value.
    """
    Parametrized fixture for datetime64 dtypes.

    * 'datetime64[ns]'
    * 'M8[ns]'
    """
    return request.param


@pytest.fixture(params=tm.TIMEDELTA64_DTYPES)
def timedelta64_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the timedelta64 dtype specifications in the testing
    # utilities (nanosecond-resolution names and codes). Returns the current value.
    """
    Parametrized fixture for timedelta64 dtypes.

    * 'timedelta64[ns]'
    * 'm8[ns]'
    """
    return request.param


@pytest.fixture
def fixed_now_ts() -> Timestamp:
    # [FUNCTION DOC]
    # Fixture returning a constant Timestamp (2021-01-01 12:04:13.000022), usable in
    # place of the current time so tests are deterministic.
    """
    Fixture emits fixed Timestamp.now()
    """
    return Timestamp(  # pyright: ignore[reportReturnType]
        year=2021, month=1, day=1, hour=12, minute=4, second=13, microsecond=22
    )


@pytest.fixture(params=tm.FLOAT_NUMPY_DTYPES)
def float_numpy_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over numpy float dtype specifications from the testing
    # utilities (the float class and the 32 and 64-bit names). Returns the current
    # value.
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    """
    return request.param


@pytest.fixture(params=tm.FLOAT_EA_DTYPES)
def float_ea_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the nullable float extension dtype names Float32 and
    # Float64. Returns the current value.
    """
    Parameterized fixture for float dtypes.

    * 'Float32'
    * 'Float64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_FLOAT_DTYPES)
def any_float_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over all float dtype specifications (numpy and nullable
    # extension). Returns the current value.
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


@pytest.fixture(params=tm.COMPLEX_DTYPES)
def complex_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over complex dtype specifications (the complex class and
    # the 64 and 128-bit names). Returns the current value.
    """
    Parameterized fixture for complex dtypes.

    * complex
    * 'complex64'
    * 'complex128'
    """
    return request.param


@pytest.fixture(params=tm.COMPLEX_FLOAT_DTYPES)
def complex_or_float_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over complex dtype specifications plus numpy float dtype
    # specifications. Returns the current value.
    """
    Parameterized fixture for complex and numpy float dtypes.

    * complex
    * 'complex64'
    * 'complex128'
    * float
    * 'float32'
    * 'float64'
    """
    return request.param


@pytest.fixture(params=tm.SIGNED_INT_NUMPY_DTYPES)
def any_signed_int_numpy_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over signed integer numpy dtype specifications (the int
    # class and the 8, 16, 32 and 64-bit names). Returns the current value.
    """
    Parameterized fixture for signed integer dtypes.

    * int
    * 'int8'
    * 'int16'
    * 'int32'
    * 'int64'
    """
    return request.param


@pytest.fixture(params=tm.UNSIGNED_INT_NUMPY_DTYPES)
def any_unsigned_int_numpy_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over unsigned integer numpy dtype names (8, 16, 32 and
    # 64-bit). Returns the current value.
    """
    Parameterized fixture for unsigned integer dtypes.

    * 'uint8'
    * 'uint16'
    * 'uint32'
    * 'uint64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_NUMPY_DTYPES)
def any_int_numpy_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over all signed and unsigned integer numpy dtype
    # specifications. Returns the current value.
    """
    Parameterized fixture for any integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_EA_DTYPES)
def any_int_ea_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the nullable integer extension dtype names (signed
    # and unsigned, 8 to 64 bits). Returns the current value.
    """
    Parameterized fixture for any nullable integer dtype.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_DTYPES)
def any_int_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over all integer dtype specifications: numpy signed and
    # unsigned plus nullable extension dtypes. Returns the current value.
    # KNOWN ISSUE: the docstring below describes it as any nullable integer dtype,
    # but the parameters also include the plain numpy integer dtypes.
    """
    Parameterized fixture for any nullable integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_EA_DTYPES + tm.FLOAT_EA_DTYPES)
def any_numeric_ea_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over nullable integer extension dtype names followed by the
    # nullable float extension dtype names. Returns the current value.
    """
    Parameterized fixture for any nullable integer dtype and
    any float ea dtypes.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


#  Unsupported operand types for + ("List[Union[str, ExtensionDtype, dtype[Any],
#  Type[object]]]" and "List[str]")
@pytest.fixture(
    params=tm.ALL_INT_EA_DTYPES
    + tm.FLOAT_EA_DTYPES
    + tm.ALL_INT_PYARROW_DTYPES_STR_REPR
    + tm.FLOAT_PYARROW_DTYPES_STR_REPR  # type: ignore[operator]
)
def any_numeric_ea_and_arrow_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over nullable integer and float extension dtype names plus
    # the Arrow-backed integer and float dtype strings. Returns the current value.
    """
    Parameterized fixture for any nullable integer dtype and
    any float ea dtypes.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    * 'uint8[pyarrow]'
    * 'int8[pyarrow]'
    * 'uint16[pyarrow]'
    * 'int16[pyarrow]'
    * 'uint32[pyarrow]'
    * 'int32[pyarrow]'
    * 'uint64[pyarrow]'
    * 'int64[pyarrow]'
    * 'float32[pyarrow]'
    * 'float64[pyarrow]'
    """
    return request.param


@pytest.fixture(params=tm.SIGNED_INT_EA_DTYPES)
def any_signed_int_ea_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the signed nullable integer extension dtype names
    # Int8, Int16, Int32, Int64. Returns the current value.
    """
    Parameterized fixture for any signed nullable integer dtype.

    * 'Int8'
    * 'Int16'
    * 'Int32'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_NUMPY_DTYPES)
def any_real_numpy_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over numpy dtype specifications of real (non-complex)
    # numbers: integers and floats. Returns the current value.
    """
    Parameterized fixture for any (purely) real numeric dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_DTYPES)
def any_real_numeric_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over real numeric dtype specifications: numpy integers and
    # floats plus the matching nullable extension dtypes. Returns the current value.
    """
    Parameterized fixture for any (purely) real numeric dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'

    and associated ea dtypes.
    """
    return request.param


@pytest.fixture(params=tm.ALL_NUMPY_DTYPES)
def any_numpy_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over numpy dtype specifications of every kind: boolean,
    # integer, float, complex, string, bytes, datetime64, timedelta64 and object.
    # Returns the current value.
    """
    Parameterized fixture for all numpy dtypes.

    * bool
    * 'bool'
    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    * complex
    * 'complex64'
    * 'complex128'
    * str
    * 'str'
    * 'U'
    * bytes
    * 'bytes'
    * 'datetime64[ns]'
    * 'M8[ns]'
    * 'timedelta64[ns]'
    * 'm8[ns]'
    * object
    * 'object'
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_NULLABLE_DTYPES)
def any_real_nullable_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over real-number dtype specifications that can hold
    # missing values: numpy floats, nullable float and integer extension dtypes and
    # Arrow-backed integer and float dtypes. Returns the current value.
    """
    Parameterized fixture for all real dtypes that can hold NA.

    * float
    * 'float32'
    * 'float64'
    * 'Float32'
    * 'Float64'
    * 'UInt8'
    * 'UInt16'
    * 'UInt32'
    * 'UInt64'
    * 'Int8'
    * 'Int16'
    * 'Int32'
    * 'Int64'
    * 'uint8[pyarrow]'
    * 'uint16[pyarrow]'
    * 'uint32[pyarrow]'
    * 'uint64[pyarrow]'
    * 'int8[pyarrow]'
    * 'int16[pyarrow]'
    * 'int32[pyarrow]'
    * 'int64[pyarrow]'
    * 'float[pyarrow]'
    * 'double[pyarrow]'
    """
    return request.param


@pytest.fixture(params=tm.ALL_NUMERIC_DTYPES)
def any_numeric_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture over numeric dtype specifications: numpy integers, floats
    # and complex numbers plus nullable integer and float extension dtypes. Returns
    # the current value.
    """
    Parameterized fixture for all numeric dtypes.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    * complex
    * 'complex64'
    * 'complex128'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


# categoricals are handled separately
_any_skipna_inferred_dtype = [
    ("string", ["a", np.nan, "c"]),
    ("string", ["a", pd.NA, "c"]),
    ("mixed", ["a", pd.NaT, "c"]),  # pd.NaT not considered valid by is_string_array
    ("bytes", [b"a", np.nan, b"c"]),
    ("empty", [np.nan, np.nan, np.nan]),
    ("empty", []),
    ("mixed-integer", ["a", np.nan, 2]),
    ("mixed", ["a", np.nan, 2.0]),
    ("floating", [1.0, np.nan, 2.0]),
    ("integer", [1, np.nan, 2]),
    ("mixed-integer-float", [1, np.nan, 2.0]),
    ("decimal", [Decimal(1), np.nan, Decimal(2)]),
    ("boolean", [True, np.nan, False]),
    ("boolean", [True, pd.NA, False]),
    ("datetime64", [np.datetime64("2013-01-01"), np.nan, np.datetime64("2018-01-01")]),
    ("datetime", [Timestamp("20130101"), np.nan, Timestamp("20180101")]),
    ("date", [date(2013, 1, 1), np.nan, date(2018, 1, 1)]),
    ("complex", [1 + 1j, np.nan, 2 + 2j]),
    # The following dtype is commented out due to GH 23554
    # ('timedelta64', [np.timedelta64(1, 'D'),
    #                  np.nan, np.timedelta64(2, 'D')]),
    ("timedelta", [timedelta(1), np.nan, timedelta(2)]),
    ("time", [time(1), np.nan, time(2)]),
    ("period", [Period(2013), pd.NaT, Period(2018)]),
    ("interval", [Interval(0, 1), np.nan, Interval(0, 2)]),
]
ids = [
    pair[0] for pair in _any_skipna_inferred_dtype
]  # use inferred type as fixture-id


@pytest.fixture(params=_any_skipna_inferred_dtype, ids=ids)
def any_skipna_inferred_dtype(request):
    # [FUNCTION DOC]
    # Parametrized fixture (ids are the inferred type names) over pairs of an
    # expected inferred dtype name and a sample list containing missing values, from
    # the module-level table (string, mixed, bytes, empty, mixed-integer, floating,
    # integer, mixed-integer-float, decimal, boolean, datetime64, datetime, date,
    # complex, timedelta, time, period, interval). Converts the sample to an
    # object-dtype array so no casting occurs. Returns the pair (name, object
    # array). Correctness of the inference is tested elsewhere.
    # KNOWN ISSUE: the docstring list of covered inferred types omits the complex
    # entry that the table contains.
    """
    Fixture for all inferred dtypes from _libs.lib.infer_dtype

    The covered (inferred) types are:
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'
    * 'mixed-integer-float'
    * 'floating'
    * 'integer'
    * 'decimal'
    * 'boolean'
    * 'datetime64'
    * 'datetime'
    * 'date'
    * 'timedelta'
    * 'time'
    * 'period'
    * 'interval'

    Returns
    -------
    inferred_dtype : str
        The string for the inferred dtype from _libs.lib.infer_dtype
    values : np.ndarray
        An array of object dtype that will be inferred to have
        `inferred_dtype`

    Examples
    --------
    >>> from pandas._libs import lib
    >>>
    >>> def test_something(any_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_skipna_inferred_dtype
    ...     # will pass
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    """
    inferred_dtype, values = request.param
    values = np.array(values, dtype=object)  # object dtype to avoid casting

    # correctness of inference tested in tests/dtypes/test_inference.py
    return inferred_dtype, values


# ----------------------------------------------------------------
# Misc
# ----------------------------------------------------------------
@pytest.fixture
def ip():
    # [FUNCTION DOC]
    # Fixture returning an interactive shell instance. Skips the test if IPython
    # (version 6.0.0 or newer) is not installed. The shell is configured to keep its
    # history in memory (to avoid leaking a sqlite file handle). Returns the shell.
    """
    Get an instance of IPython.InteractiveShell.

    Will raise a skip if IPython is not installed.
    """
    pytest.importorskip("IPython", minversion="6.0.0")
    from IPython.core.interactiveshell import InteractiveShell

    # GH#35711 make sure sqlite history file handle is not leaked
    from traitlets.config import Config  # isort:skip

    c = Config()
    c.HistoryManager.hist_file = ":memory:"

    return InteractiveShell(config=c)


@pytest.fixture
def mpl_cleanup():
    # [FUNCTION DOC]
    # Generator fixture wrapping a test with matplotlib setup and cleanup. Skips if
    # matplotlib is missing. Setup: remembers the units registry, enters a settings
    # context and selects the non-GUI template backend. After the test (always, even
    # on failure): restores the units registry, closes all figures and runs a
    # garbage collection of the youngest generation. Yields nothing.
    """
    Ensure Matplotlib is cleaned up around a test.

    Before a test is run:

    1) Set the backend to "template" to avoid requiring a GUI.

    After a test is run:

    1) Reset units registry
    2) Reset rc_context
    3) Close all figures

    See matplotlib/testing/decorators.py#L24.
    """
    mpl = pytest.importorskip("matplotlib")
    mpl_units = pytest.importorskip("matplotlib.units")
    plt = pytest.importorskip("matplotlib.pyplot")
    orig_units_registry = mpl_units.registry.copy()
    try:
        with mpl.rc_context():
            mpl.use("template")
            yield
    finally:
        mpl_units.registry.clear()
        mpl_units.registry.update(orig_units_registry)
        plt.close("all")
        # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.6.0.html#garbage-collection-is-no-longer-run-on-figure-close  # noqa: E501
        gc.collect(1)


@pytest.fixture(
    params=[
        getattr(pd.offsets, o)
        for o in pd.offsets.__all__
        if issubclass(getattr(pd.offsets, o), pd.offsets.Tick) and o != "Tick"
    ]
)
def tick_classes(request):
    # [FUNCTION DOC]
    # Parametrized fixture over every Tick-based offset class exported by the
    # offsets namespace except the generic Tick base. Returns the current class.
    """
    Fixture for Tick based datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    # [FUNCTION DOC]
    # Parametrized fixture over no key function and the identity function, for sort
    # key tests. Returns the current value.
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


@pytest.fixture(
    params=[
        ("foo", None, None),
        ("Egon", "Venkman", None),
        ("NCC1701D", "NCC1701D", "NCC1701D"),
        # possibly-matching NAs
        (np.nan, np.nan, np.nan),
        (np.nan, pd.NaT, None),
        (np.nan, pd.NA, None),
        (pd.NA, pd.NA, pd.NA),
    ]
)
def names(request) -> tuple[Hashable, Hashable, Hashable]:
    # [FUNCTION DOC]
    # Parametrized fixture over triples of names (left operand, right operand,
    # expected result) covering equal names, unequal names, and combinations of
    # missing markers. Returns the current triple.
    """
    A 3-tuple of names, the first two for operands, the last for a result.
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.loc, tm.iloc])
def indexer_sli(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the testing helpers for plain item assignment, label
    # based and position based indexers. Returns the current helper.
    """
    Parametrize over __setitem__, loc.__setitem__, iloc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.loc, tm.iloc])
def indexer_li(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the testing helpers for label based and position
    # based indexers. Returns the current helper.
    """
    Parametrize over loc.__getitem__, iloc.__getitem__
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.iloc])
def indexer_si(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the testing helpers for plain item assignment and
    # position based indexers. Returns the current helper.
    """
    Parametrize over __setitem__, iloc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.loc])
def indexer_sl(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the testing helpers for plain item assignment and
    # label based indexers. Returns the current helper.
    """
    Parametrize over __setitem__, loc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.at, tm.loc])
def indexer_al(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the testing helpers for scalar label access and label
    # based indexers. Returns the current helper.
    """
    Parametrize over at.__setitem__, loc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.iat, tm.iloc])
def indexer_ial(request):
    # [FUNCTION DOC]
    # Parametrized fixture over the testing helpers for scalar position access and
    # position based indexers. Returns the current helper.
    """
    Parametrize over iat.__setitem__, iloc.__setitem__
    """
    return request.param


@pytest.fixture(params=[True, False])
def performance_warning(request) -> Iterator[bool | type[Warning]]:
    # [FUNCTION DOC]
    # Generator fixture parametrized over true and false. While the test runs, the
    # global performance warnings option is set to the parameter (restored
    # afterwards). Yields the PerformanceWarning class when enabled and false when
    # disabled, for use as the expected warning argument of warning assertions.
    """
    Fixture to check if performance warnings are enabled. Either produces
    ``PerformanceWarning`` if they are enabled, otherwise ``False``.
    """
    with pd.option_context("mode.performance_warnings", request.param):
        yield pd.errors.PerformanceWarning if request.param else False


@pytest.fixture
def using_infer_string() -> bool:
    # [FUNCTION DOC]
    # Fixture returning true if the future string inference option is exactly true at
    # the time the fixture runs, else false.
    """
    Fixture to check if infer string option is enabled.
    """
    return pd.options.future.infer_string is True


@pytest.fixture
def using_python_scalars() -> bool:
    # [FUNCTION DOC]
    # Fixture returning true if the future Python-scalars option is exactly true at
    # the time the fixture runs, else false.
    return pd.options.future.python_scalars is True


_warsaws: list[Any] = ["Europe/Warsaw", "dateutil/Europe/Warsaw"]
if pytz is not None:
    _warsaws.append(pytz.timezone("Europe/Warsaw"))


@pytest.fixture(params=_warsaws)
def warsaw(request) -> str:
    # [FUNCTION DOC]
    # Parametrized fixture over ways to specify the Europe/Warsaw zone: the IANA
    # string, the dateutil-prefixed string and (if pytz is installed) a pytz zone
    # object. Returns the current value.
    # KNOWN ISSUE: the return annotation says string, but one parameter is a tzinfo
    # object; the docstring also mentions zoneinfo, which is not among the params.
    """
    tzinfo for Europe/Warsaw using pytz, dateutil, or zoneinfo.
    """
    return request.param


@pytest.fixture
def temp_file(tmp_path):
    # [FUNCTION DOC]
    # Fixture depending on pytest's temporary directory fixture. Creates an empty
    # file with a random unique name in it (side effect: file creation) and returns
    # its path object.
    """
    Generate a unique file for testing use. See link for removal policy.
    https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-default-base-temporary-directory
    """
    file_path = tmp_path / str(uuid.uuid4())
    file_path.touch()
    return file_path


@pytest.fixture(scope="session")
def monkeysession():
    # [FUNCTION DOC]
    # Session-scoped generator fixture providing a monkeypatch helper that lives for
    # the whole test session (the built-in monkeypatch fixture is function scoped).
    # All patches made through it are undone when the session ends. Yields the
    # helper.
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(params=[True, False])
def using_nan_is_na(request):
    # [FUNCTION DOC]
    # Generator fixture parametrized over true and false. While the test runs, the
    # global option for distinguishing NaN from NA is set to the negation of the
    # parameter (restored afterwards). Yields the parameter itself, meaning true
    # when NaN is treated as NA.
    opt = request.param
    with pd.option_context("future.distinguish_nan_and_na", not opt):
        yield opt
