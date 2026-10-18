# [MODULE DOC]
# Tests for groupby aggregation with the numba engine: user-defined function signature
# checks, keyword handling, comparison with the cython engine across engine options,
# function caching, the global use-numba option, multiple functions and named
# aggregations, multi-index keys, and reductions.
# Import-time behavior: the module is skipped if numba is not installed; a module-wide
# marker list restricts tests to a single cpu and skips them on ARM platforms with numba
# version 0.61 (segfaults). Uses the nogil, parallel, nopython, as_index and
# frame_or_series fixtures.
import numpy as np
import pytest

from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError

from pandas import (
    DataFrame,
    Index,
    NamedAgg,
    Series,
    option_context,
)
import pandas._testing as tm
from pandas.util.version import Version

pytestmark = [pytest.mark.single_cpu]

numba = pytest.importorskip("numba")
pytestmark.append(
    pytest.mark.skipif(
        Version(numba.__version__) == Version("0.61") and is_platform_arm(),
        reason=f"Segfaults on ARM platforms with numba {numba.__version__}",
    )
)


def test_correct_function_signature():
    # [FUNCTION DOC]
    # Verifies a user function that does not take the required leading values and index
    # arguments raises the numba utility error ("The first 2 ...") for frame and column
    # aggregation with the numba engine.
    pytest.importorskip("numba")

    def incorrect_function(x):
        # [FUNCTION DOC]
        # HELPER: one-argument function returning a scaled sum; deliberately has the
        # wrong signature for the numba engine.
        return sum(x) * 2.7

    data = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )
    with pytest.raises(NumbaUtilError, match="The first 2"):
        data.groupby("key").agg(incorrect_function, engine="numba")

    with pytest.raises(NumbaUtilError, match="The first 2"):
        data.groupby("key")["data"].agg(incorrect_function, engine="numba")


def test_check_nopython_kwargs():
    # [FUNCTION DOC]
    # Verifies keyword argument handling with the numba engine: missing required
    # arguments raise a type error from signature binding (for both keyword-only and
    # regular parameters), and a function with a keyword-only parameter raises the numba
    # utility error ("numba does not support") once bound, while the correct function
    # returns the sum times 2.7 plus one, for frame and column aggregation.
    pytest.importorskip("numba")

    def incorrect_function(values, index, *, a):
        # [FUNCTION DOC]
        # HELPER: function with a keyword-only parameter a; returns the sum of values
        # scaled by 2.7 plus a. Used to trigger the numba signature error.
        return sum(values) * 2.7 + a

    def correct_function(values, index, a):
        # [FUNCTION DOC]
        # HELPER: function taking values, index and a; returns the sum of values scaled
        # by 2.7 plus a. The valid signature for numba.
        return sum(values) * 2.7 + a

    data = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )
    expected = data.groupby("key").sum() * 2.7

    # py signature binding
    with pytest.raises(
        TypeError, match="missing a required (keyword-only argument|argument): 'a'"
    ):
        data.groupby("key").agg(incorrect_function, engine="numba", b=1)
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        data.groupby("key").agg(correct_function, engine="numba", b=1)

    with pytest.raises(
        TypeError, match="missing a required (keyword-only argument|argument): 'a'"
    ):
        data.groupby("key")["data"].agg(incorrect_function, engine="numba", b=1)
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        data.groupby("key")["data"].agg(correct_function, engine="numba", b=1)

    # numba signature check after binding
    with pytest.raises(NumbaUtilError, match="numba does not support"):
        data.groupby("key").agg(incorrect_function, engine="numba", a=1)
    actual = data.groupby("key").agg(correct_function, engine="numba", a=1)
    tm.assert_frame_equal(expected + 1, actual)

    with pytest.raises(NumbaUtilError, match="numba does not support"):
        data.groupby("key")["data"].agg(incorrect_function, engine="numba", a=1)
    actual = data.groupby("key")["data"].agg(correct_function, engine="numba", a=1)
    tm.assert_series_equal(expected["data"] + 1, actual)


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
@pytest.mark.parametrize("jit", [True, False])
def test_numba_vs_cython(jit, frame_or_series, nogil, parallel, nopython, as_index):
    # [FUNCTION DOC]
    # Verifies the numba engine result equals the cython engine for a
    # mean-times-constant function on a frame or column, with engine options nogil,
    # parallel and nopython and the as_index setting. Parametrized over whether the
    # function is pre-compiled with the numba jit. All warnings are ignored (parallel
    # cannot always be applied).
    pytest.importorskip("numba")

    def func_numba(values, index):
        # [FUNCTION DOC]
        # HELPER: function of values and index returning the mean of the values times
        # 2.7; the numba-compatible aggregation under test (optionally jit-compiled by
        # the enclosing test).
        return np.mean(values) * 2.7

    if jit:
        # Test accepted jitted functions
        import numba

        func_numba = numba.jit(func_numba)

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    grouped = data.groupby(0, as_index=as_index)
    if frame_or_series is Series:
        grouped = grouped[1]

    result = grouped.agg(func_numba, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine="cython")

    tm.assert_equal(result, expected)


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
@pytest.mark.parametrize("jit", [True, False])
def test_cache(jit, frame_or_series, nogil, parallel, nopython):
    # [FUNCTION DOC]
    # Verifies numba-compiled function caching when switching functions: aggregate with
    # the first function, then the second, then the first again; each result must equal
    # the cython-engine equivalent. Parametrized over jit or not and over engine
    # options; warnings are ignored.
    # Test that the functions are cached correctly if we switch functions
    pytest.importorskip("numba")

    def func_1(values, index):
        # [FUNCTION DOC]
        # HELPER: function of values and index returning the mean minus 3.4.
        return np.mean(values) - 3.4

    def func_2(values, index):
        # [FUNCTION DOC]
        # HELPER: function of values and index returning the mean times 2.7.
        return np.mean(values) * 2.7

    if jit:
        import numba

        func_1 = numba.jit(func_1)
        func_2 = numba.jit(func_2)

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    grouped = data.groupby(0)
    if frame_or_series is Series:
        grouped = grouped[1]

    result = grouped.agg(func_1, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) - 3.4, engine="cython")
    tm.assert_equal(result, expected)

    # Add func_2 to the cache
    result = grouped.agg(func_2, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine="cython")
    tm.assert_equal(result, expected)

    # Retest func_1 which should use the cache
    result = grouped.agg(func_1, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) - 3.4, engine="cython")
    tm.assert_equal(result, expected)


def test_use_global_config():
    # [FUNCTION DOC]
    # Verifies that enabling the global compute option for numba makes aggregation with
    # no engine given produce the same result as explicitly requesting the numba engine.
    pytest.importorskip("numba")

    def func_1(values, index):
        # [FUNCTION DOC]
        # HELPER: function of values and index returning the mean minus 3.4.
        return np.mean(values) - 3.4

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    grouped = data.groupby(0)
    expected = grouped.agg(func_1, engine="numba")
    with option_context("compute.use_numba", True):
        result = grouped.agg(func_1, engine=None)
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "agg_kwargs",
    [
        {"func": ["min", "max"]},
        {"func": "min"},
        {"func": {1: ["min", "max"], 2: "sum"}},
        {"bmin": NamedAgg(column=1, aggfunc="min")},
    ],
)
def test_multifunc_numba_vs_cython_frame(agg_kwargs):
    # [FUNCTION DOC]
    # Verifies that numba and cython engines give equal frames for several
    # specifications: a list of names, a single name, a dictionary mapping columns to
    # names or lists, and a named aggregation. Parametrized over the aggregation keyword
    # dictionaries.
    pytest.importorskip("numba")
    data = DataFrame(
        {
            0: ["a", "a", "b", "b", "a"],
            1: [1.0, 2.0, 3.0, 4.0, 5.0],
            2: [1, 2, 3, 4, 5],
        },
        columns=[0, 1, 2],
    )
    grouped = data.groupby(0)
    result = grouped.agg(**agg_kwargs, engine="numba")
    expected = grouped.agg(**agg_kwargs, engine="cython")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["sum", "mean", "var", "std", "min", "max"])
def test_multifunc_numba_vs_cython_frame_noskipna(func):
    # [FUNCTION DOC]
    # Verifies numba and cython engines agree for sum, mean, var, std, min and max with
    # skipna disabled on data containing a missing value. Parametrized over the function
    # name.
    pytest.importorskip("numba")
    data = DataFrame(
        {
            0: ["a", "a", "b", "b", "a"],
            1: [1.0, np.nan, 3.0, 4.0, 5.0],
            2: [1, 2, 3, 4, 5],
        },
        columns=[0, 1, 2],
    )
    grouped = data.groupby(0)
    result = grouped.agg(func, skipna=False, engine="numba")
    expected = grouped.agg(func, skipna=False, engine="cython")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg_kwargs,expected_func",
    [
        ({"func": lambda values, index: values.sum()}, "sum"),
        # FIXME
        pytest.param(
            {
                "func": [
                    lambda values, index: values.sum(),
                    lambda values, index: values.min(),
                ]
            },
            ["sum", "min"],
            marks=pytest.mark.xfail(
                reason="This doesn't work yet! Fails in nopython pipeline!"
            ),
        ),
    ],
)
def test_multifunc_numba_udf_frame(agg_kwargs, expected_func):
    # [FUNCTION DOC]
    # Verifies a user-defined function (or list of them) with the numba engine matches
    # the equivalent built-in reductions in the cython engine, ignoring dtype
    # differences (UDFs always return float64). The list-of-functions case is an
    # expected failure because it does not work yet in the numba pipeline. Parametrized
    # over the aggregation keywords and the expected function names.
    pytest.importorskip("numba")
    data = DataFrame(
        {
            0: ["a", "a", "b", "b", "a"],
            1: [1.0, 2.0, 3.0, 4.0, 5.0],
            2: [1, 2, 3, 4, 5],
        },
        columns=[0, 1, 2],
    )
    grouped = data.groupby(0)
    result = grouped.agg(**agg_kwargs, engine="numba")
    expected = grouped.agg(expected_func, engine="cython")
    # check_dtype can be removed if GH 44952 is addressed
    # Currently, UDFs still always return float64 while reductions can preserve dtype
    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "agg_kwargs",
    [{"func": ["min", "max"]}, {"func": "min"}, {"min_val": "min", "max_val": "max"}],
)
def test_multifunc_numba_vs_cython_series(agg_kwargs):
    # [FUNCTION DOC]
    # Verifies numba and cython engines agree on a series for a list of names, a single
    # name, and keyword-named aggregations. Compares frames or series depending on the
    # result type.
    # KNOWN ISSUE: the test mutates the parametrized keyword dictionary by adding the
    # engine key, so the shared parameter object is changed across runs.
    pytest.importorskip("numba")
    labels = ["a", "a", "b", "b", "a"]
    data = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    grouped = data.groupby(labels)
    agg_kwargs["engine"] = "numba"
    result = grouped.agg(**agg_kwargs)
    agg_kwargs["engine"] = "cython"
    expected = grouped.agg(**agg_kwargs)
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)


@pytest.mark.single_cpu
@pytest.mark.parametrize(
    "data,agg_kwargs",
    [
        (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {"func": ["min", "max"]}),
        (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {"func": "min"}),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": ["min", "max"]},
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": "min"},
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": {1: ["min", "max"], 2: "sum"}},
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"min_col": NamedAgg(column=1, aggfunc="min")},
        ),
    ],
)
def test_multifunc_numba_kwarg_propagation(data, agg_kwargs):
    # [FUNCTION DOC]
    # Verifies that passing engine options with parallel enabled gives the same result
    # as the default numba options, for series and frame inputs with name, list,
    # dictionary and named aggregation specifications. Marked single cpu.
    pytest.importorskip("numba")
    labels = ["a", "a", "b", "b", "a"]
    grouped = data.groupby(labels)
    result = grouped.agg(**agg_kwargs, engine="numba", engine_kwargs={"parallel": True})
    expected = grouped.agg(**agg_kwargs, engine="numba")
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)


def test_args_not_cached():
    # [FUNCTION DOC]
    # Verifies (GH 41647) that positional arguments passed to a numba aggregation are
    # not cached: calling with last-one and last-two element sums gives 1.0 and 2.0 per
    # group.
    # GH 41647
    pytest.importorskip("numba")

    def sum_last(values, index, n):
        # [FUNCTION DOC]
        # HELPER: function of values, index and n returning the sum of the last n
        # values.
        return values[-n:].sum()

    df = DataFrame({"id": [0, 0, 1, 1], "x": [1, 1, 1, 1]})
    grouped_x = df.groupby("id")["x"]
    result = grouped_x.agg(sum_last, 1, engine="numba")
    expected = Series([1.0] * 2, name="x", index=Index([0, 1], name="id"))
    tm.assert_series_equal(result, expected)

    result = grouped_x.agg(sum_last, 2, engine="numba")
    expected = Series([2.0] * 2, name="x", index=Index([0, 1], name="id"))
    tm.assert_series_equal(result, expected)


def test_index_data_correctly_passed():
    # [FUNCTION DOC]
    # Verifies (GH 43133) that the index passed to a numba aggregation is the frame's
    # index: the mean of a negative index per group is -1.5 and -3.0.
    # GH 43133
    pytest.importorskip("numba")

    def f(values, index):
        # [FUNCTION DOC]
        # HELPER: function of values and index returning the mean of the index.
        return np.mean(index)

    df = DataFrame({"group": ["A", "A", "B"], "v": [4, 5, 6]}, index=[-1, -2, -3])
    result = df.groupby("group").aggregate(f, engine="numba")
    expected = DataFrame(
        [-1.5, -3.0], columns=["v"], index=Index(["A", "B"], name="group")
    )
    tm.assert_frame_equal(result, expected)


def test_engine_kwargs_not_cached():
    # [FUNCTION DOC]
    # Verifies different engine options are not served from a cached compiled function:
    # a function adding three boolean-like closure values returns 2.0 per group, then
    # 1.0 after one flag is changed to false.
    # If the user passes a different set of engine_kwargs don't return the same
    # jitted function
    pytest.importorskip("numba")
    nogil = True
    parallel = False
    nopython = True

    def func_kwargs(values, index):
        # [FUNCTION DOC]
        # HELPER: function of values and index returning the sum of the three enclosing
        # option variables (nogil, parallel, nopython) at call time.
        return nogil + parallel + nopython

    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    df = DataFrame({"value": [0, 0, 0]})
    result = df.groupby(level=0).aggregate(
        func_kwargs, engine="numba", engine_kwargs=engine_kwargs
    )
    expected = DataFrame({"value": [2.0, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)

    nogil = False
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    result = df.groupby(level=0).aggregate(
        func_kwargs, engine="numba", engine_kwargs=engine_kwargs
    )
    expected = DataFrame({"value": [1.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings("ignore")
def test_multiindex_one_key(nogil, parallel, nopython):
    # [FUNCTION DOC]
    # Verifies the numba engine works when grouping a multi-indexed frame by one level:
    # a constant-one function gives float 1.0 for the remaining column. Parametrized
    # over engine options; warnings ignored.
    pytest.importorskip("numba")

    def numba_func(values, index):
        # [FUNCTION DOC]
        # HELPER: function of values and index returning the constant one.
        return 1

    df = DataFrame([{"A": 1, "B": 2, "C": 3}]).set_index(["A", "B"])
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    result = df.groupby("A").agg(
        numba_func, engine="numba", engine_kwargs=engine_kwargs
    )
    expected = DataFrame([1.0], index=Index([1], name="A"), columns=["C"])
    tm.assert_frame_equal(result, expected)


def test_multiindex_multi_key_not_supported(nogil, parallel, nopython):
    # [FUNCTION DOC]
    # Verifies grouping by more than one level with the numba engine raises a
    # not-implemented error ("more than 1 grouping labels"). Parametrized over engine
    # options.
    pytest.importorskip("numba")

    def numba_func(values, index):
        # [FUNCTION DOC]
        # HELPER: function of values and index returning the constant one.
        return 1

    df = DataFrame([{"A": 1, "B": 2, "C": 3}]).set_index(["A", "B"])
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    with pytest.raises(NotImplementedError, match="more than 1 grouping labels"):
        df.groupby(["A", "B"]).agg(
            numba_func, engine="numba", engine_kwargs=engine_kwargs
        )


def test_multilabel_numba_vs_cython(numba_supported_reductions):
    # [FUNCTION DOC]
    # Verifies that numba and cython engines agree when grouping by two keys, via
    # aggregate and via calling the reduction method directly, for each supported
    # reduction and keyword dictionary from the numba_supported_reductions fixture.
    pytest.importorskip("numba")
    reduction, kwargs = numba_supported_reductions
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )
    gb = df.groupby(["A", "B"])
    res_agg = gb.agg(reduction, engine="numba", **kwargs)
    expected_agg = gb.agg(reduction, engine="cython", **kwargs)
    tm.assert_frame_equal(res_agg, expected_agg)
    # Test that calling the aggregation directly also works
    direct_res = getattr(gb, reduction)(engine="numba", **kwargs)
    direct_expected = getattr(gb, reduction)(engine="cython", **kwargs)
    tm.assert_frame_equal(direct_res, direct_expected)


def test_multilabel_udf_numba_vs_cython():
    # [FUNCTION DOC]
    # Verifies a user-defined minimum function with the numba engine equals the
    # cython-engine minimum when grouping by two keys.
    pytest.importorskip("numba")
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )
    gb = df.groupby(["A", "B"])
    result = gb.agg(lambda values, index: values.min(), engine="numba")
    expected = gb.agg(lambda x: x.min(), engine="cython")
    tm.assert_frame_equal(result, expected)
