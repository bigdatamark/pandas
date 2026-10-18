# [MODULE DOC]
# Tests for the numba execution engine of window operations (rolling,
# expanding, ewm, including grouped and the table method): agreement with the
# cython engine, caching of compiled functions, argument forwarding, engine
# option validation, global configuration, the table method, and numba-based
# min and max with custom indexers. Import-time side effects: the whole module is
# skipped when numba is missing, marked single-cpu, and skipped on ARM with
# numba 0.61 because it segfaults there. Defines a method fixture, a fixture
# listing operations supported by numba with their keyword arguments, a frame
# fixture, and a prescribed-bounds indexer helper class.
import numpy as np
import pytest

from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    option_context,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.util.version import Version

pytestmark = [pytest.mark.single_cpu]

numba = pytest.importorskip("numba")
pytestmark.append(
    pytest.mark.skipif(
        Version(numba.__version__) == Version("0.61") and is_platform_arm(),
        reason=f"Segfaults on ARM platforms with numba {numba.__version__}",
    )
)


@pytest.fixture(params=["single", "table"])
def method(request):
    # [FUNCTION DOC]
    # Fixture parametrized over "single" and "table"; yields the value for the
    # method keyword of the rolling, expanding and ewm constructors.
    """method keyword in rolling/expanding/ewm constructor"""
    return request.param


@pytest.fixture(
    params=[
        ["sum", {}],
        ["mean", {}],
        ["median", {}],
        ["max", {}],
        ["min", {}],
        ["var", {}],
        ["var", {"ddof": 0}],
        ["std", {}],
        ["std", {"ddof": 0}],
    ]
)
def arithmetic_numba_supported_operators(request):
    # [FUNCTION DOC]
    # Fixture parametrized over (operation name, keyword arguments) pairs for the
    # operations numba supports: sum, mean, median, max, min, var and std (the
    # latter two also with ddof zero). Yields the pair as a list.
    return request.param


@pytest.fixture
def roll_frame():
    # [FUNCTION DOC]
    # Fixture returning a 40-row frame with group column A (20 ones, 12 twos, 8
    # threes) and column B holding the integers 0 to 39.
    return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})


@td.skip_if_no("numba")
@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestEngine:
    # [CLASS DOC]
    # Tests that the numba engine matches the cython engine and that compiled
    # functions are cached properly. Requires numba (skipped otherwise); warnings
    # are ignored because numba warns when parallel mode cannot be applied. Has no
    # state.
    @pytest.mark.parametrize("jit", [True, False])
    def test_numba_vs_cython_apply(self, jit, nogil, parallel, nopython, center, step):
        # [FUNCTION DOC]
        # Parametrized over whether the user function is numba-compiled (jit true or
        # false) plus the nogil, parallel, nopython, center and step fixtures. Verifies
        # rolling apply with window 2 over ten numbers using the numba engine and engine
        # options gives the same series as the cython engine, with a user function
        # adding a sum of extra positional arguments (here 2) to the window mean.
        def f(x, *args):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes an array x and any number of extra
            # positional numbers; returns the mean of x plus the sum of the extra numbers.
            arg_sum = 0
            for arg in args:
                arg_sum += arg
            return np.mean(x) + arg_sum

        if jit:
            import numba

            f = numba.jit(f)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        args = (2,)

        s = Series(range(10))
        result = s.rolling(2, center=center, step=step).apply(
            f, args=args, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = s.rolling(2, center=center, step=step).apply(
            f, engine="cython", args=args, raw=True
        )
        tm.assert_series_equal(result, expected)

    def test_apply_numba_with_kwargs(self, roll_frame):
        # [FUNCTION DOC]
        # Regression test for GH 58995. Verifies extra keyword arguments (a mapping with
        # a constant of 1) and extra positional arguments (a one-element tuple) are
        # passed through to the numba function for rolling apply, expanding apply, and
        # the grouped versions of each, equal to the corresponding sum plus 1. The
        # function returns the window sum plus the optional constant.
        # GH 58995
        # rolling apply
        def func(sr, a=0):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes an array sr and an optional constant a
            # (default 0); returns the sum of sr plus a.
            return sr.sum() + a

        data = DataFrame(range(10))

        result = data.rolling(5).apply(func, engine="numba", raw=True, kwargs={"a": 1})
        expected = data.rolling(5).sum() + 1
        tm.assert_frame_equal(result, expected)

        result = data.rolling(5).apply(func, engine="numba", raw=True, args=(1,))
        tm.assert_frame_equal(result, expected)

        # expanding apply

        result = data.expanding().apply(func, engine="numba", raw=True, kwargs={"a": 1})
        expected = data.expanding().sum() + 1
        tm.assert_frame_equal(result, expected)

        result = data.expanding().apply(func, engine="numba", raw=True, args=(1,))
        tm.assert_frame_equal(result, expected)

        # groupby rolling
        result = (
            roll_frame.groupby("A")
            .rolling(5)
            .apply(func, engine="numba", raw=True, kwargs={"a": 1})
        )
        expected = roll_frame.groupby("A").rolling(5).sum() + 1
        tm.assert_frame_equal(result, expected)

        result = (
            roll_frame.groupby("A")
            .rolling(5)
            .apply(func, engine="numba", raw=True, args=(1,))
        )
        tm.assert_frame_equal(result, expected)
        # groupby expanding

        result = (
            roll_frame.groupby("A")
            .expanding()
            .apply(func, engine="numba", raw=True, kwargs={"a": 1})
        )
        expected = roll_frame.groupby("A").expanding().sum() + 1
        tm.assert_frame_equal(result, expected)

        result = (
            roll_frame.groupby("A")
            .expanding()
            .apply(func, engine="numba", raw=True, args=(1,))
        )
        tm.assert_frame_equal(result, expected)

    def test_numba_min_periods(self):
        # [FUNCTION DOC]
        # Regression test for GH 58868. Four-row frame with the table method, window 3
        # and minimum periods 3: the user function (which asserts it only ever receives
        # windows of exactly 3 rows) must never see partial windows; the first two rows
        # are missing and the rest are the last row of each window.
        # GH 58868
        def last_row(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes a two-dimensional array x, asserts it has
            # exactly three rows, and returns its last row.
            assert len(x) == 3
            return x[-1]

        df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])

        result = df.rolling(3, method="table", min_periods=3).apply(
            last_row, raw=True, engine="numba"
        )

        expected = DataFrame([[np.nan, np.nan], [np.nan, np.nan], [5, 6], [7, 8]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            DataFrame(np.eye(5)),
            DataFrame(
                [
                    [5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3],
                    [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3],
                    [np.nan, np.nan, 5, 6, 7, 5, 5, 5, 5, 5],
                ]
            ).T,
            Series(range(5), name="foo"),
            Series([20, 10, 10, np.inf, 1, 1, 2, 3]),
            Series([20, 10, 10, np.nan, 10, 1, 2, 3]),
        ],
    )
    def test_numba_vs_cython_rolling_methods(
        self,
        data,
        nogil,
        parallel,
        nopython,
        arithmetic_numba_supported_operators,
        step,
    ):
        # [FUNCTION DOC]
        # Parametrized over five data objects (identity frame, a transposed frame with
        # missing and infinite values, and three series including infinities and
        # missing values) and the supported operation fixture, with nogil, parallel,
        # nopython and step. Verifies rolling (window 3) results using the numba engine
        # equal those from the cython engine.
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        roll = data.rolling(3, step=step)
        result = getattr(roll, method)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(roll, method)(engine="cython", **kwargs)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "data", [DataFrame(np.eye(5)), Series(range(5), name="foo")]
    )
    def test_numba_vs_cython_expanding_methods(
        self, data, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):
        # [FUNCTION DOC]
        # Parametrized over data objects (an identity frame and a named series) and the
        # supported operation fixture, with nogil, parallel and nopython. Verifies
        # expanding results with the numba engine equal those from the cython engine.
        # KNOWN ISSUE: the data parameter is overwritten with a fixed five by five identity frame before use, so the parametrization over the series case is ineffective and only the identity frame is tested.
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        data = DataFrame(np.eye(5))
        expand = data.expanding()
        result = getattr(expand, method)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(expand, method)(engine="cython", **kwargs)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("jit", [True, False])
    def test_cache_apply(self, jit, nogil, parallel, nopython, step):
        # [FUNCTION DOC]
        # Parametrized over jit true or false, with nogil, parallel, nopython and step.
        # Verifies compiled function caching: applies a first function with the numba
        # engine, then a second function, then the first again (which must reuse the
        # cached compiled version without returning results of the second), each time
        # comparing with the cython engine result.
        # Test that the functions are cached correctly if we switch functions
        def func_1(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes an array x and returns its mean plus 4.
            return np.mean(x) + 4

        def func_2(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes an array x and returns its standard
            # deviation times 5.
            return np.std(x) * 5

        if jit:
            import numba

            func_1 = numba.jit(func_1)
            func_2 = numba.jit(func_2)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        roll = Series(range(10)).rolling(2, step=step)
        result = roll.apply(
            func_1, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_1, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)

        result = roll.apply(
            func_2, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_2, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)
        # This run should use the cached func_1
        result = roll.apply(
            func_1, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_1, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "window,window_kwargs",
        [
            ["rolling", {"window": 3, "min_periods": 0}],
            ["expanding", {}],
        ],
    )
    def test_dont_cache_args(
        self, window, window_kwargs, nogil, parallel, nopython, method
    ):
        # [FUNCTION DOC]
        # Regression test for GH 42287. Parametrized over rolling (window 3, minimum
        # periods 0) and expanding, with the method fixture. Applies a function adding
        # an extra positional argument, first with 1 and then with 2, to a frame of
        # three zeros using the numba engine; verifies results are all 1.0 and then all
        # 2.0, so the extra argument value is not cached inside the compiled function.
        # GH 42287

        def add(values, x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes an array values and a number x; returns
            # the sum of values plus x.
            return np.sum(values) + x

        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        df = DataFrame({"value": [0, 0, 0]})
        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(1,)
        )
        expected = DataFrame({"value": [1.0, 1.0, 1.0]})
        tm.assert_frame_equal(result, expected)

        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(2,)
        )
        expected = DataFrame({"value": [2.0, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)

    def test_dont_cache_engine_kwargs(self):
        # [FUNCTION DOC]
        # Verifies that different engine option sets do not share a compiled function:
        # a function summing three module-level-like booleans (no-gil false, parallel,
        # no-python true) is applied once with parallel true (expected 2.0 everywhere) and
        # once after switching parallel to false (expected 1.0), which only holds if the
        # function is recompiled for the new options.
        # If the user passes a different set of engine_kwargs don't return the same
        # jitted function
        nogil = False
        parallel = True
        nopython = True

        def func(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Ignores its argument and returns the sum of the
            # enclosing test's three boolean option variables, read at compile time.
            return nogil + parallel + nopython

        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        df = DataFrame({"value": [0, 0, 0]})
        result = df.rolling(1).apply(
            func, raw=True, engine="numba", engine_kwargs=engine_kwargs
        )
        expected = DataFrame({"value": [2.0, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)

        parallel = False
        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        result = df.rolling(1).apply(
            func, raw=True, engine="numba", engine_kwargs=engine_kwargs
        )
        expected = DataFrame({"value": [1.0, 1.0, 1.0]})
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numba")
class TestEWM:
    # [CLASS DOC]
    # Tests for the numba engine of exponentially weighted windows, including
    # grouped ewm. Requires numba (skipped otherwise). Has no state.
    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_invalid_engine(self, grouper, method):
        # [FUNCTION DOC]
        # Parametrized over plain or grouped frames and the methods mean and sum.
        # Verifies an unknown engine name raises a value error saying the engine must be
        # one of the allowed values.
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        with pytest.raises(ValueError, match="engine must be either"):
            getattr(grouper(df).ewm(com=1.0), method)(engine="foo")

    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_invalid_engine_kwargs(self, grouper, method):
        # [FUNCTION DOC]
        # Parametrized over plain or grouped frames and the methods mean and sum.
        # Verifies passing engine options with the cython engine raises a value error
        # saying the cython engine does not accept them.
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        with pytest.raises(ValueError, match="cython engine does not"):
            getattr(grouper(df).ewm(com=1.0), method)(
                engine="cython", engine_kwargs={"nopython": True}
            )

    @pytest.mark.parametrize("grouper", ["None", "groupby"])
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_cython_vs_numba(
        self, grouper, method, nogil, parallel, nopython, ignore_na, adjust
    ):
        # [FUNCTION DOC]
        # Parametrized over plain or grouped data and the methods mean and sum, with
        # nogil, parallel, nopython, ignore_na and adjust. Verifies the numba engine
        # result equals the cython engine result for ewm with center of mass 1 (sum is
        # only tested with adjust true).
        df = DataFrame({"B": range(4)})
        if grouper == "None":
            grouper = lambda x: x
        else:
            df["A"] = ["a", "b", "a", "b"]
            grouper = lambda x: x.groupby("A")
        if method == "sum":
            adjust = True
        ewm = grouper(df).ewm(com=1.0, adjust=adjust, ignore_na=ignore_na)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        result = getattr(ewm, method)(engine="numba", engine_kwargs=engine_kwargs)
        expected = getattr(ewm, method)(engine="cython")

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("grouper", ["None", "groupby"])
    def test_cython_vs_numba_times(self, grouper, nogil, parallel, nopython, ignore_na):
        # [FUNCTION DOC]
        # Regression test for GH 40951. Parametrized over plain or grouped data. With
        # irregular timestamps and a 23 days half-life, verifies the ewm mean using
        # numba equals the cython mean.
        # GH 40951

        df = DataFrame({"B": [0, 0, 1, 1, 2, 2]})
        if grouper == "None":
            grouper = lambda x: x
        else:
            grouper = lambda x: x.groupby("A")
            df["A"] = ["a", "b", "a", "b", "b", "a"]

        halflife = "23 days"
        times = to_datetime(
            [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-10",
                "2020-02-23",
                "2020-01-03",
            ]
        )
        ewm = grouper(df).ewm(
            halflife=halflife, adjust=True, ignore_na=ignore_na, times=times
        )

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        result = ewm.mean(engine="numba", engine_kwargs=engine_kwargs)
        expected = ewm.mean(engine="cython")

        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numba")
def test_use_global_config():
    # [FUNCTION DOC]
    # Verifies that setting the global option to compute with numba makes rolling
    # apply with no explicit engine give the same result as asking for the numba
    # engine explicitly.
    def f(x):
        # [FUNCTION DOC]
        # HELPER: nested user function. Takes an array x and returns its mean plus 2.
        return np.mean(x) + 2

    s = Series(range(10))
    with option_context("compute.use_numba", True):
        result = s.rolling(2).apply(f, engine=None, raw=True)
    expected = s.rolling(2).apply(f, engine="numba", raw=True)
    tm.assert_series_equal(expected, result)


@td.skip_if_no("numba")
def test_invalid_kwargs_nopython():
    # [FUNCTION DOC]
    # Verifies keyword argument handling of numba apply: an unknown keyword raises a
    # type error naming the unexpected argument; a function with a keyword-only
    # parameter raises a numba utility error saying keyword-only arguments are
    # unsupported; and a function with a normal parameter receiving a keyword
    # argument works, giving the data plus one as floats.
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'a'"):
        Series(range(1)).rolling(1).apply(
            lambda x: x, kwargs={"a": 1}, engine="numba", raw=True
        )
    with pytest.raises(
        NumbaUtilError, match="numba does not support keyword-only arguments"
    ):
        Series(range(1)).rolling(1).apply(
            lambda x, *, a: x, kwargs={"a": 1}, engine="numba", raw=True
        )

    tm.assert_series_equal(
        Series(range(1), dtype=float) + 1,
        Series(range(1))
        .rolling(1)
        .apply(lambda x, a: (x + a).sum(), kwargs={"a": 1}, engine="numba", raw=True),
    )


@td.skip_if_no("numba")
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestTableMethod:
    # [CLASS DOC]
    # Tests for the table method (whole-frame windows) of the numba engine for
    # rolling, expanding and ewm. Requires numba, marked slow, warnings ignored.
    # Has no state.
    def test_table_series_valueerror(self):
        # [FUNCTION DOC]
        # Verifies using the table method on a series raises a value error saying the
        # table method is not applicable for series objects.
        def f(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes a two-dimensional array x and returns the
            # column sums plus 1.
            return np.sum(x, axis=0) + 1

        with pytest.raises(
            ValueError, match="method='table' not applicable for Series objects."
        ):
            Series(range(1)).rolling(1, method="table").apply(
                f, engine="numba", raw=True
            )

    def test_table_method_rolling_methods(
        self,
        nogil,
        parallel,
        nopython,
        arithmetic_numba_supported_operators,
        step,
    ):
        # [FUNCTION DOC]
        # Parametrized over supported operations with nogil, parallel, nopython and
        # step. For var and std, verifies the table method raises a not implemented
        # error saying the operation is not supported; for other operations verifies
        # the table method result on an identity frame (window 2, minimum periods 0)
        # equals the single method result.
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(np.eye(3))
        roll_table = df.rolling(2, method="table", min_periods=0, step=step)
        if method in ("var", "std"):
            with pytest.raises(NotImplementedError, match=f"{method} not supported"):
                getattr(roll_table, method)(
                    engine_kwargs=engine_kwargs, engine="numba", **kwargs
                )
        else:
            roll_single = df.rolling(2, method="single", min_periods=0, step=step)
            result = getattr(roll_table, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            expected = getattr(roll_single, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_apply(self, nogil, parallel, nopython, step):
        # [FUNCTION DOC]
        # Verifies rolling apply with the table method equals the single method for a
        # function returning column sums plus 1, with numba engine options and step.
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        def f(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes a two-dimensional array x and returns the
            # column sums plus 1.
            return np.sum(x, axis=0) + 1

        df = DataFrame(np.eye(3))
        result = df.rolling(2, method="table", min_periods=0, step=step).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        expected = df.rolling(2, method="single", min_periods=0, step=step).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_apply_col_order(self):
        # [FUNCTION DOC]
        # Regression test for GH#59666. Verifies a table rolling apply on a selection of
        # two columns uses the selected column order: the function returns the mean of
        # (first column minus second column) so the result for the order a, b is the
        # negation of that for b, a, matching expected values for window 3.
        # GH#59666
        def f(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes a two-column array x; returns the nan-aware
            # mean of the first column minus the second.
            return np.nanmean(x[:, 0] - x[:, 1])

        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [6, 7, 8, 5, 6, 7],
            }
        )
        result = df.rolling(3, method="table", min_periods=0)[["a", "b"]].apply(
            f, raw=True, engine="numba"
        )
        expected = DataFrame(
            {
                "a": [-5, -5, -5, -3.66667, -2.33333, -1],
                "b": [-5, -5, -5, -3.66667, -2.33333, -1],
            }
        )
        tm.assert_almost_equal(result, expected)
        result = df.rolling(3, method="table", min_periods=0)[["b", "a"]].apply(
            f, raw=True, engine="numba"
        )
        expected = DataFrame(
            {
                "b": [5, 5, 5, 3.66667, 2.33333, 1],
                "a": [5, 5, 5, 3.66667, 2.33333, 1],
            }
        )
        tm.assert_almost_equal(result, expected)

    def test_table_method_rolling_weighted_mean(self, step):
        # [FUNCTION DOC]
        # Verifies the table method with a function returning a full row: a weighted mean
        # of the first two columns by the third, with ones elsewhere. Window 2, minimum
        # periods 0, sliced by step, matching the expected frame.
        def weighted_mean(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes a two-dimensional window x whose third
            # column holds weights; returns a one-row array of ones whose first two entries
            # are the weighted means of the first two columns.
            arr = np.ones((1, x.shape[1]))
            arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
            return arr

        df = DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])
        result = df.rolling(2, method="table", min_periods=0, step=step).apply(
            weighted_mean, raw=True, engine="numba"
        )
        expected = DataFrame(
            [
                [1.0, 2.0, 1.0],
                [1.8, 2.0, 1.0],
                [3.333333, 2.333333, 1.0],
                [1.555556, 7, 1.0],
            ]
        )[::step]
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_apply(self, nogil, parallel, nopython):
        # [FUNCTION DOC]
        # Verifies expanding apply with the table method equals the single method for a
        # column-sums-plus-one function, using numba engine options.
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        def f(x):
            # [FUNCTION DOC]
            # HELPER: nested user function. Takes a two-dimensional array x and returns the
            # column sums plus 1.
            return np.sum(x, axis=0) + 1

        df = DataFrame(np.eye(3))
        result = df.expanding(method="table").apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        expected = df.expanding(method="single").apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_methods(
        self, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):
        # [FUNCTION DOC]
        # Parametrized over supported operations. For var and std, verifies the table
        # method raises a not implemented error saying the operation is not supported;
        # for others verifies the table result on an identity frame equals the single
        # method result.
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(np.eye(3))
        expand_table = df.expanding(method="table")
        if method in ("var", "std"):
            with pytest.raises(NotImplementedError, match=f"{method} not supported"):
                getattr(expand_table, method)(
                    engine_kwargs=engine_kwargs, engine="numba", **kwargs
                )
        else:
            expand_single = df.expanding(method="single")
            result = getattr(expand_table, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            expected = getattr(expand_single, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("data", [np.eye(3), np.ones((2, 3)), np.ones((3, 2))])
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_table_method_ewm(self, data, method, nogil, parallel, nopython):
        # [FUNCTION DOC]
        # Parametrized over three array shapes (square, wide, tall) and mean or sum.
        # Verifies ewm (center of mass 1) with the table method equals the single method
        # using the numba engine.
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(data)

        result = getattr(df.ewm(com=1, method="table"), method)(
            engine_kwargs=engine_kwargs, engine="numba"
        )
        expected = getattr(df.ewm(com=1, method="single"), method)(
            engine_kwargs=engine_kwargs, engine="numba"
        )
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numba")
def test_npfunc_no_warnings():
    # [FUNCTION DOC]
    # Verifies that applying a numpy function (product) with the numba engine in
    # rolling apply emits no warnings.
    df = DataFrame({"col1": [1, 2, 3, 4, 5]})
    with tm.assert_produces_warning(False):
        df.col1.rolling(2).apply(np.prod, raw=True, engine="numba")


class PrescribedWindowIndexer(BaseIndexer):
    # [CLASS DOC]
    # HELPER: test indexer whose window bounds are given explicitly as two arrays
    # (start and end positions per row) at construction; extends the base indexer.
    # Bounds are clipped to the valid range when requested. State: start and end
    # arrays.
    def __init__(self, start, end):
        # [FUNCTION DOC]
        # HELPER: constructor. Takes start and end integer arrays, stores them in
        # private attributes, then runs the base class constructor with defaults.
        self._start = start
        self._end = end
        super().__init__()

    def get_window_bounds(
        self, num_values=None, min_periods=None, center=None, closed=None, step=None
    ):
        # [FUNCTION DOC]
        # HELPER: returns the stored bounds. Inputs: number of values (defaults to the
        # length of the stored start array when none), and unused minimum periods,
        # center, closed and step. Returns start and end arrays clipped to the range
        # zero through number of values.
        if num_values is None:
            num_values = len(self._start)
        start = np.clip(self._start, 0, num_values)
        end = np.clip(self._end, 0, num_values)
        return start, end


@td.skip_if_no("numba")
class TestMinMaxNumba:
    # [CLASS DOC]
    # Tests for the numba implementation of rolling min and max with arbitrary
    # (non-monotonic) window bounds. Requires numba. Has no state.
    @pytest.mark.parametrize(
        "is_max, has_nan, exp_list",
        [
            (True, False, [3.0, 5.0, 2.0, 5.0, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            (True, True, [3.0, 4.0, 2.0, 4.0, 1.0, 4.0, 6.0, 7.0, 7.0, 9.0]),
            (False, False, [3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 7.0, 0.0]),
            (False, True, [3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 6.0, 6.0, 7.0, 1.0]),
        ],
    )
    def test_minmax(self, is_max, has_nan, exp_list):
        # [FUNCTION DOC]
        # Parametrized over max/min, whether some data values are missing, and expected
        # results. A ten-row frame with prescribed start and end bounds per row (via the
        # indexer helper) is rolled with the numba engine; the max or min per window
        # (ignoring missing values) must equal the expected series.
        nan_idx = [0, 5, 8]
        df = DataFrame(
            {
                "data": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0, 7.0, 8.0, 9.0],
                "start": [2, 0, 3, 0, 4, 0, 5, 5, 7, 3],
                "end": [3, 4, 4, 5, 5, 6, 7, 8, 9, 10],
            }
        )
        if has_nan:
            df.loc[nan_idx, "data"] = np.nan
        expected = Series(exp_list, name="data")
        r = df.data.rolling(
            PrescribedWindowIndexer(df.start.to_numpy(), df.end.to_numpy())
        )
        if is_max:
            result = r.max(engine="numba")
        else:
            result = r.min(engine="numba")

        tm.assert_series_equal(result, expected)

    def test_wrong_order(self):
        # [FUNCTION DOC]
        # Verifies that bounds whose start or end positions are not monotonic (the
        # fourth window ends where the third does and starts before the third's start)
        # make numba max raise a value error saying the start/end ordering requirement
        # is violated at row 3.
        start = np.array(range(5), dtype=np.int64)
        end = start + 1
        end[3] = end[2]
        start[3] = start[2] - 1

        df = DataFrame({"data": start * 1.0, "start": start, "end": end})

        r = df.data.rolling(PrescribedWindowIndexer(start, end))
        with pytest.raises(
            ValueError, match="Start/End ordering requirement is violated at index 3"
        ):
            r.max(engine="numba")
