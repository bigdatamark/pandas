# [MODULE DOC]
# Tests for the numba execution engine on groupby reductions. Module is skipped when numba is not installed, runs on a single CPU, and is skipped on ARM with the numba release known to segfault. Reductions are supplied by the numba-supported reductions fixture.
import pytest

from pandas.compat import is_platform_arm

from pandas import (
    DataFrame,
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


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestEngine:
    # [CLASS DOC]
    # Groups engine tests comparing Cython and numba results; ignores all warnings (parallel mode may not be parallelizable).
    def test_cython_vs_numba_frame(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        # [FUNCTION DOC]
        # Compares numba with the default engine for a grouped frame reduction, parametrized over sort and the numba options nogil, parallel and nopython; results must be equal frames.
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = df.groupby("a", sort=sort)
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_frame_equal(result, expected)

    def test_cython_vs_numba_getitem(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        # [FUNCTION DOC]
        # Same comparison on a selected column, producing a series.
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = df.groupby("a", sort=sort)["c"]
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_series_equal(result, expected)

    def test_cython_vs_numba_series(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        # [FUNCTION DOC]
        # Same comparison on a series grouped by index level 0.
        func, kwargs = numba_supported_reductions
        ser = Series(range(3), index=[1, 2, 1], name="foo")
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = ser.groupby(level=0, sort=sort)
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_series_equal(result, expected)

    def test_as_index_false_unsupported(self, numba_supported_reductions):
        # [FUNCTION DOC]
        # Using the numba engine with as_index False raises a not-implemented error mentioning that option.
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        gb = df.groupby("a", as_index=False)
        with pytest.raises(NotImplementedError, match="as_index=False"):
            getattr(gb, func)(engine="numba", **kwargs)

    def test_no_engine_doesnt_raise(self):
        # [FUNCTION DOC]
        # Issue 55520: aggregating with a dict and no engine argument does not raise when the global use-numba option is enabled and equals the result with the option disabled.
        # GH55520
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        gb = df.groupby("a")
        # Make sure behavior of functions w/out engine argument don't raise
        # when the global use_numba option is set
        with option_context("compute.use_numba", True):
            res = gb.agg({"b": "first"})
        expected = gb.agg({"b": "first"})
        tm.assert_frame_equal(res, expected)
