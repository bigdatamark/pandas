# [MODULE DOC]
# Tests for the online (incremental) exponentially weighted window, which
# requires numba: the whole module is skipped when numba is missing, marked
# single-cpu, and skipped on ARM platforms with numba version 0.61 because it
# segfaults there. Import-time side effects: imports numba (or skips) and builds
# the module-level mark list. Covers update validation, equality with the
# non-online computation, update with times, and unsupported methods.
import numpy as np
import pytest

from pandas.compat import is_platform_arm

from pandas import (
    DataFrame,
    Series,
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
class TestEWM:
    # [CLASS DOC]
    # Test class for the online ewm object; all warnings are ignored (numba warns
    # when parallel mode cannot be applied). Has no state; each test builds its own
    # data. The nogil, parallel and nopython flags come from shared fixtures.
    def test_invalid_update(self):
        # [FUNCTION DOC]
        # Verifies calling the online mean with an update before the first call without
        # update raises a value error saying mean must be called with no update first.
        df = DataFrame({"a": range(5), "b": range(5)})
        online_ewm = df.head(2).ewm(0.5).online()
        with pytest.raises(
            ValueError,
            match="Must call mean with update=None first before passing update",
        ):
            online_ewm.mean(update=df.head(1))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "obj", [DataFrame({"a": range(5), "b": range(5)}), Series(range(5), name="foo")]
    )
    def test_online_vs_non_online_mean(
        self, obj, nogil, parallel, nopython, adjust, ignore_na
    ):
        # [FUNCTION DOC]
        # Marked slow; parametrized over a two-column frame and a named series, and the
        # nogil, parallel, nopython, adjust and ignore_na fixtures. Verifies the online
        # ewm (center of mass 0.5, built from the first two rows with engine options)
        # reproduces the first two results of the full ewm mean, then, after updating
        # with the last three rows, reproduces the last three results. The sequence is
        # repeated after a reset to verify that reset restores the initial state.
        expected = obj.ewm(0.5, adjust=adjust, ignore_na=ignore_na).mean()
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        online_ewm = (
            obj.head(2)
            .ewm(0.5, adjust=adjust, ignore_na=ignore_na)
            .online(engine_kwargs=engine_kwargs)
        )
        # Test resetting once
        for _ in range(2):
            result = online_ewm.mean()
            tm.assert_equal(result, expected.head(2))

            result = online_ewm.mean(update=obj.tail(3))
            tm.assert_equal(result, expected.tail(3))

            online_ewm.reset()

    @pytest.mark.xfail(raises=NotImplementedError)
    @pytest.mark.parametrize(
        "obj", [DataFrame({"a": range(5), "b": range(5)}), Series(range(5), name="foo")]
    )
    def test_update_times_mean(
        self, obj, nogil, parallel, nopython, adjust, ignore_na, halflife_with_times
    ):
        # [FUNCTION DOC]
        # Marked as expected to fail with a not implemented error. Same as the online
        # versus non-online check but with irregular timestamps and a half-life given
        # as a duration (halflife_with_times fixture): the online object is built with
        # the first two times and updated with the last three values and times; expected
        # to match the full time-weighted mean, repeated after reset. The expected
        # failure reflects that online ewm does not support times yet.
        times = Series(
            np.array(
                ["2020-01-01", "2020-01-05", "2020-01-07", "2020-01-17", "2020-01-21"],
                dtype="datetime64[ns]",
            )
        )
        expected = obj.ewm(
            0.5,
            adjust=adjust,
            ignore_na=ignore_na,
            times=times,
            halflife=halflife_with_times,
        ).mean()

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        online_ewm = (
            obj.head(2)
            .ewm(
                0.5,
                adjust=adjust,
                ignore_na=ignore_na,
                times=times.head(2),
                halflife=halflife_with_times,
            )
            .online(engine_kwargs=engine_kwargs)
        )
        # Test resetting once
        for _ in range(2):
            result = online_ewm.mean()
            tm.assert_equal(result, expected.head(2))

            result = online_ewm.mean(update=obj.tail(3), update_times=times.tail(3))
            tm.assert_equal(result, expected.tail(3))

            online_ewm.reset()

    @pytest.mark.parametrize("method", ["aggregate", "std", "corr", "cov", "var"])
    def test_ewm_notimplementederror_raises(self, method):
        # [FUNCTION DOC]
        # Parametrized over aggregate, std, corr, cov and var. Verifies each of these
        # methods on the online ewm object raises a not implemented error whose message
        # ends with "is not implemented." (the aggregate case is given a function).
        ser = Series(range(10))
        kwargs = {}
        if method == "aggregate":
            kwargs["func"] = lambda x: x

        with pytest.raises(NotImplementedError, match=".* is not implemented."):
            getattr(ser.ewm(1).online(), method)(**kwargs)
