# [MODULE DOC]
# Tests for the plotting backend option and backend discovery: setting the
# option to a missing or incomplete module fails and leaves the default,
# a user-provided module can be used through the option, the backend keyword
# of a plot call, or a registered entry point, and non-matplotlib backends may
# accept arbitrary plot kinds. The whole module is marked to run in a single
# CPU process (global option state). No other import-time side effects.

import sys
import types

import pytest

import pandas.util._test_decorators as td

import pandas

pytestmark = pytest.mark.single_cpu


@pytest.fixture
def dummy_backend():
    # [FUNCTION DOC]
    # Fixture returning a fresh module object named as a dummy plotting backend
    # with a plot function that accepts anything and returns the marker string
    # "used_dummy". Not registered anywhere; tests insert it into the module table.
    db = types.ModuleType("pandas_dummy_backend")
    setattr(db, "plot", lambda *args, **kwargs: "used_dummy")
    return db


@pytest.fixture
def restore_backend():
    # [FUNCTION DOC]
    # Fixture that yields, then restores the plotting backend option to matplotlib
    # by running inside an option context. Ensures tests that change the backend
    # do not leak the change.
    """Restore the plotting backend to matplotlib"""
    with pandas.option_context("plotting.backend", "matplotlib"):
        yield


def test_backend_is_not_module():
    # [FUNCTION DOC]
    # Setting the backend option to a nonexistent module raises a value error that
    # the backend cannot be found; the option remains matplotlib afterwards.
    msg = "Could not find plotting backend 'not_an_existing_module'."
    with pytest.raises(ValueError, match=msg):
        pandas.set_option("plotting.backend", "not_an_existing_module")

    assert pandas.options.plotting.backend == "matplotlib"


def test_backend_is_correct(monkeypatch, restore_backend, dummy_backend):
    # [FUNCTION DOC]
    # Registers the dummy backend as an importable module (using monkeypatch, so
    # the module table is restored) and sets the option to it. Checks the option
    # value changes and that the backend lookup function returns the dummy module.
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)

    pandas.set_option("plotting.backend", "pandas_dummy_backend")
    assert pandas.get_option("plotting.backend") == "pandas_dummy_backend"
    assert (
        pandas.plotting._core._get_plot_backend("pandas_dummy_backend") is dummy_backend
    )


def test_backend_can_be_set_in_plot_call(monkeypatch, restore_backend, dummy_backend):
    # [FUNCTION DOC]
    # Registers the dummy backend and calls frame plotting with the backend
    # keyword naming it. Option stays matplotlib; the call returns the dummy
    # plot function's result.
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)
    df = pandas.DataFrame([1, 2, 3])

    assert pandas.get_option("plotting.backend") == "matplotlib"
    assert df.plot(backend="pandas_dummy_backend") == "used_dummy"


def test_register_entrypoint(restore_backend, tmp_path, monkeypatch, dummy_backend):
    # [FUNCTION DOC]
    # Creates a fake installed distribution (metadata directory with an entry
    # points file under a temporary path put on the import path) that maps a
    # plotting backend entry point name, distinct from the module name, to the dummy
    # module. Checks the lookup by entry point name returns the dummy module, and
    # that setting the option to that name makes the default lookup return it.
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)

    dist_info = tmp_path / "my_backend-0.0.0.dist-info"
    dist_info.mkdir()
    # entry_point name should not match module name - otherwise pandas will
    # fall back to backend lookup by module name
    (dist_info / "entry_points.txt").write_bytes(
        b"[pandas_plotting_backends]\nmy_ep_backend = pandas_dummy_backend\n"
    )

    assert pandas.plotting._core._get_plot_backend("my_ep_backend") is dummy_backend

    with pandas.option_context("plotting.backend", "my_ep_backend"):
        assert pandas.plotting._core._get_plot_backend() is dummy_backend


def test_setting_backend_without_plot_raises(monkeypatch):
    # [FUNCTION DOC]
    # GH-28163. A registered module lacking a plot function is not a valid
    # backend: setting the option raises a value error saying the backend could not
    # be found, and the option stays matplotlib.
    # GH-28163
    module = types.ModuleType("pandas_plot_backend")
    monkeypatch.setitem(sys.modules, "pandas_plot_backend", module)

    assert pandas.options.plotting.backend == "matplotlib"
    with pytest.raises(
        ValueError, match="Could not find plotting backend 'pandas_plot_backend'."
    ):
        pandas.set_option("plotting.backend", "pandas_plot_backend")

    assert pandas.options.plotting.backend == "matplotlib"


@td.skip_if_installed("matplotlib")
def test_no_matplotlib_ok():
    # [FUNCTION DOC]
    # Only runs when matplotlib is NOT installed. Looking up the matplotlib backend
    # must raise an import error saying matplotlib is required for plotting when
    # the default backend is selected.
    msg = (
        'matplotlib is required for plotting when the default backend "matplotlib" is '
        "selected."
    )
    with pytest.raises(ImportError, match=msg):
        pandas.plotting._core._get_plot_backend("matplotlib")


def test_extra_kinds_ok(monkeypatch, restore_backend, dummy_backend):
    # [FUNCTION DOC]
    # Pandas pull 28647. With the dummy backend active, plotting a frame with an
    # arbitrary unknown plot kind must not raise, since extra kinds are the
    # backend's responsibility.
    # https://github.com/pandas-dev/pandas/pull/28647
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)
    pandas.set_option("plotting.backend", "pandas_dummy_backend")
    df = pandas.DataFrame({"A": [1, 2, 3]})
    df.plot(kind="not a real kind")
