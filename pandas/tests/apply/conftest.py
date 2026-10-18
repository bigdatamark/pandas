# [MODULE DOC]
# HELPER: pytest fixtures and mock execution engine for the apply tests.
# Defines a mock execution engine that delegates to the default python implementation, a decorator
# class that advertises it through the user-defined-function protocol attribute, and an engine fixture.
# Relies on the executors base class from the public executors API. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
from pandas.api.executors import BaseExecutionEngine


class MockExecutionEngine(BaseExecutionEngine):
    # [CLASS DOC]
    # Test double for an execution engine, subclassing the public base execution engine.
    # Used to verify that map and apply forward all user supplied parameters to a custom engine.
    # It reproduces the default python behavior by calling the regular data methods without an engine.
    # The two methods are written without a self argument; they are invoked on the class itself.
    """
    Execution Engine to test if the execution engine interface receives and
    uses all parameters provided by the user.

    Making this engine work as the default Python engine by calling it, no extra
    functionality is implemented here.

    When testing, this will be called when this engine is provided, and then the
    same pandas.map and pandas.apply function will be called, but without engine,
    executing the default behavior from the python engine.
    """

    def map(data, func, args, kwargs, decorator, skip_na):
        # [FUNCTION DOC]
        # HELPER: mock engine map implementation.
        # Takes data (frame or series), the function, positional args, keyword args, decorator and skip_na flag.
        # Passes the keyword args through only for frames (series get none) and calls the data's own map
        # method, using the ignore na action when skip_na is true and no na action otherwise.
        # Returns the mapped result.
        kwargs_to_pass = kwargs if isinstance(data, DataFrame) else {}
        return data.map(func, na_action="ignore" if skip_na else None, **kwargs_to_pass)

    def apply(data, func, args, kwargs, decorator, axis):
        # [FUNCTION DOC]
        # HELPER: mock engine apply implementation.
        # Takes data, function, args, kwargs, decorator and axis.
        # For a series, calls its apply with dtype conversion on, the given args, and by_row disabled.
        # For a frame, calls its apply with the axis, raw off, no result type, the args, by_row set to the
        # compat mode, plus the kwargs. Otherwise the data must be a numpy array (asserted) and the function is
        # applied along the axis through the numpy apply-along-axis routine, wrapped so string results become
        # object dtype arrays. Returns the produced result.
        if isinstance(data, Series):
            return data.apply(func, convert_dtype=True, args=args, by_row=False)
        elif isinstance(data, DataFrame):
            return data.apply(
                func,
                axis=axis,
                raw=False,
                result_type=None,
                args=args,
                by_row="compat",
                **kwargs,
            )
        else:
            assert isinstance(data, np.ndarray)

            def wrap_function(func):
                # [FUNCTION DOC]
                # HELPER: wraps a function so that a string result is converted into a zero-dimensional object array.
                # Works around a numpy issue where apply-along-axis mishandles string results. Returns the wrapper.
                # https://github.com/numpy/numpy/issues/8352
                def wrapper(*args, **kwargs):
                    # [FUNCTION DOC]
                    # HELPER: the wrapper that calls the wrapped function with all given arguments and converts a
                    # string result to an object dtype numpy array before returning it; other results are returned as is.
                    result = func(*args, **kwargs)
                    if isinstance(result, str):
                        result = np.array(result, dtype=object)
                    return result

                return wrapper

            return np.apply_along_axis(wrap_function(func), axis, data, *args, **kwargs)


class MockEngineDecorator:
    # [CLASS DOC]
    # HELPER: decorator stand-in whose user-defined-function protocol attribute points at the mock
    # execution engine, so passing it as the engine selects the mock. Holds no other state.
    __pandas_udf__ = MockExecutionEngine


@pytest.fixture(params=[None, MockEngineDecorator])
def engine(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over two engines: none (default engine) and the mock engine decorator.
    # Returns the current parameter so tests run both with and without a custom engine.
    return request.param
