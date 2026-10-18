# [PACKAGE DOC]
# Package implementing pandas window operations (rolling, expanding and exponentially
# weighted windows) for Series and DataFrame.
# Submodules: rolling (fixed, variable, calendar-aware and weighted rolling windows
# plus their groupby variants), expanding, ewm (exponentially weighted windows),
# common (shared helpers), doc (shared docstring templates), online (online
# exponentially weighted computation), numba_ (numba kernels) and indexers (window
# bound computation).
# Re-exports (listed in the public name list): Expanding, ExpandingGroupby,
# ExponentialMovingWindow, ExponentialMovingWindowGroupby, Rolling, RollingGroupby
# and Window.
# Import-time side effects: importing this package imports the ewm, expanding and
# rolling modules only.

from pandas.core.window.ewm import (
    ExponentialMovingWindow,
    ExponentialMovingWindowGroupby,
)
from pandas.core.window.expanding import (
    Expanding,
    ExpandingGroupby,
)
from pandas.core.window.rolling import (
    Rolling,
    RollingGroupby,
    Window,
)

__all__ = [
    "Expanding",
    "ExpandingGroupby",
    "ExponentialMovingWindow",
    "ExponentialMovingWindowGroupby",
    "Rolling",
    "RollingGroupby",
    "Window",
]
