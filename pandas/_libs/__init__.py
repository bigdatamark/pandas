# [PACKAGE DOC]
# Package holding the compiled (Cython and C) extension modules that implement
# the performance-critical parts of pandas: algorithms, hashing, indexing
# engines, parsers, interval handling, the time series library (tslibs
# subpackage), window functions (window subpackage) and more. Most submodules
# are compiled extensions, so their source files are not python.
# Exports (listed in __all__): Interval, and from the time series subpackage
# NaT, NaTType, OutOfBoundsDatetime, Period, Timedelta, Timestamp and iNaT.
# Import-time side effects (order matters, per the comment in the file): the
# parser extension and the datetime extension are imported first, because the
# datetime extension's module initialization installs its C-level API capsule
# onto the pandas top-level module so that other extensions can use it. Only
# then are the interval and tslibs names imported.
__all__ = [
    "Interval",
    "NaT",
    "NaTType",
    "OutOfBoundsDatetime",
    "Period",
    "Timedelta",
    "Timestamp",
    "iNaT",
]


# Below imports needs to happen first to ensure pandas top level
# module gets monkeypatched with the pandas_datetime_CAPI
# see pandas_datetime_exec in pd_datetime.c
import pandas._libs.pandas_parser  # isort: skip # type: ignore[reportUnusedImport]
import pandas._libs.pandas_datetime  # noqa: F401 # isort: skip # type: ignore[reportUnusedImport]
from pandas._libs.interval import Interval
from pandas._libs.tslibs import (
    NaT,
    NaTType,
    OutOfBoundsDatetime,
    Period,
    Timedelta,
    Timestamp,
    iNaT,
)
