"""
Public API classes that store intermediate results useful for type-hinting.
"""

# [PACKAGE DOC]
# Public package of classes that appear as intermediate results of pandas
# operations (so users can reference them in type hints). Re-exports (all in
# __all__): the NaT type, the no-default sentinel type, the missing-value
# marker type, the column Expression class, group-by classes for series and
# frames, the frozen list class, the resampler classes (generic, per-index-type
# group-by variants, and the time grouper), the window classes (rolling,
# expanding, exponentially weighted, fixed-window and their group-by variants),
# and the reader classes for JSON (line-based reading), SAS and Stata files.
# The Styler class is intentionally not included, per a TODO comment: importing
# it would require importing the templating library. Submodule aliases holds
# type aliases. No import-time side effects beyond the imports.
from pandas._libs import NaTType
from pandas._libs.lib import NoDefault
from pandas._libs.missing import NAType

from pandas.core.col import Expression
from pandas.core.groupby import (
    DataFrameGroupBy,
    SeriesGroupBy,
)
from pandas.core.indexes.frozen import FrozenList
from pandas.core.resample import (
    DatetimeIndexResamplerGroupby,
    PeriodIndexResamplerGroupby,
    Resampler,
    TimedeltaIndexResamplerGroupby,
    TimeGrouper,
)
from pandas.core.window import (
    Expanding,
    ExpandingGroupby,
    ExponentialMovingWindow,
    ExponentialMovingWindowGroupby,
    Rolling,
    RollingGroupby,
    Window,
)

# TODO: Can't import Styler without importing jinja2
# from pandas.io.formats.style import Styler
from pandas.io.json._json import JsonReader
from pandas.io.sas.sasreader import SASReader
from pandas.io.stata import StataReader

__all__ = [
    "DataFrameGroupBy",
    "DatetimeIndexResamplerGroupby",
    "Expanding",
    "ExpandingGroupby",
    "ExponentialMovingWindow",
    "ExponentialMovingWindowGroupby",
    "Expression",
    "FrozenList",
    "JsonReader",
    "NAType",
    "NaTType",
    "NoDefault",
    "PeriodIndexResamplerGroupby",
    "Resampler",
    "Rolling",
    "RollingGroupby",
    "SASReader",
    "SeriesGroupBy",
    "StataReader",
    "TimeGrouper",
    "TimedeltaIndexResamplerGroupby",
    "Window",
]
