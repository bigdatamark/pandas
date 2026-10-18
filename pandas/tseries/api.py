"""
Timeseries API
"""

# [MODULE DOC]
# Public time series API namespace. Pure re-export module with no logic of its
# own: it re-exports the datetime format guessing function from the compiled
# tslibs parsing module, the frequency inference function from the sibling
# frequencies module, and the whole offsets module. The public name list
# contains exactly these three names. Importing it imports those modules but
# has no other side effects.
from pandas._libs.tslibs.parsing import guess_datetime_format

from pandas.tseries import offsets
from pandas.tseries.frequencies import infer_freq

__all__ = ["guess_datetime_format", "infer_freq", "offsets"]
