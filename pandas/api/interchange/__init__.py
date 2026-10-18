"""
Public API for DataFrame interchange protocol.
"""

# [PACKAGE DOC]
# Public package for the dataframe interchange protocol (exchanging tabular
# data with other libraries). Re-exports (all listed in __all__): DataFrame (the
# abstract protocol class from the protocol definition module, not the pandas
# frame) and from_dataframe (builds a pandas frame from any object implementing
# the protocol). Defines nothing itself; no import-time side effects beyond the
# imports.
from pandas.core.interchange.dataframe_protocol import DataFrame
from pandas.core.interchange.from_dataframe import from_dataframe

__all__ = ["DataFrame", "from_dataframe"]
