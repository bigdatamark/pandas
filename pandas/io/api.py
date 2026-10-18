"""
Data I/O API
"""

# [MODULE DOC]
# Aggregation module for the public I/O API: re-exports the reader functions,
# writer helpers and classes from the individual io submodules (clipboard,
# excel, feather, html, iceberg, json, orc, parquet, parsers, pickle, pytables,
# sas, spss, sql, stata, xml) so the top-level pandas namespace can import them
# from a single place. The all list names each re-exported object.
# Import-time side effects: importing all those submodules (and thus their
# own import-time effects); no logic of its own.

from pandas.io.clipboards import read_clipboard
from pandas.io.excel import (
    ExcelFile,
    ExcelWriter,
    read_excel,
)
from pandas.io.feather_format import read_feather
from pandas.io.html import read_html
from pandas.io.iceberg import read_iceberg
from pandas.io.json import read_json
from pandas.io.orc import read_orc
from pandas.io.parquet import read_parquet
from pandas.io.parsers import (
    read_csv,
    read_fwf,
    read_table,
)
from pandas.io.pickle import (
    read_pickle,
    to_pickle,
)
from pandas.io.pytables import (
    HDFStore,
    read_hdf,
)
from pandas.io.sas import read_sas
from pandas.io.spss import read_spss
from pandas.io.sql import (
    read_sql,
    read_sql_query,
    read_sql_table,
)
from pandas.io.stata import read_stata
from pandas.io.xml import read_xml

__all__ = [
    "ExcelFile",
    "ExcelWriter",
    "HDFStore",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_hdf",
    "read_html",
    "read_iceberg",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "to_pickle",
]
