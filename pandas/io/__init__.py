# ruff: noqa: TC004
# [PACKAGE DOC]
# The pandas input/output package: readers and writers for the many file and
# storage formats (csv, excel, json, parquet, sql, hdf, stata, etc.), each in
# its own submodule, plus shared helpers (common, formats, parsers, clipboard).
# This __init__ exports nothing at runtime. Only for static type checking it
# imports the formats, json and stata submodules and declares them as the
# public module names. The user-facing functions are collected in the api
# submodule and re-exported at the top-level pandas namespace.
# No import-time side effects.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import modules that have public classes/functions
    from pandas.io import (
        formats,
        json,
        stata,
    )

    # mark only those modules as public
    __all__ = ["formats", "json", "stata"]
