# [PACKAGE DOC]
# JSON input/output package of pandas.io. Its implementation lives in the
# private submodule _json (reading, writing, and the underlying ujson-based
# encoder/decoder wrappers) and the private submodule _table_schema (building
# Table Schema descriptions).
# Re-exports: read_json, to_json, ujson_dumps, ujson_loads (from _json) and
# build_table_schema (from _table_schema); all five names are listed in the
# module's public export list.
# Import-time side effects: none beyond importing those submodules.
from pandas.io.json._json import (
    read_json,
    to_json,
    ujson_dumps,
    ujson_loads,
)
from pandas.io.json._table_schema import build_table_schema

__all__ = [
    "build_table_schema",
    "read_json",
    "to_json",
    "ujson_dumps",
    "ujson_loads",
]
