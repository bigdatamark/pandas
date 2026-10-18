# [PACKAGE DOC]
# Text-file parsing package behind the delimited and fixed-width readers.
# Submodules: readers (public entry points and the TextFileReader class, plus
# option validation and engine selection), base_parser (shared parser base
# class and conversion helpers), c_parser_wrapper (adapter over the compiled
# C tokenizer engine), python_parser (pure-Python engine, including the
# fixed-width variant), and arrow_parser_wrapper (adapter over the pyarrow
# CSV reader).
# Re-exports and lists in the public names: TextFileReader, TextParser,
# read_csv, read_fwf and read_table, all taken from the readers submodule.
# Import-time side effects: importing the readers submodule, which pulls in
# the other engine modules and registers nothing else globally.
from pandas.io.parsers.readers import (
    TextFileReader,
    TextParser,
    read_csv,
    read_fwf,
    read_table,
)

__all__ = ["TextFileReader", "TextParser", "read_csv", "read_fwf", "read_table"]
