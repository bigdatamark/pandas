# [PACKAGE DOC]
# Package for reading SAS files (XPORT transport files and sas7bdat binary
# datasets) into data frames.
# Submodules: sasreader (the public read function and the abstract reader base
# class), sas7bdat (reader for the sas7bdat binary format), sas_xport (reader
# for the XPORT transport format), sas_constants (numeric and byte constants
# describing the sas7bdat layout).
# Re-exports the read_sas function and lists it as the only public name.
# Import-time side effect: importing this package imports the sasreader module;
# the format-specific reader modules are imported lazily when a file is read.
from pandas.io.sas.sasreader import read_sas

__all__ = ["read_sas"]
