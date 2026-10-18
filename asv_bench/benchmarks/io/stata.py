# [MODULE DOC]
# Benchmarks for reading and writing Stata files with each Stata date conversion
# code, plus a variant with extra columns containing missing values. Public
# contents: the benchmark classes Stata and StataMissing, both derived from the
# shared file-benchmark base class (the file __test__.dta in the working
# directory is removed after each run). Final line re-exports the common asv setup
# hook.
import numpy as np

from pandas import (
    DataFrame,
    Index,
    date_range,
    read_stata,
)

from ..pandas_vb_common import BaseIO


class Stata(BaseIO):
    # [CLASS DOC]
    # Benchmark of writing a mixed frame to a Stata file and reading it back,
    # converting the datetime index with one of the Stata date formats.
    # Parametrized by the date conversion code (tc, td, tm, tw, th, tq, ty). State:
    # fname, N, C, df and convert_dates (a mapping of the index to the code).
    params = ["tc", "td", "tm", "tw", "th", "tq", "ty"]
    param_names = ["convert_dates"]

    def setup(self, convert_dates):
        # [FUNCTION DOC]
        # Sets the file name __test__.dta, N to 100000 and C to 5. Builds a frame
        # with an hourly index from 2000-01-01, five random float columns, an
        # object column of distinct strings, and integer columns int8_, int16_,
        # int32_ drawn within each type's range but stopping 27 below the maximum
        # (to avoid values the Stata format reserves for missing codes), and a
        # float32_ column. Stores the conversion mapping for the index and writes the
        # file (side effect: creates it) so the read benchmark has input.
        self.fname = "__test__.dta"
        N = self.N = 100000
        C = self.C = 5
        self.df = DataFrame(
            np.random.randn(N, C),
            columns=[f"float{i}" for i in range(C)],
            index=date_range("20000101", periods=N, freq="h"),
        )
        self.df["object"] = Index([f"i-{i}" for i in range(self.N)], dtype=object)
        self.df["int8_"] = np.random.randint(
            np.iinfo(np.int8).min, np.iinfo(np.int8).max - 27, N
        )
        self.df["int16_"] = np.random.randint(
            np.iinfo(np.int16).min, np.iinfo(np.int16).max - 27, N
        )
        self.df["int32_"] = np.random.randint(
            np.iinfo(np.int32).min, np.iinfo(np.int32).max - 27, N
        )
        self.df["float32_"] = np.array(np.random.randn(N), dtype=np.float32)
        self.convert_dates = {"index": convert_dates}
        self.df.to_stata(self.fname, convert_dates=self.convert_dates)

    def time_read_stata(self, convert_dates):
        # [FUNCTION DOC]
        # Times reading the file. The parameter is unused here.
        read_stata(self.fname)

    def time_write_stata(self, convert_dates):
        # [FUNCTION DOC]
        # Times writing the frame to the file with the date conversion mapping
        # (overwrites the file). The parameter is unused here (the mapping from
        # setup is used).
        self.df.to_stata(self.fname, convert_dates=self.convert_dates)


class StataMissing(Stata):
    # [CLASS DOC]
    # Variant of the Stata benchmark whose frame also contains ten columns with
    # many missing values. Inherits parameters, timing methods and file handling
    # from the parent class.
    def setup(self, convert_dates):
        # [FUNCTION DOC]
        # Runs the parent setup, then adds ten columns missing_0 to missing_9 of
        # random normal numbers in which every negative value is replaced by a
        # missing marker, and rewrites the file with the same date conversion
        # mapping (overwriting the parent's file).
        super().setup(convert_dates)
        for i in range(10):
            missing_data = np.random.randn(self.N)
            missing_data[missing_data < 0] = np.nan
            self.df[f"missing_{i}"] = missing_data
        self.df.to_stata(self.fname, convert_dates=self.convert_dates)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
