# [MODULE DOC]
# Benchmarks for reading and writing a mixed frame with the pickle interface, for
# time and peak memory. Public contents: the benchmark class Pickle, derived from
# the shared file-benchmark base class (the file __test__.pkl in the working
# directory is removed after each run). Final line re-exports the common asv setup
# hook.
import numpy as np

from pandas import (
    DataFrame,
    Index,
    date_range,
    read_pickle,
)

from ..pandas_vb_common import BaseIO


class Pickle(BaseIO):
    # [CLASS DOC]
    # Benchmark of pickling a frame to a file and loading it back. No parameters.
    # State: fname and df.
    def setup(self):
        # [FUNCTION DOC]
        # Sets the file name __test__.pkl, builds a frame of 100000 rows (hourly
        # index from 2000-01-01, five random float columns, and an object column of
        # distinct strings) as df, and writes it to the file (side effect: creates
        # the file) so the read benchmarks have input.
        self.fname = "__test__.pkl"
        N = 100000
        C = 5
        self.df = DataFrame(
            np.random.randn(N, C),
            columns=[f"float{i}" for i in range(C)],
            index=date_range("20000101", periods=N, freq="h"),
        )
        self.df["object"] = Index([f"i-{i}" for i in range(N)], dtype=object)
        self.df.to_pickle(self.fname)

    def time_read_pickle(self):
        # [FUNCTION DOC]
        # Times loading the frame from the file.
        read_pickle(self.fname)

    def time_write_pickle(self):
        # [FUNCTION DOC]
        # Times writing the frame to the file (overwrites it).
        self.df.to_pickle(self.fname)

    def peakmem_read_pickle(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of loading the frame from the file.
        read_pickle(self.fname)

    def peakmem_write_pickle(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of writing the frame to the file.
        self.df.to_pickle(self.fname)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
