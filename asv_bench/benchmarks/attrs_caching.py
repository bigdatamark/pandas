# [MODULE DOC]
# Benchmarks for cheap attribute access: getting and setting the row index of a
# frame, and getting the extension array of a series with or without the internal
# helper that extracts arrays. Public contents are the benchmark classes
# DataFrameAttributes and SeriesArrayAttribute. At import time it tries to import
# the internal extract-array helper from the core construction module; on failure
# (older library versions) the module-level name is set to none, which would make
# the extract benchmarks fail. The last line re-exports the common asv setup hook.
import numpy as np

import pandas as pd
from pandas import DataFrame

try:
    from pandas.core.construction import extract_array
except ImportError:
    extract_array = None


class DataFrameAttributes:
    # [CLASS DOC]
    # Benchmark of reading and assigning the index attribute of a small frame.
    # No parameters. State: df (10 by 6 random frame) and cur_index (its index).
    def setup(self):
        # [FUNCTION DOC]
        # Creates a ten by six random normal frame as df and keeps a reference to
        # its current index as cur_index.
        self.df = DataFrame(np.random.randn(10, 6))
        self.cur_index = self.df.index

    def time_get_index(self):
        # [FUNCTION DOC]
        # Times reading the index attribute of the frame.
        self.df.index

    def time_set_index(self):
        # [FUNCTION DOC]
        # Times assigning the saved index back to the frame (mutates the frame
        # but with an identical value).
        self.df.index = self.cur_index


class SeriesArrayAttribute:
    # [CLASS DOC]
    # Benchmark of retrieving the underlying array of a three-element series of
    # various dtypes, through the public array property and the internal
    # extract helper with and without conversion of numpy-backed arrays.
    # Parametrized by dtype kind (numeric, object, category, datetime64,
    # datetime64tz). State: series.
    params = [["numeric", "object", "category", "datetime64", "datetime64tz"]]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Creates a three-element series matching the dtype kind: integers,
        # object-dtype letters, categorical letters, daily timestamps from 2013,
        # or the same timestamps localized to UTC. An unrecognized kind leaves
        # series unset. Stores it as series.
        if dtype == "numeric":
            self.series = pd.Series([1, 2, 3])
        elif dtype == "object":
            self.series = pd.Series(["a", "b", "c"], dtype=object)
        elif dtype == "category":
            self.series = pd.Series(["a", "b", "c"], dtype="category")
        elif dtype == "datetime64":
            self.series = pd.Series(pd.date_range("2013", periods=3))
        elif dtype == "datetime64tz":
            self.series = pd.Series(pd.date_range("2013", periods=3, tz="UTC"))

    def time_array(self, dtype):
        # [FUNCTION DOC]
        # Times reading the public array property of the series. The dtype
        # parameter is unused.
        self.series.array

    def time_extract_array(self, dtype):
        # [FUNCTION DOC]
        # Times the internal extract helper on the series, leaving numpy-backed
        # data in its wrapper form. The dtype parameter is unused.
        extract_array(self.series)

    def time_extract_array_numpy(self, dtype):
        # [FUNCTION DOC]
        # Times the internal extract helper with numpy extraction enabled, which
        # unwraps numpy-backed arrays to plain numpy arrays. The dtype parameter
        # is unused.
        extract_array(self.series, extract_numpy=True)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
