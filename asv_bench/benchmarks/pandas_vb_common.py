# [MODULE DOC]
# Shared helper module for the benchmark suite (internal; not a benchmark itself).
# Public contents: the compiled-library handle lib (imported from the current or the
# legacy location); the testing module handle tm (current or legacy location);
# dtype collections numeric_dtypes (ten numpy integer and float types),
# datetime_dtypes (numpy datetime and timedelta base types), string_dtypes (the
# object type) and extension_dtypes (nullable integer, categorical, interval,
# timezone-aware datetime and period dtypes, or an empty list when the library is
# too old); the function setup, which benchmark modules import to seed the global
# random generator; and the base class BaseIO for file-based benchmarks. Import-time
# side effects: imports the library and numpy and probes for old/new module
# locations; if neither location of the compiled library module can be imported the
# name lib stays undefined.
from importlib import import_module
import os

import numpy as np

import pandas as pd

# Compatibility import for lib
for imp in ["pandas._libs.lib", "pandas.lib"]:
    try:
        lib = import_module(imp)
        break
    except (ImportError, TypeError, ValueError):
        pass

# Compatibility import for the testing module
try:
    import pandas._testing as tm
except ImportError:
    import pandas.util.testing as tm  # noqa: F401


numeric_dtypes = [
    np.int64,
    np.int32,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.int16,
    np.int8,
    np.uint16,
    np.uint8,
]
datetime_dtypes = [np.datetime64, np.timedelta64]
string_dtypes = [object]
try:
    extension_dtypes = [
        pd.Int8Dtype,
        pd.Int16Dtype,
        pd.Int32Dtype,
        pd.Int64Dtype,
        pd.UInt8Dtype,
        pd.UInt16Dtype,
        pd.UInt32Dtype,
        pd.UInt64Dtype,
        pd.CategoricalDtype,
        pd.IntervalDtype,
        pd.DatetimeTZDtype("ns", "UTC"),
        pd.PeriodDtype("D"),
    ]
except AttributeError:
    extension_dtypes = []


def setup(*args, **kwargs):
    # [FUNCTION DOC]
    # Per-benchmark setup hook that benchmark modules import under the name setup so
    # that asv runs it before each benchmark. Accepts and ignores any positional and
    # keyword arguments. Side effect: seeds numpy's global random generator with the
    # fixed value 1234 so that benchmark data are reproducible. Returns nothing.
    # This function just needs to be imported into each benchmark file to
    # set up the random seed before each function.
    # https://asv.readthedocs.io/en/latest/writing_benchmarks.html
    np.random.seed(1234)


class BaseIO:
    # [CLASS DOC]
    # Base class for benchmarks that write temporary files. Subclasses set the class
    # attribute fname (default none) to the path of the file they create; the
    # inherited teardown hook removes that file after each benchmark run. Key
    # methods: remove (safe file deletion) and teardown (asv hook).
    """
    Base class for IO benchmarks
    """

    fname = None

    def remove(self, f):
        # [FUNCTION DOC]
        # Deletes the file at path f. Input f is a path (string or path-like).
        # Returns nothing. Operating-system errors (for example a missing file, or
        # on Windows a file still in use) are silently swallowed. Side effect:
        # removes the file if possible.
        """Remove created files"""
        try:
            os.remove(f)
        except OSError:
            # On Windows, attempting to remove a file that is in use
            # causes an exception to be raised
            pass

    def teardown(self, *args, **kwargs):
        # [FUNCTION DOC]
        # asv teardown hook run after each benchmark. Accepts and ignores any
        # arguments (the benchmark parameters). Removes the file named by the
        # instance's fname attribute using the remove helper.
        self.remove(self.fname)
