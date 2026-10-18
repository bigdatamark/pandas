"""support numpy compatibility across versions"""

# [PACKAGE DOC]
# Package for numpy version compatibility. Submodule function holds the
# machinery validating numpy-style keyword arguments of the library's reduction
# and other methods. This file defines: _np_version (numpy version text),
# boolean flags for numpy at least 2.0, 2.3, 2.5 and 2.6 (dev) named
# np_version_gt2, np_version_gt2_3, np_version_gt2_5 and np_version_gt2_6,
# is_numpy_dev (whether the installed numpy version has a development
# component), and np_long and np_ulong (the C long scalar types: numpy's long
# types on numpy 2 if available, else the integer and unsigned integer types,
# silencing a possible future warning).
# Import-time side effects: raises an import error if numpy is older than the
# minimum supported version (1.26.0). __all__ lists only _np_version and
# is_numpy_dev.
# KNOWN ISSUE: the numpy 2.5 and 2.6 flags and the long type aliases are
# defined but not listed in __all__.

import warnings

import numpy as np

from pandas.util.version import Version

# numpy versioning
_np_version = np.__version__
_nlv = Version(_np_version)
np_version_gt2 = _nlv >= Version("2.0.0")
np_version_gt2_3 = _nlv >= Version("2.3.0")
np_version_gt2_5 = _nlv >= Version("2.5.0")
np_version_gt2_6 = _nlv >= Version("2.6.0.dev0")
is_numpy_dev = _nlv.dev is not None
_min_numpy_ver = "1.26.0"


if _nlv < Version(_min_numpy_ver):
    raise ImportError(
        f"Please upgrade numpy to >= {_min_numpy_ver} to use this pandas version.\n"
        f"Your numpy version is {_np_version}."
    )


np_long: type
np_ulong: type

if np_version_gt2:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                r".*In the future `np\.long` will be defined as.*",
                FutureWarning,
            )
            np_long = np.long
            np_ulong = np.ulong
    except AttributeError:
        np_long = np.int_
        np_ulong = np.uint
else:
    np_long = np.int_
    np_ulong = np.uint


__all__ = [
    "_np_version",
    "is_numpy_dev",
]
