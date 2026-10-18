"""
compat
======

Cross-compatible functions for different versions of Python.

Other items:
* platform checker
"""

# [PACKAGE DOC]
# Compatibility package: version/platform feature flags and small platform
# helper functions used across pandas. Submodules: _constants (interpreter and
# platform constants, including reference counts used by the chained
# assignment checks), _optional (optional dependency importing and minimum
# versions), numpy (numpy version flags and the numpy long type aliases),
# pyarrow (pyarrow version flags and a safe null-filling wrapper), and
# pickle_compat (unpickling of objects from older pandas versions).
# Re-exports (listed in __all__): the constants for 64-bit, musl, python 3.12,
# 3.14 and 3.15 or later, PyPy, WebAssembly, and the chained-warning-disabled
# flag; the numpy development flag; and the pyarrow installed/usable flags,
# minimum version and the many "version under X" booleans (note one of the
# private-to-pyarrow flags, for version 15, is not re-exported here).
# Defines: set_function_name and the platform predicates (little endian,
# windows, linux, mac, ARM, Power, riscv64) and the CI-environment check.
# Import-time side effects: importing the numpy and pyarrow submodules, which
# import pyarrow if available and raise an import error if numpy is too old.
from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING

from pandas.compat._constants import (
    CHAINED_WARNING_DISABLED,
    IS64,
    ISMUSL,
    PY312,
    PY314,
    PY315,
    PYPY,
    WASM,
)
from pandas.compat.numpy import is_numpy_dev
from pandas.compat.pyarrow import (
    HAS_PYARROW,
    PYARROW_INSTALLED,
    PYARROW_MIN_VERSION,
    pa_version_under14p0,
    pa_version_under14p1,
    pa_version_under16p0,
    pa_version_under17p0,
    pa_version_under18p0,
    pa_version_under19p0,
    pa_version_under20p0,
    pa_version_under21p0,
    pa_version_under22p0,
    pa_version_under23p0,
    pa_version_under25p0,
    pa_version_under26p0,
)

if TYPE_CHECKING:
    from pandas._typing import F


def set_function_name(f: F, name: str, cls: type) -> F:
    # [FUNCTION DOC]
    # Renames a function in place so it looks like a method of a class.
    # Parameters: f (function), name (new short name) and cls (the class it
    # will belong to). Sets the function's name to name, its qualified name to
    # the class name, a dot and name, and its module to the class's module.
    # Mutates and returns the same function object.
    """
    Bind the name/qualname attributes of the function.
    """
    f.__name__ = name
    f.__qualname__ = f"{cls.__name__}.{name}"
    f.__module__ = cls.__module__
    return f


def is_platform_little_endian() -> bool:
    # [FUNCTION DOC]
    # Returns true if the interpreter's native byte order is little endian.
    """
    Checking if the running platform is little endian.

    Returns
    -------
    bool
        True if the running platform is little endian.
    """
    return sys.byteorder == "little"


def is_platform_windows() -> bool:
    # [FUNCTION DOC]
    # Returns true if the platform identifier is windows or cygwin (so Cygwin
    # counts as windows, which the docstring does not mention).
    """
    Checking if the running platform is windows.

    Returns
    -------
    bool
        True if the running platform is windows.
    """
    return sys.platform in ["win32", "cygwin"]


def is_platform_linux() -> bool:
    # [FUNCTION DOC]
    # Returns true if the platform identifier is exactly linux.
    """
    Checking if the running platform is linux.

    Returns
    -------
    bool
        True if the running platform is linux.
    """
    return sys.platform == "linux"


def is_platform_mac() -> bool:
    # [FUNCTION DOC]
    # Returns true if the platform identifier is darwin (macOS).
    """
    Checking if the running platform is mac.

    Returns
    -------
    bool
        True if the running platform is mac.
    """
    return sys.platform == "darwin"


def is_platform_arm() -> bool:
    # [FUNCTION DOC]
    # Returns true if the machine type reported by the platform module is arm64
    # or aarch64, or starts with armv.
    """
    Checking if the running platform use ARM architecture.

    Returns
    -------
    bool
        True if the running platform uses ARM architecture.
    """
    return platform.machine() in ("arm64", "aarch64") or platform.machine().startswith(
        "armv"
    )


def is_platform_power() -> bool:
    # [FUNCTION DOC]
    # Returns true if the machine type is ppc64 or ppc64le (Power architecture).
    # KNOWN ISSUE: the existing docstring's Returns section says it returns true
    # if the platform uses ARM architecture (copy and paste error); the code
    # tests for Power.
    """
    Checking if the running platform use Power architecture.

    Returns
    -------
    bool
        True if the running platform uses ARM architecture.
    """
    return platform.machine() in ("ppc64", "ppc64le")


def is_platform_riscv64() -> bool:
    # [FUNCTION DOC]
    # Returns true if the machine type is exactly riscv64.
    """
    Checking if the running platform use riscv64 architecture.

    Returns
    -------
    bool
        True if the running platform uses riscv64 architecture.
    """
    return platform.machine() == "riscv64"


def is_ci_environment() -> bool:
    # [FUNCTION DOC]
    # Returns true if the environment variable named PANDAS_CI is set to the
    # exact text 1 (default when unset is 0).
    """
    Checking if running in a continuous integration environment by checking
    the PANDAS_CI environment variable.

    Returns
    -------
    bool
        True if the running in a continuous integration environment.
    """
    return os.environ.get("PANDAS_CI", "0") == "1"


__all__ = [
    "CHAINED_WARNING_DISABLED",
    "HAS_PYARROW",
    "IS64",
    "ISMUSL",
    "PY312",
    "PY314",
    "PY315",
    "PYARROW_INSTALLED",
    "PYARROW_MIN_VERSION",
    "PYPY",
    "WASM",
    "is_numpy_dev",
    "pa_version_under14p0",
    "pa_version_under14p1",
    "pa_version_under16p0",
    "pa_version_under17p0",
    "pa_version_under18p0",
    "pa_version_under19p0",
    "pa_version_under20p0",
    "pa_version_under21p0",
    "pa_version_under22p0",
    "pa_version_under23p0",
    "pa_version_under25p0",
    "pa_version_under26p0",
]
