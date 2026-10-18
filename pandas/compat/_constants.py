"""
_constants
======

Constants relevant for the Python implementation.
"""

# [MODULE DOC]
# HELPER: constants describing the running Python implementation and platform,
# computed once at import time. Defined: IS64 (64-bit interpreter), PY312,
# PY314 and PY315 (interpreter version at least that), PYPY (implementation is
# PyPy), WASM (running under WebAssembly: emscripten platform or wasm machine
# type), ISMUSL (the host triple mentions musl), REF_COUNT (expected reference
# count of an object in a chained setitem call; 2 on python 3.14 or later else
# 3), REF_COUNT_IDX (2, for chained indexing) and REF_COUNT_METHOD (1 on 3.14 or
# later else 2), used by the chained assignment warning logic, and
# CHAINED_WARNING_DISABLED (true on PyPy, where reference counts are not
# reliable). The public package re-exports the first seven plus the disabled
# flag. Import-time side effects: queries platform and sysconfig.
# KNOWN ISSUE: __all__ lists only seven of the constants; the reference count
# constants and the chained warning flag are not listed although the latter is
# re-exported by the compat package.
# KNOWN ISSUE: the existing docstring underline is shorter than its title text.
from __future__ import annotations

import platform
import sys
import sysconfig

IS64 = sys.maxsize > 2**32

PY312 = sys.version_info >= (3, 12)
PY314 = sys.version_info >= (3, 14)
PY315 = sys.version_info >= (3, 15)
PYPY = platform.python_implementation() == "PyPy"
WASM = (sys.platform == "emscripten") or (platform.machine() in ["wasm32", "wasm64"])
ISMUSL = "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or "")
# the refcount for self in a chained __setitem__/.(i)loc indexing/method call
REF_COUNT = 2 if PY314 else 3
REF_COUNT_IDX = 2
REF_COUNT_METHOD = 1 if PY314 else 2
CHAINED_WARNING_DISABLED = PYPY


__all__ = [
    "IS64",
    "ISMUSL",
    "PY312",
    "PY314",
    "PY315",
    "PYPY",
    "WASM",
]
