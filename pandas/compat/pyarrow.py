"""support pyarrow compatibility across versions"""

# [MODULE DOC]
# HELPER: pyarrow availability and version flags, plus one safe wrapper.
# Public contents: PYARROW_MIN_VERSION (13.0.0); boolean flags
# pa_version_under14p0, 14p1, 15p0, 16p0 through 23p0 (except 24), 25p0 and 26p0
# (true if the installed pyarrow, ignoring pre-release and local parts, is older
# than that version; all true when pyarrow is missing); PYARROW_INSTALLED
# (pyarrow can be imported); HAS_PYARROW (installed and at least the minimum
# version); and _safe_fill_null. The compat package re-exports most of these.
# Import-time side effects: tries to import pyarrow and computes the flags.
# KNOWN ISSUE: the version-under flag for 15.0.0 is defined here but not
# re-exported by the compat package, and there is no flag for 24.0.0.

from __future__ import annotations

import sys
from typing import Any

from pandas.util.version import Version

PYARROW_MIN_VERSION = "13.0.0"
try:
    import pyarrow as pa

    _palv = Version(Version(pa.__version__).base_version)
    pa_version_under14p0 = _palv < Version("14.0.0")
    pa_version_under14p1 = _palv < Version("14.0.1")
    pa_version_under15p0 = _palv < Version("15.0.0")
    pa_version_under16p0 = _palv < Version("16.0.0")
    pa_version_under17p0 = _palv < Version("17.0.0")
    pa_version_under18p0 = _palv < Version("18.0.0")
    pa_version_under19p0 = _palv < Version("19.0.0")
    pa_version_under20p0 = _palv < Version("20.0.0")
    pa_version_under21p0 = _palv < Version("21.0.0")
    pa_version_under22p0 = _palv < Version("22.0.0")
    pa_version_under23p0 = _palv < Version("23.0.0")
    pa_version_under25p0 = _palv < Version("25.0.0")
    pa_version_under26p0 = _palv < Version("26.0.0")
    PYARROW_INSTALLED = True
    HAS_PYARROW = _palv >= Version(PYARROW_MIN_VERSION)
except ImportError:
    pa_version_under14p0 = True
    pa_version_under14p1 = True
    pa_version_under15p0 = True
    pa_version_under16p0 = True
    pa_version_under17p0 = True
    pa_version_under18p0 = True
    pa_version_under19p0 = True
    pa_version_under20p0 = True
    pa_version_under21p0 = True
    pa_version_under22p0 = True
    pa_version_under23p0 = True
    pa_version_under25p0 = True
    pa_version_under26p0 = True
    PYARROW_INSTALLED = False
    HAS_PYARROW = False


def _safe_fill_null(
    arr: pa.Array | pa.ChunkedArray, fill_value: Any
) -> pa.Array | pa.ChunkedArray:
    # [FUNCTION DOC]
    # HELPER: wrapper around the pyarrow null-filling compute function that works
    # around a bug in pyarrow 21.x on Windows. Parameters: arr (an array or
    # chunked array) and fill_value (a scalar, or an array to fill from).
    # The fallback is used only when pyarrow is usable, the platform is
    # windows/cygwin, and the version is at least 21 but under 22. Otherwise, or
    # if the fill value is itself an array, delegates to the standard function.
    # Fallback: converts the fill value to a scalar of the array's type; for
    # duration types, replaces nulls by first setting them to zero and then
    # selecting the fill scalar at the null positions (done per chunk for chunked
    # arrays); for other types, selects the fill scalar where the array is null
    # (per chunk for chunked arrays). Returns an array or chunked array of the
    # same kind as the input.
    """
    Safe wrapper for pyarrow.compute.fill_null with fallback for Windows + pyarrow 21.

    pyarrow 21.0.0 on Windows has a bug in fill_null that incorrectly fills null values.
    This function uses a fallback implementation for that specific case, otherwise uses
    the standard pyarrow.compute.fill_null.

    Parameters
    ----------
    arr : pyarrow.Array | pyarrow.ChunkedArray
        Input array with potential null values.
    fill_value : Any
        Value to fill nulls with.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Array with nulls filled with fill_value.
    """
    import pyarrow.compute as pc

    is_windows = sys.platform in ["win32", "cygwin"]
    use_fallback = (
        HAS_PYARROW and is_windows and not pa_version_under21p0 and pa_version_under22p0
    )
    if not use_fallback or isinstance(fill_value, (pa.Array, pa.ChunkedArray)):
        return pc.fill_null(arr, fill_value)

    fill_scalar = pa.scalar(fill_value, type=arr.type)

    if pa.types.is_duration(arr.type):

        def fill_null_duration(arr: pa.Array, fill_scalar: pa.Scalar) -> pa.Array:
            # [FUNCTION DOC]
            # HELPER: for the null-filling fallback with duration types.
            # Parameters: arr (a plain array) and fill_scalar (scalar of the same
            # duration type). Builds a mask of null positions, replaces the nulls
            # by a zero duration, then replaces the masked positions by the fill
            # scalar. Returns the new array. The extra zeroing step exists to
            # avoid the buggy built-in behavior for durations.
            mask = pc.is_null(arr)
            zero_duration = pa.scalar(0, type=arr.type)
            arr_zeroed = pc.if_else(mask, zero_duration, arr)
            return pc.if_else(mask, fill_scalar, arr_zeroed)

        if isinstance(arr, pa.ChunkedArray):
            return pa.chunked_array(
                [fill_null_duration(chunk, fill_scalar) for chunk in arr.chunks]
            )
        return fill_null_duration(arr, fill_scalar)

    if isinstance(arr, pa.ChunkedArray):
        return pa.chunked_array(
            [pc.if_else(pc.is_null(chunk), fill_scalar, chunk) for chunk in arr.chunks]
        )
    return pc.if_else(pc.is_null(arr), fill_scalar, arr)
