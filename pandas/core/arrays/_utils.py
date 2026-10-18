# [MODULE DOC]
# Internal helper module for extension arrays. Its single content is a function
# that infers the numpy dtype and missing-value sentinel to use when an
# extension array is converted to a numpy array.
# Used by the masked and arrow-backed extension arrays when implementing the
# conversion to numpy. Depends on the dtype utilities in the core dtypes
# package, the missing-value singleton, and the nan-as-missing configuration
# switch. No import-time side effects.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._config import is_nan_na

from pandas._libs import lib
from pandas._libs.missing import NA
from pandas.errors import LossySetitemError

from pandas.core.dtypes.cast import np_can_hold_element
from pandas.core.dtypes.common import is_numeric_dtype

if TYPE_CHECKING:
    from pandas._typing import (
        npt,
    )

    from pandas.core.arrays.base import ExtensionArray


def to_numpy_dtype_inference(
    arr: ExtensionArray,
    dtype: npt.DTypeLike | None,
    na_value,
    hasna: bool,
) -> tuple[np.dtype | None, Any]:
    # [FUNCTION DOC]
    # HELPER: shared by the extension arrays' conversion-to-numpy methods.
    # Decides the resulting numpy dtype and the value used to represent missing
    # entries.
    # Parameters: arr is the extension array being converted (only its dtype is
    # inspected); dtype is the user-requested dtype or None for inference;
    # na_value is the requested missing-value sentinel, where the library's
    # "no default" marker means to infer it; hasna says whether the array
    # actually contains missing values.
    # Returns a pair of (numpy dtype or None, missing-value sentinel). The dtype
    # is None when nothing was requested and the array dtype is not numeric.
    # Behavior: when no dtype is requested and the array dtype is numeric, the
    # dtype is inferred. Without missing values it is the array's underlying
    # numpy dtype. With missing values, boolean arrays become object, integer
    # arrays become float64 and float arrays keep their numpy dtype. For the
    # non-boolean case with an unspecified sentinel, the sentinel is NaN when
    # NaN is treated as missing and otherwise the NA singleton.
    # KNOWN ISSUE: in the branch where NaN is not treated as missing, the local
    # dtype variable is reassigned to object but is never read afterwards, so
    # that assignment is dead and the result dtype stays numeric (it is only
    # widened later if the sentinel cannot be held).
    # If a dtype is requested it is converted to a numpy dtype. If the sentinel
    # is still unspecified: it is the dtype's own missing value when there is no
    # result dtype or no missing data; NaN for float results; a NaT of the
    # matching unit for datetime and timedelta results; otherwise the array
    # dtype's missing value.
    # Finally, if the dtype was inferred as numeric and there are missing
    # values, the sentinel is checked against the inferred dtype and, if it
    # cannot be held without loss, the result dtype falls back to object.
    # No side effects, no exceptions propagated (the lossy-setitem error is
    # caught internally).
    result_dtype: np.dtype | None
    inferred_numeric_dtype = False
    if dtype is None and is_numeric_dtype(arr.dtype):
        inferred_numeric_dtype = True
        if hasna:
            if arr.dtype.kind == "b":
                result_dtype = np.dtype(np.object_)
            else:
                if arr.dtype.kind in "iu":
                    result_dtype = np.dtype(np.float64)
                else:
                    result_dtype = arr.dtype.numpy_dtype  # type: ignore[attr-defined]
                if na_value is lib.no_default:
                    if not is_nan_na():
                        na_value = NA
                        dtype = np.dtype(object)
                    else:
                        na_value = np.nan
        else:
            result_dtype = arr.dtype.numpy_dtype  # type: ignore[attr-defined]
    elif dtype is not None:
        result_dtype = np.dtype(dtype)
    else:
        result_dtype = None

    if na_value is lib.no_default:
        if result_dtype is None or not hasna:
            na_value = arr.dtype.na_value
        elif result_dtype.kind == "f":
            na_value = np.nan
        elif result_dtype.kind == "M":
            unit = np.datetime_data(result_dtype)[0]  # type: ignore[arg-type]
            na_value = np.datetime64("NaT", unit)  # type: ignore[call-overload]
        elif result_dtype.kind == "m":
            unit = np.datetime_data(result_dtype)[0]  # type: ignore[arg-type]
            na_value = np.timedelta64("NaT", unit)  # type: ignore[call-overload]
        else:
            na_value = arr.dtype.na_value

    if inferred_numeric_dtype and hasna:
        try:
            np_can_hold_element(result_dtype, na_value)  # type: ignore[arg-type]
        except LossySetitemError:
            result_dtype = np.dtype(np.object_)
    return result_dtype, na_value
