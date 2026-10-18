# [MODULE DOC]
# Parsing of time-of-day strings into time objects.
# Public contents: to_time, which converts a scalar, list-like, index or series of
# strings (or time objects) into time objects. Private contents: the fixed list of
# accepted format strings and a helper that guesses a format from an array.
# Relationships: uses the library list-like check, the generic Series/Index
# abstract classes and the not-null check from the core dtypes package.
# No import-time side effects beyond defining the module-level format list.
from __future__ import annotations

from datetime import (
    datetime,
    time,
)
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.lib import is_list_like

from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import notna

if TYPE_CHECKING:
    from pandas._typing import DateTimeErrorChoices


def to_time(
    arg,
    format: str | None = None,
    infer_time_format: bool = False,
    errors: DateTimeErrorChoices = "raise",
):
    # [FUNCTION DOC]
    # Convert time strings to datetime.time objects.
    # Parameters: arg is a scalar string, a time object, None, a list, tuple, 1-d
    # array, Index or Series; format is an optional strptime format (default none,
    # meaning the fixed module-level formats are tried); infer_time_format (default
    # false) guesses one format from the first non-missing element when no format is
    # given; errors is either "raise" (default) or "coerce".
    # Raises a value error up front if errors is any other value. Raises a value error
    # when an element cannot be parsed and errors is "raise"; raises a type error for
    # list-like input with more than one dimension.
    # Returns: None for None input; the same object for a time input; for a Series a
    # new Series (same constructor, index and name) of time objects; for an Index or
    # other list-like a plain list of time objects; for any other scalar a single time
    # object. With "coerce", unparseable elements become None.
    # No input is mutated. Without a format, each element is first tried as an
    # ISO-format time and then against the fixed formats; the fixed format list is
    # copied per call and the first format that succeeds is moved to the front for
    # the remaining elements.
    # KNOWN ISSUE: the docstring says the return type is only a time object, but list
    # and Series inputs return lists and Series respectively.
    """
    Parse time strings to time objects using fixed strptime formats ("%H:%M",
    "%H%M", "%I:%M%p", "%I%M%p", "%H:%M:%S", "%H%M%S", "%I:%M:%S%p",
    "%I%M%S%p")

    Use infer_time_format if all the strings are in the same format to speed
    up conversion.

    Parameters
    ----------
    arg : string in time format, datetime.time, list, tuple, 1-d array,  Series
    format : str, default None
        Format used to convert arg into a time object.  If None, fixed formats
        are used.
    infer_time_format: bool, default False
        Infer the time format based on the first non-NaN element.  If all
        strings are in the same format, this will speed up conversion.
    errors : {'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception
        - If 'coerce', then invalid parsing will be set as None

    Returns
    -------
    datetime.time
    """
    if errors not in ("raise", "coerce"):
        raise ValueError("errors must be one of 'raise', or 'coerce'.")

    def _convert_listlike(arg, format):
        # [FUNCTION DOC]
        # HELPER: to_time. Converts a list-like of strings to a list of time objects.
        # Inputs: arg is a list, tuple or array-like (lists and tuples are turned into
        # an object array; other inputs must have at most one dimension, otherwise a
        # type error is raised); format is an optional strptime format, which is
        # guessed from the array when it is None and infer_time_format is set in the
        # enclosing scope.
        # Returns a list the same length as the input, with None for failed elements
        # when the enclosing errors setting is "coerce". Raises a value error for an
        # unparseable element when errors is "raise" (the message names the element
        # when a format was used, and the whole array when none was).
        if isinstance(arg, (list, tuple)):
            arg = np.array(arg, dtype="O")

        elif getattr(arg, "ndim", 1) > 1:
            raise TypeError(
                "arg must be a string, datetime, list, tuple, 1-d array, or Series"
            )

        arg = np.asarray(arg, dtype="O")

        if infer_time_format and format is None:
            format = _guess_time_format_for_array(arg)

        times: list[time | None] = []
        if format is not None:
            for element in arg:
                try:
                    times.append(datetime.strptime(element, format).time())
                except (ValueError, TypeError) as err:
                    if errors == "raise":
                        msg = (
                            f"Cannot convert {element} to a time with given "
                            f"format {format}"
                        )
                        raise ValueError(msg) from err
                    times.append(None)
        else:
            formats = _time_formats[:]
            format_found = False
            for element in arg:
                time_object = None
                try:
                    time_object = time.fromisoformat(element)
                except (ValueError, TypeError):
                    for time_format in formats:
                        try:
                            time_object = datetime.strptime(element, time_format).time()
                            if not format_found:
                                # Put the found format in front
                                fmt = formats.pop(formats.index(time_format))
                                formats.insert(0, fmt)
                                format_found = True
                            break
                        except (ValueError, TypeError):
                            continue

                if time_object is not None:
                    times.append(time_object)
                elif errors == "raise":
                    raise ValueError(f"Cannot convert arg {arg} to a time")
                else:
                    times.append(None)

        return times

    if arg is None:
        return arg
    elif isinstance(arg, time):
        return arg
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, format)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, format)
    elif is_list_like(arg):
        return _convert_listlike(arg, format)

    return _convert_listlike(np.array([arg]), format)[0]


# Fixed time formats for time parsing
_time_formats = [
    "%H:%M",
    "%H%M",
    "%I:%M%p",
    "%I%M%p",
    "%H:%M:%S",
    "%H%M%S",
    "%I:%M:%S%p",
    "%I%M%S%p",
]


def _guess_time_format_for_array(arr):
    # [FUNCTION DOC]
    # HELPER: to_time (format inference). Guesses a strptime format for an array of
    # time strings.
    # Input: arr, an object array of strings that may contain missing values.
    # Looks only at the first non-missing element and returns the first format in the
    # module-level format list that parses it; returns None if the array has no
    # non-missing element or no format parses it. Non-value-error failures (such as a
    # type error for non-string elements) are not caught and propagate.
    # Try to guess the format based on the first non-NaN element
    non_nan_elements = notna(arr).nonzero()[0]
    if len(non_nan_elements):
        element = arr[non_nan_elements[0]]
        for time_format in _time_formats:
            try:
                datetime.strptime(element, time_format)
                return time_format
            except ValueError:
                pass

    return None
