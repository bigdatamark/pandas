# [MODULE DOC]
# Test-only extension array storing calendar dates as three parallel unsigned integer numpy
# arrays (year, month, day) instead of date objects. Public contents: DateDtype (registered with
# pandas at import time, so the string DateDtype can be resolved to it) and DateArray. The array
# supports only a minimal interface: construction, integer item access and assignment, length,
# copy, view, missing detection (the minimum representable date acts as the missing value), astype,
# byte size and repr. Import-time side effect: the dtype class is registered in the global
# extension dtype registry.
from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from pandas.core.dtypes.dtypes import register_extension_dtype

from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
)
from pandas.api.types import pandas_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        Dtype,
        PositionalIndexer,
    )


@register_extension_dtype
class DateDtype(ExtensionDtype):
    # [CLASS DOC]
    # Dtype for the date test array. Scalar type is the standard library date type; the name
    # is the string DateDtype; the missing-value marker is the minimum representable date. Registered
    # via the extension dtype registration decorator. Stateless.
    @property
    def type(self):
        # [FUNCTION DOC]
        # Property returning the scalar type, the standard library date class.
        return dt.date

    @property
    def name(self):
        # [FUNCTION DOC]
        # Property returning the dtype name, the string DateDtype.
        return "DateDtype"

    @classmethod
    def construct_from_string(cls, string: str):
        # [FUNCTION DOC]
        # Class method that builds the dtype from its name. Input: string. Raises a type error if the
        # input is not a string, and a type error if it is a string other than the class name. Returns
        # a new dtype instance when the string equals the class name.
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string == cls.__name__:
            return cls()
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    def construct_array_type(self):
        # [FUNCTION DOC]
        # Returns the array class for this dtype, DateArray.
        return DateArray

    @property
    def na_value(self):
        # [FUNCTION DOC]
        # Property returning the missing-value marker: the minimum representable date.
        return dt.date.min

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the dtype name as the representation.
        return self.name


class DateArray(ExtensionArray):
    # [CLASS DOC]
    # Extension array of dates. State: three numpy arrays of equal length, _year (16-bit unsigned),
    # _month and _day (8-bit unsigned). Subclasses the base extension array. Missing entries are stored
    # as the minimum date. Only integer scalar positions are supported for item access and
    # assignment; slices and arrays raise a not-implemented error.
    def __init__(
        self,
        dates: (
            dt.date
            | Sequence[dt.date]
            | tuple[np.ndarray, np.ndarray, np.ndarray]
            | np.ndarray
        ),
    ) -> None:
        # [FUNCTION DOC]
        # Builds the array from one of several input forms. A single date object creates a
        # one-element array. A list of dates creates arrays by reading year, month, day from each. A
        # tuple must contain exactly three numpy arrays of the same length (year, month, day), which are
        # cast to the storage types; a tuple of the wrong size raises a value error, a non-array member
        # raises a type error, and unequal lengths raise a value error listing the lengths. A numpy array
        # of 10-character unicode strings in the year-month-day form separated by dashes is split and
        # parsed into integers. Any other input raises a type error stating the unsupported type.
        # KNOWN ISSUE: for the single date input the day array is filled with the year instead of the
        # day, so a one-element array built from a date has the wrong day.
        # KNOWN ISSUE: the inline range comments on the month and day storage arrays are swapped:
        # the month array is annotated with the range 1 to 31 and the day array with the range 1
        # to 12.
        if isinstance(dates, dt.date):
            self._year = np.array([dates.year])
            self._month = np.array([dates.month])
            self._day = np.array([dates.year])
            return

        ldates = len(dates)
        if isinstance(dates, list):
            # pre-allocate the arrays since we know the size before hand
            self._year = np.zeros(ldates, dtype=np.uint16)  # 65535 (0, 9999)
            self._month = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 31)
            self._day = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 12)
            # populate them
            for i, (y, m, d) in enumerate(
                (date.year, date.month, date.day) for date in dates
            ):
                self._year[i] = y
                self._month[i] = m
                self._day[i] = d

        elif isinstance(dates, tuple):
            # only support triples
            if ldates != 3:
                raise ValueError("only triples are valid")
            # check if all elements have the same type
            if any(not isinstance(x, np.ndarray) for x in dates):
                raise TypeError("invalid type")
            ly, lm, ld = (len(cast(np.ndarray, d)) for d in dates)
            if not ly == lm == ld:
                raise ValueError(
                    f"tuple members must have the same length: {(ly, lm, ld)}"
                )
            self._year = dates[0].astype(np.uint16)
            self._month = dates[1].astype(np.uint8)
            self._day = dates[2].astype(np.uint8)

        elif isinstance(dates, np.ndarray) and dates.dtype == "U10":
            self._year = np.zeros(ldates, dtype=np.uint16)  # 65535 (0, 9999)
            self._month = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 31)
            self._day = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 12)

            # error: "object_" object is not iterable
            obj = np.char.split(dates, sep="-")
            for (i,), (y, m, d) in np.ndenumerate(obj):
                self._year[i] = int(y)
                self._month[i] = int(m)
                self._day[i] = int(d)

        else:
            raise TypeError(f"{type(dates)} is not supported")

    @property
    def dtype(self) -> ExtensionDtype:
        # [FUNCTION DOC]
        # Property returning a new DateDtype instance.
        return DateDtype()

    def astype(self, dtype, copy=True):
        # [FUNCTION DOC]
        # Conversion. Inputs: dtype (anything resolvable to a dtype), copy (default true). A date dtype
        # returns a copy (when copy is true) or self; any other dtype is converted via the numpy
        # conversion method with the minimum date as the missing value. Returns the result.
        dtype = pandas_dtype(dtype)

        if isinstance(dtype, DateDtype):
            data = self.copy() if copy else self
        else:
            data = self.to_numpy(dtype=dtype, copy=copy, na_value=dt.date.min)

        return data

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: total bytes of the three storage arrays.
        return self._year.nbytes + self._month.nbytes + self._day.nbytes

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the length of the year array (all three arrays have the same length).
        return len(self._year)  # all 3 arrays are enforced to have the same length

    def __getitem__(self, item: PositionalIndexer):
        # [FUNCTION DOC]
        # Integer item access: returns a date built from the year, month and day at that position.
        # Any non-integer indexer raises a not-implemented error saying only integers are supported.
        if isinstance(item, int):
            return dt.date(self._year[item], self._month[item], self._day[item])
        else:
            raise NotImplementedError("only ints are supported as indexes")

    def __setitem__(self, key: int | slice | np.ndarray, value: Any) -> None:
        # [FUNCTION DOC]
        # Assignment at one position. Inputs: key (must be an int), value (must be a date). Raises a
        # value error if the array is read-only, a not-implemented error for a non-integer key, and a
        # type error for a non-date value. Mutates the three storage arrays in place.
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        if not isinstance(key, int):
            raise NotImplementedError("only ints are supported as indexes")

        if not isinstance(value, dt.date):
            raise TypeError("you can only set datetime.date types")

        self._year[key] = value.year
        self._month[key] = value.month
        self._day[key] = value.day

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the class name followed by the list of (year, month, day) triples.
        return f"DateArray{list(zip(self._year, self._month, self._day, strict=True))}"

    def copy(self) -> DateArray:
        # [FUNCTION DOC]
        # Returns a new array with copies of the three storage arrays.
        return DateArray((self._year.copy(), self._month.copy(), self._day.copy()))

    def view(self, dtype: Dtype | None = None) -> DateArray:
        # [FUNCTION DOC]
        # Returns a new array over views of the three storage arrays (shared memory). The dtype
        # argument is accepted for interface compatibility and ignored.
        return DateArray((self._year.view(), self._month.view(), self._day.view()))

    def isna(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns a boolean numpy array, true where year, month and day all equal the minimum date.
        return np.logical_and(
            np.logical_and(
                self._year == dt.date.min.year, self._month == dt.date.min.month
            ),
            self._day == dt.date.min.day,
        )

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy=False):
        # [FUNCTION DOC]
        # HELPER: alternate constructor used by the extension array machinery. Inputs: scalars, dtype
        # (default none), copy (default false). A single date raises a type error. An existing date array
        # is returned converted with astype if a dtype is given, as a copy if copy is true, or as a full
        # slice of itself otherwise (which raises, since slicing is unsupported, for any array).
        # A numpy array is cast to 10-character unicode strings and parsed.
        # KNOWN ISSUE: any other input type falls off the end and returns nothing instead of raising.
        # KNOWN ISSUE: the date-array branch uses the full slice, which this array's item access does
        # not support (it raises a not-implemented error), so that branch fails for the plain case.
        if isinstance(scalars, dt.date):
            raise TypeError
        elif isinstance(scalars, DateArray):
            if dtype is not None:
                return scalars.astype(dtype, copy=copy)
            if copy:
                return scalars.copy()
            return scalars[:]
        elif isinstance(scalars, np.ndarray):
            scalars = scalars.astype("U10")  # 10 chars for yyyy-mm-dd
            return DateArray(scalars)
