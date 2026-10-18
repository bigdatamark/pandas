# [MODULE DOC]
# Minimal, test-local extension dtype and array for UUID values, backed by a numpy
# array of sixteen void bytes (128 bits, every pattern valid). Contains the dtype
# class, the array class, and two tests that the array can be constructed directly and
# placed in a Series. Only the essential extension API is implemented. No import-time
# side effects other than defining the module-level private numpy dtype.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    ClassVar,
    Self,
)
from uuid import UUID

import numpy as np

from pandas.core.dtypes.dtypes import ExtensionDtype

import pandas as pd
from pandas.core.arrays.base import ExtensionArray

if TYPE_CHECKING:
    import builtins
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from pandas._typing import (
        Dtype,
        ScalarIndexer,
    )


# 16 void bytes: 128 bit, every pattern valid, no funky behavior like 0 stripping.
_UuidNumpyDtype = np.dtype("V16")


class UuidDtype(ExtensionDtype):
    # [CLASS DOC]
    # Extension dtype for UUID scalars. Class attributes: name is the string uuid, type is
    # the standard UUID class, kind is taken from the backing void numpy dtype. Associates
    # itself with the array class via the construct-array-type hook.
    # ExtensionDtype essential API (3 class attrs and methods)

    name: ClassVar[str] = "uuid"
    type: ClassVar[builtins.type[UUID]] = UUID

    @classmethod
    def construct_array_type(cls) -> builtins.type[UuidExtensionArray]:
        # [FUNCTION DOC]
        # Returns the array class associated with this dtype (the UUID extension array class).
        return UuidExtensionArray

    # ExtensionDtype overrides
    kind: ClassVar[str] = _UuidNumpyDtype.kind


class UuidExtensionArray(ExtensionArray):
    # [CLASS DOC]
    # Extension array storing UUIDs as sixteen-byte void entries in a private numpy
    # array (the data attribute). Implements only item lookup, length, construction from
    # a sequence and the dtype class attribute; other base-class features are unsupported.
    # Implementation details and convenience

    _data: NDArray[np.void]

    def __init__(self, values: Iterable[UUID], *, copy: bool = False) -> None:
        # [FUNCTION DOC]
        # Builds the array from an iterable of UUID objects by taking each one's bytes and
        # storing them as a numpy array of the void dtype. The keyword-only copy flag
        # (default false) is accepted but ignored.
        self._data = np.array([x.bytes for x in values], dtype=_UuidNumpyDtype)

    # Parts of ExtensionArray's essential API required for tests:

    dtype: ClassVar[UuidDtype] = UuidDtype()

    @classmethod
    def _from_sequence(
        cls,
        scalars: Iterable[UUID],
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # Alternate constructor from an iterable of UUID scalars. The dtype argument (default
        # none) and keyword copy flag are accepted; dtype is replaced with a UUID dtype when
        # none but is not otherwise used. Returns a new array built from the scalars.
        if dtype is None:
            dtype = UuidDtype()
        return cls(scalars, copy=copy)

    def __getitem__(self, index: ScalarIndexer) -> UUID:  # type: ignore[override]
        # [FUNCTION DOC]
        # Scalar lookup only. Asserts the index is an integer (python or numpy), otherwise an
        # assertion error is raised, and returns the UUID built from the bytes at that position.
        assert isinstance(index, int | np.integer)
        return UUID(bytes=self._data[index].tobytes())

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of stored entries.
        return len(self._data)


def test_construct() -> None:
    # [FUNCTION DOC]
    # Verifies the array can be built from a list of valid UUIDs: the first item has an
    # integer value of zero and the second equals the random UUID that was supplied.
    """Tests that we can construct UuidExtensionArray from a list of valid values."""
    from uuid import uuid4

    a = UuidExtensionArray([UUID(int=0), u := uuid4()])
    assert a[0].int == 0
    assert a[1] == u


def test_series() -> None:
    # [FUNCTION DOC]
    # Verifies a Series can be constructed with the UUID extension dtype (void-backed) and
    # that the scalar string form appears in the Series string representation.
    """Tests that Series accepts (unstructured) void ExtensionDtypes."""
    from uuid import uuid4

    s = pd.Series([u := uuid4()], dtype=UuidDtype(), name="s")
    assert str(u) in str(s)
