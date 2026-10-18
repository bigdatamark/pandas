# [MODULE DOC]
# Concrete buffer implementations for the dataframe interchange protocol.
# Public contents: a buffer class wrapping a numpy array and a buffer class
# wrapping a pyarrow buffer; both subclass the abstract buffer defined in the
# protocol module and are used by the column implementation to expose raw
# memory (data, validity mask, offsets). No import-time side effects; numpy and
# pyarrow are imported only for type checking.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    DlpackDeviceType,
)

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa


class PandasBuffer(Buffer):
    # [CLASS DOC]
    # Interchange buffer backed by a numpy array. Subclass of the protocol buffer.
    # State: a single private reference to a contiguous one-dimensional numpy
    # array holding the data. Invariant: the stored array is contiguous (a copy is
    # made at construction if needed and permitted). Exposes size in bytes, start
    # address, a DLPack export and the (always CPU) device.
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(self, x: np.ndarray, allow_copy: bool = True) -> None:
        # [FUNCTION DOC]
        # Wrap a numpy array as a buffer.
        # Parameters: x is the numpy array to wrap; allow_copy (default true) says
        # whether a copy may be made when the array is strided.
        # If the array has a nonzero first stride that differs from its item size
        # (i.e. it is not contiguous 1D), the protocol cannot represent it, so a
        # contiguous copy is stored when copying is allowed; otherwise a runtime error
        # is raised stating that the export cannot be zero-copy. Otherwise the array
        # itself is stored without copying. Returns nothing.
        """
        Handle only regular columns (= numpy arrays) for now.
        """
        if x.strides[0] and not x.strides == (x.dtype.itemsize,):
            # The protocol does not support strided buffers, so a copy is
            # necessary. If that's not allowed, we need to raise an exception.
            if allow_copy:
                x = x.copy()
            else:
                raise RuntimeError(
                    "Exports cannot be zero-copy in the case of a non-contiguous buffer"
                )

        # Store the numpy array in which the data resides as a private
        # attribute, so we can use it to retrieve the public attributes
        self._x = x

    @property
    def bufsize(self) -> int:
        # [FUNCTION DOC]
        # Property: buffer size in bytes, computed as element count times item size
        # of the wrapped array.
        """
        Buffer size in bytes.
        """
        return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        # [FUNCTION DOC]
        # Property: start address of the wrapped array's memory as an integer,
        # read from the array interface data entry.
        """
        Pointer to start of the buffer as an integer.
        """
        return self._x.__array_interface__["data"][0]

    def __dlpack__(self) -> Any:
        # [FUNCTION DOC]
        # Return the DLPack capsule of the wrapped numpy array by delegating to the
        # array's own DLPack export. No arguments.
        """
        Represent this structure as DLPack interface.
        """
        return self._x.__dlpack__()

    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        # [FUNCTION DOC]
        # Return a pair of (device type, device id) for the data location. Always the
        # CPU device type with a device id of None.
        """
        Device type and device ID for where the data in the buffer resides.
        """
        return (DlpackDeviceType.CPU, None)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return a string such as the class name followed by a mapping of buffer size,
        # pointer and device name (taken from the DLPack device type).
        return (
            "PandasBuffer("
            + str(
                {
                    "bufsize": self.bufsize,
                    "ptr": self.ptr,
                    "device": self.__dlpack_device__()[0].name,
                }
            )
            + ")"
        )


class PandasBufferPyarrow(Buffer):
    # [CLASS DOC]
    # Interchange buffer backed by a pyarrow buffer. Subclass of the protocol
    # buffer. State: the pyarrow buffer and a stored length. Invariant: the data
    # is contiguous. Always reports the CPU device; DLPack export is unsupported.
    # KNOWN ISSUE: the length given at construction is stored but never read by any method of this class.
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(
        self,
        buffer: pa.Buffer,
        *,
        length: int,
    ) -> None:
        # [FUNCTION DOC]
        # Wrap a pyarrow buffer. Parameters: buffer is the pyarrow buffer; length
        # (keyword-only, required) is an integer stored on the instance for later use.
        # Performs no validation or copying and returns nothing.
        """
        Handle pyarrow chunked arrays.
        """
        self._buffer = buffer
        self._length = length

    @property
    def bufsize(self) -> int:
        # [FUNCTION DOC]
        # Property: buffer size in bytes as reported by the wrapped pyarrow buffer.
        """
        Buffer size in bytes.
        """
        return self._buffer.size

    @property
    def ptr(self) -> int:
        # [FUNCTION DOC]
        # Property: start address of the wrapped pyarrow buffer as an integer.
        """
        Pointer to start of the buffer as an integer.
        """
        return self._buffer.address

    def __dlpack__(self) -> Any:
        # [FUNCTION DOC]
        # Always raises a not-implemented error; DLPack export is not supported for
        # pyarrow-backed buffers.
        """
        Represent this structure as DLPack interface.
        """
        raise NotImplementedError

    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        # [FUNCTION DOC]
        # Return a pair of (device type, device id): always the CPU device type and a
        # device id of None.
        """
        Device type and device ID for where the data in the buffer resides.
        """
        return (DlpackDeviceType.CPU, None)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return a string containing the class label for pyarrow buffers followed by a
        # mapping of buffer size, pointer and a fixed CPU device label.
        return (
            "PandasBuffer[pyarrow]("
            + str(
                {
                    "bufsize": self.bufsize,
                    "ptr": self.ptr,
                    "device": "CPU",
                }
            )
            + ")"
        )
