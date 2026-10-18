# [MODULE DOC]
# Legacy pyarrow extension types used to store pandas period and interval data in
# parquet and feather files, plus a safety patch for old pyarrow versions.
# Public contents: the period extension type, the interval extension type, an
# error-message template and the patch function. Requires pyarrow at import.
# Import-time side effects: registers a dummy instance of each extension type
# with pyarrow's global registry, and runs the patch function, which on pyarrow
# older than 14.0.1 (unless the external hotfix is installed) replaces the
# legacy pickle-based extension type with one that refuses to deserialize and
# sets a marker attribute on the pyarrow module.
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow

from pandas.compat import pa_version_under14p1

from pandas.core.dtypes.dtypes import (
    IntervalDtype,
    PeriodDtype,
)

from pandas.core.arrays.interval import VALID_CLOSED

if TYPE_CHECKING:
    from pandas._typing import IntervalClosedType


class ArrowPeriodType(pyarrow.ExtensionType):
    # [CLASS DOC]
    # Legacy pyarrow extension type representing pandas period data: 64-bit integer
    # storage plus a frequency string. Registered under the extension name for periods.
    # State: the frequency. Equality and hashing include the frequency; metadata is
    # serialized as a JSON string holding the frequency.
    def __init__(self, freq) -> None:
        # [FUNCTION DOC]
        # Constructor. Parameter freq is the period frequency (a JSON-serializable
        # value, typically a string). The frequency must be stored before the parent
        # constructor runs because that call serializes the metadata. Storage type is
        # 64-bit integer.
        # attributes need to be set first before calling
        # super init (as that calls serialize)
        self._freq = freq
        pyarrow.ExtensionType.__init__(self, pyarrow.int64(), "pandas.period")

    @property
    def freq(self):
        # [FUNCTION DOC]
        # Property: the stored period frequency.
        return self._freq

    def __arrow_ext_serialize__(self) -> bytes:
        # [FUNCTION DOC]
        # Serialization hook for pyarrow. Returns the frequency wrapped in a JSON
        # object, encoded as bytes.
        metadata = {"freq": self.freq}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowPeriodType:
        # [FUNCTION DOC]
        # Deserialization hook for pyarrow. Parameters: storage_type (unused) and
        # serialized (bytes). Decodes the JSON and returns a new period type with the
        # stored frequency.
        metadata = json.loads(serialized.decode())
        return ArrowPeriodType(metadata["freq"])

    def __eq__(self, other):
        # [FUNCTION DOC]
        # Equality. For other pyarrow extension types returns true when the type classes
        # and frequencies are equal; for anything else returns the not-implemented
        # sentinel.
        if isinstance(other, pyarrow.BaseExtensionType):
            return type(self) == type(other) and self.freq == other.freq
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        # [FUNCTION DOC]
        # Inequality: the negation of equality.
        return not self == other

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash based on the string form of the type and the frequency.
        return hash((str(self), self.freq))

    def to_pandas_dtype(self) -> PeriodDtype:
        # [FUNCTION DOC]
        # Returns the corresponding pandas period dtype with this frequency.
        return PeriodDtype(freq=self.freq)


# register the type with a dummy instance
_period_type = ArrowPeriodType("D")
pyarrow.register_extension_type(_period_type)


class ArrowIntervalType(pyarrow.ExtensionType):
    # [CLASS DOC]
    # Legacy pyarrow extension type representing pandas interval data: storage is a
    # struct of left and right fields of the subtype. Registered under the extension
    # name for intervals. State: the closed side and the subtype. Equality and hashing
    # include both; metadata is serialized as JSON with the subtype string and closed
    # side.
    def __init__(self, subtype, closed: IntervalClosedType) -> None:
        # [FUNCTION DOC]
        # Constructor. Parameters: subtype is a pyarrow type or something convertible to
        # one by alias lookup of its string form; closed is one of the valid closed
        # values, enforced by an assertion. Attributes are stored before the parent
        # constructor runs because it serializes the metadata. Storage type is a struct
        # with left and right fields of the subtype.
        # attributes need to be set first before calling
        # super init (as that calls serialize)
        assert closed in VALID_CLOSED
        self._closed: IntervalClosedType = closed
        if not isinstance(subtype, pyarrow.DataType):
            subtype = pyarrow.type_for_alias(str(subtype))
        self._subtype = subtype

        storage_type = pyarrow.struct([("left", subtype), ("right", subtype)])
        pyarrow.ExtensionType.__init__(self, storage_type, "pandas.interval")

    @property
    def subtype(self):
        # [FUNCTION DOC]
        # Property: the pyarrow subtype of the interval endpoints.
        return self._subtype

    @property
    def closed(self) -> IntervalClosedType:
        # [FUNCTION DOC]
        # Property: which side(s) of the interval are closed.
        return self._closed

    def __arrow_ext_serialize__(self) -> bytes:
        # [FUNCTION DOC]
        # Serialization hook for pyarrow. Returns a JSON object holding the subtype
        # string and the closed side, encoded as bytes.
        metadata = {"subtype": str(self.subtype), "closed": self.closed}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowIntervalType:
        # [FUNCTION DOC]
        # Deserialization hook for pyarrow. Parameters: storage_type (unused) and
        # serialized (bytes). Decodes the JSON, resolves the subtype by alias and
        # returns a new interval type.
        metadata = json.loads(serialized.decode())
        subtype = pyarrow.type_for_alias(metadata["subtype"])
        closed = metadata["closed"]
        return ArrowIntervalType(subtype, closed)

    def __eq__(self, other):
        # [FUNCTION DOC]
        # Equality. For other pyarrow extension types returns true when the type
        # classes, subtypes and closed sides are equal; for anything else returns the
        # not-implemented sentinel.
        if isinstance(other, pyarrow.BaseExtensionType):
            return (
                type(self) == type(other)
                and self.subtype == other.subtype
                and self.closed == other.closed
            )
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        # [FUNCTION DOC]
        # Inequality: the negation of equality.
        return not self == other

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash based on the string form of the type, the string form of the subtype and
        # the closed side.
        return hash((str(self), str(self.subtype), self.closed))

    def to_pandas_dtype(self) -> IntervalDtype:
        # [FUNCTION DOC]
        # Returns the corresponding pandas interval dtype built from the subtype's
        # pandas dtype and the closed side.
        return IntervalDtype(self.subtype.to_pandas_dtype(), self.closed)


# register the type with a dummy instance
_interval_type = ArrowIntervalType(pyarrow.int64(), "left")
pyarrow.register_extension_type(_interval_type)


_ERROR_MSG = """\
Disallowed deserialization of 'arrow.py_extension_type':
storage_type = {storage_type}
serialized = {serialized}
pickle disassembly:\n{pickle_disassembly}

Reading of untrusted Parquet or Feather files with a PyExtensionType column
allows arbitrary code execution.
If you trust this file, you can enable reading the extension type by one of:

- upgrading to pyarrow >= 14.0.1, and call `pa.PyExtensionType.set_auto_load(True)`
- install pyarrow-hotfix (`pip install pyarrow-hotfix`) and disable it by running
  `import pyarrow_hotfix; pyarrow_hotfix.uninstall()`

We strongly recommend updating your Parquet/Feather files to use extension types
derived from `pyarrow.ExtensionType` instead, and register this type explicitly.
"""


def patch_pyarrow() -> None:
    # [FUNCTION DOC]
    # Import-time safety patch for old pyarrow versions. Takes no arguments, returns
    # nothing. Does nothing if the pyarrow version is at least 14.0.1 (which has its own
    # protection) or if the external hotfix package already patched pyarrow. Otherwise
    # replaces the registered legacy pickle-based extension type with a forbidden type
    # whose deserialization raises a runtime error containing a disassembly of the
    # pickle and instructions, and sets a marker attribute on the pyarrow module. Side
    # effects: unregisters and registers a pyarrow extension type, sets a module
    # attribute.
    if not pa_version_under14p1:
        return

    # if https://github.com/pitrou/pyarrow-hotfix was installed and enabled
    if getattr(pyarrow, "_hotfix_installed", False):
        return

    class ForbiddenExtensionType(pyarrow.ExtensionType):
        # [CLASS DOC]
        # HELPER: nested extension type, created only inside the patch function, that
        # refuses to deserialize data written with the unsafe legacy pickle-based
        # extension type. Serialization produces empty bytes; deserialization always
        # raises.
        def __arrow_ext_serialize__(self) -> bytes:
            # [FUNCTION DOC]
            # Serialization hook: returns empty bytes.
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            # [FUNCTION DOC]
            # Deserialization hook. Parameters: storage_type and serialized (bytes).
            # Never returns: disassembles the pickled bytes to text and raises a runtime
            # error whose message (from the module-level template) names the storage
            # type, the serialized bytes and the disassembly, with instructions for
            # explicitly trusting the file.
            import io
            import pickletools

            out = io.StringIO()
            pickletools.dis(serialized, out)
            raise RuntimeError(
                _ERROR_MSG.format(
                    storage_type=storage_type,
                    serialized=serialized,
                    pickle_disassembly=out.getvalue(),
                )
            )

    pyarrow.unregister_extension_type("arrow.py_extension_type")
    pyarrow.register_extension_type(
        ForbiddenExtensionType(pyarrow.null(), "arrow.py_extension_type")
    )

    pyarrow._hotfix_installed = True


patch_pyarrow()
