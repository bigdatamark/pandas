"""
This is a pseudo-public API for downstream libraries.  We ask that downstream
authors

1) Try to avoid using internals directly altogether, and failing that,
2) Use only functions exposed here (or in core.internals)

"""
# [MODULE DOC]
# Pseudo-public API of the internals for downstream libraries (for example
# pyarrow, which uses the block factory). Contents: a private block factory
# that normalizes shape and datetime-like classes, a private tz-aware datetime
# block subclass kept for backward compatibility, the deprecated public block
# factory, and a private plus a deprecated public function for inferring block
# dimensionality. Builds on the blocks module and the block placement type.
# No import-time side effects.

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs.internals import BlockPlacement
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)

from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)
from pandas.core.construction import extract_array
from pandas.core.internals.blocks import (
    DatetimeLikeBlock,
    check_ndim,
    ensure_block_shape,
    extract_pandas_array,
    get_block_type,
    maybe_coerce_values,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Dtype,
    )

    from pandas.core.internals.blocks import Block


def _make_block(values: ArrayLike, placement: np.ndarray) -> Block:
    # [FUNCTION DOC]
    # HELPER: internal analogue of the new-block constructors used inside pandas.
    # Parameters: values (numpy array already shaped as columns by rows, or an
    # extension array that may be one-dimensional), placement (integer array of
    # positions). Returns a block of the class chosen for the values dtype.
    # Wraps placement in a block placement; for extension dtypes that support
    # two dimensions, and for datetime and timedelta arrays, reshapes values to
    # two dimensions; then converts datetime-like numpy values to their array
    # classes. No other validation is done.
    """
    This is an analogue to blocks.new_block(_2d) that ensures:
    1) correct dimension for EAs that support 2D (`ensure_block_shape`), and
    2) correct EA class for datetime64/timedelta64 (`maybe_coerce_values`).

    The input `values` is assumed to be either numpy array or ExtensionArray:
    - In case of a numpy array, it is assumed to already be in the expected
      shape for Blocks (2D, (cols, rows)).
    - In case of an ExtensionArray the input can be 1D, also for EAs that are
      internally stored as 2D.

    For the rest no preprocessing or validation is done, except for those dtypes
    that are internally stored as EAs but have an exact numpy equivalent (and at
    the moment use that numpy dtype), i.e. datetime64/timedelta64.
    """
    dtype = values.dtype
    klass = get_block_type(dtype)
    placement_obj = BlockPlacement(placement)

    if (isinstance(dtype, ExtensionDtype) and dtype._supports_2d) or isinstance(
        values, (DatetimeArray, TimedeltaArray)
    ):
        values = ensure_block_shape(values, ndim=2)

    values = maybe_coerce_values(values)
    return klass(values, ndim=2, placement=placement_obj)


class _DatetimeTZBlock(DatetimeLikeBlock):
    # [CLASS DOC]
    # HELPER: datetime-with-timezone block kept only so legacy callers can still
    # request it by class. Subclass of the datetime-like block; holds a datetime
    # array; defines no slots or behavior of its own.
    """implement a datetime64 block with a tz attribute"""

    values: DatetimeArray

    __slots__ = ()


def make_block(
    values, placement, klass=None, ndim=None, dtype: Dtype | None = None
) -> Block:
    # [FUNCTION DOC]
    # Deprecated downstream-facing block factory.
    # Parameters: values (array-like), placement (positions or a block placement),
    # klass (optional block class, default None meaning infer from dtype), ndim
    # (optional int, inferred if None), dtype (optional dtype, default None).
    # Always emits the pandas 4 deprecation warning at stack level 2. Converts
    # dtype via the dtype lookup, unwraps pandas-numpy arrays, and chooses the
    # block class: the extension block is replaced by inference for period dtype
    # data; with no class given the class comes from the dtype; if the legacy
    # tz-aware class is requested for values lacking a tz dtype, values are wrapped
    # in a datetime array with the given dtype. Wraps placement, infers ndim,
    # reshapes period and tz-aware values to the block shape, checks ndim against
    # placement, coerces values and returns a block built from them.
    """
    This is a pseudo-public analogue to blocks.new_block.

    We ask that downstream libraries use this rather than any fully-internal
    APIs, including but not limited to:

    - core.internals.blocks.make_block
    - Block.make_block
    - Block.make_block_same_class
    - Block.__init__
    """
    warnings.warn(
        # GH#56815
        "make_block is deprecated and will be removed in a future version. "
        "Use pd.api.internals.create_dataframe_from_blocks or "
        "(recommended) higher-level public APIs instead.",
        Pandas4Warning,
        stacklevel=2,
    )

    if dtype is not None:
        dtype = pandas_dtype(dtype)

    values, dtype = extract_pandas_array(values, dtype, ndim)

    from pandas.core.internals.blocks import ExtensionBlock

    if klass is ExtensionBlock and isinstance(values.dtype, PeriodDtype):
        # GH-44681 changed PeriodArray to be stored in the 2D
        # NDArrayBackedExtensionBlock instead of ExtensionBlock
        # -> still allow ExtensionBlock to be passed in this case for back compat
        klass = None

    if klass is None:
        dtype = dtype or values.dtype
        klass = get_block_type(dtype)

    elif klass is _DatetimeTZBlock and not isinstance(values.dtype, DatetimeTZDtype):
        # pyarrow calls get here (pyarrow<15)
        values = DatetimeArray._simple_new(
            # error: Argument "dtype" to "_simple_new" of "DatetimeArray" has
            # incompatible type "Union[ExtensionDtype, dtype[Any], None]";
            # expected "Union[dtype[datetime64], DatetimeTZDtype]"
            values,
            dtype=dtype,  # type: ignore[arg-type]
        )

    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)

    ndim = _maybe_infer_ndim(values, placement, ndim)
    if isinstance(values.dtype, (PeriodDtype, DatetimeTZDtype)):
        # GH#41168 ensure we can pass 1D dt64tz values
        # More generally, any EA dtype that isn't is_1d_only_ea_dtype
        values = extract_array(values, extract_numpy=True)
        values = ensure_block_shape(values, ndim)

    check_ndim(values, placement, ndim)
    values = maybe_coerce_values(values)
    return klass(values, ndim=ndim, placement=placement)


def _maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> int:
    # [FUNCTION DOC]
    # HELPER: infer dimensionality for block creation. Parameters: values, placement,
    # ndim (optional). If ndim is given it is returned unchanged. Otherwise, for
    # extension dtypes returns 1 when the placement has other than one item and 2
    # when it has one; for numpy dtypes returns the values dimensionality.
    """
    If `ndim` is not provided, infer it from placement and values.
    """
    if ndim is None:
        # GH#38134 Block constructor now assumes ndim is not None
        if not isinstance(values.dtype, np.dtype):
            if len(placement) != 1:
                ndim = 1
            else:
                ndim = 2
        else:
            ndim = values.ndim
    return ndim


def maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> int:
    # [FUNCTION DOC]
    # Deprecated public wrapper: emits the pandas 4 deprecation warning at stack
    # level 2 and returns the result of the private inference helper for the same
    # arguments (values, placement, ndim).
    """
    If `ndim` is not provided, infer it from placement and values.
    """
    warnings.warn(
        "maybe_infer_ndim is deprecated and will be removed in a future version.",
        Pandas4Warning,
        stacklevel=2,
    )
    return _maybe_infer_ndim(values, placement, ndim)
