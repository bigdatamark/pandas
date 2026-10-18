# [MODULE DOC]
# Tests for astype on object-dtype Index.
import pytest

from pandas import (
    Index,
    NaT,
)


def test_astype_invalid_nas_to_tdt64_raises():
    # [FUNCTION DOC]
    # Casting an object index holding numpy datetime NaT values to timedelta raises a
    # type error about invalid timedelta scalar, instead of silently converting
    # (GH#45722).
    # GH#45722 don't cast np.datetime64 NaTs to timedelta64 NaT
    idx = Index([NaT.asm8] * 2, dtype=object)

    msg = r"Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
    with pytest.raises(TypeError, match=msg):
        idx.astype("m8[ns]")
