# [MODULE DOC]
# Tests for arithmetic on boolean data. Contains one test for divmod and floor division on bool dtype.

import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


def test_divmod_bool_raises(box_with_array):
    # [FUNCTION DOC]
    # Verifies that floor division and divmod on boolean data raise a not-implemented error, including
    # the reflected divmod with a Python bool on the left.
    # Parametrized by the box_with_array fixture (container type). The setup boxes a boolean series.
    # Expected: floor division always raises with the message about floordiv; divmod raises with the
    # floordiv message when the container is a DataFrame, otherwise with the divmod message.
    # GH#46043 // raises, so divmod should too
    ser = Series([True, False])
    obj = tm.box_expected(ser, box_with_array)

    msg = "operator 'floordiv' not implemented for bool dtypes"
    with pytest.raises(NotImplementedError, match=msg):
        obj // obj

    if box_with_array is DataFrame:
        msg = "operator 'floordiv' not implemented for bool dtypes"
    else:
        msg = "operator 'divmod' not implemented for bool dtypes"
    with pytest.raises(NotImplementedError, match=msg):
        divmod(obj, obj)

    # go through __rdivmod__
    with pytest.raises(NotImplementedError, match=msg):
        divmod(True, obj)
