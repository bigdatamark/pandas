# [MODULE DOC]
# Tests for the byte-swapping binary readers from the compiled byteswap extension used by the SAS reader.
# Public contents: two hypothesis-driven slow tests (integer and floating point) and one private helper that does the shared round trip check.
# Depends on the compiled readers for 16, 32 and 64 bit unsigned integers, float and double, plus the testing helpers module.
from hypothesis import (
    assume,
    example,
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas._libs.byteswap import (
    read_double_with_byteswap,
    read_float_with_byteswap,
    read_uint16_with_byteswap,
    read_uint32_with_byteswap,
    read_uint64_with_byteswap,
)

import pandas._testing as tm


@pytest.mark.slow
@given(read_offset=st.integers(0, 11), number=st.integers(min_value=0))
@example(number=2**16, read_offset=0)
@example(number=2**32, read_offset=0)
@example(number=2**64, read_offset=0)
@pytest.mark.parametrize("int_type", [np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("should_byteswap", [True, False])
def test_int_byteswap(read_offset, number, int_type, should_byteswap):
    # [FUNCTION DOC]
    # Property test: for unsigned 16, 32 and 64 bit integer types, a random offset in the range zero to eleven and a random non-negative number, with and without byte swapping.
    # Numbers that do not fit in the chosen type are discarded via an assumption. Explicit examples cover the first value that overflows each width.
    # Delegates to the shared helper; passes if the helper's assertions hold. Marked slow.
    assume(number < 2 ** (8 * int_type(0).itemsize))
    _test(number, int_type, read_offset, should_byteswap)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@given(read_offset=st.integers(0, 11), number=st.floats())
@pytest.mark.parametrize("float_type", [np.float32, np.float64])
@pytest.mark.parametrize("should_byteswap", [True, False])
def test_float_byteswap(read_offset, number, float_type, should_byteswap):
    # [FUNCTION DOC]
    # Property test: same as the integer test but for 32 and 64 bit floats with arbitrary float values (including nan and infinities), random read offset zero to eleven, and both byte swap settings.
    # Overflow runtime warnings from narrowing to float32 are ignored. Delegates to the shared helper. Marked slow.
    _test(number, float_type, read_offset, should_byteswap)


def _test(number, number_type, read_offset, should_byteswap):
    # [FUNCTION DOC]
    # HELPER: shared check for the byteswap tests.
    # Parameters: number is the value to encode, number_type the numpy scalar type, read_offset the byte position at which the value is embedded, should_byteswap whether the reader should swap bytes.
    # Builds twenty random bytes (fixed seed), overwrites the bytes at the offset with the raw bytes of the number, picks the matching reader function by type, reads the value back from the bytes object and asserts it equals the number byte-swapped when swapping is requested, else the number unchanged.
    # Returns nothing.
    number = number_type(number)
    data = np.random.default_rng(2).integers(0, 256, size=20, dtype="uint8")
    data[read_offset : read_offset + number.itemsize] = number[None].view("uint8")
    swap_func = {
        np.float32: read_float_with_byteswap,
        np.float64: read_double_with_byteswap,
        np.uint16: read_uint16_with_byteswap,
        np.uint32: read_uint32_with_byteswap,
        np.uint64: read_uint64_with_byteswap,
    }[type(number)]
    output_number = number_type(swap_func(bytes(data), read_offset, should_byteswap))
    if should_byteswap:
        tm.assert_equal(output_number, number.byteswap())
    else:
        tm.assert_equal(output_number, number)
