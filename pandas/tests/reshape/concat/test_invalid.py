# [MODULE DOC]
# Tests for invalid and unusual inputs to the concatenate function: wrong element
# types, a non-iterable first argument, generators and text-file chunk readers.
# Uses only pandas public API and the testing helpers.
from io import StringIO

import numpy as np
import pytest

from pandas import (
    DataFrame,
    concat,
    read_csv,
)
import pandas._testing as tm


class TestInvalidConcat:
    # [CLASS DOC]
    # Test collection for invalid input handling in concatenation. Stateless.
    @pytest.mark.parametrize("obj", [1, {}, [1, 2], (1, 2)])
    def test_concat_invalid(self, obj):
        # [FUNCTION DOC]
        # Parametrized over a number, an empty dict, a list and a tuple. Verifies that
        # concatenating a frame with a non-pandas object raises a type error naming the
        # offending type and stating only series and frames are valid.
        # trying to concat an ndframe with a non-ndframe
        df1 = DataFrame(range(2))
        msg = (
            f"cannot concatenate object of type '{type(obj)}'; "
            "only Series and DataFrame objs are valid"
        )
        with pytest.raises(TypeError, match=msg):
            concat([df1, obj])

    def test_concat_invalid_first_argument(self):
        # [FUNCTION DOC]
        # Verifies that passing a single frame instead of an iterable raises a type error
        # saying the first argument must be an iterable of pandas objects and naming the
        # type passed.
        df1 = DataFrame(range(2))
        msg = (
            "first argument must be an iterable of pandas "
            'objects, you passed an object of type "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            concat(df1)

    def test_concat_generator_obj(self):
        # [FUNCTION DOC]
        # Smoke test: a generator of frames is accepted without error.
        # generator ok though
        concat(DataFrame(np.random.default_rng(2).random((5, 5))) for _ in range(3))

    def test_concat_textreader_obj(self):
        # [FUNCTION DOC]
        # Verifies that the chunked reader from reading delimited text with chunk size one
        # can be concatenated (with ignored index) into a frame equal to reading the whole
        # text at once (GH 6583).
        # text reader ok
        # GH6583
        data = """index,A,B,C,D
                  foo,2,3,4,5
                  bar,7,8,9,10
                  baz,12,13,14,15
                  qux,12,13,14,15
                  foo2,12,13,14,15
                  bar2,12,13,14,15
               """

        with read_csv(StringIO(data), chunksize=1) as reader:
            result = concat(reader, ignore_index=True)
        expected = read_csv(StringIO(data))
        tm.assert_frame_equal(result, expected)
