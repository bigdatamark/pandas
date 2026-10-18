# [MODULE DOC]
# Tests for the pipe method on DataFrame and Series.

import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestPipe:
    # [CLASS DOC]
    # Groups the pipe tests.
    def test_pipe(self, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over frame and series. Applies a power function through pipe with an extra positional argument; verifies the result equals the elementwise power of the object.
        obj = DataFrame({"A": [1, 2, 3]})
        expected = DataFrame({"A": [1, 4, 9]})
        if frame_or_series is Series:
            obj = obj["A"]
            expected = expected["A"]

        f = lambda x, y: x**y
        result = obj.pipe(f, 2)
        tm.assert_equal(result, expected)

    def test_pipe_tuple(self, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over frame and series. Pipes a function given as a callable-and-name pair so that the object is passed by keyword name; verifies the function receives the object and the result equals the input object.
        obj = DataFrame({"A": [1, 2, 3]})
        obj = tm.get_obj(obj, frame_or_series)

        f = lambda x, y: y
        result = obj.pipe((f, "y"), 0)
        tm.assert_equal(result, obj)

    def test_pipe_tuple_error(self, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over frame and series. Verifies that supplying the target keyword name also as an explicit keyword argument raises a value error saying that name is both the pipe target and a keyword argument.
        obj = DataFrame({"A": [1, 2, 3]})
        obj = tm.get_obj(obj, frame_or_series)

        f = lambda x, y: y

        msg = "y is both the pipe target and a keyword argument"

        with pytest.raises(ValueError, match=msg):
            obj.pipe((f, "y"), x=1, y=0)
