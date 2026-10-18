# [MODULE DOC]
# Base tests for the textual representation of an extension array in repr, series and frame
# display, and frame info. Public contents: BasePrintingTests mixin. Uses the data fixture.
import io

import pytest

import pandas as pd


class BasePrintingTests:
    # [CLASS DOC]
    # Mixin with formatting tests.
    """Tests checking the formatting of your EA when printed."""

    @pytest.mark.parametrize("size", ["big", "small"])
    def test_array_repr(self, data, size):
        # [FUNCTION DOC]
        # Parametrized over size small (first five elements) and big (twenty concatenated copies).
        # The array repr must contain the class name, the length text with the correct length, and the
        # dtype's string; for the big case it must contain an ellipsis for truncation.
        if size == "small":
            data = data[:5]
        else:
            data = type(data)._concat_same_type([data] * 20)

        result = repr(data)
        assert type(data).__name__ in result
        assert f"Length: {len(data)}" in result
        assert str(data.dtype) in result
        if size == "big":
            assert "..." in result

    def test_array_repr_unicode(self, data):
        # [FUNCTION DOC]
        # Asserts converting the array with the string builtin returns a string without raising.
        result = str(data)
        assert isinstance(result, str)

    def test_series_repr(self, data):
        # [FUNCTION DOC]
        # Asserts the series repr contains the dtype name.
        ser = pd.Series(data)
        assert data.dtype.name in repr(ser)

    def test_dataframe_repr(self, data):
        # [FUNCTION DOC]
        # Smoke test: the frame repr must not raise.
        df = pd.DataFrame({"A": data})
        repr(df)

    def test_dtype_name_in_info(self, data):
        # [FUNCTION DOC]
        # Writes frame info to an in-memory buffer and asserts the dtype name appears in the text.
        buf = io.StringIO()
        pd.DataFrame({"A": data}).info(buf=buf)
        result = buf.getvalue()
        assert data.dtype.name in result
