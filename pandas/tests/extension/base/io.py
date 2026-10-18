# [MODULE DOC]
# Base tests for reading extension-array columns back from CSV text.
# Public contents: BaseParsingTests mixin. Requires the data fixture and the request fixture.
from io import StringIO

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray


class BaseParsingTests:
    # [CLASS DOC]
    # Mixin with a CSV round-trip parsing test for the extension array under test.
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        # [FUNCTION DOC]
        # Round-trips a one-column frame holding the data fixture through CSV text and back using
        # the extension dtype name as the requested read dtype. Parametrized over the C and Python
        # parser engines. Writes the frame without an index and with NaN as the missing marker, reads it
        # back with a dtype mapping, and asserts the result equals the original frame.
        # Marks the test as an expected failure (not-implemented error) when the array class has not
        # overridden the string-based constructor inherited from the base extension array; categorical
        # and numpy-wrapping dtypes are exempt because the parser special-cases them internally.
        if isinstance(data.dtype, pd.CategoricalDtype):
            # in parsers.pyx _convert_with_dtype there is special-casing for
            #  Categorical that preempts _from_sequence_of_strings
            pass
        elif isinstance(data.dtype, pd.core.dtypes.dtypes.NumpyEADtype):
            # These get unwrapped internally so are treated as numpy dtypes
            #  in the parsers.pyx code
            pass
        elif (
            type(data)._from_sequence_of_strings.__func__
            is ExtensionArray._from_sequence_of_strings.__func__
        ):
            # i.e. the EA hasn't overridden _from_sequence_of_strings
            mark = pytest.mark.xfail(
                reason="_from_sequence_of_strings not implemented",
                raises=NotImplementedError,
            )
            request.node.add_marker(mark)

        df = pd.DataFrame({"with_dtype": pd.Series(data, dtype=str(data.dtype))})
        csv_output = df.to_csv(index=False, na_rep=np.nan)
        result = pd.read_csv(
            StringIO(csv_output), dtype={"with_dtype": str(data.dtype)}, engine=engine
        )
        expected = df
        tm.assert_frame_equal(result, expected)
