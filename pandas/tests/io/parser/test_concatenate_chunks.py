# [MODULE DOC]
# Tests of the internal helper that concatenates the per-chunk column arrays produced by
# the C parser wrapper, covering arrow-backed arrays: numeric widening and the
# mixed-type warning. Imports the private concatenation function from the C parser
# wrapper module.

import numpy as np
import pytest

from pandas.errors import DtypeWarning

import pandas._testing as tm
from pandas.core.arrays import ArrowExtensionArray

from pandas.io.parsers.c_parser_wrapper import _concatenate_chunks


def test_concatenate_chunks_pyarrow():
    # [FUNCTION DOC]
    # Skipped without pyarrow. Verifies concatenating an arrow float chunk with an arrow
    # integer chunk for one column gives a single arrow float array of four values
    # (integers widened to floats).
    # GH#51876
    pa = pytest.importorskip("pyarrow")
    chunks = [
        {0: ArrowExtensionArray(pa.array([1.5, 2.5]))},
        {0: ArrowExtensionArray(pa.array([1, 2]))},
    ]
    result = _concatenate_chunks(chunks, ["column_0", "column_1"])
    expected = ArrowExtensionArray(pa.array([1.5, 2.5, 1.0, 2.0]))
    tm.assert_extension_array_equal(result[0], expected)


def test_concatenate_chunks_pyarrow_strings():
    # [FUNCTION DOC]
    # Skipped without pyarrow. Verifies concatenating an arrow float chunk with an arrow
    # string chunk emits a dtype warning saying the column has mixed types (naming the
    # column) and returns a plain object array holding the floats and the strings.
    # GH#51876
    pa = pytest.importorskip("pyarrow")
    chunks = [
        {0: ArrowExtensionArray(pa.array([1.5, 2.5]))},
        {0: ArrowExtensionArray(pa.array(["a", "b"]))},
    ]
    with tm.assert_produces_warning(
        DtypeWarning, match="Columns \\(0: column_0\\) have mixed types"
    ):
        result = _concatenate_chunks(chunks, ["column_0", "column_1"])
    expected = np.concatenate(
        [np.array([1.5, 2.5], dtype=object), np.array(["a", "b"])]
    )
    tm.assert_numpy_array_equal(result[0], expected)
