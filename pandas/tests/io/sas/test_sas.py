# [MODULE DOC]
# Format-independent tests of the read_sas entry point: argument validation when the format cannot be determined, and reading a compressed archive.
# Contents: a test class with two tests and one standalone archive test. Uses the datapath and temp_file fixtures.
from io import StringIO

import pytest

from pandas import read_sas
import pandas._testing as tm


class TestSas:
    # [CLASS DOC]
    # Tests for read_sas error handling when the SAS format is not provided or inferable.
    def test_sas_buffer_format(self):
        # [FUNCTION DOC]
        # Verifies that passing a buffer object (an in-memory text stream) without a format raises a value error saying a format string must be specified (GH 14947).
        # see gh-14947
        b = StringIO("")

        msg = (
            "If this is a buffer object rather than a string "
            "name, you must specify a format string"
        )
        with pytest.raises(ValueError, match=msg):
            read_sas(b)

    def test_sas_read_no_format_or_extension(self, temp_file):
        # [FUNCTION DOC]
        # Verifies that reading from a temporary file path with no extension and no format argument raises a value error stating the format of the SAS file could not be inferred (GH 24548).
        # see gh-24548
        msg = "unable to infer format of SAS file.+"
        with pytest.raises(ValueError, match=msg):
            read_sas(temp_file)


def test_sas_archive(datapath):
    # [FUNCTION DOC]
    # Reads the airline sas7bdat dataset both uncompressed and from its gzip-compressed copy (the latter with the format given explicitly) and asserts the two frames are equal.
    fname_uncompressed = datapath("io", "sas", "data", "airline.sas7bdat")
    df_uncompressed = read_sas(fname_uncompressed)
    fname_compressed = datapath("io", "sas", "data", "airline.sas7bdat.gz")
    df_compressed = read_sas(fname_compressed, format="sas7bdat")
    tm.assert_frame_equal(df_uncompressed, df_compressed)
