# [MODULE DOC]
# Tests for reading SAS XPORT (.xpt) transport files using the generic SAS reader
# function with the xport format. Compares results to reference CSVs produced with the R
# foreign library. Numbers in xport files are always float64, so integer columns of the
# references are converted before comparison.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

from pandas.io.sas.sasreader import read_sas

# CSV versions of test xpt files were obtained using the R foreign library

# Numbers in a SAS xport file are always float64, so need to convert
# before making comparisons.


def numeric_as_float(data):
    # [FUNCTION DOC]
    # HELPER: converts reference CSV data for comparison.
    # Takes a frame and converts, in place, every column of int64 dtype to float64.
    # Returns nothing; mutates the argument.
    for v in data.columns:
        if data[v].dtype is np.dtype("int64"):
            data[v] = data[v].astype(np.float64)


class TestXport:
    # [CLASS DOC]
    # Tests of the xport reader on several NHANES sample files, covering full, indexed,
    # incremental, binary-handle and error cases.
    @pytest.mark.slow
    def test1_basic(self, datapath):
        # [FUNCTION DOC]
        # Marked slow. Uses the all-numeric DEMO_G file. Verifies: full read equals the
        # CSV; reading more rows than exist returns all rows; read of 10 rows and the
        # chunk fetch method with chunksize 10 both equal the first 10 rows; looping
        # over chunks of 100 totals the full row count; and a read with format inferred
        # equals the CSV.
        # Tests with DEMO_G.xpt (all numeric file)

        # Compare to this
        file01 = datapath("io", "sas", "data", "DEMO_G.xpt")
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        numeric_as_float(data_csv)

        # Read full file
        data = read_sas(file01, format="xport")
        tm.assert_frame_equal(data, data_csv)
        num_rows = data.shape[0]

        # Test reading beyond end of file
        with read_sas(file01, format="xport", iterator=True) as reader:
            data = reader.read(num_rows + 100)
        assert data.shape[0] == num_rows

        # Test incremental read with `read` method.
        with read_sas(file01, format="xport", iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

        # Test incremental read with `get_chunk` method.
        with read_sas(file01, format="xport", chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

        # Test read in loop
        m = 0
        with read_sas(file01, format="xport", chunksize=100) as reader:
            for x in reader:
                m += x.shape[0]
        assert m == num_rows

        # Read full file with `read_sas` method
        data = read_sas(file01)
        tm.assert_frame_equal(data, data_csv)

    def test1_index(self, datapath):
        # [FUNCTION DOC]
        # Same file as the basic test with the SEQN column used as the index (both in
        # the reference and via the index argument). Checks full read, a 10-row
        # incremental read and a chunk fetch against the reference, ignoring index type.
        # Tests with DEMO_G.xpt using index (all numeric file)

        # Compare to this
        file01 = datapath("io", "sas", "data", "DEMO_G.xpt")
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        data_csv = data_csv.set_index("SEQN")
        numeric_as_float(data_csv)

        # Read full file
        data = read_sas(file01, index="SEQN", format="xport")
        tm.assert_frame_equal(data, data_csv, check_index_type=False)

        # Test incremental read with `read` method.
        with read_sas(file01, index="SEQN", format="xport", iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

        # Test incremental read with `get_chunk` method.
        with read_sas(file01, index="SEQN", format="xport", chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

    def test1_incremental(self, datapath):
        # [FUNCTION DOC]
        # Reads DEMO_G with SEQN index in chunks of 1000, concatenates all chunks along
        # rows and asserts equality with the reference (index type ignored).
        # Test with DEMO_G.xpt, reading full file incrementally

        file01 = datapath("io", "sas", "data", "DEMO_G.xpt")
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        data_csv = data_csv.set_index("SEQN")
        numeric_as_float(data_csv)

        with read_sas(file01, index="SEQN", chunksize=1000) as reader:
            all_data = list(reader)
        data = pd.concat(all_data, axis=0)

        tm.assert_frame_equal(data, data_csv, check_index_type=False)

    def test2(self, datapath):
        # [FUNCTION DOC]
        # Reads SSHSV1_A file with format inferred and compares to the CSV after
        # converting ints to floats.
        # Test with SSHSV1_A.xpt

        file02 = datapath("io", "sas", "data", "SSHSV1_A.xpt")
        # Compare to this
        data_csv = pd.read_csv(file02.replace(".xpt", ".csv"))
        numeric_as_float(data_csv)

        data = read_sas(file02)
        tm.assert_frame_equal(data, data_csv)

    def test2_binary(self, datapath):
        # [FUNCTION DOC]
        # GH 35693. Passes an already-open binary file object to the reader; asserts the
        # result equals the CSV. The comment states the open handle must not be closed
        # incorrectly by the reader (the test itself does not check closure).
        # Test with SSHSV1_A.xpt, read as a binary file

        # Compare to this
        file02 = datapath("io", "sas", "data", "SSHSV1_A.xpt")
        data_csv = pd.read_csv(file02.replace(".xpt", ".csv"))
        numeric_as_float(data_csv)

        with open(file02, "rb") as fd:
            # GH#35693 ensure that if we pass an open file, we
            #  dont incorrectly close it in read_sas
            data = read_sas(fd, format="xport")

        tm.assert_frame_equal(data, data_csv)

    def test_multiple_types(self, datapath):
        # [FUNCTION DOC]
        # Reads DRXFCD_G (text and numeric variables) with utf-8 encoding and compares
        # to the CSV.
        # Test with DRXFCD_G.xpt (contains text and numeric variables)

        # Compare to this
        file03 = datapath("io", "sas", "data", "DRXFCD_G.xpt")
        data_csv = pd.read_csv(file03.replace(".xpt", ".csv"))

        data = read_sas(file03, encoding="utf-8")
        tm.assert_frame_equal(data, data_csv)

    def test_truncated_float_support(self, datapath):
        # [FUNCTION DOC]
        # GH 11713. Reads paxraw_d_short, whose floats are truncated to 5 bytes; asserts
        # the result cast to int64 equals the CSV.
        # Test with paxraw_d_short.xpt, a shortened version of:
        # http://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/PAXRAW_D.ZIP
        # This file has truncated floats (5 bytes in this case).

        # GH 11713
        file04 = datapath("io", "sas", "data", "paxraw_d_short.xpt")
        data_csv = pd.read_csv(file04.replace(".xpt", ".csv"))

        data = read_sas(file04, format="xport")
        tm.assert_frame_equal(data.astype("int64"), data_csv)

    def test_cport_header_found_raises(self, datapath):
        # [FUNCTION DOC]
        # Reading a file whose header says it is a CPORT file (despite its extension)
        # with the xport format must raise a value error stating that CPORT files are
        # not readable.
        # Test with DEMO_PUF.cpt, the beginning of puf2019_1_fall.xpt
        # from https://www.cms.gov/files/zip/puf2019.zip
        # (despite the extension, it's a cpt file)
        msg = "Header record indicates a CPORT file, which is not readable."
        with pytest.raises(ValueError, match=msg):
            read_sas(datapath("io", "sas", "data", "DEMO_PUF.cpt"), format="xport")
