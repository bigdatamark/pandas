# [MODULE DOC]
# Benchmarks for reading SAS files (native datasets and transport files) using
# sample data files shipped with the pandas test suite. Public contents: the module
# constant ROOT (the data directory of the SAS test files, computed relative to
# this file's location three directories up) and the benchmark class SAS. Does not
# re-export the common asv setup hook and has no import-time side effects beyond
# computing ROOT.
from pathlib import Path

from pandas import read_sas

ROOT = Path(__file__).parents[3] / "pandas" / "tests" / "io" / "sas" / "data"


class SAS:
    # [CLASS DOC]
    # Benchmark of the SAS reader on four sample files. No parameters and no setup;
    # each method reads a file from the test data directory.
    def time_read_sas7bdat(self):
        # [FUNCTION DOC]
        # Times reading the whole native dataset file test1.sas7bdat.
        read_sas(ROOT / "test1.sas7bdat")

    def time_read_xpt(self):
        # [FUNCTION DOC]
        # Times reading the whole transport file paxraw_d_short.xpt.
        read_sas(ROOT / "paxraw_d_short.xpt")

    def time_read_sas7bdat_2(self):
        # [FUNCTION DOC]
        # Times opening the compressed dataset 0x00controlbyte.sas7bdat.bz2 in
        # chunks of 11000 rows and reading only the first chunk.
        next(read_sas(ROOT / "0x00controlbyte.sas7bdat.bz2", chunksize=11000))

    def time_read_sas7bdat_2_chunked(self):
        # [FUNCTION DOC]
        # Times reading the compressed dataset in chunks of 1000 rows, stopping
        # after the eleventh chunk (zero-based counter reaches ten).
        for i, _ in enumerate(
            read_sas(ROOT / "0x00controlbyte.sas7bdat.bz2", chunksize=1000)
        ):
            if i == 10:
                break
