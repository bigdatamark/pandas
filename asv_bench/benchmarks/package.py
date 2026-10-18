"""
Benchmarks for pandas at the package-level.
"""

# [MODULE DOC]
# Package-level benchmark: the time to import the library in a fresh interpreter.
# Public contents: the benchmark class TimeImport. Spawns a child interpreter at
# run time; no import-time side effects in this module.
import subprocess
import sys


class TimeImport:
    # [CLASS DOC]
    # Benchmark of the library import time measured by the interpreter's own import
    # timing report, in a child process. No parameters and no state.
    def time_import(self):
        # [FUNCTION DOC]
        # Runs a fresh interpreter that imports the library with the interpreter's
        # import-timing flag, which prints a table of import times to the error
        # stream. Raises a called-process error if the child exits with failure.
        # Takes the last line of the error output (the top-level import of the
        # library), reads its second-to-last pipe-separated field as the cumulative
        # time in microseconds, and returns that integer.
        # KNOWN ISSUE: asv ignores the return value of a time-prefixed benchmark and
        # measures the elapsed time of the whole call, including the child process
        # start-up, so the carefully extracted microsecond total does not become
        # the benchmark result (the comment below claims it gives a more precise
        # measurement).
        # on py37+ we the "-X importtime" usage gives us a more precise
        #  measurement of the import time we actually care about,
        #  without the subprocess or interpreter overhead
        cmd = [sys.executable, "-X", "importtime", "-c", "import pandas as pd"]
        p = subprocess.run(cmd, stderr=subprocess.PIPE, check=True)

        line = p.stderr.splitlines()[-1]
        field = line.split(b"|")[-2].strip()
        total = int(field)  # microseconds
        return total
