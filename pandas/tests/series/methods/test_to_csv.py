# [MODULE DOC]
# Tests for the Series method that writes comma-separated text, covering round trips through the CSV reader, encodings, compression, float formatting and index types.
# Uses shared fixtures for a datetime-indexed series, a string-indexed series, a temporary file path and a compression parametrization (provided by conftest files).
# Defines a single test class; no import-time side effects.

from datetime import datetime
from io import StringIO

import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm

from pandas.io.common import get_handle


class TestSeriesToCSV:
    # [CLASS DOC]
    # Groups the tests of writing a Series to CSV text or file. Stateless; the only non-test member is a reader helper used to read back written files.
    def read_csv(self, path, **kwargs):
        # [FUNCTION DOC]
        # HELPER: reads a CSV file or buffer back into a Series for the round-trip tests in this class.
        # Takes a path or buffer plus keyword arguments forwarded to the CSV reader; defaults are the first column as index and no header row, and caller keywords override those defaults.
        # The single resulting column is squeezed into a Series. When the header parameter ends up unset (no header), the series name and index name are both cleared to nothing.
        # Returns the resulting Series.
        params = {"index_col": 0, "header": None}
        params.update(**kwargs)

        header = params.get("header")
        out = pd.read_csv(path, **params).squeeze("columns")

        if header is None:
            out.name = out.index.name = None

        return out

    def test_from_csv(self, datetime_series, string_series, temp_file):
        # [FUNCTION DOC]
        # Verifies that a Series written to CSV with and without a header reads back equal to the original.
        # Uses the datetime series (frequency stripped from its index because it does not survive a round trip; expected index is converted to microsecond resolution) and the string-indexed series.
        # Checks that names are absent when no header is written and equal the series names ("ts", "series") when a header is written (issue 10483).
        # Finally writes a pipe-separated file by hand and checks it parses to a Series with datetime keys when date parsing is on and with plain string keys when it is off.
        # freq doesn't round-trip
        datetime_series.index = datetime_series.index._with_freq(None)

        path = temp_file
        datetime_series.to_csv(path, header=False)
        ts = self.read_csv(path, parse_dates=True)
        expected = datetime_series.copy()
        expected.index = expected.index.as_unit("us")
        tm.assert_series_equal(expected, ts, check_names=False)

        assert ts.name is None
        assert ts.index.name is None

        # see gh-10483
        datetime_series.to_csv(path, header=True)
        ts_h = self.read_csv(path, header=0)
        assert ts_h.name == "ts"

        string_series.to_csv(path, header=False)
        series = self.read_csv(path)
        tm.assert_series_equal(string_series, series, check_names=False)

        assert series.name is None
        assert series.index.name is None

        string_series.to_csv(path, header=True)
        series_h = self.read_csv(path, header=0)
        assert series_h.name == "series"

        with open(path, "w", encoding="utf-8") as outfile:
            outfile.write("1998-01-01|1.0\n1999-01-01|2.0")

        series = self.read_csv(path, sep="|", parse_dates=True)
        check_series = Series({datetime(1998, 1, 1): 1.0, datetime(1999, 1, 1): 2.0})
        tm.assert_series_equal(check_series, series)

        series = self.read_csv(path, sep="|", parse_dates=False)
        check_series = Series({"1998-01-01": 1.0, "1999-01-01": 2.0})
        tm.assert_series_equal(check_series, series)

    def test_to_csv(self, datetime_series, temp_file):
        # [FUNCTION DOC]
        # Verifies writing the datetime series to a temporary file: the second line of the output is not blank (no spurious blank lines with universal newlines), and writing without index or header yields values that load numerically as almost equal to the series values.
        datetime_series.to_csv(temp_file, header=False)

        with open(temp_file, newline=None, encoding="utf-8") as f:
            lines = f.readlines()
        assert lines[1] != "\n"

        datetime_series.to_csv(temp_file, index=False, header=False)
        arr = np.loadtxt(temp_file)
        tm.assert_almost_equal(arr, datetime_series.values)

    def test_to_csv_unicode_index(self):
        # [FUNCTION DOC]
        # Verifies that a Series with non-ASCII (Hebrew) values and index labels written to an in-memory text buffer with UTF-8 encoding and no header reads back equal to the original.
        buf = StringIO()
        s = Series(["\u05d0", "d2"], index=["\u05d0", "\u05d1"])

        s.to_csv(buf, encoding="UTF-8", header=False)
        buf.seek(0)

        s2 = self.read_csv(buf, index_col=0, encoding="UTF-8")
        tm.assert_series_equal(s, s2)

    def test_to_csv_float_format(self, temp_file):
        # [FUNCTION DOC]
        # Verifies that the float format option (two decimal places) is applied when writing: floats are written rounded and read back equal to the rounded values.
        ser = Series([0.123456, 0.234567, 0.567567])
        ser.to_csv(temp_file, float_format="%.2f", header=False)

        rs = self.read_csv(temp_file)
        xp = Series([0.12, 0.23, 0.57])
        tm.assert_series_equal(rs, xp)

    def test_to_csv_list_entries(self):
        # [FUNCTION DOC]
        # Smoke test: a Series whose elements are lists (from splitting strings on a pattern) can be written to a text buffer without header without raising. No output assertion is made.
        s = Series(["jack and jill", "jesse and frank"])

        split = s.str.split(r"\s+and\s+")

        buf = StringIO()
        split.to_csv(buf, header=False)

    def test_to_csv_path_is_none(self):
        # [FUNCTION DOC]
        # Verifies that when the target path is None, writing returns the CSV text as a string instead of None (issue 8215), matching DataFrame behavior.
        # GH 8215
        # Series.to_csv() was returning None, inconsistent with
        # DataFrame.to_csv() which returned string
        s = Series([1, 2, 3])
        csv_str = s.to_csv(path_or_buf=None, header=False)
        assert isinstance(csv_str, str)

    @pytest.mark.parametrize(
        "s,encoding",
        [
            (
                Series([0.123456, 0.234567, 0.567567], index=["A", "B", "C"], name="X"),
                None,
            ),
            # GH 21241, 21118
            (Series(["abc", "def", "ghi"], name="X"), "ascii"),
            (Series(["123", "你好", "世界"], name="中文"), "gb2312"),
            (
                Series(["123", "Γειά σου", "Κόσμε"], name="Ελληνικά"),  # noqa: RUF001
                "cp737",
            ),
        ],
    )
    def test_to_csv_compression(self, s, encoding, compression, temp_file):
        # [FUNCTION DOC]
        # Verifies CSV writing with each compression method and several encodings (none, ascii, gb2312 for Chinese, cp737 for Greek), parametrized over a series and its encoding; the compression comes from a fixture.
        # Checks three round trips: write to a path then read; write through an opened compression-aware file handle then read; and decompress the raw file to confirm the series name appears in the decoded text and that the decompressed content also reads back equal.
        filename = temp_file
        s.to_csv(filename, compression=compression, encoding=encoding, header=True)
        # test the round trip - to_csv -> read_csv
        result = pd.read_csv(
            filename,
            compression=compression,
            encoding=encoding,
            index_col=0,
        ).squeeze("columns")
        tm.assert_series_equal(s, result)

        # test the round trip using file handle - to_csv -> read_csv
        with get_handle(
            filename, "w", compression=compression, encoding=encoding
        ) as handles:
            s.to_csv(handles.handle, encoding=encoding, header=True)

        result = pd.read_csv(
            filename,
            compression=compression,
            encoding=encoding,
            index_col=0,
        ).squeeze("columns")
        tm.assert_series_equal(s, result)

        # explicitly ensure file was compressed
        with tm.decompress_file(filename, compression) as fh:
            text = fh.read().decode(encoding or "utf8")
            assert s.name in text

        with tm.decompress_file(filename, compression) as fh:
            tm.assert_series_equal(
                s,
                pd.read_csv(fh, index_col=0, encoding=encoding).squeeze("columns"),
            )

    def test_to_csv_interval_index(self, using_infer_string, temp_file):
        # [FUNCTION DOC]
        # Verifies writing a Series with an interval index (issue 28210): since an interval index cannot round trip through the reader, the read-back index is compared against the original index cast to string type.
        # The infer-string fixture parameter is accepted but not used in the body.
        # GH 28210
        s = Series(["foo", "bar", "baz"], index=pd.interval_range(0, 3))

        s.to_csv(temp_file, header=False)
        result = self.read_csv(temp_file, index_col=0)

        # can't roundtrip intervalindex via read_csv so check string repr (GH 23595)
        expected = s
        expected.index = expected.index.astype("str")
        tm.assert_series_equal(result, expected)
