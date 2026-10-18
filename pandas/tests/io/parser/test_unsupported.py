"""
Tests that features that are currently unsupported in
either the Python or C parser are actually enforced
and are clearly communicated to the user.

Ultimately, the goal is to remove test cases from this
test suite as new feature support is added to the parsers.
"""
# [MODULE DOC]
# Tests that read_csv options unsupported by a given engine (C, python, python-fwf, pyarrow) are rejected or warned about with clear messages. Goal stated upstream: tests are removed as feature support is added.
# Contents: a fixture for the two python-based engine names, a class of engine-specific tests, and several standalone tests for file handle closing, invalid inputs, separator-sniffing fallback and dtype backend validation. Module-level filter ignores the BlockManager deprecation warning.

from io import StringIO
import os
from pathlib import Path

import pytest

from pandas.errors import ParserError

from pandas import DataFrame
import pandas._testing as tm

from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture(params=["python", "python-fwf"], ids=lambda val: val)
def python_engine(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the python and python-fwf engine names (ids are the names); returns the engine name for the test.
    return request.param


class TestUnsupportedFeatures:
    # [CLASS DOC]
    # Tests for options that specific parser engines do not support: removed arguments, C-engine limits, python and pyarrow unsupported option lists, and callable bad-line handlers.
    def test_mangle_dupe_cols_false(self):
        # [FUNCTION DOC]
        # Verifies the removed mangle_dupe_cols argument is rejected: for both the C and python engines, passing it raises a type error mentioning an unexpected keyword (gh-12935).
        # KNOWN ISSUE: the test name says false but the argument is passed as true; the outcome does not depend on the value.
        # see gh-12935
        data = "a b c\n1 2 3"

        for engine in ("c", "python"):
            with pytest.raises(TypeError, match="unexpected keyword"):
                read_csv(StringIO(data), engine=engine, mangle_dupe_cols=True)

    def test_c_engine(self):
        # [FUNCTION DOC]
        # Covers C-engine limitations (gh-6607). Explicit C engine with a regex-like separator, a non-ascii quote character, or footer skipping raises a value error saying it does not support the option; with the default engine the same options only emit a parser warning (falling back to python). A whitespace-separated ragged table raises a parser error about tokenizing data for both default and C engine. Multi-character or empty thousands separators and multi-character line terminators raise value errors that only length-1 values are supported.
        # see gh-6607
        data = "a b c\n1 2 3"
        msg = "does not support"

        # specify C engine with unsupported options (raise)
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep=r"\s")
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep="\t", quotechar=chr(128))
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", skipfooter=1)

        # specify C-unsupported options without python-unsupported options
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep=r"\s")
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep="\t", quotechar=chr(128))
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), skipfooter=1)

        text = """                      A       B       C       D        E
one two three   four
a   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640
a   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744
x   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838"""
        msg = "Error tokenizing data"

        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), sep="\\s+")
        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), engine="c", sep="\\s+")

        msg = "Only length-1 thousands markers supported"
        data = """A|B|C
1|2,334|5
10|13|10.
"""
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), thousands=",,")
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), thousands="")

        msg = "Only length-1 line terminators supported"
        data = "a,b,c~~1,2,3~~4,5,6"
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), lineterminator="~~")

    def test_python_engine(self, python_engine):
        # [FUNCTION DOC]
        # Parametrized over the two python-based engines: for every option in the engine's private unsupported-options collection, passes a placeholder object value and asserts a value error stating that option is not supported with the engine.
        from pandas.io.parsers.readers import _python_unsupported as py_unsupported

        data = """1,2,3,,
1,2,3,4,
1,2,3,4,5
1,2,,,
1,2,3,4,"""

        for default in py_unsupported:
            msg = (
                f"The {default!r} option is not "
                f"supported with the {python_engine!r} engine"
            )

            kwargs = {default: object()}
            with pytest.raises(ValueError, match=msg):
                read_csv(StringIO(data), engine=python_engine, **kwargs)

    def test_python_engine_file_no_iter(self, python_engine):
        # [FUNCTION DOC]
        # Defines a file-like object without iteration support and asserts that read_csv with a python-based engine raises a type error saying it is not iterable (or must be an iterator, depending on interpreter version) (gh-16530).
        # see gh-16530
        class NoNextBuffer:
            # [CLASS DOC]
            # HELPER: minimal fake file object used by test_python_engine_file_no_iter. Stores the data; read and readline return it unchanged and next delegates to the data's own next, but it defines no iteration protocol so the parser cannot iterate it.
            def __init__(self, csv_data) -> None:
                # [FUNCTION DOC]
                # HELPER: stores the given csv data string as the data attribute.
                self.data = csv_data

            def __next__(self):
                # [FUNCTION DOC]
                # HELPER: returns the next item of the stored data by calling its own next method (which a plain string does not have, so this would raise an attribute error if invoked).
                return self.data.__next__()

            def read(self):
                # [FUNCTION DOC]
                # HELPER: returns the stored data unchanged, ignoring any size argument (none accepted).
                return self.data

            def readline(self):
                # [FUNCTION DOC]
                # HELPER: returns the stored data unchanged as if it were a single line.
                return self.data

        data = "a\n1"
        msg = "'NoNextBuffer' object is not iterable|argument 1 must be an iterator"

        with pytest.raises(TypeError, match=msg):
            read_csv(NoNextBuffer(data), engine=python_engine)

    def test_pyarrow_engine(self):
        # [FUNCTION DOC]
        # For every option in the pyarrow unsupported-options collection, builds a keyword value (a placeholder object, a sample dialect name, true for the boolean-valued deprecated bad-line options, or the warn string for on_bad_lines) and asserts read_csv with the pyarrow engine raises a value error saying that option is not supported with pyarrow.
        from pandas.io.parsers.readers import _pyarrow_unsupported as pa_unsupported

        data = """1,2,3,,
        1,2,3,4,
        1,2,3,4,5
        1,2,,,
        1,2,3,4,"""

        for default in pa_unsupported:
            msg = f"The {default!r} option is not supported with the 'pyarrow' engine"
            kwargs = {default: object()}
            default_needs_bool = {"warn_bad_lines", "error_bad_lines"}
            if default == "dialect":
                kwargs[default] = "excel"  # test a random dialect
            elif default in default_needs_bool:
                kwargs[default] = True
            elif default == "on_bad_lines":
                kwargs[default] = "warn"

            with pytest.raises(ValueError, match=msg):
                read_csv(StringIO(data), engine="pyarrow", **kwargs)

    def test_on_bad_lines_callable_python_or_pyarrow(self, all_parsers):
        # [FUNCTION DOC]
        # GH 5686 and 54643. Passing a callable for on_bad_lines raises a value error (callable only allowed for python or pyarrow engines) on the C engine; with python or pyarrow it parses without error.
        # GH 5686
        # GH 54643
        sio = StringIO("a,b\n1,2")
        bad_lines_func = lambda x: x
        parser = all_parsers
        if all_parsers.engine not in ["python", "pyarrow"]:
            msg = (
                "on_bad_line can only be a callable "
                "function if engine='python' or 'pyarrow'"
            )
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(sio, on_bad_lines=bad_lines_func)
        else:
            parser.read_csv(sio, on_bad_lines=bad_lines_func)


def test_close_file_handle_on_invalid_usecols(all_parsers, temp_file):
    # [FUNCTION DOC]
    # GH 45384. Skipped on pyarrow (upstream arrow key error). Writes a small csv to the temp file, asserts reading with a usecols entry naming a missing column raises a value error mentioning the column and emits no warnings, then deletes the file, which would fail on Windows if the handle were left open.
    # GH 45384
    parser = all_parsers

    error = ValueError
    if parser.engine == "pyarrow":
        # Raises pyarrow.lib.ArrowKeyError
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    fname = temp_file
    Path(fname).write_text("col1,col2\na,b\n1,2", encoding="utf-8")
    with tm.assert_produces_warning(False):
        with pytest.raises(error, match="col3"):
            parser.read_csv(fname, usecols=["col1", "col2", "col3"])
    # unlink fails on windows if file handles still point to it
    os.unlink(fname)


def test_invalid_file_inputs(request, all_parsers):
    # [FUNCTION DOC]
    # GH#45957. Asserts read_csv given an empty list raises a value error with an invalid-input message; expected to fail for the python engine because it supports lists.
    # GH#45957
    parser = all_parsers
    if parser.engine == "python":
        request.applymarker(
            pytest.mark.xfail(reason=f"{parser.engine} engine supports lists.")
        )

    with pytest.raises(ValueError, match="Invalid"):
        parser.read_csv([])


def test_sep_none_falls_back_to_python_engine():
    # [FUNCTION DOC]
    # GH#66639. With the default engine and no separator (sniffing), asserts a parser warning mentioning that case is emitted and the data parses with the sniffed semicolon; with engine python explicitly, asserts no warning and the same frame.
    # GH#66639 sniffing the separator is python-engine only, so the default
    # engine falls back to it rather than handing sep=None to the C parser
    data = "a;b\n1;2\n"
    expected = DataFrame({"a": [1], "b": [2]})

    with tm.assert_produces_warning(parsers.ParserWarning, match="sep=None"):
        result = read_csv(StringIO(data), sep=None)
    tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(None):
        result = read_csv(StringIO(data), sep=None, engine="python")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ["c", "pyarrow"])
def test_sep_none_explicit_engine_raises(engine):
    # [FUNCTION DOC]
    # GH#66639. Parametrized over C and pyarrow: explicitly requesting an engine that cannot sniff the separator and passing no separator raises a value error saying the engine does not support that.
    # GH#66639 an explicitly requested engine that cannot sniff the separator
    # reports that rather than falling back
    msg = f"the '{engine}' engine does not support sep=None"

    with pytest.raises(ValueError, match=msg):
        read_csv(StringIO("a;b\n1;2\n"), sep=None, engine=engine)


def test_invalid_dtype_backend(all_parsers):
    # [FUNCTION DOC]
    # Asserts read_csv with the dtype_backend value numpy raises a value error stating only numpy_nullable and pyarrow are allowed.
    parser = all_parsers
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        parser.read_csv("test", dtype_backend="numpy")
