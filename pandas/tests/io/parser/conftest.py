# [MODULE DOC]
# HELPER: shared fixtures and parser wrapper classes for the tests in the parser test
# package and its subpackages.
# Defines wrapper classes that bind a parsing engine (and C-engine memory mode) to the
# CSV and table readers, the lists of parsers and their ids, the fixtures that
# parametrize tests over all engines (all_parsers, the C-only, python-only and
# pyarrow-only variants, and a combination with float precision), data-location
# fixtures, encoding-related fixtures, a fixture of numeric strings with decimal marker
# and digit-separator edge cases, and fixtures that mark pyarrow tests as
# expected-to-fail or skipped.
# Import-time side effects: builds module-level lists of parser parameters; the pyarrow
# parameters are wrapped to skip when pyarrow is not installed and to run on a single
# cpu.

from __future__ import annotations

import os

import pytest

from pandas.compat import HAS_PYARROW
from pandas.compat._optional import VERSIONS

from pandas import (
    read_csv,
    read_table,
)
import pandas._testing as tm


class BaseParser:
    # [CLASS DOC]
    # HELPER: base class for the engine wrappers. Holds three class attributes: the
    # engine name (none in the base), the low-memory flag (true by default) and the list
    # of allowed float precision choices (empty). Its methods call the module-level csv
    # and table readers after injecting the engine and low-memory settings into the
    # keyword arguments, and offer variants that assert a warning is produced during the
    # read. Subclasses set the attributes.
    engine: str | None = None
    low_memory = True
    float_precision_choices: list[str | None] = []

    def update_kwargs(self, kwargs):
        # [FUNCTION DOC]
        # HELPER: takes a keyword-argument dictionary and returns a copy to which the
        # engine name and the low-memory flag of this wrapper have been added
        # (overriding any existing entries for those keys). The input dictionary is not
        # modified.
        kwargs = kwargs.copy()
        kwargs.update({"engine": self.engine, "low_memory": self.low_memory})

        return kwargs

    def read_csv(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: reads delimited data with the csv reader, passing all positional and
        # keyword arguments through after adding this wrapper's engine and low-memory
        # settings. Returns the resulting frame or reader.
        kwargs = self.update_kwargs(kwargs)
        return read_csv(*args, **kwargs)

    def read_csv_check_warnings(
        self,
        warn_type: type[Warning],
        warn_msg: str,
        *args,
        raise_on_extra_warnings=True,
        check_stacklevel: bool = True,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: reads with the csv reader (with engine settings injected) inside a
        # context that asserts a warning is produced.
        # Parameters: the expected warning class, a pattern the warning message must
        # match, then positional arguments for the reader; keyword-only options control
        # whether extra unexpected warnings raise (default true) and whether the
        # warning's stack level is verified (default true); remaining keyword arguments
        # go to the reader. Returns the reader's result. The stack-level check is done
        # here because this is the call site that the warning should point to. Fails the
        # test if the expected warning is not raised.
        # We need to check the stacklevel here instead of in the tests
        # since this is where read_csv is called and where the warning
        # should point to.
        kwargs = self.update_kwargs(kwargs)
        with tm.assert_produces_warning(
            warn_type,
            match=warn_msg,
            raise_on_extra_warnings=raise_on_extra_warnings,
            check_stacklevel=check_stacklevel,
        ):
            return read_csv(*args, **kwargs)

    def read_table(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: same as the csv wrapper but calls the table reader: injects the engine
        # and low-memory settings and returns the result.
        kwargs = self.update_kwargs(kwargs)
        return read_table(*args, **kwargs)

    def read_table_check_warnings(
        self,
        warn_type: type[Warning],
        warn_msg: str,
        *args,
        raise_on_extra_warnings=True,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: reads with the table reader (with engine settings injected) inside a
        # context that asserts a warning is produced.
        # Parameters: the expected warning class, a pattern for the message, positional
        # arguments for the reader, a keyword-only flag controlling whether extra
        # warnings raise (default true); remaining keyword arguments go to the reader.
        # Returns the reader's result.
        # KNOWN ISSUE: unlike the csv counterpart this method has no stack-level
        # option and never passes one on, even though the comment in the body speaks of
        # checking the stack level here.
        # We need to check the stacklevel here instead of in the tests
        # since this is where read_table is called and where the warning
        # should point to.
        kwargs = self.update_kwargs(kwargs)
        with tm.assert_produces_warning(
            warn_type, match=warn_msg, raise_on_extra_warnings=raise_on_extra_warnings
        ):
            return read_table(*args, **kwargs)


class CParser(BaseParser):
    # [CLASS DOC]
    # HELPER: wrapper for the C engine; sets the engine name to c and the allowed float
    # precision choices to default, high and round-trip.
    engine = "c"
    float_precision_choices = [None, "high", "round_trip"]


class CParserHighMemory(CParser):
    # [CLASS DOC]
    # HELPER: C engine wrapper with low-memory mode turned off.
    low_memory = False


class CParserLowMemory(CParser):
    # [CLASS DOC]
    # HELPER: C engine wrapper with low-memory mode turned on (same as the inherited
    # default).
    low_memory = True


class PythonParser(BaseParser):
    # [CLASS DOC]
    # HELPER: wrapper for the python engine; the only allowed float precision choice is
    # the default.
    engine = "python"
    float_precision_choices = [None]


class PyArrowParser(BaseParser):
    # [CLASS DOC]
    # HELPER: wrapper for the pyarrow engine; the only allowed float precision choice is
    # the default.
    engine = "pyarrow"
    float_precision_choices = [None]


@pytest.fixture
def csv_dir_path(datapath):
    # [FUNCTION DOC]
    # HELPER: fixture returning the path of the directory holding the data files used by
    # the parser tests, resolved by the data-path fixture.
    """
    The directory path to the data files needed for parser tests.
    """
    return datapath("io", "parser", "data")


@pytest.fixture
def csv1(datapath):
    # [FUNCTION DOC]
    # HELPER: fixture returning the path of the sample data file test1 in the shared CSV
    # data directory.
    """
    The path to the data file "test1.csv" needed for parser tests.
    """
    return os.path.join(datapath("io", "data", "csv"), "test1.csv")


_cParserHighMemory = CParserHighMemory
_cParserLowMemory = CParserLowMemory
_pythonParser = PythonParser
_pyarrowParser = PyArrowParser

_py_parsers_only = [_pythonParser]
_c_parsers_only = [_cParserHighMemory, _cParserLowMemory]
_pyarrow_parsers_only = [
    pytest.param(
        _pyarrowParser,
        marks=[
            pytest.mark.single_cpu,
            pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow is not installed"),
        ],
    )
]

_all_parsers = [*_c_parsers_only, *_py_parsers_only, *_pyarrow_parsers_only]

_py_parser_ids = ["python"]
_c_parser_ids = ["c_high", "c_low"]
_pyarrow_parsers_ids = ["pyarrow"]

_all_parser_ids = [*_c_parser_ids, *_py_parser_ids, *_pyarrow_parsers_ids]


@pytest.fixture(params=_all_parsers, ids=_all_parser_ids)
def all_parsers(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over all parser wrappers (two C engine variants,
    # python, and pyarrow); yields an instance of the wrapper. For pyarrow it skips the
    # test if the required pyarrow version is not available and sets the pyarrow thread
    # count to one for more stable runs.
    """
    Fixture all of the CSV parsers.
    """
    parser = request.param()
    if parser.engine == "pyarrow":
        pytest.importorskip("pyarrow", VERSIONS["pyarrow"])
        # Try finding a way to disable threads all together
        # for more stable CI runs
        import pyarrow

        pyarrow.set_cpu_count(1)
    return parser


@pytest.fixture(params=_c_parsers_only, ids=_c_parser_ids)
def c_parser_only(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the two C engine wrapper classes; returns an
    # instance.
    """
    Fixture all of the CSV parsers using the C engine.
    """
    return request.param()


@pytest.fixture(params=_py_parsers_only, ids=_py_parser_ids)
def python_parser_only(request):
    # [FUNCTION DOC]
    # HELPER: fixture returning an instance of the python engine wrapper.
    """
    Fixture all of the CSV parsers using the Python engine.
    """
    return request.param()


@pytest.fixture(params=_pyarrow_parsers_only, ids=_pyarrow_parsers_ids)
def pyarrow_parser_only(request):
    # [FUNCTION DOC]
    # HELPER: fixture returning an instance of the pyarrow engine wrapper (parameter
    # carries the single-cpu and installed-check marks).
    """
    Fixture all of the CSV parsers using the Pyarrow engine.
    """
    return request.param()


def _get_all_parser_float_precision_combinations():
    # [FUNCTION DOC]
    # HELPER: builds the parametrization for the combined parser-and-precision fixture.
    # For every parser class and each of its float precision choices, creates one
    # parameter holding an instance and the precision (wrapping pyarrow with single-cpu
    # and installed-check marks) and an id made of the parser id and the precision.
    # Returns a dictionary with the parameter list and the id list.
    """
    Return all allowable parser and float precision
    combinations and corresponding ids.
    """
    params = []
    ids = []
    for parser, parser_id in zip(_all_parsers, _all_parser_ids):
        if hasattr(parser, "values"):
            # Wrapped in pytest.param, get the actual parser back
            parser = parser.values[0]
        for precision in parser.float_precision_choices:
            # Re-wrap in pytest.param for pyarrow
            mark = (
                [
                    pytest.mark.single_cpu,
                    pytest.mark.skipif(
                        not HAS_PYARROW, reason="pyarrow is not installed"
                    ),
                ]
                if parser.engine == "pyarrow"
                else ()
            )
            param = pytest.param((parser(), precision), marks=mark)
            params.append(param)
            ids.append(f"{parser_id}-{precision}")

    return {"params": params, "ids": ids}


@pytest.fixture(
    params=_get_all_parser_float_precision_combinations()["params"],
    ids=_get_all_parser_float_precision_combinations()["ids"],
)
def all_parsers_all_precisions(request):
    # [FUNCTION DOC]
    # HELPER: fixture returning a pair of parser instance and float precision for every
    # allowed combination.
    """
    Fixture for all allowable combinations of parser
    and float precision
    """
    return request.param


_utf_values = [8, 16, 32]

_encoding_seps = ["", "-", "_"]
_encoding_prefixes = ["utf", "UTF"]

_encoding_fmts = [
    f"{prefix}{sep}{{0}}" for sep in _encoding_seps for prefix in _encoding_prefixes
]


@pytest.fixture(params=_utf_values)
def utf_value(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the integer values 8, 16 and 32, the widths used
    # in UTF encoding names.
    """
    Fixture for all possible integer values for a UTF encoding.
    """
    return request.param


@pytest.fixture(params=_encoding_fmts)
def encoding_fmt(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over format templates for UTF encoding names (upper
    # and lower case prefix, with no separator, a dash or an underscore); each has a
    # placeholder for the width.
    """
    Fixture for all possible string formats of a UTF encoding.
    """
    return request.param


@pytest.fixture(
    params=[
        ("-1,0", -1.0),
        ("-1,2e0", -1.2),
        ("-1e0", -1.0),
        ("+1e0", 1.0),
        ("+1e+0", 1.0),
        ("+1e-1", 0.1),
        ("+,1e1", 1.0),
        ("+1,e0", 1.0),
        ("-,1e1", -1.0),
        ("-1,e0", -1.0),
        ("0,1", 0.1),
        ("1,", 1.0),
        (",1", 0.1),
        ("-,1", -0.1),
        ("1_,", 1.0),
        ("1_234,56", 1234.56),
        ("1_234,56e0", 1234.56),
        # negative cases; must not parse as float
        ("_", "_"),
        ("-_", "-_"),
        ("-_1", "-_1"),
        ("-_1e0", "-_1e0"),
        ("_1", "_1"),
        ("_1,", "_1,"),
        ("_1,_", "_1,_"),
        ("_1e0", "_1e0"),
        ("1,2e_1", "1,2e_1"),
        ("1,2e1_0", "1,2e1_0"),
        ("1,_2", "1,_2"),
        (",1__2", ",1__2"),
        (",1e", ",1e"),
        ("-,1e", "-,1e"),
        ("1_000,000_000", "1_000,000_000"),
        ("1,e1_2", "1,e1_2"),
        ("e11,2", "e11,2"),
        ("1e11,2", "1e11,2"),
        ("1,2,2", "1,2,2"),
        ("1,2_1", "1,2_1"),
        ("1,2e-10e1", "1,2e-10e1"),
        ("--1,2", "--1,2"),
        ("1a_2,1", "1a_2,1"),
        ("1,2E-1", 0.12),
        ("1,2E1", 12.0),
    ]
)
def numeric_decimal(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over pairs of an input string and its expected parse
    # result with a comma decimal marker and underscore digit separators. The first
    # group should parse to a float; the negative cases are malformed numbers whose
    # expected result is the same string unchanged.
    """
    Fixture for all numeric formats which should get recognized. The first entry
    represents the value to read while the second represents the expected result.
    """
    return request.param


@pytest.fixture
def pyarrow_xfail(request):
    # [FUNCTION DOC]
    # HELPER: fixture used to mark tests expected to fail for pyarrow. Looks up the
    # parser (from the all-parsers fixture, or the first element of the combined
    # fixture) among the test's fixtures; if none is used it does nothing; if the engine
    # is pyarrow it applies an expected-failure mark with a reason that pyarrow does not
    # support this.
    """
    Fixture that xfails a test if the engine is pyarrow.

    Use if failure is do to unsupported keywords or inconsistent results.
    """
    if "all_parsers" in request.fixturenames:
        parser = request.getfixturevalue("all_parsers")
    elif "all_parsers_all_precisions" in request.fixturenames:
        # Return value is tuple of (engine, precision)
        parser = request.getfixturevalue("all_parsers_all_precisions")[0]
    else:
        return
    if parser.engine == "pyarrow":
        mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
        request.applymarker(mark)


@pytest.fixture
def pyarrow_skip(request):
    # [FUNCTION DOC]
    # HELPER: fixture used to skip tests on pyarrow. Same parser lookup as the
    # expected-failure fixture, but skips the test for the pyarrow engine with a reason
    # pointing to the upstream arrow issue about csv parse failures.
    """
    Fixture that skips a test if the engine is pyarrow.

    Use if failure is do a parsing failure from pyarrow.csv.read_csv
    """
    if "all_parsers" in request.fixturenames:
        parser = request.getfixturevalue("all_parsers")
    elif "all_parsers_all_precisions" in request.fixturenames:
        # Return value is tuple of (engine, precision)
        parser = request.getfixturevalue("all_parsers_all_precisions")[0]
    else:
        return
    if parser.engine == "pyarrow":
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")
