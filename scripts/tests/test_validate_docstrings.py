# [MODULE DOC]
# Tests for the docstring validation script. Contents: a sample class whose methods each have a
# deliberately bad docstring (test data, not real documentation); tests of the single-object
# validator against it; tests of whole-API validation and result reporting with the validator
# monkeypatched; tests of the API-page parser on an in-memory documentation sample; a test that the
# flake8 helper yields nothing for two real string-method docstrings; and tests of the main
# entry point's exit statuses with mocked validators.
import io
import textwrap

import pytest

from scripts import validate_docstrings


# fmt: off
class BadDocstrings:
    # [CLASS DOC]
    # HELPER: test data. A class whose methods are named after the specific docstring problem each one
    # demonstrates (private class mentioned, redundant pandas prefix in see-also, redundant imports in
    # examples, unused import, missing whitespace around operator, wrong indentation, missing
    # whitespace after comma, underscored array-like, example that leaves files behind). The method
    # bodies are empty; only the docstrings matter, and they must not be fixed.
    """Everything here has a bad docstring"""

    def private_classes(self) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Docstring mentions a private base class name.
        """
        This mentions NDFrame, which is not correct.
        """

    def prefix_pandas(self) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Docstring has a see-also entry with a redundant pandas prefix.
        """
        Have `pandas` prefix in See Also section.

        See Also
        --------
        pandas.Series.rename : Alter Series index labels or name.
        DataFrame.head : The first `n` rows of the caller object.
        """

    def redundant_import(self, paramx=None, paramy=None) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Docstring examples import numpy and pandas explicitly, which is redundant.
        """
        A sample DataFrame method.

        Should not import numpy and pandas.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = pd.DataFrame(np.ones((3, 3)),
        ...                   columns=('a', 'b', 'c'))
        >>> df.all(axis=1)
        0    True
        1    True
        2    True
        dtype: bool
        >>> df.all(bool_only=True)
        Series([], dtype: bool)
        """

    def unused_import(self) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Example imports pandas under another name that is never used, and uses
        # undefined short names.
        """
        Examples
        --------
        >>> import pandas as pdf
        >>> df = pd.DataFrame(np.ones((3, 3)), columns=('a', 'b', 'c'))
        """

    def missing_whitespace_around_arithmetic_operator(self) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Example has an arithmetic expression without surrounding whitespace.
        """
        Examples
        --------
        >>> 2+5
        7
        """

    def indentation_is_not_a_multiple_of_four(self) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Example has a continuation line indented by two spaces.
        """
        Examples
        --------
        >>> if 2 + 5:
        ...   pass
        """

    def missing_whitespace_after_comma(self) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Example has missing whitespace after commas.
        """
        Examples
        --------
        >>> df = pd.DataFrame(np.ones((3,3)),columns=('a','b', 'c'))
        """

    def write_array_like_with_hyphen_not_underscore(self) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Docstring uses the underscored spelling of array-like.
        """
        In docstrings, use array-like over array_like
        """

    def leftover_files(self) -> None:
        # [FUNCTION DOC]
        # HELPER: test data. Example creates a file in the working directory and never removes it.
        """
        Examples
        --------
        >>> import pathlib
        >>> pathlib.Path("foo.txt").touch()
        """
# fmt: on


class TestValidator:
    # [CLASS DOC]
    # Test class for the single-object validation function and the whole-API validation and printing
    # functions. Has one private helper to build dotted import paths into this module.
    def _import_path(self, klass=None, func=None):
        # [FUNCTION DOC]
        # HELPER: builds the dotted path of an object defined in this module. Inputs: klass, optional class
        # name; func, optional function name. Returns the module path with the class and function names
        # appended when given.
        """
        Build the required import path for tests in this module.

        Parameters
        ----------
        klass : str
            Class name of object in module.
        func : str
            Function name of object in module.

        Returns
        -------
        str
            Import path of specified object in this module
        """
        base_path = "scripts.tests.test_validate_docstrings"

        if klass:
            base_path = f"{base_path}.{klass}"

        if func:
            base_path = f"{base_path}.{func}"

        return base_path

    def test_bad_class(self, capsys) -> None:
        # [FUNCTION DOC]
        # Verifies that validating the whole sample bad class yields a non-empty list of errors.
        errors = validate_docstrings.pandas_validate(
            self._import_path(klass="BadDocstrings")
        )["errors"]
        assert isinstance(errors, list)
        assert errors

    @pytest.mark.parametrize(
        "klass,func,msgs",
        [
            (
                "BadDocstrings",
                "private_classes",
                (
                    "Private classes (NDFrame) should not be mentioned in public "
                    "docstrings",
                ),
            ),
            (
                "BadDocstrings",
                "prefix_pandas",
                (
                    "pandas.Series.rename in `See Also` section "
                    "does not need `pandas` prefix",
                ),
            ),
            # Examples tests
            (
                "BadDocstrings",
                "redundant_import",
                ("Do not import numpy, as it is imported automatically",),
            ),
            (
                "BadDocstrings",
                "redundant_import",
                ("Do not import pandas, as it is imported automatically",),
            ),
            (
                "BadDocstrings",
                "unused_import",
                (
                    "flake8 error: line 1, col 1: F401 'pandas as pdf' "
                    "imported but unused",
                ),
            ),
            (
                "BadDocstrings",
                "missing_whitespace_around_arithmetic_operator",
                (
                    "flake8 error: line 1, col 2: "
                    "E226 missing whitespace around arithmetic operator",
                ),
            ),
            (
                "BadDocstrings",
                "indentation_is_not_a_multiple_of_four",
                # with flake8 3.9.0, the message ends with four spaces,
                #  whereas in earlier versions, it ended with "four"
                (
                    "flake8 error: line 2, col 3: E111 indentation is not a "
                    "multiple of 4",
                ),
            ),
            (
                "BadDocstrings",
                "missing_whitespace_after_comma",
                ("flake8 error: line 1, col 33: E231 missing whitespace after ','",),
            ),
            (
                "BadDocstrings",
                "write_array_like_with_hyphen_not_underscore",
                ("Use 'array-like' rather than 'array_like' in docstrings",),
            ),
        ],
    )
    def test_bad_docstrings(self, capsys, klass, func, msgs) -> None:
        # [FUNCTION DOC]
        # Verifies, for each sample method, that the validation result contains the expected message text
        # among the joined error messages: private class, pandas prefix, redundant numpy and pandas imports,
        # unused import, spacing and indentation style errors from the example linter (with line and column),
        # and underscored array-like. Parametrized over class, method and expected message fragments.
        result = validate_docstrings.pandas_validate(
            self._import_path(klass=klass, func=func)
        )
        for msg in msgs:
            assert msg in " ".join([err[1] for err in result["errors"]])

    def test_validate_all_ignore_deprecated(self, monkeypatch) -> None:
        # [FUNCTION DOC]
        # Verifies that when the single-object validator is replaced by a stub returning a deprecated
        # object, whole-API validation with the ignore-deprecated option returns an empty result.
        monkeypatch.setattr(
            validate_docstrings,
            "pandas_validate",
            lambda func_name: {
                "docstring": "docstring1",
                "errors": [
                    ("ER01", "err desc"),
                    ("ER02", "err desc"),
                    ("ER03", "err desc"),
                ],
                "warnings": [],
                "examples_errors": "",
                "deprecated": True,
            },
        )
        result = validate_docstrings.validate_all(prefix=None, ignore_deprecated=True)
        assert len(result) == 0

    def test_validate_all_ignore_errors(self, monkeypatch):
        # [FUNCTION DOC]
        # Verifies error-ignoring in the whole-API printer with stubs for the validator and the API listing
        # (two objects, three errors each). With a global ignore of one code the returned count is two objects
        # times two remaining errors; adding an object-specific ignore of another code removes one more, and an
        # ignore entry for an error not raised on that object has no effect on the count.
        monkeypatch.setattr(
            validate_docstrings,
            "pandas_validate",
            lambda func_name: {
                "docstring": "docstring1",
                "errors": [
                    ("ER01", "err desc"),
                    ("ER02", "err desc"),
                    ("ER03", "err desc"),
                ],
                "warnings": [],
                "examples_errors": "",
                "deprecated": True,
                "file": "file1",
                "file_line": "file_line1",
            },
        )
        monkeypatch.setattr(
            validate_docstrings,
            "get_all_api_items",
            lambda: [
                (
                    "pandas.DataFrame.align",
                    "func",
                    "current_section",
                    "current_subsection",
                ),
                (
                    "pandas.Index.all",
                    "func",
                    "current_section",
                    "current_subsection",
                ),
            ],
        )

        exit_status = validate_docstrings.print_validate_all_results(
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={None: {"ER03"}},
        )
        # two functions * two not ignored errors
        assert exit_status == 2 * 2

        exit_status = validate_docstrings.print_validate_all_results(
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={
                None: {"ER03"},
                "pandas.DataFrame.align": {"ER01"},
                # ignoring an error that is not requested should be of no effect
                "pandas.Index.all": {"ER03"},
            },
        )
        # two functions * two not global ignored errors - one function ignored error
        assert exit_status == 2 * 2 - 1


class TestApiItems:
    # [CLASS DOC]
    # Tests for the API documentation page parser, using an in-memory page with two modules, sections
    # and subsections. Provides the sample page as a property.
    @property
    def api_doc(self):
        # [FUNCTION DOC]
        # HELPER: property returning a fresh in-memory text stream over the sample documentation page
        # (itertools cycle and count under one subsection, chain under another; random seed and randint
        # under a second section).
        return io.StringIO(
            textwrap.dedent(
                """
            .. currentmodule:: itertools

            Itertools
            ---------

            Infinite
            ~~~~~~~~

            .. autosummary::

                cycle
                count

            Finite
            ~~~~~~

            .. autosummary::

                chain

            .. currentmodule:: random

            Random
            ------

            All
            ~~~

            .. autosummary::

                seed
                randint
            """
            )
        )

    @pytest.mark.parametrize(
        "idx,name",
        [
            (0, "itertools.cycle"),
            (1, "itertools.count"),
            (2, "itertools.chain"),
            (3, "random.seed"),
            (4, "random.randint"),
        ],
    )
    def test_item_name(self, idx, name) -> None:
        # [FUNCTION DOC]
        # Verifies the first element of each yielded item is the module-qualified name, in order.
        # Parametrized over index and expected name.
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][0] == name

    @pytest.mark.parametrize(
        "idx,func",
        [(0, "cycle"), (1, "count"), (2, "chain"), (3, "seed"), (4, "randint")],
    )
    def test_item_function(self, idx, func) -> None:
        # [FUNCTION DOC]
        # Verifies the second element of each yielded item is the callable with the expected name.
        # Parametrized over index and expected function name.
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert callable(result[idx][1])
        assert result[idx][1].__name__ == func

    @pytest.mark.parametrize(
        "idx,section",
        [
            (0, "Itertools"),
            (1, "Itertools"),
            (2, "Itertools"),
            (3, "Random"),
            (4, "Random"),
        ],
    )
    def test_item_section(self, idx, section) -> None:
        # [FUNCTION DOC]
        # Verifies the third element is the expected section name for each item.
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][2] == section

    @pytest.mark.parametrize(
        "idx,subsection",
        [(0, "Infinite"), (1, "Infinite"), (2, "Finite"), (3, "All"), (4, "All")],
    )
    def test_item_subsection(self, idx, subsection) -> None:
        # [FUNCTION DOC]
        # Verifies the fourth element is the expected subsection name for each item.
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][3] == subsection


class TestPandasDocstringClass:
    # [CLASS DOC]
    # Tests for the validator subclass.
    @pytest.mark.parametrize(
        "name", ["pandas.Series.str.isdecimal", "pandas.Series.str.islower"]
    )
    def test_encode_content_write_to_file(self, name) -> None:
        # [FUNCTION DOC]
        # Regression test (issue 25466): running the example linter for two real string-method
        # docstrings yields no style errors. Parametrized over the two dotted names.
        # GH25466
        docstr = validate_docstrings.PandasDocstring(name).validate_pep8()
        # the list of pep8 errors should be empty
        assert not list(docstr)


class TestMainFunction:
    # [CLASS DOC]
    # Tests for the main entry point's exit statuses, with the underlying validators replaced by
    # stubs through monkeypatching.
    def test_exit_status_for_main(self, monkeypatch) -> None:
        # [FUNCTION DOC]
        # Verifies that for a single object with three stubbed errors, main returns three.
        monkeypatch.setattr(
            validate_docstrings,
            "pandas_validate",
            lambda func_name: {
                "docstring": "docstring1",
                "errors": [
                    ("ER01", "err desc"),
                    ("ER02", "err desc"),
                    ("ER03", "err desc"),
                ],
                "examples_errs": "",
            },
        )
        exit_status = validate_docstrings.main(
            func_name="docstring1",
            prefix=None,
            output_format="default",
            ignore_deprecated=False,
            ignore_errors={},
        )
        assert exit_status == 3

    def test_exit_status_errors_for_validate_all(self, monkeypatch) -> None:
        # [FUNCTION DOC]
        # Verifies that for the whole API with stubbed results (three and two errors) in default output
        # format, main returns five.
        monkeypatch.setattr(
            validate_docstrings,
            "validate_all",
            lambda prefix, ignore_deprecated=False, ignore_functions=None: {
                "docstring1": {
                    "errors": [
                        ("ER01", "err desc"),
                        ("ER02", "err desc"),
                        ("ER03", "err desc"),
                    ],
                    "file": "module1.py",
                    "file_line": 23,
                },
                "docstring2": {
                    "errors": [("ER04", "err desc"), ("ER05", "err desc")],
                    "file": "module2.py",
                    "file_line": 925,
                },
            },
        )
        exit_status = validate_docstrings.main(
            func_name=None,
            prefix=None,
            output_format="default",
            ignore_deprecated=False,
            ignore_errors={},
        )
        assert exit_status == 5

    def test_no_exit_status_noerrors_for_validate_all(self, monkeypatch) -> None:
        # [FUNCTION DOC]
        # Verifies that when the whole-API stub has no errors (only a warning), main returns zero.
        monkeypatch.setattr(
            validate_docstrings,
            "validate_all",
            lambda prefix, ignore_deprecated=False, ignore_functions=None: {
                "docstring1": {"errors": [], "warnings": [("WN01", "warn desc")]},
                "docstring2": {"errors": []},
            },
        )
        exit_status = validate_docstrings.main(
            func_name=None,
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={},
        )
        assert exit_status == 0

    def test_exit_status_for_validate_all_json(self, monkeypatch) -> None:
        # [FUNCTION DOC]
        # Verifies that in JSON output format main returns zero even though the stubbed results contain
        # errors.
        monkeypatch.setattr(
            validate_docstrings,
            "validate_all",
            lambda prefix, ignore_deprecated=False, ignore_functions=None: {
                "docstring1": {
                    "errors": [
                        ("ER01", "err desc"),
                        ("ER02", "err desc"),
                        ("ER03", "err desc"),
                    ]
                },
                "docstring2": {"errors": [("ER04", "err desc"), ("ER05", "err desc")]},
            },
        )
        exit_status = validate_docstrings.main(
            func_name=None,
            output_format="json",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={},
        )
        assert exit_status == 0

    def test_errors_param_filters_errors(self, monkeypatch) -> None:
        # [FUNCTION DOC]
        # Verifies error filtering through the ignore mapping: with the global ignore of two codes, the
        # stub's errors reduce to three remaining (one per object); ignoring two other codes leaves one.
        # Replaces the error message table with three entries.
        monkeypatch.setattr(
            validate_docstrings,
            "validate_all",
            lambda prefix, ignore_deprecated=False, ignore_functions=None: {
                "Series.foo": {
                    "errors": [
                        ("ER01", "err desc"),
                        ("ER02", "err desc"),
                        ("ER03", "err desc"),
                    ],
                    "file": "series.py",
                    "file_line": 142,
                },
                "DataFrame.bar": {
                    "errors": [("ER01", "err desc"), ("ER02", "err desc")],
                    "file": "frame.py",
                    "file_line": 598,
                },
                "Series.foobar": {
                    "errors": [("ER01", "err desc")],
                    "file": "series.py",
                    "file_line": 279,
                },
            },
        )
        monkeypatch.setattr(
            validate_docstrings,
            "ERROR_MSGS",
            {
                "ER01": "err desc",
                "ER02": "err desc",
                "ER03": "err desc",
            },
        )
        exit_status = validate_docstrings.main(
            func_name=None,
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={None: {"ER02", "ER03"}},
        )
        assert exit_status == 3

        exit_status = validate_docstrings.main(
            func_name=None,
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={None: {"ER01", "ER02"}},
        )
        assert exit_status == 1
