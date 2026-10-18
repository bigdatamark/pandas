# [MODULE DOC]
# Pytest configuration for the scripts test directory. Its only purpose is to
# register the strict-data-files command line option, so that running these
# tests in isolation does not fail on the option that the project
# configuration always passes. Defined and really used in the main pandas
# conftest module.
# pyproject.toml defines addopts: --strict-data-files
# strict-data-files is defined & used in pandas/conftest.py
def pytest_addoption(parser) -> None:
    # [FUNCTION DOC]
    # HELPER: pytest hook registering a no-op option. Input: parser, the
    # pytest option parser. Adds a boolean flag option named strict data files
    # (help text says unused) that has no effect here. Returns nothing.
    parser.addoption(
        "--strict-data-files",
        action="store_true",
        help="Unused",
    )
