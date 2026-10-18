# [MODULE DOC]
# HELPER: support for optional third-party dependencies. Public contents:
# VERSIONS (mapping from import name to the minimum supported version string;
# the file's comment says several other files must be updated in step),
# INSTALL_MAPPING (import name to installation package name where they differ),
# get_version and import_optional_dependency (the main entry point, with two
# typing overloads). Relationships: uses the library's own version parser and
# the helper that finds the right warning stack level. Import-time side
# effects: none beyond imports.
from __future__ import annotations

import importlib
import sys
from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)
import warnings

from pandas.util._exceptions import find_stack_level

from pandas.util.version import Version

if TYPE_CHECKING:
    import types

# Update install.rst, actions-311-minimum_versions.yaml,
# deps_minimum.toml & pyproject.toml when updating versions!

VERSIONS = {
    "adbc-driver-postgresql": "1.2.0",
    "adbc-driver-sqlite": "1.2.0",
    "bs4": "4.12.3",
    "bottleneck": "1.4.2",
    "fastparquet": "2024.11.0",
    "fsspec": "2024.10.0",
    "html5lib": "1.1",
    "hypothesis": "6.116.0",
    "gcsfs": "2024.10.0",
    "jinja2": "3.1.5",
    "lxml.etree": "5.3.0",
    "matplotlib": "3.9.3",
    "numba": "0.60.0",
    "numexpr": "2.10.2",
    "odfpy": "1.4.1",
    "openpyxl": "3.1.5",
    "psycopg2": "2.9.10",  # (dt dec pq3 ext lo64)
    "pymysql": "1.1.1",
    "pyarrow": "13.0.0",
    "pyiceberg": "0.8.1",
    "pyreadstat": "1.2.8",
    "pytest": "8.3.4",
    "python-calamine": "0.3.0",
    "pytz": "2020.1",  # keep this pinned (https://github.com/pandas-dev/pandas/pull/65133)
    "pyxlsb": "1.0.10",
    "s3fs": "2024.10.0",
    "scipy": "1.14.1",
    "sqlalchemy": "2.0.36",
    "tables": "3.10.1",
    "tabulate": "0.9.0",
    "xarray": "2024.10.0",
    "xlrd": "2.0.1",
    "xlsxwriter": "3.2.0",
    "zstandard": "0.23.0",
    "qtpy": "2.4.2",
    "pyqt5": "5.15.9",
}

# A mapping from import name to package name (on PyPI) for packages where
# these two names are different.

INSTALL_MAPPING = {
    "bs4": "beautifulsoup4",
    "bottleneck": "Bottleneck",
    "jinja2": "Jinja2",
    "lxml.etree": "lxml",
    "odf": "odfpy",
    "python_calamine": "python-calamine",
    "sqlalchemy": "SQLAlchemy",
    "tables": "pytables",
}


def get_version(module: types.ModuleType) -> str:
    # [FUNCTION DOC]
    # HELPER: reads the version string of an imported module. Parameter module is
    # the module object. Returns its version attribute; for the psycopg2 module
    # only the first whitespace-separated word is kept (it appends extra build
    # info). Raises an import error if the module has no version attribute.
    version = getattr(module, "__version__", None)

    if version is None:
        raise ImportError(f"Can't determine version for {module.__name__}")
    if module.__name__ == "psycopg2":
        # psycopg2 appends " (dt dec pq3 ext lo64)" to it's version
        version = version.split()[0]
    return version


@overload
def import_optional_dependency(
    name: str,
    extra: str = ...,
    min_version: str | None = ...,
    *,
    errors: Literal["raise"] = ...,
) -> types.ModuleType:
    # [FUNCTION DOC]
    # Typing overload stub: when errors is raise (the default) the result is
    # always a module. See the implementation below for behavior.
    ...


@overload
def import_optional_dependency(
    name: str,
    extra: str = ...,
    min_version: str | None = ...,
    *,
    errors: Literal["warn", "ignore"],
) -> types.ModuleType | None:
    # [FUNCTION DOC]
    # Typing overload stub: when errors is warn or ignore (required keyword) the
    # result may be none. See the implementation below for behavior.
    ...


def import_optional_dependency(
    name: str,
    extra: str = "",
    min_version: str | None = None,
    *,
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> types.ModuleType | None:
    # [FUNCTION DOC]
    # Imports an optional dependency and checks its minimum version.
    # Parameters: name (dotted module name; for a submodule the version is taken
    # from the top-level package), extra (text added to the failure message,
    # default empty), min_version (overrides the table's minimum; default none
    # means use the table entry, if any) and the keyword-only errors (raise, warn
    # or ignore; default raise; any other value fails an assertion).
    # Flow: look up the installation name for messages; try the import; if it
    # fails, raise an import error with an install hint (chained to the original)
    # when errors is raise, otherwise return none. If the minimum version is
    # known, read the installed version (import error if it cannot be
    # determined) and compare; if too old: errors raise -> raises an import
    # error with a message giving the required and installed versions; errors warn
    # -> emits a user warning at the caller's stack level and returns none;
    # errors ignore -> returns none. Otherwise returns the module.
    # KNOWN ISSUE: the existing docstring says that with errors set to ignore a
    # module that is too old is still returned, but the code returns none in that
    # case.
    # KNOWN ISSUE: the existing docstring's Returns section says none is
    # returned "when errors is False", but errors takes string values.
    # KNOWN ISSUE: the version table entry is looked up by the parent package
    # name, so the table key lxml.etree is never matched for the submodule name
    # (the lookup uses the part before the first dot), meaning that entry is
    # effectively keyed by a name that is not used for the lookup.
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found or its version is too old.

        * raise : Raise an ImportError
        * warn : Only applicable when a module's version is to old.
          Warns that the version is too old and returns None
        * ignore: If the module is not installed, return None, otherwise,
          return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``errors="ignore"`` (see. ``io/html.py``)
    min_version : str, default None
        Specify a minimum version that is different from the global pandas
        minimum version required.
    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `errors`
        is False, or when the package's version is too old and `errors`
        is ``'warn'`` or ``'ignore'``.
    """
    assert errors in {"warn", "raise", "ignore"}

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"`Import {install_name}` failed. {extra} "
        f"Use pip or conda to install the {install_name} package."
    )
    try:
        module = importlib.import_module(name)
    except ImportError as err:
        if errors == "raise":
            raise ImportError(msg) from err
        return None

    # Handle submodules: if we have submodule, grab parent module from sys.modules
    parent = name.split(".", maxsplit=1)[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        version = get_version(module_to_get)
        if version and Version(version) < Version(minimum_version):
            msg = (
                f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
                f"(version '{version}' currently installed)."
            )
            if errors == "warn":
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                return None
            elif errors == "raise":
                raise ImportError(msg)
            else:
                return None

    return module
