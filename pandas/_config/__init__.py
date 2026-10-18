"""
pandas._config is considered explicitly upstream of everything else in pandas,
should have no intra-pandas dependencies.

importing `dates` and `display` ensures that keys needed by _libs
are initialized.
"""

# [PACKAGE DOC]
# Option (configuration) subsystem of pandas. Deliberately the lowest layer of
# the library: it must not depend on other parts of pandas, so the compiled
# extension modules can use it. Submodules: config (the option registry and
# the user-facing functions), dates (registers the day-first and year-first
# parsing options), display (registers the console encoding option and holds the
# console encoding detector) and localization (locale helpers, not imported
# here).
# Exports (listed in __all__): the config module itself, describe_option,
# detect_console_encoding, get_option, option_context, options (the attribute
# style access object), reset_option and set_option. Also provides three small
# feature-flag accessors defined below, and imports a private registry mapping
# (the global config dictionary) for them.
# Import-time side effects: importing the dates and display submodules
# registers the options display.date_dayfirst, display.date_yearfirst and
# display.encoding in the global registry (the encoding default is computed
# by inspecting the console and locale at import time).
# KNOWN ISSUE: __all__ is placed before the import statements it refers to,
# which is legal but unusual.
__all__ = [
    "config",
    "describe_option",
    "detect_console_encoding",
    "get_option",
    "option_context",
    "options",
    "reset_option",
    "set_option",
]
from pandas._config import config
from pandas._config import dates  # pyright: ignore[reportUnusedImport]  # noqa: F401
from pandas._config.config import (
    _global_config,
    describe_option,
    get_option,
    option_context,
    options,
    reset_option,
    set_option,
)
from pandas._config.display import detect_console_encoding


def using_string_dtype() -> bool:
    # [FUNCTION DOC]
    # Returns the current boolean value of the future option that enables
    # inferring the dedicated string dtype for string data. Reads the value
    # directly from the stored option tree under the future section, without
    # validation or deprecation handling. Raises a key error if the option has
    # not yet been registered (before the core configuration init module runs).
    _mode_options = _global_config["future"]
    return _mode_options["infer_string"]


def using_python_scalars() -> bool:
    # [FUNCTION DOC]
    # Returns the current boolean value of the future option controlling whether
    # reductions and element access return builtin Python scalars. Read
    # directly from the stored option tree like the string dtype accessor;
    # raises a key error if the option is not yet registered.
    _mode_options = _global_config["future"]
    return _mode_options["python_scalars"]


def is_nan_na() -> bool:
    # [FUNCTION DOC]
    # Returns true when floating point NaN values are treated as the same as the
    # missing-value marker. This is the logical negation of the future option
    # that asks to distinguish NaN from the missing marker; read directly from
    # the stored option tree, raising a key error if not yet registered.
    _mode_options = _global_config["future"]
    return not _mode_options["distinguish_nan_and_na"]
