"""
Internal module for console introspection
"""

# [MODULE DOC]
# HELPER: internal module for console and interactive environment introspection, used by
#   display code to decide output width and height.
# Public contents: get_console_size, in_interactive_session and in_ipython_frontend. The
#   display options and the default option lookup are imported lazily from the top level
#   package inside functions.
# Import-time side effects: none.

from __future__ import annotations

from shutil import get_terminal_size


def get_console_size() -> tuple[int | None, int | None]:
    # [FUNCTION DOC]
    # Determine the console size for display purposes. Takes no arguments. Returns a
    #   pair (width, height) where each element is an integer or none.
    # Behavior: reads the configured display width and the configured maximum row count
    #   options. If running in an interactive session: in an IPython graphical frontend
    #   (which cannot detect terminal size) uses the default values of those two
    #   options; in a plain terminal asks the operating system for the terminal size. If
    #   not interactive, the detected size is none for both. The result is, for each
    #   dimension, the configured option value if it is truthy, otherwise the detected
    #   value.
    # KNOWN ISSUE: the docstring says none is returned for a non-interactive session,
    #   but configured option values take precedence, so a pair of nones only results
    #   when the options are themselves unset (set to none to request auto detection)
    #   and the session is non-interactive.
    """
    Return console size as tuple = (width, height).

    Returns (None,None) in non-interactive session.
    """
    from pandas import get_option

    display_width = get_option("display.width")
    display_height = get_option("display.max_rows")

    # Consider
    # interactive shell terminal, can detect term size
    # interactive non-shell terminal (ipnb/ipqtconsole), cannot detect term
    # size non-interactive script, should disregard term size

    # in addition
    # width,height have default values, but setting to 'None' signals
    # should use Auto-Detection, But only in interactive shell-terminal.
    # Simple. yeah.

    if in_interactive_session():
        if in_ipython_frontend():
            # sane defaults for interactive non-shell terminal
            # match default for width,height in config_init
            from pandas._config.config import get_default_val

            terminal_width = get_default_val("display.width")
            terminal_height = get_default_val("display.max_rows")
        else:
            # pure terminal
            terminal_width, terminal_height = get_terminal_size()
    else:
        terminal_width, terminal_height = None, None

    # Note if the User sets width/Height to None (auto-detection)
    # and we're in a script (non-inter), this will return (None,None)
    # caller needs to deal.
    return display_width or terminal_width, display_height or terminal_height


# ----------------------------------------------------------------------
# Detect our environment


def in_interactive_session() -> bool:
    # [FUNCTION DOC]
    # Report whether the code runs in an interactive Python shell. Returns a boolean.
    #   True when running under IPython (detected through a name that IPython injects
    #   into builtins), otherwise the result of the nested main-module check. Catches
    #   the name error that occurs when not in IPython.
    """
    Check if we're running in an interactive shell.

    Returns
    -------
    bool
        True if running under python/ipython interactive shell.
    """
    from pandas import get_option

    def check_main() -> bool:
        # [FUNCTION DOC]
        # HELPER: nested check used by the interactive session detector. Tries to import
        #   the main module; if that is impossible returns the simulated-interactive
        #   option value. Otherwise returns true when the main module has no file
        #   attribute (as in an interactive prompt) or, failing that, the simulated-
        #   interactive option value.
        try:
            import __main__ as main
        except ModuleNotFoundError:
            return get_option("mode.sim_interactive")
        return not hasattr(main, "__file__") or get_option("mode.sim_interactive")

    try:
        # error: Name '__IPYTHON__' is not defined
        return __IPYTHON__ or check_main()  # type: ignore[name-defined]
    except NameError:
        return check_main()


def in_ipython_frontend() -> bool:
    # [FUNCTION DOC]
    # Report whether the code runs inside an IPython frontend that talks over a zmq
    #   kernel connection (notebook, qtconsole). Returns a boolean: true if the lower
    #   cased type name of the injected interactive shell object contains zmq; false if
    #   that object does not exist (name error) or does not match.
    """
    Check if we're inside an IPython zmq frontend.

    Returns
    -------
    bool
    """
    try:
        # error: Name 'get_ipython' is not defined
        ip = get_ipython()  # type: ignore[name-defined]
        return "zmq" in str(type(ip)).lower()
    except NameError:
        pass

    return False
