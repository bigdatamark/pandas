"""
Pyperclip

A cross-platform clipboard module for Python,
with copy & paste functions for plain text.
By Al Sweigart al@inventwithpython.com
Licence at LICENSES/PYPERCLIP_LICENSE

Usage:
  import pyperclip
  pyperclip.copy('The text to be copied to the clipboard.')
  spam = pyperclip.paste()

  if not pyperclip.is_available():
    print("Copy functionality unavailable!")

On Windows, no additional modules are needed.
On Mac, the pyobjc module is used, falling back to the pbcopy and pbpaste cli
    commands. (These commands should come with OS X.).
On Linux, install xclip, xsel, or wl-clipboard (for "wayland" sessions) via
package manager.
For example, in Debian:
    sudo apt-get install xclip
    sudo apt-get install xsel
    sudo apt-get install wl-clipboard

Otherwise on Linux, you will need the PyQt5 modules installed.

This module does not work with PyGObject yet.

Cygwin is currently not supported.

Security Note: This module runs programs with these names:
    - pbcopy
    - pbpaste
    - xclip
    - xsel
    - wl-copy/wl-paste
    - klipper
    - qdbus
A malicious user could rename or add programs with these names, tricking
Pyperclip into running them with whatever permissions the Python process has.

"""

# [PACKAGE DOC]
# HELPER: vendored copy of the third-party clipboard library, used by the
# pandas clipboard reader and writer. It provides cross-platform plain-text
# copy and paste through platform-specific backends: macOS (pyobjc or the
# pbcopy/pbpaste commands), Windows (ctypes calls), Windows Subsystem for
# Linux (clip.exe and powershell), Cygwin (device file), and Linux (wl-clipboard,
# xsel, xclip, klipper via qdbus, or Qt bindings). The backend is chosen
# lazily on first use of copy or paste, or explicitly through the setter
# function. Public names (all): copy, paste, wait-for-paste helpers,
# set_clipboard and determine_clipboard; pandas also exposes the aliases
# clipboard_get and clipboard_set for paste and copy.
# KNOWN ISSUE: the two aliases are bound at import time to the lazy stub
# functions, not to the real backend functions, so every call through them
# re-runs backend detection (the stubs rebind the module-level copy and paste
# but the aliases are never updated), and a later explicit backend selection
# is only honoured through the module-level names.
# Import-time side effects: reads the DISPLAY environment variable into a
# module constant and installs the lazy stub functions as the initial copy and
# paste; no clipboard access happens at import.

__version__ = "1.8.2"


import contextlib
import ctypes
from ctypes import (
    c_size_t,
    c_wchar,
    c_wchar_p,
    get_errno,
    sizeof,
)
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings

from pandas.errors import (
    PyperclipException,
    PyperclipWindowsException,
)
from pandas.util._exceptions import find_stack_level

# `import PyQt4` sys.exit()s if DISPLAY is not in the environment.
# Thus, we need to detect the presence of $DISPLAY manually
# and not load PyQt4 if it is absent.
HAS_DISPLAY = os.getenv("DISPLAY")

EXCEPT_MSG = """
    Pyperclip could not find a copy/paste mechanism for your system.
    For more information, please visit
    https://pyperclip.readthedocs.io/en/latest/index.html#not-implemented-error
    """

ENCODING = "utf-8"


class PyperclipTimeoutException(PyperclipException):
    # [CLASS DOC]
    # Exception raised by the wait-for-paste helpers when the timeout elapses
    # before the clipboard satisfies the wait condition. Subclass of the
    # pandas clipboard base exception; adds no state or behavior.
    pass


def _stringifyText(text) -> str:
    # [FUNCTION DOC]
    # HELPER: validates and converts a value to text for copying. Input: any
    # value; only text, integer, float and boolean values are accepted and are
    # converted to their string form. Returns the string. Raises the clipboard
    # base exception for any other type.
    acceptedTypes = (str, int, float, bool)
    if not isinstance(text, acceptedTypes):
        raise PyperclipException(
            f"only str, int, float, and bool values "
            f"can be copied to the clipboard, not {type(text).__name__}"
        )
    return str(text)


def init_osx_pbcopy_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for macOS using the pbcopy and pbpaste commands.
    # Returns a pair of (copy function, paste function). No side effects until
    # the returned functions are called.
    def copy_osx_pbcopy(text):
        # [FUNCTION DOC]
        # HELPER: copies text (converted to a string, raising the clipboard
        # exception for unsupported types) by piping utf-8 bytes into the
        # copy command. Returns nothing.
        text = _stringifyText(text)  # Converts non-str values to str.
        with subprocess.Popen(
            ["pbcopy", "w"], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_osx_pbcopy():
        # [FUNCTION DOC]
        # HELPER: runs the paste command and returns its output decoded as
        # utf-8 text.
        with subprocess.Popen(
            ["pbpaste", "r"], stdout=subprocess.PIPE, close_fds=True
        ) as p:
            stdout = p.communicate()[0]
        return stdout.decode(ENCODING)

    return copy_osx_pbcopy, paste_osx_pbcopy


def init_osx_pyobjc_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for macOS using the pyobjc bindings (modules
    # imported at selection time into module globals by the detection
    # function). Returns a pair of (copy function, paste function).
    def copy_osx_pyobjc(text):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception for
        # unsupported types), encodes it as utf-8 data and places it on the
        # general pasteboard as string type. Returns nothing.
        """Copy string argument to clipboard"""
        text = _stringifyText(text)  # Converts non-str values to str.
        newStr = Foundation.NSString.stringWithString_(text).nsstring()
        newData = newStr.dataUsingEncoding_(Foundation.NSUTF8StringEncoding)
        board = AppKit.NSPasteboard.generalPasteboard()
        board.declareTypes_owner_([AppKit.NSStringPboardType], None)
        board.setData_forType_(newData, AppKit.NSStringPboardType)

    def paste_osx_pyobjc():
        # [FUNCTION DOC]
        # HELPER: returns the string content of the general pasteboard (may be
        # none if it holds no string).
        """Returns contents of clipboard"""
        board = AppKit.NSPasteboard.generalPasteboard()
        content = board.stringForType_(AppKit.NSStringPboardType)
        return content

    return copy_osx_pyobjc, paste_osx_pyobjc


def init_qt_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for Qt. Imports the application class from qtpy,
    # falling back to PyQt5 then PyQt4 (the last import error propagates), and
    # stores it in a module global. Reuses the running Qt application or
    # creates a new one (side effect). Returns a pair of (copy function, paste
    # function) bound to that application's clipboard.
    global QApplication
    # $DISPLAY should exist

    # Try to import from qtpy, but if that fails try PyQt5 then PyQt4
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            from PyQt4.QtGui import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    def copy_qt(text):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception on
        # unsupported types) and sets it as the Qt clipboard text.
        text = _stringifyText(text)  # Converts non-str values to str.
        cb = app.clipboard()
        cb.setText(text)

    def paste_qt() -> str:
        # [FUNCTION DOC]
        # HELPER: returns the Qt clipboard text as a string.
        cb = app.clipboard()
        return str(cb.text())

    return copy_qt, paste_qt


def init_xclip_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for X11 using the xclip command. Returns a pair
    # of (copy function, paste function). The clipboard selection is used by
    # default and the primary selection when requested.
    DEFAULT_SELECTION = "c"
    PRIMARY_SELECTION = "p"

    def copy_xclip(text, primary=False):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception on
        # unsupported types) and pipes it as utf-8 into xclip. primary
        # (default false) chooses the primary selection instead of the
        # clipboard selection.
        text = _stringifyText(text)  # Converts non-str values to str.
        selection = DEFAULT_SELECTION
        if primary:
            selection = PRIMARY_SELECTION
        with subprocess.Popen(
            ["xclip", "-selection", selection], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_xclip(primary=False):
        # [FUNCTION DOC]
        # HELPER: reads the clipboard (or the primary selection when primary,
        # default false, is true) through xclip and returns utf-8 decoded
        # text. Error output is captured and ignored, so an empty clipboard
        # yields an empty string.
        selection = DEFAULT_SELECTION
        if primary:
            selection = PRIMARY_SELECTION
        with subprocess.Popen(
            ["xclip", "-selection", selection, "-o"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]
        # Intentionally ignore extraneous output on stderr when clipboard is empty
        return stdout.decode(ENCODING)

    return copy_xclip, paste_xclip


def init_xsel_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for X11 using the xsel command. Returns a pair
    # of (copy function, paste function); uses the clipboard flag by default
    # and the primary-selection flag when requested.
    DEFAULT_SELECTION = "-b"
    PRIMARY_SELECTION = "-p"

    def copy_xsel(text, primary=False):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception on
        # unsupported types) and pipes it as utf-8 into xsel input mode.
        # primary (default false) selects the primary selection.
        text = _stringifyText(text)  # Converts non-str values to str.
        selection_flag = DEFAULT_SELECTION
        if primary:
            selection_flag = PRIMARY_SELECTION
        with subprocess.Popen(
            ["xsel", selection_flag, "-i"], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_xsel(primary=False):
        # [FUNCTION DOC]
        # HELPER: reads the clipboard (primary selection when primary, default
        # false, is true) through xsel output mode and returns utf-8 decoded
        # text.
        selection_flag = DEFAULT_SELECTION
        if primary:
            selection_flag = PRIMARY_SELECTION
        with subprocess.Popen(
            ["xsel", selection_flag, "-o"], stdout=subprocess.PIPE, close_fds=True
        ) as p:
            stdout = p.communicate()[0]
        return stdout.decode(ENCODING)

    return copy_xsel, paste_xsel


def init_wl_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for Wayland using the wl-copy and wl-paste
    # commands. Returns a pair of (copy function, paste function).
    PRIMARY_SELECTION = "-p"

    def copy_wl(text, primary=False):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception on
        # unsupported types). An empty string clears the clipboard through the
        # clear option and waits for the command; otherwise utf-8 text is piped
        # to the copy command. primary (default false) targets the primary
        # selection.
        # KNOWN ISSUE: the non-empty branch starts the process without a
        # context manager and never closes its pipes explicitly, unlike the
        # other backends.
        text = _stringifyText(text)  # Converts non-str values to str.
        args = ["wl-copy"]
        if primary:
            args.append(PRIMARY_SELECTION)
        if not text:
            args.append("--clear")
            subprocess.check_call(args, close_fds=True)
        else:
            p = subprocess.Popen(args, stdin=subprocess.PIPE, close_fds=True)
            p.communicate(input=text.encode(ENCODING))

    def paste_wl(primary=False):
        # [FUNCTION DOC]
        # HELPER: runs the paste command without a trailing newline (primary
        # selection when primary, default false, is true) and returns the
        # utf-8 decoded output.
        args = ["wl-paste", "-n"]
        if primary:
            args.append(PRIMARY_SELECTION)
        p = subprocess.Popen(args, stdout=subprocess.PIPE, close_fds=True)
        stdout, _stderr = p.communicate()
        return stdout.decode(ENCODING)

    return copy_wl, paste_wl


def init_klipper_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for KDE Klipper using the qdbus command.
    # Returns a pair of (copy function, paste function).
    def copy_klipper(text):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception on
        # unsupported types) and passes its utf-8 bytes as the argument of the
        # set-contents method over the message bus. Returns nothing.
        text = _stringifyText(text)  # Converts non-str values to str.
        with subprocess.Popen(
            [
                "qdbus",
                "org.kde.klipper",
                "/klipper",
                "setClipboardContents",
                text.encode(ENCODING),
            ],
            stdin=subprocess.PIPE,
            close_fds=True,
        ) as p:
            p.communicate(input=None)

    def paste_klipper():
        # [FUNCTION DOC]
        # HELPER: fetches clipboard contents over the message bus, decodes
        # utf-8 and strips the single trailing newline that Klipper always
        # appends. Returns the text.
        # KNOWN ISSUE: relies on assertions to verify the output is non-empty
        # and ends with a newline (disabled under optimised mode), and the
        # subsequent conditional strip is therefore redundant.
        with subprocess.Popen(
            ["qdbus", "org.kde.klipper", "/klipper", "getClipboardContents"],
            stdout=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]

        # Workaround for https://bugs.kde.org/show_bug.cgi?id=342874
        # TODO: https://github.com/asweigart/pyperclip/issues/43
        clipboardContents = stdout.decode(ENCODING)
        # even if blank, Klipper will append a newline at the end
        assert len(clipboardContents) > 0
        # make sure that newline is there
        assert clipboardContents.endswith("\n")
        if clipboardContents.endswith("\n"):
            clipboardContents = clipboardContents[:-1]
        return clipboardContents

    return copy_klipper, paste_klipper


def init_dev_clipboard_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for Cygwin using the clipboard device file.
    # Returns a pair of (copy function, paste function).
    def copy_dev_clipboard(text):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception on
        # unsupported types), warns when it is empty (effectively a no-op) or
        # contains carriage returns (unsupported), then writes it to the device
        # file as utf-8.
        text = _stringifyText(text)  # Converts non-str values to str.
        if text == "":
            warnings.warn(
                "Pyperclip cannot copy a blank string to the clipboard on Cygwin. "
                "This is effectively a no-op.",
                stacklevel=find_stack_level(),
            )
        if "\r" in text:
            warnings.warn(
                "Pyperclip cannot handle \\r characters on Cygwin.",
                stacklevel=find_stack_level(),
            )

        with open("/dev/clipboard", "w", encoding="utf-8") as fd:
            fd.write(text)

    def paste_dev_clipboard() -> str:
        # [FUNCTION DOC]
        # HELPER: reads and returns the full text of the clipboard device file.
        with open("/dev/clipboard", encoding="utf-8") as fd:
            content = fd.read()
        return content

    return copy_dev_clipboard, paste_dev_clipboard


def init_no_clipboard():
    # [FUNCTION DOC]
    # HELPER: fallback backend factory used when no mechanism is found.
    # Returns a pair of two unavailable-clipboard placeholder objects (one for
    # copy, one for paste) that are falsy and raise the clipboard exception
    # with the standard help message when called.
    class ClipboardUnavailable:
        # [CLASS DOC]
        # HELPER: placeholder callable standing in for copy and paste when no
        # clipboard backend exists. Stateless; calling it with any arguments
        # raises the clipboard exception carrying the help message; it
        # evaluates as false in boolean context.
        def __call__(self, *args, **kwargs):
            # [FUNCTION DOC]
            # Always raises the clipboard exception with the help message,
            # ignoring all arguments.
            raise PyperclipException(EXCEPT_MSG)

        def __bool__(self) -> bool:
            # [FUNCTION DOC]
            # Returns false so callers can test whether a mechanism exists.
            return False

    return ClipboardUnavailable(), ClipboardUnavailable()


# Windows-related clipboard functions:
class CheckedCall:
    # [CLASS DOC]
    # HELPER: wrapper around a foreign (ctypes) function that raises the
    # Windows clipboard exception when the call returns a falsy value while the
    # C error number is set. State: the wrapped function stored as attribute
    # f. Attribute assignment on the wrapper is forwarded to the wrapped
    # function (used to set argument and result types); only the initial
    # constructor assignment bypasses the forwarding.
    def __init__(self, f) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the function to wrap, bypassing the forwarding
        # attribute setter.
        super().__setattr__("f", f)

    def __call__(self, *args):
        # [FUNCTION DOC]
        # HELPER: calls the wrapped function with the given positional
        # arguments and returns its result; raises the Windows clipboard
        # exception naming the function if the result is falsy and the C error
        # number is non-zero.
        ret = self.f(*args)
        if not ret and get_errno():
            raise PyperclipWindowsException("Error calling " + self.f.__name__)
        return ret

    def __setattr__(self, key, value):
        # [FUNCTION DOC]
        # HELPER: forwards attribute assignment to the wrapped function (for
        # example its argument and return type declarations).
        setattr(self.f, key, value)


def init_windows_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for Windows using ctypes calls into the user,
    # kernel and C runtime libraries. Imports the Windows type aliases into
    # module globals, wraps the needed API functions with error checking and
    # declared signatures, and returns a pair of (copy function, paste
    # function). Only works on Windows (the Windows ctypes loader is absent
    # elsewhere).
    global HGLOBAL, LPVOID, DWORD, LPCSTR, INT
    global HWND, HINSTANCE, HMENU, BOOL, UINT, HANDLE
    from ctypes.wintypes import (
        BOOL,
        DWORD,
        HANDLE,
        HGLOBAL,
        HINSTANCE,
        HMENU,
        HWND,
        INT,
        LPCSTR,
        LPVOID,
        UINT,
    )

    windll = ctypes.windll
    msvcrt = ctypes.CDLL("msvcrt")

    safeCreateWindowExA = CheckedCall(windll.user32.CreateWindowExA)
    safeCreateWindowExA.argtypes = [
        DWORD,
        LPCSTR,
        LPCSTR,
        DWORD,
        INT,
        INT,
        INT,
        INT,
        HWND,
        HMENU,
        HINSTANCE,
        LPVOID,
    ]
    safeCreateWindowExA.restype = HWND

    safeDestroyWindow = CheckedCall(windll.user32.DestroyWindow)
    safeDestroyWindow.argtypes = [HWND]
    safeDestroyWindow.restype = BOOL

    OpenClipboard = windll.user32.OpenClipboard
    OpenClipboard.argtypes = [HWND]
    OpenClipboard.restype = BOOL

    safeCloseClipboard = CheckedCall(windll.user32.CloseClipboard)
    safeCloseClipboard.argtypes = []
    safeCloseClipboard.restype = BOOL

    safeEmptyClipboard = CheckedCall(windll.user32.EmptyClipboard)
    safeEmptyClipboard.argtypes = []
    safeEmptyClipboard.restype = BOOL

    safeGetClipboardData = CheckedCall(windll.user32.GetClipboardData)
    safeGetClipboardData.argtypes = [UINT]
    safeGetClipboardData.restype = HANDLE

    safeSetClipboardData = CheckedCall(windll.user32.SetClipboardData)
    safeSetClipboardData.argtypes = [UINT, HANDLE]
    safeSetClipboardData.restype = HANDLE

    safeGlobalAlloc = CheckedCall(windll.kernel32.GlobalAlloc)
    safeGlobalAlloc.argtypes = [UINT, c_size_t]
    safeGlobalAlloc.restype = HGLOBAL

    safeGlobalLock = CheckedCall(windll.kernel32.GlobalLock)
    safeGlobalLock.argtypes = [HGLOBAL]
    safeGlobalLock.restype = LPVOID

    safeGlobalUnlock = CheckedCall(windll.kernel32.GlobalUnlock)
    safeGlobalUnlock.argtypes = [HGLOBAL]
    safeGlobalUnlock.restype = BOOL

    wcslen = CheckedCall(msvcrt.wcslen)
    wcslen.argtypes = [c_wchar_p]
    wcslen.restype = UINT

    GMEM_MOVEABLE = 0x0002
    CF_UNICODETEXT = 13

    @contextlib.contextmanager
    def window():
        # [FUNCTION DOC]
        # HELPER: context manager that creates a hidden window of the
        # predefined static class, yields its handle (needed as clipboard
        # owner) and destroys the window on exit, even on error.
        """
        Context that provides a valid Windows hwnd.
        """
        # we really just need the hwnd, so setting "STATIC"
        # as predefined lpClass is just fine.
        hwnd = safeCreateWindowExA(
            0, b"STATIC", None, 0, 0, 0, 0, 0, None, None, None, None
        )
        try:
            yield hwnd
        finally:
            safeDestroyWindow(hwnd)

    @contextlib.contextmanager
    def clipboard(hwnd):
        # [FUNCTION DOC]
        # HELPER: context manager that opens the clipboard for the given window
        # handle (none allowed for reading), retrying every ten milliseconds
        # for up to half a second, and closes it on exit. Raises the Windows
        # clipboard exception if it cannot be opened in time. Yields nothing.
        """
        Context manager that opens the clipboard and prevents
        other applications from modifying the clipboard content.
        """
        # We may not get the clipboard handle immediately because
        # some other application is accessing it (?)
        # We try for at least 500ms to get the clipboard.
        t = time.time() + 0.5
        success = False
        while time.time() < t:
            success = OpenClipboard(hwnd)
            if success:
                break
            time.sleep(0.01)
        if not success:
            raise PyperclipWindowsException("Error calling OpenClipboard")

        try:
            yield
        finally:
            safeCloseClipboard()

    def copy_windows(text):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception on
        # unsupported types), opens the clipboard with a temporary window as
        # owner, empties it and, if the text is non-empty, allocates a movable
        # global memory block sized for the wide-character text plus
        # terminator, copies the text into it and sets it as unicode text data.
        # Returns nothing.
        # This function is heavily based on
        # http://msdn.com/ms649016#_win32_Copying_Information_to_the_Clipboard

        text = _stringifyText(text)  # Converts non-str values to str.

        with window() as hwnd:
            # http://msdn.com/ms649048
            # If an application calls OpenClipboard with hwnd set to NULL,
            # EmptyClipboard sets the clipboard owner to NULL;
            # this causes SetClipboardData to fail.
            # => We need a valid hwnd to copy something.
            with clipboard(hwnd):
                safeEmptyClipboard()

                if text:
                    # http://msdn.com/ms649051
                    # If the hMem parameter identifies a memory object,
                    # the object must have been allocated using the
                    # function with the GMEM_MOVEABLE flag.
                    count = wcslen(text) + 1
                    handle = safeGlobalAlloc(GMEM_MOVEABLE, count * sizeof(c_wchar))
                    locked_handle = safeGlobalLock(handle)

                    ctypes.memmove(
                        c_wchar_p(locked_handle),
                        c_wchar_p(text),
                        count * sizeof(c_wchar),
                    )

                    safeGlobalUnlock(handle)
                    safeSetClipboardData(CF_UNICODETEXT, handle)

    def paste_windows():
        # [FUNCTION DOC]
        # HELPER: opens the clipboard without an owner window and returns its
        # unicode text, or an empty string when no data handle is available.
        with clipboard(None):
            handle = safeGetClipboardData(CF_UNICODETEXT)
            if not handle:
                # GetClipboardData may return NULL with errno == NO_ERROR
                # if the clipboard is empty.
                # (Also, it may return a handle to an empty buffer,
                # but technically that's not empty)
                return ""
            return c_wchar_p(handle).value

    return copy_windows, paste_windows


def init_wsl_clipboard():
    # [FUNCTION DOC]
    # HELPER: backend factory for Windows Subsystem for Linux using the
    # Windows clip executable to copy and a PowerShell command to paste.
    # Returns a pair of (copy function, paste function).
    def copy_wsl(text):
        # [FUNCTION DOC]
        # HELPER: converts text to a string (clipboard exception on
        # unsupported types) and pipes it as utf-8 into the clip executable.
        text = _stringifyText(text)  # Converts non-str values to str.
        with subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE, close_fds=True) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_wsl():
        # [FUNCTION DOC]
        # HELPER: runs the PowerShell clipboard getter and returns its output
        # decoded as utf-8 after dropping the final two characters (carriage
        # return and newline that are always appended).
        with subprocess.Popen(
            ["powershell.exe", "-command", "Get-Clipboard"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]
        # WSL appends "\r\n" to the contents.
        return stdout[:-2].decode(ENCODING)

    return copy_wsl, paste_wsl


# Automatic detection of clipboard mechanisms
# and importing is done in determine_clipboard():
def determine_clipboard():
    # [FUNCTION DOC]
    # Detects the platform and returns the (copy function, paste function)
    # pair of the first workable backend; it does not install them as module
    # globals itself (the caller does). Order: Cygwin (only if the clipboard
    # device exists, with a warning), Windows, WSL on Linux, macOS (pyobjc if
    # importable, else the pb commands), then on Linux with a display: Wayland
    # tool when its environment variable and tool exist, xsel, xclip, klipper
    # with qdbus, then Qt via qtpy, PyQt5 or PyQt4 if importable. Otherwise
    # returns the unavailable placeholders.
    # Side effects: may import macOS or Qt bindings into module globals and
    # emit a warning on Cygwin.
    # KNOWN ISSUE: the docstring below says it sets the copy and paste
    # functions, but the code only returns them; the caller installs them.
    """
    Determine the OS/platform and set the copy() and paste() functions
    accordingly.
    """
    global Foundation, AppKit, qtpy, PyQt4, PyQt5

    # Setup for the CYGWIN platform:
    if (
        "cygwin" in platform.system().lower()
    ):  # Cygwin has a variety of values returned by platform.system(),
        # such as 'CYGWIN_NT-6.1'
        # FIXME(pyperclip#55): pyperclip currently does not support Cygwin,
        # see https://github.com/asweigart/pyperclip/issues/55
        if os.path.exists("/dev/clipboard"):
            warnings.warn(
                "Pyperclip's support for Cygwin is not perfect, "
                "see https://github.com/asweigart/pyperclip/issues/55",
                stacklevel=find_stack_level(),
            )
            return init_dev_clipboard_clipboard()

    # Setup for the WINDOWS platform:
    elif os.name == "nt" or platform.system() == "Windows":
        return init_windows_clipboard()

    if platform.system() == "Linux":
        if _executable_exists("wslconfig.exe"):
            return init_wsl_clipboard()

    # Setup for the macOS platform:
    if os.name == "mac" or platform.system() == "Darwin":
        try:
            import AppKit
            import Foundation  # check if pyobjc is installed
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()

    # Setup for the LINUX platform:
    if HAS_DISPLAY:
        if os.environ.get("WAYLAND_DISPLAY") and _executable_exists("wl-copy"):
            return init_wl_clipboard()
        if _executable_exists("xsel"):
            return init_xsel_clipboard()
        if _executable_exists("xclip"):
            return init_xclip_clipboard()
        if _executable_exists("klipper") and _executable_exists("qdbus"):
            return init_klipper_clipboard()

        try:
            # qtpy is a small abstraction layer that lets you write applications
            # using a single api call to either PyQt or PySide.
            # https://pypi.python.org/project/QtPy
            import qtpy  # check if qtpy is installed
        except ImportError:
            # If qtpy isn't installed, fall back on importing PyQt4.
            try:
                import PyQt5  # check if PyQt5 is installed
            except ImportError:
                try:
                    import PyQt4  # check if PyQt4 is installed
                except ImportError:
                    pass  # We want to fail fast for all non-ImportError exceptions.
                else:
                    return init_qt_clipboard()
            else:
                return init_qt_clipboard()
        else:
            return init_qt_clipboard()

    return init_no_clipboard()


def set_clipboard(clipboard):
    # [FUNCTION DOC]
    # Explicitly selects the clipboard backend by name. Input: clipboard, one
    # of the strings for the pbcopy, pyobjc, qt, xclip, xsel, wl-clipboard,
    # klipper, windows or no backends. Raises a value error listing the
    # allowed names for anything else. Side effect: rebinds the module-level
    # copy and paste functions to the pair returned by that backend's factory.
    # KNOWN ISSUE: the docstring list omits the wl-clipboard choice that is
    # accepted by the code, and Cygwin's device backend is not selectable here.
    """
    Explicitly sets the clipboard mechanism. The "clipboard mechanism" is how
    the copy() and paste() functions interact with the operating system to
    implement the copy/paste feature. The clipboard parameter must be one of:
        - pbcopy
        - pyobjc (default on macOS)
        - qt
        - xclip
        - xsel
        - klipper
        - windows (default on Windows)
        - no (this is what is set when no clipboard mechanism can be found)
    """
    global copy, paste

    clipboard_types = {
        "pbcopy": init_osx_pbcopy_clipboard,
        "pyobjc": init_osx_pyobjc_clipboard,
        "qt": init_qt_clipboard,  # TODO - split this into 'qtpy', 'pyqt4', and 'pyqt5'
        "xclip": init_xclip_clipboard,
        "xsel": init_xsel_clipboard,
        "wl-clipboard": init_wl_clipboard,
        "klipper": init_klipper_clipboard,
        "windows": init_windows_clipboard,
        "no": init_no_clipboard,
    }

    if clipboard not in clipboard_types:
        allowed_clipboard_types = [repr(_) for _ in clipboard_types]
        raise ValueError(
            f"Argument must be one of {', '.join(allowed_clipboard_types)}"
        )

    # Sets pyperclip's copy() and paste() functions:
    copy, paste = clipboard_types[clipboard]()


def lazy_load_stub_copy(text):
    # [FUNCTION DOC]
    # Initial stand-in for the copy function. Input: text to copy. Detects the
    # backend, rebinds the module-level copy and paste to it, then delegates
    # the copy to the new function and returns its result.
    """
    A stub function for copy(), which will load the real copy() function when
    called so that the real copy() function is used for later calls.

    This allows users to import pyperclip without having determine_clipboard()
    automatically run, which will automatically select a clipboard mechanism.
    This could be a problem if it selects, say, the memory-heavy PyQt4 module
    but the user was just going to immediately call set_clipboard() to use a
    different clipboard mechanism.

    The lazy loading this stub function implements gives the user a chance to
    call set_clipboard() to pick another clipboard mechanism. Or, if the user
    simply calls copy() or paste() without calling set_clipboard() first,
    will fall back on whatever clipboard mechanism that determine_clipboard()
    automatically chooses.
    """
    global copy, paste
    copy, paste = determine_clipboard()
    return copy(text)


def lazy_load_stub_paste():
    # [FUNCTION DOC]
    # Initial stand-in for the paste function. Detects the backend, rebinds
    # the module-level copy and paste, then delegates to the new paste and
    # returns the clipboard text.
    """
    A stub function for paste(), which will load the real paste() function when
    called so that the real paste() function is used for later calls.

    This allows users to import pyperclip without having determine_clipboard()
    automatically run, which will automatically select a clipboard mechanism.
    This could be a problem if it selects, say, the memory-heavy PyQt4 module
    but the user was just going to immediately call set_clipboard() to use a
    different clipboard mechanism.

    The lazy loading this stub function implements gives the user a chance to
    call set_clipboard() to pick another clipboard mechanism. Or, if the user
    simply calls copy() or paste() without calling set_clipboard() first,
    will fall back on whatever clipboard mechanism that determine_clipboard()
    automatically chooses.
    """
    global copy, paste
    copy, paste = determine_clipboard()
    return paste()


def is_available() -> bool:
    # [FUNCTION DOC]
    # Returns true if a clipboard backend has already been resolved, meaning
    # neither module-level copy nor paste is still a lazy stub. It does not
    # trigger detection, so it is false until the first use or explicit
    # selection.
    return copy != lazy_load_stub_copy and paste != lazy_load_stub_paste


# Initially, copy() and paste() are set to lazy loading wrappers which will
# set `copy` and `paste` to real functions the first time they're used, unless
# set_clipboard() or determine_clipboard() is called first.
copy, paste = lazy_load_stub_copy, lazy_load_stub_paste


def waitForPaste(timeout=None):
    # [FUNCTION DOC]
    # Polls the clipboard every ten milliseconds until non-empty text is
    # found and returns it. timeout (seconds, default none meaning wait
    # forever) bounds the wait; on expiry raises the timeout exception. Reads
    # the clipboard through the module-level paste function.
    """This function call blocks until a non-empty text string exists on the
    clipboard. It returns this text.

    This function raises PyperclipTimeoutException if timeout was set to
    a number of seconds that has elapsed without non-empty text being put on
    the clipboard."""
    startTime = time.time()
    while True:
        clipboardText = paste()
        if clipboardText != "":
            return clipboardText
        time.sleep(0.01)

        if timeout is not None and time.time() > startTime + timeout:
            raise PyperclipTimeoutException(
                "waitForPaste() timed out after " + str(timeout) + " seconds."
            )


def waitForNewPaste(timeout=None):
    # [FUNCTION DOC]
    # Polls the clipboard every ten milliseconds until its text differs from
    # the text present at call time and returns the new text. timeout
    # (seconds, default none meaning wait forever) bounds the wait; on expiry
    # raises the timeout exception.
    # KNOWN ISSUE: the docstring says the timeout error occurs when no
    # non-empty text appears, but the code raises it when no different text
    # appears.
    """This function call blocks until a new text string exists on the
    clipboard that is different from the text that was there when the function
    was first called. It returns this text.

    This function raises PyperclipTimeoutException if timeout was set to
    a number of seconds that has elapsed without non-empty text being put on
    the clipboard."""
    startTime = time.time()
    originalText = paste()
    while True:
        currentText = paste()
        if currentText != originalText:
            return currentText
        time.sleep(0.01)

        if timeout is not None and time.time() > startTime + timeout:
            raise PyperclipTimeoutException(
                "waitForNewPaste() timed out after " + str(timeout) + " seconds."
            )


__all__ = [
    "copy",
    "paste",
    "waitForPaste",
    "waitForNewPaste",
    "set_clipboard",
    "determine_clipboard",
]

# pandas aliases
clipboard_get = paste
clipboard_set = copy
