# [MODULE DOC]
# Tests for the console encoding detection helper exported by the config
# package. Uses a mock object standing in for the standard streams and
# monkeypatching of the stream attributes, the locale preferred-encoding
# function and the interpreter default-encoding function, so no real console
# is needed. Covers the order of preference: stream encodings first, then
# the locale encoding, then the interpreter default.
import locale

import pytest

from pandas._config import detect_console_encoding


class MockEncoding:
    # [CLASS DOC]
    # HELPER: stand-in for a standard stream object used by the console
    # encoding tests. State: a single stored value that is either a string or
    # an exception class/instance. Reading its encoding attribute returns the
    # string, or raises the stored exception. Has a static helper that applies
    # the same return-or-raise rule to any value.
    """
    Used to add a side effect when accessing the 'encoding' property. If the
    side effect is a str in nature, the value will be returned. Otherwise, the
    side effect should be an exception that will be raised.
    """

    def __init__(self, encoding) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the given value (a string, or an exception to raise on
        # access) on the instance. Calls the parent initializer.
        super().__init__()
        self.val = encoding

    @property
    def encoding(self):
        # [FUNCTION DOC]
        # HELPER: property returning the stored string, or raising the stored
        # exception, by delegating to the static return-or-raise helper.
        return self.raise_or_return(self.val)

    @staticmethod
    def raise_or_return(val):
        # [FUNCTION DOC]
        # HELPER: returns val unchanged when it is a string; otherwise raises
        # val (which must be an exception class or instance).
        if isinstance(val, str):
            return val
        else:
            raise val


@pytest.mark.parametrize("empty,filled", [["stdin", "stdout"], ["stdout", "stdin"]])
def test_detect_console_encoding_from_stdout_stdin(monkeypatch, empty, filled):
    # [FUNCTION DOC]
    # Verifies the detector uses whichever of the input or output streams has
    # a non-empty encoding. Parametrized over which stream is made empty and
    # which is filled; the filled stream reports its own name as encoding.
    # Expected: the detected encoding equals the filled stream's name.
    # Ensures that when sys.stdout.encoding or sys.stdin.encoding is used when
    # they have values filled.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr(f"sys.{empty}", MockEncoding(""))
        context.setattr(f"sys.{filled}", MockEncoding(filled))
        assert detect_console_encoding() == filled


@pytest.mark.parametrize("encoding", [AttributeError, OSError, "ascii"])
def test_detect_console_encoding_fallback_to_locale(monkeypatch, encoding):
    # [FUNCTION DOC]
    # Verifies fallback to the locale preferred encoding. Parametrized over the
    # output stream failing with an attribute error, failing with an OS error,
    # or reporting ascii. The locale function is patched to return a dummy
    # name; the detector must return that name.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr("locale.getpreferredencoding", lambda: "foo")
        context.setattr("sys.stdout", MockEncoding(encoding))
        assert detect_console_encoding() == "foo"


@pytest.mark.parametrize(
    "std,locale",
    [
        ["ascii", "ascii"],
        ["ascii", locale.Error],
        [AttributeError, "ascii"],
        [AttributeError, locale.Error],
        [OSError, "ascii"],
        [OSError, locale.Error],
    ],
)
def test_detect_console_encoding_fallback_to_default(monkeypatch, std, locale):
    # [FUNCTION DOC]
    # Verifies the final fallback to the interpreter default encoding when both
    # the stream and the locale checks fail or yield ascii. Parametrized over
    # the stream value (ascii, attribute error, OS error) and the locale
    # result (ascii or a locale error raised by the patched function). The
    # interpreter default-encoding function is patched to a sentinel name which
    # must be returned. Note the parameter named locale shadows the imported
    # module inside this function.
    # When both the stdout/stdin encoding and locale preferred encoding checks
    # fail (or return 'ascii', we should default to the sys default encoding.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr(
            "locale.getpreferredencoding", lambda: MockEncoding.raise_or_return(locale)
        )
        context.setattr("sys.stdout", MockEncoding(std))
        context.setattr("sys.getdefaultencoding", lambda: "sysDefaultEncoding")
        assert detect_console_encoding() == "sysDefaultEncoding"
