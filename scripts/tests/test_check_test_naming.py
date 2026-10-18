# [MODULE DOC]
# Tests for the test-naming checker script. One parametrized test runs the checker's main
# function on small source snippets and compares what it prints and its return value.
import pytest

from scripts.check_test_naming import main


@pytest.mark.parametrize(
    "src, expected_out, expected_ret",
    [
        (
            "def foo(): pass\n",
            "t.py:1:0 found test function which does not start with 'test'\n",
            1,
        ),
        (
            "class Foo:\n    def test_foo(): pass\n",
            "t.py:1:0 found test class which does not start with 'Test'\n",
            1,
        ),
        ("def test_foo(): pass\n", "", 0),
        ("class TestFoo:\n    def test_foo(): pass\n", "", 0),
        (
            "def foo():\n    pass\ndef test_foo():\n    foo()\n",
            "",
            0,
        ),
        (
            "class Foo:  # not a test\n    pass\ndef test_foo():\n    Class.foo()\n",
            "",
            0,
        ),
        ("@pytest.fixture\ndef foo(): pass\n", "", 0),
        ("@pytest.fixture()\ndef foo(): pass\n", "", 0),
        ("@register_extension_dtype\nclass Foo: pass\n", "", 0),
    ],
)
def test_main(capsys, src, expected_out, expected_ret):
    # [FUNCTION DOC]
    # Verifies the checker on snippets: a top-level function not named as a test, and a test-less
    # named class, are reported with file, line and column and give a failing status; correctly
    # named functions and classes, helpers used by a test function, objects with the exemption
    # comment, fixtures (bare or called) and classes decorated for dtype registration are accepted
    # with empty output and a zero status. Parametrized over source text, expected output and
    # expected return value; uses output capture.
    ret = main(src, "t.py")
    out, _ = capsys.readouterr()
    assert out == expected_out
    assert ret == expected_ret
