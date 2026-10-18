# [MODULE DOC]
# Tests for the optional-dependency import helper in the compat package: error
# messages, the ignore and warn error modes, minimum version checks, submodule
# handling, and missing version attributes. Several tests register fake modules in
# the global module table and patch the minimum-versions mapping with monkeypatch.
import sys
import types

import pytest

from pandas.compat._optional import (
    VERSIONS,
    import_optional_dependency,
)

import pandas._testing as tm


def test_import_optional():
    # [FUNCTION DOC]
    # Verifies that importing a nonexistent package raises an import error whose message
    # mentions installing it with pip or conda, that the original import error is kept as
    # the exception context, and that the ignore mode returns None instead.
    match = "Import .*notapackage.* pip .* conda .* notapackage"
    with pytest.raises(ImportError, match=match) as exc_info:
        import_optional_dependency("notapackage")
    # The original exception should be there as context:
    assert isinstance(exc_info.value.__context__, ImportError)

    result = import_optional_dependency("notapackage", errors="ignore")
    assert result is None


def test_xlrd_version_fallback():
    # [FUNCTION DOC]
    # Skipped when xlrd is absent. Verifies that importing xlrd through the helper
    # succeeds (exercising the fallback path for determining its version) without error.
    pytest.importorskip("xlrd")
    import_optional_dependency("xlrd")


def test_bad_version(monkeypatch):
    # [FUNCTION DOC]
    # Registers a fake module whose version is older than the required one (patching the
    # minimum-version mapping). Verifies: an import error for the too-old version; an
    # explicit lower minimum version accepts it; the warn mode emits a user warning and
    # returns None; an exactly equal version is accepted; an explicit higher minimum
    # version raises, warns, or returns None depending on the error mode.
    name = "fakemodule"
    module = types.ModuleType(name)
    module.__version__ = "0.9.0"
    sys.modules[name] = module
    monkeypatch.setitem(VERSIONS, name, "1.0.0")

    match = "Pandas requires .*1.0.0.* of .fakemodule.*'0.9.0'"
    with pytest.raises(ImportError, match=match):
        import_optional_dependency("fakemodule")

    # Test min_version parameter
    result = import_optional_dependency("fakemodule", min_version="0.8")
    assert result is module

    with tm.assert_produces_warning(UserWarning, match=match):
        result = import_optional_dependency("fakemodule", errors="warn")
    assert result is None

    module.__version__ = "1.0.0"  # exact match is OK
    result = import_optional_dependency("fakemodule")
    assert result is module

    with pytest.raises(ImportError, match="Pandas requires version '1.1.0'"):
        import_optional_dependency("fakemodule", min_version="1.1.0")

    with tm.assert_produces_warning(UserWarning, match="Pandas requires version"):
        result = import_optional_dependency(
            "fakemodule", errors="warn", min_version="1.1.0"
        )
    assert result is None

    result = import_optional_dependency(
        "fakemodule", errors="ignore", min_version="1.1.0"
    )
    assert result is None


def test_submodule(monkeypatch):
    # Create a fake module with a submodule
    # [FUNCTION DOC]
    # Registers a fake module with a submodule. Verifies that the version check applies
    # to the top-level package when importing the dotted submodule name: error when too
    # old, warning and None in warn mode, and the submodule object returned once the
    # version is sufficient.
    name = "fakemodule"
    module = types.ModuleType(name)
    module.__version__ = "0.9.0"
    sys.modules[name] = module
    sub_name = "submodule"
    submodule = types.ModuleType(sub_name)
    setattr(module, sub_name, submodule)
    sys.modules[f"{name}.{sub_name}"] = submodule
    monkeypatch.setitem(VERSIONS, name, "1.0.0")

    match = "Pandas requires .*1.0.0.* of .fakemodule.*'0.9.0'"
    with pytest.raises(ImportError, match=match):
        import_optional_dependency("fakemodule.submodule")

    with tm.assert_produces_warning(UserWarning, match=match):
        result = import_optional_dependency("fakemodule.submodule", errors="warn")
    assert result is None

    module.__version__ = "1.0.0"  # exact match is OK
    result = import_optional_dependency("fakemodule.submodule")
    assert result is submodule


def test_no_version_raises(monkeypatch):
    # [FUNCTION DOC]
    # Registers a fake module with no version attribute while a minimum version is
    # required. Verifies an import error saying the version cannot be determined.
    name = "fakemodule"
    module = types.ModuleType(name)
    sys.modules[name] = module
    monkeypatch.setitem(VERSIONS, name, "1.0.0")

    with pytest.raises(ImportError, match="Can't determine .* fakemodule"):
        import_optional_dependency(name)
