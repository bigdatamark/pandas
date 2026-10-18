# [MODULE DOC]
# Test that tab completion on a Categorical in an interactive IPython session emits no warnings.
# Uses the ip fixture (an interactive shell) from the shared conftest; IPython is imported lazily and skipped
# if too old or missing. No import-time side effects.

import pytest

import pandas._testing as tm


class TestCategoricalWarnings:
    # [CLASS DOC]
    # Groups warning tests. Stateless.
    def test_tab_complete_warning(self, ip):
        # [FUNCTION DOC]
        # Skips if IPython (version 6 or newer) is not available. Runs a cell in the shell that imports pandas and creates an
        # empty Categorical, then requests completions for the variable and expects no warning to be emitted (extra
        # warnings are tolerated). Side effect: executes code in the shared interactive shell.
        # https://github.com/pandas-dev/pandas/issues/16409
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; c = pd.Categorical([])"
        ip.run_cell(code)

        # GH 31324 newer jedi version raises Deprecation warning;
        #  appears resolved 2021-02-02
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):
                list(ip.Completer.completions("c.", 1))
