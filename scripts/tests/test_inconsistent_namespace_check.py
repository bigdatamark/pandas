# [MODULE DOC]
# Tests for the pandas-namespace consistency checker. Defines sample source texts that mix or
# do not mix the bare name Categorical with its namespaced forms, and tests for reporting and
# for automatic replacement.
import pytest

from scripts.check_for_inconsistent_pandas_namespace import (
    check_for_inconsistent_pandas_namespace,
)

BAD_FILE_0 = (
    "from pandas import Categorical\ncat_0 = Categorical()\ncat_1 = pd.Categorical()"
)
BAD_FILE_1 = (
    "from pandas import Categorical\ncat_0 = pd.Categorical()\ncat_1 = Categorical()"
)
BAD_FILE_2 = (
    "from pandas import Categorical\n"
    "cat_0 = pandas.Categorical()\n"
    "cat_1 = Categorical()"
)
GOOD_FILE_0 = (
    "from pandas import Categorical\ncat_0 = Categorical()\ncat_1 = Categorical()"
)
GOOD_FILE_1 = "cat_0 = pd.Categorical()\ncat_1 = pd.Categorical()"
GOOD_FILE_2 = "from array import array\nimport pandas as pd\narr = pd.array([])"
PATH = "t.py"


@pytest.mark.parametrize(
    "content, expected",
    [
        (BAD_FILE_0, "t.py:3:8: Found both 'pd.Categorical' and 'Categorical' in t.py"),
        (BAD_FILE_1, "t.py:2:8: Found both 'pd.Categorical' and 'Categorical' in t.py"),
        (
            BAD_FILE_2,
            "t.py:2:8: Found both 'pandas.Categorical' and 'Categorical' in t.py",
        ),
    ],
)
def test_inconsistent_usage(content, expected, capsys):
    # [FUNCTION DOC]
    # Verifies that each bad sample (namespaced under the alias or the full name, before or after
    # the bare use) makes the check exit via a system exit and write the exact message with
    # line, column, prefix and name to standard output. Parametrized over source and expected message.
    with pytest.raises(SystemExit):
        check_for_inconsistent_pandas_namespace(content, PATH, replace=False)
    result, _ = capsys.readouterr()
    assert result == expected


@pytest.mark.parametrize("content", [GOOD_FILE_0, GOOD_FILE_1, GOOD_FILE_2])
@pytest.mark.parametrize("replace", [True, False])
def test_consistent_usage(content, replace):
    # [FUNCTION DOC]
    # Verifies that consistent samples (only bare names, only namespaced names, or an unrelated
    # import) produce no error in both check and replace modes. Parametrized over sample and replace flag.
    # should not raise
    check_for_inconsistent_pandas_namespace(content, PATH, replace=replace)


@pytest.mark.parametrize("content", [BAD_FILE_0, BAD_FILE_1, BAD_FILE_2])
def test_inconsistent_usage_with_replace(content):
    # [FUNCTION DOC]
    # Verifies that in replace mode each bad sample is rewritten to the consistent form using
    # only the bare name. Parametrized over the three bad samples.
    result = check_for_inconsistent_pandas_namespace(content, PATH, replace=True)
    expected = (
        "from pandas import Categorical\ncat_0 = Categorical()\ncat_1 = Categorical()"
    )
    assert result == expected
