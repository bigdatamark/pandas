# [MODULE DOC]
# Pytest configuration for the plotting tests: an automatic fixture ensuring
# matplotlib state is cleaned up around each test and a shared data fixture
# with grouped, categorical and datetime columns used by the histogram and
# box plot tests. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    to_datetime,
)


@pytest.fixture(autouse=True)
def autouse_mpl_cleanup(mpl_cleanup):
    # [FUNCTION DOC]
    # Autouse fixture that depends on the global matplotlib cleanup fixture so that
    # every plotting test gets its figures closed afterwards. Takes the mpl_cleanup
    # fixture and does nothing itself.
    pass


@pytest.fixture
def hist_df():
    # [FUNCTION DOC]
    # Fixture returning a frame of 50 rows built from a fixed-seed random generator:
    # random gender labels, classroom labels A to C, normally distributed height
    # and weight, an integer category from 0 to 3, and a datetime column drawn
    # from a fixed nanosecond range in 1995.
    n = 50
    rng = np.random.default_rng(10)
    gender = rng.choice(["Male", "Female"], size=n)
    classroom = rng.choice(["A", "B", "C"], size=n)

    hist_df = DataFrame(
        {
            "gender": gender,
            "classroom": classroom,
            "height": rng.normal(66, 4, size=n),
            "weight": rng.normal(161, 32, size=n),
            "category": rng.integers(4, size=n),
            "datetime": to_datetime(
                rng.integers(
                    812419200000000000,
                    819331200000000000,
                    size=n,
                    dtype=np.int64,
                )
            ),
        }
    )
    return hist_df
