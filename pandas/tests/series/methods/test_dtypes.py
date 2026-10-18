# [MODULE DOC]
# Test for the dtype and dtypes attributes of Series. Uses the datetime_series
# fixture. No import-time side effects.

import numpy as np


class TestSeriesDtypes:
    # [CLASS DOC]
    # Groups the tests of the Series dtype attributes. Stateless.
    def test_dtype(self, datetime_series):
        # [FUNCTION DOC]
        # The dtype and dtypes attributes of the datetime_series fixture both equal the
        # 64-bit float dtype.
        assert datetime_series.dtype == np.dtype("float64")
        assert datetime_series.dtypes == np.dtype("float64")
