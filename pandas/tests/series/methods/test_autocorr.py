# [MODULE DOC]
# Test for Series.autocorr, comparing it against corr of the shifted Series.
# Uses the datetime_series fixture. No import-time side effects.

import numpy as np


class TestAutoCorr:
    # [CLASS DOC]
    # Groups the tests of Series.autocorr. Stateless.
    def test_autocorr(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies autocorr on the datetime_series fixture: the default call equals the
        # call with an explicit lag of one, and a call with a random lag between 1 and
        # length minus 2 equals the correlation of the Series with the same Series
        # shifted by that lag. Series of length two or less must give NaN in both cases.
        # The random lag comes from a fixed-seed generator.
        # Just run the function
        corr1 = datetime_series.autocorr()

        # Now run it with the lag parameter
        corr2 = datetime_series.autocorr(lag=1)

        # corr() with lag needs Series of at least length 2
        if len(datetime_series) <= 2:
            assert np.isnan(corr1)
            assert np.isnan(corr2)
        else:
            assert corr1 == corr2

        # Choose a random lag between 1 and length of Series - 2
        # and compare the result with the Series corr() function
        n = 1 + np.random.default_rng(2).integers(max(1, len(datetime_series) - 2))
        corr1 = datetime_series.corr(datetime_series.shift(n))
        corr2 = datetime_series.autocorr(lag=n)

        # corr() with lag needs Series of at least length 2
        if len(datetime_series) <= 2:
            assert np.isnan(corr1)
            assert np.isnan(corr2)
        else:
            assert corr1 == corr2
