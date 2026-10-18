# [MODULE DOC]
# Benchmarks for value replacement and missing-value filling: filling versus
# replacing missing values, replacing with a mapping, replacing with a list of
# values, and replacing integers with scalar type objects. Public contents are the
# benchmark classes FillNa, ReplaceDict, ReplaceList and Convert. No import-time
# side effects; the final line re-exports the common asv setup hook.
import numpy as np

import pandas as pd


class FillNa:
    # [CLASS DOC]
    # Benchmark of filling missing values in a large time series versus replacing
    # NaN values by a constant, in place or not. Parametrized by the inplace flag.
    # State: ts.
    params = [True, False]
    param_names = ["inplace"]

    def setup(self, inplace):
        # [FUNCTION DOC]
        # Builds a series of a million random normal values on a minute index from
        # 2000-01-01 and sets every second value to missing; stores as ts. The
        # inplace flag is unused here.
        N = 10**6
        rng = pd.date_range("1/1/2000", periods=N, freq="min")
        data = np.random.randn(N)
        data[::2] = np.nan
        self.ts = pd.Series(data, index=rng)

    def time_fillna(self, inplace):
        # [FUNCTION DOC]
        # Times filling missing values with 0.0 (mutates ts if inplace).
        self.ts.fillna(0.0, inplace=inplace)

    def time_replace(self, inplace):
        # [FUNCTION DOC]
        # Times replacing NaN values with 0.0 (mutates ts if inplace).
        self.ts.replace(np.nan, 0.0, inplace=inplace)


class ReplaceDict:
    # [CLASS DOC]
    # Benchmark of replacing values using a large mapping. Parametrized by the
    # inplace flag. State: to_rep and s.
    params = [True, False]
    param_names = ["inplace"]

    def setup(self, inplace):
        # [FUNCTION DOC]
        # Builds to_rep, a mapping of 100000 integer keys (0 to 99999) to values
        # 100000 to 199999, and s, a series of 1000 random integers below 100000.
        # The inplace flag is unused here.
        N = 10**5
        start_value = 10**5
        self.to_rep = dict(enumerate(np.arange(N) + start_value))
        self.s = pd.Series(np.random.randint(N, size=10**3))

    def time_replace_series(self, inplace):
        # [FUNCTION DOC]
        # Times replacing values in the series with the mapping (mutates s if
        # inplace).
        self.s.replace(self.to_rep, inplace=inplace)


class ReplaceList:
    # [CLASS DOC]
    # Benchmark (issue 28099) of replacing a list of values by a single value in
    # a huge frame, in place or not. Parametrized by the inplace flag (given as a
    # one-element list holding a tuple of options). State: df.
    # GH#28099

    params = [(True, False)]
    param_names = ["inplace"]

    def setup(self, inplace):
        # [FUNCTION DOC]
        # Builds a frame of ten million rows with two integer columns A and B both
        # zero, stores as df. The inplace flag is unused here.
        self.df = pd.DataFrame({"A": 0, "B": 0}, index=range(10**7))

    def time_replace_list(self, inplace):
        # [FUNCTION DOC]
        # Times replacing positive and negative infinity with missing (mutates df
        # if inplace). None of the values are present in the data.
        self.df.replace([np.inf, -np.inf], np.nan, inplace=inplace)

    def time_replace_list_one_match(self, inplace):
        # [FUNCTION DOC]
        # Times the same replacement with the additional value one in the list. The
        # integer one can be held by the first internal block while the infinities
        # cannot, which exercises the mixed path. (The original note refers to an
        # attribute name that differs from the real attribute name.)
        # the 1 can be held in self._df.blocks[0], while the inf and -inf can't
        self.df.replace([np.inf, -np.inf, 1], np.nan, inplace=inplace)


class Convert:
    # [CLASS DOC]
    # Benchmark of replacing every integer of a series or frame by a time scalar
    # type. Parametrized by container kind (DataFrame, Series) and the name of the
    # replacement type (Timestamp, Timedelta). State: to_replace and data.
    params = (["DataFrame", "Series"], ["Timestamp", "Timedelta"])
    param_names = ["constructor", "replace_data"]

    def setup(self, constructor, replace_data):
        # [FUNCTION DOC]
        # Builds a series of 1000 random integers below 1000 and a two-column frame
        # (A and B) of the same kind of data, and selects the one named by
        # constructor as data. Builds to_replace, a mapping from each integer 0 to
        # 999 to the library attribute named by replace_data (the Timestamp or
        # Timedelta class itself, not an instance).
        N = 10**3
        data = {
            "Series": pd.Series(np.random.randint(N, size=N)),
            "DataFrame": pd.DataFrame(
                {"A": np.random.randint(N, size=N), "B": np.random.randint(N, size=N)}
            ),
        }
        self.to_replace = {i: getattr(pd, replace_data) for i in range(N)}
        self.data = data[constructor]

    def time_replace(self, constructor, replace_data):
        # [FUNCTION DOC]
        # Times replacing values in data using the mapping.
        self.data.replace(self.to_replace)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
