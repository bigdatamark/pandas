# [MODULE DOC]
# Benchmarks for reshaping: melt, pivot, stack and unstack (plain, extension
# dtypes, categorical, sparse index), wide-to-long, pivot tables, cross
# tabulation, dummy encoding, binning, and exploding lists. Public contents are
# the benchmark classes Melt, Pivot, SimpleReshape, ReshapeExtensionDtype,
# ReshapeMaskedArrayDtype, Unstack, SparseIndex, WideToLong, PivotTable, Crosstab,
# GetDummies, Cut and Explode. No import-time side effects; the final line
# re-exports the common asv setup hook.
from itertools import product
import string

import numpy as np

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    date_range,
    melt,
    wide_to_long,
)
from pandas.api.types import CategoricalDtype


class Melt:
    # [CLASS DOC]
    # Benchmark of unpivoting a frame to long form with two identifier columns.
    # Parametrized by value dtype (float64, nullable Float64). State: df.
    params = ["float64", "Float64"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a 100000 by 3 random normal frame in the dtype with columns A, B, C,
        # then adds identifier columns id1 (random integers below 10) and id2
        # (random integers 100 to 999) from series of only 10000 elements.
        # KNOWN ISSUE: the identifier series have 10000 entries while the frame has
        # 100000 rows, so aligning on the index fills only the first 10000 rows and
        # leaves the other 90000 identifiers missing.
        self.df = DataFrame(
            np.random.randn(100_000, 3), columns=["A", "B", "C"], dtype=dtype
        )
        self.df["id1"] = pd.Series(np.random.randint(0, 10, 10000))
        self.df["id2"] = pd.Series(np.random.randint(100, 1000, 10000))

    def time_melt_dataframe(self, dtype):
        # [FUNCTION DOC]
        # Times unpivoting with id1 and id2 as identifier columns.
        melt(self.df, id_vars=["id1", "id2"])


class Pivot:
    # [CLASS DOC]
    # Benchmark of pivoting a long time-series frame into a wide one. No
    # parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a long frame of 500000 rows: random values, a variable column
        # (the integers below 50, each repeated 10000 times) and a date column
        # (10000 hourly timestamps from 2000-01-01 tiled 50 times).
        N = 10000
        index = date_range("1/1/2000", periods=N, freq="h")
        data = {
            "value": np.random.randn(N * 50),
            "variable": np.arange(50).repeat(N),
            "date": np.tile(index.values, 50),
        }
        self.df = DataFrame(data)

    def time_reshape_pivot_time_series(self):
        # [FUNCTION DOC]
        # Times pivoting with dates as the index, variables as the columns and the
        # values column as the cell contents.
        self.df.pivot(index="date", columns="variable", values="value")


class SimpleReshape:
    # [CLASS DOC]
    # Benchmark of unstacking a two-level indexed frame and stacking it back. No
    # parameters. State: df and udf (the unstacked version).
    def setup(self):
        # [FUNCTION DOC]
        # Builds a two-level index of 10000 entries (the first level repeats each of
        # 100 integers 100 times; the second is a rolled tiling of 100 integers),
        # a 10000 by 4 random frame on it as df, and udf as df with the second
        # level unstacked to columns.
        arrays = [np.arange(100).repeat(100), np.roll(np.tile(np.arange(100), 100), 25)]
        index = MultiIndex.from_arrays(arrays)
        self.df = DataFrame(np.random.randn(10000, 4), index=index)
        self.udf = self.df.unstack(1)

    def time_stack(self):
        # [FUNCTION DOC]
        # Times stacking the unstacked frame's columns into the index.
        self.udf.stack()

    def time_unstack(self):
        # [FUNCTION DOC]
        # Times unstacking level one of the stacked frame into columns.
        self.df.unstack(1)


class ReshapeExtensionDtype:
    # [CLASS DOC]
    # Benchmark of stack, unstack and transpose for extension dtypes
    # (timezone-aware datetimes, periods). Parametrized by dtype name. State: ser
    # (multi-indexed series) and df (its unstacked frame). Subclassed by the
    # masked-array variant, which reuses these timing methods.
    params = ["datetime64[ns, US/Pacific]", "Period[s]"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a two-level index named foo and bar (ten letters crossed with 1000
        # integers), 10000 per-second timestamps from 2016-01-01 in the US Pacific
        # zone (dropped to naive and converted to second periods for the Period
        # dtype), a series of them on the index as ser and ser unstacked on level
        # bar as df. Stacking df gives back ser.
        lev = pd.Index(list("ABCDEFGHIJ"))
        ri = pd.Index(range(1000))
        mi = MultiIndex.from_product([lev, ri], names=["foo", "bar"])

        index = date_range("2016-01-01", periods=10000, freq="s", tz="US/Pacific")
        if dtype == "Period[s]":
            index = index.tz_localize(None).to_period("s")

        ser = pd.Series(index, index=mi)
        df = ser.unstack("bar")
        # roundtrips -> df.stack().equals(ser)

        self.ser = ser
        self.df = df

    def time_stack(self, dtype):
        # [FUNCTION DOC]
        # Times stacking the unstacked frame.
        self.df.stack()

    def time_unstack_fast(self, dtype):
        # [FUNCTION DOC]
        # Times unstacking the last index level, which avoids copying.
        # last level -> doesn't have to make copies
        self.ser.unstack("bar")

    def time_unstack_slow(self, dtype):
        # [FUNCTION DOC]
        # Times unstacking the first index level, which must copy the data.
        # first level -> must make copies
        self.ser.unstack("foo")

    def time_transpose(self, dtype):
        # [FUNCTION DOC]
        # Times transposing the unstacked frame.
        self.df.T


class ReshapeMaskedArrayDtype(ReshapeExtensionDtype):
    # [CLASS DOC]
    # Variant of the extension-dtype reshape benchmark for nullable numeric dtypes
    # (Int64, Float64). Inherits the timing methods from the parent class and
    # overrides the parameters and setup.
    params = ["Int64", "Float64"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds the same two-level index as the parent (ten letters crossed with
        # 1000 integers), a series of 10000 random normal numbers truncated to
        # integers in the nullable dtype as ser, and ser unstacked on level bar as
        # df.
        lev = pd.Index(list("ABCDEFGHIJ"))
        ri = pd.Index(range(1000))
        mi = MultiIndex.from_product([lev, ri], names=["foo", "bar"])

        values = np.random.randn(10_000).astype(int)

        ser = pd.Series(values, dtype=dtype, index=mi)
        df = ser.unstack("bar")
        # roundtrips -> df.stack().equals(ser)

        self.ser = ser
        self.df = df


class Unstack:
    # [CLASS DOC]
    # Benchmark of unstacking a frame whose two-level index is a full product or
    # lacks its final row, with integer or categorical columns. Parametrized by
    # dtype kind (int, category); there is no parameter name list. State: df and
    # df2 (df without its last row).
    params = ["int", "category"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Uses a two-level index of the product of the integers below 100 with
        # themselves (10000 rows). For int: a frame of 1000 integer columns with
        # consecutive values. Otherwise: only 50 columns (the categorical case is
        # about twenty times slower, so size is reduced) of categoricals built
        # from random letters, which are about three times slower than the int
        # case. Stores as df and df2 (df without its last row).
        m = 100
        n = 1000

        levels = np.arange(m)
        index = MultiIndex.from_product([levels] * 2)
        columns = np.arange(n)
        if dtype == "int":
            values = np.arange(m * m * n).reshape(m * m, n)
            self.df = DataFrame(values, index, columns)
        else:
            # the category branch is ~20x slower than int. So we
            # cut down the size a bit. Now it's only ~3x slower.
            n = 50
            columns = columns[:n]
            indices = np.random.randint(0, 52, size=(m * m, n))
            values = np.take(list(string.ascii_letters), indices)
            values = [pd.Categorical(v) for v in values.T]

            self.df = DataFrame(dict(enumerate(values)), index, columns)

        self.df2 = self.df.iloc[:-1]

    def time_full_product(self, dtype):
        # [FUNCTION DOC]
        # Times unstacking the full-product frame.
        self.df.unstack()

    def time_without_last_row(self, dtype):
        # [FUNCTION DOC]
        # Times unstacking the frame missing its last row (an incomplete product).
        self.df2.unstack()


class SparseIndex:
    # [CLASS DOC]
    # Benchmark of unstacking a frame with a sparse five-level index. No
    # parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 1000-row frame with integer columns A and B (below 50), C and D
        # (minus 10 to 9), E (below 10) and a random float column F, then sets the
        # five integer columns as a five-level index.
        NUM_ROWS = 1000
        self.df = DataFrame(
            {
                "A": np.random.randint(50, size=NUM_ROWS),
                "B": np.random.randint(50, size=NUM_ROWS),
                "C": np.random.randint(-10, 10, size=NUM_ROWS),
                "D": np.random.randint(-10, 10, size=NUM_ROWS),
                "E": np.random.randint(10, size=NUM_ROWS),
                "F": np.random.randn(NUM_ROWS),
            }
        )
        self.df = self.df.set_index(["A", "B", "C", "D", "E"])

    def time_unstack(self):
        # [FUNCTION DOC]
        # Times unstacking the last index level.
        self.df.unstack()


class WideToLong:
    # [CLASS DOC]
    # Benchmark of converting a wide frame with year-suffixed variable columns to
    # long form. No parameters. State: letters and df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a frame of 5000 rows with 20 plain columns (named 0 to 19) and 80
        # stub columns (letters A to D followed by the years 1 to 20), filled with
        # random normal numbers, plus an id column equal to the row label. Stores
        # the stub letters as letters.
        nyrs = 20
        nidvars = 20
        N = 5000
        self.letters = list("ABCD")
        yrvars = [
            letter + str(num)
            for letter, num in product(self.letters, range(1, nyrs + 1))
        ]
        columns = [str(i) for i in range(nidvars)] + yrvars
        self.df = DataFrame(np.random.randn(N, nidvars + len(yrvars)), columns=columns)
        self.df["id"] = self.df.index

    def time_wide_to_long_big(self):
        # [FUNCTION DOC]
        # Times converting to long form using the stub letters, id as identifier and
        # year as the new variable name.
        wide_to_long(self.df, self.letters, i="id", j="year")


class PivotTable:
    # [CLASS DOC]
    # Benchmark of pivot tables with multiple aggregation functions, margins and
    # categorical keys. No parameters. State: df (100000 rows, three string keys
    # and three value columns) and df2 (five rows with categorical keys).
    def setup(self):
        # [FUNCTION DOC]
        # Builds df with 100000 rows: key1 (A, B or C at random), key2 and key3
        # (one or two at random, identical to each other) and three random normal
        # value columns. Builds df2: five rows with categorical columns col1 (a to
        # e) and col2 (f to j) and an integer column col3 (1 to 5).
        N = 100000
        fac1 = np.array(["A", "B", "C"], dtype="O")
        fac2 = np.array(["one", "two"], dtype="O")
        ind1 = np.random.randint(0, 3, size=N)
        ind2 = np.random.randint(0, 2, size=N)
        self.df = DataFrame(
            {
                "key1": fac1.take(ind1),
                "key2": fac2.take(ind2),
                "key3": fac2.take(ind2),
                "value1": np.random.randn(N),
                "value2": np.random.randn(N),
                "value3": np.random.randn(N),
            }
        )
        self.df2 = DataFrame(
            {"col1": list("abcde"), "col2": list("fghij"), "col3": [1, 2, 3, 4, 5]}
        )
        self.df2.col1 = self.df2.col1.astype("category")
        self.df2.col2 = self.df2.col2.astype("category")

    def time_pivot_table(self):
        # [FUNCTION DOC]
        # Times a pivot table with key1 as rows and key2 plus key3 as columns using
        # the default mean aggregation.
        self.df.pivot_table(index="key1", columns=["key2", "key3"])

    def time_pivot_table_agg(self):
        # [FUNCTION DOC]
        # Times the same pivot table with the sum and mean aggregation functions.
        self.df.pivot_table(
            index="key1", columns=["key2", "key3"], aggfunc=["sum", "mean"]
        )

    def time_pivot_table_margins(self):
        # [FUNCTION DOC]
        # Times the same pivot table with total margins.
        self.df.pivot_table(index="key1", columns=["key2", "key3"], margins=True)

    def time_pivot_table_categorical(self):
        # [FUNCTION DOC]
        # Times a pivot table of col3 summed over categorical keys, filling empty
        # cells with zero, including unobserved category combinations.
        self.df2.pivot_table(
            index="col1", values="col3", columns="col2", aggfunc="sum", fill_value=0
        )

    def time_pivot_table_categorical_observed(self):
        # [FUNCTION DOC]
        # Times the same categorical pivot table restricted to observed
        # combinations.
        self.df2.pivot_table(
            index="col1",
            values="col3",
            columns="col2",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )

    def time_pivot_table_margins_only_column(self):
        # [FUNCTION DOC]
        # Times a pivot table with no row key, the three keys as columns, and
        # margins.
        self.df.pivot_table(columns=["key1", "key2", "key3"], margins=True)


class Crosstab:
    # [CLASS DOC]
    # Benchmark of cross tabulation of two categorical-like vectors, with values,
    # normalization and margins. No parameters. State: ind1, ind2, vec1 and vec2.
    def setup(self):
        # [FUNCTION DOC]
        # Draws 100000 random codes in two ranges (three and two values) as ind1 and
        # ind2, and maps them to the strings A, B, C (vec1) and one, two (vec2).
        N = 100000
        fac1 = np.array(["A", "B", "C"], dtype="O")
        fac2 = np.array(["one", "two"], dtype="O")
        self.ind1 = np.random.randint(0, 3, size=N)
        self.ind2 = np.random.randint(0, 2, size=N)
        self.vec1 = fac1.take(self.ind1)
        self.vec2 = fac2.take(self.ind2)

    def time_crosstab(self):
        # [FUNCTION DOC]
        # Times counting the occurrences of each pair of values.
        pd.crosstab(self.vec1, self.vec2)

    def time_crosstab_values(self):
        # [FUNCTION DOC]
        # Times summing the integer codes ind1 for each pair.
        pd.crosstab(self.vec1, self.vec2, values=self.ind1, aggfunc="sum")

    def time_crosstab_normalize(self):
        # [FUNCTION DOC]
        # Times the counts normalized by the grand total.
        pd.crosstab(self.vec1, self.vec2, normalize=True)

    def time_crosstab_normalize_margins(self):
        # [FUNCTION DOC]
        # Times the normalized counts with margins.
        pd.crosstab(self.vec1, self.vec2, normalize=True, margins=True)


class GetDummies:
    # [CLASS DOC]
    # Benchmark of dummy (indicator) encoding of a categorical series with 12
    # categories, dense or sparse. No parameters. State: s.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a categorical series of a million random draws among the first 12
        # ASCII letters, with those letters as the category set.
        categories = list(string.ascii_letters[:12])
        s = pd.Series(
            np.random.choice(categories, size=1000000),
            dtype=CategoricalDtype(categories),
        )
        self.s = s

    def time_get_dummies_1d(self):
        # [FUNCTION DOC]
        # Times dense dummy encoding of the series.
        pd.get_dummies(self.s, sparse=False)

    def time_get_dummies_1d_sparse(self):
        # [FUNCTION DOC]
        # Times sparse dummy encoding of the series.
        pd.get_dummies(self.s, sparse=True)


class Cut:
    # [CLASS DOC]
    # Benchmark of binning by value (cut) and by quantile (qcut) for integer,
    # float, timedelta and datetime series, plus binning into prepared interval
    # bins. Parametrized by the bin count. State: int_series, float_series,
    # timedelta_series, datetime_series and interval_bins.
    params = [[4, 10, 1000]]
    param_names = ["bins"]

    def setup(self, bins):
        # [FUNCTION DOC]
        # With N equal to 100000: int_series (the integers below N, each repeated
        # five times) and float_series (random normal values each repeated five
        # times), both 500000 long; timedelta_series and datetime_series (N random
        # integers below N interpreted as nanoseconds); and interval_bins, an
        # interval index built from evenly spaced breaks between 0 and N.
        N = 10**5
        self.int_series = pd.Series(np.arange(N).repeat(5))
        self.float_series = pd.Series(np.random.randn(N).repeat(5))
        self.timedelta_series = pd.Series(
            np.random.randint(N, size=N), dtype="timedelta64[ns]"
        )
        self.datetime_series = pd.Series(
            np.random.randint(N, size=N), dtype="datetime64[ns]"
        )
        self.interval_bins = pd.IntervalIndex.from_breaks(np.linspace(0, N, bins))

    def time_cut_int(self, bins):
        # [FUNCTION DOC]
        # Times cutting the integer series into the given number of equal-width bins.
        pd.cut(self.int_series, bins)

    def time_cut_float(self, bins):
        # [FUNCTION DOC]
        # Times cutting the float series into equal-width bins.
        pd.cut(self.float_series, bins)

    def time_cut_timedelta(self, bins):
        # [FUNCTION DOC]
        # Times cutting the timedelta series into equal-width bins.
        pd.cut(self.timedelta_series, bins)

    def time_cut_datetime(self, bins):
        # [FUNCTION DOC]
        # Times cutting the datetime series into equal-width bins.
        pd.cut(self.datetime_series, bins)

    def time_qcut_int(self, bins):
        # [FUNCTION DOC]
        # Times cutting the integer series into quantile bins.
        pd.qcut(self.int_series, bins)

    def time_qcut_float(self, bins):
        # [FUNCTION DOC]
        # Times cutting the float series into quantile bins.
        pd.qcut(self.float_series, bins)

    def time_qcut_timedelta(self, bins):
        # [FUNCTION DOC]
        # Times cutting the timedelta series into quantile bins.
        pd.qcut(self.timedelta_series, bins)

    def time_qcut_datetime(self, bins):
        # [FUNCTION DOC]
        # Times cutting the datetime series into quantile bins.
        pd.qcut(self.datetime_series, bins)

    def time_cut_interval(self, bins):
        # [FUNCTION DOC]
        # Times cutting the integer series into the prepared interval bins (issue
        # 27668).
        # GH 27668
        pd.cut(self.int_series, self.interval_bins)

    def peakmem_cut_interval(self, bins):
        # [FUNCTION DOC]
        # Peak-memory benchmark of cutting into the prepared interval bins (issue
        # 27668).
        # GH 27668
        pd.cut(self.int_series, self.interval_bins)


class Explode:
    # [CLASS DOC]
    # Benchmark of expanding list-like elements of a series into separate rows.
    # Parametrized by the number of rows and the maximum list length. State:
    # series.
    param_names = ["n_rows", "max_list_length"]
    params = [[100, 1000, 10000], [3, 5, 10]]

    def setup(self, n_rows, max_list_length):
        # [FUNCTION DOC]
        # Builds a series of n_rows arrays, each holding the integers from zero up
        # to a random length below max_list_length (so lengths range from empty to
        # one less than the maximum).
        data = [np.arange(np.random.randint(max_list_length)) for _ in range(n_rows)]
        self.series = pd.Series(data)

    def time_explode(self, n_rows, max_list_length):
        # [FUNCTION DOC]
        # Times exploding the series into one row per list element.
        self.series.explode()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
