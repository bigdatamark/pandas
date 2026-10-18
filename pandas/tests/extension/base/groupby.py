# [MODULE DOC]
# Base tests for groupby behavior with extension arrays as grouping keys and values.
# Public contents: BaseGroupbyTests mixin. Relies on the data_for_grouping fixture (eight
# values with a defined grouping layout, documented in the conftest) and groupby_apply_op.
import re

import pytest

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

import pandas as pd
import pandas._testing as tm


@pytest.mark.filterwarnings(
    "ignore:The default of observed=False is deprecated:FutureWarning"
)
class BaseGroupbyTests:
    # [CLASS DOC]
    # Mixin with groupby tests. The class is decorated to ignore the future warning about the
    # observed default for categorical groupers.
    """Groupby-specific tests."""

    def test_grouping_grouper(self, data_for_grouping):
        # [FUNCTION DOC]
        # Groups a frame by an object column and by the extension column and asserts the first
        # grouping's key vector is the numpy values for the object column and the original extension
        # array for the extension column.
        df = pd.DataFrame(
            {
                "A": pd.Series(
                    ["B", "B", None, None, "A", "A", "B", "C"], dtype=object
                ),
                "B": data_for_grouping,
            }
        )
        gr1 = df.groupby("A")._grouper.groupings[0]
        gr2 = df.groupby("B")._grouper.groupings[0]

        tm.assert_numpy_array_equal(gr1.grouping_vector, df.A.values)
        tm.assert_extension_array_equal(gr2.grouping_vector, data_for_grouping)

    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        # [FUNCTION DOC]
        # Parametrized over as_index. Averages a numeric column grouped by the extension column.
        # Expected groups are the sorted unique values with means 3, 1, 4. For boolean dtypes (only two
        # unique values) the last row is dropped and the last expected value too. as_index true gives
        # a named series indexed by the unique values; false gives a frame with the key column and the
        # mean column.
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values, and the final entry has c==b
            #  (see data_for_grouping docstring)
            df = df.iloc[:-1]

        result = df.groupby("B", as_index=as_index).A.mean()
        _, uniques = pd.factorize(data_for_grouping, sort=True)

        exp_vals = [3.0, 1.0, 4.0]
        if is_bool:
            exp_vals = exp_vals[:-1]
        if as_index:
            index = pd.Index(uniques, name="B")
            expected = pd.Series(exp_vals, index=index, name="A")
            tm.assert_series_equal(result, expected)
        else:
            expected = pd.DataFrame({"B": uniques, "A": exp_vals})
            tm.assert_frame_equal(result, expected)

    def test_groupby_agg_extension(self, data_for_grouping):
        # [FUNCTION DOC]
        # Regression for an upstream issue (aggregation fails on non-numeric extension types).
        # Grouping by an integer column and aggregating the extension column with first, by dict, by
        # name and by method, must equal the rows at the first occurrence of each group, indexed by the
        # key column.
        # GH#38980 groupby agg on extension type fails for non-numeric types
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        expected = df.iloc[[0, 2, 4, 7]]
        expected = expected.set_index("A")

        result = df.groupby("A").agg({"B": "first"})
        tm.assert_frame_equal(result, expected)

        result = df.groupby("A").agg("first")
        tm.assert_frame_equal(result, expected)

        result = df.groupby("A").first()
        tm.assert_frame_equal(result, expected)

    def test_groupby_extension_no_sort(self, data_for_grouping):
        # [FUNCTION DOC]
        # Like the aggregation test but with sort disabled: means come in order of first appearance
        # (1, 3, 4; boolean drops the last row and value), with the index from unsorted factorization of
        # the extension array.
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values, and the final entry has c==b
            #  (see data_for_grouping docstring)
            df = df.iloc[:-1]

        result = df.groupby("B", sort=False).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=False)

        index = pd.Index(index, name="B")
        exp_vals = [1.0, 3.0, 4.0]
        if is_bool:
            exp_vals = exp_vals[:-1]
        expected = pd.Series(exp_vals, index=index, name="A")
        tm.assert_series_equal(result, expected)

    def test_groupby_extension_transform(self, data_for_grouping):
        # [FUNCTION DOC]
        # Drops missing entries of the data, groups by an integer column, and transforms the value
        # column with a length function. Expected group sizes are 3, 3, 2, 2, 3, 1 (boolean drops the
        # last element).
        # KNOWN ISSUE: the boolean-dtype flag is assigned twice (the first assignment is redundant).
        is_bool = data_for_grouping.dtype._is_boolean

        valid = data_for_grouping[~data_for_grouping.isna()]
        df = pd.DataFrame({"A": [1, 1, 3, 3, 1, 4], "B": valid})
        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values, and the final entry has c==b
            #  (see data_for_grouping docstring)
            df = df.iloc[:-1]

        result = df.groupby("B").A.transform(len)
        expected = pd.Series([3, 3, 2, 2, 3, 1], name="A")
        if is_bool:
            expected = expected[:-1]

        tm.assert_series_equal(result, expected)

    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        # [FUNCTION DOC]
        # Smoke test: applies the fixture-provided function on groupby objects keyed by the
        # extension column and by the integer column, both for the frame and for selected columns. Only
        # checks that no exception is raised; results are discarded.
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        df.groupby("B", group_keys=False, observed=False).apply(groupby_apply_op)
        df.groupby("B", group_keys=False, observed=False).A.apply(groupby_apply_op)
        df.groupby("A", group_keys=False, observed=False).apply(groupby_apply_op)
        df.groupby("A", group_keys=False, observed=False).B.apply(groupby_apply_op)

    def test_groupby_apply_identity(self, data_for_grouping):
        # [FUNCTION DOC]
        # Grouping by an integer column and applying a function that returns each group's backing
        # array gives a series of arrays equal to the groups' element arrays (rows 0,1,6; 2,3; 4,5; 7),
        # indexed by the group key with the value column's name.
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("A").B.apply(lambda x: x.array)
        expected = pd.Series(
            [
                df.B.iloc[[0, 1, 6]].array,
                df.B.iloc[[2, 3]].array,
                df.B.iloc[[4, 5]].array,
                df.B.iloc[[7]].array,
            ],
            index=pd.Index([1, 2, 3, 4], name="A"),
            name="B",
        )
        tm.assert_series_equal(result, expected)

    def test_in_numeric_groupby(self, data_for_grouping):
        # [FUNCTION DOC]
        # Checks which columns survive a summing groupby. If the dtype is numeric, boolean,
        # decimal-named, string, object or timedelta-kind, the sum result has both extension and
        # integer columns. Otherwise summing must raise a type error (message alternatives for period,
        # datetime, and the generic aggregation failure that mentions the dtype), and restricting to
        # numeric columns leaves only the integer column.
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 4],
                "B": data_for_grouping,
                "C": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

        dtype = data_for_grouping.dtype
        if (
            is_numeric_dtype(dtype)
            or is_bool_dtype(dtype)
            or dtype.name == "decimal"
            or is_string_dtype(dtype)
            or is_object_dtype(dtype)
            or dtype.kind == "m"  # in particular duration[*][pyarrow]
        ):
            expected = pd.Index(["B", "C"])
            result = df.groupby("A").sum().columns
        else:
            expected = pd.Index(["C"])

            msg = "|".join(
                [
                    # period
                    "does not support sum operations",
                    # datetime
                    "does not support operation 'sum'",
                    # all others
                    re.escape(f"agg function failed [how->sum,dtype->{dtype}"),
                ]
            )
            with pytest.raises(TypeError, match=msg):
                df.groupby("A").sum()
            result = df.groupby("A").sum(numeric_only=True).columns
        tm.assert_index_equal(result, expected)
