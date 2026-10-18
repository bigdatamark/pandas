# [MODULE DOC]
# Tests for the testing-utility function that asserts two data frames are equal: row and column order handling, shape and index mismatches, dtype handling
# (including extension, interval and datetime-like dtypes), block-wise comparison, unicode content messages, the duplicate-labels flag, mixed-type and categorical
# indexes, messages naming extension columns, sets and nested frames, and the deprecated datetime-like compatibility keyword. Several tests also exercise the generic equality assertion and the series assertion.

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm


@pytest.fixture(params=[True, False])
def by_blocks_fixture(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false; supplies the by-blocks option for block-wise comparison tests.
    return request.param


def _assert_frame_equal_both(a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts equality in both directions.
    # Parameters a and b are the two frames; extra keyword arguments are forwarded to the frame equality assertion. Calls the assertion for (a, b) and then for (b, a). Raises an assertion error if either direction fails; returns nothing.
    """
    Check that two DataFrame equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : DataFrame
        The first DataFrame to compare.
    b : DataFrame
        The second DataFrame to compare.
    kwargs : dict
        The arguments passed to `tm.assert_frame_equal`.
    """
    tm.assert_frame_equal(a, b, **kwargs)
    tm.assert_frame_equal(b, a, **kwargs)


@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_row_order_mismatch(check_like, frame_or_series):
    # [FUNCTION DOC]
    # Parametrized over the like-ordering flag and the frame-or-series fixture (used only for the object label). Frames with reversed rows: if the flag is off, an assertion error says the index differs; if on, equality holds in both directions.
    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [3, 2, 1], "B": [6, 5, 4]}, index=["c", "b", "a"])

    if not check_like:  # Do not ignore row-column orderings.
        msg = f"{frame_or_series.__name__}.index are different"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(
                df1, df2, check_like=check_like, obj=frame_or_series.__name__
            )
    else:
        _assert_frame_equal_both(
            df1, df2, check_like=check_like, obj=frame_or_series.__name__
        )


@pytest.mark.parametrize(
    "df1,df2",
    [
        ({"A": [1, 2, 3]}, {"A": [1, 2, 3, 4]}),
        ({"A": [1, 2, 3], "B": [4, 5, 6]}, {"A": [1, 2, 3]}),
    ],
)
def test_frame_equal_shape_mismatch(df1, df2, frame_or_series):
    # [FUNCTION DOC]
    # Parametrized over frames with different row counts or column counts. Raises an assertion error whose message begins with the object label followed by "are different".
    df1 = DataFrame(df1)
    df2 = DataFrame(df2)
    msg = f"{frame_or_series.__name__} are different"

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, obj=frame_or_series.__name__)


@pytest.mark.parametrize(
    "df1,df2,msg",
    [
        # Index
        (
            DataFrame.from_records({"a": [1, 2], "c": ["l1", "l2"]}, index=["a"]),
            DataFrame.from_records({"a": [1.0, 2.0], "c": ["l1", "l2"]}, index=["a"]),
            "DataFrame\\.index are different",
        ),
        # MultiIndex
        (
            DataFrame.from_records(
                {"a": [1, 2], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ),
            DataFrame.from_records(
                {"a": [1.0, 2.0], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ),
            "DataFrame\\.index level \\[0\\] are different",
        ),
    ],
)
def test_frame_equal_index_dtype_mismatch(df1, df2, msg, check_index_type):
    # [FUNCTION DOC]
    # Parametrized over a single-index and multi-index pair differing in integer versus float index values, and the index-type check fixture. With the check on, an assertion error naming the index (or index level 0) is raised; with it off, the frames compare equal.
    kwargs = {"check_index_type": check_index_type}

    if check_index_type:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        tm.assert_frame_equal(df1, df2, **kwargs)


def test_empty_dtypes(check_dtype):
    # [FUNCTION DOC]
    # Uses the dtype check fixture. Two empty frames with the same columns where one column was cast to 64-bit integers: raises an assertion error about attributes differing when the check is on; passes when off.
    columns = ["col1", "col2"]
    df1 = DataFrame(columns=columns)
    df2 = DataFrame(columns=columns)

    kwargs = {"check_dtype": check_dtype}
    df1["col1"] = df1["col1"].astype("int64")

    if check_dtype:
        msg = r"Attributes of DataFrame\..* are different"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        tm.assert_frame_equal(df1, df2, **kwargs)


@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_index_mismatch(check_like, frame_or_series, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over the like-ordering flag and the frame-or-series label; the expected string dtype depends on the infer-string fixture. Frames with indexes differing in the last label raise an assertion error whose message shows 33.33 percent differing, both index listings, and the first differing position with the two labels.
    if using_infer_string:
        dtype = "str"
    else:
        dtype = "object"
    msg = f"""{frame_or_series.__name__}\\.index are different

{frame_or_series.__name__}\\.index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\['a', 'b', 'c'\\], dtype='{dtype}'\\)
\\[right\\]: Index\\(\\['a', 'b', 'd'\\], dtype='{dtype}'\\)
At positional index 2, first diff: c != d"""

    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "d"])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(
            df1, df2, check_like=check_like, obj=frame_or_series.__name__
        )


@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_columns_mismatch(check_like, frame_or_series, using_infer_string):
    # [FUNCTION DOC]
    # Same parametrization; frames whose second column labels differ in case raise an assertion error showing the columns differ in 50 percent with both listings.
    if using_infer_string:
        dtype = "str"
    else:
        dtype = "object"
    msg = f"""{frame_or_series.__name__}\\.columns are different

{frame_or_series.__name__}\\.columns values are different \\(50\\.0 %\\)
\\[left\\]:  Index\\(\\['A', 'B'\\], dtype='{dtype}'\\)
\\[right\\]: Index\\(\\['A', 'b'\\], dtype='{dtype}'\\)"""

    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [1, 2, 3], "b": [4, 5, 6]}, index=["a", "b", "c"])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(
            df1, df2, check_like=check_like, obj=frame_or_series.__name__
        )


def test_frame_equal_block_mismatch(by_blocks_fixture, frame_or_series):
    # [FUNCTION DOC]
    # Uses the by-blocks fixture and the label fixture. Frames differing in one value of the second column raise an assertion error showing the column position and name, the 33.33 percent difference, the index, and left and right values.
    obj = frame_or_series.__name__
    msg = f"""{obj}\\.iloc\\[:, 1\\] \\(column name="B"\\) are different

{obj}\\.iloc\\[:, 1\\] \\(column name="B"\\) values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[4, 5, 6\\]
\\[right\\]: \\[4, 5, 7\\]"""

    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, by_blocks=by_blocks_fixture, obj=obj)


@pytest.mark.parametrize(
    "df1,df2,msg",
    [
        (
            {"A": ["á", "à", "ä"], "E": ["é", "è", "ë"]},
            {"A": ["á", "à", "ä"], "E": ["é", "è", "e̊"]},
            """{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) are different

{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[é, è, ë\\]
\\[right\\]: \\[é, è, e̊\\]""",
        ),
        (
            {"A": ["á", "à", "ä"], "E": ["é", "è", "ë"]},
            {"A": ["a", "a", "a"], "E": ["e", "e", "e"]},
            """{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) are different

{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) values are different \\(100\\.0 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[a, a, a\\]""",
        ),
    ],
)
def test_frame_equal_unicode(df1, df2, msg, by_blocks_fixture, frame_or_series):
    # [FUNCTION DOC]
    # Regression test (gh-20503). Parametrized over two pairs of frames with differing unicode strings (a combining-character difference in one column; entirely different accent-stripped values in another), expected message templates filled with the object label, by-blocks fixture and label fixture. Asserts the assertion error message matches, showing the offending column and values.
    # see gh-20503
    #
    # Test ensures that `tm.assert_frame_equals` raises the right exception
    # when comparing DataFrames containing differing unicode objects.
    df1 = DataFrame(df1)
    df2 = DataFrame(df2)
    msg = msg.format(obj=frame_or_series.__name__)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(
            df1, df2, by_blocks=by_blocks_fixture, obj=frame_or_series.__name__
        )


def test_assert_frame_equal_extension_dtype_mismatch():
    # [FUNCTION DOC]
    # Issue 32747. A nullable Int64 frame and its cast to plain integers compare equal with the dtype check off, and with it on raise an assertion error naming the first column's dtype attribute difference.
    # KNOWN ISSUE: the expected message pattern writes the integer width choice as a bracketed set of characters with a vertical bar, which matches any one of those characters rather than the alternatives 32 or 64, so the pattern is looser than intended.
    # https://github.com/pandas-dev/pandas/issues/32747
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = left.astype(int)

    msg = (
        "Attributes of DataFrame\\.iloc\\[:, 0\\] "
        '\\(column name="a"\\) are different\n\n'
        'Attribute "dtype" are different\n'
        "\\[left\\]:  Int64\n"
        "\\[right\\]: int[32|64]"
    )

    tm.assert_frame_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)


def test_assert_frame_equal_interval_dtype_mismatch():
    # [FUNCTION DOC]
    # Issue 32747. An interval-dtype frame versus its object-cast copy compares equal with the dtype check off and raises (naming the interval and object dtypes) with it on.
    # https://github.com/pandas-dev/pandas/issues/32747
    left = DataFrame({"a": [pd.Interval(0, 1)]}, dtype="interval")
    right = left.astype(object)

    msg = (
        "Attributes of DataFrame\\.iloc\\[:, 0\\] "
        '\\(column name="a"\\) are different\n\n'
        'Attribute "dtype" are different\n'
        "\\[left\\]:  interval\\[int64, right\\]\n"
        "\\[right\\]: object"
    )

    tm.assert_frame_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)


def test_assert_frame_equal_ignore_extension_dtype_mismatch():
    # [FUNCTION DOC]
    # Issue 35715. Nullable Int64 and Int32 frames compare equal with the dtype check off.
    # https://github.com/pandas-dev/pandas/issues/35715
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = DataFrame({"a": [1, 2, 3]}, dtype="Int32")
    tm.assert_frame_equal(left, right, check_dtype=False)


def test_assert_frame_equal_ignore_extension_dtype_mismatch_cross_class():
    # [FUNCTION DOC]
    # Issue 35715. Nullable Int64 and numpy int64 frames compare equal with the dtype check off.
    # https://github.com/pandas-dev/pandas/issues/35715
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = DataFrame({"a": [1, 2, 3]}, dtype="int64")
    tm.assert_frame_equal(left, right, check_dtype=False)


@pytest.mark.parametrize(
    "dtype", ["timedelta64[ns]", "datetime64[ns, UTC]", "Period[D]"]
)
def test_assert_frame_equal_datetime_like_dtype_mismatch(dtype):
    # [FUNCTION DOC]
    # Parametrized over timedelta, UTC datetime and period dtypes. An empty frame with that dtype and an empty default-dtype frame compare equal with the dtype check off.
    df1 = DataFrame({"a": []}, dtype=dtype)
    df2 = DataFrame({"a": []})
    tm.assert_frame_equal(df1, df2, check_dtype=False)


def test_allows_duplicate_labels():
    # [FUNCTION DOC]
    # Frames equal themselves; an empty frame and one with duplicate labels disallowed compare equal when flags are not checked (both orders), and raise an assertion error mentioning the flags object when flags are checked.
    # KNOWN ISSUE: the last two assertion blocks are identical (both compare left to right); the second probably was meant to swap the arguments.
    left = DataFrame()
    right = DataFrame().set_flags(allows_duplicate_labels=False)
    tm.assert_frame_equal(left, left)
    tm.assert_frame_equal(right, right)
    tm.assert_frame_equal(left, right, check_flags=False)
    tm.assert_frame_equal(right, left, check_flags=False)

    with pytest.raises(AssertionError, match="<Flags"):
        tm.assert_frame_equal(left, right)

    with pytest.raises(AssertionError, match="<Flags"):
        tm.assert_frame_equal(left, right)


def test_assert_frame_equal_columns_mixed_dtype():
    # [FUNCTION DOC]
    # GH#39168. A frame with mixed-type column labels and index labels equals itself with the like-ordering option.
    # GH#39168
    df = DataFrame([[0, 1, 2]], columns=["foo", "bar", 42], index=[1, "test", 2])
    tm.assert_frame_equal(df, df, check_like=True)


def test_frame_equal_extension_dtype(frame_or_series, any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # GH#39410. Parametrized over frame-or-series and numeric extension dtypes. The generic equality assertion with exact comparison passes for an object compared with itself.
    # GH#39410
    obj = frame_or_series([1, 2], dtype=any_numeric_ea_dtype)
    tm.assert_equal(obj, obj, check_exact=True)


@pytest.mark.parametrize("indexer", [(0, 1), (1, 0)])
def test_frame_equal_mixed_dtypes(frame_or_series, any_numeric_ea_dtype, indexer):
    # [FUNCTION DOC]
    # Parametrized over the order of dtypes (extension numeric, int64). Objects with equal values but different integer dtypes pass the generic equality with exact comparison and the dtype check off.
    dtypes = (any_numeric_ea_dtype, "int64")
    obj1 = frame_or_series([1, 2], dtype=dtypes[indexer[0]])
    obj2 = frame_or_series([1, 2], dtype=dtypes[indexer[1]])
    tm.assert_equal(obj1, obj2, check_exact=True, check_dtype=False)


def test_assert_frame_equal_check_like_different_indexes():
    # [FUNCTION DOC]
    # GH#39739. Empty frames with an empty object index versus an empty range index raise an assertion error that the index differs even with the like-ordering option.
    # GH#39739
    df1 = DataFrame(index=pd.Index([], dtype="object"))
    df2 = DataFrame(index=pd.RangeIndex(start=0, stop=0, step=1))
    with pytest.raises(AssertionError, match="DataFrame.index are different"):
        tm.assert_frame_equal(df1, df2, check_like=True)


def test_assert_frame_equal_checking_allow_dups_flag():
    # [FUNCTION DOC]
    # GH#45554. Frames differing only in the duplicate-labels flag pass with flag checking off and raise an assertion error mentioning the flag when it is on.
    # GH#45554
    left = DataFrame([[1, 2], [3, 4]])
    left.flags.allows_duplicate_labels = False

    right = DataFrame([[1, 2], [3, 4]])
    right.flags.allows_duplicate_labels = True
    tm.assert_frame_equal(left, right, check_flags=False)

    with pytest.raises(AssertionError, match="allows_duplicate_labels"):
        tm.assert_frame_equal(left, right, check_flags=True)


def test_assert_frame_equal_check_like_categorical_midx():
    # [FUNCTION DOC]
    # GH#48975. Frames with categorical multi-level indexes in reversed order compare equal with the like-ordering option.
    # GH#48975
    left = DataFrame(
        [[1], [2], [3]],
        index=pd.MultiIndex.from_arrays(
            [
                pd.Categorical(["a", "b", "c"]),
                pd.Categorical(["a", "b", "c"]),
            ]
        ),
    )
    right = DataFrame(
        [[3], [2], [1]],
        index=pd.MultiIndex.from_arrays(
            [
                pd.Categorical(["c", "b", "a"]),
                pd.Categorical(["c", "b", "a"]),
            ]
        ),
    )
    tm.assert_frame_equal(left, right, check_like=True)


def test_assert_frame_equal_ea_column_definition_in_exception_mask():
    # [FUNCTION DOC]
    # GH#50323. Nullable integer columns differing in a missing value raise an assertion error whose message names the column and says NA mask values differ.
    # GH#50323
    df1 = DataFrame({"a": pd.Series([pd.NA, 1], dtype="Int64")})
    df2 = DataFrame({"a": pd.Series([1, 1], dtype="Int64")})

    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) NA mask values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)


def test_assert_frame_equal_ea_column_definition_in_exception():
    # [FUNCTION DOC]
    # GH#50323. Nullable integer columns differing in a value raise an assertion error naming the column and saying values differ, both by default and with exact comparison.
    # GH#50323
    df1 = DataFrame({"a": pd.Series([pd.NA, 1], dtype="Int64")})
    df2 = DataFrame({"a": pd.Series([pd.NA, 2], dtype="Int64")})

    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_exact=True)


def test_assert_frame_equal_ts_column():
    # [FUNCTION DOC]
    # GH#50323. Timestamp columns that differ raise an assertion error naming the column and saying values differ.
    # GH#50323
    df1 = DataFrame({"a": [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-12-31")]})
    df2 = DataFrame({"a": [pd.Timestamp("2020-12-31"), pd.Timestamp("2020-12-31")]})

    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)


def test_assert_frame_equal_set():
    # [FUNCTION DOC]
    # GH#51727. Frames holding equal sets in a column compare equal.
    # GH#51727
    df1 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})
    df2 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})
    tm.assert_frame_equal(df1, df2)


def test_assert_frame_equal_set_mismatch():
    # [FUNCTION DOC]
    # GH#51727. Frames holding sets that differ raise an assertion error naming the column and saying values differ.
    # GH#51727
    df1 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})
    df2 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 7}]})

    msg = r'DataFrame.iloc\[:, 0\] \(column name="set_column"\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)


def test_datetimelike_compat_deprecated():
    # [FUNCTION DOC]
    # GH#55638. Passing the datetime-like compatibility keyword (true or false) to the frame and series equality assertions emits the future-removal warning that the keyword is deprecated.
    # GH#55638
    df = DataFrame({"a": [1]})

    msg = "the 'check_datetimelike_compat' keyword is deprecated"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        tm.assert_frame_equal(df, df, check_datetimelike_compat=True)
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        tm.assert_frame_equal(df, df, check_datetimelike_compat=False)

    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        tm.assert_series_equal(df["a"], df["a"], check_datetimelike_compat=True)
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        tm.assert_series_equal(df["a"], df["a"], check_datetimelike_compat=False)


@pytest.mark.parametrize("na_value", [pd.NA, np.nan, None])
def test_assert_frame_equal_nested_df_na(na_value):
    # [FUNCTION DOC]
    # GH#43022. Parametrized over NA, NaN and None. Frames containing a nested frame with that missing value compare equal.
    # GH#43022
    inner = DataFrame({"a": [1, na_value]})
    df1 = DataFrame({"df": [inner]})
    df2 = DataFrame({"df": [inner]})
    tm.assert_frame_equal(df1, df2)
