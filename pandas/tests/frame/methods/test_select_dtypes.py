# [MODULE DOC]
# Tests for DataFrame select_dtypes: include and exclude as scalars and lists, numpy and pandas dtype aliases, duplicate columns, strings and object, extension dtypes, datetime specificity, errors and copy semantics.
# Defines a minimal extension dtype and array pair for numeric-ness tests.

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas.core.dtypes.dtypes import ExtensionDtype

import pandas as pd
from pandas import (
    DataFrame,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray


class DummyDtype(ExtensionDtype):
    # [CLASS DOC]
    # HELPER: minimal extension dtype used to test whether select_dtypes honors the numeric flag of extension dtypes. Stores a numeric boolean given at construction; its name is always Dummy and its type is int.
    type = int

    def __init__(self, numeric) -> None:
        # [FUNCTION DOC]
        # HELPER: stores whether the dtype is considered numeric.
        self._numeric = numeric

    @property
    def name(self):
        # [FUNCTION DOC]
        # HELPER: property returning the constant name of the dummy dtype.
        return "Dummy"

    @property
    def _is_numeric(self):
        # [FUNCTION DOC]
        # HELPER: property returning the numeric flag given at construction.
        return self._numeric


class DummyArray(ExtensionArray):
    # [CLASS DOC]
    # HELPER: minimal extension array wrapping a sequence and a given dtype, used to test numeric selection of extension arrays. Only length, array conversion, dtype, copy and view are implemented; element access is a stub.
    def __init__(self, data, dtype) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the data and dtype.
        self.data = data
        self._dtype = dtype

    def __array__(self, dtype=None, copy=None):
        # [FUNCTION DOC]
        # HELPER: converts the stored data to a numpy array with the optionally requested dtype; the copy argument is ignored.
        return np.asarray(self.data, dtype=dtype)

    @property
    def dtype(self):
        # [FUNCTION DOC]
        # HELPER: property returning the dtype given at construction.
        return self._dtype

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # HELPER: returns the number of stored elements.
        return len(self.data)

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # HELPER: stub that ignores its argument and returns nothing; only present to satisfy the abstract interface.
        pass

    def copy(self):
        # [FUNCTION DOC]
        # HELPER: returns the array itself without copying.
        return self

    def view(self):
        # [FUNCTION DOC]
        # HELPER: returns the array itself.
        return self


class TestSelectDtypes:
    # [CLASS DOC]
    # Groups the select_dtypes tests.
    def test_select_dtypes_include_using_list_like(self, using_infer_string):
        # [FUNCTION DOC]
        # Frame with columns of many dtypes (string, int, unsigned, float, bool, categorical, naive and timezone-aware datetime, period, timedelta). Verifies include lists: numbers (including timedelta) , numbers excluding timedelta, numbers plus categorical, datetime, datetime64 and datetimetz selections; period raises a not-implemented error with an empty message; with inferred strings, str and the str type select the string column; selecting object emits a deprecation warning (Pandas4Warning) about str inclusion under inferred strings and selects the string column.
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(include=[np.number])
        ei = df[["b", "c", "d", "k"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=[np.number], exclude=["timedelta"])
        ei = df[["b", "c", "d"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=[np.number, "category"], exclude=["timedelta"])
        ei = df[["b", "c", "d", "f"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=["datetime"])
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=["datetime64"])
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=["datetimetz"])
        ei = df[["h", "i"]]
        tm.assert_frame_equal(ri, ei)

        with pytest.raises(NotImplementedError, match=r"^$"):
            df.select_dtypes(include=["period"])

        if using_infer_string:
            ri = df.select_dtypes(include=["str"])
            ei = df[["a"]]
            tm.assert_frame_equal(ri, ei)

            ri = df.select_dtypes(include=[str])
            tm.assert_frame_equal(ri, ei)

        msg = "For backward compatibility, 'str' dtypes are included"
        warn = None
        if using_infer_string:
            warn = Pandas4Warning
        with tm.assert_produces_warning(warn, match=msg):
            ri = df.select_dtypes(include=["object"])
        ei = df[["a"]]
        tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_exclude_using_list_like(self):
        # [FUNCTION DOC]
        # Verifies excluding numbers from a frame with strings, ints, floats and bools keeps the string and bool columns (bool is not numeric here).
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
            }
        )
        re = df.select_dtypes(exclude=[np.number])
        ee = df[["a", "e"]]
        tm.assert_frame_equal(re, ee)

    def test_select_dtypes_exclude_include_using_list_like(self):
        # [FUNCTION DOC]
        # Verifies combining include (bool and integer) with exclude (datetime64) selects the integer and bool columns, also when given as tuples of names.
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6, dtype="u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        exclude = (np.datetime64,)
        include = np.bool_, "integer"
        r = df.select_dtypes(include=include, exclude=exclude)
        e = df[["b", "c", "e"]]
        tm.assert_frame_equal(r, e)

        exclude = ("datetime",)
        include = "bool", "int64", "int32"
        r = df.select_dtypes(include=include, exclude=exclude)
        e = df[["b", "e"]]
        tm.assert_frame_equal(r, e)

    @pytest.mark.parametrize(
        "include", [(np.bool_, "int"), (np.bool_, "integer"), ("bool", int)]
    )
    def test_select_dtypes_exclude_include_int(self, include):
        # [FUNCTION DOC]
        # Fix for #36596. Parametrized over include tuples mixing bool with int, integer or the int type. Verifies integer columns of different widths and booleans are selected while datetimes are excluded.
        # Fix select_dtypes(include='int') for Windows, FYI #36596
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6, dtype="int32"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        exclude = (np.datetime64,)
        result = df.select_dtypes(include=include, exclude=exclude)
        expected = df[["b", "c", "e"]]
        tm.assert_frame_equal(result, expected)

    def test_select_dtypes_include_using_scalars(self, using_infer_string):
        # [FUNCTION DOC]
        # Same wide frame as the list test but passing scalar includes: number, datetime, datetime64, category; period raises not-implemented; with inferred strings, str selects the string column.
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(include=np.number)
        ei = df[["b", "c", "d", "k"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include="datetime")
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include="datetime64")
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include="category")
        ei = df[["f"]]
        tm.assert_frame_equal(ri, ei)

        with pytest.raises(NotImplementedError, match=r"^$"):
            df.select_dtypes(include="period")

        if using_infer_string:
            ri = df.select_dtypes(include="str")
            ei = df[["a"]]
            tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_exclude_using_scalars(self):
        # [FUNCTION DOC]
        # Verifies scalar exclude: excluding numbers, excluding category, and excluding period raising a not-implemented error.
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(exclude=np.number)
        ei = df[["a", "e", "f", "g", "h", "i", "j"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(exclude="category")
        ei = df[["a", "b", "c", "d", "e", "g", "h", "i", "j", "k"]]
        tm.assert_frame_equal(ri, ei)

        with pytest.raises(NotImplementedError, match=r"^$"):
            df.select_dtypes(exclude="period")

    def test_select_dtypes_include_exclude_using_scalars(self):
        # [FUNCTION DOC]
        # Verifies scalar include of numbers together with scalar exclude of floating selects integer, unsigned and timedelta columns.
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(include=np.number, exclude="floating")
        ei = df[["b", "c", "k"]]
        tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_include_exclude_mixed_scalars_lists(self):
        # [FUNCTION DOC]
        # Verifies mixing scalar and list arguments: numbers excluding floating and timedelta, and numbers plus category excluding floating.
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(include=np.number, exclude=["floating", "timedelta"])
        ei = df[["b", "c"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=[np.number, "category"], exclude="floating")
        ei = df[["b", "c", "f", "k"]]
        tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_duplicate_columns(self):
        # [FUNCTION DOC]
        # GH20839. Frame with duplicated column labels. Verifies selection works with duplicates, returning every column of the selected dtypes under their repeated labels.
        # GH20839
        df = DataFrame(
            {
                "a": ["a", "b", "c"],
                "b": [1, 2, 3],
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        df.columns = ["a", "a", "b", "b", "b", "c"]

        expected = DataFrame(
            {"a": list(range(1, 4)), "b": np.arange(3, 6).astype("u1")}
        )

        result = df.select_dtypes(include=[np.number], exclude=["floating"])
        tm.assert_frame_equal(result, expected)

    def test_select_dtypes_not_an_attr_but_still_valid_dtype(self, using_infer_string):
        # [FUNCTION DOC]
        # Checks short numpy dtype codes work even if not attributes of numpy (i8, O, timedelta). Verifies the selected columns, and under inferred strings that the object code emits the deprecation warning about str inclusion.
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        df["g"] = df.f.diff()
        assert not hasattr(np, "u8")

        msg = "For backward compatibility, 'str' dtypes are included"
        warn = None
        if using_infer_string:
            warn = Pandas4Warning
        with tm.assert_produces_warning(warn, match=msg):
            r = df.select_dtypes(include=["i8", "O"], exclude=["timedelta"])
        e = df[["a", "b"]]
        tm.assert_frame_equal(r, e)

        with tm.assert_produces_warning(warn, match=msg):
            r = df.select_dtypes(include=["i8", "O", "timedelta64[ns]"])
        e = df[["a", "b", "g"]]
        tm.assert_frame_equal(r, e)

    def test_select_dtypes_empty(self):
        # [FUNCTION DOC]
        # Verifies calling with neither include nor exclude raises a value error saying at least one must be nonempty.
        df = DataFrame({"a": list("abc"), "b": list(range(1, 4))})
        msg = "at least one of include or exclude must be nonempty"
        with pytest.raises(ValueError, match=msg):
            df.select_dtypes()

    def test_select_dtypes_bad_datetime64(self):
        # [FUNCTION DOC]
        # Verifies datetime64 dtypes with a specific unit (days or attoseconds) raise a value error saying the dtype is too specific.
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        with pytest.raises(ValueError, match=".+ is too specific"):
            df.select_dtypes(include=["datetime64[D]"])

        with pytest.raises(ValueError, match=".+ is too specific"):
            df.select_dtypes(exclude=["datetime64[as]"])

    def test_select_dtypes_datetime_with_tz(self):
        # [FUNCTION DOC]
        # Verifies selecting datetime64 of nanoseconds does not select timezone-aware columns (empty-column result with the same index).
        df2 = DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        )
        df3 = pd.concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
        result = df3.select_dtypes(include=["datetime64[ns]"])
        expected = df3.reindex(columns=[])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", [str, "str", np.bytes_, "S1", np.str_, "U1"])
    @pytest.mark.parametrize("arg", ["include", "exclude"])
    def test_select_dtypes_str_raises(self, dtype, arg, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized over string-like dtypes (str, bytes and unicode types and codes) and include versus exclude. Verifies a type error saying string dtypes are not allowed; skipped for str and "str" when inferred strings are active because those work then.
        if using_infer_string and (dtype == "str" or dtype is str):
            # this is tested below
            pytest.skip("Selecting string columns works with future strings")
        df = DataFrame(
            {
                "a": list("abc"),
                "g": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        msg = "string dtypes are not allowed"
        kwargs = {arg: [dtype]}

        with pytest.raises(TypeError, match=msg):
            df.select_dtypes(**kwargs)

    def test_select_dtypes_bad_arg_raises(self):
        # [FUNCTION DOC]
        # Verifies an unrecognized dtype string raises a type error saying the data type was not understood.
        df = DataFrame(
            {
                "a": list("abc"),
                "g": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )

        msg = "data type.*not understood"
        with pytest.raises(TypeError, match=msg):
            df.select_dtypes(["blargy, blarg, blarg"])

    def test_select_dtypes_typecodes(self):
        # [FUNCTION DOC]
        # GH 11990. Verifies selecting by the list of all float type codes returns the whole float frame.
        # GH 11990
        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        FLOAT_TYPES = list(np.typecodes["AllFloat"])
        tm.assert_frame_equal(df.select_dtypes(FLOAT_TYPES), df)

    @pytest.mark.parametrize(
        "arr,expected",
        (
            (np.array([1, 2], dtype=np.int32), True),
            (pd.array([1, 2], dtype="Int32"), True),
            (DummyArray([1, 2], dtype=DummyDtype(numeric=True)), True),
            (DummyArray([1, 2], dtype=DummyDtype(numeric=False)), False),
        ),
    )
    def test_select_dtypes_numeric(self, arr, expected):
        # [FUNCTION DOC]
        # GH 35340. Parametrized over arrays (numpy int32, nullable Int32, numeric dummy extension array, non-numeric dummy extension array) and expected selection flag. Verifies selecting numbers keeps the column exactly when the dtype is numeric.
        # GH 35340

        df = DataFrame(arr)
        is_selected = df.select_dtypes(np.number).shape == df.shape
        assert is_selected == expected

    def test_select_dtypes_numeric_nullable_string(self, nullable_string_dtype):
        # [FUNCTION DOC]
        # Uses the nullable_string_dtype fixture. Verifies selecting numbers does not select a nullable string column.
        arr = pd.array(["a", "b"], dtype=nullable_string_dtype)
        df = DataFrame(arr)
        is_selected = df.select_dtypes(np.number).shape == df.shape
        assert not is_selected

    @pytest.mark.parametrize(
        "expected, float_dtypes",
        [
            [
                DataFrame(
                    {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)}
                ).astype(dtype={"A": float, "B": np.float64, "C": np.float32}),
                float,
            ],
            [
                DataFrame(
                    {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)}
                ).astype(dtype={"A": float, "B": np.float64, "C": np.float32}),
                "float",
            ],
            [DataFrame({"C": range(10, 7, -1)}, dtype=np.float32), np.float32],
            [
                DataFrame({"A": range(3), "B": range(5, 8)}).astype(
                    dtype={"A": float, "B": np.float64}
                ),
                np.float64,
            ],
        ],
    )
    def test_select_dtypes_float_dtype(self, expected, float_dtypes):
        # [FUNCTION DOC]
        # GH#42452. Parametrized over expected frames and float dtype specifiers (float type, float string, 32-bit and 64-bit numpy floats). Verifies including a float specifier selects the columns of exactly that dtype (the generic float means 64-bit).
        # GH#42452
        dtype_dict = {"A": float, "B": np.float64, "C": np.float32}
        df = DataFrame(
            {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)},
        )
        df = df.astype(dtype_dict)
        result = df.select_dtypes(include=float_dtypes)
        tm.assert_frame_equal(result, expected)

    def test_np_bool_ea_boolean_include_number(self):
        # [FUNCTION DOC]
        # GH 46870. Verifies including numbers selects only the integer column, not boolean (numpy, nullable, categorical or sparse boolean) columns.
        # GH 46870
        df = DataFrame(
            {
                "a": [1, 2, 3],
                "b": pd.Series([True, False, True], dtype="boolean"),
                "c": np.array([True, False, True]),
                "d": pd.Categorical([True, False, True]),
                "e": pd.arrays.SparseArray([True, False, True]),
            }
        )
        result = df.select_dtypes(include="number")
        expected = DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_select_dtypes_no_view(self):
        # [FUNCTION DOC]
        # Issue 48090. Verifies the selection result is not a view: changing it leaves the original frame unchanged.
        # https://github.com/pandas-dev/pandas/issues/48090
        # result of this method is not a view on the original dataframe
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_orig = df.copy()
        result = df.select_dtypes(include=["number"])
        result.iloc[0, 0] = 0
        tm.assert_frame_equal(df, df_orig)

    def test_select_dtype_object_and_str(self, using_infer_string):
        # [FUNCTION DOC]
        # Issue 61916. Frame with default string, integer and nullable string columns. Verifies including object selects only the default string column (warning under inferred strings), and including string selects the nullable string column, plus the default string column when inferred strings are active.
        # https://github.com/pandas-dev/pandas/issues/61916
        df = DataFrame(
            {
                "a": ["a", "b", "c"],
                "b": [1, 2, 3],
                "c": pd.array(["a", "b", "c"], dtype="string"),
            }
        )

        # with "object" -> only select the object or default str dtype column
        msg = "For backward compatibility, 'str' dtypes are included"
        warn = None
        if using_infer_string:
            warn = Pandas4Warning
        with tm.assert_produces_warning(warn, match=msg):
            result = df.select_dtypes(include=["object"])
        expected = df[["a"]]
        tm.assert_frame_equal(result, expected)

        # with "string" -> select both the default 'str' and the nullable 'string'
        result = df.select_dtypes(include=["string"])
        if using_infer_string:
            expected = df[["a", "c"]]
        else:
            expected = df[["c"]]
        tm.assert_frame_equal(result, expected)
