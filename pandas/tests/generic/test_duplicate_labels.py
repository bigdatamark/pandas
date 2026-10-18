"""Tests dealing with the NDFrame.allows_duplicates."""
# [MODULE DOC]
# Tests for the flag that controls whether an object may carry duplicate labels
# (the allows-duplicate-labels flag on Series and DataFrame): which operations
# preserve the flag, which raise a duplicate-label error when it is disabled,
# and the formatting of the duplicate message. Defines an expected-failure marker
# for not-yet-implemented cases. Two test classes plus standalone tests.

import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

not_implemented = pytest.mark.xfail(reason="Not implemented.")

# ----------------------------------------------------------------------------
# Preservation


class TestPreserves:
    # [CLASS DOC]
    # Groups tests that the disallow-duplicates flag is preserved or correctly combined
    # by operations. Stateless.
    @pytest.mark.parametrize(
        "cls, data",
        [
            (pd.Series, np.array([])),
            (pd.Series, [1, 2]),
            (pd.DataFrame, {}),
            (pd.DataFrame, {"A": [1, 2]}),
        ],
    )
    def test_construction_ok(self, cls, data):
        # [FUNCTION DOC]
        # Verifies a newly constructed Series or DataFrame (empty or not) allows
        # duplicate labels by default,
        # and that disabling it via the set-flags method is reflected. Parametrized over
        # class and data.
        result = cls(data)
        assert result.flags.allows_duplicate_labels is True

        result = cls(data).set_flags(allows_duplicate_labels=False)
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize(
        "func",
        [
            operator.itemgetter(["a"]),
            operator.methodcaller("add", 1),
            operator.methodcaller("rename", str.upper),
            operator.methodcaller("rename", "name"),
            operator.methodcaller("abs"),
            np.abs,
        ],
    )
    def test_preserved_series(self, func):
        # [FUNCTION DOC]
        # Verifies Series operations (list getitem, add, rename with function and name,
        # abs, numpy abs) keep the flag disabled. Parametrized over the operation.
        s = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)
        assert func(s).flags.allows_duplicate_labels is False

    @pytest.mark.parametrize("index", [["a", "b", "c"], ["a", "b"]])
    # TODO: frame
    @not_implemented
    def test_align(self, index):
        # [FUNCTION DOC]
        # Verifies aligning a flag-disabled Series with another series keeps the flag
        # disabled on both outputs.
        # Parametrized over the other's index. Marked expected failure (not implemented;
        # frame case is a TODO).
        other = pd.Series(0, index=index)
        s = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)
        a, b = s.align(other)
        assert a.flags.allows_duplicate_labels is False
        assert b.flags.allows_duplicate_labels is False

    def test_preserved_frame(self):
        # [FUNCTION DOC]
        # Verifies row and column selection with loc on a flag-disabled frame keeps the
        # flag disabled.
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).set_flags(
            allows_duplicate_labels=False
        )
        assert df.loc[["a"]].flags.allows_duplicate_labels is False
        assert df.loc[:, ["A", "B"]].flags.allows_duplicate_labels is False

    def test_to_frame(self):
        # [FUNCTION DOC]
        # Verifies converting a flag-disabled empty Series to a frame keeps the flag
        # disabled.
        ser = pd.Series(dtype=float).set_flags(allows_duplicate_labels=False)
        assert ser.to_frame().flags.allows_duplicate_labels is False

    @pytest.mark.parametrize("func", ["add", "sub"])
    @pytest.mark.parametrize("frame", [False, True])
    @pytest.mark.parametrize("other", [1, pd.Series([1, 2], name="A")])
    def test_binops(self, func, other, frame):
        # [FUNCTION DOC]
        # Verifies add and subtract with a scalar or a series (converted to frame when
        # needed) keep the flag disabled.
        # Parametrized over method, operand and whether the left side is a frame.
        df = pd.Series([1, 2], name="A", index=["a", "b"]).set_flags(
            allows_duplicate_labels=False
        )
        if frame:
            df = df.to_frame()
        if isinstance(other, pd.Series) and frame:
            other = other.to_frame()
        func = operator.methodcaller(func, other)
        assert df.flags.allows_duplicate_labels is False
        assert func(df).flags.allows_duplicate_labels is False

    def test_preserve_getitem(self):
        # [FUNCTION DOC]
        # Verifies several getitem forms on a flag-disabled frame (column list, column,
        # loc row, loc row list, loc with row and column list) keep the flag disabled.
        df = pd.DataFrame({"A": [1, 2]}).set_flags(allows_duplicate_labels=False)
        assert df[["A"]].flags.allows_duplicate_labels is False
        assert df["A"].flags.allows_duplicate_labels is False
        assert df.loc[0].flags.allows_duplicate_labels is False
        assert df.loc[[0]].flags.allows_duplicate_labels is False
        assert df.loc[0, ["A"]].flags.allows_duplicate_labels is False

    @pytest.mark.parametrize(
        "objs, kwargs",
        [
            # Series
            (
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["c", "d"]),
                ],
                {},
            ),
            (
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["a", "b"]),
                ],
                {"ignore_index": True},
            ),
            (
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["a", "b"]),
                ],
                {"axis": 1},
            ),
            # Frame
            (
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"A": [1, 2]}, index=["c", "d"]),
                ],
                {},
            ),
            (
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                ],
                {"ignore_index": True},
            ),
            (
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"B": [1, 2]}, index=["a", "b"]),
                ],
                {"axis": 1},
            ),
            # Series / Frame
            (
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.Series([1, 2], index=["a", "b"], name="B"),
                ],
                {"axis": 1},
            ),
        ],
    )
    def test_concat(self, objs, kwargs):
        # [FUNCTION DOC]
        # Verifies concatenation of flag-disabled objects (series, frames, mixed;
        # stacked, ignoring index, or along columns) yields a result with the flag
        # disabled.
        # Parametrized over the objects and concat keywords; all inputs have the flag
        # disabled first.
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        result = pd.concat(objs, **kwargs)
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            # false false false
            pytest.param(
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
                    allows_duplicate_labels=False
                ),
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]).set_flags(
                    allows_duplicate_labels=False
                ),
                False,
            ),
            # false true false
            pytest.param(
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
                    allows_duplicate_labels=False
                ),
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]),
                False,
            ),
            # true true true
            (
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]),
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]),
                True,
            ),
        ],
    )
    def test_merge(self, left, right, expected):
        # [FUNCTION DOC]
        # Verifies the index-on-index merge flag rule: the result's flag is disabled if
        # either input has it disabled, enabled only if both allow duplicates.
        # Parametrized over left frame, right frame and the expected boolean.
        result = pd.merge(left, right, left_index=True, right_index=True)
        assert result.flags.allows_duplicate_labels is expected

    @not_implemented
    def test_groupby(self):
        # [FUNCTION DOC]
        # Verifies a groupby aggregation of a flag-disabled frame keeps the flag
        # disabled. Marked expected failure (not implemented; the body notes it is
        # under-tested with TODOs for apply and transform).
        # XXX: This is under tested
        # TODO:
        #  - apply
        #  - transform
        #  - Should passing a grouper that disallows duplicates propagate?
        df = pd.DataFrame({"A": [1, 2, 3]}).set_flags(allows_duplicate_labels=False)
        result = df.groupby([0, 0, 1]).agg("count")
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize("frame", [True, False])
    @not_implemented
    def test_window(self, frame):
        # [FUNCTION DOC]
        # Verifies rolling, ewm and expanding means keep the flag disabled, for series
        # or frame (parametrized). Marked expected failure.
        # KNOWN ISSUE: the series constructor call passes a keyword for the flag, which
        # the constructor does not accept, so the test would fail at setup regardless of
        # window behavior; the expected-failure marker hides this.
        df = pd.Series(
            1,
            index=pd.date_range("2000", periods=12),
            name="A",
            allows_duplicate_labels=False,
        )
        if frame:
            df = df.to_frame()
        assert df.rolling(3).mean().flags.allows_duplicate_labels is False
        assert df.ewm(3).mean().flags.allows_duplicate_labels is False
        assert df.expanding(3).mean().flags.allows_duplicate_labels is False


# ----------------------------------------------------------------------------
# Raises


class TestRaises:
    # [CLASS DOC]
    # Groups tests where disabling or violating the flag must raise a duplicate-label
    # error. Stateless.
    @pytest.mark.parametrize(
        "cls, axes",
        [
            (pd.Series, {"index": ["a", "a"], "dtype": float}),
            (pd.DataFrame, {"index": ["a", "a"]}),
            (pd.DataFrame, {"index": ["a", "a"], "columns": ["b", "b"]}),
            (pd.DataFrame, {"columns": ["b", "b"]}),
        ],
    )
    def test_set_flags_with_duplicates(self, cls, axes):
        # [FUNCTION DOC]
        # Verifies objects with duplicate labels can be built (flag enabled), but
        # disabling the flag via set-flags raises a duplicate label error mentioning
        # duplicates.
        # Parametrized over class and constructor keywords (duplicate index, columns or
        # both).
        result = cls(**axes)
        assert result.flags.allows_duplicate_labels is True

        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            cls(**axes).set_flags(allows_duplicate_labels=False)

    @pytest.mark.parametrize(
        "data",
        [
            pd.Series(index=[0, 0], dtype=float),
            pd.DataFrame(index=[0, 0]),
            pd.DataFrame(columns=[0, 0]),
        ],
    )
    def test_setting_allows_duplicate_labels_raises(self, data):
        # [FUNCTION DOC]
        # Verifies assigning false to the flag on an object that already has duplicate
        # labels raises a duplicate label error and leaves the flag enabled.
        # Parametrized over series, frame with duplicate index, frame with duplicate
        # columns.
        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            data.flags.allows_duplicate_labels = False

        assert data.flags.allows_duplicate_labels is True

    def test_series_raises(self):
        # [FUNCTION DOC]
        # Verifies concatenating a plain Series with a flag-disabled Series whose labels
        # collide raises a duplicate label error.
        a = pd.Series(0, index=["a", "b"])
        b = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)
        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat([a, b])

    @pytest.mark.parametrize(
        "getter, target",
        [
            (operator.itemgetter(["A", "A"]), None),
            # loc
            (operator.itemgetter(["a", "a"]), "loc"),
            pytest.param(operator.itemgetter(("a", ["A", "A"])), "loc"),
            (operator.itemgetter((["a", "a"], "A")), "loc"),
            # iloc
            (operator.itemgetter([0, 0]), "iloc"),
            pytest.param(operator.itemgetter((0, [0, 0])), "iloc"),
            pytest.param(operator.itemgetter(([0, 0], 0)), "iloc"),
        ],
    )
    def test_getitem_raises(self, getter, target):
        # [FUNCTION DOC]
        # Verifies getitem forms that would produce duplicate labels (repeated column
        # names, repeated loc or iloc labels or positions, on rows or columns) raise a
        # duplicate label error on a flag-disabled frame.
        # Parametrized over a getter callable and an optional accessor name (none, loc
        # or iloc), which is applied to the frame first.
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).set_flags(
            allows_duplicate_labels=False
        )
        if target:
            # df, df.loc, or df.iloc
            target = getattr(df, target)
        else:
            target = df

        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            getter(target)

    def test_concat_raises(self):
        # [FUNCTION DOC]
        # Verifies concatenating flag-disabled series with equal names along columns
        # raises a duplicate label error.
        objs = [
            pd.Series(1, index=[0, 1], name="a"),
            pd.Series(2, index=[0, 1], name="a"),
        ]
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat(objs, axis=1)

    def test_merge_raises(self):
        # [FUNCTION DOC]
        # Verifies a merge of a flag-disabled frame with a frame whose index has
        # duplicates raises a duplicate label error.
        a = pd.DataFrame({"A": [0, 1, 2]}, index=["a", "b", "c"]).set_flags(
            allows_duplicate_labels=False
        )
        b = pd.DataFrame({"B": [0, 1, 2]}, index=["a", "b", "b"])
        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.merge(a, b, left_index=True, right_index=True)


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 1]),
        pd.Index(["a", "a"]),
        pd.Index([1.1, 1.1]),
        pd.PeriodIndex([pd.Period("2000", "D")] * 2),
        pd.DatetimeIndex([pd.Timestamp("2000")] * 2),
        pd.TimedeltaIndex([pd.Timedelta("1D")] * 2),
        pd.CategoricalIndex(["a", "a"]),
        pd.IntervalIndex([pd.Interval(0, 1)] * 2),
        pd.MultiIndex.from_tuples([("a", 1), ("a", 1)]),
    ],
    ids=lambda x: type(x).__name__,
)
def test_raises_basic(idx):
    # [FUNCTION DOC]
    # Verifies that for many index types (integer, string, float, period, datetime,
    # timedelta, categorical, interval, multi-index) with duplicates,
    # disabling the flag raises a duplicate label error for a Series index, a frame
    # index and frame columns.
    # Parametrized over the index; test ids are the index class names.
    msg = "Index has duplicates."
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.Series(1, index=idx).set_flags(allows_duplicate_labels=False)

    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame({"A": [1, 1]}, index=idx).set_flags(allows_duplicate_labels=False)

    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame([[1, 2]], columns=idx).set_flags(allows_duplicate_labels=False)


def test_format_duplicate_labels_message():
    # [FUNCTION DOC]
    # Verifies the private duplicate-message formatter on an index with repeated labels
    # returns a frame of label (index, named label) to a list of positions column,
    # listing only duplicated labels in order of first appearance.
    idx = pd.Index(["a", "b", "a", "b", "c"])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame(
        {"positions": [[0, 2], [1, 3]]}, index=pd.Index(["a", "b"], name="label")
    )
    tm.assert_frame_equal(result, expected)


def test_format_duplicate_labels_message_multi():
    # [FUNCTION DOC]
    # Same as the preceding message-format test but for a multi-index, expecting a
    # multi-index of the duplicated tuples as result index.
    idx = pd.MultiIndex.from_product([["A"], ["a", "b", "a", "b", "c"]])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame(
        {"positions": [[0, 2], [1, 3]]},
        index=pd.MultiIndex.from_product([["A"], ["a", "b"]]),
    )
    tm.assert_frame_equal(result, expected)


def test_dataframe_insert_raises():
    # [FUNCTION DOC]
    # Verifies inserting a duplicate column with duplicates allowed raises a value error
    # ("Cannot specify") on a flag-disabled frame.
    df = pd.DataFrame({"A": [1, 2]}).set_flags(allows_duplicate_labels=False)
    msg = "Cannot specify"
    with pytest.raises(ValueError, match=msg):
        df.insert(0, "A", [3, 4], allow_duplicates=True)


@pytest.mark.parametrize(
    "method, frame_only",
    [
        (operator.methodcaller("set_index", "A", inplace=True), True),
        (operator.methodcaller("reset_index", inplace=True), True),
        (operator.methodcaller("rename", lambda x: x, inplace=True), False),
    ],
)
def test_inplace_raises(method, frame_only):
    # [FUNCTION DOC]
    # Verifies in-place set_index, reset_index and rename on a flag-disabled frame (and
    # a flag-disabled series for rename) raise a value error ("Cannot specify").
    # Parametrized over the method and a flag saying whether it is frame-only.
    df = pd.DataFrame({"A": [0, 0], "B": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    s = df["A"]
    s.flags.allows_duplicate_labels = False
    msg = "Cannot specify"

    with pytest.raises(ValueError, match=msg):
        method(df)
    if not frame_only:
        with pytest.raises(ValueError, match=msg):
            method(s)


def test_pickle(temp_file):
    # [FUNCTION DOC]
    # Verifies the flag survives a pickle round trip for a Series and an empty-column
    # DataFrame, using the temp_file fixture for the pickle path.
    a = pd.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    b = tm.round_trip_pickle(a, temp_file)
    tm.assert_series_equal(a, b)

    a = pd.DataFrame({"A": []}).set_flags(allows_duplicate_labels=False)
    b = tm.round_trip_pickle(a, temp_file)
    tm.assert_frame_equal(a, b)
