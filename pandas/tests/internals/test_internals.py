# [MODULE DOC]
# Tests for the internal storage layer: blocks, the two-dimensional block manager, the single-block manager used by series, and block placement objects, plus the can-hold-element and should-store checks.
# Defines builder helpers at module level that create blocks and managers from compact text descriptions (a type code per block, comma-separated item names), which are shared by all tests here. Imports the internal block factories (the internal new block function and the pseudo-public make block function).
# The indexing test class builds a list of managers at class-definition (import) time.

from datetime import (
    date,
    datetime,
)
import itertools
import re

import numpy as np
import pytest

from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import is_scalar

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
    period_range,
)
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
    DatetimeArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.internals import (
    BlockManager,
    SingleBlockManager,
    make_block,
)
from pandas.core.internals.blocks import (
    ensure_block_shape,
    maybe_coerce_values,
    new_block,
)


@pytest.fixture(params=[new_block, make_block])
def block_maker(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the internal block factory and the pseudo-public block factory, returning whichever is selected so a test can run against both.
    """
    Fixture to test both the internal new_block and pseudo-public make_block.
    """
    return request.param


@pytest.fixture
def mgr():
    # [FUNCTION DOC]
    # Fixture returning a block manager built from a text description with float, object, bool, integer, complex, two naive datetime and two timezone-aware datetime blocks over items a through l (ten rows each).
    return create_mgr(
        "a: f8; b: object; c: f8; d: object; e: f8;"
        "f: bool; g: i8; h: complex; i: datetime-1; j: datetime-2;"
        "k: M8[ns, US/Eastern]; l: M8[ns, CET];"
    )


def assert_block_equal(left, right):
    # [FUNCTION DOC]
    # HELPER: asserts two blocks are equal.
    # Compares values as numpy arrays, the dtypes, requires that both placements are block placement objects and that their placement arrays are equal. Raises an assertion error on any difference.
    tm.assert_numpy_array_equal(left.values, right.values)
    assert left.dtype == right.dtype
    assert isinstance(left.mgr_locs, BlockPlacement)
    assert isinstance(right.mgr_locs, BlockPlacement)
    tm.assert_numpy_array_equal(left.mgr_locs.as_array, right.mgr_locs.as_array)


def get_numeric_mat(shape):
    # [FUNCTION DOC]
    # HELPER: builds a numeric matrix of the given shape in which every row along the first axis is constant and equal to the row number.
    # Input: shape, a sequence of dimension sizes. Returns an integer array (an independent copy) where element values depend only on the index along the first axis.
    arr = np.arange(shape[0])
    return np.lib.stride_tricks.as_strided(
        x=arr, shape=shape, strides=(arr.itemsize,) + (0,) * (len(shape) - 1)
    ).copy()


N = 10


def create_block(typestr, placement, item_shape=None, num_offset=0, maker=new_block):
    # [FUNCTION DOC]
    # HELPER: builds a single block of a given type code for use by the manager builders.
    # Inputs: typestr, a type code (floats, signed and unsigned integers by width, complex, bool, object or string, naive datetime, timezone-aware datetime given as a code with the zone, timedelta, sparse with fill zero, sparse with missing fill, two categorical variants); placement, anything convertible to a block placement; item_shape, the shape of each item, default a single axis of length ten; num_offset, an integer added to the generated values (default zero); maker, the block factory (default the internal one).
    # Behavior: the number of items is the placement length and the block has a leading item axis followed by item_shape. Numeric blocks are filled with the row number plus the offset; complex are imaginary; object blocks hold text labels derived from the numbers; bool is all true; datetime and timedelta come from scaled integers (the zone code is parsed with a regular expression and the timezone-aware variant must have exactly one item); categorical variants use fixed ten-element categoricals; sparse variants need a last dimension of ten and all other dimensions one, else a not-implemented error is raised.
    # Raises a value error for an unsupported type code; asserts when a zone code is malformed or more than one item is requested for a timezone-aware block. Values are passed through the coercion helper before the factory is called, and the number of dimensions is the length of the full shape.
    """
    Supported typestr:

        * float, f8, f4, f2
        * int, i8, i4, i2, i1
        * uint, u8, u4, u2, u1
        * complex, c16, c8
        * bool
        * object, string, O
        * datetime, dt, M8[ns], M8[ns, tz]
        * timedelta, td, m8[ns]
        * sparse (SparseArray with fill_value=0.0)
        * sparse_na (SparseArray with fill_value=np.nan)
        * category, category2

    """
    placement = BlockPlacement(placement)
    num_items = len(placement)

    if item_shape is None:
        item_shape = (N,)

    shape = (num_items, *item_shape)

    mat = get_numeric_mat(shape)

    if typestr in (
        "float",
        "f8",
        "f4",
        "f2",
        "int",
        "i8",
        "i4",
        "i2",
        "i1",
        "uint",
        "u8",
        "u4",
        "u2",
        "u1",
    ):
        values = mat.astype(typestr) + num_offset
    elif typestr in ("complex", "c16", "c8"):
        values = 1.0j * (mat.astype(typestr) + num_offset)
    elif typestr in ("object", "string", "O"):
        values = np.reshape([f"A{i:d}" for i in mat.ravel() + num_offset], shape)
    elif typestr in ("b", "bool"):
        values = np.ones(shape, dtype=np.bool_)
    elif typestr in ("datetime", "dt", "M8[ns]"):
        values = (mat * 1e9).astype("M8[ns]")
    elif typestr.startswith("M8[ns"):
        # datetime with tz
        m = re.search(r"M8\[ns,\s*(\w+\/?\w*)\]", typestr)
        assert m is not None, f"incompatible typestr -> {typestr}"
        tz = m.groups()[0]
        assert num_items == 1, "must have only 1 num items for a tz-aware"
        values = DatetimeIndex(np.arange(N) * 10**9, tz=tz)._data
        values = ensure_block_shape(values, ndim=len(shape))
    elif typestr in ("timedelta", "td", "m8[ns]"):
        values = (mat * 1).astype("m8[ns]")
    elif typestr in ("category",):
        values = Categorical([1, 1, 2, 2, 3, 3, 3, 3, 4, 4])
    elif typestr in ("category2",):
        values = Categorical(["a", "a", "a", "a", "b", "b", "c", "c", "c", "d"])
    elif typestr in ("sparse", "sparse_na"):
        if shape[-1] != 10:
            # We also are implicitly assuming this in the category cases above
            raise NotImplementedError

        assert all(s == 1 for s in shape[:-1])
        if typestr.endswith("_na"):
            fill_value = np.nan
        else:
            fill_value = 0.0
        values = SparseArray(
            [fill_value, fill_value, 1, 2, 3, fill_value, 4, 5, fill_value, 6],
            fill_value=fill_value,
        )
        arr = values.sp_values.view()
        arr += num_offset - 1
    else:
        raise ValueError(f'Unsupported typestr: "{typestr}"')

    values = maybe_coerce_values(values)
    return maker(values, placement=placement, ndim=len(shape))


def create_single_mgr(typestr, num_rows=None):
    # [FUNCTION DOC]
    # HELPER: builds a single-block manager (the one-dimensional container behind a series).
    # Inputs: typestr, a type code as for the block builder; num_rows, the length (default ten). The block has placement covering all rows with scalar items, and the index is a range of that length.
    if num_rows is None:
        num_rows = N

    return SingleBlockManager(
        create_block(typestr, placement=slice(0, num_rows), item_shape=()),
        Index(np.arange(num_rows)),
    )


def create_mgr(descr, item_shape=None):
    # [FUNCTION DOC]
    # HELPER: builds a two-dimensional block manager from a text description.
    # Input: descr, semicolon-separated components of the form comma-separated names, a colon and a type code (see the block builder); item_shape, the shape of each item (default one axis of length ten). Whitespace is trimmed and empty components are skipped. Components with the same type code are merged into one block; a dash followed by any suffix on the code forces separate blocks of the same dtype. Item names become the first axis, with range indexes for the remaining axes; each block receives an increasing numeric offset (the count of items placed before it) added to its values; blocks are sorted by their first placement and passed to the manager constructor.
    # Returns the new manager.
    """
    Construct BlockManager from string description.

    String description syntax looks similar to np.matrix initializer.  It looks
    like this::

        a,b,c: f8; d,e,f: i8

    Rules are rather simple:

    * see list of supported datatypes in `create_block` method
    * components are semicolon-separated
    * each component is `NAME,NAME,NAME: DTYPE_ID`
    * whitespace around colons & semicolons are removed
    * components with same DTYPE_ID are combined into single block
    * to force multiple blocks with same dtype, use '-SUFFIX'::

        "a:f8-1; b:f8-2; c:f8-foobar"

    """
    if item_shape is None:
        item_shape = (N,)

    offset = 0
    mgr_items = []
    block_placements = {}
    for d in descr.split(";"):
        d = d.strip()
        if not len(d):
            continue
        names, blockstr = d.partition(":")[::2]
        blockstr = blockstr.strip()
        names = names.strip().split(",")

        mgr_items.extend(names)
        placement = list(np.arange(len(names)) + offset)
        try:
            block_placements[blockstr].extend(placement)
        except KeyError:
            block_placements[blockstr] = placement
        offset += len(names)

    mgr_items = Index(mgr_items)

    blocks = []
    num_offset = 0
    for blockstr, placement in block_placements.items():
        typestr = blockstr.split("-")[0]
        blocks.append(
            create_block(
                typestr, placement, item_shape=item_shape, num_offset=num_offset
            )
        )
        num_offset += len(placement)

    sblocks = sorted(blocks, key=lambda b: b.mgr_locs[0])
    return BlockManager(
        tuple(sblocks),
        [mgr_items] + [Index(np.arange(n)) for n in item_shape],
    )


@pytest.fixture
def fblock():
    # [FUNCTION DOC]
    # Fixture returning a float block with three items at positions zero, two and four.
    return create_block("float", [0, 2, 4])


class TestBlock:
    # [CLASS DOC]
    # Tests for single block behavior: construction, pickling, placement, attributes, copying, deletion and splitting.
    def test_constructor(self):
        # [FUNCTION DOC]
        # Verifies a block created with a 32-bit integer type code has that dtype.
        int32block = create_block("i4", [0])
        assert int32block.dtype == np.int32

    @pytest.mark.parametrize(
        "typ, data",
        [
            ["float", [0, 2, 4]],
            ["complex", [7]],
            ["object", [1, 3]],
            ["bool", [5]],
        ],
    )
    def test_pickle(self, typ, data, temp_file):
        # [FUNCTION DOC]
        # Verifies blocks of float, complex, object and bool types survive a pickle round trip (via the temp file fixture), as judged by the block equality helper.
        blk = create_block(typ, data)
        assert_block_equal(tm.round_trip_pickle(blk, temp_file), blk)

    def test_mgr_locs(self, fblock):
        # [FUNCTION DOC]
        # Verifies the block placement is a block placement object holding the positions zero, two and four as a platform integer array.
        assert isinstance(fblock.mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(
            fblock.mgr_locs.as_array, np.array([0, 2, 4], dtype=np.intp)
        )

    def test_attrs(self, fblock):
        # [FUNCTION DOC]
        # Verifies the shape, dtype and length of a block equal those of its underlying values.
        assert fblock.shape == fblock.values.shape
        assert fblock.dtype == fblock.values.dtype
        assert len(fblock) == len(fblock.values)

    def test_copy(self, fblock):
        # [FUNCTION DOC]
        # Verifies a deep copy of a block is a different object but equal to the original.
        cop = fblock.copy(deep=True)
        assert cop is not fblock
        assert_block_equal(fblock, cop)

    def test_delete(self, fblock):
        # [FUNCTION DOC]
        # Verifies deleting items from a float block by position.
        # Deleting the first item returns a single new block with placement two and four, leaving the original placement unchanged and its values untouched (the new block's values reflect the second row). Deleting the middle item returns two blocks with placements zero and four. Deleting the last item returns one block with placements zero and two. Deleting an out-of-range position raises an index error.
        newb = fblock.copy(deep=True)
        locs = newb.mgr_locs
        nb = newb.delete(0)[0]
        assert newb.mgr_locs is locs

        assert nb is not newb

        tm.assert_numpy_array_equal(
            nb.mgr_locs.as_array, np.array([2, 4], dtype=np.intp)
        )
        assert not (newb.values[0] == 1).all()
        assert (nb.values[0] == 1).all()

        newb = fblock.copy(deep=True)
        locs = newb.mgr_locs
        nb = newb.delete(1)
        assert len(nb) == 2
        assert newb.mgr_locs is locs

        tm.assert_numpy_array_equal(
            nb[0].mgr_locs.as_array, np.array([0], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            nb[1].mgr_locs.as_array, np.array([4], dtype=np.intp)
        )
        assert not (newb.values[1] == 2).all()
        assert (nb[1].values[0] == 2).all()

        newb = fblock.copy(deep=True)
        nb = newb.delete(2)
        assert len(nb) == 1
        tm.assert_numpy_array_equal(
            nb[0].mgr_locs.as_array, np.array([0, 2], dtype=np.intp)
        )
        assert (nb[0].values[1] == 1).all()

        newb = fblock.copy(deep=True)

        with pytest.raises(IndexError, match=None):
            newb.delete(3)

    def test_delete_datetimelike(self):
        # [FUNCTION DOC]
        # Verifies that deleting from blocks holding datetime-like arrays keeps the extension array type (timedelta and datetime arrays) and returns two blocks when deleting one or two non-adjacent positions, rather than coercing to plain numpy arrays.
        # dont use np.delete on values, as that will coerce from DTA/TDA to ndarray
        arr = np.arange(20, dtype="i8").reshape(5, 4).view("m8[ns]")
        df = DataFrame(arr)
        blk = df._mgr.blocks[0]
        assert isinstance(blk.values, TimedeltaArray)

        nb = blk.delete(1)
        assert len(nb) == 2
        assert isinstance(nb[0].values, TimedeltaArray)
        assert isinstance(nb[1].values, TimedeltaArray)

        df = DataFrame(arr.view("M8[ns]"))
        blk = df._mgr.blocks[0]
        assert isinstance(blk.values, DatetimeArray)

        nb = blk.delete([1, 3])
        assert len(nb) == 2
        assert isinstance(nb[0].values, DatetimeArray)
        assert isinstance(nb[1].values, DatetimeArray)

    def test_split(self):
        # [FUNCTION DOC]
        # Verifies the block split helper (issue 37799): a three-item block with placements three, one and six splits into three one-item blocks with the matching placement and values; the pieces are views (writing into the original values changes the block values).
        # GH#37799
        values = np.random.default_rng(2).standard_normal((3, 4))
        blk = new_block(values, placement=BlockPlacement([3, 1, 6]), ndim=2)
        result = list(blk._split())

        # check that we get views, not copies
        values[:] = -9999
        assert (blk.values == -9999).all()

        assert len(result) == 3
        expected = [
            new_block(values[[0]], placement=BlockPlacement([3]), ndim=2),
            new_block(values[[1]], placement=BlockPlacement([1]), ndim=2),
            new_block(values[[2]], placement=BlockPlacement([6]), ndim=2),
        ]
        for res, exp in zip(result, expected):
            assert_block_equal(res, exp)


class TestBlockManager:
    # [CLASS DOC]
    # Tests for the block manager: attributes, pickling, item access, setting, copying, array conversion, astype, convert, interleave, consolidation, reindex, numeric and bool selection, equality and split-set helpers.
    def test_attrs(self):
        # [FUNCTION DOC]
        # Verifies a manager with two float blocks of three items has two blocks and length six.
        mgr = create_mgr("a,b,c: f8-1; d,e,f: f8-2")
        assert mgr.nblocks == 2
        assert len(mgr) == 6

    def test_duplicate_ref_loc_failure(self):
        # [FUNCTION DOC]
        # Verifies detection of overlapping placements.
        # Sets two blocks to the same placement; rebuilding the position maps raises an assertion error about gaps in block reference locations. After assigning non-overlapping placements again, a new manager can fetch the second item without error.
        tmp_mgr = create_mgr("a:bool; a: f8")

        axes, blocks = tmp_mgr.axes, tmp_mgr.blocks

        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([0]))

        # test trying to create block manager with overlapping ref locs

        msg = "Gaps in blk ref_locs"

        mgr = BlockManager(blocks, axes)
        with pytest.raises(AssertionError, match=msg):
            mgr._rebuild_blknos_and_blklocs()

        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([1]))
        mgr = BlockManager(blocks, axes)
        mgr.iget(1)

    def test_pickle(self, mgr, temp_file):
        # [FUNCTION DOC]
        # Verifies pickling of a mixed manager (issue 2431): the frame built from the unpickled manager equals the original, the unpickled manager has the consolidation flags and both are reset to false on load.
        mgr2 = tm.round_trip_pickle(mgr, temp_file)
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

        # GH2431
        assert hasattr(mgr2, "_is_consolidated")
        assert hasattr(mgr2, "_known_consolidated")

        # reset to False on load
        assert not mgr2._is_consolidated
        assert not mgr2._known_consolidated

    @pytest.mark.parametrize("mgr_string", ["a,a,a:f8", "a: f8; a: i8"])
    def test_non_unique_pickle(self, mgr_string, temp_file):
        # [FUNCTION DOC]
        # Verifies pickle round trip for managers with duplicate item names (three equal names in one block, or the same name in two blocks of different dtypes); frames built from the managers must be equal.
        mgr = create_mgr(mgr_string)
        mgr2 = tm.round_trip_pickle(mgr, temp_file)
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

    def test_categorical_block_pickle(self, temp_file):
        # [FUNCTION DOC]
        # Verifies pickle round trip of a manager with a categorical block and of a single-block manager of a categorical type; the resulting frame and series are equal.
        mgr = create_mgr("a: category")
        mgr2 = tm.round_trip_pickle(mgr, temp_file)
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

        smgr = create_single_mgr("category")
        smgr2 = tm.round_trip_pickle(smgr, temp_file)
        tm.assert_series_equal(
            Series()._constructor_from_mgr(smgr, axes=smgr.axes),
            Series()._constructor_from_mgr(smgr2, axes=smgr2.axes),
        )

    def test_iget(self):
        # [FUNCTION DOC]
        # Verifies positional item retrieval from a manager with one float block of three items: each item's internal values equal the matching row of the source matrix.
        cols = Index(list("abc"))
        values = np.random.default_rng(2).random((3, 3))
        block = new_block(
            values=values.copy(),
            placement=BlockPlacement(np.arange(3, dtype=np.intp)),
            ndim=values.ndim,
        )
        mgr = BlockManager(blocks=(block,), axes=[cols, Index(np.arange(3))])

        tm.assert_almost_equal(mgr.iget(0).internal_values(), values[0])
        tm.assert_almost_equal(mgr.iget(1).internal_values(), values[1])
        tm.assert_almost_equal(mgr.iget(2).internal_values(), values[2])

    def test_set(self):
        # [FUNCTION DOC]
        # Verifies insert and set by position on an integer manager with three items of three values: inserting a text item at the end and replacing the second item with text. The untouched integer items keep their values and the text items are object arrays.
        mgr = create_mgr("a,b,c: int", item_shape=(3,))

        mgr.insert(len(mgr.items), "d", np.array(["foo"] * 3))
        mgr.iset(1, np.array(["bar"] * 3))
        tm.assert_numpy_array_equal(mgr.iget(0).internal_values(), np.array([0] * 3))
        tm.assert_numpy_array_equal(
            mgr.iget(1).internal_values(), np.array(["bar"] * 3, dtype=np.object_)
        )
        tm.assert_numpy_array_equal(mgr.iget(2).internal_values(), np.array([2] * 3))
        tm.assert_numpy_array_equal(
            mgr.iget(3).internal_values(), np.array(["foo"] * 3, dtype=np.object_)
        )

    def test_set_change_dtype(self, mgr):
        # [FUNCTION DOC]
        # Verifies dtype changes when setting items on a manager.
        # Inserts a bool item, replaces it by text (object dtype expected); after consolidation, repeats; inserts an integer item (integer dtype) and replaces it with floats (float dtype expected).
        mgr.insert(len(mgr.items), "baz", np.zeros(N, dtype=bool))

        mgr.iset(mgr.items.get_loc("baz"), np.repeat("foo", N))
        idx = mgr.items.get_loc("baz")
        assert mgr.iget(idx).dtype == np.object_

        mgr2 = mgr.consolidate()
        mgr2.iset(mgr2.items.get_loc("baz"), np.repeat("foo", N))
        idx = mgr2.items.get_loc("baz")
        assert mgr2.iget(idx).dtype == np.object_

        mgr2.insert(
            len(mgr2.items),
            "quux",
            np.random.default_rng(2).standard_normal(N).astype(int),
        )
        idx = mgr2.items.get_loc("quux")
        assert mgr2.iget(idx).dtype == np.dtype(int)

        mgr2.iset(
            mgr2.items.get_loc("quux"), np.random.default_rng(2).standard_normal(N)
        )
        assert mgr2.iget(idx).dtype == np.float64

    def test_copy(self, mgr):
        # [FUNCTION DOC]
        # Verifies shallow and deep manager copies.
        # Shallow copy: block values are equal to the originals and share the same underlying memory base (for datetime-like arrays, the same numpy base). Then the manager is consolidated in place (deep copy consolidates, so block-wise comparison needs it) and a deep copy must hold equal values with independent memory (no base at all for plain arrays, a different base for extension arrays; for datetime arrays either no base or different bases).
        cp = mgr.copy(deep=False)
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            # view assertion
            tm.assert_equal(cp_blk.values, blk.values)
            if isinstance(blk.values, np.ndarray):
                assert cp_blk.values.base.base is blk.values.base
            else:
                # DatetimeTZBlock has DatetimeIndex values
                assert cp_blk.values._ndarray.base is blk.values._ndarray.base

        # copy(deep=True) consolidates, so the block-wise assertions will
        #  fail is mgr is not consolidated
        mgr._consolidate_inplace()
        cp = mgr.copy(deep=True)
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            bvals = blk.values
            cpvals = cp_blk.values

            tm.assert_equal(cpvals, bvals)

            if isinstance(cpvals, np.ndarray):
                lbase = cpvals.base
                rbase = bvals.base
            else:
                lbase = cpvals._ndarray.base
                rbase = bvals._ndarray.base

            # copy assertion we either have a None for a base or in case of
            # some blocks it is an array (e.g. datetimetz), but was copied
            if isinstance(cpvals, DatetimeArray):
                assert (lbase is None and rbase is None) or (lbase is not rbase)
            elif not isinstance(cpvals, np.ndarray):
                assert lbase is not rbase
            else:
                assert lbase is None and rbase is None

    def test_sparse(self):
        # [FUNCTION DOC]
        # Verifies that a manager of two sparse blocks converts to a float64 array.
        mgr = create_mgr("a: sparse-1; b: sparse-2")
        assert mgr.as_array().dtype == np.float64

    def test_sparse_mixed(self):
        # [FUNCTION DOC]
        # Verifies that a manager made of two sparse blocks and a float block keeps three separate blocks.
        mgr = create_mgr("a: sparse-1; b: sparse-2; c: f8")
        assert len(mgr.blocks) == 3
        assert isinstance(mgr, BlockManager)

    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [("c: f4; d: f2", np.float32), ("c: f4; d: f2; e: f8", np.float64)],
    )
    def test_as_array_float(self, mgr_string, dtype):
        # [FUNCTION DOC]
        # Verifies conversion to a single array yields the common float dtype.
        # Parametrized over (description, dtype): 32-bit and 16-bit floats give 32-bit; adding a 64-bit float gives 64-bit.
        mgr = create_mgr(mgr_string)
        assert mgr.as_array().dtype == dtype

    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [
            ("a: bool-1; b: bool-2", np.bool_),
            ("a: i8-1; b: i8-2; c: i4; d: i2; e: u1", np.int64),
            ("c: i4; d: i2; e: u1", np.int32),
        ],
    )
    def test_as_array_int_bool(self, mgr_string, dtype):
        # [FUNCTION DOC]
        # Verifies conversion to a single array for bool and integer blocks: two bool blocks give bool, mixed 64, 32, 16-bit and unsigned 8-bit integers give 64-bit, and 32-bit, 16-bit and unsigned 8-bit give 32-bit. Parametrized over (description, dtype).
        mgr = create_mgr(mgr_string)
        assert mgr.as_array().dtype == dtype

    def test_as_array_datetime(self):
        # [FUNCTION DOC]
        # Verifies conversion of two naive datetime blocks to an array gives nanosecond datetime dtype.
        mgr = create_mgr("h: datetime-1; g: datetime-2")
        assert mgr.as_array().dtype == "M8[ns]"

    def test_as_array_datetime_tz(self):
        # [FUNCTION DOC]
        # Verifies two timezone-aware blocks keep their own dtypes per item, while the combined array has object dtype.
        mgr = create_mgr("h: M8[ns, US/Eastern]; g: M8[ns, CET]")
        assert mgr.iget(0).dtype == "datetime64[ns, US/Eastern]"
        assert mgr.iget(1).dtype == "datetime64[ns, CET]"
        assert mgr.as_array().dtype == "object"

    @pytest.mark.parametrize("t", ["float16", "float32", "float64", "int32", "int64"])
    def test_astype(self, t):
        # [FUNCTION DOC]
        # Verifies casting the entire manager to a numeric type, parametrized over 16, 32, 64-bit floats and 32, 64-bit integers.
        # First on an all-float manager where every item takes the dtype. Then on a mixed manager with object, bool, datetime, 32-bit, 16-bit and 64-bit float blocks, using the mode that ignores errors: numeric items and bool take the target; object items stay object; the datetime stays datetime unless the target is the 64-bit integer, in which case it is converted.
        # coerce all
        mgr = create_mgr("c: f4; d: f2; e: f8")

        t = np.dtype(t)
        tmgr = mgr.astype(t)
        assert tmgr.iget(0).dtype.type == t
        assert tmgr.iget(1).dtype.type == t
        assert tmgr.iget(2).dtype.type == t

        # mixed
        mgr = create_mgr("a,b: object; c: bool; d: datetime; e: f4; f: f2; g: f8")

        t = np.dtype(t)
        tmgr = mgr.astype(t, errors="ignore")
        assert tmgr.iget(2).dtype.type == t
        assert tmgr.iget(4).dtype.type == t
        assert tmgr.iget(5).dtype.type == t
        assert tmgr.iget(6).dtype.type == t

        assert tmgr.iget(0).dtype.type == np.object_
        assert tmgr.iget(1).dtype.type == np.object_
        if t != np.int64:
            assert tmgr.iget(3).dtype.type == np.datetime64
        else:
            assert tmgr.iget(3).dtype.type == t

    def test_convert(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies the manager convert method (type inference for object blocks), parametrized by the infer-string fixture.
        # No-op case: numeric blocks stay as they are. Conversion case: object items holding the texts "1", "2." and "foo." remain object (or the string dtype when inference of strings is on) while integer and float items keep their dtypes. A second manager with more dtypes checks the preserved dtypes of 32-bit integer, bool, 64-bit integer, 64-bit float and 16-bit float items.
        # KNOWN ISSUE: the assertion for the datetime item passes the dtype type and the expected type as the assertion message rather than comparing them, so it always succeeds.
        def _compare(old_mgr, new_mgr):
            # [FUNCTION DOC]
            # HELPER: checks two managers hold the same set of blocks irrespective of order. Requires the same number of blocks and that every block of each manager has some block in the other with fully equal values.
            """compare the blocks, numeric compare ==, object don't"""
            old_blocks = set(old_mgr.blocks)
            new_blocks = set(new_mgr.blocks)
            assert len(old_blocks) == len(new_blocks)

            # compare non-numeric
            for b in old_blocks:
                found = False
                for nb in new_blocks:
                    if (b.values == nb.values).all():
                        found = True
                        break
                assert found

            for b in new_blocks:
                found = False
                for ob in old_blocks:
                    if (b.values == ob.values).all():
                        found = True
                        break
                assert found

        # noops
        mgr = create_mgr("f: i8; g: f8")
        new_mgr = mgr.convert()
        _compare(mgr, new_mgr)

        # convert
        mgr = create_mgr("a,b,foo: object; f: i8; g: f8")
        mgr.iset(0, np.array(["1"] * N, dtype=np.object_))
        mgr.iset(1, np.array(["2."] * N, dtype=np.object_))
        mgr.iset(2, np.array(["foo."] * N, dtype=np.object_))
        new_mgr = mgr.convert()
        dtype = "str" if using_infer_string else np.object_
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int64
        assert new_mgr.iget(4).dtype == np.float64

        mgr = create_mgr(
            "a,b,foo: object; f: i4; bool: bool; dt: datetime; i: i8; g: f8; h: f2"
        )
        mgr.iset(0, np.array(["1"] * N, dtype=np.object_))
        mgr.iset(1, np.array(["2."] * N, dtype=np.object_))
        mgr.iset(2, np.array(["foo."] * N, dtype=np.object_))
        new_mgr = mgr.convert()
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int32
        assert new_mgr.iget(4).dtype == np.bool_
        assert new_mgr.iget(5).dtype.type, np.datetime64
        assert new_mgr.iget(6).dtype == np.int64
        assert new_mgr.iget(7).dtype == np.float64
        assert new_mgr.iget(8).dtype == np.float16

    def test_interleave(self):
        # [FUNCTION DOC]
        # Verifies that for each of float, integer, object, bool, complex, datetime and timedelta types, a manager with one or two items of that type converts to an array of the same dtype.
        # self
        for dtype in ["f8", "i8", "object", "bool", "complex", "M8[ns]", "m8[ns]"]:
            mgr = create_mgr(f"a: {dtype}")
            assert mgr.as_array().dtype == dtype
            mgr = create_mgr(f"a: {dtype}; b: {dtype}")
            assert mgr.as_array().dtype == dtype

    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [
            ("a: category", "i8"),
            ("a: category; b: category", "i8"),
            ("a: category; b: category2", "object"),
            ("a: category2", "object"),
            ("a: category2; b: category2", "object"),
            ("a: f8", "f8"),
            ("a: f8; b: i8", "f8"),
            ("a: f4; b: i8", "f8"),
            ("a: f4; b: i8; d: object", "object"),
            ("a: bool; b: i8", "object"),
            ("a: complex", "complex"),
            ("a: f8; b: category", "object"),
            ("a: M8[ns]; b: category", "object"),
            ("a: M8[ns]; b: bool", "object"),
            ("a: M8[ns]; b: i8", "object"),
            ("a: m8[ns]; b: bool", "object"),
            ("a: m8[ns]; b: i8", "object"),
            ("a: M8[ns]; b: m8[ns]", "object"),
        ],
    )
    def test_interleave_dtype(self, mgr_string, dtype):
        # [FUNCTION DOC]
        # Verifies the common dtype produced when converting managers with mixed block types to a single array.
        # Parametrized over (description, dtype) pairs: categoricals of integers give 64-bit integer, string categoricals give object, floats with integers give 64-bit float, bool with integer gives object, datetimes or timedeltas mixed with other types give object, and so on.
        # KNOWN ISSUE: the parameters are never used. The body repeats the same combinations as hard-coded constructions, so every parametrized case runs the identical full body.
        # will be converted according the actual dtype of the underlying
        mgr = create_mgr("a: category")
        assert mgr.as_array().dtype == "i8"
        mgr = create_mgr("a: category; b: category2")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: category2")
        assert mgr.as_array().dtype == "object"

        # combinations
        mgr = create_mgr("a: f8")
        assert mgr.as_array().dtype == "f8"
        mgr = create_mgr("a: f8; b: i8")
        assert mgr.as_array().dtype == "f8"
        mgr = create_mgr("a: f4; b: i8")
        assert mgr.as_array().dtype == "f8"
        mgr = create_mgr("a: f4; b: i8; d: object")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: bool; b: i8")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: complex")
        assert mgr.as_array().dtype == "complex"
        mgr = create_mgr("a: f8; b: category")
        assert mgr.as_array().dtype == "f8"
        mgr = create_mgr("a: M8[ns]; b: category")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: M8[ns]; b: bool")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: M8[ns]; b: i8")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: m8[ns]; b: bool")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: m8[ns]; b: i8")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: M8[ns]; b: m8[ns]")
        assert mgr.as_array().dtype == "object"

    def test_consolidate_ordering_issues(self, mgr):
        # [FUNCTION DOC]
        # Verifies consolidation behavior on a manager with datetime and timezone-aware blocks after several items were replaced by random floats.
        # Consolidation gives four blocks; the numeric part after consolidation is a single block whose placement is the full range of positions.
        mgr.iset(mgr.items.get_loc("f"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("d"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("b"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("g"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("h"), np.random.default_rng(2).standard_normal(N))

        # we have datetime/tz blocks in mgr
        cons = mgr.consolidate()
        assert cons.nblocks == 4
        cons = mgr.consolidate().get_numeric_data()
        assert cons.nblocks == 1
        assert isinstance(cons.blocks[0].mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(
            cons.blocks[0].mgr_locs.as_array, np.arange(len(cons.items), dtype=np.intp)
        )

    def test_reindex_items(self):
        # [FUNCTION DOC]
        # Verifies reindexing a non-consolidated manager along the item axis with an item list that reorders and drops items: the result is not consolidated, has the new item index, and each result item equals the corresponding source item.
        # mgr is not consolidated, f8 & f8-2 blocks
        mgr = create_mgr("a: f8; b: i8; c: f8; d: i8; e: f8; f: bool; g: f8-2")

        reindexed = mgr.reindex_axis(["g", "c", "a", "d"], axis=0)
        assert not reindexed.is_consolidated()

        tm.assert_index_equal(reindexed.items, Index(["g", "c", "a", "d"]))
        tm.assert_almost_equal(
            mgr.iget(6).internal_values(), reindexed.iget(0).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(2).internal_values(), reindexed.iget(1).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(0).internal_values(), reindexed.iget(2).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(3).internal_values(), reindexed.iget(3).internal_values()
        )

    def test_get_numeric_data(self):
        # [FUNCTION DOC]
        # Verifies selecting numeric items from a manager (integer, float, complex, bool; text, object and datetime excluded after one object item is replaced by objects). The selected items equal the originals, and an in-place set on the selection does not change the original (the data was copied).
        mgr = create_mgr(
            "int: int; float: float; complex: complex;"
            "str: object; bool: bool; obj: object; dt: datetime",
            item_shape=(3,),
        )
        mgr.iset(5, np.array([1, 2, 3], dtype=np.object_))

        numeric = mgr.get_numeric_data()
        tm.assert_index_equal(numeric.items, Index(["int", "float", "complex", "bool"]))
        tm.assert_almost_equal(
            mgr.iget(mgr.items.get_loc("float")).internal_values(),
            numeric.iget(numeric.items.get_loc("float")).internal_values(),
        )

        # Check sharing
        numeric.iset(
            numeric.items.get_loc("float"),
            np.array([100.0, 200.0, 300.0]),
            inplace=True,
        )
        tm.assert_almost_equal(
            mgr.iget(mgr.items.get_loc("float")).internal_values(),
            np.array([1.0, 1.0, 1.0]),
        )

    def test_get_bool_data(self):
        # [FUNCTION DOC]
        # Verifies selecting only bool items; an in-place set on the selection writes through to the original manager (the selection shares memory), unlike the numeric selection case. Object-typed bool values placed in a non-bool block are not selected.
        mgr = create_mgr(
            "int: int; float: float; complex: complex;"
            "str: object; bool: bool; obj: object; dt: datetime",
            item_shape=(3,),
        )
        mgr.iset(6, np.array([True, False, True], dtype=np.object_))

        bools = mgr.get_bool_data()
        tm.assert_index_equal(bools.items, Index(["bool"]))
        tm.assert_almost_equal(
            mgr.iget(mgr.items.get_loc("bool")).internal_values(),
            bools.iget(bools.items.get_loc("bool")).internal_values(),
        )

        bools.iset(0, np.array([True, False, True]), inplace=True)
        tm.assert_numpy_array_equal(
            mgr.iget(mgr.items.get_loc("bool")).internal_values(),
            np.array([True, True, True]),
        )

    def test_unicode_repr_doesnt_raise(self):
        # [FUNCTION DOC]
        # Verifies that taking the text representation of a manager with a non-ASCII item name does not raise.
        repr(create_mgr("b,\u05d0: object"))

    @pytest.mark.parametrize(
        "mgr_string", ["a,b,c: i8-1; d,e,f: i8-2", "a,a,a: i8-1; b,b,b: i8-2"]
    )
    def test_equals(self, mgr_string):
        # [FUNCTION DOC]
        # Verifies that managers with the same blocks in reverse order compare equal, for unique names and for duplicate names.
        # unique items
        bm1 = create_mgr(mgr_string)
        bm2 = BlockManager(bm1.blocks[::-1], bm1.axes)
        assert bm1.equals(bm2)

    @pytest.mark.parametrize(
        "mgr_string",
        [
            "a:i8;b:f8",  # basic case
            "a:i8;b:f8;c:c8;d:b",  # many types
            "a:i8;e:dt;f:td;g:string",  # more types
            "a:i8;b:category;c:category2",  # categories
            "c:sparse;d:sparse_na;b:f8",  # sparse
        ],
    )
    def test_equals_block_order_different_dtypes(self, mgr_string):
        # [FUNCTION DOC]
        # Verifies manager equality ignores block order for many dtype mixes (issue 9330).
        # Parametrized over descriptions (basic, many types, more types, categoricals, sparse). Every permutation of the blocks gives a manager equal to the original in both directions.
        # GH 9330
        bm = create_mgr(mgr_string)
        block_perms = itertools.permutations(bm.blocks)
        for bm_perm in block_perms:
            bm_this = BlockManager(tuple(bm_perm), bm.axes)
            assert bm.equals(bm_this)
            assert bm_this.equals(bm)

    def test_single_mgr_ctor(self):
        # [FUNCTION DOC]
        # Verifies a single-block float manager of five rows has the external values zero through four.
        mgr = create_single_mgr("f8", num_rows=5)
        assert mgr.external_values().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        # [FUNCTION DOC]
        # Verifies that passing a non-bool value (an integer, text, list or float) as the in-place argument to the replace-list method raises a value error naming the argument, the expected bool type and the received type name.
        bm1 = create_mgr("a,b,c: i8-1; d,e,f: i8-2")

        msg = (
            'For argument "inplace" expected type bool, '
            f"received type {type(value).__name__}."
        )
        with pytest.raises(ValueError, match=msg):
            bm1.replace_list([1], [2], inplace=value)

    def test_iset_split_block(self):
        # [FUNCTION DOC]
        # Verifies the helper that splits a block when one item is set.
        # Starting with three integer items and one float item, splitting the first item keeps two blocks and the per-item positions in blocks (all zero except the second block) and block numbers as expected (platform integer width).
        bm = create_mgr("a,b,c: i8; d: f8")
        bm._iset_split_block(0, np.array([0]))
        tm.assert_numpy_array_equal(
            bm.blklocs, np.array([0, 0, 1, 0], dtype="int64" if IS64 else "int32")
        )
        # First indexer currently does not have a block associated with it in case
        tm.assert_numpy_array_equal(
            bm.blknos, np.array([0, 0, 0, 1], dtype="int64" if IS64 else "int32")
        )
        assert len(bm.blocks) == 2

    def test_iset_split_block_values(self):
        # [FUNCTION DOC]
        # Same as the previous test but with replacement values supplied: three blocks result, and the new item gets its own block number.
        bm = create_mgr("a,b,c: i8; d: f8")
        bm._iset_split_block(0, np.array([0]), np.array([list(range(10))]))
        tm.assert_numpy_array_equal(
            bm.blklocs, np.array([0, 0, 1, 0], dtype="int64" if IS64 else "int32")
        )
        # First indexer currently does not have a block associated with it in case
        tm.assert_numpy_array_equal(
            bm.blknos, np.array([0, 2, 2, 1], dtype="int64" if IS64 else "int32")
        )
        assert len(bm.blocks) == 3


def _as_array(mgr):
    # [FUNCTION DOC]
    # HELPER: converts a manager to a numpy array oriented like the equivalent frame or series. For a one-dimensional manager returns the external values; otherwise returns the transposed interleaved array. Used by the indexing tests.
    if mgr.ndim == 1:
        return mgr.external_values()
    return mgr.as_array().T


class TestIndexing:
    # [CLASS DOC]
    # Data-driven tests applying slicing, take and reindex operations to block managers and comparing with the same operations on numpy arrays. The managers cover single float and integer managers and several two-dimensional ones; sparse blocks are excluded because they fail many checks.
    # Nosetests-style data-driven tests.
    #
    # This test applies different indexing routines to block managers and
    # compares the outcome to the result of same operations on np.ndarray.
    #
    # NOTE: sparse (SparseBlock with fill_value != np.nan) fail a lot of tests
    #       and are disabled.

    MANAGERS = [
        create_single_mgr("f8", N),
        create_single_mgr("i8", N),
        # 2-dim
        create_mgr("a,b,c,d,e,f: f8", item_shape=(N,)),
        create_mgr("a,b,c,d,e,f: i8", item_shape=(N,)),
        create_mgr("a,b: f8; c,d: i8; e,f: string", item_shape=(N,)),
        create_mgr("a,b: f8; c,d: i8; e,f: f8", item_shape=(N,)),
    ]

    @pytest.mark.parametrize("mgr", MANAGERS)
    def test_get_slice(self, mgr):
        # [FUNCTION DOC]
        # Verifies slicing of managers (and boolean masks for one-dimensional ones) against numpy results; parametrized over the class-level managers.
        # For each axis tries full, head, over-long, middle and reversed strided slices; for one-dimensional managers also all-true, all-false and periodic boolean masks.
        def assert_slice_ok(mgr, axis, slobj):
            # [FUNCTION DOC]
            # HELPER: asserts one slice or mask on one axis matches numpy behavior. Pads a too-short mask with false values to the axis length; uses the manager slice method for slices and the mask-based row selection for one-dimensional bool arrays, otherwise raises a type error. Compares the resulting array (dtype not checked) and the sliced axis index.
            mat = _as_array(mgr)

            # we maybe using an ndarray to test slicing and
            # might not be the full length of the axis
            if isinstance(slobj, np.ndarray):
                ax = mgr.axes[axis]
                if len(ax) and len(slobj) and len(slobj) != len(ax):
                    slobj = np.concatenate(
                        [slobj, np.zeros(len(ax) - len(slobj), dtype=bool)]
                    )

            if isinstance(slobj, slice):
                sliced = mgr.get_slice(slobj, axis=axis)
            elif (
                mgr.ndim == 1
                and axis == 0
                and isinstance(slobj, np.ndarray)
                and slobj.dtype == bool
            ):
                sliced = mgr.get_rows_with_mask(slobj)
            else:
                # BlockManager doesn't support non-slice, SingleBlockManager
                #  doesn't support axis > 0
                raise TypeError(slobj)

            mat_slobj = (slice(None),) * axis + (slobj,)
            tm.assert_numpy_array_equal(
                mat[mat_slobj], _as_array(sliced), check_dtype=False
            )
            tm.assert_index_equal(mgr.axes[axis][slobj], sliced.axes[axis])

        assert mgr.ndim <= 2, mgr.ndim
        for ax in range(mgr.ndim):
            # slice
            assert_slice_ok(mgr, ax, slice(None))
            assert_slice_ok(mgr, ax, slice(3))
            assert_slice_ok(mgr, ax, slice(100))
            assert_slice_ok(mgr, ax, slice(1, 4))
            assert_slice_ok(mgr, ax, slice(3, 0, -2))

            if mgr.ndim < 2:
                # 2D only support slice objects

                # boolean mask
                assert_slice_ok(mgr, ax, np.ones(mgr.shape[ax], dtype=np.bool_))
                assert_slice_ok(mgr, ax, np.zeros(mgr.shape[ax], dtype=np.bool_))

                if mgr.shape[ax] >= 3:
                    assert_slice_ok(mgr, ax, np.arange(mgr.shape[ax]) % 3 == 0)
                    assert_slice_ok(
                        mgr, ax, np.array([True, True, False], dtype=np.bool_)
                    )

    @pytest.mark.parametrize("mgr", MANAGERS)
    def test_take(self, mgr):
        # [FUNCTION DOC]
        # Verifies the take operation on managers along each axis with an empty indexer, repeated positions, all positions, the first three positions and negative positions; parametrized over the class-level managers.
        def assert_take_ok(mgr, axis, indexer):
            # [FUNCTION DOC]
            # HELPER: asserts the take result matches numpy take on the array form and that the taken axis index matches the index take.
            mat = _as_array(mgr)
            taken = mgr.take(indexer, axis)
            tm.assert_numpy_array_equal(
                np.take(mat, indexer, axis), _as_array(taken), check_dtype=False
            )
            tm.assert_index_equal(mgr.axes[axis].take(indexer), taken.axes[axis])

        for ax in range(mgr.ndim):
            # take/fancy indexer
            assert_take_ok(mgr, ax, indexer=np.array([], dtype=np.intp))
            assert_take_ok(mgr, ax, indexer=np.array([0, 0, 0], dtype=np.intp))
            assert_take_ok(
                mgr, ax, indexer=np.array(list(range(mgr.shape[ax])), dtype=np.intp)
            )

            if mgr.shape[ax] >= 3:
                assert_take_ok(mgr, ax, indexer=np.array([0, 1, 2], dtype=np.intp))
                assert_take_ok(mgr, ax, indexer=np.array([-1, -2, -3], dtype=np.intp))

    @pytest.mark.parametrize("mgr", MANAGERS)
    @pytest.mark.parametrize("fill_value", [None, np.nan, 100.0])
    def test_reindex_axis(self, fill_value, mgr):
        # [FUNCTION DOC]
        # Verifies reindexing along an axis against the numpy-style take-with-fill helper, parametrized over the managers and fill values none, nan and 100.0.
        # Labels tried: empty, same, repeated, all-new, partially new, and for axes of length at least three: dropping the last three, reversed, and repeated.
        def assert_reindex_axis_is_ok(mgr, axis, new_labels, fill_value):
            # [FUNCTION DOC]
            # HELPER: asserts the reindexed manager equals the take-with-fill of the array form (using the index indexer for the new labels) and has the new labels as the axis.
            mat = _as_array(mgr)
            indexer = mgr.axes[axis].get_indexer_for(new_labels)

            reindexed = mgr.reindex_axis(new_labels, axis, fill_value=fill_value)
            tm.assert_numpy_array_equal(
                algos.take_nd(mat, indexer, axis, fill_value=fill_value),
                _as_array(reindexed),
                check_dtype=False,
            )
            tm.assert_index_equal(reindexed.axes[axis], new_labels)

        for ax in range(mgr.ndim):
            assert_reindex_axis_is_ok(mgr, ax, Index([]), fill_value)
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax], fill_value)
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][[0, 0, 0]], fill_value)
            assert_reindex_axis_is_ok(mgr, ax, Index(["foo", "bar", "baz"]), fill_value)
            assert_reindex_axis_is_ok(
                mgr, ax, Index(["foo", mgr.axes[ax][0], "baz"]), fill_value
            )

            if mgr.shape[ax] >= 3:
                assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][:-3], fill_value)
                assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][-3::-1], fill_value)
                assert_reindex_axis_is_ok(
                    mgr, ax, mgr.axes[ax][[0, 1, 2, 0, 1, 2]], fill_value
                )

    @pytest.mark.parametrize("mgr", MANAGERS)
    @pytest.mark.parametrize("fill_value", [None, np.nan, 100.0])
    def test_reindex_indexer(self, fill_value, mgr):
        # [FUNCTION DOC]
        # Verifies reindexing with explicit indexers against the take-with-fill helper, parametrized over the managers and fill values none, nan and 100.0.
        # Uses empty, identity, new labels with identity indexer, reversed labels, reversed indexer, repeated positions, positions with missing markers and, for long axes, three distinct positions.
        def assert_reindex_indexer_is_ok(mgr, axis, new_labels, indexer, fill_value):
            # [FUNCTION DOC]
            # HELPER: asserts the reindex-indexer result equals the take-with-fill of the array form and its axis equals the new labels.
            mat = _as_array(mgr)
            reindexed_mat = algos.take_nd(mat, indexer, axis, fill_value=fill_value)
            reindexed = mgr.reindex_indexer(
                new_labels, indexer, axis, fill_value=fill_value
            )
            tm.assert_numpy_array_equal(
                reindexed_mat, _as_array(reindexed), check_dtype=False
            )
            tm.assert_index_equal(reindexed.axes[axis], new_labels)

        for ax in range(mgr.ndim):
            assert_reindex_indexer_is_ok(
                mgr, ax, Index([]), np.array([], dtype=np.intp), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax], np.arange(mgr.shape[ax]), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr,
                ax,
                Index(["foo"] * mgr.shape[ax]),
                np.arange(mgr.shape[ax]),
                fill_value,
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax][::-1], np.arange(mgr.shape[ax]), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax], np.arange(mgr.shape[ax])[::-1], fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, Index(["foo", "bar", "baz"]), np.array([0, 0, 0]), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, Index(["foo", "bar", "baz"]), np.array([-1, 0, -1]), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr,
                ax,
                Index(["foo", mgr.axes[ax][0], "baz"]),
                np.array([-1, -1, -1]),
                fill_value,
            )

            if mgr.shape[ax] >= 3:
                assert_reindex_indexer_is_ok(
                    mgr,
                    ax,
                    Index(["foo", "bar", "baz"]),
                    np.array([0, 1, 2]),
                    fill_value,
                )


class TestBlockPlacement:
    # [CLASS DOC]
    # Tests for the block placement type (slice or array of positions), covering length, validation, slice-likeness, conversion and addition.
    @pytest.mark.parametrize(
        "slc, expected",
        [
            (slice(0, 4), 4),
            (slice(0, 4, 2), 2),
            (slice(0, 3, 2), 2),
            (slice(0, 1, 2), 1),
            (slice(1, 0, -1), 1),
        ],
    )
    def test_slice_len(self, slc, expected):
        # [FUNCTION DOC]
        # Verifies length of placements made from slices: parametrized over stepped and reversed slices with expected lengths four, two, two, one, one.
        assert len(BlockPlacement(slc)) == expected

    @pytest.mark.parametrize("slc", [slice(1, 1, 0), slice(1, 2, 0)])
    def test_zero_step_raises(self, slc):
        # [FUNCTION DOC]
        # Verifies that a placement built from a slice with step zero raises a value error saying the step cannot be zero.
        msg = "slice step cannot be zero"
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    def test_slice_canonize_negative_stop(self):
        # [FUNCTION DOC]
        # Verifies a slice with positive start, negative stop and negative step is canonicalized by replacing the stop with an open end (issue 37524).
        # GH#37524 negative stop is OK with negative step and positive start
        slc = slice(3, -1, -2)

        bp = BlockPlacement(slc)
        assert bp.indexer == slice(3, None, -2)

    @pytest.mark.parametrize(
        "slc",
        [
            slice(None, None),
            slice(10, None),
            slice(None, None, -1),
            slice(None, 10, -1),
            # These are "unbounded" because negative index will
            #  change depending on container shape.
            slice(-1, None),
            slice(None, -1),
            slice(-1, -1),
            slice(-1, None, -1),
            slice(None, -1, -1),
            slice(-1, -1, -1),
        ],
    )
    def test_unbounded_slice_raises(self, slc):
        # [FUNCTION DOC]
        # Verifies that unbounded slices (missing start or stop, or negative bounds whose meaning depends on the container) raise a value error mentioning an unbounded slice.
        msg = "unbounded slice"
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    @pytest.mark.parametrize(
        "slc",
        [
            slice(0, 0),
            slice(100, 0),
            slice(100, 100),
            slice(100, 100, -1),
            slice(0, 100, -1),
        ],
    )
    def test_not_slice_like_slices(self, slc):
        # [FUNCTION DOC]
        # Verifies that empty or degenerate slices make placements that are not slice-like.
        assert not BlockPlacement(slc).is_slice_like

    @pytest.mark.parametrize(
        "arr, slc",
        [
            ([0], slice(0, 1, 1)),
            ([100], slice(100, 101, 1)),
            ([0, 1, 2], slice(0, 3, 1)),
            ([0, 5, 10], slice(0, 15, 5)),
            ([0, 100], slice(0, 200, 100)),
            ([2, 1], slice(2, 0, -1)),
        ],
    )
    def test_array_to_slice_conversion(self, arr, slc):
        # [FUNCTION DOC]
        # Verifies that placements built from monotonic evenly spaced position lists are converted to the equal slice (including a descending pair).
        assert BlockPlacement(arr).as_slice == slc

    @pytest.mark.parametrize(
        "arr",
        [
            [],
            [-1],
            [-1, -2, -3],
            [-10],
            [-1, 0, 1, 2],
            [-2, 0, 2, 4],
            [1, 0, -1],
            [1, 1, 1],
        ],
    )
    def test_not_slice_like_arrays(self, arr):
        # [FUNCTION DOC]
        # Verifies that empty lists, negative positions, repeated positions and non-monotonic lists are not slice-like.
        assert not BlockPlacement(arr).is_slice_like

    @pytest.mark.parametrize(
        "slc, expected",
        [(slice(0, 3), [0, 1, 2]), (slice(0, 0), []), (slice(3, 0), [])],
    )
    def test_slice_iter(self, slc, expected):
        # [FUNCTION DOC]
        # Verifies iterating over a placement from a slice yields the covered positions (empty for empty or reversed-bound slices).
        assert list(BlockPlacement(slc)) == expected

    @pytest.mark.parametrize(
        "slc, arr",
        [
            (slice(0, 3), [0, 1, 2]),
            (slice(0, 0), []),
            (slice(3, 0), []),
            (slice(3, 0, -1), [3, 2, 1]),
        ],
    )
    def test_slice_to_array_conversion(self, slc, arr):
        # [FUNCTION DOC]
        # Verifies the integer array form of placements from slices, including empty and reversed ones, equals the expected array of platform integer dtype.
        tm.assert_numpy_array_equal(
            BlockPlacement(slc).as_array, np.asarray(arr, dtype=np.intp)
        )

    def test_blockplacement_add(self):
        # [FUNCTION DOC]
        # Verifies adding to a placement: adding one gives the shifted slice, adding an array gives the strided slice, and adding a descending array gives constant positions.
        bpl = BlockPlacement(slice(0, 5))
        assert bpl.add(1).as_slice == slice(1, 6, 1)
        assert bpl.add(np.arange(5)).as_slice == slice(0, 10, 2)
        assert list(bpl.add(np.arange(5, 0, -1))) == [5, 5, 5, 5, 5]

    @pytest.mark.parametrize(
        "val, inc, expected",
        [
            (slice(0, 0), 0, []),
            (slice(1, 4), 0, [1, 2, 3]),
            (slice(3, 0, -1), 0, [3, 2, 1]),
            ([1, 2, 4], 0, [1, 2, 4]),
            (slice(0, 0), 10, []),
            (slice(1, 4), 10, [11, 12, 13]),
            (slice(3, 0, -1), 10, [13, 12, 11]),
            ([1, 2, 4], 10, [11, 12, 14]),
            (slice(0, 0), -1, []),
            (slice(1, 4), -1, [0, 1, 2]),
            ([1, 2, 4], -1, [0, 1, 3]),
        ],
    )
    def test_blockplacement_add_int(self, val, inc, expected):
        # [FUNCTION DOC]
        # Verifies adding an integer increment to placements from slices and lists, including empty ones, zero, positive and negative increments.
        assert list(BlockPlacement(val).add(inc)) == expected

    @pytest.mark.parametrize("val", [slice(1, 4), [1, 2, 4]])
    def test_blockplacement_add_int_raises(self, val):
        # [FUNCTION DOC]
        # Verifies adding a negative number that would make positions negative raises a value error mentioning an in-place add that causes a length change.
        msg = "iadd causes length change"
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(val).add(-10)


class TestCanHoldElement:
    # [CLASS DOC]
    # Tests that blocks report correctly whether they can hold an element without casting, using datetime, interval and period blocks; includes helpers that compare with the series set-item behavior.
    @pytest.fixture(
        params=[
            lambda x: x,
            lambda x: x.to_series(),
            lambda x: x._data,
            lambda x: list(x),
            lambda x: x.astype(object),
            lambda x: np.asarray(x),
            lambda x: x[0],
            lambda x: x[:0],
        ]
    )
    def element(self, request):
        # [FUNCTION DOC]
        # Fixture parametrized over eight functions that convert an index into candidate elements: the index itself, a series, the underlying array, a list, an object-dtype copy, a numpy array, the first scalar and an empty slice.
        """
        Functions that take an Index and return an element that should have
        blk._can_hold_element(element) for a Block with this index's dtype.
        """
        return request.param

    def test_datetime_block_can_hold_element(self):
        # [FUNCTION DOC]
        # Verifies the datetime block can-hold-element rules and that they agree with array set-item.
        # An empty list and none are acceptable (none becomes not-a-time); numpy datetime64 and datetime objects are acceptable; a plain date is not acceptable and setting it raises a type error saying the value should be a timestamp, not-a-time or an array of those, got date.
        block = create_block("datetime", [0])

        assert block._can_hold_element([])

        # We will check that block._can_hold_element iff arr.__setitem__ works
        arr = pd.array(block.values.ravel())

        # coerce None
        assert block._can_hold_element(None)
        arr[0] = None
        assert arr[0] is pd.NaT

        # coerce different types of datetime objects
        vals = [np.datetime64("2010-10-10"), datetime(2010, 10, 10)]
        for val in vals:
            assert block._can_hold_element(val)
            arr[0] = val

        val = date(2010, 10, 10)
        assert not block._can_hold_element(val)

        msg = (
            "value should be a 'Timestamp', 'NaT', "
            "or array of those. Got 'date' instead."
        )
        with pytest.raises(TypeError, match=msg):
            arr[0] = val

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_interval_can_hold_element_emptylist(self, dtype, element):
        # [FUNCTION DOC]
        # Verifies an interval block can hold an empty list for several numeric subtypes.
        # KNOWN ISSUE: the test requests the element fixture but never uses it, so each case runs eight times identically; a trailing note says the same check should hold for all blocks.
        arr = np.array([1, 3, 4], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)

        assert blk._can_hold_element([])
        # TODO: check this holds for all blocks

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_interval_can_hold_element(self, dtype, element):
        # [FUNCTION DOC]
        # Verifies the can-hold-element results of an interval block agree with series set-item for several elements.
        # Parametrized over the dtype (signed, unsigned, float) and the element-making fixture. The same-interval elements can be held in place; elements with different closure, with timestamp bounds and with timedelta bounds cannot, and setting them raises a type error about an invalid value.
        arr = np.array([1, 3, 4, 9], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)

        elem = element(ii)
        self.check_series_setitem(elem, ii, True)
        assert blk._can_hold_element(elem)

        # Careful: to get the expected Series-inplace behavior we need
        # `elem` to not have the same length as `arr`
        ii2 = IntervalIndex.from_breaks(arr[:-1], closed="neither")
        elem = element(ii2)
        with pytest.raises(TypeError, match="Invalid value"):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)

        ii3 = IntervalIndex.from_breaks([Timestamp(1), Timestamp(3), Timestamp(4)])
        elem = element(ii3)
        with pytest.raises(TypeError, match="Invalid value"):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)

        ii4 = IntervalIndex.from_breaks([Timedelta(1), Timedelta(3), Timedelta(4)])
        elem = element(ii4)
        with pytest.raises(TypeError, match="Invalid value"):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)

    def test_period_can_hold_element_emptylist(self):
        # [FUNCTION DOC]
        # Verifies a period block can hold an empty list.
        pi = period_range("2016", periods=3, freq="Y")
        blk = new_block(pi._data.reshape(1, 3), BlockPlacement([1]), ndim=2)

        assert blk._can_hold_element([])

    def test_period_can_hold_element(self, element):
        # [FUNCTION DOC]
        # Verifies can-hold-element for period arrays: same-frequency elements are held in place; elements with a different frequency or timestamps cannot be held, and setting raises an invalid-value type error.
        pi = period_range("2016", periods=3, freq="Y")

        elem = element(pi)
        self.check_series_setitem(elem, pi, True)

        # Careful: to get the expected Series-inplace behavior we need
        # `elem` to not have the same length as `arr`
        pi2 = pi.asfreq("D")[:-1]
        elem = element(pi2)
        with pytest.raises(TypeError, match="Invalid value"):
            self.check_series_setitem(elem, pi, False)

        dti = pi.to_timestamp("s")[:-1]
        elem = element(dti)
        with pytest.raises(TypeError, match="Invalid value"):
            self.check_series_setitem(elem, pi, False)

    def test_period_reindex_axis(self):
        # [FUNCTION DOC]
        # Verifies reindexing a manager containing a period-dtype block along the item axis (issue 60273) and that the frame built from the result equals the expected frame.
        # GH#60273 Test reindexing of block with PeriodDtype
        pi = period_range("2020", periods=5, freq="Y")
        blk = new_block(pi._data.reshape(5, 1), BlockPlacement(slice(5)), ndim=2)
        mgr = BlockManager(blocks=(blk,), axes=[Index(np.arange(5)), Index(["a"])])
        reindexed = mgr.reindex_axis(Index([0, 2, 4]), axis=0)
        result = DataFrame._from_mgr(reindexed, axes=reindexed.axes)
        expected = DataFrame([[pi[0], pi[2], pi[4]]], columns=[0, 2, 4], index=["a"])
        tm.assert_frame_equal(result, expected)

    def check_can_hold_element(self, obj, elem, inplace: bool):
        # [FUNCTION DOC]
        # HELPER: asserts that the first block of a series' manager can or cannot hold an element, according to the in-place flag.
        blk = obj._mgr.blocks[0]
        if inplace:
            assert blk._can_hold_element(elem)
        else:
            assert not blk._can_hold_element(elem)

    def check_series_setitem(self, elem, index: Index, inplace: bool):
        # [FUNCTION DOC]
        # HELPER: for an index and candidate element, checks the block can-hold result and then performs the series set-item (scalar at position zero, otherwise a leading range). If in-place is true the series still uses the original array; otherwise the dtype becomes object.
        arr = index._data.copy()
        ser = Series(arr, copy=False)

        self.check_can_hold_element(ser, elem, inplace)

        if is_scalar(elem):
            ser[0] = elem
        else:
            ser[: len(elem)] = elem

        if inplace:
            assert ser._values is arr  # i.e. setting was done inplace
        else:
            assert ser.dtype == object


class TestShouldStore:
    # [CLASS DOC]
    # Tests for the block should-store check.
    def test_should_store_categorical(self):
        # [FUNCTION DOC]
        # Verifies a categorical block should store a categorical with the same dtype (and a slice of it) but not an ordered variant or a plain numpy array.
        cat = Categorical(["A", "B", "C"])
        df = DataFrame(cat)
        blk = df._mgr.blocks[0]

        # matching dtype
        assert blk.should_store(cat)
        assert blk.should_store(cat[:-1])

        # different dtype
        assert not blk.should_store(cat.as_ordered())

        # ndarray instead of Categorical
        assert not blk.should_store(np.asarray(cat))


def test_validate_ndim():
    # [FUNCTION DOC]
    # Verifies the pseudo-public block factory raises a value error when the values have fewer dimensions than requested (message about wrong number of dimensions), and also emits its deprecation warning.
    values = np.array([1.0, 2.0])
    placement = BlockPlacement(slice(2))
    msg = r"Wrong number of dimensions. values.ndim != ndim \[1 != 2\]"

    depr_msg = "make_block is deprecated"
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(Pandas4Warning, match=depr_msg):
            make_block(values, placement, ndim=2)


def test_block_shape():
    # [FUNCTION DOC]
    # Verifies that a reindexed integer series and a reindexed categorical series have identical block placement indexers.
    idx = Index([0, 1, 2, 3, 4])
    a = Series([1, 2, 3]).reindex(idx)
    b = Series(Categorical([1, 2, 3])).reindex(idx)

    assert a._mgr.blocks[0].mgr_locs.indexer == b._mgr.blocks[0].mgr_locs.indexer


def test_make_block_no_pandas_array(block_maker):
    # [FUNCTION DOC]
    # Verifies the block factories unwrap numpy-extension arrays: the result has an integer dtype kind. Parametrized by the block-maker fixture; the pseudo-public factory also emits a deprecation warning.
    # When the pseudo-public factory is used, the block is not an extension block, with or without an explicit dtype, and with a plain numpy array together with the numpy extension dtype (the internal factory needs callers to unwrap and no longer accepts a dtype).
    # https://github.com/pandas-dev/pandas/pull/24866
    arr = pd.arrays.NumpyExtensionArray(np.array([1, 2]))

    depr_msg = "make_block is deprecated"
    warn = DeprecationWarning if block_maker is make_block else None

    # NumpyExtensionArray, no dtype
    with tm.assert_produces_warning(warn, match=depr_msg):
        result = block_maker(arr, BlockPlacement(slice(len(arr))), ndim=arr.ndim)
    assert result.dtype.kind in ["i", "u"]

    if block_maker is make_block:
        # new_block requires caller to unwrap NumpyExtensionArray
        assert result.is_extension is False

        # NumpyExtensionArray, NumpyEADtype
        with tm.assert_produces_warning(warn, match=depr_msg):
            result = block_maker(arr, slice(len(arr)), dtype=arr.dtype, ndim=arr.ndim)
        assert result.dtype.kind in ["i", "u"]
        assert result.is_extension is False

        # new_block no longer accepts dtype keyword
        # ndarray, NumpyEADtype
        with tm.assert_produces_warning(warn, match=depr_msg):
            result = block_maker(
                arr.to_numpy(), slice(len(arr)), dtype=arr.dtype, ndim=arr.ndim
            )
        assert result.dtype.kind in ["i", "u"]
        assert result.is_extension is False
