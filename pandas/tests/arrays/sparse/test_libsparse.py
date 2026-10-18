# [MODULE DOC]
# Tests for the low-level sparse index classes (BlockIndex and IntIndex), the
# make_sparse_index factory, and the compiled sparse arithmetic kernels in the
# pandas._libs.sparse extension. Covers union, intersection, lookup, integrity
# validation, conversions between the two index representations, and agreement
# of the kernel results with ordinary Series arithmetic. Defines the shared
# fixtures test_length and cases. No import-time side effects.

import operator

import numpy as np
import pytest

import pandas._libs.sparse as splib
import pandas.util._test_decorators as td

from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
    BlockIndex,
    IntIndex,
    make_sparse_index,
)


@pytest.fixture
def test_length():
    # [FUNCTION DOC]
    # HELPER: fixture returning the logical length (20) used for every index built
    # in this module.
    return 20


@pytest.fixture(
    params=[
        [
            [0, 7, 15],
            [3, 5, 5],
            [2, 9, 14],
            [2, 3, 5],
            [2, 9, 15],
            [1, 3, 4],
        ],
        [
            [0, 5],
            [4, 4],
            [1],
            [4],
            [1],
            [3],
        ],
        [
            [0],
            [10],
            [0, 5],
            [3, 7],
            [0, 5],
            [3, 5],
        ],
        [
            [10],
            [5],
            [0, 12],
            [5, 3],
            [12],
            [3],
        ],
        [
            [0, 10],
            [4, 6],
            [5, 17],
            [4, 2],
            [],
            [],
        ],
        [
            [0],
            [5],
            [],
            [],
            [],
            [],
        ],
    ],
    ids=[
        "plain_case",
        "delete_blocks",
        "split_blocks",
        "skip_block",
        "no_intersect",
        "one_empty",
    ],
)
def cases(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture (ids: plain_case, delete_blocks, split_blocks,
    # skip_block, no_intersect, one_empty). Each parameter is a list of six lists:
    # block start locations and block lengths of index x, the same for index y, and
    # the expected block locations and lengths of the intersection of x and y.
    # Returns the current parameter.
    return request.param


class TestSparseIndexUnion:
    # [CLASS DOC]
    # Tests union of two sparse indexes.
    @pytest.mark.parametrize(
        "xloc, xlen, yloc, ylen, eloc, elen",
        [
            [[0], [5], [5], [4], [0], [9]],
            [[0, 10], [5, 5], [2, 17], [5, 2], [0, 10, 17], [7, 5, 2]],
            [[1], [5], [3], [5], [1], [7]],
            [[2, 10], [4, 4], [4], [8], [2], [12]],
            [[0, 5], [3, 5], [0], [7], [0], [10]],
            [[2, 10], [4, 4], [4, 13], [8, 4], [2], [15]],
            [[2], [15], [4, 9, 14], [3, 2, 2], [2], [15]],
            [[0, 10], [3, 3], [5, 15], [2, 2], [0, 5, 10, 15], [3, 2, 3, 2]],
        ],
    )
    def test_index_make_union(self, xloc, xlen, yloc, ylen, eloc, elen, test_length):
        # [FUNCTION DOC]
        # Parametrized over eight layouts (sketched in the comments) of two block
        # indexes x and y, given as locations and lengths, with the expected union
        # locations and lengths. Verifies the block-index union has the expected block
        # locations and lengths (32-bit integers) and that the integer-index union of
        # the converted indexes lists the same positions.
        # Case 1
        # x: ----
        # y:     ----
        # r: --------
        # Case 2
        # x: -----     -----
        # y:   -----          --
        # Case 3
        # x: ------
        # y:    -------
        # r: ----------
        # Case 4
        # x: ------  -----
        # y:    -------
        # r: -------------
        # Case 5
        # x: ---  -----
        # y: -------
        # r: -------------
        # Case 6
        # x: ------  -----
        # y:    -------  ---
        # r: -------------
        # Case 7
        # x: ----------------------
        # y:   ----  ----   ---
        # r: ----------------------
        # Case 8
        # x: ----       ---
        # y:       ---       ---
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)
        bresult = xindex.make_union(yindex)
        assert isinstance(bresult, BlockIndex)
        tm.assert_numpy_array_equal(bresult.blocs, np.array(eloc, dtype=np.int32))
        tm.assert_numpy_array_equal(bresult.blengths, np.array(elen, dtype=np.int32))

        ixindex = xindex.to_int_index()
        iyindex = yindex.to_int_index()
        iresult = ixindex.make_union(iyindex)
        assert isinstance(iresult, IntIndex)
        tm.assert_numpy_array_equal(iresult.indices, bresult.to_int_index().indices)

    def test_int_index_make_union(self):
        # [FUNCTION DOC]
        # Verifies integer-index union for overlapping, one-empty, both-empty, and
        # identical indexes (compared with the index equality method), and that unioning
        # indexes of different underlying lengths raises a value error ("Indices must
        # reference same underlying length").
        a = IntIndex(5, np.array([0, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2, 3, 4], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 1, 2, 3, 4], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([0, 1], dtype=np.int32))
        b = IntIndex(4, np.array([0, 1], dtype=np.int32))

        msg = "Indices must reference same underlying length"
        with pytest.raises(ValueError, match=msg):
            a.make_union(b)


class TestSparseIndexIntersect:
    # [CLASS DOC]
    # Tests intersection of two sparse indexes.
    @td.skip_if_windows
    def test_intersect(self, cases, test_length):
        # [FUNCTION DOC]
        # Uses the cases fixture. Verifies block-index intersection and integer-index
        # intersection both equal the expected block index (converted to integer index for
        # the latter), and that intersecting with an index of a different length raises
        # (message "Indices must reference same underlying length"). Skipped on Windows.
        xloc, xlen, yloc, ylen, eloc, elen = cases
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)
        expected = BlockIndex(test_length, eloc, elen)
        longer_index = BlockIndex(test_length + 1, yloc, ylen)

        result = xindex.intersect(yindex)
        assert result.equals(expected)
        result = xindex.to_int_index().intersect(yindex.to_int_index())
        assert result.equals(expected.to_int_index())

        msg = "Indices must reference same underlying length"
        with pytest.raises(Exception, match=msg):
            xindex.intersect(longer_index)
        with pytest.raises(Exception, match=msg):
            xindex.to_int_index().intersect(longer_index.to_int_index())

    def test_intersect_empty(self):
        # [FUNCTION DOC]
        # Verifies that intersecting an empty index with a non-empty one (either order)
        # gives the empty index, for both integer and block representations.
        xindex = IntIndex(4, np.array([], dtype=np.int32))
        yindex = IntIndex(4, np.array([2, 3], dtype=np.int32))
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

        xindex = xindex.to_block_index()
        yindex = yindex.to_block_index()
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

    @pytest.mark.parametrize(
        "case",
        [
            IntIndex(5, np.array([1, 2], dtype=np.int32)),
            IntIndex(5, np.array([0, 2, 4], dtype=np.int32)),
            IntIndex(0, np.array([], dtype=np.int32)),
            IntIndex(5, np.array([], dtype=np.int32)),
        ],
    )
    def test_intersect_identical(self, case):
        # [FUNCTION DOC]
        # Parametrized over four integer indexes (including empty and zero-length).
        # Verifies that intersecting an index with itself equals the index, in both
        # integer and block representation.
        assert case.intersect(case).equals(case)
        case = case.to_block_index()
        assert case.intersect(case).equals(case)


class TestSparseIndexCommon:
    # [CLASS DOC]
    # Tests behavior common to both index representations, built via the
    # make_sparse_index factory.
    def test_int_internal(self):
        # [FUNCTION DOC]
        # Verifies that the factory with kind integer returns an integer index with the
        # right number of points and index array, for two, zero, and four positions.
        # KNOWN ISSUE: this test is an exact duplicate of the integer-index class's
        # internal-construction test; one of them is redundant.
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))

        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind="integer"
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_block_internal(self):
        # [FUNCTION DOC]
        # Verifies that the factory with kind block collapses consecutive positions into
        # blocks: checks point count, block locations and block lengths for two
        # adjacent positions, none, all four, and a mix of a single and a run.
        # KNOWN ISSUE: this test is an exact duplicate of the block-index class's
        # internal-construction test; one of them is redundant.
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    @pytest.mark.parametrize("kind", ["integer", "block"])
    def test_lookup(self, kind):
        # [FUNCTION DOC]
        # Parametrized over kind (integer, block). Verifies the scalar lookup method
        # returns the position of a logical index within the stored values, or -1 when
        # the logical index is absent or out of range (including negative), for several
        # index layouts (partial, empty, full, mixed).
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == -1
        assert idx.lookup(1) == -1
        assert idx.lookup(2) == 0
        assert idx.lookup(3) == 1
        assert idx.lookup(4) == -1

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)

        for i in range(-1, 5):
            assert idx.lookup(i) == -1

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == 0
        assert idx.lookup(1) == 1
        assert idx.lookup(2) == 2
        assert idx.lookup(3) == 3
        assert idx.lookup(4) == -1

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == 0
        assert idx.lookup(1) == -1
        assert idx.lookup(2) == 1
        assert idx.lookup(3) == 2
        assert idx.lookup(4) == -1

    @pytest.mark.parametrize("kind", ["integer", "block"])
    def test_lookup_array(self, kind):
        # [FUNCTION DOC]
        # Parametrized over kind (integer, block). Verifies the vectorized lookup returns
        # an int32 array giving, for each requested logical index, its position in the
        # stored values or -1 when missing/out of range, for partial, empty, full and
        # mixed layouts and unsorted queries.
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)

        res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
        exp = np.array([-1, -1, 0], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
        exp = np.array([-1, 0, -1, 1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([-1, 0, 2, 4], dtype=np.int32))
        exp = np.array([-1, -1, -1, -1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
        exp = np.array([-1, 0, 2], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
        exp = np.array([-1, 2, 1, 3], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([2, 1, 3, 0], dtype=np.int32))
        exp = np.array([1, -1, 2, 0], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        res = idx.lookup_array(np.array([1, 4, 2, 5], dtype=np.int32))
        exp = np.array([-1, -1, 1, -1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

    @pytest.mark.parametrize(
        "idx, expected",
        [
            [0, -1],
            [5, 0],
            [7, 2],
            [8, -1],
            [9, -1],
            [10, -1],
            [11, -1],
            [12, 3],
            [17, 8],
            [18, -1],
        ],
    )
    def test_lookup_basics(self, idx, expected):
        # [FUNCTION DOC]
        # Parametrized over (logical index, expected position). For a block index of
        # length 20 with blocks starting at 5 (length 3) and 12 (length 6), verifies
        # lookup gives the expected position or -1, and that the converted integer
        # index agrees.
        bindex = BlockIndex(20, [5, 12], [3, 6])
        assert bindex.lookup(idx) == expected

        iindex = bindex.to_int_index()
        assert iindex.lookup(idx) == expected


class TestBlockIndex:
    # [CLASS DOC]
    # Tests specific to BlockIndex.
    def test_block_internal(self):
        # [FUNCTION DOC]
        # Same checks as the common-class block test: construction via the factory with
        # kind block yields the right point count, block locations and lengths.
        # KNOWN ISSUE: exact duplicate of the common-class block internal test.
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    @pytest.mark.parametrize("i", [5, 10, 100, 101])
    def test_make_block_boundary(self, i):
        # [FUNCTION DOC]
        # Parametrized over lengths (5, 10, 100, 101). Verifies that a block index built
        # from every second position has one block per position of length one, so block
        # locations equal the positions and lengths are all ones.
        idx = make_sparse_index(i, np.arange(0, i, 2, dtype=np.int32), kind="block")

        exp = np.arange(0, i, 2, dtype=np.int32)
        tm.assert_numpy_array_equal(idx.blocs, exp)
        tm.assert_numpy_array_equal(idx.blengths, np.ones(len(exp), dtype=np.int32))

    def test_equals(self):
        # [FUNCTION DOC]
        # Verifies the equality method: an index equals itself and differs from an index
        # with a different block length.
        index = BlockIndex(10, [0, 4], [2, 5])

        assert index.equals(index)
        assert not index.equals(BlockIndex(10, [0, 4], [2, 6]))

    def test_check_integrity(self):
        # [FUNCTION DOC]
        # Verifies construction validation: empty block lists are accepted for length
        # zero and one, a block that extends past the end raises a value error ("Block 0
        # extends beyond end"), and overlapping blocks raise a value error ("Block 0
        # overlaps").
        locs = []
        lengths = []

        # 0-length OK
        BlockIndex(0, locs, lengths)

        # also OK even though empty
        BlockIndex(1, locs, lengths)

        msg = "Block 0 extends beyond end"
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [5], [10])

        msg = "Block 0 overlaps"
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [2, 5], [5, 3])

    def test_to_int_index(self):
        # [FUNCTION DOC]
        # Verifies conversion of a block index (blocks at 0 length 4 and 10 length 6) to
        # an integer index lists all covered positions as 32-bit integers.
        locs = [0, 10]
        lengths = [4, 6]
        exp_inds = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]

        block = BlockIndex(20, locs, lengths)
        dense = block.to_int_index()

        tm.assert_numpy_array_equal(dense.indices, np.array(exp_inds, dtype=np.int32))

    def test_to_block_index(self):
        # [FUNCTION DOC]
        # Verifies that converting a block index to a block index returns the very same
        # object.
        index = BlockIndex(10, [0, 5], [4, 5])
        assert index.to_block_index() is index


class TestIntIndex:
    # [CLASS DOC]
    # Tests specific to IntIndex.
    def test_check_integrity(self):
        # [FUNCTION DOC]
        # Verifies construction validation raising value errors: more indices than the
        # length ("Too many indices"), negative index ("No index can be less than zero"),
        # index equal to or greater than the length ("All indices must be less than the
        # length"), and non-strictly-increasing indices including duplicates ("Indices
        # must be strictly increasing").
        # KNOWN ISSUE: the negative-index check appears twice with identical input and
        # message; the second copy is redundant.
        # Too many indices than specified in self.length
        msg = "Too many indices"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=1, indices=[1, 2, 3])

        # No index can be negative.
        msg = "No index can be less than zero"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])

        # No index can be negative.
        msg = "No index can be less than zero"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])

        # All indices must be less than the length.
        msg = "All indices must be less than the length"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 5])

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 6])

        # Indices must be strictly ascending.
        msg = "Indices must be strictly increasing"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 2])

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 3])

    def test_int_internal(self):
        # [FUNCTION DOC]
        # Same checks as the common-class integer test: construction via the factory with
        # kind integer yields the right point count and indices.
        # KNOWN ISSUE: exact duplicate of the common-class integer internal test.
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))

        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind="integer"
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_equals(self):
        # [FUNCTION DOC]
        # Verifies the equality method: equal to itself, not equal to an index with fewer
        # positions.
        index = IntIndex(10, [0, 1, 2, 3, 4])
        assert index.equals(index)
        assert not index.equals(IntIndex(10, [0, 1, 2, 3]))

    def test_to_block_index(self, cases, test_length):
        # [FUNCTION DOC]
        # Uses the cases fixture. Verifies that block indexes for x and y survive a round
        # trip through integer index and back to block index (equal to the originals).
        xloc, xlen, yloc, ylen, _, _ = cases
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)

        # see if survive the round trip
        xbindex = xindex.to_int_index().to_block_index()
        ybindex = yindex.to_int_index().to_block_index()
        assert isinstance(xbindex, BlockIndex)
        assert xbindex.equals(xindex)
        assert ybindex.equals(yindex)

    def test_to_int_index(self):
        # [FUNCTION DOC]
        # Verifies that converting an integer index to an integer index returns the very
        # same object.
        index = IntIndex(10, [2, 3, 4, 5, 6])
        assert index.to_int_index() is index


class TestSparseOperators:
    # [CLASS DOC]
    # Tests the compiled sparse arithmetic kernels.
    @pytest.mark.parametrize("opname", ["add", "sub", "mul", "truediv", "floordiv"])
    def test_op(self, opname, cases, test_length):
        # [FUNCTION DOC]
        # Parametrized over opname (add, sub, mul, truediv, floordiv) and the cases
        # fixture. Looks up the float64 kernel for the operation in the extension module
        # and the matching Python operator. Runs the kernel with block indexes and with
        # equivalent integer indexes (fill values 0 for x and 2 for y; values are
        # arithmetic ramps) and verifies the result indexes agree, the value arrays are
        # equal and fill results are equal. Then verifies the values equal the same
        # operation done with Series reindexed to the full length and filled with the
        # respective fill values, restricted to the result index positions.
        xloc, xlen, yloc, ylen, _, _ = cases
        sparse_op = getattr(splib, f"sparse_{opname}_float64")
        python_op = getattr(operator, opname)

        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)

        xdindex = xindex.to_int_index()
        ydindex = yindex.to_int_index()

        x = np.arange(xindex.npoints) * 10.0 + 1
        y = np.arange(yindex.npoints) * 100.0 + 1

        xfill = 0
        yfill = 2

        result_block_vals, rb_index, bfill = sparse_op(
            x, xindex, xfill, y, yindex, yfill
        )
        result_int_vals, ri_index, ifill = sparse_op(
            x, xdindex, xfill, y, ydindex, yfill
        )

        assert rb_index.to_int_index().equals(ri_index)
        tm.assert_numpy_array_equal(result_block_vals, result_int_vals)
        assert bfill == ifill

        # check versus Series...
        xseries = Series(x, xdindex.indices)
        xseries = xseries.reindex(np.arange(test_length)).fillna(xfill)

        yseries = Series(y, ydindex.indices)
        yseries = yseries.reindex(np.arange(test_length)).fillna(yfill)

        series_result = python_op(xseries, yseries)
        series_result = series_result.reindex(ri_index.indices)

        tm.assert_numpy_array_equal(result_block_vals, series_result.values)
        tm.assert_numpy_array_equal(result_int_vals, series_result.values)
