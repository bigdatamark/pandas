# [MODULE DOC]
# Scalar Interval type and helpers. Defines IntervalMixin (closed-side and
# length/midpoint properties shared by Interval, IntervalArray and IntervalIndex),
# the immutable Interval scalar (left/right bounds plus a "closed" side string from
# VALID_CLOSED = left/right/both/neither), a converter from an object array of
# Intervals to separate left/right bound arrays, and, via the trailing include of
# the generated intervaltree.pxi, the IntervalTree family used by IntervalIndex for
# point and overlap queries.
import numbers
from operator import (
    le,
    lt,
)

from cpython.datetime cimport (
    PyDelta_Check,
    import_datetime,
)

import_datetime()

cimport cython
from cpython.object cimport PyObject_RichCompare
from cython cimport Py_ssize_t

import numpy as np

cimport numpy as cnp
from numpy cimport (
    NPY_QUICKSORT,
    PyArray_ArgSort,
    PyArray_Take,
    float64_t,
    int64_t,
    ndarray,
    uint64_t,
)

cnp.import_array()


from pandas._libs cimport util
from pandas._libs.hashtable cimport Int64Vector
from pandas._libs.tslibs.timedeltas cimport _Timedelta
from pandas._libs.tslibs.timestamps cimport _Timestamp
from pandas._libs.tslibs.timezones cimport tz_compare
from pandas._libs.tslibs.util cimport (
    is_float_object,
    is_integer_object,
)

VALID_CLOSED = frozenset(["left", "right", "both", "neither"])


cdef class IntervalMixin:
    # [CLASS DOC]
    # Mixin providing derived properties based on the attributes left, right and
    # closed, which the concrete class must supply (scalars for Interval, arrays for
    # IntervalArray/IntervalIndex, so most operations work elementwise on arrays too).

    @property
    def closed_left(self):
        # [FUNCTION DOC]
        # Property: whether the left endpoint is included.
        # Inputs: none (reads self.closed).
        # Outputs: True if closed is "left" or "both", else False.
        """
        Check if the interval is closed on the left side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is closed on the left-side.

        See Also
        --------
        Interval.closed_right : Check if the interval is closed on the right side.
        Interval.open_left : Boolean inverse of closed_left.

        Examples
        --------
        >>> iv = pd.Interval(0, 5, closed='left')
        >>> iv.closed_left
        True

        >>> iv = pd.Interval(0, 5, closed='right')
        >>> iv.closed_left
        False
        """
        return self.closed in ("left", "both")

    @property
    def closed_right(self):
        # [FUNCTION DOC]
        # Property: whether the right endpoint is included.
        # Inputs: none (reads self.closed).
        # Outputs: True if closed is "right" or "both", else False.
        # KNOWN ISSUE: the docstring's Returns section says "closed on the left-side";
        #   the
        # code (correctly) tests the right side.
        """
        Check if the interval is closed on the right side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is closed on the left-side.

        See Also
        --------
        Interval.closed_left : Check if the interval is closed on the left side.
        Interval.open_right : Boolean inverse of closed_right.

        Examples
        --------
        >>> iv = pd.Interval(0, 5, closed='both')
        >>> iv.closed_right
        True

        >>> iv = pd.Interval(0, 5, closed='left')
        >>> iv.closed_right
        False
        """
        return self.closed in ("right", "both")

    @property
    def open_left(self):
        # [FUNCTION DOC]
        # Property: whether the left endpoint is excluded.
        # Inputs: none.
        # Outputs: logical negation of closed_left.
        """
        Check if the interval is open on the left side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is not closed on the left-side.

        See Also
        --------
        Interval.open_right : Check if the interval is open on the right side.
        Interval.closed_left : Boolean inverse of open_left.

        Examples
        --------
        >>> iv = pd.Interval(0, 5, closed='neither')
        >>> iv.open_left
        True

        >>> iv = pd.Interval(0, 5, closed='both')
        >>> iv.open_left
        False
        """
        return not self.closed_left

    @property
    def open_right(self):
        # [FUNCTION DOC]
        # Property: whether the right endpoint is excluded.
        # Inputs: none.
        # Outputs: logical negation of closed_right.
        # KNOWN ISSUE: the docstring's Returns section says "not closed on the left-
        #   side";
        # the code tests the right side.
        """
        Check if the interval is open on the right side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is not closed on the left-side.

        See Also
        --------
        Interval.open_left : Check if the interval is open on the left side.
        Interval.closed_right : Boolean inverse of open_right.

        Examples
        --------
        >>> iv = pd.Interval(0, 5, closed='left')
        >>> iv.open_right
        True

        >>> iv = pd.Interval(0, 5)
        >>> iv.open_right
        False
        """
        return not self.closed_right

    @property
    def mid(self):
        # [FUNCTION DOC]
        # Property: midpoint of the interval.
        # Inputs: none.
        # Outputs: 0.5 times (left + right); if adding the bounds raises TypeError (e.g.
        # Timestamps cannot be added), returns left + 0.5 * length instead.
        # Exceptions: TypeError from the fallback if bounds still do not support it.
        """
        Return the midpoint of the Interval.

        See Also
        --------
        Interval.left : Return the left bound for the interval.
        Interval.right : Return the right bound for the interval.
        Interval.length : Return the length of the interval.

        Examples
        --------
        >>> iv = pd.Interval(0, 5)
        >>> iv.mid
        2.5
        """
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * self.length

    @property
    def length(self):
        # [FUNCTION DOC]
        # Property: length of the interval.
        # Inputs: none.
        # Outputs: right minus left (number, Timedelta, or array thereof).
        """
        Return the length of the Interval.

        See Also
        --------
        Interval.is_empty : Indicates if an interval contains no points.

        Examples
        --------
        >>> interval = pd.Interval(left=1, right=2, closed='left')
        >>> interval
        Interval(1, 2, closed='left')
        >>> interval.length
        1
        """
        return self.right - self.left

    @property
    def is_empty(self):
        # [FUNCTION DOC]
        # Property: whether the interval contains no points.
        # Inputs: none.
        # Outputs: (right equals left) AND (closed is not "both"), combined with bitwise
        # and so it works elementwise for arrays (NaN bounds compare unequal -> False).
        """
        Indicates if an interval is empty, meaning it contains no points.

        An interval is considered empty if its `left` and `right` endpoints
        are equal, and it is not closed on both sides. This means that the
        interval does not include any real points. In the case of an
        :class:`pandas.arrays.IntervalArray` or :class:`IntervalIndex`, the
        property returns a boolean array indicating the emptiness of each interval.

        Returns
        -------
        bool or ndarray
            A boolean indicating if a scalar :class:`Interval` is empty, or a
            boolean ``ndarray`` positionally indicating if an ``Interval`` in
            an :class:`~arrays.IntervalArray` or :class:`IntervalIndex` is
            empty.

        See Also
        --------
        Interval.length : Return the length of the Interval.

        Examples
        --------
        An :class:`Interval` that contains points is not empty:

        >>> pd.Interval(0, 1, closed='right').is_empty
        False

        An ``Interval`` that does not contain any points is empty:

        >>> pd.Interval(0, 0, closed='right').is_empty
        True
        >>> pd.Interval(0, 0, closed='left').is_empty
        True
        >>> pd.Interval(0, 0, closed='neither').is_empty
        True

        An ``Interval`` that contains a single point is not empty:

        >>> pd.Interval(0, 0, closed='both').is_empty
        False

        An :class:`~arrays.IntervalArray` or :class:`IntervalIndex` returns a
        boolean ``ndarray`` positionally indicating if an ``Interval`` is
        empty:

        >>> ivs = [pd.Interval(0, 0, closed='neither'),
        ...        pd.Interval(1, 2, closed='neither')]
        >>> pd.arrays.IntervalArray(ivs).is_empty
        array([ True, False])

        Missing values are not considered empty:

        >>> ivs = [pd.Interval(0, 0, closed='neither'), np.nan]
        >>> pd.IntervalIndex(ivs).is_empty
        array([ True, False])
        """
        return (self.right == self.left) & (self.closed != "both")

    def _check_closed_matches(self, other, name="other"):
        # [FUNCTION DOC]
        # HELPER: Validate that another interval-like object uses the same closed side.
        # Inputs: other - object with a closed attribute; name - label used in the
        #   message.
        # Outputs: none.
        # Exceptions: ValueError if self.closed differs from other.closed.
        """
        Check if the closed attribute of `other` matches.

        Note that 'left' and 'right' are considered different from 'both'.

        Parameters
        ----------
        other : Interval, IntervalIndex, IntervalArray
        name : str
            Name to use for 'other' in the error message.

        Raises
        ------
        ValueError
            When `other` is not closed exactly the same as self.
        """
        if self.closed != other.closed:
            raise ValueError(f"'{name}.closed' is {repr(other.closed)}, "
                             f"expected {repr(self.closed)}.")


cdef bint _interval_like(other):
    # [FUNCTION DOC]
    # HELPER: Duck-type check for interval-like objects.
    # Inputs: other - any object.
    # Outputs: true if other has attributes left, right and closed, else false.
    return (hasattr(other, "left")
            and hasattr(other, "right")
            and hasattr(other, "closed"))


cdef class Interval(IntervalMixin):
    # [CLASS DOC]
    # Immutable scalar interval. Fields (read-only): left, right - bounds (numeric,
    # Timestamp or Timedelta, left <= right); closed - one of VALID_CLOSED.
    # Supports membership tests for points and nested intervals, hashing, rich
    # comparison by the tuple (left, right, closed), pickling, and arithmetic with
    # scalars that shifts/scales both bounds.
    """
    Immutable object implementing an Interval, a bounded slice-like interval.

    Attributes
    ----------
    left : orderable scalar
        Left bound for the interval.
    right : orderable scalar
        Right bound for the interval.
    closed : {'right', 'left', 'both', 'neither'}, default 'right'
        Whether the interval is closed on the left-side, right-side, both or
        neither. See the Notes for more detailed explanation.

    See Also
    --------
    IntervalIndex : An Index of Interval objects that are all closed on the
        same side.
    cut : Convert continuous data into discrete bins (Categorical
        of Interval objects).
    qcut : Convert continuous data into bins (Categorical of Interval objects)
        based on quantiles.
    Period : Represents a period of time.

    Notes
    -----
    The parameters `left` and `right` must be from the same type, you must be
    able to compare them and they must satisfy ``left <= right``.

    A closed interval (in mathematics denoted by square brackets) contains
    its endpoints, i.e. the closed interval ``[0, 5]`` is characterized by the
    conditions ``0 <= x <= 5``. This is what ``closed='both'`` stands for.
    An open interval (in mathematics denoted by parentheses) does not contain
    its endpoints, i.e. the open interval ``(0, 5)`` is characterized by the
    conditions ``0 < x < 5``. This is what ``closed='neither'`` stands for.
    Intervals can also be half-open or half-closed, i.e. ``[0, 5)`` is
    described by ``0 <= x < 5`` (``closed='left'``) and ``(0, 5]`` is
    described by ``0 < x <= 5`` (``closed='right'``).

    Examples
    --------
    It is possible to build Intervals of different types, like numeric ones:

    >>> iv = pd.Interval(left=0, right=5)
    >>> iv
    Interval(0, 5, closed='right')

    You can check if an element belongs to it, or if it contains another interval:

    >>> 2.5 in iv
    True
    >>> pd.Interval(left=2, right=5, closed='both') in iv
    True

    You can test the bounds (``closed='right'``, so ``0 < x <= 5``):

    >>> 0 in iv
    False
    >>> 5 in iv
    True
    >>> 0.0001 in iv
    True

    Calculate its length

    >>> iv.length
    5

    You can operate with `+` and `*` over an Interval and the operation
    is applied to each of its bounds, so the result depends on the type
    of the bound elements

    >>> shifted_iv = iv + 3
    >>> shifted_iv
    Interval(3, 8, closed='right')
    >>> extended_iv = iv * 10.0
    >>> extended_iv
    Interval(0.0, 50.0, closed='right')

    To create a time interval you can use Timestamps as the bounds

    >>> year_2017 = pd.Interval(pd.Timestamp('2017-01-01 00:00:00'),
    ...                         pd.Timestamp('2018-01-01 00:00:00'),
    ...                         closed='left')
    >>> pd.Timestamp('2017-01-01 00:00') in year_2017
    True
    >>> year_2017.length
    Timedelta('365 days 00:00:00')
    """
    __module__ = "pandas"
    _typ = "interval"
    __array_priority__ = 1000

    cdef readonly object left
    """
    Left bound for the interval.

    See Also
    --------
    Interval.right : Return the right bound for the interval.
    numpy.ndarray.left : A similar method in numpy for obtaining
        the left endpoint(s) of intervals.

    Examples
    --------
    >>> interval = pd.Interval(left=1, right=2, closed='left')
    >>> interval
    Interval(1, 2, closed='left')
    >>> interval.left
    1
    """

    cdef readonly object right
    """
    Right bound for the interval.

    See Also
    --------
    Interval.left : Return the left bound for the interval.
    numpy.ndarray.right : A similar method in numpy for obtaining
        the right endpoint(s) of intervals.

    Examples
    --------
    >>> interval = pd.Interval(left=1, right=2, closed='left')
    >>> interval
    Interval(1, 2, closed='left')
    >>> interval.right
    2
    """

    cdef readonly str closed
    """
    String describing the inclusive side the intervals.

    Either ``left``, ``right``, ``both`` or ``neither``.

    See Also
    --------
    Interval.closed_left : Check if the interval is closed on the left side.
    Interval.closed_right : Check if the interval is closed on the right side.
    Interval.open_left : Check if the interval is open on the left side.
    Interval.open_right : Check if the interval is open on the right side.

    Examples
    --------
    >>> interval = pd.Interval(left=1, right=2, closed='left')
    >>> interval
    Interval(1, 2, closed='left')
    >>> interval.closed
    'left'
    """

    def __init__(self, left, right, str closed="right"):
        # [FUNCTION DOC]
        # Construct an Interval.
        # Inputs: left, right - endpoints; closed - string, default "right".
        # Outputs: none.
        # Exceptions: ValueError if an endpoint is not an integer, float, Timestamp or
        # Timedelta; if closed is invalid; if not left <= right; or if left is a
        #   Timestamp
        # whose time zone differs from right's.
        # Side effects: sets left, right, closed.
        # KNOWN ISSUE: the time-zone error message contains a stray single quote after
        #   the
        # left tzinfo repr (unbalanced quoting).
        # note: it is faster to just do these checks than to use a special
        # constructor (__cinit__/__new__) to avoid them

        self._validate_endpoint(left)
        self._validate_endpoint(right)

        if closed not in VALID_CLOSED:
            raise ValueError(f"invalid option for 'closed': {closed}")
        if not left <= right:
            raise ValueError("left side of interval must be <= right side")
        if (isinstance(left, _Timestamp) and
                not tz_compare(left.tzinfo, right.tzinfo)):
            # GH 18538
            raise ValueError("left and right must have the same time zone, got "
                             f"{repr(left.tzinfo)}' and {repr(right.tzinfo)}")
        self.left = left
        self.right = right
        self.closed = closed

    def _validate_endpoint(self, endpoint):
        # [FUNCTION DOC]
        # HELPER: Validate an endpoint type.
        # Inputs: endpoint - scalar.
        # Outputs: none.
        # Exceptions: ValueError unless endpoint is an integer, float (bool excluded by
        #   the
        # integer check), Timestamp or Timedelta.
        # GH 23013
        if not (is_integer_object(endpoint) or is_float_object(endpoint) or
                isinstance(endpoint, (_Timestamp, _Timedelta))):
            raise ValueError("Only numeric, Timestamp and Timedelta endpoints "
                             "are allowed when constructing an Interval.")

    def __hash__(self):
        # [FUNCTION DOC]
        # Hash of the tuple (left, right, closed).
        # Inputs: none. Outputs: integer hash.
        return hash((self.left, self.right, self.closed))

    def __contains__(self, key) -> bool:
        # [FUNCTION DOC]
        # Membership test.
        # Inputs: key - a point or an interval-like object (has left/right/closed).
        # Outputs: for interval-like keys, True if key lies within self: each side of
        #   key
        # must be strictly inside when self is open on that side and key is closed on
        #   it,
        # otherwise non-strict comparison is used. For points, compares left < key (or
        #   <=
        # when closed on the left) and key < right (or <= when closed on the right).
        # Exceptions: TypeError if the comparisons are unsupported for the key type.
        if _interval_like(key):
            key_closed_left = key.closed in ("left", "both")
            key_closed_right = key.closed in ("right", "both")
            if self.open_left and key_closed_left:
                left_contained = self.left < key.left
            else:
                left_contained = self.left <= key.left
            if self.open_right and key_closed_right:
                right_contained = key.right < self.right
            else:
                right_contained = key.right <= self.right
            return left_contained and right_contained
        return ((self.left < key if self.open_left else self.left <= key) and
                (key < self.right if self.open_right else key <= self.right))

    def __richcmp__(self, other, op: int):
        # [FUNCTION DOC]
        # Rich comparison.
        # Inputs: other - object; op - comparison opcode.
        # Outputs: for another Interval, compares the tuples (left, right, closed)
        # lexicographically with op; for a numpy array, returns a boolean array
        #   comparing
        # self against each element; otherwise NotImplemented.
        if isinstance(other, Interval):
            self_tuple = (self.left, self.right, self.closed)
            other_tuple = (other.left, other.right, other.closed)
            return PyObject_RichCompare(self_tuple, other_tuple, op)
        elif util.is_array(other):
            return np.array(
                [PyObject_RichCompare(self, x, op) for x in other],
                dtype=bool,
            )

        return NotImplemented

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support.
        # Outputs: tuple (Interval class, (left, right, closed)).
        args = (self.left, self.right, self.closed)
        return (type(self), args)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Representation "Interval(left, right, closed='...')". Bounds are rendered with
        # str when left is a numpy scalar or Timestamp, otherwise with repr. Uses the
        # actual subclass name.
        disp = str if isinstance(self.left, (np.generic, _Timestamp)) else repr
        name = type(self).__name__
        repr_str = f"{name}({disp(self.left)}, {disp(self.right)}, closed={repr(self.closed)})"  # noqa: E501
        return repr_str

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # String form in mathematical notation, using "[" / "(" for a closed / open left
        # side and "]" / ")" for the right side, e.g. "(0, 5]".
        start_symbol = "[" if self.closed_left else "("
        end_symbol = "]" if self.closed_right else ")"
        return f"{start_symbol}{self.left}, {self.right}{end_symbol}"

    def __add__(self, y):
        # [FUNCTION DOC]
        # Addition: shift both bounds.
        # Inputs: y - number, datetime.timedelta or numpy timedelta64.
        # Outputs: new Interval(left + y, right + y, same closed); NotImplemented
        #   otherwise.
        # Exceptions: ValueError from the constructor if results are invalid.
        if (
            isinstance(y, numbers.Number)
            or PyDelta_Check(y)
            or cnp.is_timedelta64_object(y)
        ):
            return Interval(self.left + y, self.right + y, closed=self.closed)
        return NotImplemented

    def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; same as __add__ with other on the left.
        # Inputs: other - number, timedelta or timedelta64.
        # Outputs: shifted Interval or NotImplemented.
        if (
                isinstance(other, numbers.Number)
                or PyDelta_Check(other)
                or cnp.is_timedelta64_object(other)
        ):
            return Interval(self.left + other, self.right + other, closed=self.closed)
        return NotImplemented

    def __sub__(self, y):
        # [FUNCTION DOC]
        # Subtraction: shift both bounds down by y.
        # Inputs: y - number, timedelta or timedelta64.
        # Outputs: new Interval(left - y, right - y, closed) or NotImplemented.
        if (
            isinstance(y, numbers.Number)
            or PyDelta_Check(y)
            or cnp.is_timedelta64_object(y)
        ):
            return Interval(self.left - y, self.right - y, closed=self.closed)
        return NotImplemented

    def __mul__(self, y):
        # [FUNCTION DOC]
        # Multiplication by a number: scales both bounds.
        # Inputs: y - numbers.Number. Outputs: new Interval or NotImplemented.
        # Exceptions: ValueError if y is negative (left would exceed right).
        if isinstance(y, numbers.Number):
            return Interval(self.left * y, self.right * y, closed=self.closed)
        return NotImplemented

    def __rmul__(self, other):
        # [FUNCTION DOC]
        # Reflected multiplication; same as __mul__.
        # Inputs: other - number. Outputs: scaled Interval or NotImplemented.
        if isinstance(other, numbers.Number):
            return Interval(self.left * other, self.right * other, closed=self.closed)
        return NotImplemented

    def __truediv__(self, y):
        # [FUNCTION DOC]
        # True division of both bounds by a number.
        # Inputs: y - number. Outputs: new Interval or NotImplemented.
        # Exceptions: ZeroDivisionError for 0; ValueError if y negative.
        if isinstance(y, numbers.Number):
            return Interval(self.left / y, self.right / y, closed=self.closed)
        return NotImplemented

    def __floordiv__(self, y):
        # [FUNCTION DOC]
        # Floor division of both bounds by a number.
        # Inputs: y - number. Outputs: new Interval or NotImplemented.
        # Exceptions: ZeroDivisionError for 0; ValueError if y negative.
        if isinstance(y, numbers.Number):
            return Interval(
                self.left // y, self.right // y, closed=self.closed)
        return NotImplemented

    def overlaps(self, other):
        # [FUNCTION DOC]
        # Test whether two Intervals share at least one point.
        # Inputs: other - Interval.
        # Outputs: True if self.left is before other.right and other.left is before
        # self.right, where "before" is <= when both touching endpoints are closed and <
        # otherwise.
        # Exceptions: TypeError if other is not an Interval.
        """
        Check whether two Interval objects overlap.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : Interval
            Interval to check against for an overlap.

        Returns
        -------
        bool
            True if the two intervals overlap.

        See Also
        --------
        IntervalArray.overlaps : The corresponding method for IntervalArray.
        IntervalIndex.overlaps : The corresponding method for IntervalIndex.

        Examples
        --------
        >>> i1 = pd.Interval(0, 2)
        >>> i2 = pd.Interval(1, 3)
        >>> i1.overlaps(i2)
        True
        >>> i3 = pd.Interval(4, 5)
        >>> i1.overlaps(i3)
        False

        Intervals that share closed endpoints overlap:

        >>> i4 = pd.Interval(0, 1, closed='both')
        >>> i5 = pd.Interval(1, 2, closed='both')
        >>> i4.overlaps(i5)
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> i6 = pd.Interval(1, 2, closed='neither')
        >>> i4.overlaps(i6)
        False
        """
        if not isinstance(other, Interval):
            raise TypeError("`other` must be an Interval, "
                            f"got {type(other).__name__}")

        # equality is okay if both endpoints are closed (overlap at a point)
        op1 = le if (self.closed_left and other.closed_right) else lt
        op2 = le if (other.closed_left and self.closed_right) else lt

        # overlaps is equivalent negation of two interval being disjoint:
        # disjoint = (A.left > B.right) or (B.left > A.right)
        # (simplifying the negation allows this to be done in less operations)
        return op1(self.left, other.right) and op2(other.left, self.right)


@cython.wraparound(False)
@cython.boundscheck(False)
def intervals_to_interval_bounds(ndarray intervals, bint validate_closed=True):
    # [FUNCTION DOC]
    # Split an object array of Intervals into left and right bound arrays.
    # Inputs: intervals - 1-D object ndarray of Interval / None / NaN; validate_closed -
    # whether mismatched closed sides are an error.
    # Outputs: tuple (left, right, closed): left/right are new arrays with the same
    # dtype as intervals (NaN where the element is null); closed is the shared closed
    # string, or None if all elements are null or (when not validating) sides differ.
    # Exceptions: TypeError if a non-null element is not an Interval; ValueError on
    # mismatched closed when validate_closed is true.
    # Side effects: allocates two arrays; bounds/wraparound checks disabled.
    """
    Parameters
    ----------
    intervals : ndarray
        Object array of Intervals / nulls.

    validate_closed: bool, default True
        Boolean indicating if all intervals must be closed on the same side.
        Mismatching closed will raise if True, else return None for closed.

    Returns
    -------
    tuple of
        left : ndarray
        right : ndarray
        closed: IntervalClosedType
    """
    cdef:
        object closed = None, interval
        Py_ssize_t i, n = len(intervals)
        ndarray left, right
        bint seen_closed = False

    left = np.empty(n, dtype=intervals.dtype)
    right = np.empty(n, dtype=intervals.dtype)

    for i in range(n):
        interval = intervals[i]
        if interval is None or util.is_nan(interval):
            left[i] = np.nan
            right[i] = np.nan
            continue

        if not isinstance(interval, Interval):
            raise TypeError(f"type {type(interval)} with value "
                            f"{interval} is not an interval")

        left[i] = interval.left
        right[i] = interval.right
        if not seen_closed:
            seen_closed = True
            closed = interval.closed
        elif closed != interval.closed:
            closed = None
            if validate_closed:
                raise ValueError("intervals must all be closed on the same side")

    return left, right, closed


include "intervaltree.pxi"
