# Disable type checking for this module since numba's internals
# are not typed, and we use numba's internals via its extension API
# mypy: ignore-errors
"""
Utility classes/functions to let numba recognize
pandas Index/Series/DataFrame

Mostly vendored from https://github.com/numba/numba/blob/main/numba/tests/pdlike_usecase.py
"""

# [MODULE DOC]
# HELPER: internal numba extension module that registers pandas Index and Series
# (and the positional indexer of a Series) with numba's typing, data model,
# lowering, boxing and unboxing systems so jitted user functions can receive and
# return them.
# Public contents: the context manager set_numba_data, the numba type classes
# IndexType, SeriesType and IlocType, the data model classes IndexModel,
# SeriesModel and ILocModel, typing hooks (typeof and constructor typers),
# lowering functions for constructors, unbox and box conversions, helper overloads
# (string cast, reductions, binary operators, label lookup, item access) and the
# module level lists series_reductions and series_binops.
# Relationships: depends on numba and on the Index, Series, internals and indexing
# modules. Import-time side effects: importing requires numba; all the registration
# decorators run immediately and register global numba extensions, and the loops at
# the bottom of the reductions/binops lists register overloads for sum, mean, min,
# max and for add, subtract, multiply and true divide on Series.
from __future__ import annotations

from contextlib import contextmanager
import operator
from typing import Self

import numba
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.extending import (
    NativeValue,
    box,
    lower_builtin,
    make_attribute_wrapper,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)
from numba.core.imputils import impl_ret_borrowed
import numpy as np

from pandas._libs import lib

from pandas.core.indexes.base import Index
from pandas.core.indexing import _iLocIndexer
from pandas.core.internals import SingleBlockManager
from pandas.core.series import Series


# Helper function to hack around fact that Index casts numpy string dtype to object
#
# Idea is to set an attribute on a Index called _numba_data
# that is the original data, or the object data casted to numpy string dtype,
# with a context manager that is unset afterwards
@contextmanager
def set_numba_data(index: Index):
    # [FUNCTION DOC]
    # HELPER: context manager that temporarily attaches a private attribute holding
    # numba-friendly data to an Index so the typeof/unbox hooks can read it.
    # Input: index, an Index. If its underlying data has object or string dtype it
    # is converted to a numpy array and must contain only strings (otherwise a
    # value error saying the numba engine only supports string or numeric column
    # names is raised), then converted to a fixed width unicode dtype. Other dtypes
    # are used as is.
    # Yields the same index with the attribute set; the attribute is always removed
    # on exit, even when the body raises.
    numba_data = index._data
    if numba_data.dtype in (object, "string"):
        numba_data = np.asarray(numba_data)
        if not lib.is_string_array(numba_data):
            raise ValueError(
                "The numba engine only supports using string or numeric column names"
            )
        numba_data = numba_data.astype("U")
    try:
        index._numba_data = numba_data
        yield index
    finally:
        del index._numba_data


# TODO: Range index support
# (this currently lowers OK, but does not round-trip)
class IndexType(types.Type):
    # [CLASS DOC]
    # HELPER: numba type describing a one dimensional Index. Subclasses the numba
    # base type. State: dtype (element type), layout (array memory layout) and
    # pyclass (the Python Index class it came from). The numba type name is built
    # from the dtype and layout; the type identity key is the class, dtype and
    # layout. Methods: key and as_array properties and copy.
    """
    The type class for Index objects.
    """

    def __init__(self, dtype, layout, pyclass: any) -> None:
        # [FUNCTION DOC]
        # HELPER: stores dtype, layout and pyclass and registers the type name with
        # the base class. KNOWN ISSUE: the annotation for pyclass uses the builtin
        # any function rather than a typing Any; it is harmless at runtime because
        # annotations are deferred, but it is not a valid type.
        self.pyclass = pyclass
        name = f"index({dtype}, {layout})"
        self.dtype = dtype
        self.layout = layout
        super().__init__(name)

    @property
    def key(self):
        # [FUNCTION DOC]
        # Property returning the tuple (python class, dtype, layout) used by numba
        # to compare and hash this type.
        return self.pyclass, self.dtype, self.layout

    @property
    def as_array(self):
        # [FUNCTION DOC]
        # Property returning the equivalent one dimensional numba array type with
        # the same dtype and layout.
        return types.Array(self.dtype, 1, self.layout)

    def copy(self, dtype=None, ndim: int = 1, layout=None) -> Self:
        # [FUNCTION DOC]
        # Returns a new IndexType of the same python class, optionally with a
        # different dtype (default: keep) and layout (default: keep). ndim must be
        # one; an assertion error is raised otherwise.
        assert ndim == 1
        if dtype is None:
            dtype = self.dtype
        layout = layout or self.layout
        return type(self)(dtype, layout, self.pyclass)


class SeriesType(types.Type):
    # [CLASS DOC]
    # HELPER: numba type describing a Series. Subclasses the numba base type.
    # State: dtype (element type), index (an IndexType), values (a one dimensional
    # C-contiguous numba array of that dtype) and namety (numba type of the name).
    # Methods: key and as_array properties and copy.
    """
    The type class for Series objects.
    """

    def __init__(self, dtype, index, namety) -> None:
        # [FUNCTION DOC]
        # HELPER: stores dtype, index type, name type and builds the values array
        # type and the type name. The index must be an IndexType (assertion error
        # otherwise).
        assert isinstance(index, IndexType)
        self.dtype = dtype
        self.index = index
        self.values = types.Array(self.dtype, 1, "C")
        self.namety = namety
        name = f"series({dtype}, {index}, {namety})"
        super().__init__(name)

    @property
    def key(self):
        # [FUNCTION DOC]
        # Property returning the tuple (dtype, index type, name type) used by numba
        # to compare and hash this type.
        return self.dtype, self.index, self.namety

    @property
    def as_array(self):
        # [FUNCTION DOC]
        # Property returning the numba array type of the values.
        return self.values

    def copy(self, dtype=None, ndim: int = 1, layout: str = "C") -> Self:
        # [FUNCTION DOC]
        # Returns a new SeriesType with the same index and name types and optionally
        # a different dtype (default: keep). ndim must be one and layout must be
        # the C layout; assertion errors are raised otherwise.
        assert ndim == 1
        assert layout == "C"
        if dtype is None:
            dtype = self.dtype
        return type(self)(dtype, self.index, self.namety)


@typeof_impl.register(Index)
def typeof_index(val, c) -> IndexType:
    # [FUNCTION DOC]
    # Numba typeof hook for Index. Input: val, an Index that currently has the
    # temporary numba data attribute set (see set_numba_data), and c, the numba
    # typing context. Returns an IndexType built from the numba type of that data;
    # asserts that the data is one dimensional. If the attribute is missing an
    # attribute error is raised.
    """
    This will assume that only strings are in object dtype
    index.
    (you should check this before this gets lowered down to numba)
    """
    # arrty = typeof_impl(val._data, c)
    arrty = typeof_impl(val._numba_data, c)
    assert arrty.ndim == 1
    return IndexType(arrty.dtype, arrty.layout, type(val))


@typeof_impl.register(Series)
def typeof_series(val, c) -> SeriesType:
    # [FUNCTION DOC]
    # Numba typeof hook for Series. Computes the numba types of the index, the
    # values and the name, asserts the values are one dimensional and C-contiguous
    # and returns the matching SeriesType.
    index = typeof_impl(val.index, c)
    arrty = typeof_impl(val.values, c)
    namety = typeof_impl(val.name, c)
    assert arrty.ndim == 1
    assert arrty.layout == "C"
    return SeriesType(arrty.dtype, index, namety)


@type_callable(Series)
def type_series_constructor(context):
    # [FUNCTION DOC]
    # Numba type_callable hook for calling the Series constructor inside jitted
    # code. Returns a typer function (see below) for the given typing context.
    def typer(data, index, name=None):
        # [FUNCTION DOC]
        # HELPER: computes the return type of the Series constructor call. Inputs:
        # data (numba array type, one dimensional), index (IndexType) and optional
        # name type (default: platform integer type). Returns a SeriesType when
        # data is an array and index an IndexType; otherwise returns nothing, so
        # typing fails for other signatures.
        if isinstance(index, IndexType) and isinstance(data, types.Array):
            assert data.ndim == 1
            if name is None:
                name = types.intp
            return SeriesType(data.dtype, index, name)

    return typer


@type_callable(Index)
def type_index_constructor(context):
    # [FUNCTION DOC]
    # Numba type_callable hook for calling the Index constructor inside jitted
    # code. Returns a typer function for the given typing context.
    def typer(data, hashmap=None):
        # [FUNCTION DOC]
        # HELPER: computes the return type of the Index constructor call. Inputs:
        # data (one dimensional C-contiguous numba array type) and an optional
        # hashmap that must be absent or a numba typed dictionary type (assertion
        # errors otherwise). Returns an IndexType; returns nothing if data is not
        # an array.
        if isinstance(data, types.Array):
            assert data.layout == "C"
            assert data.ndim == 1
            assert hashmap is None or isinstance(hashmap, types.DictType)
            return IndexType(data.dtype, layout=data.layout, pyclass=Index)

    return typer


# Backend extensions for Index and Series and Frame
@register_model(IndexType)
class IndexModel(models.StructModel):
    # [CLASS DOC]
    # HELPER: numba data model for IndexType, a struct with three members: data
    # (the underlying array), hashmap (a typed dictionary from value to integer
    # position, filled lazily) and parent (a pointer to the originating Python
    # object, used to avoid rebuilding on boxing).
    def __init__(self, dmm, fe_type) -> None:
        # [FUNCTION DOC]
        # HELPER: declares the struct members for the given frontend type and
        # forwards to the base struct model constructor.
        # We don't want the numpy string scalar type in our hashmap
        members = [
            ("data", fe_type.as_array),
            # This is an attempt to emulate our hashtable code with a numba
            # typed dict
            # It maps from values in the index to their integer positions in the array
            ("hashmap", types.DictType(fe_type.dtype, types.intp)),
            # Pointer to the Index object this was created from, or that it
            # boxes to
            # https://numba.discourse.group/t/qst-how-to-cache-the-boxing-of-an-object/2128/2?u=lithomas1
            ("parent", types.pyobject),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(SeriesType)
class SeriesModel(models.StructModel):
    # [CLASS DOC]
    # HELPER: numba data model for SeriesType, a struct with members index, values
    # and name.
    def __init__(self, dmm, fe_type) -> None:
        # [FUNCTION DOC]
        # HELPER: declares the index, values and name struct members for the given
        # frontend type and forwards to the base struct model constructor.
        members = [
            ("index", fe_type.index),
            ("values", fe_type.as_array),
            ("name", fe_type.namety),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IndexType, "data", "_data")
make_attribute_wrapper(IndexType, "hashmap", "hashmap")

make_attribute_wrapper(SeriesType, "index", "index")
make_attribute_wrapper(SeriesType, "values", "values")
make_attribute_wrapper(SeriesType, "name", "name")


@lower_builtin(Series, types.Array, IndexType)
def pdseries_constructor(context, builder, sig, args):
    # [FUNCTION DOC]
    # Numba lowering for constructing a Series from an array and an index. Builds
    # the struct with the given values and index and a default name of the integer
    # zero; returns a borrowed reference to the new value.
    data, index = args
    series = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    series.index = index
    series.values = data
    series.name = context.get_constant(types.intp, 0)
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())


@lower_builtin(Series, types.Array, IndexType, types.intp)
@lower_builtin(Series, types.Array, IndexType, types.float64)
@lower_builtin(Series, types.Array, IndexType, types.unicode_type)
def pdseries_constructor_with_name(context, builder, sig, args):
    # [FUNCTION DOC]
    # Numba lowering for constructing a Series from an array, an index and a name
    # (integer, float or string). Builds the struct with those fields and returns
    # a borrowed reference.
    data, index, name = args
    series = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    series.index = index
    series.values = data
    series.name = name
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())


@lower_builtin(Index, types.Array, types.DictType, types.pyobject)
def index_constructor_2arg(context, builder, sig, args):
    # [FUNCTION DOC]
    # Numba lowering for constructing an Index from an array, a typed dictionary
    # hashmap and a parent Python object. Fills all three struct members and
    # returns a borrowed reference.
    # KNOWN ISSUE: the function name says two arguments, but it takes three; the
    # next function, named as the variant taking a parent, actually takes only the
    # array and the hashmap, so the two names are swapped relative to their
    # behavior.
    (data, hashmap, parent) = args
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)

    index.data = data
    index.hashmap = hashmap
    index.parent = parent
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())


@lower_builtin(Index, types.Array, types.DictType)
def index_constructor_2arg_parent(context, builder, sig, args):
    # [FUNCTION DOC]
    # Numba lowering for constructing an Index from an array and a typed dictionary
    # hashmap. Fills the data and hashmap members only (the parent pointer is left
    # unset, i.e. null) and returns a borrowed reference.
    # KNOWN ISSUE: the explanatory comment below says this variant lets you specify
    # the parent object, but the code does not set a parent at all.
    # Basically same as index_constructor_1arg, but also lets you specify the
    # parent object
    (data, hashmap) = args
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)

    index.data = data
    index.hashmap = hashmap
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())


@lower_builtin(Index, types.Array)
def index_constructor_1arg(context, builder, sig, args):
    # [FUNCTION DOC]
    # Numba lowering for constructing an Index from just an array. Compiles an
    # internal implementation that creates an empty typed dictionary keyed by the
    # index dtype with integer positions as values and calls the two-argument
    # constructor. Returns the compiled result.
    from numba.typed import Dict

    key_type = sig.return_type.dtype
    value_type = types.intp

    def index_impl(data):
        # [FUNCTION DOC]
        # HELPER: jitted body that builds an Index from data and an empty typed
        # dictionary; compiled by the lowering function above.
        return Index(data, Dict.empty(key_type, value_type))

    return context.compile_internal(builder, index_impl, sig, args)


# Helper to convert the unicodecharseq (numpy string scalar) into a unicode_type
# (regular string)
def maybe_cast_str(x):
    # [FUNCTION DOC]
    # HELPER: placeholder function that numba overloads (see the overload below);
    # in plain Python it does nothing and returns nothing. In jitted code it
    # converts a numpy fixed-width string scalar into a regular string and leaves
    # other types unchanged.
    # Dummy function that numba can overload
    pass


@overload(maybe_cast_str)
def maybe_cast_str_impl(x):
    # [FUNCTION DOC]
    # Numba overload implementation selector for the string cast placeholder.
    # Input: x, the numba type of the argument. Returns a jitted implementation
    # that converts to a string when x is a fixed-width unicode character sequence
    # type, otherwise one that returns the value unchanged.
    """Converts numba UnicodeCharSeq (numpy string scalar) -> unicode type (string).
    Is a no-op for other types."""
    if isinstance(x, types.UnicodeCharSeq):
        return lambda x: str(x)
    else:
        return lambda x: x


@unbox(IndexType)
def unbox_index(typ, obj, c):
    # [FUNCTION DOC]
    # Numba unbox hook: converts a Python Index into the native struct. Reads the
    # temporary numba data attribute, unboxes it as the data array, creates an
    # empty typed dictionary for the hashmap (filled lazily on first label lookup),
    # stores the original object as parent so boxing back is cheap, releases the
    # temporary references and returns a native value. Object dtype is only valid
    # if it was already validated and converted to strings.
    """
    Convert a Index object to a native structure.

    Note: Object dtype is not allowed here
    """
    data_obj = c.pyapi.object_getattr_string(obj, "_numba_data")
    index = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    # If we see an object array, assume its been validated as only containing strings
    # We still need to do the conversion though
    index.data = c.unbox(typ.as_array, data_obj).value
    typed_dict_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numba.typed.Dict))
    # Create an empty typed dict in numba for the hashmap for indexing
    # equiv of numba.typed.Dict.empty(typ.dtype, types.intp)
    arr_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.dtype))
    intp_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(types.intp))
    hashmap_obj = c.pyapi.call_method(
        typed_dict_obj, "empty", (arr_type_obj, intp_type_obj)
    )
    index.hashmap = c.unbox(types.DictType(typ.dtype, types.intp), hashmap_obj).value
    # Set the parent for speedy boxing.
    index.parent = obj

    # Decrefs
    c.pyapi.decref(data_obj)
    c.pyapi.decref(arr_type_obj)
    c.pyapi.decref(intp_type_obj)
    c.pyapi.decref(typed_dict_obj)

    return NativeValue(index._getvalue())


@unbox(SeriesType)
def unbox_series(typ, obj, c):
    # [FUNCTION DOC]
    # Numba unbox hook: converts a Python Series into the native struct by reading
    # its index, values and name attributes and unboxing each into the matching
    # member; the temporary references are released; returns a native value.
    """
    Convert a Series object to a native structure.
    """
    index_obj = c.pyapi.object_getattr_string(obj, "index")
    values_obj = c.pyapi.object_getattr_string(obj, "values")
    name_obj = c.pyapi.object_getattr_string(obj, "name")

    series = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    series.index = c.unbox(typ.index, index_obj).value
    series.values = c.unbox(typ.values, values_obj).value
    series.name = c.unbox(typ.namety, name_obj).value

    # Decrefs
    c.pyapi.decref(index_obj)
    c.pyapi.decref(values_obj)
    c.pyapi.decref(name_obj)

    return NativeValue(series._getvalue())


@box(IndexType)
def box_index(typ, val, c):
    # [FUNCTION DOC]
    # Numba box hook: converts the native Index struct back to a Python Index.
    # If a parent object is present it is reused (its reference count is
    # incremented). Otherwise the data array is boxed (converted to object dtype if
    # it was a fixed-width string array) and a plain Index is created through its
    # internal simple constructor; the new object is stored as the parent.
    # KNOWN ISSUE: a comment says the original Index subclass and name should be
    # preserved, but the code always creates the base Index class with no name (a
    # TODO that changes the result type).
    """
    Convert a native index structure to a Index object.

    If our native index is of a numpy string dtype, we'll cast it to
    object.
    """
    # First build a Numpy array object, then wrap it in a Index
    index = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    res = cgutils.alloca_once_value(c.builder, index.parent)

    # Does parent exist?
    # (it means already boxed once, or Index same as original df.index or df.columns)
    # xref https://github.com/numba/numba/blob/596e8a55334cc46854e3192766e643767bd7c934/numba/core/boxing.py#L593C17-L593C17
    with c.builder.if_else(cgutils.is_not_null(c.builder, index.parent)) as (
        has_parent,
        otherwise,
    ):
        with has_parent:
            c.pyapi.incref(index.parent)
        with otherwise:
            # TODO: preserve the original class for the index
            # Also need preserve the name of the Index
            # class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.pyclass))
            class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Index))
            array_obj = c.box(typ.as_array, index.data)
            if isinstance(typ.dtype, types.UnicodeCharSeq):
                # We converted to numpy string dtype, convert back
                # to object since _simple_new won't do that for uss
                object_str_obj = c.pyapi.unserialize(c.pyapi.serialize_object("object"))
                array_obj = c.pyapi.call_method(array_obj, "astype", (object_str_obj,))
                c.pyapi.decref(object_str_obj)
            # this is basically Index._simple_new(array_obj, name_obj) in python
            index_obj = c.pyapi.call_method(class_obj, "_simple_new", (array_obj,))
            index.parent = index_obj
            c.builder.store(index_obj, res)

            # Decrefs
            c.pyapi.decref(class_obj)
            c.pyapi.decref(array_obj)
    return c.builder.load(res)


@box(SeriesType)
def box_series(typ, val, c):
    # [FUNCTION DOC]
    # Numba box hook: converts the native Series struct back to a Python Series.
    # Boxes the index, values and name, builds a single block manager from the
    # array and index (skipping validation for speed), constructs the Series from
    # that manager, sets its name attribute directly, releases temporary
    # references and returns the new object.
    """
    Convert a native series structure to a Series object.
    """
    series = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    series_const_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Series._from_mgr))
    mgr_const_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(SingleBlockManager.from_array)
    )
    index_obj = c.box(typ.index, series.index)
    array_obj = c.box(typ.as_array, series.values)
    name_obj = c.box(typ.namety, series.name)
    # This is basically equivalent of
    # pd.Series(data=array_obj, index=index_obj)
    # To improve perf, we will construct the Series from a manager
    # object to avoid checks.
    # We'll also set the name attribute manually to avoid validation
    mgr_obj = c.pyapi.call_function_objargs(
        mgr_const_obj,
        (
            array_obj,
            index_obj,
        ),
    )
    mgr_axes_obj = c.pyapi.object_getattr_string(mgr_obj, "axes")
    # Series._constructor_from_mgr(mgr, axes)
    series_obj = c.pyapi.call_function_objargs(
        series_const_obj, (mgr_obj, mgr_axes_obj)
    )
    c.pyapi.object_setattr_string(series_obj, "_name", name_obj)

    # Decrefs
    c.pyapi.decref(series_const_obj)
    c.pyapi.decref(mgr_axes_obj)
    c.pyapi.decref(mgr_obj)
    c.pyapi.decref(mgr_const_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(array_obj)
    c.pyapi.decref(name_obj)

    return series_obj


# Add common series reductions (e.g. mean, sum),
# and also add common binops (e.g. add, sub, mul, div)
def generate_series_reduction(ser_reduction, ser_method):
    # [FUNCTION DOC]
    # HELPER: registers a reduction method on SeriesType. Inputs: ser_reduction,
    # the method name to expose, and ser_method, the numpy function applied to the
    # values array. Side effect: registers a numba method overload. Returns the
    # overload function.
    @overload_method(SeriesType, ser_reduction)
    def series_reduction(series):
        # [FUNCTION DOC]
        # HELPER: overload selector returning the jitted implementation.
        def series_reduction_impl(series):
            # [FUNCTION DOC]
            # HELPER: jitted body; applies the numpy reduction to the values.
            return ser_method(series.values)

        return series_reduction_impl

    return series_reduction


def generate_series_binop(binop):
    # [FUNCTION DOC]
    # HELPER: registers a binary operator overload for Series. Input: binop, an
    # operator function from the standard operator module. Side effect: registers a
    # numba overload. Returns the overload function.
    @overload(binop)
    def series_binop(series1, value):
        # [FUNCTION DOC]
        # HELPER: overload selector. When the left operand is a Series, returns a
        # jitted implementation applying the operator to the values: with another
        # Series on the right it operates on both values arrays, otherwise on the
        # values and the right operand; the result keeps the left index and name.
        # Returns nothing for other left operand types.
        if isinstance(series1, SeriesType):
            if isinstance(value, SeriesType):

                def series_binop_impl(series1, series2):
                    # [FUNCTION DOC]
                    # HELPER: jitted body for Series-with-Series operation.
                    # KNOWN ISSUE: the index of the right operand is not checked or
                    # aligned (a TODO), unlike regular pandas arithmetic.
                    # TODO: Check index matching?
                    return Series(
                        binop(series1.values, series2.values),
                        series1.index,
                        series1.name,
                    )

                return series_binop_impl
            else:

                def series_binop_impl(series1, value):
                    # [FUNCTION DOC]
                    # HELPER: jitted body for Series-with-scalar/array operation.
                    return Series(
                        binop(series1.values, value), series1.index, series1.name
                    )

                return series_binop_impl

    return series_binop


series_reductions = [
    ("sum", np.sum),
    ("mean", np.mean),
    # Disabled due to discrepancies between numba std. dev
    # and pandas std. dev (no way to specify dof)
    # ("std", np.std),
    # ("var", np.var),
    ("min", np.min),
    ("max", np.max),
]
for reduction, reduction_method in series_reductions:
    generate_series_reduction(reduction, reduction_method)

series_binops = [operator.add, operator.sub, operator.mul, operator.truediv]

for ser_binop in series_binops:
    generate_series_binop(ser_binop)


# get_loc on Index
@overload_method(IndexType, "get_loc")
def index_get_loc(index, item):
    # [FUNCTION DOC]
    # Numba method overload selector for label lookup on an Index; returns the
    # jitted implementation.
    def index_get_loc_impl(index, item):
        # [FUNCTION DOC]
        # HELPER: jitted body. Lazily fills the typed dictionary hashmap with
        # label to position pairs on first use (empty map), then returns the
        # position for the given item; a missing label raises a key error. For
        # duplicate labels the last position wins.
        # Initialize the hash table if not initialized
        if len(index.hashmap) == 0:
            for i, val in enumerate(index._data):
                index.hashmap[val] = i
        return index.hashmap[item]

    return index_get_loc_impl


# Indexing for Series/Index
@overload(operator.getitem)
def series_indexing(series, item):
    # [FUNCTION DOC]
    # Numba overload selector for item access on a Series by label. Returns a
    # jitted implementation when the first operand is a Series, else nothing.
    if isinstance(series, SeriesType):

        def series_getitem(series, item):
            # [FUNCTION DOC]
            # HELPER: jitted body; looks up the label position in the index and
            # returns the value at that position.
            loc = series.index.get_loc(item)
            return series.iloc[loc]

        return series_getitem


@overload(operator.getitem)
def index_indexing(index, idx):
    # [FUNCTION DOC]
    # Numba overload selector for item access on an Index by position. Returns a
    # jitted implementation when the first operand is an Index, else nothing.
    if isinstance(index, IndexType):

        def index_getitem(index, idx):
            # [FUNCTION DOC]
            # HELPER: jitted body; returns the element of the underlying array
            # at the given position.
            return index._data[idx]

        return index_getitem


class IlocType(types.Type):
    # [CLASS DOC]
    # HELPER: numba type for the positional indexer of a Series. Subclasses the
    # numba base type. State: obj_type, the numba type of the indexed object; the
    # identity key is that type.
    def __init__(self, obj_type) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the indexed object's type and sets the type name.
        self.obj_type = obj_type
        name = f"iLocIndexer({obj_type})"
        super().__init__(name=name)

    @property
    def key(self):
        # [FUNCTION DOC]
        # Property returning the wrapped object type used for type identity.
        return self.obj_type


@typeof_impl.register(_iLocIndexer)
def typeof_iloc(val, c) -> IlocType:
    # [FUNCTION DOC]
    # Numba typeof hook for the positional indexer: returns an IlocType wrapping
    # the numba type of the indexed object.
    objtype = typeof_impl(val.obj, c)
    return IlocType(objtype)


@type_callable(_iLocIndexer)
def type_iloc_constructor(context):
    # [FUNCTION DOC]
    # Numba type_callable hook for constructing the positional indexer in jitted
    # code; returns a typer function.
    def typer(obj):
        # [FUNCTION DOC]
        # HELPER: returns an IlocType when the argument is a SeriesType, otherwise
        # nothing (typing fails).
        if isinstance(obj, SeriesType):
            return IlocType(obj)

    return typer


@lower_builtin(_iLocIndexer, SeriesType)
def iloc_constructor(context, builder, sig, args):
    # [FUNCTION DOC]
    # Numba lowering for constructing the positional indexer from a Series: builds
    # the struct holding the Series and returns a borrowed reference.
    (obj,) = args
    iloc_indexer = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    iloc_indexer.obj = obj
    return impl_ret_borrowed(
        context, builder, sig.return_type, iloc_indexer._getvalue()
    )


@register_model(IlocType)
class ILocModel(models.StructModel):
    # [CLASS DOC]
    # HELPER: numba data model for IlocType, a struct with one member, obj (the
    # indexed Series).
    def __init__(self, dmm, fe_type) -> None:
        # [FUNCTION DOC]
        # HELPER: declares the single obj member and forwards to the base class.
        members = [("obj", fe_type.obj_type)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IlocType, "obj", "obj")


@overload_attribute(SeriesType, "iloc")
def series_iloc(series):
    # [FUNCTION DOC]
    # Numba attribute overload exposing the positional indexer on a Series;
    # returns the jitted getter.
    def get(series):
        # [FUNCTION DOC]
        # HELPER: jitted getter; creates the positional indexer for the Series.
        return _iLocIndexer(series)

    return get


@overload(operator.getitem)
def iloc_getitem(iloc_indexer, i):
    # [FUNCTION DOC]
    # Numba overload selector for item access on the positional indexer; returns a
    # jitted implementation when the first operand is an IlocType, else nothing.
    if isinstance(iloc_indexer, IlocType):

        def getitem_impl(iloc_indexer, i):
            # [FUNCTION DOC]
            # HELPER: jitted body; returns the element of the Series values at
            # position i.
            return iloc_indexer.obj.values[i]

        return getitem_impl
