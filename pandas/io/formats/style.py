"""
Module for applying conditional formatting to DataFrames and Series.
"""
# [MODULE DOC]
# Conditional formatting of frames and series: defines the public Styler class,
# the object returned by the frame and series style accessors. Styler extends the
# renderer base class from the style_render module with the user-facing API:
# output methods (HTML, LaTeX, Typst, string, Excel), style application (apply,
# map, and their index/header variants), table-level settings (attributes,
# captions, uuid, sticky headers, table styles, hiding), export/use of settings,
# built-in styles (gradients, bars, highlighting), subclass templating and a pipe
# helper. Private module functions implement the built-in styles (gradient
# coloring, between/quantile/extreme highlighting, in-cell bars) and argument
# validation for them.
# Dependencies: requires the optional package jinja2 (an import error with an
# explanatory message is raised at import if it is missing); matplotlib is imported
# lazily only when gradients or colormapped bars are used.

from __future__ import annotations

import copy
from functools import partial
import operator
from typing import (
    TYPE_CHECKING,
    Concatenate,
    Self,
    overload,
)

import numpy as np

from pandas._config import get_option

from pandas.compat._optional import import_optional_dependency

import pandas as pd
from pandas import (
    IndexSlice,
    RangeIndex,
)
import pandas.core.common as com
from pandas.core.frame import (
    DataFrame,
    Series,
)

from pandas.io.formats.format import save_to_buffer

jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")

from pandas.io.formats.style_render import (
    CSSProperties,
    CSSStyles,
    ExtFormatter,
    StylerRenderer,
    Subset,
    Tooltips,
    format_table_styles,
    maybe_convert_css_to_tuples,
    non_reducing_slice,
    refactor_levels,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )

    from matplotlib.colors import Colormap

    from pandas._typing import (
        Any,
        Axis,
        AxisInt,
        ExcelWriterMergeCells,
        FilePath,
        IndexLabel,
        IntervalClosedType,
        Level,
        P,
        QuantileInterpolation,
        Scalar,
        StorageOptions,
        T,
        WriteBuffer,
        WriteExcelBuffer,
    )

    from pandas import ExcelWriter
    from pandas.core.generic import NDFrame


class Styler(StylerRenderer):
    # [CLASS DOC]
    # User-facing styling object for a frame or series. Inherits all rendering state
    # and machinery from the renderer base class (data, uuid, table styles, caption,
    # cell style contexts, display functions, concatenated stylers, tooltips, hidden
    # rows and columns) and adds the public API.
    # Style functions are not run immediately: the apply and map families append a
    # deferred call description to an internal list, and the styles are computed only
    # when rendering (or exporting to Excel/LaTeX/HTML). Methods that configure the
    # object return the object itself for chaining. Output methods work on a deep copy so
    # the original is not modified by render-time table style changes.
    # Constructor options mirror the renderer's and additionally apply default
    # formatting (na representation, decimal, thousands, escape, formatter) taken from
    # option settings when not given.
    r"""
    Helps style a DataFrame or Series according to the data with HTML and CSS.

    This class provides methods for styling and formatting a Pandas DataFrame or Series.
    The styled output can be rendered as HTML or LaTeX, and it supports CSS-based
    styling, allowing users to control colors, font styles, and other visual aspects of
    tabular data. It is particularly useful for presenting DataFrame objects in a
    Jupyter Notebook environment or when exporting styled tables for reports and

    Parameters
    ----------
    data : Series or DataFrame
        Data to be styled - either a Series or DataFrame.
    precision : int, optional
        Precision to round floats to. If not given defaults to
        ``pandas.options.styler.format.precision``.
    table_styles : list-like, default None
        List of {selector: (attr, value)} dicts; see Notes.
    uuid : str, default None
        A unique identifier to avoid CSS collisions; generated automatically.
    caption : str, tuple, default None
        String caption to attach to the table. Tuple only used for LaTeX dual captions.
    table_attributes : str, default None
        Items that show up in the opening ``<table>`` tag
        in addition to automatic (by default) id.
    cell_ids : bool, default True
        If True, each cell will have an ``id`` attribute in their HTML tag.
        The ``id`` takes the form ``T_<uuid>_row<num_row>_col<num_col>``
        where ``<uuid>`` is the unique identifier, ``<num_row>`` is the row
        number and ``<num_col>`` is the column number.
    na_rep : str, optional
        Representation for missing values.
        If ``na_rep`` is None, no special formatting is applied, and falls back to
        ``pandas.options.styler.format.na_rep``.

    uuid_len : int, default 5
        If ``uuid`` is not specified, the length of the ``uuid`` to randomly generate
        expressed in hex characters, in range [0, 32].
    decimal : str, optional
        Character used as decimal separator for floats, complex and integers. If not
        given uses ``pandas.options.styler.format.decimal``.
    thousands : str, optional, default None
        Character used as thousands separator for floats, complex and integers. If not
        given uses ``pandas.options.styler.format.thousands``.
    escape : str, optional
        Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
        in cell display string with HTML-safe sequences.

        Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
        ``{``, ``}``, ``~``, ``^``, and ``\`` in the cell display string with
        LaTeX-safe sequences. Use 'latex-math' to replace the characters
        the same way as in 'latex' mode, except for math substrings,
        which either are surrounded by two characters ``$`` or start with
        the character ``\(`` and end with ``\)``.
        If not given uses ``pandas.options.styler.format.escape``.
    formatter : str, callable, dict, optional
        Object to define how values are displayed. See ``Styler.format``. If not given
        uses ``pandas.options.styler.format.formatter``.

    Attributes
    ----------
    index : data.index Index
    columns : data.columns Index
    env : Jinja2 jinja2.Environment
    template_html : Jinja2 Template
    template_html_table : Jinja2 Template
    template_html_style : Jinja2 Template
    template_latex : Jinja2 Template
    loader : Jinja2 Loader

    See Also
    --------
    DataFrame.style : Return a Styler object containing methods for building
        a styled HTML representation for the DataFrame.

    Notes
    -----
    .. warning::

       ``Styler`` is primarily intended for use on safe input that you control.
       When using ``Styler`` on untrusted, user-provided input to serve HTML,
       you should set ``escape="html"`` to prevent security vulnerabilities.
       See the Jinja2 documentation on escaping HTML for more.

    Most styling will be done by passing style functions into
    ``Styler.apply`` or ``Styler.map``. Style functions should
    return values with strings containing CSS ``'attr: value'`` that will
    be applied to the indicated cells.

    If using in the Jupyter notebook, Styler has defined a ``_repr_html_``
    to automatically render itself. Otherwise call Styler.to_html to get
    the generated HTML.

    CSS classes are attached to the generated HTML

    * Index and Column names include ``index_name`` and ``level<k>``
      where `k` is its level in a MultiIndex
    * Index label cells include

      * ``row_heading``
      * ``row<n>`` where `n` is the numeric position of the row
      * ``level<k>`` where `k` is the level in a MultiIndex

    * Column label cells include

      * ``col_heading``
      * ``col<n>`` where `n` is the numeric position of the column
      * ``level<k>`` where `k` is the level in a MultiIndex

    * Blank cells include ``blank``
    * Data cells include ``data``
    * Trimmed cells include ``col_trim`` or ``row_trim``.

    Any, or all, or these classes can be renamed by using the ``css_class_names``
    argument in ``Styler.set_table_styles``, giving a value such as
    *{"row": "MY_ROW_CLASS", "col_trim": "", "row_trim": ""}*.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     [[1.0, 2.0, 3.0], [4, 5, 6]], index=["a", "b"], columns=["A", "B", "C"]
    ... )
    >>> pd.io.formats.style.Styler(
    ...     df, precision=2, caption="My table"
    ... )  # doctest: +SKIP

    Please see:
    `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
    """

    def __init__(
        self,
        data: DataFrame | Series,
        precision: int | None = None,
        table_styles: CSSStyles | None = None,
        uuid: str | None = None,
        caption: str | tuple | list | None = None,
        table_attributes: str | None = None,
        cell_ids: bool = True,
        na_rep: str | None = None,
        uuid_len: int = 5,
        decimal: str | None = None,
        thousands: str | None = None,
        escape: str | None = None,
        formatter: ExtFormatter | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the styler. Parameters: data (frame or series), precision, table_styles,
        # uuid, caption, table_attributes, cell_ids, uuid_len are passed to the base
        # class (see its documentation); na_rep, decimal, thousands, escape and formatter
        # are formatting defaults (each falls back to the matching styler option when
        # not given or falsy). Calls the base constructor, resolves the format option
        # defaults, then applies a default display format to all cells using them (the
        # precision is handled by the base class). Raises a type error for invalid data
        # or uuid length.
        super().__init__(
            data=data,
            uuid=uuid,
            uuid_len=uuid_len,
            table_styles=table_styles,
            table_attributes=table_attributes,
            caption=caption,
            cell_ids=cell_ids,
            precision=precision,
        )

        # validate ordered args
        thousands = thousands or get_option("styler.format.thousands")
        decimal = decimal or get_option("styler.format.decimal")
        na_rep = na_rep or get_option("styler.format.na_rep")
        escape = escape or get_option("styler.format.escape")
        formatter = formatter or get_option("styler.format.formatter")
        # precision is handled by superclass as default for performance

        self.format(
            formatter=formatter,
            precision=precision,
            na_rep=na_rep,
            escape=escape,
            decimal=decimal,
            thousands=thousands,
        )

    def concat(self, other: Styler) -> Styler:
        # [FUNCTION DOC]
        # Appends another styler below this one in the rendered output (summary rows or
        # similar). other must be a styler (type error otherwise), have columns equal to this
        # one's (value error otherwise) and the same number of index levels (value error).
        # Mutates this styler by adding other to its list of concatenated stylers and
        # returns this styler. Does not copy other.
        """
        Append another Styler to combine the output into a single table.

        Parameters
        ----------
        other : Styler
            The other Styler object which has already been styled and formatted. The
            data for this Styler must have the same columns as the original, and the
            number of index levels must also be the same to render correctly.

        Returns
        -------
        Styler
            Instance of class with specified Styler appended.

        See Also
        --------
        Styler.clear : Reset the ``Styler``, removing any previously applied styles.
        Styler.export : Export the styles applied to the current Styler.

        Notes
        -----
        The purpose of this method is to extend existing styled dataframes with other
        metrics that may be useful but may not conform to the original's structure.
        For example adding a sub total row, or displaying metrics such as means,
        variance or counts.

        Styles that are applied using the ``apply``, ``map``, ``apply_index``
        and ``map_index``, and formatting applied with ``format`` and
        ``format_index`` will be preserved.

        .. warning::
            Only the output methods ``to_html``, ``to_string`` and ``to_latex``
            currently work with concatenated Stylers.

            Other output methods, including ``to_excel``, **do not** work with
            concatenated Stylers.

        The following should be noted:

          - ``table_styles``, ``table_attributes``, ``caption`` and ``uuid`` are all
            inherited from the original Styler and not ``other``.
          - hidden columns and hidden index levels will be inherited from the
            original Styler
          - ``css`` will be inherited from the original Styler, and the value of
            keys ``data``, ``row_heading`` and ``row`` will be prepended with
            ``foot0_``. If more concats are chained, their styles will be prepended
            with ``foot1_``, ''foot_2'', etc., and if a concatenated style have
            another concatenated style, the second style will be prepended with
            ``foot{parent}_foot{child}_``.

        A common use case is to concatenate user defined functions with
        ``DataFrame.agg`` or with described statistics via ``DataFrame.describe``.
        See examples.

        Examples
        --------
        A common use case is adding totals rows, or otherwise, via methods calculated
        in ``DataFrame.agg``.

        >>> df = pd.DataFrame(
        ...     [[4, 6], [1, 9], [3, 4], [5, 5], [9, 6]],
        ...     columns=["Mike", "Jim"],
        ...     index=["Mon", "Tue", "Wed", "Thurs", "Fri"],
        ... )
        >>> styler = df.style.concat(df.agg(["sum"]).style)  # doctest: +SKIP

        .. figure:: ../../_static/style/footer_simple.png

        Since the concatenated object is a Styler the existing functionality can be
        used to conditionally format it as well as the original.

        >>> descriptors = df.agg(["sum", "mean", lambda s: s.dtype])
        >>> descriptors.index = ["Total", "Average", "dtype"]
        >>> other = (
        ...     descriptors.style.highlight_max(
        ...         axis=1, subset=(["Total", "Average"], slice(None))
        ...     )
        ...     .format(subset=("Average", slice(None)), precision=2, decimal=",")
        ...     .map(lambda v: "font-weight: bold;")
        ... )
        >>> styler = df.style.highlight_max(color="salmon").set_table_styles(
        ...     [{"selector": ".foot_row0", "props": "border-top: 1px solid black;"}]
        ... )
        >>> styler.concat(other)  # doctest: +SKIP

        .. figure:: ../../_static/style/footer_extended.png

        When ``other`` has fewer index levels than the original Styler it is possible
        to extend the index in ``other``, with placeholder levels.

        >>> df = pd.DataFrame(
        ...     [[1], [2]], index=pd.MultiIndex.from_product([[0], [1, 2]])
        ... )
        >>> descriptors = df.agg(["sum"])
        >>> descriptors.index = pd.MultiIndex.from_product([[""], descriptors.index])
        >>> df.style.concat(descriptors.style)  # doctest: +SKIP
        """
        if not isinstance(other, Styler):
            raise TypeError("`other` must be of type `Styler`")
        if not self.data.columns.equals(other.data.columns):
            raise ValueError("`other.data` must have same columns as `Styler.data`")
        if not self.data.index.nlevels == other.data.index.nlevels:
            raise ValueError(
                "number of index levels must be same in `other` "
                "as in `Styler`. See documentation for suggestions."
            )
        self.concatenated.append(other)
        return self

    def _repr_html_(self) -> str | None:
        # [FUNCTION DOC]
        # Notebook display hook: returns the HTML rendering when the render representation
        # option is html, otherwise none.
        """
        Hooks into Jupyter notebook rich display system, which calls _repr_html_ by
        default if an object is returned at the end of a cell.
        """
        if get_option("styler.render.repr") == "html":
            return self.to_html()
        return None

    def _repr_latex_(self) -> str | None:
        # [FUNCTION DOC]
        # Notebook display hook: returns the LaTeX rendering when the render
        # representation option is latex, otherwise none.
        if get_option("styler.render.repr") == "latex":
            return self.to_latex()
        return None

    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = None,
        css_class: str | None = None,
        as_title_attribute: bool = False,
    ) -> Styler:
        # [FUNCTION DOC]
        # Attaches hover tooltips to cells. Parameters: ttips is a frame of strings
        # aligned with the styled data (missing or empty entries mean no tooltip); props
        # sets CSS declarations for the tooltip class (default none: unchanged);
        # css_class sets the tooltip class name (default none: unchanged);
        # as_title_attribute (default false) uses the native title attribute instead of CSS
        # rendering. Raises a not-implemented error if cell ids are disabled, and a key
        # error if ttips has a non-unique index or columns. Creates a tooltips helper if
        # none exists, stores ttips in it, and updates the class properties and name only
        # when not in title mode; in title mode only the mode flag is set. Returns this
        # styler.
        # KNOWN ISSUE: calling it again with the title mode off does not turn the title mode flag back off if it was set earlier, since the flag is assigned only in title mode.
        """
        Set the DataFrame of strings on ``Styler`` generating ``:hover`` tooltips.

        These string based tooltips are only applicable to ``<td>`` HTML elements,
        and cannot be used for column or index headers.

        Parameters
        ----------
        ttips : DataFrame
            DataFrame containing strings that will be translated to tooltips, mapped
            by identical column and index values that must exist on the underlying
            Styler data. None, NaN values, and empty strings will be ignored and
            not affect the rendered HTML.
        props : list-like or str, optional
            List of (attr, value) tuples or a valid CSS string. If ``None`` adopts
            the internal default values described in notes.
        css_class : str, optional
            Name of the tooltip class used in CSS, should conform to HTML standards.
            Only useful if integrating tooltips with external CSS. If ``None`` uses the
            internal default value 'pd-t'.
        as_title_attribute : bool, default False
            Add the tooltip text as title attribute to resultant <td> element. If True
            then props and css_class arguments are ignored.

        Returns
        -------
        Styler
            Instance of class with DataFrame set for strings on ``Styler``
                generating ``:hover`` tooltips.

        See Also
        --------
        Styler.set_table_attributes : Set the table attributes added to the
            ``<table>`` HTML element.
        Styler.set_table_styles : Set the table styles included within the
            ``<style>`` HTML element.

        Notes
        -----
        Tooltips are created by adding `<span class="pd-t"></span>` to each data cell
        and then manipulating the table level CSS to attach pseudo hover and pseudo
        after selectors to produce the required the results.

        The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

        The property 'visibility: hidden;' is a key prerequisite to the hover
        functionality, and should always be included in any manual properties
        specification, using the ``props`` argument.

        Tooltips are not designed to be efficient, and can add large amounts of
        additional HTML for larger tables, since they also require that ``cell_ids``
        is forced to `True`.

        If multiline tooltips are required, or if styling is not required and/or
        space is of concern, then utilizing as_title_attribute as True will store
        the tooltip on the <td> title attribute. This will cause no CSS
        to be generated nor will the <span> elements. Storing tooltips through
        the title attribute will mean that tooltip styling effects do not apply.

        Examples
        --------
        Basic application

        >>> df = pd.DataFrame(data=[[0, 1], [2, 3]])
        >>> ttips = pd.DataFrame(
        ...     data=[["Min", ""], [np.nan, "Max"]], columns=df.columns, index=df.index
        ... )
        >>> s = df.style.set_tooltips(ttips).to_html()

        Optionally controlling the tooltip visual display

        >>> df.style.set_tooltips(
        ...     ttips,
        ...     css_class="tt-add",
        ...     props=[
        ...         ("visibility", "hidden"),
        ...         ("position", "absolute"),
        ...         ("z-index", 1),
        ...     ],
        ... )  # doctest: +SKIP
        >>> df.style.set_tooltips(
        ...     ttips,
        ...     css_class="tt-add",
        ...     props="visibility:hidden; position:absolute; z-index:1;",
        ... )
        ... # doctest: +SKIP

        Multiline tooltips with smaller size footprint

        >>> df.style.set_tooltips(ttips, as_title_attribute=True)  # doctest: +SKIP
        """
        if not self.cell_ids:
            # tooltips not optimised for individual cell check. requires reasonable
            # redesign and more extensive code for a feature that might be rarely used.
            raise NotImplementedError(
                "Tooltips can only render with 'cell_ids' is True."
            )
        if not ttips.index.is_unique or not ttips.columns.is_unique:
            raise KeyError(
                "Tooltips render only if `ttips` has unique index and columns."
            )
        if self.tooltips is None:  # create a default instance if necessary
            self.tooltips = Tooltips()
        self.tooltips.tt_data = ttips
        if not as_title_attribute:
            if props:
                self.tooltips.class_properties = props
            if css_class:
                self.tooltips.class_name = css_class
        else:
            self.tooltips.as_title_attribute = as_title_attribute

        return self

    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: str | None = None,
        merge_cells: ExcelWriterMergeCells = True,
        encoding: str | None = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
        autofilter: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Writes the styled data to an Excel sheet. Parameters mirror the frame Excel
        # writer: excel_writer (path, buffer or writer), sheet_name (default "Sheet1"),
        # na_rep (default empty), float_format, columns, header, index, index_label,
        # startrow, startcol, engine, merge_cells, encoding, inf_rep (default "inf"),
        # verbose, freeze_panes, storage_options, autofilter (default false).
        # Creates an Excel formatter bound to this styler (so CSS styles are converted
        # to Excel styles) and writes through it. Returns nothing; the file/buffer is
        # written as a side effect.
        # KNOWN ISSUE: the encoding and verbose arguments are accepted but never passed on, so they have no effect.
        """
        Write Styler to an Excel sheet.

        To write a single Styler to an Excel .xlsx file it is only necessary to
        specify a target file name. To write to multiple sheets it is necessary to
        create an `ExcelWriter` object with a target file name, and specify a sheet
        in the file to write to.

        Multiple sheets may be written to by specifying unique `sheet_name`.
        With all data written to the file it is necessary to save the changes.
        Note that creating an `ExcelWriter` object with a file name that already exists
        will overwrite the existing file because the default mode is write.

        Parameters
        ----------
        excel_writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter.
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, optional
            Format string for floating point numbers. For example
            ``float_format="%.2f"`` will format 0.1234 to 0.12.
        columns : sequence or list of str, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of string is given it is
            assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : str or sequence, optional
            Column label for index column(s) if desired. If not specified, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the DataFrame uses MultiIndex.
        startrow : int, default 0
            Upper left cell row to dump data frame.
        startcol : int, default 0
            Upper left cell column to dump data frame.
        engine : str, optional
            Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this
            via the options ``io.excel.xlsx.writer`` or
            ``io.excel.xlsm.writer``.
        merge_cells : bool or 'columns', default False
            If True, write MultiIndex index and columns as merged cells.
            If 'columns', merge MultiIndex column cells only.
        encoding : str or None, default None
            Unused parameter, present for compatibility.
        inf_rep : str, default 'inf'
            Representation for infinity (there is no native representation for
            infinity in Excel).
        verbose : str, default True
            Optional unused parameter, present for compatibility.
        freeze_panes : tuple of int (length 2), optional
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen.
        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        autofilter : bool, default False
            If True, add automatic filters to all columns.

        See Also
        --------
        to_csv : Write DataFrame to a comma-separated values (csv) file.
        ExcelWriter : Class for writing DataFrame objects into excel sheets.
        read_excel : Read an Excel file into a pandas DataFrame.
        read_csv : Read a comma-separated values (csv) file into DataFrame.
        io.formats.style.Styler.to_excel : Add styles to Excel sheet.

        Notes
        -----
        For compatibility with :meth:`~DataFrame.to_csv`,
        to_excel serializes lists and dicts to strings before writing.

        Once a workbook has been saved it is not possible to write further
        data without rewriting the whole workbook.

        pandas will check the number of rows, columns,
        and cell character count does not exceed Excel's limitations.
        All other limitations must be checked by the user.

        Examples
        --------

        Create, write to and save a workbook:

        >>> df1 = pd.DataFrame(
        ...     [["a", "b"], ["c", "d"]],
        ...     index=["row 1", "row 2"],
        ...     columns=["col 1", "col 2"],
        ... )
        >>> df1.to_excel("output.xlsx")  # doctest: +SKIP

        To specify the sheet name:

        >>> df1.to_excel("output.xlsx", sheet_name="Sheet_name_1")  # doctest: +SKIP

        If you wish to write to more than one sheet in the workbook, it is
        necessary to specify an ExcelWriter object:

        >>> df2 = df1.copy()
        >>> with pd.ExcelWriter("output.xlsx") as writer:  # doctest: +SKIP
        ...     df1.to_excel(writer, sheet_name="Sheet_name_1")
        ...     df2.to_excel(writer, sheet_name="Sheet_name_2")

        ExcelWriter can also be used to append to an existing Excel file:

        >>> with pd.ExcelWriter("output.xlsx", mode="a") as writer:  # doctest: +SKIP
        ...     df1.to_excel(writer, sheet_name="Sheet_name_3")

        To set the library that is used to write the Excel file,
        you can pass the `engine` keyword (the default engine is
        automatically chosen depending on the file extension):

        >>> df1.to_excel("output1.xlsx", engine="xlsxwriter")  # doctest: +SKIP
        """
        from pandas.io.formats.excel import ExcelFormatter

        formatter = ExcelFormatter(
            self,
            na_rep=na_rep,
            cols=columns,
            header=header,
            float_format=float_format,
            index=index,
            index_label=index_label,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
            autofilter=autofilter,
        )
        formatter.write(
            excel_writer,
            sheet_name=sheet_name,
            startrow=startrow,
            startcol=startcol,
            freeze_panes=freeze_panes,
            engine=engine,
            storage_options=storage_options,
        )

    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> None: ...
        # [FUNCTION DOC]
        # Typing overload only: with a target buffer, returns nothing. See the
        # implementation below.

    @overload
    def to_latex(
        self,
        buf: None = ...,
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> str: ...
        # [FUNCTION DOC]
        # Typing overload only: without a buffer, returns the LaTeX string. See the
        # implementation below.

    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        column_format: str | None = None,
        position: str | None = None,
        position_float: str | None = None,
        hrules: bool | None = None,
        clines: str | None = None,
        label: str | None = None,
        caption: str | tuple | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        multirow_align: str | None = None,
        multicol_align: str | None = None,
        siunitx: bool = False,
        environment: str | None = None,
        encoding: str | None = None,
        convert_css: bool = False,
    ) -> str | None:
        # [FUNCTION DOC]
        # Renders as LaTeX. Works on a deep copy. Parameters (keyword-only except buf):
        # buf is a path or writable buffer, or none to return text; column_format is the
        # column specification (default none: reuse a previously set table style, else
        # generate one with left alignment for index levels and text columns and right
        # alignment, or the siunitx S type when enabled, for numeric columns, skipping
        # hidden items); position and position_float (centering, raggedleft or
        # raggedright; value error for others or when the longtable environment is
        # used); hrules adds top/mid/bottom rules (default from option); clines
        # selects horizontal line style; label (colons are protected from CSS parsing);
        # caption; sparse_index and sparse_columns (default from options); multirow_align
        # and multicol_align (default from options); siunitx; environment (default from
        # option); encoding (only used when buf is a path string, default from option);
        # convert_css converts CSS properties to LaTeX commands.
        # Each setting is stored as a table style on the copy, the copy is rendered, and
        # the text is written to buf or returned.
        # Side effects: temporarily replaces the data's column labels with a plain range
        # to find numeric columns and restores them (not exception safe).
        # Returns the string when buf is none, else none.
        r"""
        Write Styler to a file, buffer or string in LaTeX format.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result
            is returned as a string.
        column_format : str, optional
            The LaTeX column specification placed in location:

            \\begin{tabular}{<column_format>}

            Defaults to 'l' for index and
            non-numeric data columns, and, for numeric data columns,
            to 'r' by default, or 'S' if ``siunitx`` is ``True``.
        position : str, optional
            The LaTeX positional argument (e.g. 'h!') for tables, placed in location:

            ``\\begin{table}[<position>]``.
        position_float : {"centering", "raggedleft", "raggedright"}, optional
            The LaTeX float command placed in location:

            \\begin{table}[<position>]

            \\<position_float>

            Cannot be used if ``environment`` is "longtable".
        hrules : bool
            Set to `True` to add \\toprule, \\midrule and \\bottomrule from the
            {booktabs} LaTeX package.
            Defaults to ``pandas.options.styler.latex.hrules``, which is `False`.
        clines : str, optional
            Use to control adding \\cline commands for the index labels separation.
            Possible values are:

              - `None`: no cline commands are added (default).
              - `"all;data"`: a cline is added for every index value extending the
                width of the table, including data entries.
              - `"all;index"`: as above with lines extending only the width of the
                index entries.
              - `"skip-last;data"`: a cline is added for each index value except the
                last level (which is never sparsified), extending the widtn of the
                table.
              - `"skip-last;index"`: as above with lines extending only the width of the
                index entries.

        label : str, optional
            The LaTeX label included as: \\label{<label>}.
            This is used with \\ref{<label>} in the main .tex file.
        caption : str, tuple, optional
            If string, the LaTeX table caption included as: \\caption{<caption>}.
            If tuple, i.e ("full caption", "short caption"), the caption included
            as: \\caption[<caption[1]>]{<caption[0]>}.
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index``, which is `True`.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns``, which
            is `True`.
        multirow_align : {"c", "t", "b", "naive"}, optional
            If sparsifying hierarchical MultiIndexes whether to align text centrally,
            at the top or bottom using the multirow package. If not given defaults to
            ``pandas.options.styler.latex.multirow_align``, which is `"c"`.
            If "naive" is given renders without multirow.
        multicol_align : {"r", "c", "l", "naive-l", "naive-r"}, optional
            If sparsifying hierarchical MultiIndex columns whether to align text at
            the left, centrally, or at the right. If not given defaults to
            ``pandas.options.styler.latex.multicol_align``, which is "r".
            If a naive option is given renders without multicol.
            Pipe decorators can also be added to non-naive values to draw vertical
            rules, e.g. "\|r" will draw a rule on the left side of right aligned merged
            cells.
        siunitx : bool, default False
            Set to ``True`` to structure LaTeX compatible with the {siunitx} package.
        environment : str, optional
            If given, the environment that will replace 'table' in ``\\begin{table}``.
            If 'longtable' is specified then a more suitable template is
            rendered. If not given defaults to
            ``pandas.options.styler.latex.environment``, which is `None`.
        encoding : str, optional
            Character encoding setting. Defaults
            to ``pandas.options.styler.render.encoding``, which is "utf-8".
        convert_css : bool, default False
            Convert simple cell-styles from CSS to LaTeX format. Any CSS not found in
            conversion table is dropped. A style can be forced by adding option
            `--latex`. See notes.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        Styler.format: Format the text display value of cells.

        Notes
        -----
        **Latex Packages**

        For the following features we recommend the following LaTeX inclusions:

        ===================== ==========================================================
        Feature               Inclusion
        ===================== ==========================================================
        sparse columns        none: included within default {tabular} environment
        sparse rows           \\usepackage{multirow}
        hrules                \\usepackage{booktabs}
        colors                \\usepackage[table]{xcolor}
        siunitx               \\usepackage{siunitx}
        bold (with siunitx)   | \\usepackage{etoolbox}
                              | \\robustify\\bfseries
                              | \\sisetup{detect-all = true}  *(within {document})*
        italic (with siunitx) | \\usepackage{etoolbox}
                              | \\robustify\\itshape
                              | \\sisetup{detect-all = true}  *(within {document})*
        environment           \\usepackage{longtable} if arg is "longtable"
                              | or any other relevant environment package
        hyperlinks            \\usepackage{hyperref}
        ===================== ==========================================================

        **Cell Styles**

        LaTeX styling can only be rendered if the accompanying styling functions have
        been constructed with appropriate LaTeX commands. All styling
        functionality is built around the concept of a CSS ``(<attribute>, <value>)``
        pair (see `Table Visualization <../../user_guide/style.ipynb>`_), and this
        should be replaced by a LaTeX
        ``(<command>, <options>)`` approach. Each cell will be styled individually
        using nested LaTeX commands with their accompanied options.

        For example the following code will highlight and bold a cell in HTML-CSS:

        >>> df = pd.DataFrame([[1, 2], [3, 4]])
        >>> s = df.style.highlight_max(axis=None,
        ...                            props='background-color:red; font-weight:bold;')
        >>> s.to_html()  # doctest: +SKIP

        The equivalent using LaTeX only commands is the following:

        >>> s = df.style.highlight_max(axis=None,
        ...                            props='cellcolor:{red}; bfseries: ;')
        >>> s.to_latex()  # doctest: +SKIP

        Internally these structured LaTeX ``(<command>, <options>)`` pairs
        are translated to the
        ``display_value`` with the default structure:
        ``\<command><options> <display_value>``.
        Where there are multiple commands the latter is nested recursively, so that
        the above example highlighted cell is rendered as
        ``\cellcolor{red} \bfseries 4``.

        Occasionally this format does not suit the applied command, or
        combination of LaTeX packages that is in use, so additional flags can be
        added to the ``<options>``, within the tuple, to result in different
        positions of required braces (the **default** being the same as ``--nowrap``):

        =================================== ============================================
        Tuple Format                           Output Structure
        =================================== ============================================
        (<command>,<options>)               \\<command><options> <display_value>
        (<command>,<options> ``--nowrap``)  \\<command><options> <display_value>
        (<command>,<options> ``--rwrap``)   \\<command><options>{<display_value>}
        (<command>,<options> ``--wrap``)    {\\<command><options> <display_value>}
        (<command>,<options> ``--lwrap``)   {\\<command><options>} <display_value>
        (<command>,<options> ``--dwrap``)   {\\<command><options>}{<display_value>}
        =================================== ============================================

        For example the `textbf` command for font-weight
        should always be used with `--rwrap` so ``('textbf', '--rwrap')`` will render a
        working cell, wrapped with braces, as ``\textbf{<display_value>}``.

        A more comprehensive example is as follows:

        >>> df = pd.DataFrame([[1, 2.2, "dogs"], [3, 4.4, "cats"], [2, 6.6, "cows"]],
        ...                   index=["ix1", "ix2", "ix3"],
        ...                   columns=["Integers", "Floats", "Strings"])
        >>> s = df.style.highlight_max(
        ...     props='cellcolor:[HTML]{FFFF00}; color:{red};'
        ...           'textit:--rwrap; textbf:--rwrap;'
        ... )
        >>> s.to_latex()  # doctest: +SKIP

        .. figure:: ../../_static/style/latex_1.png

        **Table Styles**

        Internally Styler uses its ``table_styles`` object to parse the
        ``column_format``, ``position``, ``position_float``, and ``label``
        input arguments. These arguments are added to table styles in the format:

        .. code-block:: python

            set_table_styles([
                {"selector": "column_format", "props": f":{column_format};"},
                {"selector": "position", "props": f":{position};"},
                {"selector": "position_float", "props": f":{position_float};"},
                {"selector": "label", "props": f":{{{label.replace(':','§')}}};"}
            ], overwrite=False)

        Exception is made for the ``hrules`` argument which, in fact, controls all three
        commands: ``toprule``, ``bottomrule`` and ``midrule`` simultaneously. Instead of
        setting ``hrules`` to ``True``, it is also possible to set each
        individual rule definition, by manually setting the ``table_styles``,
        for example below we set a regular ``toprule``, set an ``hline`` for
        ``bottomrule`` and exclude the ``midrule``:

        .. code-block:: python

            set_table_styles([
                {'selector': 'toprule', 'props': ':toprule;'},
                {'selector': 'bottomrule', 'props': ':hline;'},
            ], overwrite=False)

        If other ``commands`` are added to table styles they will be detected, and
        positioned immediately above the '\\begin{tabular}' command. For example to
        add odd and even row coloring, from the {colortbl} package, in format
        ``\rowcolors{1}{pink}{red}``, use:

        .. code-block:: python

            set_table_styles([
                {'selector': 'rowcolors', 'props': ':{1}{pink}{red};'}
            ], overwrite=False)

        A more comprehensive example using these arguments is as follows:

        >>> df.columns = pd.MultiIndex.from_tuples([
        ...     ("Numeric", "Integers"),
        ...     ("Numeric", "Floats"),
        ...     ("Non-Numeric", "Strings")
        ... ])
        >>> df.index = pd.MultiIndex.from_tuples([
        ...     ("L0", "ix1"), ("L0", "ix2"), ("L1", "ix3")
        ... ])
        >>> s = df.style.highlight_max(
        ...     props='cellcolor:[HTML]{FFFF00}; color:{red}; itshape:; bfseries:;'
        ... )
        >>> s.to_latex(
        ...     column_format="rrrrr", position="h", position_float="centering",
        ...     hrules=True, label="table:5", caption="Styled LaTeX Table",
        ...     multirow_align="t", multicol_align="r"
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/latex_2.png

        **Formatting**

        To format values :meth:`Styler.format` should be used prior to calling
        `Styler.to_latex`, as well as other methods such as :meth:`Styler.hide`
        for example:

        >>> s.clear()
        >>> s.table_styles = []
        >>> s.caption = None
        >>> s.format({
        ...     ("Numeric", "Integers"): '\\${}',
        ...     ("Numeric", "Floats"): '{:.3f}',
        ...     ("Non-Numeric", "Strings"): str.upper
        ... })  # doctest: +SKIP
                        Numeric      Non-Numeric
                  Integers   Floats    Strings
        L0    ix1       $1   2.200      DOGS
              ix2       $3   4.400      CATS
        L1    ix3       $2   6.600      COWS

        >>> s.to_latex()  # doctest: +SKIP
        \begin{tabular}{llrrl}
        {} & {} & \multicolumn{2}{r}{Numeric} & {Non-Numeric} \\
        {} & {} & {Integers} & {Floats} & {Strings} \\
        \multirow[c]{2}{*}{L0} & ix1 & \\$1 & 2.200 & DOGS \\
         & ix2 & \$3 & 4.400 & CATS \\
        L1 & ix3 & \$2 & 6.600 & COWS \\
        \end{tabular}

        **CSS Conversion**

        This method can convert a Styler constructured with HTML-CSS to LaTeX using
        the following limited conversions.

        ================== ==================== ============= ==========================
        CSS Attribute      CSS value            LaTeX Command LaTeX Options
        ================== ==================== ============= ==========================
        font-weight        | bold               | bfseries
                           | bolder             | bfseries
        font-style         | italic             | itshape
                           | oblique            | slshape
        background-color   | red                cellcolor     | {red}--lwrap
                           | #fe01ea                          | [HTML]{FE01EA}--lwrap
                           | #f0e                             | [HTML]{FF00EE}--lwrap
                           | rgb(128,255,0)                   | [rgb]{0.5,1,0}--lwrap
                           | rgba(128,0,0,0.5)                | [rgb]{0.5,0,0}--lwrap
                           | rgb(25%,255,50%)                 | [rgb]{0.25,1,0.5}--lwrap
        color              | red                color         | {red}
                           | #fe01ea                          | [HTML]{FE01EA}
                           | #f0e                             | [HTML]{FF00EE}
                           | rgb(128,255,0)                   | [rgb]{0.5,1,0}
                           | rgba(128,0,0,0.5)                | [rgb]{0.5,0,0}
                           | rgb(25%,255,50%)                 | [rgb]{0.25,1,0.5}
        ================== ==================== ============= ==========================

        It is also possible to add user-defined LaTeX only styles to a HTML-CSS Styler
        using the ``--latex`` flag, and to add LaTeX parsing options that the
        converter will detect within a CSS-comment.

        >>> df = pd.DataFrame([[1]])
        >>> df.style.set_properties(
        ...     **{"font-weight": "bold /* --dwrap */", "Huge": "--latex--rwrap"}
        ... ).to_latex(convert_css=True)  # doctest: +SKIP
        \begin{tabular}{lr}
        {} & {0} \\
        0 & {\bfseries}{\Huge{1}} \\
        \end{tabular}

        Examples
        --------
        Below we give a complete step by step example adding some advanced features
        and noting some common gotchas.

        First we create the DataFrame and Styler as usual, including MultiIndex rows
        and columns, which allow for more advanced formatting options:

        >>> cidx = pd.MultiIndex.from_arrays([
        ...     ["Equity", "Equity", "Equity", "Equity",
        ...      "Stats", "Stats", "Stats", "Stats", "Rating"],
        ...     ["Energy", "Energy", "Consumer", "Consumer", "", "", "", "", ""],
        ...     ["BP", "Shell", "H&M", "Unilever",
        ...      "Std Dev", "Variance", "52w High", "52w Low", ""]
        ... ])
        >>> iidx = pd.MultiIndex.from_arrays([
        ...     ["Equity", "Equity", "Equity", "Equity"],
        ...     ["Energy", "Energy", "Consumer", "Consumer"],
        ...     ["BP", "Shell", "H&M", "Unilever"]
        ... ])
        >>> styler = pd.DataFrame([
        ...     [1, 0.8, 0.66, 0.72, 32.1678, 32.1678**2, 335.12, 240.89, "Buy"],
        ...     [0.8, 1.0, 0.69, 0.79, 1.876, 1.876**2, 14.12, 19.78, "Hold"],
        ...     [0.66, 0.69, 1.0, 0.86, 7, 7**2, 210.9, 140.6, "Buy"],
        ...     [0.72, 0.79, 0.86, 1.0, 213.76, 213.76**2, 2807, 3678, "Sell"],
        ... ], columns=cidx, index=iidx).style

        Second we will format the display and, since our table is quite wide, will
        hide the repeated level-0 of the index:

        >>> (styler.format(subset="Equity", precision=2)
        ...     .format(subset="Stats", precision=1, thousands=",")
        ...     .format(subset="Rating", formatter=str.upper)
        ...     .format_index(escape="latex", axis=1)
        ...     .format_index(escape="latex", axis=0)
        ...     .hide(level=0, axis=0))  # doctest: +SKIP

        Note that one of the string entries of the index and column headers is "H&M".
        Without applying the `escape="latex"` option to the `format_index` method the
        resultant LaTeX will fail to render, and the error returned is quite
        difficult to debug. Using the appropriate escape the "&" is converted to "\\&".

        Thirdly we will apply some (CSS-HTML) styles to our object. We will use a
        builtin method and also define our own method to highlight the stock
        recommendation:

        >>> def rating_color(v):
        ...     if v == "Buy":
        ...         color = "#33ff85"
        ...     elif v == "Sell":
        ...         color = "#ff5933"
        ...     else:
        ...         color = "#ffdd33"
        ...     return f"color: {color}; font-weight: bold;"
        >>> (styler.background_gradient(cmap="inferno", subset="Equity", vmin=0, vmax=1)
        ...     .map(rating_color, subset="Rating"))  # doctest: +SKIP

        All the above styles will work with HTML (see below) and LaTeX upon conversion:

        .. figure:: ../../_static/style/latex_stocks_html.png

        However, we finally want to add one LaTeX only style
        (from the {graphicx} package), that is not easy to convert from CSS and
        pandas does not support it. Notice the `--latex` flag used here,
        as well as `--rwrap` to ensure this is formatted correctly and
        not ignored upon conversion.

        >>> styler.map_index(
        ...     lambda v: "rotatebox:{45}--rwrap--latex;", level=2, axis=1
        ... )  # doctest: +SKIP

        Finally we render our LaTeX adding in other options as required:

        >>> styler.to_latex(
        ...     caption="Selected stock correlation and simple statistics.",
        ...     clines="skip-last;data",
        ...     convert_css=True,
        ...     position_float="centering",
        ...     multicol_align="|c|",
        ...     hrules=True,
        ... )  # doctest: +SKIP
        \begin{table}
        \centering
        \caption{Selected stock correlation and simple statistics.}
        \begin{tabular}{llrrrrrrrrl}
        \toprule
         &  & \multicolumn{4}{|c|}{Equity} & \multicolumn{4}{|c|}{Stats} & Rating \\
         &  & \multicolumn{2}{|c|}{Energy} & \multicolumn{2}{|c|}{Consumer} &
        \multicolumn{4}{|c|}{} &  \\
         &  & \rotatebox{45}{BP} & \rotatebox{45}{Shell} & \rotatebox{45}{H\&M} &
        \rotatebox{45}{Unilever} & \rotatebox{45}{Std Dev} & \rotatebox{45}{Variance} &
        \rotatebox{45}{52w High} & \rotatebox{45}{52w Low} & \rotatebox{45}{} \\
        \midrule
        \multirow[c]{2}{*}{Energy} & BP & {\cellcolor[HTML]{FCFFA4}}
        \color[HTML]{000000} 1.00 & {\cellcolor[HTML]{FCA50A}} \color[HTML]{000000}
        0.80 & {\cellcolor[HTML]{EB6628}} \color[HTML]{F1F1F1} 0.66 &
        {\cellcolor[HTML]{F68013}} \color[HTML]{F1F1F1} 0.72 & 32.2 & 1,034.8 & 335.1
        & 240.9 & \color[HTML]{33FF85} \bfseries BUY \\
         & Shell & {\cellcolor[HTML]{FCA50A}} \color[HTML]{000000} 0.80 &
        {\cellcolor[HTML]{FCFFA4}} \color[HTML]{000000} 1.00 &
        {\cellcolor[HTML]{F1731D}} \color[HTML]{F1F1F1} 0.69 &
        {\cellcolor[HTML]{FCA108}} \color[HTML]{000000} 0.79 & 1.9 & 3.5 & 14.1 &
        19.8 & \color[HTML]{FFDD33} \bfseries HOLD \\
        \cline{1-11}
        \multirow[c]{2}{*}{Consumer} & H\&M & {\cellcolor[HTML]{EB6628}}
        \color[HTML]{F1F1F1} 0.66 & {\cellcolor[HTML]{F1731D}} \color[HTML]{F1F1F1}
        0.69 & {\cellcolor[HTML]{FCFFA4}} \color[HTML]{000000} 1.00 &
        {\cellcolor[HTML]{FAC42A}} \color[HTML]{000000} 0.86 & 7.0 & 49.0 & 210.9 &
        140.6 & \color[HTML]{33FF85} \bfseries BUY \\
         & Unilever & {\cellcolor[HTML]{F68013}} \color[HTML]{F1F1F1} 0.72 &
        {\cellcolor[HTML]{FCA108}} \color[HTML]{000000} 0.79 &
        {\cellcolor[HTML]{FAC42A}} \color[HTML]{000000} 0.86 &
        {\cellcolor[HTML]{FCFFA4}} \color[HTML]{000000} 1.00 & 213.8 & 45,693.3 &
        2,807.0 & 3,678.0 & \color[HTML]{FF5933} \bfseries SELL \\
        \cline{1-11}
        \bottomrule
        \end{tabular}
        \end{table}

        .. figure:: ../../_static/style/latex_stocks.png
        """
        obj = self._copy(deepcopy=True)  # manipulate table_styles on obj, not self

        table_selectors = (
            [style["selector"] for style in self.table_styles]
            if self.table_styles is not None
            else []
        )

        if column_format is not None:
            # add more recent setting to table_styles
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )
        elif "column_format" in table_selectors:
            pass  # adopt what has been previously set in table_styles
        else:
            # create a default: set float, complex, int cols to 'r' ('S'), index to 'l'
            _original_columns = self.data.columns
            self.data.columns = RangeIndex(stop=len(self.data.columns))
            numeric_cols = self.data._get_numeric_data().columns.to_list()
            self.data.columns = _original_columns
            column_format = ""
            for level in range(self.index.nlevels):
                column_format += "" if self.hide_index_[level] else "l"
            for ci, _ in enumerate(self.data.columns):
                if ci not in self.hidden_columns:
                    column_format += (
                        ("r" if not siunitx else "S") if ci in numeric_cols else "l"
                    )
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )

        if position:
            obj.set_table_styles(
                [{"selector": "position", "props": f":{position}"}],
                overwrite=False,
            )

        if position_float:
            if environment == "longtable":
                raise ValueError(
                    "`position_float` cannot be used in 'longtable' `environment`"
                )
            if position_float not in ["raggedright", "raggedleft", "centering"]:
                raise ValueError(
                    f"`position_float` should be one of "
                    f"'raggedright', 'raggedleft', 'centering', "
                    f"got: '{position_float}'"
                )
            obj.set_table_styles(
                [{"selector": "position_float", "props": f":{position_float}"}],
                overwrite=False,
            )

        hrules = get_option("styler.latex.hrules") if hrules is None else hrules
        if hrules:
            obj.set_table_styles(
                [
                    {"selector": "toprule", "props": ":toprule"},
                    {"selector": "midrule", "props": ":midrule"},
                    {"selector": "bottomrule", "props": ":bottomrule"},
                ],
                overwrite=False,
            )

        if label:
            obj.set_table_styles(
                [{"selector": "label", "props": f":{{{label.replace(':', '§')}}}"}],
                overwrite=False,
            )

        if caption:
            obj.set_caption(caption)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")
        environment = environment or get_option("styler.latex.environment")
        multicol_align = multicol_align or get_option("styler.latex.multicol_align")
        multirow_align = multirow_align or get_option("styler.latex.multirow_align")
        latex = obj._render_latex(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            multirow_align=multirow_align,
            multicol_align=multicol_align,
            environment=environment,
            convert_css=convert_css,
            siunitx=siunitx,
            clines=clines,
        )

        encoding = (
            (encoding or get_option("styler.render.encoding"))
            if isinstance(buf, str)  # i.e. a filepath
            else encoding
        )
        return save_to_buffer(latex, buf=buf, encoding=encoding)

    @overload
    def to_typst(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
    ) -> None: ...
        # [FUNCTION DOC]
        # Typing overload only: with a target buffer, returns nothing. See the
        # implementation below.

    @overload
    def to_typst(
        self,
        buf: None = ...,
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
    ) -> str: ...
        # [FUNCTION DOC]
        # Typing overload only: without a buffer, returns the Typst string. See the
        # implementation below.

    def to_typst(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Renders as Typst markup. Parameters (keyword-only except buf): buf is a path or
        # buffer (default none: return a string); encoding (used only when writing to
        # buf); sparse_index and sparse_columns (default from options); max_rows and
        # max_columns limit rendered size (default none: option-based). Works on a deep copy
        # and returns the string when buf is none, else writes it and returns none.
        """
        Write Styler to a file, buffer or string in Typst format.

        .. versionadded:: 3.0.0

        Parameters
        ----------
        buf : str, path object, file-like object, optional
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If ``None``, the result
            is returned as a string.
        encoding : str, optional
              Character encoding setting for file output (and meta tags if available).
              Defaults to ``pandas.options.styler.render.encoding`` value of "utf-8".
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows``, which is None.
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_typst : Write a DataFrame to a file,
            buffer or string in Typst format.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.to_typst()  # doctest: +SKIP

        .. code-block:: typst

            #table(
              columns: 3,
              [], [A], [B],

              [0], [1], [3],
              [1], [2], [4],
            )
        """
        obj = self._copy(deepcopy=True)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        text = obj._render_typst(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
        )
        return save_to_buffer(
            text, buf=buf, encoding=(encoding if buf is not None else None)
        )

    @overload
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs,
    ) -> None: ...
        # [FUNCTION DOC]
        # Typing overload only: with a target buffer, returns nothing. See the
        # implementation below.

    @overload
    def to_html(
        self,
        buf: None = ...,
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs,
    ) -> str: ...
        # [FUNCTION DOC]
        # Typing overload only: without a buffer, returns the HTML string. See the
        # implementation below.

    def to_html(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        bold_headers: bool = False,
        caption: str | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
        **kwargs,
    ) -> str | None:
        # [FUNCTION DOC]
        # Renders as HTML. Works on a deep copy. Parameters (keyword-only except buf):
        # buf (path, buffer or none to return text); table_uuid replaces the uuid (applied
        # if truthy); table_attributes replaces table attributes (if truthy);
        # sparse_index and sparse_columns (default from options); bold_headers adds a
        # bold style for header cells (default false); caption overrides the caption if
        # not none; max_rows and max_columns limit rendered size; encoding (the meta
        # charset value, default from option, and the file encoding when buf is set);
        # doctype_html emits a full document with doctype (default false); exclude_styles
        # omits the style block (default false); extra keyword arguments are passed on to the
        # HTML template. Returns the string when buf is none, else writes and returns none.
        """
        Write Styler to a file, buffer or string in HTML-CSS format.

        Parameters
        ----------
        buf : str, path object, file-like object, optional
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If ``None``, the result
            is returned as a string.
        table_uuid : str, optional
            Id attribute assigned to the <table> HTML element in the format:

            ``<table id="T_<table_uuid>" ..>``

            If not given uses Styler's initially assigned value.
        table_attributes : str, optional
            Attributes to assign within the `<table>` HTML element in the format:

            ``<table .. <table_attributes> >``

            If not given defaults to Styler's preexisting value.
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.
        bold_headers : bool, optional
            Adds "font-weight: bold;" as a CSS property to table style header cells.
        caption : str, optional
            Set, or overwrite, the caption on Styler before rendering.
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows/max_columns``.
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).
        encoding : str, optional
              Character encoding setting for file output (and meta tags if available).
              Defaults to ``pandas.options.styler.render.encoding`` value of "utf-8".
        doctype_html : bool, default False
            Whether to output a fully structured HTML file including all
            HTML elements, or just the core ``<style>`` and ``<table>`` elements.
        exclude_styles : bool, default False
            Whether to include the ``<style>`` element and all associated element
            ``class`` and ``id`` identifiers, or solely the ``<table>`` element without
            styling identifiers.
        **kwargs
            Any additional keyword arguments are passed through to the jinja2
            ``self.template.render`` process. This is useful when you need to provide
            additional variables for a custom template.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_html: Write a DataFrame to a file, buffer or string in HTML format.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> print(df.style.to_html())  # doctest: +SKIP
        <style type="text/css">
        </style>
        <table id="T_1e78e">
          <thead>
            <tr>
              <th class="blank level0" >&nbsp;</th>
              <th id="T_1e78e_level0_col0" class="col_heading level0 col0" >A</th>
              <th id="T_1e78e_level0_col1" class="col_heading level0 col1" >B</th>
            </tr>
        ...
        """
        obj = self._copy(deepcopy=True)  # manipulate table_styles on obj, not self

        if table_uuid:
            obj.set_uuid(table_uuid)

        if table_attributes:
            obj.set_table_attributes(table_attributes)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        if bold_headers:
            obj.set_table_styles(
                [{"selector": "th", "props": "font-weight: bold;"}], overwrite=False
            )

        if caption is not None:
            obj.set_caption(caption)

        # Build HTML string..
        html = obj._render_html(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            max_rows=max_rows,
            max_cols=max_columns,
            exclude_styles=exclude_styles,
            encoding=encoding or get_option("styler.render.encoding"),
            doctype_html=doctype_html,
            **kwargs,
        )

        return save_to_buffer(
            html, buf=buf, encoding=(encoding if buf is not None else None)
        )

    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        delimiter: str = ...,
    ) -> None: ...
        # [FUNCTION DOC]
        # Typing overload only: with a target buffer, returns nothing. See the
        # implementation below.

    @overload
    def to_string(
        self,
        buf: None = ...,
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        delimiter: str = ...,
    ) -> str: ...
        # [FUNCTION DOC]
        # Typing overload only: without a buffer, returns the text. See the
        # implementation below.

    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        delimiter: str = " ",
    ) -> str | None:
        # [FUNCTION DOC]
        # Renders as plain text. Works on a deep copy. Parameters (keyword-only except
        # buf): buf (path, buffer or none to return text); encoding (used only when
        # writing to buf); sparse_index and sparse_columns (default from options);
        # max_rows and max_columns limit size; delimiter separates columns (default one
        # space). Returns the string when buf is none, else writes it and returns none.
        """
        Write Styler to a file, buffer or string in text format.

        Parameters
        ----------
        buf : str, path object, file-like object, optional
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If ``None``, the result
            is returned as a string.
        encoding : str, optional
              Character encoding setting for file output (and meta tags if available).
              Defaults to ``pandas.options.styler.render.encoding`` value of "utf-8".
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows``, which is None.
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).
        delimiter : str, default single space
            The separator between data elements.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_string : Render a DataFrame to a console-friendly tabular output.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.to_string()
        ' A B\\n0 1 3\\n1 2 4\\n'
        """
        obj = self._copy(deepcopy=True)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        text = obj._render_string(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
            delimiter=delimiter,
        )
        return save_to_buffer(
            text, buf=buf, encoding=(encoding if buf is not None else None)
        )

    def set_td_classes(self, classes: DataFrame) -> Styler:
        # [FUNCTION DOC]
        # Assigns CSS classes to individual data cells. classes is a frame of class
        # name strings (aligned to the data; missing or empty entries mean no class); its
        # index and columns must be unique, otherwise a key error is raised. The frame is
        # reindexed like the data and, for each non-empty entry, stored in the cell
        # context keyed by integer position (replacing prior entries for the same cell).
        # Returns this styler.
        """
        Set the ``class`` attribute of ``<td>`` HTML elements.

        Parameters
        ----------
        classes : DataFrame
            DataFrame containing strings that will be translated to CSS classes,
            mapped by identical column and index key values that must exist on the
            underlying Styler data. None, NaN values, and empty strings will
            be ignored and not affect the rendered HTML.

        Returns
        -------
        Styler
            Instance of class with ``class`` attribute set for ``<td>``
                HTML elements.

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_table_attributes: Set the table attributes added to the ``<table>``
            HTML element.

        Notes
        -----
        Can be used in combination with ``Styler.set_table_styles`` to define an
        internal CSS solution without reference to external CSS files.

        Examples
        --------
        >>> df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        >>> classes = pd.DataFrame(
        ...     [["min-val red", "", "blue"], ["red", None, "blue max-val"]],
        ...     index=df.index,
        ...     columns=df.columns,
        ... )
        >>> df.style.set_td_classes(classes)  # doctest: +SKIP

        Using `MultiIndex` columns and a `classes` `DataFrame` as a subset of the
        underlying,

        >>> df = pd.DataFrame(
        ...     [[1, 2], [3, 4]],
        ...     index=["a", "b"],
        ...     columns=[["level0", "level0"], ["level1a", "level1b"]],
        ... )
        >>> classes = pd.DataFrame(
        ...     ["min-val"], index=["a"], columns=[["level0"], ["level1a"]]
        ... )
        >>> df.style.set_td_classes(classes)  # doctest: +SKIP

        Form of the output with new additional css classes,

        >>> from pandas.io.formats.style import Styler
        >>> df = pd.DataFrame([[1]])
        >>> css = pd.DataFrame([["other-class"]])
        >>> s = Styler(df, uuid="_", cell_ids=False).set_td_classes(css)
        >>> s.hide(axis=0).to_html()  # doctest: +SKIP
        '<style type="text/css"></style>'
        '<table id="T__">'
        '  <thead>'
        '    <tr><th class="col_heading level0 col0" >0</th></tr>'
        '  </thead>'
        '  <tbody>'
        '    <tr><td class="data row0 col0 other-class" >1</td></tr>'
        '  </tbody>'
        '</table>'
        """
        if not classes.index.is_unique or not classes.columns.is_unique:
            raise KeyError(
                "Classes render only if `classes` has unique index and columns."
            )
        classes = classes.reindex_like(self.data)

        for r, row_tup in enumerate(classes.itertuples()):
            for c, value in enumerate(row_tup[1:]):
                if not (pd.isna(value) or value == ""):
                    self.cell_context[(r, c)] = str(value)

        return self

    def _update_ctx(self, attrs: DataFrame) -> None:
        # [FUNCTION DOC]
        # HELPER: records the CSS produced by a data-cell style function. attrs is a frame
        # of CSS strings or pair lists. Raises a key error if this styler's index or
        # columns are not unique. For each non-empty, non-missing entry converts the CSS
        # to a list of pairs and appends it to the data cell's style context at the integer
        # position located from the labels. Returns nothing.
        """
        Update the state of the ``Styler`` for data cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : DataFrame
            should contain strings of '<property>: <value>;<prop2>: <val2>'
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        """
        if not self.index.is_unique or not self.columns.is_unique:
            raise KeyError(
                "`Styler.apply` and `.map` are not compatible "
                "with non-unique index or columns."
            )

        for cn in attrs.columns:
            j = self.columns.get_loc(cn)
            ser = attrs[cn]
            for rn, c in ser.items():
                if not c or pd.isna(c):
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                i = self.index.get_loc(rn)
                self.ctx[(i, j)].extend(css_list)

    def _update_ctx_header(self, attrs: DataFrame, axis: AxisInt) -> None:
        # [FUNCTION DOC]
        # HELPER: records the CSS produced by an index or column header style function.
        # attrs is a frame whose columns are level numbers and whose rows are label
        # positions; axis 0 stores into the index style context keyed by (position, level),
        # otherwise into the columns style context keyed by (level, position). Skips
        # empty or missing entries; converts text CSS to pairs. Returns nothing.
        """
        Update the state of the ``Styler`` for header cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : Series
            Should contain strings of '<property>: <value>;<prop2>: <val2>', and an
            integer index.
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        axis : int
            Identifies whether the ctx object being updated is the index or columns
        """
        for j in attrs.columns:
            ser = attrs[j]
            for i, c in ser.items():
                if not c or pd.isna(c):
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                if axis == 0:
                    self.ctx_index[(i, j)].extend(css_list)
                else:
                    self.ctx_columns[(j, i)].extend(css_list)

    def _copy(self, deepcopy: bool = False) -> Styler:
        # [FUNCTION DOC]
        # HELPER: creates a new styler of the same class over the same data and copies
        # settings. Parameter deepcopy (default false) chooses whether the nested
        # structures (css names, display functions, hidden rows/columns, style contexts,
        # cell context, deferred functions, table styles, tooltips, concatenated stylers)
        # are deep-copied or shared; simple attributes (hide flags, table attributes, cell
        # ids, caption, uuid and the possibly customized html and latex templates) are
        # shared directly. Returns the new styler.
        # KNOWN ISSUE: the Typst and string templates are not copied even though the other templates are copied in case of customization.
        """
        Copies a Styler, allowing for deepcopy or shallow copy

        Copying a Styler aims to recreate a new Styler object which contains the same
        data and styles as the original.

        Data dependent attributes [copied and NOT exported]:
          - formatting (._display_funcs)
          - hidden index values or column values (.hidden_rows, .hidden_columns)
          - tooltips
          - cell_context (cell css classes)
          - ctx (cell css styles)
          - caption
          - concatenated stylers

        Non-data dependent attributes [copied and exported]:
          - css
          - hidden index state and hidden columns state (.hide_index_, .hide_columns_)
          - table_attributes
          - table_styles
          - applied styles (_todo)

        """
        # GH 40675, 52728
        styler = type(self)(
            self.data,  # populates attributes 'data', 'columns', 'index' as shallow
        )
        shallow = [  # simple string or boolean immutables
            "hide_index_",
            "hide_columns_",
            "hide_column_names",
            "hide_index_names",
            "table_attributes",
            "cell_ids",
            "caption",
            "uuid",
            "uuid_len",
            "template_latex",  # also copy templates if these have been customised
            "template_html_style",
            "template_html_table",
            "template_html",
        ]
        deep = [  # nested lists or dicts
            "css",
            "concatenated",
            "_display_funcs",
            "_display_funcs_index",
            "_display_funcs_columns",
            "_display_funcs_index_names",
            "_display_funcs_column_names",
            "hidden_rows",
            "hidden_columns",
            "ctx",
            "ctx_index",
            "ctx_columns",
            "cell_context",
            "_todo",
            "table_styles",
            "tooltips",
        ]

        for attr in shallow:
            setattr(styler, attr, getattr(self, attr))

        for attr in deep:
            val = getattr(self, attr)
            setattr(styler, attr, copy.deepcopy(val) if deepcopy else val)

        return styler

    def __copy__(self) -> Styler:
        # [FUNCTION DOC]
        # Copy-protocol hook: returns a shallow copy (nested state shared).
        return self._copy(deepcopy=False)

    def __deepcopy__(self, memo) -> Styler:
        # [FUNCTION DOC]
        # Deep-copy-protocol hook: returns a deep copy; the memo argument is unused.
        return self._copy(deepcopy=True)

    def clear(self) -> None:
        # [FUNCTION DOC]
        # Resets this styler to a clean default state (keeping data and uuid): builds a
        # fresh styler over the same data and uuid, copies all its attributes onto this
        # object and deletes any attributes this object has that the fresh one lacks.
        # Mutates in place; returns nothing. Format settings revert to option defaults.
        # KNOWN ISSUE: the attribute selection filters out callables by testing the attribute names, which are strings and never callable, so the filter does nothing.
        """
        Reset the ``Styler``, removing any previously applied styles.

        Returns None.

        See Also
        --------
        Styler.apply : Apply a CSS-styling function column-wise, row-wise,
            or table-wise.
        Styler.export : Export the styles applied to the current Styler.
        Styler.map : Apply a CSS-styling function elementwise.
        Styler.use : Set the styles on the current Styler.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, np.nan]})

        After any added style:

        >>> df.style.highlight_null(color="yellow")  # doctest: +SKIP

        Remove it with:

        >>> df.style.clear()  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        # create default GH 40675
        clean_copy = Styler(self.data, uuid=self.uuid)
        clean_attrs = [a for a in clean_copy.__dict__ if not callable(a)]
        self_attrs = [a for a in self.__dict__ if not callable(a)]  # maybe more attrs
        for attr in clean_attrs:
            setattr(self, attr, getattr(clean_copy, attr))
        for attr in set(self_attrs).difference(clean_attrs):
            delattr(self, attr)

    def _apply(
        self,
        func: Callable,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        **kwargs,
    ) -> Styler:
        # [FUNCTION DOC]
        # HELPER: runs a style function immediately (the deferred target of the apply
        # method). Parameters: func is the style function; axis is 0 (per column), 1 (per
        # row) or none (whole selection at once); subset limits cells; extra keyword
        # arguments are forwarded to func. Normalizes the subset to a frame selection;
        # with an empty selection uses an empty result. With axis none the function gets
        # the selection and must return a frame or array of the same shape (type error,
        # value error otherwise; arrays are wrapped with the selection's labels). Otherwise
        # applies column-wise, or row-wise via the transposed data. Raises a value error if the
        # result collapsed to a series or contains labels not in the selection. Stores the
        # resulting CSS into the style context and returns this styler.
        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]
        if data.empty:
            result = DataFrame()
        elif axis is None:
            result = func(data, **kwargs)
            if not isinstance(result, DataFrame):
                if not isinstance(result, np.ndarray):
                    raise TypeError(
                        f"Function {func!r} must return a DataFrame or ndarray "
                        f"when passed to `Styler.apply` with axis=None"
                    )
                if data.shape != result.shape:
                    raise ValueError(
                        f"Function {func!r} returned ndarray with wrong shape.\n"
                        f"Result has shape: {result.shape}\n"
                        f"Expected shape: {data.shape}"
                    )
                result = DataFrame(result, index=data.index, columns=data.columns)
        else:
            axis = self.data._get_axis_number(axis)
            if axis == 0:
                result = data.apply(func, axis=0, **kwargs)
            else:
                result = data.T.apply(func, axis=0, **kwargs).T  # see GH 42005

        if isinstance(result, Series):
            raise ValueError(
                f"Function {func!r} resulted in the apply method collapsing to a "
                f"Series.\nUsually, this is the result of the function returning a "
                f"single value, instead of list-like."
            )
        msg = (
            f"Function {func!r} created invalid {{0}} labels.\nUsually, this is "
            f"the result of the function returning a "
            f"{'Series' if axis is not None else 'DataFrame'} which contains invalid "
            f"labels, or returning an incorrectly shaped, list-like object which "
            f"cannot be mapped to labels, possibly due to applying the function along "
            f"the wrong axis.\n"
            f"Result {{0}} has shape: {{1}}\n"
            f"Expected {{0}} shape:   {{2}}"
        )
        if not all(result.index.isin(data.index)):
            raise ValueError(msg.format("index", result.index.shape, data.index.shape))
        if not all(result.columns.isin(data.columns)):
            raise ValueError(
                msg.format("columns", result.columns.shape, data.columns.shape)
            )
        self._update_ctx(result)
        return self

    def apply(
        self,
        func: Callable,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        **kwargs,
    ) -> Styler:
        # [FUNCTION DOC]
        # Queues a style function to be applied at render time (public). func takes a
        # series (axis 0 or 1) or the whole selection (axis none) and returns CSS strings of
        # the same shape; axis defaults to 0; subset limits cells (default all); extra
        # keyword arguments are passed to func. Appends a deferred call for the private
        # apply routine and returns this styler; nothing is computed now.
        """
        Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a Series if ``axis`` in [0,1] and return a list-like
            object of same length, or a Series, not necessarily of same length, with
            valid index labels considering ``subset``.
            ``func`` should take a DataFrame if ``axis`` is ``None`` and return either
            an ndarray with the same shape or a DataFrame, not necessarily of the same
            shape, with valid index and columns labels considering ``subset``.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
            Instance of class with CSS applied to its HTML representation.

        See Also
        --------
        Styler.map_index: Apply a CSS-styling function to headers elementwise.
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.map: Apply a CSS-styling function elementwise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        This is similar to ``DataFrame.apply``, except that ``axis=None``
        applies the function to the entire DataFrame at once,
        rather than column-wise or row-wise.

        Examples
        --------
        >>> def highlight_max(x, color):
        ...     return np.where(x == np.nanmax(x.to_numpy()), f"color: {color};", None)
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.apply(highlight_max, color="red")  # doctest: +SKIP
        >>> df.style.apply(highlight_max, color="blue", axis=1)  # doctest: +SKIP
        >>> df.style.apply(highlight_max, color="green", axis=None)  # doctest: +SKIP

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.apply(highlight_max, color="red", subset="A")
        ... # doctest: +SKIP
        >>> df.style.apply(highlight_max, color="red", subset=["A", "B"])
        ... # doctest: +SKIP

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.apply(highlight_max, color="red", subset=([0, 1, 2], slice(None)))
        ... # doctest: +SKIP
        >>> df.style.apply(highlight_max, color="red", subset=(slice(0, 5, 2), "A"))
        ... # doctest: +SKIP

        Using a function which returns a Series / DataFrame of unequal length but
        containing valid index labels

        >>> df = pd.DataFrame([[1, 2], [3, 4], [4, 6]], index=["A1", "A2", "Total"])
        >>> total_style = pd.Series("font-weight: bold;", index=["Total"])
        >>> df.style.apply(lambda s: total_style)  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        self._todo.append(
            (lambda instance: instance._apply, (func, axis, subset), kwargs)
        )
        return self

    def _apply_index(
        self,
        func: Callable,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        method: str = "apply",
        **kwargs,
    ) -> Styler:
        # [FUNCTION DOC]
        # HELPER: runs an index or header style function immediately. Parameters: func is
        # the function; axis is 0 (index) or 1 (columns); level selects levels (default none:
        # all); method is apply or map selecting how func is applied to the label
        # frame; extra keyword arguments go to func. Builds a frame of the labels for the
        # selected levels, applies func column-wise or elementwise accordingly, and stores the
        # CSS in the header style context. Returns this styler.
        # KNOWN ISSUE: if method is neither apply nor map, no result is assigned and the following statement fails with an unbound-variable error; only the two internal callers exist.
        axis = self.data._get_axis_number(axis)
        obj = self.index if axis == 0 else self.columns

        levels_ = refactor_levels(level, obj)
        data = DataFrame(obj.to_list()).loc[:, levels_]

        if method == "apply":
            result = data.apply(func, axis=0, **kwargs)
        elif method == "map":
            result = data.map(func, **kwargs)

        self._update_ctx_header(result, axis)
        return self

    def apply_index(
        self,
        func: Callable,
        axis: AxisInt | str = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
    ) -> Styler:
        # [FUNCTION DOC]
        # Queues a style function for index or column header labels (public), applied
        # column-wise on each selected level. Parameters: func, axis (0 index, 1 columns;
        # default 0), level (default all), extra keyword arguments. Appends a deferred call
        # and returns this styler.
        """
        Apply a CSS-styling function to the index or column headers, level-wise.

        Updates the HTML representation with the result.

        .. versionadded:: 2.1.0
           Styler.applymap_index was deprecated and renamed to Styler.map_index.

        Parameters
        ----------
        func : function
            ``func`` should take a Series and return a string array of the same length.
        axis : {0, 1, "index", "columns"}
            The headers over which to apply the function.
        level : int, str, list, optional
            If index is MultiIndex the level(s) over which to apply the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
            Instance of class with CSS applied to its HTML representation.

        See Also
        --------
        Styler.map_index: Apply a CSS-styling function to headers elementwise.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.
        Styler.map: Apply a CSS-styling function elementwise.

        Notes
        -----
        Each input to ``func`` will be the index as a Series, if an Index, or a level
        of a MultiIndex. The output of ``func`` should be an identically sized array
        of CSS styles as strings, in the format
        'attribute: value; attribute2: value2; ...' or, if nothing is to be applied
        to that element, an empty string or ``None``.

        Examples
        --------
        Basic usage to conditionally highlight values in the index.

        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"])
        >>> def color_b(label):
        ...     return np.where(label == "B", "background-color: yellow;", "")
        >>> df.style.apply_index(color_b)  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead1.png

        Selectively applying to specific levels of MultiIndex columns.

        >>> midx = pd.MultiIndex.from_product([["ix", "jy"], [0, 1], ["x3", "z4"]])
        >>> df = pd.DataFrame([np.arange(8)], columns=midx)
        >>> def highlight_x(label):
        ...     return ["background-color: yellow;" if "x" in v else "" for v in label]
        >>> df.style.apply_index(
        ...     highlight_x, axis="columns", level=[0, 2]
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead2.png
        """
        self._todo.append(
            (
                lambda instance: instance._apply_index,
                (func, axis, level, "apply"),
                kwargs,
            )
        )
        return self

    def map_index(
        self,
        func: Callable,
        axis: AxisInt | str = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
    ) -> Styler:
        # [FUNCTION DOC]
        # Queues an elementwise style function for index or column header labels
        # (public). Parameters: func, axis (default 0), level (default all), extra keyword
        # arguments. Appends a deferred call and returns this styler.
        """
        Apply a CSS-styling function to the index or column headers, elementwise.

        Updates the HTML representation with the result.

        .. versionadded:: 2.1.0
           Styler.applymap_index was deprecated and renamed to Styler.map_index.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        axis : {0, 1, "index", "columns"}
            The headers over which to apply the function.
        level : int, str, list, optional
            If index is MultiIndex the level(s) over which to apply the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
            Instance of class with CSS applied to its HTML representation.

        See Also
        --------
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.
        Styler.map: Apply a CSS-styling function elementwise.

        Notes
        -----
        Each input to ``func`` will be an index value, if an Index, or a level value of
        a MultiIndex. The output of ``func`` should be CSS styles as a string, in the
        format 'attribute: value; attribute2: value2; ...' or, if nothing is to be
        applied to that element, an empty string or ``None``.

        Examples
        --------
        Basic usage to conditionally highlight values in the index.

        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"])
        >>> def color_b(label):
        ...     return "background-color: yellow;" if label == "B" else None
        >>> df.style.map_index(color_b)  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead1.png

        Selectively applying to specific levels of MultiIndex columns.

        >>> midx = pd.MultiIndex.from_product([["ix", "jy"], [0, 1], ["x3", "z4"]])
        >>> df = pd.DataFrame([np.arange(8)], columns=midx)
        >>> def highlight_x(label):
        ...     return "background-color: yellow;" if "x" in label else None
        >>> df.style.map_index(
        ...     highlight_x, axis="columns", level=[0, 2]
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead2.png
        """
        self._todo.append(
            (
                lambda instance: instance._apply_index,
                (func, axis, level, "map"),
                kwargs,
            )
        )
        return self

    def _map(self, func: Callable, subset: Subset | None = None, **kwargs) -> Styler:
        # [FUNCTION DOC]
        # HELPER: runs an elementwise style function immediately (deferred target of the
        # map method). Binds extra keyword arguments to func, selects the subset (default
        # all cells, normalized to a frame selection), maps the function over every cell,
        # and stores the resulting CSS. Returns this styler.
        func = partial(func, **kwargs)  # map doesn't take kwargs?
        if subset is None:
            subset = IndexSlice[:]
        subset = non_reducing_slice(subset)
        result = self.data.loc[subset].map(func)
        self._update_ctx(result)
        return self

    def map(self, func: Callable, subset: Subset | None = None, **kwargs) -> Styler:
        # [FUNCTION DOC]
        # Queues an elementwise style function (public). func takes a scalar and returns a
        # CSS string; subset limits cells (default all); extra keyword arguments are
        # passed to func. Appends a deferred call to the private map routine and returns
        # this styler.
        """
        Apply a CSS-styling function elementwise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
            Instance of class with CSS-styling function applied elementwise.

        See Also
        --------
        Styler.map_index: Apply a CSS-styling function to headers elementwise.
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        Examples
        --------
        >>> def color_negative(v, color):
        ...     return f"color: {color};" if v < 0 else None
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.map(color_negative, color="red")  # doctest: +SKIP

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.map(color_negative, color="red", subset="A")
        ... # doctest: +SKIP
        >>> df.style.map(color_negative, color="red", subset=["A", "B"])
        ... # doctest: +SKIP

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.map(
        ...     color_negative, color="red", subset=([0, 1, 2], slice(None))
        ... )  # doctest: +SKIP
        >>> df.style.map(color_negative, color="red", subset=(slice(0, 5, 2), "A"))
        ... # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        self._todo.append((lambda instance: instance._map, (func, subset), kwargs))
        return self

    def set_table_attributes(self, attributes: str) -> Styler:
        # [FUNCTION DOC]
        # Sets the extra attributes string for the opening table tag (replacing any previous
        # value). Returns this styler.
        """
        Set the table attributes added to the ``<table>`` HTML element.

        These are items in addition to automatic (by default) ``id`` attribute.

        Parameters
        ----------
        attributes : str
            Table attributes to be added to the ``<table>`` HTML element.

        Returns
        -------
        Styler
            Instance of class with specified table attributes set.

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_table_attributes('class="pure-table"')  # doctest: +SKIP
        # ... <table class="pure-table"> ...
        """
        self.table_attributes = attributes
        return self

    def export(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns a dictionary of the reusable settings: copies of the deferred style
        # functions, table attributes, table styles and css class names, plus booleans for
        # whether the entire index and columns are hidden and the hide-names flags. Data,
        # caption, uuid, formatting and tooltips are not exported.
        """
        Export the styles applied to the current Styler.

        Can be applied to a second Styler with ``Styler.use``.

        Returns
        -------
        dict
            Contains data-independent (exportable) styles applied to current Styler.

        See Also
        --------
        Styler.use: Set the styles on the current Styler.
        Styler.copy: Create a copy of the current Styler.

        Notes
        -----
        This method is designed to copy non-data dependent attributes of
        one Styler to another. It differs from ``Styler.copy`` where data and
        data dependent attributes are also copied.

        The following items are exported since they are not generally data dependent:

          - Styling functions added by the ``apply`` and ``map``
          - Whether axes and names are hidden from the display, if unambiguous.
          - Table attributes
          - Table styles

        The following attributes are considered data dependent and therefore not
        exported:

          - Caption
          - UUID
          - Tooltips
          - Any hidden rows or columns identified by Index labels
          - Any formatting applied using ``Styler.format``
          - Any CSS classes added using ``Styler.set_td_classes``

        Examples
        --------

        >>> styler = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> styler2 = pd.DataFrame([[9, 9, 9]]).style
        >>> styler.hide(axis=0).highlight_max(axis=1)  # doctest: +SKIP
        >>> export = styler.export()
        >>> styler2.use(export)  # doctest: +SKIP
        """
        return {
            "apply": copy.copy(self._todo),
            "table_attributes": self.table_attributes,
            "table_styles": copy.copy(self.table_styles),
            "hide_index": all(self.hide_index_),
            "hide_columns": all(self.hide_columns_),
            "hide_index_names": self.hide_index_names,
            "hide_column_names": self.hide_column_names,
            "css": copy.copy(self.css),
        }

    def use(self, styles: dict[str, Any]) -> Styler:
        # [FUNCTION DOC]
        # Applies an exported settings dictionary to this styler (appending, not
        # replacing, where lists are involved). Adds the deferred functions; combines table
        # attributes by concatenation with a space; adds table styles without overwriting;
        # sets hide flags for index and columns (a boolean is expanded to all levels);
        # sets the hide-names flags (defaulting to false when absent, i.e. resetting them);
        # replaces css class names if present. Returns this styler.
        """
        Set the styles on the current Styler.

        Possibly uses styles from ``Styler.export``.

        Parameters
        ----------
        styles : dict(str, Any)
            List of attributes to add to Styler. Dict keys should contain only:
              - "apply": list of styler functions, typically added with ``apply`` or
                ``map``.
              - "table_attributes": HTML attributes, typically added with
                ``set_table_attributes``.
              - "table_styles": CSS selectors and properties, typically added with
                ``set_table_styles``.
              - "hide_index":  whether the index is hidden, typically added with
                ``hide_index``, or a boolean list for hidden levels.
              - "hide_columns": whether column headers are hidden, typically added with
                ``hide_columns``, or a boolean list for hidden levels.
              - "hide_index_names": whether index names are hidden.
              - "hide_column_names": whether column header names are hidden.
              - "css": the css class names used.

        Returns
        -------
        Styler
            Instance of class with defined styler attributes added.

        See Also
        --------
        Styler.export : Export the non data dependent attributes to the current Styler.

        Examples
        --------

        >>> styler = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> styler2 = pd.DataFrame([[9, 9, 9]]).style
        >>> styler.hide(axis=0).highlight_max(axis=1)  # doctest: +SKIP
        >>> export = styler.export()
        >>> styler2.use(export)  # doctest: +SKIP
        """
        self._todo.extend(styles.get("apply", []))
        table_attributes: str = self.table_attributes or ""
        obj_table_atts: str = (
            ""
            if styles.get("table_attributes") is None
            else str(styles.get("table_attributes"))
        )
        self.set_table_attributes((table_attributes + " " + obj_table_atts).strip())
        if styles.get("table_styles"):
            self.set_table_styles(styles.get("table_styles"), overwrite=False)

        for obj in ["index", "columns"]:
            hide_obj = styles.get("hide_" + obj)
            if hide_obj is not None:
                if isinstance(hide_obj, bool):
                    n = getattr(self, obj).nlevels
                    setattr(self, "hide_" + obj + "_", [hide_obj] * n)
                else:
                    setattr(self, "hide_" + obj + "_", hide_obj)

        self.hide_index_names = styles.get("hide_index_names", False)
        self.hide_column_names = styles.get("hide_column_names", False)
        if styles.get("css"):
            self.css = styles.get("css")  # type: ignore[assignment]
        return self

    def set_uuid(self, uuid: str) -> Styler:
        # [FUNCTION DOC]
        # Sets the table identifier used in cell ids and CSS selectors. Returns this styler.
        """
        Set the uuid applied to ``id`` attributes of HTML elements.

        Parameters
        ----------
        uuid : str
            The uuid to be applied to ``id`` attributes of HTML elements.

        Returns
        -------
        Styler
            Instance of class with specified uuid for `id` attributes set.

        See Also
        --------
        Styler.set_caption : Set the text added to a ``<caption>`` HTML element.
        Styler.set_td_classes : Set the ``class`` attribute of ``<td>`` HTML elements.
        Styler.set_tooltips : Set the DataFrame of strings on ``Styler`` generating
            ``:hover`` tooltips.

        Notes
        -----
        Almost all HTML elements within the table, and including the ``<table>`` element
        are assigned ``id`` attributes. The format is ``T_uuid_<extra>`` where
        ``<extra>`` is typically a more specific identifier, such as ``row1_col2``.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["c1", "c2"])

        You can get the `id` attributes with the following:

        >>> print((df).style.to_html())  # doctest: +SKIP

        To add a title to column `c1`, its `id` is T_20a7d_level0_col0:

        >>> df.style.set_uuid("T_20a7d_level0_col0").set_caption("Test")
        ... # doctest: +SKIP

        Please see:
        `Table visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        self.uuid = uuid
        return self

    def set_caption(self, caption: str | tuple | list) -> Styler:
        # [FUNCTION DOC]
        # Sets the caption. caption must be a string or a two-item tuple/list of strings
        # (short and long caption for LaTeX); anything else raises a value error.
        # Returns this styler.
        """
        Set the text added to a ``<caption>`` HTML element.

        Parameters
        ----------
        caption : str, tuple, list
            For HTML output either the string input is used or the first element of the
            tuple. For LaTeX the string input provides a caption and the additional
            tuple input allows for full captions and short captions, in that order.

        Returns
        -------
        Styler
            Instance of class with text set for ``<caption>`` HTML element.

        See Also
        --------
        Styler.set_td_classes : Set the ``class`` attribute of ``<td>`` HTML elements.
        Styler.set_tooltips : Set the DataFrame of strings on ``Styler`` generating
            ``:hover`` tooltips.
        Styler.set_uuid : Set the uuid applied to ``id`` attributes of HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.set_caption("test")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        msg = "`caption` must be either a string or 2-tuple of strings."
        if isinstance(caption, (list, tuple)):
            if (
                len(caption) != 2
                or not isinstance(caption[0], str)
                or not isinstance(caption[1], str)
            ):
                raise ValueError(msg)
        elif not isinstance(caption, str):
            raise ValueError(msg)
        self.caption = caption
        return self

    def set_sticky(
        self,
        axis: Axis = 0,
        pixel_size: int | None = None,
        levels: Level | list[Level] | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Makes index or column headers stick while scrolling by adding table styles
        # (without overwriting existing ones). Parameters: axis (0 index, 1 columns;
        # default 0); pixel_size is the width/height of each sticky header cell used
        # for offsets of multi-level labels (default 75 for index, 25 for columns); levels
        # selects which levels stick for multi-level labels (default all; value error
        # from level lookup for invalid values). Single-level and multi-level labels use different
        # CSS rules; when stickying columns and the index has names, the names row is also
        # made sticky. Returns this styler.
        """
        Add CSS to permanently display the index or column headers in a scrolling frame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to make the index or column headers sticky.
        pixel_size : int, optional
            Required to configure the width of index cells or the height of column
            header cells when sticking a MultiIndex (or with a named Index).
            Defaults to 75 and 25 respectively.
        levels : int, str, list, optional
            If ``axis`` is a MultiIndex the specific levels to stick. If ``None`` will
            stick all levels.

        Returns
        -------
        Styler
            Instance of class with CSS set for permanently displaying headers
                in scrolling frame.

        See Also
        --------
        Styler.set_properties : Set defined CSS-properties to each ``<td>``
            HTML element for the given subset.

        Notes
        -----
        This method uses the CSS 'position: sticky;' property to display. It is
        designed to work with visible axes, therefore both:

          - `styler.set_sticky(axis="index").hide(axis="index")`
          - `styler.set_sticky(axis="columns").hide(axis="columns")`

        may produce strange behaviour due to CSS controls with missing elements.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.set_sticky(axis="index")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        axis = self.data._get_axis_number(axis)
        obj = self.data.index if axis == 0 else self.data.columns
        pixel_size = (75 if axis == 0 else 25) if not pixel_size else pixel_size

        props = "position:sticky; background-color:inherit;"
        if not isinstance(obj, pd.MultiIndex):
            # handling MultiIndexes requires different CSS

            if axis == 1:
                # stick the first <tr> of <head> and, if index names, the second <tr>
                # if self._hide_columns then no <thead><tr> here will exist: no conflict
                styles: CSSStyles = [
                    {
                        "selector": "thead tr:nth-child(1) th",
                        "props": props + "top:0px; z-index:2;",
                    }
                ]
                if self.index.names[0] is not None:
                    styles[0]["props"] = (
                        props + f"top:0px; z-index:2; height:{pixel_size}px;"
                    )
                    styles.append(
                        {
                            "selector": "thead tr:nth-child(2) th",
                            "props": props
                            + f"top:{pixel_size}px; z-index:2; height:{pixel_size}px; ",
                        }
                    )
            else:
                # stick the first <th> of each <tr> in both <thead> and <tbody>
                # if self._hide_index then no <th> will exist in <tbody>: no conflict
                # but <th> will exist in <thead>: conflict with initial element
                styles = [
                    {
                        "selector": "thead tr th:nth-child(1)",
                        "props": props + "left:0px; z-index:3 !important;",
                    },
                    {
                        "selector": "tbody tr th:nth-child(1)",
                        "props": props + "left:0px; z-index:1;",
                    },
                ]

        else:
            # handle the MultiIndex case
            range_idx = list(range(obj.nlevels))
            levels_: list[int] = refactor_levels(levels, obj) if levels else range_idx
            levels_ = sorted(levels_)

            if axis == 1:
                styles = []
                for i, level in enumerate(levels_):
                    styles.append(
                        {
                            "selector": f"thead tr:nth-child({level + 1}) th",
                            "props": props
                            + (
                                f"top:{i * pixel_size}px; height:{pixel_size}px; "
                                "z-index:2;"
                            ),
                        }
                    )
                if not all(name is None for name in self.index.names):
                    styles.append(
                        {
                            "selector": f"thead tr:nth-child({obj.nlevels + 1}) th",
                            "props": props
                            + (
                                f"top:{(len(levels_)) * pixel_size}px; "
                                f"height:{pixel_size}px; z-index:2;"
                            ),
                        }
                    )

            else:
                styles = []
                for i, level in enumerate(levels_):
                    props_ = props + (
                        f"left:{i * pixel_size}px; "
                        f"min-width:{pixel_size}px; "
                        f"max-width:{pixel_size}px; "
                    )
                    styles.extend(
                        [
                            {
                                "selector": f"thead tr th:nth-child({level + 1})",
                                "props": props_ + "z-index:3 !important;",
                            },
                            {
                                "selector": f"tbody tr th.level{level}",
                                "props": props_ + "z-index:1;",
                            },
                        ]
                    )

        return self.set_table_styles(styles, overwrite=False)

    def set_table_styles(
        self,
        table_styles: dict[Any, CSSStyles] | CSSStyles | None = None,
        axis: AxisInt = 0,
        overwrite: bool = True,
        css_class_names: dict[str, str] | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Sets CSS table styles. Parameters: table_styles is none (no change), a list of
        # rule dictionaries with a selector and properties, or a dictionary mapping a
        # label to such a list (rules apply to that row (axis 1) or column (axis 0)
        # positionally via class selectors); properties may be CSS text or pairs and are
        # converted to pairs; comma-separated selectors in dictionary mode are split.
        # axis (default 0) applies to the dictionary form; overwrite (default true) replaces
        # the existing styles, otherwise appends; css_class_names optionally overrides the
        # css class names used in rendering (merged into the existing mapping, even if
        # table_styles is none). Returns this styler.
        """
        Set the table styles included within the ``<style>`` HTML element.

        This function can be used to style the entire table, columns, rows or
        specific HTML selectors.

        Parameters
        ----------
        table_styles : list or dict
            If supplying a list, each individual table_style should be a
            dictionary with ``selector`` and ``props`` keys. ``selector``
            should be a CSS selector that the style will be applied to
            (automatically prefixed by the table's UUID) and ``props``
            should be a list of tuples with ``(attribute, value)``.
            If supplying a dict, the dict keys should correspond to
            column names or index values, depending upon the specified
            `axis` argument. These will be mapped to row or col CSS
            selectors. MultiIndex values as dict keys should be
            in their respective tuple form. The dict values should be
            a list as specified in the form with CSS selectors and
            props that will be applied to the specified row or column.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``). Only used if `table_styles` is
            dict.
        overwrite : bool, default True
            Styles are replaced if `True`, or extended if `False`. CSS
            rules are preserved so most recent styles set will dominate
            if selectors intersect.
        css_class_names : dict, optional
            A dict of strings used to replace the default CSS classes described below.

        Returns
        -------
        Styler
            Instance of class with specified table styles set.

        See Also
        --------
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.
        Styler.set_table_attributes: Set the table attributes added to the ``<table>``
            HTML element.

        Notes
        -----
        The default CSS classes dict, whose values can be replaced is as follows:

        .. code-block:: python

            css_class_names = {
                "row_heading": "row_heading",
                "col_heading": "col_heading",
                "index_name": "index_name",
                "col": "col",
                "row": "row",
                "col_trim": "col_trim",
                "row_trim": "row_trim",
                "level": "level",
                "data": "data",
                "blank": "blank",
                "foot": "foot",
            }

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
        >>> df.style.set_table_styles(
        ...     [{"selector": "tr:hover", "props": [("background-color", "yellow")]}]
        ... )  # doctest: +SKIP

        Or with CSS strings

        >>> df.style.set_table_styles(
        ...     [
        ...         {
        ...             "selector": "tr:hover",
        ...             "props": "background-color: yellow; font-size: 1em;",
        ...         }
        ...     ]
        ... )  # doctest: +SKIP

        Adding column styling by name

        >>> df.style.set_table_styles(
        ...     {
        ...         "A": [{"selector": "", "props": [("color", "red")]}],
        ...         "B": [{"selector": "td", "props": "color: blue;"}],
        ...     },
        ...     overwrite=False,
        ... )  # doctest: +SKIP

        Adding row styling

        >>> df.style.set_table_styles(
        ...     {0: [{"selector": "td:hover", "props": [("font-size", "25px")]}]},
        ...     axis=1,
        ...     overwrite=False,
        ... )  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        if css_class_names is not None:
            self.css = {**self.css, **css_class_names}

        if table_styles is None:
            return self
        elif isinstance(table_styles, dict):
            axis = self.data._get_axis_number(axis)
            obj = self.data.index if axis == 1 else self.data.columns
            idf = f".{self.css['row']}" if axis == 1 else f".{self.css['col']}"

            table_styles = [
                {
                    "selector": str(s["selector"]) + idf + str(idx),
                    "props": maybe_convert_css_to_tuples(s["props"]),
                }
                for key, styles in table_styles.items()
                for idx in obj.get_indexer_for([key])
                for s in format_table_styles(styles)
            ]
        else:
            table_styles = [
                {
                    "selector": s["selector"],
                    "props": maybe_convert_css_to_tuples(s["props"]),
                }
                for s in table_styles
            ]

        if not overwrite and self.table_styles is not None:
            self.table_styles.extend(table_styles)
        else:
            self.table_styles = table_styles
        return self

    def hide(
        self,
        subset: Subset | None = None,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        names: bool = False,
    ) -> Styler:
        # [FUNCTION DOC]
        # Hides index or columns (or their names) from the rendered output. Parameters:
        # subset limits the labels to hide (default none); axis 0 for index/rows or 1 for
        # columns (default 0); level selects hidden levels (default none); names hides the
        # level names (default false). Raises a value error if both subset and level are
        # given. With neither subset nor level and names true, only the names are hidden.
        # With a subset, replaces the hidden row/column positions with the positions of the
        # selected labels (replacing, not adding to, earlier ones). Otherwise the per-level
        # hide flags are set so only the selected (or all) levels are hidden (replacing
        # previous flags). Returns this styler.
        """
        Hide the entire index / column headers, or specific rows / columns from display.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 1d input or single key along the axis within
            `DataFrame.loc[<subset>, :]` or `DataFrame.loc[:, <subset>]` depending
            upon ``axis``, to limit ``data`` to select hidden rows / columns.
        axis : {"index", 0, "columns", 1}
            Apply to the index or columns.
        level : int, str, list
            The level(s) to hide in a MultiIndex if hiding the entire index / column
            headers. Cannot be used simultaneously with ``subset``.
        names : bool
            Whether to hide the level name(s) of the index / columns headers in the case
            it (or at least one the levels) remains visible.

        Returns
        -------
        Styler
            Instance of class with specified headers/rows/columns hidden from display.

        See Also
        --------
        Styler.apply : Apply a CSS-styling function column-wise, row-wise,
            or table-wise.
        Styler.map : Apply a CSS-styling function elementwise.

        Notes
        -----
        .. warning::
           This method only works with the output methods ``to_html``, ``to_string``
           and ``to_latex``.

           Other output methods, including ``to_excel``, ignore this hiding method
           and will display all data.

        This method has multiple functionality depending upon the combination
        of the ``subset``, ``level`` and ``names`` arguments (see examples). The
        ``axis`` argument is used only to control whether the method is applied to row
        or column headers:

        .. list-table:: Argument combinations
           :widths: 10 20 10 60
           :header-rows: 1

           * - ``subset``
             - ``level``
             - ``names``
             - Effect
           * - None
             - None
             - False
             - The axis-Index is hidden entirely.
           * - None
             - None
             - True
             - Only the axis-Index names are hidden.
           * - None
             - Int, Str, List
             - False
             - Specified axis-MultiIndex levels are hidden entirely.
           * - None
             - Int, Str, List
             - True
             - Specified axis-MultiIndex levels are hidden entirely and the names of
               remaining axis-MultiIndex levels.
           * - Subset
             - None
             - False
             - The specified data rows/columns are hidden, but the axis-Index itself,
               and names, remain unchanged.
           * - Subset
             - None
             - True
             - The specified data rows/columns and axis-Index names are hidden, but
               the axis-Index itself remains unchanged.
           * - Subset
             - Int, Str, List
             - Boolean
             - ValueError: cannot supply ``subset`` and ``level`` simultaneously.

        Note this method only hides the identified elements so can be chained to hide
        multiple elements in sequence.

        Examples
        --------
        Simple application hiding specific rows:

        >>> df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=["a", "b", "c"])
        >>> df.style.hide(["a", "b"])  # doctest: +SKIP
             0    1
        c    5    6

        Hide the index and retain the data values:

        >>> midx = pd.MultiIndex.from_product([["x", "y"], ["a", "b", "c"]])
        >>> df = pd.DataFrame(np.random.randn(6, 6), index=midx, columns=midx)
        >>> df.style.format("{:.1f}").hide()  # doctest: +SKIP
                         x                    y
           a      b      c      a      b      c
         0.1    0.0    0.4    1.3    0.6   -1.4
         0.7    1.0    1.3    1.5   -0.0   -0.2
         1.4   -0.8    1.6   -0.2   -0.4   -0.3
         0.4    1.0   -0.2   -0.8   -1.2    1.1
        -0.6    1.2    1.8    1.9    0.3    0.3
         0.8    0.5   -0.3    1.2    2.2   -0.8

        Hide specific rows in a MultiIndex but retain the index:

        >>> df.style.format("{:.1f}").hide(subset=(slice(None), ["a", "c"]))
        ... # doctest: +SKIP
                                 x                    y
                   a      b      c      a      b      c
        x   b    0.7    1.0    1.3    1.5   -0.0   -0.2
        y   b   -0.6    1.2    1.8    1.9    0.3    0.3

        Hide specific rows and the index through chaining:

        >>> df.style.format("{:.1f}").hide(subset=(slice(None), ["a", "c"])).hide()
        ... # doctest: +SKIP
                         x                    y
           a      b      c      a      b      c
         0.7    1.0    1.3    1.5   -0.0   -0.2
        -0.6    1.2    1.8    1.9    0.3    0.3

        Hide a specific level:

        >>> df.style.format("{:,.1f}").hide(level=1)  # doctest: +SKIP
                             x                    y
               a      b      c      a      b      c
        x    0.1    0.0    0.4    1.3    0.6   -1.4
             0.7    1.0    1.3    1.5   -0.0   -0.2
             1.4   -0.8    1.6   -0.2   -0.4   -0.3
        y    0.4    1.0   -0.2   -0.8   -1.2    1.1
            -0.6    1.2    1.8    1.9    0.3    0.3
             0.8    0.5   -0.3    1.2    2.2   -0.8

        Hiding just the index level names:

        >>> df.index.names = ["lev0", "lev1"]
        >>> df.style.format("{:,.1f}").hide(names=True)  # doctest: +SKIP
                                 x                    y
                   a      b      c      a      b      c
        x   a    0.1    0.0    0.4    1.3    0.6   -1.4
            b    0.7    1.0    1.3    1.5   -0.0   -0.2
            c    1.4   -0.8    1.6   -0.2   -0.4   -0.3
        y   a    0.4    1.0   -0.2   -0.8   -1.2    1.1
            b   -0.6    1.2    1.8    1.9    0.3    0.3
            c    0.8    0.5   -0.3    1.2    2.2   -0.8

        Examples all produce equivalently transposed effects with ``axis="columns"``.
        """
        axis = self.data._get_axis_number(axis)
        if axis == 0:
            obj, objs, alt = "index", "index", "rows"
        else:
            obj, objs, alt = "column", "columns", "columns"

        if level is not None and subset is not None:
            raise ValueError("`subset` and `level` cannot be passed simultaneously")

        if subset is None:
            if level is None and names:
                # this combination implies user shows the index and hides just names
                setattr(self, f"hide_{obj}_names", True)
                return self

            levels_ = refactor_levels(level, getattr(self, objs))
            setattr(
                self,
                f"hide_{objs}_",
                [lev in levels_ for lev in range(getattr(self, objs).nlevels)],
            )
        else:
            if axis == 0:
                subset_ = IndexSlice[subset, :]  # new var so mypy reads not Optional
            else:
                subset_ = IndexSlice[:, subset]  # new var so mypy reads not Optional
            subset = non_reducing_slice(subset_)
            hide = self.data.loc[subset]
            h_els = getattr(self, objs).get_indexer_for(getattr(hide, objs))
            setattr(self, f"hidden_{alt}", h_els)

        if names:
            setattr(self, f"hide_{obj}_names", True)
        return self

    # -----------------------------------------------------------------------
    # A collection of "builtin" styles
    # -----------------------------------------------------------------------

    def _get_numeric_subset_default(self):
        # [FUNCTION DOC]
        # HELPER: returns the default subset of the built-in styles: a boolean mask over
        # the columns that are numeric (using a mask rather than labels so that boolean
        # labels work).
        # Returns a boolean mask indicating where `self.data` has numerical columns.
        # Choosing a mask as opposed to the column names also works for
        # boolean column labels (GH47838).
        return self.data.columns.isin(self.data.select_dtypes(include=np.number))

    def background_gradient(
        self,
        cmap: str | Colormap = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        text_color_threshold: float = 0.408,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Colors cell backgrounds (and chooses light or dark text) according to a colormap
        # scaled by the values. Parameters: cmap (colormap name or object, default
        # "PuBu"); low and high extend the scale ends as fractions of the range (default
        # 0); axis (0 per column, 1 per row, none for the whole selection; default 0);
        # subset (default none: numeric columns, unless gmap is given and then all);
        # text_color_threshold (luminance cutoff for switching text to light, default 0.408);
        # vmin and vmax override the scale limits; gmap is an alternative array of values
        # to color by (shape must match). Queues the private gradient function through the
        # apply method and returns this styler. Needs matplotlib at render time.
        """
        Color the background in a gradient style.

        The background color is determined according
        to the data in each column, row or frame, or by a given
        gradient map. Requires matplotlib.

        Parameters
        ----------
        cmap : str or colormap
            Matplotlib colormap.
        low : float
            Compress the color range at the low end. This is a multiple of the data
            range to extend below the minimum; good values usually in [0, 1],
            defaults to 0.
        high : float
            Compress the color range at the high end. This is a multiple of the data
            range to extend above the maximum; good values usually in [0, 1],
            defaults to 0.
        axis : {0, 1, "index", "columns", None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        text_color_threshold : float or int
            Luminance threshold for determining text color in [0, 1]. Facilitates text
            visibility across varying background colors. All text is dark if 0, and
            light if 1, defaults to 0.408.
        vmin : float, optional
            Minimum data value that corresponds to colormap minimum value.
            If not specified the minimum value of the data (or gmap) will be used.
        vmax : float, optional
            Maximum data value that corresponds to colormap maximum value.
            If not specified the maximum value of the data (or gmap) will be used.
        gmap : array-like, optional
            Gradient map for determining the background colors. If not supplied
            will use the underlying data from rows, columns or frame. If given as an
            ndarray or list-like must be an identical shape to the underlying data
            considering ``axis`` and ``subset``. If given as DataFrame or Series must
            have same index and column labels considering ``axis`` and ``subset``.
            If supplied, ``vmin`` and ``vmax`` should be given relative to this
            gradient map.

        Returns
        -------
        Styler
            Instance of class with background colored in gradient style.

        See Also
        --------
        Styler.text_gradient: Color the text in a gradient style.

        Notes
        -----
        When using ``low`` and ``high`` the range
        of the gradient, given by the data if ``gmap`` is not given or by ``gmap``,
        is extended at the low end effectively by
        `map.min - low * map.range` and at the high end by
        `map.max + high * map.range` before the colors are normalized and determined.

        If combining with ``vmin`` and ``vmax`` the `map.min`, `map.max` and
        `map.range` are replaced by values according to the values derived from
        ``vmin`` and ``vmax``.

        This method will preselect numeric columns and ignore non-numeric columns
        unless a ``gmap`` is supplied in which case no preselection occurs.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     columns=["City", "Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ...     data=[
        ...         ["Stockholm", 21.6, 5.0, 3.2],
        ...         ["Oslo", 22.4, 13.3, 3.1],
        ...         ["Copenhagen", 24.5, 0.0, 6.7],
        ...     ],
        ... )

        Shading the values column-wise, with ``axis=0``, preselecting numeric columns

        >>> df.style.background_gradient(axis=0)  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_ax0.png

        Shading all values collectively using ``axis=None``

        >>> df.style.background_gradient(axis=None)  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_axNone.png

        Compress the color map from the both ``low`` and ``high`` ends

        >>> df.style.background_gradient(
        ...     axis=None, low=0.75, high=1.0
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_axNone_lowhigh.png

        Manually setting ``vmin`` and ``vmax`` gradient thresholds

        >>> df.style.background_gradient(
        ...     axis=None, vmin=6.7, vmax=21.6
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_axNone_vminvmax.png

        Setting a ``gmap`` and applying to all columns with another ``cmap``

        >>> df.style.background_gradient(axis=0, gmap=df["Temp (c)"], cmap="YlOrRd")
        ... # doctest: +SKIP

        .. figure:: ../../_static/style/bg_gmap.png

        Setting the gradient map for a dataframe (i.e. ``axis=None``), we need to
        explicitly state ``subset`` to match the ``gmap`` shape

        >>> gmap = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        >>> df.style.background_gradient(
        ...     axis=None,
        ...     gmap=gmap,
        ...     cmap="YlOrRd",
        ...     subset=["Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_axNone_gmap.png
        """
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()

        self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            text_color_threshold=text_color_threshold,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
        )
        return self

    def text_gradient(
        self,
        cmap: str | Colormap = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Like the background gradient but colors the text instead of the background (no
        # text color threshold). Parameters: cmap, low, high, axis, subset, vmin, vmax, gmap
        # as for the background gradient. Queues the private gradient function in
        # text-only mode and returns the styler.
        """
        Color the text in a gradient style.

        The text color is determined according
        to the data in each column, row or frame, or by a given
        gradient map. Requires matplotlib.

        Parameters
        ----------
        cmap : str or colormap
            Matplotlib colormap.
        low : float
            Compress the color range at the low end. This is a multiple of the data
            range to extend below the minimum; good values usually in [0, 1],
            defaults to 0.
        high : float
            Compress the color range at the high end. This is a multiple of the data
            range to extend above the maximum; good values usually in [0, 1],
            defaults to 0.
        axis : {0, 1, "index", "columns", None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        vmin : float, optional
            Minimum data value that corresponds to colormap minimum value.
            If not specified the minimum value of the data (or gmap) will be used.
        vmax : float, optional
            Maximum data value that corresponds to colormap maximum value.
            If not specified the maximum value of the data (or gmap) will be used.
        gmap : array-like, optional
            Gradient map for determining the text colors. If not supplied
            will use the underlying data from rows, columns or frame. If given as an
            ndarray or list-like must be an identical shape to the underlying data
            considering ``axis`` and ``subset``. If given as DataFrame or Series must
            have same index and column labels considering ``axis`` and ``subset``.
            If supplied, ``vmin`` and ``vmax`` should be given relative to this
            gradient map.

        Returns
        -------
        Styler
            Instance of class with background colored in gradient style.

        See Also
        --------
        Styler.background_gradient: Color the background in a gradient style.

        Notes
        -----
        When using ``low`` and ``high`` the range
        of the gradient, given by the data if ``gmap`` is not given or by ``gmap``,
        is extended at the low end effectively by
        `map.min - low * map.range` and at the high end by
        `map.max + high * map.range` before the colors are normalized and determined.

        If combining with ``vmin`` and ``vmax`` the `map.min`, `map.max` and
        `map.range` are replaced by values according to the values derived from
        ``vmin`` and ``vmax``.

        This method will preselect numeric columns and ignore non-numeric columns
        unless a ``gmap`` is supplied in which case no preselection occurs.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     columns=["City", "Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ...     data=[
        ...         ["Stockholm", 21.6, 5.0, 3.2],
        ...         ["Oslo", 22.4, 13.3, 3.1],
        ...         ["Copenhagen", 24.5, 0.0, 6.7],
        ...     ],
        ... )

        Shading the values column-wise, with ``axis=0``, preselecting numeric columns

        >>> df.style.text_gradient(axis=0)  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_ax0.png

        Shading all values collectively using ``axis=None``

        >>> df.style.text_gradient(axis=None)  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_axNone.png

        Compress the color map from the both ``low`` and ``high`` ends

        >>> df.style.text_gradient(axis=None, low=0.75, high=1.0)  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_axNone_lowhigh.png

        Manually setting ``vmin`` and ``vmax`` gradient thresholds

        >>> df.style.text_gradient(axis=None, vmin=6.7, vmax=21.6)  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_axNone_vminvmax.png

        Setting a ``gmap`` and applying to all columns with another ``cmap``

        >>> df.style.text_gradient(axis=0, gmap=df["Temp (c)"], cmap="YlOrRd")
        ... # doctest: +SKIP

        .. figure:: ../../_static/style/tg_gmap.png

        Setting the gradient map for a dataframe (i.e. ``axis=None``), we need to
        explicitly state ``subset`` to match the ``gmap`` shape

        >>> gmap = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        >>> df.style.text_gradient(
        ...     axis=None,
        ...     gmap=gmap,
        ...     cmap="YlOrRd",
        ...     subset=["Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_axNone_gmap.png
        """
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()

        return self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
            text_only=True,
        )

    def set_properties(self, subset: Subset | None = None, **kwargs) -> Styler:
        # [FUNCTION DOC]
        # Applies fixed CSS properties to cells. subset limits cells; keyword arguments
        # are CSS property names and values (underscores are not converted, so names must
        # already be valid CSS names). Builds one declaration string and queues an
        # elementwise map returning it for every cell. Returns this styler.
        """
        Set defined CSS-properties to each ``<td>`` HTML element for the given subset.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        **kwargs : dict
            A dictionary of property, value pairs to be set for each cell.

        Returns
        -------
        Styler
            Instance of class with CSS-properties set for each ``<td>`` HTML element
                in the given subset

        See Also
        --------
        Styler.set_sticky : Add CSS to permanently display the index or column
            headers in a scrolling frame.

        Notes
        -----
        This is a convenience methods which wraps the :meth:`Styler.map` calling a
        function returning the CSS-properties independently of the data.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_properties(color="white", align="right")  # doctest: +SKIP
        >>> df.style.set_properties(**{"background-color": "yellow"})  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        values = "".join([f"{p}: {v};" for p, v in kwargs.items()])
        return self.map(lambda x: values, subset=subset)

    def bar(
        self,
        subset: Subset | None = None,
        axis: Axis | None = 0,
        *,
        color: str | list | tuple | None = None,
        cmap: Any | None = None,
        width: float = 100,
        height: float = 100,
        align: str | float | Callable = "mid",
        vmin: float | None = None,
        vmax: float | None = None,
        props: str = "width: 10em;",
    ) -> Styler:
        # [FUNCTION DOC]
        # Draws bars in cells. Parameters: subset (default numeric columns); axis (0, 1 or
        # none; default 0); color (one color string or a list/tuple of two for negative and
        # positive; default red-ish when neither color nor cmap is given); cmap
        # (matplotlib colormap, alternative to color); width and height as percentages in
        # 0..100 (default 100); align (left, right, mid, zero, mean, a number or a callable;
        # default mid); vmin and vmax override the data window; props extra CSS (default a
        # fixed width).
        # Raises a value error when both color and cmap are given, when color has the wrong
        # type or more than two entries, or when width or height lie outside 0..100.
        # Queues the private bar function via the apply method. Returns this styler.
        """
        Draw bar chart in the cell backgrounds.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        color : str or 2-tuple/list
            If a str is passed, the color is the same for both
            negative and positive numbers. If 2-tuple/list is used, the
            first element is the color_negative and the second is the
            color_positive (eg: ['#d65f5f', '#5fba7d']).
        cmap : str, matplotlib.cm.ColorMap
            A string name of a matplotlib Colormap, or a Colormap object. Cannot be
            used together with ``color``.
        width : float, default 100
            The percentage of the cell, measured from the left, in which to draw the
            bars, in [0, 100].
        height : float, default 100
            The percentage height of the bar in the cell, centrally aligned, in [0,100].
        align : str, int, float, callable, default 'mid'
            How to align the bars within the cells relative to a width adjusted center.
            If string must be one of:

            - 'left' : bars are drawn rightwards from the minimum data value.
            - 'right' : bars are drawn leftwards from the maximum data value.
            - 'zero' : a value of zero is located at the center of the cell.
            - 'mid' : a value of (max-min)/2 is located at the center of the cell,
              or if all values are negative (positive) the zero is
              aligned at the right (left) of the cell.
            - 'mean' : the mean value of the data is located at the center of the cell.

            If a float or integer is given this will indicate the center of the cell.

            If a callable should take a 1d or 2d array and return a scalar.

        vmin : float, optional
            Minimum bar value, defining the left hand limit
            of the bar drawing range, lower values are clipped to `vmin`.
            When None (default): the minimum value of the data will be used.
        vmax : float, optional
            Maximum bar value, defining the right hand limit
            of the bar drawing range, higher values are clipped to `vmax`.
            When None (default): the maximum value of the data will be used.
        props : str, optional
            The base CSS of the cell that is extended to add the bar chart. Defaults to
            `"width: 10em;"`.

        Returns
        -------
        Styler
            Contains list-like attribute with bar chart data as formatted CSS.

        See Also
        --------
        PlotAccessor.bar : Vertical bar plot.
        PlotAccessor.line : Plot Series or DataFrame as lines.
        PlotAccessor.pie : Generate a pie plot.

        Notes
        -----
        This section of the user guide:
        `Table Visualization <../../user_guide/style.ipynb>`_ gives
        a number of examples for different settings and color coordination.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
        >>> df.style.bar(subset=["A"], color="gray")  # doctest: +SKIP
        """
        if color is None and cmap is None:
            color = "#d65f5f"
        elif color is not None and cmap is not None:
            raise ValueError("`color` and `cmap` cannot both be given")
        elif color is not None:
            if (isinstance(color, (list, tuple)) and len(color) > 2) or not isinstance(
                color, (str, list, tuple)
            ):
                raise ValueError(
                    "`color` must be string or list or tuple of 2 strings,"
                    "(eg: color=['#d65f5f', '#5fba7d'])"
                )

        if not 0 <= width <= 100:
            raise ValueError(f"`width` must be a value in [0, 100], got {width}")
        if not 0 <= height <= 100:
            raise ValueError(f"`height` must be a value in [0, 100], got {height}")

        if subset is None:
            subset = self._get_numeric_subset_default()

        self.apply(
            _bar,
            subset=subset,
            axis=axis,
            align=align,
            colors=color,
            cmap=cmap,
            width=width / 100,
            height=height / 100,
            vmin=vmin,
            vmax=vmax,
            base_css=props,
        )

        return self

    def highlight_null(
        self,
        color: str = "red",
        subset: Subset | None = None,
        props: str | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Highlights missing values. color (default red) is used for a background
        # declaration unless props (full CSS text) is given; subset limits cells.
        # Queues a whole-selection style marking missing cells and returns this styler.
        """
        Highlight missing values with a style.

        Parameters
        ----------
        color : str, default 'red'
           Background color to use for highlighting.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        props : str, default None
           CSS properties to use for highlighting. If ``props`` is given, ``color``
           is not used.

        Returns
        -------
        Styler
            Instance of class where null values are highlighted with given style.

        See Also
        --------
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, np.nan]})
        >>> df.style.highlight_null(color="yellow")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """

        def f(data: DataFrame, props: str) -> np.ndarray:
            # [FUNCTION DOC]
            # HELPER: style function used by the null highlighter. Returns an array of the same
            # shape as data with the CSS text where the value is missing and an empty string
            # elsewhere.
            return np.where(pd.isna(data).to_numpy(), props, "")

        if props is None:
            props = f"background-color: {color};"
        return self.apply(f, axis=None, subset=subset, props=props)

    def highlight_max(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Highlights the maximum value. Parameters: subset; color (default yellow);
        # axis (0 per column, 1 per row, none for the whole selection; default 0); props
        # (full CSS text overriding the color). Queues the private value-highlight helper
        # with the max operation via the apply method. Returns this styler.
        """
        Highlight the maximum with a style.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        color : str, default 'yellow'
           Background color to use for highlighting.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        props : str, default None
           CSS properties to use for highlighting. If ``props`` is given, ``color``
           is not used.

        Returns
        -------
        Styler
            Instance of class where max value is highlighted in given style.

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [2, 1], "B": [3, 4]})
        >>> df.style.highlight_max(color="yellow")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """

        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            partial(_highlight_value, op="max"),
            axis=axis,
            subset=subset,
            props=props,
        )

    def highlight_min(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Highlights the minimum value; same parameters and behavior as the maximum
        # highlighter but with the min operation.
        """
        Highlight the minimum with a style.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        color : str, default 'yellow'
           Background color to use for highlighting.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        props : str, default None
           CSS properties to use for highlighting. If ``props`` is given, ``color``
           is not used.

        Returns
        -------
        Styler
            Instance of class where min value is highlighted in given style.

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [2, 1], "B": [3, 4]})
        >>> df.style.highlight_min(color="yellow")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """

        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            partial(_highlight_value, op="min"),
            axis=axis,
            subset=subset,
            props=props,
        )

    def highlight_between(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        left: Scalar | Sequence | None = None,
        right: Scalar | Sequence | None = None,
        inclusive: IntervalClosedType = "both",
        props: str | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Highlights values within a range. Parameters: subset; color (default yellow);
        # axis (default 0, none for whole selection); left and right are the bounds
        # (scalars, or sequences/frames matching the data shape; none means unbounded);
        # inclusive is both (default), neither, left or right; props full CSS text.
        # Queues the private range helper through the apply method and returns this styler.
        # Invalid inclusive values raise a value error when the style is computed.
        """
        Highlight a defined range with a style.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        color : str, default 'yellow'
           Background color to use for highlighting.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            If ``left`` or ``right`` given as sequence, axis along which to apply those
            boundaries. See examples.
        left : scalar or datetime-like, or sequence or array-like, default None
            Left bound for defining the range.
        right : scalar or datetime-like, or sequence or array-like, default None
            Right bound for defining the range.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether bounds are closed or open.
        props : str, default None
           CSS properties to use for highlighting. If ``props`` is given, ``color``
           is not used.

        Returns
        -------
        Styler
            Instance of class with range highlighted in given style.

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Notes
        -----
        If ``left`` is ``None`` only the right bound is applied.
        If ``right`` is ``None`` only the left bound is applied. If both are ``None``
        all values are highlighted.

        ``axis`` is only needed if ``left`` or ``right`` are provided as a sequence or
        an array-like object for aligning the shapes. If ``left`` and ``right`` are
        both scalars then all ``axis`` inputs will give the same result.

        This function only works with compatible ``dtypes``. For example a datetime-like
        region can only use equivalent datetime-like ``left`` and ``right`` arguments.
        Use ``subset`` to control regions which have multiple ``dtypes``.

        Examples
        --------
        Basic usage

        >>> df = pd.DataFrame(
        ...     {
        ...         "One": [1.2, 1.6, 1.5],
        ...         "Two": [2.9, 2.1, 2.5],
        ...         "Three": [3.1, 3.2, 3.8],
        ...     }
        ... )
        >>> df.style.highlight_between(left=2.1, right=2.9)  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_basic.png

        Using a range input sequence along an ``axis``, in this case setting a ``left``
        and ``right`` for each column individually

        >>> df.style.highlight_between(
        ...     left=[1.4, 2.4, 3.4], right=[1.6, 2.6, 3.6], axis=1, color="#fffd75"
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_seq.png

        Using ``axis=None`` and providing the ``left`` argument as an array that
        matches the input DataFrame, with a constant ``right``

        >>> df.style.highlight_between(
        ...     left=[[2, 2, 3], [2, 2, 3], [3, 3, 3]],
        ...     right=3.5,
        ...     axis=None,
        ...     color="#fffd75",
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_axNone.png

        Using ``props`` instead of default background coloring

        >>> df.style.highlight_between(
        ...     left=1.5, right=3.5, props="font-weight:bold;color:#e83e8c"
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_props.png
        """
        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            _highlight_between,
            axis=axis,
            subset=subset,
            props=props,
            left=left,
            right=right,
            inclusive=inclusive,
        )

    def highlight_quantile(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        q_left: float = 0.0,
        q_right: float = 1.0,
        interpolation: QuantileInterpolation = "linear",
        inclusive: IntervalClosedType = "both",
        props: str | None = None,
    ) -> Styler:
        # [FUNCTION DOC]
        # Highlights values within a quantile range. Parameters: subset; color (default
        # yellow); axis (0 or 1 or none; default 0); q_left and q_right quantile bounds
        # (defaults 0.0 and 1.0); interpolation (default linear); inclusive (default both);
        # props. Computes the quantiles immediately over the selected subset (flattened when
        # axis is none, else along the axis, non-numeric allowed), then queues the range
        # helper with those bounds applied along the other axis. Returns this styler.
        """
        Highlight values defined by a quantile with a style.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        color : str, default 'yellow'
            Background color to use for highlighting.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis along which to determine and highlight quantiles. If ``None`` quantiles
            are measured over the entire DataFrame. See examples.
        q_left : float, default 0
            Left bound, in [0, q_right), for the target quantile range.
        q_right : float, default 1
            Right bound, in (q_left, 1], for the target quantile range.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Argument passed to ``Series.quantile`` or ``DataFrame.quantile`` for
            quantile estimation.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether quantile bounds are closed or open.
        props : str, default None
            CSS properties to use for highlighting. If ``props`` is given, ``color``
            is not used.

        Returns
        -------
        Styler
            Instance of class where values in quantile highlighted with given style.

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.

        Notes
        -----
        This function does not work with ``str`` dtypes.

        Examples
        --------
        Using ``axis=None`` and apply a quantile to all collective data

        >>> df = pd.DataFrame(np.arange(10).reshape(2, 5) + 1)
        >>> df.style.highlight_quantile(axis=None, q_left=0.8, color="#fffd75")
        ... # doctest: +SKIP

        .. figure:: ../../_static/style/hq_axNone.png

        Or highlight quantiles row-wise or column-wise, in this case by row-wise

        >>> df.style.highlight_quantile(axis=1, q_left=0.8, color="#fffd75")
        ... # doctest: +SKIP

        .. figure:: ../../_static/style/hq_ax1.png

        Use ``props`` instead of default background coloring

        >>> df.style.highlight_quantile(
        ...     axis=None,
        ...     q_left=0.2,
        ...     q_right=0.8,
        ...     props="font-weight:bold;color:#e83e8c",
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/hq_props.png
        """
        subset_ = slice(None) if subset is None else subset
        subset_ = non_reducing_slice(subset_)
        data = self.data.loc[subset_]

        # after quantile is found along axis, e.g. along rows,
        # applying the calculated quantile to alternate axis, e.g. to each column
        quantiles = [q_left, q_right]
        if axis is None:
            q = Series(data.to_numpy().ravel()).quantile(
                q=quantiles, interpolation=interpolation
            )
            axis_apply: int | None = None
        else:
            axis = self.data._get_axis_number(axis)
            q = data.quantile(
                axis=axis, numeric_only=False, q=quantiles, interpolation=interpolation
            )
            axis_apply = 1 - axis

        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            _highlight_between,
            axis=axis_apply,
            subset=subset,
            props=props,
            left=q.iloc[0],
            right=q.iloc[1],
            inclusive=inclusive,
        )

    @classmethod
    def from_custom_template(
        cls,
        searchpath: Sequence[str],
        html_table: str | None = None,
        html_style: str | None = None,
    ) -> type[Styler]:
        # [FUNCTION DOC]
        # Class method that builds a Styler subclass with custom templates. Parameters:
        # searchpath is a directory or list of directories searched (before the built-in
        # templates) for template files; html_table and html_style are optional template
        # file names overriding the table and style templates. Returns the new subclass
        # (not an instance).
        """
        Factory function for creating a subclass of ``Styler``.

        Uses custom templates and Jinja environment.

        Parameters
        ----------
        searchpath : str or list
            Path or paths of directories containing the templates.
        html_table : str
            Name of your custom template to replace the html_table template.
        html_style : str
            Name of your custom template to replace the html_style template.

        Returns
        -------
        MyStyler : subclass of Styler
            Has the correct ``env``, ``template_html``, ``template_html_table`` and
            ``template_html_style`` class attributes set.

        See Also
        --------
        Styler.export : Export the styles applied to the current Styler.
        Styler.use : Set the styles on the current Styler.

        Examples
        --------
        >>> from pandas.io.formats.style import Styler
        >>> EasyStyler = Styler.from_custom_template(
        ...     "path/to/template",
        ...     "template.tpl",
        ... )  # doctest: +SKIP
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> EasyStyler(df)  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(searchpath), cls.loader])

        # mypy doesn't like dynamically-defined classes
        # error: Variable "cls" is not valid as a type
        # error: Invalid base class "cls"
        class MyStyler(cls):  # type: ignore[valid-type,misc]
            # [CLASS DOC]
            # Dynamically created subclass of the invoking class, holding a template
            # environment that searches the custom directories first and, when names were
            # given, overriding the html table and html style templates.
            env = jinja2.Environment(loader=loader)
            if html_table:
                template_html_table = env.get_template(html_table)
            if html_style:
                template_html_style = env.get_template(html_style)

        return MyStyler

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
        # [FUNCTION DOC]
        # Typing overload only: function taking the styler first. See the implementation
        # below.

    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
        # [FUNCTION DOC]
        # Typing overload only: function given as a pair of callable and keyword name for the
        # styler. See the implementation below.

    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # [FUNCTION DOC]
        # Applies a function to this styler for chaining. func is a callable taking the
        # styler first, or a pair of callable and keyword name under which the styler
        # is passed; extra positional and keyword arguments are passed along. Returns
        # whatever the function returns; delegates to the shared pipe helper.
        """
        Apply ``func(self, *args, **kwargs)``, and return the result.

        Parameters
        ----------
        func : function
            Function to apply to the Styler.  Alternatively, a
            ``(callable, keyword)`` tuple where ``keyword`` is a string
            indicating the keyword of ``callable`` that expects the Styler.
        *args : optional
            Arguments passed to `func`.
        **kwargs : optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object :
            The value returned by ``func``.

        See Also
        --------
        DataFrame.pipe : Analogous method for DataFrame.
        Styler.apply : Apply a CSS-styling function column-wise, row-wise, or
            table-wise.

        Notes
        -----
        Like :meth:`DataFrame.pipe`, this method can simplify the
        application of several user-defined functions to a styler.  Instead
        of writing:

        .. code-block:: python

            f(g(df.style.format(precision=3), arg1=a), arg2=b, arg3=c)

        users can write:

        .. code-block:: python

            (df.style.format(precision=3).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c))

        In particular, this allows users to define functions that take a
        styler object, along with other parameters, and return the styler after
        making styling changes (such as calling :meth:`Styler.apply` or
        :meth:`Styler.set_properties`).

        Examples
        --------

        **Common Use**

        A common usage pattern is to pre-define styling operations which
        can be easily applied to a generic styler in a single ``pipe`` call.

        >>> def some_highlights(styler, min_color="red", max_color="blue"):
        ...     styler.highlight_min(color=min_color, axis=None)
        ...     styler.highlight_max(color=max_color, axis=None)
        ...     styler.highlight_null()
        ...     return styler
        >>> df = pd.DataFrame([[1, 2, 3, pd.NA], [pd.NA, 4, 5, 6]], dtype="Int64")
        >>> df.style.pipe(some_highlights, min_color="green")  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_hl.png

        Since the method returns a ``Styler`` object it can be chained with other
        methods as if applying the underlying highlighters directly.

        >>> (
        ...     df.style.format("{:.1f}")
        ...     .pipe(some_highlights, min_color="green")
        ...     .highlight_between(left=2, right=5)
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_hl2.png

        **Advanced Use**

        Sometimes it may be necessary to pre-define styling functions, but in the case
        where those functions rely on the styler, data or context. Since
        ``Styler.use`` and ``Styler.export`` are designed to be non-data dependent,
        they cannot be used for this purpose. Additionally the ``Styler.apply``
        and ``Styler.format`` type methods are not context aware, so a solution
        is to use ``pipe`` to dynamically wrap this functionality.

        Suppose we want to code a generic styling function that highlights the final
        level of a MultiIndex. The number of levels in the Index is dynamic so we
        need the ``Styler`` context to define the level.

        >>> def highlight_last_level(styler):
        ...     return styler.apply_index(
        ...         lambda v: "background-color: pink; color: yellow",
        ...         axis="columns",
        ...         level=styler.columns.nlevels - 1,
        ...     )  # doctest: +SKIP
        >>> df.columns = pd.MultiIndex.from_product([["A", "B"], ["X", "Y"]])
        >>> df.style.pipe(highlight_last_level)  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_applymap.png

        Additionally suppose we want to highlight a column header if there is any
        missing data in that column.
        In this case we need the data object itself to determine the effect on the
        column headers.

        >>> def highlight_header_missing(styler, level):
        ...     def dynamic_highlight(s):
        ...         return np.where(
        ...             styler.data.isna().any(), "background-color: red;", ""
        ...         )
        ...
        ...     return styler.apply_index(dynamic_highlight, axis=1, level=level)
        >>> df.style.pipe(highlight_header_missing, level=1)  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_applydata.png
        """
        return com.pipe(self, func, *args, **kwargs)


def _validate_apply_axis_arg(
    arg: NDFrame | Sequence | np.ndarray,
    arg_name: str,
    dtype: Any | None,
    data: NDFrame,
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: validates and converts a user-supplied argument (bounds or gradient map)
    # for the built-in styles. Parameters: arg is a sequence, series or frame; arg_name
    # is used in error messages; dtype optionally forces the array dtype; data is the
    # selection being styled. Raises a value error if a series is given for frame data
    # or a frame for series data. Series/frames are aligned to data by reindexing; other
    # inputs are converted to an array and must have exactly the shape of data
    # (value error otherwise). Returns the array.
    """
    For the apply-type methods, ``axis=None`` creates ``data`` as DataFrame, and for
    ``axis=[1,0]`` it creates a Series. Where ``arg`` is expected as an element
    of some operator with ``data`` we must make sure that the two are compatible shapes,
    or raise.

    Parameters
    ----------
    arg : sequence, Series or DataFrame
        the user input arg
    arg_name : string
        name of the arg for use in error messages
    dtype : numpy dtype, optional
        forced numpy dtype if given
    data : Series or DataFrame
        underling subset of Styler data on which operations are performed

    Returns
    -------
    ndarray
    """
    dtype = {"dtype": dtype} if dtype else {}
    # raise if input is wrong for axis:
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(
            f"'{arg_name}' is a Series but underlying data for operations "
            f"is a DataFrame since 'axis=None'"
        )
    if isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(
            f"'{arg_name}' is a DataFrame but underlying data for "
            f"operations is a Series with 'axis in [0,1]'"
        )
    if isinstance(arg, (Series, DataFrame)):  # align indx / cols to data
        arg = arg.reindex_like(data).to_numpy(**dtype)
    else:
        arg = np.asarray(arg, **dtype)
        assert isinstance(arg, np.ndarray)  # mypy requirement
        if arg.shape != data.shape:  # check valid input
            raise ValueError(
                f"supplied '{arg_name}' is not correct shape for data over "
                f"selected 'axis': got {arg.shape}, "
                f"expected {data.shape}"
            )
    return arg


def _background_gradient(
    data,
    cmap: str | Colormap = "PuBu",
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: float | None = None,
    vmax: float | None = None,
    gmap: Sequence | np.ndarray | DataFrame | Series | None = None,
    text_only: bool = False,
) -> list[str] | DataFrame:
    # [FUNCTION DOC]
    # HELPER: computes CSS for the gradient styles. Parameters: data (series or frame
    # selection); cmap (colormap name or object; none uses matplotlib's default);
    # low and high widen the color range (defaults 0); text_color_threshold (default
    # 0.408) is the luminance below which text turns light; vmin and vmax override
    # limits; gmap an alternative array for color mapping (validated against data;
    # default: the data converted to floating point); text_only colors the text only
    # (default false). Requires matplotlib (import error otherwise). Normalizes the
    # values, maps them to colors and returns a list of CSS strings for one-dimensional data or
    # a frame of CSS strings.
    """
    Color background in a range according to the data or a gradient map
    """
    if gmap is None:  # the data is used the gmap
        gmap = data.to_numpy(dtype=float, na_value=np.nan)
    else:  # else validate gmap against the underlying data
        gmap = _validate_apply_axis_arg(gmap, "gmap", float, data)

    smin = np.nanmin(gmap) if vmin is None else vmin
    smax = np.nanmax(gmap) if vmax is None else vmax
    rng = smax - smin
    _matplotlib = import_optional_dependency(
        "matplotlib", extra="Styler.background_gradient requires matplotlib."
    )
    # extend lower / upper bounds, compresses color range
    norm = _matplotlib.colors.Normalize(smin - (rng * low), smax + (rng * high))

    if cmap is None:
        rgbas = _matplotlib.colormaps[_matplotlib.rcParams["image.cmap"]](norm(gmap))
    else:
        rgbas = _matplotlib.colormaps.get_cmap(cmap)(norm(gmap))

    def relative_luminance(rgba) -> float:
        # [FUNCTION DOC]
        # HELPER: relative luminance of a color from its red, green and blue
        # components (first three of an RGBA sequence) in the sRGB model: linearizes each
        # channel with the standard threshold and returns the weighted sum.
        """
        Calculate relative luminance of a color.

        The calculation adheres to the W3C standards
        (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

        Parameters
        ----------
        color : rgb or rgba tuple

        Returns
        -------
        float
            The relative luminance as a value from 0 to 1
        """
        r, g, b = (
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
            for x in rgba[:3]
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba, text_only) -> str:
        # [FUNCTION DOC]
        # HELPER: builds the CSS for one color. rgba is the color; text_only says whether
        # to color only the text. When coloring the background also sets text color to
        # light or dark based on whether the relative luminance is below the threshold of
        # the enclosing function. Returns the CSS string.
        if not text_only:
            dark = relative_luminance(rgba) < text_color_threshold
            text_color = "#f1f1f1" if dark else "#000000"
            return (
                f"background-color: {_matplotlib.colors.rgb2hex(rgba)};"
                f"color: {text_color};"
            )
        else:
            return f"color: {_matplotlib.colors.rgb2hex(rgba)};"

    if data.ndim == 1:
        return [css(rgba, text_only) for rgba in rgbas]
    else:
        return DataFrame(
            [[css(rgba, text_only) for rgba in row] for row in rgbas],
            index=data.index,
            columns=data.columns,
        )


def _highlight_between(
    data: NDFrame,
    props: str,
    left: Scalar | Sequence | np.ndarray | NDFrame | None = None,
    right: Scalar | Sequence | np.ndarray | NDFrame | None = None,
    inclusive: bool | str = True,
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: marks cells lying between bounds. Parameters: data; props (CSS text
    # returned for matching cells); left and right bounds (none means unbounded; iterable
    # bounds are validated to match data); inclusive (default true, meaning both;
    # accepted values both, neither, left, right). Raises a value error for other
    # inclusive values. Missing comparison results count as false. Returns an array of
    # props text for matching cells and empty strings elsewhere.
    # KNOWN ISSUE: the inclusive parameter is annotated as a boolean or text, but a boolean true is not an accepted value; only the four text values pass, so the default value would raise the error.
    """
    Return an array of css props based on condition of data values within given range.
    """
    if np.iterable(left) and not isinstance(left, str):
        left = _validate_apply_axis_arg(left, "left", None, data)

    if np.iterable(right) and not isinstance(right, str):
        right = _validate_apply_axis_arg(right, "right", None, data)

    # get ops with correct boundary attribution
    if inclusive == "both":
        ops = (operator.ge, operator.le)
    elif inclusive == "neither":
        ops = (operator.gt, operator.lt)
    elif inclusive == "left":
        ops = (operator.ge, operator.lt)
    elif inclusive == "right":
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(
            f"'inclusive' values can be 'both', 'left', 'right', or 'neither' "
            f"got {inclusive}"
        )

    g_left = (
        # error: Argument 2 to "ge" has incompatible type "Union[str, float,
        # Period, Timedelta, Interval[Any], datetime64, timedelta64, datetime,
        # Sequence[Any], ndarray[Any, Any], NDFrame]"; expected "Union
        # [SupportsDunderLE, SupportsDunderGE, SupportsDunderGT, SupportsDunderLT]"
        ops[0](data, left)  # type: ignore[arg-type]
        if left is not None
        else np.full(data.shape, True, dtype=bool)
    )
    if isinstance(g_left, (DataFrame, Series)):
        g_left = g_left.where(pd.notna(g_left), False)
    l_right = (
        # error: Argument 2 to "le" has incompatible type "Union[str, float,
        # Period, Timedelta, Interval[Any], datetime64, timedelta64, datetime,
        # Sequence[Any], ndarray[Any, Any], NDFrame]"; expected "Union
        # [SupportsDunderLE, SupportsDunderGE, SupportsDunderGT, SupportsDunderLT]"
        ops[1](data, right)  # type: ignore[arg-type]
        if right is not None
        else np.full(data.shape, True, dtype=bool)
    )
    if isinstance(l_right, (DataFrame, Series)):
        l_right = l_right.where(pd.notna(l_right), False)
    return np.where(g_left & l_right, props, "")


def _highlight_value(data: DataFrame | Series, op: str, props: str) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: marks the cells equal to the extreme value. op names the reduction
    # (max or min); props is the CSS text. The extreme is computed ignoring missing values
    # (reduced twice for frames to obtain a scalar). Returns an array with props at
    # matching cells and empty strings elsewhere (missing comparisons are false).
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(skipna=True)
    if isinstance(data, DataFrame):  # min/max must be done twice to return scalar
        value = getattr(value, op)(skipna=True)
    cond = data == value
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, "")


def _bar(
    data: NDFrame,
    align: str | float | Callable,
    colors: str | list | tuple,
    cmap: Any,
    width: float,
    height: float,
    vmin: float | None,
    vmax: float | None,
    base_css: str,
):
    # [FUNCTION DOC]
    # HELPER: computes bar CSS for the bar style. Parameters: data (series or frame
    # selection); align (left, right, mid, zero, mean, a number or a callable giving
    # the center value); colors (one color or two colors for negative and positive
    # values); cmap (colormap name or object, overriding colors per cell); width and
    # height as fractions of the cell; vmin and vmax window overrides; base_css text
    # included in every cell. Determines the window from data or vmin/vmax. Alignment
    # mid acts as left if all values are non-negative and as right if all are
    # non-positive; mean, a callable or a number shifts the data to be centered on
    # that value and aligns at zero; unknown alignments raise a value error.
    # Optionally maps the values through the colormap (matplotlib required). Returns a
    # list of CSS strings for one-dimensional data or a two-dimensional array.
    """
    Draw bar chart in data cells using HTML CSS linear gradient.

    Parameters
    ----------
    data : Series or DataFrame
        Underling subset of Styler data on which operations are performed.
    align : str in {"left", "right", "mid", "zero", "mean"}, int, float, callable
        Method for how bars are structured or scalar value of centre point.
    colors : list-like of str
        Two listed colors as string in valid CSS.
    width : float in [0,1]
        The percentage of the cell, measured from left, where drawn bars will reside.
    height : float in [0,1]
        The percentage of the cell's height where drawn bars will reside, centrally
        aligned.
    vmin : float, optional
        Overwrite the minimum value of the window.
    vmax : float, optional
        Overwrite the maximum value of the window.
    base_css : str
        Additional CSS that is included in the cell before bars are drawn.
    """

    def css_bar(start: float, end: float, color: str) -> str:
        # [FUNCTION DOC]
        # HELPER: builds the CSS for one bar. start and end are fractions between 0 and 1
        # of the cell width; color the bar color. Returns the base CSS followed, when end is
        # greater than start, by a linear gradient with transparent parts outside the bar.
        """
        Generate CSS code to draw a bar from start to end in a table cell.

        Uses linear-gradient.

        Parameters
        ----------
        start : float
            Relative positional start of bar coloring in [0,1]
        end : float
            Relative positional end of the bar coloring in [0,1]
        color : str
            CSS valid color to apply.

        Returns
        -------
        str : The CSS applicable to the cell.

        Notes
        -----
        Uses ``base_css`` from outer scope.
        """
        cell_css = base_css
        if end > start:
            cell_css += "background: linear-gradient(90deg,"
            if start > 0:
                cell_css += (
                    f" transparent {start * 100:.1f}%, {color} {start * 100:.1f}%,"
                )
            cell_css += f" {color} {end * 100:.1f}%, transparent {end * 100:.1f}%)"
        return cell_css

    def css_calc(x, left: float, right: float, align: str, color: str | list | tuple):
        # [FUNCTION DOC]
        # HELPER: computes the CSS for one cell. x is the value (centered), left and right
        # the window bounds, align the alignment mode, color one color or a pair for
        # negative and positive values. Returns base CSS if x is missing. Clips x to the
        # window, computes the start and end fractions from the alignment (left from the
        # left, right to the right, zero symmetric around zero, mid around the window
        # center), delegates to the bar drawer, and, when the height is below one, adds
        # no-repeat centering and a background size for the bar height.
        """
        Return the correct CSS for bar placement based on calculated values.

        Parameters
        ----------
        x : float
            Value which determines the bar placement.
        left : float
            Value marking the left side of calculation, usually minimum of data.
        right : float
            Value marking the right side of the calculation, usually maximum of data
            (left < right).
        align : {"left", "right", "zero", "mid"}
            How the bars will be positioned.
            "left", "right", "zero" can be used with any values for ``left``, ``right``.
            "mid" can only be used where ``left <= 0`` and ``right >= 0``.
            "zero" is used to specify a center when all values ``x``, ``left``,
            ``right`` are translated, e.g. by say a mean or median.

        Returns
        -------
        str : Resultant CSS with linear gradient.

        Notes
        -----
        Uses ``colors``, ``width`` and ``height`` from outer scope.
        """
        if pd.isna(x):
            return base_css

        if isinstance(color, (list, tuple)):
            color = color[0] if x < 0 else color[1]
        assert isinstance(color, str)  # mypy redefinition

        x = left if x < left else x
        x = right if x > right else x  # trim data if outside of the window

        start: float = 0
        end: float = 1

        if align == "left":
            # all proportions are measured from the left side between left and right
            end = (x - left) / (right - left)

        elif align == "right":
            # all proportions are measured from the right side between left and right
            start = (x - left) / (right - left)

        else:
            z_frac: float = 0.5  # location of zero based on the left-right range
            if align == "zero":
                # all proportions are measured from the center at zero
                limit: float = max(abs(left), abs(right))
                left, right = -limit, limit
            elif align == "mid":
                # bars drawn from zero either leftwards or rightwards with center at mid
                mid: float = (left + right) / 2
                z_frac = (
                    -mid / (right - left) + 0.5 if mid < 0 else -left / (right - left)
                )

            if x < 0:
                start, end = (x - left) / (right - left), z_frac
            else:
                start, end = z_frac, (x - left) / (right - left)

        ret = css_bar(start * width, end * width, color)
        if height < 1 and "background: linear-gradient(" in ret:
            return (
                ret + f" no-repeat center; background-size: 100% {height * 100:.1f}%;"
            )
        else:
            return ret

    values = data.to_numpy()
    # A tricky way to address the issue where np.nanmin/np.nanmax fail to handle pd.NA.
    left = np.nanmin(data.min(skipna=True)) if vmin is None else vmin
    right = np.nanmax(data.max(skipna=True)) if vmax is None else vmax
    z: float = 0  # adjustment to translate data

    if align == "mid":
        if left >= 0:  # "mid" is documented to act as "left" if all values positive
            align, left = "left", 0 if vmin is None else vmin
        elif right <= 0:  # "mid" is documented to act as "right" if all values negative
            align, right = "right", 0 if vmax is None else vmax
    elif align == "mean":
        z, align = np.nanmean(values), "zero"
    elif callable(align):
        z, align = align(values), "zero"
    elif isinstance(align, (float, int)):
        z, align = float(align), "zero"
    elif align not in ("left", "right", "zero"):
        raise ValueError(
            "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or be a "
            "value defining the center line or a callable that returns a float"
        )

    rgbas = None
    if cmap is not None:
        # use the matplotlib colormap input
        _matplotlib = import_optional_dependency(
            "matplotlib", extra="Styler.bar requires matplotlib."
        )
        cmap = (
            _matplotlib.colormaps[cmap]
            if isinstance(cmap, str)
            else cmap  # assumed to be a Colormap instance as documented
        )
        norm = _matplotlib.colors.Normalize(left, right)
        rgbas = cmap(norm(values))
        if data.ndim == 1:
            rgbas = [_matplotlib.colors.rgb2hex(rgba) for rgba in rgbas]
        else:
            rgbas = [
                [_matplotlib.colors.rgb2hex(rgba) for rgba in row] for row in rgbas
            ]

    assert isinstance(align, str)  # mypy: should now be in [left, right, mid, zero]
    if data.ndim == 1:
        return [
            css_calc(
                x - z, left - z, right - z, align, colors if rgbas is None else rgbas[i]
            )
            for i, x in enumerate(values)
        ]
    else:
        return np.array(
            [
                [
                    css_calc(
                        x - z,
                        left - z,
                        right - z,
                        align,
                        colors if rgbas is None else rgbas[i][j],
                    )
                    for j, x in enumerate(row)
                ]
                for i, row in enumerate(values)
            ]
        )
