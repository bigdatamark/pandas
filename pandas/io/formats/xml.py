"""
:mod:`pandas.io.formats.xml` is a module for formatting data in XML.
"""
# [MODULE DOC]
# Formatting of a DataFrame as an XML document for the frame-to-xml writer.
# Public contents: the abstract base formatter, the standard-library based
# formatter (EtreeXMLFormatter) and the lxml based formatter (LxmlXMLFormatter,
# which additionally supports XSLT stylesheets). All three classes live here;
# the io-level writer chooses one by parser name and calls its output method.
# Relationships: uses file-handle utilities from the io common module and the
# stylesheet reader from the xml reader module. lxml and the xml standard
# library modules are imported lazily inside methods, so import has no side
# effects.

from __future__ import annotations

import codecs
import io
from typing import (
    TYPE_CHECKING,
    Any,
    final,
)

from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna

from pandas.io.common import get_handle
from pandas.io.xml import get_data_from_filepath

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import DataFrame


class _BaseXMLFormatter:
    # [CLASS DOC]
    # Abstract base class holding the shared logic for XML formatters.
    # State: all constructor arguments are stored as attributes of the same name;
    # orig_cols is the original column labels of the frame; frame_dicts is a mapping
    # from row position to a mapping of column to value (index included when
    # requested, missing values filled when a replacement is given); prefix_uri is the
    # brace-wrapped namespace URI for the chosen prefix (empty if none).
    # Subclasses must implement the tree builder, the prefix URI lookup and the
    # sub-element class property (all raise an abstract-method error here).
    # Key public method: write_output. Invariant: the constructor validates
    # columns, encoding, and computes prefix_uri before index columns are
    # prepended to the attribute/element column lists.
    # KNOWN ISSUE: the class docstring describes the arguments as row_name being the root
    # element name and root_name being the row element name (swapped relative to the
    # code), uses the name attrs_cols instead of attr_cols, and omits frame.
    """
    Subclass for formatting data in XML.

    Parameters
    ----------
    path_or_buffer : str or file-like
        This can be either a string of raw XML, a valid URL,
        file or file-like object.

    index : bool
        Whether to include index in xml document.

    row_name : str
        Name for root of xml document. Default is 'data'.

    root_name : str
        Name for row elements of xml document. Default is 'row'.

    na_rep : str
        Missing data representation.

    attrs_cols : list
        List of columns to write as attributes in row element.

    elem_cols : list
        List of columns to write as children in row element.

    namespaces : dict
        The namespaces to define in XML document as dicts with key
        being namespace and value the URI.

    prefix : str
        The prefix for each element in XML document including root.

    encoding : str
        Encoding of xml object or document.

    xml_declaration : bool
        Whether to include xml declaration at top line item in xml.

    pretty_print : bool
        Whether to write xml document with line breaks and indentation.

    stylesheet : str or file-like
        A URL, file, file-like object, or a raw string containing XSLT.

    compression : str or dict, default 'infer'
        For on-the-fly compression of the output data. If 'infer' and 'path_or_buffer'
        is path-like, then detect compression from the following extensions: '.gz',
        '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
        (otherwise no compression).
        Set to ``None`` for no compression.
        Can also be a dict with key ``'method'`` set
        to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``}
        and other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for faster compression and to
        create a reproducible gzip archive:
        ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

    See also
    --------
    pandas.io.formats.xml.EtreeXMLFormatter
    pandas.io.formats.xml.LxmlXMLFormatter

    """

    def __init__(
        self,
        frame: DataFrame,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        index: bool = True,
        root_name: str | None = "data",
        row_name: str | None = "row",
        na_rep: str | None = None,
        attr_cols: list[str] | None = None,
        elem_cols: list[str] | None = None,
        namespaces: dict[str | None, str] | None = None,
        prefix: str | None = None,
        encoding: str = "utf-8",
        xml_declaration: bool | None = True,
        pretty_print: bool | None = True,
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the options and precompute derived state.
        # Parameters: frame is the frame to serialize; path_or_buffer is a path or
        # writable buffer, or none to return a string (default none); index says whether
        # the index is written as extra columns (default true); root_name is the root
        # element name (default "data"); row_name is the per-row element name (default
        # "row"); na_rep is the replacement for missing values (default none, which
        # leaves them as-is); attr_cols and elem_cols are lists of columns to write as
        # attributes and child elements (default none); namespaces maps prefix to URI
        # (default none); prefix selects the namespace prefix used for all elements
        # (default none); encoding is the output encoding (default utf-8);
        # xml_declaration says whether to emit the declaration (default true);
        # pretty_print says whether to indent (default true); stylesheet is an XSLT
        # source (default none); compression and storage_options are forwarded to file
        # opening (defaults infer and none).
        # Effects: records orig_cols, builds frame_dicts, validates the column lists and
        # encoding (raising type error for non list-like column lists, lookup error for
        # unknown encodings), computes prefix_uri, then prepends index column names to the
        # attribute and element column lists when those are non-empty and the index is
        # included.
        self.frame = frame
        self.path_or_buffer = path_or_buffer
        self.index = index
        self.root_name = root_name
        self.row_name = row_name
        self.na_rep = na_rep
        self.attr_cols = attr_cols
        self.elem_cols = elem_cols
        self.namespaces = namespaces
        self.prefix = prefix
        self.encoding = encoding
        self.xml_declaration = xml_declaration
        self.pretty_print = pretty_print
        self.stylesheet = stylesheet
        self.compression: CompressionOptions = compression
        self.storage_options = storage_options

        self.orig_cols = self.frame.columns.tolist()
        self.frame_dicts = self._process_dataframe()

        self._validate_columns()
        self._validate_encoding()
        self.prefix_uri = self._get_prefix_uri()
        self._handle_indexes()

    def _build_tree(self) -> bytes:
        # [FUNCTION DOC]
        # HELPER: abstract hook for write_output. Subclasses build the whole XML
        # document and return it encoded as bytes. The base implementation always raises an
        # abstract-method error.
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        raise AbstractMethodError(self)

    @final
    def _validate_columns(self) -> None:
        # [FUNCTION DOC]
        # HELPER: constructor validation for the attribute and element column lists.
        # Raises a type error when either list is truthy but not list-like; otherwise
        # returns nothing.
        # KNOWN ISSUE: the docstring says it raises a value error, but the code raises a type error.
        """
        Validate elems_cols and attrs_cols.

        This method will check if columns is list-like.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
        if self.attr_cols and not is_list_like(self.attr_cols):
            raise TypeError(
                f"{type(self.attr_cols).__name__} is not a valid type for attr_cols"
            )

        if self.elem_cols and not is_list_like(self.elem_cols):
            raise TypeError(
                f"{type(self.elem_cols).__name__} is not a valid type for elem_cols"
            )

    @final
    def _validate_encoding(self) -> None:
        # [FUNCTION DOC]
        # HELPER: constructor validation of the encoding name. Looks the encoding up in
        # the codec registry; an unknown name raises a lookup error. Returns nothing.
        """
        Validate encoding.

        This method will check if encoding is among listed under codecs.

        Raises
        ------
        LookupError
            * If encoding is not available in codecs.
        """

        codecs.lookup(self.encoding)

    @final
    def _process_dataframe(self) -> dict[int | str, dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: prepares the frame for output. Resets the index into columns when
        # index output is enabled, replaces missing values with na_rep when it is not
        # none, and returns a mapping from row position to a dict of column to value.
        """
        Adjust Data Frame to fit xml output.

        This method will adjust underlying data frame for xml output,
        including optionally replacing missing values and including indexes.
        """

        df = self.frame

        if self.index:
            df = df.reset_index()

        if self.na_rep is not None:
            df = df.fillna(self.na_rep)

        return df.to_dict(orient="index")

    @final
    def _handle_indexes(self) -> None:
        # [FUNCTION DOC]
        # HELPER: when the index is written, determines the index column names (keys of
        # the first row mapping that are not among the original columns) and prepends
        # them to the attribute and element column lists, each only if non-empty.
        # Does nothing if the index is not written.
        # Raises a stop-iteration error if the frame has no rows, since the first row
        # is taken from the row mapping.
        """
        Handle indexes.

        This method will add indexes into attr_cols or elem_cols.
        """

        if not self.index:
            return

        first_key = next(iter(self.frame_dicts))
        indexes: list[str] = [
            x for x in self.frame_dicts[first_key].keys() if x not in self.orig_cols
        ]

        if self.attr_cols:
            self.attr_cols = indexes + self.attr_cols

        if self.elem_cols:
            self.elem_cols = indexes + self.elem_cols

    def _get_prefix_uri(self) -> str:
        # [FUNCTION DOC]
        # HELPER: abstract hook returning the namespace URI (wrapped in braces) matching
        # the configured prefix. Subclasses implement it; the base raises an
        # abstract-method error.
        # KNOWN ISSUE: the docstring mentions a key error for a missing prefix, but this base version only raises the abstract-method error; the key error comes from subclasses.
        """
        Get uri of namespace prefix.

        This method retrieves corresponding URI to prefix in namespaces.

        Raises
        ------
        KeyError
            *If prefix is not included in namespace dict.
        """

        raise AbstractMethodError(self)

    @final
    def _other_namespaces(self) -> dict:
        # [FUNCTION DOC]
        # HELPER: builds the extra namespace declarations for the root element. Returns
        # a mapping of declaration attribute names (default one for an empty prefix, or
        # prefixed ones) to URIs, skipping the URI that is already used as the element
        # prefix. Returns an empty mapping when no namespaces are configured.
        """
        Define other namespaces.

        This method will build dictionary of namespaces attributes
        for root element, conditionally with optional namespaces and
        prefix.
        """

        nmsp_dict: dict[str, str] = {}
        if self.namespaces:
            nmsp_dict = {
                f"xmlns{p if p == '' else f':{p}'}": n
                for p, n in self.namespaces.items()
                if n != self.prefix_uri[1:-1]
            }

        return nmsp_dict

    @final
    def _build_attribs(self, d: dict[str, Any], elem_row: Any) -> Any:
        # [FUNCTION DOC]
        # HELPER: writes the attribute columns onto a row element. Parameters: d is the
        # row mapping of column to value; elem_row is the row element. For every column
        # in attr_cols, sets an attribute (name from the flattened column name) to the
        # string form of the value unless it is missing. Returns the same element
        # (unchanged if attr_cols is empty). Raises a key error naming the column when it
        # is absent from the row mapping.
        """
        Create attributes of row.

        This method adds attributes using attr_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """

        if not self.attr_cols:
            return elem_row

        for col in self.attr_cols:
            attr_name = self._get_flat_col_name(col)
            try:
                if not isna(d[col]):
                    elem_row.attrib[attr_name] = str(d[col])
            except KeyError as err:
                raise KeyError(f"no valid column, {col}") from err
        return elem_row

    @final
    def _get_flat_col_name(self, col: str | tuple) -> str:
        # [FUNCTION DOC]
        # HELPER: converts a column label into an element/attribute name prefixed with
        # prefix_uri. Tuple labels (multi-level columns) are joined by concatenation if
        # any part is an empty string, otherwise joined with underscores, then stripped.
        # Non-tuple labels are used as-is.
        flat_col = col
        if isinstance(col, tuple):
            flat_col = (
                "".join([str(c) for c in col]).strip()
                if "" in col
                else "_".join([str(c) for c in col]).strip()
            )
        return f"{self.prefix_uri}{flat_col}"

    @cache_readonly
    def _sub_element_cls(self):
        # [FUNCTION DOC]
        # HELPER: cached abstract property returning the sub-element factory of the
        # backing XML library. The base raises an abstract-method error.
        raise AbstractMethodError(self)

    @final
    def _build_elems(self, d: dict[str, Any], elem_row: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the element columns as children of a row element. Parameters:
        # d is the row mapping; elem_row is the row element. For each column in
        # elem_cols creates a child named by the flattened column name whose text is the
        # string form of the value, or no text when the value is missing or an empty
        # string. Does nothing if elem_cols is empty; raises a key error naming the
        # column if it is absent. Returns nothing.
        """
        Create child elements of row.

        This method adds child elements using elem_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """
        sub_element_cls = self._sub_element_cls

        if not self.elem_cols:
            return

        for col in self.elem_cols:
            elem_name = self._get_flat_col_name(col)
            try:
                val = None if isna(d[col]) or d[col] == "" else str(d[col])
                sub_element_cls(elem_row, elem_name).text = val
            except KeyError as err:
                raise KeyError(f"no valid column, {col}") from err

    @final
    def write_output(self) -> str | None:
        # [FUNCTION DOC]
        # Builds the XML document and emits it. If path_or_buffer is set, writes the
        # bytes to it in binary mode using the configured compression and storage
        # options and returns none. Otherwise returns the document decoded with the
        # configured encoding with trailing whitespace removed.
        xml_doc = self._build_tree()

        if self.path_or_buffer is not None:
            with get_handle(
                self.path_or_buffer,
                "wb",
                compression=self.compression,
                storage_options=self.storage_options,
                is_text=False,
            ) as handles:
                handles.handle.write(xml_doc)
            return None

        else:
            return xml_doc.decode(self.encoding).rstrip()


class EtreeXMLFormatter(_BaseXMLFormatter):
    # [CLASS DOC]
    # Concrete formatter built on the standard library element tree and minidom
    # modules. Implements the tree building, prefix lookup, sub-element class, and
    # pretty printing hooks. Extra state: root (root element) and out_xml (the
    # serialized bytes) set while building. Does not support stylesheets.
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    def _build_tree(self) -> bytes:
        # [FUNCTION DOC]
        # Builds the document using the standard library: creates the root element with
        # the extra namespace declarations, adds one row element per entry in
        # frame_dicts; if neither attribute nor element columns were given, all keys of
        # the row mapping become element columns (this assignment persists on the
        # instance) and are written, otherwise attributes and elements are written.
        # Serializes with the configured encoding and declaration setting, optionally
        # pretty-prints (reparsing with minidom), and returns the bytes.
        # Raises a value error if a stylesheet was given (needs lxml). Sets root and
        # out_xml.
        # KNOWN ISSUE: the stylesheet error is raised only after the tree has been serialized and pretty-printed, so the work is wasted.
        from xml.etree.ElementTree import (
            Element,
            SubElement,
            tostring,
        )

        self.root = Element(
            f"{self.prefix_uri}{self.root_name}", attrib=self._other_namespaces()
        )

        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f"{self.prefix_uri}{self.row_name}")

            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)

            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)

        self.out_xml = tostring(
            self.root,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )

        if self.pretty_print:
            self.out_xml = self._prettify_tree()

        if self.stylesheet is not None:
            raise ValueError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )

        return self.out_xml

    def _get_prefix_uri(self) -> str:
        # [FUNCTION DOC]
        # HELPER: returns the brace-wrapped URI for the prefix, or the empty string.
        # When namespaces are given, registers every string prefix/URI pair with the
        # library's global namespace registry (global side effect). If a prefix is set, uses
        # its URI (raising a key error saying the prefix is not in namespaces if absent);
        # else uses the default (empty-prefix) namespace if present; else empty.
        from xml.etree.ElementTree import register_namespace

        uri = ""
        if self.namespaces:
            for p, n in self.namespaces.items():
                if isinstance(p, str) and isinstance(n, str):
                    register_namespace(p, n)
            if self.prefix:
                try:
                    uri = f"{{{self.namespaces[self.prefix]}}}"
                except KeyError as err:
                    raise KeyError(
                        f"{self.prefix} is not included in namespaces"
                    ) from err
            elif "" in self.namespaces:
                uri = f"{{{self.namespaces['']}}}"
            else:
                uri = ""

        return uri

    @cache_readonly
    def _sub_element_cls(self):
        # [FUNCTION DOC]
        # HELPER: cached property returning the standard library sub-element factory.
        from xml.etree.ElementTree import SubElement

        return SubElement

    def _prettify_tree(self) -> bytes:
        # [FUNCTION DOC]
        # HELPER: reparses the serialized document with minidom and returns it as
        # indented bytes (two spaces) in the configured encoding.
        """
        Output tree for pretty print format.

        This method will pretty print xml with line breaks and indentation.
        """

        from xml.dom.minidom import parseString

        dom = parseString(self.out_xml)

        return dom.toprettyxml(indent="  ", encoding=self.encoding)


class LxmlXMLFormatter(_BaseXMLFormatter):
    # [CLASS DOC]
    # Concrete formatter built on lxml. Implements tree building with native
    # namespace map support and optional XSLT transformation of the result.
    # State: root (lxml root element) and out_xml (serialized bytes) set while
    # building. Replaces an empty-string namespace key by none at construction.
    # KNOWN ISSUE: the class docstring is copied from the standard library formatter and wrongly says it uses the standard library modules.
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Runs the base constructor with all arguments, then converts an empty-string
        # namespace prefix key into none (the form lxml requires). Arguments are
        # passed through unchanged.
        super().__init__(*args, **kwargs)

        self._convert_empty_str_key()

    def _build_tree(self) -> bytes:
        # [FUNCTION DOC]
        # Builds the document using lxml: creates the root element with the namespace
        # map as given, adds a row element per entry (same default-all-columns behavior
        # as the base standard library formatter, which stores the column list on the
        # instance), writes attributes/elements, and serializes with pretty printing,
        # encoding, and declaration options. If a stylesheet is set, the result is
        # transformed with it. Returns bytes; sets root and out_xml.
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        from lxml.etree import (
            Element,
            SubElement,
            tostring,
        )

        self.root = Element(f"{self.prefix_uri}{self.root_name}", nsmap=self.namespaces)

        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f"{self.prefix_uri}{self.row_name}")

            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)

            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)

        self.out_xml = tostring(
            self.root,
            pretty_print=self.pretty_print,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )

        if self.stylesheet is not None:
            self.out_xml = self._transform_doc()

        return self.out_xml

    def _convert_empty_str_key(self) -> None:
        # [FUNCTION DOC]
        # HELPER: if the namespaces mapping has an empty-string key, moves its value to a
        # none key (default placeholder if popping fails). Mutates the caller's mapping in
        # place. Returns nothing.
        """
        Replace zero-length string in `namespaces`.

        This method will replace '' with None to align to `lxml`
        requirement that empty string prefixes are not allowed.
        """

        if self.namespaces and "" in self.namespaces.keys():
            self.namespaces[None] = self.namespaces.pop("", "default")

    def _get_prefix_uri(self) -> str:
        # [FUNCTION DOC]
        # HELPER: returns the brace-wrapped URI for the configured prefix, else for
        # the empty-prefix namespace, else the empty string. Raises a key error saying
        # the prefix is not in namespaces if a prefix is set but absent. Unlike the
        # standard library variant, does not register namespaces globally.
        uri = ""
        if self.namespaces:
            if self.prefix:
                try:
                    uri = f"{{{self.namespaces[self.prefix]}}}"
                except KeyError as err:
                    raise KeyError(
                        f"{self.prefix} is not included in namespaces"
                    ) from err
            elif "" in self.namespaces:
                uri = f"{{{self.namespaces['']}}}"
            else:
                uri = ""

        return uri

    @cache_readonly
    def _sub_element_cls(self):
        # [FUNCTION DOC]
        # HELPER: cached property returning the lxml sub-element factory.
        from lxml.etree import SubElement

        return SubElement

    def _transform_doc(self) -> bytes:
        # [FUNCTION DOC]
        # HELPER: applies the XSLT stylesheet to the built tree. Reads the stylesheet
        # (path, URL, buffer or raw string) through the xml reader's data loader with the
        # configured encoding, compression and storage options, parses it (from encoded
        # text for string buffers, else directly), applies it to root, and returns the
        # result as bytes. Raises parser or transform errors from lxml for invalid
        # stylesheets.
        """
        Parse stylesheet from file or buffer and run it.

        This method will parse stylesheet object into tree for parsing
        conditionally by its specific object type, then transforms
        original tree with XSLT script.
        """
        from lxml.etree import (
            XSLT,
            XMLParser,
            fromstring,
            parse,
        )

        style_doc = self.stylesheet
        assert style_doc is not None  # is ensured by caller

        handle_data = get_data_from_filepath(
            filepath_or_buffer=style_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        with handle_data as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)

            if isinstance(xml_data, io.StringIO):
                xsl_doc = fromstring(
                    xml_data.getvalue().encode(self.encoding), parser=curr_parser
                )
            else:
                xsl_doc = parse(xml_data, parser=curr_parser)

        transformer = XSLT(xsl_doc)
        new_doc = transformer(self.root)

        return bytes(new_doc)
