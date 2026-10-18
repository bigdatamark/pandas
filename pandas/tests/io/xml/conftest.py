# [MODULE DOC]
# Fixtures for the XML IO tests: each provides the path to an example data file
# under the shared test data directory for XML (books, Chinese UTF-8 document,
# baby names, CTA rail lines KML, and two XSL stylesheets). Used by the read and
# write XML test modules in this package. No import-time side effects.
from pathlib import Path

import pytest


@pytest.fixture
def xml_data_path():
    # [FUNCTION DOC]
    # Fixture returning a path object for the XML example data directory, located
    # relative to this file in the shared data folder one level above the package.
    """
    Returns a Path object to the XML example directory.

    Examples
    --------
    >>> def test_read_xml(xml_data_path):
    ...     pd.read_xml(xml_data_path / "file.xsl")
    """
    return Path(__file__).parent.parent / "data" / "xml"


@pytest.fixture
def xml_books(xml_data_path, datapath):
    # [FUNCTION DOC]
    # Fixture taking the data directory fixture and the data path factory; returns the
    # path, as a string, of the books example XML file.
    """
    Returns the path (as `str`) to the `books.xml` example file.

    Examples
    --------
    >>> def test_read_xml(xml_books):
    ...     pd.read_xml(xml_books)
    """
    return datapath(xml_data_path / "books.xml")


@pytest.fixture
def xml_doc_ch_utf(xml_data_path, datapath):
    # [FUNCTION DOC]
    # Fixture returning the path, as a string, of the Chinese UTF-8 document example XML file.
    """
    Returns the path (as `str`) to the `doc_ch_utf.xml` example file.

    Examples
    --------
    >>> def test_read_xml(xml_doc_ch_utf):
    ...     pd.read_xml(xml_doc_ch_utf)
    """
    return datapath(xml_data_path / "doc_ch_utf.xml")


@pytest.fixture
def xml_baby_names(xml_data_path, datapath):
    # [FUNCTION DOC]
    # Fixture returning the path, as a string, of the baby names example XML file.
    """
    Returns the path (as `str`) to the `baby_names.xml` example file.

    Examples
    --------
    >>> def test_read_xml(xml_baby_names):
    ...     pd.read_xml(xml_baby_names)
    """
    return datapath(xml_data_path / "baby_names.xml")


@pytest.fixture
def kml_cta_rail_lines(xml_data_path, datapath):
    # [FUNCTION DOC]
    # Fixture returning the path, as a string, of the CTA rail lines example KML file.
    """
    Returns the path (as `str`) to the `cta_rail_lines.kml` example file.

    Examples
    --------
    >>> def test_read_xml(kml_cta_rail_lines):
    ...     pd.read_xml(
    ...         kml_cta_rail_lines,
    ...         xpath=".//k:Placemark",
    ...         namespaces={"k": "http://www.opengis.net/kml/2.2"},
    ...         stylesheet=xsl_flatten_doc,
    ...     )
    """
    return datapath(xml_data_path / "cta_rail_lines.kml")


@pytest.fixture
def xsl_flatten_doc(xml_data_path, datapath):
    # [FUNCTION DOC]
    # Fixture returning the path, as a string, of the flatten document example XSL stylesheet.
    """
    Returns the path (as `str`) to the `flatten_doc.xsl` example file.

    Examples
    --------
    >>> def test_read_xsl(xsl_flatten_doc, mode):
    ...     with open(
    ...         xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None
    ...     ) as f:
    ...         xsl_obj = f.read()
    """
    return datapath(xml_data_path / "flatten_doc.xsl")


@pytest.fixture
def xsl_row_field_output(xml_data_path, datapath):
    # [FUNCTION DOC]
    # Fixture returning the path, as a string, of the row and field output example XSL stylesheet.
    """
    Returns the path (as `str`) to the `row_field_output.xsl` example file.

    Examples
    --------
    >>> def test_read_xsl(xsl_row_field_output, mode):
    ...     with open(
    ...         xsl_row_field_output, mode, encoding="utf-8" if mode == "r" else None
    ...     ) as f:
    ...         xsl_obj = f.read()
    """
    return datapath(xml_data_path / "row_field_output.xsl")
