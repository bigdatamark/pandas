# [MODULE DOC]
# Tests for the interactive notebook display integration of frames and series,
# in particular the table-schema (data resource) mime representation. Uses the
# shared ip fixture providing an interactive shell, and the option-context
# manager of the config module. Contains one test class.
import numpy as np

import pandas._config.config as cf

from pandas import (
    DataFrame,
    MultiIndex,
)


class TestTableSchemaRepr:
    # [CLASS DOC]
    # Groups tests of the data-resource mime output controlled by the table
    # schema display option and the styler render representation option. No
    # state; plain test class using the shell fixture for some tests.
    def test_publishes(self, ip):
        # [FUNCTION DOC]
        # With the table schema option on, formatting a series publishes plain
        # text and the data resource mime types, and formatting a frame also
        # publishes html. Then with the styler representation option set to
        # latex as well, the last object (the frame) additionally publishes a
        # latex mime type. The option context is rebuilt inside the loop
        # because a single context object cannot be reused on all systems
        # (GH#58055).
        ipython = ip.instance(config=ip.config)
        df = DataFrame({"A": [1, 2]})
        objects = [df["A"], df]  # dataframe / series
        expected_keys = [
            {"text/plain", "application/vnd.dataresource+json"},
            {"text/plain", "text/html", "application/vnd.dataresource+json"},
        ]

        opt = cf.option_context("display.html.table_schema", True)
        last_obj = None
        for obj, expected in zip(objects, expected_keys):
            last_obj = obj
            with cf.option_context("display.html.table_schema", True):
                # Can't reuse opt on all systems GH#58055
                formatted = ipython.display_formatter.format(obj)
            assert set(formatted[0].keys()) == expected

        with_latex = cf.option_context("styler.render.repr", "latex")

        with opt, with_latex:
            formatted = ipython.display_formatter.format(last_obj)

        expected = {
            "text/plain",
            "text/html",
            "text/latex",
            "application/vnd.dataresource+json",
        }
        assert set(formatted[0].keys()) == expected

    def test_publishes_not_implemented(self, ip):
        # [FUNCTION DOC]
        # Regression test (GH#15996): a frame with a two-level column index of
        # random normal values cannot be expressed as a data resource, so even
        # with the table schema option on only plain text and html mime types
        # are published.
        # column MultiIndex
        # GH#15996
        midx = MultiIndex.from_product([["A", "B"], ["a", "b", "c"]])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, len(midx))), columns=midx
        )

        opt = cf.option_context("display.html.table_schema", True)

        with opt:
            formatted = ip.instance(config=ip.config).display_formatter.format(df)

        expected = {"text/plain", "text/html"}
        assert set(formatted[0].keys()) == expected

    def test_config_on(self):
        # [FUNCTION DOC]
        # With the table schema option on, the frame's data resource
        # representation method returns a non-empty (not none) result.
        df = DataFrame({"A": [1, 2]})
        with cf.option_context("display.html.table_schema", True):
            result = df._repr_data_resource_()

        assert result is not None

    def test_config_default_off(self):
        # [FUNCTION DOC]
        # With the table schema option explicitly off, the frame's data
        # resource representation method returns none.
        df = DataFrame({"A": [1, 2]})
        with cf.option_context("display.html.table_schema", False):
            result = df._repr_data_resource_()

        assert result is None

    def test_enable_data_resource_formatter(self, ip):
        # [FUNCTION DOC]
        # Regression test (GH#10491): the data resource formatter is registered
        # on the shell and enabled only while the table schema option is on;
        # outside the option context it stays registered but disabled, and it
        # can be enabled again by re-entering the context. A smoke call to
        # format an arbitrary object (the config module) must not fail.
        # GH#10491
        formatters = ip.instance(config=ip.config).display_formatter.formatters
        mimetype = "application/vnd.dataresource+json"

        with cf.option_context("display.html.table_schema", True):
            assert "application/vnd.dataresource+json" in formatters
            assert formatters[mimetype].enabled

        # still there, just disabled
        assert "application/vnd.dataresource+json" in formatters
        assert not formatters[mimetype].enabled

        # able to re-set
        with cf.option_context("display.html.table_schema", True):
            assert "application/vnd.dataresource+json" in formatters
            assert formatters[mimetype].enabled
            # smoke test that it works
            ip.instance(config=ip.config).display_formatter.format(cf)
