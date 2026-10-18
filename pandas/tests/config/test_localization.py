# [MODULE DOC]
# Tests for locale utilities in pandas._config.localization: can_set_locale,
# get_locales and the set_locale context manager, plus the detected display
# encoding. At import time it queries the system locales; if none exist the
# module-level pytestmark skips every test, and a separate marker skips tests
# that need more than one locale. Side effects of the tests are limited to
# temporarily changing the process locale, which must be restored.

import codecs
import locale
import os

import pytest

from pandas._config.localization import (
    can_set_locale,
    get_locales,
    set_locale,
)

from pandas.compat import ISMUSL

import pandas as pd

_all_locales = get_locales()

# Don't run any of these tests if we have no locales.
pytestmark = pytest.mark.skipif(not _all_locales, reason="Need locales")

_skip_if_only_one_locale = pytest.mark.skipif(
    len(_all_locales) <= 1, reason="Need multiple locales for meaningful test"
)


def _get_current_locale(lc_var: int = locale.LC_ALL) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the current locale setting for a locale category as a string.
    # Input: lc_var, the locale category constant (default: all categories). Queries
    # the C library setting call without changing anything, because the Python
    # getlocale function is not always consistent (GH#46595).
    # getlocale is not always compliant with setlocale, use setlocale. GH#46595
    return locale.setlocale(lc_var)


@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_current_locale(lc_var):
    # [FUNCTION DOC]
    # Parametrized over locale categories (all, ctype, time). Verifies that the
    # current locale can be set again and that the locale is unchanged afterwards.
    # Can set the current locale
    before_locale = _get_current_locale(lc_var)
    assert can_set_locale(before_locale, lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_locale_valid_set(lc_var):
    # [FUNCTION DOC]
    # Parametrized over locale categories. Verifies that the default (empty) locale can
    # be set and that the locale is unchanged afterwards.
    # Can set the default locale.
    before_locale = _get_current_locale(lc_var)
    assert can_set_locale("", lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize(
    "lc_var",
    (
        locale.LC_ALL,
        locale.LC_CTYPE,
        pytest.param(
            locale.LC_TIME,
            marks=pytest.mark.skipif(
                ISMUSL, reason="MUSL allows setting invalid LC_TIME."
            ),
        ),
    ),
)
def test_can_set_locale_invalid_set(lc_var):
    # [FUNCTION DOC]
    # Parametrized over locale categories (the time category is skipped on musl
    # which accepts invalid values). Verifies that setting a non-existent locale returns
    # false and leaves the locale unchanged.
    # Cannot set an invalid locale.
    before_locale = _get_current_locale(lc_var)
    assert not can_set_locale("non-existent_locale", lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize(
    "lang,enc",
    [
        ("it_CH", "UTF-8"),
        ("en_US", "ascii"),
        ("zh_CN", "GB2312"),
        ("it_IT", "ISO-8859-1"),
    ],
)
@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_locale_no_leak(lang, enc, lc_var):
    # [FUNCTION DOC]
    # Regression test (GH#46595). Parametrized over four language and encoding pairs
    # and locale categories. Verifies that trying to set a locale pair (even failing)
    # does not leave the locale changed.
    # KNOWN ISSUE: the lc_var parameter only selects which category is read before
    # and after; the attempt to set the locale always uses the all-categories constant.
    # Test that can_set_locale does not leak even when returning False. See GH#46595
    before_locale = _get_current_locale(lc_var)
    can_set_locale((lang, enc), locale.LC_ALL)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


def test_can_set_locale_invalid_get(monkeypatch):
    # [FUNCTION DOC]
    # Regression test (GH#22129). Replaces the getlocale function by one raising a
    # value error (restored after the test) and verifies can_set_locale returns false.
    # see GH#22129
    # In some cases, an invalid locale can be set,
    #  but a subsequent getlocale() raises a ValueError.

    def mock_get_locale():
        # [FUNCTION DOC]
        # HELPER: replacement for the locale getter. Takes no arguments and always raises
        # a value error.
        raise ValueError

    with monkeypatch.context() as m:
        m.setattr(locale, "getlocale", mock_get_locale)
        assert not can_set_locale("")


def test_get_locales_at_least_one():
    # [FUNCTION DOC]
    # Regression test (GH#9744). Verifies at least one locale is available.
    # see GH#9744
    assert len(_all_locales) > 0


@_skip_if_only_one_locale
def test_get_locales_prefix():
    # [FUNCTION DOC]
    # Skipped when only one locale exists. Verifies get_locales filtered by the
    # two-letter prefix of the first locale returns at least one locale.
    first_locale = _all_locales[0]
    assert len(get_locales(prefix=first_locale[:2])) > 0


@_skip_if_only_one_locale
@pytest.mark.parametrize(
    "lang,enc",
    [
        ("it_CH", "UTF-8"),
        ("en_US", "ascii"),
        ("zh_CN", "GB2312"),
        ("it_IT", "ISO-8859-1"),
    ],
)
def test_set_locale(lang, enc):
    # [FUNCTION DOC]
    # Skipped when only one locale exists. Parametrized over four language and
    # encoding pairs. If the locale cannot be set, verifies the context manager raises
    # a locale error ("unsupported locale setting"); otherwise verifies the manager yields
    # the normalized locale name whose language and (normalized) encoding equal the
    # requested pair. In both cases verifies the locale is restored on exit.
    # KNOWN ISSUE: the encoding taken from the yielded name is overwritten with the
    # normalized requested encoding before the comparison, so the comparison does not
    # check the encoding actually reported.
    before_locale = _get_current_locale()

    enc = codecs.lookup(enc).name
    new_locale = lang, enc

    if not can_set_locale(new_locale):
        msg = "unsupported locale setting"

        with pytest.raises(locale.Error, match=msg):
            with set_locale(new_locale):
                pass
    else:
        with set_locale(new_locale) as normalized_locale:
            new_lang, new_enc = normalized_locale.split(".")
            new_enc = codecs.lookup(enc).name

            normalized_locale = new_lang, new_enc
            assert normalized_locale == new_locale

    # Once we exit the "with" statement, locale should be back to what it was.
    after_locale = _get_current_locale()
    assert before_locale == after_locale


def test_encoding_detected():
    # [FUNCTION DOC]
    # Verifies the default display encoding option equals the encoding from the
    # environment locale variable (or utf-8 when unset), compared after codec
    # normalization.
    system_locale = os.environ.get("LC_ALL")
    system_encoding = system_locale.split(".")[-1] if system_locale else "utf-8"

    assert (
        codecs.lookup(pd.options.display.encoding).name
        == codecs.lookup(system_encoding).name
    )
