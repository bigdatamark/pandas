# [MODULE DOC]
# Tests for the internal ultra-fast JSON encoder/decoder extension (imported under the
# alias ujson) that backs the JSON reading and writing.
# Organized in three test classes: general encoder/decoder behavior for built-in Python
# types (numbers, strings, unicode, dates, objects, error handling), behavior for numpy
# scalars and arrays, and behavior for pandas objects (frames, series, indexes, sets,
# timedeltas, periods).
# Most tests encode a value, decode it, and compare against the standard library JSON
# module or the original value.
# Defines a helper to stringify dictionary keys and a fixture parametrizing the
# orientation.

import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time

import dateutil
import numpy as np
import pytest

import pandas._libs.json as ujson
from pandas.compat import IS64

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    PeriodIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


def _clean_dict(d):
    # [FUNCTION DOC]
    # HELPER: used by the pandas round-trip tests for the split orientation.
    # Takes a dictionary and returns a new dictionary whose keys have all been converted
    # to strings, with values unchanged, so it can be unpacked as keyword arguments to a
    # constructor.
    """
    Sanitize dictionary for JSON by converting all keys to strings.

    Parameters
    ----------
    d : dict
        The dictionary to convert.

    Returns
    -------
    cleaned_dict : dict
    """
    return {str(k): v for k, v in d.items()}


@pytest.fixture(
    params=[None, "split", "records", "values", "index"]  # Column indexed by default.
)
def orient(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the encoder orientation: none (the default,
    # column indexed), split, records, values and index. Returns the current orientation
    # value.
    return request.param


class TestUltraJSONTests:
    # [CLASS DOC]
    # Tests of the JSON encoder and decoder on plain Python values: decimals, floats and
    # precision, strings and unicode, ints and longs, booleans and None, dates and
    # times, objects with custom conversion hooks, and malformed-input errors.
    @pytest.mark.skipif(not IS64, reason="not compliant on 32-bit, xref #15865")
    @pytest.mark.parametrize(
        "value, double_precision",
        [
            ("1337.1337", 15),
            ("0.95", 1),
            ("0.94", 1),
            ("1.95", 1),
            ("-1.95", 1),
            ("0.995", 2),
            ("0.9995", 3),
            ("0.99999999999999944", 15),
        ],
    )
    def test_encode_decimal(self, value, double_precision):
        # [FUNCTION DOC]
        # Verifies that a decimal value encodes with the given double precision and
        # decodes back to the same string representation. Skipped on 32-bit platforms.
        # Parametrized over decimal strings and the precision needed to preserve them.
        sut = decimal.Decimal(value)
        encoded = ujson.ujson_dumps(sut, double_precision=double_precision)
        decoded = ujson.ujson_loads(encoded)
        assert decoded == value

    @pytest.mark.parametrize("ensure_ascii", [True, False])
    def test_encode_string_conversion(self, ensure_ascii):
        # [FUNCTION DOC]
        # Verifies the encoding of a string containing backslash, slash, control
        # characters, a closing script tag and an ampersand, for both ASCII-only and
        # non-ASCII-only output settings.
        # Uses a nested checker that compares the output to the expected text and checks
        # that both the standard library and the decoder restore the original. Without
        # the option, or with HTML encoding disabled, the tag and ampersand stay
        # literal; with HTML encoding enabled they are written as unicode escapes.
        string_input = "A string \\ / \b \f \n \r \t </script> &"
        not_html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
        html_encoded = (
            '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'
        )

        def helper(expected_output, **encode_kwargs):
            # [FUNCTION DOC]
            # HELPER: nested checker for the string-conversion test. Takes the expected
            # encoded text and extra encoder keyword options, encodes the shared input
            # string using the parametrized ASCII setting plus the options, and asserts
            # the output equals the expected text and that decoding it with both the
            # standard library and the ujson decoder restores the input.
            output = ujson.ujson_dumps(
                string_input, ensure_ascii=ensure_ascii, **encode_kwargs
            )

            assert output == expected_output
            assert string_input == json.loads(output)
            assert string_input == ujson.ujson_loads(output)

        # Default behavior assumes encode_html_chars=False.
        helper(not_html_encoded)

        # Make sure explicit encode_html_chars=False works.
        helper(not_html_encoded, encode_html_chars=False)

        # Make sure explicit encode_html_chars=True does the encoding.
        helper(html_encoded, encode_html_chars=True)

    @pytest.mark.parametrize(
        "long_number", [-4342969734183514, -12345678901234.56789012, -528656961.4399388]
    )
    def test_double_long_numbers(self, long_number):
        # [FUNCTION DOC]
        # Verifies that long negative numbers (a large integer and two floats with many
        # digits) stored in a dictionary survive an encode and decode round trip with
        # precision fifteen.
        sut = {"a": long_number}
        encoded = ujson.ujson_dumps(sut, double_precision=15)

        decoded = ujson.ujson_loads(encoded)
        assert sut == decoded

    def test_encode_non_c_locale(self):
        # [FUNCTION DOC]
        # Verifies that encoding and decoding floats works while the numeric locale is
        # switched to an Italian locale (decimal comma), also with precise float
        # decoding.
        # Tries two locale names and uses the first one the system supports; if neither
        # can be set, nothing is asserted.
        lc_category = locale.LC_NUMERIC

        # We just need one of these locales to work.
        for new_locale in ("it_IT.UTF-8", "Italian_Italy"):
            if tm.can_set_locale(new_locale, lc_category):
                with tm.set_locale(new_locale, lc_category):
                    assert ujson.ujson_loads(ujson.ujson_dumps(4.78e60)) == 4.78e60
                    assert ujson.ujson_loads("4.78", precise_float=True) == 4.78
                break

    def test_decimal_decode_test_precise(self):
        # [FUNCTION DOC]
        # Verifies that decoding with the precise float option round trips a dictionary
        # holding a float.
        sut = {"a": 4.56}
        encoded = ujson.ujson_dumps(sut)
        decoded = ujson.ujson_loads(encoded, precise_float=True)
        assert sut == decoded

    def test_encode_double_tiny_exponential(self):
        # [FUNCTION DOC]
        # Verifies that very small floats (with exponent down to minus one hundred
        # forty-five) round trip, using exact equality for the first three and
        # approximate equality for the smallest.
        num = 1e-40
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = 1e-100
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-45
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-145
        assert np.allclose(num, ujson.ujson_loads(ujson.ujson_dumps(num)))

    @pytest.mark.parametrize("unicode_key", ["key1", "بن"])
    def test_encode_dict_with_unicode_keys(self, unicode_key):
        # [FUNCTION DOC]
        # Verifies a dictionary with an ASCII or Arabic key round trips unchanged.
        unicode_dict = {unicode_key: "value1"}
        assert unicode_dict == ujson.ujson_loads(ujson.ujson_dumps(unicode_dict))

    @pytest.mark.parametrize("double_input", [math.pi, -math.pi])
    def test_encode_double_conversion(self, double_input):
        # [FUNCTION DOC]
        # Verifies that pi and its negative encode to text that both the standard
        # library and the ujson decoder read back equal after rounding to five places.
        output = ujson.ujson_dumps(double_input)
        assert round(double_input, 5) == round(json.loads(output), 5)
        assert round(double_input, 5) == round(ujson.ujson_loads(output), 5)

    def test_encode_with_decimal(self):
        # [FUNCTION DOC]
        # Verifies that encoding the value one point zero gives the text one point zero.
        # KNOWN ISSUE: the test name and the local variable name say decimal, but the
        # input is a plain float, so no decimal object is exercised.
        decimal_input = 1.0
        output = ujson.ujson_dumps(decimal_input)

        assert output == "1.0"

    def test_encode_array_of_nested_arrays(self):
        # [FUNCTION DOC]
        # Verifies a list of twenty copies of a four-level nested empty list encodes and
        # decodes equal to the original, matching the standard library.
        nested_input = [[[[]]]] * 20
        output = ujson.ujson_dumps(nested_input)

        assert nested_input == json.loads(output)
        assert nested_input == ujson.ujson_loads(output)

    def test_encode_array_of_doubles(self):
        # [FUNCTION DOC]
        # Verifies a list of forty identical doubles round trips and matches the
        # standard library's decode of the encoded text.
        doubles_input = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
        output = ujson.ujson_dumps(doubles_input)

        assert doubles_input == json.loads(output)
        assert doubles_input == ujson.ujson_loads(output)

    def test_double_precision(self):
        # [FUNCTION DOC]
        # Verifies a double with many digits round trips exactly at precision fifteen,
        # and that with precision three and nine the decoded value equals the input
        # rounded to that many places.
        double_input = 30.012345678901234
        output = ujson.ujson_dumps(double_input, double_precision=15)

        assert double_input == json.loads(output)
        assert double_input == ujson.ujson_loads(output)

        for double_precision in (3, 9):
            output = ujson.ujson_dumps(double_input, double_precision=double_precision)
            rounded_input = round(double_input, double_precision)

            assert rounded_input == json.loads(output)
            assert rounded_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "invalid_val",
        [
            20,
            -1,
            "9",
            None,
        ],
    )
    def test_invalid_double_precision(self, invalid_val):
        # [FUNCTION DOC]
        # Verifies that invalid precision values raise errors: integers out of range
        # (twenty, minus one) raise a value error citing a maximum of fifteen; a string
        # or None raises a type error about needing an integer.
        double_input = 30.12345678901234567890
        expected_exception = ValueError if isinstance(invalid_val, int) else TypeError
        msg = (
            r"Invalid value '.*' for option 'double_precision', max is '15'|"
            r"an integer is required \(got type |"
            r"object cannot be interpreted as an integer"
        )
        with pytest.raises(expected_exception, match=msg):
            ujson.ujson_dumps(double_input, double_precision=invalid_val)

    def test_encode_string_conversion2(self):
        # [FUNCTION DOC]
        # Verifies a string with backslash, slash and control characters round trips and
        # encodes to the exact expected escaped text.
        string_input = "A string \\ / \b \f \n \r \t"
        output = ujson.ujson_dumps(string_input)

        assert string_input == json.loads(output)
        assert string_input == ujson.ujson_loads(output)
        assert output == '"A string \\\\ \\/ \\b \\f \\n \\r \\t"'

    @pytest.mark.parametrize(
        "unicode_input",
        ["Räksmörgås اسامة بن محمد بن عوض بن لادن", "\xe6\x97\xa5\xd1\x88"],
    )
    def test_encode_unicode_conversion(self, unicode_input):
        # [FUNCTION DOC]
        # Verifies that non-ASCII strings encode identically to the standard library's
        # output and decode to the same value.
        enc = ujson.ujson_dumps(unicode_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(unicode_input)
        assert dec == json.loads(enc)

    def test_encode_control_escaping(self):
        # [FUNCTION DOC]
        # Verifies that a control character (code point twenty-five) is escaped
        # identically to the standard library and round trips.
        escaped_input = "\x19"
        enc = ujson.ujson_dumps(escaped_input)
        dec = ujson.ujson_loads(enc)

        assert escaped_input == dec
        assert enc == json.dumps(escaped_input)

    def test_encode_unicode_surrogate_pair(self):
        # [FUNCTION DOC]
        # Verifies a four-character string, written as latin-1 code points that spell
        # out a UTF-8 surrogate-pair byte sequence, encodes identically to the standard
        # library and round trips.
        surrogate_input = "\xf0\x90\x8d\x86"
        enc = ujson.ujson_dumps(surrogate_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(surrogate_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8(self):
        # [FUNCTION DOC]
        # Verifies a string starting with four latin-1 code points (spelling a four-byte
        # UTF-8 sequence) followed by ASCII text encodes identically to the standard
        # library and round trips.
        four_bytes_input = "\xf0\x91\x80\xb0TRAILINGNORMAL"
        enc = ujson.ujson_dumps(four_bytes_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8highest(self):
        # [FUNCTION DOC]
        # Same as the four-byte test but with the highest four-byte sequence; verifies
        # identical encoding to the standard library and a round trip.
        four_bytes_input = "\xf3\xbf\xbf\xbfTRAILINGNORMAL"
        enc = ujson.ujson_dumps(four_bytes_input)

        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_error(self):
        # [FUNCTION DOC]
        # Verifies that encoding a list holding a string with a lone surrogate raises a
        # unicode encode error with a message saying surrogates are not allowed in
        # utf-8.
        string = "'\udac0'"
        msg = (
            r"'utf-8' codec can't encode character '\\udac0' "
            r"in position 1: surrogates not allowed"
        )
        with pytest.raises(UnicodeEncodeError, match=msg):
            ujson.ujson_dumps([string])

    def test_encode_array_in_array(self):
        # [FUNCTION DOC]
        # Verifies a deeply nested empty list encodes to the same text as the standard
        # library and decodes back equal.
        arr_in_arr_input = [[[[]]]]
        output = ujson.ujson_dumps(arr_in_arr_input)

        assert arr_in_arr_input == json.loads(output)
        assert output == json.dumps(arr_in_arr_input)
        assert arr_in_arr_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "num_input",
        [
            31337,
            -31337,  # Negative number.
            -9223372036854775808,  # Large negative number.
        ],
    )
    def test_encode_num_conversion(self, num_input):
        # [FUNCTION DOC]
        # Verifies ints (positive, negative, and the minimum signed 64-bit value) encode
        # identically to the standard library and round trip.
        output = ujson.ujson_dumps(num_input)
        assert num_input == json.loads(output)
        assert output == json.dumps(num_input)
        assert num_input == ujson.ujson_loads(output)

    def test_encode_list_conversion(self):
        # [FUNCTION DOC]
        # Verifies a list of four ints round trips and matches the standard library's
        # decode of the output.
        list_input = [1, 2, 3, 4]
        output = ujson.ujson_dumps(list_input)

        assert list_input == json.loads(output)
        assert list_input == ujson.ujson_loads(output)

    def test_encode_dict_conversion(self):
        # [FUNCTION DOC]
        # Verifies a dictionary of four string keys to ints round trips and matches the
        # standard library's decode of the output.
        dict_input = {"k1": 1, "k2": 2, "k3": 3, "k4": 4}
        output = ujson.ujson_dumps(dict_input)

        assert dict_input == json.loads(output)
        assert dict_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("builtin_value", [None, True, False])
    def test_encode_builtin_values_conversion(self, builtin_value):
        # [FUNCTION DOC]
        # Verifies None, True and False encode identically to the standard library and
        # round trip.
        output = ujson.ujson_dumps(builtin_value)
        assert builtin_value == json.loads(output)
        assert output == json.dumps(builtin_value)
        assert builtin_value == ujson.ujson_loads(output)

    def test_encode_datetime_conversion(self):
        # [FUNCTION DOC]
        # Verifies a datetime (from the current time) encodes with second date unit to
        # the integer epoch seconds computed from its UTC time tuple.
        datetime_input = datetime.datetime.fromtimestamp(time.time())
        output = ujson.ujson_dumps(datetime_input, date_unit="s")
        expected = calendar.timegm(datetime_input.utctimetuple())

        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    def test_encode_date_conversion(self):
        # [FUNCTION DOC]
        # Verifies a date (from the current time) encodes with second date unit to the
        # integer epoch seconds of midnight of that date.
        date_input = datetime.date.fromtimestamp(time.time())
        output = ujson.ujson_dumps(date_input, date_unit="s")

        tup = (date_input.year, date_input.month, date_input.day, 0, 0, 0)
        expected = calendar.timegm(tup)

        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "test",
        [datetime.time(), datetime.time(1, 2, 3), datetime.time(10, 12, 15, 343243)],
    )
    def test_encode_time_conversion_basic(self, test):
        # [FUNCTION DOC]
        # Verifies time-of-day values (default, with seconds, with microseconds) encode
        # to a quoted ISO format string.
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_pytz(self):
        # [FUNCTION DOC]
        # Verifies a timezone-aware time (UTC from pytz) encodes to a quoted ISO string
        # without crashing (regression for a segfault). Skipped when pytz is not
        # installed.
        # see gh-11473: to_json segfaults with timezone-aware datetimes
        pytz = pytest.importorskip("pytz")
        test = datetime.time(10, 12, 15, 343243, pytz.utc)
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_dateutil(self):
        # [FUNCTION DOC]
        # Verifies a timezone-aware time (UTC from dateutil) encodes to a quoted ISO
        # string without crashing (regression for a segfault).
        # see gh-11473: to_json segfaults with timezone-aware datetimes
        test = datetime.time(10, 12, 15, 343243, dateutil.tz.tzutc())
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    @pytest.mark.parametrize(
        "decoded_input", [NaT, np.datetime64("NaT", "ns"), np.nan, np.inf, -np.inf]
    )
    def test_encode_as_null(self, decoded_input):
        # [FUNCTION DOC]
        # Verifies missing-value markers (NaT, a numpy NaT datetime, NaN, positive and
        # negative infinity) all encode to the JSON null literal.
        assert ujson.ujson_dumps(decoded_input) == "null", "Expected null"

    def test_datetime_units(self):
        # [FUNCTION DOC]
        # Verifies that encoding a datetime with each date unit (second, millisecond,
        # microsecond, nanosecond) yields the nanosecond epoch value scaled down
        # accordingly, and that an unknown unit raises a value error saying the option
        # value is invalid.
        val = datetime.datetime(2013, 8, 17, 21, 17, 12, 215504)
        stamp = Timestamp(val).as_unit("ns")

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="s"))
        assert roundtrip == stamp._value // 10**9

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="ms"))
        assert roundtrip == stamp._value // 10**6

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="us"))
        assert roundtrip == stamp._value // 10**3

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="ns"))
        assert roundtrip == stamp._value

        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_dumps(val, date_unit="foo")

    def test_encode_to_utf8(self):
        # [FUNCTION DOC]
        # Verifies that encoding with ASCII-only output disabled matches the standard
        # library with the same setting and round trips.
        unencoded = "\xe6\x97\xa5\xd1\x88"

        enc = ujson.ujson_dumps(unencoded, ensure_ascii=False)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(unencoded, ensure_ascii=False)
        assert dec == json.loads(enc)

    def test_decode_from_unicode(self):
        # [FUNCTION DOC]
        # Verifies decoding the same JSON text as given and as a re-created string gives
        # equal results.
        unicode_input = '{"obj": 31337}'

        dec1 = ujson.ujson_loads(unicode_input)
        dec2 = ujson.ujson_loads(str(unicode_input))

        assert dec1 == dec2

    def test_encode_recursion_max(self):
        # [FUNCTION DOC]
        # Verifies that encoding an object graph that refers back to itself raises an
        # overflow error saying the maximum recursion level was reached.
        # Uses two locally defined classes with a member attribute to build the cycle.
        # 8 is the max recursion depth

        class O2:
            # [CLASS DOC]
            # HELPER: throwaway class for the recursion test, with a class attribute
            # member defaulting to zero.
            member = 0

        class O1:
            # [CLASS DOC]
            # HELPER: throwaway class for the recursion test, with a class attribute
            # member defaulting to zero; the test links an instance of this class and an
            # instance of the other class in a cycle.
            member = 0

        decoded_input = O1()
        decoded_input.member = O2()
        decoded_input.member.member = decoded_input

        with pytest.raises(OverflowError, match="Maximum recursion level reached"):
            ujson.ujson_dumps(decoded_input)

    def test_decode_jibberish(self):
        # [FUNCTION DOC]
        # Verifies decoding nonsense text raises a value error about an unexpected
        # character found when decoding false.
        jibberish = "fdsa sda v9sa fdsa"
        msg = "Unexpected character found when decoding 'false'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(jibberish)

    @pytest.mark.parametrize(
        "broken_json",
        [
            "[",  # Broken array start.
            "{",  # Broken object start.
            "]",  # Broken array end.
            "}",  # Broken object end.
        ],
    )
    def test_decode_broken_json(self, broken_json):
        # [FUNCTION DOC]
        # Verifies lone bracket or brace characters raise a value error saying an object
        # or value was expected.
        msg = "Expected object or value"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize("too_big_char", ["[", "{"])
    def test_decode_depth_too_big(self, too_big_char):
        # [FUNCTION DOC]
        # Verifies decoding a million opening brackets or braces raises a value error
        # about reaching the object decoding depth limit.
        with pytest.raises(ValueError, match="Reached object decoding depth limit"):
            ujson.ujson_loads(too_big_char * (1024 * 1024))

    @pytest.mark.parametrize(
        "bad_string",
        [
            '"TESTING',  # Unterminated.
            '"TESTING\\"',  # Unterminated escape.
            "tru",  # Broken True.
            "fa",  # Broken False.
            "n",  # Broken None.
        ],
    )
    def test_decode_bad_string(self, bad_string):
        # [FUNCTION DOC]
        # Verifies unterminated strings, unterminated escapes, and truncated true, false
        # and null literals raise a value error with one of the two expected
        # decode-error messages.
        msg = (
            "Unexpected character found when decoding|"
            "Unmatched ''\"' when when decoding 'string'"
        )
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(bad_string)

    @pytest.mark.parametrize(
        "broken_json, err_msg",
        [
            (
                '{{1337:""}}',
                "Key name of object must be 'string' when decoding 'object'",
            ),
            ('{{"key":"}', "Unmatched ''\"' when when decoding 'string'"),
            ("[[[true", "Unexpected character found when decoding array value (2)"),
        ],
    )
    def test_decode_broken_json_leak(self, broken_json, err_msg):
        # [FUNCTION DOC]
        # Verifies that each of three malformed documents raises the specific expected
        # value error message, repeated one thousand times to exercise memory cleanup
        # paths (a leak regression).
        for _ in range(1000):
            with pytest.raises(ValueError, match=re.escape(err_msg)):
                ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize(
        "invalid_dict",
        [
            "{{{{31337}}}}",  # No key.
            '{{{{"key":}}}}',  # No value.
            '{{{{"key"}}}}',  # No colon or value.
        ],
    )
    def test_decode_invalid_dict(self, invalid_dict):
        # [FUNCTION DOC]
        # Verifies malformed objects (missing key, missing value, missing colon and
        # value) raise a value error with one of three expected messages.
        msg = (
            "Key name of object must be 'string' when decoding 'object'|"
            "No ':' found when decoding object value|"
            "Expected object or value"
        )
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_dict)

    @pytest.mark.parametrize("numeric_int_as_str", ["31337", "-31337"])
    def test_decode_numeric_int(self, numeric_int_as_str):
        # [FUNCTION DOC]
        # Verifies positive and negative integer text decodes to the same integer as the
        # built-in conversion.
        assert int(numeric_int_as_str) == ujson.ujson_loads(numeric_int_as_str)

    def test_encode_null_character(self):
        # [FUNCTION DOC]
        # Verifies that strings with an embedded or lone null character encode
        # identically to the standard library and round trip, and that a string of
        # spaces, a null, a carriage return and a newline encodes to the exact expected
        # escaped text.
        wrapped_input = "31337 \x00 1337"
        output = ujson.ujson_dumps(wrapped_input)

        assert wrapped_input == json.loads(output)
        assert output == json.dumps(wrapped_input)
        assert wrapped_input == ujson.ujson_loads(output)

        alone_input = "\x00"
        output = ujson.ujson_dumps(alone_input)

        assert alone_input == json.loads(output)
        assert output == json.dumps(alone_input)
        assert alone_input == ujson.ujson_loads(output)
        assert '"  \\u0000\\r\\n "' == ujson.ujson_dumps("  \u0000\r\n ")

    def test_decode_null_character(self):
        # [FUNCTION DOC]
        # Verifies decoding a JSON string containing an escaped null character matches
        # the standard library.
        wrapped_input = '"31337 \\u0000 31337"'
        assert ujson.ujson_loads(wrapped_input) == json.loads(wrapped_input)

    def test_encode_list_long_conversion(self):
        # [FUNCTION DOC]
        # Verifies a list of six maximum signed 64-bit integers round trips and matches
        # the standard library's decode.
        long_input = [
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
        ]
        output = ujson.ujson_dumps(long_input)

        assert long_input == json.loads(output)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("long_input", [9223372036854775807, 18446744073709551615])
    def test_encode_long_conversion(self, long_input):
        # [FUNCTION DOC]
        # Verifies the maximum signed and unsigned 64-bit integers encode identically to
        # the standard library and round trip.
        output = ujson.ujson_dumps(long_input)

        assert long_input == json.loads(output)
        assert output == json.dumps(long_input)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("bigNum", [2**64, -(2**63) - 1])
    def test_dumps_ints_larger_than_maxsize(self, bigNum):
        # [FUNCTION DOC]
        # Verifies integers beyond the 64-bit range (two to the sixty-fourth and below
        # the minimum signed) encode to their decimal text but decoding that text raises
        # a value error saying the value is too big or too small.
        encoding = ujson.ujson_dumps(bigNum)
        assert str(bigNum) == encoding

        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            assert ujson.ujson_loads(encoding) == bigNum

    @pytest.mark.parametrize(
        "int_exp", ["1337E40", "1.337E40", "1337E+9", "1.337e+40", "1.337E-4"]
    )
    def test_decode_numeric_int_exp(self, int_exp):
        # [FUNCTION DOC]
        # Verifies integers and floats written with exponents decode to the same values
        # as the standard library.
        assert ujson.ujson_loads(int_exp) == json.loads(int_exp)

    def test_loads_non_str_bytes_raises(self):
        # [FUNCTION DOC]
        # Verifies decoding None raises a type error saying a bytes-like object is
        # required.
        msg = "a bytes-like object is required, not 'NoneType'"
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_loads(None)

    @pytest.mark.parametrize("val", [3590016419, 2**31, 2**32, (2**32) - 1])
    def test_decode_number_with_32bit_sign_bit(self, val):
        # [FUNCTION DOC]
        # Verifies numbers between two to the thirty-first and two to the thirty-second
        # (which would have the 32-bit sign bit set) decode correctly from a small
        # document.
        # Test that numbers that fit within 32 bits but would have the
        # sign bit set (2**31 <= x < 2**32) are decoded properly.
        doc = f'{{"id": {val}}}'
        assert ujson.ujson_loads(doc)["id"] == val

    def test_encode_big_escape(self):
        # [FUNCTION DOC]
        # Smoke test: encodes a very large (two megabyte of two-byte characters) bytes
        # value ten times and expects no exception.
        # Make sure no Exception is raised.
        for _ in range(10):
            base = "\u00e5".encode()
            escape_input = base * 1024 * 1024 * 2
            ujson.ujson_dumps(escape_input)

    def test_decode_big_escape(self):
        # [FUNCTION DOC]
        # Smoke test: decodes a very large quoted bytes document ten times and expects
        # no exception.
        # Make sure no Exception is raised.
        for _ in range(10):
            base = "\u00e5".encode()
            quote = b'"'

            escape_input = quote + (base * 1024 * 1024 * 2) + quote
            ujson.ujson_loads(escape_input)

    def test_to_dict(self):
        # [FUNCTION DOC]
        # Verifies an object exposing a dictionary-conversion method is encoded as the
        # dictionary that method returns.
        d = {"key": 31337}

        class DictTest:
            # [CLASS DOC]
            # HELPER: local class for the dictionary-conversion test, providing a method
            # that returns the enclosing test's dictionary.
            def toDict(self):
                # [FUNCTION DOC]
                # HELPER: returns the dictionary captured from the enclosing test; used
                # by the encoder as the object's JSON form.
                return d

        o = DictTest()
        output = ujson.ujson_dumps(o)

        dec = ujson.ujson_loads(output)
        assert dec == d

    def test_default_handler(self):
        # [FUNCTION DOC]
        # Verifies the default handler option of the encoder: without a handler an
        # object with a self-producing attribute raises an overflow error (maximum
        # recursion); with the string conversion as handler it is encoded as a quoted
        # string; custom handlers returning a string, an integer or a datetime are used
        # for the encoded value; a handler that raises propagates its type error; and a
        # list of objects encodes like the standard library with a string default.
        class _TestObject:
            # [CLASS DOC]
            # HELPER: local class for the default-handler test. Stores a single value;
            # its string conversion returns the string form of the stored value; a
            # property returns a new instance each access, making the object infinitely
            # recursive for the encoder unless a handler is given.
            def __init__(self, val) -> None:
                # [FUNCTION DOC]
                # HELPER: constructor storing the given value in the val attribute.
                self.val = val

            @property
            def recursive_attr(self):
                # [FUNCTION DOC]
                # HELPER: property returning a new instance of the enclosing test object
                # class with a fixed string value; this makes the encoder recurse until
                # it hits its depth limit.
                return _TestObject("recursive_attr")

            def __str__(self) -> str:
                # [FUNCTION DOC]
                # HELPER: returns the string form of the stored value.
                return str(self.val)

        msg = "Maximum recursion level reached"
        with pytest.raises(OverflowError, match=msg):
            ujson.ujson_dumps(_TestObject("foo"))
        assert '"foo"' == ujson.ujson_dumps(_TestObject("foo"), default_handler=str)

        def my_handler(_):
            # [FUNCTION DOC]
            # HELPER: default handler ignoring its argument and returning the fixed
            # string foobar.
            return "foobar"

        assert '"foobar"' == ujson.ujson_dumps(
            _TestObject("foo"), default_handler=my_handler
        )

        def my_handler_raises(_):
            # [FUNCTION DOC]
            # HELPER: default handler ignoring its argument and raising a type error
            # with a fixed message.
            raise TypeError("I raise for anything")

        with pytest.raises(TypeError, match="I raise for anything"):
            ujson.ujson_dumps(_TestObject("foo"), default_handler=my_handler_raises)

        def my_int_handler(_):
            # [FUNCTION DOC]
            # HELPER: default handler ignoring its argument and returning the integer
            # forty-two.
            return 42

        assert (
            ujson.ujson_loads(
                ujson.ujson_dumps(_TestObject("foo"), default_handler=my_int_handler)
            )
            == 42
        )

        def my_obj_handler(_):
            # [FUNCTION DOC]
            # HELPER: default handler ignoring its argument and returning a fixed
            # datetime (third of February 2013).
            return datetime.datetime(2013, 2, 3)

        assert ujson.ujson_loads(
            ujson.ujson_dumps(datetime.datetime(2013, 2, 3))
        ) == ujson.ujson_loads(
            ujson.ujson_dumps(_TestObject("foo"), default_handler=my_obj_handler)
        )

        obj_list = [_TestObject("foo"), _TestObject("bar")]
        assert json.loads(json.dumps(obj_list, default=str)) == ujson.ujson_loads(
            ujson.ujson_dumps(obj_list, default_handler=str)
        )

    def test_encode_object(self):
        # [FUNCTION DOC]
        # Verifies that encoding a plain object produces a mapping of only its public,
        # non-callable attributes (those not starting with an underscore), so methods
        # and underscore attributes are omitted.
        class _TestObject:
            # [CLASS DOC]
            # HELPER: local class for the object-encoding test, with four stored
            # attributes (one underscore-prefixed) and a method returning five.
            def __init__(self, a, b, _c, d) -> None:
                # [FUNCTION DOC]
                # HELPER: constructor storing the four arguments as attributes a, b, a
                # private one, and d.
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self):
                # [FUNCTION DOC]
                # HELPER: method returning the constant five; present to confirm
                # callables are skipped by the encoder.
                return 5

        # JSON keys should be all non-callable non-underscore attributes, see GH-42768
        test_object = _TestObject(a=1, b=2, _c=3, d=4)
        assert ujson.ujson_loads(ujson.ujson_dumps(test_object)) == {
            "a": 1,
            "b": 2,
            "d": 4,
        }

    def test_ujson__name__(self):
        # [FUNCTION DOC]
        # Verifies the extension module reports its own name as the full pandas internal
        # path for the JSON library.
        # GH 52898
        assert ujson.__name__ == "pandas._libs.json"


class TestNumpyJSONTests:
    # [CLASS DOC]
    # Tests of the JSON encoder and decoder on numpy scalars and arrays (booleans,
    # integer and float dtypes, multidimensional and object arrays, and unsupported
    # scalar types).
    @pytest.mark.parametrize("bool_input", [True, False])
    def test_bool(self, bool_input):
        # [FUNCTION DOC]
        # Verifies a boolean converted from the parametrized value round trips.
        b = bool(bool_input)
        assert ujson.ujson_loads(ujson.ujson_dumps(b)) == b

    def test_bool_array(self):
        # [FUNCTION DOC]
        # Verifies a one-dimensional boolean array round trips through encode and
        # decode.
        bool_array = np.array(
            [True, False, True, True, False, True, False, False], dtype=bool
        )
        output = np.array(ujson.ujson_loads(ujson.ujson_dumps(bool_array)), dtype=bool)
        tm.assert_numpy_array_equal(bool_array, output)

    def test_int(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies a numpy integer scalar of each integer dtype with value one round
        # trips (compared after converting back to the same scalar type).
        klass = np.dtype(any_int_numpy_dtype).type
        num = klass(1)

        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_int_array(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies an array of the first hundred integers cast to each integer dtype
        # round trips.
        arr = np.arange(100, dtype=int)
        arr_input = arr.astype(any_int_numpy_dtype)

        arr_output = np.array(
            ujson.ujson_loads(ujson.ujson_dumps(arr_input)), dtype=any_int_numpy_dtype
        )
        tm.assert_numpy_array_equal(arr_input, arr_output)

    def test_int_max(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies the maximum value of each integer dtype round trips, except the
        # unsigned 64-bit type, for which the signed 64-bit maximum is used since larger
        # values are encoded as signed. Skips 64-bit types on 32-bit platforms.
        if any_int_numpy_dtype in ("int64", "uint64") and not IS64:
            pytest.skip("Cannot test 64-bit integer on 32-bit platform")

        klass = np.dtype(any_int_numpy_dtype).type

        # uint64 max will always overflow,
        # as it's encoded to signed.
        if any_int_numpy_dtype == "uint64":
            num = np.iinfo("int64").max
        else:
            num = np.iinfo(any_int_numpy_dtype).max

        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float(self, float_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies a numpy float scalar of each float dtype round trips exactly.
        klass = np.dtype(float_numpy_dtype).type
        num = klass(256.2013)

        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float_array(self, float_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies an array of evenly spaced floats cast to each float dtype round trips
        # approximately with precision fifteen.
        arr = np.arange(12.5, 185.72, 1.7322, dtype=float)
        float_input = arr.astype(float_numpy_dtype)

        float_output = np.array(
            ujson.ujson_loads(ujson.ujson_dumps(float_input, double_precision=15)),
            dtype=float_numpy_dtype,
        )
        tm.assert_almost_equal(float_input, float_output)

    def test_float_max(self, float_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies one tenth of the maximum value of each float dtype round trips
        # approximately at precision fifteen.
        klass = np.dtype(float_numpy_dtype).type
        num = klass(np.finfo(float_numpy_dtype).max / 10)

        tm.assert_almost_equal(
            klass(ujson.ujson_loads(ujson.ujson_dumps(num, double_precision=15))), num
        )

    def test_array_basic(self):
        # [FUNCTION DOC]
        # Verifies a six-dimensional integer array (96 elements) round trips preserving
        # shape.
        arr = np.arange(96)
        arr = arr.reshape((2, 2, 2, 2, 3, 2))

        tm.assert_numpy_array_equal(
            np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr
        )

    @pytest.mark.parametrize("shape", [(10, 10), (5, 5, 4), (100, 1)])
    def test_array_reshaped(self, shape):
        # [FUNCTION DOC]
        # Verifies an array of the first hundred integers reshaped to each parametrized
        # shape round trips preserving shape.
        arr = np.arange(100)
        arr = arr.reshape(shape)

        tm.assert_numpy_array_equal(
            np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr
        )

    def test_array_list(self):
        # [FUNCTION DOC]
        # Verifies an object array containing a string, empty lists and dictionaries,
        # numbers, a list and a dictionary round trips.
        arr_list = [
            "a",
            [],
            {},
            {},
            [],
            42,
            97.8,
            ["a", "b"],
            {"key": "val"},
        ]
        arr = np.array(arr_list, dtype=object)
        result = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=object)
        tm.assert_numpy_array_equal(result, arr)

    def test_array_float(self):
        # [FUNCTION DOC]
        # Verifies a three-dimensional single-precision float array round trips
        # approximately.
        dtype = np.float32

        arr = np.arange(100.202, 200.202, 1, dtype=dtype)
        arr = arr.reshape((5, 5, 4))

        arr_out = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=dtype)
        tm.assert_almost_equal(arr, arr_out)

    def test_0d_array(self):
        # [FUNCTION DOC]
        # Verifies encoding a zero-dimensional array raises a type error stating the
        # numpy scalar is not JSON serializable at the moment (regression for an older
        # issue).
        # gh-18878
        msg = re.escape(
            "array(1) (numpy-scalar) is not JSON serializable at the moment"
        )
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.array(1))

    def test_array_long_double(self):
        # [FUNCTION DOC]
        # Verifies encoding an extended-precision float scalar raises a type error
        # stating it is not JSON serializable at the moment.
        msg = re.compile(
            "1234.5.* \\(numpy-scalar\\) is not JSON serializable at the moment"
        )
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.longdouble(1234.5))


class TestPandasJSONTests:
    # [CLASS DOC]
    # Tests of the JSON encoder and decoder on pandas objects (DataFrame, Series, Index,
    # DatetimeIndex, PeriodIndex, Timedelta), sets, and edge cases of numeric and array
    # decoding. Many are parametrized on the orientation fixture.
    def test_dataframe(self, orient):
        # [FUNCTION DOC]
        # Verifies a small int64 DataFrame round trips for each orientation: encodes
        # with the orientation, decodes, rebuilds a frame (unpacking the dictionary for
        # the split orientation), adjusts the expected frame's labels to what each
        # orientation preserves, and checks that dtypes remain int64 and the frames are
        # equal.
        dtype = np.int64

        df = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            index=["a", "b"],
            columns=["x", "y", "z"],
            dtype=dtype,
        )
        encode_kwargs = {} if orient is None else {"orient": orient}
        assert (df.dtypes == dtype).all()

        output = ujson.ujson_loads(ujson.ujson_dumps(df, **encode_kwargs))
        assert (df.dtypes == dtype).all()

        # Ensure proper DataFrame initialization.
        if orient == "split":
            dec = _clean_dict(output)
            output = DataFrame(**dec)
        else:
            output = DataFrame(output)

        # Corrections to enable DataFrame comparison.
        if orient == "values":
            df.columns = [0, 1, 2]
            df.index = [0, 1]
        elif orient == "records":
            df.index = [0, 1]
        elif orient == "index":
            df = df.transpose()

        assert (df.dtypes == dtype).all()
        tm.assert_frame_equal(output, df)

    def test_dataframe_nested(self, orient):
        # [FUNCTION DOC]
        # Verifies that a dictionary containing two DataFrames encodes to the same
        # nested structure as encoding each frame separately, for each orientation.
        df = DataFrame(
            [[1, 2, 3], [4, 5, 6]], index=["a", "b"], columns=["x", "y", "z"]
        )

        nested = {"df1": df, "df2": df.copy()}
        kwargs = {} if orient is None else {"orient": orient}

        exp = {
            "df1": ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)),
            "df2": ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)),
        }
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_series(self, orient):
        # [FUNCTION DOC]
        # Verifies an int64 Series round trips for each orientation, rebuilding a Series
        # (unpacking the dictionary for split) and adjusting the expected name and index
        # for what each orientation preserves (string index labels for default and index
        # orientations, a default range index for records and values), then compares
        # series and dtype.
        dtype = np.int64
        s = Series(
            [10, 20, 30, 40, 50, 60],
            name="series",
            index=[6, 7, 8, 9, 10, 15],
            dtype=dtype,
        ).sort_values()
        assert s.dtype == dtype

        encode_kwargs = {} if orient is None else {"orient": orient}

        output = ujson.ujson_loads(ujson.ujson_dumps(s, **encode_kwargs))
        assert s.dtype == dtype

        if orient == "split":
            dec = _clean_dict(output)
            output = Series(**dec)
        else:
            output = Series(output)

        if orient in (None, "index"):
            s.name = None
            output = output.sort_values()
            s.index = ["6", "7", "8", "9", "10", "15"]
        elif orient in ("records", "values"):
            s.name = None
            s.index = [0, 1, 2, 3, 4, 5]

        assert s.dtype == dtype
        tm.assert_series_equal(output, s)

    def test_series_nested(self, orient):
        # [FUNCTION DOC]
        # Verifies that a dictionary containing two Series encodes to the same nested
        # structure as encoding each series separately, for each orientation.
        s = Series(
            [10, 20, 30, 40, 50, 60], name="series", index=[6, 7, 8, 9, 10, 15]
        ).sort_values()
        nested = {"s1": s, "s2": s.copy()}
        kwargs = {} if orient is None else {"orient": orient}

        exp = {
            "s1": ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)),
            "s2": ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)),
        }
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_index(self):
        # [FUNCTION DOC]
        # Verifies a named integer Index round trips for the default, split, values,
        # records and index orientations.
        # KNOWN ISSUE: the split-orientation comparison of the index and its name is
        # performed twice in a row, which is redundant.
        i = Index([23, 45, 18, 98, 43, 11], name="index")

        # Column indexed.
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i)), name="index")
        tm.assert_index_equal(i, output)

        dec = _clean_dict(ujson.ujson_loads(ujson.ujson_dumps(i, orient="split")))
        output = Index(**dec)

        tm.assert_index_equal(i, output)
        assert i.name == output.name

        tm.assert_index_equal(i, output)
        assert i.name == output.name

        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="values")), name="index"
        )
        tm.assert_index_equal(i, output)

        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="records")), name="index"
        )
        tm.assert_index_equal(i, output)

        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="index")), name="index"
        )
        tm.assert_index_equal(i, output)

    def test_datetime_index(self):
        # [FUNCTION DOC]
        # Verifies a DatetimeIndex with nanosecond unit and no frequency encodes with
        # nanosecond date unit and decodes to an equal index, and that a Series indexed
        # by it round trips (decoded index values are converted back from integers).
        date_unit = "ns"

        # freq doesn't round-trip
        rng = DatetimeIndex(
            list(date_range("1/1/2000", periods=20, unit="ns")), freq=None
        )
        encoded = ujson.ujson_dumps(rng, date_unit=date_unit)

        decoded = DatetimeIndex(np.array(ujson.ujson_loads(encoded)))
        tm.assert_index_equal(rng, decoded)

        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        decoded = Series(ujson.ujson_loads(ujson.ujson_dumps(ts, date_unit=date_unit)))

        idx_values = decoded.index.values.astype(np.int64)
        decoded.index = DatetimeIndex(idx_values)
        tm.assert_series_equal(ts, decoded)

    @pytest.mark.parametrize(
        "invalid_arr",
        [
            "[31337,]",  # Trailing comma.
            "[,31337]",  # Leading comma.
            "[]]",  # Unmatched bracket.
            "[,]",  # Only comma.
        ],
    )
    def test_decode_invalid_array(self, invalid_arr):
        # [FUNCTION DOC]
        # Verifies invalid array text (trailing comma, leading comma, unmatched closing
        # bracket, only comma) raises a value error with one of the expected messages.
        msg = (
            "Expected object or value|Trailing data|"
            "Unexpected character found when decoding array value"
        )
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_arr)

    @pytest.mark.parametrize("arr", [[], [31337]])
    def test_decode_array(self, arr):
        # [FUNCTION DOC]
        # Verifies an empty list and a single-element list decode from their string
        # forms.
        assert arr == ujson.ujson_loads(str(arr))

    @pytest.mark.parametrize("extreme_num", [9223372036854775807, -9223372036854775808])
    def test_decode_extreme_numbers(self, extreme_num):
        # [FUNCTION DOC]
        # Verifies the maximum and minimum signed 64-bit integers decode correctly from
        # their string forms.
        assert extreme_num == ujson.ujson_loads(str(extreme_num))

    @pytest.mark.parametrize("too_extreme_num", [f"{2**64}", f"{-(2**63) - 1}"])
    def test_decode_too_extreme_numbers(self, too_extreme_num):
        # [FUNCTION DOC]
        # Verifies integer text just outside the signed 64-bit range raises a value
        # error saying the value is too big or too small.
        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            ujson.ujson_loads(too_extreme_num)

    def test_decode_with_trailing_whitespaces(self):
        # [FUNCTION DOC]
        # Verifies an empty object followed by whitespace characters decodes to an empty
        # dictionary.
        assert {} == ujson.ujson_loads("{}\n\t ")

    def test_decode_with_trailing_non_whitespaces(self):
        # [FUNCTION DOC]
        # Verifies an object followed by whitespace and a non-whitespace character
        # raises a value error about trailing data.
        with pytest.raises(ValueError, match="Trailing data"):
            ujson.ujson_loads("{}\n\t a")

    @pytest.mark.parametrize("value", [f"{2**64}", f"{-(2**63) - 1}"])
    def test_decode_array_with_big_int(self, value):
        # [FUNCTION DOC]
        # Verifies integer text outside the signed 64-bit range raises a value error
        # saying the value is too big or too small.
        # KNOWN ISSUE: despite its name this test decodes a bare number, not an array,
        # and duplicates the previous too-extreme-numbers test.
        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            ujson.ujson_loads(value)

    @pytest.mark.parametrize(
        "float_number",
        [
            1.1234567893,
            1.234567893,
            1.34567893,
            1.4567893,
            1.567893,
            1.67893,
            1.7893,
            1.893,
            1.3,
        ],
    )
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_decode_floating_point(self, sign, float_number):
        # [FUNCTION DOC]
        # Verifies positive and negative floats with many digits decode from their
        # string form to within a relative tolerance of 1e-15 (sign and number both
        # parametrized).
        float_number *= sign
        tm.assert_almost_equal(
            float_number, ujson.ujson_loads(str(float_number)), rtol=1e-15
        )

    def test_encode_big_set(self):
        # [FUNCTION DOC]
        # Smoke test: encoding a set of 100000 integers raises no exception.
        s = set(range(100000))
        # Make sure no Exception is raised.
        ujson.ujson_dumps(s)

    def test_encode_empty_set(self):
        # [FUNCTION DOC]
        # Verifies an empty set encodes to an empty JSON array.
        assert "[]" == ujson.ujson_dumps(set())

    def test_encode_set(self):
        # [FUNCTION DOC]
        # Verifies a set of nine ints encodes to an array whose decoded elements are all
        # members of the set.
        s = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        enc = ujson.ujson_dumps(s)
        dec = ujson.ujson_loads(enc)

        for v in dec:
            assert v in s

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta(days=366),
            Timedelta(days=-1),
            Timedelta(hours=13, minutes=5, seconds=5),
            Timedelta(hours=13, minutes=20, seconds=30),
            Timedelta(days=-1, nanoseconds=5),
            Timedelta(nanoseconds=1),
            Timedelta(microseconds=1, nanoseconds=1),
            Timedelta(milliseconds=1, microseconds=1, nanoseconds=1),
            Timedelta(milliseconds=999, microseconds=999, nanoseconds=999),
        ],
    )
    def test_encode_timedelta_iso(self, td):
        # [FUNCTION DOC]
        # Verifies Timedelta values of varying magnitudes (days, hours, negative, down
        # to nanoseconds) encode with ISO dates enabled to the quoted ISO duration
        # string produced by the Timedelta itself.
        # GH 28256
        result = ujson.ujson_dumps(td, iso_dates=True)
        expected = f'"{td.isoformat()}"'

        assert result == expected

    def test_encode_periodindex(self):
        # [FUNCTION DOC]
        # Verifies a DataFrame with no columns indexed by a daily PeriodIndex serializes
        # to an empty JSON object.
        # GH 46683
        p = PeriodIndex(["2022-04-06", "2022-04-07"], freq="D")
        df = DataFrame(index=p)
        assert df.to_json() == "{}"
