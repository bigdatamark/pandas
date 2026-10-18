"""Tests that the tslibs API is locked down"""

# [MODULE DOC]
# Locks down the public namespace of the internal time-series library package: verifies the set of names exposed
# by that package is exactly the expected set of submodules and API names. Imports the package at import time.

from pandas._libs import tslibs


def test_namespace():
    # [FUNCTION DOC]
    # Builds the expected set from a list of submodule names (base, ccalendar, conversion, dtypes, fields, nattype, np_datetime, offsets,
    # parsing, period, strptime, vectorized, timedeltas, timestamps, timezones, tzconversion) plus a list of API names (offset base classes,
    # missing-time types and sentinels, error types, period and timedelta/timestamp classes, conversion, resolution, timezone and unit helper
    # functions, and format guessing helpers). Takes all attributes of the package whose name does not start with a double underscore and asserts
    # the set equals the expected set, so any addition or removal of an exported name fails the test.
    submodules = [
        "base",
        "ccalendar",
        "conversion",
        "dtypes",
        "fields",
        "nattype",
        "np_datetime",
        "offsets",
        "parsing",
        "period",
        "strptime",
        "vectorized",
        "timedeltas",
        "timestamps",
        "timezones",
        "tzconversion",
    ]

    api = [
        "BaseOffset",
        "NaT",
        "NaTType",
        "iNaT",
        "nat_strings",
        "Day",
        "OutOfBoundsDatetime",
        "OutOfBoundsTimedelta",
        "Period",
        "IncompatibleFrequency",
        "Resolution",
        "Tick",
        "Timedelta",
        "dt64arr_to_periodarr",
        "Timestamp",
        "is_date_array_normalized",
        "ints_to_pydatetime",
        "normalize_i8_timestamps",
        "get_resolution",
        "delta_to_nanoseconds",
        "ints_to_pytimedelta",
        "localize_pydatetime",
        "tz_convert_from_utc",
        "tz_convert_from_utc_single",
        "to_offset",
        "tz_compare",
        "is_unitless",
        "astype_overflowsafe",
        "get_unit_from_dtype",
        "periods_per_day",
        "periods_per_second",
        "guess_datetime_format",
        "add_overflowsafe",
        "get_supported_dtype",
        "is_supported_dtype",
    ]

    expected = set(submodules + api)
    names = [x for x in dir(tslibs) if not x.startswith("__")]
    assert set(names) == expected
