# [MODULE DOC]
# Tests for the internal Series name-setting method and the name attribute. No
# import-time side effects.

from datetime import datetime

from pandas import Series


class TestSetName:
    # [CLASS DOC]
    # Groups the tests of Series name setting. Stateless.
    def test_set_name(self):
        # [FUNCTION DOC]
        # The internal name setter returns a new Series with the name, leaving the
        # original unnamed.
        ser = Series([1, 2, 3])
        ser2 = ser._set_name("foo")
        assert ser2.name == "foo"
        assert ser.name is None
        assert ser is not ser2

    def test_set_name_attribute(self):
        # [FUNCTION DOC]
        # Assigning any hashable (int, float, string, datetime, tuple, non-ASCII text) to
        # the name attribute sets it, both for an unnamed Series and one with a name.
        ser = Series([1, 2, 3])
        ser2 = Series([1, 2, 3], name="bar")
        for name in [7, 7.0, "name", datetime(2001, 1, 1), (1,), "\u05d0"]:
            ser.name = name
            assert ser.name == name
            ser2.name = name
            assert ser2.name == name
