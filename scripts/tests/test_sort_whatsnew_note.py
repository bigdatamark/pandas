# [MODULE DOC]
# Test of the whatsnew note sorting script on a small document.
from scripts.sort_whatsnew_note import sort_whatsnew_note


def test_sort_whatsnew_note() -> None:
    # [FUNCTION DOC]
    # Verifies that two adjacent bullet entries ending in issue references are swapped into
    # ascending issue number order while the headings and blank lines stay in place.
    content = (
        ".. _whatsnew_200:\n"
        "\n"
        "What's new in 2.0.0 (March XX, 2023)\n"
        "------------------------------------\n"
        "\n"
        "Timedelta\n"
        "^^^^^^^^^\n"
        "- Bug in :meth:`Timedelta.round` (:issue:`51494`)\n"
        "- Bug in :class:`TimedeltaIndex` (:issue:`51575`)\n"
        "\n"
    )
    expected = (
        ".. _whatsnew_200:\n"
        "\n"
        "What's new in 2.0.0 (March XX, 2023)\n"
        "------------------------------------\n"
        "\n"
        "Timedelta\n"
        "^^^^^^^^^\n"
        "- Bug in :class:`TimedeltaIndex` (:issue:`51575`)\n"
        "- Bug in :meth:`Timedelta.round` (:issue:`51494`)\n"
        "\n"
    )
    result = sort_whatsnew_note(content)
    assert result == expected
