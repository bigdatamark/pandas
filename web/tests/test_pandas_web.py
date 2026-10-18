# [MODULE DOC]
# Test of the website generator's release-list preprocessor. Provides a fake HTTP response class, fixtures for a minimal context and
# for replacing the HTTP get function with one returning a parametrized fake response, a sample list of releases and one test.
from unittest.mock import (  # noqa: TID251
    mock_open,
    patch,
)

import pytest
import requests

from web.pandas_web import Preprocessors


class MockResponse:
    # [CLASS DOC]
    # HELPER: fake HTTP response. State: status_code and the stored response payload. Provides json() returning the payload and a static
    # raise_for_status doing nothing.
    def __init__(self, status_code: int, response: dict) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the status code and response payload.
        self.status_code = status_code
        self._resp = response

    def json(self):
        # [FUNCTION DOC]
        # HELPER: returns the stored payload.
        return self._resp

    @staticmethod
    def raise_for_status() -> None:
        # [FUNCTION DOC]
        # HELPER: static; pretends the response was successful (does nothing).
        return


@pytest.fixture
def context() -> dict:
    # [FUNCTION DOC]
    # HELPER: fixture returning a minimal context dictionary with the repository name and a target path.
    return {
        "main": {"github_repo_url": "pandas-dev/pandas"},
        "target_path": "test_target_path",
    }


@pytest.fixture
def mock_response(monkeypatch, request) -> None:
    # [FUNCTION DOC]
    # HELPER: fixture that replaces the HTTP get function of the requests library (via monkeypatch) with a function returning a fake
    # response built from the fixture's parameter (a status code and payload pair). Parametrized indirectly by the test.
    def mocked_resp(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: replacement for the HTTP get function; ignores its arguments and returns the fake response.
        status_code, response = request.param
        return MockResponse(status_code, response)

    monkeypatch.setattr(requests, "get", mocked_resp)


_releases_list = [
    {
        "prerelease": False,
        "published_at": "2024-01-19T03:34:05Z",
        "tag_name": "v1.5.6",
        "assets": None,
    },
    {
        "prerelease": False,
        "published_at": "2023-11-10T19:07:37Z",
        "tag_name": "v2.1.3",
        "assets": None,
    },
    {
        "prerelease": False,
        "published_at": "2023-08-30T13:24:32Z",
        "tag_name": "v2.1.0",
        "assets": None,
    },
    {
        "prerelease": False,
        "published_at": "2023-04-30T13:24:32Z",
        "tag_name": "v2.0.0",
        "assets": None,
    },
    {
        "prerelease": True,
        "published_at": "2023-01-19T03:34:05Z",
        "tag_name": "v1.5.3xd",
        "assets": None,
    },
    {
        "prerelease": False,
        "published_at": "2027-01-19T03:34:05Z",
        "tag_name": "v10.0.1",
        "assets": None,
    },
]


@pytest.mark.parametrize("mock_response", [(200, _releases_list)], indirect=True)
def test_web_preprocessor_creates_releases(mock_response, context) -> None:
    # [FUNCTION DOC]
    # Verifies that the release preprocessor, given a successful fake response listing six releases (including a prerelease and
    # a far-future one) and with file opening mocked out, produces release names for only the latest patch per minor series, sorted
    # from newest to oldest: ten point zero point one, two point one point three, two point zero point zero, one point five point six. The older 2.1.0
    # and the prerelease are excluded.
    m = mock_open()
    with patch("builtins.open", m):
        context = Preprocessors.home_add_releases(context)
        release_versions = [release["name"] for release in context["releases"]]
        assert release_versions == ["10.0.1", "2.1.3", "2.0.0", "1.5.6"]
