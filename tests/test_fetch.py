"""Tests for sank.fetch.

clean_html() is a pure function and is tested directly against a realistic
static fixture. fetch_raw_html()/fetch_and_clean() touch the network, so
they're tested by monkeypatching the network call — this sandbox (and most
CI runners) can't reach arbitrary external URLs, and a unit test shouldn't
depend on a live website's uptime anyway.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from sank.exceptions import FetchError
from sank.fetch import clean_html, fetch_and_clean, fetch_raw_html
from sank.models import FetchMethod, Source


class TestCleanHtml:
    def test_strips_script_and_style(self, sample_changelog_html):
        cleaned = clean_html(sample_changelog_html)
        assert "console.log" not in cleaned
        assert "color: red" not in cleaned

    def test_strips_nav_and_footer(self, sample_changelog_html):
        cleaned = clean_html(sample_changelog_html)
        assert "Home" not in cleaned
        assert "Terms" not in cleaned
        assert "All rights reserved" not in cleaned

    def test_keeps_real_content(self, sample_changelog_html):
        cleaned = clean_html(sample_changelog_html)
        assert "Design Mode" in cleaned
        assert "Auto-review" in cleaned

    def test_empty_input_returns_empty_string(self):
        assert clean_html("") == ""
        assert clean_html("   ") == ""

    def test_truncates_overlong_pages(self):
        huge = "<p>" + ("word " * 10_000) + "</p>"
        cleaned = clean_html(huge, max_chars=500)
        assert len(cleaned) <= 520  # 500 + the "...[truncated]" marker
        assert cleaned.endswith("[truncated]")


class TestFetchRawHtml:
    def test_returns_text_on_success(self):
        with patch("httpx.get") as mock_get:
            mock_get.return_value.text = "<html>ok</html>"
            mock_get.return_value.raise_for_status.return_value = None
            result = fetch_raw_html("https://example.com")
        assert result == "<html>ok</html>"

    def test_sends_a_real_browser_user_agent(self):
        """Sites commonly 403 the default httpx UA — this is the single
        most common silent failure for this kind of fetch, worth its own
        test rather than trusting it stayed correct."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.raise_for_status.return_value = None
            mock_get.return_value.text = "ok"
            fetch_raw_html("https://example.com")
        _, kwargs = mock_get.call_args
        assert "Mozilla" in kwargs["headers"]["User-Agent"]

    def test_retries_on_transport_error_then_raises(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("boom")) as mock_get:
            with pytest.raises(httpx.ConnectError):
                fetch_raw_html("https://example.com")
        assert mock_get.call_count == 3  # stop_after_attempt(3)


class TestFetchAndClean:
    def test_happy_path(self, sample_entity, sample_changelog_html):
        with patch("sank.fetch.fetch_raw_html", return_value=sample_changelog_html):
            raw_signal = fetch_and_clean(sample_entity.sources[0], sample_entity.name)
        assert raw_signal.entity_name == "Cursor"
        assert "Design Mode" in raw_signal.clean_text

    def test_network_failure_becomes_fetch_error(self, sample_entity):
        with patch("sank.fetch.fetch_raw_html", side_effect=httpx.ConnectError("down")):
            with pytest.raises(FetchError, match="Cursor"):
                fetch_and_clean(sample_entity.sources[0], sample_entity.name)

    def test_page_with_no_extractable_text_raises_fetch_error(self, sample_entity):
        with patch("sank.fetch.fetch_raw_html", return_value="<html><script>only js</script></html>"):
            with pytest.raises(FetchError, match="no readable text"):
                fetch_and_clean(sample_entity.sources[0], sample_entity.name)

    def test_unimplemented_fetch_method_fails_loudly(self):
        source = Source(url="https://example.com/feed", fetch_method=FetchMethod.RSS)
        with pytest.raises(FetchError, match="not implemented"):
            fetch_and_clean(source, "SomeEntity")
