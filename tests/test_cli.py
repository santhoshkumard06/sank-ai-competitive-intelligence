"""Tests for the CLI, using Click's official CliRunner (no subprocess
needed). Covers the three things most likely to break for a real user:
a missing API key, an invalid config file, and the full happy path."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner

from sank.cli import cli
from sank.llm_client import MockLLMClient

CONFIG = dict(
    watchlist="config/watchlist.example.yaml",
    reference="config/reference_corpus.example.yaml",
    domain="config/domains/competitive_intelligence.yaml",
)


@pytest.fixture
def runner():
    return CliRunner()


class TestValidate:
    def test_valid_watchlist(self, runner):
        result = runner.invoke(cli, ["validate", "--watchlist", CONFIG["watchlist"]])
        assert result.exit_code == 0
        assert "3 entities" in result.output

    def test_invalid_watchlist_exits_nonzero_with_clear_message(self, runner, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text('entities:\n  - name: ""\n    sources: []\n')
        result = runner.invoke(cli, ["validate", "--watchlist", str(bad_file)])
        assert result.exit_code == 1
        assert "Invalid" in result.output


class TestRun:
    def test_missing_gemini_key_fails_clearly_not_with_a_traceback(self, runner, monkeypatch):
        """gemini is the default provider (genuine free tier, no card needed)."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = runner.invoke(cli, ["run", *_flatten(CONFIG)])
        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_missing_anthropic_key_fails_clearly_when_that_provider_is_chosen(self, runner, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(cli, ["run", *_flatten(CONFIG), "--provider", "anthropic"])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output

    def test_full_happy_path_through_the_cli_with_gemini(self, runner, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-for-test")

        def fake_fetch(url: str, timeout: float = 15.0) -> str:
            if "lovable" in url:
                return "<p>Shipped one-click custom domains with automatic SSL.</p>"
            if "cursor" in url:
                raise httpx.ConnectError("blocked")
            return "<p>Fixed a minor CSS issue.</p>"

        responses = [
            '{"skipped": false, "plain_summary": "Shipped one-click custom domains with SSL."}',
            '{"severity": "high", "reason": "Matches our roadmap directly."}',
            '{"skipped": true, "skip_reason": "minor css fix"}',
            "Lovable shipped custom domains — high severity, matches our roadmap.",
        ]

        with patch("sank.cli.GeminiLLMClient", return_value=MockLLMClient(responses)), patch(
            "sank.fetch.fetch_raw_html", side_effect=fake_fetch
        ):
            result = runner.invoke(cli, ["run", *_flatten(CONFIG)])

        assert result.exit_code == 0
        assert "Lovable shipped custom domains" in result.output
        assert "2/3 succeeded" in result.output

    def test_full_happy_path_through_the_cli_with_anthropic(self, runner, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-test")

        def fake_fetch(url: str, timeout: float = 15.0) -> str:
            return "<p>Shipped one-click custom domains with automatic SSL.</p>"

        responses = [
            '{"skipped": false, "plain_summary": "Shipped one-click custom domains with SSL."}',
            '{"severity": "high", "reason": "Matches our roadmap directly."}',
        ] * 3 + ["All quiet today."]

        with patch("sank.cli.AnthropicLLMClient", return_value=MockLLMClient(responses)), patch(
            "sank.fetch.fetch_raw_html", side_effect=fake_fetch
        ):
            result = runner.invoke(cli, ["run", *_flatten(CONFIG), "--provider", "anthropic"])

        assert result.exit_code == 0
        assert "3/3 succeeded" in result.output


def _flatten(config: dict) -> list[str]:
    flags = []
    for key, value in config.items():
        flags += [f"--{key}", value]
    return flags
