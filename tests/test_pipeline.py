"""End-to-end pipeline tests.

test_one_broken_source_does_not_take_down_the_run is the formalized version
of the manual test that caught both real bugs during development (the
float-precision clamp in vector_store.py and the strict validator in
models.py) — keeping it as a permanent regression test, not a one-off script.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from sank.exceptions import SankError
from sank.llm_client import MockLLMClient
from sank.pipeline import run_pipeline
from sank.vector_store import LocalVectorStore


def _fake_fetch(url: str, timeout: float = 15.0) -> str:
    if "lovable" in url:
        return (
            "<html><body><article><h2>Jun 10</h2>"
            "<p>Shipped one-click custom domains with automatic SSL setup.</p>"
            "</article></body></html>"
        )
    if "cursor" in url:
        raise httpx.ConnectError("simulated: site blocked the request")
    if "replit" in url:
        return (
            "<html><body><article><h2>Jun 9</h2>"
            "<p>Fixed a minor CSS alignment issue on the dashboard.</p>"
            "</article></body></html>"
        )
    raise AssertionError(f"unexpected url in test fixture: {url}")


_LLM_RESPONSES = [
    '{"skipped": false, "date": "2026-06-10", "kind": "feature", '
    '"raw_excerpt": "one-click custom domains", '
    '"plain_summary": "Shipped one-click custom domain connection with automatic SSL."}',
    '{"severity": "high", "reason": "This is exactly our planned custom-domain roadmap item."}',
    '{"skipped": true, "skip_reason": "Only a minor CSS fix"}',
    "Lovable shipped one-click custom domains with auto SSL — directly overlaps "
    "our own roadmap item. Nothing else notable today.",
]


@pytest.fixture
def watchlist():
    from sank.config import load_watchlist
    from pathlib import Path

    return load_watchlist(Path(__file__).parent.parent / "config" / "watchlist.example.yaml")


@pytest.fixture
def reference_corpus():
    from sank.config import load_reference_corpus
    from pathlib import Path

    return load_reference_corpus(
        Path(__file__).parent.parent / "config" / "reference_corpus.example.yaml"
    )


class TestPipeline:
    def test_one_broken_source_does_not_take_down_the_run(
        self, domain_config, watchlist, reference_corpus
    ):
        llm = MockLLMClient(_LLM_RESPONSES)
        vs = LocalVectorStore()

        with patch("sank.fetch.fetch_raw_html", side_effect=_fake_fetch):
            result = run_pipeline(watchlist, reference_corpus, domain_config, llm, vs)

        assert result.sources_attempted == 3
        assert result.sources_succeeded == 2
        assert len(result.errors) == 1
        assert "Cursor" in result.errors[0]
        assert result.digest.high_severity_count() == 1
        assert "Lovable" in result.digest.summary_text

    def test_a_fully_clean_run_has_no_errors(self, domain_config, watchlist, reference_corpus):
        def all_succeed(url: str, timeout: float = 15.0) -> str:
            return "<html><body><p>Shipped one-click custom domains with SSL.</p></body></html>"

        responses = [
            '{"skipped": false, "plain_summary": "Shipped one-click custom domains."}',
            '{"severity": "medium", "reason": "Partial overlap."}',
        ] * 3 + ["All quiet, nothing urgent."]
        llm = MockLLMClient(responses)
        vs = LocalVectorStore()

        with patch("sank.fetch.fetch_raw_html", side_effect=all_succeed):
            result = run_pipeline(watchlist, reference_corpus, domain_config, llm, vs)

        assert result.errors == []
        assert result.sources_succeeded == result.sources_attempted == 3

    def test_total_gemini_exhaustion_is_isolated_not_a_crash(
        self, domain_config, watchlist, reference_corpus
    ):
        """Integration test across the llm_client.py / pipeline.py boundary:
        if Gemini is down hard enough that every model and every retry
        fails, that must still only cost the pipeline this one source —
        not crash the whole run. This is the exact bug found during
        development: the fallback loop's final exception wasn't a
        SankError, so it wasn't caught by pipeline.py's isolation."""
        from unittest.mock import patch as mock_patch

        from google.genai import errors

        from sank.llm_client import GeminiLLMClient

        def all_pages_fetch_fine(url: str, timeout: float = 15.0) -> str:
            return "<html><body><p>Shipped one-click custom domains with SSL.</p></body></html>"

        with mock_patch("google.genai.Client") as MockClient, mock_patch(
            "sank.fetch.fetch_raw_html", side_effect=all_pages_fetch_fine
        ), mock_patch("time.sleep", return_value=None):
            MockClient.return_value.models.generate_content.side_effect = errors.ServerError(
                503, {"message": "This model is currently experiencing high demand."}
            )
            llm = GeminiLLMClient(api_key="dummy")
            vs = LocalVectorStore()
            result = run_pipeline(watchlist, reference_corpus, domain_config, llm, vs)

        # Every source failed (Gemini is "fully down" in this scenario),
        # but the call returned a result rather than raising — that's the
        # isolation property being tested.
        assert result.sources_succeeded == 0
        assert result.sources_attempted == 3
        assert len(result.errors) == 3
        assert "Nothing notable" in result.digest.summary_text

    def test_an_empty_reference_corpus_fails_fast_and_clearly(self, domain_config, watchlist):
        """A misconfigured run (no reference corpus at all) should fail
        immediately with a clear error, not produce a confusing partial
        digest three steps later."""
        llm = MockLLMClient("irrelevant")
        vs = LocalVectorStore()
        with pytest.raises(SankError):
            run_pipeline(watchlist, [], domain_config, llm, vs)
