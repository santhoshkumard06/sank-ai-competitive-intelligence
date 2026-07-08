"""Tests for sank.agents — the three reasoning steps."""

from __future__ import annotations

import pytest

from sank.agents import BriefingAgent, ExtractionAgent, ScoringAgent
from sank.exceptions import ExtractionError, ScoringError
from sank.llm_client import MockLLMClient
from sank.models import ExtractedSignal, RawSignal, Severity
from sank.vector_store import LocalVectorStore


def _refuse_to_be_called(*_args, **_kwargs):
    raise AssertionError("LLM should not have been called for this case")


class TestExtractionAgent:
    def test_extracts_a_meaningful_signal(self, domain_config, sample_entity):
        llm = MockLLMClient(
            '{"skipped": false, "date": "2026-06-10", "kind": "feature", '
            '"raw_excerpt": "x", "plain_summary": "Shipped a visual editor."}'
        )
        agent = ExtractionAgent(llm, domain_config)
        raw = RawSignal(entity_name="Cursor", source=sample_entity.sources[0], clean_text="...")
        result = agent.run(raw)
        assert result.skipped is False
        assert result.plain_summary == "Shipped a visual editor."

    def test_marks_uninteresting_pages_as_skipped(self, domain_config, sample_entity):
        llm = MockLLMClient('{"skipped": true, "skip_reason": "Only a typo fix"}')
        agent = ExtractionAgent(llm, domain_config)
        raw = RawSignal(entity_name="Cursor", source=sample_entity.sources[0], clean_text="...")
        result = agent.run(raw)
        assert result.skipped is True

    def test_malformed_json_raises_extraction_error(self, domain_config, sample_entity):
        llm = MockLLMClient("this is not json")
        agent = ExtractionAgent(llm, domain_config)
        raw = RawSignal(entity_name="Cursor", source=sample_entity.sources[0], clean_text="...")
        with pytest.raises(ExtractionError):
            agent.run(raw)

    def test_wrong_shaped_json_raises_extraction_error(self, domain_config, sample_entity):
        """The exact bug found during manual testing: valid JSON, wrong shape."""
        llm = MockLLMClient('{"severity": "high", "reason": "wrong agent"}')
        agent = ExtractionAgent(llm, domain_config)
        raw = RawSignal(entity_name="Cursor", source=sample_entity.sources[0], clean_text="...")
        with pytest.raises(ExtractionError):
            agent.run(raw)


class TestScoringAgent:
    def test_scores_a_real_signal_using_the_vector_match(self, domain_config, sample_reference_items):
        vs = LocalVectorStore()
        vs.index_reference_corpus(sample_reference_items)
        llm = MockLLMClient('{"severity": "high", "reason": "Overlaps our roadmap."}')
        agent = ScoringAgent(llm, vs, domain_config)
        signal = ExtractedSignal(entity_name="Cursor", plain_summary="real-time team collaboration")
        scored = agent.run(signal)
        assert scored.severity == Severity.HIGH
        assert scored.best_match.reference_item_id == "rm-002"

    def test_skipped_signal_never_calls_llm_or_vector_store(self, domain_config):
        vs = LocalVectorStore()  # deliberately never indexed
        llm = MockLLMClient(_refuse_to_be_called)
        agent = ScoringAgent(llm, vs, domain_config)
        signal = ExtractedSignal(entity_name="Cursor", skipped=True, skip_reason="typo fix")
        scored = agent.run(signal)  # must not raise, must not touch llm/vs
        assert scored.severity == Severity.LOW
        assert scored.reason == "typo fix"

    def test_llm_failure_raises_scoring_error(self, domain_config, sample_reference_items):
        vs = LocalVectorStore()
        vs.index_reference_corpus(sample_reference_items)
        llm = MockLLMClient("not valid json")
        agent = ScoringAgent(llm, vs, domain_config)
        signal = ExtractedSignal(entity_name="Cursor", plain_summary="something")
        with pytest.raises(ScoringError):
            agent.run(signal)


class TestBriefingAgent:
    def test_writes_a_digest_from_scored_signals(self, domain_config):
        from sank.models import MatchResult, ScoredSignal

        llm = MockLLMClient("Cursor shipped a visual editor. High severity.")
        agent = BriefingAgent(llm, domain_config)
        scored = ScoredSignal(
            signal=ExtractedSignal(entity_name="Cursor", plain_summary="Shipped editor"),
            best_match=MatchResult(reference_item_id="rm-001", reference_item_text="x", similarity=0.9),
            severity=Severity.HIGH,
            reason="Matches our roadmap",
        )
        digest = agent.run([scored])
        assert "Cursor" in digest.summary_text
        assert digest.high_severity_count() == 1

    def test_all_skipped_signals_skip_the_llm_call_entirely(self, domain_config):
        llm = MockLLMClient(_refuse_to_be_called)
        agent = BriefingAgent(llm, domain_config)
        skipped_signal = ExtractedSignal(entity_name="Cursor", skipped=True, skip_reason="nothing")
        from sank.models import ScoredSignal

        digest = agent.run([ScoredSignal(signal=skipped_signal)])
        assert "Nothing notable" in digest.summary_text

    def test_high_severity_items_are_sent_to_the_llm_first(self, domain_config):
        """Verifies the deterministic pre-sort: the pipeline shouldn't rely
        on the LLM alone to prioritize correctly."""
        from sank.models import ScoredSignal

        captured_payload = {}

        def capture(system, user):
            captured_payload["user"] = user
            return "digest text"

        llm = MockLLMClient(capture)
        agent = BriefingAgent(llm, domain_config)
        low = ScoredSignal(
            signal=ExtractedSignal(entity_name="LowCo", plain_summary="minor thing"),
            severity=Severity.LOW,
            reason="minor",
        )
        high = ScoredSignal(
            signal=ExtractedSignal(entity_name="HighCo", plain_summary="big thing"),
            severity=Severity.HIGH,
            reason="big",
        )
        agent.run([low, high])
        # HighCo's entry should appear before LowCo's in the serialized payload
        assert captured_payload["user"].index("HighCo") < captured_payload["user"].index("LowCo")
