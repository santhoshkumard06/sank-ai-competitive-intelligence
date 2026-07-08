"""Tests for sank.models — the data contracts everything else relies on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sank.models import (
    Entity,
    ExtractedSignal,
    MatchResult,
    ReferenceItem,
    ScoredSignal,
    Severity,
    Source,
)


class TestEntity:
    def test_valid_entity(self):
        e = Entity(name="Lovable", sources=[Source(url="https://lovable.dev/changelog")])
        assert e.name == "Lovable"
        assert len(e.sources) == 1

    def test_blank_name_rejected(self):
        with pytest.raises(ValidationError):
            Entity(name="   ", sources=[])

    def test_name_is_stripped(self):
        e = Entity(name="  Cursor  ", sources=[])
        assert e.name == "Cursor"

    def test_invalid_url_rejected(self):
        with pytest.raises(ValidationError):
            Source(url="not-a-url")


class TestExtractedSignal:
    def test_skipped_signal_does_not_need_a_summary(self):
        sig = ExtractedSignal(entity_name="Lovable", skipped=True, skip_reason="typo fix")
        assert sig.skipped is True

    def test_non_skipped_signal_requires_a_summary(self):
        """This is the exact bug found during manual pipeline testing:
        a wrongly-shaped LLM response must fail loudly here, not produce
        a hollow 'valid' signal that breaks confusingly downstream."""
        with pytest.raises(ValidationError, match="plain_summary"):
            ExtractedSignal(entity_name="Lovable", skipped=False, plain_summary="")

    def test_unexpected_fields_are_rejected(self):
        """Guards against an agent receiving another agent's response shape
        (e.g. a scoring response handed to the extraction model)."""
        with pytest.raises(ValidationError):
            ExtractedSignal(entity_name="Lovable", severity="high", reason="wrong shape")

    def test_well_formed_signal(self):
        sig = ExtractedSignal(
            entity_name="Lovable",
            kind="feature",
            plain_summary="Shipped one-click custom domains.",
        )
        assert sig.kind.value == "feature"


class TestMatchResult:
    @pytest.mark.parametrize("similarity", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_valid_similarity_range(self, similarity):
        MatchResult(reference_item_id="x", reference_item_text="t", similarity=similarity)

    @pytest.mark.parametrize("similarity", [1.0000000000000002, 1.5, -1.1])
    def test_out_of_range_similarity_rejected(self, similarity):
        """The exact failure mode hit during pipeline testing — a cosine
        similarity computation can return a value a hair outside [-1, 1]
        due to float precision. This model is intentionally strict; the
        caller (vector_store.py) is responsible for clamping first."""
        with pytest.raises(ValidationError):
            MatchResult(reference_item_id="x", reference_item_text="t", similarity=similarity)


class TestScoredSignalAndReferenceItem:
    def test_reference_item_round_trip(self):
        item = ReferenceItem(id="rm-001", text="Some roadmap item", metadata={"pillar": "growth"})
        assert item.metadata["pillar"] == "growth"

    def test_scored_signal_defaults(self):
        sig = ExtractedSignal(entity_name="Lovable", skipped=True, skip_reason="nothing")
        scored = ScoredSignal(signal=sig)
        assert scored.severity == Severity.LOW
        assert scored.best_match is None
