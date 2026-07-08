"""The three agents. Domain-agnostic: every word that's specific to
"competitive intelligence" vs "creator comparison" lives in the
DomainConfig's prompt strings, never in this file.

ExtractionAgent  : RawSignal            -> ExtractedSignal
ScoringAgent     : ExtractedSignal      -> ScoredSignal   (calls the vector store)
BriefingAgent    : list[ScoredSignal]   -> Digest
"""

from __future__ import annotations

import json
import logging

from sank.config import DomainConfig
from sank.exceptions import ExtractionError, ScoringError
from sank.llm_client import LLMClient
from sank.models import (
    Digest,
    ExtractedSignal,
    RawSignal,
    ScoredSignal,
    Severity,
)
from sank.vector_store import VectorStore

logger = logging.getLogger("sank.agents")


class ExtractionAgent:
    """Reads unstructured page text, decides if anything meaningful happened."""

    def __init__(self, llm: LLMClient, domain: DomainConfig) -> None:
        self._llm = llm
        self._domain = domain

    def run(self, raw_signal: RawSignal) -> ExtractedSignal:
        try:
            parsed = self._llm.complete_json(
                system=self._domain.extraction_system_prompt,
                user=raw_signal.clean_text,
            )
        except Exception as exc:
            raise ExtractionError(
                f"Extraction failed for {raw_signal.entity_name} "
                f"({raw_signal.source.url}): {exc}"
            ) from exc

        try:
            return ExtractedSignal(entity_name=raw_signal.entity_name, **parsed)
        except Exception as exc:
            raise ExtractionError(
                f"Model output for {raw_signal.entity_name} didn't match the "
                f"expected schema: {parsed!r} ({exc})"
            ) from exc


class ScoringAgent:
    """Judges whether an extracted signal threatens/matters to our reference corpus."""

    def __init__(self, llm: LLMClient, vector_store: VectorStore, domain: DomainConfig) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._domain = domain

    def run(self, signal: ExtractedSignal) -> ScoredSignal:
        if signal.skipped:
            # No meaningful change was extracted — no point spending an LLM
            # call or a vector query judging "how severe is nothing".
            return ScoredSignal(
                signal=signal,
                best_match=None,
                severity=Severity.LOW,
                reason=signal.skip_reason or "No meaningful change detected.",
            )

        try:
            match = self._vector_store.find_best_match(signal.plain_summary, top_k=1)[0]
        except Exception as exc:
            raise ScoringError(
                f"Vector match failed for {signal.entity_name}: {exc}"
            ) from exc

        user_payload = json.dumps(
            {
                "signal_summary": signal.plain_summary,
                "matched_reference_item": match.reference_item_text,
                "similarity": round(match.similarity, 3),
            }
        )

        try:
            parsed = self._llm.complete_json(
                system=self._domain.scoring_system_prompt, user=user_payload
            )
            return ScoredSignal(
                signal=signal,
                best_match=match,
                severity=Severity(parsed["severity"]),
                reason=parsed["reason"],
            )
        except Exception as exc:
            raise ScoringError(
                f"Scoring failed for {signal.entity_name}: {exc}"
            ) from exc


class BriefingAgent:
    """Turns today's scored signals into one short, prioritized digest."""

    _SEVERITY_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}

    def __init__(self, llm: LLMClient, domain: DomainConfig) -> None:
        self._llm = llm
        self._domain = domain

    def run(self, scored_signals: list[ScoredSignal]) -> Digest:
        reportable = [s for s in scored_signals if not s.signal.skipped]
        reportable.sort(key=lambda s: self._SEVERITY_ORDER[s.severity])

        if not reportable:
            # Don't spend an LLM call writing a briefing about nothing.
            return Digest(
                entries=scored_signals,
                summary_text="Nothing notable today across the watchlist.",
            )

        payload = json.dumps(
            [
                {
                    "entity_name": s.signal.entity_name,
                    "summary": s.signal.plain_summary,
                    "severity": s.severity.value,
                    "reason": s.reason,
                }
                for s in reportable
            ]
        )
        summary_text = self._llm.complete(
            system=self._domain.briefing_system_prompt, user=payload, max_tokens=600
        )
        return Digest(entries=scored_signals, summary_text=summary_text.strip())
