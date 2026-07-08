"""Orchestrates the full run: fetch -> extract -> score -> brief.

The one piece of "industrial" behavior that matters most here: a single
broken source (site redesign, transient 500, blocked request) must not
take down the whole run. Each (entity, source) pair fails in isolation;
its error is collected and surfaced in PipelineResult.errors, and the
digest still gets built from everything that succeeded. This is the
"edge case" called out explicitly in the PRD: a page blocking automated
fetching should degrade gracefully, not crash the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sank.agents import BriefingAgent, ExtractionAgent, ScoringAgent
from sank.config import DomainConfig
from sank.exceptions import SankError
from sank.fetch import fetch_and_clean
from sank.llm_client import LLMClient
from sank.models import Digest, Entity, ReferenceItem, ScoredSignal
from sank.vector_store import VectorStore

logger = logging.getLogger("sank.pipeline")


@dataclass
class PipelineResult:
    digest: Digest
    scored_signals: list[ScoredSignal]
    errors: list[str] = field(default_factory=list)
    sources_attempted: int = 0
    sources_succeeded: int = 0


def run_pipeline(
    entities: list[Entity],
    reference_corpus: list[ReferenceItem],
    domain: DomainConfig,
    llm: LLMClient,
    vector_store: VectorStore,
) -> PipelineResult:
    """Run one full Sank cycle across every Source of every Entity.

    Raises SankError only for failures that make the *entire* run
    meaningless (e.g. the reference corpus itself fails to index).
    Per-source failures are caught, logged, and collected — they do not
    raise.
    """
    vector_store.index_reference_corpus(reference_corpus)

    extraction_agent = ExtractionAgent(llm, domain)
    scoring_agent = ScoringAgent(llm, vector_store, domain)
    briefing_agent = BriefingAgent(llm, domain)

    scored_signals: list[ScoredSignal] = []
    errors: list[str] = []
    attempted = 0
    succeeded = 0

    for entity in entities:
        for source in entity.sources:
            attempted += 1
            try:
                raw = fetch_and_clean(source, entity.name)
                extracted = extraction_agent.run(raw)
                scored = scoring_agent.run(extracted)
                scored_signals.append(scored)
                succeeded += 1
            except SankError as exc:
                msg = f"{entity.name} ({source.url}): {exc}"
                logger.warning("Source failed, continuing run: %s", msg)
                errors.append(msg)
                continue

    digest = briefing_agent.run(scored_signals)

    return PipelineResult(
        digest=digest,
        scored_signals=scored_signals,
        errors=errors,
        sources_attempted=attempted,
        sources_succeeded=succeeded,
    )
