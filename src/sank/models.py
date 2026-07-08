"""Core data models for Sank.

These models are intentionally domain-agnostic. The word "competitor" never
appears here — it's `Entity`. The word "roadmap" never appears here — it's
`ReferenceItem`. That's what lets the exact same pipeline run a competitive
intelligence workflow today and a creator-comparison or pricing-watch
workflow tomorrow, with no code changes — only a new domain config and a
new watchlist (see config.py and config/domains/).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class SourceType(str, Enum):
    """What kind of public page a Source points at.

    This is used only to give the LLM agents a hint about what they're
    reading — it does not change how fetching/cleaning works.
    """

    CHANGELOG = "changelog"
    PRICING = "pricing"
    REVIEWS = "reviews"
    SOCIAL = "social"
    BLOG = "blog"
    JOB_POSTINGS = "job_postings"
    OTHER = "other"


class FetchMethod(str, Enum):
    """How a Source should be retrieved."""

    HTML = "html"
    RSS = "rss"
    MANUAL = "manual"  # text pasted/maintained by the user instead of fetched


class Source(BaseModel):
    """One concrete page or feed to monitor for a single Entity."""

    url: HttpUrl
    source_type: SourceType = SourceType.OTHER
    fetch_method: FetchMethod = FetchMethod.HTML
    label: Optional[str] = Field(
        default=None, description="Human-readable note, e.g. 'public changelog'."
    )


class Entity(BaseModel):
    """Something being watched — a competitor, a creator, a product, anything.

    `category` is free text used purely for display/grouping; the pipeline
    logic never branches on it.
    """

    name: str
    category: Optional[str] = None
    sources: list[Source] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Entity.name cannot be blank")
        return v.strip()


class ReferenceItem(BaseModel):
    """One item in the user's own reference corpus.

    For competitive intelligence this is a roadmap item. For a
    creator-comparison domain it might be one of *your* content pillars.
    Whatever it is, it gets embedded once and every incoming signal is
    matched against the whole set.
    """

    id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class RawSignal(BaseModel):
    """The output of fetch + clean: plain text, not yet interpreted."""

    entity_name: str
    source: Source
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    clean_text: str


class SignalKind(str, Enum):
    FEATURE = "feature"
    PRICING = "pricing"
    HIRING = "hiring"
    REVIEW_SIGNAL = "review_signal"
    OTHER = "other"


class ExtractedSignal(BaseModel):
    """Agent 1's output: a raw page turned into one structured, meaningful change.

    `skipped=True` means Agent 1 read the page and decided nothing
    meaningful happened (e.g. a typo fix) — this is a normal, expected
    outcome, not an error.

    `extra="forbid"` and the validator below are deliberate: if the LLM
    (or a test, or a future caller) hands this model JSON that doesn't
    actually match the extraction contract — e.g. fields from a *different*
    agent's response shape — we want a loud, immediate, easy-to-diagnose
    ValidationError right here, not a silently-defaulted, hollow signal
    that fails confusingly three steps downstream instead.
    """

    model_config = ConfigDict(extra="forbid")

    entity_name: str
    date: Optional[str] = None
    kind: SignalKind = SignalKind.OTHER
    raw_excerpt: str = ""
    plain_summary: str = ""
    skipped: bool = False
    skip_reason: Optional[str] = None

    @model_validator(mode="after")
    def _non_skipped_must_have_summary(self) -> "ExtractedSignal":
        if not self.skipped and not self.plain_summary.strip():
            raise ValueError(
                "ExtractedSignal is not marked skipped but plain_summary is "
                "empty — this almost always means the upstream JSON didn't "
                "match the expected extraction schema."
            )
        return self


class MatchResult(BaseModel):
    """One nearest-neighbour result from the vector store."""

    reference_item_id: str
    reference_item_text: str
    similarity: float = Field(ge=-1.0, le=1.0)


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ScoredSignal(BaseModel):
    """Agent 2's output: an extracted signal + why it matters (or doesn't)."""

    signal: ExtractedSignal
    best_match: Optional[MatchResult] = None
    severity: Severity = Severity.LOW
    reason: str = ""


class Digest(BaseModel):
    """Agent 3's output: the final, human-readable briefing."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entries: list[ScoredSignal] = Field(default_factory=list)
    summary_text: str = ""

    def high_severity_count(self) -> int:
        return sum(1 for e in self.entries if e.severity == Severity.HIGH)
