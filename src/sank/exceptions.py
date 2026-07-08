"""Custom exception hierarchy for Sank.

Every error raised by this package is a subclass of SankError, so callers
can do a single `except SankError` if they just want to know "something in
the pipeline failed", or catch a specific subclass if they need to react
differently (e.g. retry a FetchError but abort on a ConfigError).
"""

from __future__ import annotations


class SankError(Exception):
    """Base class for every exception raised by the sank package."""


class ConfigError(SankError):
    """Raised when a YAML/JSON config file is missing, malformed, or fails validation."""


class FetchError(SankError):
    """Raised when retrieving or cleaning a source document fails.

    Carries the offending URL so the pipeline can log which source broke
    without killing the whole run (see pipeline.run_pipeline's per-source
    isolation).
    """

    def __init__(self, message: str, url: str | None = None) -> None:
        super().__init__(message)
        self.url = url


class LLMError(SankError):
    """Raised when an LLM call fails or returns something we can't parse."""


class VectorStoreError(SankError):
    """Raised when an embedding upsert or similarity query fails."""


class ExtractionError(SankError):
    """Raised when Agent 1 (signal extraction) cannot produce valid output."""


class ScoringError(SankError):
    """Raised when Agent 2 (threat/relevance scoring) cannot produce valid output."""
