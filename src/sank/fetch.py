"""Fetching and cleaning public source pages.

Deliberately split into two layers:

  fetch_raw_html()  - the only function that touches the network. Retries
                       with backoff via tenacity, raises FetchError on
                       final failure. Cannot be meaningfully unit-tested
                       without a live network call or an HTTP mock server,
                       so tests exercise it via a monkeypatched httpx client.

  clean_html()       - a pure function: HTML string in, plain text out.
                       Fully unit-testable with static fixture files, no
                       network needed. This is where almost all the real
                       logic (and almost all the bugs) live, so it's kept
                       separate on purpose.

fetch_and_clean() wires the two together into a RawSignal.
"""

from __future__ import annotations

import logging
import re

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sank.exceptions import FetchError
from sank.models import RawSignal, Source

logger = logging.getLogger("sank.fetch")

# A real browser-like user agent. Many sites 403 the default httpx/requests
# UA outright, which is the single most common reason this kind of fetch
# silently "fails" — worth getting right.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 SankBot/0.1 "
    "(+https://github.com/; respectful low-frequency polling)"
)

# Tags whose text is never the actual content of the page.
_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "svg", "noscript", "form"]

# A hard cap so one huge page can't blow up token usage downstream when this
# text gets handed to an LLM. ~12k chars is comfortably enough for a
# changelog entry while staying cheap.
MAX_CLEAN_TEXT_CHARS = 12_000


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
)
def fetch_raw_html(url: str, timeout: float = 15.0) -> str:
    """Fetch a URL and return its raw HTML, retrying transient failures.

    Raises FetchError (not a tenacity/httpx exception) on final failure,
    so callers only ever need to catch one exception type.
    """
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    except (httpx.TransportError, httpx.HTTPStatusError):
        # Let tenacity see and retry the real exception type.
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise FetchError(f"Unexpected error fetching {url}: {exc}", url=url) from exc


def clean_html(raw_html: str, max_chars: int = MAX_CLEAN_TEXT_CHARS) -> str:
    """Turn raw HTML into plain, whitespace-normalized text.

    Pure function — no network, fully deterministic, fully unit-testable.
    """
    if not raw_html or not raw_html.strip():
        return ""

    soup = BeautifulSoup(raw_html, "lxml")

    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines and trailing/leading whitespace per line.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n...[truncated]"

    return cleaned


def fetch_and_clean(source: Source, entity_name: str) -> RawSignal:
    """Fetch one Source and return a RawSignal, or raise FetchError.

    RSS/manual fetch methods are not yet implemented in v1 — they raise a
    clear FetchError rather than silently doing nothing, so a misconfigured
    watchlist fails loudly instead of producing an empty, misleading digest.
    """
    if source.fetch_method.value != "html":
        raise FetchError(
            f"fetch_method={source.fetch_method.value!r} is not implemented yet "
            f"in this v1 (only 'html' is). Use a manual paste step for now, "
            f"or extend fetch.py.",
            url=str(source.url),
        )

    try:
        raw_html = fetch_raw_html(str(source.url))
    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
        raise FetchError(
            f"Failed to fetch {source.url} for {entity_name} after retries: {exc}",
            url=str(source.url),
        ) from exc

    clean_text = clean_html(raw_html)
    if not clean_text:
        raise FetchError(
            f"Fetched {source.url} but extracted no readable text "
            f"(page may be JS-rendered or blocked).",
            url=str(source.url),
        )

    logger.info("Fetched and cleaned %s (%d chars)", source.url, len(clean_text))
    return RawSignal(entity_name=entity_name, source=source, clean_text=clean_text)
