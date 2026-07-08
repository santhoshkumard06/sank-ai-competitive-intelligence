"""LLM client abstraction.

Every agent in agents.py talks to an `LLMClient`, never directly to a
provider's package. That's what makes the agents testable without an API
key (inject `MockLLMClient`) and swappable to a different provider
(implement one more class, like GeminiLLMClient below) without touching
agent logic.

Two real implementations:
  GeminiLLMClient    — gemini-2.5-flash by default. Genuinely free tier,
                       no credit card required (see aistudio.google.com).
                       The CLI defaults to this one for exactly that reason.
  AnthropicLLMClient — claude-sonnet-4-6 by default. Needs a funded account.

Both retry transient failures (429 rate limits, 5xx overload — Gemini's
free tier genuinely does return "high demand" 503s) with backoff, and
fail immediately on non-transient ones (bad key, malformed request) rather
than burning 60+ seconds retrying something that will never succeed.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sank.exceptions import LLMError

logger = logging.getLogger("sank.llm")

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-2.5-flash-lite"

# Kept for backwards compatibility with anything importing the old name.
DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL

# Generous: free-tier "high demand" 503s and 429s are real and documented,
# and can persist for tens of seconds, not just one or two retries' worth.
_RETRY_ATTEMPTS = 6
_RETRY_WAIT = wait_exponential(multiplier=2, min=2, max=60)


class _RetryableProviderError(Exception):
    """Internal marker raised for transient provider failures (429, 5xx)
    so the retry decorator can retry *only* these, not e.g. a bad API key
    or a malformed request — those will never succeed no matter how many
    times we ask, and retrying them just wastes the user's remaining time."""


class LLMClient(ABC):
    """Anything that can turn (system prompt, user message) into text."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024, json_mode: bool = False) -> str:
        """Return the model's raw text response.

        json_mode=True is a hint, not a guarantee: providers that support
        a native structured-output mode (Gemini's response_mime_type)
        should use it. Providers without one (or MockLLMClient) just
        ignore the hint — complete_json()'s defensive parsing below is
        the fallback either way.
        """
        raise NotImplementedError

    def complete_json(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        """Call complete() in JSON mode and parse the result.

        Always requests json_mode=True — on a provider with a real native
        JSON mode (Gemini) this is a hard syntactic guarantee, not just a
        prompt suggestion, which is the fix for a fast/cheap model
        wrapping the JSON in a sentence despite being told not to.

        Still defensive on top of that, in case a provider's "JSON mode"
        wraps the object in markdown fences anyway, or returns a
        partial/truncated object.
        """
        raw = self.complete(system, user, max_tokens=max_tokens, json_mode=True)
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Model did not return valid JSON even in JSON mode. "
                f"Raw output ({len(raw)} chars): {raw[:500]!r}"
            ) from exc


class GeminiLLMClient(LLMClient):
    """Real implementation backed by Google's Gemini API via Google AI Studio.

    Default model is gemini-2.5-flash. As of 2026, Gemini is the only
    major provider with a genuinely permanent free tier: no credit card,
    no expiring trial credits, roughly 15 requests/minute and 1,500
    requests/day on Flash models. Get a key at https://aistudio.google.com
    by signing in with a Google account.

    Two real-world Gemini-specific issues this class deliberately works
    around, both confirmed against current reports, not assumed:

    1. Gemini 2.5/3 Flash models have "thinking" turned on by default, and
       thinking tokens are deducted from max_output_tokens *before* the
       visible answer — well documented (Google's own python-genai repo,
       multiple independent reports) to silently truncate or empty out
       short structured-output responses. Fixed by explicitly setting
       thinking_budget=0 — this task needs classification, not deliberation.
    2. The free tier returns real 429s (rate limit) and 503s ("high
       demand") under load. These are retried with real backoff; anything
       else (bad key, bad request) fails immediately instead.

    Uses the current `google-genai` package (`from google import genai`),
    not the older `google-generativeai` package, which Google has marked
    deprecated/legacy.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        fallback_models: Optional[list[str]] = None,
    ) -> None:
        from google import genai  # lazy import, see AnthropicLLMClient below

        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        # gemini-2.5-flash-lite is a separate, lighter deployment from
        # gemini-2.5-flash — different capacity pool, so a "high demand"
        # 503 on one doesn't necessarily mean the other is also down.
        # De-duplicated and order-preserved in case `model` is itself one
        # of the defaults.
        defaults = [DEFAULT_GEMINI_MODEL, DEFAULT_GEMINI_FALLBACK_MODEL]
        self.fallback_models = (
            fallback_models if fallback_models is not None else [m for m in defaults if m != model]
        )

    def complete(self, system: str, user: str, max_tokens: int = 1024, json_mode: bool = False) -> str:
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]
        last_exc: Exception | None = None

        for i, model_name in enumerate(models_to_try):
            try:
                return self._complete_with_one_model(model_name, system, user, max_tokens, json_mode)
            except _RetryableProviderError as exc:
                # Retries on this model are exhausted (real, persistent
                # 429/503), so this specific model's capacity is the
                # problem — worth trying a different model's capacity pool.
                last_exc = exc
                if i < len(models_to_try) - 1:
                    logger.warning(
                        "%s exhausted its retries (%s); falling back to %s",
                        model_name, exc, models_to_try[i + 1],
                    )
            # Anything else (LLMError for a bad key, malformed request,
            # safety block, truncation) is NOT caught here deliberately:
            # those are about the request/account, not which model
            # received it, so the fallback model would fail identically —
            # propagate immediately instead of wasting that call.
        assert last_exc is not None  # models_to_try always has >= 1 entry
        # last_exc here is always a _RetryableProviderError (anything else
        # propagated immediately above, never reaching this line). Wrap it
        # into an LLMError before it leaves this function — _RetryableProviderError
        # is a private implementation detail, not a SankError subclass, and
        # pipeline.py's per-source error isolation only catches SankError.
        # Raising the raw marker here would crash the *entire* run instead
        # of just marking this one source as failed.
        raise LLMError(
            f"All {len(models_to_try)} model(s) ({', '.join(models_to_try)}) exhausted "
            f"their retries: {last_exc}"
        ) from last_exc

    @retry(
        reraise=True,
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=_RETRY_WAIT,
        retry=retry_if_exception_type(_RetryableProviderError),
    )
    def _complete_with_one_model(
        self, model_name: str, system: str, user: str, max_tokens: int, json_mode: bool
    ) -> str:
        from google.genai import errors, types

        config_kwargs = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            # Disable thinking: this pipeline classifies/extracts, it
            # doesn't need multi-step deliberation, and leaving the
            # default on is what causes issue #1 above.
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if json_mode:
            # Real, enforced constraint on Gemini's decoding — not a
            # prompt suggestion — which is what actually fixes "Model did
            # not return valid JSON": Flash-tier models otherwise
            # sometimes wrap the object in a sentence regardless of
            # being told in plain English not to.
            config_kwargs["response_mime_type"] = "application/json"

        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=user,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except errors.APIError as exc:
            if exc.code == 429 or (exc.code and 500 <= exc.code < 600):
                logger.warning("Gemini (%s) HTTP %s (retryable): %s", model_name, exc.code, exc.message)
                raise _RetryableProviderError(f"HTTP {exc.code}: {exc.message}") from exc
            # Auth, bad request, etc — retrying won't help.
            raise LLMError(
                f"Gemini API ({model_name}) rejected the request (HTTP {exc.code}): {exc.message}"
            ) from exc
        except _RetryableProviderError:
            raise
        except Exception as exc:
            # A network-level hiccup (DNS, reset, timeout) rather than a
            # structured API error — still worth a retry, not an
            # immediate failure.
            raise _RetryableProviderError(f"Unexpected error calling Gemini ({model_name}): {exc}") from exc

        candidate = response.candidates[0] if getattr(response, "candidates", None) else None
        finish_reason = str(getattr(candidate, "finish_reason", "")) if candidate else ""

        if "MAX_TOKENS" in finish_reason:
            raise LLMError(
                f"Gemini ({model_name}) hit max_output_tokens ({max_tokens}) before "
                f"finishing the response. Thinking is already disabled here, so this "
                f"means the actual content needs more room — try raising max_tokens."
            )
        if not response.text:
            raise LLMError(f"Gemini ({model_name}) response had no text (finish_reason={finish_reason!r}).")
        return response.text


class AnthropicLLMClient(LLMClient):
    """Real implementation backed by the Anthropic Messages API.

    Needs a funded Anthropic console account — there is no permanent free
    tier as of this writing. Use GeminiLLMClient if you don't have credits.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        # Imported lazily so the rest of the package — and all tests that
        # use MockLLMClient — never require the `anthropic` package or an
        # API key to be importable.
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    @retry(
        reraise=True,
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=_RETRY_WAIT,
        retry=retry_if_exception_type(_RetryableProviderError),
    )
    def complete(self, system: str, user: str, max_tokens: int = 1024, json_mode: bool = False) -> str:
        import anthropic  # local import, see __init__

        # json_mode is accepted for interface compatibility but unused:
        # Claude follows an explicit "respond with only JSON" instruction
        # reliably enough in practice that this hasn't needed a forced
        # mode. If that ever changes, forcing a single JSON-schema tool
        # choice is the equivalent native mechanism to reach for.
        del json_mode

        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIStatusError as exc:
            status = exc.status_code
            if status == 429 or (status and 500 <= status < 600):
                raise _RetryableProviderError(f"HTTP {status}: {exc}") from exc
            raise LLMError(f"Anthropic API returned an error: {exc}") from exc
        except _RetryableProviderError:
            raise
        except anthropic.APIConnectionError as exc:
            raise _RetryableProviderError(f"Connection error: {exc}") from exc

        text_parts = [block.text for block in message.content if block.type == "text"]
        if not text_parts:
            raise LLMError("Anthropic response contained no text content block")
        return "".join(text_parts)


class MockLLMClient(LLMClient):
    """Deterministic stand-in for tests and offline development.

    `responses` can be:
      - a single string: always returned
      - a list of strings: returned in order, one per call (cycles if exhausted)
      - a callable(system, user) -> str: full control for edge-case tests
    """

    def __init__(self, responses: "str | list[str] | Callable[[str, str], str]") -> None:
        self._responses = responses
        self._call_count = 0
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1024, json_mode: bool = False) -> str:
        del json_mode  # accepted for interface compatibility, not needed by the mock
        self.calls.append((system, user))
        if callable(self._responses):
            result = self._responses(system, user)
        elif isinstance(self._responses, str):
            result = self._responses
        else:
            if not self._responses:
                raise LLMError("MockLLMClient has no responses configured")
            result = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return result


def make_default_llm_client(api_key: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """Convenience factory: Gemini if a Gemini key is present (the free,
    no-card path), otherwise Anthropic. cli.py uses an explicit --provider
    flag rather than this factory, so the choice is always visible in the
    command rather than silently inferred — this function exists for
    scripts/notebooks that just want a sensible default.
    """
    import os

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return GeminiLLMClient(api_key=api_key, **({"model": model} if model else {}))
    return AnthropicLLMClient(api_key=api_key, **({"model": model} if model else {}))
