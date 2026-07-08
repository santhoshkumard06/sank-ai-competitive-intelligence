"""Tests for sank.llm_client.

Includes GeminiLLMClient (added as the free-tier alternative after
Anthropic credits ran out during real testing) and the retry/thinking-
budget fixes added after *that* — both found via actual user runs, not
hypothesized in advance, which is why each has a regression test tied to
the specific symptom that was reported.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sank.exceptions import LLMError
from sank.llm_client import AnthropicLLMClient, GeminiLLMClient, MockLLMClient

# Every test in this file that exercises retry logic would otherwise sleep
# for real (the backoff is deliberately generous in production — up to 60s
# — because Gemini's free-tier "high demand" 503s are real and can persist).
# Patching time.sleep keeps the whole suite fast without weakening the
# production backoff at all.
pytestmark = pytest.mark.usefixtures("_no_real_sleeping")


@pytest.fixture(autouse=False)
def _no_real_sleeping():
    with patch("time.sleep", return_value=None):
        yield


class TestMockLLMClient:
    def test_single_string_response(self):
        client = MockLLMClient("hello")
        assert client.complete("s", "u") == "hello"
        assert client.complete("s", "u") == "hello"

    def test_cycles_through_list_of_responses(self):
        client = MockLLMClient(["a", "b"])
        assert [client.complete("s", "u") for _ in range(3)] == ["a", "b", "a"]

    def test_records_every_call(self):
        client = MockLLMClient("x")
        client.complete("sys1", "user1")
        client.complete("sys2", "user2")
        assert client.calls == [("sys1", "user1"), ("sys2", "user2")]

    def test_complete_json_strips_markdown_fences(self):
        client = MockLLMClient('```json\n{"a": 1}\n```')
        assert client.complete_json("s", "u") == {"a": 1}

    def test_complete_json_raises_llm_error_on_invalid_json(self):
        client = MockLLMClient("not json")
        with pytest.raises(LLMError):
            client.complete_json("s", "u")


class TestAnthropicLLMClient:
    def test_constructs_without_a_live_call(self):
        client = AnthropicLLMClient(api_key="dummy")
        assert client.model  # has some default model set

    def test_happy_path_extracts_text_from_content_blocks(self):
        with patch("anthropic.Anthropic") as MockAnthropic:
            instance = MockAnthropic.return_value
            text_block = MagicMock(type="text", text="hello from claude")
            instance.messages.create.return_value = MagicMock(content=[text_block])
            client = AnthropicLLMClient(api_key="dummy")
            assert client.complete("sys", "user") == "hello from claude"

    def test_429_is_retried_then_succeeds(self):
        import anthropic
        import httpx

        def make_status_error(code):
            resp = httpx.Response(status_code=code, request=httpx.Request("POST", "https://api.anthropic.com"))
            return anthropic.APIStatusError("rate limited", response=resp, body=None)

        with patch("anthropic.Anthropic") as MockAnthropic:
            instance = MockAnthropic.return_value
            text_block = MagicMock(type="text", text="ok now")
            instance.messages.create.side_effect = [
                make_status_error(429),
                MagicMock(content=[text_block]),
            ]
            client = AnthropicLLMClient(api_key="dummy")
            assert client.complete("sys", "user") == "ok now"
        assert instance.messages.create.call_count == 2

    def test_bad_request_fails_immediately_without_retrying(self):
        import anthropic
        import httpx

        resp = httpx.Response(status_code=400, request=httpx.Request("POST", "https://api.anthropic.com"))
        with patch("anthropic.Anthropic") as MockAnthropic:
            instance = MockAnthropic.return_value
            instance.messages.create.side_effect = anthropic.APIStatusError(
                "bad request", response=resp, body=None
            )
            client = AnthropicLLMClient(api_key="dummy")
            with pytest.raises(LLMError):
                client.complete("sys", "user")
        assert instance.messages.create.call_count == 1  # no retries wasted on a non-retryable error


class TestGeminiLLMClient:
    """Added as the free-tier fallback: Gemini's free tier needs no credit
    card and no expiring trial credits, unlike Anthropic/OpenAI."""

    def test_constructs_without_a_live_call(self):
        client = GeminiLLMClient(api_key="dummy")
        assert client.model == "gemini-2.5-flash"

    def test_happy_path(self):
        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(text="hello from gemini")
            client = GeminiLLMClient(api_key="dummy")
            result = client.complete("system prompt", "user message", max_tokens=500)
        assert result == "hello from gemini"

    def test_passes_system_instruction_and_max_tokens_correctly(self):
        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(text="ok")
            client = GeminiLLMClient(api_key="dummy", model="gemini-2.5-flash-lite")
            client.complete("be concise", "hello", max_tokens=42)
            kwargs = instance.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash-lite"
        assert kwargs["contents"] == "hello"
        assert kwargs["config"].system_instruction == "be concise"
        assert kwargs["config"].max_output_tokens == 42

    def test_thinking_is_disabled_on_every_call(self):
        """Regression test for the exact reported bug: Gemini 2.5 Flash
        has 'thinking' on by default, and thinking tokens are deducted
        from max_output_tokens *before* the visible answer — which is
        what produced the truncated '{ "skipped": false, "date": ...'
        response the user saw. Must be off on every call, not just
        json_mode ones (the briefing agent's plain-prose call can be hit
        by this too)."""
        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(text="ok")
            client = GeminiLLMClient(api_key="dummy")
            client.complete("s", "u")
            config = instance.models.generate_content.call_args.kwargs["config"]
        assert config.thinking_config.thinking_budget == 0

    def test_max_tokens_finish_reason_raises_a_specific_error(self):
        """If truncation still happens (e.g. thinking_budget being
        ignored, which has been reported upstream), the error should name
        the actual cause instead of falling through to a generic
        'invalid JSON' message that hides what's really going on."""
        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            candidate = MagicMock(finish_reason="MAX_TOKENS")
            instance.models.generate_content.return_value = MagicMock(
                text='{"skipped": false, "date": "2026-06-17", "', candidates=[candidate]
            )
            client = GeminiLLMClient(api_key="dummy")
            with pytest.raises(LLMError, match="max_output_tokens"):
                client.complete("s", "u")

    def test_429_is_retried_then_succeeds(self):
        from google.genai import errors

        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.side_effect = [
                errors.ClientError(429, {"message": "quota exceeded"}),
                MagicMock(text="ok now"),
            ]
            client = GeminiLLMClient(api_key="dummy")
            result = client.complete("s", "u")
        assert result == "ok now"
        assert instance.models.generate_content.call_count == 2

    def test_falls_back_to_second_model_when_first_exhausts_retries(self):
        """The implementation tries gemini-2.5-flash-lite automatically if
        gemini-2.5-flash's retries are exhausted — a 'high demand' 503 on
        one deployment doesn't mean the other is also overloaded. This
        was implemented but had no direct test; closing that gap."""
        from google.genai import errors

        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            always_503 = errors.ServerError(503, {"message": "high demand"})
            instance.models.generate_content.side_effect = [
                always_503,  # flash, attempt 1
                always_503,  # flash, attempt 2
                always_503,  # flash, attempt 3
                always_503,  # flash, attempt 4
                always_503,  # flash, attempt 5
                always_503,  # flash, attempt 6 (retry budget exhausted)
                MagicMock(text="ok from the fallback model"),  # flash-lite, attempt 1
            ]
            client = GeminiLLMClient(api_key="dummy")
            result = client.complete("s", "u")

        assert result == "ok from the fallback model"
        models_called = [c.kwargs["model"] for c in instance.models.generate_content.call_args_list]
        assert models_called == ["gemini-2.5-flash"] * 6 + ["gemini-2.5-flash-lite"]

    def test_explicit_fallback_models_list_is_respected(self):
        with patch("google.genai.Client"):
            client = GeminiLLMClient(api_key="dummy", model="gemini-2.5-pro", fallback_models=["gemini-2.5-flash"])
        assert client.fallback_models == ["gemini-2.5-flash"]

    def test_503_high_demand_is_retried_then_succeeds(self):
        """The exact symptom reported: 'This model is currently
        experiencing high demand' is a 503, which must be retried, not
        failed immediately."""
        from google.genai import errors

        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.side_effect = [
                errors.ServerError(503, {"message": "This model is currently experiencing high demand."}),
                errors.ServerError(503, {"message": "This model is currently experiencing high demand."}),
                MagicMock(text="ok now"),
            ]
            client = GeminiLLMClient(api_key="dummy")
            result = client.complete("s", "u")
        assert result == "ok now"
        assert instance.models.generate_content.call_count == 3

    def test_persistent_429_eventually_gives_up_with_a_clear_error(self):
        """Persistent 429 on the primary model is exactly the case the
        fallback model exists for — it should try gemini-2.5-flash-lite
        too (a different quota pool) before giving up. The final error
        must be an LLMError (a SankError), not the internal retry-marker
        exception, or pipeline.py's per-source isolation won't catch it
        and one bad source would crash the entire run."""
        from google.genai import errors

        from sank.exceptions import LLMError as PublicLLMError
        from sank.exceptions import SankError

        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.side_effect = errors.ClientError(
                429, {"message": "quota exceeded"}
            )
            client = GeminiLLMClient(api_key="dummy")
            with pytest.raises(PublicLLMError, match="429") as exc_info:
                client.complete("s", "u")
            assert isinstance(exc_info.value, SankError)
        # 6 attempts on gemini-2.5-flash, then 6 more on the fallback
        # gemini-2.5-flash-lite, before finally giving up.
        assert instance.models.generate_content.call_count == 12

    def test_bad_request_fails_immediately_without_retrying(self):
        """The other half of the fix: a 400 (malformed request) or 401
        (bad key) will never succeed no matter how many times we ask —
        must fail on the first attempt, not burn the retry budget."""
        from google.genai import errors

        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.side_effect = errors.ClientError(
                400, {"message": "malformed request"}
            )
            client = GeminiLLMClient(api_key="dummy")
            with pytest.raises(LLMError, match="rejected"):
                client.complete("s", "u")
        assert instance.models.generate_content.call_count == 1

    def test_empty_response_text_raises_llm_error(self):
        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(text="", candidates=[])
            client = GeminiLLMClient(api_key="dummy")
            with pytest.raises(LLMError, match="no text"):
                client.complete("s", "u")

    def test_plain_complete_does_not_force_json_mode(self):
        """Regular complete() calls (e.g. the briefing agent, which wants
        prose) must not have response_mime_type set — that would force
        Gemini to emit JSON for what's supposed to be a plain-English digest."""
        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(text="Plain prose digest.")
            client = GeminiLLMClient(api_key="dummy")
            client.complete("write a digest", "signals...")
            config = instance.models.generate_content.call_args.kwargs["config"]
        assert config.response_mime_type is None

    def test_complete_json_forces_native_json_mode(self):
        """This is the actual fix for 'Model did not return valid JSON':
        complete_json() must request json_mode=True, and GeminiLLMClient
        must translate that into response_mime_type='application/json',
        which is a real decoding constraint on Gemini's side, not just a
        prompt suggestion the model can ignore."""
        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(text='{"skipped": true}')
            client = GeminiLLMClient(api_key="dummy")
            result = client.complete_json("extract stuff", "page text")
            config = instance.models.generate_content.call_args.kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert result == {"skipped": True}

    def test_complete_json_still_works_if_json_mode_output_has_fences_anyway(self):
        """Defense in depth: even with native JSON mode requested, keep
        stripping markdown fences in case a provider's 'JSON mode' still
        wraps the object."""
        with patch("google.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(
                text='```json\n{"skipped": false, "plain_summary": "x"}\n```'
            )
            client = GeminiLLMClient(api_key="dummy")
            result = client.complete_json("extract stuff", "page text")
        assert result == {"skipped": False, "plain_summary": "x"}
