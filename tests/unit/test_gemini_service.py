"""Unit tests for GeminiService using mocked LLM responses."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import BaseModel, Field

from backend.app.config import Settings
from backend.app.services.gemini_service import (
    GeminiService,
    GeminiAPIKeyMissingError,
    GeminiInvalidModelError,
    GeminiTimeoutError,
    GeminiMalformedResponseError,
    GeminiAPIError,
    GeminiRequestLimiter,
)


class DummyOutputSchema(BaseModel):
    analysis: str = Field(..., description="Test field")
    score: float = Field(..., ge=0.0, le=100.0)


@pytest.fixture
def settings_with_key():
    return Settings(
        gemini_api_key="AIzaSyTestSecretKey987654321",
        gemini_model="gemini-3.8-flash"
    )


@pytest.fixture
def settings_missing_key():
    return Settings(
        gemini_api_key="",
        gemini_model="gemini-3.8-flash"
    )


@pytest.fixture
def gemini_service(settings_with_key):
    return GeminiService(
        settings=settings_with_key,
        request_limiter=GeminiRequestLimiter(requests_per_minute=1000, max_concurrency=5),
    )


class TestGeminiServiceInitAndValidation:
    def test_missing_api_key_raises_error(self, settings_missing_key):
        service = GeminiService(settings=settings_missing_key)
        with pytest.raises(GeminiAPIKeyMissingError) as exc_info:
            service.get_llm()
        assert "GEMINI_API_KEY is not configured" in str(exc_info.value)

    def test_deprecated_model_raises_error(self, settings_with_key):
        with pytest.raises(GeminiInvalidModelError) as exc_info:
            GeminiService(settings=settings_with_key, model_name="gemini-1.0-pro")
        assert "deprecated" in str(exc_info.value).lower()

    def test_empty_model_raises_error(self, settings_with_key):
        with pytest.raises(GeminiInvalidModelError) as exc_info:
            GeminiService(settings=settings_with_key, model_name="")
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_configurable_model(self, settings_with_key):
        service = GeminiService(settings=settings_with_key, model_name="gemini-2.5-flash")
        assert service.model_name == "gemini-2.5-flash"

    def test_configured_gemini_38_model_is_accepted(self, settings_with_key):
        service = GeminiService(settings=settings_with_key)
        assert service.model_name == "gemini-3.8-flash"


class TestGeminiServiceStructuredOutput:
    @pytest.mark.asyncio
    async def test_agenerate_structured_success(self, gemini_service):
        mock_output = DummyOutputSchema(analysis="Positive signals detected", score=94.5)
        mock_runnable = MagicMock()
        mock_runnable.ainvoke = AsyncMock(return_value=mock_output)

        with patch.object(gemini_service, "get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_runnable
            mock_get_llm.return_value = mock_llm

            result = await gemini_service.agenerate_structured(
                prompt="Analyze this article headline",
                schema=DummyOutputSchema,
                system_instruction="You are a media analyst."
            )

            assert isinstance(result, DummyOutputSchema)
            assert result.analysis == "Positive signals detected"
            assert result.score == 94.5
            mock_llm.with_structured_output.assert_called_once_with(DummyOutputSchema)

    def test_generate_structured_sync_success(self, gemini_service):
        mock_output = DummyOutputSchema(analysis="Sync analysis ok", score=82.0)
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = mock_output

        with patch.object(gemini_service, "get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_runnable
            mock_get_llm.return_value = mock_llm

            result = gemini_service.generate_structured(
                prompt="Evaluate article",
                schema=DummyOutputSchema
            )

            assert isinstance(result, DummyOutputSchema)
            assert result.analysis == "Sync analysis ok"
            assert result.score == 82.0

    @pytest.mark.asyncio
    async def test_agenerate_structured_null_response_raises_error(self, gemini_service):
        mock_runnable = MagicMock()
        mock_runnable.ainvoke = AsyncMock(return_value=None)

        with patch.object(gemini_service, "get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_runnable
            mock_get_llm.return_value = mock_llm

            with pytest.raises(GeminiMalformedResponseError) as exc_info:
                await gemini_service.agenerate_structured(
                    prompt="Test",
                    schema=DummyOutputSchema
                )
            assert "null or empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_timeout_raises_gemini_timeout_error(self, gemini_service):
        mock_runnable = MagicMock()
        mock_runnable.ainvoke = AsyncMock(side_effect=TimeoutError("Deadline exceeded"))

        with patch.object(gemini_service, "get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_runnable
            mock_get_llm.return_value = mock_llm

            with pytest.raises(GeminiTimeoutError) as exc_info:
                await gemini_service.agenerate_structured(
                    prompt="Test",
                    schema=DummyOutputSchema
                )
            assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_upstream_api_error(self, gemini_service):
        mock_runnable = MagicMock()
        mock_runnable.ainvoke = AsyncMock(side_effect=RuntimeError("Google API 503 Backend Error"))

        with patch.object(gemini_service, "get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_runnable
            mock_get_llm.return_value = mock_llm

            with pytest.raises(GeminiAPIError) as exc_info:
                await gemini_service.agenerate_structured(
                    prompt="Test",
                    schema=DummyOutputSchema
                )
            assert "Google API 503 Backend Error" in str(exc_info.value)


class TestGeminiServiceTextAndSanitization:
    @pytest.mark.asyncio
    async def test_agenerate_text_success(self, gemini_service):
        mock_response = MagicMock()
        mock_response.content = "Summary of breaking world news."

        with patch.object(gemini_service, "get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            text = await gemini_service.agenerate_text(prompt="Summarize this")
            assert text == "Summary of breaking world news."

    def test_api_key_sanitization_in_errors(self, settings_with_key):
        service = GeminiService(settings=settings_with_key)
        secret_key = settings_with_key.gemini_api_key
        leaked_message = f"Error connecting to https://generativelanguage.googleapis.com?key={secret_key}"

        sanitized = service._sanitize(leaked_message)
        assert secret_key not in sanitized
        assert "[REDACTED_GEMINI_KEY]" in sanitized


class TestGeminiRateLimits:
    def test_service_instances_use_shared_configured_limiter(self, settings_with_key):
        first = GeminiService(settings=settings_with_key)
        second = GeminiService(settings=settings_with_key)

        assert first.request_limiter is second.request_limiter
        assert first.request_limiter.requests_per_minute == 5
        assert settings_with_key.max_articles_to_rank == 1

    @pytest.mark.asyncio
    async def test_shared_limiter_enforces_requests_per_minute(self, settings_with_key, monkeypatch):
        limiter = GeminiRequestLimiter(requests_per_minute=1, max_concurrency=1)
        first = GeminiService(settings=settings_with_key, request_limiter=limiter)
        second = GeminiService(settings=settings_with_key, request_limiter=limiter)
        clock = [100.0]
        delays = []

        monkeypatch.setattr("backend.app.services.gemini_service.time.monotonic", lambda: clock[0])

        async def advance(delay):
            delays.append(delay)
            clock[0] += delay

        monkeypatch.setattr("backend.app.services.gemini_service.asyncio.sleep", advance)
        for service in (first, second):
            async with service.request_limiter.request_slot(deadline=clock[0] + 120):
                pass

        assert delays == [60.0]

    @pytest.mark.asyncio
    async def test_rate_limit_retries_after_provider_delay(self, settings_with_key, monkeypatch):
        service = GeminiService(
            settings=settings_with_key,
            request_limiter=GeminiRequestLimiter(requests_per_minute=60, max_concurrency=1),
        )
        quota_error = RuntimeError("429 RESOURCE_EXHAUSTED")
        quota_error.code = 429
        quota_error.response = type("Response", (), {"headers": {"retry-after": "2"}})()
        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(side_effect=[quota_error, "ok"])
        delays = []

        async def capture_delay(delay):
            delays.append(delay)

        monkeypatch.setattr("backend.app.services.gemini_service.asyncio.sleep", capture_delay)
        result = await service._ainvoke_with_retry(runnable, [])

        assert result == "ok"
        assert runnable.ainvoke.await_count == 2
        assert delays == [2.0]

    @pytest.mark.asyncio
    async def test_exhausted_quota_has_bounded_retries_and_safe_error(self, settings_with_key, monkeypatch):
        service = GeminiService(
            settings=settings_with_key,
            max_retries=1,
            request_limiter=GeminiRequestLimiter(requests_per_minute=60, max_concurrency=1),
        )
        quota_error = RuntimeError("429 RESOURCE_EXHAUSTED sensitive-upstream-detail")
        quota_error.code = 429
        quota_error.response = type("Response", (), {"headers": {"retry-after": "0.1"}})()
        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(side_effect=[quota_error, quota_error])
        monkeypatch.setattr("backend.app.services.gemini_service.asyncio.sleep", AsyncMock())

        with pytest.raises(GeminiAPIError) as exc_info:
            await service._ainvoke_with_retry(runnable, [])

        assert exc_info.value.status_code == 429
        assert runnable.ainvoke.await_count == 2
        assert "free-tier quota is exhausted" in str(exc_info.value)
        assert "sensitive-upstream-detail" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_delay_beyond_budget_returns_quota_error_without_retry(self, settings_with_key):
        service = GeminiService(
            settings=settings_with_key,
            timeout=0.1,
            max_retries=2,
            request_limiter=GeminiRequestLimiter(requests_per_minute=60, max_concurrency=1),
        )
        quota_error = RuntimeError("429 RESOURCE_EXHAUSTED")
        quota_error.code = 429
        quota_error.response = type("Response", (), {"headers": {"retry-after": "5"}})()
        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(side_effect=quota_error)

        with pytest.raises(GeminiAPIError) as exc_info:
            await service._ainvoke_with_retry(runnable, [])

        assert exc_info.value.status_code == 429
        runnable.ainvoke.assert_awaited_once()
