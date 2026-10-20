"""Google Gemini LLM service using LangChain's Google Generative AI integration."""

import logging
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, Type, TypeVar, Any, List, Union
from pydantic import BaseModel, ValidationError
import asyncio
from email.utils import parsedate_to_datetime
import math

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.app.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiServiceException(Exception):
    """Base exception for Gemini service operations."""
    pass


class GeminiAPIKeyMissingError(GeminiServiceException):
    """Raised when GEMINI_API_KEY is not configured."""
    pass


class GeminiInvalidModelError(GeminiServiceException):
    """Raised when model name is invalid or deprecated."""
    pass


class GeminiTimeoutError(GeminiServiceException):
    """Raised when request to Gemini API times out."""
    pass


class GeminiMalformedResponseError(GeminiServiceException):
    """Raised when the LLM produces unparseable or malformed output."""
    pass


class GeminiAPIError(GeminiServiceException):
    """Raised when Google Gemini API returns an upstream error."""
    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after


class GeminiRequestLimiter:
    """Share a concurrency bound and rolling requests-per-minute window."""

    def __init__(self, requests_per_minute: int, max_concurrency: int):
        self.requests_per_minute = max(1, requests_per_minute)
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrency))
        self._timestamps = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        self._semaphore.acquire()
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                wait_for = max(0.01, 60 - (now - self._timestamps[0]))
            time.sleep(wait_for)

    @asynccontextmanager
    async def request_slot(self, deadline: float):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            if not self._semaphore.acquire(blocking=False):
                await asyncio.sleep(min(0.05, remaining))
                continue
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    break
                wait_for = max(0.01, 60 - (now - self._timestamps[0]))
            self._semaphore.release()
            await asyncio.sleep(min(wait_for, remaining))
        try:
            yield
        finally:
            self._semaphore.release()

    @contextmanager
    def request_slot_sync(self):
        self.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


@lru_cache(maxsize=16)
def _shared_limiter(requests_per_minute: int, max_concurrency: int) -> GeminiRequestLimiter:
    return GeminiRequestLimiter(requests_per_minute, max_concurrency)


class GeminiService:
    """
    Reusable service for interacting with Google Gemini models.
    Supports structured Pydantic outputs, prompt execution, and error handling.
    """

    # List of known deprecated/unavailable models to proactively warn or reject
    DEPRECATED_MODELS = {
        "gemini-1.0-pro",
        "gemini-pro",
        "gemini-pro-vision",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "text-bison",
        "chat-bison"
    }

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        request_limiter: Optional[GeminiRequestLimiter] = None
    ):
        self.settings = settings or get_settings()
        if model_name is not None:
            raw_model = model_name
        else:
            raw_model = self.settings.gemini_model or "gemini-3.8-flash"

        self.model_name = (raw_model or "").strip()
        self.timeout = max(0.1, timeout if timeout is not None else self.settings.gemini_request_timeout)
        self.max_retries = max(0, max_retries if max_retries is not None else self.settings.gemini_max_retries)
        self.request_limiter = request_limiter or _shared_limiter(
            self.settings.gemini_requests_per_minute,
            self.settings.gemini_max_concurrency,
        )
        self._validate_model_name(self.model_name)

    @property
    def api_key(self) -> str:
        return (self.settings.gemini_api_key or "").strip()

    def _validate_api_key(self) -> None:
        if not self.api_key:
            logger.error("Gemini invocation aborted: GEMINI_API_KEY is not set.")
            raise GeminiAPIKeyMissingError("GEMINI_API_KEY is not configured in the application environment.")

    def _validate_model_name(self, model: str) -> None:
        if not model:
            raise GeminiInvalidModelError("Gemini model name cannot be empty.")
        if model.lower() in self.DEPRECATED_MODELS:
            raise GeminiInvalidModelError(
                f"Model '{model}' is deprecated or unavailable on the current API version. "
                f"Please configure a supported model such as 'gemini-3.8-flash' or 'gemini-2.5-flash'."
            )

    def _sanitize(self, text: str) -> str:
        """Sanitize text to guarantee API key is never logged or exposed in exceptions."""
        if not text:
            return ""
        if self.api_key and len(self.api_key) > 4 and self.api_key in text:
            return text.replace(self.api_key, "[REDACTED_GEMINI_KEY]")
        return text

    def get_llm(
        self,
        temperature: float = 0.0,
        model_name: Optional[str] = None
    ) -> ChatGoogleGenerativeAI:
        """Create and return a configured ChatGoogleGenerativeAI instance.

        Uses the API endpoint selected by the installed Google GenAI integration.
        """
        self._validate_api_key()
        active_model = (model_name or self.model_name).strip()
        self._validate_model_name(active_model)

        return ChatGoogleGenerativeAI(
            model=active_model,
            google_api_key=self.api_key,
            temperature=temperature,
            timeout=self.timeout,
            # The Google SDK retries 429 responses internally by default. Disable
            # those retries so this service owns one bounded, delay-aware policy.
            max_retries=1
        )

    def _retry_delay(self, exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass
        for name, divisor in (("retry-after", 1.0), ("retry-after-ms", 1000.0)):
            value = headers.get(name) if hasattr(headers, "get") else None
            if value is not None:
                try:
                    return max(0.0, float(value) / divisor)
                except (TypeError, ValueError):
                    if name == "retry-after":
                        try:
                            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
                        except (TypeError, ValueError, OverflowError):
                            pass
        message = str(exc)
        match = re.search(
            r"retry(?:\s+in|\s*delay)[\s:'\"]+([0-9.]+)\s*(ms|s|seconds?)?",
            message,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            return value / 1000 if (match.group(2) or "s").lower() == "ms" else value
        return None

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        message = str(exc).upper()
        return status == 429 or "429" in message or "RESOURCE_EXHAUSTED" in message

    def _quota_error(self, exc: Exception, retry_after: Optional[float]) -> GeminiAPIError:
        rounded_delay = max(1, math.ceil(retry_after)) if retry_after is not None else None
        delay_text = f" Wait {rounded_delay} seconds before retrying." if rounded_delay is not None else " Please wait before retrying."
        return GeminiAPIError(
            f"Gemini free-tier quota is exhausted (HTTP 429).{delay_text}",
            status_code=429,
            retry_after=rounded_delay,
        )

    async def _ainvoke_with_retry(self, runnable: Any, messages: List[BaseMessage]) -> Any:
        deadline = time.monotonic() + self.timeout
        for attempt in range(self.max_retries + 1):
            try:
                async with self.request_limiter.request_slot(deadline):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    return await asyncio.wait_for(runnable.ainvoke(messages), timeout=remaining)
            except Exception as exc:
                if not self._is_rate_limit_error(exc):
                    raise
                delay = self._retry_delay(exc)
                if delay is None:
                    delay = min(2 ** attempt, 5)
                remaining = deadline - time.monotonic()
                if attempt >= self.max_retries or delay >= remaining:
                    raise self._quota_error(exc, delay) from exc
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    def _invoke_with_retry(self, runnable: Any, messages: List[BaseMessage]) -> Any:
        deadline = time.monotonic() + self.timeout
        for attempt in range(self.max_retries + 1):
            try:
                with self.request_limiter.request_slot_sync():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError()
                    return runnable.invoke(messages, timeout=remaining, max_retries=1)
            except Exception as exc:
                if not self._is_rate_limit_error(exc):
                    raise
                delay = self._retry_delay(exc)
                if delay is None:
                    delay = min(2 ** attempt, 5)
                if attempt >= self.max_retries or delay >= deadline - time.monotonic():
                    raise self._quota_error(exc, delay) from exc
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        if system_instruction and system_instruction.strip():
            messages.append(SystemMessage(content=system_instruction.strip()))
        messages.append(HumanMessage(content=prompt.strip()))
        return messages

    async def agenerate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        model_name: Optional[str] = None
    ) -> T:
        """
        Asynchronously invoke Gemini with guaranteed structured Pydantic output.
        """
        self._validate_api_key()
        llm = self.get_llm(temperature=temperature, model_name=model_name)
        structured_llm = llm.with_structured_output(schema)
        messages = self._build_messages(prompt, system_instruction)

        logger.info("Executing structured LLM prompt against model '%s'", getattr(llm, "model", self.model_name))

        try:
            result = await self._ainvoke_with_retry(structured_llm, messages)
            if result is None:
                raise GeminiMalformedResponseError("LLM returned null or empty response for structured schema.")
            if not isinstance(result, schema):
                # Attempt validation if dictionary was returned
                if isinstance(result, dict):
                    result = schema.model_validate(result)
                else:
                    raise GeminiMalformedResponseError(
                        f"Expected {schema.__name__} output, got {type(result).__name__}"
                    )
            return result
        except asyncio.TimeoutError as exc:
            logger.error("Gemini request timed out after %ss", self.timeout)
            raise GeminiTimeoutError(f"Gemini request timed out after {self.timeout}s.") from exc
        except (ValidationError, TypeError) as exc:
            logger.error("Failed to parse Gemini structured response: %s", self._sanitize(str(exc)))
            raise GeminiMalformedResponseError(f"Malformed LLM structured output: {self._sanitize(str(exc))}") from exc
        except (GeminiServiceException, GeminiAPIKeyMissingError, GeminiInvalidModelError, GeminiTimeoutError, GeminiMalformedResponseError):
            raise
        except Exception as exc:
            sanitized_err = self._sanitize(str(exc))
            logger.error("Gemini API error during structured invocation: %s", sanitized_err)
            if "timeout" in sanitized_err.lower():
                raise GeminiTimeoutError(sanitized_err) from exc
            if "not found" in sanitized_err.lower() or "invalid model" in sanitized_err.lower():
                raise GeminiInvalidModelError(sanitized_err) from exc
            raise GeminiAPIError(f"Gemini API failure: {sanitized_err}") from exc

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        model_name: Optional[str] = None
    ) -> T:
        """
        Synchronously invoke Gemini with guaranteed structured Pydantic output.
        """
        self._validate_api_key()
        llm = self.get_llm(temperature=temperature, model_name=model_name)
        structured_llm = llm.with_structured_output(schema)
        messages = self._build_messages(prompt, system_instruction)

        try:
            result = self._invoke_with_retry(structured_llm, messages)
            if result is None:
                raise GeminiMalformedResponseError("LLM returned null or empty response for structured schema.")
            if not isinstance(result, schema):
                if isinstance(result, dict):
                    result = schema.model_validate(result)
                else:
                    raise GeminiMalformedResponseError(
                        f"Expected {schema.__name__} output, got {type(result).__name__}"
                    )
            return result
        except TimeoutError as exc:
            logger.error("Gemini request timed out after %ss", self.timeout)
            raise GeminiTimeoutError(f"Gemini request timed out after {self.timeout}s.") from exc
        except (ValidationError, TypeError) as exc:
            sanitized_err = self._sanitize(str(exc))
            logger.error("Failed to parse Gemini structured response: %s", sanitized_err)
            raise GeminiMalformedResponseError(f"Malformed LLM structured output: {sanitized_err}") from exc
        except (GeminiServiceException, GeminiAPIKeyMissingError, GeminiInvalidModelError, GeminiTimeoutError, GeminiMalformedResponseError):
            raise
        except Exception as exc:
            sanitized_err = self._sanitize(str(exc))
            logger.error("Gemini API error during synchronous structured invocation: %s", sanitized_err)
            raise GeminiAPIError(f"Gemini API failure: {sanitized_err}") from exc

    async def agenerate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        model_name: Optional[str] = None
    ) -> str:
        """
        Asynchronously invoke Gemini for plain-text response.
        """
        self._validate_api_key()
        llm = self.get_llm(temperature=temperature, model_name=model_name)
        messages = self._build_messages(prompt, system_instruction)

        try:
            response = await self._ainvoke_with_retry(llm, messages)
            content = response.content
            if isinstance(content, list):
                # If multi-part content returned
                content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
            return str(content or "").strip()
        except asyncio.TimeoutError as exc:
            raise GeminiTimeoutError(f"Gemini request timed out after {self.timeout}s.") from exc
        except Exception as exc:
            sanitized_err = self._sanitize(str(exc))
            if "timeout" in sanitized_err.lower():
                raise GeminiTimeoutError(sanitized_err) from exc
            raise GeminiAPIError(f"Gemini API failure: {sanitized_err}") from exc
