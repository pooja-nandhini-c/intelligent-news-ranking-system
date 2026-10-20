"""Relevance Agent using LangChain and Google Gemini Service.

Evaluates semantic alignment between a user's search query or topic
and a news article based strictly on its title, description, and available content.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.models.article import Article
from backend.app.services.gemini_service import (
    GeminiAPIError,
    GeminiAPIKeyMissingError,
    GeminiInvalidModelError,
    GeminiMalformedResponseError,
    GeminiService,
    GeminiServiceException,
    GeminiTimeoutError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Output Schema & Exceptions
# ---------------------------------------------------------------------------


class RelevanceOutput(BaseModel):
    """Structured output returned by the Relevance Agent."""

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relevance score from 0 to 100 indicating semantic alignment with the user query or topic."
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Short, grounded explanation of why this relevance score was assigned based on the article's text."
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases_and_fallbacks(cls, data: Any) -> Any:
        """Handle field aliases or alternate keys gracefully from LLM responses."""
        if isinstance(data, dict):
            # Normalize relevance_score key
            if "relevance_score" not in data:
                for alias in ("score", "relevance", "relevancy_score"):
                    if alias in data:
                        data["relevance_score"] = data[alias]
                        break
            # Normalize reason key
            if "reason" not in data:
                for alias in ("explanation", "rationale", "summary"):
                    if alias in data:
                        data["reason"] = data[alias]
                        break
        return data

    @field_validator("relevance_score", mode="before")
    @classmethod
    def validate_score(cls, v: Any) -> float:
        try:
            val = float(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"relevance_score must be a valid number, got {v}") from exc
        if val < 0.0 or val > 100.0:
            raise ValueError(f"relevance_score must be between 0.0 and 100.0, got {val}")
        return round(val, 2)

    @field_validator("reason", mode="before")
    @classmethod
    def clean_reason(cls, v: Any) -> str:
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("reason cannot be empty or whitespace only.")
            return cleaned
        if v is None:
            raise ValueError("reason cannot be None.")
        cleaned = str(v).strip()
        if not cleaned:
            raise ValueError("reason cannot be empty.")
        return cleaned


class RelevanceAgentError(Exception):
    """Base exception for Relevance Agent operations."""
    pass


class RelevanceValidationError(RelevanceAgentError, ValueError):
    """Raised when input query or article fails validation."""
    pass


class RelevanceExecutionError(RelevanceAgentError, RuntimeError):
    """Raised when relevance evaluation encounters an unhandled LLM error."""
    pass


# ---------------------------------------------------------------------------
# System and User Prompts
# ---------------------------------------------------------------------------

RELEVANCE_SYSTEM_PROMPT = """You are an objective, expert News Relevance Evaluator.
Your sole purpose is to evaluate how relevant a given news article is to a user's search query or topic.

Evaluation Guidelines:
1. Assess topical and semantic alignment between the user's query and the article.
2. Grounding & Anti-Hallucination:
   - Rely strictly and exclusively on the explicit facts, claims, and context present in the provided article text (Title, Description, and Content).
   - Do NOT invent, extrapolate, or assume facts, events, or implications that are not directly mentioned in the article text.
   - If the article text is truncated, brief, or missing specific details, evaluate strictly based on the available text without assuming external facts.
   - Do not assume real-world developments or knowledge not explicitly stated in the text.
3. Scoring Criteria (0 - 100):
   - 90 - 100: Highly Relevant. The article directly and centrally addresses the specific query or topic with strong focus.
   - 70 - 89: Substantially Relevant. The article is closely related to the query/topic and discusses major relevant aspects.
   - 40 - 69: Moderately / Tangentially Relevant. The article mentions the query or related concepts, but it is a minor focus or secondary theme.
   - 10 - 39: Marginally Relevant. Only a passing, peripheral, or superficial mention of terms related to the query.
   - 0 - 9: Irrelevant. The article is completely unrelated to the query or topic.
4. Reason Requirements:
   - Provide a concise explanation (1 to 3 sentences) justifying the score.
   - Specifically reference the article's title, description, or content to explain what is or isn't relevant to the query.
   - Ground the explanation only in what the article explicitly states.
"""

RELEVANCE_USER_TEMPLATE = """Evaluate the relevance of the following news article to the given user query/topic:

User Query / Topic:
"{query}"

Article Details:
- Title: {title}
- Description: {description}
- Content: {content}
- Source: {source_name}

Provide your assessment strictly as structured output containing:
- relevance_score: float or integer between 0 and 100
- reason: short explanation grounded strictly in the provided text above
"""


# ---------------------------------------------------------------------------
# Relevance Agent Implementation
# ---------------------------------------------------------------------------


class RelevanceAgent:
    """
    Independent agent responsible for evaluating news article relevance to a user query.
    Utilizes LangChain prompt templates and the shared GeminiService.
    """

    def __init__(
        self,
        gemini_service: Optional[GeminiService] = None,
        temperature: float = 0.0,
        model_name: Optional[str] = None
    ):
        self.gemini_service = gemini_service or GeminiService(model_name=model_name)
        self.temperature = temperature
        self.model_name = model_name
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", RELEVANCE_SYSTEM_PROMPT),
            ("human", RELEVANCE_USER_TEMPLATE)
        ])

    def get_chain(self) -> Runnable:
        """
        Build and return a LangChain LCEL Runnable chain for relevance evaluation.
        """
        llm = self.gemini_service.get_llm(
            temperature=self.temperature,
            model_name=self.model_name
        )
        structured_llm = llm.with_structured_output(RelevanceOutput)
        return self.prompt_template | structured_llm

    def _validate_and_normalize_inputs(
        self,
        query: Any,
        article: Any
    ) -> Tuple[str, Dict[str, str]]:
        """
        Validate and sanitize user query and article inputs.

        Raises:
            RelevanceValidationError: If inputs are missing, invalid, or empty.
        """
        if not isinstance(query, str):
            raise RelevanceValidationError("Query must be a string.")

        clean_query = query.strip()
        if not clean_query:
            raise RelevanceValidationError("Query cannot be empty or whitespace only.")

        if article is None:
            raise RelevanceValidationError("Article cannot be None.")

        if isinstance(article, (str, int, float, bool, list, set, tuple)):
            raise RelevanceValidationError(
                f"Unsupported article type: {type(article).__name__}. Expected Article instance or dict."
            )

        # Extract fields depending on input type
        if isinstance(article, Article):
            title = article.title
            description = article.description
            content = article.content
            source_name = article.source_name
        elif isinstance(article, dict):
            if "title" not in article or article["title"] is None:
                raise RelevanceValidationError("Article title is required and cannot be empty.")
            title = str(article["title"]).strip()
            if not title:
                raise RelevanceValidationError("Article title cannot be empty or whitespace only.")
            description = article.get("description")
            content = article.get("content")

            source_val = article.get("source")
            if isinstance(source_val, dict):
                source_name = source_val.get("name") or "Unknown"
            else:
                source_name = article.get("source_name") or "Unknown"
        elif hasattr(article, "title") and not callable(getattr(article, "title")):
            raw_title = getattr(article, "title")
            if raw_title is None or not str(raw_title).strip():
                raise RelevanceValidationError("Article title is required and cannot be empty.")
            title = str(raw_title).strip()
            description = getattr(article, "description", None)
            content = getattr(article, "content", None)
            source_name = getattr(article, "source_name", "Unknown")
        else:
            raise RelevanceValidationError(
                f"Unsupported article input type: {type(article).__name__}. Expected Article instance or dict."
            )

        clean_title = (title or "").strip()
        if not clean_title:
            raise RelevanceValidationError("Article title cannot be empty.")

        clean_desc = (description or "").strip() if description else "Not provided"
        clean_content = (content or "").strip() if content else "Not provided"
        clean_source = (source_name or "").strip() if source_name else "Unknown"

        return clean_query, {
            "title": clean_title,
            "description": clean_desc,
            "content": clean_content,
            "source_name": clean_source
        }

    async def aevaluate(
        self,
        query: str,
        article: Union[Article, dict, Any],
        fallback_on_error: bool = True
    ) -> RelevanceOutput:
        """
        Asynchronously evaluate the relevance of an article to the user query.

        Args:
            query: The user search query or topic string.
            article: The article to evaluate (Article instance or dict with title/description/content).
            fallback_on_error: If True, returns a safe default score on LLM errors instead of raising.

        Returns:
            RelevanceOutput containing relevance_score (0-100) and short explanation.

        Raises:
            RelevanceValidationError: If input query or article is invalid.
            RelevanceExecutionError: If LLM invocation fails and fallback_on_error is False.
        """
        clean_query, article_data = self._validate_and_normalize_inputs(query, article)

        user_prompt = RELEVANCE_USER_TEMPLATE.format(
            query=clean_query,
            title=article_data["title"],
            description=article_data["description"],
            content=article_data["content"],
            source_name=article_data["source_name"]
        )

        try:
            return await self.gemini_service.agenerate_structured(
                prompt=user_prompt,
                schema=RelevanceOutput,
                system_instruction=RELEVANCE_SYSTEM_PROMPT,
                temperature=self.temperature,
                model_name=self.model_name
            )
        except GeminiServiceException as exc:
            logger.error("GeminiService error in RelevanceAgent.aevaluate: %s", exc)
            if not fallback_on_error:
                raise RelevanceExecutionError(f"Relevance evaluation failed: {exc}") from exc
            return RelevanceOutput(
                relevance_score=0.0,
                reason=f"Relevance evaluation unavailable due to service error: {exc}"
            )
        except Exception as exc:
            logger.exception("Unexpected error in RelevanceAgent.aevaluate: %s", exc)
            if not fallback_on_error:
                raise RelevanceExecutionError(f"Relevance evaluation failed due to unexpected error: {exc}") from exc
            return RelevanceOutput(
                relevance_score=0.0,
                reason=f"Relevance evaluation unavailable due to unexpected error: {exc}"
            )

    def evaluate(
        self,
        query: str,
        article: Union[Article, dict, Any],
        fallback_on_error: bool = True
    ) -> RelevanceOutput:
        """
        Synchronously evaluate the relevance of an article to the user query.

        Args:
            query: The user search query or topic string.
            article: The article to evaluate (Article instance or dict with title/description/content).
            fallback_on_error: If True, returns a safe default score on LLM errors instead of raising.

        Returns:
            RelevanceOutput containing relevance_score (0-100) and short explanation.

        Raises:
            RelevanceValidationError: If input query or article is invalid.
            RelevanceExecutionError: If LLM invocation fails and fallback_on_error is False.
        """
        clean_query, article_data = self._validate_and_normalize_inputs(query, article)

        user_prompt = RELEVANCE_USER_TEMPLATE.format(
            query=clean_query,
            title=article_data["title"],
            description=article_data["description"],
            content=article_data["content"],
            source_name=article_data["source_name"]
        )

        try:
            return self.gemini_service.generate_structured(
                prompt=user_prompt,
                schema=RelevanceOutput,
                system_instruction=RELEVANCE_SYSTEM_PROMPT,
                temperature=self.temperature,
                model_name=self.model_name
            )
        except GeminiServiceException as exc:
            logger.error("GeminiService error in RelevanceAgent.evaluate: %s", exc)
            if not fallback_on_error:
                raise RelevanceExecutionError(f"Relevance evaluation failed: {exc}") from exc
            return RelevanceOutput(
                relevance_score=0.0,
                reason=f"Relevance evaluation unavailable due to service error: {exc}"
            )
        except Exception as exc:
            logger.exception("Unexpected error in RelevanceAgent.evaluate: %s", exc)
            if not fallback_on_error:
                raise RelevanceExecutionError(f"Relevance evaluation failed due to unexpected error: {exc}") from exc
            return RelevanceOutput(
                relevance_score=0.0,
                reason=f"Relevance evaluation unavailable due to unexpected error: {exc}"
            )

    async def aevaluate_batch(
        self,
        query: str,
        articles: List[Union[Article, dict, Any]],
        max_concurrency: int = 5,
        fallback_on_error: bool = True
    ) -> List[RelevanceOutput]:
        """
        Evaluate multiple articles concurrently for a given query.
        Does not perform ranking or score synthesis.

        Args:
            query: The user search query or topic string.
            articles: List of articles to evaluate.
            max_concurrency: Maximum number of concurrent LLM requests.
            fallback_on_error: If True, individual failures return safe fallbacks.

        Returns:
            List of RelevanceOutput in the same order as input articles.
        """
        if not articles:
            return []

        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _eval_with_sem(art: Union[Article, dict, Any]) -> RelevanceOutput:
            async with semaphore:
                return await self.aevaluate(query, art, fallback_on_error=fallback_on_error)

        tasks = [_eval_with_sem(art) for art in articles]
        return await asyncio.gather(*tasks)
