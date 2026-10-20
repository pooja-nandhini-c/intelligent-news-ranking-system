"""Sentiment Agent using LangChain and Google Gemini Service.

Evaluates the sentiment, emotional polarity, and tonal orientation of news articles
based strictly on explicit article content without political bias or outside assumptions.
"""

import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.models.article import Article
from backend.app.models.ranking import SentimentLabel
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

SentimentType = Literal["positive", "neutral", "negative"]


class SentimentOutput(BaseModel):
    """Structured output returned by the Sentiment Agent.

    Represents objective sentiment classification, normalized score, and grounded explanation.
    This score is independent and kept strictly separate from any final ranking score.
    """

    sentiment: SentimentType = Field(
        ...,
        description="Sentiment classification: 'positive', 'neutral', or 'negative'."
    )
    sentiment_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Sentiment score normalized between 0.0 and 100.0 (0=negative, 50=neutral, 100=positive)."
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Short, grounded explanation of the sentiment evaluation based strictly on article content."
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases_and_fallbacks(cls, data: Any) -> Any:
        """Handle field aliases or alternate keys gracefully from LLM responses."""
        if isinstance(data, dict):
            # Normalize sentiment key
            if "sentiment" not in data:
                for alias in ("polarity", "label", "classification", "sentiment_label"):
                    if alias in data:
                        data["sentiment"] = data[alias]
                        break

            # Normalize sentiment_score key
            if "sentiment_score" not in data:
                for alias in ("score", "sentiment_value", "polarity_score", "val"):
                    if alias in data:
                        data["sentiment_score"] = data[alias]
                        break

            # Normalize reason key
            if "reason" not in data:
                for alias in ("explanation", "rationale", "summary", "justification"):
                    if alias in data:
                        data["reason"] = data[alias]
                        break
        return data

    @field_validator("sentiment", mode="before")
    @classmethod
    def clean_sentiment(cls, v: Any) -> str:
        """Strip whitespace, lowercase, and validate allowed sentiment labels."""
        if isinstance(v, SentimentLabel):
            v = v.value

        if not isinstance(v, str):
            raise ValueError(f"sentiment must be a string, got {type(v).__name__}")

        cleaned = v.strip().lower()
        # Handle common variants
        if cleaned in ("positive", "pos"):
            return "positive"
        if cleaned in ("negative", "neg"):
            return "negative"
        if cleaned in ("neutral", "neu", "mixed"):
            return "neutral"

        if cleaned not in ("positive", "neutral", "negative"):
            raise ValueError(f"sentiment must be 'positive', 'neutral', or 'negative', got '{cleaned}'")
        return cleaned

    @field_validator("sentiment_score", mode="before")
    @classmethod
    def validate_score(cls, v: Any) -> float:
        """Validate and round sentiment score to 2 decimal places within [0.0, 100.0]."""
        try:
            val = float(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"sentiment_score must be a valid number, got {v}") from exc

        if val < 0.0 or val > 100.0:
            raise ValueError(f"sentiment_score must be between 0.0 and 100.0, got {val}")
        return round(val, 2)

    @field_validator("reason", mode="before")
    @classmethod
    def clean_reason(cls, v: Any) -> str:
        """Strip whitespace and verify explanation is non-empty."""
        if v is None:
            raise ValueError("reason cannot be None.")
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("reason cannot be empty or whitespace only.")
            return cleaned
        cleaned = str(v).strip()
        if not cleaned:
            raise ValueError("reason cannot be empty.")
        return cleaned


class SentimentAgentError(Exception):
    """Base exception for Sentiment Agent operations."""
    pass


class SentimentValidationError(SentimentAgentError, ValueError):
    """Raised when input article or article content fails validation."""
    pass


class SentimentExecutionError(SentimentAgentError, RuntimeError):
    """Raised when sentiment evaluation encounters an unhandled LLM error."""
    pass


# ---------------------------------------------------------------------------
# System and User Prompts
# ---------------------------------------------------------------------------

SENTIMENT_SYSTEM_PROMPT = """You are an objective, expert News Sentiment Analyst.
Your sole purpose is to analyze the sentiment and emotional tone of a given news article based strictly and exclusively on its content.

Strict Analysis Rules:
1. Strict Content Grounding & Anti-Hallucination:
   - Rely strictly and exclusively on the explicit facts, claims, language, and emotional framing present in the provided article text (Title, Description, and Content).
   - Do NOT invent, extrapolate, or assume facts, consequences, or sentiments that are not directly stated in the article text.
   - Do NOT use outside knowledge, personal beliefs, or prior assumptions about real-world entities, topics, or events.

2. Political Neutrality and Non-Bias:
   - Do NOT infer sentiment from political preferences, partisan leanings, ideological viewpoints, or subjective political opinions.
   - Do NOT assume an article is positive or negative simply because it covers a particular political figure, party, policy, government, or ideology.
   - An article reporting on politics, elections, government actions, or policy debates must be evaluated objectively based on tone, language, and reported outcomes—NOT on agreement or disagreement with any political position.
   - Maintain complete journalistic and analytical neutrality.

3. Handling Missing Article Content:
   - If full body content is truncated, brief, or missing, evaluate strictly using the available headline (Title) and summary (Description) without hallucinating unmentioned text.
   - If the available text is purely factual, neutral headline reporting, or has no detectable emotional polarity, classify it as "neutral".

4. Scoring Criteria (0 - 100 Scale):
   - sentiment_score is a continuous metric between 0.0 and 100.0 representing the tonal orientation of the text:
     * 0 to 39: Negative Sentiment. Focuses primarily on adverse events, crises, failures, tragedy, conflict, loss, severe criticism, or distress.
       - 0 to 19: Strongly negative (e.g., catastrophic loss, major corruption scandal, severe economic collapse).
       - 20 to 39: Moderately negative (e.g., regulatory setback, quarterly loss, critical review, diplomatic tension).
     * 40 to 60: Neutral Sentiment. Factual, matter-of-fact, objective reporting, or balanced presentation of opposing views without clear positive or negative emotional slant.
       - 45 to 55: Purely objective / factual news reporting.
     * 61 to 100: Positive Sentiment. Focuses on achievements, breakthroughs, recovery, beneficial outcomes, growth, praise, or solutions.
       - 61 to 80: Moderately positive (e.g., steady growth, product launch, constructive agreement, steady progress).
       - 81 to 100: Strongly positive (e.g., medical cure breakthrough, historic peace agreement, life-saving innovation).
   - Ensure the sentiment classification matches the sentiment_score:
     * "negative" for scores 0.0 to 39.99
     * "neutral" for scores 40.0 to 60.0
     * "positive" for scores 60.01 to 100.0

5. Reason Requirements:
   - Provide a concise explanation (1 to 3 sentences) justifying the sentiment classification and score.
   - Ground the explanation directly in specific words, tone, or reported events from the article text.
   - Do NOT speculate or mention extraneous details outside the article.
"""

SENTIMENT_USER_TEMPLATE = """Analyze the sentiment of the following news article based strictly on its content:

Article Details:
- Title: {title}
- Description: {description}
- Content: {content}
- Source: {source_name}

Provide your assessment strictly as structured output containing:
- sentiment: "positive" | "neutral" | "negative"
- sentiment_score: float or integer between 0 and 100
- reason: short explanation grounded strictly in the provided text above
"""


# ---------------------------------------------------------------------------
# Sentiment Agent Implementation
# ---------------------------------------------------------------------------

class SentimentAgent:
    """Independent agent responsible for evaluating news article sentiment.

    Utilizes LangChain prompt templates and the shared GeminiService.
    Produces an isolated sentiment score, completely decoupled from final ranking scores.
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
            ("system", SENTIMENT_SYSTEM_PROMPT),
            ("human", SENTIMENT_USER_TEMPLATE)
        ])

    def get_chain(self) -> Runnable:
        """Build and return a LangChain LCEL Runnable chain for sentiment analysis."""
        llm = self.gemini_service.get_llm(
            temperature=self.temperature,
            model_name=self.model_name
        )
        structured_llm = llm.with_structured_output(SentimentOutput)
        return self.prompt_template | structured_llm

    def _validate_and_normalize_inputs(
        self,
        article: Any
    ) -> Tuple[Dict[str, str], bool]:
        """Validate and sanitize article input and detect if body content is missing.

        Supports Article instances, dictionaries, and raw text strings.

        Returns:
            Tuple of (article_dict, is_content_missing)

        Raises:
            SentimentValidationError: If article is None, wrong type, or has no text.
        """
        if article is None:
            raise SentimentValidationError("Article cannot be None.")

        if isinstance(article, (int, float, bool, list, set, tuple)):
            raise SentimentValidationError(
                f"Unsupported article input type: {type(article).__name__}. Expected Article instance, dict, or text string."
            )

        if isinstance(article, str):
            clean_text = article.strip()
            if not clean_text:
                raise SentimentValidationError("Article text cannot be empty or whitespace only.")
            return {
                "title": "News Article",
                "description": "Not provided",
                "content": clean_text,
                "source_name": "Unknown"
            }, False

        if isinstance(article, Article):
            title = article.title
            description = article.description
            content = article.content
            source_name = article.source_name
        elif isinstance(article, dict):
            title = article.get("title")
            description = article.get("description")
            content = article.get("content")
            source_val = article.get("source")
            if isinstance(source_val, dict):
                source_name = source_val.get("name") or "Unknown"
            else:
                source_name = article.get("source_name") or "Unknown"
        elif hasattr(article, "__dict__") or (hasattr(article, "title") and not callable(getattr(article, "title"))):
            title = getattr(article, "title", None)
            description = getattr(article, "description", None)
            content = getattr(article, "content", None)
            source_name = getattr(article, "source_name", "Unknown")
        else:
            raise SentimentValidationError(
                f"Unsupported article input type: {type(article).__name__}. Expected Article instance, dict, or text string."
            )

        clean_title = str(title).strip() if title is not None else ""
        clean_desc = str(description).strip() if description is not None else ""
        clean_content = str(content).strip() if content is not None else ""
        clean_source = str(source_name).strip() if source_name else "Unknown"

        # Check if the article contains any readable text at all
        if not clean_title and not clean_desc and not clean_content:
            raise SentimentValidationError(
                "Article contains no readable title, description, or content to analyze."
            )

        is_content_missing = not bool(clean_content)

        formatted_title = clean_title if clean_title else "Not provided"
        formatted_desc = clean_desc if clean_desc else "Not provided"
        formatted_content = (
            clean_content
            if clean_content
            else "Not provided (evaluate sentiment strictly based on available title and description)"
        )

        return {
            "title": formatted_title,
            "description": formatted_desc,
            "content": formatted_content,
            "source_name": clean_source
        }, is_content_missing

    async def aevaluate(
        self,
        article: Union[Article, dict, str, Any],
        fallback_on_error: bool = True
    ) -> SentimentOutput:
        """Asynchronously analyze the sentiment of a news article.

        Args:
            article: News article (Article instance, dict, or raw text string).
            fallback_on_error: If True, returns a safe neutral fallback on failure instead of raising.

        Returns:
            SentimentOutput containing sentiment, sentiment_score (0-100), and short explanation.

        Raises:
            SentimentValidationError: If input article is invalid or completely empty (when fallback_on_error is False).
            SentimentExecutionError: If LLM invocation fails (when fallback_on_error is False).
        """
        try:
            article_data, is_content_missing = self._validate_and_normalize_inputs(article)
        except SentimentValidationError as exc:
            logger.warning("Sentiment validation error: %s", exc)
            if not fallback_on_error:
                raise
            return SentimentOutput(
                sentiment="neutral",
                sentiment_score=50.0,
                reason="Article content is missing or invalid; unable to determine sentiment."
            )

        user_prompt = SENTIMENT_USER_TEMPLATE.format(
            title=article_data["title"],
            description=article_data["description"],
            content=article_data["content"],
            source_name=article_data["source_name"]
        )

        try:
            return await self.gemini_service.agenerate_structured(
                prompt=user_prompt,
                schema=SentimentOutput,
                system_instruction=SENTIMENT_SYSTEM_PROMPT,
                temperature=self.temperature,
                model_name=self.model_name
            )
        except GeminiServiceException as exc:
            logger.error("GeminiService error in SentimentAgent.aevaluate: %s", exc)
            if not fallback_on_error:
                raise SentimentExecutionError(f"Sentiment analysis failed: {exc}") from exc
            return SentimentOutput(
                sentiment="neutral",
                sentiment_score=50.0,
                reason=f"Sentiment analysis unavailable due to service error: {exc}"
            )
        except Exception as exc:
            logger.exception("Unexpected error in SentimentAgent.aevaluate: %s", exc)
            if not fallback_on_error:
                raise SentimentExecutionError(f"Sentiment analysis failed due to unexpected error: {exc}") from exc
            return SentimentOutput(
                sentiment="neutral",
                sentiment_score=50.0,
                reason=f"Sentiment analysis unavailable due to unexpected error: {exc}"
            )

    def evaluate(
        self,
        article: Union[Article, dict, str, Any],
        fallback_on_error: bool = True
    ) -> SentimentOutput:
        """Synchronously analyze the sentiment of a news article.

        Args:
            article: News article (Article instance, dict, or raw text string).
            fallback_on_error: If True, returns a safe neutral fallback on failure instead of raising.

        Returns:
            SentimentOutput containing sentiment, sentiment_score (0-100), and short explanation.

        Raises:
            SentimentValidationError: If input article is invalid or completely empty (when fallback_on_error is False).
            SentimentExecutionError: If LLM invocation fails (when fallback_on_error is False).
        """
        try:
            article_data, is_content_missing = self._validate_and_normalize_inputs(article)
        except SentimentValidationError as exc:
            logger.warning("Sentiment validation error: %s", exc)
            if not fallback_on_error:
                raise
            return SentimentOutput(
                sentiment="neutral",
                sentiment_score=50.0,
                reason="Article content is missing or invalid; unable to determine sentiment."
            )

        user_prompt = SENTIMENT_USER_TEMPLATE.format(
            title=article_data["title"],
            description=article_data["description"],
            content=article_data["content"],
            source_name=article_data["source_name"]
        )

        try:
            return self.gemini_service.generate_structured(
                prompt=user_prompt,
                schema=SentimentOutput,
                system_instruction=SENTIMENT_SYSTEM_PROMPT,
                temperature=self.temperature,
                model_name=self.model_name
            )
        except GeminiServiceException as exc:
            logger.error("GeminiService error in SentimentAgent.evaluate: %s", exc)
            if not fallback_on_error:
                raise SentimentExecutionError(f"Sentiment analysis failed: {exc}") from exc
            return SentimentOutput(
                sentiment="neutral",
                sentiment_score=50.0,
                reason=f"Sentiment analysis unavailable due to service error: {exc}"
            )
        except Exception as exc:
            logger.exception("Unexpected error in SentimentAgent.evaluate: %s", exc)
            if not fallback_on_error:
                raise SentimentExecutionError(f"Sentiment analysis failed due to unexpected error: {exc}") from exc
            return SentimentOutput(
                sentiment="neutral",
                sentiment_score=50.0,
                reason=f"Sentiment analysis unavailable due to unexpected error: {exc}"
            )

    async def aevaluate_batch(
        self,
        articles: List[Union[Article, dict, str, Any]],
        max_concurrency: int = 5,
        fallback_on_error: bool = True
    ) -> List[SentimentOutput]:
        """Analyze sentiment for multiple articles concurrently.

        Does not perform ranking or composite score synthesis.

        Args:
            articles: List of articles to evaluate.
            max_concurrency: Maximum number of concurrent LLM requests.
            fallback_on_error: If True, individual failures return safe neutral fallbacks.

        Returns:
            List of SentimentOutput in the same order as input articles.
        """
        if not articles:
            return []

        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _eval_with_sem(art: Union[Article, dict, str, Any]) -> SentimentOutput:
            async with semaphore:
                return await self.aevaluate(art, fallback_on_error=fallback_on_error)

        tasks = [_eval_with_sem(art) for art in articles]
        return await asyncio.gather(*tasks)
