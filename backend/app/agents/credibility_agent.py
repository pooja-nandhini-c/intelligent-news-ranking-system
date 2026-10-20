"""Credibility Assessment Agent using LangChain and Google Gemini Service.

Evaluates observable journalistic credibility indicators based strictly on available
article metadata and content, without claiming factual truth verification or inventing
source reputations.
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

class CredibilityOutput(BaseModel):
    """Structured output returned by the Credibility Agent.

    Represents objective assessment of observable credibility indicators,
    confidence based on metadata completeness, and explicit limitations.
    This score is independent and kept strictly separate from any final ranking score.
    """

    credibility_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Observable credibility indicator score between 0.0 and 100.0 based on structural quality and attribution."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Assessment confidence level between 0.0 and 100.0 reflecting metadata and content completeness."
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Short, grounded explanation of the credibility score based strictly on observable indicators."
    )
    limitations: str = Field(
        ...,
        min_length=1,
        description="Short explanation of assessment limitations, distinguishing missing data from low credibility."
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases_and_fallbacks(cls, data: Any) -> Any:
        """Handle field aliases or alternate keys gracefully from LLM responses."""
        if isinstance(data, dict):
            # Normalize credibility_score key
            if "credibility_score" not in data:
                for alias in ("score", "credibility", "credibility_val", "quality_score"):
                    if alias in data:
                        data["credibility_score"] = data[alias]
                        break

            # Normalize confidence key
            if "confidence" not in data:
                for alias in ("confidence_score", "conf", "confidence_level", "certainty"):
                    if alias in data:
                        data["confidence"] = data[alias]
                        break

            # Normalize reason key
            if "reason" not in data:
                for alias in ("explanation", "rationale", "summary", "justification"):
                    if alias in data:
                        data["reason"] = data[alias]
                        break

            # Normalize limitations key
            if "limitations" not in data:
                for alias in ("limitation", "caveats", "constraints", "assessment_limitations"):
                    if alias in data:
                        data["limitations"] = data[alias]
                        break
        return data

    @field_validator("credibility_score", "confidence", mode="before")
    @classmethod
    def validate_scores(cls, v: Any) -> float:
        """Validate and round scores to 2 decimal places within [0.0, 100.0]."""
        try:
            val = float(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Score must be a valid number, got {v}") from exc

        if val < 0.0 or val > 100.0:
            raise ValueError(f"Score must be between 0.0 and 100.0, got {val}")
        return round(val, 2)

    @field_validator("reason", "limitations", mode="before")
    @classmethod
    def clean_text_fields(cls, v: Any) -> str:
        """Strip whitespace and verify explanation and limitations are non-empty."""
        if v is None:
            raise ValueError("Field cannot be None.")
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("Field cannot be empty or whitespace only.")
            return cleaned
        cleaned = str(v).strip()
        if not cleaned:
            raise ValueError("Field cannot be empty.")
        return cleaned


class CredibilityAgentError(Exception):
    """Base exception for Credibility Agent operations."""
    pass


class CredibilityValidationError(CredibilityAgentError, ValueError):
    """Raised when input article or metadata fails validation."""
    pass


class CredibilityExecutionError(CredibilityAgentError, RuntimeError):
    """Raised when credibility evaluation encounters an unhandled LLM error."""
    pass


# ---------------------------------------------------------------------------
# System and User Prompts
# ---------------------------------------------------------------------------

CREDIBILITY_SYSTEM_PROMPT = """You are an objective, expert News Credibility Analyst.
Your role is to assess observable credibility indicators of a news article based strictly on its available metadata and content.

Core Assessment Rules:
1. Observable Indicators, Not Factual Truth Verification:
   - Do NOT claim to prove absolute factual truth or label an article as definitively true or false. You are evaluating journalistic quality indicators, structural rigor, and source attribution, not conducting external investigative fact-checking.
   - Evaluate observable signals:
     * Clear attribution of statements, quotes, and primary sources (e.g., named officials, institutions, scientific reports, official data).
     * Specificity of dates, locations, figures, and technical or factual claims vs. vague assertions.
     * Balanced, professional, and transparent journalistic presentation vs. sensationalism, hyperbole, or unsubstantiated rumors.
     * Available publication metadata (author byline, publisher/source name, canonical URL, publication timestamp).

2. No Hallucinated Reputation or Cross-Source Verification:
   - Do NOT invent or assume external reputation, historical track record, or biases for the publisher, author, or source.
   - Do NOT claim cross-source verification, fact-checking database confirmation, or external corroboration, as no external live web verification was conducted.
   - Evaluate strictly what is present in the provided text and metadata.

3. Distinguishing Missing Information from Low Credibility:
   - Missing information (e.g., missing author byline, missing full body content, lack of publication date) is NOT necessarily evidence of low credibility or disinformation; it represents an informational limitation.
   - When content is truncated, missing, or brief, lower the 'confidence' score and document this explicitly in 'limitations', while evaluating the 'credibility_score' fairly based on whatever observable indicators exist in the available text.
   - A low credibility score should be reserved for explicit observable red flags: anonymous unverified hearsay presented as absolute fact, blatant internal contradictions, sensationalist clickbait headlines decoupled from facts, or total absence of attribution for extraordinary claims.

4. Scoring Guidelines (0 - 100 Scale):
   - credibility_score (0.0 to 100.0):
     * 80 - 100: High Observable Credibility. Strong attribution to named primary sources/experts, precise verifiable details (dates, numbers, entities), clear publisher and metadata, and objective tone.
     * 60 - 79: Moderate Observable Credibility. Professional reporting with reasonable attribution and factual context, though some details may be secondary or general.
     * 40 - 59: Limited / Average Observable Credibility. Basic factual report or headline with minimal attribution or vague sources ("sources said"), though no obvious bad-faith framing.
     * 20 - 39: Questionable Credibility Indicators. Heavy reliance on anonymous or unverified rumors, lack of sourcing for bold claims, heavily sensationalized framing.
     * 0 - 19: Extremely Low Observable Credibility. Completely ungrounded, conspiratorial, fabricated tone, or obvious disinformation patterns.
   - confidence (0.0 to 100.0):
     * 80 - 100: Comprehensive article content, author, publication date, and detailed text available.
     * 50 - 79: Partial content or brief summary available with basic metadata.
     * 20 - 49: Severe information scarcity (e.g., headline only, truncated snippet, missing author/date).
     * 0 - 19: Negligible information available to form an assessment.

5. Explanations & Limitations:
   - reason: 1 to 3 concise sentences explaining the assigned credibility_score based on observable attributes (attribution, specificity, tone).
   - limitations: 1 to 2 concise sentences explicitly stating what could not be verified (e.g., truncated content, lack of external fact-checking, missing author byline).
"""

CREDIBILITY_USER_TEMPLATE = """Evaluate the observable credibility indicators of the following news article:

Article Metadata & Content:
- Title: {title}
- Description: {description}
- Content: {content}
- Publisher / Source: {source_name}
- Source URL: {source_url}
- Article URL: {article_url}
- Author: {author}
- Published At: {published_at}

Provide your assessment strictly as structured output containing:
- credibility_score: float or integer between 0 and 100
- confidence: float or integer between 0 and 100
- reason: short explanation grounded in observable text indicators
- limitations: short explanation of limitations and unverified factors
"""


# ---------------------------------------------------------------------------
# Credibility Agent Implementation
# ---------------------------------------------------------------------------

class CredibilityAgent:
    """Independent agent responsible for evaluating observable news article credibility.

    Utilizes LangChain prompt templates and the shared GeminiService.
    Produces an isolated credibility score, completely decoupled from final ranking scores.
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
            ("system", CREDIBILITY_SYSTEM_PROMPT),
            ("human", CREDIBILITY_USER_TEMPLATE)
        ])

    def get_chain(self) -> Runnable:
        """Build and return a LangChain LCEL Runnable chain for credibility evaluation."""
        llm = self.gemini_service.get_llm(
            temperature=self.temperature,
            model_name=self.model_name
        )
        structured_llm = llm.with_structured_output(CredibilityOutput)
        return self.prompt_template | structured_llm

    def _validate_and_normalize_inputs(
        self,
        article: Any
    ) -> Tuple[Dict[str, str], bool]:
        """Validate and sanitize article input and metadata.

        Supports Article instances, dictionaries, and raw text strings.

        Returns:
            Tuple of (article_data_dict, is_content_missing)

        Raises:
            CredibilityValidationError: If article is None, wrong type, or has no text.
        """
        if article is None:
            raise CredibilityValidationError("Article cannot be None.")

        if isinstance(article, (int, float, bool, list, set, tuple)):
            raise CredibilityValidationError(
                f"Unsupported article input type: {type(article).__name__}. Expected Article instance, dict, or text string."
            )

        if isinstance(article, str):
            clean_text = article.strip()
            if not clean_text:
                raise CredibilityValidationError("Article text cannot be empty or whitespace only.")
            return {
                "title": "News Article",
                "description": "Not provided",
                "content": clean_text,
                "source_name": "Unknown",
                "source_url": "Not provided",
                "article_url": "Not provided",
                "author": "Not provided",
                "published_at": "Not provided"
            }, False

        if isinstance(article, Article):
            title = article.title
            description = article.description
            content = article.content
            source_name = article.source_name
            source_url = article.source_url
            article_url = article.article_url
            author = article.author
            published_at = article.published_at.isoformat() if article.published_at else None
        elif isinstance(article, dict):
            title = article.get("title")
            description = article.get("description")
            content = article.get("content")
            author = article.get("author")
            published_at = article.get("published_at")
            if hasattr(published_at, "isoformat"):
                published_at = published_at.isoformat()

            source_val = article.get("source")
            if isinstance(source_val, dict):
                source_name = source_val.get("name") or "Unknown"
                source_url = source_val.get("url") or article.get("source_url")
            else:
                source_name = article.get("source_name") or "Unknown"
                source_url = article.get("source_url")

            article_url = article.get("article_url") or article.get("url")
        elif hasattr(article, "__dict__") or (hasattr(article, "title") and not callable(getattr(article, "title"))):
            title = getattr(article, "title", None)
            description = getattr(article, "description", None)
            content = getattr(article, "content", None)
            source_name = getattr(article, "source_name", "Unknown")
            source_url = getattr(article, "source_url", None)
            article_url = getattr(article, "article_url", getattr(article, "url", None))
            author = getattr(article, "author", None)
            published_raw = getattr(article, "published_at", None)
            published_at = published_raw.isoformat() if hasattr(published_raw, "isoformat") else published_raw
        else:
            raise CredibilityValidationError(
                f"Unsupported article input type: {type(article).__name__}. Expected Article instance, dict, or text string."
            )

        clean_title = str(title).strip() if title is not None else ""
        clean_desc = str(description).strip() if description is not None else ""
        clean_content = str(content).strip() if content is not None else ""
        clean_source = str(source_name).strip() if source_name else "Unknown"
        clean_source_url = str(source_url).strip() if source_url else "Not provided"
        clean_article_url = str(article_url).strip() if article_url else "Not provided"
        clean_author = str(author).strip() if author else "Not provided"
        clean_published = str(published_at).strip() if published_at else "Not provided"

        # Check if the article contains any readable text at all
        if not clean_title and not clean_desc and not clean_content:
            raise CredibilityValidationError(
                "Article contains no readable title, description, or content to analyze."
            )

        is_content_missing = not bool(clean_content)

        formatted_title = clean_title if clean_title else "Not provided"
        formatted_desc = clean_desc if clean_desc else "Not provided"
        formatted_content = (
            clean_content
            if clean_content
            else "Not provided (evaluate observable credibility strictly based on available title and description)"
        )

        return {
            "title": formatted_title,
            "description": formatted_desc,
            "content": formatted_content,
            "source_name": clean_source,
            "source_url": clean_source_url,
            "article_url": clean_article_url,
            "author": clean_author,
            "published_at": clean_published,
        }, is_content_missing

    async def aevaluate(
        self,
        article: Union[Article, dict, str, Any],
        fallback_on_error: bool = True
    ) -> CredibilityOutput:
        """Asynchronously evaluate the observable credibility of a news article.

        Args:
            article: News article (Article instance, dict, or raw text string).
            fallback_on_error: If True, returns a safe neutral fallback on failure instead of raising.

        Returns:
            CredibilityOutput containing credibility_score, confidence, reason, and limitations.

        Raises:
            CredibilityValidationError: If input article is invalid or completely empty (when fallback_on_error is False).
            CredibilityExecutionError: If LLM invocation fails (when fallback_on_error is False).
        """
        try:
            article_data, is_content_missing = self._validate_and_normalize_inputs(article)
        except CredibilityValidationError as exc:
            logger.warning("Credibility validation error: %s", exc)
            if not fallback_on_error:
                raise
            return CredibilityOutput(
                credibility_score=50.0,
                confidence=0.0,
                reason="Article content is missing or invalid; unable to assess credibility indicators.",
                limitations="No usable article text or metadata available to evaluate."
            )

        user_prompt = CREDIBILITY_USER_TEMPLATE.format(
            title=article_data["title"],
            description=article_data["description"],
            content=article_data["content"],
            source_name=article_data["source_name"],
            source_url=article_data["source_url"],
            article_url=article_data["article_url"],
            author=article_data["author"],
            published_at=article_data["published_at"]
        )

        try:
            return await self.gemini_service.agenerate_structured(
                prompt=user_prompt,
                schema=CredibilityOutput,
                system_instruction=CREDIBILITY_SYSTEM_PROMPT,
                temperature=self.temperature,
                model_name=self.model_name
            )
        except GeminiServiceException as exc:
            logger.error("GeminiService error in CredibilityAgent.aevaluate: %s", exc)
            if not fallback_on_error:
                raise CredibilityExecutionError(f"Credibility evaluation failed: {exc}") from exc
            return CredibilityOutput(
                credibility_score=50.0,
                confidence=0.0,
                reason=f"Credibility assessment unavailable due to service error: {exc}",
                limitations="Evaluation incomplete due to upstream service failure."
            )
        except Exception as exc:
            logger.exception("Unexpected error in CredibilityAgent.aevaluate: %s", exc)
            if not fallback_on_error:
                raise CredibilityExecutionError(f"Credibility evaluation failed due to unexpected error: {exc}") from exc
            return CredibilityOutput(
                credibility_score=50.0,
                confidence=0.0,
                reason=f"Credibility assessment unavailable due to unexpected error: {exc}",
                limitations="Evaluation could not be completed."
            )

    def evaluate(
        self,
        article: Union[Article, dict, str, Any],
        fallback_on_error: bool = True
    ) -> CredibilityOutput:
        """Synchronously evaluate the observable credibility of a news article.

        Args:
            article: News article (Article instance, dict, or raw text string).
            fallback_on_error: If True, returns a safe neutral fallback on failure instead of raising.

        Returns:
            CredibilityOutput containing credibility_score, confidence, reason, and limitations.

        Raises:
            CredibilityValidationError: If input article is invalid or completely empty (when fallback_on_error is False).
            CredibilityExecutionError: If LLM invocation fails (when fallback_on_error is False).
        """
        try:
            article_data, is_content_missing = self._validate_and_normalize_inputs(article)
        except CredibilityValidationError as exc:
            logger.warning("Credibility validation error: %s", exc)
            if not fallback_on_error:
                raise
            return CredibilityOutput(
                credibility_score=50.0,
                confidence=0.0,
                reason="Article content is missing or invalid; unable to assess credibility indicators.",
                limitations="No usable article text or metadata available to evaluate."
            )

        user_prompt = CREDIBILITY_USER_TEMPLATE.format(
            title=article_data["title"],
            description=article_data["description"],
            content=article_data["content"],
            source_name=article_data["source_name"],
            source_url=article_data["source_url"],
            article_url=article_data["article_url"],
            author=article_data["author"],
            published_at=article_data["published_at"]
        )

        try:
            return self.gemini_service.generate_structured(
                prompt=user_prompt,
                schema=CredibilityOutput,
                system_instruction=CREDIBILITY_SYSTEM_PROMPT,
                temperature=self.temperature,
                model_name=self.model_name
            )
        except GeminiServiceException as exc:
            logger.error("GeminiService error in CredibilityAgent.evaluate: %s", exc)
            if not fallback_on_error:
                raise CredibilityExecutionError(f"Credibility evaluation failed: {exc}") from exc
            return CredibilityOutput(
                credibility_score=50.0,
                confidence=0.0,
                reason=f"Credibility assessment unavailable due to service error: {exc}",
                limitations="Evaluation incomplete due to upstream service failure."
            )
        except Exception as exc:
            logger.exception("Unexpected error in CredibilityAgent.evaluate: %s", exc)
            if not fallback_on_error:
                raise CredibilityExecutionError(f"Credibility evaluation failed due to unexpected error: {exc}") from exc
            return CredibilityOutput(
                credibility_score=50.0,
                confidence=0.0,
                reason=f"Credibility assessment unavailable due to unexpected error: {exc}",
                limitations="Evaluation could not be completed."
            )

    async def aevaluate_batch(
        self,
        articles: List[Union[Article, dict, str, Any]],
        max_concurrency: int = 5,
        fallback_on_error: bool = True
    ) -> List[CredibilityOutput]:
        """Evaluate credibility for multiple articles concurrently.

        Does not perform ranking or composite score synthesis.

        Args:
            articles: List of articles to evaluate.
            max_concurrency: Maximum number of concurrent LLM requests.
            fallback_on_error: If True, individual failures return safe neutral fallbacks.

        Returns:
            List of CredibilityOutput in the same order as input articles.
        """
        if not articles:
            return []

        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _eval_with_sem(art: Union[Article, dict, str, Any]) -> CredibilityOutput:
            async with semaphore:
                return await self.aevaluate(art, fallback_on_error=fallback_on_error)

        tasks = [_eval_with_sem(art) for art in articles]
        return await asyncio.gather(*tasks)
