"""Pydantic data models for article analysis and ranking."""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional, List
from enum import Enum

from backend.app.models.article import Article


class SentimentLabel(str, Enum):
    """Common sentiment classifications."""
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    MIXED = "Mixed"


class ArticleAnalysis(BaseModel):
    """Multi-dimensional analysis scores and explanation for a news article."""

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Semantic alignment score between article and query/category (0.0 to 100.0)"
    )
    sentiment: str = Field(
        ...,
        min_length=1,
        description="Sentiment classification label (e.g., Positive, Neutral, Negative)"
    )
    sentiment_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Sentiment metric normalized between 0.0 and 100.0"
    )
    credibility_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Observable journalistic quality and credibility score (0.0 to 100.0)"
    )
    information_quality_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Assessment of structure, citations, and informativeness (0.0 to 100.0)"
    )
    explanation: str = Field(
        ...,
        min_length=1,
        description="Explanatory rationale synthesizing the agent evaluations"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Confidence level in the credibility assessment (0.0 to 100.0)"
    )
    limitations: Optional[str] = Field(
        default=None,
        description="Observable limitations or data constraints if applicable"
    )

    @field_validator("sentiment", mode="before")
    @classmethod
    def clean_sentiment(cls, v: Any) -> str:
        """Strip whitespace and normalize sentiment string."""
        if isinstance(v, SentimentLabel):
            return v.value
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("Sentiment cannot be empty or whitespace only.")
            return cleaned.title()
        raise ValueError(f"Invalid sentiment type: {type(v).__name__}")

    @field_validator("explanation", mode="before")
    @classmethod
    def clean_explanation(cls, v: Any) -> str:
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("Explanation cannot be empty or whitespace only.")
            return cleaned
        return str(v)


class RankedArticle(BaseModel):
    """Article coupled with its multi-dimensional analysis, composite score, and rank."""

    article: Article = Field(
        ...,
        description="The source article data"
    )
    analysis: ArticleAnalysis = Field(
        ...,
        description="The multi-agent evaluation output"
    )
    final_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Final composite score synthesized across dimensions (0.0 to 100.0)"
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Final rank position (1-indexed integer, >= 1)"
    )

    @field_validator("final_score")
    @classmethod
    def round_final_score(cls, v: float) -> float:
        """Keep final score formatted cleanly to 2 decimal places."""
        return round(v, 2)


class RankedNewsSearchResponse(BaseModel):
    """Structured response containing ranked news articles."""

    status: str = Field(default="ok", description="Status string: 'ok' or 'error'")
    query: str = Field(..., description="The evaluated search query")
    total_results: int = Field(default=0, ge=0, description="Total count of ranked articles")
    articles: List[RankedArticle] = Field(default_factory=list, description="Ranked articles in descending score order")
