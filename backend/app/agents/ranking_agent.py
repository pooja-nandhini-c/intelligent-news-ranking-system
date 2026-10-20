"""Ranking Agent for multi-agent news evaluation and composite score synthesis.

Synthesizes relevance, credibility, freshness, and information quality into a final
ranking score using configurable weights. Assigns ranks and orders articles from highest
to lowest score without allowing sentiment to dominate ranking.
"""

from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.config import Settings, get_settings
from backend.app.models.article import Article
from backend.app.models.ranking import ArticleAnalysis, RankedArticle, SentimentLabel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Input Data Models
# ---------------------------------------------------------------------------

class RankingWeights(BaseModel):
    """Configurable weights for final score synthesis.

    Normalized to sum to 1.0 during score calculation.
    Default formula: 40% Relevance, 25% Credibility, 20% Freshness, 15% Information Quality.
    """

    relevance: float = Field(default=0.40, ge=0.0, description="Weight for relevance score (0.0 to 1.0+)")
    credibility: float = Field(default=0.25, ge=0.0, description="Weight for credibility score (0.0 to 1.0+)")
    freshness: float = Field(default=0.20, ge=0.0, description="Weight for freshness score (0.0 to 1.0+)")
    information_quality: float = Field(default=0.15, ge=0.0, description="Weight for information quality score (0.0 to 1.0+)")

    @model_validator(mode="after")
    def validate_positive_sum(self) -> "RankingWeights":
        total = self.relevance + self.credibility + self.freshness + self.information_quality
        if total <= 0.0:
            raise ValueError("Sum of ranking weights must be strictly greater than 0.")
        return self

    def normalized(self) -> Dict[str, float]:
        """Return normalized weights that sum strictly to 1.0."""
        total = self.relevance + self.credibility + self.freshness + self.information_quality
        return {
            "relevance": self.relevance / total,
            "credibility": self.credibility / total,
            "freshness": self.freshness / total,
            "information_quality": self.information_quality / total,
        }


class ArticleEvaluationInput(BaseModel):
    """Container for an article and its multi-agent evaluation outputs."""

    article: Union[Article, Dict[str, Any]] = Field(..., description="The source Article or dictionary")
    relevance: Any = Field(..., description="RelevanceOutput, score dict, or numeric score")
    credibility: Any = Field(..., description="CredibilityOutput, score dict, or numeric score")
    sentiment: Any = Field(..., description="SentimentOutput, sentiment dict, or label string")
    freshness: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Optional pre-computed freshness score")
    information_quality: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Optional pre-computed quality score")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RankingAgentError(Exception):
    """Base exception for Ranking Agent operations."""
    pass


class RankingValidationError(RankingAgentError, ValueError):
    """Raised when ranking inputs, scores, or query fail validation."""
    pass


# ---------------------------------------------------------------------------
# Utility Scoring Functions (Freshness & Information Quality)
# ---------------------------------------------------------------------------

def calculate_freshness_score(
    published_at: Optional[Union[datetime, str]],
    reference_time: Optional[datetime] = None,
    half_life_days: float = 7.0,
    default_score: float = 50.0
) -> float:
    """Calculate normalized freshness score (0.0 - 100.0) from article publication timestamp.

    Uses continuous exponential decay over article age:
    score = 100.0 * exp(-age_days / half_life_days * ln(2)).
    At age 0 -> 100.0, at 7 days -> 50.0, at 14 days -> 25.0.
    If published_at is missing or unparseable, returns default_score (50.0).
    """
    if published_at is None:
        return default_score

    dt: Optional[datetime] = None
    if isinstance(published_at, datetime):
        dt = published_at
    elif isinstance(published_at, str):
        clean_str = published_at.strip()
        if not clean_str or clean_str.lower() in ("not provided", "unknown", "none"):
            return default_score
        try:
            dt = datetime.fromisoformat(clean_str.replace("Z", "+00:00"))
        except Exception:
            return default_score
    else:
        return default_score

    now = reference_time or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_seconds = max(0.0, (now - dt).total_seconds())
    age_days = age_seconds / 86400.0

    decay = math.exp(-age_days * (math.log(2) / max(0.1, half_life_days)))
    score = round(max(0.0, min(100.0, 100.0 * decay)), 2)
    return score


def calculate_information_quality_score(
    article: Union[Article, Dict[str, Any], Any],
    default_score: float = 60.0
) -> float:
    """Assess observable information quality score (0.0 - 100.0) from article structure and completeness.

    Considers:
    - Content depth and length
    - Description clarity
    - Named author byline presence
    - Source publisher presence
    - Canonical article URL
    """
    if isinstance(article, Article):
        title = article.title
        desc = article.description
        content = article.content
        author = article.author
        source_name = article.source_name
        article_url = article.article_url
    elif isinstance(article, dict):
        title = article.get("title")
        desc = article.get("description")
        content = article.get("content")
        author = article.get("author")
        source_val = article.get("source")
        if isinstance(source_val, dict):
            source_name = source_val.get("name")
        else:
            source_name = article.get("source_name")
        article_url = article.get("article_url") or article.get("url")
    else:
        title = getattr(article, "title", None)
        desc = getattr(article, "description", None)
        content = getattr(article, "content", None)
        author = getattr(article, "author", None)
        source_name = getattr(article, "source_name", None)
        article_url = getattr(article, "article_url", getattr(article, "url", None))

    score = 40.0  # Factual baseline

    # Title quality
    if title and len(str(title).strip()) >= 15:
        score += 10.0

    # Description quality
    if desc and len(str(desc).strip()) >= 30:
        score += 15.0

    # Content completeness
    if content:
        c_str = str(content).strip()
        if len(c_str) >= 200:
            score += 20.0
        elif len(c_str) > 0:
            score += 10.0

    # Author transparency
    if author:
        a_str = str(author).strip()
        if a_str and a_str.lower() not in ("unknown", "none", "not provided"):
            score += 10.0

    # Source credibility presence
    if source_name:
        s_str = str(source_name).strip()
        if s_str and s_str.lower() not in ("unknown", "none"):
            score += 5.0

    return min(100.0, max(0.0, round(score, 2)))


# ---------------------------------------------------------------------------
# Ranking Agent Implementation
# ---------------------------------------------------------------------------

TieStrategy = Literal["standard", "competition", "dense", "sequential"]


class RankingAgent:
    """Agent responsible for synthesizing multi-agent evaluations into final ranked articles.

    Applies normalized linear weighting across relevance, credibility, freshness,
    and information quality. Keeps sentiment separate from final score to prevent bias.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        weights: Optional[Union[RankingWeights, Dict[str, float]]] = None,
        tie_strategy: TieStrategy = "standard"
    ):
        self.settings = settings or get_settings()

        if weights is not None:
            if isinstance(weights, RankingWeights):
                self.weights = weights
            elif isinstance(weights, dict):
                self.weights = RankingWeights(**weights)
            else:
                raise RankingValidationError(f"Invalid weights type: {type(weights).__name__}")
        else:
            cfg_weights = self.settings.get_ranking_weights()
            self.weights = RankingWeights(**cfg_weights)

        self.tie_strategy = tie_strategy

    def calculate_final_score(
        self,
        relevance_score: float,
        credibility_score: float,
        freshness_score: float,
        information_quality_score: float
    ) -> float:
        """Calculate normalized composite final score (0.0 to 100.0) using configured weights."""
        norm = self.weights.normalized()
        raw = (
            norm["relevance"] * relevance_score +
            norm["credibility"] * credibility_score +
            norm["freshness"] * freshness_score +
            norm["information_quality"] * information_quality_score
        )
        return round(max(0.0, min(100.0, raw)), 2)

    def _extract_score(self, val: Any, name: str, default: Optional[float] = None) -> float:
        """Extract and validate a numeric score between 0.0 and 100.0."""
        if val is None:
            if default is not None:
                return default
            raise RankingValidationError(f"Missing required score for '{name}'.")

        # Object attribute check (e.g., RelevanceOutput, CredibilityOutput)
        target = val
        if hasattr(val, f"{name}_score"):
            target = getattr(val, f"{name}_score")
        elif hasattr(val, "score"):
            target = getattr(val, "score")
        elif isinstance(val, dict):
            if f"{name}_score" in val:
                target = val[f"{name}_score"]
            elif "score" in val:
                target = val["score"]
            elif name in val:
                target = val[name]

        try:
            num = float(target)
        except (ValueError, TypeError) as exc:
            raise RankingValidationError(f"Invalid non-numeric score for '{name}': {target}") from exc

        if math.isnan(num) or math.isinf(num):
            raise RankingValidationError(f"Score for '{name}' must be a finite number, got {num}")

        if num < 0.0 or num > 100.0:
            raise RankingValidationError(f"Score for '{name}' must be between 0.0 and 100.0, got {num}")

        return round(num, 2)

    def _extract_sentiment(self, val: Any) -> Tuple[str, float]:
        """Extract sentiment label and score from SentimentOutput, dict, or string."""
        if val is None:
            raise RankingValidationError("Missing required sentiment analysis.")

        sentiment_label = "Neutral"
        sentiment_score = 50.0

        if hasattr(val, "sentiment") and hasattr(val, "sentiment_score"):
            raw_label = getattr(val, "sentiment")
            sentiment_label = str(raw_label).title()
            sentiment_score = float(getattr(val, "sentiment_score"))
        elif isinstance(val, dict):
            if "sentiment" in val:
                sentiment_label = str(val["sentiment"]).title()
            elif "label" in val:
                sentiment_label = str(val["label"]).title()
            elif "polarity" in val:
                sentiment_label = str(val["polarity"]).title()

            if "sentiment_score" in val:
                sentiment_score = float(val["sentiment_score"])
            elif "score" in val:
                sentiment_score = float(val["score"])
        elif isinstance(val, str):
            clean = val.strip()
            if not clean:
                raise RankingValidationError("Sentiment label cannot be empty.")
            sentiment_label = clean.title()
        elif isinstance(val, SentimentLabel):
            sentiment_label = val.value
        else:
            raise RankingValidationError(f"Unsupported sentiment analysis input type: {type(val).__name__}")

        if sentiment_score < 0.0 or sentiment_score > 100.0:
            raise RankingValidationError(f"sentiment_score must be between 0.0 and 100.0, got {sentiment_score}")

        return sentiment_label, round(sentiment_score, 2)

    def _normalize_article(self, article: Any) -> Article:
        """Validate and convert article input to an Article model instance."""
        if article is None:
            raise RankingValidationError("Article cannot be None.")

        if isinstance(article, Article):
            return article

        if isinstance(article, dict):
            try:
                # Ensure id exists or generate
                art_dict = dict(article)
                if "id" not in art_dict or not art_dict["id"]:
                    art_dict["id"] = Article.generate_id(
                        url=art_dict.get("article_url") or art_dict.get("url"),
                        title=art_dict.get("title", "")
                    )
                return Article.model_validate(art_dict)
            except Exception as exc:
                raise RankingValidationError(f"Failed to parse article dictionary into Article model: {exc}") from exc

        raise RankingValidationError(
            f"Unsupported article type: {type(article).__name__}. Expected Article model or dict."
        )

    def _generate_explanation(
        self,
        final_score: float,
        relevance_score: float,
        credibility_score: float,
        freshness_score: float,
        information_quality_score: float,
        sentiment: str,
        sentiment_score: float
    ) -> str:
        """Generate a concise, objective explanation of why the article received its score."""
        drivers = []
        if relevance_score >= 80.0:
            drivers.append(f"high topical relevance ({relevance_score:.1f})")
        elif relevance_score <= 40.0:
            drivers.append(f"limited relevance ({relevance_score:.1f})")

        if credibility_score >= 80.0:
            drivers.append(f"strong credibility indicators ({credibility_score:.1f})")
        elif credibility_score <= 40.0:
            drivers.append(f"unverified/informal sourcing ({credibility_score:.1f})")

        if freshness_score >= 85.0:
            drivers.append("recent publication")
        elif freshness_score <= 30.0:
            drivers.append("older publication date")

        driver_str = f" Driven primarily by {', '.join(drivers)}." if drivers else ""

        return (
            f"Final score {final_score:.2f} synthesized across relevance ({relevance_score:.1f}), "
            f"credibility ({credibility_score:.1f}), freshness ({freshness_score:.1f}), and "
            f"information quality ({information_quality_score:.1f}).{driver_str} "
            f"Sentiment is {sentiment} ({sentiment_score:.1f}) and did not influence the ranking score."
        )

    def rank_article(
        self,
        query: str,
        article: Union[Article, Dict[str, Any]],
        relevance: Any,
        sentiment: Any,
        credibility: Any,
        freshness: Optional[float] = None,
        information_quality: Optional[float] = None,
        rank: int = 1
    ) -> RankedArticle:
        """Rank a single article and return a structured RankedArticle object.

        Args:
            query: User search query or topic string.
            article: Article instance or dictionary.
            relevance: Relevance evaluation output, dict, or numeric score.
            sentiment: Sentiment evaluation output, dict, or label.
            credibility: Credibility evaluation output, dict, or numeric score.
            freshness: Optional pre-computed freshness score (0-100).
            information_quality: Optional pre-computed quality score (0-100).
            rank: Assigned rank index (>= 1).

        Returns:
            RankedArticle containing source article, multi-agent analysis, final score, and rank.

        Raises:
            RankingValidationError: If any input score or metadata fails validation.
        """
        if not isinstance(query, str) or not query.strip():
            raise RankingValidationError("Query must be a non-empty string.")

        if rank < 1:
            raise RankingValidationError(f"Rank must be an integer >= 1, got {rank}")

        article_obj = self._normalize_article(article)
        clean_rel = self._extract_score(relevance, "relevance")
        clean_cred = self._extract_score(credibility, "credibility")
        clean_sent_label, clean_sent_score = self._extract_sentiment(sentiment)

        # Freshness score
        if freshness is not None:
            clean_fresh = self._extract_score(freshness, "freshness")
        else:
            clean_fresh = calculate_freshness_score(article_obj.published_at)

        # Information quality score
        if information_quality is not None:
            clean_iq = self._extract_score(information_quality, "information_quality")
        else:
            clean_iq = calculate_information_quality_score(article_obj)

        final_score = self.calculate_final_score(
            relevance_score=clean_rel,
            credibility_score=clean_cred,
            freshness_score=clean_fresh,
            information_quality_score=clean_iq
        )

        explanation = self._generate_explanation(
            final_score=final_score,
            relevance_score=clean_rel,
            credibility_score=clean_cred,
            freshness_score=clean_fresh,
            information_quality_score=clean_iq,
            sentiment=clean_sent_label,
            sentiment_score=clean_sent_score
        )

        # Extract credibility confidence and limitations if present
        cred_conf = None
        cred_lim = None
        if hasattr(credibility, "confidence"):
            cred_conf = getattr(credibility, "confidence")
        elif isinstance(credibility, dict):
            cred_conf = credibility.get("confidence")

        if hasattr(credibility, "limitations"):
            cred_lim = getattr(credibility, "limitations")
        elif isinstance(credibility, dict):
            cred_lim = credibility.get("limitations")

        analysis = ArticleAnalysis(
            relevance_score=clean_rel,
            sentiment=clean_sent_label,
            sentiment_score=clean_sent_score,
            credibility_score=clean_cred,
            information_quality_score=clean_iq,
            explanation=explanation,
            confidence=cred_conf,
            limitations=cred_lim
        )

        return RankedArticle(
            article=article_obj,
            analysis=analysis,
            final_score=final_score,
            rank=rank
        )

    def rank_articles(
        self,
        query: str,
        items: List[Union[ArticleEvaluationInput, Dict[str, Any], Tuple[Any, ...]]],
        tie_strategy: Optional[TieStrategy] = None
    ) -> List[RankedArticle]:
        """Rank and sort multiple articles from highest score to lowest score.

        Assigns ranks based on the specified tie_strategy (defaulting to competition ranking).

        Args:
            query: User search query or topic string.
            items: List of evaluated articles (ArticleEvaluationInput, dict, or tuple).
            tie_strategy: 'standard'/'competition' (1,2,2,4), 'dense' (1,2,2,3), or 'sequential' (1,2,3,4).

        Returns:
            List of RankedArticle sorted descending by final_score.
        """
        if not isinstance(query, str) or not query.strip():
            raise RankingValidationError("Query must be a non-empty string.")

        if not items:
            return []

        active_strategy = tie_strategy or self.tie_strategy
        evaluated: List[RankedArticle] = []

        for item in items:
            if isinstance(item, ArticleEvaluationInput):
                ranked = self.rank_article(
                    query=query,
                    article=item.article,
                    relevance=item.relevance,
                    sentiment=item.sentiment,
                    credibility=item.credibility,
                    freshness=item.freshness,
                    information_quality=item.information_quality,
                    rank=1  # Temporary rank before sorting
                )
            elif isinstance(item, dict):
                ranked = self.rank_article(
                    query=query,
                    article=item.get("article"),
                    relevance=item.get("relevance") if "relevance" in item else item.get("relevance_score"),
                    sentiment=item.get("sentiment") if "sentiment" in item else item.get("sentiment_score"),
                    credibility=item.get("credibility") if "credibility" in item else item.get("credibility_score"),
                    freshness=item.get("freshness") or item.get("freshness_score"),
                    information_quality=item.get("information_quality") or item.get("information_quality_score"),
                    rank=1
                )
            elif isinstance(item, (list, tuple)):
                if len(item) < 4:
                    raise RankingValidationError(
                        f"Tuple evaluation item must have at least 4 elements (article, relevance, sentiment, credibility), got {len(item)}"
                    )
                art = item[0]
                rel = item[1]
                sent = item[2]
                cred = item[3]
                fresh = item[4] if len(item) > 4 else None
                iq = item[5] if len(item) > 5 else None
                ranked = self.rank_article(
                    query=query,
                    article=art,
                    relevance=rel,
                    sentiment=sent,
                    credibility=cred,
                    freshness=fresh,
                    information_quality=iq,
                    rank=1
                )
            else:
                raise RankingValidationError(f"Unsupported evaluation item type: {type(item).__name__}")

            evaluated.append(ranked)

        # Sort descending by final_score, then tie-break deterministically
        evaluated.sort(
            key=lambda r: (
                -r.final_score,
                -r.analysis.relevance_score,
                -r.analysis.credibility_score,
                -r.analysis.information_quality_score,
                r.article.title
            )
        )

        # Assign final ranks based on tie_strategy
        ranked_results: List[RankedArticle] = []
        for i, item in enumerate(evaluated):
            if i == 0:
                curr_rank = 1
            else:
                prev = evaluated[i - 1]
                if item.final_score == prev.final_score:
                    if active_strategy in ("standard", "competition"):
                        # In competition ranking, equal scores share previous rank
                        curr_rank = ranked_results[i - 1].rank
                    elif active_strategy == "dense":
                        # In dense ranking, equal scores share previous rank
                        curr_rank = ranked_results[i - 1].rank
                    elif active_strategy == "sequential":
                        curr_rank = i + 1
                    else:
                        curr_rank = ranked_results[i - 1].rank
                else:
                    if active_strategy in ("standard", "competition"):
                        curr_rank = i + 1
                    elif active_strategy == "dense":
                        curr_rank = ranked_results[i - 1].rank + 1
                    elif active_strategy == "sequential":
                        curr_rank = i + 1
                    else:
                        curr_rank = i + 1

            # Build final RankedArticle with computed rank
            ranked_results.append(
                RankedArticle(
                    article=item.article,
                    analysis=item.analysis,
                    final_score=item.final_score,
                    rank=curr_rank
                )
            )

        return ranked_results
