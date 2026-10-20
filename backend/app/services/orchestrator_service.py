"""Agent Orchestration service for Intelligent News Ranking System.

Coordinates article retrieval via NewsAPIService, normalization and deduplication,
concurrent evaluation across Relevance, Sentiment, and Credibility agents,
and final score synthesis and sorting via RankingAgent.
"""

import asyncio
from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from backend.app.agents.credibility_agent import CredibilityAgent, CredibilityOutput
from backend.app.agents.ranking_agent import ArticleEvaluationInput, RankingAgent
from backend.app.agents.relevance_agent import RelevanceAgent, RelevanceOutput
from backend.app.agents.sentiment_agent import SentimentAgent, SentimentOutput
from backend.app.config import Settings, get_settings
from backend.app.models.article import Article
from backend.app.models.ranking import RankedArticle
from backend.app.services.news_service import (
    NewsAPIError,
    NewsAPIException,
    NewsAPIKeyMissingError,
    NewsAPIService,
    NewsAPITimeoutError,
)
from backend.app.services.gemini_service import GeminiServiceException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OrchestratorError(Exception):
    """Base exception for Orchestration Service operations."""
    pass


class OrchestratorQueryError(OrchestratorError, ValueError):
    """Raised when user query is missing, empty, or invalid."""
    pass


# ---------------------------------------------------------------------------
# URL Normalization & Deduplication Utilities
# ---------------------------------------------------------------------------

def normalize_canonical_url(url: Optional[str]) -> str:
    """Normalize a URL to canonical form for duplicate detection.

    Strips tracking query parameters (utm_*, ref, etc.), fragments, trailing slashes,
    default ports, and www prefix.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove standard port specifications
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif netloc.endswith(":443"):
            netloc = netloc[:-4]

        # Strip www. prefix for consistent matching
        if netloc.startswith("www."):
            netloc = netloc[4:]

        path = parsed.path.rstrip("/")

        # Filter tracking and campaign parameters
        query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
        clean_pairs = [
            (k, v) for k, v in query_pairs
            if not k.lower().startswith("utm_") and k.lower() not in ("ref", "source", "fbclid", "gclid")
        ]
        clean_query = urlencode(sorted(clean_pairs))

        return urlunparse((scheme, netloc, path, "", clean_query, ""))
    except Exception:
        return url.strip().lower().rstrip("/")


def deduplicate_articles(articles: List[Article]) -> List[Article]:
    """Remove duplicate articles using stable IDs, canonical URLs, and normalized titles.

    Preserves the earliest/most complete article occurrence.
    """
    if not articles:
        return []

    seen_ids = set()
    seen_urls = set()
    seen_titles = set()
    unique_articles: List[Article] = []

    for art in articles:
        if not art or not art.title:
            continue

        canon_url = normalize_canonical_url(art.article_url)
        # Normalize title: remove punctuation and whitespace
        norm_title = re.sub(r"[^\w\s]", "", art.title.lower()).strip()

        is_dup = False
        if art.id and art.id in seen_ids:
            is_dup = True
        elif canon_url and canon_url in seen_urls:
            is_dup = True
        elif norm_title and norm_title in seen_titles:
            is_dup = True

        if not is_dup:
            if art.id:
                seen_ids.add(art.id)
            if canon_url:
                seen_urls.add(canon_url)
            if norm_title:
                seen_titles.add(norm_title)
            unique_articles.append(art)
        else:
            logger.debug("Filtered duplicate article: id=%s, title='%s'", art.id, art.title)

    return unique_articles


# ---------------------------------------------------------------------------
# Orchestration Service Implementation
# ---------------------------------------------------------------------------

class NewsOrchestratorService:
    """Orchestration layer coordinating retrieval, multi-agent evaluation, and ranking."""

    def __init__(
        self,
        news_service: Optional[NewsAPIService] = None,
        relevance_agent: Optional[RelevanceAgent] = None,
        sentiment_agent: Optional[SentimentAgent] = None,
        credibility_agent: Optional[CredibilityAgent] = None,
        ranking_agent: Optional[RankingAgent] = None,
        settings: Optional[Settings] = None,
        max_articles: Optional[int] = None,
        concurrency_limit: Optional[int] = None
    ):
        self.settings = settings or get_settings()
        self.news_service = news_service or NewsAPIService(settings=self.settings)
        self.relevance_agent = relevance_agent or RelevanceAgent()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        self.credibility_agent = credibility_agent or CredibilityAgent()
        self.ranking_agent = ranking_agent or RankingAgent(settings=self.settings)

        self.max_articles = max_articles or self.settings.max_articles_to_rank
        self.concurrency_limit = concurrency_limit or self.settings.agent_concurrency_limit

    async def _evaluate_article(
        self,
        query: str,
        article: Article,
        semaphore: asyncio.Semaphore
    ) -> ArticleEvaluationInput:
        """Evaluate a single article concurrently across Relevance, Sentiment, and Credibility.

        Propagates required agent failures so the system never ranks default scores.
        """
        async with semaphore:
            tasks = [
                asyncio.create_task(self.relevance_agent.aevaluate(
                    query=query, article=article, fallback_on_error=False
                )),
                asyncio.create_task(self.sentiment_agent.aevaluate(
                    article=article, fallback_on_error=False
                )),
                asyncio.create_task(self.credibility_agent.aevaluate(
                    article=article, fallback_on_error=False
                )),
            ]
            try:
                results = await asyncio.gather(*tasks)
            except Exception as result:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                cause = result
                while cause is not None:
                    if isinstance(cause, GeminiServiceException):
                        raise cause
                    cause = cause.__cause__
                logger.error("Required agent failed for article '%s': %s", article.id, result)
                raise OrchestratorError(
                    "A required article evaluation failed; no ranked results were produced."
                ) from result

            rel_out, sent_out, cred_out = results

            return ArticleEvaluationInput(
                article=article,
                relevance=rel_out,
                sentiment=sent_out,
                credibility=cred_out
            )

    async def rank_news(
        self,
        query: str,
        max_articles: Optional[int] = None,
        page: int = 1,
        language: str = "en",
        sort_by: str = "relevancy",
        articles: Optional[List[Article]] = None
    ) -> List[RankedArticle]:
        """Execute end-to-end news search, evaluation, and ranking for a user query.

        Args:
            query: User search query keywords.
            max_articles: Optional limit on the number of articles to rank.
            page: Page number for NewsAPI pagination.
            language: 2-letter ISO language code (default 'en').
            sort_by: Sorting criteria for NewsAPI ('relevancy', 'popularity', 'publishedAt').
            articles: Optional pre-fetched article list (skips NewsAPI retrieval if provided).

        Returns:
            List of RankedArticle structured objects sorted from highest to lowest score.

        Raises:
            OrchestratorQueryError: If query is missing or empty.
            NewsAPIKeyMissingError: If NEWS_API_KEY is not configured and retrieval is attempted.
            NewsAPITimeoutError: If NewsAPI connection times out.
            NewsAPIError: If NewsAPI returns an upstream error.
        """
        if not isinstance(query, str) or not query.strip():
            raise OrchestratorQueryError("Query must be a non-empty string.")

        clean_query = query.strip()
        limit = max_articles if max_articles is not None else self.max_articles
        limit = max(1, limit)

        # Step 1: Article Retrieval
        if articles is not None:
            raw_articles = articles
        else:
            logger.info("Fetching articles from NewsAPI for query: '%s' (limit=%d)", clean_query, limit)
            # Fetch up to max(limit * 2, 20) to have sufficient buffer for deduplication
            fetch_size = min(100, max(limit * 2, 20))
            raw_articles = await self.news_service.search_articles(
                query=clean_query,
                page=page,
                page_size=fetch_size,
                sort_by=sort_by,
                language=language
            )

        # Step 2: Normalization and Deduplication
        unique_articles = deduplicate_articles(raw_articles)
        selected_articles = unique_articles[:limit]

        if not selected_articles:
            logger.info("No articles available for query '%s' after retrieval and deduplication.", clean_query)
            return []

        logger.info(
            "Evaluating %d unique articles with bounded concurrency (limit=%d) for query '%s'",
            len(selected_articles), self.concurrency_limit, clean_query
        )

        # Step 3: Concurrent Multi-Agent Evaluation
        semaphore = asyncio.Semaphore(max(1, self.concurrency_limit))
        eval_tasks = [
            asyncio.create_task(self._evaluate_article(clean_query, art, semaphore))
            for art in selected_articles
        ]
        try:
            evaluation_inputs: List[ArticleEvaluationInput] = await asyncio.gather(*eval_tasks)
        except Exception:
            for task in eval_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*eval_tasks, return_exceptions=True)
            raise

        # Step 4: Final Score Synthesis & Sorting via RankingAgent
        ranked_results = self.ranking_agent.rank_articles(
            query=clean_query,
            items=evaluation_inputs
        )

        # Step 5: Preserve Agent Limitations & Fallback Warnings
        # Ensure fallback scores are never silently presented as reliable assessments
        eval_map = {
            (item.article.id if hasattr(item.article, "id") else item.article.get("id")): item
            for item in evaluation_inputs
        }
        for item in ranked_results:
            eval_in = eval_map.get(item.article.id)
            if eval_in:
                limitations_notes = []
                # Check credibility limitations
                cred = eval_in.credibility
                if hasattr(cred, "confidence") and cred.confidence == 0.0:
                    limitations_notes.append("Credibility assessment unavailable (fallback score applied)")
                elif hasattr(cred, "limitations") and cred.limitations and "incomplete" in str(cred.limitations).lower():
                    limitations_notes.append(f"Credibility limitation: {cred.limitations}")

                # Check relevance failure
                rel = eval_in.relevance
                if hasattr(rel, "reason") and "unavailable" in str(rel.reason).lower():
                    limitations_notes.append("Relevance evaluation failed (fallback applied)")

                # If any limitations/failures occurred, append to explanation
                if limitations_notes:
                    notice = " [Assessment Notice: " + "; ".join(limitations_notes) + "]"
                    if notice not in item.analysis.explanation:
                        item.analysis.explanation += notice

        return ranked_results


# Convenience Alias
AgentOrchestrator = NewsOrchestratorService
