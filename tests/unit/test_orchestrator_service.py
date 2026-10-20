"""Unit tests for NewsOrchestratorService and deduplication utilities."""

import asyncio
from datetime import datetime, timezone, timedelta
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.agents.credibility_agent import CredibilityAgent, CredibilityOutput
from backend.app.agents.ranking_agent import RankingAgent, RankingWeights
from backend.app.agents.relevance_agent import RelevanceAgent, RelevanceOutput
from backend.app.agents.sentiment_agent import SentimentAgent, SentimentOutput
from backend.app.config import Settings
from backend.app.models.article import Article
from backend.app.models.ranking import RankedArticle
from backend.app.services.news_service import (
    NewsAPIError,
    NewsAPIKeyMissingError,
    NewsAPIService,
    NewsAPITimeoutError,
)
from backend.app.services.orchestrator_service import (
    NewsOrchestratorService,
    AgentOrchestrator,
    OrchestratorError,
    OrchestratorQueryError,
    deduplicate_articles,
    normalize_canonical_url,
)
from backend.app.services.gemini_service import GeminiAPIError
from backend.app.agents.credibility_agent import CredibilityExecutionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_articles():
    """List of sample normalized articles."""
    return [
        Article(
            id="art-orch-001",
            title="Fusion Reactor Sets Sustained Heat Record",
            description="Facility reaches 100M degrees for 48 seconds.",
            content="Researchers at the Tokamak facility confirmed stable plasma confinement at record temperatures.",
            article_url="https://example.com/fusion-record",
            source_name="Energy Science",
            author="Alice Zhang",
            published_at=datetime.now(timezone.utc) - timedelta(hours=3)
        ),
        Article(
            id="art-orch-002",
            title="Solar Cell Breakthrough Achieves 32% Efficiency",
            description="Perovskite tandem cells set commercial benchmark.",
            content="A joint engineering consortium demonstrated high-yield production of record-breaking solar cells.",
            article_url="https://example.com/solar-cells",
            source_name="Renewable Tech",
            author="Bob Miller",
            published_at=datetime.now(timezone.utc) - timedelta(hours=6)
        ),
        Article(
            id="art-orch-003",
            title="Wind Turbine Factory Expands Production in Midwest",
            description="New industrial plant creates 1,200 manufacturing jobs.",
            content="Heavy turbine components will be produced locally to supply offshore and onshore wind farms.",
            article_url="https://example.com/wind-factory",
            source_name="Industrial Gazette",
            author="Carol Evans",
            published_at=datetime.now(timezone.utc) - timedelta(days=2)
        ),
    ]


@pytest.fixture
def mock_news_service(sample_articles):
    service = MagicMock(spec=NewsAPIService)
    service.search_articles = AsyncMock(return_value=sample_articles)
    return service


@pytest.fixture
def mock_relevance_agent():
    agent = MagicMock(spec=RelevanceAgent)
    # Default to high relevance
    agent.aevaluate = AsyncMock(
        return_value=RelevanceOutput(relevance_score=90.0, reason="Strong semantic overlap.")
    )
    return agent


@pytest.fixture
def mock_sentiment_agent():
    agent = MagicMock(spec=SentimentAgent)
    agent.aevaluate = AsyncMock(
        return_value=SentimentOutput(sentiment="positive", sentiment_score=80.0, reason="Constructive news.")
    )
    return agent


@pytest.fixture
def mock_credibility_agent():
    agent = MagicMock(spec=CredibilityAgent)
    agent.aevaluate = AsyncMock(
        return_value=CredibilityOutput(
            credibility_score=85.0,
            confidence=90.0,
            reason="Verified institutional attribution.",
            limitations="No live web replication."
        )
    )
    return agent


@pytest.fixture
def mock_ranking_agent():
    agent = RankingAgent()
    return agent


# ---------------------------------------------------------------------------
# Deduplication & URL Normalization Tests
# ---------------------------------------------------------------------------

class TestDeduplicationAndNormalization:
    def test_normalize_canonical_url_strips_tracking_params(self):
        url1 = "https://example.com/news/fusion?utm_source=twitter&utm_medium=social"
        url2 = "https://www.example.com/news/fusion/?fbclid=12345"
        norm1 = normalize_canonical_url(url1)
        norm2 = normalize_canonical_url(url2)
        assert norm1 == norm2
        assert norm1 == "https://example.com/news/fusion"

    def test_normalize_canonical_url_preserves_meaningful_params(self):
        url = "https://example.com/article?id=42&category=tech"
        norm = normalize_canonical_url(url)
        assert "category=tech" in norm
        assert "id=42" in norm

    def test_deduplicate_articles_removes_identical_urls(self):
        art1 = Article(id="id-1", title="Headline 1", article_url="https://example.com/story?utm_source=mail")
        art2 = Article(id="id-2", title="Headline 2", article_url="https://example.com/story")
        unique = deduplicate_articles([art1, art2])
        assert len(unique) == 1
        assert unique[0].id == "id-1"

    def test_deduplicate_articles_removes_identical_ids(self):
        art1 = Article(id="stable-id-100", title="Headline A", article_url="https://example.com/a")
        art2 = Article(id="stable-id-100", title="Headline B", article_url="https://example.com/b")
        unique = deduplicate_articles([art1, art2])
        assert len(unique) == 1

    def test_deduplicate_articles_removes_normalized_identical_titles(self):
        art1 = Article(id="id-1", title="Quantum Computer Achieves Coherence", article_url="https://example.com/1")
        art2 = Article(id="id-2", title="Quantum Computer Achieves Coherence!", article_url="https://example.com/2")
        unique = deduplicate_articles([art1, art2])
        assert len(unique) == 1

    def test_deduplicate_articles_empty_or_none(self):
        assert deduplicate_articles([]) == []
        assert deduplicate_articles([None]) == []


# ---------------------------------------------------------------------------
# Successful Orchestration Tests
# ---------------------------------------------------------------------------

class TestSuccessfulOrchestration:
    @pytest.mark.asyncio
    async def test_rank_news_end_to_end_success(
        self, mock_news_service, mock_relevance_agent, mock_sentiment_agent,
        mock_credibility_agent, mock_ranking_agent
    ):
        orchestrator = NewsOrchestratorService(
            news_service=mock_news_service,
            relevance_agent=mock_relevance_agent,
            sentiment_agent=mock_sentiment_agent,
            credibility_agent=mock_credibility_agent,
            ranking_agent=mock_ranking_agent
        )

        results = await orchestrator.rank_news(query="clean energy breakthrough", max_articles=3)

        assert len(results) == 3
        mock_news_service.search_articles.assert_called_once()
        assert mock_relevance_agent.aevaluate.call_count == 3
        assert mock_sentiment_agent.aevaluate.call_count == 3
        assert mock_credibility_agent.aevaluate.call_count == 3

        # Verify structured RankedArticle return
        assert all(isinstance(r, RankedArticle) for r in results)
        # Verify rank order (1, 2, 3)
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        # Scores descending
        assert results[0].final_score >= results[1].final_score >= results[2].final_score

    @pytest.mark.asyncio
    async def test_rank_news_with_pre_fetched_articles(
        self, mock_news_service, mock_relevance_agent, mock_sentiment_agent,
        mock_credibility_agent, mock_ranking_agent, sample_articles
    ):
        orchestrator = NewsOrchestratorService(
            news_service=mock_news_service,
            relevance_agent=mock_relevance_agent,
            sentiment_agent=mock_sentiment_agent,
            credibility_agent=mock_credibility_agent,
            ranking_agent=mock_ranking_agent
        )

        results = await orchestrator.rank_news(
            query="fusion power",
            articles=sample_articles[:2],
            max_articles=2,
        )

        assert len(results) == 2
        # Should NOT call NewsAPI when articles are provided
        mock_news_service.search_articles.assert_not_called()
        assert mock_relevance_agent.aevaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_alias_agent_orchestrator(self):
        assert AgentOrchestrator is NewsOrchestratorService


# ---------------------------------------------------------------------------
# Configurable Limits & Concurrency Tests
# ---------------------------------------------------------------------------

class TestConfigurableLimitsAndConcurrency:
    @pytest.mark.asyncio
    async def test_max_articles_configuration_limits_output(
        self, mock_news_service, mock_relevance_agent, mock_sentiment_agent,
        mock_credibility_agent, mock_ranking_agent
    ):
        orchestrator = NewsOrchestratorService(
            news_service=mock_news_service,
            relevance_agent=mock_relevance_agent,
            sentiment_agent=mock_sentiment_agent,
            credibility_agent=mock_credibility_agent,
            ranking_agent=mock_ranking_agent,
            max_articles=2
        )

        results = await orchestrator.rank_news(query="energy")

        assert len(results) == 2
        assert mock_relevance_agent.aevaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_concurrency_bounded_by_semaphore(
        self, mock_news_service, mock_relevance_agent, mock_sentiment_agent,
        mock_credibility_agent, mock_ranking_agent
    ):
        concurrency_limit = 2
        active_calls = 0
        max_seen_calls = 0

        async def tracked_eval(*args, **kwargs):
            nonlocal active_calls, max_seen_calls
            active_calls += 1
            max_seen_calls = max(max_seen_calls, active_calls)
            await asyncio.sleep(0.01)
            active_calls -= 1
            return RelevanceOutput(relevance_score=80.0, reason="Ok")

        mock_relevance_agent.aevaluate = AsyncMock(side_effect=tracked_eval)

        orchestrator = NewsOrchestratorService(
            news_service=mock_news_service,
            relevance_agent=mock_relevance_agent,
            sentiment_agent=mock_sentiment_agent,
            credibility_agent=mock_credibility_agent,
            ranking_agent=mock_ranking_agent,
            concurrency_limit=concurrency_limit
        )

        await orchestrator.rank_news(query="energy")
        assert max_seen_calls <= concurrency_limit


# ---------------------------------------------------------------------------
# Empty Results & Validation Tests
# ---------------------------------------------------------------------------

class TestEmptyResultsAndValidation:
    @pytest.mark.asyncio
    async def test_empty_news_api_results_returns_empty_list(
        self, mock_news_service, mock_relevance_agent, mock_sentiment_agent,
        mock_credibility_agent, mock_ranking_agent
    ):
        mock_news_service.search_articles = AsyncMock(return_value=[])

        orchestrator = NewsOrchestratorService(
            news_service=mock_news_service,
            relevance_agent=mock_relevance_agent,
            sentiment_agent=mock_sentiment_agent,
            credibility_agent=mock_credibility_agent,
            ranking_agent=mock_ranking_agent
        )

        results = await orchestrator.rank_news(query="nonexistent query")
        assert results == []
        mock_relevance_agent.aevaluate.assert_not_called()

    @pytest.mark.parametrize("invalid_query", ["", "   ", None])
    @pytest.mark.asyncio
    async def test_invalid_or_empty_query_raises_orchestrator_query_error(self, invalid_query):
        orchestrator = NewsOrchestratorService()
        with pytest.raises(OrchestratorQueryError):
            await orchestrator.rank_news(query=invalid_query)


# ---------------------------------------------------------------------------
# Partial Agent Failure & Limitations Preservation Tests
# ---------------------------------------------------------------------------

class TestAgentFailureHandling:
    @pytest.mark.asyncio
    async def test_quota_failure_propagates_without_ranking_fallback_scores(
        self, mock_news_service, mock_relevance_agent, mock_sentiment_agent,
        mock_credibility_agent, mock_ranking_agent
    ):
        """Quota exhaustion propagates and the ranking stage is not called."""
        quota_error = GeminiAPIError("Gemini free-tier quota is exhausted (HTTP 429).", status_code=429)
        agent_error = CredibilityExecutionError("Credibility evaluation failed due to quota exhaustion.")
        agent_error.__cause__ = quota_error
        mock_credibility_agent.aevaluate = AsyncMock(
            side_effect=agent_error
        )

        orchestrator = NewsOrchestratorService(
            news_service=mock_news_service,
            relevance_agent=mock_relevance_agent,
            sentiment_agent=mock_sentiment_agent,
            credibility_agent=mock_credibility_agent,
            ranking_agent=mock_ranking_agent
        )

        with patch.object(orchestrator.ranking_agent, "rank_articles", wraps=orchestrator.ranking_agent.rank_articles) as rank_mock:
            with pytest.raises(GeminiAPIError) as exc_info:
                await orchestrator.rank_news(query="energy")

        assert exc_info.value.status_code == 429
        rank_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_unexpected_agent_exception_handled_gracefully(
        self, mock_news_service, mock_relevance_agent, mock_sentiment_agent,
        mock_credibility_agent, mock_ranking_agent
    ):
        """If an agent fails unexpectedly, no ranked response is synthesized."""
        mock_relevance_agent.aevaluate = AsyncMock(side_effect=RuntimeError("Unexpected connection drop"))

        orchestrator = NewsOrchestratorService(
            news_service=mock_news_service,
            relevance_agent=mock_relevance_agent,
            sentiment_agent=mock_sentiment_agent,
            credibility_agent=mock_credibility_agent,
            ranking_agent=mock_ranking_agent
        )

        with patch.object(orchestrator.ranking_agent, "rank_articles", wraps=orchestrator.ranking_agent.rank_articles) as rank_mock:
            with pytest.raises(OrchestratorError, match="no ranked results"):
                await orchestrator.rank_news(query="energy")
        rank_mock.assert_not_called()


# ---------------------------------------------------------------------------
# NewsAPI Error Handling Tests
# ---------------------------------------------------------------------------

class TestNewsAPIErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_news_api_key_propagates_error(self, mock_news_service):
        mock_news_service.search_articles = AsyncMock(
            side_effect=NewsAPIKeyMissingError("NEWS_API_KEY is not configured.")
        )
        orchestrator = NewsOrchestratorService(news_service=mock_news_service)

        with pytest.raises(NewsAPIKeyMissingError):
            await orchestrator.rank_news(query="test")

    @pytest.mark.asyncio
    async def test_news_api_timeout_propagates_error(self, mock_news_service):
        mock_news_service.search_articles = AsyncMock(
            side_effect=NewsAPITimeoutError("NewsAPI request timed out after 10.0s.")
        )
        orchestrator = NewsOrchestratorService(news_service=mock_news_service)

        with pytest.raises(NewsAPITimeoutError):
            await orchestrator.rank_news(query="test")

    @pytest.mark.asyncio
    async def test_news_api_upstream_error_propagates_error(self, mock_news_service):
        mock_news_service.search_articles = AsyncMock(
            side_effect=NewsAPIError("Rate limit exceeded", status_code=429)
        )
        orchestrator = NewsOrchestratorService(news_service=mock_news_service)

        with pytest.raises(NewsAPIError) as exc_info:
            await orchestrator.rank_news(query="test")
        assert exc_info.value.status_code == 429
