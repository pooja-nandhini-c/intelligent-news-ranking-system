"""Unit tests for ranked news FastAPI endpoints using TestClient and mocks."""

import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.article import Article
from backend.app.models.ranking import ArticleAnalysis, RankedArticle, RankedNewsSearchResponse
from backend.app.api.ranked_news_routes import get_orchestrator_service
from backend.app.config import get_settings
from backend.app.services.orchestrator_service import (
    NewsOrchestratorService,
    OrchestratorQueryError,
    OrchestratorError,
)
from backend.app.services.news_service import (
    NewsAPIKeyMissingError,
    NewsAPITimeoutError,
    NewsAPIError,
)
from backend.app.services.gemini_service import (
    GeminiAPIKeyMissingError,
    GeminiTimeoutError,
    GeminiAPIError,
    GeminiMalformedResponseError,
    GeminiServiceException,
)

client = TestClient(app)


def make_test_ranked_article(
    title: str = "Test Headline",
    url: str = "https://example.com/test",
    final_score: float = 85.5,
    rank: int = 1,
    relevance_score: float = 90.0,
    sentiment: str = "Positive",
    sentiment_score: float = 75.0,
    credibility_score: float = 80.0,
    quality_score: float = 85.0,
    explanation: str = "High relevance and strong factual reporting.",
    confidence: float = 85.0,
    limitations: str = None
) -> RankedArticle:
    """Helper to construct a valid RankedArticle model instance."""
    article = Article(
        id=Article.generate_id(url, title),
        title=title,
        description="A test article summary.",
        content="Clean article body content for testing.",
        url=url,
        image_url="https://example.com/image.png",
        published_at="2026-10-10T12:00:00Z",
        source_name="Tech Dispatch",
        author="Reporter Smith"
    )
    analysis = ArticleAnalysis(
        relevance_score=relevance_score,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        credibility_score=credibility_score,
        information_quality_score=quality_score,
        explanation=explanation,
        confidence=confidence,
        limitations=limitations
    )
    return RankedArticle(
        article=article,
        analysis=analysis,
        final_score=final_score,
        rank=rank
    )


# ===========================================================================
# Health & Root Endpoint Tests
# ===========================================================================

class TestHealthAndRootEndpoints:
    """Tests for /health, /api/health, and / endpoints."""

    def test_health_endpoint_success(self):
        """GET /health returns 200 with status ok and does not require API keys."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data
        # Ensure no secrets leaked
        assert "api_key" not in response.text
        assert "gemini_api_key" not in response.text

    def test_legacy_api_health_endpoint(self):
        """GET /api/health returns 200 with service operational status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "gemini_configured" in data
        assert "newsapi_configured" in data

    def test_root_endpoint(self):
        """GET / returns 200 with service welcome info and documentation links."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["docs_url"] == "/docs"
        assert data["health_check"] == "/health"
        assert data["api_v1_search"] == "/api/v1/news/search"


# ===========================================================================
# Search Endpoint - Success Scenarios
# ===========================================================================

class TestRankedNewsSearchSuccess:
    """Tests for successful searches and structured ranking output."""

    def test_search_valid_query_returns_ranked_articles(self):
        """GET /api/v1/news/search with valid q returns 200 and structured results in descending order."""
        article_1 = make_test_ranked_article(
            title="Top AI Milestone",
            url="https://example.com/ai1",
            final_score=92.5,
            rank=1,
            relevance_score=95.0,
            confidence=90.0,
            limitations=None
        )
        article_2 = make_test_ranked_article(
            title="Secondary AI Update",
            url="https://example.com/ai2",
            final_score=78.0,
            rank=2,
            relevance_score=80.0,
            confidence=70.0,
            limitations="Limited citations."
        )

        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.return_value = [article_1, article_2]

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=Artificial+Intelligence")
            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "ok"
            assert data["query"] == "Artificial Intelligence"
            assert data["total_results"] == 2
            assert len(data["articles"]) == 2

            # Descending order check
            assert data["articles"][0]["final_score"] >= data["articles"][1]["final_score"]
            assert data["articles"][0]["rank"] == 1
            assert data["articles"][1]["rank"] == 2

            # First article verification
            first = data["articles"][0]
            assert first["article"]["title"] == "Top AI Milestone"
            assert first["article"]["article_url"] == "https://example.com/ai1"
            assert first["analysis"]["relevance_score"] == 95.0
            assert first["analysis"]["sentiment"] == "Positive"
            assert first["analysis"]["confidence"] == 90.0
            assert first["analysis"]["explanation"] == "High relevance and strong factual reporting."
            assert first["final_score"] == 92.5

            # Second article limitations verification
            second = data["articles"][1]
            assert second["analysis"]["limitations"] == "Limited citations."

            # Verify orchestrator was called with default config limit
            mock_orchestrator.rank_news.assert_awaited_once_with(
                query="Artificial Intelligence",
                max_articles=get_settings().max_articles_to_rank
            )
        finally:
            app.dependency_overrides.clear()

    def test_search_with_custom_limit(self):
        """GET /api/v1/news/search with custom limit forwards limit to orchestrator."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.return_value = [make_test_ranked_article()]

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=quantum+computing&limit=7")
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 1

            mock_orchestrator.rank_news.assert_awaited_once_with(
                query="quantum computing",
                max_articles=7
            )
        finally:
            app.dependency_overrides.clear()

    def test_search_empty_results_returns_200(self):
        """GET /api/v1/news/search returning zero articles yields 200 with empty list."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.return_value = []

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=obscuretopicneverbeforeheard")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["query"] == "obscuretopicneverbeforeheard"
            assert data["total_results"] == 0
            assert data["articles"] == []
        finally:
            app.dependency_overrides.clear()


# ===========================================================================
# Search Endpoint - Input Validation
# ===========================================================================

class TestRankedNewsSearchValidation:
    """Tests for parameter validation on /api/v1/news/search."""

    def test_missing_query_parameter_returns_422(self):
        """Missing required 'q' parameter returns 422 Unprocessable Entity."""
        response = client.get("/api/v1/news/search")
        assert response.status_code == 422

    def test_empty_string_query_returns_422(self):
        """Empty string 'q' fails min_length=1 validation and returns 422."""
        response = client.get("/api/v1/news/search?q=")
        assert response.status_code == 422

    def test_whitespace_only_query_returns_400(self):
        """Whitespace-only query passes min_length but fails stripped check returning 400."""
        response = client.get("/api/v1/news/search?q=%20%20%20")
        assert response.status_code == 400
        assert "empty or whitespace only" in response.json()["detail"].lower()

    def test_limit_zero_returns_422(self):
        """limit=0 violates ge=1 boundary and returns 422."""
        response = client.get("/api/v1/news/search?q=test&limit=0")
        assert response.status_code == 422

    def test_limit_negative_returns_422(self):
        """Negative limit violates ge=1 boundary and returns 422."""
        response = client.get("/api/v1/news/search?q=test&limit=-5")
        assert response.status_code == 422

    def test_limit_exceeding_max_returns_422(self):
        """limit=51 violates le=50 boundary and returns 422."""
        response = client.get("/api/v1/news/search?q=test&limit=51")
        assert response.status_code == 422

    def test_non_integer_limit_returns_422(self):
        """Non-integer limit returns 422."""
        response = client.get("/api/v1/news/search?q=test&limit=notanumber")
        assert response.status_code == 422

    def test_orchestrator_query_error_returns_400(self):
        """OrchestratorQueryError maps to 400 Bad Request."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = OrchestratorQueryError("Query must be a non-empty string.")

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=validlookingquery")
            assert response.status_code == 400
            assert "non-empty string" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


# ===========================================================================
# Search Endpoint - Upstream and Service Error Handling
# ===========================================================================

class TestRankedNewsSearchErrorHandling:
    """Tests for safe error handling and mapping across upstream dependencies."""

    def test_news_api_key_missing_returns_503(self):
        """NewsAPIKeyMissingError maps to 503 without leaking configuration details."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = NewsAPIKeyMissingError("NEWS_API_KEY not configured.")

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=geopolitics")
            assert response.status_code == 503
            data = response.json()
            assert "News retrieval service is unavailable" in data["detail"]
            assert "NEWS_API_KEY" not in response.text
        finally:
            app.dependency_overrides.clear()

    def test_gemini_api_key_missing_returns_503(self):
        """GeminiAPIKeyMissingError maps to 503 without leaking secrets."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = GeminiAPIKeyMissingError("GEMINI_API_KEY missing.")

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=robotics")
            assert response.status_code == 503
            data = response.json()
            assert "Language model evaluation service is unavailable" in data["detail"]
            assert "GEMINI_API_KEY" not in response.text
        finally:
            app.dependency_overrides.clear()

    def test_news_api_timeout_returns_504(self):
        """NewsAPITimeoutError maps to 504 Gateway Timeout."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = NewsAPITimeoutError("NewsAPI timeout after 10s.")

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=macroeconomics")
            assert response.status_code == 504
            assert "timed out" in response.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()

    def test_gemini_timeout_returns_504(self):
        """GeminiTimeoutError maps to 504 Gateway Timeout."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = GeminiTimeoutError("Gemini call timed out.")

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=aerospace")
            assert response.status_code == 504
            assert "timed out" in response.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()

    def test_news_api_rate_limit_returns_429(self):
        """NewsAPI 429 status maps to 429 Too Many Requests."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = NewsAPIError("Rate limit exceeded", status_code=429)

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=fintech")
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_news_api_auth_error_maps_to_502(self):
        """NewsAPI 401/403 upstream errors map to 502 Bad Gateway."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = NewsAPIError("Invalid api key", status_code=401)

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=telecom")
            assert response.status_code == 502
            assert "rejected authentication" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_news_api_general_error_maps_to_502(self):
        """NewsAPI 500 error maps to 502 Bad Gateway."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = NewsAPIError("Internal server error", status_code=500)

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=cloud")
            assert response.status_code == 502
            assert "Upstream news retrieval provider returned an error" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_gemini_api_rate_limit_returns_429(self):
        """GeminiAPIError with 429 maps to 429 Too Many Requests."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = GeminiAPIError(
            "Gemini free-tier quota is exhausted (HTTP 429). Wait 12 seconds before retrying.",
            status_code=429,
            retry_after=12,
        )

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=cybersecurity")
            assert response.status_code == 429
            assert "free-tier quota is exhausted" in response.json()["detail"]
            assert response.headers["retry-after"] == "12"
        finally:
            app.dependency_overrides.clear()

    def test_gemini_malformed_response_returns_502(self):
        """GeminiMalformedResponseError maps to 502 Bad Gateway."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = GeminiMalformedResponseError("Unparseable output.")

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=biotech")
            assert response.status_code == 502
            assert "unparseable response" in response.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()

    def test_gemini_service_exception_returns_502(self):
        """GeminiServiceException maps to 502 Bad Gateway."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = GeminiServiceException("Gemini upstream service down.")

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=genomics")
            assert response.status_code == 502
            assert "Language model evaluation service encountered an error" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_orchestrator_error_returns_502(self):
        """OrchestratorError maps to 502 Bad Gateway."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = OrchestratorError("Evaluation pipeline stalled.")

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=semiconductors")
            assert response.status_code == 502
            assert "Orchestration service encountered an error" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_unexpected_exception_returns_500_without_leaks(self):
        """Unhandled internal exceptions return 500 without leaking stack traces or secrets."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.side_effect = RuntimeError(
            "Secret database leak: password=SUPER_SECRET_123 host=10.0.0.1"
        )

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=finance")
            assert response.status_code == 500
            assert "unexpected internal error" in response.json()["detail"].lower()
            # Ensure sensitive information and stack traces are absent
            assert "SUPER_SECRET_123" not in response.text
            assert "Traceback" not in response.text
        finally:
            app.dependency_overrides.clear()


# ===========================================================================
# Secret Leak & Security Verification
# ===========================================================================

class TestSecurityAndConfidentiality:
    """Verifies that internal credentials and stack traces are never exposed."""

    def test_responses_never_contain_credentials_or_tracebacks(self):
        """Ensure responses under multiple conditions never include secret tokens."""
        mock_orchestrator = AsyncMock(spec=NewsOrchestratorService)
        mock_orchestrator.rank_news.return_value = [make_test_ranked_article()]

        app.dependency_overrides[get_orchestrator_service] = lambda: mock_orchestrator

        try:
            response = client.get("/api/v1/news/search?q=technology")
            text = response.text
            assert "gemini_api_key" not in text
            assert "news_api_key" not in text
            assert "Authorization" not in text
            assert "Bearer" not in text
            assert "Traceback" not in text
        finally:
            app.dependency_overrides.clear()


# ===========================================================================
# Dependency Injection and OpenAPI Verification
# ===========================================================================

class TestArchitectureAndDocumentation:
    """Verifies dependency injection and OpenAPI specifications."""

    def test_dependency_injection_provider(self):
        """get_orchestrator_service returns a configured NewsOrchestratorService."""
        service = get_orchestrator_service()
        assert isinstance(service, NewsOrchestratorService)
        assert service.settings is not None

    def test_openapi_schema_generation(self):
        """OpenAPI JSON contains documentation for /health and /api/v1/news/search."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "/health" in schema["paths"]
        assert "/api/v1/news/search" in schema["paths"]

        search_op = schema["paths"]["/api/v1/news/search"]["get"]
        assert search_op["summary"] == "Search and Rank News Articles"
        assert any(p["name"] == "q" for p in search_op["parameters"])
        assert any(p["name"] == "limit" for p in search_op["parameters"])
        assert "200" in search_op["responses"]
        assert "400" in search_op["responses"]
        assert "422" in search_op["responses"]
        assert "502" in search_op["responses"]
        assert "503" in search_op["responses"]
        assert "504" in search_op["responses"]
