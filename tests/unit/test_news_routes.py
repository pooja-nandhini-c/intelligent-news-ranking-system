"""Unit tests for news API routes using FastAPI TestClient."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.article import Article
from backend.app.services.news_service import (
    NewsAPIKeyMissingError,
    NewsAPITimeoutError,
    NewsAPIError
)
from backend.app.api.news_routes import get_news_service

client = TestClient(app)


def fake_article(title="Test Headline", url="https://example.com/test"):
    return Article(
        id=Article.generate_id(url, title),
        title=title,
        description="A test article summary.",
        content="Clean content.",
        url=url,
        image_url="https://example.com/image.png",
        published_at=None,
        source_name="Test Publisher",
        author="Reporter Smith"
    )


def test_search_endpoint_success():
    """Test /api/news/search endpoint returns 200 and formatted response."""
    mock_service = AsyncMock()
    mock_service.search_articles.return_value = [fake_article()]

    app.dependency_overrides[get_news_service] = lambda: mock_service

    try:
        response = client.get("/api/news/search?q=Artificial+Intelligence")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["query"] == "Artificial Intelligence"
        assert data["total_results"] == 1
        assert len(data["articles"]) == 1
        assert data["articles"][0]["title"] == "Test Headline"
        # Ensure API key is not present in response
        assert "api_key" not in response.text
        assert "apiKey" not in response.text
    finally:
        app.dependency_overrides.clear()


def test_top_headlines_endpoint_success():
    """Test /api/news/top endpoint returns 200 and formatted response."""
    mock_service = AsyncMock()
    mock_service.get_top_headlines.return_value = [
        fake_article("Tech News 1", "https://example.com/tech1"),
        fake_article("Tech News 2", "https://example.com/tech2")
    ]

    app.dependency_overrides[get_news_service] = lambda: mock_service

    try:
        response = client.get("/api/news/top?category=technology")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["category"] == "technology"
        assert data["total_results"] == 2
        assert len(data["articles"]) == 2
    finally:
        app.dependency_overrides.clear()


def test_search_missing_query_validation():
    """Test search endpoint returns 422 if 'q' parameter is omitted."""
    response = client.get("/api/news/search")
    assert response.status_code == 422


def test_missing_api_key_returns_503():
    """Test missing API key error maps to 503 Service Unavailable."""
    mock_service = AsyncMock()
    mock_service.search_articles.side_effect = NewsAPIKeyMissingError("NEWS_API_KEY is not configured.")

    app.dependency_overrides[get_news_service] = lambda: mock_service

    try:
        response = client.get("/api/news/search?q=economy")
        assert response.status_code == 503
        data = response.json()
        assert "NEWS_API_KEY is not configured" in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_timeout_returns_504():
    """Test timeout error maps to 504 Gateway Timeout."""
    mock_service = AsyncMock()
    mock_service.search_articles.side_effect = NewsAPITimeoutError("NewsAPI request timed out after 10s.")

    app.dependency_overrides[get_news_service] = lambda: mock_service

    try:
        response = client.get("/api/news/search?q=crypto")
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_upstream_429_rate_limit():
    """Test upstream 429 maps to 429."""
    mock_service = AsyncMock()
    mock_service.get_top_headlines.side_effect = NewsAPIError("Too many requests", status_code=429)

    app.dependency_overrides[get_news_service] = lambda: mock_service

    try:
        response = client.get("/api/news/top?category=business")
        assert response.status_code == 429
        assert "Too many requests" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
