"""Unit tests for NewsAPIService with mocked HTTP responses."""

import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime, timezone
import httpx

from backend.app.config import Settings
from backend.app.services.news_service import (
    NewsAPIService,
    NewsAPIKeyMissingError,
    NewsAPITimeoutError,
    NewsAPIError
)
from backend.app.models.article import Article


@pytest.fixture
def settings_with_key():
    return Settings(news_api_key="valid-test-key-12345")


@pytest.fixture
def settings_missing_key():
    return Settings(news_api_key="")


@pytest.fixture
def news_service(settings_with_key):
    return NewsAPIService(settings=settings_with_key)


@pytest.fixture
def sample_newsapi_raw_response():
    return {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {"id": "wired", "name": "Wired"},
                "author": "Jane Doe",
                "title": "Breakthrough in Renewable Energy Storage",
                "description": "Researchers achieve novel solid-state battery milestone.",
                "url": "https://www.wired.com/story/renewable-energy-breakthrough/",
                "urlToImage": "https://www.wired.com/images/battery.jpg",
                "publishedAt": "2026-09-29T10:15:30Z",
                "content": "A team of materials scientists have developed... [+1420 chars]"
            },
            {
                "source": {"id": None, "name": "TechCrunch"},
                "author": None,
                "title": "Quantum Computing Startup Raises Series A",
                "description": "Funding round led by prominent deep tech investors.",
                "url": "https://techcrunch.com/2026/09/29/quantum-startup-series-a/",
                "urlToImage": None,
                "publishedAt": "2026-09-29T08:00:00Z",
                "content": "The startup plans to expand its cryogenic laboratory."
            }
        ]
    }


@pytest.mark.asyncio
async def test_missing_api_key_raises_error(settings_missing_key):
    """Test that missing NEWS_API_KEY immediately raises NewsAPIKeyMissingError."""
    service = NewsAPIService(settings=settings_missing_key)
    with pytest.raises(NewsAPIKeyMissingError) as exc_info:
        await service.search_articles("AI")
    assert "NEWS_API_KEY is not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_articles_success(news_service, sample_newsapi_raw_response):
    """Test searching articles normalizes and parses fields correctly."""
    mock_resp = httpx.Response(
        status_code=200,
        json=sample_newsapi_raw_response,
        request=httpx.Request("GET", "https://newsapi.org/v2/everything")
    )
    
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        
        articles = await news_service.search_articles(query="Renewable Energy", page_size=10)
        
        assert len(articles) == 2
        assert isinstance(articles[0], Article)
        assert articles[0].title == "Breakthrough in Renewable Energy Storage"
        assert articles[0].source_name == "Wired"
        assert articles[0].author == "Jane Doe"
        assert articles[0].image_url == "https://www.wired.com/images/battery.jpg"
        assert articles[0].published_at == datetime(2026, 9, 29, 10, 15, 30, tzinfo=timezone.utc)
        # Content should have [+1420 chars] marker stripped
        assert "[+1420 chars]" not in (articles[0].content or "")
        assert articles[0].id is not None
        
        assert articles[1].author is None
        assert articles[1].source_name == "TechCrunch"


@pytest.mark.asyncio
async def test_get_top_headlines_success(news_service, sample_newsapi_raw_response):
    """Test top headlines endpoint returns normalized articles."""
    mock_resp = httpx.Response(
        status_code=200,
        json=sample_newsapi_raw_response,
        request=httpx.Request("GET", "https://newsapi.org/v2/top-headlines")
    )
    
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        
        articles = await news_service.get_top_headlines(category="technology", country="us")
        assert len(articles) == 2
        assert articles[0].title == "Breakthrough in Renewable Energy Storage"


@pytest.mark.asyncio
async def test_empty_query_returns_empty_list(news_service):
    """Test that empty or whitespace query returns empty list without calling API."""
    articles = await news_service.search_articles("")
    assert articles == []
    
    articles = await news_service.search_articles("   ")
    assert articles == []


@pytest.mark.asyncio
async def test_empty_results_from_api(news_service):
    """Test API returning 0 articles returns empty list."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"status": "ok", "totalResults": 0, "articles": []},
        request=httpx.Request("GET", "https://newsapi.org/v2/everything")
    )
    
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        articles = await news_service.search_articles("nonexistenttopic12345")
        assert articles == []


@pytest.mark.asyncio
async def test_removed_and_malformed_articles_filtered(news_service):
    """Test that [Removed] articles, articles without titles or URLs are ignored."""
    raw_response = {
        "status": "ok",
        "totalResults": 4,
        "articles": [
            {
                "title": "[Removed]",
                "url": "https://removed.com",
                "description": "[Removed]"
            },
            {
                "title": "",
                "url": "https://example.com/notitle"
            },
            {
                "title": "Valid Article",
                "url": "https://example.com/valid",
                "source": {"name": "BBC"}
            },
            "invalid_non_dict_item"
        ]
    }
    
    mock_resp = httpx.Response(
        status_code=200,
        json=raw_response,
        request=httpx.Request("GET", "https://newsapi.org/v2/everything")
    )
    
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        articles = await news_service.search_articles("test")
        assert len(articles) == 1
        assert articles[0].title == "Valid Article"


@pytest.mark.asyncio
async def test_timeout_handling(news_service):
    """Test that httpx.TimeoutException triggers NewsAPITimeoutError."""
    with patch.object(httpx.AsyncClient, "get", side_effect=httpx.ReadTimeout("Connection timed out")):
        with pytest.raises(NewsAPITimeoutError) as exc_info:
            await news_service.search_articles("technology")
        assert "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_api_401_error(news_service):
    """Test that 401 Unauthorized raises NewsAPIError with 401."""
    mock_resp = httpx.Response(
        status_code=401,
        json={
            "status": "error",
            "code": "apiKeyInvalid",
            "message": "Your API key is invalid or incorrect."
        },
        request=httpx.Request("GET", "https://newsapi.org/v2/everything")
    )
    
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        with pytest.raises(NewsAPIError) as exc_info:
            await news_service.search_articles("test")
        assert exc_info.value.status_code == 401
        assert "invalid or incorrect" in exc_info.value.message


@pytest.mark.asyncio
async def test_api_429_rate_limit_error(news_service):
    """Test that 429 Rate Limit raises NewsAPIError with 429."""
    mock_resp = httpx.Response(
        status_code=429,
        json={
            "status": "error",
            "code": "rateLimited",
            "message": "You have made too many requests recently."
        },
        request=httpx.Request("GET", "https://newsapi.org/v2/everything")
    )
    
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        with pytest.raises(NewsAPIError) as exc_info:
            await news_service.search_articles("test")
        assert exc_info.value.status_code == 429
        assert "too many requests" in exc_info.value.message


@pytest.mark.asyncio
async def test_non_json_response_handling(news_service):
    """Test that HTML / non-JSON response raises NewsAPIError."""
    mock_resp = httpx.Response(
        status_code=502,
        content=b"<html>Bad Gateway</html>",
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://newsapi.org/v2/everything")
    )
    
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        with pytest.raises(NewsAPIError) as exc_info:
            await news_service.search_articles("test")
        assert exc_info.value.status_code == 502
        assert "malformed or non-json" in exc_info.value.message.lower()
