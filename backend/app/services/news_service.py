"""NewsAPI client service for fetching and normalizing news articles."""

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import httpx

from backend.app.config import Settings, get_settings
from backend.app.models.article import Article

logger = logging.getLogger(__name__)


class NewsAPIException(Exception):
    """Base exception for NewsAPI operations."""
    pass


class NewsAPIKeyMissingError(NewsAPIException):
    """Raised when NEWS_API_KEY is not configured."""
    pass


class NewsAPITimeoutError(NewsAPIException):
    """Raised when request to NewsAPI times out."""
    pass


class NewsAPIError(NewsAPIException):
    """Raised when NewsAPI returns an error response."""
    def __init__(self, message: str, status_code: int = 502, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class NewsAPIService:
    """Service to interact with NewsAPI.org and normalize article responses."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, settings: Optional[Settings] = None, timeout: float = 10.0):
        self.settings = settings or get_settings()
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        return (self.settings.news_api_key or "").strip()

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            logger.error("NewsAPI operation aborted: NEWS_API_KEY is missing")
            raise NewsAPIKeyMissingError("NEWS_API_KEY is not configured in the application environment.")
        return {
            "X-Api-Key": self.api_key,
            "User-Agent": "IntelligentNewsRanker/1.0"
        }

    async def _send_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP GET request to NewsAPI with error handling."""
        headers = self._get_headers()
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        # Filter out None values from params
        clean_params = {k: v for k, v in params.items() if v is not None}
        
        logger.info("Calling NewsAPI endpoint: %s with params: %s", endpoint, {k: v for k, v in clean_params.items() if k != "apiKey"})

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=clean_params, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error("NewsAPI request timed out on endpoint %s: %s", endpoint, str(exc))
            raise NewsAPITimeoutError(f"NewsAPI request timed out after {self.timeout}s.") from exc
        except httpx.RequestError as exc:
            logger.error("Network communication error with NewsAPI: %s", str(exc))
            raise NewsAPIError(f"Network error while connecting to NewsAPI: {str(exc)}", status_code=502) from exc

        # Parse JSON
        try:
            data = response.json()
        except Exception as exc:
            logger.error("Malformed non-JSON response from NewsAPI (HTTP %s): %s", response.status_code, response.text[:200])
            raise NewsAPIError("Received malformed or non-JSON response from NewsAPI.", status_code=502) from exc

        if not isinstance(data, dict):
            logger.error("Unexpected response format from NewsAPI: expected dict, got %s", type(data).__name__)
            raise NewsAPIError("Received invalid response structure from NewsAPI.", status_code=502)

        # Check API status field
        status = data.get("status")
        if status == "error" or response.is_error:
            error_code = data.get("code", "unknown_error")
            error_message = data.get("message", f"NewsAPI returned error HTTP {response.status_code}")
            logger.error("NewsAPI returned error: code=%s, message=%s (HTTP %s)", error_code, error_message, response.status_code)
            
            # Map NewsAPI status codes
            http_status = 502
            if response.status_code in (401, 403):
                http_status = 401
            elif response.status_code == 429:
                http_status = 429
            elif response.status_code == 400:
                http_status = 400
                
            raise NewsAPIError(error_message, status_code=http_status, error_code=error_code)

        return data

    def _clean_content(self, content: Optional[str]) -> Optional[str]:
        """Strip truncated markers like '[+1234 chars]' from article content."""
        if not content:
            return None
        cleaned = re.sub(r"\[\+\d+\s*chars?\]", "", content).strip()
        return cleaned or None

    def _normalize_article(self, raw: Dict[str, Any]) -> Optional[Article]:
        """Normalize a raw NewsAPI article dictionary into an Article model."""
        if not isinstance(raw, dict):
            return None

        title = (raw.get("title") or "").strip()
        url = (raw.get("url") or "").strip()

        # Discard empty, placeholder or removed articles
        if not title or not url or title == "[Removed]" or "removed.com" in url:
            return None

        # Source extraction
        source_data = raw.get("source") or {}
        source_name = "Unknown"
        if isinstance(source_data, dict):
            source_name = (source_data.get("name") or "Unknown").strip()

        # Parse publication date
        published_at = None
        raw_date = raw.get("publishedAt")
        if raw_date and isinstance(raw_date, str):
            try:
                published_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Could not parse article publication date '%s'", raw_date)

        # Clean description and content
        description = (raw.get("description") or "").strip() or None
        content = self._clean_content(raw.get("content"))
        author = (raw.get("author") or "").strip() or None
        image_url = (raw.get("urlToImage") or "").strip() or None

        article_id = Article.generate_id(article_url=url, title=title)

        return Article(
            id=article_id,
            title=title,
            description=description,
            content=content,
            source_name=source_name,
            source_url=None,
            article_url=url,
            image_url=image_url,
            published_at=published_at,
            author=author
        )

    def normalize_articles(self, raw_articles: Any) -> List[Article]:
        """Normalize a list of raw articles and skip invalid entries."""
        if not isinstance(raw_articles, list):
            logger.warning("Expected list for 'articles' key, got %s", type(raw_articles).__name__)
            return []

        articles: List[Article] = []
        seen_ids = set()

        for raw in raw_articles:
            try:
                article = self._normalize_article(raw)
                if article and article.id not in seen_ids:
                    seen_ids.add(article.id)
                    articles.append(article)
            except Exception as exc:
                logger.warning("Failed to normalize article: %s", exc)

        return articles

    async def search_articles(
        self,
        query: str,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "relevancy",
        language: str = "en"
    ) -> List[Article]:
        """
        Search articles using /v2/everything endpoint.
        """
        if not query or not query.strip():
            return []

        params = {
            "q": query.strip(),
            "page": max(1, page),
            "pageSize": min(100, max(1, page_size)),
            "sortBy": sort_by,
            "language": language
        }

        data = await self._send_request("everything", params)
        raw_articles = data.get("articles", [])
        return self.normalize_articles(raw_articles)

    async def get_top_headlines(
        self,
        category: Optional[str] = None,
        country: Optional[str] = "us",
        query: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Article]:
        """
        Retrieve top headlines using /v2/top-headlines endpoint.
        """
        params = {
            "category": category.strip().lower() if category and category.strip() else None,
            "country": country,
            "q": query.strip() if query and query.strip() else None,
            "page": max(1, page),
            "pageSize": min(100, max(1, page_size))
        }

        # If both category and country are omitted and query is omitted, default country to "us"
        if not params["category"] and not params["country"] and not params["q"]:
            params["country"] = "us"

        data = await self._send_request("top-headlines", params)
        raw_articles = data.get("articles", [])
        return self.normalize_articles(raw_articles)
