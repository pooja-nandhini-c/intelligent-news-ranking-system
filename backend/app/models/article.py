"""Article and News response data models."""

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional, List, Any
from datetime import datetime
from urllib.parse import urlparse
import hashlib


class Article(BaseModel):
    """Normalized article representation decoupled from upstream NewsAPI schema."""

    id: str = Field(..., min_length=1, description="Unique deterministic identifier for the article")
    title: str = Field(..., min_length=1, description="Headline of the article")
    description: Optional[str] = Field(default=None, description="Summary or snippet of the article")
    content: Optional[str] = Field(default=None, description="Truncated article content if available")
    source_name: str = Field(default="Unknown", min_length=1, description="Name of the news publisher or domain")
    source_url: Optional[str] = Field(default=None, description="Homepage or base domain URL of the source")
    article_url: str = Field(..., min_length=1, description="Canonical URL to original article")
    image_url: Optional[str] = Field(default=None, description="Lead image URL")
    author: Optional[str] = Field(default=None, description="Byline or author name")
    published_at: Optional[datetime] = Field(default=None, description="Publication timestamp")

    @model_validator(mode="before")
    @classmethod
    def handle_url_compatibility_and_defaults(cls, data: Any) -> Any:
        """Allow backwards compatibility with 'url' and auto-derive source_url if omitted."""
        if isinstance(data, dict):
            # Support 'url' as alias/fallback for 'article_url'
            if "article_url" not in data and "url" in data:
                data["article_url"] = data["url"]

            # Auto-derive source_url from article_url if not explicitly provided
            if not data.get("source_url") and data.get("article_url"):
                try:
                    parsed = urlparse(str(data["article_url"]))
                    if parsed.scheme and parsed.netloc:
                        data["source_url"] = f"{parsed.scheme}://{parsed.netloc}"
                except Exception:
                    pass
        return data

    @field_validator("title", "source_name", "article_url", "id", mode="before")
    @classmethod
    def strip_required_strings(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
        return v

    @property
    def url(self) -> str:
        """Convenience property for backwards compatibility with code expecting .url."""
        return self.article_url

    @classmethod
    def generate_id(cls, url: Optional[str] = None, title: str = "", article_url: Optional[str] = None) -> str:
        """Generate a deterministic hash ID from article URL and title."""
        target_url = (article_url or url or "").strip().lower()
        clean_title = (title or "").strip().lower()
        raw = f"{target_url}:{clean_title}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ArticleListResponse(BaseModel):
    """Structured response containing normalized articles and query metadata."""

    status: str = Field(default="ok", description="Status string: 'ok' or 'error'")
    total_results: int = Field(default=0, ge=0, description="Total count of retrieved articles")
    query: Optional[str] = Field(default=None, description="Search query if applicable")
    category: Optional[str] = Field(default=None, description="News category if applicable")
    articles: List[Article] = Field(default_factory=list, description="List of normalized articles")
