"""NewsAPI routing endpoints."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status

from backend.app.config import Settings, get_settings
from backend.app.models.article import ArticleListResponse
from backend.app.services.news_service import (
    NewsAPIService,
    NewsAPIKeyMissingError,
    NewsAPITimeoutError,
    NewsAPIError
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news", tags=["news"])


def get_news_service(settings: Settings = Depends(get_settings)) -> NewsAPIService:
    """Dependency provider for NewsAPIService."""
    return NewsAPIService(settings=settings)


@router.get("/search", response_model=ArticleListResponse, summary="Search News Articles")
async def search_news(
    q: str = Query(..., min_length=1, description="Search query keywords"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Articles per page"),
    sort_by: str = Query(default="relevancy", description="Sorting criteria: relevancy, popularity, publishedAt"),
    language: str = Query(default="en", description="2-letter ISO-639-1 language code"),
    service: NewsAPIService = Depends(get_news_service)
):
    """
    Search for articles from NewsAPI using the /v2/everything endpoint.
    Articles are normalized and cleaned.
    """
    try:
        articles = await service.search_articles(
            query=q,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            language=language
        )
        return ArticleListResponse(
            status="ok",
            total_results=len(articles),
            query=q,
            articles=articles
        )
    except NewsAPIKeyMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)
        ) from exc
    except NewsAPITimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc)
        ) from exc
    except NewsAPIError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /api/news/search: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while searching news."
        ) from exc


@router.get("/top", response_model=ArticleListResponse, summary="Get Top Headlines")
async def get_top_headlines(
    category: Optional[str] = Query(default=None, description="News category: business, entertainment, general, health, science, sports, technology"),
    country: Optional[str] = Query(default="us", description="2-letter ISO 3166-1 country code"),
    q: Optional[str] = Query(default=None, description="Optional search query filter"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Articles per page"),
    service: NewsAPIService = Depends(get_news_service)
):
    """
    Retrieve top headlines from NewsAPI using the /v2/top-headlines endpoint.
    Articles are normalized and cleaned.
    """
    try:
        articles = await service.get_top_headlines(
            category=category,
            country=country,
            query=q,
            page=page,
            page_size=page_size
        )
        return ArticleListResponse(
            status="ok",
            total_results=len(articles),
            category=category,
            query=q,
            articles=articles
        )
    except NewsAPIKeyMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)
        ) from exc
    except NewsAPITimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc)
        ) from exc
    except NewsAPIError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /api/news/top: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while retrieving top headlines."
        ) from exc
