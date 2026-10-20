"""FastAPI routing endpoints for ranked news search."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status

from backend.app.config import Settings, get_settings
from backend.app.models.ranking import RankedNewsSearchResponse
from backend.app.services.orchestrator_service import (
    NewsOrchestratorService,
    OrchestratorQueryError,
    OrchestratorError
)
from backend.app.services.news_service import (
    NewsAPIKeyMissingError,
    NewsAPITimeoutError,
    NewsAPIError
)
from backend.app.services.gemini_service import (
    GeminiAPIKeyMissingError,
    GeminiTimeoutError,
    GeminiAPIError,
    GeminiMalformedResponseError,
    GeminiServiceException
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/news", tags=["Ranked News"])


def get_orchestrator_service(
    settings: Settings = Depends(get_settings)
) -> NewsOrchestratorService:
    """Dependency provider for NewsOrchestratorService."""
    resolved_settings = settings if isinstance(settings, Settings) else get_settings()
    return NewsOrchestratorService(settings=resolved_settings)


@router.get(
    "/search",
    response_model=RankedNewsSearchResponse,
    summary="Search and Rank News Articles",
    description=(
        "Retrieves news articles for a user query via NewsAPI, deduplicates results, "
        "evaluates them concurrently across Relevance, Sentiment, and Credibility agents, "
        "and synthesizes a multi-dimensional composite ranking score."
    ),
    responses={
        200: {
            "description": "Ranked news articles in descending order of composite score.",
            "model": RankedNewsSearchResponse,
        },
        400: {
            "description": "Bad Request - Invalid, missing, or whitespace-only search query.",
        },
        422: {
            "description": "Validation Error - Missing required parameter or invalid limit boundary.",
        },
        429: {
            "description": "Too Many Requests - Upstream NewsAPI or Gemini rate limit exceeded.",
        },
        502: {
            "description": "Bad Gateway - Upstream NewsAPI or Gemini provider failure.",
        },
        503: {
            "description": "Service Unavailable - NewsAPI or Gemini API key missing/unconfigured.",
        },
        504: {
            "description": "Gateway Timeout - Upstream NewsAPI or Gemini request timed out.",
        },
    }
)
async def search_and_rank_news(
    q: str = Query(
        ...,
        min_length=1,
        description="Search query keywords (must be non-empty and non-whitespace)"
    ),
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        le=50,
        description="Maximum number of articles to rank (1 to 50; defaults to configured max_articles_to_rank)"
    ),
    orchestrator: NewsOrchestratorService = Depends(get_orchestrator_service),
    settings: Settings = Depends(get_settings)
) -> RankedNewsSearchResponse:
    """
    Search, evaluate, and rank news articles for the given query.

    Returns ranked articles ordered by final composite score descending.
    """
    clean_query = q.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' cannot be empty or whitespace only."
        )

    effective_limit = limit if limit is not None else settings.max_articles_to_rank

    try:
        ranked_articles = await orchestrator.rank_news(
            query=clean_query,
            max_articles=effective_limit
        )

        return RankedNewsSearchResponse(
            status="ok",
            query=clean_query,
            total_results=len(ranked_articles),
            articles=ranked_articles
        )

    except OrchestratorQueryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc

    except NewsAPIKeyMissingError as exc:
        logger.warning("NewsAPI key missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News retrieval service is unavailable due to missing API key configuration."
        ) from exc

    except GeminiAPIKeyMissingError as exc:
        logger.warning("Gemini API key missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model evaluation service is unavailable due to missing API key configuration."
        ) from exc

    except NewsAPITimeoutError as exc:
        logger.error("NewsAPI timeout: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="News retrieval request timed out while communicating with upstream provider."
        ) from exc

    except GeminiTimeoutError as exc:
        logger.error("Gemini timeout: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Language model evaluation timed out."
        ) from exc

    except NewsAPIError as exc:
        logger.error("NewsAPI error [%s]: %s", exc.status_code, exc.message)
        if exc.status_code == 429:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for upstream news retrieval provider."
            ) from exc
        if exc.status_code in (401, 403):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Upstream news provider rejected authentication."
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream news retrieval provider returned an error."
        ) from exc

    except GeminiAPIError as exc:
        logger.error("Gemini API error [%s]: %s", exc.status_code, exc.message)
        if exc.status_code == 429:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=exc.message,
                headers={"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream language model provider returned an error."
        ) from exc

    except GeminiMalformedResponseError as exc:
        logger.error("Gemini malformed response: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model returned an unparseable response."
        ) from exc

    except GeminiServiceException as exc:
        logger.error("Gemini service exception: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model evaluation service encountered an error."
        ) from exc

    except OrchestratorError as exc:
        logger.error("Orchestrator error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc) if "no ranked results" in str(exc).lower() else "Orchestration service encountered an error."
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error in /api/v1/news/search: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing the news search."
        ) from exc
