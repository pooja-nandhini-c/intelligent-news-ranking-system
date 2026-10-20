"""API routes package."""

from backend.app.api.news_routes import router as news_router
from backend.app.api.ranked_news_routes import (
    router as ranked_news_router,
    get_orchestrator_service
)

__all__ = ["news_router", "ranked_news_router", "get_orchestrator_service"]
