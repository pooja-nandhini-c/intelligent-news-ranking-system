"""FastAPI entrypoint for Intelligent News Ranking System."""

from datetime import datetime, timezone
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.config import Settings, get_settings
from backend.app.api.news_routes import router as news_router
from backend.app.api.ranked_news_routes import router as ranked_news_router


class HealthStatusResponse(BaseModel):
    """Simple health check status response model."""
    status: str = Field(default="ok", description="Operational health status: 'ok'")
    service: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp")


tags_metadata = [
    {
        "name": "Health",
        "description": "Liveness and operational health check endpoints.",
    },
    {
        "name": "Ranked News",
        "description": "Multi-agent news evaluation and composite ranking endpoints.",
    },
    {
        "name": "news",
        "description": "Direct NewsAPI search and headline retrieval endpoints.",
    },
    {
        "name": "General",
        "description": "Root endpoint and service overview.",
    },
]

app = FastAPI(
    title="Intelligent News Ranking System API",
    description=(
        "Intelligent News Ranking System using LLM Agents. "
        "Coordinates retrieval via NewsAPI, deduplication, concurrent multi-agent evaluation "
        "(Relevance, Sentiment, Credibility), and composite score synthesis via LangChain and Google Gemini."
    ),
    version="0.1.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local development and frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(ranked_news_router)
app.include_router(news_router)


@app.get(
    "/health",
    response_model=HealthStatusResponse,
    tags=["Health"],
    summary="Application Health Check",
    description="Returns a simple operational health status without requiring API keys or authentication."
)
async def health(settings: Settings = Depends(get_settings)) -> HealthStatusResponse:
    """Return simple health status without requiring API keys."""
    return HealthStatusResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get(
    "/api/health",
    tags=["Health"],
    summary="Detailed Health Check (Legacy)",
    description="Health check endpoint indicating whether the backend service is operational and configuration flags."
)
async def health_check_legacy(settings: Settings = Depends(get_settings)):
    """Health check endpoint indicating whether the backend service is operational."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gemini_model": settings.gemini_model,
        "gemini_configured": bool(settings.gemini_api_key),
        "newsapi_configured": bool(settings.news_api_key)
    }


import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve frontend UI static files if directory exists
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get(
        "/app",
        tags=["General"],
        summary="Frontend Web Dashboard",
        description="Serves the interactive HTML5/Vanilla JS frontend dashboard."
    )
    async def serve_dashboard():
        """Serve the frontend dashboard index.html."""
        index_file = os.path.join(frontend_path, "index.html")
        return FileResponse(index_file)


@app.get(
    "/",
    tags=["General"],
    summary="Root API Information",
    description="Root endpoint providing service overview and documentation links."
)
async def root(settings: Settings = Depends(get_settings)):
    """Root endpoint providing service overview."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs_url": "/docs",
        "health_check": "/health",
        "api_v1_search": "/api/v1/news/search",
        "frontend_app": "/app"
    }
