"""Configuration settings using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration settings loaded from environment or .env file."""
    
    # AI / LLM Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    
    # News Ingestion Configuration
    news_api_key: str = ""
    
    # Ranking Weights Configuration (Default formula: 40% Relevance, 25% Credibility, 20% Freshness, 15% Quality)
    ranking_weight_relevance: float = 0.40
    ranking_weight_credibility: float = 0.25
    ranking_weight_freshness: float = 0.20
    ranking_weight_information_quality: float = 0.15

    # Orchestration Configuration
    max_articles_to_rank: int = 1
    agent_concurrency_limit: int = 5
    gemini_requests_per_minute: int = 5
    gemini_max_concurrency: int = 1
    gemini_max_retries: int = 1
    gemini_request_timeout: float = 30.0

    # Application & Server Settings
    app_name: str = "Intelligent News Ranking System Using LLM Agents"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    def get_ranking_weights(self) -> dict:
        """Return ranking weights dictionary."""
        return {
            "relevance": self.ranking_weight_relevance,
            "credibility": self.ranking_weight_credibility,
            "freshness": self.ranking_weight_freshness,
            "information_quality": self.ranking_weight_information_quality,
        }
    
    model_config = SettingsConfigDict(
    env_file=Path(__file__).resolve().parents[2] / ".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)

@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
