"""Integration tests verifying frontend static asset delivery and security."""

import os
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)


class TestFrontendDeliveryAndSecurity:
    """Verifies frontend HTML, CSS, JS delivery and confidentiality."""

    def test_frontend_files_exist(self):
        """Ensure all required frontend files exist on disk."""
        assert os.path.isfile(os.path.join(FRONTEND_DIR, "index.html"))
        assert os.path.isfile(os.path.join(FRONTEND_DIR, "styles.css"))
        assert os.path.isfile(os.path.join(FRONTEND_DIR, "app.js"))
        assert os.path.isfile(os.path.join(FRONTEND_DIR, "config.js"))

    def test_app_dashboard_route_serves_html(self):
        """GET /app serves index.html with 200 OK and text/html content type."""
        response = client.get("/app")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Intelligent News Ranking System" in response.text
        assert "search-form" in response.text
        assert "articles-container" in response.text

    def test_static_index_html_route(self):
        """GET /static/index.html serves index.html."""
        response = client.get("/static/index.html")
        assert response.status_code == 200
        assert "Intelligent News Ranking System" in response.text

    def test_static_styles_css_route(self):
        """GET /static/styles.css serves valid stylesheet."""
        response = client.get("/static/styles.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        assert "--bg-dark" in response.text
        assert ".article-card" in response.text

    def test_static_config_js_route(self):
        """GET /static/config.js serves client config."""
        response = client.get("/static/config.js")
        assert response.status_code == 200
        assert "APP_CONFIG" in response.text
        assert "/api/v1/news/search" in response.text
        assert "DEFAULT_LIMIT: 1" in response.text
        assert "TIMEOUT_MS: 60000" in response.text

    def test_static_app_js_route(self):
        """GET /static/app.js serves client application script."""
        response = client.get("/static/app.js")
        assert response.status_code == 200
        assert "fetchRankedNews" in response.text
        assert "createArticleCard" in response.text
        assert "errorMessage.textContent = message" in response.text

    def test_frontend_does_not_contain_secrets_or_env_values(self):
        """Verify that frontend assets never contain hardcoded API keys or env secrets."""
        for filename in ["index.html", "styles.css", "app.js", "config.js"]:
            filepath = os.path.join(FRONTEND_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Ensure neither real API key format nor raw secrets are present
            assert "AIzaSy" not in content  # Common Google Gemini API key prefix
            assert "NEWS_API_KEY=" not in content
            assert "GEMINI_API_KEY=" not in content
