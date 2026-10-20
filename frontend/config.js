/**
 * Intelligent News Ranking System - Frontend Configuration
 *
 * Configurable API base URL and defaults.
 * Allows switching between local FastAPI server, Docker, or custom remote endpoints.
 */

const CONFIG = {
  // Dynamically resolve API Base URL:
  // 1. Check for manual override in localStorage
  // 2. If served directly by FastAPI (port 8000), use same origin
  // 3. Fallback to default local backend (http://127.0.0.1:8000)
  getApiBaseUrl() {
    const savedUrl = localStorage.getItem('news_rank_api_base_url');
    if (savedUrl && savedUrl.trim()) {
      return savedUrl.trim().replace(/\/+$/, '');
    }
    if (window.location.protocol.startsWith('http') && window.location.port === '8000') {
      return window.location.origin;
    }
    return 'http://127.0.0.1:8000';
  },

  setApiBaseUrl(newUrl) {
    if (newUrl && newUrl.trim()) {
      localStorage.setItem('news_rank_api_base_url', newUrl.trim().replace(/\/+$/, ''));
    } else {
      localStorage.removeItem('news_rank_api_base_url');
    }
  },

  // API Endpoints
  SEARCH_PATH: '/api/v1/news/search',
  HEALTH_PATH: '/health',

  // Limits
  DEFAULT_LIMIT: 1,
  MIN_LIMIT: 1,
  MAX_LIMIT: 50,

  // Keep the request bounded while allowing three RPM-limited agent calls.
  TIMEOUT_MS: 60000,

  // Example topic queries for fast demonstration
  EXAMPLE_QUERIES: [
    { label: '🤖 AI & Robotics', query: 'Artificial Intelligence robotics' },
    { label: '🌱 Clean Energy', query: 'Renewable clean energy transition' },
    { label: '🚀 Space Missions', query: 'Space exploration NASA Artemis' },
    { label: '💻 Quantum Tech', query: 'Quantum computing breakthroughs' },
    { label: '📈 Global Markets', query: 'Global economy central bank inflation' },
    { label: '🛡️ Cybersecurity', query: 'Cybersecurity zero-day threats cloud' }
  ]
};

// Export to window
window.APP_CONFIG = CONFIG;
