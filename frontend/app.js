/**
 * Intelligent News Ranking System - Frontend Application Logic
 *
 * Handles API integration, state management, UI rendering,
 * and error visualization using Vanilla JavaScript.
 */

(() => {
  'use strict';

  // =========================================================================
  // DOM Elements
  // =========================================================================

  const searchForm = document.getElementById('search-form');
  const searchInput = document.getElementById('search-input');
  const limitSelect = document.getElementById('limit-select');
  const btnSearch = document.getElementById('btn-search');
  const searchBtnText = document.getElementById('search-btn-text');
  const searchBtnIcon = document.getElementById('search-btn-icon');
  const examplePillsContainer = document.getElementById('example-pills-container');

  const apiStatusBadge = document.getElementById('api-status-badge');
  const apiStatusText = document.getElementById('api-status-text');

  const initialState = document.getElementById('initial-state');
  const loadingState = document.getElementById('loading-state');
  const loadingStatusText = document.getElementById('loading-status-text');
  const errorState = document.getElementById('error-state');
  const errorTitle = document.getElementById('error-title');
  const errorMessage = document.getElementById('error-message');
  const errorAdvice = document.getElementById('error-advice');
  const btnErrorRetry = document.getElementById('btn-error-retry');
  const emptyResultsState = document.getElementById('empty-results-state');
  const emptyQueryMessage = document.getElementById('empty-query-message');

  const resultsSection = document.getElementById('results-section');
  const resultsCountText = document.getElementById('results-count-text');
  const resultsMetaPill = document.getElementById('results-meta-pill');
  const articlesContainer = document.getElementById('articles-container');

  const configModal = document.getElementById('config-modal');
  const btnOpenConfig = document.getElementById('btn-open-config');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const inputApiUrl = document.getElementById('input-api-url');
  const btnSaveApiUrl = document.getElementById('btn-save-api-url');
  const btnResetApiUrl = document.getElementById('btn-reset-api-url');

  // =========================================================================
  // Application State
  // =========================================================================

  const state = {
    isLoading: false,
    lastQuery: '',
    lastLimit: 10,
    apiConnected: false,
    loadingInterval: null
  };

  // =========================================================================
  // Helper Utilities
  // =========================================================================

  /**
   * Escape HTML special characters to prevent XSS vulnerabilities.
   */
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * Format ISO date string into human-friendly relative or local date.
   */
  function formatDate(isoString) {
    if (!isoString) return 'Date unknown';
    try {
      const date = new Date(isoString);
      if (isNaN(date.getTime())) return 'Date unknown';

      // Relative calculation
      const now = new Date();
      const diffHours = Math.round((now - date) / (1000 * 60 * 60));
      if (diffHours >= 0 && diffHours < 24) {
        if (diffHours === 0) return 'Just now';
        return `${diffHours}h ago`;
      }
      if (diffHours >= 24 && diffHours < 48) {
        return 'Yesterday';
      }

      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    } catch {
      return 'Date unknown';
    }
  }

  /**
   * Get semantic color class based on normalized 0-100 score.
   */
  function getScoreColorClass(score) {
    const val = Number(score) || 0;
    if (val >= 80) return 'score-high';
    if (val >= 60) return 'score-mid';
    if (val >= 40) return 'score-mod';
    return 'score-low';
  }

  /**
   * Format sentiment class name.
   */
  function getSentimentClass(sentiment) {
    const clean = String(sentiment || '').toLowerCase();
    if (clean.includes('pos')) return 'positive';
    if (clean.includes('neg')) return 'negative';
    return 'neutral';
  }

  // =========================================================================
  // API Communication
  // =========================================================================

  /**
   * Ping backend /health endpoint to verify connectivity.
   */
  async function checkBackendHealth() {
    const baseUrl = window.APP_CONFIG.getApiBaseUrl();
    const healthUrl = `${baseUrl}${window.APP_CONFIG.HEALTH_PATH}`;

    try {
      const response = await fetch(healthUrl, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        signal: AbortSignal.timeout(4000)
      });

      if (response.ok) {
        state.apiConnected = true;
        apiStatusBadge.classList.remove('offline');
        apiStatusText.textContent = 'API Connected';
        apiStatusBadge.title = `Backend active at ${baseUrl}`;
      } else {
        throw new Error(`Status ${response.status}`);
      }
    } catch (err) {
      state.apiConnected = false;
      apiStatusBadge.classList.add('offline');
      apiStatusText.textContent = 'API Offline';
      apiStatusBadge.title = `Unable to connect to ${baseUrl} - Click settings to adjust`;
    }
  }

  /**
   * Fetch ranked news articles from GET /api/v1/news/search.
   */
  async function fetchRankedNews(query, limit) {
    const baseUrl = window.APP_CONFIG.getApiBaseUrl();
    const params = new URLSearchParams({
      q: query.trim(),
      limit: String(limit)
    });
    const requestUrl = `${baseUrl}${window.APP_CONFIG.SEARCH_PATH}?${params.toString()}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), window.APP_CONFIG.TIMEOUT_MS);

    try {
      const response = await fetch(requestUrl, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      const contentType = response.headers.get('content-type') || '';
      let data = null;
      if (contentType.includes('application/json')) {
        data = await response.json();
      }

      if (!response.ok) {
        const errorDetail = data && data.detail ? data.detail : `Server error (HTTP ${response.status})`;
        const error = new Error(errorDetail);
        error.status = response.status;
        throw error;
      }

      return data;
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        const timeoutErr = new Error('The request timed out while multi-agent evaluation was executing.');
        timeoutErr.status = 504;
        throw timeoutErr;
      }
      throw err;
    }
  }

  // =========================================================================
  // State Rendering Functions
  // =========================================================================

  function hideAllStates() {
    initialState.style.display = 'none';
    loadingState.style.display = 'none';
    errorState.style.display = 'none';
    emptyResultsState.style.display = 'none';
    resultsSection.style.display = 'none';
  }

  function showLoading(query) {
    hideAllStates();
    loadingState.style.display = 'block';

    const stages = [
      `Retrieving news articles for "${escapeHtml(query)}" via NewsAPI...`,
      'Normalizing content and pruning duplicate canonical URLs...',
      'Relevance Agent analyzing query intent and semantic overlap...',
      'Credibility Agent evaluating attribution and journalistic signals...',
      'Sentiment Agent assessing objectivity and emotional polarity...',
      'Ranking Agent synthesizing weighted composite scores...'
    ];

    let stageIdx = 0;
    loadingStatusText.textContent = stages[0];

    if (state.loadingInterval) clearInterval(state.loadingInterval);
    state.loadingInterval = setInterval(() => {
      stageIdx = (stageIdx + 1) % stages.length;
      loadingStatusText.textContent = stages[stageIdx];
    }, 2500);

    btnSearch.disabled = true;
    searchBtnText.textContent = 'Evaluating...';
    searchBtnIcon.textContent = '⏳';
  }

  function stopLoading() {
    if (state.loadingInterval) {
      clearInterval(state.loadingInterval);
      state.loadingInterval = null;
    }
    loadingState.style.display = 'none';
    btnSearch.disabled = false;
    searchBtnText.textContent = 'Rank News';
    searchBtnIcon.textContent = '⚡';
  }

  function showError(error) {
    stopLoading();
    hideAllStates();
    errorState.style.display = 'flex';

    const status = error.status || 0;
    let title = 'Evaluation Error';
    let message = error.message || 'An unexpected error occurred.';
    let advice = 'Please check the backend server logs for more details.';

    if (status === 400 || status === 422) {
      title = 'Invalid Query Request';
      advice = 'Please enter a non-empty search phrase between 1 and 50 requested articles.';
    } else if (status === 429) {
      title = 'Rate Limit Exceeded';
      advice = 'The free-tier Gemini quota is exhausted. Wait for the provider retry interval, then try again.';
    } else if (status === 503) {
      title = 'Service Configuration Required';
      advice = 'Make sure NEWS_API_KEY and GEMINI_API_KEY are configured in your backend .env file.';
    } else if (status === 504) {
      title = 'Gateway Timeout';
      advice = 'NewsAPI or Google Gemini took too long to reply. Try a more specific search or retry.';
    } else if (status === 502) {
      title = 'Upstream Provider Error';
      advice = 'An upstream service (NewsAPI or Gemini) reported an issue. Please try again.';
    } else if (error.message && error.message.includes('Failed to fetch')) {
      title = 'Cannot Connect to Backend';
      message = `Failed to contact the backend server at ${window.APP_CONFIG.getApiBaseUrl()}`;
      advice = 'Ensure your FastAPI server is running with "python -m uvicorn backend.app.main:app --reload". Click the ⚙️ icon to adjust the backend URL.';
    }

    errorTitle.textContent = title;
    errorMessage.textContent = message;
    errorAdvice.textContent = advice;
  }

  function showEmptyResults(query) {
    stopLoading();
    hideAllStates();
    emptyResultsState.style.display = 'block';
    emptyQueryMessage.innerHTML = `No articles found matching <strong>"${escapeHtml(query)}"</strong> after retrieval and deduplication.`;
  }

  function showResults(data) {
    stopLoading();
    hideAllStates();
    resultsSection.style.display = 'block';

    const query = data.query || state.lastQuery;
    const total = data.total_results || (data.articles ? data.articles.length : 0);
    const articles = data.articles || [];

    resultsCountText.innerHTML = `Ranked <span>${total}</span> articles for <strong>"${escapeHtml(query)}"</strong>`;
    resultsMetaPill.textContent = 'Sorted by Multi-Agent Composite Score (Highest First)';

    articlesContainer.innerHTML = '';

    articles.forEach((item, index) => {
      const card = createArticleCard(item, index);
      articlesContainer.appendChild(card);
    });

    // Smooth scroll down to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // =========================================================================
  // Article Card DOM Generation
  // =========================================================================

  function createArticleCard(rankedItem, index) {
    const art = rankedItem.article || {};
    const analysis = rankedItem.analysis || {};
    const finalScore = Number(rankedItem.final_score !== undefined ? rankedItem.final_score : 0).toFixed(1);
    const rank = rankedItem.rank || (index + 1);

    const relScore = Number(analysis.relevance_score || 0).toFixed(1);
    const credScore = Number(analysis.credibility_score || 0).toFixed(1);
    const qualScore = Number(analysis.information_quality_score || 0).toFixed(1);
    const sentScore = Number(analysis.sentiment_score || 50).toFixed(1);
    const confidence = analysis.confidence !== null && analysis.confidence !== undefined
      ? `${Number(analysis.confidence).toFixed(0)}%`
      : 'N/A';

    const sentiment = analysis.sentiment || 'Neutral';
    const sentimentClass = getSentimentClass(sentiment);
    const scoreColorClass = getScoreColorClass(finalScore);

    const articleUrl = art.article_url || art.url || '#';
    const sourceName = art.source_name || 'News Source';
    const author = art.author ? `By ${art.author}` : '';
    const pubDate = formatDate(art.published_at);

    const rankClass = rank === 1 ? 'rank-1' : (rank === 2 ? 'rank-2' : (rank === 3 ? 'rank-3' : ''));

    const card = document.createElement('article');
    card.className = `article-card ${rankClass}`;
    card.setAttribute('aria-label', `Rank #${rank}: ${art.title}`);

    // Thumbnail markup if available
    const thumbnailMarkup = art.image_url ? `
      <div class="article-thumbnail-wrapper">
        <img
          src="${escapeHtml(art.image_url)}"
          alt="${escapeHtml(art.title)}"
          class="article-thumbnail"
          loading="lazy"
          onerror="this.parentElement.style.display='none'"
        />
      </div>
    ` : '';

    // Limitations markup if present
    const limitationsMarkup = analysis.limitations ? `
      <div class="limitations-notice" role="note">
        <span class="limitations-icon" aria-hidden="true">⚠️</span>
        <div>
          <strong>Credibility Limitation:</strong> ${escapeHtml(analysis.limitations)}
        </div>
      </div>
    ` : '';

    card.innerHTML = `
      <!-- Top Row: Content & Prominent Score -->
      <div class="card-top-row">
        <div class="card-main-content">
          <div class="card-meta-line">
            <span class="rank-badge ${rankClass}">#${rank}</span>
            <span class="source-tag">${escapeHtml(sourceName)}</span>
            ${author ? `<span class="author-tag">&bull; ${escapeHtml(author)}</span>` : ''}
            <span class="date-tag">&bull; ${escapeHtml(pubDate)}</span>
          </div>

          <h3 class="article-headline">
            <a href="${escapeHtml(articleUrl)}" target="_blank" rel="noopener noreferrer">
              ${escapeHtml(art.title || 'Untitled Article')}
            </a>
          </h3>

          <p class="article-description">
            ${escapeHtml(art.description || art.content || 'No summary available for this article.')}
          </p>
        </div>

        ${thumbnailMarkup}

        <!-- Prominent Composite Score Dial -->
        <div class="score-prominent-box" title="Composite multi-agent ranking score (0-100)">
          <div class="score-prominent-number ${scoreColorClass}">${finalScore}</div>
          <div class="score-prominent-label">Composite</div>
        </div>
      </div>

      <!-- Score Breakdown Section -->
      <div class="score-breakdown-section">
        <div class="breakdown-header">
          <span class="breakdown-title">
            <span>Agent Metrics</span>
          </span>
          <div class="sentiment-chip ${sentimentClass}" title="Sentiment score: ${sentScore}/100">
            <span>Tone: ${escapeHtml(sentiment)}</span>
            <span style="font-weight: 500; opacity: 0.85;">(${sentScore})</span>
          </div>
        </div>

        <div class="metrics-grid">
          <!-- Relevance -->
          <div class="metric-item">
            <div class="metric-labels">
              <span class="metric-name">Relevance <span class="metric-weight-tag">(40%)</span></span>
              <span class="metric-val ${getScoreColorClass(relScore)}">${relScore}</span>
            </div>
            <div class="metric-bar-bg">
              <div class="metric-bar-fill fill-relevance" style="width: ${Math.min(100, relScore)}%;"></div>
            </div>
          </div>

          <!-- Credibility -->
          <div class="metric-item">
            <div class="metric-labels">
              <span class="metric-name">Credibility <span class="metric-weight-tag">(25%)</span></span>
              <span class="metric-val ${getScoreColorClass(credScore)}">${credScore}</span>
            </div>
            <div class="metric-bar-bg">
              <div class="metric-bar-fill fill-credibility" style="width: ${Math.min(100, credScore)}%;"></div>
            </div>
          </div>

          <!-- Information Quality -->
          <div class="metric-item">
            <div class="metric-labels">
              <span class="metric-name">Info Quality <span class="metric-weight-tag">(15%)</span></span>
              <span class="metric-val ${getScoreColorClass(qualScore)}">${qualScore}</span>
            </div>
            <div class="metric-bar-bg">
              <div class="metric-bar-fill fill-quality" style="width: ${Math.min(100, qualScore)}%;"></div>
            </div>
          </div>

          <!-- Credibility Confidence -->
          <div class="metric-item">
            <div class="metric-labels">
              <span class="metric-name">Cred. Confidence</span>
              <span class="metric-val">${confidence}</span>
            </div>
            <div class="metric-bar-bg">
              <div class="metric-bar-fill fill-confidence" style="width: ${analysis.confidence !== null ? Math.min(100, analysis.confidence) : 50}%;"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Agent Explanation Box -->
      <div class="agent-explanation-box">
        <div class="explanation-header">
          <span>🧠 Agent Rationale</span>
        </div>
        <div>
          ${escapeHtml(analysis.explanation || 'Article evaluation synthesized across relevance, credibility, and content metrics.')}
        </div>
      </div>

      <!-- Limitations Banner if present -->
      ${limitationsMarkup}

      <!-- Bottom Card Action Footer -->
      <div class="card-footer">
        <div class="credibility-confidence-badge">
          <span>Credibility Confidence: <strong>${confidence}</strong></span>
        </div>
        <a
          href="${escapeHtml(articleUrl)}"
          target="_blank"
          rel="noopener noreferrer"
          class="btn-read-article"
          aria-label="Read full article on ${escapeHtml(sourceName)}"
        >
          <span>Read Full Article on ${escapeHtml(sourceName)}</span>
          <span aria-hidden="true">↗</span>
        </a>
      </div>
    `;

    return card;
  }

  // =========================================================================
  // Search Flow
  // =========================================================================

  async function executeSearch(query, limit) {
    const cleanQuery = query ? query.trim() : '';
    if (!cleanQuery) {
      showError({ status: 400, message: 'Please enter a search topic or keyword.' });
      return;
    }

    state.lastQuery = cleanQuery;
    state.lastLimit = limit || 10;
    state.isLoading = true;

    showLoading(cleanQuery);

    try {
      const data = await fetchRankedNews(cleanQuery, state.lastLimit);

      if (!data.articles || data.articles.length === 0) {
        showEmptyResults(cleanQuery);
      } else {
        showResults(data);
      }
    } catch (err) {
      showError(err);
    } finally {
      state.isLoading = false;
    }
  }

  // =========================================================================
  // Initialization & Event Binding
  // =========================================================================

  function renderExamplePills() {
    examplePillsContainer.innerHTML = '';
    const examples = window.APP_CONFIG.EXAMPLE_QUERIES || [];

    examples.forEach(item => {
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = 'example-pill';
      pill.textContent = item.label;
      pill.title = `Search for: "${item.query}"`;
      pill.addEventListener('click', () => {
        searchInput.value = item.query;
        executeSearch(item.query, parseInt(limitSelect.value, 10) || 10);
      });
      examplePillsContainer.appendChild(pill);
    });
  }

  function bindEvents() {
    // Form submission
    searchForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const query = searchInput.value;
      const limit = parseInt(limitSelect.value, 10) || 10;
      executeSearch(query, limit);
    });

    // Error retry button
    btnErrorRetry.addEventListener('click', () => {
      if (state.lastQuery) {
        executeSearch(state.lastQuery, state.lastLimit);
      } else {
        searchInput.focus();
      }
    });

    // Configuration Modal
    btnOpenConfig.addEventListener('click', () => {
      inputApiUrl.value = window.APP_CONFIG.getApiBaseUrl();
      configModal.classList.add('active');
    });

    btnCloseModal.addEventListener('click', () => {
      configModal.classList.remove('active');
    });

    configModal.addEventListener('click', (e) => {
      if (e.target === configModal) {
        configModal.classList.remove('active');
      }
    });

    btnSaveApiUrl.addEventListener('click', async () => {
      const newUrl = inputApiUrl.value.trim();
      if (newUrl) {
        window.APP_CONFIG.setApiBaseUrl(newUrl);
        configModal.classList.remove('active');
        await checkBackendHealth();
      }
    });

    btnResetApiUrl.addEventListener('click', async () => {
      window.APP_CONFIG.setApiBaseUrl('');
      inputApiUrl.value = window.APP_CONFIG.getApiBaseUrl();
      configModal.classList.remove('active');
      await checkBackendHealth();
    });
  }

  // Bootstrap Application
  document.addEventListener('DOMContentLoaded', () => {
    renderExamplePills();
    bindEvents();
    checkBackendHealth();
  });

})();
