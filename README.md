# Intelligent News Ranking System Using LLM Agents

A portfolio-grade web application that retrieves real-time news articles and orchestrates specialized LLM agents (Google Gemini via LangChain) to analyze, score, and rank news according to multi-dimensional criteria.

---

## 🌟 Overview

Traditional news aggregators rely on keyword matching or opaque recommendation algorithms. This system employs an agent-based architecture to provide explainable, multi-perspective news evaluation:

1. **Relevance Agent**: Measures semantic alignment against user query intent and domain topics.
2. **Sentiment Agent**: Assesses emotional polarity, tone, and editorial subjectivity.
3. **Credibility Assessment Agent**: Evaluates observable structural and stylistic quality signals (author bylines, transparent citations, non-sensational phrasing). *Explicitly does not claim to guarantee factual truth.*
4. **Ranking & Synthesis Agent**: Computes transparent composite scores and generates comparative rationale.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.13+, FastAPI, Uvicorn, Pydantic V2
- **AI & Agents**: Google Gemini API, LangChain
- **News Source**: NewsAPI
- **Frontend**: Modern Vanilla HTML5, CSS3, JavaScript (ES6+)
- **Testing**: pytest (unit & integration), Selenium WebDriver (E2E)

---

## 📁 Repository Structure

```
d:/news_rank/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── pytest.ini               # Pytest configuration
│
├── backend/
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       ├── config.py         # Application settings (Pydantic Settings)
│       └── main.py           # FastAPI entrypoint and routes
│
├── frontend/                 # Client UI (HTML, CSS, Vanilla JS)
├── selenium_tests/           # Selenium WebDriver E2E test suite
└── docs/                     # Architecture & design documentation
```

---

## ⚙️ Configuration & Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API Key | `your_gemini_api_key_here` |
| `GEMINI_MODEL` | Gemini Model (configurable, non-deprecated) | `gemini-3.8-flash` |
| `GEMINI_REQUESTS_PER_MINUTE` | Shared Gemini request limit | `5` |
| `GEMINI_MAX_CONCURRENCY` | Concurrent Gemini requests | `1` |
| `GEMINI_MAX_RETRIES` | Maximum service-level retries for quota responses | `1` |
| `GEMINI_REQUEST_TIMEOUT` | Total time budget for one Gemini request, in seconds | `30` |
| `MAX_ARTICLES_TO_RANK` | Default article count for free-tier testing | `1` |
| `NEWS_API_KEY` | NewsAPI developer key | `your_news_api_key_here` |
| `HOST` | Server bind host | `127.0.0.1` |
| `PORT` | Server bind port | `8000` |
| `DEBUG` | Enable debug mode | `False` |

---

## 🚀 Quickstart

### 1. Set Up Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run FastAPI Backend

```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Verify Health Check

Visit `http://127.0.0.1:8000/api/health` or run:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "Intelligent News Ranking System Using LLM Agents",
  "version": "0.1.0",
  "timestamp": "2026-09-29T16:30:00.000000+00:00",
  "gemini_model": "gemini-3.8-flash",
  "gemini_configured": false,
  "newsapi_configured": false
}
```

Interactive API documentation is accessible at `http://127.0.0.1:8000/docs`.
