"""Unit tests for Article, ArticleAnalysis, and RankedArticle Pydantic models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from backend.app.models.article import Article
from backend.app.models.ranking import ArticleAnalysis, RankedArticle, SentimentLabel


@pytest.fixture
def valid_article():
    return Article(
        id="art-12345",
        title="Breakthrough in Quantum Computing",
        description="A major leap forward in fault-tolerant qubits.",
        content="Scientists at leading institutes published...",
        source_name="Science Daily",
        source_url="https://sciencedaily.com",
        article_url="https://sciencedaily.com/articles/2026/quantum-breakthrough",
        image_url="https://sciencedaily.com/images/quantum.jpg",
        author="Dr. Alan Turing",
        published_at=datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def valid_analysis():
    return ArticleAnalysis(
        relevance_score=92.5,
        sentiment="Positive",
        sentiment_score=80.0,
        credibility_score=89.0,
        information_quality_score=94.0,
        explanation="Highly relevant to quantum research with verified institutional attribution."
    )


class TestArticleModel:
    def test_valid_article_creation(self, valid_article):
        assert valid_article.id == "art-12345"
        assert valid_article.title == "Breakthrough in Quantum Computing"
        assert valid_article.source_name == "Science Daily"
        assert valid_article.source_url == "https://sciencedaily.com"
        assert valid_article.article_url == "https://sciencedaily.com/articles/2026/quantum-breakthrough"
        assert valid_article.url == valid_article.article_url
        assert valid_article.author == "Dr. Alan Turing"

    def test_auto_derives_source_url_from_article_url(self):
        article = Article(
            id="art-999",
            title="Tech Headline",
            article_url="https://news.ycombinator.com/item?id=12345"
        )
        assert article.source_url == "https://news.ycombinator.com"

    def test_url_alias_backward_compatibility(self):
        article = Article(
            id="art-888",
            title="Another Headline",
            url="https://techcrunch.com/article"
        )
        assert article.article_url == "https://techcrunch.com/article"
        assert article.url == "https://techcrunch.com/article"

    def test_missing_required_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            # Missing id, title, article_url
            Article()

    def test_empty_strings_for_required_fields_fail(self):
        with pytest.raises(ValidationError):
            Article(id=" ", title="Valid Title", article_url="https://example.com")
        with pytest.raises(ValidationError):
            Article(id="art-1", title="   ", article_url="https://example.com")
        with pytest.raises(ValidationError):
            Article(id="art-1", title="Valid Title", article_url="   ")


class TestArticleAnalysisModel:
    def test_valid_analysis_creation(self, valid_analysis):
        assert valid_analysis.relevance_score == 92.5
        assert valid_analysis.sentiment == "Positive"
        assert valid_analysis.sentiment_score == 80.0
        assert valid_analysis.credibility_score == 89.0
        assert valid_analysis.information_quality_score == 94.0
        assert "institutional attribution" in valid_analysis.explanation

    def test_boundary_scores_accepted(self):
        analysis_min = ArticleAnalysis(
            relevance_score=0.0,
            sentiment="Neutral",
            sentiment_score=0.0,
            credibility_score=0.0,
            information_quality_score=0.0,
            explanation="Minimum boundary test."
        )
        assert analysis_min.relevance_score == 0.0

        analysis_max = ArticleAnalysis(
            relevance_score=100.0,
            sentiment="Positive",
            sentiment_score=100.0,
            credibility_score=100.0,
            information_quality_score=100.0,
            explanation="Maximum boundary test."
        )
        assert analysis_max.relevance_score == 100.0

    @pytest.mark.parametrize("field_name, invalid_val", [
        ("relevance_score", -0.1),
        ("relevance_score", 100.1),
        ("sentiment_score", -5.0),
        ("sentiment_score", 105.0),
        ("credibility_score", -1.0),
        ("credibility_score", 100.01),
        ("information_quality_score", -0.01),
        ("information_quality_score", 150.0),
    ])
    def test_out_of_bounds_scores_fail(self, field_name, invalid_val):
        params = {
            "relevance_score": 50.0,
            "sentiment": "Neutral",
            "sentiment_score": 50.0,
            "credibility_score": 50.0,
            "information_quality_score": 50.0,
            "explanation": "Valid rationale."
        }
        params[field_name] = invalid_val
        with pytest.raises(ValidationError):
            ArticleAnalysis(**params)

    def test_empty_sentiment_or_explanation_fails(self):
        with pytest.raises(ValidationError):
            ArticleAnalysis(
                relevance_score=50.0,
                sentiment="   ",
                sentiment_score=50.0,
                credibility_score=50.0,
                information_quality_score=50.0,
                explanation="Valid explanation"
            )
        with pytest.raises(ValidationError):
            ArticleAnalysis(
                relevance_score=50.0,
                sentiment="Positive",
                sentiment_score=50.0,
                credibility_score=50.0,
                information_quality_score=50.0,
                explanation="   "
            )

    def test_sentiment_enum_accepted(self):
        analysis = ArticleAnalysis(
            relevance_score=75.0,
            sentiment=SentimentLabel.POSITIVE,
            sentiment_score=70.0,
            credibility_score=80.0,
            information_quality_score=85.0,
            explanation="Enum supported."
        )
        assert analysis.sentiment == "Positive"


class TestRankedArticleModel:
    def test_valid_ranked_article(self, valid_article, valid_analysis):
        ranked = RankedArticle(
            article=valid_article,
            analysis=valid_analysis,
            final_score=89.456,
            rank=1
        )
        assert ranked.article.title == "Breakthrough in Quantum Computing"
        assert ranked.analysis.relevance_score == 92.5
        assert ranked.final_score == 89.46  # Rounded to 2 decimals
        assert ranked.rank == 1

    def test_final_score_boundary_values(self, valid_article, valid_analysis):
        ranked_min = RankedArticle(
            article=valid_article,
            analysis=valid_analysis,
            final_score=0.0,
            rank=10
        )
        assert ranked_min.final_score == 0.0

        ranked_max = RankedArticle(
            article=valid_article,
            analysis=valid_analysis,
            final_score=100.0,
            rank=1
        )
        assert ranked_max.final_score == 100.0

    @pytest.mark.parametrize("invalid_score", [-0.01, -10.0, 100.01, 105.0])
    def test_invalid_final_score_fails(self, valid_article, valid_analysis, invalid_score):
        with pytest.raises(ValidationError):
            RankedArticle(
                article=valid_article,
                analysis=valid_analysis,
                final_score=invalid_score,
                rank=1
            )

    @pytest.mark.parametrize("invalid_rank", [0, -1, -5])
    def test_invalid_rank_fails(self, valid_article, valid_analysis, invalid_rank):
        with pytest.raises(ValidationError):
            RankedArticle(
                article=valid_article,
                analysis=valid_analysis,
                final_score=85.0,
                rank=invalid_rank
            )
