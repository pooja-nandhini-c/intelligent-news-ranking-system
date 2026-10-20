"""Unit tests for RankingAgent, RankingWeights, and composite ranking logic."""

from datetime import datetime, timezone, timedelta
import pytest
from pydantic import ValidationError

from backend.app.agents.ranking_agent import (
    RankingAgent,
    RankingWeights,
    ArticleEvaluationInput,
    RankingAgentError,
    RankingValidationError,
    calculate_freshness_score,
    calculate_information_quality_score,
)
from backend.app.agents.relevance_agent import RelevanceOutput
from backend.app.agents.sentiment_agent import SentimentOutput
from backend.app.agents.credibility_agent import CredibilityOutput
from backend.app.models.article import Article
from backend.app.models.ranking import RankedArticle, ArticleAnalysis, SentimentLabel
from backend.app.config import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_article():
    return Article(
        id="art-rank-001",
        title="Breakthrough in Fusion Energy Net Gain Confirmed",
        description="National ignition laboratory confirms sustained net positive fusion reaction.",
        content="In a historic milestone for clean limitless power, researchers achieved a record energy yield exceeding target values by 150%.",
        article_url="https://example.com/fusion-gain",
        source_name="Science Today",
        source_url="https://example.com",
        author="Dr. Rachel Stone",
        published_at=datetime.now(timezone.utc) - timedelta(hours=2)
    )


@pytest.fixture
def high_quality_article():
    return Article(
        id="art-rank-high",
        title="NASA Telescope Discovers Water Vapor on Earth-Sized Exoplanet",
        description="Spectroscopic measurements reveal atmospheric moisture on habitable-zone planet.",
        content="Astronomers analyzing transmission spectra from deep space confirmed clear absorption bands corresponding to atmospheric water vapor.",
        article_url="https://example.com/nasa-exoplanet",
        source_name="Astrophysical Journal",
        author="Prof. Carl Sagan Jr.",
        published_at=datetime.now(timezone.utc) - timedelta(hours=4)
    )


@pytest.fixture
def moderate_article():
    return Article(
        id="art-rank-mod",
        title="City Council Debates Downtown Zoning Changes",
        description="Public hearing attended by local business owners and urban planners.",
        content="Debate centered on commercial building height restrictions and affordable housing allocations in the historic district.",
        article_url="https://example.com/city-zoning",
        source_name="Daily Tribune",
        author="Jane Reporter",
        published_at=datetime.now(timezone.utc) - timedelta(days=3)
    )


@pytest.fixture
def low_quality_article():
    return Article(
        id="art-rank-low",
        title="Weird Rumor About Ancient Aliens Trending Online",
        description="Social media posts speculate about lost pyramids.",
        content=None,
        article_url="https://example.com/alien-rumor",
        source_name="Gossip Net",
        author=None,
        published_at=datetime.now(timezone.utc) - timedelta(days=20)
    )


# ---------------------------------------------------------------------------
# RankingWeights Tests
# ---------------------------------------------------------------------------

class TestRankingWeights:
    def test_default_weights(self):
        weights = RankingWeights()
        assert weights.relevance == 0.40
        assert weights.credibility == 0.25
        assert weights.freshness == 0.20
        assert weights.information_quality == 0.15

        norm = weights.normalized()
        assert pytest.approx(sum(norm.values()), 0.0001) == 1.0
        assert norm["relevance"] == 0.40
        assert norm["credibility"] == 0.25

    def test_custom_unnormalized_weights(self):
        weights = RankingWeights(
            relevance=40.0,
            credibility=30.0,
            freshness=20.0,
            information_quality=10.0
        )
        norm = weights.normalized()
        assert pytest.approx(sum(norm.values()), 0.0001) == 1.0
        assert pytest.approx(norm["relevance"], 0.001) == 0.40
        assert pytest.approx(norm["credibility"], 0.001) == 0.30

    def test_zero_total_weights_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RankingWeights(relevance=0.0, credibility=0.0, freshness=0.0, information_quality=0.0)

    def test_negative_weight_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RankingWeights(relevance=-0.1)


# ---------------------------------------------------------------------------
# Freshness & Information Quality Scoring Tests
# ---------------------------------------------------------------------------

class TestFreshnessAndQualityScoring:
    def test_freshness_immediate_publication(self):
        now = datetime.now(timezone.utc)
        score = calculate_freshness_score(published_at=now, reference_time=now)
        assert score == 100.0

    def test_freshness_exponential_decay_at_seven_days(self):
        now = datetime.now(timezone.utc)
        published = now - timedelta(days=7)
        score = calculate_freshness_score(published_at=published, reference_time=now, half_life_days=7.0)
        assert pytest.approx(score, 0.5) == 50.0

    def test_freshness_missing_date_returns_default(self):
        assert calculate_freshness_score(None, default_score=50.0) == 50.0
        assert calculate_freshness_score("Not provided", default_score=50.0) == 50.0

    def test_freshness_iso_string_parsed(self):
        now = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
        published_str = "2026-10-11T06:00:00Z"
        score = calculate_freshness_score(published_str, reference_time=now)
        assert score > 90.0

    def test_information_quality_full_article(self, high_quality_article):
        score = calculate_information_quality_score(high_quality_article)
        # Baseline 40 + title 10 + desc 15 + content 20 + author 10 + source 5 = 100
        assert score >= 90.0

    def test_information_quality_minimal_article(self, low_quality_article):
        score = calculate_information_quality_score(low_quality_article)
        assert score < 80.0


# ---------------------------------------------------------------------------
# Normal Ranking Tests
# ---------------------------------------------------------------------------

class TestNormalRanking:
    def test_rank_single_article_with_agent_outputs(self, base_article):
        agent = RankingAgent()

        rel_out = RelevanceOutput(relevance_score=90.0, reason="Strong match.")
        cred_out = CredibilityOutput(credibility_score=80.0, confidence=85.0, reason="Clear sources.", limitations="None.")
        sent_out = SentimentOutput(sentiment="positive", sentiment_score=85.0, reason="Optimistic.")

        ranked = agent.rank_article(
            query="clean fusion energy",
            article=base_article,
            relevance=rel_out,
            sentiment=sent_out,
            credibility=cred_out,
            freshness=90.0,
            information_quality=80.0,
            rank=1
        )

        assert isinstance(ranked, RankedArticle)
        assert ranked.article.id == base_article.id
        assert ranked.rank == 1
        # Formula: 0.40*90 + 0.25*80 + 0.20*90 + 0.15*80 = 36 + 20 + 18 + 12 = 86.0
        assert ranked.final_score == 86.0
        assert ranked.analysis.relevance_score == 90.0
        assert ranked.analysis.credibility_score == 80.0
        assert ranked.analysis.sentiment == "Positive"
        assert ranked.analysis.sentiment_score == 85.0
        assert "clean fusion energy" in ranked.analysis.explanation or "relevance" in ranked.analysis.explanation.lower()

    def test_rank_multiple_articles_sorted_highest_to_lowest(
        self, high_quality_article, moderate_article, low_quality_article
    ):
        agent = RankingAgent()

        items = [
            ArticleEvaluationInput(
                article=moderate_article,
                relevance=60.0,
                credibility=60.0,
                sentiment="neutral",
                freshness=50.0,
                information_quality=60.0
            ),
            ArticleEvaluationInput(
                article=high_quality_article,
                relevance=95.0,
                credibility=90.0,
                sentiment="positive",
                freshness=95.0,
                information_quality=90.0
            ),
            ArticleEvaluationInput(
                article=low_quality_article,
                relevance=20.0,
                credibility=15.0,
                sentiment="negative",
                freshness=20.0,
                information_quality=30.0
            ),
        ]

        results = agent.rank_articles(query="astronomy discovery", items=items)

        assert len(results) == 3
        # Should be sorted: High, Moderate, Low
        assert results[0].article.id == high_quality_article.id
        assert results[0].rank == 1
        assert results[1].article.id == moderate_article.id
        assert results[1].rank == 2
        assert results[2].article.id == low_quality_article.id
        assert results[2].rank == 3

        # Scores strictly descending
        assert results[0].final_score > results[1].final_score > results[2].final_score

    def test_configurable_weights_via_settings(self, base_article):
        custom_settings = Settings(
            ranking_weight_relevance=0.70,
            ranking_weight_credibility=0.10,
            ranking_weight_freshness=0.10,
            ranking_weight_information_quality=0.10
        )
        agent = RankingAgent(settings=custom_settings)

        ranked = agent.rank_article(
            query="fusion",
            article=base_article,
            relevance=100.0,
            credibility=50.0,
            sentiment="neutral",
            freshness=50.0,
            information_quality=50.0
        )

        # 0.70*100 + 0.10*50 + 0.10*50 + 0.10*50 = 70 + 5 + 5 + 5 = 85.0
        assert ranked.final_score == 85.0


# ---------------------------------------------------------------------------
# Sentiment Non-Dominance Tests
# ---------------------------------------------------------------------------

class TestSentimentNonDominance:
    def test_negative_news_with_high_merit_outranks_positive_fluff(self):
        agent = RankingAgent()

        # Hard-hitting investigation (negative sentiment, but high relevance/credibility)
        serious_negative = Article(
            id="neg-investigation",
            title="Investigation Exposes Critical Safety Vulnerability in Medical Devices",
            article_url="https://example.com/safety-probe"
        )
        # Positive fluff (positive sentiment, but low relevance/credibility)
        positive_fluff = Article(
            id="pos-fluff",
            title="Sunny Day at the Local Dog Park Brings Joy",
            article_url="https://example.com/dog-park"
        )

        items = [
            ArticleEvaluationInput(
                article=positive_fluff,
                relevance=30.0,
                credibility=40.0,
                sentiment=SentimentOutput(sentiment="positive", sentiment_score=99.0, reason="Pure joy."),
                freshness=80.0,
                information_quality=50.0
            ),
            ArticleEvaluationInput(
                article=serious_negative,
                relevance=95.0,
                credibility=90.0,
                sentiment=SentimentOutput(sentiment="negative", sentiment_score=10.0, reason="Critical flaws."),
                freshness=80.0,
                information_quality=85.0
            ),
        ]

        results = agent.rank_articles(query="medical device safety", items=items)

        # Serious negative article MUST rank #1 despite low sentiment score
        assert results[0].article.id == "neg-investigation"
        assert results[0].rank == 1
        assert results[1].article.id == "pos-fluff"
        assert results[1].rank == 2

    def test_sentiment_score_change_does_not_affect_final_ranking_score(self, base_article):
        agent = RankingAgent()

        ranked_pos = agent.rank_article(
            query="fusion",
            article=base_article,
            relevance=80.0,
            credibility=80.0,
            sentiment={"sentiment": "positive", "sentiment_score": 95.0},
            freshness=80.0,
            information_quality=80.0
        )

        ranked_neg = agent.rank_article(
            query="fusion",
            article=base_article,
            relevance=80.0,
            credibility=80.0,
            sentiment={"sentiment": "negative", "sentiment_score": 15.0},
            freshness=80.0,
            information_quality=80.0
        )

        assert ranked_pos.final_score == ranked_neg.final_score == 80.0


# ---------------------------------------------------------------------------
# Equal Scores & Tie-Breaking Tests
# ---------------------------------------------------------------------------

class TestEqualScoresAndTieBreaking:
    def test_equal_scores_competition_strategy(self):
        agent = RankingAgent(tie_strategy="standard")

        art1 = Article(id="art-1", title="Alpha News", article_url="https://example.com/1")
        art2 = Article(id="art-2", title="Beta News", article_url="https://example.com/2")
        art3 = Article(id="art-3", title="Gamma News", article_url="https://example.com/3")

        # art1 and art2 have exact same scores (80.0), art3 has lower (60.0)
        items = [
            {"article": art1, "relevance": 80.0, "credibility": 80.0, "sentiment": "neutral", "freshness": 80.0, "information_quality": 80.0},
            {"article": art2, "relevance": 80.0, "credibility": 80.0, "sentiment": "neutral", "freshness": 80.0, "information_quality": 80.0},
            {"article": art3, "relevance": 60.0, "credibility": 60.0, "sentiment": "neutral", "freshness": 60.0, "information_quality": 60.0},
        ]

        results = agent.rank_articles(query="test", items=items)

        assert results[0].final_score == 80.0
        assert results[1].final_score == 80.0
        assert results[2].final_score == 60.0

        # Competition ranking ties: 1, 1, 3
        assert results[0].rank == 1
        assert results[1].rank == 1
        assert results[2].rank == 3

    def test_equal_scores_dense_strategy(self):
        agent = RankingAgent(tie_strategy="dense")

        art1 = Article(id="art-1", title="A News", article_url="https://example.com/1")
        art2 = Article(id="art-2", title="B News", article_url="https://example.com/2")
        art3 = Article(id="art-3", title="C News", article_url="https://example.com/3")

        items = [
            {"article": art1, "relevance": 80.0, "credibility": 80.0, "sentiment": "neutral", "freshness": 80.0, "information_quality": 80.0},
            {"article": art2, "relevance": 80.0, "credibility": 80.0, "sentiment": "neutral", "freshness": 80.0, "information_quality": 80.0},
            {"article": art3, "relevance": 60.0, "credibility": 60.0, "sentiment": "neutral", "freshness": 60.0, "information_quality": 60.0},
        ]

        results = agent.rank_articles(query="test", items=items)

        # Dense ranking: 1, 1, 2
        assert results[0].rank == 1
        assert results[1].rank == 1
        assert results[2].rank == 2

    def test_equal_scores_sequential_strategy(self):
        agent = RankingAgent(tie_strategy="sequential")

        art1 = Article(id="art-1", title="A News", article_url="https://example.com/1")
        art2 = Article(id="art-2", title="B News", article_url="https://example.com/2")

        items = [
            {"article": art1, "relevance": 80.0, "credibility": 80.0, "sentiment": "neutral", "freshness": 80.0, "information_quality": 80.0},
            {"article": art2, "relevance": 80.0, "credibility": 80.0, "sentiment": "neutral", "freshness": 80.0, "information_quality": 80.0},
        ]

        results = agent.rank_articles(query="test", items=items)

        # Sequential: 1, 2
        assert results[0].rank == 1
        assert results[1].rank == 2


# ---------------------------------------------------------------------------
# Missing Values & Input Validation Tests
# ---------------------------------------------------------------------------

class TestMissingValuesAndValidation:
    def test_empty_query_raises_validation_error(self, base_article):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError) as exc:
            agent.rank_article(
                query="   ",
                article=base_article,
                relevance=80.0,
                sentiment="neutral",
                credibility=80.0
            )
        assert "Query" in str(exc.value)

    def test_none_article_raises_validation_error(self):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError) as exc:
            agent.rank_article(
                query="test",
                article=None,
                relevance=80.0,
                sentiment="neutral",
                credibility=80.0
            )
        assert "Article" in str(exc.value)

    def test_missing_relevance_raises_validation_error(self, base_article):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError) as exc:
            agent.rank_article(
                query="test",
                article=base_article,
                relevance=None,
                sentiment="neutral",
                credibility=80.0
            )
        assert "relevance" in str(exc.value).lower()

    def test_missing_credibility_raises_validation_error(self, base_article):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError) as exc:
            agent.rank_article(
                query="test",
                article=base_article,
                relevance=80.0,
                sentiment="neutral",
                credibility=None
            )
        assert "credibility" in str(exc.value).lower()

    def test_missing_sentiment_raises_validation_error(self, base_article):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError) as exc:
            agent.rank_article(
                query="test",
                article=base_article,
                relevance=80.0,
                sentiment=None,
                credibility=80.0
            )
        assert "sentiment" in str(exc.value).lower()

    def test_empty_items_returns_empty_list(self):
        agent = RankingAgent()
        assert agent.rank_articles(query="test", items=[]) == []


# ---------------------------------------------------------------------------
# Invalid Scores Handling Tests
# ---------------------------------------------------------------------------

class TestInvalidScoresHandling:
    @pytest.mark.parametrize("invalid_score", [-0.1, -10.0, 100.1, 150.0])
    def test_out_of_bounds_relevance_raises_validation_error(self, base_article, invalid_score):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError):
            agent.rank_article(
                query="test",
                article=base_article,
                relevance=invalid_score,
                sentiment="neutral",
                credibility=80.0
            )

    @pytest.mark.parametrize("invalid_score", [-1.0, 101.0])
    def test_out_of_bounds_credibility_raises_validation_error(self, base_article, invalid_score):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError):
            agent.rank_article(
                query="test",
                article=base_article,
                relevance=80.0,
                sentiment="neutral",
                credibility=invalid_score
            )

    @pytest.mark.parametrize("invalid_score", [-5.0, 110.0])
    def test_out_of_bounds_freshness_raises_validation_error(self, base_article, invalid_score):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError):
            agent.rank_article(
                query="test",
                article=base_article,
                relevance=80.0,
                sentiment="neutral",
                credibility=80.0,
                freshness=invalid_score
            )

    def test_non_numeric_score_raises_validation_error(self, base_article):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError):
            agent.rank_article(
                query="test",
                article=base_article,
                relevance="not_a_number",
                sentiment="neutral",
                credibility=80.0
            )

    def test_nan_or_inf_score_raises_validation_error(self, base_article):
        agent = RankingAgent()
        with pytest.raises(RankingValidationError):
            agent.rank_article(
                query="test",
                article=base_article,
                relevance=float("nan"),
                sentiment="neutral",
                credibility=80.0
            )


# ---------------------------------------------------------------------------
# Ranking Order & Input Flexibility Tests
# ---------------------------------------------------------------------------

class TestRankingOrderAndFlexibility:
    def test_tuple_input_format(self, high_quality_article, moderate_article):
        agent = RankingAgent()
        items = [
            (moderate_article, 50.0, "neutral", 50.0, 50.0, 50.0),
            (high_quality_article, 90.0, "positive", 90.0, 90.0, 90.0),
        ]
        results = agent.rank_articles(query="test", items=items)
        assert len(results) == 2
        assert results[0].article.id == high_quality_article.id
        assert results[0].rank == 1
        assert results[1].article.id == moderate_article.id
        assert results[1].rank == 2

    def test_dict_input_format(self, high_quality_article, moderate_article):
        agent = RankingAgent()
        items = [
            {"article": moderate_article, "relevance_score": 40.0, "credibility_score": 40.0, "sentiment": "neutral"},
            {"article": high_quality_article, "relevance_score": 85.0, "credibility_score": 85.0, "sentiment": "positive"},
        ]
        results = agent.rank_articles(query="test", items=items)
        assert results[0].article.id == high_quality_article.id
        assert results[1].article.id == moderate_article.id

    def test_boundary_scores_min_and_max(self, base_article):
        agent = RankingAgent()

        # Minimum scores (all 0.0)
        ranked_min = agent.rank_article(
            query="test",
            article=base_article,
            relevance=0.0,
            credibility=0.0,
            sentiment={"sentiment": "negative", "sentiment_score": 0.0},
            freshness=0.0,
            information_quality=0.0
        )
        assert ranked_min.final_score == 0.0

        # Maximum scores (all 100.0)
        ranked_max = agent.rank_article(
            query="test",
            article=base_article,
            relevance=100.0,
            credibility=100.0,
            sentiment={"sentiment": "positive", "sentiment_score": 100.0},
            freshness=100.0,
            information_quality=100.0
        )
        assert ranked_max.final_score == 100.0
