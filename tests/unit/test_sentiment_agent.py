"""Unit tests for SentimentAgent and SentimentOutput using mocked LLM responses."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import ValidationError

from backend.app.agents.sentiment_agent import (
    SentimentAgent,
    SentimentOutput,
    SentimentAgentError,
    SentimentValidationError,
    SentimentExecutionError,
    SENTIMENT_SYSTEM_PROMPT,
    SENTIMENT_USER_TEMPLATE,
    SentimentType,
)
from backend.app.models.article import Article
from backend.app.models.ranking import SentimentLabel
from backend.app.services.gemini_service import (
    GeminiService,
    GeminiTimeoutError,
    GeminiAPIKeyMissingError,
    GeminiMalformedResponseError,
    GeminiAPIError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gemini_service():
    """Mocked GeminiService fixture."""
    service = MagicMock(spec=GeminiService)
    service.model_name = "gemini-1.5-flash"
    service.timeout = 30.0
    return service


@pytest.fixture
def sample_positive_article():
    """Sample article with positive news content."""
    return Article(
        id="art-pos-001",
        title="Breakthrough Solar Cell Reaches Record 34% Efficiency",
        description="Researchers develop tandem perovskite cell setting world record in clean energy generation.",
        content="In an inspiring milestone for renewable energy, international scientists demonstrated unprecedented solar conversion efficiency, lowering future production costs.",
        article_url="https://example.com/solar-record",
        source_name="CleanTech Review"
    )


@pytest.fixture
def sample_negative_article():
    """Sample article with negative news content."""
    return Article(
        id="art-neg-001",
        title="Severe Drought Causes Extensive Crop Failure Across Midwest",
        description="Agricultural losses mount as record heat wave devastates local farming communities.",
        content="Farmers face catastrophic economic hardship as prolonged water shortages wipe out millions of bushels of corn and soy.",
        article_url="https://example.com/drought-crisis",
        source_name="AgriNews"
    )


@pytest.fixture
def sample_neutral_article():
    """Sample article with neutral/factual reporting."""
    return Article(
        id="art-neu-001",
        title="Central Bank Holds Interest Rates Steady at 4.5%",
        description="Monetary policy committee voted 7-2 to maintain the current benchmark rate.",
        content="The Federal Reserve announced Wednesday that its target benchmark rate remains unchanged, citing balanced labor market and inflation indicators.",
        article_url="https://example.com/central-bank-decision",
        source_name="Financial Times"
    )


@pytest.fixture
def sample_article_dict():
    """Sample article represented as a dictionary."""
    return {
        "title": "Municipal Council Approves New City Park Design",
        "description": "Council members voted in favor of the urban green space proposal.",
        "content": "Construction for the ten-acre community park is scheduled to begin this spring.",
        "source_name": "Metro Gazette"
    }


# ---------------------------------------------------------------------------
# SentimentOutput Schema Tests
# ---------------------------------------------------------------------------

class TestSentimentOutputModel:
    @pytest.mark.parametrize("sentiment,score,reason", [
        ("positive", 88.0, "Reports historic milestone in renewable energy."),
        ("negative", 15.5, "Details catastrophic economic devastation."),
        ("neutral", 50.0, "Matter-of-fact factual reporting on rate decision."),
    ])
    def test_valid_sentiment_outputs(self, sentiment, score, reason):
        output = SentimentOutput(sentiment=sentiment, sentiment_score=score, reason=reason)
        assert output.sentiment == sentiment
        assert output.sentiment_score == score
        assert output.reason == reason

    def test_boundary_scores_accepted(self):
        out_zero = SentimentOutput(sentiment="negative", sentiment_score=0.0, reason="Total collapse.")
        out_hundred = SentimentOutput(sentiment="positive", sentiment_score=100.0, reason="Unprecedented triumph.")
        out_fifty = SentimentOutput(sentiment="neutral", sentiment_score=50.0, reason="Balanced neutral facts.")
        assert out_zero.sentiment_score == 0.0
        assert out_hundred.sentiment_score == 100.0
        assert out_fifty.sentiment_score == 50.0

    def test_integer_score_coerced(self):
        output = SentimentOutput(sentiment="positive", sentiment_score=75, reason="Constructive news.")
        assert output.sentiment_score == 75.0
        assert isinstance(output.sentiment_score, float)

    def test_score_rounded_to_two_decimals(self):
        output = SentimentOutput(sentiment="positive", sentiment_score=82.456, reason="Strong growth.")
        assert output.sentiment_score == 82.46

    @pytest.mark.parametrize("invalid_score", [-0.1, -25.0, 100.1, 120.0])
    def test_out_of_bounds_score_raises_validation_error(self, invalid_score):
        with pytest.raises(ValidationError):
            SentimentOutput(sentiment="neutral", sentiment_score=invalid_score, reason="Valid reason")

    def test_non_numeric_score_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SentimentOutput(sentiment="positive", sentiment_score="not_a_number", reason="Valid reason")

    @pytest.mark.parametrize("raw_label,expected", [
        ("Positive", "positive"),
        ("POSITIVE", "positive"),
        ("  positive  ", "positive"),
        ("pos", "positive"),
        ("Negative", "negative"),
        ("NEGATIVE", "negative"),
        ("neg", "negative"),
        ("Neutral", "neutral"),
        ("NEUTRAL", "neutral"),
        ("neu", "neutral"),
        ("mixed", "neutral"),
    ])
    def test_sentiment_label_normalization(self, raw_label, expected):
        output = SentimentOutput(sentiment=raw_label, sentiment_score=50.0, reason="Valid reason")
        assert output.sentiment == expected

    def test_sentiment_label_enum_coercion(self):
        output = SentimentOutput(
            sentiment=SentimentLabel.POSITIVE,
            sentiment_score=85.0,
            reason="Positive technological breakthrough."
        )
        assert output.sentiment == "positive"

    @pytest.mark.parametrize("invalid_sentiment", ["happy", "angry", "terrible", "excellent", "unknown", ""])
    def test_invalid_sentiment_raises_validation_error(self, invalid_sentiment):
        with pytest.raises(ValidationError):
            SentimentOutput(sentiment=invalid_sentiment, sentiment_score=50.0, reason="Valid reason")

    @pytest.mark.parametrize("invalid_reason", ["", "   ", None])
    def test_empty_or_whitespace_reason_raises_validation_error(self, invalid_reason):
        with pytest.raises(ValidationError):
            SentimentOutput(sentiment="neutral", sentiment_score=50.0, reason=invalid_reason)

    def test_reason_whitespace_is_stripped(self):
        output = SentimentOutput(
            sentiment="positive",
            sentiment_score=70.0,
            reason="   Progress reported on community housing project.   "
        )
        assert output.reason == "Progress reported on community housing project."

    def test_alias_mapping_polarity_score_and_explanation(self):
        data = {
            "polarity": "positive",
            "score": 90.0,
            "explanation": "Clear optimistic framing and record success."
        }
        output = SentimentOutput.model_validate(data)
        assert output.sentiment == "positive"
        assert output.sentiment_score == 90.0
        assert output.reason == "Clear optimistic framing and record success."

    def test_alias_mapping_label_sentiment_value_and_rationale(self):
        data = {
            "label": "negative",
            "sentiment_value": 22.0,
            "rationale": "Severe conflict and casualties reported."
        }
        output = SentimentOutput.model_validate(data)
        assert output.sentiment == "negative"
        assert output.sentiment_score == 22.0
        assert output.reason == "Severe conflict and casualties reported."


# ---------------------------------------------------------------------------
# Input Normalization & Missing Content Tests
# ---------------------------------------------------------------------------

class TestSentimentAgentInputValidation:
    @pytest.mark.parametrize("invalid_article", [
        None,
        12345,
        ["Not", "an", "article"],
        True,
        False,
        {"key": "value_without_article_fields"},
    ])
    def test_invalid_article_raises_validation_error_when_fallback_disabled(self, mock_gemini_service, invalid_article):
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        with pytest.raises(SentimentValidationError):
            agent.evaluate(article=invalid_article, fallback_on_error=False)

    def test_empty_string_article_raises_validation_error(self, mock_gemini_service):
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        with pytest.raises(SentimentValidationError) as exc_info:
            agent.evaluate(article="   ", fallback_on_error=False)
        assert "empty" in str(exc_info.value).lower()

    def test_raw_text_string_article_normalized_successfully(self, mock_gemini_service):
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        text = "Electric vehicle sales surged 45% this quarter, exceeding analyst expectations."
        data, is_missing = agent._validate_and_normalize_inputs(text)
        assert data["content"] == text
        assert data["title"] == "News Article"
        assert not is_missing

    def test_article_instance_normalization(self, mock_gemini_service, sample_positive_article):
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        data, is_missing = agent._validate_and_normalize_inputs(sample_positive_article)
        assert data["title"] == sample_positive_article.title
        assert data["description"] == sample_positive_article.description
        assert data["content"] == sample_positive_article.content
        assert data["source_name"] == sample_positive_article.source_name
        assert not is_missing

    def test_article_dict_normalization(self, mock_gemini_service, sample_article_dict):
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        data, is_missing = agent._validate_and_normalize_inputs(sample_article_dict)
        assert data["title"] == sample_article_dict["title"]
        assert data["description"] == sample_article_dict["description"]
        assert data["content"] == sample_article_dict["content"]
        assert data["source_name"] == sample_article_dict["source_name"]
        assert not is_missing

    def test_article_with_nested_source_dict(self, mock_gemini_service):
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        article = {
            "title": "Summit On Ocean Conservation Begins",
            "description": "Delegates gather to discuss plastic waste limits.",
            "source": {"id": "bbc-news", "name": "BBC News"}
        }
        data, is_missing = agent._validate_and_normalize_inputs(article)
        assert data["source_name"] == "BBC News"
        assert is_missing  # Content is missing

    def test_custom_object_normalization(self, mock_gemini_service):
        class CustomNews:
            def __init__(self):
                self.title = "Hospital Opens Pediatric Cancer Wing"
                self.description = "State-of-the-art facility will treat hundreds of young patients."
                self.content = "Funded by regional donations, the pediatric wing began operations today."
                self.source_name = "Healthcare Weekly"

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        data, is_missing = agent._validate_and_normalize_inputs(CustomNews())
        assert data["title"] == "Hospital Opens Pediatric Cancer Wing"
        assert data["source_name"] == "Healthcare Weekly"
        assert not is_missing


class TestSentimentAgentMissingContentHandling:
    def test_missing_body_content_with_title_and_description(self, mock_gemini_service):
        """When article content is missing/None, analysis proceeds using title and description."""
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        article_without_content = {
            "title": "Global Trade Volume Climbs 3% in Q3",
            "description": "Shipping bottlenecks eased during the late summer months.",
            "content": None
        }
        data, is_missing = agent._validate_and_normalize_inputs(article_without_content)
        assert is_missing is True
        assert data["title"] == "Global Trade Volume Climbs 3% in Q3"
        assert data["description"] == "Shipping bottlenecks eased during the late summer months."
        assert "Not provided" in data["content"]
        assert "title and description" in data["content"]

    def test_missing_body_content_and_description_with_title_only(self, mock_gemini_service):
        """When content and description are missing, analysis proceeds using title."""
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        title_only = {"title": "Company Announces Q2 Earnings Date"}
        data, is_missing = agent._validate_and_normalize_inputs(title_only)
        assert is_missing is True
        assert data["title"] == "Company Announces Q2 Earnings Date"
        assert data["description"] == "Not provided"
        assert "Not provided" in data["content"]

    def test_completely_missing_content_raises_when_fallback_disabled(self, mock_gemini_service):
        """When title, description, and content are all empty/None, raise SentimentValidationError."""
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        empty_article = {"title": "", "description": "", "content": None}
        with pytest.raises(SentimentValidationError) as exc_info:
            agent.evaluate(article=empty_article, fallback_on_error=False)
        assert "no readable title, description, or content" in str(exc_info.value)

    def test_completely_missing_content_returns_neutral_fallback(self, mock_gemini_service):
        """When article has no readable content and fallback_on_error=True, return neutral default."""
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        empty_article = {"title": None, "description": "", "content": "   "}
        result = agent.evaluate(article=empty_article, fallback_on_error=True)
        assert isinstance(result, SentimentOutput)
        assert result.sentiment == "neutral"
        assert result.sentiment_score == 50.0
        assert "missing or invalid" in result.reason.lower()


# ---------------------------------------------------------------------------
# Mocked LLM Sentiment Evaluation Tests
# ---------------------------------------------------------------------------

class TestSentimentAgentExecutionWithMockedLLM:
    @pytest.mark.asyncio
    async def test_aevaluate_positive_article(self, mock_gemini_service, sample_positive_article):
        expected_output = SentimentOutput(
            sentiment="positive",
            sentiment_score=88.0,
            reason="Article emphasizes record-breaking clean energy efficiency and lower production costs."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_positive_article)

        assert isinstance(result, SentimentOutput)
        assert result.sentiment == "positive"
        assert result.sentiment_score == 88.0
        assert "clean energy" in result.reason

        mock_gemini_service.agenerate_structured.assert_called_once()
        call_kwargs = mock_gemini_service.agenerate_structured.call_args.kwargs
        assert call_kwargs["schema"] == SentimentOutput
        assert call_kwargs["system_instruction"] == SENTIMENT_SYSTEM_PROMPT
        assert sample_positive_article.title in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_aevaluate_negative_article(self, mock_gemini_service, sample_negative_article):
        expected_output = SentimentOutput(
            sentiment="negative",
            sentiment_score=15.0,
            reason="Reports widespread economic hardship, severe drought, and devastating crop failures."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_negative_article)

        assert result.sentiment == "negative"
        assert result.sentiment_score == 15.0
        assert "hardship" in result.reason or "crop" in result.reason

    @pytest.mark.asyncio
    async def test_aevaluate_neutral_article(self, mock_gemini_service, sample_neutral_article):
        expected_output = SentimentOutput(
            sentiment="neutral",
            sentiment_score=50.0,
            reason="Factual and balanced reporting on a central bank interest rate decision."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_neutral_article)

        assert result.sentiment == "neutral"
        assert result.sentiment_score == 50.0
        assert "factual" in result.reason.lower() or "balanced" in result.reason.lower()

    def test_evaluate_synchronous_success(self, mock_gemini_service, sample_positive_article):
        expected_output = SentimentOutput(
            sentiment="positive",
            sentiment_score=92.0,
            reason="Significant technical breakthrough with positive future outlook."
        )
        mock_gemini_service.generate_structured = MagicMock(return_value=expected_output)

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = agent.evaluate(article=sample_positive_article)

        assert result.sentiment == "positive"
        assert result.sentiment_score == 92.0
        mock_gemini_service.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_aevaluate_article_with_missing_content_calls_llm_with_fallback_prompt(
        self, mock_gemini_service
    ):
        """Verify that when body content is missing, prompt explicitly clarifies content status."""
        article_missing_content = {
            "title": "New Tech Hub Creates 2,000 Engineering Jobs",
            "description": "Tech campus ribbon-cutting ceremony held today.",
            "content": None
        }
        expected_output = SentimentOutput(
            sentiment="positive",
            sentiment_score=78.0,
            reason="Job creation and campus launch indicate positive economic development."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=article_missing_content)

        assert result.sentiment == "positive"
        call_kwargs = mock_gemini_service.agenerate_structured.call_args.kwargs
        prompt_text = call_kwargs["prompt"]
        assert "New Tech Hub Creates 2,000 Engineering Jobs" in prompt_text
        assert "title and description" in prompt_text

    @pytest.mark.asyncio
    async def test_aevaluate_batch_evaluates_multiple_articles(
        self, mock_gemini_service, sample_positive_article, sample_negative_article, sample_neutral_article
    ):
        out_pos = SentimentOutput(sentiment="positive", sentiment_score=85.0, reason="Positive.")
        out_neg = SentimentOutput(sentiment="negative", sentiment_score=20.0, reason="Negative.")
        out_neu = SentimentOutput(sentiment="neutral", sentiment_score=50.0, reason="Neutral.")

        mock_gemini_service.agenerate_structured = AsyncMock(side_effect=[out_pos, out_neg, out_neu])

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        results = await agent.aevaluate_batch(
            articles=[sample_positive_article, sample_negative_article, sample_neutral_article],
            max_concurrency=3
        )

        assert len(results) == 3
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"
        assert results[2].sentiment == "neutral"
        assert mock_gemini_service.agenerate_structured.call_count == 3

    @pytest.mark.asyncio
    async def test_aevaluate_batch_empty_list(self, mock_gemini_service):
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        results = await agent.aevaluate_batch(articles=[])
        assert results == []


# ---------------------------------------------------------------------------
# LLM Error Handling Tests
# ---------------------------------------------------------------------------

class TestSentimentAgentFailureHandling:
    @pytest.mark.asyncio
    async def test_timeout_fallback_on_error_true(self, mock_gemini_service, sample_positive_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiTimeoutError("Request timed out after 30.0s")
        )

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_positive_article, fallback_on_error=True)

        assert isinstance(result, SentimentOutput)
        assert result.sentiment == "neutral"
        assert result.sentiment_score == 50.0
        assert "timed out" in result.reason.lower() or "unavailable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_timeout_raises_when_fallback_disabled(self, mock_gemini_service, sample_positive_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiTimeoutError("Request timed out after 30.0s")
        )

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        with pytest.raises(SentimentExecutionError) as exc_info:
            await agent.aevaluate(article=sample_positive_article, fallback_on_error=False)
        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_api_key_missing_fallback(self, mock_gemini_service, sample_positive_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiAPIKeyMissingError("GEMINI_API_KEY is not configured.")
        )

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_positive_article, fallback_on_error=True)

        assert result.sentiment == "neutral"
        assert result.sentiment_score == 50.0
        assert "unavailable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_malformed_llm_response_fallback(self, mock_gemini_service, sample_positive_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiMalformedResponseError("Malformed JSON structure.")
        )

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_positive_article, fallback_on_error=True)

        assert result.sentiment == "neutral"
        assert result.sentiment_score == 50.0
        assert "malformed" in result.reason.lower() or "unavailable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_upstream_api_503_error_fallback(self, mock_gemini_service, sample_positive_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiAPIError("Google API 503 Backend Error")
        )

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_positive_article, fallback_on_error=True)

        assert result.sentiment == "neutral"
        assert result.sentiment_score == 50.0
        assert "unavailable" in result.reason.lower()

    def test_sync_evaluate_fallback_on_error_true(self, mock_gemini_service, sample_positive_article):
        mock_gemini_service.generate_structured = MagicMock(
            side_effect=GeminiTimeoutError("Sync call timed out.")
        )

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        result = agent.evaluate(article=sample_positive_article, fallback_on_error=True)

        assert result.sentiment == "neutral"
        assert result.sentiment_score == 50.0
        assert "unavailable" in result.reason.lower()

    def test_sync_evaluate_raises_when_fallback_disabled(self, mock_gemini_service, sample_positive_article):
        mock_gemini_service.generate_structured = MagicMock(
            side_effect=RuntimeError("Connection terminated")
        )

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        with pytest.raises(SentimentExecutionError):
            agent.evaluate(article=sample_positive_article, fallback_on_error=False)


# ---------------------------------------------------------------------------
# Prompt Rules, Political Neutrality & Anti-Hallucination Tests
# ---------------------------------------------------------------------------

class TestSentimentAgentPromptRulesAndNeutrality:
    def test_system_prompt_strictly_prohibits_political_bias(self):
        """Verify prompt explicitly bans inferring sentiment from political preferences or ideologies."""
        prompt_text = SENTIMENT_SYSTEM_PROMPT.lower()
        assert "political neutrality" in prompt_text
        assert "political preference" in prompt_text
        assert "partisan" in prompt_text
        assert "ideology" in prompt_text or "ideological" in prompt_text

    def test_system_prompt_enforces_content_grounding(self):
        """Verify prompt requires basing analysis strictly on article content without external assumptions."""
        prompt_text = SENTIMENT_SYSTEM_PROMPT.lower()
        assert "content grounding" in prompt_text or "strictly and exclusively" in prompt_text
        assert "anti-hallucination" in prompt_text or "do not invent" in prompt_text
        assert "outside knowledge" in prompt_text or "external" in prompt_text

    def test_system_prompt_defines_0_to_100_score_scale(self):
        """Verify scoring criteria spans 0 to 100 with clear intervals for negative, neutral, and positive."""
        prompt_text = SENTIMENT_SYSTEM_PROMPT
        assert "0 - 100" in prompt_text or "0 to 100" in prompt_text
        assert "Negative" in prompt_text
        assert "Neutral" in prompt_text
        assert "Positive" in prompt_text

    def test_user_template_placeholders(self):
        """Verify template accepts title, description, content, and source_name."""
        assert "{title}" in SENTIMENT_USER_TEMPLATE
        assert "{description}" in SENTIMENT_USER_TEMPLATE
        assert "{content}" in SENTIMENT_USER_TEMPLATE
        assert "{source_name}" in SENTIMENT_USER_TEMPLATE

    def test_prompt_template_structure(self, mock_gemini_service):
        agent = SentimentAgent(gemini_service=mock_gemini_service)
        assert len(agent.prompt_template.messages) == 2
        human_msg = agent.prompt_template.messages[1]
        variables = human_msg.prompt.input_variables
        assert "title" in variables
        assert "description" in variables
        assert "content" in variables
        assert "source_name" in variables

    def test_get_chain_returns_runnable_with_sentiment_output(self, mock_gemini_service):
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_gemini_service.get_llm.return_value = mock_llm

        agent = SentimentAgent(gemini_service=mock_gemini_service)
        chain = agent.get_chain()

        assert chain is not None
        mock_gemini_service.get_llm.assert_called_once_with(temperature=0.0, model_name=None)
        mock_llm.with_structured_output.assert_called_once_with(SentimentOutput)


# ---------------------------------------------------------------------------
# Sentiment Score Independence & Non-Interference with Ranking Tests
# ---------------------------------------------------------------------------

class TestSentimentAgentIndependence:
    def test_sentiment_score_isolated_from_ranking_score(self):
        """Verify SentimentOutput does not compute or contain final_score or rank fields."""
        output = SentimentOutput(sentiment="positive", sentiment_score=78.5, reason="Constructive news.")
        data = output.model_dump()
        assert "final_score" not in data
        assert "rank" not in data
        assert "relevance_score" not in data
        assert data["sentiment_score"] == 78.5

    def test_sentiment_agent_module_independence(self):
        """Verify sentiment_agent does not import ranking agent or final ranking entities."""
        import backend.app.agents.sentiment_agent as sent_module
        source_code = open(sent_module.__file__, "r", encoding="utf-8").read()

        forbidden_references = [
            "ranking_agent",
            "relevance_agent",
            "credibility_agent",
            "RankedArticle",
            "final_score",
        ]
        for term in forbidden_references:
            assert term not in source_code, f"sentiment_agent should not import or reference '{term}'"
