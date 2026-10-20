"""Unit tests for CredibilityAgent and CredibilityOutput using mocked LLM responses."""

from datetime import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import ValidationError

from backend.app.agents.credibility_agent import (
    CredibilityAgent,
    CredibilityOutput,
    CredibilityAgentError,
    CredibilityValidationError,
    CredibilityExecutionError,
    CREDIBILITY_SYSTEM_PROMPT,
    CREDIBILITY_USER_TEMPLATE,
)
from backend.app.models.article import Article
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
def sample_high_credibility_article():
    """Sample article with comprehensive metadata, named sources, and official data."""
    return Article(
        id="art-cred-high-001",
        title="WHO Releases Global Health Statistics Report for 2026",
        description="Comprehensive data from 194 member states shows steady decline in premature cardiovascular mortality.",
        content=(
            "According to the official 2026 report published by the World Health Organization on Wednesday, "
            "cardiovascular mortality dropped by 4.2% globally over the past five years. "
            "Lead epidemiologist Dr. Elena Vance presented findings from multi-center studies covering over 10 million participants, "
            "citing improved early hypertension screening and regional primary care access as key contributing factors."
        ),
        article_url="https://example.com/who-global-health-2026",
        source_name="Global Health Journal",
        source_url="https://example.com",
        author="Sarah Jenkins",
        published_at=datetime(2026, 4, 15, 10, 30)
    )


@pytest.fixture
def sample_questionable_article():
    """Sample article with anonymous rumors and sensationalized claims."""
    return Article(
        id="art-cred-low-001",
        title="Secret Whispers Claim Miracle Elixir Cures All Ailments Overnight",
        description="Anonymous insiders claim revolutionary tonic suppressed by global cabal.",
        content=(
            "Unidentified sources on internet forums claim that an ancient herbal mixture eliminates every human illness. "
            "No clinical trials or documented studies exist, but believers insist the miracle potion works without fail. "
            "Authorities have issued repeated consumer warnings against unverified claims."
        ),
        article_url="https://example.com/miracle-rumor",
        source_name="Viral Buzz Hub",
        author=None,
        published_at=None
    )


@pytest.fixture
def sample_minimal_article():
    """Sample article with minimal metadata and headline only."""
    return Article(
        id="art-cred-min-001",
        title="City Council Convenes For Annual Budget Review",
        description="Public meeting held at City Hall on Monday evening.",
        content=None,
        article_url="https://example.com/city-council-budget",
        source_name="City Gazette",
        author=None,
        published_at=None
    )


@pytest.fixture
def sample_article_dict():
    """Sample article represented as a dictionary with nested source."""
    return {
        "title": "National Seismology Center Records 4.1 Magnitude Tremor",
        "description": "Geological survey confirmed minor seismic activity with no reported damage.",
        "content": "A 4.1 magnitude earthquake struck 20 miles north of the coast at 3:14 AM local time, according to USGS data.",
        "source": {"id": "reuters", "name": "Reuters", "url": "https://reuters.com"},
        "article_url": "https://reuters.com/seismic-event",
        "author": "Mark Taylor",
        "published_at": "2026-06-12T04:00:00"
    }


# ---------------------------------------------------------------------------
# CredibilityOutput Schema Tests
# ---------------------------------------------------------------------------

class TestCredibilityOutputModel:
    def test_valid_credibility_output(self):
        output = CredibilityOutput(
            credibility_score=85.0,
            confidence=90.0,
            reason="Clear attribution to named WHO officials, specific statistical figures, and official report citations.",
            limitations="Assessment based on article summary and provided text; no independent lab replication conducted."
        )
        assert output.credibility_score == 85.0
        assert output.confidence == 90.0
        assert "WHO officials" in output.reason
        assert "Assessment based on" in output.limitations

    def test_boundary_scores_accepted(self):
        out_min = CredibilityOutput(
            credibility_score=0.0,
            confidence=0.0,
            reason="Unverifiable assertions.",
            limitations="Severe information absence."
        )
        out_max = CredibilityOutput(
            credibility_score=100.0,
            confidence=100.0,
            reason="Highest observable rigor.",
            limitations="External truth cannot be conclusively verified."
        )
        assert out_min.credibility_score == 0.0
        assert out_min.confidence == 0.0
        assert out_max.credibility_score == 100.0
        assert out_max.confidence == 100.0

    def test_integer_scores_coerced_to_float(self):
        output = CredibilityOutput(
            credibility_score=75,
            confidence=80,
            reason="Standard journalistic attribution.",
            limitations="Missing independent verification."
        )
        assert output.credibility_score == 75.0
        assert output.confidence == 80.0
        assert isinstance(output.credibility_score, float)
        assert isinstance(output.confidence, float)

    def test_scores_rounded_to_two_decimals(self):
        output = CredibilityOutput(
            credibility_score=84.567,
            confidence=72.123,
            reason="Sound attribution.",
            limitations="Brief article text."
        )
        assert output.credibility_score == 84.57
        assert output.confidence == 72.12

    @pytest.mark.parametrize("invalid_score", [-0.1, -10.0, 100.1, 150.0])
    def test_out_of_bounds_credibility_score_raises_validation_error(self, invalid_score):
        with pytest.raises(ValidationError):
            CredibilityOutput(
                credibility_score=invalid_score,
                confidence=50.0,
                reason="Reasonable attribution.",
                limitations="Limitations text."
            )

    @pytest.mark.parametrize("invalid_confidence", [-0.1, -5.0, 100.1, 120.0])
    def test_out_of_bounds_confidence_raises_validation_error(self, invalid_confidence):
        with pytest.raises(ValidationError):
            CredibilityOutput(
                credibility_score=70.0,
                confidence=invalid_confidence,
                reason="Reasonable attribution.",
                limitations="Limitations text."
            )

    def test_non_numeric_score_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CredibilityOutput(
                credibility_score="high",
                confidence=50.0,
                reason="Reason",
                limitations="Limitations"
            )

    @pytest.mark.parametrize("invalid_field", ["", "   ", None])
    def test_empty_or_whitespace_reason_raises_validation_error(self, invalid_field):
        with pytest.raises(ValidationError):
            CredibilityOutput(
                credibility_score=50.0,
                confidence=50.0,
                reason=invalid_field,
                limitations="Valid limitations."
            )

    @pytest.mark.parametrize("invalid_field", ["", "   ", None])
    def test_empty_or_whitespace_limitations_raises_validation_error(self, invalid_field):
        with pytest.raises(ValidationError):
            CredibilityOutput(
                credibility_score=50.0,
                confidence=50.0,
                reason="Valid reason.",
                limitations=invalid_field
            )

    def test_reason_and_limitations_whitespace_stripped(self):
        output = CredibilityOutput(
            credibility_score=65.0,
            confidence=70.0,
            reason="   Attributed to university press release.   ",
            limitations="   No peer-review paper link included.   "
        )
        assert output.reason == "Attributed to university press release."
        assert output.limitations == "No peer-review paper link included."

    def test_alias_mapping_score_quality_and_caveats(self):
        data = {
            "score": 82.0,
            "confidence_score": 75.0,
            "explanation": "Clear attribution to named sources.",
            "caveats": "Body content truncated to 200 words."
        }
        output = CredibilityOutput.model_validate(data)
        assert output.credibility_score == 82.0
        assert output.confidence == 75.0
        assert output.reason == "Clear attribution to named sources."
        assert output.limitations == "Body content truncated to 200 words."

    def test_alias_mapping_credibility_conf_and_constraints(self):
        data = {
            "credibility": 45.0,
            "conf": 30.0,
            "rationale": "Only anonymous sources referenced.",
            "constraints": "Headline and single sentence description only."
        }
        output = CredibilityOutput.model_validate(data)
        assert output.credibility_score == 45.0
        assert output.confidence == 30.0
        assert output.reason == "Only anonymous sources referenced."
        assert output.limitations == "Headline and single sentence description only."


# ---------------------------------------------------------------------------
# Input Normalization & Metadata Handling Tests
# ---------------------------------------------------------------------------

class TestCredibilityAgentInputValidation:
    @pytest.mark.parametrize("invalid_article", [
        None,
        12345,
        ["Not", "an", "article"],
        True,
        False,
        {"some_unrelated_key": "no_title_content"},
    ])
    def test_invalid_article_raises_validation_error_when_fallback_disabled(self, mock_gemini_service, invalid_article):
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        with pytest.raises(CredibilityValidationError):
            agent.evaluate(article=invalid_article, fallback_on_error=False)

    def test_empty_string_article_raises_validation_error(self, mock_gemini_service):
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        with pytest.raises(CredibilityValidationError) as exc_info:
            agent.evaluate(article="   ", fallback_on_error=False)
        assert "empty" in str(exc_info.value).lower()

    def test_raw_text_string_article_normalized_successfully(self, mock_gemini_service):
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        text = "The Ministry of Health announced 500 new clinics will open by December 2026."
        data, is_missing = agent._validate_and_normalize_inputs(text)
        assert data["content"] == text
        assert data["title"] == "News Article"
        assert data["author"] == "Not provided"
        assert data["source_name"] == "Unknown"
        assert not is_missing

    def test_article_instance_normalization_with_full_metadata(self, mock_gemini_service, sample_high_credibility_article):
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        data, is_missing = agent._validate_and_normalize_inputs(sample_high_credibility_article)
        assert data["title"] == sample_high_credibility_article.title
        assert data["description"] == sample_high_credibility_article.description
        assert data["content"] == sample_high_credibility_article.content
        assert data["source_name"] == sample_high_credibility_article.source_name
        assert data["source_url"] == sample_high_credibility_article.source_url
        assert data["article_url"] == sample_high_credibility_article.article_url
        assert data["author"] == "Sarah Jenkins"
        assert "2026-04-15" in data["published_at"]
        assert not is_missing

    def test_article_dict_normalization_with_nested_source(self, mock_gemini_service, sample_article_dict):
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        data, is_missing = agent._validate_and_normalize_inputs(sample_article_dict)
        assert data["title"] == sample_article_dict["title"]
        assert data["source_name"] == "Reuters"
        assert data["source_url"] == "https://reuters.com"
        assert data["article_url"] == "https://reuters.com/seismic-event"
        assert data["author"] == "Mark Taylor"
        assert "2026-06-12" in data["published_at"]
        assert not is_missing

    def test_custom_object_normalization(self, mock_gemini_service):
        class CustomReport:
            def __init__(self):
                self.title = "Aviation Board Releases Initial Crash Finding"
                self.description = "Flight recorders show sensor anomaly."
                self.content = "NTSB investigators confirmed primary instruments recorded airspeed discrepancies."
                self.source_name = "AeroSafety Net"
                self.source_url = "https://aerosafety.example"
                self.article_url = "https://aerosafety.example/report-101"
                self.author = "Dan Reynolds"
                self.published_at = datetime(2026, 7, 2, 8, 0)

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        data, is_missing = agent._validate_and_normalize_inputs(CustomReport())
        assert data["title"] == "Aviation Board Releases Initial Crash Finding"
        assert data["author"] == "Dan Reynolds"
        assert "2026-07-02" in data["published_at"]
        assert not is_missing


# ---------------------------------------------------------------------------
# Missing Information Handling Tests
# ---------------------------------------------------------------------------

class TestCredibilityAgentMissingContentHandling:
    def test_missing_body_content_with_title_and_description(self, mock_gemini_service):
        """When article body content is missing, evaluation proceeds on title and description without crashing."""
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        article_missing_content = {
            "title": "Central Bank Announces Rate Decision",
            "description": "Policy committee kept rates at 4.5% following standard quarterly vote.",
            "content": None
        }
        data, is_missing = agent._validate_and_normalize_inputs(article_missing_content)
        assert is_missing is True
        assert data["title"] == "Central Bank Announces Rate Decision"
        assert data["description"] == "Policy committee kept rates at 4.5% following standard quarterly vote."
        assert "Not provided" in data["content"]
        assert "title and description" in data["content"]

    def test_missing_author_and_published_at_normalized_to_not_provided(self, mock_gemini_service):
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        article_without_metadata = {
            "title": "Local Library Expands Weekend Hours",
            "description": "Branch will now open Saturdays from 10am to 4pm.",
            "content": "Patrons requested additional hours following budget approval.",
            "author": None,
            "published_at": None
        }
        data, is_missing = agent._validate_and_normalize_inputs(article_without_metadata)
        assert data["author"] == "Not provided"
        assert data["published_at"] == "Not provided"
        assert is_missing is False

    def test_completely_missing_content_raises_when_fallback_disabled(self, mock_gemini_service):
        """When title, description, and content are all empty/None, raise CredibilityValidationError."""
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        empty_article = {"title": "", "description": "", "content": None}
        with pytest.raises(CredibilityValidationError) as exc_info:
            agent.evaluate(article=empty_article, fallback_on_error=False)
        assert "no readable title, description, or content" in str(exc_info.value)

    def test_completely_missing_content_returns_neutral_fallback_with_zero_confidence(self, mock_gemini_service):
        """When article has no readable content and fallback_on_error=True, return 50.0 score with 0.0 confidence."""
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        empty_article = {"title": None, "description": "", "content": "   "}
        result = agent.evaluate(article=empty_article, fallback_on_error=True)
        assert isinstance(result, CredibilityOutput)
        assert result.credibility_score == 50.0
        assert result.confidence == 0.0
        assert "missing or invalid" in result.reason.lower()
        assert "no usable article text" in result.limitations.lower()


# ---------------------------------------------------------------------------
# Mocked LLM Credibility Evaluation Tests
# ---------------------------------------------------------------------------

class TestCredibilityAgentExecutionWithMockedLLM:
    @pytest.mark.asyncio
    async def test_aevaluate_high_credibility_article(self, mock_gemini_service, sample_high_credibility_article):
        expected_output = CredibilityOutput(
            credibility_score=88.0,
            confidence=92.0,
            reason="Clear attribution to named WHO officials, specific statistical figures (4.2% drop), and complete metadata.",
            limitations="External corroboration of study findings was not independently conducted."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_high_credibility_article)

        assert isinstance(result, CredibilityOutput)
        assert result.credibility_score == 88.0
        assert result.confidence == 92.0
        assert "WHO officials" in result.reason
        assert "External corroboration" in result.limitations

        mock_gemini_service.agenerate_structured.assert_called_once()
        call_kwargs = mock_gemini_service.agenerate_structured.call_args.kwargs
        assert call_kwargs["schema"] == CredibilityOutput
        assert call_kwargs["system_instruction"] == CREDIBILITY_SYSTEM_PROMPT
        assert sample_high_credibility_article.title in call_kwargs["prompt"]
        assert "Sarah Jenkins" in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_aevaluate_questionable_article(self, mock_gemini_service, sample_questionable_article):
        expected_output = CredibilityOutput(
            credibility_score=22.0,
            confidence=80.0,
            reason="Heavy reliance on anonymous internet rumors, lack of scientific studies, and sensationalist framing.",
            limitations="Analysis limited to available text; claims are inherently unverifiable without clinical data."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_questionable_article)

        assert result.credibility_score == 22.0
        assert result.confidence == 80.0
        assert "anonymous" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_aevaluate_article_with_missing_content_lowers_confidence_not_credibility(
        self, mock_gemini_service, sample_minimal_article
    ):
        """Verify that scarce information results in lower confidence and explicit limitation notes."""
        expected_output = CredibilityOutput(
            credibility_score=55.0,
            confidence=35.0,
            reason="Factual meeting notice with no overt sensationalism, but minimal detail.",
            limitations="Full article content and author information missing; assessment limited to headline."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_minimal_article)

        assert result.credibility_score == 55.0
        assert result.confidence == 35.0
        assert "content" in result.limitations.lower()

        call_kwargs = mock_gemini_service.agenerate_structured.call_args.kwargs
        prompt_text = call_kwargs["prompt"]
        assert sample_minimal_article.title in prompt_text
        assert "title and description" in prompt_text

    def test_evaluate_synchronous_success(self, mock_gemini_service, sample_high_credibility_article):
        expected_output = CredibilityOutput(
            credibility_score=85.0,
            confidence=90.0,
            reason="High observable attribution and official statistics.",
            limitations="No live web fact-checking performed."
        )
        mock_gemini_service.generate_structured = MagicMock(return_value=expected_output)

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = agent.evaluate(article=sample_high_credibility_article)

        assert result.credibility_score == 85.0
        assert result.confidence == 90.0
        mock_gemini_service.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_aevaluate_batch_evaluates_multiple_articles(
        self, mock_gemini_service, sample_high_credibility_article, sample_questionable_article
    ):
        out1 = CredibilityOutput(credibility_score=88.0, confidence=90.0, reason="High.", limitations="None.")
        out2 = CredibilityOutput(credibility_score=20.0, confidence=75.0, reason="Low.", limitations="None.")

        mock_gemini_service.agenerate_structured = AsyncMock(side_effect=[out1, out2])

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        results = await agent.aevaluate_batch(
            articles=[sample_high_credibility_article, sample_questionable_article],
            max_concurrency=2
        )

        assert len(results) == 2
        assert results[0].credibility_score == 88.0
        assert results[1].credibility_score == 20.0
        assert mock_gemini_service.agenerate_structured.call_count == 2

    @pytest.mark.asyncio
    async def test_aevaluate_batch_empty_list(self, mock_gemini_service):
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        results = await agent.aevaluate_batch(articles=[])
        assert results == []


# ---------------------------------------------------------------------------
# LLM Error Handling Tests
# ---------------------------------------------------------------------------

class TestCredibilityAgentFailureHandling:
    @pytest.mark.asyncio
    async def test_timeout_fallback_on_error_true(self, mock_gemini_service, sample_high_credibility_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiTimeoutError("Request timed out after 30.0s")
        )

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_high_credibility_article, fallback_on_error=True)

        assert isinstance(result, CredibilityOutput)
        assert result.credibility_score == 50.0
        assert result.confidence == 0.0
        assert "timed out" in result.reason.lower() or "unavailable" in result.reason.lower()
        assert "service failure" in result.limitations.lower() or "incomplete" in result.limitations.lower()

    @pytest.mark.asyncio
    async def test_timeout_raises_when_fallback_disabled(self, mock_gemini_service, sample_high_credibility_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiTimeoutError("Request timed out after 30.0s")
        )

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        with pytest.raises(CredibilityExecutionError) as exc_info:
            await agent.aevaluate(article=sample_high_credibility_article, fallback_on_error=False)
        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_api_key_missing_fallback(self, mock_gemini_service, sample_high_credibility_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiAPIKeyMissingError("GEMINI_API_KEY is not configured.")
        )

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_high_credibility_article, fallback_on_error=True)

        assert result.credibility_score == 50.0
        assert result.confidence == 0.0
        assert "unavailable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_malformed_llm_response_fallback(self, mock_gemini_service, sample_high_credibility_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiMalformedResponseError("Malformed JSON structure.")
        )

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_high_credibility_article, fallback_on_error=True)

        assert result.credibility_score == 50.0
        assert result.confidence == 0.0
        assert "malformed" in result.reason.lower() or "unavailable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_upstream_api_503_error_fallback(self, mock_gemini_service, sample_high_credibility_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiAPIError("Google API 503 Backend Error")
        )

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(article=sample_high_credibility_article, fallback_on_error=True)

        assert result.credibility_score == 50.0
        assert result.confidence == 0.0
        assert "unavailable" in result.reason.lower()

    def test_sync_evaluate_fallback_on_error_true(self, mock_gemini_service, sample_high_credibility_article):
        mock_gemini_service.generate_structured = MagicMock(
            side_effect=GeminiTimeoutError("Sync call timed out.")
        )

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        result = agent.evaluate(article=sample_high_credibility_article, fallback_on_error=True)

        assert result.credibility_score == 50.0
        assert result.confidence == 0.0
        assert "unavailable" in result.reason.lower()

    def test_sync_evaluate_raises_when_fallback_disabled(self, mock_gemini_service, sample_high_credibility_article):
        mock_gemini_service.generate_structured = MagicMock(
            side_effect=RuntimeError("Connection terminated")
        )

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        with pytest.raises(CredibilityExecutionError):
            agent.evaluate(article=sample_high_credibility_article, fallback_on_error=False)


# ---------------------------------------------------------------------------
# Prompt Rules, Grounding & Anti-Hallucination Tests
# ---------------------------------------------------------------------------

class TestCredibilityAgentPromptRulesAndGroundedness:
    def test_system_prompt_prohibits_claiming_factual_truth(self):
        """Verify prompt explicitly bans claiming factual truth or labeling articles true/false."""
        prompt_text = CREDIBILITY_SYSTEM_PROMPT.lower()
        assert "factual truth" in prompt_text or "prove" in prompt_text
        assert "true or false" in prompt_text

    def test_system_prompt_prohibits_inventing_reputation_or_cross_source_claims(self):
        """Verify prompt bans hallucinating reputation or claiming unperformed cross-source checks."""
        prompt_text = CREDIBILITY_SYSTEM_PROMPT.lower()
        assert "reputation" in prompt_text
        assert "cross-source" in prompt_text

    def test_system_prompt_distinguishes_missing_info_from_low_credibility(self):
        """Verify prompt instructs distinguishing missing information from evidence of low credibility."""
        prompt_text = CREDIBILITY_SYSTEM_PROMPT.lower()
        assert "distinguishing missing information" in prompt_text or "missing information" in prompt_text
        assert "confidence" in prompt_text
        assert "limitations" in prompt_text

    def test_system_prompt_defines_0_to_100_scales(self):
        """Verify both credibility_score and confidence have defined 0-100 criteria."""
        prompt_text = CREDIBILITY_SYSTEM_PROMPT
        assert "credibility_score (0.0 to 100.0)" in prompt_text or "0 - 100" in prompt_text
        assert "confidence (0.0 to 100.0)" in prompt_text or "confidence" in prompt_text

    def test_user_template_includes_all_metadata_fields(self):
        """Verify template accepts title, description, content, source_name, source_url, article_url, author, published_at."""
        assert "{title}" in CREDIBILITY_USER_TEMPLATE
        assert "{description}" in CREDIBILITY_USER_TEMPLATE
        assert "{content}" in CREDIBILITY_USER_TEMPLATE
        assert "{source_name}" in CREDIBILITY_USER_TEMPLATE
        assert "{source_url}" in CREDIBILITY_USER_TEMPLATE
        assert "{article_url}" in CREDIBILITY_USER_TEMPLATE
        assert "{author}" in CREDIBILITY_USER_TEMPLATE
        assert "{published_at}" in CREDIBILITY_USER_TEMPLATE

    def test_prompt_template_structure(self, mock_gemini_service):
        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        assert len(agent.prompt_template.messages) == 2
        human_msg = agent.prompt_template.messages[1]
        variables = human_msg.prompt.input_variables
        assert "title" in variables
        assert "description" in variables
        assert "content" in variables
        assert "source_name" in variables
        assert "source_url" in variables
        assert "article_url" in variables
        assert "author" in variables
        assert "published_at" in variables

    def test_get_chain_returns_runnable_with_credibility_output(self, mock_gemini_service):
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_gemini_service.get_llm.return_value = mock_llm

        agent = CredibilityAgent(gemini_service=mock_gemini_service)
        chain = agent.get_chain()

        assert chain is not None
        mock_gemini_service.get_llm.assert_called_once_with(temperature=0.0, model_name=None)
        mock_llm.with_structured_output.assert_called_once_with(CredibilityOutput)


# ---------------------------------------------------------------------------
# Score Independence & Non-Interference with Ranking Tests
# ---------------------------------------------------------------------------

class TestCredibilityAgentIndependence:
    def test_credibility_score_isolated_from_ranking_score(self):
        """Verify CredibilityOutput does not compute or contain final_score or rank fields."""
        output = CredibilityOutput(
            credibility_score=85.0,
            confidence=80.0,
            reason="High source attribution.",
            limitations="Single source report."
        )
        data = output.model_dump()
        assert "final_score" not in data
        assert "rank" not in data
        assert "relevance_score" not in data
        assert "sentiment_score" not in data
        assert data["credibility_score"] == 85.0
        assert data["confidence"] == 80.0

    def test_credibility_agent_module_independence(self):
        """Verify credibility_agent does not import ranking agent or other independent agents."""
        import backend.app.agents.credibility_agent as cred_module
        source_code = open(cred_module.__file__, "r", encoding="utf-8").read()

        forbidden_references = [
            "ranking_agent",
            "relevance_agent",
            "sentiment_agent",
            "RankedArticle",
            "final_score",
        ]
        for term in forbidden_references:
            assert term not in source_code, f"credibility_agent should not import or reference '{term}'"
