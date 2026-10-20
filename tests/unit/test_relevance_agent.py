"""Unit tests for RelevanceAgent and RelevanceOutput using mocked LLM responses."""

import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import ValidationError

from backend.app.agents.relevance_agent import (
    RelevanceAgent,
    RelevanceOutput,
    RelevanceAgentError,
    RelevanceValidationError,
    RelevanceExecutionError,
    RELEVANCE_SYSTEM_PROMPT,
    RELEVANCE_USER_TEMPLATE,
)
from backend.app.models.article import Article
from backend.app.services.gemini_service import (
    GeminiService,
    GeminiTimeoutError,
    GeminiAPIKeyMissingError,
    GeminiMalformedResponseError,
    GeminiAPIError,
)
from backend.app.config import Settings


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
def sample_article():
    """Sample normalized Article instance."""
    return Article(
        id="art-test-001",
        title="Breakthrough in Quantum Computing Coherence Announced",
        description="Physicists at National Lab demonstrate 10x coherence time improvement for qubits.",
        content="In a major advance for quantum technology, researchers have sustained superconducting qubit states for over ten minutes.",
        article_url="https://example.com/quantum-news",
        source_name="Quantum Chronicle"
    )


@pytest.fixture
def sample_article_dict():
    """Sample article dictionary."""
    return {
        "title": "Autonomous AI Agents Reshaping Software Engineering",
        "description": "New benchmarks show agentic coding frameworks accelerating development cycles.",
        "content": "A new industry report highlights how pairing LLM agents with automated tools leads to faster feature shipping.",
        "source_name": "Tech Insider"
    }


# ---------------------------------------------------------------------------
# RelevanceOutput Model Tests
# ---------------------------------------------------------------------------

class TestRelevanceOutputModel:
    def test_valid_relevance_output(self):
        output = RelevanceOutput(
            relevance_score=88.5,
            reason="Article directly discusses quantum coherence advancements, matching the query."
        )
        assert output.relevance_score == 88.5
        assert "quantum coherence" in output.reason

    def test_boundary_scores_accepted(self):
        out_zero = RelevanceOutput(relevance_score=0.0, reason="Completely irrelevant topic.")
        out_hundred = RelevanceOutput(relevance_score=100.0, reason="Exact topic match.")
        assert out_zero.relevance_score == 0.0
        assert out_hundred.relevance_score == 100.0

    def test_integer_score_coerced(self):
        output = RelevanceOutput(relevance_score=75, reason="Solid topical relevance.")
        assert output.relevance_score == 75.0
        assert isinstance(output.relevance_score, float)

    @pytest.mark.parametrize("invalid_score", [-0.1, -10.0, 100.1, 150.0])
    def test_out_of_bounds_score_raises_validation_error(self, invalid_score):
        with pytest.raises(ValidationError):
            RelevanceOutput(relevance_score=invalid_score, reason="Some reason")

    def test_non_numeric_score_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RelevanceOutput(relevance_score="not_a_number", reason="Some reason")

    @pytest.mark.parametrize("invalid_reason", ["", "   ", None])
    def test_empty_or_whitespace_reason_raises_validation_error(self, invalid_reason):
        with pytest.raises(ValidationError):
            RelevanceOutput(relevance_score=50.0, reason=invalid_reason)

    def test_reason_whitespace_is_stripped(self):
        output = RelevanceOutput(relevance_score=60.0, reason="  Trimmed explanation  ")
        assert output.reason == "Trimmed explanation"

    def test_alias_mapping_score_and_explanation(self):
        data = {
            "score": 92.0,
            "explanation": "High semantic overlap with user query."
        }
        output = RelevanceOutput.model_validate(data)
        assert output.relevance_score == 92.0
        assert output.reason == "High semantic overlap with user query."

    def test_alias_mapping_relevance_and_rationale(self):
        data = {
            "relevance": 45.0,
            "rationale": "Only tangentially mentions the topic in passing."
        }
        output = RelevanceOutput.model_validate(data)
        assert output.relevance_score == 45.0
        assert output.reason == "Only tangentially mentions the topic in passing."


# ---------------------------------------------------------------------------
# Input Validation Tests
# ---------------------------------------------------------------------------

class TestRelevanceAgentInputValidation:
    @pytest.mark.parametrize("invalid_query", ["", "   ", None, 12345, ["topic"], {"q": "ai"}])
    def test_invalid_query_raises_validation_error(self, mock_gemini_service, sample_article, invalid_query):
        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        with pytest.raises(RelevanceValidationError) as exc_info:
            agent.evaluate(query=invalid_query, article=sample_article)
        assert "Query" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_article", [
        None,
        {},
        {"title": ""},
        {"title": "   "},
        {"title": None},
        12345,
        ["Not", "an", "article"],
        "A string headline"
    ])
    def test_invalid_article_raises_validation_error(self, mock_gemini_service, invalid_article):
        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        with pytest.raises(RelevanceValidationError) as exc_info:
            agent.evaluate(query="quantum computing", article=invalid_article)
        assert "Article" in str(exc_info.value) or "Unsupported article" in str(exc_info.value)

    def test_article_instance_normalization(self, mock_gemini_service, sample_article):
        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        clean_q, art_data = agent._validate_and_normalize_inputs("  quantum computing  ", sample_article)
        assert clean_q == "quantum computing"
        assert art_data["title"] == sample_article.title
        assert art_data["description"] == sample_article.description
        assert art_data["content"] == sample_article.content
        assert art_data["source_name"] == sample_article.source_name

    def test_article_dict_normalization(self, mock_gemini_service, sample_article_dict):
        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        clean_q, art_data = agent._validate_and_normalize_inputs("AI agents", sample_article_dict)
        assert clean_q == "AI agents"
        assert art_data["title"] == sample_article_dict["title"]
        assert art_data["description"] == sample_article_dict["description"]
        assert art_data["content"] == sample_article_dict["content"]
        assert art_data["source_name"] == "Tech Insider"

    def test_article_with_newsapi_nested_source(self, mock_gemini_service):
        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        article_with_nested_source = {
            "title": "Global Climate Summit Concludes",
            "description": "World leaders sign new treaty.",
            "source": {"id": "reuters", "name": "Reuters"}
        }
        clean_q, art_data = agent._validate_and_normalize_inputs("climate change", article_with_nested_source)
        assert art_data["source_name"] == "Reuters"
        assert art_data["content"] == "Not provided"

    def test_article_with_title_only(self, mock_gemini_service):
        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        article_title_only = {"title": "Mars Rover Discovers Ancient Riverbed"}
        clean_q, art_data = agent._validate_and_normalize_inputs("Mars exploration", article_title_only)
        assert art_data["title"] == "Mars Rover Discovers Ancient Riverbed"
        assert art_data["description"] == "Not provided"
        assert art_data["content"] == "Not provided"
        assert art_data["source_name"] == "Unknown"

    def test_custom_object_normalization(self, mock_gemini_service):
        class CustomArticle:
            def __init__(self):
                self.title = "Electric Aircraft Passes Test Flight"
                self.description = "Zero-emission regional flight test successful."
                self.content = None
                self.source_name = "AeroNews"

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        clean_q, art_data = agent._validate_and_normalize_inputs("electric planes", CustomArticle())
        assert art_data["title"] == "Electric Aircraft Passes Test Flight"
        assert art_data["content"] == "Not provided"


# ---------------------------------------------------------------------------
# Mocked LLM Evaluation Tests
# ---------------------------------------------------------------------------

class TestRelevanceAgentExecutionWithMockedLLM:
    @pytest.mark.asyncio
    async def test_aevaluate_highly_relevant_article(self, mock_gemini_service, sample_article):
        expected_output = RelevanceOutput(
            relevance_score=96.0,
            reason="The article specifically details quantum computing coherence breakthroughs, directly answering the query."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(
            query="quantum computing breakthroughs",
            article=sample_article
        )

        assert isinstance(result, RelevanceOutput)
        assert result.relevance_score == 96.0
        assert "quantum computing" in result.reason
        mock_gemini_service.agenerate_structured.assert_called_once()

        # Check call arguments
        call_kwargs = mock_gemini_service.agenerate_structured.call_args.kwargs
        assert call_kwargs["schema"] == RelevanceOutput
        assert RELEVANCE_SYSTEM_PROMPT == call_kwargs["system_instruction"]
        assert "quantum computing breakthroughs" in call_kwargs["prompt"]
        assert sample_article.title in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_aevaluate_irrelevant_article(self, mock_gemini_service, sample_article):
        expected_output = RelevanceOutput(
            relevance_score=4.0,
            reason="The article is about quantum computing and has no connection to celebrity culinary shows."
        )
        mock_gemini_service.agenerate_structured = AsyncMock(return_value=expected_output)

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(
            query="Celebrity MasterChef 2026 winners",
            article=sample_article
        )

        assert result.relevance_score == 4.0
        assert "no connection" in result.reason

    def test_evaluate_synchronous_success(self, mock_gemini_service, sample_article):
        expected_output = RelevanceOutput(
            relevance_score=85.0,
            reason="Substantial alignment with quantum physics research topic."
        )
        mock_gemini_service.generate_structured = MagicMock(return_value=expected_output)

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        result = agent.evaluate(
            query="quantum physics research",
            article=sample_article
        )

        assert result.relevance_score == 85.0
        mock_gemini_service.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_aevaluate_batch_evaluates_all_articles(self, mock_gemini_service, sample_article, sample_article_dict):
        out1 = RelevanceOutput(relevance_score=90.0, reason="Matches quantum topic.")
        out2 = RelevanceOutput(relevance_score=20.0, reason="AI agents unrelated to quantum.")

        mock_gemini_service.agenerate_structured = AsyncMock(side_effect=[out1, out2])

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        results = await agent.aevaluate_batch(
            query="quantum algorithms",
            articles=[sample_article, sample_article_dict]
        )

        assert len(results) == 2
        assert results[0].relevance_score == 90.0
        assert results[1].relevance_score == 20.0
        assert mock_gemini_service.agenerate_structured.call_count == 2

    @pytest.mark.asyncio
    async def test_aevaluate_batch_empty_list(self, mock_gemini_service):
        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        results = await agent.aevaluate_batch(query="test", articles=[])
        assert results == []


# ---------------------------------------------------------------------------
# Graceful LLM Failure Handling Tests
# ---------------------------------------------------------------------------

class TestRelevanceAgentFailureHandling:
    @pytest.mark.asyncio
    async def test_timeout_fallback_on_error_true(self, mock_gemini_service, sample_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiTimeoutError("Request timed out after 30.0s")
        )

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(
            query="quantum computing",
            article=sample_article,
            fallback_on_error=True
        )

        assert isinstance(result, RelevanceOutput)
        assert result.relevance_score == 0.0
        assert "timed out" in result.reason.lower() or "unavailable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_timeout_raises_when_fallback_disabled(self, mock_gemini_service, sample_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiTimeoutError("Request timed out after 30.0s")
        )

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        with pytest.raises(RelevanceExecutionError) as exc_info:
            await agent.aevaluate(
                query="quantum computing",
                article=sample_article,
                fallback_on_error=False
            )
        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_api_key_missing_graceful_handling(self, mock_gemini_service, sample_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiAPIKeyMissingError("GEMINI_API_KEY is not configured.")
        )

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(
            query="quantum computing",
            article=sample_article,
            fallback_on_error=True
        )

        assert result.relevance_score == 0.0
        assert "unavailable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_malformed_llm_response_graceful_handling(self, mock_gemini_service, sample_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiMalformedResponseError("Malformed JSON structure from model.")
        )

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(
            query="quantum computing",
            article=sample_article,
            fallback_on_error=True
        )

        assert result.relevance_score == 0.0
        assert "malformed" in result.reason.lower() or "unavailable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_upstream_api_503_error_graceful_handling(self, mock_gemini_service, sample_article):
        mock_gemini_service.agenerate_structured = AsyncMock(
            side_effect=GeminiAPIError("Google API 503 Backend Error")
        )

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        result = await agent.aevaluate(
            query="quantum computing",
            article=sample_article,
            fallback_on_error=True
        )

        assert result.relevance_score == 0.0
        assert "unavailable" in result.reason.lower()

    def test_sync_evaluate_fallback_on_error_true(self, mock_gemini_service, sample_article):
        mock_gemini_service.generate_structured = MagicMock(
            side_effect=GeminiTimeoutError("Sync call timed out.")
        )

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        result = agent.evaluate(
            query="quantum computing",
            article=sample_article,
            fallback_on_error=True
        )

        assert result.relevance_score == 0.0
        assert "unavailable" in result.reason.lower()

    def test_sync_evaluate_raises_when_fallback_disabled(self, mock_gemini_service, sample_article):
        mock_gemini_service.generate_structured = MagicMock(
            side_effect=RuntimeError("Connection aborted")
        )

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        with pytest.raises(RelevanceExecutionError):
            agent.evaluate(
                query="quantum computing",
                article=sample_article,
                fallback_on_error=False
            )


# ---------------------------------------------------------------------------
# LangChain Chain Integration & Independence Tests
# ---------------------------------------------------------------------------

class TestRelevanceAgentLangChainIntegration:
    def test_prompt_template_structure(self, mock_gemini_service):
        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        assert len(agent.prompt_template.messages) == 2

        # Verify system prompt content has anti-hallucination and score range
        sys_msg = agent.prompt_template.messages[0]
        assert "Anti-Hallucination" in sys_msg.prompt.template or "Anti-Hallucination" in RELEVANCE_SYSTEM_PROMPT
        assert "0 - 100" in sys_msg.prompt.template or "0 - 100" in RELEVANCE_SYSTEM_PROMPT

        # Verify human message variables
        human_msg = agent.prompt_template.messages[1]
        variables = human_msg.prompt.input_variables
        assert "query" in variables
        assert "title" in variables
        assert "description" in variables
        assert "content" in variables
        assert "source_name" in variables

    def test_get_chain_returns_runnable(self, mock_gemini_service):
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_gemini_service.get_llm.return_value = mock_llm

        agent = RelevanceAgent(gemini_service=mock_gemini_service)
        chain = agent.get_chain()

        assert chain is not None
        mock_gemini_service.get_llm.assert_called_once_with(
            temperature=0.0,
            model_name=None
        )
        mock_llm.with_structured_output.assert_called_once_with(RelevanceOutput)

    def test_relevance_agent_independence(self):
        """Verify Relevance Agent has no imports from other agents or ranking synthesis."""
        import backend.app.agents.relevance_agent as rel_module

        # Inspect module symbols
        forbidden_terms = ["sentiment_agent", "credibility_agent", "ranking_agent", "RankedArticle"]
        module_source = open(rel_module.__file__, "r", encoding="utf-8").read()
        for term in forbidden_terms:
            assert term not in module_source, f"Relevance agent should not import or reference {term}"
