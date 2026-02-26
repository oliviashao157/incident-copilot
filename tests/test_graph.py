"""Tests for LangGraph-based RAG pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.graph import (
    build_result,
    fallback,
    should_retry_generation,
    should_retry_retrieval,
    validate_output,
)
from src.rag.graph_state import PipelineState
from src.schema import (
    AnalysisResult,
    Category,
    IncidentInput,
    Severity,
    SimilarIncident,
    Incident,
    IncidentSource,
)


@pytest.fixture
def base_state() -> dict:
    """Create a base pipeline state for testing."""
    return {
        "input_data": IncidentInput(
            title="High latency on API",
            description="Response times spiked to 5000ms on the payment endpoint",
        ),
        "top_k": 5,
        "similarity_threshold": 0.3,
        "predicted_category": Category.LATENCY,
        "predicted_severity": Severity.HIGH,
        "category_confidence": 0.85,
        "query_text": "High latency on API Response times spiked to 5000ms",
        "similar_incidents": [],
        "retrieval_attempt": 0,
        "llm_response": None,
        "generation_attempt": 0,
        "generation_error": None,
        "validation_passed": False,
        "result": None,
    }


@pytest.fixture
def valid_llm_response():
    """A valid LLM response dict."""
    return {
        "root_cause_hypothesis": "Database connection pool exhaustion",
        "recommended_actions": ["Scale connection pool", "Add circuit breaker"],
        "estimated_impact": "High - affects all payment transactions",
        "analysis_summary": "Latency caused by DB pool saturation",
        "citations": [
            {
                "incident_id": "inc-1",
                "text": "Similar DB pool issue",
                "relevance": "Same root cause pattern",
            }
        ],
    }


class TestValidateOutput:
    """Tests for the validate_output node."""

    def test_valid_response_passes(self, base_state, valid_llm_response):
        state = {**base_state, "llm_response": valid_llm_response}
        result = validate_output(state)
        assert result["validation_passed"] is True
        assert result["generation_error"] is None

    def test_none_response_fails(self, base_state):
        state = {**base_state, "llm_response": None}
        result = validate_output(state)
        assert result["validation_passed"] is False
        assert "None" in result["generation_error"]

    def test_missing_field_fails(self, base_state, valid_llm_response):
        del valid_llm_response["root_cause_hypothesis"]
        state = {**base_state, "llm_response": valid_llm_response}
        result = validate_output(state)
        assert result["validation_passed"] is False
        assert "root_cause_hypothesis" in result["generation_error"]

    def test_wrong_type_fails(self, base_state, valid_llm_response):
        valid_llm_response["recommended_actions"] = "not a list"
        state = {**base_state, "llm_response": valid_llm_response}
        result = validate_output(state)
        assert result["validation_passed"] is False
        assert "recommended_actions" in result["generation_error"]

    def test_non_string_actions_fail(self, base_state, valid_llm_response):
        valid_llm_response["recommended_actions"] = ["valid", 123]
        state = {**base_state, "llm_response": valid_llm_response}
        result = validate_output(state)
        assert result["validation_passed"] is False
        assert "recommended_actions" in result["generation_error"]


class TestShouldRetryRetrieval:
    """Tests for the should_retry_retrieval conditional edge."""

    def test_retry_when_few_results_first_attempt(self, base_state):
        state = {**base_state, "similar_incidents": [], "retrieval_attempt": 1}
        assert should_retry_retrieval(state) == "retry"

    def test_proceed_when_enough_results(self, base_state):
        incidents = [
            SimilarIncident(
                incident=Incident(
                    id=f"inc-{i}", title="t", description="d", source=IncidentSource.SYNTHETIC
                ),
                similarity_score=0.8,
            )
            for i in range(3)
        ]
        state = {**base_state, "similar_incidents": incidents, "retrieval_attempt": 1}
        assert should_retry_retrieval(state) == "proceed"

    def test_proceed_after_max_retries(self, base_state):
        state = {**base_state, "similar_incidents": [], "retrieval_attempt": 2}
        assert should_retry_retrieval(state) == "proceed"


class TestShouldRetryGeneration:
    """Tests for the should_retry_generation conditional edge."""

    def test_proceed_when_passed(self, base_state):
        state = {**base_state, "validation_passed": True, "generation_attempt": 1}
        assert should_retry_generation(state) == "proceed"

    def test_retry_when_failed_under_max(self, base_state):
        state = {**base_state, "validation_passed": False, "generation_attempt": 1}
        assert should_retry_generation(state) == "retry"

    def test_fallback_when_max_attempts(self, base_state):
        state = {**base_state, "validation_passed": False, "generation_attempt": 3}
        assert should_retry_generation(state) == "fallback"


class TestBuildResult:
    """Tests for the build_result node."""

    def test_builds_analysis_result(self, base_state, valid_llm_response):
        state = {**base_state, "llm_response": valid_llm_response}
        result = build_result(state)

        assert "result" in result
        ar = result["result"]
        assert isinstance(ar, AnalysisResult)
        assert ar.root_cause_hypothesis == valid_llm_response["root_cause_hypothesis"]
        assert ar.recommended_actions == valid_llm_response["recommended_actions"]
        assert ar.predicted_category == Category.LATENCY
        assert len(ar.citations) == 1

    def test_builds_result_without_citations(self, base_state):
        response = {
            "root_cause_hypothesis": "Test",
            "recommended_actions": [],
            "estimated_impact": "Low",
            "analysis_summary": "Summary",
        }
        state = {**base_state, "llm_response": response}
        result = build_result(state)
        assert result["result"].citations == []


class TestFallback:
    """Tests for the fallback node."""

    def test_fallback_produces_result(self, base_state):
        state = {**base_state, "generation_error": "JSON parse error"}
        result = fallback(state)

        ar = result["result"]
        assert isinstance(ar, AnalysisResult)
        assert "JSON parse error" in ar.root_cause_hypothesis
        assert "Investigate manually" in ar.recommended_actions
        assert ar.analysis_summary == "LLM analysis unavailable"


class TestBuildAnalysisGraph:
    """Tests for graph compilation."""

    def test_graph_compiles(self):
        """Verify the graph can be compiled without errors."""
        from src.rag.graph import build_analysis_graph

        mock_pipeline = MagicMock()
        mock_pipeline.classifier = None
        mock_pipeline._estimate_severity.return_value = Severity.MEDIUM
        mock_pipeline._load_prompt_template.return_value = "template {title} {description} {category} {severity} {similar_incidents}"
        mock_pipeline._format_similar_incidents.return_value = ""

        graph = build_analysis_graph(mock_pipeline)
        assert graph is not None
