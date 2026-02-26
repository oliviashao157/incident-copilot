"""LangGraph-based RAG pipeline with retry logic and conditional branching."""

import json
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from src.rag.graph_state import PipelineState
from src.schema import (
    AnalysisResult,
    Category,
    Citation,
    Incident,
    IncidentSource,
    Severity,
    SimilarIncident,
)

if TYPE_CHECKING:
    from src.rag.pipeline import RAGPipeline


def _make_classify_node(pipeline: "RAGPipeline"):
    """Create the classify node function bound to the pipeline instance."""

    def classify(state: PipelineState) -> dict:
        input_data = state["input_data"]

        temp_incident = Incident(
            id="temp",
            title=input_data.title,
            description=input_data.description,
            source=IncidentSource.MANUAL,
        )

        if pipeline.classifier:
            category, confidence = pipeline.classifier.predict(temp_incident)
        else:
            from src.classifier.categories import infer_category_from_text

            category, confidence = infer_category_from_text(
                f"{input_data.title} {input_data.description}"
            )

        severity = pipeline._estimate_severity(input_data.title, input_data.description)
        query_text = f"{input_data.title} {input_data.description}"

        return {
            "predicted_category": category,
            "predicted_severity": severity,
            "category_confidence": confidence,
            "query_text": query_text,
        }

    return classify


def _make_retrieve_node(pipeline: "RAGPipeline"):
    """Create the retrieve node function bound to the pipeline instance."""

    def retrieve(state: PipelineState) -> dict:
        attempt = state.get("retrieval_attempt", 0)
        top_k = state["top_k"]
        threshold = state["similarity_threshold"]

        # On retry, broaden search parameters
        if attempt > 0:
            threshold = max(0.0, threshold - 0.1)
            top_k = top_k + 5

        similar_incidents = pipeline.retriever.retrieve(
            query=state["query_text"],
            k=top_k,
            threshold=threshold,
        )

        return {
            "similar_incidents": similar_incidents,
            "retrieval_attempt": attempt + 1,
        }

    return retrieve


def _make_generate_node(pipeline: "RAGPipeline"):
    """Create the generate node function bound to the pipeline instance."""

    def generate(state: PipelineState) -> dict:
        attempt = state.get("generation_attempt", 0)

        prompt_template = pipeline._load_prompt_template("rag_analysis")
        prompt = prompt_template.format(
            title=state["input_data"].title,
            description=state["input_data"].description,
            category=state["predicted_category"].value,
            severity=state["predicted_severity"].value,
            similar_incidents=pipeline._format_similar_incidents(state["similar_incidents"]),
        )

        # On retry, include previous error as feedback
        prev_error = state.get("generation_error")
        if prev_error and attempt > 0:
            prompt += (
                f"\n\nPrevious attempt failed validation: {prev_error}\n"
                "Please fix the issue and provide a valid JSON response."
            )

        try:
            llm_response = pipeline.llm_client.generate_json(
                prompt=prompt,
                system_prompt="You are an expert SRE incident analyst. Respond only with valid JSON.",
            )
        except Exception as e:
            return {
                "llm_response": None,
                "generation_attempt": attempt + 1,
                "generation_error": str(e),
                "validation_passed": False,
            }

        return {
            "llm_response": llm_response,
            "generation_attempt": attempt + 1,
            "generation_error": None,
        }

    return generate


def validate_output(state: PipelineState) -> dict:
    """Validate the LLM response has all required fields with correct types."""
    llm_response = state.get("llm_response")

    if llm_response is None:
        return {"validation_passed": False, "generation_error": "LLM response is None"}

    required_fields = {
        "root_cause_hypothesis": str,
        "recommended_actions": list,
        "estimated_impact": str,
        "analysis_summary": str,
    }

    errors = []
    for field, expected_type in required_fields.items():
        if field not in llm_response:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(llm_response[field], expected_type):
            errors.append(
                f"Field '{field}' expected {expected_type.__name__}, "
                f"got {type(llm_response[field]).__name__}"
            )

    if llm_response.get("recommended_actions") and not all(
        isinstance(a, str) for a in llm_response["recommended_actions"]
    ):
        errors.append("All items in 'recommended_actions' must be strings")

    if errors:
        return {"validation_passed": False, "generation_error": "; ".join(errors)}

    return {"validation_passed": True, "generation_error": None}


def build_result(state: PipelineState) -> dict:
    """Assemble the final AnalysisResult from validated state."""
    llm_response = state["llm_response"]

    citations = []
    for cit in llm_response.get("citations", []):
        if isinstance(cit, dict):
            citations.append(
                Citation(
                    incident_id=cit.get("incident_id", "unknown"),
                    text=cit.get("text", ""),
                    relevance=cit.get("relevance", ""),
                )
            )

    result = AnalysisResult(
        query_title=state["input_data"].title,
        query_description=state["input_data"].description,
        predicted_category=state["predicted_category"],
        predicted_severity=state["predicted_severity"],
        category_confidence=state["category_confidence"],
        similar_incidents=state["similar_incidents"],
        root_cause_hypothesis=llm_response.get("root_cause_hypothesis", ""),
        recommended_actions=llm_response.get("recommended_actions", []),
        estimated_impact=llm_response.get("estimated_impact", ""),
        citations=citations,
        analysis_summary=llm_response.get("analysis_summary", ""),
        raw_llm_response=json.dumps(llm_response),
    )

    return {"result": result}


def fallback(state: PipelineState) -> dict:
    """Build a hardcoded fallback response when generation fails after max attempts."""
    error_msg = state.get("generation_error", "Unknown error")

    result = AnalysisResult(
        query_title=state["input_data"].title,
        query_description=state["input_data"].description,
        predicted_category=state["predicted_category"],
        predicted_severity=state["predicted_severity"],
        category_confidence=state["category_confidence"],
        similar_incidents=state["similar_incidents"],
        root_cause_hypothesis=f"Unable to generate analysis: {error_msg}",
        recommended_actions=["Investigate manually", "Check logs", "Review recent changes"],
        estimated_impact="Unknown - requires manual assessment",
        citations=[],
        analysis_summary="LLM analysis unavailable",
        raw_llm_response=None,
    )

    return {"result": result}


def should_retry_retrieval(state: PipelineState) -> str:
    """Decide whether to retry retrieval or proceed to generation."""
    similar = state.get("similar_incidents", [])
    attempt = state.get("retrieval_attempt", 0)

    if len(similar) < 2 and attempt <= 1:
        return "retry"
    return "proceed"


def should_retry_generation(state: PipelineState) -> str:
    """Decide whether to retry generation, use fallback, or proceed."""
    if state.get("validation_passed"):
        return "proceed"

    attempt = state.get("generation_attempt", 0)
    if attempt >= 3:
        return "fallback"
    return "retry"


def build_analysis_graph(pipeline: "RAGPipeline") -> StateGraph:
    """Build and compile the LangGraph analysis pipeline.

    Args:
        pipeline: RAGPipeline instance whose components are used by nodes.

    Returns:
        Compiled LangGraph StateGraph.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("classify", _make_classify_node(pipeline))
    graph.add_node("retrieve", _make_retrieve_node(pipeline))
    graph.add_node("generate", _make_generate_node(pipeline))
    graph.add_node("validate_output", validate_output)
    graph.add_node("build_result", build_result)
    graph.add_node("fallback", fallback)

    # Set entry point
    graph.set_entry_point("classify")

    # classify → retrieve
    graph.add_edge("classify", "retrieve")

    # retrieve → conditional: retry or proceed to generate
    graph.add_conditional_edges(
        "retrieve",
        should_retry_retrieval,
        {"retry": "retrieve", "proceed": "generate"},
    )

    # generate → validate
    graph.add_edge("generate", "validate_output")

    # validate → conditional: proceed, retry, or fallback
    graph.add_conditional_edges(
        "validate_output",
        should_retry_generation,
        {"proceed": "build_result", "retry": "generate", "fallback": "fallback"},
    )

    # Terminal nodes
    graph.add_edge("build_result", END)
    graph.add_edge("fallback", END)

    return graph.compile()
