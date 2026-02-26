"""Pipeline state schema for LangGraph-based RAG orchestration."""

from typing import Optional, TypedDict

from src.schema import (
    AnalysisResult,
    Category,
    IncidentInput,
    Severity,
    SimilarIncident,
)


class PipelineState(TypedDict):
    """State flowing through the LangGraph analysis pipeline."""

    # Inputs
    input_data: IncidentInput
    top_k: int
    similarity_threshold: float

    # Classification
    predicted_category: Optional[Category]
    predicted_severity: Optional[Severity]
    category_confidence: float

    # Retrieval
    query_text: str
    similar_incidents: list[SimilarIncident]
    retrieval_attempt: int

    # Generation
    llm_response: Optional[dict]
    generation_attempt: int
    generation_error: Optional[str]

    # Validation
    validation_passed: bool

    # Output
    result: Optional[AnalysisResult]
