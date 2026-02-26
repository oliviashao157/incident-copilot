"""Main RAG pipeline for incident analysis."""

import json
from pathlib import Path
from typing import Optional

from src.classifier.model import IncidentClassifier
from src.config import get_settings
from src.rag.llm_client import LLMClient
from src.rag.retriever import IncidentRetriever
from src.schema import (
    AnalysisResult,
    Category,
    Citation,
    Incident,
    IncidentInput,
    IncidentSource,
    Severity,
    SimilarIncident,
)


class RAGPipeline:
    """RAG pipeline for incident classification and analysis."""

    def __init__(
        self,
        retriever: Optional[IncidentRetriever] = None,
        classifier: Optional[IncidentClassifier] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize RAG pipeline.

        Args:
            retriever: IncidentRetriever instance
            classifier: IncidentClassifier instance
            llm_client: LLMClient instance
        """
        self.settings = get_settings()
        self.retriever = retriever
        self.classifier = classifier
        self.llm_client = llm_client or LLMClient()

        # Lazy initialization
        self._retriever_initialized = retriever is not None
        self._classifier_initialized = classifier is not None
        self._graph = None

    def _ensure_retriever(self) -> None:
        """Ensure retriever is initialized."""
        if not self._retriever_initialized:
            self.retriever = IncidentRetriever()
            self._retriever_initialized = True

    def _ensure_classifier(self) -> None:
        """Ensure classifier is initialized."""
        if not self._classifier_initialized:
            classifier_path = self.settings.classifier_dir
            if (classifier_path / "model.joblib").exists():
                self.classifier = IncidentClassifier(model_path=classifier_path)
            else:
                self.classifier = None
            self._classifier_initialized = True

    def _ensure_graph(self):
        """Lazily compile the LangGraph pipeline."""
        if self._graph is None:
            from src.rag.graph import build_analysis_graph

            self._graph = build_analysis_graph(self)
        return self._graph

    def _load_prompt_template(self, name: str) -> str:
        """Load a prompt template from the prompts directory."""
        prompt_path = self.settings.prompts_dir / f"{name}.txt"
        if prompt_path.exists():
            return prompt_path.read_text()

        # Default RAG analysis prompt
        return """You are an expert Site Reliability Engineer analyzing an incident.

Given the following new incident and similar past incidents, provide a detailed analysis.

## New Incident
Title: {title}
Description: {description}
Category: {category}
Severity: {severity}

## Similar Past Incidents
{similar_incidents}

## Your Analysis

Provide your analysis in the following JSON format:
{{
    "root_cause_hypothesis": "Your hypothesis about the root cause based on patterns in similar incidents",
    "recommended_actions": ["Action 1", "Action 2", "Action 3"],
    "estimated_impact": "Assessment of the potential impact",
    "analysis_summary": "A concise summary of your analysis",
    "citations": [
        {{
            "incident_id": "ID of referenced incident",
            "text": "Relevant text from that incident",
            "relevance": "Why this is relevant"
        }}
    ]
}}

Base your analysis on the similar incidents provided. Include citations to support your recommendations."""

    def _format_similar_incidents(self, similar: list[SimilarIncident]) -> str:
        """Format similar incidents for the prompt."""
        if not similar:
            return "No similar incidents found."

        parts = []
        for i, sim in enumerate(similar, 1):
            incident = sim.incident
            parts.append(f"""
### Similar Incident {i} (Similarity: {sim.similarity_score:.2f})
- ID: {incident.id}
- Title: {incident.title}
- Category: {incident.category.value}
- Severity: {incident.severity.value}
- Description: {incident.description[:500]}...
- Resolution: {incident.resolution or 'Not resolved yet'}
""")
        return "\n".join(parts)

    def analyze(
        self,
        input_data: IncidentInput,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ) -> AnalysisResult:
        """Analyze an incident using the LangGraph RAG pipeline.

        Args:
            input_data: Incident input (title and description)
            top_k: Number of similar incidents to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            AnalysisResult with classification, similar incidents, and LLM analysis
        """
        self._ensure_retriever()
        self._ensure_classifier()

        graph = self._ensure_graph()

        initial_state = {
            "input_data": input_data,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "predicted_category": None,
            "predicted_severity": None,
            "category_confidence": 0.0,
            "query_text": "",
            "similar_incidents": [],
            "retrieval_attempt": 0,
            "llm_response": None,
            "generation_attempt": 0,
            "generation_error": None,
            "validation_passed": False,
            "result": None,
        }

        final_state = graph.invoke(initial_state)

        return final_state["result"]

    def _estimate_severity(self, title: str, description: str) -> Severity:
        """Estimate severity based on keywords."""
        text = f"{title} {description}".lower()

        if any(
            word in text
            for word in ["critical", "urgent", "emergency", "complete outage", "all users"]
        ):
            return Severity.CRITICAL
        if any(
            word in text
            for word in ["high", "major", "severe", "production down", "data loss"]
        ):
            return Severity.HIGH
        if any(
            word in text for word in ["degraded", "slow", "intermittent", "some users"]
        ):
            return Severity.MEDIUM
        if any(word in text for word in ["minor", "low", "cosmetic", "typo"]):
            return Severity.LOW

        return Severity.MEDIUM  # Default to medium
