"""Pydantic models for incident data and analysis results."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Incident severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Category(str, Enum):
    """Incident category types."""

    LATENCY = "latency"
    OUTAGE = "outage"
    DEPLOYMENT = "deployment"
    CONFIG = "config"
    CAPACITY = "capacity"
    DATA = "data"
    SECURITY = "security"
    DEPENDENCY = "dependency"
    NETWORK = "network"
    UNKNOWN = "unknown"


class IncidentSource(str, Enum):
    """Source of the incident data."""

    SYNTHETIC = "synthetic"
    GITHUB = "github"
    KAGGLE = "kaggle"
    MANUAL = "manual"


class Incident(BaseModel):
    """Unified incident schema."""

    id: str = Field(..., description="Unique incident identifier")
    title: str = Field(..., description="Incident title/summary")
    description: str = Field(..., description="Detailed incident description")
    category: Category = Field(default=Category.UNKNOWN, description="Incident category")
    severity: Severity = Field(default=Severity.UNKNOWN, description="Severity level")
    source: IncidentSource = Field(..., description="Data source")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolution: Optional[str] = Field(None, description="Resolution description")
    labels: list[str] = Field(default_factory=list, description="Additional labels/tags")
    metadata: dict = Field(default_factory=dict, description="Source-specific metadata")

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [self.title, self.description]
        if self.resolution:
            parts.append(f"Resolution: {self.resolution}")
        return " ".join(parts)


class SimilarIncident(BaseModel):
    """A similar incident returned from retrieval."""

    incident: Incident
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class Citation(BaseModel):
    """A citation referencing a similar incident."""

    incident_id: str
    text: str = Field(..., description="Relevant text from the incident")
    relevance: str = Field(..., description="Why this is relevant")


class AnalysisResult(BaseModel):
    """RAG analysis output."""

    query_title: str
    query_description: str
    predicted_category: Category
    predicted_severity: Severity
    category_confidence: float = Field(..., ge=0.0, le=1.0)
    similar_incidents: list[SimilarIncident]
    root_cause_hypothesis: str
    recommended_actions: list[str]
    estimated_impact: str
    citations: list[Citation]
    analysis_summary: str
    raw_llm_response: Optional[str] = None


class IncidentInput(BaseModel):
    """Input for incident analysis."""

    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1, max_length=10000)
