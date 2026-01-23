"""API request and response models."""

from typing import Optional

from pydantic import BaseModel, Field

from src.schema import AnalysisResult, Category, Incident, Severity


class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint."""

    title: str = Field(..., min_length=1, max_length=500, description="Incident title")
    description: str = Field(
        ..., min_length=1, max_length=10000, description="Incident description"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of similar incidents")
    similarity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class AnalyzeResponse(BaseModel):
    """Response body for /analyze endpoint."""

    success: bool
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None


class IncidentResponse(BaseModel):
    """Response body for /incident/{id} endpoint."""

    success: bool
    incident: Optional[Incident] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    version: str
    index_size: int
    classifier_loaded: bool


class ClassifyRequest(BaseModel):
    """Request body for /classify endpoint."""

    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1, max_length=10000)


class ClassifyResponse(BaseModel):
    """Response body for /classify endpoint."""

    success: bool
    category: Optional[Category] = None
    severity: Optional[Severity] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class SearchRequest(BaseModel):
    """Request body for /search endpoint."""

    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=50)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """Response body for /search endpoint."""

    success: bool
    results: list[dict] = Field(default_factory=list)
    error: Optional[str] = None
