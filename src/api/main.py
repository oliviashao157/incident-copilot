"""FastAPI application for incident analysis."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    IncidentResponse,
    SearchRequest,
    SearchResponse,
)
from src.classifier.model import IncidentClassifier
from src.config import get_settings
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import IncidentRetriever
from src.schema import Incident, IncidentInput, IncidentSource

# Global instances
pipeline: Optional[RAGPipeline] = None
retriever: Optional[IncidentRetriever] = None
classifier: Optional[IncidentClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global pipeline, retriever, classifier

    settings = get_settings()

    # Try to initialize retriever
    try:
        if (settings.index_dir / "index.faiss").exists():
            retriever = IncidentRetriever()
            print(f"Loaded index with {retriever.index.size} incidents")
        else:
            print("Warning: No index found. Run 'make build-index' first.")
    except Exception as e:
        print(f"Warning: Failed to load retriever: {e}")

    # Try to initialize classifier
    try:
        if (settings.classifier_dir / "model.joblib").exists():
            classifier = IncidentClassifier(model_path=settings.classifier_dir)
            print("Loaded classifier model")
        else:
            print("Warning: No classifier found. Run 'make train-classifier' first.")
    except Exception as e:
        print(f"Warning: Failed to load classifier: {e}")

    # Initialize pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        classifier=classifier,
    )

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Incident Copilot API",
    description="AI-powered incident classification and RAG-based analysis",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()

    return HealthResponse(
        status="healthy",
        version=__version__,
        index_size=retriever.index.size if retriever else 0,
        classifier_loaded=classifier is not None,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_incident(request: AnalyzeRequest):
    """Analyze an incident using the RAG pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        input_data = IncidentInput(
            title=request.title,
            description=request.description,
        )

        result = pipeline.analyze(
            input_data=input_data,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )

        return AnalyzeResponse(success=True, result=result)

    except Exception as e:
        return AnalyzeResponse(success=False, error=str(e))


@app.get("/incident/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: str):
    """Get an incident by ID."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        incident = retriever.get_incident(incident_id)
        if incident is None:
            return IncidentResponse(
                success=False,
                error=f"Incident not found: {incident_id}",
            )
        return IncidentResponse(success=True, incident=incident)

    except Exception as e:
        return IncidentResponse(success=False, error=str(e))


@app.post("/classify", response_model=ClassifyResponse)
async def classify_incident(request: ClassifyRequest):
    """Classify an incident without full RAG analysis."""
    try:
        temp_incident = Incident(
            id="temp",
            title=request.title,
            description=request.description,
            source=IncidentSource.MANUAL,
        )

        if classifier:
            category, confidence = classifier.predict(temp_incident)
        else:
            from src.classifier.categories import infer_category_from_text

            category, confidence = infer_category_from_text(
                f"{request.title} {request.description}"
            )

        # Estimate severity
        if pipeline:
            severity = pipeline._estimate_severity(request.title, request.description)
        else:
            from src.schema import Severity

            severity = Severity.UNKNOWN

        return ClassifyResponse(
            success=True,
            category=category,
            severity=severity,
            confidence=confidence,
        )

    except Exception as e:
        return ClassifyResponse(success=False, error=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_incidents(request: SearchRequest):
    """Search for similar incidents."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Parse optional date filters
        from datetime import datetime

        after_dt = datetime.fromisoformat(request.after) if request.after else None
        before_dt = datetime.fromisoformat(request.before) if request.before else None

        results = retriever.retrieve(
            query=request.query,
            k=request.top_k,
            threshold=request.threshold,
            categories=request.categories,
            severities=request.severities,
            after=after_dt,
            before=before_dt,
        )

        return SearchResponse(
            success=True,
            results=[
                {
                    "incident_id": r.incident.id,
                    "title": r.incident.title,
                    "category": r.incident.category.value,
                    "severity": r.incident.severity.value,
                    "similarity_score": r.similarity_score,
                }
                for r in results
            ],
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))
