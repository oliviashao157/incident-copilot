# Incident Copilot — Project Reference

## Project Overview

AI-powered incident classification and analysis copilot using RAG (Retrieval-Augmented Generation). Classifies infrastructure incidents, retrieves similar past incidents via FAISS vector search, and generates root-cause analysis using LLMs (Anthropic Claude or OpenAI).

## Tech Stack

- **Python 3.10+**, **FastAPI**, **Pydantic v2**, **Streamlit**
- **LangGraph** for pipeline orchestration with retry/conditional logic
- **FAISS** (faiss-cpu) for vector similarity search
- **sentence-transformers** (all-MiniLM-L6-v2, 384-dim) for embeddings
- **scikit-learn** (TF-IDF + LogisticRegression) for classification
- **Anthropic / OpenAI** SDKs for LLM generation

## Directory Structure

```
src/
├── __init__.py                    # Version (0.1.0)
├── schema.py                      # Core Pydantic models & enums
├── config.py                      # Settings (pydantic-settings, .env)
├── api/
│   ├── main.py                    # FastAPI app, endpoints
│   └── models.py                  # Request/response models
├── rag/
│   ├── graph_state.py             # PipelineState TypedDict
│   ├── graph.py                   # LangGraph nodes, edges, build_analysis_graph()
│   ├── pipeline.py                # RAGPipeline (public interface, invokes graph)
│   ├── retriever.py               # IncidentRetriever (FAISS + metadata filtering)
│   └── llm_client.py              # LLMClient (Anthropic/OpenAI abstraction)
├── embedding/
│   ├── embedder.py                # TextEmbedder (sentence-transformers)
│   └── index.py                   # FAISSIndex (build, search, save/load)
├── classifier/
│   ├── model.py                   # IncidentClassifier (TF-IDF + LogisticRegression)
│   └── categories.py              # Keyword/label-based category inference
└── data/
    ├── loader.py                  # IncidentLoader (JSONL read/write)
    ├── incident_store.py          # IncidentStore (metadata indices + filtering)
    ├── github_ingestion.py        # GitHub issue → Incident converter
    └── synthetic_generator.py     # LLM/template-based synthetic data

tests/
├── test_graph.py                  # Graph nodes, edges, conditional logic
├── test_incident_store.py         # Store loading and metadata filtering
├── test_retrieval.py              # Embedder, FAISS index, filtered retrieval
├── test_schema.py                 # Pydantic model validation
├── test_classifier.py             # Classifier training and prediction
└── test_github_ingestion.py       # GitHub issue parsing

ui/app.py                          # Streamlit UI
scripts/                           # CLI utilities (build_index, train_classifier, etc.)
prompts/                           # LLM prompt templates (loaded by name)
data/{raw,processed,sample}/       # Incident data (JSONL)
```

## Core Data Models (`src/schema.py`)

### Enums
- **Category**: `latency`, `outage`, `deployment`, `config`, `capacity`, `data`, `security`, `dependency`, `network`, `unknown`
- **Severity**: `critical`, `high`, `medium`, `low`, `unknown`
- **IncidentSource**: `synthetic`, `github`, `kaggle`, `manual`

### Key Models
- **Incident**: `id`, `title`, `description`, `category`, `severity`, `source`, `created_at`, `resolved_at`, `resolution`, `labels`, `metadata`. Has `to_embedding_text()`.
- **IncidentInput**: `title` (1-500 chars), `description` (1-10000 chars). Used as pipeline input.
- **SimilarIncident**: `incident` + `similarity_score` (0.0-1.0)
- **AnalysisResult**: Full RAG output — `predicted_category`, `predicted_severity`, `category_confidence`, `similar_incidents`, `root_cause_hypothesis`, `recommended_actions`, `estimated_impact`, `citations`, `analysis_summary`, `raw_llm_response`
- **Citation**: `incident_id`, `text`, `relevance`

## LangGraph Pipeline Architecture

### State Schema (`src/rag/graph_state.py`)

`PipelineState` is a TypedDict flowing through all graph nodes:

| Field | Type | Set By |
|-------|------|--------|
| `input_data` | IncidentInput | initial state |
| `top_k`, `similarity_threshold` | int, float | initial state |
| `predicted_category`, `predicted_severity` | Category, Severity | classify |
| `category_confidence` | float | classify |
| `query_text` | str | classify |
| `similar_incidents` | list[SimilarIncident] | retrieve |
| `retrieval_attempt` | int | retrieve |
| `llm_response` | dict or None | generate |
| `generation_attempt` | int | generate |
| `generation_error` | str or None | generate, validate_output |
| `validation_passed` | bool | validate_output |
| `result` | AnalysisResult or None | build_result, fallback |

### Graph Flow (`src/rag/graph.py`)

```
classify → retrieve ──→ should_retry_retrieval? ──→ generate → validate_output
              ↑              │                        ↑              │
              └──── retry ───┘                        │    should_retry_generation?
            (< 2 results                              │       /      |        \
             AND attempt ≤ 1)                      retry   proceed   fallback
                                                   (< 3)     │      (≥ 3)
                                                     └───┘    │        │
                                                        build_result  fallback → END
                                                              └──→ END
```

### Node Functions

All pipeline-bound nodes are created via closure factories (`_make_*_node(pipeline)`) that capture the `RAGPipeline` instance.

| Node | Function | Logic |
|------|----------|-------|
| **classify** | `_make_classify_node()` | Uses `IncidentClassifier.predict()` if trained model exists, else falls back to `infer_category_from_text()`. Calls `_estimate_severity()` for keyword-based severity. |
| **retrieve** | `_make_retrieve_node()` | Calls `retriever.retrieve()`. On retry (attempt > 0): threshold -= 0.1, top_k += 5. |
| **generate** | `_make_generate_node()` | Calls `llm_client.generate_json()`. On retry: appends previous `generation_error` as feedback to the prompt. Catches exceptions and sets `validation_passed = False`. |
| **validate_output** | standalone | Checks `llm_response` for required fields (`root_cause_hypothesis`: str, `recommended_actions`: list[str], `estimated_impact`: str, `analysis_summary`: str). |
| **build_result** | standalone | Assembles `AnalysisResult` from validated state, parses citations. |
| **fallback** | standalone | Produces hardcoded fallback `AnalysisResult` with error message and generic actions. |

### Conditional Edges

| Edge Function | From Node | Returns | Condition |
|---------------|-----------|---------|-----------|
| `should_retry_retrieval` | retrieve | `"retry"` | `len(similar_incidents) < 2 AND retrieval_attempt ≤ 1` |
| | | `"proceed"` | otherwise |
| `should_retry_generation` | validate_output | `"proceed"` | `validation_passed == True` |
| | | `"retry"` | `validation_passed == False AND generation_attempt < 3` |
| | | `"fallback"` | `validation_passed == False AND generation_attempt ≥ 3` |

### Pipeline Entry Point (`src/rag/pipeline.py`)

`RAGPipeline.analyze(input_data, top_k, similarity_threshold) → AnalysisResult`

- Same public signature as before — API and UI require zero changes.
- Internally builds initial `PipelineState` dict and calls `self._graph.invoke(initial_state)`.
- Graph is lazily compiled via `_ensure_graph()` → `build_analysis_graph(self)`.
- Helper methods (`_load_prompt_template`, `_format_similar_incidents`, `_estimate_severity`) remain on the class, called by node closures.

## IncidentStore (`src/data/incident_store.py`)

Wraps `IncidentLoader` with in-memory metadata indices for filtered queries.

### Indices
- `_by_category: dict[Category, set[str]]` — category → incident IDs
- `_by_severity: dict[Severity, set[str]]` — severity → incident IDs
- `_by_date: list[tuple[datetime, str]]` — sorted (datetime, ID) pairs

### Filtering
`filter_ids(categories?, severities?, after?, before?) → set[str]`
- Each non-None param produces a set of matching IDs.
- All sets are intersected (AND logic).
- No params → returns all IDs.
- Data stays validated through `Incident` Pydantic model on load.

## Filtered Retrieval (`src/rag/retriever.py`)

`IncidentRetriever.retrieve()` signature:
```python
def retrieve(self, query, k=5, threshold=0.3, *,
             categories=None, severities=None, after=None, before=None)
```

- **No filters** → existing FAISS search behavior (search k, threshold).
- **With filters** → over-fetch from FAISS (`k * 3`), post-filter by `store.filter_ids()`, take top `k`.
- Retriever uses `IncidentStore` internally (replaces raw `IncidentLoader`). Accepts legacy `loader` param for backward compat (wraps it in a store).

## API Endpoints (`src/api/main.py`)

| Method | Path | Request Model | Response Model | Notes |
|--------|------|---------------|----------------|-------|
| GET | `/health` | — | HealthResponse | Status, version, index size, classifier loaded |
| POST | `/analyze` | AnalyzeRequest | AnalyzeResponse | Full RAG pipeline via LangGraph |
| GET | `/incident/{id}` | — | IncidentResponse | Lookup by ID |
| POST | `/classify` | ClassifyRequest | ClassifyResponse | Classification only (no RAG) |
| POST | `/search` | SearchRequest | SearchResponse | Similarity search with optional filters |

### Search Filters (added to `SearchRequest`)
All optional, backward compatible:
- `categories: list[Category]` — filter by category
- `severities: list[Severity]` — filter by severity
- `after: str` — ISO date string, only incidents after
- `before: str` — ISO date string, only incidents before

Date strings are parsed via `datetime.fromisoformat()` in the endpoint handler.

## Configuration (`src/config.py`)

Uses `pydantic-settings` with `.env` file support.

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `"anthropic"` | `"anthropic"` or `"openai"` |
| `LLM_MODEL` | `"claude-sonnet-4-20250514"` | Anthropic model name |
| `OPENAI_MODEL` | `"gpt-4o-mini"` | OpenAI model name |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence-transformers model |
| `RAG_TOP_K` | `5` | Default similar incidents to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity |

Path properties: `project_root`, `data_dir`, `raw_data_dir`, `processed_data_dir`, `artifacts_dir`, `classifier_dir`, `index_dir`, `prompts_dir`.

## Lazy Initialization Pattern

All heavy components use lazy init to avoid slow startup and allow graceful degradation:
- `RAGPipeline._ensure_retriever()`, `_ensure_classifier()`, `_ensure_graph()`
- `TextEmbedder.model` (property, loads sentence-transformers on first access)
- `LLMClient._get_client()` (creates API client on first call)
- `FAISSIndex` loads from disk via `index_path` constructor param

## Testing

Run all tests: `pytest tests/ -v`

| Test File | Count | What It Covers |
|-----------|-------|----------------|
| `test_graph.py` | 15 | validate_output (5), should_retry_retrieval (3), should_retry_generation (3), build_result (2), fallback (1), graph compilation (1) |
| `test_incident_store.py` | 17 | Loading (4), category filter (3), severity filter (2), date filter (3), combined AND filters (4), stats (1) |
| `test_retrieval.py` | 18 | TextEmbedder (4), FAISSIndex (7), FilteredRetrieval (5 — no filter, category, severity, date, combined) |
| `test_schema.py` | 12 | Incident, IncidentInput, SimilarIncident, AnalysisResult, enums |
| `test_classifier.py` | 17 | Keywords, label inference, text inference, train/predict, save/load |
| `test_github_ingestion.py` | 7 | Issue parsing, severity/category inference (1 skipped: needs GitHub token) |

**Total: 84 passed, 1 skipped**

## Key Design Decisions

1. **Graph nodes use closures** — `_make_*_node(pipeline)` factories bind the `RAGPipeline` instance so nodes can access classifier, retriever, llm_client, and helper methods without global state.
2. **Retrieval retry broadens params** — On retry: `threshold -= 0.1`, `top_k += 5`. Max 1 retry (attempt ≤ 1).
3. **Generation retry includes error feedback** — Previous validation error is appended to the prompt so the LLM can self-correct. Max 3 attempts before fallback.
4. **Filter uses over-fetch + post-filter** — Fetches `k * 3` from FAISS, then filters by metadata. This avoids modifying the FAISS index itself.
5. **IncidentStore wraps IncidentLoader** — Adds index layer without changing on-disk JSONL format. No migration needed.
6. **Backward compatible API** — All new filter fields are optional. Existing callers see no change.
