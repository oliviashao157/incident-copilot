# AI Incident & On-Call Copilot - Implementation Plan

## Project Overview
Build a portfolio-grade AI/ML system that ingests incident-like data, performs incident classification and RAG-based analysis, and serves results through an API + Streamlit UI.

## Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI Incident Copilot                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   GitHub    │    │  Synthetic  │    │   Kaggle    │  Data Sources   │
│  │   Issues    │    │  Generator  │    │  (Optional) │                 │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                 │
│         │                  │                  │                         │
│         └──────────────────┼──────────────────┘                         │
│                            ▼                                            │
│                 ┌──────────────────────┐                                │
│                 │   Unified Schema     │  data/processed/               │
│                 │   (JSONL files)      │  incidents.jsonl               │
│                 └──────────┬───────────┘                                │
│                            │                                            │
│         ┌──────────────────┼──────────────────┐                         │
│         ▼                  ▼                  ▼                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐               │
│  │  Embedding  │   │  Classifier │   │  Incident Store │               │
│  │  Index      │   │  TF-IDF+LR  │   │  (JSON lookup)  │               │
│  │  (FAISS)    │   │             │   │                 │               │
│  └──────┬──────┘   └──────┬──────┘   └────────┬────────┘               │
│         │                 │                   │                         │
│         └─────────────────┼───────────────────┘                         │
│                           ▼                                             │
│                 ┌──────────────────────┐                                │
│                 │    RAG Pipeline      │                                │
│                 │  1. Classify         │                                │
│                 │  2. Retrieve Top-K   │                                │
│                 │  3. LLM Analysis     │                                │
│                 └──────────┬───────────┘                                │
│                            │                                            │
│         ┌──────────────────┴──────────────────┐                         │
│         ▼                                     ▼                         │
│  ┌─────────────────┐                 ┌─────────────────┐               │
│  │   FastAPI       │                 │   Streamlit     │               │
│  │   /analyze      │◄────────────────│   UI            │               │
│  │   /incident/{id}│                 │                 │               │
│  │   /health       │                 │                 │               │
│  └─────────────────┘                 └─────────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Repository Structure
```
incident-copilot/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Lint + tests
├── src/
│   ├── __init__.py
│   ├── schema.py                  # Pydantic models for Incident
│   ├── config.py                  # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── github_ingestion.py    # GitHub issues fetcher
│   │   ├── synthetic_generator.py # LLM-based synthetic data
│   │   └── loader.py              # Unified data loading
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── embedder.py            # Text embedding (sentence-transformers)
│   │   └── index.py               # FAISS index management
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── model.py               # TF-IDF + LogisticRegression
│   │   └── categories.py          # Category definitions
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── pipeline.py            # Main RAG orchestration
│   │   ├── retriever.py           # Similarity search
│   │   └── llm_client.py          # OpenAI/Anthropic wrapper
│   └── api/
│       ├── __init__.py
│       ├── main.py                # FastAPI app
│       └── models.py              # Request/Response models
├── scripts/
│   ├── fetch_github_issues.py     # CLI for GitHub ingestion
│   ├── generate_synthetic.py      # CLI for synthetic data
│   ├── build_index.py             # Build FAISS index
│   ├── train_classifier.py        # Train ML classifier
│   ├── eval_classifier.py         # Evaluate classifier
│   └── run_all.py                 # Full pipeline runner
├── prompts/
│   ├── synthetic_incident.txt     # Template for generating incidents
│   └── rag_analysis.txt           # Template for RAG analysis
├── tests/
│   ├── __init__.py
│   ├── test_schema.py
│   ├── test_github_ingestion.py
│   ├── test_classifier.py
│   └── test_retrieval.py
├── eval/
│   ├── golden_cases.json          # 20-30 test cases
│   ├── eval_rag.py                # Evaluation script
│   └── report.md                  # Generated report
├── ui/
│   └── app.py                     # Streamlit UI
├── data/
│   ├── raw/                       # Raw fetched data (gitignored)
│   ├── processed/                 # Processed JSONL
│   └── sample/                    # Small sample for testing
├── artifacts/
│   ├── classifier/                # Trained models
│   └── index/                     # FAISS index + mappings
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml                 # Dependencies (using uv/pip)
├── requirements.txt               # Fallback for pip
└── README.md
```

## Implementation Steps (in order)

### Step 1: Repository Scaffold + Environment
**Files to create:**
- `pyproject.toml` - Dependencies: fastapi, uvicorn, streamlit, pydantic, requests, sentence-transformers, faiss-cpu, scikit-learn, openai, anthropic, python-dotenv, pytest, ruff
- `requirements.txt` - Fallback pip requirements
- `.env.example` - Template for API keys
- `.gitignore` - Standard Python + data/artifacts
- `Makefile` - Commands for all operations
- `README.md` - Skeleton with sections
- `src/__init__.py`, etc. - Package structure

### Step 2: Data Schema + Configuration
**Files:**
- `src/schema.py` - Pydantic models:
  - `Incident` with all required fields
  - `AnalysisResult` for RAG output
- `src/config.py` - Settings from env vars
- `src/classifier/categories.py` - Category definitions

**Design decisions:**
- Use Pydantic v2 for validation
- Categories: latency, outage, deployment, config, capacity, data, security, dependency, network, unknown

### Step 3: Synthetic Data Generator
**Files:**
- `src/data/synthetic_generator.py`
- `scripts/generate_synthetic.py`
- `prompts/synthetic_incident.txt`

**Approach:**
- Use LLM (Claude/OpenAI) to generate realistic incidents
- Configurable: categories, count per category, seed
- Output: JSONL matching schema
- Include severity estimation and mock resolutions

### Step 4: GitHub Issues Ingestion
**Files:**
- `src/data/github_ingestion.py`
- `scripts/fetch_github_issues.py`

**Approach:**
- Use GitHub REST API (requests, not PyGithub for simplicity)
- Target repos: kubernetes/kubernetes, apache/airflow, prometheus/prometheus
- Cache responses in data/raw/{repo}.json
- Map GitHub labels to our categories using keyword matching
- Extract comments and infer resolution from closing comments/linked PRs

### Step 5: Data Loader + Processing
**Files:**
- `src/data/loader.py`

**Approach:**
- Combine all sources into unified JSONL
- Deduplicate by title similarity if needed
- Save to data/processed/incidents.jsonl

### Step 6: Embedding + Index Builder
**Files:**
- `src/embedding/embedder.py`
- `src/embedding/index.py`
- `scripts/build_index.py`

**Approach:**
- Use sentence-transformers (all-MiniLM-L6-v2) for embeddings
- FAISS IndexFlatIP for similarity search
- Store mapping: vector_id -> incident_id in JSON

### Step 7: Baseline Classifier
**Files:**
- `src/classifier/model.py`
- `scripts/train_classifier.py`
- `scripts/eval_classifier.py`

**Approach:**
- TF-IDF vectorizer + LogisticRegression
- Train on title + description
- Save with joblib to artifacts/classifier/
- Eval: accuracy, macro-F1, confusion matrix

### Step 8: RAG Pipeline
**Files:**
- `src/rag/llm_client.py`
- `src/rag/retriever.py`
- `src/rag/pipeline.py`
- `prompts/rag_analysis.txt`

**Approach:**
- Retriever: query FAISS index, return top-k with scores
- LLM client: abstract OpenAI/Anthropic with config flag
- Pipeline: classify -> retrieve -> prompt LLM -> parse response
- Faithfulness: check similarity threshold, require citations
- Output structured JSON with all required fields

### Step 9: FastAPI Backend
**Files:**
- `src/api/models.py`
- `src/api/main.py`

**Endpoints:**
- `POST /analyze` - Main RAG endpoint
- `GET /incident/{id}` - Retrieve incident by ID
- `GET /health` - Health check

### Step 10: Streamlit UI
**Files:**
- `ui/app.py`

**Features:**
- Text input for title/description
- Display category, severity
- Show similar incidents with expandable details
- Show LLM analysis with citations highlighted

### Step 11: Tests + CI
**Files:**
- `tests/test_schema.py`
- `tests/test_github_ingestion.py`
- `tests/test_classifier.py`
- `tests/test_retrieval.py`
- `.github/workflows/ci.yml`

**CI Pipeline:**
- Lint with ruff
- Run pytest
- Use sample data for tests

### Step 12: Evaluation Harness
**Files:**
- `eval/golden_cases.json`
- `eval/eval_rag.py`
- `eval/report.md`

**Metrics:**
- Retrieval hit rate
- Citation presence
- Latency

### Step 13: Final README Polish
Complete documentation with:
- Architecture diagram
- Setup instructions
- Demo walkthrough
- Evaluation results
- Data ethics statement

## Key Design Decisions

1. **Embedding Model**: `all-MiniLM-L6-v2` - good balance of quality/speed, runs locally
2. **Vector Store**: FAISS (IndexFlatIP) - simple, no external dependencies
3. **Classifier**: TF-IDF + LogisticRegression - interpretable baseline
4. **LLM Provider**: Swappable via config (ANTHROPIC or OPENAI)
5. **UI**: Streamlit - fast to build, good enough for portfolio

## Verification Plan

1. **Data Pipeline**:
   ```bash
   make generate-synthetic  # Creates 100+ synthetic incidents
   make fetch-github        # Fetches real GitHub issues
   make process-data        # Combines into unified format
   ```

2. **Index + Classifier**:
   ```bash
   make build-index         # Creates FAISS index
   make train-classifier    # Trains and evaluates model
   ```

3. **API + UI**:
   ```bash
   make run-api            # Start FastAPI on :8000
   make run-ui             # Start Streamlit on :8501
   ```

4. **Tests**:
   ```bash
   make test               # Run pytest
   make lint               # Run ruff
   ```

5. **Full Demo**:
   ```bash
   make all                # Complete pipeline
   curl -X POST localhost:8000/analyze -d '{"title":"High latency","description":"API response times increased"}'
   ```

## Dependencies
- Python 3.10+
- fastapi, uvicorn
- streamlit
- pydantic>=2.0
- requests
- sentence-transformers
- faiss-cpu
- scikit-learn
- joblib
- openai
- anthropic
- python-dotenv
- pytest
- ruff

## Implementation Status

- [x] Step 1: Repository Scaffold + Environment
- [x] Step 2: Data Schema + Configuration
- [x] Step 3: Synthetic Data Generator
- [x] Step 4: GitHub Issues Ingestion
- [x] Step 5: Data Loader + Processing
- [x] Step 6: Embedding + Index Builder
- [x] Step 7: Baseline Classifier
- [x] Step 8: RAG Pipeline
- [x] Step 9: FastAPI Backend
- [x] Step 10: Streamlit UI
- [x] Step 11: Tests + CI
- [x] Step 12: Evaluation Harness
- [x] Step 13: Final README Polish

**All steps completed!**
