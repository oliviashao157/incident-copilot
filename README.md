# AI Incident Copilot

An AI-powered incident classification and analysis system that helps on-call engineers quickly understand and respond to production incidents using RAG (Retrieval-Augmented Generation).

## Features

- **Incident Classification**: Automatically categorize incidents using ML (TF-IDF + Logistic Regression)
- **Similar Incident Retrieval**: Find relevant past incidents using vector similarity search (FAISS)
- **AI-Powered Analysis**: Get root cause hypotheses and recommended actions via LLM (Claude/OpenAI)
- **Multiple Data Sources**: Ingest from GitHub issues or generate synthetic data
- **REST API**: FastAPI backend for integration
- **Interactive UI**: Streamlit dashboard for incident analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI Incident Copilot                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   GitHub    │    │  Synthetic  │    │   Manual    │  Data Sources   │
│  │   Issues    │    │  Generator  │    │   Input     │                 │
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

## Quick Start

### Prerequisites

- Python 3.10+
- API key for Claude (Anthropic) or OpenAI

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/incident-copilot.git
cd incident-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run the Pipeline

```bash
# Option 1: Run everything at once
make all

# Option 2: Run steps individually
make generate-synthetic  # Generate 100 synthetic incidents
make process-data        # Process into unified format
make build-index         # Build FAISS vector index
make train-classifier    # Train the ML classifier
```

### Start the Services

```bash
# Terminal 1: Start the API
make run-api

# Terminal 2: Start the UI
make run-ui
```

Then open http://localhost:8501 in your browser.

## Usage

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Analyze an incident
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "title": "High latency on checkout API",
    "description": "P99 response times increased to 5000ms"
  }'

# Search for similar incidents
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database connection timeout", "top_k": 5}'

# Get incident by ID
curl http://localhost:8000/incident/sample-001
```

### Python SDK

```python
from src.rag.pipeline import RAGPipeline
from src.schema import IncidentInput

# Initialize pipeline
pipeline = RAGPipeline()

# Analyze an incident
result = pipeline.analyze(
    IncidentInput(
        title="API returning 503 errors",
        description="Payment service is down, all requests failing"
    )
)

print(f"Category: {result.predicted_category}")
print(f"Severity: {result.predicted_severity}")
print(f"Root Cause: {result.root_cause_hypothesis}")
print(f"Actions: {result.recommended_actions}")
```

## Project Structure

```
incident-copilot/
├── src/
│   ├── schema.py              # Pydantic data models
│   ├── config.py              # Configuration management
│   ├── data/                  # Data ingestion modules
│   │   ├── synthetic_generator.py
│   │   ├── github_ingestion.py
│   │   └── loader.py
│   ├── embedding/             # Vector embeddings
│   │   ├── embedder.py
│   │   └── index.py
│   ├── classifier/            # ML classification
│   │   ├── model.py
│   │   └── categories.py
│   ├── rag/                   # RAG pipeline
│   │   ├── pipeline.py
│   │   ├── retriever.py
│   │   └── llm_client.py
│   └── api/                   # FastAPI app
│       ├── main.py
│       └── models.py
├── scripts/                   # CLI tools
├── prompts/                   # LLM prompt templates
├── tests/                     # Test suite
├── eval/                      # Evaluation harness
├── ui/                        # Streamlit app
└── data/                      # Data storage
```

## Incident Categories

| Category | Description |
|----------|-------------|
| latency | Performance degradation, slow response times |
| outage | Complete service unavailability |
| deployment | Deployment failures, rollout issues |
| config | Configuration errors, misconfigurations |
| capacity | Resource exhaustion (CPU, memory, disk) |
| data | Database issues, data corruption |
| security | Security incidents, certificate issues |
| dependency | Third-party service failures |
| network | Network connectivity issues |

## Configuration

Key environment variables in `.env`:

```bash
# LLM Provider (anthropic or openai)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key

# Model settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=claude-sonnet-4-20250514

# RAG settings
RAG_TOP_K=5
SIMILARITY_THRESHOLD=0.3
```

## Development

```bash
# Run tests
make test

# Run linter
make lint

# Format code
make format

# Run evaluation
make eval-rag
```

## Evaluation

The project includes an evaluation harness with 20 golden test cases:

```bash
python eval/eval_rag.py --golden eval/golden_cases.json --output eval/report.md
```

Metrics evaluated:
- Classification accuracy (category and severity)
- Retrieval quality (similar incidents found)
- Analysis quality (keyword coverage, action relevance)
- Latency

## Data Ethics

This project uses:
- **Synthetic data**: Generated using templates and LLMs for realistic but fictional incidents
- **Public GitHub issues**: From open-source projects (kubernetes, prometheus, etc.)

No proprietary or private incident data is used. Generated data is for demonstration purposes only.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Streamlit](https://streamlit.io/) for the UI
