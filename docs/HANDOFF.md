# Project Handoff Document

## Current State

The AI Incident Copilot project is **fully scaffolded** with all source code written. It has NOT been run or tested yet.

## Immediate Next Steps

### 1. Environment Setup
```bash
cd incident-copilot
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env and add:
# - ANTHROPIC_API_KEY (or OPENAI_API_KEY)
# - GITHUB_TOKEN (optional, for fetching GitHub issues)
```

### 2. Run the Pipeline
```bash
# Generate data and build ML artifacts
make all

# Or step by step:
make generate-synthetic   # Creates synthetic incidents
make process-data         # Processes into unified format
make build-index          # Builds FAISS vector index
make train-classifier     # Trains the classifier
```

### 3. Start Services
```bash
# Terminal 1
make run-api   # FastAPI on http://localhost:8000

# Terminal 2
make run-ui    # Streamlit on http://localhost:8501
```

### 4. Run Tests
```bash
make test      # Run pytest
make lint      # Run ruff linter
```

## Key Files to Understand

| File | Purpose |
|------|---------|
| `src/schema.py` | All data models (Incident, AnalysisResult) |
| `src/rag/pipeline.py` | Main RAG orchestration logic |
| `src/api/main.py` | FastAPI endpoints |
| `ui/app.py` | Streamlit UI |
| `Makefile` | All available commands |

## Architecture Summary

```
User Input → FastAPI → RAG Pipeline → Response
                          ↓
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
       Classifier    FAISS Index    LLM Client
       (TF-IDF+LR)   (Embeddings)   (Claude/OpenAI)
```

## Configuration

All settings in `.env`:
- `LLM_PROVIDER`: `anthropic` or `openai`
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`: LLM API key
- `GITHUB_TOKEN`: For fetching GitHub issues (optional)
- `RAG_TOP_K`: Number of similar incidents to retrieve (default: 5)

## Data Flow

1. **Ingestion**: Synthetic generator OR GitHub issues → JSONL
2. **Processing**: Combined into `data/processed/incidents.jsonl`
3. **Indexing**: FAISS index saved to `artifacts/index/`
4. **Training**: Classifier saved to `artifacts/classifier/`
5. **Serving**: API loads index + classifier on startup

## Known Limitations / TODOs

1. **Not tested end-to-end** - Code is written but hasn't been run
2. **GitHub ingestion** requires token for higher rate limits
3. **LLM calls** require valid API key
4. **Embedding model** downloads on first run (~90MB)

## Potential Issues

- If `make build-index` fails: Check that `data/processed/incidents.jsonl` exists
- If API returns 503: Index or classifier not loaded (run `make all` first)
- If LLM analysis fails: Check API key in `.env`

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Analyze incident
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"title":"High latency on API","description":"Response times increased to 5s"}'

# Search similar
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"database connection timeout"}'
```

## File Counts

- Source files: 15 Python modules in `src/`
- Scripts: 7 CLI tools in `scripts/`
- Tests: 4 test files in `tests/`
- Total: ~45 files, ~3500 lines of code

## Contact / Questions

Refer to `docs/IMPLEMENTATION_PLAN.md` for detailed design decisions.
