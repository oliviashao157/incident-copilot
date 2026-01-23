.PHONY: install install-dev lint test clean generate-synthetic fetch-github process-data build-index train-classifier eval-classifier run-api run-ui all

# Environment setup
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Code quality
lint:
	ruff check src/ tests/ scripts/ ui/
	ruff format --check src/ tests/ scripts/ ui/

format:
	ruff check --fix src/ tests/ scripts/ ui/
	ruff format src/ tests/ scripts/ ui/

test:
	pytest tests/ -v

# Data pipeline
generate-synthetic:
	python scripts/generate_synthetic.py --count 100 --output data/processed/synthetic.jsonl

fetch-github:
	python scripts/fetch_github_issues.py --output data/raw/

process-data:
	python scripts/process_data.py --output data/processed/incidents.jsonl

# ML pipeline
build-index:
	python scripts/build_index.py --input data/processed/incidents.jsonl --output artifacts/index/

train-classifier:
	python scripts/train_classifier.py --input data/processed/incidents.jsonl --output artifacts/classifier/

eval-classifier:
	python scripts/eval_classifier.py --model artifacts/classifier/model.joblib --input data/processed/incidents.jsonl

# Evaluation
eval-rag:
	python eval/eval_rag.py --golden eval/golden_cases.json --output eval/report.md

# Run services
run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run ui/app.py --server.port 8501

# Combined targets
data-pipeline: generate-synthetic fetch-github process-data

ml-pipeline: build-index train-classifier

all: data-pipeline ml-pipeline
	@echo "Pipeline complete! Run 'make run-api' and 'make run-ui' to start services."

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-data:
	rm -rf data/raw/* data/processed/*

clean-artifacts:
	rm -rf artifacts/classifier/* artifacts/index/*

clean-all: clean clean-data clean-artifacts
