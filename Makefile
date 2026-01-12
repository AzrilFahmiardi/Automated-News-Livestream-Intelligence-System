.PHONY: help setup install download-models run clean test format lint

help:
	@echo "Automated News Livestream Intelligence System"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup          - Install system dependencies and Python packages"
	@echo "  make download-models - Download AI models"
	@echo "  make install        - Setup + download models (complete installation)"
	@echo "  make run            - Run the system"
	@echo "  make clean          - Clean temporary files and cache"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint code with ruff"
	@echo ""

setup:
	@echo "Setting up system..."
	bash scripts/setup.sh

download-models:
	@echo "Downloading AI models..."
	bash scripts/download_models.sh

install: setup download-models
	@echo "Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate venv: source .venv/bin/activate"
	@echo "  2. Run system: make run"

run:
	@echo "Starting system..."
	@if [ ! -f ".venv/bin/activate" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@. .venv/bin/activate && python main.py

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.tmp" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete"

test:
	@echo "Running tests..."
	@if [ ! -f ".venv/bin/activate" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@. .venv/bin/activate && pytest -v

format:
	@echo "Formatting code..."
	@if [ ! -f ".venv/bin/activate" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@. .venv/bin/activate && black src/ main.py
	@echo "Code formatted"

lint:
	@echo "Linting code..."
	@if [ ! -f ".venv/bin/activate" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@. .venv/bin/activate && ruff check src/ main.py
	@echo "Lint complete"

check-models:
	@echo "Checking models..."
	@if [ -f "models/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then \
		echo "LLM model found"; \
	else \
		echo "LLM model not found"; \
	fi
	@if [ -f "models/ggml-base.bin" ]; then \
		echo "Whisper model found"; \
	else \
		echo "Whisper model not found"; \
	fi

dev: format lint
	@echo "Development checks complete"
