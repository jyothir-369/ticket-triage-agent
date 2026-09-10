# =============================================================================
# Support-Ticket Triage Agent — Makefile
# =============================================================================

.PHONY: install dev db-up db-down migrate seed test lint format run \
        run-dashboard docker-up docker-down help eval

PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
RUFF := ruff
MYPY := mypy

# ── Help ────────────────────────────────────────────────────────────────────────
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────────────────
install: ## Install all dependencies (including dev)
	$(PIP) install -e ".[dev]"

dev: install ## Install + set up pre-commit hooks
	pre-commit install
	@echo "✅ Development environment ready"

# ── Database ────────────────────────────────────────────────────────────────────
db-up: ## Start PostgreSQL, Qdrant, Redis via Docker
	docker compose up -d postgres qdrant redis
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 5
	@echo "✅ Services running — Postgres:5432, Qdrant:6333, Redis:6379"

db-down: ## Stop all database services
	docker compose down

migrate: ## Run Alembic migrations (alembic upgrade head)
	alembic upgrade head

seed: ## Seed the database with sample tickets and eval fixtures
	$(PYTHON) -m scripts.seed_db

# ── Quality ─────────────────────────────────────────────────────────────────────
test: ## Run the full test suite with coverage
	$(PYTEST) --cov=src --cov-report=term-missing --cov-report=html -v

lint: ## Run linters (ruff + mypy)
	$(RUFF) check src/ tests/ scripts/
	$(MYPY) src/ --ignore-missing-imports

format: ## Auto-format code
	$(BLACK) src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

# ── Run ─────────────────────────────────────────────────────────────────────────
run: ## Start the FastAPI server (dev mode)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard: ## Start the Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

# ── Docker ──────────────────────────────────────────────────────────────────────
docker-up: ## Build and start all services
	docker compose up --build -d
	@echo "✅ All services running"

docker-down: ## Stop and remove all containers
	docker compose down -v

# ── Evaluation ──────────────────────────────────────────────────────────────────
eval: ## Run the evaluation harness
	$(PYTHON) -m scripts.eval_run
