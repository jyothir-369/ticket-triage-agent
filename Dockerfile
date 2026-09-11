# =============================================================================
# Support-Ticket Triage Agent — FastAPI Application
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    WORKERS=1

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
COPY pyproject.toml ./
# PyTorch CPU-only FIRST: Render's free tier has no GPU, and PyPI's default
# Linux torch wheel bundles the CUDA toolkit (~5-8GB of image the app can't
# use). Installing the CPU wheel here lets the `pip install .` below see torch
# already satisfied (sentence-transformers requires only torch>=X) and NOT
# re-pull the CUDA build. We install the base project only — the dev extras
# (pytest, ruff, mypy, black, pre-commit, datasets) are not needed at runtime
# and are omitted to keep the deployed image small and the free-tier build fast.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# ── Application code ───────────────────────────────────────────────────────────
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# ── Health check ───────────────────────────────────────────────────────────────
# start-period is generous: the sentence-transformer model can take 20-30s to
# load on the first /health probe (embeddings are lazy-loaded).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

EXPOSE 8000

# Exec-form CMD with a shell so PORT/WORKERS are read from the environment.
# Render assigns PORT via render.yaml; local docker-compose falls back to 8000/1.
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]
