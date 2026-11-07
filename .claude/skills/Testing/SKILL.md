---
name: Testing
description: Pytest testing best practices for FastAPI + LangGraph + async Python. Use when writing unit, integration, regression, or E2E tests. Covers fixtures, mocking, async testing, and test structure.
installation-note: Remote source (h4vzz/awesome-ai-agent-skills) was a private/inaccessible repository. This skill was authored locally to fill the gap.
---

# Testing (Pytest — FastAPI + LangGraph)

## Core Principles

- Every test has a single clear assertion about one observable behaviour.
- Prefer real implementations over mocks; mock only at system boundaries (external APIs, databases, network).
- Do not weaken assertions or delete failing tests — fix the root cause.
- Skipped tests are not passes. Running `pytest --co` should list every test; `pytest` should execute them.

## Pytest Setup

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
filterwarnings = ["error"]
```

## Async Tests

Use `pytest-asyncio` with `asyncio_mode = "auto"` — no need to decorate every test with `@pytest.mark.asyncio`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
```

## Mocking External Services

Mock at the boundary — patch the service's outbound call, not internal logic:

```python
from unittest.mock import AsyncMock, patch

async def test_classify_uses_llm(mock_ticket):
    with patch("src.services.llm.call_llm_with_fallback", new=AsyncMock(return_value="...")):
        result = await classify_ticket(mock_ticket)
    assert result.category is not None
```

## Fixtures

Define fixtures in `conftest.py`; scope to the smallest lifetime needed:

```python
@pytest.fixture
def sample_ticket():
    from src.models.schemas import Ticket
    return Ticket(id="test-001", content="My login broke", source="test")
```

## Integration vs Unit

- **Unit tests**: mock all I/O; test one function at a time.
- **Integration tests**: use a real (local/test) database and Qdrant; never mock persistence.
- **E2E tests**: spin the full FastAPI app with `AsyncClient` + real services; assert HTTP contract.

## Test Taxonomy for This Project

| Layer | Scope | Mock? |
|---|---|---|
| `test_unit_*.py` | Single function/method | Yes — all I/O |
| `test_integration.py` | API + DB + Qdrant together | No — real services |
| `test_e2e*.py` | Full HTTP round-trip | No — real services |
| `test_eval*.py` | Accuracy on labeled fixtures | No — real LLM or deterministic fallback |

## Regression Tests

Always add a regression test that reproduces the exact bug scenario before fixing it:

```python
async def test_ticket_idempotency_regression():
    """Regression: duplicate POST /tickets/:id/triage must not create two DB rows."""
    ...
```

## Acceptance Criteria Checklist

- [ ] `pytest` exits 0 with no skipped tests
- [ ] Coverage on `src/` ≥ 80% (`pytest --cov=src`)
- [ ] No test imports from `.venv` or uses internal test helpers from production code
- [ ] Eval suite produces real accuracy numbers, not mocked metrics
