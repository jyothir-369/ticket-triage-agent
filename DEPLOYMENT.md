# 🌐 Deployment — Render (Public URLs)

This deploys both the FastAPI backend and the Streamlit dashboard to Render's
free tier, so reviewers can access the project from two public URLs without
cloning the repo.

| Service | Public URL |
|---|---|
| FastAPI backend | `https://support-triage-api.onrender.com` |
| API docs (Swagger) | `https://support-triage-api.onrender.com/docs` |
| Health check | `https://support-triage-api.onrender.com/health` |
| Streamlit dashboard | `https://support-triage-dashboard.onrender.com` |

---

## 0. Prerequisites

- A [Render](https://render.com) account (free tier works).
- A GitHub account with the `ticket-triage-agent` repo pushed to it.
- The cloud services already set up (these live outside Render):

| Service | Role | Where to get it |
|---|---|---|
| **Neon** — PostgreSQL | Relational DB | https://neon.tech (free tier) |
| **Upstash** — Redis | Caching / rate limits | https://upstash.com (free tier) |
| **Qdrant Cloud** — vector DB | RAG retrieval | https://cloud.qdrant.io (free 1 GB) |
| **Gemini** (primary LLM) | Classification / drafting | https://aistudio.google.com/apikey |
| **OpenRouter** (fallback LLM) | Failover provisioning | https://openrouter.ai |

You can create the Render services before or after the cloud services exist —
but the API will report `unhealthy` on `/health` until the secrets below are
filled in.

---

## 1. Step 1 — Push the repo to GitHub

```bash
git remote -v                    # confirm origin → your GitHub repo
git push origin main
```

No fork is needed if you own the repo. (If you're deploying from a fork of
`jyothir-369/ticket-triage-agent`, fork first and push your changes there.)

---

## 2. Step 2 — Connect Render to your GitHub

1. Go to https://render.com and sign in.
2. **Dashboard → New → `Blueprint`** (this reads the checked-in `render.yaml`).
3. If Render hasn't connected to GitHub yet, click **Connect account** and
   authorize Render to access the `ticket-triage-agent` repository.

---

## 3. Step 3 — Create the two services from `render.yaml`

**Option A — Blueprint (recommended, 2 clicks):**

1. **Dashboard → New → Blueprint**.
2. Pick **`jyothir-369/ticket-triage-agent`**.
3. Render parses `render.yaml`, sees two web services
   (`support-triage-api` + `support-triage-dashboard`), and displays the
   "New Blueprint Instance" page with a form.
4. Fill in the **sync: false** secret fields (see Step 4) and click **Apply**.

**Option B — Manually:**

1. **Dashboard → New → Web Service**.
2. Choose the `ticket-triage-agent` repo, set **Runtime: Docker**,
   **Dockerfile: `./Dockerfile`** → create **`support-triage-api`**.
3. Repeat with **Dockerfile: `./Dockerfile.streamlit`** → create
   **`support-triage-dashboard`**.
4. In each service, **Environment → Add Environment Variable** for the
   variables in [`.env.production.example`](.env.production.example).
5. Set **Health Check Path** to `/health` (API) and `/_stcore/health`
   (dashboard).

> The app name you choose in Render becomes part of the URL:
> `https://<app-name>.onrender.com`. The values above assume the default
> names from `render.yaml`. If you rename them, adjust every URL in this doc.

---

## 4. Step 4 — Set the secret env vars in Render

These are marked `sync: false` so the render.yaml contains **no** real
credentials. Paste real values either in the Blueprint form, or later under
**each service → Environment**.

Both services need a **matching** `DASHBOARD_PASSWORD`.

**`support-triage-api`:**

| Variable | Example value | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql://user:pw@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require` | Your **Neon** connection string |
| `REDIS_URL` | `rediss://default:pw@xxx.upstash.io:6379` | **Upstash** — note `rediss://` (TLS) |
| `QDRANT_URL` | `https://xxx.qdrant.io:6333` | **Qdrant Cloud** URL |
| `QDRANT_API_KEY` | `xxx` | From Qdrant Cloud dashboard |
| `GEMINI_API_KEY` | `AIza…` | From https://aistudio.google.com/apikey |
| `OPENROUTER_API_KEY` | `sk-or-…` | From OpenRouter |
| `DASHBOARD_PASSWORD` | `pick-a-strong-password` | Same as dashboard service → see below |
| *(non-secret, pre-set in render.yaml)* | — | `GEMINI_MODEL`, `OPENROUTER_*`, `LLM_*`, `PORT`, `ENVIRONMENT` |

**`support-triage-dashboard`:**

| Variable | Value | Notes |
|---|---|---|
| `DASHBOARD_PASSWORD` | *must match the API service value* | Used by the Streamlit login |
| `API_URL` | `https://support-triage-api.onrender.com` | Pre-set in render.yaml |
| `DASHBOARD_USERNAME` | `admin` | Pre-set in render.yaml |

---

## 5. Step 5 — Database schema

**No action required for the first deploy.**

This project manages schema with **`Base.metadata.create_all`**, which the API
runs automatically in its startup lifespan (`src/api/main.py`) — it creates any
tables that don't exist yet and is a no-op afterward. The checked-in **alembic
scaffolding has no authored migrations** (`alembic/versions/` is empty), so no
`alembic upgrade head` step is needed. On first start against the deployed
Neon DB, all tables are created for you; every later start skips already-created
tables.

**If you later add real alembic migrations** (a `versions/` file), apply them
either of these ways:

```yaml
# Option A — render.yaml: run before each deploy (fails the deploy loudly)
preDeployCommand: alembic upgrade head
```

```bash
# Option B — run once, locally, against the PRODUCTION database:
set -a; source .env.production; set +a
alembic upgrade head
```

> `alembic` is already a runtime dependency and present in the image either way.

---

## 6. Step 6 — Seed Qdrant with initial documents

After `/health` shows Qdrant green (collection exists), index the example
tickets into the **deployed** Qdrant collection:

```bash
# From your local checkout, pointing at the PRODUCTION Qdrant Cloud:
set -a; source .env.production; set +a
python -m scripts.ingest_tickets --csv data/tickets.csv
```

See `python -m scripts.ingest_tickets --help` for input options (CSV or GitHub
Issues) and `--dry-run` to validate before writing. The script is idempotent —
tickets already indexed are skipped.

> If you skip seeding, triage still works — retrieval just returns less context
> and drafts lean more on the LLM's trained knowledge. The `/health` Qdrant
> check reports the collection as `degraded` (exists vs. not) either way.

---

## 7. Step 7 — Verify `/health`

```bash
curl -s https://support-triage-api.onrender.com/health
```

Expected shape (first request can take ~30 s if the service was asleep):

```json
{
  "status": "healthy",
  "timestamp": "...",
  "version": "0.1.0",
  "environment": "production",
  "checks": {
    "database": { "status": "healthy", "latency_ms": 12.3 },
    "qdrant":   { "status": "healthy", "latency_ms": 40.1 },
    "redis":    { "status": "healthy", "latency_ms": 8.2 },
    "llm":      { "status": "healthy", "latency_ms": 350.0 }
  }
}
```

The LLM probe result is cached for 60 s so Render's health polls don't burn
Gemini/OpenRouter quota (~2880 probes/month would otherwise be LLM calls).

- All four checks **healthy** → ready.
- **degraded** Qdrant → collection not seeded yet (see Step 6).
- **unhealthy** / **status: unhealthy** → check the logs under
  **Logs → Runtime** and re-validate the env vars.

---

## 8. Step 8 — Log in to the dashboard

1. Open `https://support-triage-dashboard.onrender.com`.
2. Log in with the `DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD` you set
   in Step 4.
3. Create a ticket, watch the trace (classify → retrieve → draft →
   escalate_check → finalize), and review the metrics page.

---

## Free Tier Notes

- Render **free** web services **sleep after 15 min** of inactivity. The first
  request after sleep takes ~30 s to wake up; browsers may need one refresh.
- Free tier gives **750 instance-hours/month** across **all** services on an
  account. Two always-on services burn ~1440 h/month, so they will *spin down*
  (or Render will refuse new deploys) unless the account has free-tier hours
  remaining or paid capacity.
- To keep services warm (optional): hit `/health` every ~14 min with
  [UptimeRobot](https://uptimerobot.com) (free) or any cron, e.g.:
  ```bash
  # on any always-on machine / GitHub Action cron:
  curl -fsS https://support-triage-api.onrender.com/health > /dev/null
  ```
  The API pings in turn wake the dashboard's `API_URL` path. A simple GET to
  the dashboard URL also works.
- Free-tier CPU is limited (~0.1 vCPU). Single uvicorn worker is intentional
  for the API (see `WORKERS` env var in `Dockerfile`); raise it to 2 only on a
  paid (512 MB+) plan.
- The API image installs **CPU-only PyTorch** (Render has no GPU) and **omits
  the dev extras** (pytest, ruff, mypy, black, pre-commit, datasets) — this
  shrinks the image from ~11 GB to ~3 GB and keeps free-tier builds fast.
  `alembic` (needed by `preDeployCommand`) is a runtime dependency and is
  included.

---

## CI / CD (automatic deploys)

`.github/workflows/ci.yml` includes a `deploy` job that runs **only on pushes
to `main`** (not PRs) after `lint` → `test` → `eval`. It POSTs to Render's
deploy hooks:

1. In Render: export a **Deploy Hook URL** per service
   (**Settings → Deploy Hooks**).
2. In GitHub: add repo secrets
   - `RENDER_DEPLOY_HOOK_API`
   - `RENDER_DEPLOY_HOOK_DASHBOARD`
3. Done — every merge to `main` redeploys both services.

Until those two secrets are configured, the `deploy` job logs a warning and
skips (it does not fail CI).