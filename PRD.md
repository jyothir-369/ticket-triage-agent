
# PRD 1 — Support-Ticket Triage Agent

## Overview
An agent that classifies incoming support tickets, retrieves related past tickets/docs, drafts a response, and escalates ambiguous cases to a human.

## Problem Statement
Manual ticket triage is slow and inconsistent. An agent that reliably handles the easy 70–80% and cleanly escalates the rest proves you can own a real automatable business process end-to-end, not just a chatbot demo.

## Goals
- Achieve a measured task-success rate with a defined escalation path for low-confidence cases.
- Produce a full, auditable decision trace per ticket.
- Handle tool failures (e.g., a lookup API timing out) without silently dropping the ticket.

## Non-Goals
- Actually sending customer-facing replies unsupervised (always human-approved for v1).
- Multi-language support (English-only for v1, documented as a limitation).

## Users & Personas
| Persona | Needs |
|---|---|
| Support agent | Reviews agent-drafted responses, approves/edits/escalates |
| Team lead | Views triage accuracy and escalation-rate dashboards |

## Tech Stack
| Layer | Choice | Notes |
|---|---|---|
| Language/runtime | Python 3.11+ | Matches your existing LangGraph/RAG repos |
| Orchestration | LangGraph | Planner/executor graph with an explicit escalation-gate node |
| Backend API | FastAPI | Consistent with `production-grade-rag-engine` |
| LLM provider | OpenAI or Anthropic API, behind a thin provider adapter | Swappable without touching agent logic |
| Retrieval / vector DB | Qdrant | Reuses your existing RAG-engine ingestion pipeline for ticket-search |
| Data store | PostgreSQL | Tickets, traces, escalation state |
| Async/background jobs | Inngest (or Celery + Redis) | Tool-call retries, background classification |
| Observability & eval | OpenTelemetry traces + a custom eval harness (task-success rate, escalation accuracy) | Purpose-built, not RAGAS (this is agent-behavior eval, not retrieval eval) |
| CI/CD | GitHub Actions | Runs the labeled-ticket eval set on every PR |
| Testing | Pytest + a labeled 50–100 ticket eval fixture | Eval set is checked into the repo, not external |
| Deployment | Docker + a lightweight review UI (Streamlit or a small Next.js app) | Human-approval queue for drafted responses |

## Functional Requirements
- FR1: Ingest a ticket (public dataset or your own inbox/GitHub Issues) and classify category + urgency.
- FR2: Retrieve related past tickets/docs via RAG before drafting a response.
- FR3: Draft a response with citations to retrieved sources.
- FR4: Route to human review when confidence is below a tunable threshold.
- FR5: Persist a full step-by-step trace (classification → retrieval → draft → routing decision) per ticket.
- FR6: Detect and break out of agent loops (e.g., repeated failed tool calls) rather than looping indefinitely.

## Non-Functional Requirements
| Category | Requirement |
|---|---|
| Reliability | Tool-call failures trigger retry with backoff, then graceful escalation |
| Auditability | Every decision traceable end-to-end with timestamps |
| Latency | p95 triage time under a defined budget (e.g., 30s) |
| Cost | Per-ticket cost tracked and reported |

## Architecture Notes
Planner/executor split (or single ReAct loop for v1) using LangGraph; tools: ticket-search (your RAG index), classification call, draft-generation call; state persisted in Postgres; escalation gate as an explicit graph node, not a prompt instruction alone.

## API Surface (representative)
```
POST /tickets/:id/triage
GET  /tickets/:id/trace
GET  /dashboard/triage-metrics
```

## Milestones
1. Single-step classifier only, no retrieval.
2. Add RAG retrieval + draft generation.
3. Add confidence-based escalation + human review UI.
4. Add loop detection, failure handling, and full tracing.
5. Load test + report task-success rate on a labeled test set.

## Success Metrics
- Task success rate (%) on a held-out labeled set of tickets.
- Escalation rate and false-escalation rate (easy tickets wrongly escalated).
- Mean/median steps to completion.

## Risks
Labeled test-set construction is manual work — budget real time for it; a small (50–100 ticket) hand-labeled set is enough to be credible.