"""LangGraph agent nodes — each node reads/writes AgentState.

Every node:
  1. Wraps its body in a try/except so a transient failure never breaks the graph.
  2. Appends a :class:`TraceStep` to ``state["trace"]`` on every path.
  3. Logs with the ticket's ``correlation_id`` (the ticket id) for distributed tracing.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import structlog

from src.agent.state import AgentState
from src.config import get_settings
from src.models.schemas import (
    DraftResponse,
    EscalationDecision,
    StepStatus,
    Ticket,
    TicketClassification,
    TraceStep,
    TriageTrace,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _ensure_trace(state: AgentState) -> TriageTrace:
    """Return the existing trace or create a minimal one bound to the ticket."""
    if "trace" in state and state["trace"] is not None:
        return state["trace"]
    ticket = state.get("ticket")
    ticket_id = ticket.id if ticket else "unknown"
    trace = TriageTrace(ticket_id=ticket_id)
    state["trace"] = trace
    return trace


def _append_step(
    state: AgentState,
    *,
    step_name: str,
    status: StepStatus,
    duration_ms: int | None = None,
    data: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Create a :class:`TraceStep` and append it to the trace."""
    trace = _ensure_trace(state)
    trace.add_step(
        TraceStep(
            step=step_name,
            status=status,
            duration_ms=duration_ms,
            data=data or {},
            error=error,
        )
    )


def _ticket_id(state: AgentState) -> str:
    """Shorthand to get the ticket id for logging / correlation."""
    ticket = state.get("ticket")
    return str(ticket.id) if ticket else "unknown"


def _now_ms() -> int:
    """Return the current wall-clock time in milliseconds."""
    return int(time.monotonic() * 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. classify_node
# ═══════════════════════════════════════════════════════════════════════════════


async def classify_node(state: AgentState) -> dict:
    """Classify the ticket into a category and urgency level.

    Calls :func:`src.services.classification.get_classifier`, updates
    ``state["classification"]`` and appends a trace step.

    **Graceful degradation**: if the LLM fails, the classifier's built-in
    keyword-based heuristic fallback is used automatically, returning a
    classification with confidence < 0.5.  This low confidence will trigger
    escalation in the escalate_check node.  Logs the degradation event.
    """
    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id)
    start = _now_ms()

    try:
        log.info("node.classify.start")
        _append_step(state, step_name="classify", status=StepStatus.RUNNING)

        ticket: Ticket = state["ticket"]
        from src.services.classification import get_classifier

        classifier = get_classifier()
        result: TicketClassification = await classifier.classify(ticket.content)

        elapsed = _now_ms() - start

        # Track whether heuristic fallback was used (confidence < 0.5 indicates fallback)
        is_degraded = result.confidence < 0.5
        degradation_reason = ""
        if is_degraded:
            degradation_reason = "LLM unavailable — used keyword heuristic fallback"
            log.warning(
                "node.classify.degraded_heuristic",
                category=result.category.value,
                urgency=result.urgency.value,
                confidence=round(result.confidence, 4),
                reason=degradation_reason,
                duration_ms=elapsed,
            )
        else:
            log.info(
                "node.classify.success",
                category=result.category.value,
                urgency=result.urgency.value,
                confidence=round(result.confidence, 4),
                duration_ms=elapsed,
            )

        _append_step(
            state,
            step_name="classify",
            status=StepStatus.COMPLETED,
            duration_ms=elapsed,
            data={
                **result.to_dict(),
                "degraded": is_degraded,
                "degradation_reason": degradation_reason,
            },
        )

        return {
            "classification": result,
            "trace": state["trace"],
        }

    except Exception as exc:
        elapsed = _now_ms() - start
        loop_count = state.get("loop_count", 0) + 1
        log.warning(
            "node.classify.failed",
            error=str(exc),
            error_type=type(exc).__name__,
            loop_count=loop_count,
            duration_ms=elapsed,
        )

        _append_step(
            state,
            step_name="classify",
            status=StepStatus.FAILED,
            duration_ms=elapsed,
            error=str(exc),
        )

        should_escalate = loop_count > settings.max_loop_retries

        return {
            "loop_count": loop_count,
            "error_message": str(exc),
            "should_escalate": should_escalate,
            "trace": state["trace"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. retrieve_node
# ═══════════════════════════════════════════════════════════════════════════════


async def retrieve_node(state: AgentState) -> dict:
    """Search Qdrant for semantically similar historical tickets.

    Uses the ticket content (and optional classification filters) to
    retrieve relevant context documents.  Results are serialised into
    ``state["retrieved_docs"]``.

    **Graceful degradation**: if Qdrant is unavailable (circuit breaker open,
    timeout, connection error), returns an empty document list so the pipeline
    continues with classification + template draft only.  Logs the degradation
    event for monitoring.
    """
    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id)
    start = _now_ms()

    try:
        log.info("node.retrieve.start")
        _append_step(state, step_name="retrieve", status=StepStatus.RUNNING)

        ticket: Ticket = state["ticket"]
        classification = state.get("classification")

        from src.services.retrieval import get_retriever

        retriever = get_retriever()

        # Build optional Qdrant filters from classification
        filters: dict[str, Any] | None = None
        if classification is not None:
            filters = {
                "category": classification.category.value,
                "urgency": classification.urgency.value,
            }

        docs = await retriever.search(
            ticket.content,
            limit=5,
            filters=filters,
            min_score=0.5,
        )

        # Serialise for downstream consumption
        doc_dicts = [
            {
                "id": d.id,
                "content": d.content,
                "metadata": d.metadata,
                "similarity_score": d.similarity_score,
                "source": d.source,
            }
            for d in docs
        ]

        elapsed = _now_ms() - start

        if not doc_dicts:
            # Graceful degradation: Qdrant unavailable or returned no results
            log.warning(
                "node.retrieve.degraded_no_results",
                duration_ms=elapsed,
                reason="Qdrant returned empty results — continuing without RAG context",
            )
            _append_step(
                state,
                step_name="retrieve",
                status=StepStatus.COMPLETED,
                duration_ms=elapsed,
                data={
                    "retrieval_count": 0,
                    "degraded": True,
                    "degradation_reason": "qdrant_unavailable",
                },
            )
        else:
            log.info(
                "node.retrieve.success",
                count=len(doc_dicts),
                duration_ms=elapsed,
            )
            _append_step(
                state,
                step_name="retrieve",
                status=StepStatus.COMPLETED,
                duration_ms=elapsed,
                data={"retrieval_count": len(doc_dicts)},
            )

        return {
            "retrieved_docs": doc_dicts,
            "trace": state["trace"],
        }

    except Exception as exc:
        elapsed = _now_ms() - start
        loop_count = state.get("loop_count", 0) + 1

        # Graceful degradation: log and continue without RAG context
        log.warning(
            "node.retrieve.degraded_error",
            error=str(exc),
            error_type=type(exc).__name__,
            loop_count=loop_count,
            duration_ms=elapsed,
            reason="Qdrant failed — continuing without RAG context",
        )

        _append_step(
            state,
            step_name="retrieve",
            status=StepStatus.COMPLETED,  # Mark as completed (degraded) not failed
            duration_ms=elapsed,
            data={
                "retrieval_count": 0,
                "degraded": True,
                "degradation_reason": str(exc),
            },
        )

        # Do NOT escalate — graceful degradation means we continue without RAG
        return {
            "retrieved_docs": [],
            "loop_count": loop_count,
            "error_message": "",
            "trace": state["trace"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. draft_node
# ═══════════════════════════════════════════════════════════════════════════════


async def draft_node(state: AgentState) -> dict:
    """Generate a draft response using the LLM with retrieved context.

    Calls :class:`src.services.drafting.DraftGenerator`, validates
    citations, and stores the result in ``state["draft"]``.  Falls back
    to a simple template-based draft when the LLM call fails.
    """
    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id)
    start = _now_ms()

    try:
        log.info("node.draft.start")
        _append_step(state, step_name="draft", status=StepStatus.RUNNING)

        ticket: Ticket = state["ticket"]
        classification = state.get("classification")
        raw_docs = state.get("retrieved_docs", [])

        from src.models.schemas import RetrievedDocument

        # Convert serialised dicts back to schema objects
        retrieved_docs: list[RetrievedDocument] = [
            RetrievedDocument(
                id=d.get("id", ""),
                content=d.get("content", ""),
                metadata=d.get("metadata", {}),
                similarity_score=d.get("similarity_score", 0.0),
                source=d.get("source", "unknown"),
            )
            for d in raw_docs
        ]

        # Fall back to a minimal classification when none is available
        if classification is None:
            from src.models.schemas import TicketCategory, UrgencyLevel

            classification = TicketClassification(
                category=TicketCategory.OTHER,
                urgency=UrgencyLevel.MEDIUM,
                confidence=0.5,
                reasoning="No classification available.",
            )

        from src.services.drafting import get_draft_generator

        generator = get_draft_generator()
        draft: DraftResponse = await generator.generate(
            ticket, classification, retrieved_docs
        )

        elapsed = _now_ms() - start
        citation_count = len(draft.citations)

        log.info(
            "node.draft.success",
            draft_length=len(draft.draft_text),
            citation_count=citation_count,
            confidence=round(draft.confidence, 4),
            citations_validated=draft.citations_validated,
            duration_ms=elapsed,
        )

        _append_step(
            state,
            step_name="draft",
            status=StepStatus.COMPLETED,
            duration_ms=elapsed,
            data={
                "draft_length": len(draft.draft_text),
                "citation_count": citation_count,
                "confidence": round(draft.confidence, 4),
            },
        )

        return {
            "draft": draft,
            "trace": state["trace"],
        }

    except Exception as exc:
        elapsed = _now_ms() - start
        log.warning(
            "node.draft.llm_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            duration_ms=elapsed,
        )

        # Fallback: generate a simple template-based draft
        fallback = _template_fallback_draft(state)
        _append_step(
            state,
            step_name="draft",
            status=StepStatus.COMPLETED,
            duration_ms=elapsed,
            data={
                "draft_length": len(fallback.draft_text),
                "citation_count": 0,
                "fallback": True,
                "error": str(exc),
            },
        )

        return {
            "draft": fallback,
            "trace": state["trace"],
        }


def _template_fallback_draft(state: AgentState) -> DraftResponse:
    """Build a minimal template-based draft when the LLM is unavailable."""
    ticket = state.get("ticket")
    classification = state.get("classification")

    category_label = (
        classification.category.value.replace("_", " ").title()
        if classification
        else "your issue"
    )
    urgency_note = (
        f" (urgency: {classification.urgency.value})"
        if classification
        else ""
    )

    draft_text = (
        f"Thank you for reaching out. We've received your {category_label} request{urgency_note} "
        f"and our support team is reviewing it. We'll get back to you as soon as possible.\n\n"
        f"If you have additional details to share, please reply to this message."
    )

    return DraftResponse(
        draft_text=draft_text,
        citations=[],
        confidence=0.3,
        reasoning="Template fallback — LLM drafting service unavailable.",
        citations_validated=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. escalate_check_node
# ═══════════════════════════════════════════════════════════════════════════════


async def escalate_check_node(state: AgentState) -> dict:
    """Decide whether the ticket should be escalated to a human reviewer.

    Combines classification confidence and draft confidence into a single
    score, then compares it against the configured threshold.  Also
    triggers escalation when the loop budget is exhausted.
    """
    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id)
    start = _now_ms()

    try:
        log.info("node.escalate_check.start")
        _append_step(state, step_name="escalate_check", status=StepStatus.RUNNING)

        classification = state.get("classification")
        draft = state.get("draft")
        loop_count = state.get("loop_count", 0)

        # Compute final confidence: weighted average of classification and
        # draft scores.  When one is missing, fall back to the other or 0.
        class_conf = classification.confidence if classification else 0.0
        draft_conf = draft.confidence if draft else 0.0

        if classification is not None and draft is not None:
            final_confidence = (class_conf * 0.6) + (draft_conf * 0.4)
        elif classification is not None:
            final_confidence = class_conf
        elif draft is not None:
            final_confidence = draft_conf
        else:
            final_confidence = 0.0

        threshold = settings.confidence_threshold
        max_retries = settings.max_loop_retries

        # Determine escalation
        reasons: list[str] = []
        should_escalate = False

        if final_confidence < threshold:
            reasons.append(
                f"Final confidence {final_confidence:.4f} is below threshold {threshold}"
            )
            should_escalate = True

        if loop_count > max_retries:
            reasons.append(
                f"Loop count {loop_count} exceeded max retries {max_retries}"
            )
            should_escalate = True

        if classification is None:
            reasons.append("Classification is missing")
            should_escalate = True

        if draft is None:
            reasons.append("Draft response is missing")
            should_escalate = True

        escalation_reason = "; ".join(reasons) if reasons else "Confidence meets threshold"

        # Build the EscalationDecision
        escalation = EscalationDecision(
            should_escalate=should_escalate,
            reason=escalation_reason,
            confidence_score=round(final_confidence, 4),
            threshold_used=threshold,
            human_review_required=should_escalate,
        )

        elapsed = _now_ms() - start
        log.info(
            "node.escalate_check.complete",
            should_escalate=should_escalate,
            final_confidence=round(final_confidence, 4),
            classification_confidence=round(class_conf, 4),
            draft_confidence=round(draft_conf, 4),
            threshold=threshold,
            loop_count=loop_count,
            duration_ms=elapsed,
        )

        _append_step(
            state,
            step_name="escalate_check",
            status=StepStatus.COMPLETED,
            duration_ms=elapsed,
            data={
                "should_escalate": should_escalate,
                "final_confidence": round(final_confidence, 4),
                "classification_confidence": round(class_conf, 4),
                "draft_confidence": round(draft_conf, 4),
                "threshold": threshold,
                "loop_count": loop_count,
                "reasons": reasons,
            },
        )

        return {
            "escalation": escalation,
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
            "trace": state["trace"],
        }

    except Exception as exc:
        elapsed = _now_ms() - start
        log.warning(
            "node.escalate_check.failed",
            error=str(exc),
            duration_ms=elapsed,
        )

        _append_step(
            state,
            step_name="escalate_check",
            status=StepStatus.FAILED,
            duration_ms=elapsed,
            error=str(exc),
        )

        # On failure, default to escalation for safety
        fallback = EscalationDecision(
            should_escalate=True,
            reason=f"Escalation check failed: {exc}",
            confidence_score=0.0,
            threshold_used=settings.confidence_threshold,
        )
        return {
            "escalation": fallback,
            "should_escalate": True,
            "escalation_reason": str(exc),
            "trace": state["trace"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. escalate_node
# ═══════════════════════════════════════════════════════════════════════════════


async def escalate_node(state: AgentState) -> dict:
    """Persist the escalation decision and update the ticket in the database.

    Writes the full :class:`EscalationDecision`, updates the ticket's
    lifecycle status to ``ESCALATED``, and saves the trace.
    """
    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id)
    start = _now_ms()

    try:
        log.info("node.escalate.start")
        _append_step(state, step_name="escalate", status=StepStatus.RUNNING)

        escalation: EscalationDecision | None = state.get("escalation")
        if escalation is None:
            escalation = EscalationDecision(
                should_escalate=True,
                reason="No escalation decision was computed.",
            )

        from src.repository import TicketRepository

        repo = TicketRepository()
        trace = _ensure_trace(state)

        elapsed = _now_ms() - start
        log.info(
            "node.escalate.completed_in_memory",
            reason=escalation.reason,
            confidence_score=escalation.confidence_score,
            duration_ms=elapsed,
        )

        # Mark escalate COMPLETED *before* persisting so the saved trace JSON
        # contains the terminal step (previously the trace was written while
        # escalate was still "running"). finalize_node runs right after this
        # node (graph edge node_escalate → node_finalize) and persists the
        # ticket status, so we no longer write a trace_steps row here.
        _append_step(
            state,
            step_name="escalate",
            status=StepStatus.COMPLETED,
            duration_ms=elapsed,
            data={
                "escalated": True,
                "reason": escalation.reason,
                "confidence_score": escalation.confidence_score,
            },
        )

        # Save full trace JSON on the ticket row
        await repo.update_ticket(
            ticket_id,
            {
                "trace": json.dumps(trace.to_dict(), default=str),
                "escalation_reason": escalation.reason,
                "loop_count": state.get("loop_count", 0),
            },
        )

        return {
            "trace": state["trace"],
        }

    except Exception as exc:
        elapsed = _now_ms() - start
        log.warning(
            "node.escalate.failed",
            error=str(exc),
            duration_ms=elapsed,
        )

        _append_step(
            state,
            step_name="escalate",
            status=StepStatus.FAILED,
            duration_ms=elapsed,
            error=str(exc),
        )

        return {
            "error_message": str(exc),
            "trace": state["trace"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. finalize_node
# ═══════════════════════════════════════════════════════════════════════════════


async def finalize_node(state: AgentState) -> dict:
    """Save the complete triage result and finalise the ticket status.

    Persists classification, draft, and escalation to the database, sets
    the ticket status to ``RESOLVED`` or ``ESCALATED``, and writes the
    full trace.
    """
    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id)
    start = _now_ms()

    try:
        log.info("node.finalize.start")
        _append_step(state, step_name="finalize", status=StepStatus.RUNNING)

        classification = state.get("classification")
        draft = state.get("draft")
        escalation = state.get("escalation")
        should_escalate = state.get("should_escalate", False)

        from src.models.schemas import TicketStatus
        from src.repository import TicketRepository

        repo = TicketRepository()
        trace = _ensure_trace(state)

        # Build dicts for save_triage_result
        classification_dict = classification.to_dict() if classification else {
            "category": "other",
            "urgency": "medium",
            "confidence": 0.0,
        }
        draft_dict = draft.to_dict() if draft else {
            "draft_text": "",
            "citations": [],
            "confidence": 0.0,
        }
        escalation_dict = escalation.to_dict() if escalation else {
            "should_escalate": should_escalate,
            "reason": state.get("escalation_reason", ""),
        }

        await repo.save_triage_result(
            ticket_id,
            classification=classification_dict,
            draft=draft_dict,
            escalation=escalation_dict,
        )

        # Determine final status
        final_status = (
            TicketStatus.ESCALATED if should_escalate else TicketStatus.RESOLVED
        )

        # Compute total duration from trace
        trace_total = 0
        for step in trace.steps:
            if step.duration_ms is not None:
                trace_total += step.duration_ms
        elapsed = _now_ms() - start
        total_ms = trace_total + elapsed

        # Update final trace state
        trace.final_status = final_status
        trace.total_duration_ms = total_ms
        trace.loop_count = state.get("loop_count", 0)
        if should_escalate:
            trace.loop_detected = state.get("loop_count", 0) > settings.max_loop_retries

        # Mark finalize COMPLETED *before* persisting so the saved trace JSON
        # contains the terminal step (previously the completed step was appended
        # after the DB write, leaving finalize stuck at "running").
        _append_step(
            state,
            step_name="finalize",
            status=StepStatus.COMPLETED,
            duration_ms=elapsed,
            data={
                "final_status": final_status.value,
                "total_duration_ms": total_ms,
                "loop_count": state.get("loop_count", 0),
                "should_escalate": should_escalate,
            },
        )

        # Save the final trace
        await repo.update_ticket(
            ticket_id,
            {
                "trace": json.dumps(trace.to_dict(), default=str),
                "status": final_status.value,
                "loop_count": state.get("loop_count", 0),
                "processed_at": datetime.now(UTC),
            },
        )

        log.info(
            "node.finalize.success",
            final_status=final_status.value,
            total_duration_ms=total_ms,
            loop_count=state.get("loop_count", 0),
        )

        return {
            "trace": state["trace"],
        }

    except Exception as exc:
        elapsed = _now_ms() - start
        log.warning(
            "node.finalize.failed",
            error=str(exc),
            duration_ms=elapsed,
        )

        _append_step(
            state,
            step_name="finalize",
            status=StepStatus.FAILED,
            duration_ms=elapsed,
            error=str(exc),
        )

        return {
            "error_message": str(exc),
            "trace": state["trace"],
        }
