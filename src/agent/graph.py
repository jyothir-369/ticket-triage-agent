"""LangGraph agent — Planner/Executor with an explicit escalation gate.

The graph has six nodes:
  1. classify        – categorize the ticket + assign urgency
  2. retrieve        – search Qdrant for related past tickets
  3. draft           – generate a suggested response
  4. escalate_check  – decide whether to route to a human or mark complete
  5. escalate        – persist the escalation decision
  6. finalize        – save the complete triage result

Edges:
  START → classify → retrieve → draft → escalate_check
  escalate_check → escalate (when should_escalate is True)
  escalate_check → finalize (when should_escalate is False)
  escalate → END
  finalize → END

Loop detection: the agent tracks ``tool_call_count`` in state and routes
directly to escalate when > 3 failed tool calls.

Retry logic: each node is wrapped with tenacity exponential backoff
(1 s → 2 s → 4 s) for transient failures.

Persistence: the graph is compiled with a ``PostgresSaver`` checkpointer
for full state persistence across restarts.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Retry decorator with exponential backoff
# ═══════════════════════════════════════════════════════════════════════════════


def _with_retry(node_fn):
    """Wrap an async node function with tenacity exponential backoff.

    Retries up to 3 times on ``Exception`` with waits of 1 s, 2 s, 4 s.
    On each failure the loop counter is incremented; when the retry budget
    is exhausted the error is propagated so the error handler can set
    ``should_escalate``.

    Parameters
    ----------
    node_fn:
        An ``async def(state) -> dict`` node function.

    Returns
    -------
    Callable
        The wrapped function with retry logic.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _retry_wrapper(state):
        return await node_fn(state)

    _retry_wrapper.__name__ = node_fn.__name__
    _retry_wrapper.__qualname__ = node_fn.__qualname__
    _retry_wrapper.__doc__ = node_fn.__doc__
    return _retry_wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Node error handler — catches failures after retries are exhausted
# ═══════════════════════════════════════════════════════════════════════════════


async def _handle_node_error(
    state: "AgentState",
    node_name: str,
    exc: Exception,
) -> dict:
    """Update state after a node fails: increment loop/tool counters and
    escalate when the retry budget is exhausted.

    Parameters
    ----------
    state:
        Current agent state.
    node_name:
        Human-readable label for logging (e.g. ``"classify"``).
    exc:
        The exception that caused the failure.

    Returns
    -------
    dict
        Partial state update to merge into ``AgentState``.
    """
    from src.agent.state import AgentState  # noqa: F811 — local import for type

    ticket_id = state.get("ticket")
    ticket_id_str = str(ticket_id.id) if ticket_id else "unknown"

    new_loop_count = state.get("loop_count", 0) + 1
    new_tool_call_count = state.get("tool_call_count", 0) + 1

    logger.warning(
        "graph.node_failed",
        node=node_name,
        ticket_id=ticket_id_str,
        error=str(exc),
        error_type=type(exc).__name__,
        loop_count=new_loop_count,
        tool_call_count=new_tool_call_count,
    )

    should_escalate = (
        new_loop_count > settings.max_loop_retries
        or new_tool_call_count > 3
    )

    return {
        "loop_count": new_loop_count,
        "tool_call_count": new_tool_call_count,
        "error_message": str(exc),
        "should_escalate": should_escalate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node functions — each reads/writes AgentState
# ═══════════════════════════════════════════════════════════════════════════════


async def classify_node(state: "AgentState") -> dict:
    """Classify the ticket into a category and urgency level.

    Calls the classification service, updates ``state["classification"]``
    and appends a trace step.  On failure, delegates to the error handler
    which increments counters and may trigger escalation.
    """
    from src.agent.nodes import _append_step, _ensure_trace, _now_ms, _ticket_id
    from src.agent.state import AgentState  # noqa: F811 — for type checking
    from src.models.schemas import StepStatus, TicketClassification

    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id, node="classify")
    start = _now_ms()

    try:
        log.info("graph.classify.start")
        _append_step(state, step_name="classify", status=StepStatus.RUNNING)

        ticket = state["ticket"]
        from src.services.classification import get_classifier

        classifier = get_classifier()
        result: TicketClassification = await classifier.classify(ticket.content)

        elapsed = _now_ms() - start
        log.info(
            "graph.classify.success",
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
            data=result.to_dict(),
        )

        return {
            "classification": result,
            "trace": state["trace"],
        }

    except Exception as exc:
        elapsed = _now_ms() - start
        _append_step(
            state,
            step_name="classify",
            status=StepStatus.FAILED,
            duration_ms=elapsed,
            error=str(exc),
        )
        raise  # let _with_retry re-raise; error handler runs in graph


async def retrieve_node(state: "AgentState") -> dict:
    """Search Qdrant for semantically similar historical tickets.

    Uses the ticket content (and optional classification filters) to
    retrieve relevant context documents.  Results are serialised into
    ``state["retrieved_docs"]``.
    """
    from src.agent.nodes import _append_step, _now_ms, _ticket_id
    from src.agent.state import AgentState  # noqa: F811

    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id, node="retrieve")
    start = _now_ms()

    try:
        log.info("graph.retrieve.start")
        _append_step(state, step_name="retrieve", status=StepStatus.RUNNING)

        ticket = state["ticket"]
        classification = state.get("classification")

        from src.models.schemas import StepStatus
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
        log.info(
            "graph.retrieve.success",
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
        _append_step(
            state,
            step_name="retrieve",
            status=StepStatus.FAILED,
            duration_ms=elapsed,
            error=str(exc),
        )
        raise


async def draft_node(state: "AgentState") -> dict:
    """Generate a draft response using the LLM with retrieved context.

    Calls the draft generator, validates citations, and stores the result
    in ``state["draft"]``.  Falls back to a simple template-based draft
    when the LLM call fails.
    """
    from src.agent.nodes import (
        _append_step,
        _now_ms,
        _template_fallback_draft,
        _ticket_id,
    )
    from src.agent.state import AgentState  # noqa: F811

    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id, node="draft")
    start = _now_ms()

    try:
        log.info("graph.draft.start")
        _append_step(state, step_name="draft", status=StepStatus.RUNNING)

        ticket = state["ticket"]
        classification = state.get("classification")
        raw_docs = state.get("retrieved_docs", [])

        from src.models.schemas import RetrievedDocument, StepStatus, TicketCategory
        from src.models.schemas import UrgencyLevel
        from src.services.drafting import get_draft_generator

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
            from src.models.schemas import TicketClassification

            classification = TicketClassification(
                category=TicketCategory.OTHER,
                urgency=UrgencyLevel.MEDIUM,
                confidence=0.5,
                reasoning="No classification available.",
            )

        generator = get_draft_generator()
        draft = await generator.generate(ticket, classification, retrieved_docs)

        elapsed = _now_ms() - start
        citation_count = len(draft.citations)

        log.info(
            "graph.draft.success",
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
            "graph.draft.llm_failed",
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


async def escalate_check_node(state: "AgentState") -> dict:
    """Decide whether the ticket should be escalated to a human reviewer.

    Combines classification confidence and draft confidence into a single
    score, then compares it against the configured threshold.  Also
    triggers escalation when the loop budget is exhausted or
    ``tool_call_count > 3``.
    """
    from src.agent.nodes import _append_step, _ensure_trace, _now_ms, _ticket_id
    from src.agent.state import AgentState  # noqa: F811

    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id, node="escalate_check")
    start = _now_ms()

    try:
        log.info("graph.escalate_check.start")
        _append_step(state, step_name="escalate_check", status=StepStatus.RUNNING)

        classification = state.get("classification")
        draft = state.get("draft")
        loop_count = state.get("loop_count", 0)
        tool_call_count = state.get("tool_call_count", 0)

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

        if tool_call_count > 3:
            reasons.append(
                f"Tool call count {tool_call_count} exceeded threshold of 3"
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
        from src.models.schemas import EscalationDecision

        escalation = EscalationDecision(
            should_escalate=should_escalate,
            reason=escalation_reason,
            confidence_score=round(final_confidence, 4),
            threshold_used=threshold,
            human_review_required=should_escalate,
        )

        elapsed = _now_ms() - start
        log.info(
            "graph.escalate_check.complete",
            should_escalate=should_escalate,
            final_confidence=round(final_confidence, 4),
            classification_confidence=round(class_conf, 4),
            draft_confidence=round(draft_conf, 4),
            threshold=threshold,
            loop_count=loop_count,
            tool_call_count=tool_call_count,
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
                "tool_call_count": tool_call_count,
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
            "graph.escalate_check.failed",
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
        from src.models.schemas import EscalationDecision

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


async def escalate_node(state: "AgentState") -> dict:
    """Persist the escalation decision and update the ticket in the database.

    Writes the full :class:`EscalationDecision`, updates the ticket's
    lifecycle status to ``ESCALATED``, and saves the trace.
    """
    from src.agent.nodes import _append_step, _ensure_trace, _now_ms, _ticket_id
    from src.agent.state import AgentState  # noqa: F811

    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id, node="escalate")
    start = _now_ms()

    try:
        log.info("graph.escalate.start")
        _append_step(state, step_name="escalate", status=StepStatus.RUNNING)

        escalation = state.get("escalation")
        if escalation is None:
            from src.models.schemas import EscalationDecision

            escalation = EscalationDecision(
                should_escalate=True,
                reason="No escalation decision was computed.",
            )

        from src.models.schemas import StepStatus, TicketStatus
        from src.repository import TicketRepository

        repo = TicketRepository()
        trace = _ensure_trace(state)

        # Update ticket status to ESCALATED
        await repo.update_ticket_status(
            ticket_id,
            TicketStatus.ESCALATED.value,
            trace={
                "step": "escalate",
                "status": TicketStatus.ESCALATED.value,
                "data": escalation.to_dict(),
            },
        )

        # Save full trace JSON on the ticket row
        import json

        await repo.update_ticket(
            ticket_id,
            {
                "trace": json.dumps(trace.to_dict(), default=str),
                "escalation_reason": escalation.reason,
                "loop_count": state.get("loop_count", 0),
            },
        )

        elapsed = _now_ms() - start
        log.info(
            "graph.escalate.success",
            reason=escalation.reason,
            confidence_score=escalation.confidence_score,
            duration_ms=elapsed,
        )

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

        return {
            "trace": state["trace"],
        }

    except Exception as exc:
        elapsed = _now_ms() - start
        log.warning(
            "graph.escalate.failed",
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


async def finalize_node(state: "AgentState") -> dict:
    """Save the complete triage result and finalise the ticket status.

    Persists classification, draft, and escalation to the database, sets
    the ticket status to ``RESOLVED`` or ``ESCALATED``, and writes the
    full trace.
    """
    from src.agent.nodes import _append_step, _ensure_trace, _now_ms, _ticket_id
    from src.agent.state import AgentState  # noqa: F811

    ticket_id = _ticket_id(state)
    log = logger.bind(ticket_id=ticket_id, node="finalize")
    start = _now_ms()

    try:
        log.info("graph.finalize.start")
        _append_step(state, step_name="finalize", status=StepStatus.RUNNING)

        classification = state.get("classification")
        draft = state.get("draft")
        escalation = state.get("escalation")
        should_escalate = state.get("should_escalate", False)

        import json
        from datetime import datetime, timezone

        from src.models.schemas import StepStatus, TicketStatus
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

        # Save the final trace
        await repo.update_ticket(
            ticket_id,
            {
                "trace": json.dumps(trace.to_dict(), default=str),
                "status": final_status.value,
                "loop_count": state.get("loop_count", 0),
                "processed_at": datetime.now(timezone.utc),
            },
        )

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

        log.info(
            "graph.finalize.success",
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
            "graph.finalize.failed",
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


# ═══════════════════════════════════════════════════════════════════════════════
# Error handler node — catches failures after retries are exhausted
# ═══════════════════════════════════════════════════════════════════════════════


async def _error_handler_node(state: "AgentState", node_name: str, exc: Exception) -> dict:
    """Update state after a node fails: increment loop/tool counters and
    escalate when the retry budget is exhausted.

    Parameters
    ----------
    state:
        Current agent state.
    node_name:
        Human-readable label for logging (e.g. ``"classify"``).
    exc:
        The exception that caused the failure.

    Returns
    -------
    dict
        Partial state update to merge into ``AgentState``.
    """
    from src.agent.state import AgentState  # noqa: F811 — local import for type

    ticket_id = state.get("ticket")
    ticket_id_str = str(ticket_id.id) if ticket_id else "unknown"

    new_loop_count = state.get("loop_count", 0) + 1
    new_tool_call_count = state.get("tool_call_count", 0) + 1

    logger.warning(
        "graph.node_failed",
        node=node_name,
        ticket_id=ticket_id_str,
        error=str(exc),
        error_type=type(exc).__name__,
        loop_count=new_loop_count,
        tool_call_count=new_tool_call_count,
    )

    should_escalate = (
        new_loop_count > settings.max_loop_retries
        or new_tool_call_count > 3
    )

    return {
        "loop_count": new_loop_count,
        "tool_call_count": new_tool_call_count,
        "error_message": str(exc),
        "should_escalate": should_escalate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node wrappers — retry + error handling
# ═══════════════════════════════════════════════════════════════════════════════


async def _classify_wrapped(state: "AgentState") -> dict:
    """classify_node with retry logic and error handling."""
    try:
        return await _with_retry(classify_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "classify", exc)


async def _retrieve_wrapped(state: "AgentState") -> dict:
    """retrieve_node with retry logic and error handling."""
    try:
        return await _with_retry(retrieve_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "retrieve", exc)


async def _draft_wrapped(state: "AgentState") -> dict:
    """draft_node with retry logic and error handling."""
    try:
        return await _with_retry(draft_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "draft", exc)


async def _escalate_check_wrapped(state: "AgentState") -> dict:
    """escalate_check_node with retry logic and error handling."""
    try:
        return await _with_retry(escalate_check_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "escalate_check", exc)


async def _escalate_wrapped(state: "AgentState") -> dict:
    """escalate_node with retry logic and error handling."""
    try:
        return await _with_retry(escalate_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "escalate", exc)


async def _finalize_wrapped(state: "AgentState") -> dict:
    """finalize_node with retry logic and error handling."""
    try:
        return await _with_retry(finalize_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "finalize", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Routing logic
# ═══════════════════════════════════════════════════════════════════════════════


def route_after_escalation_check(state: dict) -> str:
    """Route after the escalation check.

    Returns ``"escalate"`` when ``state.should_escalate`` is ``True``,
    otherwise ``"finalize"``.

    Parameters
    ----------
    state:
        The current agent state dict.  Typed as ``dict`` rather than
        ``AgentState`` to avoid forward-reference issues with LangGraph's
        type introspection on conditional routing functions.
    """
    should_escalate = state.get("should_escalate", False)
    ticket = state.get("ticket")
    if ticket is None:
        ticket_id_str = "unknown"
    elif hasattr(ticket, "id"):
        ticket_id_str = str(ticket.id)
    elif isinstance(ticket, dict):
        ticket_id_str = str(ticket.get("id", "unknown"))
    else:
        ticket_id_str = str(ticket)

    logger.info(
        "graph.route_after_escalation_check",
        ticket_id=ticket_id_str,
        should_escalate=should_escalate,
    )

    if should_escalate:
        return "escalate"  # maps to "node_escalate" in add_conditional_edges
    return "finalize"  # maps to "node_finalize" in add_conditional_edges


# ═══════════════════════════════════════════════════════════════════════════════
# Graph definition
# ═══════════════════════════════════════════════════════════════════════════════


def build_triage_graph() -> StateGraph:
    """Construct and return the triage agent graph (not compiled).

    Nodes are wrapped with ``_with_retry`` for exponential backoff
    and error handling.
    """
    from src.agent.state import AgentState  # noqa: F811

    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    # Node names must not collide with AgentState keys (e.g. "draft", "classification").
    # We prefix with "node_" where needed to avoid conflicts.
    graph.add_node("node_classify", _classify_wrapped)
    graph.add_node("node_retrieve", _retrieve_wrapped)
    graph.add_node("node_draft", _draft_wrapped)
    graph.add_node("node_escalate_check", _escalate_check_wrapped)
    graph.add_node("node_escalate", _escalate_wrapped)
    graph.add_node("node_finalize", _finalize_wrapped)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("node_classify")

    graph.add_edge("node_classify", "node_retrieve")
    graph.add_edge("node_retrieve", "node_draft")
    graph.add_edge("node_draft", "node_escalate_check")

    # Conditional routing from escalate_check
    graph.add_conditional_edges(
        "node_escalate_check",
        route_after_escalation_check,
        {
            "escalate": "node_escalate",
            "finalize": "node_finalize",
        },
    )

    # Terminal edges
    graph.add_edge("node_escalate", END)
    graph.add_edge("node_finalize", END)

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointer — PostgresSaver for persistence
# ═══════════════════════════════════════════════════════════════════════════════


def _build_checkpointer():
    """Build and return a ``PostgresSaver`` checkpointer for graph persistence.

    Uses the synchronous ``psycopg2`` connection from the configured
    ``DATABASE_URL``.  Falls back to ``MemorySaver`` when the database
    is not available (e.g. during tests).
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        # Convert the async PostgreSQL URL to a synchronous psycopg2 URL
        db_url = settings.database_url
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

        saver = PostgresSaver.from_conn_string(sync_url)
        saver.setup()
        logger.info("graph.checkpointer.postgres_configured", url=sync_url.split("@")[-1])
        return saver

    except Exception as exc:
        logger.warning(
            "graph.checkpointer.fallback_memory",
            error=str(exc),
            reason="PostgresSaver unavailable — using MemorySaver",
        )
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


# ═══════════════════════════════════════════════════════════════════════════════
# Compiled graph
# ═══════════════════════════════════════════════════════════════════════════════

checkpointer = _build_checkpointer()

triage_agent = build_triage_graph().compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# AgentExecutor — high-level interface for running the triage pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class AgentExecutor:
    """High-level interface for running the triage pipeline.

    Provides ``run()`` for one-shot execution and ``stream()`` for
    real-time state updates as each node completes.

    Usage::

        executor = AgentExecutor()
        result = await executor.run(ticket)

        # Or stream updates:
        async for state in executor.stream(ticket):
            print(state)
    """

    def __init__(self, graph=None):
        """Initialise the executor with an optional compiled graph.

        Parameters
        ----------
        graph:
            A compiled LangGraph.  When ``None`` the module-level
            ``triage_agent`` is used.
        """
        self._graph = graph or triage_agent
        logger.debug("agent_executor.initialised")

    async def run(self, ticket: "Ticket") -> "AgentState":
        """Execute the triage graph for a single ticket and return the final state.

        Parameters
        ----------
        ticket:
            The incoming support ticket to triage.

        Returns
        -------
        AgentState
            The complete state after all nodes have executed.
        """
        from src.agent.state import AgentState  # noqa: F811
        from src.models.schemas import TriageTrace, Ticket as TicketSchema

        ticket_id_str = str(ticket.id)

        logger.info(
            "agent_executor.run.start",
            ticket_id=ticket_id_str,
            source=ticket.source,
            content_length=len(ticket.content),
        )

        initial_state: AgentState = {
            "ticket": ticket,
            "classification": None,
            "retrieved_docs": [],
            "draft": None,
            "escalation": None,
            "should_escalate": False,
            "escalation_reason": "",
            "trace": TriageTrace(ticket_id=ticket_id_str),
            "loop_count": 0,
            "tool_call_count": 0,
            "error_message": "",
            "messages": [],
        }

        start = time.monotonic()

        try:
            final_state = await self._graph.ainvoke(initial_state)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            logger.info(
                "agent_executor.run.complete",
                ticket_id=ticket_id_str,
                should_escalate=final_state.get("should_escalate", False),
                loop_count=final_state.get("loop_count", 0),
                tool_call_count=final_state.get("tool_call_count", 0),
                latency_ms=elapsed_ms,
            )

            return final_state

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "agent_executor.run.failed",
                ticket_id=ticket_id_str,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=elapsed_ms,
            )
            raise

    async def stream(self, ticket: "Ticket") -> AsyncIterator["AgentState"]:
        """Stream state updates as each node in the triage pipeline completes.

        Parameters
        ----------
        ticket:
            The incoming support ticket to triage.

        Yields
        ------
        AgentState
            The state after each node executes, in execution order.
        """
        from src.agent.state import AgentState  # noqa: F811
        from src.models.schemas import TriageTrace

        ticket_id_str = str(ticket.id)

        logger.info(
            "agent_executor.stream.start",
            ticket_id=ticket_id_str,
            source=ticket.source,
            content_length=len(ticket.content),
        )

        initial_state: AgentState = {
            "ticket": ticket,
            "classification": None,
            "retrieved_docs": [],
            "draft": None,
            "escalation": None,
            "should_escalate": False,
            "escalation_reason": "",
            "trace": TriageTrace(ticket_id=ticket_id_str),
            "loop_count": 0,
            "tool_call_count": 0,
            "error_message": "",
            "messages": [],
        }

        start = time.monotonic()

        try:
            async for chunk in self._graph.astream(initial_state):
                # Each chunk is a dict mapping node_name → state_update
                for node_name, state_update in chunk.items():
                    logger.info(
                        "agent_executor.stream.chunk",
                        ticket_id=ticket_id_str,
                        node=node_name,
                    )
                    yield state_update

            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "agent_executor.stream.complete",
                ticket_id=ticket_id_str,
                latency_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "agent_executor.stream.failed",
                ticket_id=ticket_id_str,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=elapsed_ms,
            )
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions — backward-compatible entry points
# ═══════════════════════════════════════════════════════════════════════════════


async def run_triage(
    ticket_id: int | str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    """Execute the triage graph for a single ticket and return a summary dict.

    This is a convenience wrapper around ``AgentExecutor.run()`` that
    constructs a :class:`Ticket` from the provided arguments.

    Parameters
    ----------
    ticket_id:
        Unique ticket identifier.
    subject:
        The ticket subject line.
    body:
        The full ticket body text.

    Returns
    -------
    dict
        Summary of the triage result with keys: ``ticket_id``, ``category``,
        ``urgency``, ``confidence``, ``drafted_response``, ``decision``,
        ``escalation_reason``, ``latency_ms``.
    """
    from src.models.schemas import Ticket, TicketStatus

    content = f"Subject: {subject}\n\nBody:\n{body}"
    ticket = Ticket(
        id=ticket_id,
        content=content,
        source="api",
    )

    executor = AgentExecutor()
    start = time.monotonic()
    final_state = await executor.run(ticket)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    classification = final_state.get("classification")
    draft = final_state.get("draft")

    return {
        "ticket_id": ticket_id,
        "category": classification.category.value if classification else None,
        "urgency": classification.urgency.value if classification else None,
        "confidence": classification.confidence if classification else 0.0,
        "drafted_response": draft.draft_text if draft else "",
        "decision": "escalate" if final_state.get("should_escalate") else "complete",
        "escalation_reason": final_state.get("escalation_reason", ""),
        "latency_ms": elapsed_ms,
    }
