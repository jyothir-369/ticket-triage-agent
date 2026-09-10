"""AgentState — the TypedDict that flows through the LangGraph triage pipeline.

Every node reads from and writes to this shared state.  The ``add_messages``
reducer from ``langgraph.graph.message`` handles the ``messages`` field;
all other fields are simple overwrites.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from src.models.schemas import (
    DraftResponse,
    EscalationDecision,
    Ticket,
    TicketClassification,
    TriageTrace,
)


class AgentState(TypedDict, total=False):
    """Mutable state carried through the triage graph.

    Fields are grouped by pipeline stage.  Nodes are responsible for
    populating only the fields they own — downstream nodes consume them
    but never overwrite a field they don't own.
    """

    # ── Input ───────────────────────────────────────────────────────────────────

    ticket: Ticket
    """The raw incoming ticket."""

    # ── Classification stage ────────────────────────────────────────────────────

    classification: TicketClassification | None
    """Result of the classify node (category + urgency + confidence)."""

    # ── Retrieval stage ─────────────────────────────────────────────────────────

    retrieved_docs: list[dict[str, Any]]
    """Serialised list of RetrievedDocument dicts from the retrieve node."""

    # ── Drafting stage ──────────────────────────────────────────────────────────

    draft: DraftResponse | None
    """LLM-generated draft response with citations."""

    # ── Escalation gate ─────────────────────────────────────────────────────────

    escalation: EscalationDecision | None
    """Outcome of the escalation check (human review needed or not)."""

    should_escalate: bool
    """Quick boolean flag derived from escalation.should_escalate."""

    escalation_reason: str
    """Human-readable reason when escalated (empty string otherwise)."""

    # ── Observability ───────────────────────────────────────────────────────────

    trace: TriageTrace
    """Full audit trail for this triage run (steps, timings, etc.)."""

    loop_count: int
    """Current iteration count — incremented on each loop; breaks at max_loop_retries."""

    tool_call_count: int
    """Total number of failed tool/LLM calls — triggers escalation when > 3."""

    error_message: str
    """Set when a node encounters an unrecoverable error."""

    # ── LangGraph messaging ─────────────────────────────────────────────────────

    messages: Annotated[list[Any], add_messages]
    """LangGraph message list (used for multi-turn LLM interactions)."""
