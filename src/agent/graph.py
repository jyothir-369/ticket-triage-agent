"""LangGraph agent — Planner/Executor with an explicit escalation gate.

The graph has four nodes:
  1. classify   – categorize the ticket + assign urgency
  2. retrieve   – search Qdrant for related past tickets
  3. draft      – generate a suggested response
  4. escalate   – decide whether to route to a human or mark complete

Loop detection: the agent tracks `loop_count` in state and breaks out
after `MAX_LOOP_RETRIES` iterations.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.config import get_settings
from src.tools.ticket_tools import AGENT_TOOLS

logger = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TriageState(TypedDict):
    """Mutable state flowing through the graph."""

    ticket_id: int
    subject: str
    body: str

    # Populated by nodes
    category: str
    urgency: str
    confidence: float
    retrieved_docs_json: str
    drafted_response: str
    decision: Literal["complete", "escalate"]
    escalation_reason: str

    # Loop control
    loop_count: int

    # Messages (for multi-turn LLM interaction if needed)
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

async def classify_node(state: TriageState) -> dict:
    """Classify the ticket into category + urgency."""
    from src.services.classification import classify_ticket

    result = await classify_ticket(state["subject"], state["body"])
    confidence = 1.0 if result.category.value != "unknown" else 0.4
    logger.info("agent.classify", category=result.category.value, urgency=result.urgency.value)
    return {
        "category": result.category.value,
        "urgency": result.urgency.value,
        "confidence": confidence,
    }


async def retrieve_node(state: TriageState) -> dict:
    """Search for similar historical tickets."""
    from src.services.retrieval import search_similar_tickets

    query = f"{state['subject']} {state['body']}"
    docs = await search_similar_tickets(query, top_k=5)
    doc_dicts = [
        {"ticket_id": d.ticket_id, "subject": d.subject, "body": d.body, "score": d.score}
        for d in docs
    ]
    logger.info("agent.retrieve", count=len(doc_dicts))
    return {"retrieved_docs_json": json.dumps(doc_dicts)}


async def draft_node(state: TriageState) -> dict:
    """Generate a suggested response."""
    from src.services.drafting import draft_response
    from src.services.retrieval import RetrievedDoc

    raw_docs = json.loads(state.get("retrieved_docs_json", "[]"))
    docs = [
        RetrievedDoc(
            ticket_id=d.get("ticket_id", 0),
            subject=d.get("subject", ""),
            body=d.get("body", ""),
            score=d.get("score", 0.0),
        )
        for d in raw_docs
    ]
    response_text = await draft_response(state["subject"], state["body"], docs)
    logger.info("agent.draft", length=len(response_text))
    return {"drafted_response": response_text}


async def escalate_node(state: TriageState) -> dict:
    """Decide: complete or escalate based on confidence threshold."""
    confidence = state.get("confidence", 0.0)
    loop_count = state.get("loop_count", 0)

    # Escalate if confidence is below threshold OR we've looped too many times
    if confidence < settings.confidence_threshold:
        reason = f"Low confidence ({confidence:.2f} < {settings.confidence_threshold})"
        logger.info("agent.escalate", reason=reason)
        return {"decision": "escalate", "escalation_reason": reason}

    if loop_count >= settings.max_loop_retries:
        reason = f"Max loop retries reached ({loop_count})"
        logger.warning("agent.loop_break", reason=reason)
        return {"decision": "escalate", "escalation_reason": reason}

    logger.info("agent.complete", confidence=confidence)
    return {"decision": "complete", "escalation_reason": ""}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_continue(state: TriageState) -> str:
    """Route after the escalation check."""
    return state.get("decision", "complete")


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_triage_graph() -> StateGraph:
    """Construct and return the triage agent graph (not compiled)."""
    graph = StateGraph(TriageState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("draft", draft_node)
    graph.add_node("escalate", escalate_node)

    # Entry → classify → retrieve → draft → escalate → end
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "escalate")
    graph.add_conditional_edges(
        "escalate",
        should_continue,
        {
            "complete": END,
            "escalate": END,
        },
    )

    return graph


# Compile once at import time for reuse
triage_agent = build_triage_graph().compile()


async def run_triage(
    ticket_id: int,
    subject: str,
    body: str,
) -> dict[str, Any]:
    """Execute the triage graph for a single ticket and return the final state."""
    initial_state: TriageState = {
        "ticket_id": ticket_id,
        "subject": subject,
        "body": body,
        "category": "",
        "urgency": "",
        "confidence": 0.0,
        "retrieved_docs_json": "[]",
        "drafted_response": "",
        "decision": "complete",
        "escalation_reason": "",
        "loop_count": 0,
        "messages": [],
    }

    start = time.monotonic()
    final_state = await triage_agent.ainvoke(initial_state)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "agent.triage_complete",
        ticket_id=ticket_id,
        decision=final_state.get("decision"),
        latency_ms=elapsed_ms,
    )

    return {
        "ticket_id": ticket_id,
        "category": final_state.get("category"),
        "urgency": final_state.get("urgency"),
        "confidence": final_state.get("confidence"),
        "drafted_response": final_state.get("drafted_response"),
        "decision": final_state.get("decision"),
        "escalation_reason": final_state.get("escalation_reason"),
        "latency_ms": elapsed_ms,
    }
