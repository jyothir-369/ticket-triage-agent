"""Tool functions the LangGraph agent uses during triage."""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from langchain_core.tools import tool

from src.config import get_settings
from src.services.classification import TicketClassifier, get_classifier
from src.services.drafting import draft_response
from src.services.retrieval import (
    RetrievedDoc,
    search_similar_tickets,
    search_tickets,
    get_retriever,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Tool output types
# ---------------------------------------------------------------------------

@dataclass
class ClassifyOutput:
    category: str
    urgency: str
    confidence: float


@dataclass
class RetrieveOutput:
    docs: list[dict]
    count: int


@dataclass
class DraftOutput:
    response: str
    source_count: int


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def classify(subject: str, body: str) -> dict:
    """Classify a support ticket into category and urgency.

    Args:
        subject: The ticket subject line.
        body: The full ticket body text.

    Returns:
        dict with keys: category, urgency, confidence.
    """
    import asyncio

    classifier = get_classifier()
    ticket_content = f"Subject: {subject}\n\nBody:\n{body}"
    result = asyncio.get_event_loop().run_until_complete(
        classifier.classify(ticket_content)
    )
    output = ClassifyOutput(
        category=result.category.value,
        urgency=result.urgency.value,
        confidence=result.confidence,
    )
    logger.info("tool.classify", **vars(output))
    return {"category": output.category, "urgency": output.urgency, "confidence": output.confidence}


@tool
def retrieve(subject: str, body: str, top_k: int = 5) -> dict:
    """Search for similar historical tickets to provide context.

    Args:
        subject: The ticket subject line.
        body: The full ticket body text.
        top_k: Maximum number of results to return.

    Returns:
        dict with keys: docs (list of dicts), count.
    """
    import asyncio

    query = f"{subject} {body}"
    docs: list[RetrievedDoc] = asyncio.get_event_loop().run_until_complete(
        search_similar_tickets(query, top_k=top_k)
    )
    doc_dicts = [
        {"ticket_id": d.ticket_id, "subject": d.subject, "score": d.score}
        for d in docs
    ]
    output = RetrieveOutput(docs=doc_dicts, count=len(doc_dicts))
    logger.info("tool.retrieve", count=output.count)
    return {"docs": output.docs, "count": output.count}


@tool
def draft(subject: str, body: str, retrieved_docs_json: str) -> dict:
    """Draft a response using the ticket and retrieved context.

    Args:
        subject: The ticket subject line.
        body: The full ticket body text.
        retrieved_docs_json: JSON string of retrieved docs from the retrieve tool.

    Returns:
        dict with keys: response, source_count.
    """
    import asyncio
    import json

    raw_docs = json.loads(retrieved_docs_json)
    docs = [
        RetrievedDoc(
            ticket_id=d.get("ticket_id", 0),
            subject=d.get("subject", ""),
            body=d.get("body", ""),
            score=d.get("score", 0.0),
        )
        for d in raw_docs
    ]
    response_text: str = asyncio.get_event_loop().run_until_complete(
        draft_response(subject, body, docs)
    )
    output = DraftOutput(response=response_text, source_count=len(docs))
    logger.info("tool.draft", source_count=output.source_count, length=len(response_text))
    return {"response": output.response, "source_count": output.source_count}


# Exported list of tools for the agent
AGENT_TOOLS = [classify, retrieve, draft, search_tickets]
