"""LLM-based response drafting service."""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.services.llm import call_llm
from src.services.retrieval import RetrievedDoc

logger = structlog.get_logger(__name__)

DRAFT_SYSTEM_PROMPT = """\
You are a support-response drafter. Given a ticket and retrieved context from
past tickets, write a helpful, concise response for the customer.

Rules:
1. Address the customer's specific issue.
2. Reference relevant past solutions when applicable.
3. If the issue is ambiguous, acknowledge it and suggest next steps.
4. Maintain a professional, empathetic tone.
5. Keep the response under 300 words.
"""


async def draft_response(
    subject: str,
    body: str,
    retrieved_docs: list[RetrievedDoc],
) -> str:
    """Draft a customer-facing response using the ticket and retrieved context."""
    context_parts: list[str] = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"[{i}] (score: {doc.score:.2f}) {doc.subject}\n{doc.body[:500]}"
        )
    context_block = "\n\n".join(context_parts) if context_parts else "No related tickets found."

    user_msg = (
        f"## Ticket\nSubject: {subject}\n\nBody:\n{body}\n\n"
        f"## Related Past Tickets\n{context_block}"
    )

    response = await call_llm(
        [
            SystemMessage(content=DRAFT_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    logger.info("draft.generated", length=len(response))
    return response.strip()
