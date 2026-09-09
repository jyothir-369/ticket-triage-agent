"""LLM-based ticket classification service."""

from __future__ import annotations

import json
from dataclasses import dataclass

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.ticket import TicketCategory, TicketUrgency
from src.services.llm import call_llm

logger = structlog.get_logger(__name__)

CLASSIFY_SYSTEM_PROMPT = """\
You are a support-ticket classifier. Given a ticket subject and body, return a JSON
object with exactly two keys:

- "category": one of [bug, feature_request, billing, account, general, unknown]
- "urgency": one of [P0, P1, P2, P3]
  - P0 = system down / data loss
  - P1 = major feature broken, no workaround
  - P2 = issue with a workaround
  - P3 = cosmetic / question / nice-to-have

Return ONLY valid JSON. No markdown fences, no commentary.
"""


@dataclass
class Classification:
    category: TicketCategory
    urgency: TicketUrgency


async def classify_ticket(subject: str, body: str) -> Classification:
    """Classify a ticket into category + urgency via the configured LLM."""
    user_msg = f"Subject: {subject}\n\nBody:\n{body}"

    raw = await call_llm(
        [
            SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ],
        temperature=0.0,
        max_tokens=200,
    )

    try:
        parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
        category = TicketCategory(parsed["category"])
        urgency = TicketUrgency(parsed["urgency"])
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("classify.parse_failed", raw=raw, error=str(exc))
        return Classification(category=TicketCategory.UNKNOWN, urgency=TicketUrgency.P3)

    logger.info("classify.result", category=category.value, urgency=urgency.value)
    return Classification(category=category, urgency=urgency)
