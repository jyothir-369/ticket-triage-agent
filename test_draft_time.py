"""Time the actual DraftGenerator on gemini-flash-latest vs gemini-flash-lite-latest."""
import asyncio
import time

from src.config import get_settings
from src.models.schemas import RetrievedDocument, Ticket, TicketCategory, TicketClassification, UrgencyLevel
from src.services.drafting import get_draft_generator

s = get_settings()
print(f"configured gemini_model = {s.gemini_model}", flush=True)

ticket = Ticket(
    id="t1",
    content="My login fails after password reset",
    source="test",
)

classification = TicketClassification(
    category=TicketCategory.ACCOUNT_ISSUE,
    urgency=UrgencyLevel.HIGH,
    confidence=0.94,
    reasoning="User locked out after password reset",
)

doc = RetrievedDocument(
    id="d1",
    content="Ticket: user could not log in after changing password. Resolution: advise password reset link, verify email, clear cache, contact IT.",
    metadata={"ticket_id": "abc", "subject": "Login issue after reset"},
    similarity_score=0.78,
    source="past_ticket",
)


async def run_once(model: str) -> None:
    gen = get_draft_generator()
    gen.model_name = model
    t0 = time.monotonic()
    try:
        draft = await gen.generate(ticket, classification, [doc])
        dt = time.monotonic() - t0
        print(f"[{model}] draft OK in {dt:.2f}s  len={len(draft.draft_text)} conf={draft.confidence} reasoning={draft.reasoning[:70]!r}", flush=True)
    except Exception as e:  # noqa: BLE001
        dt = time.monotonic() - t0
        print(f"[{model}] draft FAILED in {dt:.2f}s  {type(e).__name__}: {str(e)[:120]}", flush=True)


async def main() -> None:
    for model in ["gemini-flash-lite-latest", "gemini-flash-latest"]:
        await run_once(model)


if __name__ == "__main__":
    asyncio.run(main())