"""Time the full TicketClassifier prompt on candidate models."""
import asyncio
import time

from src.services.classification import TicketClassifier


async def run(model: str) -> None:
    c = TicketClassifier(model_name=model, timeout_seconds=30)
    t0 = time.monotonic()
    r = await c.classify("My login fails after password reset")
    dt = time.monotonic() - t0
    print(
        f"[{model}] classify OK in {dt:.2f}s  "
        f"cat={r.category.value} urg={r.urgency.value} conf={r.confidence} "
        f"reasoning={r.reasoning[:80]!r}",
        flush=True,
    )


async def main() -> None:
    await run("gemini-flash-lite-latest")


if __name__ == "__main__":
    asyncio.run(main())