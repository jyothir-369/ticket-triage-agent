"""Quick script to time the LangChain Gemini ainvoke path used by classifiers."""
import asyncio
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.services.llm import get_llm

SYSTEM_PROMPT = (
    "You are an expert support-ticket classifier. Classify into EXACTLY one of 6 categories "
    "(BUG, FEATURE_REQUEST, ACCOUNT_ISSUE, BILLING, USAGE_HELP, OTHER) and one of 4 urgencies "
    "(CRITICAL, HIGH, MEDIUM, LOW). Return ONLY valid JSON: "
    '{"category": ..., "urgency": ..., "confidence": 0.0-1.0, "reasoning": "..."}. '
    'Example: "I cannot log in after password change" -> '
    '{"category":"ACCOUNT_ISSUE","urgency":"HIGH","confidence":0.88}'
)


async def main() -> None:
    llm = get_llm("gemini")
    msg = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Classify this ticket:\n\nMy login fails after password reset"),
    ]
    for i in range(2):
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(llm.ainvoke(msg), timeout=25)
            dt = time.monotonic() - t0
            print(f"attempt {i}: {dt:.2f}s content={resp.content[:80]!r}")
        except Exception as e:  # noqa: BLE001
            dt = time.monotonic() - t0
            print(f"attempt {i}: {dt:.2f}s ERROR {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    asyncio.run(main())