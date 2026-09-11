"""Test script to identify the async_generator context manager bug."""
import asyncio
import traceback


async def test_triage():
    """Test the triage pipeline to identify the bug."""
    try:
        from src.models.schemas import Ticket
        from src.agent.graph import AgentExecutor

        ticket = Ticket(
            id="test-123",
            content="Test ticket content",
            source="test",
        )

        executor = AgentExecutor()

        # Try to run the triage
        print("Starting triage...")
        result = await executor.run(ticket)
        print(f"Triage completed: {result}")

    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_triage())
