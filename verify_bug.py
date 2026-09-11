"""Verify the async context manager protocol error."""
import asyncio
from collections.abc import AsyncGenerator

# This is an async generator (has yield)
async def get_db_session_generator() -> AsyncGenerator[str, None]:
    """This is an async generator - CANNOT be used with async with."""
    print("Opening session")
    try:
        yield "session"
    finally:
        print("Closing session")

# This will fail with the error described
async def test_bug():
    try:
        # This is the BUG - using async with on an async generator
        async with get_db_session_generator() as session:
            print(f"Using {session}")
    except TypeError as e:
        print(f"ERROR: {e}")
        return str(e)

if __name__ == "__main__":
    result = asyncio.run(test_bug())
    print(f"\nConfirmed bug: {result}")
