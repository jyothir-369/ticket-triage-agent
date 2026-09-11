"""Populate Qdrant with seed tickets (Postgres rows already exist for these source_ids)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.ingest_tickets import (
    index_into_qdrant,
    load_from_csv,
    preprocess_ticket,
)
from src.services.embedding import get_embedding_provider


async def main() -> None:
    raws = load_from_csv("data/eval_tickets.csv")
    processed = [preprocess_ticket(t) for t in raws]
    provider = get_embedding_provider(use_cache=True)
    texts = [t["content"] for t in processed]
    embs = await provider.embed(texts)
    print(f"Embedded {len(embs)} vectors (dim={len(embs[0])})")
    result = await index_into_qdrant(processed, embs, force=True)
    print(f"Qdrant result: {result}")


if __name__ == "__main__":
    asyncio.run(main())