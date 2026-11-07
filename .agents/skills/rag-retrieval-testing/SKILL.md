---
name: rag-retrieval-testing
description: RAG retrieval quality testing — relevance, recall, citation accuracy, and metadata filtering. Use when verifying Qdrant retrieval, embedding correctness, or RAG pipeline quality for the Support Ticket Triage Agent.
installation-note: Remote source (naodeng/awesome-qa-skills) failed to install on Windows due to git checkout error (path length). This skill was authored locally to fill the gap.
---

# RAG Retrieval Testing

## What to Measure

| Metric | Description | Threshold |
|---|---|---|
| Recall@K | Fraction of relevant docs in top-K results | ≥ 0.8 at K=5 |
| Precision@K | Fraction of retrieved docs that are relevant | ≥ 0.6 at K=5 |
| MRR | Mean Reciprocal Rank of first relevant result | ≥ 0.7 |
| Latency | p95 retrieval time | ≤ 500ms |

## Qdrant-Specific Tests

### Collection Health

```python
async def test_collection_exists_with_correct_vector_size():
    client = get_qdrant_client()
    info = client.get_collection(settings.qdrant_collection)
    assert info.config.params.vectors.size == settings.embedding_vector_size
```

### Metadata Filters

Verify that `category` and `urgency` payload filters reduce result set correctly:

```python
async def test_retrieval_category_filter():
    results = await retriever.search("login broken", filters={"category": "billing"})
    for r in results:
        assert r.payload.get("category") == "billing"
```

### No Duplicate Indexing

After processing the same ticket twice, the collection count should not increase:

```python
async def test_no_duplicate_indexing(ticket):
    count_before = client.count(settings.qdrant_collection).count
    await index_ticket(ticket)
    await index_ticket(ticket)  # second time — must be idempotent
    count_after = client.count(settings.qdrant_collection).count
    assert count_after == count_before + 1
```

## Embedding Dimension Compatibility

When switching embedding providers, verify dimensions match the collection:

```python
def test_embedding_dimension_matches_collection():
    vec = embed("test text")
    assert len(vec) == settings.embedding_vector_size
```

## Citation Accuracy

The draft response must cite only retrieved document IDs:

```python
async def test_draft_citations_are_retrieved_docs(ticket, retrieved_docs):
    draft = await draft_response(ticket, retrieved_docs)
    retrieved_ids = {d.id for d in retrieved_docs}
    for citation in draft.citations:
        assert citation.source_id in retrieved_ids
```

## Regression Tests to Always Keep

- Retrieval returns empty list gracefully when collection is empty (no crash).
- Retrieval with a `None` filter falls back to unfiltered search.
- Vector size mismatch raises `ValueError`, not a silent wrong-dimension result.
