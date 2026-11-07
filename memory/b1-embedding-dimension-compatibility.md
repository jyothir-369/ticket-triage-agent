---
name: b1-embedding-dimension-compatibility
description: Make embedding dimension compatibility enforceable — ensure_collection_exists must raise on mismatch between settings.embedding_vector_size and active provider's get_dimension(); add regression tests for OpenAI/ST/Gemini dimensions + mismatch scenarios. Verified 125 tests pass.
metadata:
  type: project
  priority: high
---

Lesson: patching `src.services.embedding.settings` mock loses `embedding_vector_size`; existing MagicMock providers returning non-int `get_dimension()` fail the new comparison. Fix by explicitly setting `mock_provider.get_dimension.return_value = 1536` on every existing retrieval test patch, then writing new retrieval dimension tests via `patch("src.services.embedding.get_embedding_provider", return_value=<Provider instance>)`.
