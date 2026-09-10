"""Unit tests for each tool in isolation (mock LLM/Qdrant).

Tests the tool registry, search_tickets, classify_ticket, and generate_draft
tools with all external dependencies mocked.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.schemas import (
    RetrievedDocument,
    TicketCategory,
    UrgencyLevel,
)
from src.tools.registry import (
    ClassifyTicketInput,
    GenerateDraftInput,
    SearchTicketsInput,
    ToolRegistry,
    _classify_ticket_impl,
    _generate_draft_impl,
    _search_tickets_impl,
    get_tool_registry,
    reset_tool_registry,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the tool registry singleton before and after each test."""
    reset_tool_registry()
    yield
    reset_tool_registry()


# ═══════════════════════════════════════════════════════════════════════════════
# ToolRegistry tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    def test_registry_initializes_with_defaults(self):
        registry = ToolRegistry(use_defaults=True)
        assert registry.has_tool("search_tickets")
        assert registry.has_tool("classify_ticket")
        assert registry.has_tool("generate_draft")

    def test_registry_empty_without_defaults(self):
        registry = ToolRegistry(use_defaults=False)
        assert not registry.has_tool("search_tickets")
        assert len(registry.get_all_tools()) == 0

    def test_register_tool(self):
        registry = ToolRegistry(use_defaults=False)

        def my_tool(x: int) -> int:
            return x * 2

        registry.register_tool("double", my_tool, description="Doubles a number")
        assert registry.has_tool("double")
        assert registry.get_tool("double")(5) == 10

    def test_register_tool_empty_name_raises(self):
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(ValueError, match="non-empty string"):
            registry.register_tool("", lambda x: x)

    def test_register_tool_non_callable_raises(self):
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(ValueError, match="callable"):
            registry.register_tool("bad", "not_a_function")

    def test_register_tool_invalid_schema_raises(self):
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(ValueError, match="BaseModel"):
            registry.register_tool("bad", lambda x: x, args_schema=str)

    def test_unregister_tool(self):
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool("temp", lambda x: x)
        assert registry.has_tool("temp")
        registry.unregister_tool("temp")
        assert not registry.has_tool("temp")

    def test_unregister_nonexistent_raises(self):
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister_tool("nonexistent")

    def test_get_tool_nonexistent_raises(self):
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(KeyError, match="not registered"):
            registry.get_tool("nonexistent")

    def test_get_all_tools(self):
        registry = ToolRegistry(use_defaults=True)
        tools = registry.get_all_tools()
        assert len(tools) >= 3

    def test_get_tool_names(self):
        registry = ToolRegistry(use_defaults=True)
        names = registry.get_tool_names()
        assert "search_tickets" in names
        assert "classify_ticket" in names
        assert "generate_draft" in names

    def test_get_tools_for_langgraph(self):
        registry = ToolRegistry(use_defaults=True)
        tools = registry.get_tools_for_langgraph()
        assert len(tools) >= 3
        for tool_dict in tools:
            assert "name" in tool_dict
            assert "description" in tool_dict
            assert "function" in tool_dict

    def test_clear(self):
        registry = ToolRegistry(use_defaults=True)
        registry.clear()
        assert len(registry.get_all_tools()) == 0

    def test_singleton_get_and_reset(self):
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2
        reset_tool_registry()
        r3 = get_tool_registry()
        assert r3 is not r1


# ═══════════════════════════════════════════════════════════════════════════════
# search_tickets tool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchTickets:
    @pytest.mark.asyncio
    async def test_search_returns_documents(self):
        mock_docs = [
            RetrievedDocument(
                id="doc-1",
                content="Test document",
                similarity_score=0.9,
                source="past_ticket",
            )
        ]
        with patch("src.tools.registry.get_retriever") as mock_get:
            mock_retriever = AsyncMock()
            mock_retriever.search = AsyncMock(return_value=mock_docs)
            mock_get.return_value = mock_retriever

            results = await _search_tickets_impl("login error")

            assert len(results) == 1
            assert results[0]["id"] == "doc-1"
            assert results[0]["similarity_score"] == 0.9
            mock_retriever.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        with patch("src.tools.registry.get_retriever") as mock_get:
            mock_retriever = AsyncMock()
            mock_retriever.search = AsyncMock(return_value=[])
            mock_get.return_value = mock_retriever

            await _search_tickets_impl(
                "test query",
                category="bug",
                urgency="high",
            )

            call_kwargs = mock_retriever.search.call_args
            assert call_kwargs.kwargs.get("filters") == {"category": "bug", "urgency": "high"}

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        with patch("src.tools.registry.get_retriever") as mock_get:
            mock_retriever = AsyncMock()
            mock_retriever.search = AsyncMock(return_value=[])
            mock_get.return_value = mock_retriever

            results = await _search_tickets_impl("nonexistent topic")
            assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# classify_ticket tool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestClassifyTicket:
    @pytest.mark.asyncio
    async def test_classify_returns_result(self):
        from src.models.schemas import TicketClassification

        mock_result = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.90,
            reasoning="Bug detected.",
        )
        with patch("src.tools.registry.get_classifier") as mock_get:
            mock_classifier = MagicMock()
            mock_classifier.classify = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_classifier

            result = await _classify_ticket_impl("Login Error", "Page crashes on login")

            assert result["category"] == "bug"
            assert result["urgency"] == "high"
            assert result["confidence"] == 0.90
            assert "Bug detected" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_classify_builds_content_correctly(self):
        from src.models.schemas import TicketClassification

        mock_result = TicketClassification(
            category=TicketCategory.BILLING,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.80,
            reasoning="Billing question.",
        )
        with patch("src.tools.registry.get_classifier") as mock_get:
            mock_classifier = MagicMock()
            mock_classifier.classify = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_classifier

            await _classify_ticket_impl("Refund request", "I need a refund")

            call_args = mock_classifier.classify.call_args[0][0]
            assert "Refund request" in call_args
            assert "I need a refund" in call_args


# ═══════════════════════════════════════════════════════════════════════════════
# generate_draft tool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateDraft:
    @pytest.mark.asyncio
    async def test_generate_draft_returns_response(self):
        from src.models.schemas import DraftResponse

        mock_draft = DraftResponse(
            draft_text="We are investigating your issue.",
            confidence=0.85,
            reasoning="Standard response.",
        )
        with patch("src.tools.registry.get_draft_generator") as mock_get:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_draft)
            mock_get.return_value = mock_gen

            docs_json = json.dumps([
                {"id": "doc-1", "content": "Help article", "similarity_score": 0.8}
            ])
            result = await _generate_draft_impl(
                "Login Error",
                "Page crashes",
                docs_json,
            )

            assert "response" in result
            assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_generate_draft_parses_classification(self):
        from src.models.schemas import DraftResponse

        mock_draft = DraftResponse(
            draft_text="Here is help for your bug.",
            confidence=0.80,
            reasoning="Bug response.",
        )
        with patch("src.tools.registry.get_draft_generator") as mock_get:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_draft)
            mock_get.return_value = mock_gen

            classification_json = json.dumps({
                "category": "bug",
                "urgency": "high",
                "confidence": 0.9,
            })
            docs_json = json.dumps([])

            result = await _generate_draft_impl(
                "Subject",
                "Body",
                docs_json,
                classification_json=classification_json,
            )

            assert "response" in result

    @pytest.mark.asyncio
    async def test_generate_draft_handles_empty_docs(self):
        from src.models.schemas import DraftResponse

        mock_draft = DraftResponse(
            draft_text="We will look into this.",
            confidence=0.70,
            reasoning="No context available.",
        )
        with patch("src.tools.registry.get_draft_generator") as mock_get:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_draft)
            mock_get.return_value = mock_gen

            result = await _generate_draft_impl(
                "Help",
                "Need help",
                "[]",
            )

            assert "response" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Input schema validation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestInputSchemas:
    def test_search_tickets_input_valid(self):
        s = SearchTicketsInput(query="login error")
        assert s.query == "login error"
        assert s.limit == 5
        assert s.min_score == 0.5

    def test_search_tickets_input_empty_query_rejected(self):
        with pytest.raises(Exception):
            SearchTicketsInput(query="")

    def test_classify_ticket_input_valid(self):
        s = ClassifyTicketInput(subject="Bug", body="Login crashes")
        assert s.subject == "Bug"

    def test_generate_draft_input_valid(self):
        s = GenerateDraftInput(
            subject="Bug",
            body="Login crashes",
            retrieved_docs_json="[]",
        )
        assert s.subject == "Bug"
