"""Tests for ToolRegistry — central tool registration and DI hub."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from src.tools.registry import (
    ClassifyTicketInput,
    GenerateDraftInput,
    SearchTicketsInput,
    ToolRegistry,
    get_tool_registry,
    reset_tool_registry,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton before and after each test."""
    reset_tool_registry()
    yield
    reset_tool_registry()


class MockSearchInput(BaseModel):
    """Mock input schema for testing."""

    query: str = Field(..., description="Search query.")


def mock_search(query: str) -> list[dict]:
    """Mock search tool."""
    return [{"id": "1", "content": f"Result for {query}"}]


async def async_mock_classify(subject: str, body: str) -> dict:
    """Mock async classify tool."""
    return {"category": "bug", "urgency": "high", "confidence": 0.9}


def mock_classify(subject: str, body: str) -> dict:
    """Mock sync classify tool."""
    return {"category": "bug", "urgency": "high", "confidence": 0.9}


# ═══════════════════════════════════════════════════════════════════════════════
# ToolRegistry initialisation
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryInitialisation:
    """Test registry creation with and without default tools."""

    def test_creates_with_defaults(self):
        """Registry should auto-register the three built-in tools."""
        registry = ToolRegistry(use_defaults=True)
        names = registry.get_tool_names()

        assert "search_tickets" in names
        assert "classify_ticket" in names
        assert "generate_draft" in names
        assert len(names) == 3

    def test_creates_empty(self):
        """Registry should be empty when use_defaults=False."""
        registry = ToolRegistry(use_defaults=False)
        assert registry.get_tool_names() == []
        assert registry.get_all_tools() == []

    def test_singleton_returns_same_instance(self):
        """get_tool_registry() should return the same object."""
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2

    def test_singleton_with_defaults(self):
        """First call with defaults should populate the singleton."""
        registry = get_tool_registry(use_defaults=True)
        assert registry.has_tool("search_tickets")

    def test_reset_singleton(self):
        """reset_tool_registry() should allow creating a fresh singleton."""
        r1 = get_tool_registry()
        reset_tool_registry()
        r2 = get_tool_registry()
        assert r1 is not r2


# ═══════════════════════════════════════════════════════════════════════════════
# register_tool / get_tool
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegisterAndGetTool:
    """Test tool registration and retrieval."""

    def test_register_and_get(self):
        """Should register a tool and retrieve it by name."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool("search_tickets", mock_search, description="Search docs")

        tool = registry.get_tool("search_tickets")
        assert tool is mock_search

    def test_register_with_schema(self):
        """Should store the Pydantic args_schema."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool(
            "search_tickets", mock_search,
            args_schema=MockSearchInput,
        )
        schema = registry.get_tool_schema("search_tickets")
        assert schema is MockSearchInput

    def test_register_replaces_existing(self):
        """Re-registering the same name should replace the tool."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool("my_tool", mock_search)
        registry.register_tool("my_tool", mock_classify)

        tool = registry.get_tool("my_tool")
        assert tool is mock_classify

    def test_register_empty_name_raises(self):
        """Empty tool name should raise ValueError."""
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(ValueError, match="non-empty string"):
            registry.register_tool("", mock_search)

    def test_register_whitespace_name_raises(self):
        """Whitespace-only tool name should raise ValueError."""
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(ValueError, match="non-empty string"):
            registry.register_tool("   ", mock_search)

    def test_register_non_callable_raises(self):
        """Non-callable tool should raise ValueError."""
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(ValueError, match="must be callable"):
            registry.register_tool("bad_tool", "not a function")

    def test_register_invalid_schema_raises(self):
        """Non-BaseModel args_schema should raise ValueError."""
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(ValueError, match="BaseModel subclass"):
            registry.register_tool(
                "my_tool", mock_search, args_schema=dict  # type: ignore
            )

    def test_get_missing_tool_raises(self):
        """Getting a non-existent tool should raise KeyError."""
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(KeyError, match="not registered"):
            registry.get_tool("nonexistent")

    def test_get_tool_description(self):
        """Should return the registered description."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool("my_tool", mock_search, description="My desc")
        assert registry.get_tool_description("my_tool") == "My desc"

    def test_get_tool_description_fallback_to_doc(self):
        """Should fall back to __doc__ when no description given."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool("my_tool", mock_search)
        desc = registry.get_tool_description("my_tool")
        assert "Mock search tool" in desc

    def test_has_tool(self):
        """has_tool should return True for registered, False otherwise."""
        registry = ToolRegistry(use_defaults=False)
        assert not registry.has_tool("search_tickets")
        registry.register_tool("search_tickets", mock_search)
        assert registry.has_tool("search_tickets")


# ═══════════════════════════════════════════════════════════════════════════════
# unregister_tool / clear
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnregisterAndClear:
    """Test tool removal."""

    def test_unregister_tool(self):
        """Should remove a tool from the registry."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool("my_tool", mock_search)
        registry.unregister_tool("my_tool")

        assert not registry.has_tool("my_tool")
        assert registry.get_tool_names() == []

    def test_unregister_missing_raises(self):
        """Unregistering a non-existent tool should raise KeyError."""
        registry = ToolRegistry(use_defaults=False)
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister_tool("nonexistent")

    def test_clear_all(self):
        """clear() should remove all tools."""
        registry = ToolRegistry(use_defaults=True)
        assert len(registry.get_tool_names()) > 0
        registry.clear()
        assert registry.get_tool_names() == []


# ═══════════════════════════════════════════════════════════════════════════════
# get_tools_for_langgraph
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetToolsForLangGraph:
    """Test LangGraph-compatible output format."""

    def test_output_format(self):
        """Each tool dict should have name, description, function, and args_schema."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool(
            "search_tickets", mock_search,
            description="Search tickets",
            args_schema=MockSearchInput,
        )

        tools = registry.get_tools_for_langgraph()
        assert len(tools) == 1

        tool = tools[0]
        assert tool["name"] == "search_tickets"
        assert tool["description"] == "Search tickets"
        assert tool["function"] is mock_search
        assert tool["args_schema"] is MockSearchInput

    def test_includes_all_default_tools(self):
        """Should include all three default tools."""
        registry = ToolRegistry(use_defaults=True)
        tools = registry.get_tools_for_langgraph()
        names = [t["name"] for t in tools]

        assert "search_tickets" in names
        assert "classify_ticket" in names
        assert "generate_draft" in names

    def test_args_schema_present(self):
        """Default tools should have args_schema set."""
        registry = ToolRegistry(use_defaults=True)
        tools = registry.get_tools_for_langgraph()

        for tool in tools:
            assert "args_schema" in tool
            assert issubclass(tool["args_schema"], BaseModel)

    def test_empty_registry(self):
        """Empty registry should return an empty list."""
        registry = ToolRegistry(use_defaults=False)
        assert registry.get_tools_for_langgraph() == []


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency injection for testing
# ═══════════════════════════════════════════════════════════════════════════════


class TestDependencyInjection:
    """Test swapping tools for testing purposes."""

    def test_swap_tool_for_mock(self):
        """Should allow replacing a tool with a mock."""
        registry = ToolRegistry(use_defaults=True)
        mock_fn = MagicMock(return_value={"category": "bug"})

        registry.register_tool("classify_ticket", mock_fn, description="Mock classify")
        tool = registry.get_tool("classify_ticket")

        result = tool(subject="Test", body="Test body")
        assert result == {"category": "bug"}
        mock_fn.assert_called_once()

    def test_inject_async_mock(self):
        """Should accept async functions as tool implementations."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool("classify_ticket", async_mock_classify)

        tool = registry.get_tool("classify_ticket")
        assert tool is async_mock_classify

    def test_independent_registries(self):
        """Two registry instances should be independent."""
        r1 = ToolRegistry(use_defaults=False)
        r2 = ToolRegistry(use_defaults=False)

        r1.register_tool("my_tool", mock_search)
        assert r1.has_tool("my_tool")
        assert not r2.has_tool("my_tool")

    def test_selective_tool_override(self):
        """Should be able to override one tool while keeping others."""
        registry = ToolRegistry(use_defaults=True)
        custom_search = MagicMock(return_value=[])
        registry.register_tool("search_tickets", custom_search)

        # classify_ticket and generate_draft should still be defaults
        assert registry.get_tool("classify_ticket") is not custom_search
        assert registry.get_tool("search_tickets") is custom_search


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic input schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestInputSchemas:
    """Test Pydantic models used as args_schema for each tool."""

    def test_search_tickets_input_valid(self):
        """Valid SearchTicketsInput should pass validation."""
        schema = SearchTicketsInput(query="login error")
        assert schema.query == "login error"
        assert schema.limit == 5
        assert schema.min_score == 0.5
        assert schema.category is None
        assert schema.urgency is None

    def test_search_tickets_input_with_filters(self):
        """Should accept optional category and urgency filters."""
        schema = SearchTicketsInput(
            query="payment issue",
            limit=10,
            min_score=0.7,
            category="billing",
            urgency="high",
        )
        assert schema.category == "billing"
        assert schema.urgency == "high"

    def test_search_tickets_input_empty_query_fails(self):
        """Empty query should fail validation."""
        with pytest.raises(Exception):
            SearchTicketsInput(query="")

    def test_search_tickets_input_limit_bounds(self):
        """Limit should be between 1 and 20."""
        with pytest.raises(Exception):
            SearchTicketsInput(query="test", limit=0)
        with pytest.raises(Exception):
            SearchTicketsInput(query="test", limit=21)

    def test_classify_ticket_input_valid(self):
        """Valid ClassifyTicketInput should pass validation."""
        schema = ClassifyTicketInput(subject="Login broken", body="Cannot log in")
        assert schema.subject == "Login broken"
        assert schema.body == "Cannot log in"

    def test_classify_ticket_input_empty_fails(self):
        """Empty subject or body should fail validation."""
        with pytest.raises(Exception):
            ClassifyTicketInput(subject="", body="test")
        with pytest.raises(Exception):
            ClassifyTicketInput(subject="test", body="")

    def test_generate_draft_input_valid(self):
        """Valid GenerateDraftInput should pass validation."""
        schema = GenerateDraftInput(
            subject="Help needed",
            body="How do I reset my password?",
            retrieved_docs_json='[{"id": "1", "content": "Reset via settings"}]',
        )
        assert schema.subject == "Help needed"
        assert schema.classification_json == "{}"  # default

    def test_generate_draft_input_with_classification(self):
        """Should accept optional classification JSON."""
        classification = '{"category": "usage_help", "urgency": "low", "confidence": 0.85}'
        schema = GenerateDraftInput(
            subject="Question",
            body="How to export?",
            retrieved_docs_json="[]",
            classification_json=classification,
        )
        assert "usage_help" in schema.classification_json

    def test_generate_draft_input_empty_fields_fail(self):
        """Empty required fields should fail validation."""
        with pytest.raises(Exception):
            GenerateDraftInput(subject="", body="test", retrieved_docs_json="[]")
        with pytest.raises(Exception):
            GenerateDraftInput(subject="test", body="", retrieved_docs_json="[]")
        # retrieved_docs_json has min_length=1 — empty string should fail
        with pytest.raises(Exception):
            GenerateDraftInput(subject="test", body="test", retrieved_docs_json="")


# ═══════════════════════════════════════════════════════════════════════════════
# Tool descriptions
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolDescriptions:
    """Test that default tool descriptions are informative."""

    def test_search_tickets_description_mentions_similarity(self):
        """Search description should mention vector similarity."""
        registry = ToolRegistry(use_defaults=True)
        desc = registry.get_tool_description("search_tickets")
        assert "similarity" in desc.lower()

    def test_classify_ticket_description_mentions_categories(self):
        """Classify description should mention categories."""
        registry = ToolRegistry(use_defaults=True)
        desc = registry.get_tool_description("classify_ticket")
        assert "category" in desc.lower()

    def test_generate_draft_description_mentions_citations(self):
        """Draft description should mention citations or context."""
        registry = ToolRegistry(use_defaults=True)
        desc = registry.get_tool_description("generate_draft")
        assert "response" in desc.lower() or "draft" in desc.lower()

    def test_descriptions_are_non_empty(self):
        """All default tools should have non-empty descriptions."""
        registry = ToolRegistry(use_defaults=True)
        for name in registry.get_tool_names():
            desc = registry.get_tool_description(name)
            assert len(desc) > 10, f"Description for '{name}' is too short"


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and error paths."""

    def test_register_lambda(self):
        """Lambda functions should be registerable."""
        registry = ToolRegistry(use_defaults=False)
        registry.register_tool("lambda_tool", lambda x: x * 2)
        assert registry.get_tool("lambda_tool")(5) == 10

    def test_register_callable_object(self):
        """Callable class instances should be registerable."""
        registry = ToolRegistry(use_defaults=False)

        class MyTool:
            def __call__(self, x: int) -> int:
                return x + 1

        my_tool = MyTool()
        registry.register_tool("callable_tool", my_tool)
        assert registry.get_tool("callable_tool")(5) == 6

    def test_many_tools(self):
        """Registry should handle many tools without issues."""
        registry = ToolRegistry(use_defaults=False)
        for i in range(50):
            registry.register_tool(f"tool_{i}", lambda x, _i=i: _i)

        assert len(registry.get_tool_names()) == 50
        assert registry.get_tool("tool_42")(None) == 42

    def test_tool_ordering_preserved(self):
        """Tool order should match insertion order."""
        registry = ToolRegistry(use_defaults=False)
        names = [f"tool_{i}" for i in range(10)]
        for name in names:
            registry.register_tool(name, lambda x: x)

        assert registry.get_tool_names() == names
        langgraph_names = [t["name"] for t in registry.get_tools_for_langgraph()]
        assert langgraph_names == names
