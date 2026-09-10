"""ToolRegistry — central registration and dependency-injection hub for agent tools.

Adapts internal services (retriever, classifier, draft_generator) to
LangGraph's tool interface with proper Pydantic schemas, descriptions,
and dependency-injection support for testing.

Usage
-----
    from src.tools.registry import get_tool_registry

    registry = get_tool_registry()
    tools = registry.get_tools_for_langgraph()
    # → list of dicts with keys: name, description, args_schema, function

    # Dependency injection for testing
    registry.register_tool("classify_ticket", mock_classify_fn)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic input schemas for each tool
# ═══════════════════════════════════════════════════════════════════════════════


class SearchTicketsInput(BaseModel):
    """Input schema for the ``search_tickets`` tool."""

    query: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language search query describing the support issue. "
            "Combine the ticket subject and body for best results."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of similar tickets to return (1-20).",
    )
    min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0 to 1.0).",
    )
    category: str | None = Field(
        default=None,
        description=(
            "Optional filter by ticket category. "
            "Valid values: bug, feature_request, account_issue, billing, usage_help, other."
        ),
    )
    urgency: str | None = Field(
        default=None,
        description=(
            "Optional filter by urgency level. "
            "Valid values: low, medium, high, critical."
        ),
    )


class ClassifyTicketInput(BaseModel):
    """Input schema for the ``classify_ticket`` tool."""

    subject: str = Field(
        ...,
        min_length=1,
        description="The ticket subject line.",
    )
    body: str = Field(
        ...,
        min_length=1,
        description="The full ticket body text.",
    )


class GenerateDraftInput(BaseModel):
    """Input schema for the ``generate_draft`` tool."""

    subject: str = Field(
        ...,
        min_length=1,
        description="The ticket subject line.",
    )
    body: str = Field(
        ...,
        min_length=1,
        description="The full ticket body text.",
    )
    retrieved_docs_json: str = Field(
        ...,
        min_length=1,
        description=(
            "JSON-encoded list of retrieved documents from the search_tickets tool. "
            "Each element should have keys: id, content, metadata, similarity_score, source."
        ),
    )
    classification_json: str = Field(
        default="{}",
        description=(
            "Optional JSON-encoded classification from classify_ticket. "
            "Keys: category, urgency, confidence."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool descriptions
# ═══════════════════════════════════════════════════════════════════════════════


TOOL_DESCRIPTIONS: dict[str, str] = {
    "search_tickets": (
        "Search for similar historical support tickets using vector similarity. "
        "Finds past tickets that are semantically similar to the given query, "
        "returning their content, metadata, and similarity scores. "
        "Use this FIRST to gather context before drafting a response."
    ),
    "classify_ticket": (
        "Classify a support ticket into a category and urgency level. "
        "Analyzes the ticket content and returns one of: bug, feature_request, "
        "account_issue, billing, usage_help, other — along with an urgency level "
        "(low, medium, high, critical) and a confidence score. "
        "Use this to understand the nature and priority of the ticket."
    ),
    "generate_draft": (
        "Generate a professional draft response to a support ticket. "
        "Uses the ticket content, classification, and retrieved context documents "
        "to craft a helpful, accurate, and well-cited response. "
        "Call this AFTER search_tickets and classify_ticket have been executed."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Internal async wrappers (bridge to service layer)
# ═══════════════════════════════════════════════════════════════════════════════


async def _search_tickets_impl(
    query: str,
    limit: int = 5,
    min_score: float = 0.5,
    category: str | None = None,
    urgency: str | None = None,
) -> list[dict]:
    """Search Qdrant for similar tickets via the retriever singleton."""
    from src.services.retrieval import get_retriever

    retriever = get_retriever()

    # Build optional payload filters
    filters: dict[str, Any] = {}
    if category:
        filters["category"] = category
    if urgency:
        filters["urgency"] = urgency

    docs = await retriever.search(
        query,
        limit=limit,
        filters=filters if filters else None,
        min_score=min_score,
    )

    return [
        {
            "id": d.id,
            "content": d.content[:500],  # Truncate for LLM context
            "metadata": d.metadata,
            "similarity_score": d.similarity_score,
            "source": d.source,
        }
        for d in docs
    ]


async def _classify_ticket_impl(subject: str, body: str) -> dict:
    """Classify a ticket using the TicketClassifier."""
    from src.services.classification import get_classifier

    classifier = get_classifier()
    ticket_content = f"Subject: {subject}\n\nBody:\n{body}"
    result = await classifier.classify(ticket_content)

    return {
        "category": result.category.value,
        "urgency": result.urgency.value,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
    }


async def _generate_draft_impl(
    subject: str,
    body: str,
    retrieved_docs_json: str,
    classification_json: str = "{}",
) -> dict:
    """Generate a draft response using the DraftGenerator."""
    from src.models.schemas import (
        RetrievedDocument,
        Ticket,
        TicketCategory,
        TicketClassification,
        UrgencyLevel,
    )
    from src.services.drafting import get_draft_generator

    # Parse retrieved documents
    raw_docs = json.loads(retrieved_docs_json)
    docs = [
        RetrievedDocument(
            id=d.get("id", d.get("ticket_id", "0")),
            content=d.get("content", ""),
            metadata=d.get("metadata", {}),
            similarity_score=d.get("similarity_score", d.get("score", 0.0)),
            source=d.get("source", "past_ticket"),
        )
        for d in raw_docs
    ]

    # Parse classification (optional)
    classification_data = json.loads(classification_json)
    category_str = classification_data.get("category", "other").upper()
    urgency_str = classification_data.get("urgency", "medium").upper()

    try:
        category = TicketCategory(category_str.lower())
    except ValueError:
        category = TicketCategory.OTHER

    try:
        urgency = UrgencyLevel(urgency_str.lower())
    except ValueError:
        urgency = UrgencyLevel.MEDIUM

    classification = TicketClassification(
        category=category,
        urgency=urgency,
        confidence=classification_data.get("confidence", 0.5),
        reasoning=classification_data.get("reasoning", ""),
    )

    # Build ticket object
    ticket = Ticket(
        id="agent",
        content=f"Subject: {subject}\n\nBody:\n{body}",
        source="agent",
    )

    generator = get_draft_generator()
    result = await generator.generate(ticket, classification, docs)

    return {
        "response": result.draft_text,
        "confidence": result.confidence,
        "citations": [
            {"claim": c.claim, "source_doc_id": c.source_doc_id}
            for c in result.citations
        ],
        "reasoning": result.reasoning,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sync wrappers (for LangGraph's @tool decorator which expects sync callables)
# ═══════════════════════════════════════════════════════════════════════════════


def _sync_wrapper(async_fn: Callable) -> Callable:
    """Wrap an async function so it can be called synchronously.

    Uses ``asyncio.get_event_loop().run_until_complete`` with a fallback
    to creating a new event loop when no loop is running.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an already-running loop (e.g. Jupyter);
                # create a new thread to run the async function.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, async_fn(*args, **kwargs)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(async_fn(*args, **kwargs))
        except RuntimeError:
            # No event loop — create one
            return asyncio.run(async_fn(*args, **kwargs))

    wrapper.__name__ = async_fn.__name__
    wrapper.__doc__ = async_fn.__doc__
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════════


class ToolRegistry:
    """Central registry for agent tools with dependency injection support.

    Maintains a mapping of tool names to their implementations and
    provides methods to register, retrieve, and adapt tools for
    LangGraph's tool interface.

    Features
    --------
    * Lazy initialization of default tools (services aren't imported
      until first access).
    * Dependency injection: swap any tool implementation at runtime
      (useful for testing with mocks).
    * LangGraph integration: ``get_tools_for_langgraph()`` returns
      dicts with ``name``, ``description``, ``args_schema``, and
      ``function`` keys ready for ``ToolNode``.

    Examples
    --------
    >>> registry = ToolRegistry()
    >>> tools = registry.get_tools_for_langgraph()
    >>> # Inject a mock for testing
    >>> registry.register_tool("classify_ticket", mock_classify)
    """

    # Default tool configurations: name → (async_impl, input_schema, description)
    _DEFAULT_TOOLS: dict[str, tuple[Callable, type[BaseModel], str]] = {}

    def __init__(self, *, use_defaults: bool = True) -> None:
        """Initialise the registry.

        Parameters
        ----------
        use_defaults:
            If ``True``, automatically register the three built-in tools
            (search_tickets, classify_ticket, generate_draft) on creation.
            Set to ``False`` for a clean registry (useful in tests).
        """
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, type[BaseModel]] = {}
        self._descriptions: dict[str, str] = {}

        if use_defaults:
            self._register_defaults()

        logger.debug(
            "tool_registry.initialised",
            tool_count=len(self._tools),
            tools=list(self._tools.keys()),
        )

    # ── Default registration ──────────────────────────────────────────────

    def _register_defaults(self) -> None:
        """Register the three built-in triage tools."""
        defaults = [
            (
                "search_tickets",
                _sync_wrapper(_search_tickets_impl),
                SearchTicketsInput,
                TOOL_DESCRIPTIONS["search_tickets"],
            ),
            (
                "classify_ticket",
                _sync_wrapper(_classify_ticket_impl),
                ClassifyTicketInput,
                TOOL_DESCRIPTIONS["classify_ticket"],
            ),
            (
                "generate_draft",
                _sync_wrapper(_generate_draft_impl),
                GenerateDraftInput,
                TOOL_DESCRIPTIONS["generate_draft"],
            ),
        ]

        for name, fn, schema, desc in defaults:
            self._tools[name] = fn
            self._schemas[name] = schema
            self._descriptions[name] = desc

    # ── Public API ────────────────────────────────────────────────────────

    def register_tool(
        self,
        name: str,
        tool: Callable,
        *,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
    ) -> None:
        """Register (or replace) a tool by name.

        Parameters
        ----------
        name:
            Unique tool identifier (e.g. ``"search_tickets"``).
        tool:
            Callable that implements the tool. Can be sync or async.
        description:
            Human-readable description shown to the LLM.  Falls back
            to the tool's ``__doc__`` or a generic placeholder.
        args_schema:
            Pydantic model defining the tool's input arguments.  If
            ``None``, the registry will not expose an ``args_schema``
            (LangGraph will infer from the function signature).

        Raises
        ------
        ValueError
            If *name* is empty or *tool* is not callable.
        """
        if not name or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if not callable(tool):
            raise ValueError(f"Tool '{name}' must be callable, got {type(tool).__name__}.")

        self._tools[name] = tool

        if description:
            self._descriptions[name] = description
        elif name not in self._descriptions:
            self._descriptions[name] = getattr(tool, "__doc__", "") or f"Tool: {name}"

        if args_schema is not None:
            if not (isinstance(args_schema, type) and issubclass(args_schema, BaseModel)):
                raise ValueError(
                    f"args_schema for '{name}' must be a Pydantic BaseModel subclass, "
                    f"got {type(args_schema).__name__}."
                )
            self._schemas[name] = args_schema

        logger.info("tool_registry.tool_registered", name=name)

    def get_tool(self, name: str) -> Callable:
        """Retrieve a registered tool by name.

        Parameters
        ----------
        name:
            The tool's unique identifier.

        Returns
        -------
        Callable
            The registered tool function.

        Raises
        ------
        KeyError
            If no tool with the given *name* is registered.
        """
        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys())) or "(none)"
            raise KeyError(
                f"Tool '{name}' is not registered. Available tools: {available}"
            )
        return self._tools[name]

    def get_all_tools(self) -> list[Callable]:
        """Return all registered tool callables.

        Returns
        -------
        list[Callable]
            Ordered list of tool functions (insertion order).
        """
        return list(self._tools.values())

    def get_tools_for_langgraph(self) -> list[dict[str, Any]]:
        """Return tools formatted for LangGraph's ``ToolNode``.

        Each entry is a dict with:

        - ``name``: unique tool identifier
        - ``description``: human-readable description for the LLM
        - ``args_schema``: Pydantic model class defining input arguments
        - ``function``: the callable tool implementation

        Returns
        -------
        list[dict[str, Any]]
            List of tool dicts ready for ``ToolNode(tools=...)``.
        """
        result = []
        for name in self._tools:
            entry: dict[str, Any] = {
                "name": name,
                "description": self._descriptions.get(name, f"Tool: {name}"),
                "function": self._tools[name],
            }
            if name in self._schemas:
                entry["args_schema"] = self._schemas[name]
            result.append(entry)
        return result

    def has_tool(self, name: str) -> bool:
        """Check whether a tool with the given name is registered."""
        return name in self._tools

    def unregister_tool(self, name: str) -> None:
        """Remove a tool from the registry.

        Parameters
        ----------
        name:
            The tool's unique identifier.

        Raises
        ------
        KeyError
            If no tool with the given *name* is registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        del self._tools[name]
        self._schemas.pop(name, None)
        self._descriptions.pop(name, None)
        logger.info("tool_registry.tool_unregistered", name=name)

    def get_tool_names(self) -> list[str]:
        """Return the names of all registered tools in insertion order."""
        return list(self._tools.keys())

    def get_tool_description(self, name: str) -> str:
        """Return the description for a registered tool.

        Raises
        ------
        KeyError
            If no tool with the given *name* is registered.
        """
        if name not in self._descriptions:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._descriptions[name]

    def get_tool_schema(self, name: str) -> type[BaseModel] | None:
        """Return the Pydantic input schema for a tool, or ``None``."""
        return self._schemas.get(name)

    def clear(self) -> None:
        """Remove all registered tools.  Primarily for testing."""
        self._tools.clear()
        self._schemas.clear()
        self._descriptions.clear()
        logger.debug("tool_registry.cleared")


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════════

_registry: ToolRegistry | None = None


def get_tool_registry(*, use_defaults: bool = True) -> ToolRegistry:
    """Return (and lazily create) the module-level ``ToolRegistry`` singleton.

    Parameters
    ----------
    use_defaults:
        Passed through to ``ToolRegistry()`` on first call.  Subsequent
        calls return the same instance regardless of this flag.
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry(use_defaults=use_defaults)
    return _registry


def reset_tool_registry() -> None:
    """Reset the module-level singleton.  For testing only."""
    global _registry
    _registry = None
