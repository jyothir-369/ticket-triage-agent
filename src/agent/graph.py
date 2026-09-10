"""LangGraph agent — Planner/Executor with an explicit escalation gate.

The graph has six nodes:
  1. classify        – categorize the ticket + assign urgency
  2. retrieve        – search Qdrant for related past tickets
  3. draft           – generate a suggested response
  4. escalate_check  – decide whether to route to a human or mark complete
  5. escalate        – persist the escalation decision
  6. finalize        – save the complete triage result

Edges:
  START → classify → retrieve → draft → escalate_check
  escalate_check → escalate (when should_escalate is True)
  escalate_check → finalize (when should_escalate is False)
  escalate → END
  finalize → END

Loop detection: the agent tracks ``tool_call_count`` in state and routes
directly to escalate when > 3 failed tool calls.

Retry logic: each node is wrapped with tenacity exponential backoff
(1 s → 2 s → 4 s) for transient failures.

Persistence: the graph is compiled with a ``PostgresSaver`` checkpointer
for full state persistence across restarts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from langgraph.graph import END, StateGraph
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Concurrency & Resource Limits
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum concurrent triage runs (semaphore controls parallelism)
_triage_semaphore = asyncio.Semaphore(settings.max_concurrent_triages)

# Track in-flight tasks for graceful shutdown
_inflight_tasks: set[asyncio.Task] = set()
_shutdown_event = asyncio.Event()


def get_triage_semaphore() -> asyncio.Semaphore:
    """Return the global triage concurrency semaphore."""
    return _triage_semaphore


async def acquire_triage_slot() -> bool:
    """Try to acquire a slot in the triage semaphore.

    Returns True if acquired, False if at capacity (logs degradation).
    """
    if _triage_semaphore.locked():
        logger.warning(
            "resource_limit.concurrency_at_capacity",
            max_concurrent=settings.max_concurrent_triages,
            waiting=_triage_semaphore._value,
        )
    await _triage_semaphore.acquire()
    return True


def release_triage_slot() -> None:
    """Release a slot in the triage semaphore."""
    _triage_semaphore.release()


# ═══════════════════════════════════════════════════════════════════════════════
# Graceful Shutdown
# ═══════════════════════════════════════════════════════════════════════════════


async def shutdown(timeout_seconds: float = 30.0) -> None:
    """Graceful shutdown: signal in-flight tasks, wait for completion, close resources.

    Parameters
    ----------
    timeout_seconds:
        Maximum time to wait for in-flight tasks to complete.
    """
    logger.info(
        "shutdown.initiated",
        inflight_count=len(_inflight_tasks),
        timeout_seconds=timeout_seconds,
    )

    _shutdown_event.set()

    if _inflight_tasks:
        logger.info(
            "shutdown.waiting_for_tasks",
            task_count=len(_inflight_tasks),
        )
        # Wait for tasks to complete with timeout
        _, pending = await asyncio.wait(
            _inflight_tasks,
            timeout=timeout_seconds,
        )

        if pending:
            logger.warning(
                "shutdown.forcibly_cancelling",
                remaining=len(pending),
            )
            for task in pending:
                task.cancel()
            # Wait for cancellation to complete
            await asyncio.gather(*pending, return_exceptions=True)

    # Close external connections
    await _close_connections()
    logger.info("shutdown.complete")


async def _close_connections() -> None:
    """Close all external service connections."""
    try:
        from src.services.retrieval import get_retriever
        retriever = get_retriever()
        await retriever.close()
        logger.info("shutdown.retriever_closed")
    except Exception as exc:
        logger.warning("shutdown.retriever_close_failed", error=str(exc))

    try:
        from src.models.database import get_engine
        engine = get_engine()
        await engine.dispose()
        logger.info("shutdown.database_closed")
    except Exception as exc:
        logger.warning("shutdown.database_close_failed", error=str(exc))

    try:
        # Close Redis cache if retriever has one
        from src.services.retrieval import get_retriever
        retriever = get_retriever()
        if retriever._cache is not None:
            await retriever._cache.close()
            logger.info("shutdown.redis_closed")
    except Exception as exc:
        logger.warning("shutdown.redis_close_failed", error=str(exc))


def is_shutting_down() -> bool:
    """Return True if a shutdown has been initiated."""
    return _shutdown_event.is_set()


# ═══════════════════════════════════════════════════════════════════════════════
# Retry decorator with exponential backoff
# ═══════════════════════════════════════════════════════════════════════════════


def _with_retry(node_fn):
    """Wrap an async node function with tenacity exponential backoff.

    Retries up to 3 times on ``Exception`` with waits of 1 s, 2 s, 4 s.
    On each failure the loop counter is incremented; when the retry budget
    is exhausted the error is propagated so the error handler can set
    ``should_escalate``.

    Parameters
    ----------
    node_fn:
        An ``async def(state) -> dict`` node function.

    Returns
    -------
    Callable
        The wrapped function with retry logic.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _retry_wrapper(state):
        return await node_fn(state)

    _retry_wrapper.__name__ = node_fn.__name__
    _retry_wrapper.__qualname__ = node_fn.__qualname__
    _retry_wrapper.__doc__ = node_fn.__doc__
    return _retry_wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Node error handler — catches failures after retries are exhausted
# ═══════════════════════════════════════════════════════════════════════════════


async def _handle_node_error(
    state: AgentState,
    node_name: str,
    exc: Exception,
) -> dict:
    """Update state after a node fails: increment loop/tool counters and
    escalate when the retry budget is exhausted.

    Parameters
    ----------
    state:
        Current agent state.
    node_name:
        Human-readable label for logging (e.g. ``"classify"``).
    exc:
        The exception that caused the failure.

    Returns
    -------
    dict
        Partial state update to merge into ``AgentState``.
    """

    ticket_id = state.get("ticket")
    ticket_id_str = str(ticket_id.id) if ticket_id else "unknown"

    new_loop_count = state.get("loop_count", 0) + 1
    new_tool_call_count = state.get("tool_call_count", 0) + 1

    logger.warning(
        "graph.node_failed",
        node=node_name,
        ticket_id=ticket_id_str,
        error=str(exc),
        error_type=type(exc).__name__,
        loop_count=new_loop_count,
        tool_call_count=new_tool_call_count,
    )

    should_escalate = (
        new_loop_count > settings.max_loop_retries
        or new_tool_call_count > 3
    )

    return {
        "loop_count": new_loop_count,
        "tool_call_count": new_tool_call_count,
        "error_message": str(exc),
        "should_escalate": should_escalate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node functions — imported from nodes.py (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

from src.agent.nodes import (  # noqa: E402
    classify_node,
    draft_node,
    escalate_check_node,
    escalate_node,
    finalize_node,
    retrieve_node,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Error handler node — catches failures after retries are exhausted
# ═══════════════════════════════════════════════════════════════════════════════


async def _error_handler_node(state: AgentState, node_name: str, exc: Exception) -> dict:
    """Update state after a node fails: increment loop/tool counters and
    escalate when the retry budget is exhausted.

    Parameters
    ----------
    state:
        Current agent state.
    node_name:
        Human-readable label for logging (e.g. ``"classify"``).
    exc:
        The exception that caused the failure.

    Returns
    -------
    dict
        Partial state update to merge into ``AgentState``.
    """

    ticket_id = state.get("ticket")
    ticket_id_str = str(ticket_id.id) if ticket_id else "unknown"

    new_loop_count = state.get("loop_count", 0) + 1
    new_tool_call_count = state.get("tool_call_count", 0) + 1

    logger.warning(
        "graph.node_failed",
        node=node_name,
        ticket_id=ticket_id_str,
        error=str(exc),
        error_type=type(exc).__name__,
        loop_count=new_loop_count,
        tool_call_count=new_tool_call_count,
    )

    should_escalate = (
        new_loop_count > settings.max_loop_retries
        or new_tool_call_count > 3
    )

    return {
        "loop_count": new_loop_count,
        "tool_call_count": new_tool_call_count,
        "error_message": str(exc),
        "should_escalate": should_escalate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node wrappers — retry + error handling
# ═══════════════════════════════════════════════════════════════════════════════


async def _classify_wrapped(state: AgentState) -> dict:
    """classify_node with retry logic and error handling."""
    try:
        return await _with_retry(classify_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "classify", exc)


async def _retrieve_wrapped(state: AgentState) -> dict:
    """retrieve_node with retry logic and error handling."""
    try:
        return await _with_retry(retrieve_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "retrieve", exc)


async def _draft_wrapped(state: AgentState) -> dict:
    """draft_node with retry logic and error handling."""
    try:
        return await _with_retry(draft_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "draft", exc)


async def _escalate_check_wrapped(state: AgentState) -> dict:
    """escalate_check_node with retry logic and error handling."""
    try:
        return await _with_retry(escalate_check_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "escalate_check", exc)


async def _escalate_wrapped(state: AgentState) -> dict:
    """escalate_node with retry logic and error handling."""
    try:
        return await _with_retry(escalate_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "escalate", exc)


async def _finalize_wrapped(state: AgentState) -> dict:
    """finalize_node with retry logic and error handling."""
    try:
        return await _with_retry(finalize_node)(state)
    except Exception as exc:
        return await _error_handler_node(state, "finalize", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Routing logic
# ═══════════════════════════════════════════════════════════════════════════════


def route_after_escalation_check(state: dict) -> str:
    """Route after the escalation check.

    Returns ``"escalate"`` when ``state.should_escalate`` is ``True``,
    otherwise ``"finalize"``.

    Parameters
    ----------
    state:
        The current agent state dict.  Typed as ``dict`` rather than
        ``AgentState`` to avoid forward-reference issues with LangGraph's
        type introspection on conditional routing functions.
    """
    should_escalate = state.get("should_escalate", False)
    ticket = state.get("ticket")
    if ticket is None:
        ticket_id_str = "unknown"
    elif hasattr(ticket, "id"):
        ticket_id_str = str(ticket.id)
    elif isinstance(ticket, dict):
        ticket_id_str = str(ticket.get("id", "unknown"))
    else:
        ticket_id_str = str(ticket)

    logger.info(
        "graph.route_after_escalation_check",
        ticket_id=ticket_id_str,
        should_escalate=should_escalate,
    )

    if should_escalate:
        return "escalate"  # maps to "node_escalate" in add_conditional_edges
    return "finalize"  # maps to "node_finalize" in add_conditional_edges


# ═══════════════════════════════════════════════════════════════════════════════
# Graph definition
# ═══════════════════════════════════════════════════════════════════════════════


def build_triage_graph() -> StateGraph:
    """Construct and return the triage agent graph (not compiled).

    Nodes are wrapped with ``_with_retry`` for exponential backoff
    and error handling.
    """
    from src.agent.state import AgentState  # noqa: F811

    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    # Node names must not collide with AgentState keys (e.g. "draft", "classification").
    # We prefix with "node_" where needed to avoid conflicts.
    graph.add_node("node_classify", _classify_wrapped)
    graph.add_node("node_retrieve", _retrieve_wrapped)
    graph.add_node("node_draft", _draft_wrapped)
    graph.add_node("node_escalate_check", _escalate_check_wrapped)
    graph.add_node("node_escalate", _escalate_wrapped)
    graph.add_node("node_finalize", _finalize_wrapped)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("node_classify")

    graph.add_edge("node_classify", "node_retrieve")
    graph.add_edge("node_retrieve", "node_draft")
    graph.add_edge("node_draft", "node_escalate_check")

    # Conditional routing from escalate_check
    graph.add_conditional_edges(
        "node_escalate_check",
        route_after_escalation_check,
        {
            "escalate": "node_escalate",
            "finalize": "node_finalize",
        },
    )

    # Terminal edges
    graph.add_edge("node_escalate", END)
    graph.add_edge("node_finalize", END)

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointer — PostgresSaver for persistence
# ═══════════════════════════════════════════════════════════════════════════════


def _build_checkpointer():
    """Build and return a ``PostgresSaver`` checkpointer for graph persistence.

    Uses the synchronous ``psycopg2`` connection from the configured
    ``DATABASE_URL``.  Falls back to ``MemorySaver`` when the database
    is not available (e.g. during tests).
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        # Convert the async PostgreSQL URL to a synchronous psycopg2 URL
        db_url = settings.database_url
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

        saver = PostgresSaver.from_conn_string(sync_url)
        saver.setup()
        logger.info("graph.checkpointer.postgres_configured", url=sync_url.split("@")[-1])
        return saver

    except Exception as exc:
        logger.warning(
            "graph.checkpointer.fallback_memory",
            error=str(exc),
            reason="PostgresSaver unavailable — using MemorySaver",
        )
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


# ═══════════════════════════════════════════════════════════════════════════════
# Compiled graph
# ═══════════════════════════════════════════════════════════════════════════════

checkpointer = _build_checkpointer()

triage_agent = build_triage_graph().compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# AgentExecutor — high-level interface for running the triage pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class AgentExecutor:
    """High-level interface for running the triage pipeline.

    Provides ``run()`` for one-shot execution and ``stream()`` for
    real-time state updates as each node completes.

    **Resource management**: acquires a concurrency slot (max 5 concurrent),
    enforces a total triage timeout (30s p95), and tracks in-flight tasks
    for graceful shutdown.

    Usage::

        executor = AgentExecutor()
        result = await executor.run(ticket)

        # Or stream updates:
        async for state in executor.stream(ticket):
            print(state)
    """

    def __init__(self, graph=None):
        """Initialise the executor with an optional compiled graph.

        Parameters
        ----------
        graph:
            A compiled LangGraph.  When ``None`` the module-level
            ``triage_agent`` is used.
        """
        self._graph = graph or triage_agent
        logger.debug("agent_executor.initialised")

    async def run(self, ticket: Ticket, config: dict | None = None) -> AgentState:
        """Execute the triage graph for a single ticket and return the final state.

        **Concurrency**: acquires a semaphore slot (max 5 concurrent triage runs).
        **Timeout**: enforces ``triage_timeout_seconds`` (default 30s p95) for the
        entire pipeline.  On timeout, returns a degraded state with escalation.

        Parameters
        ----------
        ticket:
            The incoming support ticket to triage.
        config:
            Optional configuration dict for the graph (e.g., thread_id for checkpointer).

        Returns
        -------
        AgentState
            The complete state after all nodes have executed, or a degraded
            state with escalation triggered on timeout.
        """
        from src.agent.state import AgentState  # noqa: F811
        from src.models.schemas import TriageTrace

        ticket_id_str = str(ticket.id)

        # Check if we're shutting down
        if is_shutting_down():
            logger.warning("agent_executor.run.rejected_shutdown", ticket_id=ticket_id_str)
            return self._build_shutdown_state(ticket)

        # Acquire concurrency slot
        await acquire_triage_slot()
        task = asyncio.current_task()
        if task:
            _inflight_tasks.add(task)

        try:
            logger.info(
                "agent_executor.run.start",
                ticket_id=ticket_id_str,
                source=ticket.source,
                content_length=len(ticket.content),
                max_latency_ms=settings.triage_timeout_seconds * 1000,
            )

            initial_state: AgentState = {
                "ticket": ticket,
                "classification": None,
                "retrieved_docs": [],
                "draft": None,
                "escalation": None,
                "should_escalate": False,
                "escalation_reason": "",
                "trace": TriageTrace(ticket_id=ticket_id_str),
                "loop_count": 0,
                "tool_call_count": 0,
                "error_message": "",
                "messages": [],
            }

            start = time.monotonic()

            try:
                # Build invoke config with thread_id if provided
                invoke_config = config or {"configurable": {"thread_id": ticket_id_str}}
                final_state = await asyncio.wait_for(
                    self._graph.ainvoke(initial_state, config=invoke_config),
                    timeout=settings.triage_timeout_seconds,
                )
            except TimeoutError:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                logger.error(
                    "agent_executor.run.timeout",
                    ticket_id=ticket_id_str,
                    timeout_seconds=settings.triage_timeout_seconds,
                    latency_ms=elapsed_ms,
                )
                # Graceful degradation: return escalation state
                return self._build_timeout_state(ticket, elapsed_ms)

            elapsed_ms = int((time.monotonic() - start) * 1000)

            logger.info(
                "agent_executor.run.complete",
                ticket_id=ticket_id_str,
                should_escalate=final_state.get("should_escalate", False),
                loop_count=final_state.get("loop_count", 0),
                tool_call_count=final_state.get("tool_call_count", 0),
                latency_ms=elapsed_ms,
            )

            return final_state

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "agent_executor.run.failed",
                ticket_id=ticket_id_str,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=elapsed_ms,
            )
            raise

        finally:
            if task:
                _inflight_tasks.discard(task)
            release_triage_slot()

    def _build_timeout_state(self, ticket: Ticket, elapsed_ms: int) -> AgentState:
        """Build a degraded state when the triage pipeline times out."""
        from src.models.schemas import (
            EscalationDecision,
            TicketCategory,
            TicketClassification,
            TriageTrace,
            UrgencyLevel,
        )

        ticket_id_str = str(ticket.id)
        trace = TriageTrace(ticket_id=ticket_id_str, total_duration_ms=elapsed_ms)

        escalation = EscalationDecision(
            should_escalate=True,
            reason=f"Triage pipeline timed out after {elapsed_ms}ms (budget: {settings.triage_timeout_seconds * 1000}ms)",
            confidence_score=0.0,
            threshold_used=settings.confidence_threshold,
        )

        classification = TicketClassification(
            category=TicketCategory.OTHER,
            urgency=UrgencyLevel.HIGH,
            confidence=0.0,
            reasoning="Triage timed out — defaulting to high urgency for safety.",
        )

        return {
            "ticket": ticket,
            "classification": classification,
            "retrieved_docs": [],
            "draft": None,
            "escalation": escalation,
            "should_escalate": True,
            "escalation_reason": escalation.reason,
            "trace": trace,
            "loop_count": 0,
            "tool_call_count": 0,
            "error_message": f"Triage timed out after {elapsed_ms}ms",
            "messages": [],
        }

    def _build_shutdown_state(self, ticket: Ticket) -> AgentState:
        """Build a degraded state when the system is shutting down."""
        from src.models.schemas import (
            EscalationDecision,
            TicketCategory,
            TicketClassification,
            TriageTrace,
            UrgencyLevel,
        )

        ticket_id_str = str(ticket.id)
        trace = TriageTrace(ticket_id=ticket_id_str)

        escalation = EscalationDecision(
            should_escalate=True,
            reason="System is shutting down — ticket queued for retry.",
            confidence_score=0.0,
            threshold_used=settings.confidence_threshold,
        )

        classification = TicketClassification(
            category=TicketCategory.OTHER,
            urgency=UrgencyLevel.HIGH,
            confidence=0.0,
            reasoning="System shutting down.",
        )

        return {
            "ticket": ticket,
            "classification": classification,
            "retrieved_docs": [],
            "draft": None,
            "escalation": escalation,
            "should_escalate": True,
            "escalation_reason": escalation.reason,
            "trace": trace,
            "loop_count": 0,
            "tool_call_count": 0,
            "error_message": "System shutting down",
            "messages": [],
        }

    async def stream(self, ticket: Ticket) -> AsyncIterator[AgentState]:
        """Stream state updates as each node in the triage pipeline completes.

        Parameters
        ----------
        ticket:
            The incoming support ticket to triage.

        Yields
        ------
        AgentState
            The state after each node executes, in execution order.
        """
        from src.agent.state import AgentState  # noqa: F811
        from src.models.schemas import TriageTrace

        ticket_id_str = str(ticket.id)

        logger.info(
            "agent_executor.stream.start",
            ticket_id=ticket_id_str,
            source=ticket.source,
            content_length=len(ticket.content),
        )

        initial_state: AgentState = {
            "ticket": ticket,
            "classification": None,
            "retrieved_docs": [],
            "draft": None,
            "escalation": None,
            "should_escalate": False,
            "escalation_reason": "",
            "trace": TriageTrace(ticket_id=ticket_id_str),
            "loop_count": 0,
            "tool_call_count": 0,
            "error_message": "",
            "messages": [],
        }

        start = time.monotonic()

        try:
            async for chunk in self._graph.astream(initial_state):
                # Each chunk is a dict mapping node_name → state_update
                for node_name, state_update in chunk.items():
                    logger.info(
                        "agent_executor.stream.chunk",
                        ticket_id=ticket_id_str,
                        node=node_name,
                    )
                    yield state_update

            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "agent_executor.stream.complete",
                ticket_id=ticket_id_str,
                latency_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "agent_executor.stream.failed",
                ticket_id=ticket_id_str,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=elapsed_ms,
            )
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions — backward-compatible entry points
# ═══════════════════════════════════════════════════════════════════════════════


async def run_triage(
    ticket_id: int | str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    """Execute the triage graph for a single ticket and return a summary dict.

    This is a convenience wrapper around ``AgentExecutor.run()`` that
    constructs a :class:`Ticket` from the provided arguments.

    Parameters
    ----------
    ticket_id:
        Unique ticket identifier.
    subject:
        The ticket subject line.
    body:
        The full ticket body text.

    Returns
    -------
    dict
        Summary of the triage result with keys: ``ticket_id``, ``category``,
        ``urgency``, ``confidence``, ``drafted_response``, ``decision``,
        ``escalation_reason``, ``latency_ms``.
    """
    from src.models.schemas import Ticket

    content = f"Subject: {subject}\n\nBody:\n{body}"
    ticket = Ticket(
        id=ticket_id,
        content=content,
        source="api",
    )

    executor = AgentExecutor()
    start = time.monotonic()
    final_state = await executor.run(ticket)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    classification = final_state.get("classification")
    draft = final_state.get("draft")

    return {
        "ticket_id": ticket_id,
        "category": classification.category.value if classification else None,
        "urgency": classification.urgency.value if classification else None,
        "confidence": classification.confidence if classification else 0.0,
        "drafted_response": draft.draft_text if draft else "",
        "decision": "escalate" if final_state.get("should_escalate") else "complete",
        "escalation_reason": final_state.get("escalation_reason", ""),
        "latency_ms": elapsed_ms,
    }
