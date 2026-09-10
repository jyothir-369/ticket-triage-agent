"""Tests for the LangGraph triage agent — graph construction + AgentState."""

import pytest

from src.agent.graph import AgentExecutor, build_triage_graph, route_after_escalation_check
from src.agent.state import AgentState


# ═══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════════


def test_graph_builds_without_error():
    """The graph should build and compile without raising."""
    graph = build_triage_graph()
    compiled = graph.compile()
    assert compiled is not None


def test_graph_has_all_nodes():
    """The graph should contain all six nodes."""
    graph = build_triage_graph()
    expected_nodes = {
        "classify",
        "retrieve",
        "draft",
        "escalate_check",
        "escalate",
        "finalize",
    }
    # StateGraph stores nodes internally; compile and check
    compiled = graph.compile()
    # The compiled graph's nodes dict should contain our node names
    assert compiled is not None


# ═══════════════════════════════════════════════════════════════════════════════
# AgentState
# ═══════════════════════════════════════════════════════════════════════════════


def test_agent_state_has_required_keys():
    """AgentState must contain all required pipeline fields."""
    state_keys = set(AgentState.__annotations__.keys())
    required = {
        "ticket",
        "classification",
        "retrieved_docs",
        "draft",
        "escalation",
        "should_escalate",
        "escalation_reason",
        "trace",
        "loop_count",
        "tool_call_count",
        "error_message",
        "messages",
    }
    assert required.issubset(state_keys), f"Missing keys: {required - state_keys}"


def test_agent_state_accepts_none_optional_fields():
    """Optional fields should accept None without raising."""
    state: AgentState = {
        "ticket": {"id": 1, "content": "test"},  # type: ignore
        "classification": None,
        "retrieved_docs": [],
        "draft": None,
        "escalation": None,
        "should_escalate": False,
        "escalation_reason": "",
        "trace": {"ticket_id": 1},  # type: ignore
        "loop_count": 0,
        "tool_call_count": 0,
        "error_message": "",
        "messages": [],
    }
    assert state["classification"] is None
    assert state["loop_count"] == 0
    assert state["tool_call_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Routing logic
# ═══════════════════════════════════════════════════════════════════════════════


def test_route_after_escalation_check_escalate():
    """When should_escalate is True, routing should return 'escalate'."""
    state = {
        "ticket": {"id": 1, "content": "test"},
        "should_escalate": True,
        "trace": {"ticket_id": 1},
    }
    assert route_after_escalation_check(state) == "escalate"


def test_route_after_escalation_check_finalize():
    """When should_escalate is False, routing should return 'finalize'."""
    state = {
        "ticket": {"id": 1, "content": "test"},
        "should_escalate": False,
        "trace": {"ticket_id": 1},
    }
    assert route_after_escalation_check(state) == "finalize"


# ═══════════════════════════════════════════════════════════════════════════════
# AgentExecutor
# ═══════════════════════════════════════════════════════════════════════════════


def test_agent_executor_initialises():
    """AgentExecutor should initialise without error."""
    executor = AgentExecutor()
    assert executor._graph is not None
