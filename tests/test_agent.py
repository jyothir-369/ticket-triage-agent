"""Tests for the LangGraph triage agent — graph construction + AgentState."""

import pytest

from src.agent.graph import build_triage_graph
from src.agent.state import AgentState


def test_graph_builds_without_error():
    graph = build_triage_graph()
    compiled = graph.compile()
    assert compiled is not None


def test_agent_state_has_required_keys():
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
        "error_message": "",
        "messages": [],
    }
    assert state["classification"] is None
    assert state["loop_count"] == 0
