"""Tests for the LangGraph triage agent."""

import pytest

from src.agent.graph import TriageState, build_triage_graph


def test_graph_builds_without_error():
    graph = build_triage_graph()
    compiled = graph.compile()
    assert compiled is not None


def test_triage_state_has_required_keys():
    state_keys = set(TriageState.__annotations__.keys())
    required = {
        "ticket_id",
        "subject",
        "body",
        "category",
        "urgency",
        "confidence",
        "retrieved_docs_json",
        "drafted_response",
        "decision",
        "escalation_reason",
        "loop_count",
        "messages",
    }
    assert required.issubset(state_keys)
