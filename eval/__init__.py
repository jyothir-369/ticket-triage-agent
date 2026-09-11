"""Evaluation framework for the Support-Ticket Triage Agent.

This package measures the agent's accuracy against a hand-labeled set of
:math:`100` tickets (``eval/fixtures/tickets.jsonl``).

Modules
-------
- ``build_fixture``   — reproducible builder for the labeled ticket fixture.
- ``harness``         — :class:`EvalHarness`: loads tickets, runs the real
  agent graph on each, and compares against ground truth.
- ``metrics``         — pure-Python metric computation (accuracy, confusion
  matrices, ROUGE-L, escalation P/R/F1, calibration, latency percentiles).
- ``report``          — console, JSON, and self-contained HTML reports.
- ``run_eval``        — the ``python -m eval.run_eval`` CLI.

Run the full evaluation with::

    python -m eval.run_eval --source all
"""

__version__ = "1.0.0"

# Console reports / the fixture builder print non-ASCII glyphs (→ ≥ · 🎉).
# Windows CI shells default to cp1252, which crashes those prints — pin the
# console streams to UTF-8 up front (no-op where they already are UTF-8).
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass
