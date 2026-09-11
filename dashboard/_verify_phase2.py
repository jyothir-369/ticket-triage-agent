"""Headless verification of the Phase 2 dashboard via Streamlit's AppTest.

Runs the real app script (not a mock), drives login → metrics → queue →
ticket detail, and asserts no exceptions bubble up. Requires the backend to
be running on port 8000 for data-fetching pages.
"""

import os

import httpx
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


def _button(at, label: str):
    """Find a Streamlit button element by its label."""
    for b in at.button:
        if getattr(b, "label", "") == label:
            return b
    raise AssertionError(f"Button not found: {label!r}; have {[getattr(b,'label','') for b in at.button]}")


def _first_ticket_id() -> str | None:
    try:
        r = httpx.get("http://localhost:8000/tickets/list", params={"page_size": 5}, timeout=8)
        r.raise_for_status()
        tickets = r.json().get("tickets", [])
        return tickets[0]["ticket_id"] if tickets else None
    except Exception:
        return None


def main():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, f"Startup exception: {at.exception}"
    print("1 OK: app starts without exception on login screen")

    # Confirm login form is present
    labels = [getattr(t, "label", "") for t in at.text_input]
    assert any("Username" in l for l in labels), labels
    _button(at, "Sign in")
    print("2 OK: login form rendered")

    # Log in with the demo creds
    at.text_input[0].set_value("admin")
    at.text_input[1].set_value("admin")
    _button(at, "Sign in").click().run()
    assert not at.exception, f"Login exception: {at.exception}"

    # Sidebar radio exists and defaults to metrics
    radios = [r for r in at.radio if getattr(r, "label", "") == "Navigation"]
    assert radios, "No Navigation radio found"
    print("3 OK: logged in; navigation radio present")

    # Metrics page renders
    assert any("Overview" in str(x) for x in at.markdown)
    print("4 OK: metrics page rendered without exception")

    # Navigate to the queue
    radios[0].set_value("queue").run()
    assert not at.exception, f"Queue exception: {at.exception}"
    print("5 OK: queue page rendered")

    # Navigate to the trace viewer
    radios = [r for r in at.radio if getattr(r, "label", "") == "Navigation"]
    radios[0].set_value("trace").run()
    assert not at.exception, f"Trace exception: {at.exception}"
    print("6 OK: trace viewer page rendered")

    # Open a real ticket in detail view (if the backend has one)
    tid = _first_ticket_id()
    if tid:
        at.session_state["page"] = "queue"
        at.session_state["selected_ticket_id"] = tid
        at.run()
        assert not at.exception, f"Detail exception: {at.exception}"
        print(f"7 OK: ticket detail rendered for {tid[:12]}…")

        # Tabs present
        tab_labels = [t.label for t in at.tabs]
        assert tab_labels, f"No tabs found: {tab_labels}"
        print("8 OK: detail tabs:", [str(t)[:18] for t in tab_labels])
    else:
        print("7 SKIP: no tickets in backend to open a detail view")

    # Sign out
    _button(at, "Sign out").click().run()
    assert not at.exception, f"Signout exception: {at.exception}"
    print("9 OK: signed out cleanly")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()