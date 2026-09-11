You are a senior engineer who has just taken over a partially-working Support-Ticket Triage Agent. The FastAPI backend boots and the database is connected, but the agent pipeline has bugs that prevent tickets from completing triage. Additionally, three external services (Qdrant Cloud, Upstash Redis, Gemini) are showing connection errors from inside the app even though they work standalone from the terminal. Finally, the Streamlit dashboard has never been verified end-to-end.

Your job is to fix every issue so the entire system runs live: a real ticket submitted via the UI flows through classification → retrieval → drafting → escalation → persistence, and the result appears in the dashboard with a full audit trail.

Work carefully. Diagnose before you fix. Verify after each change. Do not skip steps.

---

PART 1: DIAGNOSE THE TRIAGE PIPELINE BUG

The error reported is:
    TypeError: 'async_generator' object does not support the asynchronous context manager protocol

This means somewhere the code uses `async with some_function()` where `some_function` is an async generator (defined with `async def` + `yield`) rather than an async context manager.

Investigate:
1. Read src/agent/graph.py — look for every `async with` statement and trace what function it calls
2. Read src/api/tickets.py — check the background triage task
3. Read src/services/*.py — look for any helper defined with `async def ... yield ...`
4. Read src/models/database.py — check how AsyncSession is created

Common causes:
- A FastAPI dependency written as `async def get_db() -> AsyncGenerator: ... yield session` being used with `async with` instead of `Depends(...)` or `async for`
- An agent runner that wraps graph execution in a helper that yields state instead of returning it
- A retry/circuit-breaker decorator that produces an async generator when awaited

Report which file and line causes the error. Then fix it:

- If a `Depends`-style dependency is used directly, replace it with the correct pattern
- If an async generator is wrongly wrapped in `async with`, convert to `async for` or refactor to a proper async context manager
- If a helper should return a value but has `yield`, change it to `return`

After the fix, write a minimal test that runs the graph end-to-end with a mock ticket and asserts it completes without raising. Run it before proceeding.

---

PART 2: FIX QDRANT CLOUD CONNECTION INSIDE THE APP

Verify Qdrant works standalone by running this script:

    from src.config import get_settings
    from qdrant_client import QdrantClient
    s = get_settings()
    print("URL:", s.qdrant_url)
    c = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=15)
    print("Collections:", [x.name for x in c.get_collections().collections])

If this succeeds but the app fails with "connection refused," then the app is instantiating QdrantClient differently. Inspect src/services/retrieval.py:

- Does it call `QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)` when qdrant_url is set?
- Does it fall back to `QdrantClient(host=..., port=...)` correctly when qdrant_url is empty?
- Is the client created lazily or at import time? (import-time creation can fail silently)
- Is there a `get_qdrant_client()` helper, and do all callers use it?

Fix any deviation. Then verify by starting the server and calling `curl http://127.0.0.1:8000/health`. The `qdrant` check must report `healthy`.

If the standalone script also fails, capture the exact exception. If it's a DNS or TLS issue, note it as an environment problem and configure a retry with exponential backoff. If it's an auth issue, verify the API key format and regenerate if necessary.

---

PART 3: FIX UPSTASH REDIS CONNECTION INSIDE THE APP

Verify Redis works standalone:

    import asyncio
    from src.config import get_settings
    import redis.asyncio as redis
    async def t():
        s = get_settings()
        r = redis.from_url(s.redis_url, socket_connect_timeout=10)
        print("PING:", await r.ping())
        await r.aclose()
    asyncio.run(t())

If this succeeds but the app fails with timeout, the app is likely:
- Using a sync Redis client (`import redis` instead of `import redis.asyncio`)
- Not passing `socket_connect_timeout` (defaults to blocking)
- Creating connections at import time instead of lazily
- Using a different URL (e.g., falling back to `redis://localhost`)

Inspect src/services/embedding.py and src/services/retrieval.py. Confirm both use `redis.asyncio.from_url(settings.redis_url)` with proper timeouts and TLS handling for `rediss://` URLs.

Fix any deviation. Then restart the server and verify `curl http://127.0.0.1:8000/health` reports `redis` as `healthy`.

Also patch the deprecated `close()` call to `aclose()` to remove the DeprecationWarning.

---

PART 4: FIX GEMINI TIMEOUT IN HEALTH CHECK

The health check makes a live Gemini API call and times out. Two likely causes:
1. The timeout is too short (10s default is tight for first-call TLS + model warmup)
2. The call is retrying in a loop and every attempt times out

Inspect src/api/main.py health check. If the Gemini check has a timeout under 20 seconds, raise it to 30. If it doesn't cache results, add a 60-second cache so repeated health checks don't hammer the API.

Also verify the health check hits the correct endpoint. A lightweight way to test Gemini connectivity is:

    GET https://generativelanguage.googleapis.com/v1beta/models?key={api_key}

That returns quickly and confirms the key is valid without burning inference tokens.

Fix the health check accordingly. Then verify `curl http://127.0.0.1:8000/health` reports `llm` as `healthy`.

---

PART 5: VERIFY THE FULL TRIAGE PIPELINE

Once all four fixes are applied and the server boots cleanly:

1. Start the server in one terminal:
       uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

2. Confirm the health endpoint is all green:
       curl http://127.0.0.1:8000/health

3. Submit a real ticket:
       curl -X POST http://127.0.0.1:8000/tickets/ -H "Content-Type: application/json" -d "{\"content\": \"My dashboard crashes every time I upload a CSV file. It shows a 500 error.\", \"source\": \"manual-test\"}"

4. Copy the ticket_id from the response.

5. Trigger triage:
       curl -X POST http://127.0.0.1:8000/tickets/{id}/triage

6. Wait 20-40 seconds.

7. Check status:
       curl http://127.0.0.1:8000/tickets/{id}/status

8. Fetch trace:
       curl http://127.0.0.1:8000/tickets/{id}/trace

Assert the trace contains:
- classification step with category, urgency, confidence
- retrieval step with number of documents fetched
- draft step with generated response and citation count
- escalation step with decision, threshold, and reason
- finalize step with total duration
- All steps have timestamps

If any step is missing or errored, fix the corresponding service and retry.

Also verify the ticket persisted to Neon by querying the tickets table. The row should have category, urgency, confidence, draft_text, and status populated.

---

PART 6: EMBED SEED TICKETS INTO QDRANT (FOR MEANINGFUL RETRIEVAL)

The Qdrant collection currently has zero points, so retrieval returns nothing and drafts can't cite real sources. Ingest the seeded tickets:

    python -m scripts.ingest_tickets --source data/eval_tickets.json

Or whatever the ingest script's actual interface is (check with --help). Confirm afterward that Qdrant has more than zero points:

    from qdrant_client import QdrantClient
    from src.config import get_settings
    s = get_settings()
    c = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    info = c.get_collection(s.qdrant_collection)
    print("Points:", info.points_count)

Re-submit the ticket from Part 5. The trace's retrieval step should now show documents, and the draft should include citations referencing those documents.

---

PART 7: START AND VERIFY THE STREAMLIT DASHBOARD

Open a second terminal and start the dashboard:

    streamlit run dashboard/app.py --server.port 8501

Or, if the file is at a different path:

    Get-ChildItem -Recurse -Filter "*.py" | Where-Object { $_.FullName -like "*dashboard*" -or $_.FullName -like "*app.py" }

Streamlit will open a browser at http://localhost:8501.

Verify the following works in the browser:

1. Login page appears (credentials from .env: DASHBOARD_USERNAME / DASHBOARD_PASSWORD, default admin/admin)
2. After login, the dashboard shows:
   - Total tickets counter (should be > 10)
   - Category distribution chart
   - Urgency distribution chart
   - Escalation queue listing any tickets with status ESCALATED
3. The ticket you submitted in Part 5 appears in the list
4. Clicking it shows:
   - Original content
   - Classification (category, urgency, confidence, reasoning)
   - Retrieved documents with similarity scores
   - Draft response with citations
   - Full trace with timestamps
5. The "Approve" button works and updates status to RESOLVED
6. The "Escalate" button works and updates status to ESCALATED

If any UI element is broken:
- Fix the API call it makes (check with browser DevTools Network tab)
- Fix the response parsing
- Fix the display logic

Report each UI element's pass/fail status.

---

PART 8: SCREENSHOT-WORTHY DEMO

After everything works, produce a clean demo run:

1. From a fresh terminal, submit three distinct tickets via curl:
   - A bug report: "Login fails with 'invalid token' after password reset"
   - A feature request: "Please add dark mode to the dashboard"
   - A billing question: "Why was I charged twice this month?"

2. For each, capture:
   - ticket_id
   - classification (category, urgency)
   - whether it was auto-approved or escalated
   - the generated draft (first 200 chars)

3. Open the dashboard and screenshot:
   - The metrics page (shows all three tickets counted)
   - The escalation queue (shows any that were escalated)
   - The trace view for one ticket

4. Save screenshots to docs/screenshots/ with descriptive names.

5. Update README.md with a "Live Demo" section linking to the screenshots.

---

PART 9: FINAL VERIFICATION REPORT

Produce a report with these sections:

1. Bug fixes applied:
   - File:line changed, what was wrong, what was fixed

2. Service status (all must be green):
   - Neon PostgreSQL: healthy/unhealthy + latency
   - Upstash Redis: healthy/unhealthy + latency
   - Qdrant Cloud: healthy/unhealthy + points count
   - Gemini: healthy/unhealthy + model used
   - OpenRouter: healthy/unhealthy (fallback test)

3. Triage pipeline status:
   - Ticket submitted: YES/NO
   - Classification completed: YES/NO
   - Retrieval completed: YES/NO + doc count
   - Draft generated: YES/NO + confidence
   - Escalation decision made: YES/NO + reason
   - Trace persisted: YES/NO
   - Total duration: X seconds

4. Dashboard status:
   - Login page: works/broken
   - Metrics page: works/broken
   - Escalation queue: works/broken
   - Ticket detail: works/broken
   - Approve button: works/broken
   - Escalate button: works/broken

5. Screenshots saved:
   - List paths

6. What still doesn't work (be honest):
   - List any remaining issues

Do not declare success until a real ticket has flowed end-to-end through the UI with a full trace.

---

TONE AND STANDARDS

Approach this like a debugger, not a rewriter. Read the code before changing it. Make the smallest change that fixes the problem. Verify each fix before moving to the next. If a fix doesn't work, revert and try a different approach — don't compound broken changes.

Report failures honestly. A bug you can't fix is more useful to me than a fake "it works" that fails in the demo.

When done, the entire stack — FastAPI, agent pipeline, Streamlit dashboard, and all five cloud services — should run live from your machine, with a real ticket completing end-to-end and appearing in the UI.