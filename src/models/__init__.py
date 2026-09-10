"""Domain models for the Triage Agent.

SQLAlchemy ORM models live in ``ticket`` / ``trace`` / ``database``.
Pydantic application-layer schemas live in ``schemas``.
"""

# ── SQLAlchemy layer ───────────────────────────────────────────────────────────
from src.models.database import Base, get_db_session, get_engine, get_session

# ── Pydantic schemas ──────────────────────────────────────────────────────────
from src.models.schemas import (
    Citation,
    DashboardMetrics,
    DraftResponse,
    EscalationDecision,
    RecentActivity,
    RetrievedDocument,
    StepStatus,
    Ticket,
    TicketCategory,
    TicketClassification,
    TicketMetadata,
    TicketStatus,
    TriageTrace,
    UrgencyLevel,
)
from src.models.ticket import TicketCategory as DBTicketCategory
from src.models.ticket import TicketModel
from src.models.ticket import TicketStatus as DBTicketStatus
from src.models.ticket import TicketUrgency as DBTicketUrgency
from src.models.trace import TraceModel

# ── Backward-compat aliases (used by api/main.py, seed_db.py) ────────────────
DBTicket = TicketModel
DBTraceModel = TraceModel

__all__ = [
    # SQLAlchemy
    "Base",
    "get_engine",
    "get_db_session",
    "get_session",
    "TicketModel",
    "TraceModel",
    "DBTicket",
    "DBTraceModel",
    "DBTicketCategory",
    "DBTicketStatus",
    "DBTicketUrgency",
    # Pydantic schemas
    "Citation",
    "DashboardMetrics",
    "DraftResponse",
    "EscalationDecision",
    "RecentActivity",
    "RetrievedDocument",
    "StepStatus",
    "Ticket",
    "TicketCategory",
    "TicketClassification",
    "TicketMetadata",
    "TicketStatus",
    "TriageTrace",
    "UrgencyLevel",
]
