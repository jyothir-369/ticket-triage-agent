"""Domain models for the Triage Agent.

SQLAlchemy ORM models live in ``ticket`` / ``trace`` / ``database``.
Pydantic application-layer schemas live in ``schemas``.
"""

# ── SQLAlchemy layer ───────────────────────────────────────────────────────────
from src.models.database import Base, engine, get_db_session
from src.models.ticket import Ticket as DBTicket
from src.models.ticket import TicketCategory as DBTicketCategory
from src.models.ticket import TicketStatus as DBTicketStatus
from src.models.ticket import TicketUrgency as DBTicketUrgency
from src.models.trace import TriageTrace as DBTriageTrace
from src.models.trace import TraceStep as DBTraceStep

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

__all__ = [
    # SQLAlchemy
    "Base",
    "engine",
    "get_db_session",
    "DBTicket",
    "DBTicketCategory",
    "DBTicketStatus",
    "DBTicketUrgency",
    "DBTriageTrace",
    "DBTraceStep",
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
