"""Ticket SQLAlchemy model and enums."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.database import Base

# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class TicketCategory(str, enum.Enum):
    """High-level ticket classification categories."""

    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    ACCOUNT_ISSUE = "account_issue"
    BILLING = "billing"
    USAGE_HELP = "usage_help"
    OTHER = "other"


class TicketUrgency(str, enum.Enum):
    """Urgency tiers assigned during classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, enum.Enum):
    """Lifecycle status of a support ticket."""

    OPEN = "open"
    CLASSIFIED = "classified"
    RETRIEVING = "retrieving"
    DRAFTING = "drafting"
    AWAITING_REVIEW = "awaiting_review"
    ESCALATED = "escalated"
    APPROVED = "approved"
    REJECTED = "rejected"
    RESOLVED = "resolved"
    FAILED = "failed"


# ═══════════════════════════════════════════════════════════════════════════════
# ORM Model
# ═══════════════════════════════════════════════════════════════════════════════


class TicketModel(Base):
    """Full support-ticket table — stores the ticket, its classification,
    draft response, escalation info, and trace data in a single row.
    """

    __tablename__ = "tickets"

    # ── Primary key ─────────────────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, comment="UUID ticket identifier"
    )

    # ── Ingestion fields ────────────────────────────────────────────────────────
    content: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Full text body of the ticket"
    )
    source: Mapped[str] = mapped_column(
        String(100), nullable=False, default="api", comment="Origin system"
    )
    source_id: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        unique=True,
        comment="Original ID in the source system",
    )
    metadata_: Mapped[str | None] = mapped_column(
        "metadata", Text, nullable=True, comment="JSON-serialised TicketMetadata"
    )

    # ── Classification ──────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=TicketStatus.OPEN.value,
        comment="Current lifecycle status",
    )
    category: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Predicted category"
    )
    urgency: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="Predicted urgency level"
    )
    confidence: Mapped[float | None] = mapped_column(
        nullable=True, comment="Classification confidence (0-1)"
    )

    # ── Draft response ──────────────────────────────────────────────────────────
    draft_text: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="LLM-generated draft reply"
    )
    draft_citations: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON-serialised list of citations"
    )

    # ── Escalation ──────────────────────────────────────────────────────────────
    escalation_reason: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Reason for escalation"
    )

    # ── Trace / observability ───────────────────────────────────────────────────
    trace: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON-serialised TriageTrace"
    )
    loop_count: Mapped[int] = mapped_column(
        default=0, nullable=False, comment="Agent loop iteration count"
    )

    # ── Timestamps ──────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When the ticket was created (UTC)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last modification timestamp (UTC)",
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="When triage completed (UTC)"
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="When a human reviewed (UTC)"
    )
    review_decision: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Outcome of human review (approved/rejected/reassigned)",
    )

    # ── Relationships ─────────────────────────────────────────────────────────────
    traces: Mapped[list[TraceModel]] = relationship(  # noqa: F821 — forward ref
        back_populates="ticket", order_by="TraceModel.timestamp", cascade="all, delete-orphan"
    )

    # ── Indexes ─────────────────────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_tickets_status", "status"),
        Index("ix_tickets_created_at", "created_at"),
        UniqueConstraint("source_id", name="uq_tickets_source_id"),
    )
