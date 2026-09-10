"""Trace SQLAlchemy model — one row per pipeline step."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.database import Base


class TraceModel(Base):
    """Audit-trace row — records one step in the triage pipeline for a ticket."""

    __tablename__ = "trace_steps"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="Auto-increment PK"
    )
    ticket_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the parent ticket",
    )
    step: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="Pipeline step name (classify, retrieve, …)"
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Step status (pending, running, completed, …)"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When this step was recorded (UTC)",
    )
    duration_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Wall-clock duration in milliseconds"
    )
    data: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON-serialised step-specific output data"
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Error message if the step failed"
    )

    # ── Relationships ───────────────────────────────────────────────────────────
    ticket: Mapped[TicketModel] = relationship(  # noqa: F821 — forward ref
        back_populates="traces", lazy="selectin"
    )

    # ── Indexes ─────────────────────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_trace_steps_ticket_id", "ticket_id"),
        Index("ix_trace_steps_status", "status"),
        Index("ix_trace_steps_created_at", "timestamp"),
    )
