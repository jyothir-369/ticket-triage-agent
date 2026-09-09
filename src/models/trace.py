"""Triage audit-trace models."""

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.database import Base


class TraceStepType(str, enum.Enum):
    CLASSIFY = "classify"
    RETRIEVE = "retrieve"
    DRAFT = "draft"
    ESCALATION_CHECK = "escalation_check"
    ESCALATE = "escalate"
    COMPLETE = "complete"


class TriageTrace(Base):
    """One triage run, linked to a ticket."""

    __tablename__ = "triage_traces"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    status: Mapped[str] = mapped_column(String(50), default="in_progress")
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    final_decision: Mapped[str | None] = mapped_column(String(50), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    steps: Mapped[list["TraceStep"]] = relationship(
        back_populates="trace", order_by="TraceStep.order"
    )


class TraceStep(Base):
    """Individual step within a triage trace."""

    __tablename__ = "trace_steps"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trace_id: Mapped[int] = mapped_column(ForeignKey("triage_traces.id"), index=True)
    step_type: Mapped[TraceStepType] = mapped_column(Enum(TraceStepType))
    order: Mapped[int] = mapped_column(Integer)
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    trace: Mapped["TriageTrace"] = relationship(back_populates="steps")
