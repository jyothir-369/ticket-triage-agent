"""Ticket SQLAlchemy model and enums."""

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base


class TicketCategory(str, enum.Enum):
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    BILLING = "billing"
    ACCOUNT = "account"
    GENERAL = "general"
    UNKNOWN = "unknown"


class TicketUrgency(str, enum.Enum):
    P0 = "P0"  # Critical — system down
    P1 = "P1"  # High — major feature broken
    P2 = "P2"  # Medium — workaround exists
    P3 = "P3"  # Low — cosmetic / nice-to-have


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    TRIAGED = "triaged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class Ticket(Base):
    """Represents an incoming support ticket."""

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text)
    customer_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    category: Mapped[TicketCategory] = mapped_column(
        Enum(TicketCategory), default=TicketCategory.UNKNOWN
    )
    urgency: Mapped[TicketUrgency] = mapped_column(
        Enum(TicketUrgency), default=TicketUrgency.P3
    )
    status: Mapped[TicketStatus] = mapped_column(
        Enum(TicketStatus), default=TicketStatus.OPEN
    )
    confidence: Mapped[float | None] = mapped_column(nullable=True)

    drafted_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    escalated_to: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
