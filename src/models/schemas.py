"""Pydantic schemas — request/response models with full validation.

These are the *application-layer* schemas used by the API, agent, and
dashboard.  They are **not** SQLAlchemy ORM models (see ``ticket.py`` /
``trace.py`` for those).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class UrgencyLevel(str, enum.Enum):
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


class StepStatus(str, enum.Enum):
    """Status of an individual trace step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ═══════════════════════════════════════════════════════════════════════════════
# Core domain models
# ═══════════════════════════════════════════════════════════════════════════════


class TicketMetadata(BaseModel):
    """Arbitrary metadata attached to a ticket (source-specific)."""

    customer_email: str | None = None
    tags: list[str] = Field(default_factory=list)
    custom_fields: dict[str, Any] = Field(default_factory=dict)


class Ticket(BaseModel):
    """An incoming support ticket.

    ``id`` may be a database PK or an external identifier depending on
    the ingestion source.
    """

    id: int | str = Field(..., description="Unique ticket identifier.")
    content: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Full text body of the ticket.",
    )
    source: str = Field(
        default="api",
        min_length=1,
        max_length=100,
        description="Origin system (e.g. 'email', 'github', 'intercom', 'api').",
    )
    source_id: str | None = Field(
        default=None,
        max_length=500,
        description="Original ID in the source system (e.g. GitHub issue #).",
    )
    metadata: TicketMetadata = Field(
        default_factory=TicketMetadata,
        description="Source-specific metadata (email, tags, custom fields).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the ticket was created (UTC).",
    )
    status: TicketStatus = Field(
        default=TicketStatus.OPEN,
        description="Current lifecycle status.",
    )

    # ── Validators ──────────────────────────────────────────────────────────────

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank or whitespace-only.")
        return v.strip()

    @field_validator("source")
    @classmethod
    def _source_lower(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("created_at")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════════════


class TicketClassification(BaseModel):
    """Output of the classification step.

    Contains both coarse-grained (category + urgency) and fine-grained
    (per-dimension confidence scores + reasoning) information.
    """

    category: TicketCategory = Field(..., description="Predicted ticket category.")
    urgency: UrgencyLevel = Field(..., description="Predicted urgency level.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall classification confidence (0–1).",
    )
    reasoning: str = Field(
        default="",
        max_length=2000,
        description="LLM-generated reasoning behind the classification.",
    )
    category_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the category prediction specifically.",
    )
    urgency_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the urgency prediction specifically.",
    )

    # ── Validators ──────────────────────────────────────────────────────────────

    @field_validator("reasoning")
    @classmethod
    def _reasoning_strip(cls, v: str) -> str:
        return v.strip()

    # ── Methods ─────────────────────────────────────────────────────────────────

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Return ``True`` when overall confidence meets or exceeds *threshold*."""
        return self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (useful for logging / JSON traces)."""
        return self.model_dump(mode="json")


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════════════


class RetrievedDocument(BaseModel):
    """A single document returned by the vector-search step.

    ``id`` is the vector-store point id; ``source`` identifies the
    originating system (e.g. "past_ticket", "knowledge_base").
    """

    id: str = Field(..., description="Unique document identifier from the vector store.")
    content: str = Field(
        ...,
        min_length=1,
        description="Full text content of the retrieved document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata carried from the vector store.",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine / dot-product similarity score (0–1).",
    )
    source: str = Field(
        default="unknown",
        min_length=1,
        description="Origin of the document (e.g. 'past_ticket', 'knowledge_base').",
    )

    # ── Validators ──────────────────────────────────────────────────────────────

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank.")
        return v.strip()

    @field_validator("source")
    @classmethod
    def _source_lower(cls, v: str) -> str:
        return v.strip().lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Draft response
# ═══════════════════════════════════════════════════════════════════════════════


class Citation(BaseModel):
    """A single citation linking a claim in the draft to a source document."""

    claim: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The specific claim or sentence being cited.",
    )
    source_doc_id: str = Field(
        ...,
        description="id of the RetrievedDocument supporting this claim.",
    )
    relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How relevant this citation is to the claim.",
    )


class DraftResponse(BaseModel):
    """LLM-generated draft reply to the customer.

    Includes validated citations linking claims back to retrieved documents.
    """

    draft_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The full draft response text.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations linking draft claims to source documents.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Draft quality confidence (0–1).",
    )
    reasoning: str = Field(
        default="",
        max_length=2000,
        description="LLM reasoning for the chosen response strategy.",
    )
    citations_validated: bool = Field(
        default=False,
        description="Whether citations have been cross-checked against retrieved docs.",
    )

    # ── Validators ──────────────────────────────────────────────────────────────

    @field_validator("draft_text")
    @classmethod
    def _draft_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("draft_text must not be blank.")
        return v.strip()

    @field_validator("reasoning")
    @classmethod
    def _reasoning_strip(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _validate_citations_exist_when_claimed(self) -> DraftResponse:
        if self.citations_validated and not self.citations:
            raise ValueError(
                "citations_validated is True but no citations were provided."
            )
        return self

    # ── Methods ─────────────────────────────────────────────────────────────────

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Return ``True`` when draft confidence meets or exceeds *threshold*."""
        return self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return self.model_dump(mode="json")


# ═══════════════════════════════════════════════════════════════════════════════
# Trace / audit
# ═══════════════════════════════════════════════════════════════════════════════


class TraceStep(BaseModel):
    """One step in the triage audit trail.

    ``step`` is a free-form label (e.g. "classify", "retrieve", "draft").
    """

    step: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of this pipeline step.",
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Current status of the step.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this step started (UTC).",
    )
    duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Wall-clock duration in milliseconds (set on completion).",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Step-specific output data (classification result, doc list, etc.).",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the step failed.",
    )

    # ── Validators ──────────────────────────────────────────────────────────────

    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @field_validator("error")
    @classmethod
    def _error_strip(cls, v: str | None) -> str | None:
        return v.strip() if v else None


# ═══════════════════════════════════════════════════════════════════════════════
# Escalation
# ═══════════════════════════════════════════════════════════════════════════════


class EscalationDecision(BaseModel):
    """Outcome of the escalation-gate node.

    Records whether a ticket should be routed to a human reviewer and,
    if so, who reviewed it and what decision they reached.
    """

    should_escalate: bool = Field(
        ...,
        description="True when the ticket must be routed to a human.",
    )
    reason: str = Field(
        default="",
        max_length=2000,
        description="Human-readable reason for the escalation decision.",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score that triggered (or avoided) escalation.",
    )
    threshold_used: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="The confidence threshold that was compared against.",
    )
    human_review_required: bool = Field(
        default=True,
        description="Whether a human must review before the response is sent.",
    )
    reviewed_by: str | None = Field(
        default=None,
        max_length=255,
        description="Identifier of the human reviewer (set after review).",
    )
    reviewed_at: datetime | None = Field(
        default=None,
        description="Timestamp when a human reviewed the escalation (UTC).",
    )
    review_decision: Literal["approved", "rejected", "reassigned", None] = Field(
        default=None,
        description="Outcome of the human review (set after review).",
    )

    # ── Validators ──────────────────────────────────────────────────────────────

    @field_validator("reason")
    @classmethod
    def _reason_strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("reviewed_by")
    @classmethod
    def _reviewer_strip(cls, v: str | None) -> str | None:
        return v.strip() if v else None

    @field_validator("reviewed_at")
    @classmethod
    def _ensure_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    # ── Methods ─────────────────────────────────────────────────────────────────

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Return ``True`` when the confidence score is at or above *threshold*."""
        return self.confidence_score >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return self.model_dump(mode="json")


# ═══════════════════════════════════════════════════════════════════════════════
# Full triage trace
# ═══════════════════════════════════════════════════════════════════════════════


class TriageTrace(BaseModel):
    """Complete audit trail for a single triage run.

    Aggregates every step, intermediate result, and the final decision
    into one serialisable object.
    """

    ticket_id: int | str = Field(..., description="Ticket this trace belongs to.")
    steps: list[TraceStep] = Field(
        default_factory=list,
        description="Ordered list of pipeline steps executed.",
    )
    classification: TicketClassification | None = Field(
        default=None,
        description="Classification result (populated after classify step).",
    )
    retrieved_docs: list[RetrievedDocument] = Field(
        default_factory=list,
        description="Documents retrieved during the RAG step.",
    )
    draft: DraftResponse | None = Field(
        default=None,
        description="Draft response (populated after draft step).",
    )
    escalation: EscalationDecision | None = Field(
        default=None,
        description="Escalation decision (populated after escalation gate).",
    )
    final_status: TicketStatus = Field(
        default=TicketStatus.OPEN,
        description="Final status of the ticket after this triage run.",
    )
    total_duration_ms: int = Field(
        default=0,
        ge=0,
        description="Total wall-clock time for the entire triage run.",
    )
    tool_call_count: int = Field(
        default=0,
        ge=0,
        description="Number of external tool/LLM calls made.",
    )
    loop_count: int = Field(
        default=0,
        ge=0,
        description="Number of agent loop iterations executed.",
    )
    loop_detected: bool = Field(
        default=False,
        description="True if the agent loop was broken by the retry guard.",
    )

    # ── Methods ─────────────────────────────────────────────────────────────────

    def add_step(self, step: TraceStep) -> None:
        """Append a step and keep the list ordered by timestamp."""
        self.steps.append(step)
        self.steps.sort(key=lambda s: s.timestamp)

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Shortcut — delegates to the classification result."""
        if self.classification is None:
            return False
        return self.classification.is_high_confidence(threshold)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe with ``mode='json'``)."""
        return self.model_dump(mode="json")


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard metrics
# ═══════════════════════════════════════════════════════════════════════════════


class RecentActivity(BaseModel):
    """One entry in the "recent activity" feed."""

    ticket_id: int | str
    action: str
    timestamp: datetime
    details: str = ""

    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class DashboardMetrics(BaseModel):
    """Aggregated metrics for the Streamlit dashboard / ``/dashboard/triage-metrics``."""

    total_tickets: int = Field(
        default=0,
        ge=0,
        description="Total tickets in the system.",
    )
    processed_tickets: int = Field(
        default=0,
        ge=0,
        description="Tickets that completed the triage pipeline.",
    )
    escalated_tickets: int = Field(
        default=0,
        ge=0,
        description="Tickets escalated to human review.",
    )
    approval_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of escalated tickets that were approved by a human.",
    )
    escalation_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of total tickets that were escalated.",
    )
    avg_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean classification confidence across all tickets.",
    )
    avg_steps: float = Field(
        default=0.0,
        ge=0.0,
        description="Average number of pipeline steps per ticket.",
    )
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of tickets triaged without escalation (auto-resolved).",
    )
    category_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of tickets per category.",
    )
    urgency_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of tickets per urgency level.",
    )
    recent_activity: list[RecentActivity] = Field(
        default_factory=list,
        description="Most recent triage events (newest first).",
    )
    p95_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="95th-percentile triage latency in milliseconds.",
    )
    cost_per_ticket: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated average LLM cost per triaged ticket (USD).",
    )

    # ── Validators ──────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _check_consistency(self) -> DashboardMetrics:
        if self.processed_tickets > self.total_tickets:
            raise ValueError(
                "processed_tickets cannot exceed total_tickets."
            )
        if self.escalated_tickets > self.processed_tickets:
            raise ValueError(
                "escalated_tickets cannot exceed processed_tickets."
            )
        return self

    # ── Methods ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return self.model_dump(mode="json")
