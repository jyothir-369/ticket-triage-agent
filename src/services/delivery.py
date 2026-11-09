"""Outbound reply delivery for human-approved draft responses.

FR4 / FR3 constraint: customer-facing replies are sent only after human
approval.  This module provides two delivery backends:

* ``email`` — SMTP delivery (uses ``SMTP_HOST`` / ``SMTP_PORT`` / auth).
* ``webhook`` — HMAC-signed POST to a customer-supplied endpoint.

When neither backend is configured (``reply_delivery_provider='none'``),
the ticket is marked approved but no outbound message is transmitted; the
API response explicitly states ``reply_queued`` so operators know delivery
integration is pending.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from email.message import EmailMessage
from typing import Any

import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class DeliveryResult:
    """Outcome of a delivery attempt."""

    def __init__(
        self,
        *,
        delivered: bool,
        provider: str,
        error: str | None = None,
        message_id: str | None = None,
        attempts: int = 1,
        duration_ms: int = 0,
    ) -> None:
        self.delivered = delivered
        self.provider = provider
        self.error = error
        self.message_id = message_id
        self.attempts = attempts
        self.duration_ms = duration_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivered": self.delivered,
            "provider": self.provider,
            "error": self.error,
            "message_id": self.message_id,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
        }


class DeliveryError(Exception):
    """Raised when a delivery attempt fails."""


# ---------------------------------------------------------------------------
# Email delivery (SMTP)
# ---------------------------------------------------------------------------


def _build_email_message(
    *,
    to_address: str,
    subject: str,
    body: str,
) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = settings.smtp_from_address
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.set_content(body)
    return msg


def _send_email_sync(
    *,
    to_address: str,
    subject: str,
    body: str,
) -> None:
    """Blocking SMTP send; runs inside ``asyncio.to_thread``."""
    import smtplib

    if not settings.smtp_host:
        raise DeliveryError("SMTP_HOST is not configured")

    msg = _build_email_message(
        to_address=to_address,
        subject=subject,
        body=body,
    )

    if settings.smtp_user:
        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=settings.reply_delivery_timeout_seconds)
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
    else:
        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=settings.reply_delivery_timeout_seconds)

    try:
        server.send_message(msg)
    finally:
        server.quit()


async def _deliver_email(
    *,
    to_address: str,
    subject: str,
    body: str,
) -> DeliveryResult:
    """Deliver an approved email reply via SMTP (``SMTP_HOST``/``SMTP_PORT``, ``timeout=settings.reply_delivery_timeout_seconds``).

    Runs the blocking SMTP send inside ``asyncio.to_thread`` to avoid blocking
    the event loop.  The underlying ``smtplib.SMTP(timeout=...)`` call enforces
    a hard timeout to prevent slowloris / connection hangs.
    """
    start = time.monotonic()
    try:
        await asyncio.to_thread(
            _send_email_sync,
            to_address=to_address,
            subject=subject,
            body=body,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "delivery.email.sent",
            to_address=to_address,
            duration_ms=duration_ms,
        )
        return DeliveryResult(
            delivered=True,
            provider="email",
            message_id=f"<triage-{time.time_ns()}@{settings.smtp_host}>",
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "delivery.email.failed",
            to_address=to_address,
            error=str(exc),
            error_type=type(exc).__name__,
            duration_ms=duration_ms,
        )
        return DeliveryResult(
            delivered=False,
            provider="email",
            error=str(exc),
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# Webhook delivery (HMAC-signed JSON POST)
# ---------------------------------------------------------------------------


async def _deliver_webhook(
    *,
    to_address: str,
    subject: str,
    body: str,
) -> DeliveryResult:
    """POST an approved reply to ``reply_webhook_url`` with HMAC-SHA256 signature.

    Uses ``httpx.AsyncClient(timeout=settings.reply_delivery_timeout_seconds)``
    to enforce a hard timeout against the customer-supplied endpoint, preventing
    slowloris attacks.
    """
    import httpx

    if not settings.reply_webhook_url:
        raise DeliveryError("REPLY_WEBHOOK_URL is not configured")

    payload = {
        "ticket_id": to_address,
        "subject": subject,
        "body": body,
        "provider": "webhook",
        "timestamp": time.time(),
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    secret = settings.smtp_password.encode("utf-8")  # reuse SMTP password as webhook secret
    signature = hmac.new(secret, body_bytes, hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Support-Triage-Signature": f"sha256={signature}",
        "X-Support-Triage-Event": "reply.approved",
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=settings.reply_delivery_timeout_seconds) as client:
            response = await client.post(
                str(settings.reply_webhook_url),
                content=body_bytes,
                headers=headers,
            )
        duration_ms = int((time.monotonic() - start) * 1000)
        if response.status_code >= 400:
            logger.warning(
                "delivery.webhook.rejected",
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return DeliveryResult(
                delivered=False,
                provider="webhook",
                error=f"webhook returned HTTP {response.status_code}",
                duration_ms=duration_ms,
            )
        logger.info(
            "delivery.webhook.sent",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return DeliveryResult(
            delivered=True,
            provider="webhook",
            message_id=f"webhook-{time.time_ns()}",
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "delivery.webhook.failed",
            error=str(exc),
            error_type=type(exc).__name__,
            duration_ms=duration_ms,
        )
        return DeliveryResult(
            delivered=False,
            provider="webhook",
            error=str(exc),
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def deliver_reply(
    *,
    customer_email: str,
    subject: str,
    body: str,
) -> DeliveryResult:
    """Deliver an approved reply to the customer.

    Routes to the configured provider (``email`` or ``webhook``).
    Returns a :class:`DeliveryResult`; callers inspect ``delivered`` to
    decide whether to surface ``reply_sent`` or ``reply_queued``.
    """
    provider = settings.reply_delivery_provider

    if provider == "none":
        logger.info(
            "delivery.skipped",
            reason="reply_delivery_provider='none'",
            customer_email=customer_email,
        )
        return DeliveryResult(
            delivered=False,
            provider="none",
            error="delivery provider not configured",
        )

    if provider == "email":
        return await _deliver_email(
            to_address=customer_email,
            subject=subject,
            body=body,
        )

    if provider == "webhook":
        return await _deliver_webhook(
            to_address=customer_email,
            subject=subject,
            body=body,
        )

    logger.warning(
        "delivery.unknown_provider",
        provider=provider,
    )
    return DeliveryResult(
        delivered=False,
        provider=provider,
        error=f"unknown delivery provider '{provider}'",
    )
