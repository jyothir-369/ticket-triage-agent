"""Build the 100-ticket labeled evaluation fixture (``eval/fixtures/tickets.jsonl``).

Ticket sources
--------------
1. ``huggingface`` — 60 tickets sampled from ``alemnew/customer-support-tickets``,
   mapped onto the 6-category taxonomy (10 per category, category-diverse).
2. ``github``      — 30 issues from ``qdrant/qdrant`` tagged
   ``bug`` / ``enhancement`` / ``question`` / ``documentation``.
3. ``manual``      — 10 hand-written edge cases (vague, multi-category, PII,
   non-English, one-word, very long, multi-request, hostile, spam, competitor).

Every ticket gets a ground-truth category, urgency, escalation decision and a
human-approved draft.  The check-in ``tickets.jsonl`` is authoritative; this
script reproduces it deterministically (fixed RNG seed), and **fails fast** if
it cannot look up a hand-authored label for a sampled ticket — prompting a
fresh label pass whenever the upstream data drifts.

Usage::

    python -m eval.build_fixture            # → eval/fixtures/tickets.jsonl
    python -m eval.build_fixture --check    # validate the checked-in fixture only
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# The spec names ``alemnew/customer-support-tickets``; that repo no longer
# exists on the Hub.  ``anirudhhari/customer-support-tickets`` is the nearest
# public mirror of the same Kaggle-family "customer support tickets" CSV whose
# ``Ticket Type`` values match the spec's mapping keywords.
HF_DATASET = "anirudhhari/customer-support-tickets"
GH_REPO = "qdrant/qdrant"
GH_LABEL_CATEGORIES = {
    "bug": ["bug"],
    "enhancement": ["enhancement"],
    "question": ["question"],
    "documentation": ["documentation"],
}
GH_TARGET = 30
HF_TARGET = 60
PER_CATEGORY_QUOTA = 10
RNG_SEED = 42
MAX_TICKET_CHARS = 4000
MIN_TICKET_CHARS = 60

CATEGORIES = ["BUG", "FEATURE_REQUEST", "ACCOUNT_ISSUE", "BILLING", "USAGE_HELP", "OTHER"]
URGENCIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

OUTPUT_PATH = Path(__file__).parent / "fixtures" / "tickets.jsonl"

# HF category → taxonomy mapping (per the evaluation spec).
HF_CATEGORY_MAP: list[tuple[str, list[str]]] = [
    ("BUG", ["technical issue", "bug", "error"]),
    ("FEATURE_REQUEST", ["feature request", "enhancement"]),
    ("ACCOUNT_ISSUE", ["account", "login", "access"]),
    ("BILLING", ["billing", "payment", "invoice", "refund"]),
    ("USAGE_HELP", ["how to", "product inquiry", "information"]),
    ("OTHER", []),
]


def map_hf_category(raw: str) -> str:
    """Map a raw HF category string onto our 6-category taxonomy."""
    normalized = (raw or "").strip().lower()
    for category, keywords in HF_CATEGORY_MAP:
        if any(kw in normalized for kw in keywords):
            return category
    return "OTHER"


# ═══════════════════════════════════════════════════════════════════════════════
# Hand-authored labels for sampled HF tickets
# keyed by the dataset's native row id → (urgency, should_escalate, draft, notes)
# ═══════════════════════════════════════════════════════════════════════════════

HF_LABELS: dict[str, tuple[str, bool, str, str]] = {
    # ── BUG ─────────────────────────────────────────────────────────────────
    "825": (
        "MEDIUM",
        False,
        "Thanks for reporting the intermittent freezing. Freezes that resolve on "
        "their own are usually tied to a driver or power setting — could you share "
        "your BIOS and graphics driver versions? I've flagged this to our engineers "
        "either way; the existing workaround should hold until a fix lands.",
        "Intermittent freeze on Dell XPS laptop; workaround exists.",
    ),
    "787": (
        "MEDIUM",
        False,
        "Appreciate the report on the intermittent freezing in AutoCAD. Brief, "
        "self-resolving freezes often come from graphics- or licensing-related "
        "settings; please send your driver and build versions so we can narrow it "
        "down. I've logged it with engineering in the meantime.",
        "Intermittent freeze in Autodesk AutoCAD; workaround exists.",
    ),
    "1435": (
        "LOW",
        False,
        "Thanks for the documentation feedback on advanced settings — getting "
        "confusing docs clarified is exactly the kind of input we want. I've passed "
        "the Amazon Echo page to our docs team and we'll tighten the wording.",
        "Docs feedback; no urgency, no broken behavior.",
    ),
    "1156": (
        "MEDIUM",
        False,
        "Sorry the updates have slowed down so much. A 5-minute update jumping to "
        "30 minutes usually points to network throughput or pending install "
        "corruption. Could you confirm your firmware version and connection type? "
        "We'll dig in from our side as well.",
        "Slow updates on Amazon Echo; device otherwise functional.",
    ),
    "585": (
        "HIGH",
        False,
        "That sounds frustrating — crashing twice a day and losing work is "
        "unacceptable. Please first make sure your work is saved to the cloud or an "
        "external drive as a precaution. Our team is prioritising the touchscreen "
        "stability fix, and we'll reply here with a patch timeline shortly.",
        "Reproducible touchscreen crash with data-loss impact on Dell XPS.",
    ),
    "580": (
        "HIGH",
        False,
        "That sounds frustrating — crashing twice a day and losing unsaved work is "
        "unacceptable. Please save frequently or enable autosave in the meantime. "
        "We're prioritising the touchscreen stability issue and will share a fix "
        "timeline with you shortly.",
        "Reproducible touchscreen crash with data-loss impact.",
    ),
    "566": (
        "HIGH",
        False,
        "Sorry you're hitting this — crashing twice a day and losing unsaved work "
        "is not acceptable. Please enable autosave until we ship a fix. We've "
        "logged the touchscreen issue as high priority and will update you once a "
        "patch is available.",
        "Reproducible touchscreen crash on iPhone 15 with data-loss impact.",
    ),
    "97": (
        "CRITICAL",
        True,
        "We're very sorry about the lost data — with a client presentation tomorrow "
        "we know this is urgent. This has been escalated to our data-recovery team "
        "and they are checking our backups right now. In parallel, please send us "
        "any snapshots or sync logs from the last 30 days; we'll keep you posted "
        "hourly on recovery progress.",
        "Critical data-loss incident; escalated to recovery queue.",
    ),
    "134": (
        "CRITICAL",
        True,
        "A security incident is always urgent, so this has been routed straight to "
        "our security team. In the meantime, please force sign-out of all sessions "
        "and enable 2FA if possible — we can walk you through it. Investigators "
        "will take the unauthorised login from 10 hours ago from there and keep you "
        "updated via a secure channel.",
        "Unauthorised access detected; escalates to security team.",
    ),
    "1386": (
        "LOW",
        False,
        "Happy to help you get the most out of Office 365. For battery-friendly "
        "settings, start with screen brightness, sleep timers, and background-"
        "refresh limits; I'd also recommend turning off hardware acceleration if "
        "you don't need it. Let me know if you'd like a full recommended list.",
        "General settings/usage question; everything working fine.",
    ),
    "1157": (
        "MEDIUM",
        False,
        "Sorry the updates have slowed down. An update jumping from 5 to 30 minutes "
        "usually comes down to network throughput or a pending install that can't "
        "finish cleanly. Could you confirm your firmware version and connection type? "
        "We'll dig into the device logs from our side in parallel.",
        "Slow Amazon Echo updates; device otherwise works fine.",
    ),
    "17": (
        "CRITICAL",
        True,
        "A full system outage for all 100 users is critical — we're treating this as "
        "an incident right now. Our infrastructure team has been alerted and is "
        "checking service health and the last changes to the deployment. We'll post "
        "status here regularly until access is restored.",
        "Complete system outage affecting all users; escalated as incident.",
    ),
    "1293": (
        "LOW",
        False,
        "Thanks for the feedback on the default font size in Dell XPS Laptop — we "
        "appreciate it even though it's not urgent. Since it can be adjusted in "
        "settings for now, I've passed your note to our design team for review on "
        "the default. Nothing further needed from you.",
        "Non-urgent font-size feedback; no broken behavior.",
    ),
    "1419": (
        "LOW",
        False,
        "Thanks for flagging the confusing advanced-settings documentation for "
        "Google Pixel 8 — getting docs clarified is exactly the kind of feedback we "
        "want, even when it's not urgent. I've passed the page to our docs team and "
        "we'll tighten the wording.",
        "Docs clarity feedback; no urgency or broken behavior.",
    ),
    # ── BILLING ──────────────────────────────────────────────────────────────
    "2862": (
        "LOW",
        False,
        "You can absolutely change the billing date on your Autodesk AutoCAD plan — "
        "it's under Billing settings → Cycle. Moving it from the 15th to the 1st "
        "will prorate your next charge once, then settle into the new cycle. Want "
        "me to make the change for you from here?",
        "Billing-date preference change; non-urgent.",
    ),
    "7115": (
        "MEDIUM",
        False,
        "We've received your refund request for the $149.99 renewal you didn't "
        "intend to keep. Since the service wasn't used this period, we've submitted "
        "the refund and it should appear on your original payment method within "
        "5–7 business days. I've also turned off auto-renewal so this doesn't repeat.",
        "Unwanted auto-renewal refund; clear-cut.",
    ),
    "6330": (
        "CRITICAL",
        True,
        "A $1,299 charge you never made is very serious, so we've escalated this to "
        "our fraud team and paused further activity on the account. Alongside "
        "processing a full refund, we strongly recommend contacting your card "
        "issuer. We'll confirm the refund here as soon as it's initiated.",
        "Large suspected-fraud charge; escalates to fraud team.",
    ),
    "2267": (
        "MEDIUM",
        False,
        "Thanks for flagging the unexpected $59.99 charge — we'll review how it "
        "appeared and whether it was part of a plan change you weren't shown. "
        "Please reply with a screenshot of the charge and, if it was unauthorised, "
        "we'll refund it right away.",
        "Unexpected charge needing clarification.",
    ),
    "3082": (
        "LOW",
        False,
        "Great question about annual billing! If you switch from monthly to annual, "
        "you'd be billed once for the full year at a discounted rate, and you'd keep "
        "any remaining monthly credit as a pro-rated adjustment. Nothing changes "
        "about your current plan until you opt in.",
        "General annual-billing curiosity; no action needed.",
    ),
    "6853": (
        "HIGH",
        False,
        "A duplicate charge is never fun — we've verified the two $79.00 charges on "
        "your statement and initiated a refund of the second one. It should appear "
        "on your card within 5–7 business days. You won't need to do anything else; "
        "we'll confirm here once it posts.",
        "Duplicate charge; refund initiated.",
    ),
    "1784": (
        "CRITICAL",
        True,
        "Several unauthorised charges and a possibly stolen card means we're treating "
        "this as fraud. We've frozen further transactions on the account and "
        "escalated to our fraud team for a full investigation and refund of the "
        "$999.00. Please contact your bank to freeze the card as well — we'll stay "
        "in touch here.",
        "Suspected card theft / fraudulent charges; escalates to fraud team.",
    ),
    "7511": (
        "LOW",
        False,
        "No harm in asking! While the Sony WH-1000XM5 purchase is past our standard "
        "30-day return window, I've checked and we can make an exception here. "
        "Reply with your order number and we'll start the return for you.",
        "Return-window inquiry for old purchase; handled with exception.",
    ),
    "2485": (
        "MEDIUM",
        False,
        "Happy to walk through the proration! When you upgraded plans mid-cycle, the "
        "$49.99 charge covers the difference between your old and new rate for the "
        "days already used in that billing period. I'll itemise it in an email — "
        "does that match what you saw?",
        "Proration charge confusion; explain and confirm.",
    ),
    "7614": (
        "LOW",
        False,
        "No problem — since it was a trial charge you'd forgotten about, I've gone "
        "ahead and requested the $29.99 refund anyway. It'll return to your original "
        "payment method within 5–7 business days, and you can cancel any future "
        "trial through the same menu.",
        "Small forgotten trial charge; refund requested, low stakes.",
    ),
    "7320": (
        "MEDIUM",
        False,
        "Easy mix-up to make! Since the wrong Autodesk AutoCAD version is unused "
        "and unopened, we'll refund the $149.99 and you can purchase the correct "
        "edition straight away. I'll start the refund now — look for confirmation "
        "in your inbox.",
        "Accidental purchase of wrong version; refund and repurchase.",
    ),
    "7319": (
        "MEDIUM",
        False,
        "Easy mix-up to make! Since the wrong Autodesk AutoCAD version is unused "
        "and unopened, we'll refund the $149.99 so you can buy the correct edition. "
        "I've started the refund — it typically lands within 5–7 business days.",
        "Accidental purchase of wrong version; refund processed.",
    ),
    "6628": (
        "HIGH",
        False,
        "I'm sorry this happened — four months of incorrect charges totaling "
        "$599.99 is a lot. I've opened a review to refund every charge from the "
        "period in question, and we'll confirm the full amount once audited. You "
        "won't need to pay anything else until this is resolved.",
        "Months of incorrect charges; full-refund review.",
    ),
    "7605": (
        "LOW",
        False,
        "No harm in asking! While the Microsoft Office 365 purchase is past our "
        "standard 30-day return window, I've noted your case and will see whether we "
        "can make an exception. Reply with your order number and I'll take it from "
        "there.",
        "Return-window inquiry for old purchase; possible exception.",
    ),
    "8269": (
        "HIGH",
        False,
        "No problem — we can get the invoice you need for your grant application. "
        "I've generated a detailed PDF invoice with all the line items you're "
        "missing from the confirmation email and it's on its way to your inbox now. "
        "Let me know if the grant requires any specific fields and we'll adjust.",
        "Urgent invoice for grant deadline tomorrow.",
    ),
    "2839": (
        "LOW",
        False,
        "You can absolutely change the billing date for your iPhone 15 plan — it's "
        "under Billing settings → Cycle. Moving from the 15th to the 1st will "
        "prorate your next charge once, then settle into the new monthly cycle. "
        "Happy to make the change for you if you prefer.",
        "Billing-date preference change; non-urgent.",
    ),
    "2658": (
        "MEDIUM",
        False,
        "Sorry the portal is erroring when you try to update the card. Let's sort "
        "this directly: please try clearing the page cache first, and if it still "
        "fails, share the exact error text. I can also update the payment method "
        "manually on my side once you confirm the new card details over a secure "
        "link.",
        "Payment-method update blocked by portal error.",
    ),
    # ── USAGE_HELP ───────────────────────────────────────────────────────────
    "5841": (
        "MEDIUM",
        False,
        "There are indeed a couple of bundle options for Nintendo Switch that work "
        "out cheaper than buying separately — I'll send over the current accessory "
        "bundles and pricing. Let me know which accessories you were considering "
        "and I can point you to the best-value package.",
        "Bundle-deal inquiry; share options.",
    ),
    "5111": (
        "MEDIUM",
        False,
        "Great timing to evaluate iPhone 15 for a 10-person team! I'll send a "
        "detailed feature and pricing pack today, including volume discounting for "
        "10+ seats so you have it well before the decision meeting. Would you also "
        "like a break/fix support quote?",
        "Pre-purchase evaluation with a 7-day deadline.",
    ),
    "5048": (
        "HIGH",
        False,
        "We understand the budget deadline, so I'll make this quick: I'm arranging "
        "for a product specialist to call you inside the hour to finalise pricing on "
        "Samsung Galaxy S23. Expect the call request within the next few minutes — "
        "have your order size handy.",
        "Same-day purchasing decision; requests immediate callback.",
    ),
    "5083": (
        "MEDIUM",
        False,
        "For a 25-person evaluation with a decision meeting in two days, I'll send "
        "the Google Pixel 8 feature sheet and a volume-priced quote today. I can "
        "also set up a tailored walkthrough before the meeting if that helps your "
        "team decide.",
        "Pre-purchase evaluation for 25 users, short deadline.",
    ),
    "5774": (
        "MEDIUM",
        False,
        "There are a few Dell XPS Laptop bundles that are more economical than "
        "buying the laptop and accessories separately — I'll send over what's "
        "currently available along with pricing. Tell me which accessories you want "
        "and I'll put the best-value package together.",
        "Bundle-deal inquiry; share options.",
    ),
    "6080": (
        "LOW",
        False,
        "For video editing, the Dell XPS Laptop is a solid fit — the display and "
        "CPU handle editing well, though a model with 16GB+ of RAM and a dedicated "
        "GPU is worth the upgrade if editing is serious. I can recommend a specific "
        "configuration if you share your usual workflow and budget.",
        "Casual product recommendation query.",
    ),
    "6000": (
        "LOW",
        False,
        "Good news — the Google Pixel 8 is available in your region! I've included "
        "the current price and a few notes on when it goes on sale, so you can "
        "decide whenever you're ready. There's no rush on our side; just reply when "
        "you want to move forward.",
        "Future-consideration availability inquiry; no urgency.",
    ),
    "4808": (
        "HIGH",
        False,
        "We understand the system failed this morning and you need a replacement "
        "ASAP. A specialist can walk you through standing up iPhone 15 for 25 users "
        "by tomorrow — I'll prioritise your request so you get the runbook and a "
        "consult call today. Please share the workloads you need to migrate so we "
        "can confirm fit first.",
        "Emergency procurement to replace failed system by tomorrow.",
    ),
    "5646": (
        "LOW",
        False,
        "Good question to ask before buying! Your Nintendo Switch purchase includes "
        "a standard 12-month manufacturer warranty covering hardware defects, with "
        "extended plans available at checkout. I'll send across the full warranty "
        "terms so you can see exactly what's covered.",
        "Pre-purchase warranty inquiry.",
    ),
    "5102": (
        "MEDIUM",
        False,
        "Great timing to evaluate iPhone 15 for a 5-person team! I'll send a "
        "detailed feature and pricing pack today, including volume discounting, so "
        "you have everything well before the decision meeting. Would a short "
        "walkthrough for your team also help?",
        "Pre-purchase evaluation with a 7-day deadline.",
    ),
    "4968": (
        "HIGH",
        False,
        "We understand the budget deadline, so I'll move quickly: a product "
        "specialist will call you within the hour to finalise the LG Smart TV "
        "pricing and quantities. Watch for the call request — and have your "
        "headcount ready.",
        "Same-day purchasing decision; requests immediate callback.",
    ),
    "4690": (
        "HIGH",
        False,
        "We understand the system failed this morning and you need a replacement "
        "ASAP. I'll prioritise getting you the HP LaserJet Printer readiness info "
        "and a consult call today so it can support 15 users by tomorrow. Let me "
        "know the primary workloads and we'll confirm fit.",
        "Emergency procurement to replace failed system by tomorrow.",
    ),
    "5353": (
        "MEDIUM",
        False,
        "We'd be glad to schedule a HP LaserJet Printer demo this week. I have "
        "openings Thursday and Friday — let me know your preference and how many "
        "people to include, and I'll send the calendar invitation with the demo "
        "agenda.",
        "Demo request ahead of purchasing decision.",
    ),
    "4688": (
        "HIGH",
        False,
        "We understand the urgency — with your old system down, a GoPro Hero rollout "
        "for 10 users by tomorrow is feasible. I'll send the setup runbook and "
        "schedule a priority call today to confirm hardware and licensing fit "
        "before you commit.",
        "Emergency procurement to replace failed system by tomorrow.",
    ),
    "5518": (
        "MEDIUM",
        False,
        "Happy to compare! Samsung Galaxy S23 is a single-device plan built for "
        "individuals, whereas Enterprise Suite adds management, security and "
        "support for a business fleet. I'll send a side-by-side so you can see "
        "pricing and features — happy to talk through which fits your company.",
        "Product comparison for business decision.",
    ),
    "5656": (
        "LOW",
        False,
        "Good question to ask before buying! Your Google Pixel 8 purchase includes "
        "a standard 12-month manufacturer warranty covering hardware defects, with "
        "extended plans available at checkout. I'll send across the full terms so "
        "you can confirm what's covered.",
        "Pre-purchase warranty inquiry.",
    ),
    "5207": (
        "MEDIUM",
        False,
        "We'd rather you confirm compatibility before a big purchase — yes, "
        "Nintendo Switch is supported on Ubuntu 22.04 LTS via our native client; "
        "the latest driver release added full desktop support. I'll send the setup "
        "notes and the exact build that's verified, so you can test it before "
        "committing.",
        "Compatibility check before a significant purchase.",
    ),
    # ── OTHER (cancellations / account closures) ─────────────────────────────
    "3314": (
        "HIGH",
        True,
        "We're treating the security issue seriously and cancelling all Sony "
        "WH-1000XM5 subscriptions on your account immediately. Because this "
        "involves a security matter, it's now with our security team who will "
        "confirm the shutdown and review activity on the account. Expect a "
        "confirmation from them today.",
        "Emergency termination triggered by a security issue; escalate.",
    ),
    "3320": (
        "HIGH",
        True,
        "Closing an account due to suspected suspicious activity is exactly the "
        "kind of thing we take over from here. I've routed this to our security "
        "team who will close the Fitbit Charge 6 account today and review access. "
        "They'll reach out to verify the closure and any recovery steps you need.",
        "Account closure due to suspicious activity; escalate to security.",
    ),
    "4118": (
        "MEDIUM",
        False,
        "We can downgrade your Samsung Galaxy S23 plan from premium to the free "
        "basic version — the change takes effect at the end of the current billing "
        "period, so you'll keep premium features until then. I'll process the "
        "downgrade now and confirm by email.",
        "Plan downgrade from premium to free tier.",
    ),
    "3703": (
        "MEDIUM",
        True,
        "Three months of unresolved issues is on us — before we cancel, I'd like a "
        "support lead to review your ticket history and our open fixes, because "
        "your experience should have been better. If you still want to cancel after "
        "that review, we'll do it immediately and make sure the subscription ends "
        "cleanly.",
        "Cancellation after unresolved issues; review history before processing.",
    ),
    "4013": (
        "MEDIUM",
        False,
        "No hard feelings about switching providers — we'll make the cancellation "
        "painless. Your Nintendo Switch subscription will be cancelled and I'll "
        "send a confirmation to your email as requested. You'll keep access until "
        "the end of the current billing period.",
        "Routine cancellation; confirm by email.",
    ),
    "3386": (
        "HIGH",
        True,
        "Closing an account due to suspected suspicious activity is something we "
        "handle directly. I've escalated this to our security team, who will close "
        "the GoPro Hero account today and review what triggered the alert. They'll "
        "confirm with you and help with any recovery steps.",
        "Account closure due to suspicious activity; escalate to security.",
    ),
    "3913": (
        "MEDIUM",
        False,
        "We've scheduled your Microsoft Office 365 subscription to cancel at the "
        "end of the current billing period, so no further monthly charges after "
        "that. You'll keep full access until then. Confirmation is on its way to "
        "your inbox.",
        "Routine end-of-period cancellation.",
    ),
    "3344": (
        "HIGH",
        True,
        "We've escalated this: closing an account because you don't feel safe is "
        "exactly the situation our security team handles. They'll close the HP "
        "LaserJet Printer account today and review the suspicious activity. Expect "
        "confirmation and next steps from them directly.",
        "Account closure due to suspicious activity; escalate to security.",
    ),
    "3844": (
        "MEDIUM",
        False,
        "Better late than never! I've processed the cancellation of your Nintendo "
        "Switch subscription so you won't be charged again, effective end of the "
        "current cycle. I'll also turn off auto-renewal so it doesn't reactivate.",
        "Overdue cancellation; avoid another charge.",
    ),
    "4435": (
        "LOW",
        False,
        "Happy to explain how cancellation works ahead of time! You can cancel your "
        "Google Pixel 8 subscription any time from Billing settings, and it will "
        "stay active until the end of the current period — no early-termination "
        "fees. No action needed from you yet; just reach out when you're ready.",
        "Future cancellation process inquiry; informational only.",
    ),
    "4575": (
        "LOW",
        False,
        "You can absolutely pause your LG Smart TV subscription while you're on "
        "holiday! We support pausing for 1–3 months with no charge during the pause "
        "and no loss of your history. Just tell me the dates and I'll set it up.",
        "Pause-while-away inquiry (holiday).",
    ),
    "3823": (
        "MEDIUM",
        False,
        "Done — I've started the cancellation of your Nintendo Switch subscription "
        "so you won't be charged again, and it stays active through the end of the "
        "current billing period. Auto-renewal is switched off as well, so nothing "
        "re-triggers later.",
        "Overdue cancellation; avoid another charge.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Hand-authored labels for sampled GitHub issues
# keyed by GitHub issue number → (category, urgency, should_escalate, draft, notes)
# ═══════════════════════════════════════════════════════════════════════════════

GH_TICKET_LABELS: dict[int, tuple[str, str, bool, str, str]] = {
    # ── bug ────────────────────────────────────────────────────────────────
    9372: (
        "BUG",
        "MEDIUM",
        False,
        "Thanks for the precise report — we've confirmed strict mode accepts 0 for "
        "some configuration fields while rejecting it for equivalent ones, which can "
        "let an unusable collection be created. I've logged this with the "
        "strict-mode validation team and attached your reproduction. We'll track the "
        "fix here.",
        "Strict-mode config inconsistently validates zero values.",
    ),
    10520: (
        "BUG",
        "HIGH",
        True,
        "Appreciate the detailed report — this is a real validation gap: prefetches "
        "inside group queries bypass max_query_limit under strict mode. That's "
        "security-relevant, so I've escalated it to our API engineering team for a "
        "fix and we'll update this thread once a patch is scoped.",
        "Strict-mode prefetch bypass of max_query_limit — security-relevant.",
    ),
    10607: (
        "BUG",
        "MEDIUM",
        False,
        "Thanks for the clear reproduction — unordered results when rescore=false on "
        "binary-quantized Euclidean queries is a genuine bug in the quantization "
        "path. I've forwarded it to the search team with your example so they can "
        "verify the ordering logic. We'll confirm here once it's fixed.",
        "Binary-quantized Euclidean results not sorted by score when rescore=false.",
    ),
    10127: (
        "BUG",
        "MEDIUM",
        False,
        "Great catch — count with exact=false on a match_any filter under-counts and "
        "the error grows with the number of values, while exact=true is correct. I've "
        "filed this with the payload-index team and attached your example so they can "
        "reproduce it directly. We'll update this thread when a fix is available.",
        "count exact=false under-counts match_any, error scales with value count.",
    ),
    10302: (
        "BUG",
        "HIGH",
        True,
        "Thank you for the thorough concurrency report and the repro scripts. A "
        "desynchronized payload index under concurrent set_payload is a serious "
        "consistency bug, so I've escalated it to our storage engineering team along "
        "with your docker-compose and reproduction script. We'll follow up here with "
        "the root cause.",
        "Payload keyword index desyncs under concurrent set_payload by-filter.",
    ),
    10368: (
        "BUG",
        "CRITICAL",
        True,
        "This is a critical data-safety issue and we're treating it as a release "
        "blocker. We've confirmed that snapshot recovery with priority 'replica' can "
        "replace live data instead of preserving it, and the fix is being prioritised "
        "by our storage team. We'll update this thread with the patch milestone and a "
        "safe workaround.",
        "Snapshot recovery with priority replica silently destroys live data.",
    ),
    10500: (
        "BUG",
        "MEDIUM",
        False,
        "Thanks for the detailed MMR reproduction — applying limit before offset is "
        "definitely wrong for pagination and returns too few points. I've passed this "
        "to the search team with your exact parameters so they can reproduce it "
        "cleanly. We'll update once the fix lands.",
        "MMR applies limit before offset, returning too few points.",
    ),
    10522: (
        "BUG",
        "HIGH",
        True,
        "Good find — the gRPC facet path accepting exact=true when search_allow_exact "
        "is false is both an inconsistency with REST and a strict-mode bypass. I've "
        "escalated this to our API team; we'll confirm here once a fix covers both "
        "endpoints.",
        "gRPC facet exact=true bypasses search_allow_exact=false.",
    ),
    9255: (
        "BUG",
        "MEDIUM",
        False,
        "Thanks for the report — a payload filter returning points whose payload "
        "field is missing is a bug in the filter path. I've reproduced it from your "
        "example and logged it with the payload-index team. We'll keep you posted on "
        "the fix.",
        "Payload filter returns points with a missing payload field.",
    ),
    9515: (
        "BUG",
        "MEDIUM",
        False,
        "Thanks for the analysis — scalar-quantized Euclid scores shifting under "
        "translation violates the expected invariance when rescore=false. I've "
        "forwarded the report to our quantization team with your example so they can "
        "verify. We'll update this thread with the fix.",
        "Scalar-quantized Euclid scores violate translation invariance.",
    ),
    10546: (
        "BUG",
        "HIGH",
        True,
        "This is a low-level correctness issue: an oversized HNSW m value overflows "
        "internally and can return incorrect self-matches, so we're escalating it to "
        "core engineering as a priority. We'll share a patch timeline here shortly "
        "and a workaround in the meantime.",
        "HNSW m overflow derives m0 to zero and returns incorrect self matches.",
    ),
    9417: (
        "BUG",
        "HIGH",
        False,
        "Thanks for catching this — accepting a collection creation without a vectors "
        "field produces an unusable collection. I've filed it against the "
        "collection-validation team and it should land in an upcoming release. We'll "
        "confirm here once the fix ships.",
        "Missing vectors field silently accepted at collection creation.",
    ),
    9670: (
        "BUG",
        "HIGH",
        True,
        "Thanks for the detailed root-cause writeup. A dropped restart future leaving "
        "the local shard update handler stopped is a real reliability bug, so I've "
        "escalated it to our core team with your analysis attached. We'll follow up "
        "here once it's addressed.",
        "try_join_all drops shard restart future, stopping the update handler.",
    ),
    10555: (
        "BUG",
        "LOW",
        False,
        "Thanks for the report — the generic 400 on malformed sparse vectors is a "
        "validation usability issue that names neither the field nor the point. I've "
        "logged it with the API team to improve the error message. We'll update here "
        "when it lands.",
        "Malformed sparse vector 400 lacks actionable location info.",
    ),
    10064: (
        "BUG",
        "MEDIUM",
        False,
        "Thanks for the report — a false 'point not found' error on multi-shard-key "
        "payload updates is a real routing bug, and it's useful to know the writes "
        "still apply. Escalated to our storage team with your reproduction; we'll "
        "keep you updated here.",
        "False point-not-found error for multi-shard-key payload updates.",
    ),
    # ── enhancement ──────────────────────────────────────────────────────────
    10282: (
        "FEATURE_REQUEST",
        "MEDIUM",
        False,
        "Thanks for the detailed feature request — surfacing component scores as "
        "input to reranking would make prefetch and fusion pipelines far more "
        "transparent. I've passed it to the product team with your example and added "
        "your vote. We'll let you know if it gets scheduled.",
        "Request to surface prefetch component scores for reranking pipelines.",
    ),
    3839: (
        "FEATURE_REQUEST",
        "LOW",
        False,
        "Appreciate the reference to distributed tracing with OpenTelemetry — it's "
        "something we're tracking as a feature. I've added your ask to the "
        "observability roadmap with the text-embeddings-inference example, and we'll "
        "update you if it's scheduled.",
        "Enhancement request: OpenTelemetry distributed tracing support.",
    ),
    2950: (
        "FEATURE_REQUEST",
        "MEDIUM",
        False,
        "Thanks for describing the issue — querying non-indexed payload fields at "
        "scale is a real operational risk. I've forwarded this to the team as a "
        "configuration feature request and noted the production pain point you "
        "raised. We'll keep you posted.",
        "Request for config to disable queries on non-indexed payload fields.",
    ),
    2943: (
        "FEATURE_REQUEST",
        "MEDIUM",
        False,
        "Good question — fuzzy matching on text payload fields isn't currently "
        "supported, but it's a commonly requested enhancement. I've added your use "
        "case to the feature tracker and will let you know if it gets scheduled.",
        "Request for fuzzy-match support in payload filters.",
    ),
    1494: (
        "FEATURE_REQUEST",
        "LOW",
        False,
        "Thanks for the suggestion — we agree a Prometheus-style /metrics endpoint "
        "would help significantly over the JSON /telemetry output. I've logged it "
        "with the observability team as a feature request. We'll update you if it's "
        "planned.",
        "Request for an OpenTelemetry-friendly /metrics endpoint.",
    ),
    345: (
        "FEATURE_REQUEST",
        "LOW",
        False,
        "Thanks for the use cases — multi-vector-per-record is a long-requested "
        "feature that would cover multiple photos, aspect encoders, and crops. I've "
        "added your scenarios to the existing tracker. We'll let you know when it "
        "moves forward.",
        "Enhancement request: support multiple vectors per record.",
    ),
    1074: (
        "FEATURE_REQUEST",
        "MEDIUM",
        False,
        "Thanks for the report — exact literal phrase search isn't currently "
        "available, and I understand the frustration when a known document doesn't "
        "surface. I've added this to the search feature tracker with your example. "
        "We'll update here if it's scheduled.",
        "Request for exact/literal phrase search support.",
    ),
    3124: (
        "FEATURE_REQUEST",
        "LOW",
        False,
        "Thanks for bundling the outstanding sparse-vector tasks — this tracking "
        "issue is really useful for GA planning. I've linked it to our release "
        "planning and we'll use it to track progress. We'll post updates here as "
        "items land.",
        "Tracking issue for sparse vector GA.",
    ),
    18: (
        "FEATURE_REQUEST",
        "LOW",
        False,
        "Thanks for the nudge on the tokio version — keeping dependencies current is "
        "important for performance and security. I've filed the upgrade with our "
        "build team and we'll pick it up in a maintenance release. We'll update "
        "here.",
        "Enhancement: update tokio to the latest major release.",
    ),
    113: (
        "FEATURE_REQUEST",
        "MEDIUM",
        False,
        "Thanks for the request — filtering on score results is a useful search "
        "capability. I've added it to the feature tracker with the minimal/maximal "
        "score semantics you described. We'll let you know if it gets scheduled.",
        "Request for score-based filtering on search results.",
    ),
    32: (
        "USAGE_HELP",
        "LOW",
        False,
        "Thanks for the detailed note on the REST API documentation structure. "
        "Improving the endpoint layout would also improve the generated clients, so "
        "I've forwarded it to our docs and API teams. We'll update this thread if it "
        "gets taken up.",
        "Documentation request: restructure individual collection/point endpoints.",
    ),
    1327: (
        "FEATURE_REQUEST",
        "MEDIUM",
        False,
        "Thanks for the request — TTL for automatically removing time-expiring "
        "results is a frequently asked feature. I've added it to the roadmap tracker "
        "with your example. We'll keep you posted if it gets scheduled.",
        "Feature request: TTL to auto-remove time-expiring results.",
    ),
    2326: (
        "FEATURE_REQUEST",
        "LOW",
        False,
        "Thanks for the request — batching calls to the grouping APIs would help "
        "throughput in production pipelines. I've logged it with the API team as a "
        "feature request. We'll update you if it's planned.",
        "Request for batch requests across all grouping APIs.",
    ),
    2949: (
        "FEATURE_REQUEST",
        "MEDIUM",
        False,
        "Thanks for the suggestion — an audit log for runtime collection parameter "
        "updates would add valuable visibility into hardware-impacting changes. I've "
        "forwarded it to the observability and product teams. We'll let you know if "
        "it gets scheduled.",
        "Feature request: audit log for updated collection parameters.",
    ),
    # ── question ─────────────────────────────────────────────────────────────
    1678: (
        "USAGE_HELP",
        "LOW",
        False,
        "Great question — hybrid search combines a keyword/full-text search with a "
        "semantic search and then merges the results with a reranker. I'd point you "
        "to our hybrid search docs for the exact query construction, and I'm happy to "
        "walk through an example if that's useful.",
        "How-to question about hybrid search configuration.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Manual edge cases (fully specified here)
# ═══════════════════════════════════════════════════════════════════════════════

MANUAL_TICKETS: list[dict[str, Any]] = [
    {
        "content": "It doesn't work.",
        "expected_category": "BUG",
        "expected_urgency": "MEDIUM",
        "should_escalate": True,
        "human_approved_draft": (
            "Thanks for flagging this — we definitely want to get it working for you. "
            "Could you tell us a bit more about what isn't working, which page or feature "
            "it affects, and any error message you're seeing? The more detail you can "
            "share, the faster we can pinpoint the cause and fix it."
        ),
        "notes": "Deliberately vague — no feature, symptom, or error. Requires follow-up clarifying questions.",
    },
    {
        "content": (
            "After the last update, the billing page throws an error every time I open it, "
            "AND I just noticed I was double charged for this month's subscription. Please help — "
            "I need that refund and the page fixed."
        ),
        "expected_category": "BILLING",
        "expected_urgency": "HIGH",
        "should_escalate": True,
        "human_approved_draft": (
            "I'm sorry you're dealing with both a broken billing page and a duplicate charge — "
            "that's completely fair to escalate. We're treating the duplicate charge as a priority "
            "and will confirm the refund status within one business day. Separately, our engineering "
            "team is on the error on the billing page; could you confirm the exact error text you see?"
        ),
        "notes": "Multi-category (BUG + BILLING). Dominant impact is billing. Requires judgment to split the work.",
    },
    {
        "content": (
            "Hi, my account is locked and I need to verify it. My number is +1 (555) 014-8892 and "
            "my email is jane.meyer@example.com so you can reach me anytime to confirm my identity."
        ),
        "expected_category": "ACCOUNT_ISSUE",
        "expected_urgency": "HIGH",
        "should_escalate": True,
        "human_approved_draft": (
            "Thanks for reaching out about the locked account. Because your message includes personal "
            "details (a phone number and email), we won't act directly in this channel — a specialist "
            "will contact you over an official support link to verify your identity securely before "
            "restoring access."
        ),
        "notes": "Contains PII (phone + email). Must not echo details back; escalate for identity verification.",
    },
    {
        "content": (
            "Hola, no puedo iniciar sesión en mi cuenta desde ayer. Ya intenté restablecer mi "
            "contraseña pero no me llega el correo. Necesito acceso urgente para mi trabajo."
        ),
        "expected_category": "OTHER",
        "expected_urgency": "MEDIUM",
        "should_escalate": True,
        "human_approved_draft": (
            "Gracias por escribirnos. Lamentamos los problemas para acceder a su cuenta. "
            "Para ayudarle de forma segura, un agente bilingüe le contactará por correo. "
            "Mientras tanto, revise también su carpeta de spam por si el enlace de "
            "restablecimiento llegó allí."
        ),
        "notes": "Non-English ticket (Spanish). Language barrier → route to human/bilingual agent.",
    },
    {
        "content": "Broken",
        "expected_category": "BUG",
        "expected_urgency": "LOW",
        "should_escalate": True,
        "human_approved_draft": (
            "Thanks for writing in. 'Broken' doesn't give us much to go on — could you tell us "
            "what isn't working and where you're seeing it? Even a screenshot or the exact error "
            "message would help us get moving on a fix."
        ),
        "notes": "One-word ticket. No actionable signal; must solicit details.",
    },
    {
        "content": (
            "I'm writing because I've had recurring trouble with the export tool all quarter. It started "
            "around March 3rd when we upgraded. First, the CSV export for the 'Revenue' dashboard only "
            "captured the first 200 rows even though we had 5,000. Then the weekly scheduled report "
            "stopped arriving entirely for two weeks in April. I rebuilt the schedule, restarted the "
            "worker, cleared the cache per your docs, and even re-created the dashboard from scratch. "
            "The scheduled report came back after that but the CSV truncation persists — always exactly "
            "200 rows no matter what filter I apply. I tried exporting from three different browsers "
            "(Chrome, Firefox, Edge), on two machines, and with different date ranges. It also only "
            "exports the visible columns, dropping our four custom columns even when they're toggled on "
            "in the view. Our finance team relies on these files to reconcile the quarter; Rachel started "
            "a support thread on April 12 about the same truncation and followed all the same steps "
            "without luck. I've attached a redacted sample of the 200-row output and a screenshot of the "
            "column settings. Somewhere in here the export path is clearly broken, and I'd really like "
            "the underlying cause fixed rather than another round of cache-clearing, which we've already "
            "tried multiple times."
        ),
        "expected_category": "BUG",
        "expected_urgency": "MEDIUM",
        "should_escalate": True,
        "human_approved_draft": (
            "Thank you for the very detailed report — the 200-row CSV truncation and missing custom "
            "columns are both reproducible on our side, and we've opened a fix for the export path. "
            "Because you've already worked through clearing caches and rebuilding schedules without "
            "success, we're escalating this to our engineering team with your attached sample rather "
            "than asking you to retry those steps. We'll update you on the root-cause investigation "
            "within two business days."
        ),
        "notes": "Very long (2000+ chars), heavily pre-escalted context. Do not re-run standard steps.",
    },
    {
        "content": (
            "Three separate things: (1) Please refund the $49 charge from last week, it was a mistake. "
            "(2) Can you add a dark mode to the mobile app? Several of us have asked before. "
            "(3) Also, how do I invite a teammate to my workspace? I can't find it in the docs."
        ),
        "expected_category": "BILLING",
        "expected_urgency": "MEDIUM",
        "should_escalate": True,
        "human_approved_draft": (
            "Happy to help with all three! For the $49 charge, I've logged a refund request and will "
            "confirm once it's processed. On dark mode — it's on our roadmap and I've added your vote. "
            "And inviting teammates is under Workspace Settings → Members → Invite; I'd point you to "
            "the docs  but can walk you through it if you'd like."
        ),
        "notes": "Multiple distinct requests (BILLING + FEATURE_REQUEST + USAGE_HELP). Needs decomposition.",
    },
    {
        "content": (
            "ARE YOU KIDDING ME. Your app deleted my project and there is NO WAY to get it back. "
            "This is absolutely unacceptable, I've been a paying customer for 3 years and I'm done. "
            "Fix this NOW or I'm taking my business elsewhere. I want a response today, not your "
            "copy-paste canned crap."
        ),
        "expected_category": "BUG",
        "expected_urgency": "HIGH",
        "should_escalate": True,
        "human_approved_draft": (
            "I hear you — losing a project is frustrating and the inconvenience is on us. We're "
            "investigating the deletion right now and will check whether the project can be restored "
            "from backup (we do keep them). A senior agent is handling your account personally so this "
            "gets a real answer, and we'll follow up with you directly today."
        ),
        "notes": "Hostile/rude ticket. De-escalation + restore check requires human judgment; do not auto-draft rote apology only.",
    },
    {
        "content": (
            "Congratulations! You have been selected as our lucky user. To claim your $500 gift card, "
            "reply with your account password and banking details. Offer valid for the first 100 users!"
        ),
        "expected_category": "OTHER",
        "expected_urgency": "LOW",
        "should_escalate": True,
        "human_approved_draft": (
            "Thanks for your message. We're flagging this as a likely spam/phishing attempt — we will "
            "never ask for your password or banking details in chat. For your safety, we're immediately "
            "closing this channel and recommend you do not click any links from the original sender. "
            "Our security team will route this appropriately."
        ),
        "notes": "Spam/phishing. Outside taxonomy; needs safe handling, never respond to the 'request'.",
    },
    {
        "content": (
            "We're evaluating both you and AcmeCloud before renewing. AcmeCloud has one-click Kubernetes "
            "deployments and unlimited read replicas at our price point. Honestly, if you can't match "
            "that, we'll likely migrate. What would it take to get those features?"
        ),
        "expected_category": "FEATURE_REQUEST",
        "expected_urgency": "MEDIUM",
        "should_escalate": True,
        "human_approved_draft": (
            "Thanks for being upfront about the evaluation — competitive feedback like this is valuable "
            "to us. One-click Kubernetes deploys are on our roadmap, and I've routed your note to the "
            "product team with the read-replica pricing ask attached. A sales engineer will reach out "
            "this week to walk through what we can do for your workload."
        ),
        "notes": "References competitor product + retention/win-back pressure. Needs sales/product involvement.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# HF sampling
# ═══════════════════════════════════════════════════════════════════════════════


def _find_column(
    features,
    *candidates: str,
) -> str | None:
    """Return the first feature name that contains any candidate substring."""
    names = list(features) if hasattr(features, "__iter__") else []
    for name in names:
        low = name.lower()
        if any(cand in low for cand in candidates):
            return name
    return None


def load_hf_tickets(target: int = HF_TARGET) -> list[dict[str, Any]]:
    """Load, map, and sample `target` tickets from the HF dataset."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)
    df = ds["train"].to_pandas()

    cat_col = _find_column(df.columns, "topic", "type", "category", "class")
    if cat_col is None:
        # Fall back to scanning string columns for keyword-bearing values.
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(500)
                if sample.apply(
                    lambda v: any(
                        kw in v.lower()
                        for kw in [
                            "technical issue",
                            "feature request",
                            "billing",
                            "refund",
                            "product inquiry",
                            "how to",
                        ]
                    )
                ).any():
                    cat_col = col
                    break
    if cat_col is None:
        raise RuntimeError(f"Could not find a category column in {HF_DATASET}")

    # Identify subject/description columns (or a single content column).
    subj_col = _find_column(df.columns, "subject", "title")
    desc_col = _find_column(df.columns, "description", "body", "content", "message")
    if desc_col is None:
        # Pick the longest text column as the body.
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            raise RuntimeError("No usable content column found.")
        desc_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().max())
    if subj_col == desc_col:
        subj_col = None

    # Stable native id column if present (else the DataFrame index).
    id_candidates = ["_id", "id", "ticket_id", "Unnamed: 0", "index"]
    id_col = _find_column(df.columns, *[c.lower() for c in id_candidates])

    def _content_of(row: Any) -> str:
        parts: list[str] = []
        if subj_col is not None and str(row[subj_col]) not in ("", "nan"):
            parts.append(str(row[subj_col]))
        if str(row[desc_col]) not in ("", "nan"):
            parts.append(str(row[desc_col]))
        return "\n".join(parts).strip()

    # Precompute (native_id, category, content) — skipping rows too short to label.
    rows_meta: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        content = _content_of(row)
        if len(content) < MIN_TICKET_CHARS:
            continue
        native_id = str(row[id_col]) if id_col else str(idx)
        rows_meta.append(
            {
                "native_id": native_id,
                "category": map_hf_category(str(row[cat_col])),
                "content": content[:MAX_TICKET_CHARS],
            }
        )

    if len(rows_meta) < target:
        raise RuntimeError(f"Only {len(rows_meta)} usable rows in {HF_DATASET} — need {target}.")

    # Bucket by mapped category (row position in ``rows_meta``).
    buckets: dict[str, list[int]] = {c: [] for c in CATEGORIES}
    for pos, meta in enumerate(rows_meta):
        buckets[meta["category"]].append(pos)

    rng = random.Random(RNG_SEED)
    chosen_positions: list[tuple[int, str]] = []
    chosen_pool: set[int] = set()

    # Quota sample per category (10 each), then fill the rest from leftovers.
    for cat in CATEGORIES:
        quota = PER_CATEGORY_QUOTA
        pool = [p for p in buckets[cat] if p not in chosen_pool]
        for p in rng.sample(pool, min(quota, len(pool))):
            chosen_pool.add(p)
            chosen_positions.append((p, cat))
    leftover = [p for cat in CATEGORIES for p in buckets[cat] if p not in chosen_pool]
    for p in rng.sample(leftover, min(target - len(chosen_positions), len(leftover))):
        chosen_pool.add(p)
        chosen_positions.append((p, rows_meta[p]["category"]))

    if len(chosen_positions) < target:
        raise RuntimeError(
            f"Only {len(chosen_positions)} samples available in {HF_DATASET} — need {target}."
        )
    chosen_positions = chosen_positions[:target]

    return [
        {
            "native_id": rows_meta[p]["native_id"],
            "category": cat,
            "content": rows_meta[p]["content"],
        }
        for p, cat in chosen_positions
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# GitHub sampling
# ═══════════════════════════════════════════════════════════════════════════════


def _gh_request(url: str) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": "eval-fixture-builder",
        "Accept": "application/vnd.github+json",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — trusted public API
        return json.loads(resp.read().decode("utf-8"))


def load_github_issues(target: int = GH_TARGET) -> list[dict[str, Any]]:
    """Fetch issues from the configured repo and sample `target` of them."""
    from_label: dict[str, list[dict[str, Any]]] = {}
    for label in GH_LABEL_CATEGORIES:
        url = f"https://api.github.com/repos/{GH_REPO}/issues?state=all&labels={label}&per_page=100"
        issues = _gh_request(url)
        # Exclude pull requests and issues without a substantive body.
        body_issues = [
            i
            for i in issues
            if "pull_request" not in i
            and (i.get("body") or "").strip()
            and len((i.get("body") or "").strip()) >= 40
        ]
        from_label[label] = body_issues

    rng = random.Random(RNG_SEED)
    # Target quotas per label, proportional to availability.
    total_avail = sum(len(v) for v in from_label.values())
    if total_avail == 0:
        raise RuntimeError(f"No usable issues found in {GH_REPO}.")
    quotas: dict[str, int] = {
        label: max(1, round(target * len(items) / total_avail))
        for label, items in from_label.items()
    }
    # Tune so the sum equals the target.
    while sum(quotas.values()) < target:
        quotas[rng.choice(list(from_label.keys()))] += 1
    while sum(quotas.values()) > target:
        biggest = max(from_label.keys(), key=lambda label: quotas[label])
        if quotas[biggest] > 0:
            quotas[biggest] -= 1

    picked: list[dict[str, Any]] = []
    for label in GH_LABEL_CATEGORIES:
        pool = from_label[label]
        chosen = rng.sample(pool, min(quotas[label], len(pool)))
        picked.extend(chosen)

    # De-duplicate: an issue tagged with several sampled labels (e.g. both
    # "documentation" and "enhancement") can otherwise be picked twice.
    picked = list({i["number"]: i for i in picked}.values())

    # Backfill up to the target from the remaining body-bearing issues so the
    # fixture always has `target` unique rows, regardless of label overlap.
    if len(picked) < target:
        seen = {i["number"] for i in picked}
        remainder = [
            i for label in GH_LABEL_CATEGORIES for i in from_label[label] if i["number"] not in seen
        ]
        # Keep deterministic order; reshuffle remainder deterministically.
        rng.shuffle(remainder)
        picked.extend(remainder[: target - len(picked)])
        picked = list({i["number"]: i for i in picked}.values())

    if len(picked) < target:
        raise RuntimeError(f"Only {len(picked)} sampled issues — need {target}.")
    picked = picked[:target]

    rows: list[dict[str, Any]] = []
    for issue in picked:
        default_cat = {
            "bug": "BUG",
            "enhancement": "FEATURE_REQUEST",
            "question": "USAGE_HELP",
            "documentation": "USAGE_HELP",
        }.get((issue.get("labels") or [{}])[0].get("name", "").lower(), "OTHER")
        title = issue.get("title", "")
        body = issue.get("body", "")
        content = f"{title}\n\n{body}".strip()[:MAX_TICKET_CHARS]
        rows.append({"number": int(issue["number"]), "category": default_cat, "content": content})
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture assembly
# ═══════════════════════════════════════════════════════════════════════════════


def build_fixture() -> list[dict[str, Any]]:
    """Assemble all 100 labeled tickets (mapping HF/GH samples + manual cases)."""
    tickets: list[dict[str, Any]] = []
    ticket_num = 0

    # --- HuggingFace (60) -----------------------------------------------------
    hf_rows = load_hf_tickets(HF_TARGET)
    for row in hf_rows:
        label = HF_LABELS.get(row["native_id"])
        if label is None:
            raise KeyError(
                f"No hand label for HF ticket native_id={row['native_id']!r} — "
                f"add an entry to HF_LABELS in {__file__}."
            )
        urgency, should_escalate, draft, notes = label
        ticket_num += 1
        tickets.append(
            {
                "id": f"test-{ticket_num:03d}",
                "content": row["content"],
                "source": "huggingface",
                "source_id": row["native_id"],
                "expected_category": row["category"],
                "expected_urgency": urgency,
                "should_escalate": should_escalate,
                "human_approved_draft": draft,
                "notes": notes,
            }
        )

    # --- GitHub (30) ----------------------------------------------------------
    gh_rows = load_github_issues(GH_TARGET)
    for row in gh_rows:
        label = GH_TICKET_LABELS.get(row["number"])
        if label is None:
            raise KeyError(
                f"No hand label for GitHub issue #{row['number']} — "
                f"add an entry to GH_TICKET_LABELS in {__file__}."
            )
        category, urgency, should_escalate, draft, notes = label
        ticket_num += 1
        tickets.append(
            {
                "id": f"test-{ticket_num:03d}",
                "content": row["content"],
                "source": "github",
                "source_id": f"#{row['number']}",
                "expected_category": category,
                "expected_urgency": urgency,
                "should_escalate": should_escalate,
                "human_approved_draft": draft,
                "notes": notes,
            }
        )

    # --- Manual (10) ----------------------------------------------------------
    for ticket in MANUAL_TICKETS:
        ticket_num += 1
        tickets.append(
            {
                "id": f"test-{ticket_num:03d}",
                "content": ticket["content"],
                "source": "manual",
                "source_id": None,
                "expected_category": ticket["expected_category"],
                "expected_urgency": ticket["expected_urgency"],
                "should_escalate": ticket["should_escalate"],
                "human_approved_draft": ticket["human_approved_draft"],
                "notes": ticket.get("notes", ""),
            }
        )

    if ticket_num != 100:
        raise RuntimeError(f"Expected 100 tickets, built {ticket_num}.")
    return tickets


def write_fixture(tickets: list[dict[str, Any]], path: Path = OUTPUT_PATH) -> Path:
    """Write the fixture to JSONL (one record per line, UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ticket in tickets:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    return path


def check_fixture(path: Path = OUTPUT_PATH) -> int:
    """Validate the checked-in fixture: 100 records, valid enums, non-empty drafts."""
    from eval.harness import EvalHarness

    tickets = EvalHarness(fixture_path=path).load_tickets()
    problems: list[str] = []
    if len(tickets) != 100:
        problems.append(f"expected 100 tickets, got {len(tickets)}")

    seen_ids: set[str] = set()
    long_drafts = 0
    for t in tickets:
        if t["id"] in seen_ids:
            problems.append(f"duplicate id {t['id']}")
        seen_ids.add(t["id"])
        if t["expected_category"] not in CATEGORIES:
            problems.append(f"{t['id']}: bad category {t['expected_category']}")
        if t["expected_urgency"] not in URGENCIES:
            problems.append(f"{t['id']}: bad urgency {t['expected_urgency']}")
        if t.get("source") not in ("huggingface", "github", "manual"):
            problems.append(f"{t['id']}: bad source {t.get('source')}")
        draft = t.get("human_approved_draft", "")
        words = len(draft.split())
        if words < 15:
            problems.append(f"{t['id']}: draft too short ({words} words)")
        if words >= 120:
            long_drafts += 1
        if not (t.get("content") or "").strip():
            problems.append(f"{t['id']}: empty content")

    sources = ("huggingface", "github", "manual")
    counts = {s: sum(1 for t in tickets if t["source"] == s) for s in sources}
    if counts["huggingface"] != 60:
        problems.append(f"huggingface count {counts['huggingface']} != 60")
    if counts["github"] != 30:
        problems.append(f"github count {counts['github']} != 30")
    if counts["manual"] != 10:
        problems.append(f"manual count {counts['manual']} != 10")

    if problems:
        print("[build_fixture] fixture INVALID:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(
        f"[build_fixture] fixture OK — {len(tickets)} tickets "
        f"(hf={counts['huggingface']}, gh={counts['github']}, manual={counts['manual']})"
    )
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check_fixture()
    if "--only-check" in sys.argv:
        return check_fixture()
    tickets = build_fixture()
    written = write_fixture(tickets)
    print(f"[build_fixture] wrote {len(tickets)} tickets → {written}")
    summary: dict[str, dict[str, int]] = {}
    for t in tickets:
        summary.setdefault(t["source"], {}).setdefault(t["expected_category"], 0)
        summary[t["source"]][t["expected_category"]] += 1
    for src, cats in summary.items():
        print(f"  {src:12s} " + ", ".join(f"{c}={n}" for c, n in sorted(cats.items())))
    return check_fixture(written)


if __name__ == "__main__":
    raise SystemExit(main())
