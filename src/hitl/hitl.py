"""Human-in-the-loop routing and an auditable review lifecycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from uuid import uuid4

HIGH_RISK_ACTIONS = frozenset({
    "transfer_money", "close_account", "change_password",
    "delete_data", "update_personal_info",
})


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    confidence: float
    reason: str
    priority: str
    requires_human: bool


class ConfidenceRouter:
    """Route by action risk first, then by confidence."""

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(
        self,
        response: str,
        confidence: float,
        action_type: str = "general",
    ) -> RoutingDecision:
        del response
        try:
            score = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be a number between 0 and 1") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        normalized_action = (action_type or "general").strip().casefold()
        if normalized_action in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                "escalate", score, f"High-risk action: {normalized_action}",
                "high", True,
            )
        if score >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                "auto_send", score, "High confidence and low-risk action.",
                "low", False,
            )
        if score >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                "queue_review", score,
                "Medium confidence; human review required before sending.",
                "normal", True,
            )
        return RoutingDecision(
            "escalate", score,
            "Low confidence; immediate human escalation required.",
            "high", True,
        )


@dataclass(frozen=True)
class ReviewRequest:
    """Everything a reviewer needs to judge one proposed side effect."""

    correlation_id: str
    intent: str
    action_type: str
    current_state: str
    proposed_state: str
    proposed_diff: str
    destination: str
    payload_summary: str
    risk_reason: str
    created_at: str
    expires_at: str


@dataclass(frozen=True)
class ReviewDecision:
    correlation_id: str
    status: str
    reviewer_id: str | None
    approval_id: str | None
    reason: str
    decided_at: str


@dataclass
class HITLReviewQueue:
    """In-memory demonstration of approve/reject/timeout with fail-closed semantics."""

    default_timeout_seconds: int = 300
    pending: dict[str, ReviewRequest] = field(default_factory=dict)
    decisions: list[ReviewDecision] = field(default_factory=list)

    def submit(
        self,
        *,
        intent: str,
        action_type: str,
        current_state: str,
        proposed_state: str,
        proposed_diff: str,
        destination: str,
        payload_summary: str,
        risk_reason: str,
        correlation_id: str | None = None,
        timeout_seconds: int | None = None,
    ) -> ReviewRequest:
        timeout = timeout_seconds or self.default_timeout_seconds
        if timeout < 1:
            raise ValueError("timeout_seconds must be at least 1")
        now = datetime.now(timezone.utc)
        cid = correlation_id or uuid4().hex
        if cid in self.pending:
            raise ValueError(f"duplicate pending correlation_id: {cid}")
        request = ReviewRequest(
            correlation_id=cid,
            intent=intent,
            action_type=action_type,
            current_state=current_state,
            proposed_state=proposed_state,
            proposed_diff=proposed_diff,
            destination=destination,
            payload_summary=payload_summary,
            risk_reason=risk_reason,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=timeout)).isoformat(),
        )
        self.pending[cid] = request
        return request

    def decide(
        self,
        correlation_id: str,
        *,
        approve: bool,
        reviewer_id: str,
        reason: str,
    ) -> ReviewDecision:
        request = self.pending.pop(correlation_id, None)
        if request is None:
            raise KeyError(f"no pending review: {correlation_id}")
        if not reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        status = "approved" if approve else "rejected"
        approval_id = f"HITL-{uuid4().hex[:8].upper()}" if approve else None
        decision = ReviewDecision(
            correlation_id=correlation_id,
            status=status,
            reviewer_id=reviewer_id,
            approval_id=approval_id,
            reason=reason,
            decided_at=datetime.now(timezone.utc).isoformat(),
        )
        self.decisions.append(decision)
        return decision

    def expire(self, *, now: datetime | None = None) -> list[ReviewDecision]:
        """Convert expired requests to ``timed_out`` decisions; never auto-approve."""
        current = now or datetime.now(timezone.utc)
        expired: list[ReviewDecision] = []
        for cid, request in list(self.pending.items()):
            if datetime.fromisoformat(request.expires_at) <= current:
                self.pending.pop(cid)
                decision = ReviewDecision(
                    correlation_id=cid,
                    status="timed_out",
                    reviewer_id=None,
                    approval_id=None,
                    reason="Review timeout; action denied fail-closed.",
                    decided_at=current.isoformat(),
                )
                self.decisions.append(decision)
                expired.append(decision)
        return expired


hitl_decision_points = [
    {
        "id": 1,
        "name": "High-value or unusual transfer approval",
        "trigger": "Transfer is high value, uses a new payee, or fraud rules flag it.",
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Verified customer identity, amount, payee history, device/IP risk, "
            "current balance, model rationale, destination, and payload summary."
        ),
        "example": "A 200,000,000 VND transfer to a first-time payee from a new device.",
        "approval_path": (
            "Approve creates a one-use HITL approval ID; reject cancels the action; "
            "timeout denies it fail-closed and asks the customer to contact support."
        ),
        "audit_fields": (
            "request/correlation ID, user intent, current state, proposed state/diff, "
            "destination, reviewer ID, approval ID, decision, reason, timestamps."
        ),
    },
    {
        "id": 2,
        "name": "Account closure or destructive data action",
        "trigger": "User requests account closure, data deletion, or another irreversible change.",
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Identity verification, balance/loan obligations, pending disputes, legal "
            "retention requirements, current state, and exact destructive diff."
        ),
        "example": "The assistant recognizes 'close my account now' while a loan is active.",
        "approval_path": (
            "Approve only after identity and obligations checks; reject preserves state; "
            "timeout preserves state and records no side effect."
        ),
        "audit_fields": (
            "correlation ID, intent, account-state snapshot, proposed deletion diff, "
            "reviewer ID, decision, reason, approval ID, expiry."
        ),
    },
    {
        "id": 3,
        "name": "Ambiguous policy answer or exception",
        "trigger": "Confidence is 0.7–0.9, sources conflict, or a policy exception is requested.",
        "hitl_model": "human-as-tiebreaker",
        "context_needed": (
            "Customer question, provenance-tagged policy passages, candidate answer, "
            "confidence scores, conflicts, and before/after answer diff."
        ),
        "example": "Two policy documents disagree about an early repayment fee.",
        "approval_path": (
            "Approve sends the reviewed answer; reject returns it for correction; timeout "
            "sends no answer and escalates to operations."
        ),
        "audit_fields": (
            "correlation ID, source provenance, intent, candidate/final diff, reviewer ID, "
            "decision, reason, and timestamps."
        ),
    },
]


def test_confidence_router() -> None:
    router = ConfidenceRouter()
    assert router.route("ok", 0.95, "general").action == "auto_send"
    assert router.route("ok", 0.8, "general").action == "queue_review"
    assert router.route("ok", 0.5, "general").action == "escalate"
    assert router.route("ok", 0.99, "transfer_money").action == "escalate"


def test_hitl_points() -> None:
    required = {
        "name", "trigger", "hitl_model", "context_needed", "example",
        "approval_path", "audit_fields",
    }
    assert len(hitl_decision_points) >= 3
    assert all(required <= point.keys() for point in hitl_decision_points)
