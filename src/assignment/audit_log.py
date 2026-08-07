"""Thread-safe request-correlated audit logging for security review."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
import time
from uuid import uuid4


class AuditLogPlugin:
    """Observe each request from input through output/action without blocking it."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._pending: dict[str, dict] = {}
        self._open = self._pending
        self._lock = Lock()

    def record_input(
        self,
        *,
        user_id: str,
        text: str,
        request_id: str | None = None,
        source: str = "user",
    ) -> str:
        """Open one correlated request and return its stable request ID."""
        rid = request_id or uuid4().hex
        pending = {
            "request_id": rid,
            "timestamp": utc_now_iso(),
            "user_id": user_id or "anonymous",
            "input_text": text or "",
            "source": source,
            "started_at": time.perf_counter(),
        }
        with self._lock:
            if rid in self._pending:
                raise ValueError(f"Duplicate in-flight request_id: {rid}")
            self._pending[rid] = pending
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        block_reason: str | None = None,
        plugins_triggered: list[str] | None = None,
        error: str | None = None,
        action_intent: str | None = None,
        action_destination: str | None = None,
        proposed_diff: str | None = None,
        approval_id: str | None = None,
        reviewer_id: str | None = None,
        reviewer_decision: str | None = None,
    ) -> dict:
        """Close a request and append a forensically useful JSON record."""
        rid = request_id
        with self._lock:
            if rid is None:
                candidates = [
                    key for key, item in self._pending.items()
                    if item["user_id"] == (user_id or "anonymous")
                ]
                if len(candidates) == 1:
                    rid = candidates[0]
            pending = self._pending.pop(rid, None) if rid else None
            finished = time.perf_counter()
            if pending is None:
                rid = rid or uuid4().hex
                pending = {
                    "request_id": rid,
                    "timestamp": utc_now_iso(),
                    "user_id": user_id or "anonymous",
                    "input_text": "",
                    "source": "unknown",
                    "started_at": finished,
                }

            triggered = list(dict.fromkeys(plugins_triggered or []))
            if layer and layer not in triggered:
                triggered.append(layer)

            entry = {
                "request_id": rid,
                "timestamp": pending["timestamp"],
                "user_id": pending["user_id"],
                "source": pending["source"],
                "input_text": pending["input_text"],
                "output_text": text or "",
                "plugins_triggered": triggered,
                "blocked": bool(blocked),
                "block_layer": layer,
                "block_reason": block_reason,
                "latency_ms": round(
                    max(0.0, finished - float(pending["started_at"])) * 1000, 3
                ),
                "error": error,
                "action": {
                    "intent": action_intent,
                    "destination": action_destination,
                    "proposed_diff": proposed_diff,
                    "approval_id": approval_id,
                    "reviewer_id": reviewer_id,
                    "reviewer_decision": reviewer_decision,
                } if any((
                    action_intent, action_destination, proposed_diff, approval_id,
                    reviewer_id, reviewer_decision,
                )) else None,
            }
            self.logs.append(entry)
        return entry

    def export_json(self, filepath: str = "outputs/audit_log.json") -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = list(self.logs)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_summary(self) -> dict:
        with self._lock:
            logs = list(self.logs)
            open_requests = len(self._pending)
        blocked = [entry for entry in logs if entry.get("blocked")]
        latencies = [float(entry.get("latency_ms", 0.0)) for entry in logs]
        reasons = Counter(
            entry.get("block_reason") or entry.get("block_layer")
            for entry in blocked
            if entry.get("block_reason") or entry.get("block_layer")
        )
        return {
            "total_requests": len(logs),
            "blocked_count": len(blocked),
            "block_rate": len(blocked) / len(logs) if logs else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "most_common_block_reason": reasons.most_common(1)[0][0] if reasons else None,
            "open_requests": open_requests,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
