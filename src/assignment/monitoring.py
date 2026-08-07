"""Monitoring counters, incident alerts, and replayable metric snapshots."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        for name in ("block_rate_threshold", "judge_fail_rate_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.rate_limit_hit_threshold < 1:
            raise ValueError("rate_limit_hit_threshold must be at least 1")

    def record_result(
        self,
        *,
        blocked: bool,
        layer: str | None = None,
        judge_checked: bool = False,
        judge_failed: bool = False,
        error: bool = False,
    ) -> None:
        """Update counters exactly once for a completed request."""
        self.total_requests += 1
        self.blocked_requests += int(bool(blocked))
        self.rate_limit_hits += int(layer == "rate_limiter")
        self.judge_checks += int(bool(judge_checked))
        self.judge_fails += int(bool(judge_checked and judge_failed))
        self.error_count += int(bool(error))

    def check_metrics(self) -> list[Alert]:
        """Recompute a fresh alert set, avoiding duplicate incident alerts."""
        self.alerts = []
        snapshot = self.snapshot(include_alerts=False)
        if self.total_requests and snapshot["block_rate"] >= self.block_rate_threshold:
            self.alerts.append(Alert(
                "block_rate", snapshot["block_rate"], self.block_rate_threshold,
                "High request block rate detected.",
            ))
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            self.alerts.append(Alert(
                "rate_limit_hits", float(self.rate_limit_hits),
                float(self.rate_limit_hit_threshold),
                "Repeated rate-limit activity detected.",
            ))
        if self.judge_checks and snapshot["judge_fail_rate"] >= self.judge_fail_rate_threshold:
            self.alerts.append(Alert(
                "judge_fail_rate", snapshot["judge_fail_rate"],
                self.judge_fail_rate_threshold,
                "LLM judge failure rate is above threshold.",
            ))
        return list(self.alerts)

    def snapshot(self, *, include_alerts: bool = True) -> dict:
        block_rate = self.blocked_requests / self.total_requests if self.total_requests else 0.0
        judge_fail_rate = self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        payload = {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "error_count": self.error_count,
        }
        if include_alerts:
            payload["alerts"] = [
                {"metric": a.metric, "value": a.value, "threshold": a.threshold,
                 "message": a.message}
                for a in self.alerts
            ]
        return payload

    def replay_snapshot(self, snapshot: dict) -> list[Alert]:
        """Re-evaluate a stored metric snapshot for incident-response exercises."""
        required = {
            "total_requests", "blocked_requests", "rate_limit_hits",
            "judge_checks", "judge_fails",
        }
        if not required <= snapshot.keys():
            raise ValueError("snapshot is missing required counters")
        clone = MonitoringAlert(
            block_rate_threshold=self.block_rate_threshold,
            rate_limit_hit_threshold=self.rate_limit_hit_threshold,
            judge_fail_rate_threshold=self.judge_fail_rate_threshold,
        )
        for key in required | {"error_count"}:
            if key in snapshot:
                setattr(clone, key, int(snapshot[key]))
        return clone.check_metrics()

    def export_json(self, filepath: str = "outputs/metrics.json") -> None:
        self.check_metrics()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
