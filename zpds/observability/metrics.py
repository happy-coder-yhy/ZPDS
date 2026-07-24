"""从 Run Ledger 生成可持久化运行指标。"""

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from zpds.pipeline.ledger import RunLedgerSnapshot
from zpds.pipeline.stage import StageStatus
from zpds.storage import LocalStorage, StoredFile
from zpds.utils.schema_validator import validate_with_schema


def build_run_metrics(snapshot: RunLedgerSnapshot) -> dict[str, Any]:
    """将 Ledger 快照汇总为稳定、可审计的指标对象。"""

    stage_statuses = Counter(entry.status.value for entry in snapshot.stages)
    decisions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    severities: Counter[str] = Counter()
    duration_seconds = 0.0
    for entry in snapshot.stages:
        if entry.result is None:
            continue
        duration_seconds += entry.result.duration_seconds
        for decision in entry.result.decisions:
            decisions[decision.decision.value] += 1
            reasons[decision.reason.value] += 1
            severities[decision.severity.value] += 1
    attempts = sum(entry.attempts for entry in snapshot.stages)
    retries = sum(max(entry.attempts - 1, 0) for entry in snapshot.stages)
    return {
        "zpds_version": "0.1.0",
        "run_id": snapshot.run_id,
        "session_id": snapshot.session_id,
        "status": snapshot.status.value,
        "config_hash": snapshot.config_hash,
        "code_version": snapshot.code_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "total": len(snapshot.stages),
            "pending": stage_statuses[StageStatus.PENDING.value],
            "running": stage_statuses[StageStatus.RUNNING.value],
            "succeeded": stage_statuses[StageStatus.SUCCEEDED.value],
            "failed": stage_statuses[StageStatus.FAILED.value],
            "skipped": stage_statuses[StageStatus.SKIPPED.value],
        },
        "attempts": attempts,
        "retries": retries,
        "duration_seconds": duration_seconds,
        "decisions_by_type": dict(sorted(decisions.items())),
        "reason_codes": dict(sorted(reasons.items())),
        "severities": dict(sorted(severities.items())),
    }


def persist_run_metrics(
    storage: LocalStorage,
    snapshot: RunLedgerSnapshot,
) -> StoredFile:
    """原子写入 runs/<run_id>/metrics.json。"""

    value = build_run_metrics(snapshot)
    return storage.atomic_write_json(
        f"artifact://runs/{snapshot.run_id}/metrics.json",
        value,
        validator=_validate_metrics,
    )


def _validate_metrics(value: dict[str, Any]) -> None:
    errors = validate_with_schema(value, "run_metrics")
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid run metrics:\n{details}")
