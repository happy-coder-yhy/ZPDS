"""原子持久化的本地 Pipeline Run Ledger。"""

import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from zpds.core.decisions import Decision, DecisionType, Evidence, ReasonCode, Severity
from zpds.core.quality import MetricDirection, QualityMetric
from zpds.storage import LocalStorage
from zpds.utils.schema_validator import validate_with_schema

from .stage import StageContext, StageDescriptor, StageResult, StageStatus

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class LedgerError(RuntimeError):
    """Ledger 读取、状态迁移或持久化失败。"""


class LedgerConflictError(LedgerError):
    """同一 run_id 被用于不同输入、配置、代码或 Stage 集合。"""


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class StageLedgerEntry:
    descriptor: StageDescriptor
    execution_key: str | None
    status: StageStatus
    attempts: int
    input_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    started_at: datetime | None
    finished_at: datetime | None
    last_error: str | None
    result: StageResult | None


@dataclass(frozen=True)
class RunLedgerSnapshot:
    run_id: str
    session_id: str
    config_hash: str
    code_version: str
    input_refs: tuple[str, ...]
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    stages: tuple[StageLedgerEntry, ...]

    def stage(self, stage_id: int) -> StageLedgerEntry:
        for entry in self.stages:
            if entry.descriptor.stage_id == stage_id:
                return entry
        raise KeyError(f"Stage not present in ledger: {stage_id}")


class FileRunLedger:
    """每个 run 使用一个 JSON 文件，所有更新经临时文件原子替换。"""

    def __init__(self, storage: LocalStorage) -> None:
        self._storage = storage
        self._lock = threading.RLock()

    def initialize(
        self,
        context: StageContext,
        descriptors: tuple[StageDescriptor, ...],
    ) -> RunLedgerSnapshot:
        if not RUN_ID_PATTERN.fullmatch(context.run_id):
            raise ValueError("run_id must contain only letters, digits, dot, underscore, or dash")
        if not descriptors:
            raise ValueError("descriptors must not be empty")
        reference = self._reference(context.run_id)
        with self._lock:
            if self._storage.exists(reference):
                data = self._load(context.run_id)
                self._assert_identity(data, context, descriptors)
                return _snapshot_from_data(data)
            now = _utc_now().isoformat()
            data = {
                "zpds_version": "0.1.0",
                "run_id": context.run_id,
                "session_id": context.session_id,
                "config_hash": context.config.config_hash,
                "code_version": context.code_version,
                "input_refs": list(context.input_refs),
                "status": RunStatus.PENDING.value,
                "created_at": now,
                "updated_at": now,
                "stages": [_pending_stage_data(descriptor) for descriptor in descriptors],
            }
            self._save(data)
            return _snapshot_from_data(data)

    def completed_result(
        self,
        run_id: str,
        descriptor: StageDescriptor,
        execution_key: str,
    ) -> StageResult | None:
        with self._lock:
            data = self._load(run_id)
            entry = _find_stage_data(data, descriptor.stage_id)
            if (
                entry["execution_key"] == execution_key
                and entry["status"]
                in {StageStatus.SUCCEEDED.value, StageStatus.SKIPPED.value}
                and isinstance(entry["result"], dict)
            ):
                return _result_from_data(entry["result"])
            return None

    def begin_attempt(
        self,
        run_id: str,
        descriptor: StageDescriptor,
        execution_key: str,
        input_refs: tuple[str, ...],
        *,
        started_at: datetime,
    ) -> int:
        with self._lock:
            data = self._load(run_id)
            entry = _find_stage_data(data, descriptor.stage_id)
            if entry["execution_key"] not in {None, execution_key}:
                if entry["status"] in {
                    StageStatus.SUCCEEDED.value,
                    StageStatus.SKIPPED.value,
                }:
                    raise LedgerConflictError(
                        f"Completed stage {descriptor.stage_id} has a different execution key"
                    )
                entry.update(_pending_stage_data(descriptor))
            entry["execution_key"] = execution_key
            entry["status"] = StageStatus.RUNNING.value
            entry["attempts"] = int(entry["attempts"]) + 1
            entry["input_refs"] = list(input_refs)
            entry["output_refs"] = []
            entry["started_at"] = started_at.isoformat()
            entry["finished_at"] = None
            entry["last_error"] = None
            entry["result"] = None
            data["status"] = RunStatus.RUNNING.value
            data["updated_at"] = _utc_now().isoformat()
            self._save(data)
            return int(entry["attempts"])

    def record_result(
        self,
        run_id: str,
        execution_key: str,
        result: StageResult,
    ) -> RunLedgerSnapshot:
        with self._lock:
            data = self._load(run_id)
            entry = _find_stage_data(data, result.descriptor.stage_id)
            if entry["execution_key"] != execution_key:
                raise LedgerConflictError(
                    f"Execution key changed for stage {result.descriptor.stage_id}"
                )
            if entry["status"] != StageStatus.RUNNING.value:
                raise LedgerError(
                    f"Stage {result.descriptor.stage_id} is not running"
                )
            if tuple(entry["input_refs"]) != result.input_refs:
                raise LedgerConflictError(
                    f"Stage {result.descriptor.stage_id} result input_refs changed"
                )
            if result.config_hash != data["config_hash"]:
                raise LedgerConflictError(
                    f"Stage {result.descriptor.stage_id} result config_hash changed"
                )
            entry["status"] = result.status.value
            entry["output_refs"] = list(result.output_refs)
            entry["started_at"] = result.started_at.isoformat()
            entry["finished_at"] = result.finished_at.isoformat()
            entry["last_error"] = result.error
            entry["result"] = _result_to_data(result)
            data["updated_at"] = _utc_now().isoformat()
            data["status"] = _derive_run_status(data).value
            self._save(data)
            return _snapshot_from_data(data)

    def snapshot(self, run_id: str) -> RunLedgerSnapshot:
        with self._lock:
            return _snapshot_from_data(self._load(run_id))

    @staticmethod
    def _reference(run_id: str) -> str:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("invalid run_id")
        return f"artifact://runs/{run_id}/ledger.json"

    def _load(self, run_id: str) -> dict[str, Any]:
        reference = self._reference(run_id)
        if not self._storage.exists(reference):
            raise LedgerError(f"Run ledger not found: {run_id}")
        data = self._storage.read_json(reference)
        _validate_ledger(data)
        return data

    def _save(self, data: dict[str, Any]) -> None:
        reference = self._reference(str(data["run_id"]))
        self._storage.atomic_write_json(reference, data, validator=_validate_ledger)

    @staticmethod
    def _assert_identity(
        data: dict[str, Any],
        context: StageContext,
        descriptors: tuple[StageDescriptor, ...],
    ) -> None:
        expected = {
            "session_id": context.session_id,
            "config_hash": context.config.config_hash,
            "code_version": context.code_version,
            "input_refs": list(context.input_refs),
        }
        mismatches = [
            field for field, value in expected.items() if data.get(field) != value
        ]
        stored_descriptors = [entry["descriptor"] for entry in data["stages"]]
        expected_descriptors = [_descriptor_to_data(item) for item in descriptors]
        if stored_descriptors != expected_descriptors:
            mismatches.append("stages")
        if mismatches:
            raise LedgerConflictError(
                f"run_id {context.run_id!r} conflicts on: {', '.join(mismatches)}"
            )


def _validate_ledger(data: dict[str, Any]) -> None:
    errors = validate_with_schema(data, "run_ledger")
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise LedgerError(f"Invalid run ledger:\n{details}")


def _derive_run_status(data: dict[str, Any]) -> RunStatus:
    statuses = {entry["status"] for entry in data["stages"]}
    if StageStatus.FAILED.value in statuses:
        return RunStatus.FAILED
    terminal = {StageStatus.SUCCEEDED.value, StageStatus.SKIPPED.value}
    if statuses and statuses <= terminal:
        return RunStatus.SUCCEEDED
    if StageStatus.RUNNING.value in statuses or statuses & terminal:
        return RunStatus.RUNNING
    return RunStatus.PENDING


def _find_stage_data(data: dict[str, Any], stage_id: int) -> dict[str, Any]:
    for entry in data["stages"]:
        if entry["descriptor"]["stage_id"] == stage_id:
            return entry
    raise LedgerError(f"Stage not present in ledger: {stage_id}")


def _pending_stage_data(descriptor: StageDescriptor) -> dict[str, Any]:
    return {
        "descriptor": _descriptor_to_data(descriptor),
        "execution_key": None,
        "status": StageStatus.PENDING.value,
        "attempts": 0,
        "input_refs": [],
        "output_refs": [],
        "started_at": None,
        "finished_at": None,
        "last_error": None,
        "result": None,
    }


def _descriptor_to_data(descriptor: StageDescriptor) -> dict[str, Any]:
    return {
        "stage_id": descriptor.stage_id,
        "name": descriptor.name,
        "version": descriptor.version,
    }


def _descriptor_from_data(data: dict[str, Any]) -> StageDescriptor:
    return StageDescriptor(
        stage_id=int(data["stage_id"]),
        name=str(data["name"]),
        version=str(data["version"]),
    )


def _evidence_to_data(evidence: Evidence) -> dict[str, Any]:
    return {
        "uri": evidence.uri,
        "kind": evidence.kind,
        "description": evidence.description,
        "start_ns": evidence.start_ns,
        "end_ns": evidence.end_ns,
    }


def _evidence_from_data(data: dict[str, Any]) -> Evidence:
    return Evidence(
        uri=str(data["uri"]),
        kind=str(data["kind"]),
        description=str(data["description"]),
        start_ns=data["start_ns"],
        end_ns=data["end_ns"],
    )


def _metric_to_data(metric: QualityMetric) -> dict[str, Any]:
    return {
        "name": metric.name,
        "value": metric.value,
        "unit": metric.unit,
        "direction": metric.direction.value,
        "threshold": metric.threshold,
        "lower_bound": metric.lower_bound,
        "upper_bound": metric.upper_bound,
        "applicable": metric.applicable,
    }


def _metric_from_data(data: dict[str, Any]) -> QualityMetric:
    return QualityMetric(
        name=str(data["name"]),
        value=float(data["value"]),
        unit=str(data["unit"]),
        direction=MetricDirection(str(data["direction"])),
        threshold=data["threshold"],
        lower_bound=data["lower_bound"],
        upper_bound=data["upper_bound"],
        applicable=bool(data["applicable"]),
    )


def _decision_to_data(decision: Decision) -> dict[str, Any]:
    return {
        "stage": decision.stage,
        "reason": decision.reason.value,
        "severity": decision.severity.value,
        "decision": decision.decision.value,
        "message": decision.message,
        "frame_idx": decision.frame_idx,
        "timestamp_ns": decision.timestamp_ns,
        "span_start_ns": decision.span_start_ns,
        "span_end_ns": decision.span_end_ns,
        "evidence": [_evidence_to_data(item) for item in decision.evidence],
        "detail": decision.detail,
    }


def _decision_from_data(data: dict[str, Any]) -> Decision:
    return Decision(
        stage=int(data["stage"]),
        reason=ReasonCode(str(data["reason"])),
        severity=Severity(str(data["severity"])),
        decision=DecisionType(str(data["decision"])),
        message=str(data["message"]),
        frame_idx=data["frame_idx"],
        timestamp_ns=data["timestamp_ns"],
        span_start_ns=data["span_start_ns"],
        span_end_ns=data["span_end_ns"],
        evidence=[_evidence_from_data(item) for item in data["evidence"]],
        detail=dict(data["detail"]),
    )


def _result_to_data(result: StageResult) -> dict[str, Any]:
    return {
        "descriptor": _descriptor_to_data(result.descriptor),
        "status": result.status.value,
        "input_refs": list(result.input_refs),
        "output_refs": list(result.output_refs),
        "config_hash": result.config_hash,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "metrics": [_metric_to_data(item) for item in result.metrics],
        "decisions": [_decision_to_data(item) for item in result.decisions],
        "evidence": [_evidence_to_data(item) for item in result.evidence],
        "error": result.error,
    }


def _result_from_data(data: dict[str, Any]) -> StageResult:
    return StageResult(
        descriptor=_descriptor_from_data(data["descriptor"]),
        status=StageStatus(str(data["status"])),
        input_refs=tuple(str(item) for item in data["input_refs"]),
        output_refs=tuple(str(item) for item in data["output_refs"]),
        config_hash=str(data["config_hash"]),
        started_at=datetime.fromisoformat(str(data["started_at"])),
        finished_at=datetime.fromisoformat(str(data["finished_at"])),
        metrics=tuple(_metric_from_data(item) for item in data["metrics"]),
        decisions=tuple(_decision_from_data(item) for item in data["decisions"]),
        evidence=tuple(_evidence_from_data(item) for item in data["evidence"]),
        error=data["error"],
    )


def _snapshot_from_data(data: dict[str, Any]) -> RunLedgerSnapshot:
    stages = tuple(
        StageLedgerEntry(
            descriptor=_descriptor_from_data(entry["descriptor"]),
            execution_key=entry["execution_key"],
            status=StageStatus(str(entry["status"])),
            attempts=int(entry["attempts"]),
            input_refs=tuple(str(item) for item in entry["input_refs"]),
            output_refs=tuple(str(item) for item in entry["output_refs"]),
            started_at=(
                datetime.fromisoformat(str(entry["started_at"]))
                if entry["started_at"] is not None
                else None
            ),
            finished_at=(
                datetime.fromisoformat(str(entry["finished_at"]))
                if entry["finished_at"] is not None
                else None
            ),
            last_error=entry["last_error"],
            result=(
                _result_from_data(entry["result"])
                if isinstance(entry["result"], dict)
                else None
            ),
        )
        for entry in data["stages"]
    )
    return RunLedgerSnapshot(
        run_id=str(data["run_id"]),
        session_id=str(data["session_id"]),
        config_hash=str(data["config_hash"]),
        code_version=str(data["code_version"]),
        input_refs=tuple(str(item) for item in data["input_refs"]),
        status=RunStatus(str(data["status"])),
        created_at=datetime.fromisoformat(str(data["created_at"])),
        updated_at=datetime.fromisoformat(str(data["updated_at"])),
        stages=stages,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
