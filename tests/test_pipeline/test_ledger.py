from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from zpds.config import PipelineConfigLoader
from zpds.core.decisions import (
    Decision,
    DecisionType,
    Evidence,
    ReasonCode,
    Severity,
)
from zpds.core.quality import MetricDirection, QualityMetric
from zpds.pipeline import (
    FileRunLedger,
    LedgerConflictError,
    RunStatus,
    StageContext,
    StageDescriptor,
    StageResult,
    StageStatus,
    execution_key,
)
from zpds.storage import LocalStorage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"
DESCRIPTOR = StageDescriptor(0, "file_registry", "0.1.0")


def _context(*, code_version: str = "commit-a") -> StageContext:
    return StageContext(
        run_id="run_ledger",
        session_id="session_demo",
        input_refs=("raw://session/index.jsonl",),
        config=PipelineConfigLoader().load(DEFAULT_CONFIG),
        code_version=code_version,
    )


def _ledger(tmp_path: Path) -> FileRunLedger:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    return FileRunLedger(LocalStorage(raw_root, tmp_path / "artifacts"))


def test_ledger_persists_stage_result_and_provenance(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    context = _context()
    key = execution_key(DESCRIPTOR, context)
    started_at = datetime.now(timezone.utc)
    evidence = Evidence(uri="artifact://reports/stage0.json", kind="inventory")
    decision = Decision(
        stage=0,
        reason=ReasonCode.FILE_MISSING,
        severity=Severity.WARN,
        decision=DecisionType.KEEP_WITH_FLAG,
        message="optional preview is absent",
        evidence=[evidence],
    )
    metric = QualityMetric(
        name="inventory_complete",
        value=1.0,
        direction=MetricDirection.HIGHER_IS_BETTER,
        threshold=1.0,
    )

    initial = ledger.initialize(context, (DESCRIPTOR,))
    attempt = ledger.begin_attempt(
        context.run_id,
        DESCRIPTOR,
        key,
        context.input_refs,
        started_at=started_at,
    )
    result = StageResult(
        descriptor=DESCRIPTOR,
        status=StageStatus.SUCCEEDED,
        input_refs=context.input_refs,
        output_refs=("artifact://reports/stage0.json",),
        config_hash=context.config.config_hash,
        started_at=started_at,
        finished_at=started_at + timedelta(milliseconds=5),
        metrics=(metric,),
        decisions=(decision,),
        evidence=(evidence,),
    )
    completed = ledger.record_result(context.run_id, key, result)
    reloaded = ledger.snapshot(context.run_id)

    assert initial.status is RunStatus.PENDING
    assert attempt == 1
    assert completed.status is RunStatus.SUCCEEDED
    assert reloaded.stage(0).result == result
    assert reloaded.stage(0).attempts == 1
    assert ledger.completed_result(context.run_id, DESCRIPTOR, key) == result


def test_ledger_rejects_reusing_run_id_for_different_code(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.initialize(_context(code_version="commit-a"), (DESCRIPTOR,))

    with pytest.raises(LedgerConflictError, match="code_version"):
        ledger.initialize(_context(code_version="commit-b"), (DESCRIPTOR,))


def test_ledger_keeps_running_attempt_for_interruption_recovery(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    context = _context()
    key = execution_key(DESCRIPTOR, context)
    ledger.initialize(context, (DESCRIPTOR,))

    ledger.begin_attempt(
        context.run_id,
        DESCRIPTOR,
        key,
        context.input_refs,
        started_at=datetime.now(timezone.utc),
    )
    snapshot = ledger.snapshot(context.run_id)

    assert snapshot.status is RunStatus.RUNNING
    assert snapshot.stage(0).status is StageStatus.RUNNING
    assert snapshot.stage(0).attempts == 1
