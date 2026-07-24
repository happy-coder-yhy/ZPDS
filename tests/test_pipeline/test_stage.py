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
from zpds.pipeline import (
    StageContext,
    StageDescriptor,
    StageResult,
    StageStatus,
    validate_stage_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"


class ExampleStage:
    descriptor = StageDescriptor(0, "file_registry", "0.1.0")

    def execute(self, context: StageContext) -> StageResult:
        started_at = datetime.now(timezone.utc)
        return StageResult(
            descriptor=self.descriptor,
            status=StageStatus.SUCCEEDED,
            input_refs=context.input_refs,
            output_refs=("report://stage0",),
            config_hash=context.config.config_hash,
            started_at=started_at,
            finished_at=started_at + timedelta(milliseconds=5),
        )


def _context() -> StageContext:
    return StageContext(
        run_id="run_demo",
        session_id="session_demo",
        input_refs=("raw://session/index.jsonl",),
        config=PipelineConfigLoader().load(DEFAULT_CONFIG),
        code_version="0.1.0",
    )


def test_stage_contract_and_success_result() -> None:
    stage = ExampleStage()

    assert validate_stage_contract(stage) == stage.descriptor
    result = stage.execute(_context())

    assert result.status is StageStatus.SUCCEEDED
    assert result.duration_seconds == pytest.approx(0.005)
    assert result.error is None


def test_descriptor_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="between 0 and 12"):
        StageDescriptor(13, "delivery", "0.1.0")
    with pytest.raises(ValueError, match="lower_snake_case"):
        StageDescriptor(0, "File Registry", "0.1.0")
    with pytest.raises(ValueError, match="SemVer"):
        StageDescriptor(0, "file_registry", "v1")


def test_context_requires_input_reference() -> None:
    loaded = PipelineConfigLoader().load(DEFAULT_CONFIG)

    with pytest.raises(ValueError, match="input_refs"):
        StageContext(
            run_id="run",
            session_id="session",
            input_refs=(),
            config=loaded,
            code_version="0.1.0",
        )


def test_result_rejects_non_terminal_status() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="terminal"):
        StageResult(
            descriptor=ExampleStage.descriptor,
            status=StageStatus.RUNNING,
            input_refs=("raw://session",),
            output_refs=(),
            config_hash=_context().config.config_hash,
            started_at=now,
            finished_at=now,
        )


def test_failed_result_requires_error() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="requires error"):
        StageResult(
            descriptor=ExampleStage.descriptor,
            status=StageStatus.FAILED,
            input_refs=("raw://session",),
            output_refs=(),
            config_hash=_context().config.config_hash,
            started_at=now,
            finished_at=now,
        )


def test_result_requires_timezone_aware_timestamps() -> None:
    naive = datetime(2026, 7, 24)  # noqa: DTZ001 - intentionally invalid input

    with pytest.raises(ValueError, match="timezone-aware"):
        StageResult(
            descriptor=ExampleStage.descriptor,
            status=StageStatus.SUCCEEDED,
            input_refs=("raw://session",),
            output_refs=(),
            config_hash=_context().config.config_hash,
            started_at=naive,
            finished_at=naive,
        )


def test_contract_rejects_arbitrary_object() -> None:
    with pytest.raises(TypeError, match="descriptor and execute"):
        validate_stage_contract(object())


def test_result_rejects_decision_from_another_stage() -> None:
    now = datetime.now(timezone.utc)
    decision = Decision(
        stage=1,
        reason=ReasonCode.SCHEMA_UNKNOWN,
        severity=Severity.ERROR,
        decision=DecisionType.QUARANTINE,
        message="unexpected schema",
        evidence=[Evidence(uri="report://schema", kind="structure")],
    )

    with pytest.raises(ValueError, match="descriptor stage_id"):
        StageResult(
            descriptor=ExampleStage.descriptor,
            status=StageStatus.SUCCEEDED,
            input_refs=("raw://session",),
            output_refs=("report://stage0",),
            config_hash=_context().config.config_hash,
            started_at=now,
            finished_at=now,
            decisions=(decision,),
        )
