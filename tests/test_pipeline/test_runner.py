from datetime import datetime, timezone
from pathlib import Path

import pytest

from zpds.config import PipelineConfigLoader
from zpds.pipeline import (
    FileRunLedger,
    PipelineRunner,
    RunStatus,
    StageContext,
    StageDescriptor,
    StageResult,
    StageStatus,
)
from zpds.storage import LocalStorage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"


class ControlledStage:
    def __init__(
        self,
        stage_id: int,
        name: str,
        *,
        failures: int = 0,
        interrupt_once: bool = False,
    ) -> None:
        self.descriptor = StageDescriptor(stage_id, name, "0.1.0")
        self.failures = failures
        self.interrupt_once = interrupt_once
        self.calls = 0
        self.seen_inputs: list[tuple[str, ...]] = []

    def execute(self, context: StageContext) -> StageResult:
        self.calls += 1
        self.seen_inputs.append(context.input_refs)
        if self.interrupt_once and self.calls == 1:
            raise KeyboardInterrupt
        if self.calls <= self.failures:
            raise OSError("temporary read failure")
        now = datetime.now(timezone.utc)
        return StageResult(
            descriptor=self.descriptor,
            status=StageStatus.SUCCEEDED,
            input_refs=context.input_refs,
            output_refs=(f"artifact://stage-{self.descriptor.stage_id}/result.json",),
            config_hash=context.config.config_hash,
            started_at=now,
            finished_at=now,
        )


def _context(run_id: str) -> StageContext:
    return StageContext(
        run_id=run_id,
        session_id="session_demo",
        input_refs=("raw://session/index.jsonl",),
        config=PipelineConfigLoader().load(DEFAULT_CONFIG),
        code_version="commit-a",
    )


def _ledger(tmp_path: Path) -> FileRunLedger:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    return FileRunLedger(LocalStorage(raw_root, tmp_path / "artifacts"))


def test_runner_retries_failure_and_passes_outputs_to_next_stage(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    stage0 = ControlledStage(0, "file_registry", failures=1)
    stage1 = ControlledStage(1, "container_structure")

    result = PipelineRunner((stage1, stage0), ledger).run(_context("run_retry"))

    assert result.status is RunStatus.SUCCEEDED
    assert result.executed_stage_ids == (0, 1)
    assert result.reused_stage_ids == ()
    assert stage0.calls == 2
    assert stage1.calls == 1
    assert stage1.seen_inputs == [("artifact://stage-0/result.json",)]
    assert ledger.snapshot("run_retry").stage(0).attempts == 2


def test_runner_resumes_after_interruption_and_reuses_completed_stage(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    stage0 = ControlledStage(0, "file_registry")
    stage1 = ControlledStage(1, "container_structure", interrupt_once=True)
    runner = PipelineRunner((stage0, stage1), ledger)
    context = _context("run_resume")

    with pytest.raises(KeyboardInterrupt):
        runner.run(context)

    interrupted = ledger.snapshot(context.run_id)
    assert interrupted.stage(0).status is StageStatus.SUCCEEDED
    assert interrupted.stage(1).status is StageStatus.RUNNING

    resumed = runner.run(context)
    repeated = runner.run(context)

    assert resumed.status is RunStatus.SUCCEEDED
    assert resumed.executed_stage_ids == (1,)
    assert resumed.reused_stage_ids == (0,)
    assert stage0.calls == 1
    assert stage1.calls == 2
    assert ledger.snapshot(context.run_id).stage(1).attempts == 2
    assert repeated.executed_stage_ids == ()
    assert repeated.reused_stage_ids == (0, 1)
    assert stage0.calls == 1
    assert stage1.calls == 2


def test_runner_records_terminal_failure_after_retry_budget(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    stage = ControlledStage(0, "file_registry", failures=3)

    result = PipelineRunner((stage,), ledger).run(_context("run_failed"))

    assert result.status is RunStatus.FAILED
    assert result.stage_results[0].status is StageStatus.FAILED
    assert "OSError: temporary read failure" == result.stage_results[0].error
    assert stage.calls == 2
    assert ledger.snapshot("run_failed").stage(0).attempts == 2
