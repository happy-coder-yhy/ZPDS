import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from zpds.config import PipelineConfigLoader
from zpds.observability import (
    JsonLinesObserver,
    build_run_metrics,
    persist_run_metrics,
)
from zpds.pipeline import (
    FileRunLedger,
    PipelineRunner,
    StageContext,
    StageDescriptor,
    StageResult,
    StageStatus,
)
from zpds.storage import LocalStorage
from zpds.utils.schema_validator import validate_with_schema

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"


class SuccessStage:
    descriptor = StageDescriptor(0, "file_registry", "0.1.0")

    def execute(self, context: StageContext) -> StageResult:
        now = datetime.now(timezone.utc)
        return StageResult(
            descriptor=self.descriptor,
            status=StageStatus.SUCCEEDED,
            input_refs=context.input_refs,
            output_refs=("artifact://reports/stage0.json",),
            config_hash=context.config.config_hash,
            started_at=now,
            finished_at=now,
        )


def test_json_events_and_metrics_are_machine_readable(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    storage = LocalStorage(raw_root, tmp_path / "artifacts")
    ledger = FileRunLedger(storage)
    stream = StringIO()
    context = StageContext(
        run_id="run_observable",
        session_id="session_demo",
        input_refs=("raw://session/index.jsonl",),
        config=PipelineConfigLoader().load(DEFAULT_CONFIG),
        code_version="commit-a",
    )

    PipelineRunner(
        (SuccessStage(),),
        ledger,
        observer=JsonLinesObserver(stream),
    ).run(context)
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    snapshot = ledger.snapshot(context.run_id)
    metrics = build_run_metrics(snapshot)
    persisted = persist_run_metrics(storage, snapshot)

    assert [event["event"] for event in events] == [
        "run_started",
        "stage_attempt_started",
        "stage_attempt_finished",
        "run_finished",
    ]
    assert all(event["run_id"] == context.run_id for event in events)
    assert metrics["stages"]["succeeded"] == 1
    assert metrics["attempts"] == 1
    assert metrics["retries"] == 0
    assert validate_with_schema(metrics, "run_metrics") == []
    assert persisted.reference == "artifact://runs/run_observable/metrics.json"
    assert storage.read_json(persisted.reference)["status"] == "succeeded"
