import json
import pickle
import subprocess
import sys
from pathlib import Path

from zpds.cli import main

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"


def test_config_validate_outputs_version_and_hash(capsys) -> None:
    exit_code = main(["config", "validate", str(DEFAULT_CONFIG)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "valid"
    assert output["version"] == "0.1.0"
    assert output["config_hash"].startswith("sha256:")


def test_cli_execute_creates_ledger_events_metrics_and_status(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    plugin = tmp_path / "wp1_cli_plugin.py"
    plugin.write_text(
        """from datetime import datetime, timezone
from zpds.pipeline import StageDescriptor, StageResult, StageStatus


class DemoStage:
    descriptor = StageDescriptor(0, "file_registry", "0.1.0")

    def execute(self, context):
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


def create_stage():
    return DemoStage()
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    raw_root.mkdir()

    exit_code = main(
        [
            "run",
            "execute",
            "--config",
            str(DEFAULT_CONFIG),
            "--raw-root",
            str(raw_root),
            "--artifact-root",
            str(artifact_root),
            "--run-id",
            "run_cli",
            "--session-id",
            "session_cli",
            "--input-ref",
            "raw://session/index.jsonl",
            "--code-version",
            "commit-a",
            "--stage",
            "wp1_cli_plugin:create_stage",
        ]
    )
    execute_output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert execute_output["status"] == "succeeded"
    assert Path(execute_output["ledger"]).is_file()
    assert Path(execute_output["events"]).is_file()
    assert Path(execute_output["metrics"]).is_file()

    status_exit = main(
        [
            "run",
            "status",
            "--artifact-root",
            str(artifact_root),
            "--run-id",
            "run_cli",
        ]
    )
    status_output = json.loads(capsys.readouterr().out)

    assert status_exit == 0
    assert status_output["status"] == "succeeded"
    assert status_output["stages"][0]["attempts"] == 1
    assert status_output["metrics"]["stages"]["succeeded"] == 1


def test_cli_rejects_invalid_stage_specification(tmp_path: Path, capsys) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    raw_root.mkdir()

    exit_code = main(
        [
            "run",
            "execute",
            "--config",
            str(DEFAULT_CONFIG),
            "--raw-root",
            str(raw_root),
            "--artifact-root",
            str(artifact_root),
            "--run-id",
            "run_invalid",
            "--session-id",
            "session_cli",
            "--input-ref",
            "raw://session/index.jsonl",
            "--code-version",
            "commit-a",
            "--stage",
            "invalid",
        ]
    )
    error = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert error["status"] == "error"
    assert "module:attribute" in error["error"]


def test_base_import_does_not_load_optional_dependencies() -> None:
    script = (
        "import sys; import zpds; "
        "blocked = {'torch', 'h5py', 'mcap'} & set(sys.modules); "
        "assert not blocked, blocked"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_cli_inspects_and_runs_stage_zero_to_two(
    tmp_path: Path,
    capsys,
) -> None:
    raw_root = tmp_path / "raw"
    session = raw_root / "session"
    (session / "imu").mkdir(parents=True)
    (session / "meta.json").write_text(
        json.dumps(
            {
                "session": {"output_folder_name": "cli-source"},
                "streams": {"color": {"enabled": True, "fps": 30}},
                "imu": {"csv": "imu/imu_000000.csv"},
            }
        ),
        encoding="utf-8",
    )
    (session / "index.jsonl").write_text(
        '{"schema":"guida.index.v1"}\n'
        '{"type":"segment_start","color_video":"color_000000.mkv"}\n'
        '{"type":"frame","seq":0,"timestamp_ns":1000000000}\n',
        encoding="utf-8",
    )
    (session / "color_000000.mkv").write_bytes(b"fixture")
    (session / "imu" / "imu_000000.csv").write_text(
        "timestamp_ns,ax,ay,az,gx,gy,gz\n1000000000,0,0,9.8,0,0,0\n",
        encoding="utf-8",
    )

    inspect_exit = main(
        [
            "source",
            "inspect",
            "--profile",
            "guida_ego",
            "--raw-root",
            str(raw_root),
            "--input-ref",
            "raw://session",
        ]
    )
    inspect_output = json.loads(capsys.readouterr().out)
    assert inspect_exit == 0
    assert inspect_output["inventory"]["session_id"] == "cli-source"

    artifact_root = tmp_path / "artifacts"
    run_exit = main(
        [
            "run",
            "source",
            "--profile",
            "guida_ego",
            "--config",
            str(DEFAULT_CONFIG),
            "--raw-root",
            str(raw_root),
            "--artifact-root",
            str(artifact_root),
            "--run-id",
            "run_source_cli",
            "--session-id",
            "cli-source",
            "--input-ref",
            "raw://session",
            "--code-version",
            "test-code",
        ]
    )
    run_output = json.loads(capsys.readouterr().out)

    assert run_exit == 0
    assert run_output["executed_stage_ids"] == [0, 1, 2]
    assert (artifact_root / "runs" / "run_source_cli" / "stage-0" / "inventory.json").is_file()


def test_source_scan_cli_runs_safe_pickle_full_scan(tmp_path: Path, capsys) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "epic"
    source.mkdir(parents=True)
    (source / "annotations.pkl").write_bytes(pickle.dumps({"frames": [1, 2]}))

    exit_code = main(
        [
            "source",
            "scan",
            "--profile",
            "epic100",
            "--raw-root",
            str(raw_root),
            "--input-ref",
            "raw://epic",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["report"]["checked_records"] == 1
    assert output["report"]["decoded_records"] == 1
    assert output["report"]["metadata"]["isolated_content_parse"] is True
    assert output["report"]["metadata"]["quick_validation"] is False
