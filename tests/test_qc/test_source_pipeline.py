import json
from pathlib import Path

from zpds.adapters import GuidaAdapter
from zpds.config import PipelineConfigLoader
from zpds.pipeline import FileRunLedger, PipelineRunner, StageContext
from zpds.qc.stage0_registry import InventoryStage
from zpds.qc.stage1_structure import StructureStage
from zpds.qc.stage2_time import TimeStage
from zpds.storage import LocalStorage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"


def _make_session(raw_root: Path) -> Path:
    session = raw_root / "session"
    (session / "imu").mkdir(parents=True)
    (session / "meta.json").write_text(
        json.dumps(
            {
                "session": {"output_folder_name": "pipeline-demo"},
                "streams": {"color": {"enabled": True, "fps": 30}},
                "imu": {"csv": "imu/imu_000000.csv", "sample_rate_hz": 100},
            }
        ),
        encoding="utf-8",
    )
    (session / "index.jsonl").write_text(
        '{"schema":"guida.index.v1"}\n'
        '{"type":"segment_start","color_video":"color_000000.mkv"}\n'
        '{"type":"frame","seq":0,"timestamp_ns":1000000000}\n'
        '{"type":"frame","seq":1,"timestamp_ns":1033333333}\n',
        encoding="utf-8",
    )
    (session / "color_000000.mkv").write_bytes(b"fixture")
    (session / "imu" / "imu_000000.csv").write_text(
        "timestamp_ns,ax,ay,az,gx,gy,gz\n"
        "1000000000,0,0,9.8,0,0,0\n",
        encoding="utf-8",
    )
    return session


def test_source_stages_write_traceable_artifacts_and_resume(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_session(raw_root)
    storage = LocalStorage(raw_root, tmp_path / "artifacts")
    stages = (
        InventoryStage(GuidaAdapter(), storage),
        StructureStage(GuidaAdapter(), storage),
        TimeStage(GuidaAdapter(), storage),
    )
    context = StageContext(
        run_id="run_wp2",
        session_id="pipeline-demo",
        input_refs=("raw://session",),
        config=PipelineConfigLoader().load(DEFAULT_CONFIG),
        code_version="test-code",
    )
    runner = PipelineRunner(stages, FileRunLedger(storage))

    first = runner.run(context)
    repeated = runner.run(context)

    assert first.executed_stage_ids == (0, 1, 2)
    assert repeated.reused_stage_ids == (0, 1, 2)
    inventory = storage.read_json("artifact://runs/run_wp2/stage-0/inventory.json")
    assets = inventory["inventory"]["assets"]
    assert assets
    assert all(asset["sha256"].startswith("sha256:") for asset in assets)
    assert inventory["config_hash"] == context.config.config_hash
    assert storage.exists("artifact://runs/run_wp2/stage-1/structure.json")
    assert storage.exists("artifact://runs/run_wp2/stage-2/time.json")

