"""生成多视频 Stage A provisional 阈值回归报告。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from zpds.scene.config import SceneConfig
from zpds.scene.regression import run_stage_a_regression

_detection_cli = importlib.import_module(
    "scripts.run_scene_detection" if __package__ else "run_scene_detection"
)
ConsoleProgress = _detection_cli.ConsoleProgress
read_video = _detection_cli.read_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行多数据源 Scene Stage A 阈值回归")
    parser.add_argument("--input", action="append", required=True, help="视频路径，可重复")
    parser.add_argument(
        "--profile",
        action="append",
        help="QC Profile；指定一个则应用全部输入，或与 input 等量逐项对应",
    )
    parser.add_argument("--name", action="append", help="案例名称，需与 input 等量")
    parser.add_argument("--config", default="configs/scene/default.yaml")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--output-json",
        default="output/scene/threshold_regression.json",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_profiles(profiles: Sequence[str] | None, count: int) -> list[str | None]:
    if not profiles:
        return [None] * count
    if len(profiles) == 1:
        return [profiles[0]] * count
    if len(profiles) != count:
        raise ValueError("profile 数量必须为 1，或与 input 数量一致")
    return list(profiles)


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    inputs = list(args.input)
    profiles = _expand_profiles(getattr(args, "profile", None), len(inputs))
    names = list(getattr(args, "name", None) or [])
    if names and len(names) != len(inputs):
        raise ValueError("name 数量必须与 input 数量一致")

    cases = []
    progress = None if args.quiet else ConsoleProgress()
    for index, (input_value, profile) in enumerate(zip(inputs, profiles)):
        input_path = Path(input_value).expanduser().resolve()
        config = (
            SceneConfig.load_with_profile(args.config, profile)
            if profile is not None
            else SceneConfig.load(args.config)
        )
        if not args.quiet:
            print(f"回归案例 {index + 1}/{len(inputs)}: {input_path}", file=sys.stderr)
        before_hash = _sha256(input_path)
        video = read_video(input_path, max_frames=args.max_frames)
        report = run_stage_a_regression(
            video.frames,
            fps=video.fps,
            config=config,
            progress=progress,
        )
        after_hash = _sha256(input_path)
        if before_hash != after_hash:
            raise RuntimeError(f"Raw 视频在回归过程中发生变化: {input_path}")
        report.update(
            {
                "name": names[index] if names else input_path.stem,
                "input": str(input_path),
                "input_sha256": before_hash,
                "raw_unchanged": True,
            }
        )
        cases.append(report)

    document = {
        "schema_version": "zpds.scene.threshold_regression.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "calibration_status": "provisional",
        "requires_adjudicated_gold": True,
        "thresholds_changed": False,
        "cases": cases,
    }
    _write_json_atomic(Path(args.output_json).expanduser().resolve(), document)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"scene regression failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
