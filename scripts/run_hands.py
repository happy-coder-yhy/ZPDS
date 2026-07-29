"""Hands V1 端到端命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mediapipe
import yaml

from zpds.hands.mediapipe_adapter import (
    HandEstimatorConfig,
    MediaPipeHandEstimator,
)
from zpds.hands.pipeline import HandsPipeline
from zpds.hands.preview import generate_hands_preview
from zpds.hands.segment_reader import PreparedSegmentReader
from zpds.hands.validator import validate_hands_parquet
from zpds.hands.writer import compute_config_sha256, write_hand_observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Prepared Segment 生成 Hands V1 关键点标注",
    )
    parser.add_argument("--segment", required=True, help="Prepared Segment 目录")
    parser.add_argument("--stream-id", help="RGB Stream ID；唯一 RGB 流时可省略")
    parser.add_argument("--config", default="config.yaml", help="Hands YAML 配置")
    parser.add_argument(
        "--backend",
        choices=["auto", "tasks_hand_landmarker", "solutions_hands"],
        help="覆盖配置文件中的模型后端",
    )
    parser.add_argument("--output", help="hands_2d.parquet 输出路径")
    parser.add_argument("--prep-revision", help="覆盖 Prepared Revision")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="最多处理的帧数，用于冒烟测试",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="写出后运行 Hands Validator",
    )
    parser.add_argument(
        "--report-output",
        help="Validator JSON 报告路径，默认与 Parquet 同目录",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="生成 hands_preview.mp4",
    )
    parser.add_argument(
        "--preview-output",
        help="预览视频路径，默认与 Parquet 同目录",
    )
    return parser


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"配置文件顶层必须是对象: {config_path}")
    return data


def _read_segment_json(segment_dir: Path) -> dict[str, Any]:
    segment_path = segment_dir / "segment.json"
    with segment_path.open(encoding="utf-8") as file:
        segment = json.load(file)
    if not isinstance(segment, dict):
        raise TypeError(f"segment.json 顶层必须是对象: {segment_path}")
    return segment


def _default_output_path(
    segment_id: str,
    video_stream_id: str,
) -> Path:
    return (
        Path("output")
        / "hands"
        / segment_id
        / video_stream_id
        / "hands_2d.parquet"
    )


def _image_dimensions(
    segment: dict[str, Any],
    video_stream_id: str,
) -> tuple[int | None, int | None]:
    stream = next(
        (
            item
            for item in segment.get("streams", [])
            if item.get("stream_id") == video_stream_id
        ),
        None,
    )
    if not isinstance(stream, dict):
        return None, None
    shape = stream.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        return None, None
    return int(shape[1]), int(shape[0])


def run(args: argparse.Namespace) -> int:
    segment_dir = Path(args.segment).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config_data = _load_yaml(config_path)
    estimator_config = HandEstimatorConfig.from_yaml(config_path)
    if args.backend:
        estimator_config.backend = args.backend
        hands_config = config_data.setdefault("hands", {})
        if not isinstance(hands_config, dict):
            raise ValueError("配置中的 hands 必须是对象")
        hands_config["backend"] = args.backend

    reader = PreparedSegmentReader(
        segment_dir,
        video_stream_id=args.stream_id,
    )
    segment = _read_segment_json(segment_dir)
    prep_revision = (
        args.prep_revision
        or str(segment.get("record_revision") or "r0001")
    )
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else _default_output_path(reader.segment_id, reader.video_stream_id)
    ).resolve()

    with MediaPipeHandEstimator.from_config(estimator_config) as estimator:
        pipeline = HandsPipeline(
            reader,
            estimator,
            model_name="mediapipe",
            model_version=mediapipe.__version__,
            max_frames=args.max_frames,
        )
        parquet_path = write_hand_observations(
            pipeline,
            output_path,
            prep_revision=prep_revision,
            checkpoint_sha256=estimator.model_info.sha256,
            config_sha256=compute_config_sha256(config_data),
        )
        backend_name = (
            estimator.backend_info.active_backend
            if estimator.backend_info is not None
            else "unknown"
        )
        print(f"Parquet: {parquet_path}")
        print(f"Backend: {backend_name}")
        print(
            "Stats: "
            f"frames={pipeline.stats.frames_processed}, "
            f"observations={pipeline.stats.observations_created}, "
            f"frames_with_hands={pipeline.stats.frames_with_hands}, "
            f"fps={pipeline.stats.average_fps:.2f}"
        )

    validation_status: str | None = None
    if args.validate:
        image_width, image_height = _image_dimensions(
            segment,
            reader.video_stream_id,
        )
        report = validate_hands_parquet(
            parquet_path,
            segment_json_path=str(segment_dir / "segment.json"),
            image_width=image_width,
            image_height=image_height,
        )
        report_path = (
            Path(args.report_output).expanduser()
            if args.report_output
            else output_path.with_name("hands_validation.json")
        ).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        validation_status = str(report["status"])
        print(f"Validation: {validation_status}")
        print(f"Report: {report_path}")

    if args.preview:
        preview_path = (
            Path(args.preview_output).expanduser()
            if args.preview_output
            else output_path.with_name("hands_preview.mp4")
        ).resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        generated_preview = generate_hands_preview(
            segment_dir=str(segment_dir),
            hands_parquet_path=parquet_path,
            output_path=str(preview_path),
            video_stream_id=reader.video_stream_id,
        )
        print(f"Preview: {generated_preview}")

    return 2 if validation_status == "fail" else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    # CLI 边界将第三方模型、视频和 Parquet 异常统一转换为稳定退出码。
    except Exception as error:  # noqa: BLE001
        print(f"Hands CLI failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
