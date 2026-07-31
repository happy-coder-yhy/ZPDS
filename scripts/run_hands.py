"""Hands V1 端到端命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2

from zpds.hands.config import HandsOutputPaths, HandsPipelineConfig
from zpds.hands.experience import write_hands_experience_manifest
from zpds.hands.mediapipe_adapter import (
    MediaPipeHandEstimator,
)
from zpds.hands.pipeline import HandsPipeline
from zpds.hands.preview import generate_hands_preview
from zpds.hands.segment_reader import PreparedSegmentReader
from zpds.hands.validator import validate_hands_parquet, validate_wilor_hands
from zpds.hands.writer import wilor_provenance, write_hand_observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Prepared Segment 生成 Hands V1 关键点标注",
    )
    parser.add_argument("--segment", required=True, help="Prepared Segment 目录")
    parser.add_argument("--stream-id", help="RGB Stream ID；唯一 RGB 流时可省略")
    parser.add_argument("--config", default="config.yaml", help="Hands YAML 配置")
    parser.add_argument(
        "--backend",
        choices=["auto", "tasks_hand_landmarker", "solutions_hands", "wilor"],
        help="覆盖配置文件中的模型后端",
    )
    parser.add_argument("--output", help="hands_2d.parquet 输出路径")
    parser.add_argument(
        "--experience-dir",
        help="按 Experience 标准目录写出；不能与各类 --*-output 同时使用",
    )
    parser.add_argument(
        "--experience-version",
        help="Experience 版本；默认使用 --experience-dir 目录名",
    )
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
    parser.add_argument(
        "--manifest-output",
        help="运行清单路径，默认与 Parquet 同目录",
    )
    return parser


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
    segment_dir: Path | None = None,
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
    if segment_dir is not None:
        uri = stream.get("uri")
        if isinstance(uri, str):
            capture = cv2.VideoCapture(str(segment_dir / uri))
            try:
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                capture.release()
            if width > 0 and height > 0:
                return width, height
    shape = stream.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        return None, None
    return int(shape[1]), int(shape[0])


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """原子写入 JSON，避免任务中断留下半份报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _mediapipe_version() -> str:
    try:
        return version("mediapipe")
    except PackageNotFoundError:
        return ""


def _create_wilor_estimator(runtime_config: HandsPipelineConfig):
    """Create WiLoR lazily so ordinary MediaPipe imports do not need Torch."""
    if runtime_config.wilor is None:
        raise ValueError("WiLoR 后端缺少 hands.wilor 配置")

    from zpds.hands.backends.wilor import WiLoRBackend
    from zpds.hands.wilor_adapter import WiLoRAdapter
    from zpds.hands.wilor_estimator import WiLoREstimatorConfig, WiLoRHandEstimator
    from zpds.hands.wilor_schema import WiLoRFallbackPolicy

    wilor_document = runtime_config.document["hands"].get("wilor", {})
    fallback_estimator = None
    if bool(wilor_document.get("fallback_to_mediapipe", False)):
        fallback_estimator = MediaPipeHandEstimator.from_config(runtime_config.estimator)
    fallback_policy = WiLoRFallbackPolicy(
        on_wilor_frame_failure=bool(
            wilor_document.get("fallback_on_frame_failure", True)
        ),
        on_wilor_no_hand=bool(wilor_document.get("fallback_on_no_hand", False)),
        on_invalid_input=bool(wilor_document.get("fallback_on_invalid_input", False)),
        compare_with_mediapipe=False,
    )
    backend = WiLoRBackend(runtime_config.wilor)
    return WiLoRHandEstimator(
        adapter=WiLoRAdapter(backend),
        model_info=backend.model_info,
        fallback_estimator=fallback_estimator,
        config=WiLoREstimatorConfig(
            fallback_policy=fallback_policy,
            model_version=backend.model_info.model_version,
        ),
    )


def run(args: argparse.Namespace) -> int:
    segment_dir = Path(args.segment).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    runtime_config = HandsPipelineConfig.load(
        config_path,
        backend_override=args.backend,
    )

    reader = PreparedSegmentReader(
        segment_dir,
        video_stream_id=args.stream_id,
    )
    segment = _read_segment_json(segment_dir)
    prep_revision = (
        args.prep_revision
        or str(segment.get("record_revision") or "r0001")
    )
    experience_dir_value = getattr(args, "experience_dir", None)
    experience_version = getattr(args, "experience_version", None)
    if experience_dir_value:
        conflicting_outputs = [
            name
            for name in (
                "output",
                "report_output",
                "preview_output",
                "manifest_output",
            )
            if getattr(args, name, None)
        ]
        if conflicting_outputs:
            option_names = [
                f"--{name.replace('_', '-')}"
                for name in conflicting_outputs
            ]
            raise ValueError(
                "--experience-dir 不能与以下参数同时使用: "
                f"{', '.join(option_names)}"
            )
        output_paths = HandsOutputPaths.experience(experience_dir_value)
    else:
        output_path = (
            Path(args.output).expanduser()
            if args.output
            else _default_output_path(
                reader.segment_id,
                reader.video_stream_id,
            )
        ).resolve()
        output_paths = HandsOutputPaths(
            parquet=output_path,
            validation_report=output_path.with_name("hands_validation.json"),
            preview=output_path.with_name("hands_preview.mp4"),
            run_manifest=output_path.with_name("hands_run.json"),
        )
    preview_name = (
        "wilor_hands_preview.mp4"
        if runtime_config.estimator.backend == "wilor"
        else "mediapipe_hands_preview.mp4"
    )
    output_paths = replace(output_paths, preview=output_paths.preview.with_name(preview_name))
    output_path = output_paths.parquet
    config_sha256 = runtime_config.config_sha256
    manifest_path = (
        Path(args.manifest_output).expanduser()
        if args.manifest_output
        else output_paths.run_manifest
    ).resolve()
    wilor_run_report: dict[str, Any] | None = None

    if runtime_config.estimator.backend == "wilor":
        estimator = _create_wilor_estimator(runtime_config)
        try:
            provenance, wilor_run_report = wilor_provenance(
                estimator,
                runtime_config.document,
            )
            pipeline = HandsPipeline(
                reader,
                estimator,
                model_name="wilor",
                model_version=estimator.model_info.model_version,
                max_frames=args.max_frames,
            )
            parquet_path = write_hand_observations(
                pipeline,
                output_path,
                prep_revision=prep_revision,
                checkpoint_sha256=provenance["checkpoint_sha256"],
                config_sha256=config_sha256,
                run_meta=provenance,
            )
            backend_name = "wilor"
            checkpoint_sha256 = provenance["checkpoint_sha256"]
            run_statistics = {
                "frames_processed": pipeline.stats.frames_processed,
                "observations_created": pipeline.stats.observations_created,
                "frames_with_hands": pipeline.stats.frames_with_hands,
                "average_fps": pipeline.stats.average_fps,
            }
        finally:
            wilor_run_report = estimator.build_run_report().to_dict()
            estimator.close()
        print(f"Parquet: {parquet_path}")
        print("Backend: wilor")
        print(
            "Stats: "
            f"frames={run_statistics['frames_processed']}, "
            f"observations={run_statistics['observations_created']}, "
            f"frames_with_hands={run_statistics['frames_with_hands']}, "
            f"fps={run_statistics['average_fps']:.2f}"
        )
        # WiLoR validation consumes this report.  It is enriched with the generic
        # run metadata below after validation has finished.
        _write_json_atomic(manifest_path, wilor_run_report)
    else:
        if runtime_config.wilor is not None:
            raise ValueError("非 WiLoR 后端不能携带 WiLoR 运行配置")
        with MediaPipeHandEstimator.from_config(runtime_config.estimator) as estimator:
            backend_info = estimator.backend_info
            run_meta = {
                "backend_requested": (
                    backend_info.requested_backend if backend_info else ""
                ),
                "backend_active": (
                    backend_info.active_backend if backend_info else ""
                ),
                "backend_fallback_used": (
                    backend_info.fallback_used if backend_info else False
                ),
                "backend_fallback_reason": (
                    backend_info.fallback_reason if backend_info else ""
                ),
                "backend_delegate": backend_info.delegate if backend_info else "",
            }
            pipeline = HandsPipeline(
                reader,
                estimator,
                model_name="mediapipe",
                model_version=_mediapipe_version(),
                max_frames=args.max_frames,
            )
            parquet_path = write_hand_observations(
                pipeline,
                output_path,
                prep_revision=prep_revision,
                checkpoint_sha256=estimator.model_info.sha256,
                config_sha256=config_sha256,
                run_meta=run_meta,
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
            run_statistics = {
                "frames_processed": pipeline.stats.frames_processed,
                "observations_created": pipeline.stats.observations_created,
                "frames_with_hands": pipeline.stats.frames_with_hands,
                "average_fps": pipeline.stats.average_fps,
            }
            checkpoint_sha256 = estimator.model_info.sha256

    validation_status: str | None = None
    report_path: Path | None = None
    declared_image_dimensions = _image_dimensions(
        segment,
        reader.video_stream_id,
    )
    actual_image_dimensions = _image_dimensions(
        segment,
        reader.video_stream_id,
        segment_dir,
    )
    if args.validate:
        image_width, image_height = actual_image_dimensions
        if runtime_config.estimator.backend == "wilor":
            if image_width is None or image_height is None:
                raise ValueError("WiLoR 校验需要可读取的实际视频分辨率")
            report = validate_wilor_hands(
                parquet_path,
                hands_run_path=str(manifest_path),
                segment_json_path=str(segment_dir / "segment.json"),
                image_width=image_width,
                image_height=image_height,
            )
        else:
            report = validate_hands_parquet(
                parquet_path,
                segment_json_path=str(segment_dir / "segment.json"),
                image_width=image_width,
                image_height=image_height,
            )
        report_path = (
            Path(args.report_output).expanduser()
            if args.report_output
            else output_paths.validation_report
        ).resolve()
        _write_json_atomic(report_path, report)
        validation_status = str(report["status"])
        print(f"Validation: {validation_status}")
        print(f"Report: {report_path}")

    generated_preview: str | None = None
    if args.preview:
        preview_path = (
            Path(args.preview_output).expanduser()
            if args.preview_output
            else output_paths.preview
        ).resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        generated_preview = generate_hands_preview(
            segment_dir=str(segment_dir),
            hands_parquet_path=parquet_path,
            output_path=str(preview_path),
            video_stream_id=reader.video_stream_id,
        )
        print(f"Preview: {generated_preview}")

    exit_code = 2 if validation_status == "fail" else 0
    manifest_document = {
        "schema_version": 1,
        "completed": exit_code == 0,
        "segment_id": reader.segment_id,
        "video_stream_id": reader.video_stream_id,
        "prep_revision": prep_revision,
        "backend": backend_name,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "max_frames": args.max_frames,
        "image_dimensions": {
            "declared_width": declared_image_dimensions[0],
            "declared_height": declared_image_dimensions[1],
            "actual_width": actual_image_dimensions[0],
            "actual_height": actual_image_dimensions[1],
            "metadata_matches_video": (
                declared_image_dimensions == actual_image_dimensions
            ),
        },
        "statistics": run_statistics,
        "validation_status": validation_status,
        "outputs": {
            "parquet": parquet_path,
            "validation_report": str(report_path) if report_path else None,
            "preview": generated_preview,
        },
    }
    if wilor_run_report is not None:
        wilor_run_report.update(manifest_document)
        _write_json_atomic(manifest_path, wilor_run_report)
    else:
        _write_json_atomic(manifest_path, manifest_document)
    print(f"Manifest: {manifest_path}")
    if experience_dir_value:
        manifest = write_hands_experience_manifest(
            experience_dir=experience_dir_value,
            experience_version=(
                experience_version
                or Path(experience_dir_value).expanduser().resolve().name
            ),
            segment_id=reader.segment_id,
            video_stream_id=reader.video_stream_id,
            outputs=output_paths,
            prep_revision=prep_revision,
            config_sha256=config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            validation_status=validation_status,
            model_name="wilor" if wilor_run_report is not None else "mediapipe",
        )
        print(f"Experience: {manifest}")
    return exit_code


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
