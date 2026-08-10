"""Hands V1 端到端命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2

from zpds.hands.backend_router import HandsBackendRouter
from zpds.hands.config import HandsOutputPaths, HandsPipelineConfig
from zpds.hands.estimator_factory import (
    EstimatorRuntime,
    create_hand_estimator,
    validate_estimator_runtime,
)
from zpds.hands.experience import write_hands_experience_manifest
from zpds.hands.frame_artifacts import (
    InferenceArtifactContext,
    validate_wilor_frame_artifacts,
)
from zpds.hands.orchestration import (
    InferenceWriterBundle,
    create_inference_writers,
)
from zpds.hands.pipeline import HandsPipeline
from zpds.hands.preview import generate_hands_preview
from zpds.hands.segment_reader import PreparedSegmentReader
from zpds.hands.validator import validate_hands_parquet, validate_wilor_hands
from zpds.hands.wilor_preflight import check_wilor_assets
from zpds.hands.writer import write_hand_observations

EstimatorFactory = Callable[[str, HandsPipelineConfig], EstimatorRuntime]
InferenceWriterFactory = Callable[..., InferenceWriterBundle]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Prepared Segment 生成 Hands V1 关键点标注",
    )
    parser.add_argument("--segment", required=True, help="Prepared Segment 目录")
    parser.add_argument("--stream-id", help="RGB Stream ID；唯一 RGB 流时可省略")
    parser.add_argument("--config", default="config.yaml", help="Hands YAML 配置")
    parser.add_argument(
        "--source-kind",
        choices=["ego", "non_ego"],
        default="non_ego",
        help="数据源类型；必须显式设为 ego 才会选择 ego 主后端",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "tasks_hand_landmarker", "solutions_hands"],
        help="覆盖配置文件中的模型后端",
    )
    parser.add_argument("--output", help="hands_2d.parquet 输出路径")
    parser.add_argument(
        "--frame-status-output",
        help="WiLoR 逐帧状态 Parquet 输出路径",
    )
    parser.add_argument(
        "--bbox-output",
        help="WiLoR 全帧检测 BBox Parquet 输出路径",
    )
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
    segment_dir: Path,
    video_stream_id: str,
) -> Path:
    """手部产物写入 Prepared Segment 目录下的 hands/<stream_id>/。"""
    return segment_dir / "hands" / video_stream_id / "hands_2d.parquet"


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


def _combine_validation_reports(
    **reports: dict[str, Any],
) -> dict[str, Any]:
    """将 A 的全帧资产校验与 C 的 Hands 校验合并为一个报告。"""
    statuses = [
        str(report.get("status", "fail"))
        for report in reports.values()
    ]
    status = (
        "fail"
        if "fail" in statuses
        else ("warn" if "warn" in statuses else "pass")
    )
    errors: list[str] = []
    warnings: list[str] = []
    for report in reports.values():
        errors.extend(str(item) for item in report.get("errors", []))
        warnings.extend(str(item) for item in report.get("warnings", []))
    return {
        "status": status,
        "checks": {
            name: report.get("checks", {})
            for name, report in reports.items()
        },
        "statistics": {
            name: report.get("statistics", {})
            for name, report in reports.items()
        },
        "errors": errors,
        "warnings": warnings,
    }


def run(
    args: argparse.Namespace,
    *,
    estimator_factory: EstimatorFactory = create_hand_estimator,
    inference_writer_factory: InferenceWriterFactory = create_inference_writers,
    verify_wilor_assets: bool = True,
) -> int:
    segment_dir = Path(args.segment).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    runtime_config = HandsPipelineConfig.load(
        config_path,
        backend_override=args.backend,
    )
    source_kind = getattr(args, "source_kind", "non_ego")
    router = HandsBackendRouter(runtime_config.backend_policy)
    primary_model = router.select_backend(is_ego=source_kind == "ego")

    reader = PreparedSegmentReader(
        segment_dir,
        video_stream_id=args.stream_id,
    )
    segment = _read_segment_json(segment_dir)
    prep_revision = (
        args.prep_revision
        or str(
            segment.get("prep_revision")
            or segment.get("record_revision")
            or "r0001"
        )
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
                "frame_status_output",
                "bbox_output",
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
                segment_dir,
                reader.video_stream_id,
            )
        ).resolve()
        output_paths = HandsOutputPaths(
            parquet=output_path,
            validation_report=output_path.with_name("hands_validation.json"),
            preview=output_path.with_name("hands_preview.mp4"),
            run_manifest=output_path.with_name("hands_run.json"),
            frame_status=(
                Path(args.frame_status_output).expanduser().resolve()
                if getattr(args, "frame_status_output", None)
                else output_path.with_name("wilor_frame_status.parquet")
            ),
            bbox=(
                Path(args.bbox_output).expanduser().resolve()
                if getattr(args, "bbox_output", None)
                else output_path.with_name("wilor_hands_bbox.parquet")
            ),
        )
    output_path = output_paths.parquet
    config_sha256 = runtime_config.config_sha256

    if primary_model == "wilor" and verify_wilor_assets:
        preflight = check_wilor_assets(runtime_config.wilor)
        if not preflight.ready:
            details = "; ".join(preflight.errors) or "未知资产错误"
            raise RuntimeError(f"WiLoR 资产预检失败: {details}")

    runtime = estimator_factory(primary_model, runtime_config)
    try:
        validate_estimator_runtime(primary_model, runtime, runtime_config)
    except Exception:
        runtime.estimator.close()
        raise
    frame_status_path = (
        str(output_paths.frame_status)
        if primary_model == "wilor"
        and output_paths.frame_status is not None
        else None
    )
    bbox_path = (
        str(output_paths.bbox)
        if primary_model == "wilor" and output_paths.bbox is not None
        else None
    )
    try:
        writer_context = InferenceArtifactContext(
            prep_revision=prep_revision,
            segment_id=reader.segment_id,
            video_stream_id=reader.video_stream_id,
            model_name=runtime.model_name,
            model_version=runtime.model_version,
            checkpoint_sha256=runtime.checkpoint_sha256,
            config_sha256=config_sha256,
            device=str(
                runtime.run_meta.get("device")
                or runtime.run_meta.get("backend_delegate")
                or runtime.active_backend
            ),
        )
        writers = inference_writer_factory(
            primary_model,
            frame_status_path=frame_status_path,
            bbox_path=bbox_path,
            context=writer_context,
        )
    except Exception:
        runtime.estimator.close()
        raise
    parquet_path: str | None = None
    try:
        pipeline = HandsPipeline(
            reader,
            runtime.estimator,
            model_name=runtime.model_name,
            model_version=runtime.model_version,
            active_backend=runtime.active_backend,
            max_frames=args.max_frames,
        )

        def consume_records() -> Iterator:
            for record in pipeline.run_frames():
                writers.frame_status.write(record)
                writers.bbox.write(record)
                yield from pipeline.observations_for_record(
                    record,
                    fail_on_error=primary_model == "mediapipe",
                )

        parquet_path = write_hand_observations(
            consume_records(),
            output_path,
            prep_revision=prep_revision,
            checkpoint_sha256=runtime.checkpoint_sha256,
            config_sha256=config_sha256,
            run_meta=runtime.run_meta,
        )
        print(f"Parquet: {parquet_path}")

        print(f"Primary model: {primary_model}")
        print(f"Backend: {runtime.active_backend}")
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
            "frames_no_hand": pipeline.stats.frames_no_hand,
            "frames_failed": pipeline.stats.frames_failed,
            "frames_skipped_invalid_input": (
                pipeline.stats.frames_skipped_invalid_input
            ),
            "average_fps": pipeline.stats.average_fps,
            "frame_status": pipeline.frame_statistics.to_manifest(),
            "expected_frame_count": reader.expected_frame_count,
        }
    finally:
        try:
            writers.close()
        finally:
            runtime.estimator.close()

    backend_name = runtime.active_backend
    checkpoint_sha256 = runtime.checkpoint_sha256
    declared_image_dimensions = _image_dimensions(
        segment,
        reader.video_stream_id,
    )
    actual_image_dimensions = _image_dimensions(
        segment,
        reader.video_stream_id,
        segment_dir,
    )
    manifest_path = (
        Path(args.manifest_output).expanduser()
        if args.manifest_output
        else output_paths.run_manifest
    ).resolve()
    run_mode = "smoke" if args.max_frames is not None else "production"
    frame_status_counts = pipeline.frame_statistics
    full_frame_coverage = (
        args.max_frames is None
        and frame_status_counts.requested == reader.expected_frame_count
        and frame_status_counts.is_complete
    )
    wilor_artifacts_present = (
        output_paths.frame_status is not None
        and output_paths.frame_status.is_file()
        and output_paths.bbox is not None
        and output_paths.bbox.is_file()
    )
    artifact_report: dict[str, Any] | None = None
    if primary_model == "wilor":
        if (
            output_paths.frame_status is None
            or output_paths.bbox is None
        ):
            raise RuntimeError("WiLoR 全帧资产输出路径缺失")
        artifact_report = validate_wilor_frame_artifacts(
            output_paths.frame_status,
            output_paths.bbox,
            reader.sample_map_path,
            expected_frame_count=frame_status_counts.requested,
        )

    model_device = str(
        runtime.run_meta.get("device")
        or runtime.run_meta.get("backend_delegate")
        or runtime.active_backend
    )
    manifest_document: dict[str, Any] = {
        "schema_version": 2,
        "completed": False,
        "run_mode": run_mode,
        "segment_id": reader.segment_id,
        "video_stream_id": reader.video_stream_id,
        "prep_revision": prep_revision,
        "source_kind": source_kind,
        "primary_model": primary_model,
        "backend": backend_name,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "upstream_git_commit": runtime.upstream_git_commit,
        "max_frames": args.max_frames,
        "full_frame_coverage": full_frame_coverage,
        "wilor_requirement_satisfied": False,
        "model": {
            "name": runtime.model_name,
            "version": runtime.model_version,
            "checkpoint_sha256": checkpoint_sha256,
            "device": model_device,
        },
        "coverage": {
            "decoded_frames": pipeline.stats.frames_processed,
            "failed_frames": pipeline.stats.frames_failed,
        },
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
        "frame_artifact_validation": artifact_report,
        "validation_status": None,
        "outputs": {
            "parquet": parquet_path,
            "frame_status": frame_status_path,
            "bbox": bbox_path,
            "validation_report": None,
            "preview": None,
        },
    }
    # WiLoR 专用 Validator 读取 hands_run.json，因此先写可追溯的初始版本。
    _write_json_atomic(manifest_path, manifest_document)

    validation_status: str | None = None
    report_path: Path | None = None
    if args.validate and parquet_path is not None:
        image_width, image_height = actual_image_dimensions
        if image_width is None or image_height is None:
            raise ValueError("Hands Validator 无法确定实际视频分辨率")
        if primary_model == "wilor":
            hands_report = validate_wilor_hands(
                parquet_path,
                hands_run_path=str(manifest_path),
                segment_json_path=str(segment_dir / "segment.json"),
                image_width=image_width,
                image_height=image_height,
                expected_model_version=runtime.model_version,
            )
            report = _combine_validation_reports(
                frame_artifacts=artifact_report or {},
                hands_v1=hands_report,
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
    if args.preview and parquet_path is not None:
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

    artifact_status = (
        str(artifact_report.get("status"))
        if artifact_report is not None
        else "pass"
    )
    exit_code = (
        2
        if validation_status == "fail"
        or (primary_model == "wilor" and artifact_status == "fail")
        else 0
    )
    wilor_requirement_satisfied = (
        full_frame_coverage
        and wilor_artifacts_present
        and artifact_status == "pass"
        if primary_model == "wilor"
        else None
    )
    completed = (
        exit_code == 0
        and run_mode == "production"
        and (
            primary_model != "wilor"
            or bool(wilor_requirement_satisfied)
        )
    )
    manifest_document["completed"] = completed
    manifest_document["wilor_requirement_satisfied"] = (
        wilor_requirement_satisfied
    )
    manifest_document["validation_status"] = validation_status
    manifest_document["outputs"]["validation_report"] = (
        str(report_path) if report_path else None
    )
    manifest_document["outputs"]["preview"] = generated_preview
    _write_json_atomic(manifest_path, manifest_document)
    print(f"Manifest: {manifest_path}")
    if not completed:
        print(
            "Run incomplete: smoke run or required WiLoR full-frame "
            "artifacts are not complete"
        )
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
