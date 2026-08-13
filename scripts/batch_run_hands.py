"""Hands V1 多 Prepared Segment 批处理入口。

支持断点续跑、强制重跑、失败隔离和 JSON 汇总报告。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# 直接运行脚本时确保可以导入项目根目录下的 scripts 包。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_hands as single_cli
from zpds.hands.config import HandsPipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量生成 Hands V1 关键点标注",
    )
    parser.add_argument(
        "--segments-root",
        required=True,
        help="包含多个 Prepared Segment 目录的根目录",
    )
    parser.add_argument(
        "--pattern",
        default="seg_*",
        help="Segment 目录匹配模式，默认 seg_*",
    )
    parser.add_argument("--stream-id", help="指定所有 Segment 使用的 RGB Stream ID")
    parser.add_argument("--config", default="config.yaml", help="Hands YAML 配置")
    parser.add_argument(
        "--source-kind",
        choices=["ego", "non_ego"],
        default="non_ego",
        help="所有输入 Segment 的来源类型（单后端恒 WiLoR，仅作 manifest 记录）",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "批处理输出根目录；不提供时写入各 Segment 目录下的 "
            "hands/<stream_id>/"
        ),
    )
    parser.add_argument("--summary-output", help="批处理汇总 JSON 路径")
    parser.add_argument("--max-frames", type=int, help="每个 Segment 最多处理帧数")
    parser.add_argument("--limit", type=int, help="最多处理的 Segment 数量")
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否为每个结果生成 Validator 报告，默认开启",
    )
    parser.add_argument("--preview", action="store_true", help="生成每个 Segment 的预览视频")
    parser.add_argument("--force", action="store_true", help="忽略已有结果并强制重跑")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="首个 Segment 失败后停止；默认继续处理后续 Segment",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def discover_segments(root: Path, pattern: str) -> list[Path]:
    """按名称稳定排序发现 Segment 目录。"""
    if not root.is_dir():
        raise NotADirectoryError(f"Prepared Segment 根目录不存在: {root}")
    return sorted(
        (path.resolve() for path in root.glob(pattern) if path.is_dir()),
        key=lambda path: path.name,
    )


def _segment_identity(
    segment_dir: Path,
    requested_stream_id: str | None,
) -> tuple[dict[str, Any], str, str]:
    segment = single_cli._read_segment_json(segment_dir)
    segment_id = str(segment.get("segment_id") or segment_dir.name)
    streams = segment.get("streams")
    if not isinstance(streams, list):
        raise TypeError(f"segment.json 的 streams 必须是数组: {segment_dir}")

    rgb_stream_ids = [
        str(stream["stream_id"])
        for stream in streams
        if isinstance(stream, dict)
        and stream.get("modality") == "rgb"
        and stream.get("format") == "mp4"
        and stream.get("stream_id")
    ]
    if requested_stream_id is not None:
        if requested_stream_id not in rgb_stream_ids:
            raise ValueError(
                f"RGB Stream {requested_stream_id!r} 不存在，实际为 {rgb_stream_ids}"
            )
        stream_id = requested_stream_id
    elif len(rgb_stream_ids) == 1:
        stream_id = rgb_stream_ids[0]
    else:
        raise ValueError(
            f"需要通过 --stream-id 指定 RGB Stream，候选为 {rgb_stream_ids}"
        )
    return segment, segment_id, stream_id


def _expected_provenance(
    config_path: Path,
) -> tuple[str, str, str]:
    runtime_config = HandsPipelineConfig.load(config_path)
    return (
        runtime_config.config_sha256,
        runtime_config.wilor.checkpoint_sha256,
        runtime_config.wilor.upstream_commit,
    )


def _output_paths(
    output_root: Path | None,
    segment_dir: Path,
    segment_id: str,
    stream_id: str,
) -> dict[str, Path]:
    directory = (
        output_root / segment_id / stream_id
        if output_root is not None
        else segment_dir / "hands" / stream_id
    )
    return {
        "directory": directory,
        "parquet": directory / "hands_2d.parquet",
        "report": directory / "hands_validation.json",
        "preview": directory / "hands_preview.mp4",
        "manifest": directory / "hands_run.json",
        "frame_status": directory / "wilor_frame_status.parquet",
        "bbox": directory / "wilor_hands_bbox.parquet",
    }


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return document


def _existing_output_can_be_skipped(
    *,
    segment_dir: Path,
    segment_id: str,
    stream_id: str,
    paths: dict[str, Path],
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
    max_frames: int | None,
    expected_upstream_git_commit: str = "",
) -> tuple[bool, str]:
    required = [paths["manifest"], paths["frame_status"], paths["bbox"]]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return False, f"缺少产物: {missing}"

    try:
        manifest = _read_json(paths["manifest"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return False, f"运行清单不可读: {error}"

    if not manifest.get("completed"):
        return False, "上次运行未完成"
    if manifest.get("run_mode", "production") != "production":
        return False, "上次运行不是 production"
    if manifest.get("segment_id") != segment_id:
        return False, "Segment ID 与运行清单不一致"
    if manifest.get("video_stream_id") != stream_id:
        return False, "Stream ID 与运行清单不一致"
    if manifest.get("max_frames") != max_frames:
        return False, "max_frames 与上次运行不一致"
    if manifest.get("config_sha256") != expected_config_sha256:
        return False, "配置哈希已变化"
    if manifest.get("checkpoint_sha256") != expected_checkpoint_sha256:
        return False, "模型哈希已变化"
    manifest_primary_model = manifest.get("primary_model", "mediapipe")
    if manifest_primary_model != "wilor":
        return False, "主模型已变化"
    if manifest.get("upstream_git_commit") != expected_upstream_git_commit:
        return False, "WiLoR upstream commit 已变化"
    if not manifest.get("wilor_requirement_satisfied"):
        return False, "WiLoR 全帧要求未满足"
    statistics = manifest.get("statistics", {})
    if not isinstance(statistics, dict):
        return False, "WiLoR statistics 格式错误"
    frame_status = statistics.get("frame_status", {})
    if not isinstance(frame_status, dict):
        return False, "WiLoR frame-status 统计格式错误"
    expected_frames = statistics.get("expected_frame_count")
    requested = frame_status.get("requested")
    accounted = sum(
        int(frame_status.get(name, 0))
        for name in (
            "detected",
            "no_hand",
            "failed",
            "skipped_invalid_input",
        )
    )
    if requested != expected_frames or requested != accounted:
        return False, "WiLoR frame-status 统计不完整"
    return True, "已有 WiLoR 全帧产物校验通过"


def _collect_output_statistics(
    paths: dict[str, Path],
) -> dict[str, Any]:
    manifest = _read_json(paths["manifest"])
    run_statistics = manifest.get("statistics", {})
    if not paths["parquet"].is_file():
        return {
            "rows": 0,
            "annotated_frames": 0,
            "run": run_statistics,
            "validation_status": manifest.get("validation_status"),
            "primary_model": manifest.get("primary_model"),
            "wilor_requirement_satisfied": manifest.get(
                "wilor_requirement_satisfied"
            ),
        }

    frame = pd.read_parquet(paths["parquet"])
    frames_processed = int(run_statistics.get("frames_processed", 0))
    annotated_frames = int(frame["output_frame_index"].nunique())
    confidence = frame["handedness_score"]
    return {
        "rows": len(frame),
        "annotated_frames": annotated_frames,
        "no_hand_frames": max(0, frames_processed - annotated_frames),
        "handedness": {
            str(key): int(value)
            for key, value in frame["handedness"].value_counts().to_dict().items()
        },
        "confidence": {
            "minimum": None if frame.empty else float(confidence.min()),
            "mean": None if frame.empty else float(confidence.mean()),
            "low_confidence_rows": int((confidence < 0.5).sum()),
        },
        "run": run_statistics,
        "validation_status": manifest.get("validation_status"),
    }


def _summary_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(items),
        "completed": sum(item["status"] == "completed" for item in items),
        "skipped": sum(item["status"] == "skipped" for item in items),
        "failed": sum(item["status"] == "failed" for item in items),
    }


def run(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    segments_root = Path(args.segments_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else None
    )
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else (
            output_root / "batch_summary.json"
            if output_root is not None
            else segments_root / "hands_batch_summary.json"
        )
    )
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0")

    segment_dirs = discover_segments(segments_root, args.pattern)
    if args.limit is not None:
        segment_dirs = segment_dirs[: args.limit]
    if not segment_dirs:
        raise ValueError(
            f"未发现匹配 {args.pattern!r} 的 Prepared Segment: {segments_root}"
        )

    (
        expected_config_sha256,
        expected_checkpoint_sha256,
        expected_upstream_git_commit,
    ) = _expected_provenance(config_path)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": None,
        "segments_root": str(segments_root),
        "config": str(config_path),
        "config_sha256": expected_config_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "primary_model": "wilor",
        "upstream_git_commit": expected_upstream_git_commit,
        "options": {
            "pattern": args.pattern,
            "stream_id": args.stream_id,
            "source_kind": args.source_kind,
            "max_frames": args.max_frames,
            "validate": args.validate,
            "preview": args.preview,
            "force": args.force,
        },
        "counts": {"total": 0, "completed": 0, "skipped": 0, "failed": 0},
        "items": [],
    }

    for index, segment_dir in enumerate(segment_dirs, start=1):
        item_started = time.perf_counter()
        item: dict[str, Any] = {
            "segment_dir": str(segment_dir),
            "status": "failed",
        }
        print(f"[{index}/{len(segment_dirs)}] {segment_dir.name}")
        try:
            segment, segment_id, stream_id = _segment_identity(
                segment_dir,
                args.stream_id,
            )
            paths = _output_paths(
                output_root,
                segment_dir,
                segment_id,
                stream_id,
            )
            item.update(
                {
                    "segment_id": segment_id,
                    "video_stream_id": stream_id,
                    "outputs": {
                        key: str(value)
                        for key, value in paths.items()
                        if key != "directory"
                    },
                }
            )

            can_skip, skip_reason = _existing_output_can_be_skipped(
                segment_dir=segment_dir,
                segment_id=segment_id,
                stream_id=stream_id,
                paths=paths,
                expected_config_sha256=expected_config_sha256,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                max_frames=args.max_frames,
                expected_upstream_git_commit=expected_upstream_git_commit,
            )
            if can_skip and not args.force:
                item["status"] = "skipped"
                item["reason"] = skip_reason
                item["statistics"] = _collect_output_statistics(paths)
                print(f"  跳过: {skip_reason}")
            else:
                paths["directory"].mkdir(parents=True, exist_ok=True)
                exit_code = single_cli.run(
                    argparse.Namespace(
                        segment=str(segment_dir),
                        stream_id=stream_id,
                        config=str(config_path),
                        source_kind=args.source_kind,
                        output=str(paths["parquet"]),
                        frame_status_output=str(paths["frame_status"]),
                        bbox_output=str(paths["bbox"]),
                        prep_revision=None,
                        max_frames=args.max_frames,
                        validate=args.validate,
                        report_output=str(paths["report"]),
                        preview=args.preview,
                        preview_output=str(paths["preview"]),
                        manifest_output=str(paths["manifest"]),
                        experience_dir=None,
                        experience_version=None,
                    )
                )
                if exit_code != 0:
                    raise RuntimeError(f"单 Segment 命令退出码为 {exit_code}")
                manifest = _read_json(paths["manifest"])
                if not manifest.get("completed"):
                    raise RuntimeError(
                        "单 Segment 运行未达到 production 完成条件"
                    )
                item["status"] = "completed"
                item["statistics"] = _collect_output_statistics(paths)
                if can_skip:
                    item["reason"] = "使用 --force 强制重跑"
                elif skip_reason:
                    item["reason"] = skip_reason
                print("  完成")
        except Exception as error:  # noqa: BLE001
            item["status"] = "failed"
            item["error"] = f"{type(error).__name__}: {error}"
            print(f"  失败: {item['error']}", file=sys.stderr)

        item["duration_seconds"] = round(time.perf_counter() - item_started, 3)
        summary["items"].append(item)
        summary["counts"] = _summary_counts(summary["items"])
        _write_json_atomic(summary_path, summary)
        if item["status"] == "failed" and args.fail_fast:
            break

    summary["finished_at"] = _utc_now()
    summary["counts"] = _summary_counts(summary["items"])
    _write_json_atomic(summary_path, summary)
    print(
        "Batch: "
        f"completed={summary['counts']['completed']}, "
        f"skipped={summary['counts']['skipped']}, "
        f"failed={summary['counts']['failed']}"
    )
    print(f"Summary: {summary_path}")
    return 1 if summary["counts"]["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:  # noqa: BLE001
        print(f"Hands batch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
