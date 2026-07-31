"""Run a non-production WiLoR smoke test on 20-50 Prepared frames.

This diagnostic intentionally does not create the formal Person C Parquet
artifacts. It records frame status, boxes, handedness, timing, provenance, and
CUDA memory in JSON so Person A can validate orchestration before the formal
writers are available.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zpds.hands.config import HandsPipelineConfig
from zpds.hands.estimator_factory import (
    create_hand_estimator,
    validate_estimator_runtime,
)
from zpds.hands.pipeline import HandsPipeline
from zpds.hands.segment_reader import PreparedSegmentReader
from zpds.hands.wilor_preflight import check_wilor_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", required=True, help="Prepared Segment 目录")
    parser.add_argument("--config", default="config.yaml", help="Hands 配置")
    parser.add_argument("--stream-id", help="可选 RGB stream_id")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Smoke 帧数，必须在 20 到 50 之间",
    )
    parser.add_argument(
        "--output",
        default="output/hands/wilor_segment_smoke.json",
        help="诊断 JSON 输出路径",
    )
    return parser


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _hand_document(hand: Any) -> dict[str, Any]:
    return {
        "handedness": str(hand.handedness),
        "handedness_score": float(hand.handedness_score),
        "detection_score": float(hand.detection_score),
        "bbox_xyxy_px": [
            float(hand.bbox.x1),
            float(hand.bbox.y1),
            float(hand.bbox.x2),
            float(hand.bbox.y2),
        ],
        "keypoint_count": len(hand.keypoints.pixel),
        "keypoints_clipped_count": int(hand.keypoints.clipped_count),
    }


def run(args: argparse.Namespace) -> int:
    if not 20 <= args.max_frames <= 50:
        raise ValueError("--max-frames 必须在 20 到 50 之间")

    config = HandsPipelineConfig.load(args.config)
    if not config.wilor.upstream_license_checked:
        raise RuntimeError("WiLoR 许可证尚未确认")

    preflight = check_wilor_assets(config.wilor)
    if not preflight.ready:
        raise RuntimeError(
            "WiLoR 资产预检失败: " + "; ".join(preflight.errors)
        )

    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(Path(config.wilor.source_path).parent / "yolo_config"),
    )
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    reader = PreparedSegmentReader(args.segment, args.stream_id)
    if reader.expected_frame_count < args.max_frames:
        raise RuntimeError(
            "Prepared Segment 帧数不足: "
            f"required={args.max_frames}, actual={reader.expected_frame_count}"
        )

    runtime = create_hand_estimator("wilor", config)
    validate_estimator_runtime("wilor", runtime, config)
    estimator = runtime.estimator
    pipeline = HandsPipeline(
        reader,
        estimator,
        model_name=runtime.model_name,
        model_version=runtime.model_version,
        active_backend=runtime.active_backend,
        max_frames=args.max_frames,
    )

    import torch

    torch.cuda.reset_peak_memory_stats()
    cuda_before = (
        torch.cuda.memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    frames: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    started_at = time.perf_counter()

    try:
        for record in pipeline.run_frames():
            frame = record.frame
            statuses[record.inference_status] += 1
            frames.append(
                {
                    "output_frame_index": frame.output_frame_index,
                    "timestamp_ns": frame.timestamp_ns,
                    "source_frame_index": frame.source_frame_index,
                    "source_timestamp_ns": frame.source_timestamp_ns,
                    "model_timestamp_ms": (
                        frame.timestamp_ns // 1_000_000
                    ),
                    "inference_status": record.inference_status,
                    "failure_reason": record.failure_reason,
                    "inference_ms": float(record.inference_ms),
                    "hand_count": len(record.raw_hands),
                    "hands": [
                        _hand_document(hand)
                        for hand in record.raw_hands
                    ],
                }
            )
    finally:
        estimator.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_seconds = time.perf_counter() - started_at
    cuda_peak = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    cuda_after = (
        torch.cuda.memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    processed = len(frames)
    accounted = sum(statuses.values())
    no_silent_mediapipe_fallback = (
        runtime.active_backend == "wilor"
        and runtime.run_meta.get("backend_fallback_used") is False
    )

    report = {
        "artifact_kind": "person_a_wilor_smoke_diagnostic",
        "formal_person_c_artifact": False,
        "segment_id": reader.segment_id,
        "video_stream_id": reader.video_stream_id,
        "video_path": str(reader.video_path),
        "sample_map_path": str(reader.sample_map_path),
        "sample_map_rows": reader.expected_frame_count,
        "requested_frames": args.max_frames,
        "processed_frames": processed,
        "accounted_frames": accounted,
        "full_smoke_coverage": (
            processed == args.max_frames and accounted == processed
        ),
        "statuses": dict(sorted(statuses.items())),
        "fallback_attempted_frames": 0,
        "fallback_used_frames": 0,
        "no_silent_mediapipe_fallback": no_silent_mediapipe_fallback,
        "elapsed_seconds": elapsed_seconds,
        "average_fps": (
            processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "cuda_memory_mib": {
            "before": cuda_before,
            "peak": cuda_peak,
            "after_close": cuda_after,
        },
        "model": {
            "name": runtime.model_name,
            "version": runtime.model_version,
            "active_backend": runtime.active_backend,
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "upstream_git_commit": runtime.upstream_git_commit,
            "config_sha256": config.config_sha256,
            "run_meta": runtime.run_meta,
        },
        "frames": frames,
    }
    output_path = Path(args.output).expanduser().resolve()
    _atomic_write_json(output_path, report)

    print(f"Smoke report: {output_path}")
    print(
        "Coverage: "
        f"processed={processed}, accounted={accounted}, "
        f"statuses={dict(statuses)}"
    )
    print(
        "Performance: "
        f"elapsed={elapsed_seconds:.2f}s, "
        f"fps={report['average_fps']:.2f}, "
        f"cuda_peak={cuda_peak:.1f}MiB, "
        f"cuda_after_close={cuda_after:.1f}MiB"
    )
    print(
        "No silent MediaPipe fallback: "
        f"{no_silent_mediapipe_fallback}"
    )
    return 0 if (
        report["full_smoke_coverage"]
        and no_silent_mediapipe_fallback
    ) else 2


def main() -> None:
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except Exception as error:
        print(f"WiLoR segment smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
