"""Run Hands V1 on one Prepared Segment and generate C-side QA artifacts.

This is intentionally a narrow integration utility.  It reads the existing
Prepared contract directly while the general multi-source Hands pipeline remains
owned by the segment-reader/pipeline work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import yaml

from zpds.hands.mediapipe_adapter import MediaPipeHandEstimator
from zpds.hands.preview import generate_hands_preview
from zpds.hands.validator import validate_hands_parquet
from zpds.hands.writer import (
    estimator_provenance,
    write_hands_parquet,
    write_hands_run_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hands V1 for one Prepared Segment")
    parser.add_argument("segment_dir", type=Path, help="Prepared Segment directory")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--video-stream-id", help="RGB MP4 stream ID; defaults to the first RGB MP4"
    )
    parser.add_argument("--experience-version", default="hands_v1")
    parser.add_argument("--output-root", type=Path, help="Defaults to the segment dataset root")
    parser.add_argument("--preview-output", type=Path, help="Defaults to output/hand_preview/")
    return parser.parse_args()


def _select_video_stream(segment: dict, requested_stream_id: str | None) -> dict:
    streams = [stream for stream in segment["streams"] if stream.get("format") == "mp4"]
    if requested_stream_id:
        selected = next(
            (stream for stream in streams if stream["stream_id"] == requested_stream_id), None
        )
        if selected is None:
            raise ValueError(f"Video stream not found: {requested_stream_id}")
        return selected
    rgb_streams = [stream for stream in streams if stream.get("modality") == "rgb"]
    if rgb_streams:
        return rgb_streams[0]
    if streams:
        return streams[0]
    raise ValueError("Prepared Segment has no MP4 video stream")


def _load_frame_metadata(segment_dir: Path, stream: dict) -> dict[int, dict]:
    sample_map_uri = stream.get("origin", {}).get("sample_map_uri")
    if not sample_map_uri:
        raise ValueError(f"Stream {stream['stream_id']} has no sample_map_uri")
    sample_map_path = segment_dir / sample_map_uri
    if not sample_map_path.is_file():
        raise FileNotFoundError(f"Sample map not found: {sample_map_path}")

    sample_map = pd.read_parquet(sample_map_path)
    required = {"output_frame_index", "output_timestamp_ns"}
    missing = required.difference(sample_map.columns)
    if missing:
        raise ValueError(f"Sample map missing required columns: {sorted(missing)}")

    segment_id = segment_dir.name
    return {
        int(row.output_frame_index): {
            "segment_id": segment_id,
            "video_stream_id": stream["stream_id"],
            "output_frame_index": int(row.output_frame_index),
            "timestamp_ns": int(row.output_timestamp_ns),
            "source_frame_index": _nullable_int(row.get("source_frame_index")),
            "source_timestamp_ns": _nullable_int(row.get("source_timestamp_ns")),
        }
        for _, row in sample_map.iterrows()
    }


def _nullable_int(value) -> int | None:
    return None if pd.isna(value) else int(value)


def run_segment(args: argparse.Namespace) -> dict:
    segment_dir = args.segment_dir.resolve()
    segment_path = segment_dir / "segment.json"
    if not segment_path.is_file():
        raise FileNotFoundError(f"segment.json not found: {segment_path}")
    if not args.config.is_file():
        raise FileNotFoundError(f"Hands config not found: {args.config.resolve()}")

    segment = json.loads(segment_path.read_text(encoding="utf-8"))
    stream = _select_video_stream(segment, args.video_stream_id)
    video_path = segment_dir / stream["uri"]
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    frame_metadata = _load_frame_metadata(segment_dir, stream)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    dataset_root = args.output_root.resolve() if args.output_root else segment_dir.parent.parent
    experience_root = dataset_root / "experiences" / args.experience_version
    pose_path = experience_root / "assets" / "poses" / f"{segment_dir.name}_hands_2d.parquet"
    validation_path = experience_root / "reports" / f"{segment_dir.name}_hands_validation.json"
    run_report_path = experience_root / "reports" / f"{segment_dir.name}_hands_run.json"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    image_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    observations = []
    estimator = MediaPipeHandEstimator.from_yaml(args.config)
    try:
        frame_index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            meta = frame_metadata.get(frame_index)
            if meta is None:
                raise ValueError(f"No sample-map row for output_frame_index={frame_index}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            hands = estimator.estimate(frame_rgb, timestamp_ms=meta["timestamp_ns"] // 1_000_000)
            observations.append({"frame_meta": meta, "hands": hands})
            frame_index += 1
    finally:
        capture.release()

    provenance, run_report = estimator_provenance(estimator, config)
    estimator.close()
    model_meta = {key: provenance[key] for key in (
        "model_name", "model_version", "checkpoint_sha256", "config_sha256"
    )}
    run_meta = {key: provenance[key] for key in (
        "backend_requested", "backend_active", "backend_fallback_used",
        "backend_fallback_reason", "backend_delegate"
    )}
    hands_path = write_hands_parquet(
        observations,
        str(pose_path),
        prep_revision=segment.get("prep_revision")
        or segment.get("record_revision", ""),
        model_meta=model_meta,
        run_meta=run_meta,
    )

    validation = validate_hands_parquet(
        hands_path,
        segment_json_path=str(segment_path),
        image_width=image_width,
        image_height=image_height,
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    preview_path = generate_hands_preview(
        str(segment_dir),
        hands_path,
        output_path=str(args.preview_output) if args.preview_output else None,
        video_stream_id=stream["stream_id"],
    )
    run_report.update({
        "segment_id": segment_dir.name,
        "video_stream_id": stream["stream_id"],
        "video_resolution": [image_width, image_height],
        "processed_frames": len(observations),
        "detected_hand_rows": sum(len(item["hands"]) for item in observations),
        "hands_parquet": hands_path,
        "validation_report": str(validation_path.resolve()),
        "preview_video": preview_path,
    })
    run_report_path_str = write_hands_run_report(run_report, str(run_report_path))
    return {
        "hands_parquet": hands_path,
        "validation_report": str(validation_path.resolve()),
        "run_report": run_report_path_str,
        "preview_video": preview_path,
        "validation_status": validation["status"],
    }


def main() -> None:
    result = run_segment(_parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
