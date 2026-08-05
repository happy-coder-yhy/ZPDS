"""将场景融合 JSON 渲染为带边界标记的预览视频。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成场景融合结果预览视频")
    parser.add_argument("--input", required=True, help="原始视频路径")
    parser.add_argument("--scene-json", required=True, help="scene_all JSON 路径")
    parser.add_argument("--output", required=True, help="输出 MP4 路径")
    parser.add_argument("--max-frames", type=int, help="最多渲染帧数")
    return parser


def _scene_for_timestamp(scenes: list[dict], timestamp_ns: int) -> tuple[int, dict]:
    for index, scene in enumerate(scenes):
        if int(scene["start_ns"]) <= timestamp_ns < int(scene["end_ns"]):
            return index, scene
    return len(scenes) - 1, scenes[-1]


def _sources(scene: dict) -> str:
    values = scene.get("sources") or []
    return ", ".join(str(value) for value in values) or "unknown"


def render_preview(
    input_path: Path,
    scene_json: Path,
    output_path: Path,
    max_frames: int | None = None,
) -> int:
    document = json.loads(scene_json.read_text(encoding="utf-8"))
    scenes = list(document.get("scenes") or [])
    if not scenes:
        raise ValueError("场景 JSON 中没有 scenes，需使用 --stage all 的输出")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开输入视频: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("输入视频 FPS 或分辨率无效")

    json_frame_count = int(document.get("frame_count") or 0)
    frame_limit = max_frames if max_frames is not None else json_frame_count or None
    boundary_frames = {
        max(0, round(int(scene["start_ns"]) * fps / 1_000_000_000))
        for scene in scenes[1:]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    colors = ((48, 190, 255), (110, 220, 90), (255, 170, 70), (210, 90, 255))
    frame_index = 0
    try:
        while frame_limit is None or frame_index < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ns = round(frame_index * 1_000_000_000 / fps)
            scene_index, scene = _scene_for_timestamp(scenes, timestamp_ns)
            color = colors[scene_index % len(colors)]

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, min(height, 112)), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
            cv2.rectangle(frame, (0, 0), (12, height), color, -1)

            seconds = timestamp_ns / 1_000_000_000
            confidence = float(scene.get("confidence") or 0.0)
            cv2.putText(
                frame,
                f"SCENE {scene_index + 1}/{len(scenes)}   t={seconds:06.2f}s",
                (28, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"confidence={confidence:.3f}   sources={_sources(scene)}",
                (28, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

            if any(abs(frame_index - boundary) <= 3 for boundary in boundary_frames):
                cv2.rectangle(frame, (4, 4), (width - 5, height - 5), (0, 0, 255), 8)
                cv2.putText(
                    frame,
                    "SCENE BOUNDARY",
                    (max(20, width // 2 - 180), max(145, height // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("输入视频没有可渲染帧")
    return frame_index


def main() -> int:
    args = build_parser().parse_args()
    rendered = render_preview(
        Path(args.input).expanduser().resolve(),
        Path(args.scene_json).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        args.max_frames,
    )
    print(f"预览视频已生成，共 {rendered} 帧: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
