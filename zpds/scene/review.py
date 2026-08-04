"""Scene 边界双人复核样本选择与证据帧导出。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def safe_case_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return cleaned or "case"


def _review_item_id(input_sha256: str, frame_index: int, source: str) -> str:
    payload = f"{input_sha256}:{frame_index}:{source}".encode()
    return f"scene_review_{hashlib.sha256(payload).hexdigest()[:16]}"


def _negative_frames(
    *,
    frame_count: int,
    candidate_frames: Sequence[int],
    count: int,
    exclusion_frames: int,
) -> list[int]:
    if count <= 0:
        return []
    candidates = set(candidate_frames)
    pool = np.linspace(0, frame_count - 1, max(count * 8, count), dtype=int)
    selected: list[int] = []
    for frame_index in sorted({int(value) for value in pool}):
        if any(abs(frame_index - candidate) <= exclusion_frames for candidate in candidates):
            continue
        selected.append(frame_index)
        if len(selected) == count:
            break
    return selected


def build_review_items(
    case: dict[str, Any],
    *,
    negative_count: int = 8,
    context_s: float = 0.5,
) -> list[dict[str, Any]]:
    """为融合候选和负样本审计创建空白双人复核记录。"""

    frame_count = int(case["frame_count"])
    fps = float(case["fps"])
    if frame_count <= 0 or fps <= 0:
        raise ValueError("回归案例的 frame_count 和 fps 必须大于 0")
    if negative_count < 0:
        raise ValueError("negative_count 不能为负数")
    if context_s <= 0:
        raise ValueError("context_s 必须大于 0")
    input_sha256 = str(case["input_sha256"])
    transitions = list(case.get("fused_transitions", []))
    candidate_frames = sorted({int(item["frame_index"]) for item in transitions})
    context_frames = max(1, round(context_s * fps))
    negatives = _negative_frames(
        frame_count=frame_count,
        candidate_frames=candidate_frames,
        count=negative_count,
        exclusion_frames=context_frames,
    )
    transition_by_frame = {int(item["frame_index"]): item for item in transitions}
    selections = [
        (frame_index, "detected_candidate") for frame_index in candidate_frames
    ] + [(frame_index, "negative_audit") for frame_index in negatives]

    items = []
    for frame_index, source in sorted(selections):
        evidence_indices = (
            max(0, frame_index - context_frames),
            frame_index,
            min(frame_count - 1, frame_index + context_frames),
        )
        prediction = transition_by_frame.get(frame_index)
        items.append(
            {
                "schema_version": "zpds.scene.boundary_review.v1",
                "review_item_id": _review_item_id(input_sha256, frame_index, source),
                "case_name": case["name"],
                "profile": case.get("profile"),
                "input": case["input"],
                "input_sha256": input_sha256,
                "sample_source": source,
                "predicted_boundary": source == "detected_candidate",
                "frame_index": frame_index,
                "timestamp_ns": round(frame_index * 1_000_000_000 / fps),
                "evidence_frame_indices": list(evidence_indices),
                "evidence_frame_uris": [],
                "prediction": prediction,
                "reviews": [
                    {
                        "reviewer_slot": slot,
                        "reviewer_id": "",
                        "decision": None,
                        "boundary_type": None,
                        "confidence": None,
                        "notes": "",
                    }
                    for slot in ("reviewer_1", "reviewer_2")
                ],
                "adjudication": {
                    "required": True,
                    "adjudicator_id": "",
                    "decision": None,
                    "boundary_type": None,
                    "confidence": None,
                    "notes": "",
                },
            }
        )
    return items


def export_evidence_frames(
    video_path: str | Path,
    items: Sequence[dict[str, Any]],
    *,
    evidence_dir: str | Path,
    uri_prefix: str,
) -> None:
    """顺序解码一次视频并为复核项写出前/中/后三帧 PNG。"""

    target = Path(evidence_dir)
    target.mkdir(parents=True, exist_ok=True)
    required = sorted(
        {
            int(index)
            for item in items
            for index in item["evidence_frame_indices"]
        }
    )
    by_index: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(Path(video_path).resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开复核视频: {video_path}")
    try:
        required_set = set(required)
        index = 0
        while required_set:
            ok, frame = capture.read()
            if not ok:
                break
            if index in required_set:
                by_index[index] = frame
                required_set.remove(index)
            index += 1
    finally:
        capture.release()
    missing = sorted(set(required) - set(by_index))
    if missing:
        raise RuntimeError(f"无法解码复核证据帧: {missing}")

    for item in items:
        uris = []
        for position, frame_index in zip(
            ("before", "center", "after"),
            item["evidence_frame_indices"],
        ):
            filename = f"{item['review_item_id']}_{position}_{frame_index:08d}.png"
            output = target / filename
            if not cv2.imwrite(str(output), by_index[int(frame_index)]):
                raise RuntimeError(f"写出证据帧失败: {output}")
            uris.append(f"{uri_prefix.rstrip('/')}/{filename}")
        item["evidence_frame_uris"] = uris


def write_review_jsonl(path: str | Path, items: Sequence[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    temporary.replace(output)


__all__ = [
    "build_review_items",
    "export_evidence_frames",
    "safe_case_name",
    "write_review_jsonl",
]
