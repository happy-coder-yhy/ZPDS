from __future__ import annotations

import json
from pathlib import Path

import cv2

from zpds.scene.review import (
    build_review_items,
    export_evidence_frames,
    write_review_jsonl,
)
from zpds.scene.testing import hard_cut_fixture


def _case() -> dict:
    return {
        "name": "fixture",
        "profile": "test",
        "input": "fixture.mp4",
        "input_sha256": "a" * 64,
        "frame_count": 20,
        "fps": 10.0,
        "fused_transitions": [
            {
                "frame_index": 10,
                "timestamp_ns": 1_000_000_000,
                "score": 0.9,
                "is_hard_cut": True,
                "sources": ["ssim", "optical_flow"],
            }
        ],
    }


def test_review_items_include_candidates_negatives_and_blank_dual_reviews() -> None:
    items = build_review_items(_case(), negative_count=2, context_s=0.2)

    assert sum(item["sample_source"] == "detected_candidate" for item in items) == 1
    assert sum(item["sample_source"] == "negative_audit" for item in items) == 2
    candidate = next(item for item in items if item["predicted_boundary"])
    assert candidate["evidence_frame_indices"] == [8, 10, 12]
    assert len(candidate["reviews"]) == 2
    assert all(review["decision"] is None for review in candidate["reviews"])
    assert candidate["adjudication"]["decision"] is None


def test_evidence_export_and_jsonl_round_trip(tmp_path: Path) -> None:
    fixture = hard_cut_fixture()
    video_path = tmp_path / "fixture.mp4"
    height, width = fixture.frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fixture.fps,
        (width, height),
    )
    assert writer.isOpened()
    for frame in fixture.frames:
        writer.write(frame)
    writer.release()
    items = build_review_items(_case(), negative_count=0, context_s=0.2)

    export_evidence_frames(
        video_path,
        items,
        evidence_dir=tmp_path / "evidence",
        uri_prefix="evidence/fixture",
    )
    output = tmp_path / "reviews.jsonl"
    write_review_jsonl(output, items)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert len(row["evidence_frame_uris"]) == 3
    assert len(list((tmp_path / "evidence").glob("*.png"))) == 3
