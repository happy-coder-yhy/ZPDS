from __future__ import annotations

import copy

import pytest

from zpds.scene.review import build_review_items
from zpds.scene.review_evaluation import evaluate_reviews


def _case() -> dict:
    return {
        "name": "fixture",
        "profile": "test",
        "input": "fixture.mp4",
        "input_sha256": "b" * 64,
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


def _fill(review: dict, reviewer_id: str, decision: str) -> None:
    review.update(
        {
            "reviewer_id": reviewer_id,
            "decision": decision,
            "boundary_type": "hard_cut" if decision == "true_boundary" else None,
            "confidence": 0.9,
        }
    )


def test_empty_template_reports_incomplete_without_fabricating_metrics() -> None:
    items = build_review_items(_case(), negative_count=1)

    report = evaluate_reviews(items)

    assert report["review_complete"] is False
    assert len(report["incomplete_item_ids"]) == 2
    assert report["candidate_precision"] is None
    assert report["recall"] is None
    assert report["formal_calibration_ready"] is False


def test_agreement_and_adjudication_produce_provisional_precision() -> None:
    items = build_review_items(_case(), negative_count=1)
    candidate = next(item for item in items if item["predicted_boundary"])
    negative = next(item for item in items if not item["predicted_boundary"])
    _fill(candidate["reviews"][0], "reviewer-a", "true_boundary")
    _fill(candidate["reviews"][1], "reviewer-b", "true_boundary")
    _fill(negative["reviews"][0], "reviewer-a", "true_boundary")
    _fill(negative["reviews"][1], "reviewer-b", "no_boundary")
    negative["adjudication"].update(
        {
            "adjudicator_id": "reviewer-c",
            "decision": "no_boundary",
            "boundary_type": None,
            "confidence": 0.8,
        }
    )

    report = evaluate_reviews(items)

    assert report["review_complete"] is True
    assert report["agreement_count"] == 1
    assert len(report["disagreement_item_ids"]) == 1
    assert report["candidate_precision"] == 1.0
    assert report["negative_audit_missed_boundary_count"] == 0
    assert report["formal_calibration_ready"] is False


def test_same_reviewer_cannot_fill_both_slots() -> None:
    item = build_review_items(_case(), negative_count=0)[0]
    _fill(item["reviews"][0], "same", "true_boundary")
    _fill(item["reviews"][1], "same", "true_boundary")

    with pytest.raises(ValueError, match="不同 reviewer_id"):
        evaluate_reviews([copy.deepcopy(item)])
