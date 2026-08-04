"""Scene 双人复核完整性、一致性与 provisional 指标评估。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ALLOWED_DECISIONS = {"true_boundary", "no_boundary", "uncertain"}
ALLOWED_BOUNDARY_TYPES = {"hard_cut", "gradual", "semantic", "other"}


def load_review_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"复核 JSONL 第 {line_number} 行非法") from error
        if not isinstance(row, dict):
            raise TypeError(f"复核 JSONL 第 {line_number} 行必须是对象")
        rows.append(row)
    if not rows:
        raise ValueError("复核 JSONL 不能为空")
    return rows


def _completed_review(review: dict[str, Any]) -> bool:
    return bool(review.get("reviewer_id")) and review.get("decision") is not None


def _validate_completed_review(review: dict[str, Any], *, item_id: str) -> None:
    decision = review["decision"]
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"{item_id} 包含非法复核 decision: {decision!r}")
    confidence = review.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError(f"{item_id} 已完成复核必须填写数值 confidence")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"{item_id} confidence 必须在 [0, 1]")
    boundary_type = review.get("boundary_type")
    if decision == "true_boundary" and boundary_type not in ALLOWED_BOUNDARY_TYPES:
        raise ValueError(f"{item_id} true_boundary 必须填写合法 boundary_type")
    if decision == "no_boundary" and boundary_type is not None:
        raise ValueError(f"{item_id} no_boundary 的 boundary_type 必须为 null")


def evaluate_reviews(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("复核项不能为空")
    seen_ids: set[str] = set()
    incomplete: list[str] = []
    disagreements: list[str] = []
    unresolved: list[str] = []
    effective: list[tuple[dict[str, Any], str]] = []
    paired_count = 0
    agreement_count = 0

    for item in items:
        item_id = str(item.get("review_item_id", ""))
        if not item_id or item_id in seen_ids:
            raise ValueError(f"复核项 ID 为空或重复: {item_id!r}")
        seen_ids.add(item_id)
        reviews = item.get("reviews")
        if not isinstance(reviews, list) or len(reviews) != 2:
            raise ValueError(f"{item_id} 必须包含两条独立复核")
        if not all(isinstance(review, dict) for review in reviews):
            raise TypeError(f"{item_id} reviews 必须是对象")
        if not all(_completed_review(review) for review in reviews):
            incomplete.append(item_id)
            continue
        for review in reviews:
            _validate_completed_review(review, item_id=item_id)
        reviewer_ids = {str(review["reviewer_id"]) for review in reviews}
        if len(reviewer_ids) != 2:
            raise ValueError(f"{item_id} 的两条复核必须来自不同 reviewer_id")
        paired_count += 1
        decisions = [str(review["decision"]) for review in reviews]
        agreed = decisions[0] == decisions[1] and decisions[0] != "uncertain"
        if agreed:
            agreement_count += 1
            effective.append((item, decisions[0]))
            continue

        disagreements.append(item_id)
        adjudication = item.get("adjudication")
        if not isinstance(adjudication, dict):
            unresolved.append(item_id)
            continue
        adjudication_review = {
            "reviewer_id": adjudication.get("adjudicator_id", ""),
            "decision": adjudication.get("decision"),
            "boundary_type": adjudication.get("boundary_type"),
            "confidence": adjudication.get("confidence"),
        }
        if not _completed_review(adjudication_review):
            unresolved.append(item_id)
            continue
        _validate_completed_review(adjudication_review, item_id=item_id)
        if adjudication_review["decision"] == "uncertain":
            unresolved.append(item_id)
            continue
        if adjudication_review["reviewer_id"] in reviewer_ids:
            raise ValueError(f"{item_id} 仲裁人员不能是原两名复核人员")
        effective.append((item, str(adjudication_review["decision"])))

    candidate_labels = [
        decision
        for item, decision in effective
        if item.get("sample_source") == "detected_candidate"
    ]
    true_positive = candidate_labels.count("true_boundary")
    false_positive = candidate_labels.count("no_boundary")
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    negative_labels = [
        decision
        for item, decision in effective
        if item.get("sample_source") == "negative_audit"
    ]
    audit_missed = negative_labels.count("true_boundary")
    audit_hit_rate = audit_missed / len(negative_labels) if negative_labels else None
    review_complete = not incomplete and not unresolved

    return {
        "schema_version": "zpds.scene.boundary_review_evaluation.v1",
        "review_item_count": len(items),
        "paired_review_count": paired_count,
        "agreement_count": agreement_count,
        "agreement_rate": agreement_count / paired_count if paired_count else None,
        "incomplete_item_ids": incomplete,
        "disagreement_item_ids": disagreements,
        "unresolved_item_ids": unresolved,
        "review_complete": review_complete,
        "effective_label_count": len(effective),
        "candidate_true_positive": true_positive,
        "candidate_false_positive": false_positive,
        "candidate_precision": precision,
        "negative_audit_reviewed_count": len(negative_labels),
        "negative_audit_missed_boundary_count": audit_missed,
        "negative_audit_hit_rate": audit_hit_rate,
        "recall": None,
        "recall_eligible": False,
        "recall_blocker": "需要补录并仲裁检测器未提出的全部真实边界",
        "formal_calibration_ready": False,
        "automatic_threshold_install": False,
    }


__all__ = ["ALLOWED_BOUNDARY_TYPES", "ALLOWED_DECISIONS", "evaluate_reviews", "load_review_jsonl"]
