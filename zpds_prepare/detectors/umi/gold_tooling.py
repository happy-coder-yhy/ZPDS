"""UMI gold-set sampling and review utilities; no labels are fabricated here."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UmiEpisodeCandidate:
    episode_id: str
    task: str
    camera_group: str
    duration_s: float
    outcome: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class UmiLabeledSpan:
    label: str
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if self.end_ns < self.start_ns:
            raise ValueError("labeled span end_ns must not precede start_ns")


@dataclass(frozen=True)
class UmiGoldAnnotation:
    episode_id: str
    reviewer_id: str
    dual_sync_usable: bool | None
    vio_usable: bool | None
    spans: tuple[UmiLabeledSpan, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spans"] = [asdict(span) for span in self.spans]
        return payload


def stratified_sample_episodes(
    episodes: Sequence[UmiEpisodeCandidate],
    *,
    target_count: int = 30,
) -> list[UmiEpisodeCandidate]:
    """Deterministic round-robin sample across task/camera/duration/outcome."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(episodes) <= target_count:
        return sorted(episodes, key=lambda episode: episode.episode_id)

    def duration_bucket(duration_s: float) -> str:
        if duration_s < 10:
            return "short"
        if duration_s < 30:
            return "medium"
        return "long"

    strata: dict[tuple[str, str, str, str], list[UmiEpisodeCandidate]] = (
        defaultdict(list)
    )
    for episode in sorted(episodes, key=lambda item: item.episode_id):
        key = (
            episode.task,
            episode.camera_group,
            duration_bucket(episode.duration_s),
            episode.outcome,
        )
        strata[key].append(episode)

    selected: list[UmiEpisodeCandidate] = []
    ordered_keys = sorted(strata)
    offset = 0
    while len(selected) < target_count:
        added = False
        for key in ordered_keys:
            items = strata[key]
            if offset < len(items):
                selected.append(items[offset])
                added = True
                if len(selected) == target_count:
                    break
        if not added:
            break
        offset += 1
    return selected


def compare_independent_reviews(
    annotations: Sequence[UmiGoldAnnotation],
) -> dict[str, Any]:
    """Compare reviewers and emit disagreements for explicit adjudication."""
    by_episode: dict[str, list[UmiGoldAnnotation]] = defaultdict(list)
    for annotation in annotations:
        by_episode[annotation.episode_id].append(annotation)

    agreements = 0
    disagreements: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for episode_id, episode_annotations in sorted(by_episode.items()):
        reviewer_ids = {item.reviewer_id for item in episode_annotations}
        if len(episode_annotations) != 2 or len(reviewer_ids) != 2:
            incomplete.append(episode_id)
            continue
        left, right = sorted(
            episode_annotations,
            key=lambda item: item.reviewer_id,
        )
        fields = {
            "dual_sync_usable": (
                left.dual_sync_usable,
                right.dual_sync_usable,
            ),
            "vio_usable": (left.vio_usable, right.vio_usable),
            "spans": (
                [asdict(span) for span in left.spans],
                [asdict(span) for span in right.spans],
            ),
        }
        differing = {
            name: values for name, values in fields.items() if values[0] != values[1]
        }
        if differing:
            disagreements.append(
                {
                    "episode_id": episode_id,
                    "reviewers": [left.reviewer_id, right.reviewer_id],
                    "differences": differing,
                    "adjudication_required": True,
                }
            )
        else:
            agreements += 1

    completed = agreements + len(disagreements)
    return {
        "episode_count": len(by_episode),
        "completed_pair_count": completed,
        "agreement_count": agreements,
        "agreement_rate": round(agreements / completed, 6) if completed else None,
        "disagreements": disagreements,
        "incomplete_episode_ids": incomplete,
    }


def binary_classification_metrics(
    predictions: Sequence[bool],
    labels: Sequence[bool],
) -> dict[str, float | int]:
    if len(predictions) != len(labels) or not predictions:
        raise ValueError("predictions and labels must be non-empty and equal length")
    predicted = np.asarray(predictions, dtype=bool)
    gold = np.asarray(labels, dtype=bool)
    true_positive = int(np.sum(predicted & gold))
    false_positive = int(np.sum(predicted & ~gold))
    false_negative = int(np.sum(~predicted & gold))
    true_negative = int(np.sum(~predicted & ~gold))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def evaluate_threshold_candidates(
    values: Sequence[float],
    usable_labels: Sequence[bool],
    *,
    usable_when: str = "less_equal",
) -> dict[str, Any]:
    """Evaluate threshold candidates but never install them automatically."""
    if len(values) != len(usable_labels) or not values:
        raise ValueError("values and labels must be non-empty and equal length")
    if usable_when not in {"less_equal", "greater_equal"}:
        raise ValueError("usable_when must be less_equal or greater_equal")
    if len(set(usable_labels)) < 2:
        raise ValueError("threshold evaluation requires both usable label classes")

    candidates: list[dict[str, Any]] = []
    for threshold in sorted({float(value) for value in values}):
        if usable_when == "less_equal":
            predictions = [float(value) <= threshold for value in values]
        else:
            predictions = [float(value) >= threshold for value in values]
        metrics = binary_classification_metrics(predictions, usable_labels)
        candidates.append({"threshold": threshold, **metrics})
    recommended = max(
        candidates,
        key=lambda row: (row["f1"], row["precision"], row["recall"]),
    )
    return {
        "usable_when": usable_when,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "recommended_candidate": recommended,
        "formal_threshold": False,
        "automatic_install": False,
        "requires_adjudicated_gold": True,
    }


__all__ = [
    "UmiEpisodeCandidate",
    "UmiGoldAnnotation",
    "UmiLabeledSpan",
    "binary_classification_metrics",
    "compare_independent_reviews",
    "evaluate_threshold_candidates",
    "stratified_sample_episodes",
]
