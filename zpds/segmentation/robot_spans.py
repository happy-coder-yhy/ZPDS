"""Physical span and conservative edge-idle proposal for robot observations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from zpds.core.decisions import Decision, Disposition
from zpds.core.types import SpanProposal


@dataclass(frozen=True)
class IdleCandidate:
    """A review-only edge trim proposal; middle low-motion periods are preserved."""

    start_ns: int
    end_ns: int
    edge: str
    signals_agreeing: tuple[str, ...]
    disposition: Disposition = Disposition.TRIM
    requires_review: bool = True


def propose_physical_spans(
    stream_ranges: dict[str, tuple[int, int]],
    decisions: Iterable[Decision] = (),
) -> list[SpanProposal]:
    """Intersect trusted streams and split only at explicitly bad time ranges."""
    if not stream_ranges:
        return []
    starts, ends = zip(*stream_ranges.values())
    start_ns, end_ns = max(starts), min(ends)
    if end_ns <= start_ns:
        return []

    blocked = sorted(
        (
            max(start_ns, decision.timestamp_ns),
            min(end_ns, decision.end_timestamp_ns),
        )
        for decision in decisions
        if decision.disposition in {Disposition.SPLIT, Disposition.REJECT}
        and decision.timestamp_ns is not None
        and decision.end_timestamp_ns is not None
        and decision.end_timestamp_ns > start_ns
        and decision.timestamp_ns < end_ns
    )
    spans: list[SpanProposal] = []
    cursor = start_ns
    for bad_start, bad_end in blocked:
        if bad_start > cursor:
            spans.append(SpanProposal(cursor, bad_start, reason="trusted_common_range"))
        cursor = max(cursor, bad_end)
    if cursor < end_ns:
        spans.append(SpanProposal(cursor, end_ns, reason="trusted_common_range"))
    return spans


def propose_edge_idle(
    timestamps_ns: Sequence[int],
    robot_motion_energy: Sequence[float] | None,
    gripper_event_energy: Sequence[float] | None,
    visual_change_energy: Sequence[float] | None,
    *,
    motion_max: float,
    gripper_max: float,
    visual_change_max: float,
    min_samples: int = 1,
) -> list[IdleCandidate]:
    """Propose trim candidates only where all three independent signals are idle."""
    arrays = (robot_motion_energy, gripper_event_energy, visual_change_energy)
    if not timestamps_ns or any(values is None for values in arrays):
        return []
    count = len(timestamps_ns)
    if any(len(values) != count for values in arrays if values is not None):
        raise ValueError("timestamps and idle signals must have identical lengths")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    robot, gripper, visual = (np.asarray(values, dtype=float) for values in arrays)
    idle = (
        np.isfinite(robot)
        & np.isfinite(gripper)
        & np.isfinite(visual)
        & (robot <= motion_max)
        & (gripper <= gripper_max)
        & (visual <= visual_change_max)
    )
    candidates: list[IdleCandidate] = []
    head = int(np.argmax(~idle)) if np.any(~idle) else count
    if head >= min_samples:
        candidates.append(
            IdleCandidate(
                int(timestamps_ns[0]),
                int(timestamps_ns[head - 1]),
                "head",
                ("robot_motion_energy", "gripper_event_energy", "visual_change_energy"),
            )
        )
    tail = int(np.argmax(~idle[::-1])) if np.any(~idle) else count
    if tail >= min_samples:
        start = count - tail
        candidates.append(
            IdleCandidate(
                int(timestamps_ns[start]),
                int(timestamps_ns[-1]),
                "tail",
                ("robot_motion_energy", "gripper_event_energy", "visual_change_energy"),
            )
        )
    return candidates


__all__ = ["IdleCandidate", "propose_edge_idle", "propose_physical_spans"]
