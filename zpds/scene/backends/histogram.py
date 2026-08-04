"""HSV 直方图转场检测。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import cv2
import numpy as np

from zpds.scene.backends.common import (
    finite_unit,
    select_peak_indices,
    timestamp_ns,
    to_bgr,
    validate_frames,
)
from zpds.scene.config import HistogramConfig
from zpds.scene.schemas import DetectorFrameScores, TransitionProposal


class HistogramTransitionDetector:
    source = "histogram"

    def __init__(self, config: HistogramConfig) -> None:
        self.config = config

    def _histogram(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(to_bgr(frame), cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [self.config.h_bins, self.config.s_bins],
            [0, 180, 0, 256],
        )
        return cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)

    def score_frames(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
    ) -> DetectorFrameScores:
        validate_frames(frames, fps=fps)
        if not frames:
            return DetectorFrameScores(self.source, ())
        histograms = [self._histogram(frame) for frame in frames]
        scores = [0.0]
        method = (
            cv2.HISTCMP_BHATTACHARYYA
            if self.config.method == "bhattacharyya"
            else cv2.HISTCMP_CHISQR
        )
        for previous, current in pairwise(histograms):
            raw = float(cv2.compareHist(previous, current, method))
            if self.config.method == "chi_square":
                raw = raw / (raw + 1.0) if raw >= 0 else 0.0
            scores.append(finite_unit(raw))
        return DetectorFrameScores(self.source, tuple(scores))

    def detect(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
        start_timestamp_ns: int = 0,
        frame_scores: DetectorFrameScores | None = None,
    ) -> list[TransitionProposal]:
        if frame_scores is None:
            frame_scores = self.score_frames(frames, fps=fps)
        indices = [
            index
            for index, score in enumerate(frame_scores.scores)
            if index > 0 and score >= self.config.threshold
        ]
        peaks = select_peak_indices(indices, frame_scores.scores)
        return [
            TransitionProposal(
                frame_index=index,
                timestamp_ns=timestamp_ns(
                    index,
                    fps=fps,
                    start_timestamp_ns=start_timestamp_ns,
                ),
                score=frame_scores.scores[index],
                is_hard_cut=False,
                sources=(self.source,),
            )
            for index in peaks
        ]


__all__ = ["HistogramTransitionDetector"]
