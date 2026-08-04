"""亮度突变与黑帧边界检测。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from zpds.scene.backends.common import timestamp_ns, to_gray, validate_frames
from zpds.scene.config import BrightnessConfig
from zpds.scene.schemas import DetectorFrameScores, TransitionProposal


class BrightnessTransitionDetector:
    source = "brightness"

    def __init__(self, config: BrightnessConfig) -> None:
        self.config = config

    def score_frames(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
    ) -> DetectorFrameScores:
        validate_frames(frames, fps=fps)
        if not frames:
            return DetectorFrameScores(self.source, ())
        gray_frames = [to_gray(frame) for frame in frames]
        means = [float(np.mean(frame)) / 255.0 for frame in gray_frames]
        black_ratios = [
            float(np.mean(frame <= self.config.black_pixel_value))
            for frame in gray_frames
        ]
        is_black = [ratio >= self.config.black_ratio_threshold for ratio in black_ratios]
        scores = [0.0]
        for index in range(1, len(frames)):
            jump = abs(means[index] - means[index - 1])
            black_transition = abs(black_ratios[index] - black_ratios[index - 1])
            if is_black[index] != is_black[index - 1]:
                black_transition = max(black_transition, 1.0)
            scores.append(float(np.clip(max(jump, black_transition), 0.0, 1.0)))
        return DetectorFrameScores(
            self.source,
            tuple(scores),
            diagnostics={
                "mean_luma": tuple(means),
                "black_ratio": tuple(black_ratios),
            },
        )

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
        means = frame_scores.diagnostics.get("mean_luma", ())
        black_ratios = frame_scores.diagnostics.get("black_ratio", ())
        proposals: list[TransitionProposal] = []
        for index in range(1, len(frame_scores.scores)):
            jump = abs(means[index] - means[index - 1])
            black_changed = (
                black_ratios[index] >= self.config.black_ratio_threshold
            ) != (
                black_ratios[index - 1] >= self.config.black_ratio_threshold
            )
            if jump < self.config.mean_jump_threshold and not black_changed:
                continue
            proposals.append(
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
            )
        return proposals


__all__ = ["BrightnessTransitionDetector"]
