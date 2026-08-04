"""仅依赖 OpenCV 的 SSIM 硬切和渐变检测。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import cv2
import numpy as np

from zpds.scene.backends.common import (
    finite_unit,
    select_peak_indices,
    timestamp_ns,
    to_gray,
    validate_frames,
)
from zpds.scene.config import SSIMConfig
from zpds.scene.schemas import DetectorFrameScores, TransitionProposal


class SSIMTransitionDetector:
    source = "ssim"

    def __init__(self, config: SSIMConfig) -> None:
        self.config = config

    def similarity(self, first: np.ndarray, second: np.ndarray) -> float:
        """计算两个等尺寸图像的平均结构相似度。"""

        first_gray = to_gray(first).astype(np.float32)
        second_gray = to_gray(second).astype(np.float32)
        if first_gray.shape != second_gray.shape:
            raise ValueError("SSIM 输入图像必须具有相同尺寸")
        kernel = (self.config.gaussian_window_size,) * 2
        sigma = self.config.gaussian_sigma
        mu_first = cv2.GaussianBlur(first_gray, kernel, sigma)
        mu_second = cv2.GaussianBlur(second_gray, kernel, sigma)
        mu_first_sq = mu_first * mu_first
        mu_second_sq = mu_second * mu_second
        mu_product = mu_first * mu_second
        sigma_first_sq = cv2.GaussianBlur(first_gray * first_gray, kernel, sigma) - mu_first_sq
        sigma_second_sq = cv2.GaussianBlur(second_gray * second_gray, kernel, sigma) - mu_second_sq
        sigma_product = cv2.GaussianBlur(first_gray * second_gray, kernel, sigma) - mu_product
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        numerator = (2.0 * mu_product + c1) * (2.0 * sigma_product + c2)
        denominator = (mu_first_sq + mu_second_sq + c1) * (
            sigma_first_sq + sigma_second_sq + c2
        )
        map_ssim = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=np.abs(denominator) > np.finfo(np.float32).eps,
        )
        return finite_unit(float(np.mean(map_ssim)))

    def score_frames(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
    ) -> DetectorFrameScores:
        validate_frames(frames, fps=fps)
        if not frames:
            return DetectorFrameScores(self.source, ())
        similarities = [1.0]
        for previous, current in pairwise(frames):
            similarities.append(self.similarity(previous, current))
        scores = tuple(finite_unit(1.0 - value) for value in similarities)
        return DetectorFrameScores(
            self.source,
            scores,
            diagnostics={"similarity": tuple(similarities)},
        )

    def detect(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
        start_timestamp_ns: int = 0,
    ) -> list[TransitionProposal]:
        frame_scores = self.score_frames(frames, fps=fps)
        similarities = frame_scores.diagnostics.get("similarity", ())
        hard_indices = [
            index
            for index, similarity in enumerate(similarities)
            if index > 0 and similarity <= self.config.hard_cut_similarity
        ]

        gradual_indices: list[int] = []
        gradual_scores = list(frame_scores.scores)
        window = self.config.gradual_window_frames
        for index in range(window, len(frames)):
            lag_similarity = self.similarity(frames[index - window], frames[index])
            local_similarities = similarities[index - window + 1 : index + 1]
            changed_steps = sum(similarity < 0.999 for similarity in local_similarities)
            if (
                lag_similarity <= self.config.gradual_similarity
                and changed_steps >= self.config.gradual_min_frames
                and not any(
                    similarity <= self.config.hard_cut_similarity
                    for similarity in local_similarities
                )
            ):
                midpoint = index - window // 2
                gradual_indices.append(midpoint)
                gradual_scores[midpoint] = max(
                    gradual_scores[midpoint],
                    finite_unit(1.0 - lag_similarity),
                )

        candidates = select_peak_indices(
            hard_indices + gradual_indices,
            gradual_scores,
            max_gap=max(1, self.config.gradual_min_frames),
        )
        return [
            TransitionProposal(
                frame_index=index,
                timestamp_ns=timestamp_ns(
                    index,
                    fps=fps,
                    start_timestamp_ns=start_timestamp_ns,
                ),
                score=gradual_scores[index],
                is_hard_cut=False,
                sources=(self.source,),
            )
            for index in candidates
        ]


__all__ = ["SSIMTransitionDetector"]
