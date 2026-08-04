"""Farneback 稠密光流、RANSAC 全局运动补偿与冻结检测。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from zpds.scene.config import OpticalFlowConfig
from zpds.scene.schemas import DetectorFrameScores, TransitionProposal


@dataclass(frozen=True)
class FlowPairMetrics:
    raw_motion_px: float
    residual_motion_px: float
    inlier_ratio: float


class OpticalFlowTransitionDetector:
    source = "optical_flow"

    def __init__(
        self,
        config: OpticalFlowConfig,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback

    def _analysis_gray(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = to_gray(frame)
        height, width = gray.shape
        longest = max(height, width)
        if longest <= self.config.analysis_max_dimension:
            return gray, 1.0
        scale = self.config.analysis_max_dimension / longest
        resized = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _pair_metrics(self, previous: np.ndarray, current: np.ndarray) -> FlowPairMetrics:
        previous_gray, scale = self._analysis_gray(previous)
        current_gray, current_scale = self._analysis_gray(current)
        if current_gray.shape != previous_gray.shape or current_scale != scale:
            raise ValueError("光流输入图像必须具有相同宽高")
        flow_buffer = np.zeros((*previous_gray.shape, 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            flow_buffer,
            self.config.pyr_scale,
            self.config.levels,
            self.config.window_size,
            self.config.iterations,
            self.config.poly_n,
            self.config.poly_sigma,
            0,
        )
        height, width = previous_gray.shape
        grid_step = max(1, round(self.config.grid_step * scale))
        offset = max(1, grid_step // 2)
        y_values = np.arange(offset, height, grid_step)
        x_values = np.arange(offset, width, grid_step)
        if not len(x_values) or not len(y_values):
            return FlowPairMetrics(0.0, 0.0, 0.0)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        source = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
        sampled_flow = flow[grid_y.ravel(), grid_x.ravel()].reshape(-1, 2)
        finite = np.all(np.isfinite(sampled_flow), axis=1)
        source = source[finite]
        sampled_flow = sampled_flow[finite]
        if not len(source):
            return FlowPairMetrics(0.0, 0.0, 0.0)
        raw_motion = float(np.median(np.linalg.norm(sampled_flow, axis=1)))
        destination = source + sampled_flow

        predicted_flow = np.zeros_like(sampled_flow)
        inlier_ratio = 0.0
        if len(source) >= self.config.min_correspondences:
            matrix, inliers = cv2.estimateAffinePartial2D(
                source,
                destination,
                method=cv2.RANSAC,
                ransacReprojThreshold=max(
                    0.5,
                    self.config.ransac_reproj_threshold * scale,
                ),
            )
            if matrix is not None and np.all(np.isfinite(matrix)):
                homogeneous = np.column_stack((source, np.ones(len(source), dtype=np.float32)))
                predicted_destination = homogeneous @ matrix.T
                predicted_flow = predicted_destination - source
                if inliers is not None:
                    inlier_ratio = float(np.mean(inliers.ravel() > 0))
            else:
                predicted_flow[:] = np.median(sampled_flow, axis=0)
        else:
            predicted_flow[:] = np.median(sampled_flow, axis=0)

        residual = np.linalg.norm(sampled_flow - predicted_flow, axis=1)
        residual_motion = float(np.percentile(residual, 90)) if len(residual) else 0.0
        # 配置阈值以原图像素为单位，缩放计算后换算回来。
        return FlowPairMetrics(
            raw_motion / scale,
            residual_motion / scale,
            inlier_ratio,
        )

    def _all_metrics(self, frames: Sequence[np.ndarray]) -> list[FlowPairMetrics]:
        metrics = [FlowPairMetrics(0.0, 0.0, 1.0)]
        total = max(0, len(frames) - 1)
        for completed, (previous, current) in enumerate(pairwise(frames), start=1):
            metrics.append(self._pair_metrics(previous, current))
            if self.progress_callback is not None and (
                completed == total or completed % 10 == 0
            ):
                self.progress_callback(completed, total)
        return metrics

    def score_frames(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
    ) -> DetectorFrameScores:
        validate_frames(frames, fps=fps)
        if not frames:
            return DetectorFrameScores(self.source, ())
        metrics = self._all_metrics(frames)
        scores = tuple(
            finite_unit(metric.residual_motion_px / self.config.residual_hard_scale_px)
            for metric in metrics
        )
        return DetectorFrameScores(
            self.source,
            scores,
            diagnostics={
                "raw_motion_px": tuple(metric.raw_motion_px for metric in metrics),
                "residual_motion_px": tuple(
                    metric.residual_motion_px for metric in metrics
                ),
                "ransac_inlier_ratio": tuple(metric.inlier_ratio for metric in metrics),
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
        residuals = frame_scores.diagnostics.get("residual_motion_px", ())
        raw_motion = frame_scores.diagnostics.get("raw_motion_px", ())
        transition_indices = [
            index
            for index, residual in enumerate(residuals)
            if index > 0 and residual >= self.config.residual_threshold_px
        ]
        candidate_scores = list(frame_scores.scores)
        candidates = select_peak_indices(transition_indices, candidate_scores)

        freeze_runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for index in range(1, len(raw_motion)):
            is_frozen = raw_motion[index] <= self.config.freeze_motion_threshold_px
            if is_frozen and run_start is None:
                run_start = index
            if not is_frozen and run_start is not None:
                freeze_runs.append((run_start, index - 1))
                run_start = None
        if run_start is not None:
            freeze_runs.append((run_start, len(raw_motion) - 1))
        for start, end in freeze_runs:
            if end - start + 1 >= self.config.freeze_min_frames:
                candidates.append(start)
                candidate_scores[start] = 1.0

        candidates = sorted(set(candidates))
        return [
            TransitionProposal(
                frame_index=index,
                timestamp_ns=timestamp_ns(
                    index,
                    fps=fps,
                    start_timestamp_ns=start_timestamp_ns,
                ),
                score=candidate_scores[index],
                is_hard_cut=False,
                sources=(self.source,),
            )
            for index in candidates
        ]


__all__ = ["FlowPairMetrics", "OpticalFlowTransitionDetector"]
