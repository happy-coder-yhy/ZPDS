"""有手视频的帧级分析、区间清洗与可追溯派生产物。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from zpds.core.decisions import Disposition, ReasonCode, Severity

_REQUIRED_HAND_COLUMNS = {
    "output_frame_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "keypoints_2d",
}
_SOFT_REASONS = {
    ReasonCode.BLUR_DETECTED.value,
    ReasonCode.NO_OPERATION.value,
    ReasonCode.HAND_OCCLUDED.value,
    ReasonCode.HAND_TRACK_LOST.value,
}


@dataclass(frozen=True)
class HandVideoCleaningConfig:
    """经过校验的清洗阈值，数值的唯一默认来源是 YAML。"""

    min_video_duration_s: float
    min_kept_span_duration_s: float
    black_gray_mean_max: float
    black_pixel_value_max: int
    black_dark_ratio_min: float
    black_min_duration_s: float
    blur_laplacian_var_max: float
    blur_min_duration_s: float
    no_hand_min_duration_s: float
    no_operation_center_speed_max_per_s: float
    no_operation_flow_magnitude_max_px: float
    no_operation_min_duration_s: float
    occlusion_pose_completeness_max: float
    occlusion_clipped_ratio_min: float
    occlusion_min_duration_s: float
    flow_consistency_scale_px: float
    flow_resize_width: int
    flow_grid_stride: int
    smoothness_normalized_jerk_scale: float
    merge_gap_s: float
    output_codec: str

    @classmethod
    def load(cls, path: str | Path) -> HandVideoCleaningConfig:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"手部视频清洗配置不存在: {config_path}")
        with config_path.open(encoding="utf-8") as file:
            document = yaml.safe_load(file)
        if not isinstance(document, dict) or not isinstance(
            document.get("hand_video_cleaning"), dict
        ):
            raise TypeError("配置顶层必须包含 hand_video_cleaning 对象")
        values = document["hand_video_cleaning"]
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - set(values))
        unknown = sorted(set(values) - expected)
        if missing or unknown:
            raise ValueError(f"清洗配置字段不匹配: missing={missing}, unknown={unknown}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "min_video_duration_s": self.min_video_duration_s,
            "min_kept_span_duration_s": self.min_kept_span_duration_s,
            "black_min_duration_s": self.black_min_duration_s,
            "blur_min_duration_s": self.blur_min_duration_s,
            "no_hand_min_duration_s": self.no_hand_min_duration_s,
            "no_operation_min_duration_s": self.no_operation_min_duration_s,
            "occlusion_min_duration_s": self.occlusion_min_duration_s,
            "flow_consistency_scale_px": self.flow_consistency_scale_px,
            "smoothness_normalized_jerk_scale": self.smoothness_normalized_jerk_scale,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0]
        ratios = {
            "black_dark_ratio_min": self.black_dark_ratio_min,
            "occlusion_pose_completeness_max": self.occlusion_pose_completeness_max,
            "occlusion_clipped_ratio_min": self.occlusion_clipped_ratio_min,
        }
        invalid.extend(name for name, value in ratios.items() if not 0 <= float(value) <= 1)
        if invalid:
            raise ValueError(f"清洗配置值越界: {sorted(invalid)}")
        if not 0 <= self.black_pixel_value_max <= 255:
            raise ValueError("black_pixel_value_max 必须在 [0, 255]")
        if self.flow_resize_width < 32 or self.flow_grid_stride < 1:
            raise ValueError("光流缩放宽度至少为 32，采样步长至少为 1")
        if len(self.output_codec) != 4:
            raise ValueError("output_codec 必须是四字符编码")

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class HandVideoAnalysis:
    """分析阶段的内存结果；不包含视频帧本身。"""

    report: dict[str, Any]
    frame_metrics: pd.DataFrame
    kept_spans: list[tuple[int, int]]


@dataclass(frozen=True)
class HandVideoCleaningResult:
    """清洗产物路径。"""

    report_path: Path
    frame_metrics_path: Path
    sample_map_path: Path
    cleaned_video_path: Path | None
    report: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return version("zpds")
    except PackageNotFoundError:
        return "0.1.0"


def _minimum_frames(duration_s: float, fps: float) -> int:
    return max(1, math.ceil(duration_s * fps))


def _continuous_spans(flags: np.ndarray, minimum: int = 1) -> list[tuple[int, int]]:
    if flags.size == 0:
        return []
    padded = np.pad(flags.astype(np.int8), (1, 1))
    starts = np.flatnonzero((padded[1:] == 1) & (padded[:-1] == 0))
    ends = np.flatnonzero((padded[1:] == 0) & (padded[:-1] == 1))
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= minimum]


def _sustain(flags: np.ndarray, minimum: int) -> np.ndarray:
    result = np.zeros_like(flags, dtype=bool)
    for start, end in _continuous_spans(flags, minimum):
        result[start:end] = True
    return result


def _invert_spans(length: int, excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    cursor = 0
    for start, end in excluded:
        if cursor < start:
            kept.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length:
        kept.append((cursor, length))
    return kept


def _resize_gray(gray: np.ndarray, target_width: int) -> tuple[np.ndarray, float]:
    if gray.shape[1] <= target_width:
        return gray, 1.0
    scale = target_width / gray.shape[1]
    resized = cv2.resize(gray, (target_width, max(1, round(gray.shape[0] * scale))))
    return resized, scale


def _flow_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    scale: float,
    grid_stride: int,
) -> tuple[float, float]:
    forward = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    backward = cv2.calcOpticalFlowFarneback(current, previous, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    height, width = previous.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + forward[..., 0]).astype(np.float32)
    map_y = (grid_y + forward[..., 1]).astype(np.float32)
    backward_warped = cv2.remap(
        backward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    error = np.linalg.norm(forward + backward_warped, axis=2)
    magnitude = np.linalg.norm(forward, axis=2)
    valid = (map_x >= 0) & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1)
    sampled = valid[::grid_stride, ::grid_stride]
    sampled_error = error[::grid_stride, ::grid_stride][sampled]
    fb_error = float(np.median(sampled_error)) if sampled_error.size else float("nan")
    if bbox is None:
        region = magnitude
    else:
        x1, y1, x2, y2 = (round(value * scale) for value in bbox)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        region = magnitude[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else magnitude
    flow_magnitude = float(np.median(region)) / scale if region.size else float("nan")
    return flow_magnitude, fb_error / scale


def _parse_keypoints(value: Any) -> np.ndarray:
    try:
        points = np.stack([np.asarray(point, dtype=np.float64) for point in value])
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)
    return points if points.shape == (21, 2) else np.empty((0, 2), dtype=np.float64)


def _hand_frame_metadata(
    hands: pd.DataFrame, width: int, height: int
) -> tuple[
    set[int],
    dict[int, float],
    dict[str, dict[int, np.ndarray]],
    dict[int, tuple[float, float, float, float]],
]:
    present: set[int] = set()
    completeness: dict[int, float] = {}
    tracks: dict[str, dict[int, np.ndarray]] = {}
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for frame_index, rows in hands.groupby("output_frame_index", sort=False):
        index = int(frame_index)
        if index < 0:
            continue
        candidates: list[
            tuple[float, np.ndarray, tuple[float, float, float, float], float, str]
        ] = []
        for _, row in rows.iterrows():
            points = _parse_keypoints(row["keypoints_2d"])
            if points.shape != (21, 2):
                score = 0.0
                center = np.array([np.nan, np.nan])
            else:
                valid = (
                    np.isfinite(points).all(axis=1)
                    & (points[:, 0] >= 0)
                    & (points[:, 0] < width)
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < height)
                )
                raw_clipped = row.get("keypoints_clipped_count", 0)
                clipped = int(raw_clipped) if pd.notna(raw_clipped) else 0
                score = min(float(valid.mean()), max(0.0, 1.0 - clipped / 21.0))
                palm = points[[0, 5, 9, 13, 17]]
                center = np.nanmedian(palm, axis=0)
            bbox = tuple(float(row[name]) for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
            area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
            handedness = str(row.get("handedness", "unknown")).lower()
            detection_id = int(row.get("detection_id", 0))
            track_key = handedness if handedness in {"left", "right"} else f"unknown_{detection_id}"
            candidates.append((area, center, bbox, score, track_key))
        if candidates:
            best = max(candidates, key=lambda item: item[0])
            present.add(index)
            boxes[index] = best[2]
            completeness[index] = max(item[3] for item in candidates)
            for _, center, _, _, track_key in candidates:
                tracks.setdefault(track_key, {})[index] = center
    return present, completeness, tracks, boxes


def _motion_metrics(
    tracks: dict[str, dict[int, np.ndarray]],
    frame_count: int,
    fps: float,
    diagonal: float,
    normalized_jerk_scale: float,
) -> tuple[np.ndarray, float | None, int, float | None, float | None]:
    maximum_speed = np.full(frame_count, np.nan, dtype=np.float64)
    all_jerk: list[np.ndarray] = []
    all_speed: list[np.ndarray] = []
    for track in tracks.values():
        centers = np.full((frame_count, 2), np.nan, dtype=np.float64)
        for frame_index, center in track.items():
            if 0 <= frame_index < frame_count:
                centers[frame_index] = center
        velocity = np.full((frame_count, 2), np.nan, dtype=np.float64)
        speed = np.full(frame_count, np.nan, dtype=np.float64)
        acceleration = np.full((frame_count, 2), np.nan, dtype=np.float64)
        jerk = np.full(frame_count, np.nan, dtype=np.float64)
        for index in range(1, frame_count):
            if np.isfinite(centers[index - 1 : index + 1]).all():
                velocity[index] = (centers[index] - centers[index - 1]) * fps / diagonal
                speed[index] = np.linalg.norm(velocity[index])
        valid_speed = np.isfinite(speed)
        if valid_speed.any():
            all_speed.append(speed[valid_speed])
        empty_speed = valid_speed & ~np.isfinite(maximum_speed)
        maximum_speed[empty_speed] = speed[empty_speed]
        comparable_speed = valid_speed & np.isfinite(maximum_speed)
        maximum_speed[comparable_speed] = np.maximum(
            maximum_speed[comparable_speed], speed[comparable_speed]
        )
        for index in range(2, frame_count):
            if np.isfinite(velocity[index - 1 : index + 1]).all():
                acceleration[index] = (velocity[index] - velocity[index - 1]) * fps
        for index in range(3, frame_count):
            if np.isfinite(acceleration[index - 1 : index + 1]).all():
                jerk[index] = np.linalg.norm(
                    (acceleration[index] - acceleration[index - 1]) * fps
                )
        valid_jerk = jerk[np.isfinite(jerk)]
        if valid_jerk.size:
            all_jerk.append(valid_jerk)
    if not all_jerk:
        return maximum_speed, None, 0, None, None
    combined_jerk = np.concatenate(all_jerk)
    median_jerk = float(np.median(combined_jerk))
    median_speed = float(np.median(np.concatenate(all_speed))) if all_speed else 0.0
    normalized_jerk = median_jerk / (median_speed * fps**2 + 1e-12)
    return (
        maximum_speed,
        float(math.exp(-normalized_jerk / normalized_jerk_scale)),
        int(combined_jerk.size),
        median_jerk,
        normalized_jerk,
    )


def analyze_hand_video(
    video_path: str | Path,
    hands_parquet_path: str | Path,
    config: HandVideoCleaningConfig,
    frame_status_path: str | Path | None = None,
) -> HandVideoAnalysis:
    """逐帧分析视频，返回坏区间、保留区间和四项核心质量指标。"""
    source = Path(video_path).expanduser().resolve()
    hands_path = Path(hands_parquet_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"视频不存在: {source}")
    if not hands_path.is_file():
        raise FileNotFoundError(f"手部检测结果不存在: {hands_path}")
    hands = pd.read_parquet(hands_path)
    missing = sorted(_REQUIRED_HAND_COLUMNS - set(hands.columns))
    if missing:
        raise ValueError(f"hands_2d.parquet 缺少字段: {missing}")
    frame_status: pd.DataFrame | None = None
    status_path: Path | None = None
    if frame_status_path is not None:
        status_path = Path(frame_status_path).expanduser().resolve()
        if not status_path.is_file():
            raise FileNotFoundError(f"手部全帧状态不存在: {status_path}")
        frame_status = pd.read_parquet(status_path)
        status_missing = {"output_frame_index", "inference_status"} - set(frame_status.columns)
        if status_missing:
            raise ValueError(f"手部全帧状态缺少字段: {sorted(status_missing)}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"无法解码视频: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    advertised_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"视频元数据无效: fps={fps}, size={width}x{height}")

    hand_frames, completeness_by_frame, hand_tracks, boxes_by_frame = (
        _hand_frame_metadata(hands, width, height)
    )
    gray_means: list[float] = []
    dark_ratios: list[float] = []
    laplacian_vars: list[float] = []
    flow_magnitudes: list[float] = []
    flow_errors: list[float] = []
    previous_gray: np.ndarray | None = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized, flow_scale = _resize_gray(gray, config.flow_resize_width)
        gray_means.append(float(gray.mean()))
        dark_ratios.append(float(np.mean(gray <= config.black_pixel_value_max)))
        laplacian_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if previous_gray is None:
            flow_magnitudes.append(float("nan"))
            flow_errors.append(float("nan"))
        else:
            magnitude, error = _flow_metrics(
                previous_gray,
                resized,
                boxes_by_frame.get(frame_index),
                flow_scale,
                config.flow_grid_stride,
            )
            flow_magnitudes.append(magnitude)
            flow_errors.append(error)
        previous_gray = resized
        frame_index += 1
    capture.release()
    frame_count = len(gray_means)
    if frame_count == 0:
        raise ValueError(f"视频没有可解码帧: {source}")

    present = np.fromiter(
        (index in hand_frames for index in range(frame_count)), dtype=bool, count=frame_count
    )
    completeness = np.asarray(
        [completeness_by_frame.get(index, float("nan")) for index in range(frame_count)]
    )
    gray_mean = np.asarray(gray_means)
    dark_ratio = np.asarray(dark_ratios)
    laplacian_var = np.asarray(laplacian_vars)
    flow_magnitude = np.asarray(flow_magnitudes)
    flow_error = np.asarray(flow_errors)
    speed, smoothness, smoothness_samples, median_jerk, normalized_jerk = _motion_metrics(
        hand_tracks,
        frame_count,
        fps,
        math.hypot(width, height),
        config.smoothness_normalized_jerk_scale,
    )
    black = (gray_mean <= config.black_gray_mean_max) & (
        dark_ratio >= config.black_dark_ratio_min
    )
    blur = laplacian_var <= config.blur_laplacian_var_max
    clipped_ratio = 1.0 - completeness
    occluded = present & (
        (completeness <= config.occlusion_pose_completeness_max)
        | (clipped_ratio >= config.occlusion_clipped_ratio_min)
    )
    no_operation = (
        present
        & np.isfinite(speed)
        & np.isfinite(flow_magnitude)
        & (speed <= config.no_operation_center_speed_max_per_s)
        & (flow_magnitude <= config.no_operation_flow_magnitude_max_px)
    )
    track_lost = np.zeros(frame_count, dtype=bool)
    if frame_status is None:
        no_hand = ~present
    else:
        statuses = np.full(frame_count, "missing", dtype=object)
        for _, row in frame_status.iterrows():
            index = int(row["output_frame_index"])
            if 0 <= index < frame_count:
                statuses[index] = str(row["inference_status"])
        no_hand = (statuses == "no_hand") & ~present
        track_lost = np.isin(statuses, ["failed", "skipped_invalid_input", "missing"])
        track_lost |= (statuses == "detected") & ~present
    reason_masks = {
        ReasonCode.BLACK_FRAME.value: _sustain(
            black, _minimum_frames(config.black_min_duration_s, fps)
        ),
        ReasonCode.BLUR_DETECTED.value: _sustain(
            blur & ~black, _minimum_frames(config.blur_min_duration_s, fps)
        ),
        ReasonCode.HAND_ABSENT.value: _sustain(
            no_hand, _minimum_frames(config.no_hand_min_duration_s, fps)
        ),
        ReasonCode.HAND_TRACK_LOST.value: _sustain(
            track_lost, _minimum_frames(config.no_hand_min_duration_s, fps)
        ),
        ReasonCode.NO_OPERATION.value: _sustain(
            no_operation, _minimum_frames(config.no_operation_min_duration_s, fps)
        ),
        ReasonCode.HAND_OCCLUDED.value: _sustain(
            occluded, _minimum_frames(config.occlusion_min_duration_s, fps)
        ),
    }
    too_short = frame_count / fps < config.min_video_duration_s
    if too_short:
        reason_masks[ReasonCode.VIDEO_TOO_SHORT.value] = np.ones(frame_count, dtype=bool)

    excluded_mask = np.logical_or.reduce(list(reason_masks.values()))
    merge_gap = round(config.merge_gap_s * fps)
    initial = _continuous_spans(excluded_mask)
    merged: list[tuple[int, int]] = []
    for start, end in initial:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    kept = _invert_spans(frame_count, merged)
    short_kept = [
        span for span in kept if span[1] - span[0] < _minimum_frames(
            config.min_kept_span_duration_s, fps
        )
    ]
    if short_kept:
        merged = sorted(merged + short_kept)
        collapsed: list[tuple[int, int]] = []
        for start, end in merged:
            if collapsed and start <= collapsed[-1][1]:
                collapsed[-1] = (collapsed[-1][0], max(end, collapsed[-1][1]))
            else:
                collapsed.append((start, end))
        merged = collapsed
        kept = _invert_spans(frame_count, merged)

    excluded_spans: list[dict[str, Any]] = []
    for start, end in merged:
        reasons = sorted(
            reason for reason, mask in reason_masks.items() if bool(mask[start:end].any())
        )
        if not reasons:
            reasons = [ReasonCode.VIDEO_TOO_SHORT.value]
        if start == 0 and end == frame_count:
            disposition = Disposition.REJECT
        elif set(reasons).issubset(_SOFT_REASONS):
            disposition = Disposition.QUARANTINE
        elif start == 0 or end == frame_count:
            disposition = Disposition.TRIM
        else:
            disposition = Disposition.SPLIT
        excluded_spans.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_timestamp_ns": int(start / fps * 1_000_000_000),
                "end_timestamp_ns": int(end / fps * 1_000_000_000),
                "duration_s": (end - start) / fps,
                "reasons": reasons,
                "severity": Severity.WARN.value,
                "disposition": disposition.value,
                "included_in_cleaned_video": False,
            }
        )
    kept_span_documents = [
        {
            "start_frame": start,
            "end_frame": end,
            "start_timestamp_ns": int(start / fps * 1_000_000_000),
            "end_timestamp_ns": int(end / fps * 1_000_000_000),
            "duration_s": (end - start) / fps,
            "disposition": Disposition.KEEP.value,
        }
        for start, end in kept
    ]
    final_excluded = np.zeros(frame_count, dtype=bool)
    for start, end in merged:
        final_excluded[start:end] = True

    flow_valid = flow_error[np.isfinite(flow_error)]
    flow_consistency = (
        float(np.mean(1.0 / (1.0 + flow_valid / config.flow_consistency_scale_px)))
        if flow_valid.size
        else None
    )
    pose_valid = completeness[np.isfinite(completeness)]
    frame_metrics = pd.DataFrame(
        {
            "source_frame_index": np.arange(frame_count, dtype=np.int64),
            "timestamp_ns": (np.arange(frame_count) / fps * 1_000_000_000).astype(np.int64),
            "gray_mean": gray_mean,
            "dark_pixel_ratio": dark_ratio,
            "laplacian_variance": laplacian_var,
            "hand_present": present,
            "hand_pose_completeness": completeness,
            "hand_center_speed_normalized_per_s": speed,
            "flow_magnitude_px": flow_magnitude,
            "flow_forward_backward_error_px": flow_error,
            "is_black": reason_masks[ReasonCode.BLACK_FRAME.value],
            "is_blurry": reason_masks[ReasonCode.BLUR_DETECTED.value],
            "is_no_hand": reason_masks[ReasonCode.HAND_ABSENT.value],
            "is_hand_track_lost": reason_masks[ReasonCode.HAND_TRACK_LOST.value],
            "is_no_operation": reason_masks[ReasonCode.NO_OPERATION.value],
            "is_strongly_occluded": reason_masks[ReasonCode.HAND_OCCLUDED.value],
            "is_excluded": final_excluded,
        }
    )
    report = {
        "schema_version": "zpds.hand_video_cleaning.v1",
        "producer": {"name": "zpds.hands.cleaning", "version": _package_version()},
        "provenance": {
            "source_video_uri": str(video_path),
            "source_video_sha256": _sha256_file(source),
            "hands_parquet_uri": str(hands_parquet_path),
            "hands_parquet_sha256": _sha256_file(hands_path),
            "frame_status_uri": str(frame_status_path) if frame_status_path else None,
            "frame_status_sha256": _sha256_file(status_path) if status_path else None,
            "config_sha256": config.sha256(),
        },
        "source": {
            "fps": fps,
            "width": width,
            "height": height,
            "advertised_frame_count": advertised_count,
            "decoded_frame_count": frame_count,
            "duration_s": frame_count / fps,
        },
        "quality_metrics": {
            "hand_presence_ratio": float(present.mean()),
            "motion_smoothness": smoothness,
            "motion_smoothness_sample_count": smoothness_samples,
            "motion_median_jerk_normalized_per_s3": median_jerk,
            "motion_normalized_jerk": normalized_jerk,
            "optical_flow_consistency": flow_consistency,
            "optical_flow_pair_count": int(flow_valid.size),
            "hand_pose_completeness": float(pose_valid.mean()) if pose_valid.size else None,
            "hand_pose_sample_count": int(pose_valid.size),
        },
        "thresholds": asdict(config),
        "summary": {
            "input_frames": frame_count,
            "kept_frames": int(sum(end - start for start, end in kept)),
            "excluded_frames": int(sum(end - start for start, end in merged)),
            "input_duration_s": frame_count / fps,
            "kept_duration_s": sum(end - start for start, end in kept) / fps,
            "overall_disposition": (
                Disposition.REJECT.value
                if not kept
                else Disposition.SPLIT.value
                if merged
                else Disposition.KEEP.value
            ),
        },
        "excluded_spans": excluded_spans,
        "kept_spans": kept_span_documents,
    }
    return HandVideoAnalysis(report=report, frame_metrics=frame_metrics, kept_spans=kept)


def _write_cleaned_video(
    source: Path,
    destination: Path,
    kept_spans: list[tuple[int, int]],
    fps: float,
    size: tuple[int, int],
    codec: str,
) -> pd.DataFrame:
    columns = [
        "cleaned_frame_index",
        "source_output_frame_index",
        "source_timestamp_ns",
        "mapping_method",
    ]
    if not kept_spans:
        return pd.DataFrame(columns=columns)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"无法创建派生视频: {destination}")
    rows: list[dict[str, Any]] = []
    source_index = 0
    cleaned_index = 0
    span_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            while span_index < len(kept_spans) and source_index >= kept_spans[span_index][1]:
                span_index += 1
            is_kept = (
                span_index < len(kept_spans)
                and kept_spans[span_index][0] <= source_index < kept_spans[span_index][1]
            )
            if is_kept:
                writer.write(frame)
                rows.append(
                    {
                        "cleaned_frame_index": cleaned_index,
                        "source_output_frame_index": source_index,
                        "source_timestamp_ns": int(source_index / fps * 1_000_000_000),
                        "mapping_method": "decoded_frame_identity",
                    }
                )
                cleaned_index += 1
            source_index += 1
    finally:
        capture.release()
        writer.release()
    expected_frames = sum(end - start for start, end in kept_spans)
    if cleaned_index != expected_frames:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"派生视频帧数不完整: expected={expected_frames}, actual={cleaned_index}"
        )
    temporary.replace(destination)
    return pd.DataFrame(rows, columns=columns)


def clean_hand_video(
    video_path: str | Path,
    hands_parquet_path: str | Path,
    output_dir: str | Path,
    config: HandVideoCleaningConfig,
    frame_status_path: str | Path | None = None,
) -> HandVideoCleaningResult:
    """分析并写出清洗视频、帧指标、sample map 和报告。"""
    source = Path(video_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    analysis = analyze_hand_video(source, hands_parquet_path, config, frame_status_path)
    frame_metrics_path = destination / "hand_cleaning_frames.parquet"
    sample_map_path = destination / "cleaned_sample_map.parquet"
    report_path = destination / "hand_cleaning_report.json"
    cleaned_path = destination / "cleaned.mp4" if analysis.kept_spans else None
    analysis.frame_metrics.to_parquet(frame_metrics_path, index=False)
    source_meta = analysis.report["source"]
    if cleaned_path is None:
        (destination / "cleaned.mp4").unlink(missing_ok=True)
        sample_map = pd.DataFrame(
            columns=[
                "cleaned_frame_index",
                "source_output_frame_index",
                "source_timestamp_ns",
                "mapping_method",
            ]
        )
    else:
        sample_map = _write_cleaned_video(
            source,
            cleaned_path,
            analysis.kept_spans,
            float(source_meta["fps"]),
            (int(source_meta["width"]), int(source_meta["height"])),
            config.output_codec,
        )
    sample_map.to_parquet(sample_map_path, index=False)
    report = dict(analysis.report)
    report["artifacts"] = {
        "cleaned_video_uri": cleaned_path.name if cleaned_path else None,
        "frame_metrics_uri": frame_metrics_path.name,
        "sample_map_uri": sample_map_path.name,
    }
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    return HandVideoCleaningResult(
        report_path=report_path,
        frame_metrics_path=frame_metrics_path,
        sample_map_path=sample_map_path,
        cleaned_video_path=cleaned_path,
        report=report,
    )


__all__ = [
    "HandVideoAnalysis",
    "HandVideoCleaningConfig",
    "HandVideoCleaningResult",
    "analyze_hand_video",
    "clean_hand_video",
]
