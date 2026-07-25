"""
流缺口检测。

检测 VideoStream (camera) 和 TimeSeriesStream (robot) 中的时间缺口，
生成 split（长缺口）或 keep_with_flag（短缺口）决策。
"""

from __future__ import annotations

import numpy as np

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.readers.session_model import TimeSeriesStream, VideoStream

# 默认阈值（纳秒）
DEFAULT_TIMESERIES_SHORT_GAP_NS = 500_000_000   # 0.5s
DEFAULT_TIMESERIES_LONG_GAP_NS = 2_000_000_000   # 2.0s
DEFAULT_CAMERA_SHORT_ABSENCE_NS = 1_000_000_000  # 1.0s
DEFAULT_CAMERA_LONG_ABSENCE_NS = 5_000_000_000   # 5.0s


def detect_timeseries_gaps(
    ts_stream: TimeSeriesStream,
    config: dict | None = None,
) -> list[QualityIssue]:
    """检测 TimeSeriesStream 中的时间缺口。

    短缺口 → keep_with_flag（标记但不切分）
    长缺口 → split（切分依据）

    Args:
        ts_stream: TimeSeriesStream 对象。
        config: 配置字典，支持:
            - short_gap_ns: 短缺口阈值 (默认 500ms)
            - long_gap_ns: 长缺口阈值 (默认 2s)

    Returns:
        QualityIssue 列表。
    """
    if config is None:
        config = {}

    short_gap_ns = int(config.get("short_gap_ns", DEFAULT_TIMESERIES_SHORT_GAP_NS))
    long_gap_ns = int(config.get("long_gap_ns", DEFAULT_TIMESERIES_LONG_GAP_NS))

    issues: list[QualityIssue] = []
    stream_id = ts_stream.stream_id
    timestamps = np.array(ts_stream.timestamps_ns, dtype=np.int64)

    if len(timestamps) < 2:
        return issues

    diffs = np.diff(timestamps)
    expected_interval_ns = int(1e9 / ts_stream.expected_rate_hz) if ts_stream.expected_rate_hz else None

    # 找所有缺口（间隔 > 0，排除时间回拨）
    gap_mask = diffs > 0
    gap_indices = np.where(gap_mask)[0]

    for idx in gap_indices:
        gap_ns = int(diffs[idx])
        start_ns = int(timestamps[idx])
        end_ns = int(timestamps[idx + 1])

        if gap_ns >= long_gap_ns:
            # 长缺口 → 切分
            multiplier = gap_ns / expected_interval_ns if expected_interval_ns else 0.0
            issues.append(QualityIssue(
                issue_type="timeseries_long_gap",
                stream_id=stream_id,
                start_ns=start_ns,
                end_ns=end_ns,
                severity="error",
                decision="split",
                details={
                    "gap_ns": gap_ns,
                    "gap_s": round(gap_ns / 1e9, 3),
                    "expected_interval_ns": expected_interval_ns,
                    "multiplier": round(multiplier, 1) if multiplier else None,
                    "threshold_long_ns": long_gap_ns,
                    "sample_index": int(idx),
                    "check": "timeseries_gap",
                },
            ))
        elif gap_ns >= short_gap_ns:
            # 短缺口 → 标记
            multiplier = gap_ns / expected_interval_ns if expected_interval_ns else 0.0
            issues.append(QualityIssue(
                issue_type="timeseries_short_gap",
                stream_id=stream_id,
                start_ns=start_ns,
                end_ns=end_ns,
                severity="warning",
                decision="keep_with_flag",
                details={
                    "gap_ns": gap_ns,
                    "gap_s": round(gap_ns / 1e9, 3),
                    "expected_interval_ns": expected_interval_ns,
                    "multiplier": round(multiplier, 1) if multiplier else None,
                    "threshold_short_ns": short_gap_ns,
                    "sample_index": int(idx),
                    "check": "timeseries_gap",
                },
            ))

    return issues


def detect_camera_gaps(
    video_stream: VideoStream,
    config: dict | None = None,
) -> list[QualityIssue]:
    """检测相机流中帧间时间缺口（相机缺失）。

    短缺失 → keep_with_flag（标记但不切分）
    长缺失 → split（切分依据）

    Args:
        video_stream: VideoStream 对象（image_sequence 源）。
        config: 配置字典，支持:
            - short_absence_ns: 短缺失阈值 (默认 1s)
            - long_absence_ns: 长缺失阈值 (默认 5s)

    Returns:
        QualityIssue 列表。
    """
    if config is None:
        config = {}

    short_absence_ns = int(config.get("short_absence_ns", DEFAULT_CAMERA_SHORT_ABSENCE_NS))
    long_absence_ns = int(config.get("long_absence_ns", DEFAULT_CAMERA_LONG_ABSENCE_NS))

    issues: list[QualityIssue] = []
    stream_id = video_stream.stream_id

    timestamps = np.array(video_stream.timestamps_ns, dtype=np.int64)
    if len(timestamps) < 2:
        return issues

    # 获取帧索引（image_sequence 特有）
    index_frames = video_stream.index_frames
    if not index_frames:
        return issues

    # 过滤有效时间戳（timestamp_method == "aligned_joints_index"）
    valid_indices = [
        i for i, f in enumerate(index_frames)
        if f.get("timestamp_method") != "pending_alignment"
        and f.get("source_timestamp_ns") is not None
    ]

    if len(valid_indices) < 2:
        return issues

    # 检查相邻帧间的 gap
    for i in range(len(valid_indices) - 1):
        curr_idx = valid_indices[i]
        next_idx = valid_indices[i + 1]
        curr_ts = int(timestamps[curr_idx])
        next_ts = int(timestamps[next_idx])
        gap_ns = next_ts - curr_ts

        if gap_ns <= 0:
            continue  # 非单调（已由其他检测器处理）

        # 帧间帧数差
        curr_frame = index_frames[curr_idx].get("frame_index", curr_idx)
        next_frame = index_frames[next_idx].get("frame_index", next_idx)
        frame_skip = next_frame - curr_frame - 1  # 跳过的帧数

        if gap_ns >= long_absence_ns:
            issues.append(QualityIssue(
                issue_type="camera_long_absence",
                stream_id=stream_id,
                start_ns=curr_ts,
                end_ns=next_ts,
                severity="error",
                decision="split",
                details={
                    "gap_ns": gap_ns,
                    "gap_s": round(gap_ns / 1e9, 3),
                    "frames_skipped": frame_skip,
                    "threshold_long_ns": long_absence_ns,
                    "check": "camera_absence",
                },
            ))
        elif gap_ns >= short_absence_ns:
            issues.append(QualityIssue(
                issue_type="camera_short_absence",
                stream_id=stream_id,
                start_ns=curr_ts,
                end_ns=next_ts,
                severity="warning",
                decision="keep_with_flag",
                details={
                    "gap_ns": gap_ns,
                    "gap_s": round(gap_ns / 1e9, 3),
                    "frames_skipped": frame_skip,
                    "threshold_short_ns": short_absence_ns,
                    "check": "camera_absence",
                },
            ))

    return issues


__all__ = ["detect_timeseries_gaps", "detect_camera_gaps"]
