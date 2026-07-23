"""
从 QC 结果确定 Segment 的有效时间范围。

策略（第一版）：
- 从 index.jsonl 获取全部帧时间戳
- 根据黑帧列表确定头部裁剪点（跳过连续黑屏）
- 根据 IMU 缺口确定尾部裁剪点（在长缺口前停止）
- 暂不处理中间的小异常
"""

import json
import yaml
from pathlib import Path


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_index(dataset_path: str) -> list[dict]:
    """读取 index.jsonl 中所有 type=frame 的行。"""
    index_path = Path(dataset_path) / "index.jsonl"
    frames = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "frame":
                frames.append(item)
    return frames


def load_segment_info(dataset_path: str) -> dict:
    """从 index.jsonl 提取 segment 信息（文件名映射）。"""
    index_path = Path(dataset_path) / "index.jsonl"
    segments = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "segment_start":
                seg = item["segment"]
                segments[seg] = {
                    "color_video": item.get("color_video", ""),
                    "depth_video": item.get("depth_video", ""),
                }
    return segments


def determine_span(
    dataset_path: str,
    black_frame_indices: list[int] | None = None,
    imu_gap_samples: list[tuple] | None = None,
    timestamp_gaps: list[tuple] | None = None,
    config_path: str = "config.yaml",
    black_issues: list | None = None,  # list[QualityIssue], 新版统一格式
) -> dict:
    """确定一个 Session 的有效时间区间。

    Args:
        dataset_path: 数据集根目录（含 index.jsonl）
        black_frame_indices: 黑帧序号列表（来自 video_checker）
        imu_gap_samples: IMU 中断列表 [(sample_idx, gap_s), ...]
        timestamp_gaps: 时间戳间隔异常列表 [(frame_idx, gap_ms), ...]
        config_path: YAML 配置路径

    Returns:
        {
            "source_start_ns": int,
            "source_end_ns": int,
            "duration_s": float,
            "total_frames_in_span": int,
            "reason": {"start": str, "end": str},
            "trimmed_head_frames": int,
            "trimmed_tail_frames": int,
        }
    """
    cfg = load_config(config_path)
    frames = load_index(dataset_path)

    if not frames:
        raise ValueError("index.jsonl 中没有 type=frame 的记录")

    timestamps = [f["timestamp_ns"] for f in frames]
    min_black_duration_ns = int(cfg["video"]["min_black_duration_s"] * 1_000_000_000)

    # 若传入了新版 QualityIssue 列表，从中提取头部 trim 信息
    if black_issues and not black_frame_indices:
        head_trim_issues = [
            iss for iss in black_issues
            if iss.issue_type == "continuous_black_frames" and iss.decision == "trim"
        ]
        if head_trim_issues:
            # 取最晚的头部 trim 结束点
            latest_trim_end = max(iss.end_ns for iss in head_trim_issues)
            # 计算对应的帧索引（第一个时间戳 > latest_trim_end 的帧）
            for i, ts in enumerate(timestamps):
                if ts >= latest_trim_end:
                    black_frame_indices = list(range(0, i))
                    break
            if black_frame_indices is None:
                black_frame_indices = []

    # ---- 计算头部裁剪 ----
    head_trim_ns = 0
    reason_start = "no_trim_needed"

    if black_frame_indices:
        # 找开头的连续黑帧
        consecutive_from_start = 0
        for i, idx in enumerate(black_frame_indices):
            if idx == i and idx < len(timestamps):
                consecutive_from_start += 1
            else:
                break

        if consecutive_from_start > 0 and consecutive_from_start < len(timestamps):
            first_black_ts = timestamps[black_frame_indices[0]]
            last_black_ts = timestamps[black_frame_indices[consecutive_from_start - 1]]
            black_duration = last_black_ts - first_black_ts

            if black_duration >= min_black_duration_ns:
                # 裁剪到最后一帧黑屏之后的第一帧
                head_trim_idx = black_frame_indices[consecutive_from_start - 1] + 1
                head_trim_ns = timestamps[head_trim_idx]
                reason_start = f"remove_initial_black_frames ({consecutive_from_start} frames)"
            elif consecutive_from_start >= 3:
                # 即使时间不够，连续黑帧也不该保留
                head_trim_idx = black_frame_indices[consecutive_from_start - 1] + 1
                head_trim_ns = timestamps[head_trim_idx]
                reason_start = f"remove_initial_black_frames_short ({consecutive_from_start} frames)"

    # ---- 计算尾部裁剪 ----
    tail_trim_ns = timestamps[-1]
    reason_end = "no_trim_needed"

    if imu_gap_samples:
        # 找最后一个大的 IMU 缺口，在其之前停止
        imu_gap_factor = cfg["timestamp"]["imu_gap_factor"]
        for sample_idx, gap_s in reversed(imu_gap_samples):
            if gap_s > imu_gap_factor * 0.02:  # 超过正常采样间隔的 3 倍
                # 缺口位置对应的帧索引近似（IMU 采样与视频帧有对应关系）
                # 简单处理：在缺口前的最后一个帧处停止
                # sample_idx 是去重后的 IMU 样本索引
                # 这里用比例映射到视频帧
                imu_ratio = sample_idx / 983  # 假设 ~983 唯一 IMU 时间戳
                approx_frame_idx = int(imu_ratio * len(timestamps))
                if 0 < approx_frame_idx < len(timestamps) - 10:
                    tail_trim_ns = timestamps[approx_frame_idx]
                    reason_end = f"stop_before_long_imu_gap ({gap_s:.3f}s at IMU sample {sample_idx})"
                    break

    # 默认：使用全部帧
    source_start_ns = max(timestamps[0], head_trim_ns)
    source_end_ns = max(source_start_ns + 1_000_000_000, tail_trim_ns)  # 至少 1 秒

    # 计算裁剪帧数
    trimmed_head = sum(1 for ts in timestamps if ts < source_start_ns)
    trimmed_tail = sum(1 for ts in timestamps if ts > source_end_ns)
    frames_in_span = sum(
        1 for ts in timestamps if source_start_ns <= ts <= source_end_ns
    )

    return {
        "source_start_ns": source_start_ns,
        "source_end_ns": source_end_ns,
        "duration_s": (source_end_ns - source_start_ns) / 1_000_000_000,
        "total_frames_in_span": frames_in_span,
        "reason": {"start": reason_start, "end": reason_end},
        "trimmed_head_frames": trimmed_head,
        "trimmed_tail_frames": trimmed_tail,
    }
