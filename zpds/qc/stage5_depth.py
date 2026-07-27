"""Stage 5: 深度有效性检测（D15）。

检测内容：
  - zero_ratio：零值像素比例
  - invalid_ratio：无效值比例（含零值 + 特定 invalid value）
  - valid_pixel_ratio：有效像素比例
  - all_zero_depth：全零深度帧检测
  - frozen_depth：冻结深度检测（连续帧完全相同）
  - dtype 和单位检查
  - RGB-Depth 配对率
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from zpds.core.decisions import Decision, ReasonCode, Severity
from zpds.qc.cascade import register_stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认阈值
# ---------------------------------------------------------------------------

DEFAULT_ZERO_RATIO_MAX = 0.50          # 零值比例上限
DEFAULT_INVALID_RATIO_MAX = 0.60       # 无效值比例上限
DEFAULT_VALID_RATIO_MIN = 0.40         # 有效像素比例下限
DEFAULT_FROZEN_CONSECUTIVE_MIN = 3     # 连续相同帧最少触发数
DEFAULT_RGB_DEPTH_PAIRING_MIN = 0.90   # RGB-Depth 最低配对比率


# ---------------------------------------------------------------------------
# 深度帧级别检测
# ---------------------------------------------------------------------------


def check_depth_frame(
    depth: np.ndarray,
    *,
    invalid_value: int | None = None,
    zero_ratio_max: float = DEFAULT_ZERO_RATIO_MAX,
    invalid_ratio_max: float = DEFAULT_INVALID_RATIO_MAX,
) -> dict:
    """检测单帧深度的基本指标。

    Parameters
    ----------
    depth : np.ndarray
        深度帧（uint16 或 float）。
    invalid_value : Optional[int]
        已知的无效值标记（如 0 表示无效）。
    zero_ratio_max : float
    invalid_ratio_max : float

    Returns
    -------
    dict with keys: zero_ratio, invalid_ratio, valid_ratio, all_zero, mean, std, min, max
    """
    total = depth.size
    if total == 0:
        return {
            "zero_ratio": 1.0,
            "invalid_ratio": 1.0,
            "valid_ratio": 0.0,
            "all_zero": True,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
        }

    zero_mask = depth == 0
    zero_ratio = float(np.sum(zero_mask) / total)

    if invalid_value is not None and invalid_value != 0:
        invalid_mask = (depth == 0) | (depth == invalid_value)
    else:
        invalid_mask = zero_mask

    invalid_ratio = float(np.sum(invalid_mask) / total)
    valid_ratio = 1.0 - invalid_ratio
    all_zero = bool(np.all(zero_mask))

    valid_pixels = depth[~invalid_mask]
    if len(valid_pixels) > 0:
        d_mean = float(np.mean(valid_pixels))
        d_std = float(np.std(valid_pixels))
        d_min = int(np.min(valid_pixels))
        d_max = int(np.max(valid_pixels))
    else:
        d_mean = 0.0
        d_std = 0.0
        d_min = 0
        d_max = 0

    return {
        "zero_ratio": round(zero_ratio, 6),
        "invalid_ratio": round(invalid_ratio, 6),
        "valid_ratio": round(valid_ratio, 6),
        "all_zero": all_zero,
        "mean": round(d_mean, 2),
        "std": round(d_std, 2),
        "min": d_min,
        "max": d_max,
    }


def check_depth_sequence(
    depth_frames: list[np.ndarray],
    *,
    invalid_value: int | None = None,
    zero_ratio_max: float = DEFAULT_ZERO_RATIO_MAX,
    invalid_ratio_max: float = DEFAULT_INVALID_RATIO_MAX,
    valid_ratio_min: float = DEFAULT_VALID_RATIO_MIN,
    frozen_consecutive_min: int = DEFAULT_FROZEN_CONSECUTIVE_MIN,
    timestamps_ns: list[int] | None = None,
    stream_id: str = "depth",
) -> list[Decision]:
    """对深度帧序列执行完整有效性检测。

    Parameters
    ----------
    depth_frames : list[np.ndarray]
        深度帧列表。
    invalid_value : Optional[int]
        已知无效值标记。
    zero_ratio_max, invalid_ratio_max, valid_ratio_min : float
        阈值。
    frozen_consecutive_min : int
        连续完全相同帧的最小触发帧数。
    timestamps_ns : Optional[list[int]]
        每帧的时间戳（纳秒）。
    stream_id : str
        流标识符。

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    n_frames = len(depth_frames)
    if n_frames == 0:
        decisions.append(
            Decision(
                stage=5,
                reason=ReasonCode.DEPTH_INVALID_RATIO,
                severity=Severity.WARN,
                message="Depth sequence is empty",
                detail={"stream_id": stream_id, "frame_count": 0},
            )
        )
        return decisions

    # --- 逐帧分析 ---
    frame_stats: list[dict] = []
    all_zero_count = 0
    total_invalid_px = 0
    total_px = 0

    for i, frame in enumerate(depth_frames):
        stats = check_depth_frame(
            frame,
            invalid_value=invalid_value,
            zero_ratio_max=zero_ratio_max,
            invalid_ratio_max=invalid_ratio_max,
        )
        stats["frame_idx"] = i
        if timestamps_ns and i < len(timestamps_ns):
            stats["timestamp_ns"] = timestamps_ns[i]
        frame_stats.append(stats)

        if stats["all_zero"]:
            all_zero_count += 1

        total_invalid_px += int(stats["invalid_ratio"] * frame.size)
        total_px += frame.size

    # --- 全零深度检测 ---
    all_zero_ratio = all_zero_count / n_frames if n_frames > 0 else 0.0
    if all_zero_ratio > 0.5:
        decisions.append(
            Decision(
                stage=5,
                reason=ReasonCode.DEPTH_INVALID_RATIO,
                severity=Severity.FATAL,
                message=(
                    f"Depth stream '{stream_id}': {all_zero_count}/{n_frames} "
                    f"frames are all-zero ({all_zero_ratio:.1%})"
                ),
                detail={
                    "stream_id": stream_id,
                    "all_zero_frames": all_zero_count,
                    "total_frames": n_frames,
                    "all_zero_ratio": round(all_zero_ratio, 4),
                },
            )
        )
    elif all_zero_count > 0:
        decisions.append(
            Decision(
                stage=5,
                reason=ReasonCode.DEPTH_INVALID_RATIO,
                severity=Severity.WARN,
                message=(
                    f"Depth stream '{stream_id}': {all_zero_count}/{n_frames} "
                    f"frames all-zero"
                ),
                detail={
                    "stream_id": stream_id,
                    "all_zero_frames": all_zero_count,
                    "total_frames": n_frames,
                    "all_zero_ratio": round(all_zero_ratio, 4),
                },
            )
        )

    # --- 整体无效比例 ---
    overall_invalid_ratio = total_invalid_px / total_px if total_px > 0 else 1.0
    overall_valid_ratio = 1.0 - overall_invalid_ratio

    if overall_valid_ratio < valid_ratio_min:
        decisions.append(
            Decision(
                stage=5,
                reason=ReasonCode.DEPTH_INVALID_RATIO,
                severity=Severity.WARN,
                message=(
                    f"Depth stream '{stream_id}': valid pixel ratio "
                    f"{overall_valid_ratio:.2%} < threshold {valid_ratio_min:.2%}"
                ),
                detail={
                    "stream_id": stream_id,
                    "overall_valid_ratio": round(overall_valid_ratio, 4),
                    "overall_invalid_ratio": round(overall_invalid_ratio, 4),
                    "valid_ratio_threshold": valid_ratio_min,
                    "recommended_action": "quarantine",
                },
            )
        )

    # --- 冻结深度检测 ---
    frozen_decisions = _detect_frozen_depth(
        depth_frames,
        frozen_consecutive_min=frozen_consecutive_min,
        timestamps_ns=timestamps_ns,
        stream_id=stream_id,
    )
    decisions.extend(frozen_decisions)

    # --- dtype / 单位信息 ---
    if len(depth_frames) > 0:
        d0 = depth_frames[0]
        decisions.append(
            Decision(
                stage=5,
                reason=ReasonCode.DEPTH_UNIT_UNKNOWN,
                severity=Severity.INFO,
                message=(
                    f"Depth stream '{stream_id}': dtype={d0.dtype}, "
                    f"shape={d0.shape}, value_range=[{d0.min()}, {d0.max()}]"
                ),
                detail={
                    "stream_id": stream_id,
                    "dtype": str(d0.dtype),
                    "shape": list(d0.shape),
                    "total_frames": n_frames,
                    "frame_stats_sample": frame_stats[:5],
                },
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# 冻结深度检测
# ---------------------------------------------------------------------------


def _detect_frozen_depth(
    depth_frames: list[np.ndarray],
    frozen_consecutive_min: int = DEFAULT_FROZEN_CONSECUTIVE_MIN,
    timestamps_ns: list[int] | None = None,
    stream_id: str = "depth",
) -> list[Decision]:
    """检测连续多帧完全相同（冻结深度）。"""
    decisions: list[Decision] = []
    n = len(depth_frames)
    if n < 2:
        return decisions

    frozen_start: int | None = None
    for i in range(1, n):
        same = np.array_equal(depth_frames[i], depth_frames[i - 1])
        if same:
            if frozen_start is None:
                frozen_start = i - 1
        else:
            if frozen_start is not None:
                span_len = i - frozen_start
                if span_len >= frozen_consecutive_min:
                    t0 = timestamps_ns[frozen_start] if timestamps_ns else None
                    t1 = timestamps_ns[i - 1] if timestamps_ns else None
                    decisions.append(
                        Decision(
                            stage=5,
                            reason=ReasonCode.DEPTH_INVALID_RATIO,
                            severity=Severity.WARN,
                            message=(
                                f"Depth stream '{stream_id}': frozen frames "
                                f"[{frozen_start}, {i - 1}] ({span_len} frames identical)"
                            ),
                            frame_idx=frozen_start,
                            timestamp_ns=t0,
                            detail={
                                "stream_id": stream_id,
                                "start_frame": frozen_start,
                                "end_frame": i - 1,
                                "frozen_frame_count": span_len,
                                "start_ns": t0,
                                "end_ns": t1,
                            },
                        )
                    )
                frozen_start = None

    # 尾端冻结
    if frozen_start is not None:
        span_len = n - frozen_start
        if span_len >= frozen_consecutive_min:
            t0 = timestamps_ns[frozen_start] if timestamps_ns else None
            t1 = timestamps_ns[-1] if timestamps_ns else None
            decisions.append(
                Decision(
                    stage=5,
                    reason=ReasonCode.DEPTH_INVALID_RATIO,
                    severity=Severity.WARN,
                    message=(
                        f"Depth stream '{stream_id}': frozen frames "
                        f"[{frozen_start}, {n - 1}] ({span_len} frames identical)"
                    ),
                    frame_idx=frozen_start,
                    timestamp_ns=t0,
                    detail={
                        "stream_id": stream_id,
                        "start_frame": frozen_start,
                        "end_frame": n - 1,
                        "frozen_frame_count": span_len,
                        "start_ns": t0,
                        "end_ns": t1,
                    },
                )
            )

    return decisions


# ---------------------------------------------------------------------------
# RGB-Depth 配对率
# ---------------------------------------------------------------------------


def check_rgb_depth_pairing(
    rgb_timestamps_ns: list[int],
    depth_timestamps_ns: list[int],
    max_offset_ns: int = 50_000_000,  # 50ms
    stream_id_rgb: str = "rgb",
    stream_id_depth: str = "depth",
) -> list[Decision]:
    """检查 RGB 与 Depth 时间戳的配对率。

    对每个 RGB 帧查找最近的 Depth 帧，统计配对率。

    Parameters
    ----------
    rgb_timestamps_ns : list[int]
        RGB 帧时间戳。
    depth_timestamps_ns : list[int]
        Depth 帧时间戳。
    max_offset_ns : int
        最大允许时间偏移（ns）。
    stream_id_rgb, stream_id_depth : str

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    if not rgb_timestamps_ns or not depth_timestamps_ns:
        return [
            Decision(
                stage=5,
                reason=ReasonCode.DEPTH_INVALID_RATIO,
                severity=Severity.WARN,
                message="Cannot check RGB-Depth pairing: empty timestamps",
                detail={
                    "rgb_frame_count": len(rgb_timestamps_ns),
                    "depth_frame_count": len(depth_timestamps_ns),
                },
            )
        ]

    depth_arr = np.array(depth_timestamps_ns, dtype=np.int64)
    paired = 0
    offsets_ns: list[int] = []

    for ts in rgb_timestamps_ns:
        idx = np.argmin(np.abs(depth_arr - ts))
        offset = abs(int(depth_arr[idx]) - ts)
        offsets_ns.append(offset)
        if offset <= max_offset_ns:
            paired += 1

    pairing_ratio = paired / len(rgb_timestamps_ns)
    median_offset_ns = int(np.median(offsets_ns)) if offsets_ns else 0
    max_offset_found = int(max(offsets_ns)) if offsets_ns else 0

    if pairing_ratio < DEFAULT_RGB_DEPTH_PAIRING_MIN:
        severity = Severity.ERROR if pairing_ratio < 0.5 else Severity.WARN
        decisions.append(
            Decision(
                stage=5,
                reason=ReasonCode.DEPTH_INVALID_RATIO,
                severity=severity,
                message=(
                    f"RGB-Depth pairing ratio {pairing_ratio:.2%} "
                    f"< threshold {DEFAULT_RGB_DEPTH_PAIRING_MIN:.0%}, "
                    f"median offset={median_offset_ns / 1e6:.1f}ms"
                ),
                detail={
                    f"{stream_id_rgb}_frame_count": len(rgb_timestamps_ns),
                    f"{stream_id_depth}_frame_count": len(depth_timestamps_ns),
                    "paired_frames": paired,
                    "pairing_ratio": round(pairing_ratio, 4),
                    "max_offset_ns": max_offset_ns,
                    "median_offset_ns": median_offset_ns,
                    "max_offset_found_ns": max_offset_found,
                    "recommended_action": "quarantine",
                },
            )
        )
    else:
        decisions.append(
            Decision(
                stage=5,
                reason=ReasonCode.DEPTH_INVALID_RATIO,
                severity=Severity.INFO,
                message=(
                    f"RGB-Depth pairing ratio {pairing_ratio:.2%}, "
                    f"median offset={median_offset_ns / 1e6:.1f}ms"
                ),
                detail={
                    f"{stream_id_rgb}_frame_count": len(rgb_timestamps_ns),
                    f"{stream_id_depth}_frame_count": len(depth_timestamps_ns),
                    "paired_frames": paired,
                    "pairing_ratio": round(pairing_ratio, 4),
                    "median_offset_ns": median_offset_ns,
                },
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# Stage 5 统一入口
# ---------------------------------------------------------------------------


def check(
    depth_frames: list[np.ndarray] | None = None,
    *,
    depth_dir: str | None = None,
    rgb_timestamps_ns: list[int] | None = None,
    depth_timestamps_ns: list[int] | None = None,
    invalid_value: int | None = None,
    stage_config: dict | None = None,
    stream_id: str = "depth",
) -> list[Decision]:
    """Stage 5 统一检查入口。

    可以通过 ``depth_frames`` 直接传入帧列表，或通过 ``depth_dir``
    指定 PNG 深度序列目录（自动加载）。

    Parameters
    ----------
    depth_frames : Optional[list[np.ndarray]]
        深度帧列表（内存中）。
    depth_dir : Optional[str]
        深度 PNG 序列目录。
    rgb_timestamps_ns : Optional[list[int]]
        RGB 时间戳（用于配对率检查）。
    depth_timestamps_ns : Optional[list[int]]
        深度时间戳（用于配对率检查）。
    invalid_value : Optional[int]
        已知的无效深度值标记。
    stage_config : Optional[dict]
        阈值覆盖配置。
    stream_id : str
        流标识符。

    Returns
    -------
    list[Decision]
    """
    cfg = stage_config or {}
    decisions: list[Decision] = []

    # --- 加载深度帧 ---
    frames: list[np.ndarray] = []
    if depth_frames is not None:
        frames = list(depth_frames)
    elif depth_dir is not None:
        depth_path = Path(depth_dir)
        if depth_path.is_dir():
            png_files = sorted(depth_path.glob("*.png"))
            import cv2
            for pf in png_files:
                img = cv2.imread(str(pf), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    frames.append(img)
            if not png_files:
                decisions.append(
                    Decision(
                        stage=5,
                        reason=ReasonCode.DEPTH_INVALID_RATIO,
                        severity=Severity.WARN,
                        message=f"No PNG files found in depth dir: {depth_dir}",
                    )
                )

    if not frames and not rgb_timestamps_ns:
        return decisions

    # --- 阈值 ---
    zero_ratio_max = cfg.get("zero_ratio_max", DEFAULT_ZERO_RATIO_MAX)
    invalid_ratio_max = cfg.get("invalid_ratio_max", DEFAULT_INVALID_RATIO_MAX)
    valid_ratio_min = cfg.get("valid_ratio_min", DEFAULT_VALID_RATIO_MIN)
    frozen_consecutive_min = cfg.get(
        "frozen_consecutive_min", DEFAULT_FROZEN_CONSECUTIVE_MIN
    )

    if frames:
        decisions.extend(
            check_depth_sequence(
                frames,
                invalid_value=invalid_value or cfg.get("invalid_value"),
                zero_ratio_max=zero_ratio_max,
                invalid_ratio_max=invalid_ratio_max,
                valid_ratio_min=valid_ratio_min,
                frozen_consecutive_min=frozen_consecutive_min,
                timestamps_ns=depth_timestamps_ns,
                stream_id=stream_id,
            )
        )

    # --- RGB-Depth 配对 ---
    if rgb_timestamps_ns is not None and depth_timestamps_ns is not None:
        pairing_cfg = cfg.get("rgb_depth_pairing", {})
        decisions.extend(
            check_rgb_depth_pairing(
                rgb_timestamps_ns,
                depth_timestamps_ns,
                max_offset_ns=pairing_cfg.get("max_offset_ns", 50_000_000),
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# QCCascade 注册入口
# ---------------------------------------------------------------------------


@register_stage(5)
def _check_stage5(context: dict) -> list[Decision]:
    """Stage 5 QCCascade 入口：从 context dict 提取参数并调用 check()。"""
    stage_config = context.get("stage_config", {})
    return check(
        depth_frames=context.get("depth_frames"),
        depth_dir=context.get("depth_dir"),
        rgb_timestamps_ns=context.get("rgb_timestamps_ns"),
        depth_timestamps_ns=context.get("depth_timestamps_ns"),
        invalid_value=context.get("depth_invalid_value"),
        stage_config=stage_config,
        stream_id=context.get("stream_id", "depth"),
    )
