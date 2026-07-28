"""Stage 6: IMU 异常检测（D16）。

检测内容：
  - 时间间隔异常（gap / 重复时间戳）
  - 尖峰检测（加速度 / 角速度超出统计范围）
  - 冻结检测（连续 N 个样本完全相同）
  - 静止窗口零偏估计
  - 饱和检查（仅在已知设备量程时启用）
"""

from __future__ import annotations

import logging

import numpy as np

from zpds.core.decisions import Decision, ReasonCode, Severity
from zpds.qc.cascade import register_stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认阈值
# ---------------------------------------------------------------------------

DEFAULT_EXPECTED_INTERVAL_S = 0.005      # 默认期望采样间隔 (200 Hz)
DEFAULT_GAP_FACTOR = 3.0                 # 间隔超过 N 倍期望值 -> gap
DEFAULT_SPIKE_STD_FACTOR = 6.0           # 超过 N 倍标准差 -> spike
DEFAULT_FREEZE_CONSECUTIVE_MIN = 10      # 连续相同最少触发数
DEFAULT_STATIC_WINDOW_S = 1.0            # 静止窗口时长 (s)

# 典型 IMU 量程（供参考；无明确设备量程时不判定饱和）
TYPICAL_ACCEL_RANGE_MPS2 = 156.96        # ±16g
TYPICAL_GYRO_RANGE_RPS = 34.9066         # ±2000 deg/s


# ---------------------------------------------------------------------------
# 时间间隔异常
# ---------------------------------------------------------------------------


def detect_imu_interval_anomalies(
    timestamps_ns: list[int],
    *,
    expected_interval_s: float = DEFAULT_EXPECTED_INTERVAL_S,
    gap_factor: float = DEFAULT_GAP_FACTOR,
    stream_id: str = "imu",
) -> list[Decision]:
    """检测 IMU 时间戳间隔异常（gap 和重复时间戳）。

    Parameters
    ----------
    timestamps_ns : list[int]
        IMU 采样时间戳（纳秒，已排序）。
    expected_interval_s : float
        期望采样间隔（秒）。
    gap_factor : float
        gap 倍数阈值。
    stream_id : str

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    n = len(timestamps_ns)
    if n < 2:
        return decisions

    # 去重后的唯一时间戳
    ts = np.array(timestamps_ns, dtype=np.int64)
    unique_ts, counts = np.unique(ts, return_counts=True)

    # 重复时间戳
    dup_mask = counts > 1
    dup_count = int(np.sum(dup_mask))
    total_dup_samples = int(np.sum(counts[dup_mask] - 1))
    if dup_count > 0:
        decisions.append(
            Decision(
                stage=6,
                reason=ReasonCode.IMU_GAP,
                severity=Severity.WARN,
                message=(
                    f"IMU '{stream_id}': {dup_count} duplicate timestamp(s), "
                    f"{total_dup_samples} excess sample(s)"
                ),
                detail={
                    "stream_id": stream_id,
                    "duplicate_timestamp_count": dup_count,
                    "excess_sample_count": total_dup_samples,
                },
            )
        )

    if len(unique_ts) < 2:
        return decisions

    # 计算间隔
    intervals_ns = np.diff(unique_ts)
    expected_interval_ns = int(expected_interval_s * 1_000_000_000)
    gap_threshold_ns = int(expected_interval_ns * gap_factor)

    gap_indices = np.where(intervals_ns > gap_threshold_ns)[0]

    for idx in gap_indices:
        gap_ns = int(intervals_ns[idx])
        gap_s = gap_ns / 1_000_000_000
        severity = Severity.ERROR if gap_s > 1.0 else Severity.WARN

        decisions.append(
            Decision(
                stage=6,
                reason=ReasonCode.IMU_GAP,
                severity=severity,
                message=(
                    f"IMU '{stream_id}': gap {gap_s:.3f}s "
                    f"between samples {idx} and {idx + 1} "
                    f"(expected <={expected_interval_s * gap_factor:.3f}s)"
                ),
                timestamp_ns=int(unique_ts[idx]),
                detail={
                    "stream_id": stream_id,
                    "gap_ns": gap_ns,
                    "gap_s": round(gap_s, 6),
                    "sample_idx": int(idx),
                    "start_ns": int(unique_ts[idx]),
                    "end_ns": int(unique_ts[idx + 1]),
                    "expected_interval_ns": expected_interval_ns,
                    "gap_factor": gap_factor,
                    "recommended_action": "split" if gap_s > 1.0 else "keep_with_flag",
                },
            )
        )

    # 汇总统计
    median_interval_ns = int(np.median(intervals_ns))
    actual_rate_hz = 1_000_000_000 / median_interval_ns if median_interval_ns > 0 else 0.0

    decisions.append(
        Decision(
            stage=6,
            reason=ReasonCode.IMU_GAP,
            severity=Severity.INFO,
            message=(
                f"IMU '{stream_id}': {n} samples, "
                f"median interval={median_interval_ns / 1e6:.2f}ms "
                f"(~{actual_rate_hz:.1f} Hz), "
                f"{len(gap_indices)} gap(s) detected"
            ),
            detail={
                "stream_id": stream_id,
                "total_samples": n,
                "unique_timestamps": len(unique_ts),
                "median_interval_ns": median_interval_ns,
                "actual_rate_hz": round(actual_rate_hz, 1),
                "gap_count": len(gap_indices),
                "duplicate_timestamp_count": dup_count,
            },
        )
    )

    return decisions


# ---------------------------------------------------------------------------
# 尖峰检测
# ---------------------------------------------------------------------------


def detect_imu_spikes(
    timestamps_ns: list[int],
    values: np.ndarray,
    *,
    axis_names: list[str] | None = None,
    std_factor: float = DEFAULT_SPIKE_STD_FACTOR,
    stream_id: str = "imu",
) -> list[Decision]:
    """检测 IMU 数据中的尖峰（离群值）。

    使用滚动 MAD（中位数绝对偏差）方法检测尖峰。

    Parameters
    ----------
    timestamps_ns : list[int]
    values : np.ndarray
        shape (N, C) 的 IMU 数据。
    axis_names : Optional[list[str]]
        各轴名称，如 ['ax', 'ay', 'az', 'gx', 'gy', 'gz']。
    std_factor : float
        判定尖峰的 MAD 倍数。
    stream_id : str

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    if values.size == 0:
        return decisions

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_samples, n_axes = values.shape
    if axis_names is None:
        axis_names = [f"axis_{i}" for i in range(n_axes)]
    elif len(axis_names) < n_axes:
        axis_names = list(axis_names) + [f"axis_{i}" for i in range(len(axis_names), n_axes)]

    for ax in range(n_axes):
        col = values[:, ax]
        # 排除 NaN / Inf
        finite = np.isfinite(col)
        if not np.any(finite):
            continue

        finite_vals = col[finite]
        median = np.median(finite_vals)
        mad = np.median(np.abs(finite_vals - median))
        if mad == 0:
            mad = np.std(finite_vals) or 1e-6

        threshold = std_factor * mad
        spike_mask = np.abs(col - median) > threshold
        spike_indices = np.where(spike_mask & finite)[0]

        if len(spike_indices) == 0:
            continue

        spike_ratio = len(spike_indices) / n_samples if n_samples > 0 else 0.0
        severity = Severity.WARN if spike_ratio < 0.05 else Severity.ERROR

        decisions.append(
            Decision(
                stage=6,
                reason=ReasonCode.IMU_BIAS_DRIFT,
                severity=severity,
                message=(
                    f"IMU '{stream_id}' axis '{axis_names[ax]}': "
                    f"{len(spike_indices)} spike(s) detected "
                    f"({spike_ratio:.1%}), threshold={threshold:.2f}"
                ),
                detail={
                    "stream_id": stream_id,
                    "axis": axis_names[ax],
                    "spike_count": len(spike_indices),
                    "spike_ratio": round(spike_ratio, 4),
                    "median": round(float(median), 4),
                    "mad": round(float(mad), 4),
                    "threshold": round(float(threshold), 4),
                    "std_factor": std_factor,
                    "spike_indices": spike_indices[:20].tolist(),
                },
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# 冻结检测
# ---------------------------------------------------------------------------


def detect_imu_freeze(
    timestamps_ns: list[int],
    values: np.ndarray,
    *,
    consecutive_min: int = DEFAULT_FREEZE_CONSECUTIVE_MIN,
    stream_id: str = "imu",
) -> list[Decision]:
    """检测 IMU 信号冻结（连续样本完全相同）。

    Parameters
    ----------
    timestamps_ns : list[int]
    values : np.ndarray
        shape (N, C) 的 IMU 数据。
    consecutive_min : int
        连续相同最少触发样本数。
    stream_id : str

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    n = values.shape[0]
    if n < consecutive_min:
        return decisions

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    # 按行检查是否相同
    frozen_start: int | None = None
    for i in range(1, n):
        same = np.array_equal(values[i], values[i - 1])
        if same:
            if frozen_start is None:
                frozen_start = i - 1
        else:
            _close_freeze_span(
                decisions,
                frozen_start,
                i - 1,
                timestamps_ns,
                consecutive_min,
                stream_id,
            )
            frozen_start = None

    _close_freeze_span(
        decisions, frozen_start, n - 1, timestamps_ns, consecutive_min, stream_id
    )

    return decisions


def _close_freeze_span(
    decisions: list[Decision],
    start: int | None,
    end: int,
    timestamps_ns: list[int],
    consecutive_min: int,
    stream_id: str,
) -> None:
    if start is None:
        return
    span_len = end - start + 1
    if span_len < consecutive_min:
        return
    t0 = timestamps_ns[start] if start < len(timestamps_ns) else None
    t1 = timestamps_ns[end] if end < len(timestamps_ns) else None
    decisions.append(
        Decision(
            stage=6,
            reason=ReasonCode.IMU_BIAS_DRIFT,
            severity=Severity.WARN,
            message=(
                f"IMU '{stream_id}': frozen signal [{start}, {end}] "
                f"({span_len} identical samples)"
            ),
            timestamp_ns=t0,
            detail={
                "stream_id": stream_id,
                "start_sample": start,
                "end_sample": end,
                "frozen_sample_count": span_len,
                "start_ns": t0,
                "end_ns": t1,
            },
        )
    )


# ---------------------------------------------------------------------------
# 静止窗口零偏估计
# ---------------------------------------------------------------------------


def estimate_static_bias(
    timestamps_ns: list[int],
    values: np.ndarray,
    *,
    axis_names: list[str] | None = None,
    static_window_s: float = DEFAULT_STATIC_WINDOW_S,
    stream_id: str = "imu",
) -> list[Decision]:
    """在静止窗口中估计 IMU 零偏。

    使用加速度幅值接近重力（9.8 ± 0.5 m/s²）且角速度接近 0 的窗口作为静止段。

    Parameters
    ----------
    timestamps_ns : list[int]
    values : np.ndarray
        shape (N,>=6) [ax, ay, az, gx, gy, gz]（m/s², rad/s）。
    axis_names : Optional[list[str]]
    static_window_s : float
        静止判定窗口时长。
    stream_id : str

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    n = values.shape[0]
    if n < 3 or values.shape[1] < 6:
        return decisions

    # 前 3 列视为加速度，后 3 列视为角速度
    accel = values[:, :3]
    gyro = values[:, 3:6]

    # 加速度幅值
    accel_mag = np.sqrt(np.sum(accel ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))

    # 静止判定：加速度接近 9.81 且 角速度接近 0
    static_mask = (
        (np.abs(accel_mag - 9.81) < 0.5)
        & (gyro_mag < 0.05)
        & np.all(np.isfinite(values), axis=1)
    )

    static_indices = np.where(static_mask)[0]
    if len(static_indices) < 10:
        decisions.append(
            Decision(
                stage=6,
                reason=ReasonCode.IMU_BIAS_DRIFT,
                severity=Severity.INFO,
                message=(
                    f"IMU '{stream_id}': insufficient static samples for bias "
                    f"estimation ({len(static_indices)} found)"
                ),
                detail={"stream_id": stream_id, "static_samples": len(static_indices)},
            )
        )
        return decisions

    # 在静止段上计算统计
    static_accel = accel[static_indices]
    static_gyro = gyro[static_indices]

    accel_bias = np.mean(static_accel, axis=0)
    accel_std = np.std(static_accel, axis=0)
    gyro_bias = np.mean(static_gyro, axis=0)
    gyro_std = np.std(static_gyro, axis=0)

    # 重力方向加速度幅值偏置
    expected_g = 9.81
    accel_mag_mean = np.mean(accel_mag[static_indices])
    accel_mag_bias = accel_mag_mean - expected_g

    has_sig_bias = np.abs(accel_mag_bias) > 0.5 or np.any(np.abs(gyro_bias) > 0.05)

    decisions.append(
        Decision(
            stage=6,
            reason=ReasonCode.IMU_BIAS_DRIFT,
            severity=Severity.WARN if has_sig_bias else Severity.INFO,
            message=(
                f"IMU '{stream_id}': static bias estimate from "
                f"{len(static_indices)} samples — "
                f"accel mag bias={accel_mag_bias:.3f} m/s², "
                f"gyro bias mag={np.linalg.norm(gyro_bias):.4f} rad/s"
            ),
            detail={
                "stream_id": stream_id,
                "static_sample_count": len(static_indices),
                "static_ratio": round(len(static_indices) / n, 4) if n > 0 else 0,
                "accel_bias_x": round(float(accel_bias[0]), 6),
                "accel_bias_y": round(float(accel_bias[1]), 6),
                "accel_bias_z": round(float(accel_bias[2]), 6),
                "accel_std_x": round(float(accel_std[0]), 6),
                "accel_std_y": round(float(accel_std[1]), 6),
                "accel_std_z": round(float(accel_std[2]), 6),
                "gyro_bias_x": round(float(gyro_bias[0]), 6),
                "gyro_bias_y": round(float(gyro_bias[1]), 6),
                "gyro_bias_z": round(float(gyro_bias[2]), 6),
                "gyro_std_x": round(float(gyro_std[0]), 6),
                "gyro_std_y": round(float(gyro_std[1]), 6),
                "gyro_std_z": round(float(gyro_std[2]), 6),
                "accel_mag_bias_mps2": round(float(accel_mag_bias), 6),
            },
        )
    )

    return decisions


# ---------------------------------------------------------------------------
# 饱和检查
# ---------------------------------------------------------------------------


def detect_imu_saturation(
    values: np.ndarray,
    *,
    accel_range_mps2: float | None = None,
    gyro_range_rps: float | None = None,
    axis_names: list[str] | None = None,
    stream_id: str = "imu",
) -> list[Decision]:
    """检测 IMU 数据是否达到传感器量程上限（饱和）。

    仅在明确提供量程参数时才执行此检查。
    没有明确设备量程时，不判定"传感器饱和"（符合 AGENTS.md 约定）。

    Parameters
    ----------
    values : np.ndarray
        shape (N,>=6) [ax, ay, az, gx, gy, gz]。
    accel_range_mps2 : Optional[float]
        加速度计量程 (m/s²)。
    gyro_range_rps : Optional[float]
        陀螺仪量程 (rad/s)。
    axis_names : Optional[list[str]]
    stream_id : str

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    n = values.shape[0]
    if n == 0:
        return decisions

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if axis_names is None:
        axis_names = [f"axis_{i}" for i in range(values.shape[1])]

    saturation_warned = False

    if accel_range_mps2 is not None and values.shape[1] >= 3:
        for ax in range(3):
            col = np.abs(values[:, ax])
            saturated = col >= accel_range_mps2 * 0.98
            sat_count = int(np.sum(saturated))
            if sat_count > 0:
                saturation_warned = True
                decisions.append(
                    Decision(
                        stage=6,
                        reason=ReasonCode.IMU_SATURATION,
                        severity=Severity.WARN,
                        message=(
                            f"IMU '{stream_id}' axis '{axis_names[ax]}': "
                            f"{sat_count} sample(s) near or at saturation "
                            f"(range={accel_range_mps2} m/s²)"
                        ),
                        detail={
                            "stream_id": stream_id,
                            "axis": axis_names[ax],
                            "saturated_sample_count": sat_count,
                            "saturated_ratio": round(sat_count / n, 4),
                            "accel_range_mps2": accel_range_mps2,
                        },
                    )
                )

    if gyro_range_rps is not None and values.shape[1] >= 6:
        for ax in range(3, 6):
            col = np.abs(values[:, ax])
            saturated = col >= gyro_range_rps * 0.98
            sat_count = int(np.sum(saturated))
            if sat_count > 0:
                saturation_warned = True
                decisions.append(
                    Decision(
                        stage=6,
                        reason=ReasonCode.IMU_SATURATION,
                        severity=Severity.WARN,
                        message=(
                            f"IMU '{stream_id}' axis '{axis_names[ax]}': "
                            f"{sat_count} sample(s) near or at saturation "
                            f"(range={gyro_range_rps:.1f} rad/s)"
                        ),
                        detail={
                            "stream_id": stream_id,
                            "axis": axis_names[ax],
                            "saturated_sample_count": sat_count,
                            "saturated_ratio": round(sat_count / n, 4),
                            "gyro_range_rps": round(gyro_range_rps, 4),
                        },
                    )
                )

    if not saturation_warned and (accel_range_mps2 is not None or gyro_range_rps is not None):
        decisions.append(
            Decision(
                stage=6,
                reason=ReasonCode.IMU_SATURATION,
                severity=Severity.INFO,
                message=f"IMU '{stream_id}': no saturation detected",
                detail={"stream_id": stream_id},
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# Stage 6 统一入口
# ---------------------------------------------------------------------------


def check(
    timestamps_ns: list[int] | None = None,
    values: np.ndarray | None = None,
    *,
    axis_names: list[str] | None = None,
    accel_range_mps2: float | None = None,
    gyro_range_rps: float | None = None,
    stage_config: dict | None = None,
    stream_id: str = "imu",
) -> list[Decision]:
    """Stage 6 统一检查入口：IMU 间隔 / 尖峰 / 冻结 / 零偏 / 饱和。

    Parameters
    ----------
    timestamps_ns : Optional[list[int]]
        IMU 时间戳（纳秒，已排序）。
    values : Optional[np.ndarray]
        IMU 数据，shape (N, C)。至少需要 [ax, ay, az, gx, gy, gz]。
    axis_names : Optional[list[str]]
        各轴名称。
    accel_range_mps2 : Optional[float]
        加速度计量程。None 时跳过饱和检查。
    gyro_range_rps : Optional[float]
        陀螺仪量程。None 时跳过饱和检查。
    stage_config : Optional[dict]
        阈值覆盖。
    stream_id : str

    Returns
    -------
    list[Decision]
    """
    cfg = stage_config or {}
    decisions: list[Decision] = []

    if timestamps_ns is None or len(timestamps_ns) == 0:
        return [
            Decision(
                stage=6,
                reason=ReasonCode.IMU_GAP,
                severity=Severity.WARN,
                message=f"IMU '{stream_id}': no timestamp data provided",
            )
        ]

    # --- 间隔异常 ---
    interval_cfg = cfg.get("interval", {})
    decisions.extend(
        detect_imu_interval_anomalies(
            timestamps_ns,
            expected_interval_s=interval_cfg.get(
                "expected_interval_s", DEFAULT_EXPECTED_INTERVAL_S
            ),
            gap_factor=interval_cfg.get("gap_factor", DEFAULT_GAP_FACTOR),
            stream_id=stream_id,
        )
    )

    if values is not None and values.size > 0:
        # --- 尖峰 ---
        spike_cfg = cfg.get("spike", {})
        if spike_cfg.get("enabled", True):
            decisions.extend(
                detect_imu_spikes(
                    timestamps_ns,
                    values,
                    axis_names=axis_names,
                    std_factor=spike_cfg.get(
                        "std_factor", DEFAULT_SPIKE_STD_FACTOR
                    ),
                    stream_id=stream_id,
                )
            )

        # --- 冻结 ---
        freeze_cfg = cfg.get("freeze", {})
        if freeze_cfg.get("enabled", True):
            decisions.extend(
                detect_imu_freeze(
                    timestamps_ns,
                    values,
                    consecutive_min=freeze_cfg.get(
                        "consecutive_min", DEFAULT_FREEZE_CONSECUTIVE_MIN
                    ),
                    stream_id=stream_id,
                )
            )

        # --- 静止零偏 ---
        bias_cfg = cfg.get("static_bias", {})
        if bias_cfg.get("enabled", True) and values.shape[1] >= 6:
            decisions.extend(
                estimate_static_bias(
                    timestamps_ns,
                    values,
                    axis_names=axis_names,
                    static_window_s=bias_cfg.get(
                        "static_window_s", DEFAULT_STATIC_WINDOW_S
                    ),
                    stream_id=stream_id,
                )
            )

        # --- 饱和（仅量程已知时） ---
        if accel_range_mps2 is not None or gyro_range_rps is not None:
            decisions.extend(
                detect_imu_saturation(
                    values,
                    accel_range_mps2=accel_range_mps2,
                    gyro_range_rps=gyro_range_rps,
                    axis_names=axis_names,
                    stream_id=stream_id,
                )
            )

    return decisions


# ---------------------------------------------------------------------------
# QCCascade 注册入口
# ---------------------------------------------------------------------------


@register_stage(6)
def _check_stage6(context: dict) -> list[Decision]:
    """Stage 6 QCCascade 入口：从 context dict 提取参数并调用 check()。"""
    stage_config = context.get("stage_config", {})
    return check(
        timestamps_ns=context.get("imu_timestamps_ns"),
        values=context.get("imu_values"),
        axis_names=context.get("imu_axis_names"),
        accel_range_mps2=context.get("imu_accel_range_mps2"),
        gyro_range_rps=context.get("imu_gyro_range_rps"),
        stage_config=stage_config,
        stream_id=context.get("stream_id", "imu"),
    )
