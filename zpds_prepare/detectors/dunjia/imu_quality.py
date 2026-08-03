"""B3: 遁甲 IMU 质量检测。

检查 Dunjia Session 中 IMU 流的物理有效性。

检查项：
  1. 时间戳完整性 — 重复/回退/长 gap
  2. 尖峰检测 — 加速度和角速度 MAD 离群
  3. 冻结检测 — 连续相同样本
  4. 静止窗口零偏估计 — 自动寻找 accel≈9.81 & gyro≈0 的静止段
  5. 饱和检查 — 仅在已知设备量程时判定，量程未知记 ``unavailable``

原则：
  - 使用真实 dt（跨 gap/reset 断开计算）
  - 不对跨 gap 做插值
  - 每项异常带时间戳、样本索引和原始值
  - 设备量程有来源才判定饱和；量程未知时饱和状态为 ``unavailable``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# IMU 必需列
_REQUIRED_IMU_COLUMNS = {"timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"}

# 默认阈值
_DEFAULT_SPIKE_STD_FACTOR = 6.0
_DEFAULT_FREEZE_MIN = 5
_DEFAULT_GAP_FACTOR = 3.0
_DEFAULT_STATIC_WINDOW_S = 0.5
_STATIC_ACCEL_TOLERANCE = 0.5  # |accel_mag - 9.81| < 0.5 m/s²
_STATIC_GYRO_TOLERANCE = 0.05  # gyro_mag < 0.05 rad/s


# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------


@dataclass
class IMUGapSpan:
    """IMU 时间戳缺口。"""

    start_sample: int
    end_sample: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    gap_ns: int
    gap_s: float
    expected_interval_ns: int
    factor: float


@dataclass
class IMUSpikeEvent:
    """IMU 尖峰事件。"""

    sample_index: int
    timestamp_ns: int
    field: str  # "ax", "gy", etc.
    value: float
    median: float
    mad: float
    deviation_factor: float  # |value - median| / mad


@dataclass
class IMUStaticWindow:
    """IMU 静止窗口。"""

    start_sample: int
    end_sample: int
    duration_s: float
    accel_bias: tuple[float, float, float]  # mean ax, ay, az
    gyro_bias: tuple[float, float, float]  # mean gx, gy, gz
    accel_mag_mean: float
    gyro_mag_mean: float


@dataclass
class DunjiaIMUReport:
    """遁甲 IMU 质量报告。"""

    session_id: str
    source_path: str
    schema_version: str = "zpds.dunjia_imu.v1"

    # 基本信息
    sample_count: int = 0
    sample_rate_hz: float = 0.0
    expected_interval_ns: int = 0
    median_interval_ns: int = 0
    timestamp_valid: bool = False
    has_duplicates: bool = False
    duplicate_count: int = 0
    has_regression: bool = False
    regression_count: int = 0

    # gap 统计
    gap_count: int = 0
    total_gap_duration_s: float = 0.0
    gaps: list[IMUGapSpan] = field(default_factory=list)

    # 尖峰
    spike_count: int = 0
    spikes: list[IMUSpikeEvent] = field(default_factory=list)

    # 冻结
    freeze_span_count: int = 0
    freeze_total_samples: int = 0

    # 静止零偏
    static_window_count: int = 0
    static_windows: list[IMUStaticWindow] = field(default_factory=list)

    # 饱和
    saturation_status: str = "unavailable"  # "unavailable" | "checked" | "partial"
    saturation_accel_count: int = 0
    saturation_gyro_count: int = 0
    accel_range_mps2: float | None = None
    gyro_range_rps: float | None = None

    # 聚合
    overall_disposition: str = "pass"
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def check_dunjia_imu(
    session: Any,
    *,
    spike_std_factor: float = _DEFAULT_SPIKE_STD_FACTOR,
    freeze_min: int = _DEFAULT_FREEZE_MIN,
    gap_factor: float = _DEFAULT_GAP_FACTOR,
    static_window_s: float = _DEFAULT_STATIC_WINDOW_S,
    accel_range_mps2: float | None = None,
    gyro_range_rps: float | None = None,
) -> DunjiaIMUReport:
    """检查遁甲 IMU 质量。

    Args:
        session: ``Session`` 对象（来自 ``dunjia_reader.read_session()``）
        spike_std_factor: MAD 尖峰检测倍数（默认 6.0）
        freeze_min: 冻结检测最少连续相同样本数
        gap_factor: 间隔超过 × 中位数 → gap
        static_window_s: 静止窗口最短时长（秒）
        accel_range_mps2: 加速度计量程（None = 未知）
        gyro_range_rps: 陀螺仪量程（None = 未知）

    Returns:
        DunjiaIMUReport 包含时间戳/gap/尖峰/冻结/零偏/饱和的全部指标。
    """
    source_path = session.source_path
    report = DunjiaIMUReport(
        session_id=session.session_id,
        source_path=str(source_path),
        accel_range_mps2=accel_range_mps2,
        gyro_range_rps=gyro_range_rps,
    )

    imu_stream = session.imu_streams.get("robot0_imu")
    if imu_stream is None:
        report.issues.append("IMU 流 robot0_imu 不存在")
        report.overall_disposition = "reject"
        return report

    df = imu_stream.dataframe
    if len(df) == 0:
        report.issues.append("IMU DataFrame 为空")
        report.overall_disposition = "reject"
        return report

    missing = _REQUIRED_IMU_COLUMNS - set(df.columns)
    if missing:
        report.issues.append(f"IMU 缺少列: {sorted(missing)}")
        report.overall_disposition = "reject"
        return report

    report.sample_count = len(df)
    report.sample_rate_hz = getattr(imu_stream, "sample_rate_hz", 0.0)

    ts = df["timestamp_ns"].values.astype(np.int64)

    # ---- 1. 时间戳完整性 ----
    _check_timestamps(ts, report, gap_factor)

    # ---- 2. 尖峰检测 ----
    _detect_spikes(df, ts, report, spike_std_factor)

    # ---- 3. 冻结检测 ----
    _detect_freezes(df, ts, report, freeze_min)

    # ---- 4. 静止零偏 ----
    _estimate_static_bias(df, ts, report, static_window_s)

    # ---- 5. 饱和检查 ----
    _check_saturation(df, report)

    # ---- 聚合 ----
    if not report.timestamp_valid:
        report.overall_disposition = "keep_with_flag"
    if report.gap_count > 0:
        report.overall_disposition = "keep_with_flag"
    if report.spike_count > len(df) * 0.01:  # > 1% 尖峰
        report.issues.append(f"尖峰比例异常: {report.spike_count}/{report.sample_count}")
        report.overall_disposition = "keep_with_flag"
    if report.freeze_span_count > 0:
        report.issues.append(
            f"IMU 冻结: {report.freeze_span_count} 段, 共 {report.freeze_total_samples} 样本"
        )
        report.overall_disposition = "keep_with_flag"

    return report


# ---------------------------------------------------------------------------
# 1. 时间戳完整性
# ---------------------------------------------------------------------------


def _check_timestamps(
    ts: np.ndarray,
    report: DunjiaIMUReport,
    gap_factor: float,
) -> None:
    """检查时间戳单调性、重复、回退和 gap。"""
    # 去重检查
    unique_ts, counts = np.unique(ts, return_counts=True)
    dup_mask = counts > 1
    report.has_duplicates = bool(dup_mask.any())
    report.duplicate_count = int((counts[dup_mask] - 1).sum()) if report.has_duplicates else 0

    if report.has_duplicates:
        report.issues.append(f"IMU 时间戳重复: {report.duplicate_count} 个")

    # 回退检测（在原序列中检测，不去重）
    diffs = np.diff(ts)
    neg_mask = diffs < 0
    report.has_regression = bool(neg_mask.any())
    report.regression_count = int(neg_mask.sum())
    report.timestamp_valid = not report.has_regression

    if report.has_regression:
        report.issues.append(f"IMU 时间戳回退: {report.regression_count} 处")

    # gap 检测
    if len(ts) < 2:
        return

    # 正常间隔（去重后）
    unique_diffs = np.diff(unique_ts)
    if len(unique_diffs) == 0:
        return

    median_interval = float(np.median(unique_diffs))
    report.median_interval_ns = int(median_interval)
    report.expected_interval_ns = int(median_interval)

    if report.sample_rate_hz > 0:
        report.expected_interval_ns = int(1_000_000_000 / report.sample_rate_hz)
    elif median_interval > 0:
        report.expected_interval_ns = int(median_interval)

    threshold_ns = report.expected_interval_ns * gap_factor

    # 在去重序列上找 gap
    gap_indices = np.where(unique_diffs > threshold_ns)[0]
    for idx in gap_indices:
        gap_ns = int(unique_diffs[idx])
        report.gaps.append(
            IMUGapSpan(
                start_sample=int(idx),
                end_sample=int(idx + 1),
                start_timestamp_ns=int(unique_ts[idx]),
                end_timestamp_ns=int(unique_ts[idx + 1]),
                gap_ns=gap_ns,
                gap_s=gap_ns / 1_000_000_000,
                expected_interval_ns=report.expected_interval_ns,
                factor=gap_ns / report.expected_interval_ns,
            )
        )

    report.gap_count = len(report.gaps)
    report.total_gap_duration_s = sum(g.gap_s for g in report.gaps)

    if report.gap_count > 0:
        report.issues.append(
            f"IMU gap: {report.gap_count} 处, "
            f"最宽 {max(g.gap_s for g in report.gaps):.3f}s"
        )


# ---------------------------------------------------------------------------
# 2. 尖峰检测（MAD 方法）
# ---------------------------------------------------------------------------


def _detect_spikes(
    df: pd.DataFrame,
    ts: np.ndarray,
    report: DunjiaIMUReport,
    std_factor: float,
) -> None:
    """对加速度和角速度各轴做 MAD 尖峰检测。"""
    fields = {
        "ax": "accel_x_mps2",
        "ay": "accel_y_mps2",
        "az": "accel_z_mps2",
        "gx": "gyro_x_rps",
        "gy": "gyro_y_rps",
        "gz": "gyro_z_rps",
    }

    for field_name in fields:
        col = df[field_name].values.astype(np.float64)
        finite = np.isfinite(col)
        if not finite.any():
            continue

        valid = col[finite]
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        if mad == 0:
            continue

        deviations = np.abs(valid - median) / (mad + 1e-12)
        spike_mask = deviations > std_factor

        for local_idx in np.where(spike_mask)[0]:
            report.spikes.append(
                IMUSpikeEvent(
                    sample_index=int(local_idx),
                    timestamp_ns=int(ts[finite][local_idx]) if len(ts) > local_idx else 0,
                    field=field_name,
                    value=float(valid[local_idx]),
                    median=float(median),
                    mad=float(mad),
                    deviation_factor=float(deviations[local_idx]),
                )
            )

    report.spike_count = len(report.spikes)

    if report.spike_count > 0:
        by_field: dict[str, int] = {}
        for s in report.spikes:
            by_field[s.field] = by_field.get(s.field, 0) + 1
        report.issues.append(
            f"IMU 尖峰: {report.spike_count} 个, "
            f"分布={dict(sorted(by_field.items()))}"
        )


# ---------------------------------------------------------------------------
# 3. 冻结检测
# ---------------------------------------------------------------------------


def _detect_freezes(
    df: pd.DataFrame,
    ts: np.ndarray,
    report: DunjiaIMUReport,
    freeze_min: int,
) -> None:
    """检测连续完全相同样本（使用加速度和角速度联合检查）。"""
    cols = list(_REQUIRED_IMU_COLUMNS - {"timestamp_ns"})
    n = len(df)
    if n < freeze_min:
        return

    frozen = np.zeros(n, dtype=bool)
    for col_name in cols:
        col = df[col_name].values
        # 连续相同的标志
        col_frozen = np.zeros(n, dtype=bool)
        run_start = 0
        for i in range(1, n):
            if col[i] != col[i - 1]:
                run_len = i - run_start
                if run_len >= freeze_min:
                    col_frozen[run_start:i] = True
                run_start = i
        # 末尾
        run_len = n - run_start
        if run_len >= freeze_min:
            col_frozen[run_start:] = True

        frozen |= col_frozen

    # 合并连续冻结段
    in_span = False
    span_start = 0
    for i in range(n):
        if frozen[i] and not in_span:
            in_span = True
            span_start = i
        elif not frozen[i] and in_span:
            in_span = False
            report.freeze_total_samples += i - span_start
            report.freeze_span_count += 1
    if in_span:
        report.freeze_total_samples += n - span_start
        report.freeze_span_count += 1


# ---------------------------------------------------------------------------
# 4. 静止零偏估计
# ---------------------------------------------------------------------------


def _estimate_static_bias(
    df: pd.DataFrame,
    ts: np.ndarray,
    report: DunjiaIMUReport,
    window_s: float,
) -> None:
    """自动寻找 IMU 静止窗口并估计零偏。"""
    n = len(df)
    if n < 2:
        return

    # 加速度幅值
    accel_mag = np.sqrt(
        df["ax"].values ** 2 + df["ay"].values ** 2 + df["az"].values ** 2
    )
    # 角速度幅值
    gyro_mag = np.sqrt(
        df["gx"].values ** 2 + df["gy"].values ** 2 + df["gz"].values ** 2
    )

    # 静止条件: accel_mag 接近 9.81 & gyro_mag 接近 0
    is_static = (
        np.abs(accel_mag - 9.81) < _STATIC_ACCEL_TOLERANCE
    ) & (
        gyro_mag < _STATIC_GYRO_TOLERANCE
    )

    # 计算采样间隔
    if report.median_interval_ns > 0:
        interval_s = report.median_interval_ns / 1_000_000_000
    else:
        interval_s = 0.005  # 196Hz 默认

    min_samples = max(1, int(window_s / interval_s))

    # 找连续静止段
    in_static = False
    start = 0
    for i in range(n):
        if is_static[i] and not in_static:
            in_static = True
            start = i
        elif not is_static[i] and in_static:
            in_static = False
            duration = (i - start) * interval_s
            if i - start >= min_samples:
                seg = df.iloc[start:i]
                report.static_windows.append(
                    IMUStaticWindow(
                        start_sample=start,
                        end_sample=i,
                        duration_s=round(duration, 3),
                        accel_bias=(
                            float(seg["ax"].mean()),
                            float(seg["ay"].mean()),
                            float(seg["az"].mean()),
                        ),
                        gyro_bias=(
                            float(seg["gx"].mean()),
                            float(seg["gy"].mean()),
                            float(seg["gz"].mean()),
                        ),
                        accel_mag_mean=float(accel_mag[start:i].mean()),
                        gyro_mag_mean=float(gyro_mag[start:i].mean()),
                    )
                )
    # 末尾
    if in_static and n - start >= min_samples:
        seg = df.iloc[start:]
        report.static_windows.append(
            IMUStaticWindow(
                start_sample=start,
                end_sample=n,
                duration_s=round((n - start) * interval_s, 3),
                accel_bias=(
                    float(seg["ax"].mean()),
                    float(seg["ay"].mean()),
                    float(seg["az"].mean()),
                ),
                gyro_bias=(
                    float(seg["gx"].mean()),
                    float(seg["gy"].mean()),
                    float(seg["gz"].mean()),
                ),
                accel_mag_mean=float(accel_mag[start:].mean()),
                gyro_mag_mean=float(gyro_mag[start:].mean()),
            )
        )

    report.static_window_count = len(report.static_windows)


# ---------------------------------------------------------------------------
# 5. 饱和检查
# ---------------------------------------------------------------------------


def _check_saturation(
    df: pd.DataFrame,
    report: DunjiaIMUReport,
) -> None:
    """检查 IMU 饱和（仅在已知量程时判定）。"""
    accel_range = report.accel_range_mps2
    gyro_range = report.gyro_range_rps

    if accel_range is not None:
        accel_cols = ["ax", "ay", "az"]
        for col_name in accel_cols:
            col = df[col_name].values
            sat_mask = np.abs(col) >= accel_range * 0.98
            report.saturation_accel_count += int(sat_mask.sum())
        report.saturation_status = "checked"

        if report.saturation_accel_count > 0:
            report.issues.append(
                f"加速度计饱和: {report.saturation_accel_count} 样本 "
                f"(量程 ±{accel_range} m/s²)"
            )

    if gyro_range is not None:
        gyro_cols = ["gx", "gy", "gz"]
        for col_name in gyro_cols:
            col = df[col_name].values
            sat_mask = np.abs(col) >= gyro_range * 0.98
            report.saturation_gyro_count += int(sat_mask.sum())
        report.saturation_status = "checked"

        if report.saturation_gyro_count > 0:
            report.issues.append(
                f"陀螺仪饱和: {report.saturation_gyro_count} 样本 "
                f"(量程 ±{gyro_range} rad/s)"
            )

    if accel_range is None and gyro_range is None:
        report.saturation_status = "unavailable"
    elif accel_range is None or gyro_range is None:
        report.saturation_status = "partial"


__all__ = [
    "DunjiaIMUReport",
    "IMUGapSpan",
    "IMUSpikeEvent",
    "IMUStaticWindow",
    "check_dunjia_imu",
]
