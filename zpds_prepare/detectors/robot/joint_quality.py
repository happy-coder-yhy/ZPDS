"""
Joint State 质量检查。

针对 robot_state / gripper_state 流：
  - position 是否为有限值
  - velocity 极值
  - 温度是否缺失 / 异常
  - 信号是否长时间冻结
  - error_code 检查（如可用）
"""

from __future__ import annotations

import numpy as np

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.readers.session_model import TimeSeriesStream

# 默认配置
DEFAULT_FREEZE_MIN_DURATION_S = 2.0
DEFAULT_FREEZE_EPSILON = 1.0e-6
DEFAULT_MAX_ABS_VELOCITY_ENABLED = False


def detect_joint_quality(
    ts_stream: TimeSeriesStream,
    config: dict | None = None,
) -> list[QualityIssue]:
    """检查关节状态流的质量。

    Args:
        ts_stream: robot_state 或 gripper_state TimeSeriesStream。
        config: 配置字典，支持:
            - freeze.min_duration_s: 冻结检测最短持续秒数 (默认 2.0)
            - freeze.epsilon: 变化阈值 (默认 1.0e-6)
            - max_abs_velocity.enabled: 是否启用速度极值检查 (默认 false)

    Returns:
        QualityIssue 列表。
    """
    if config is None:
        config = {}

    freeze_cfg = config.get("freeze", {})
    freeze_min_s = float(freeze_cfg.get("min_duration_s", DEFAULT_FREEZE_MIN_DURATION_S))
    freeze_eps = float(freeze_cfg.get("epsilon", DEFAULT_FREEZE_EPSILON))
    velocity_check = config.get("max_abs_velocity", {}).get("enabled", DEFAULT_MAX_ABS_VELOCITY_ENABLED)

    issues: list[QualityIssue] = []
    stream_id = ts_stream.stream_id
    timestamps = np.array(ts_stream.timestamps_ns, dtype=np.int64)
    if hasattr(ts_stream.rows, "select_dtypes"):
        numeric_frame = ts_stream.rows.select_dtypes(include=[np.number])
        rows = (
            numeric_frame.to_numpy(dtype=np.float64)
            if not numeric_frame.empty
            else np.zeros((len(ts_stream.rows), 0), dtype=np.float64)
        )
    else:
        rows = np.asarray(ts_stream.rows, dtype=np.float64)
    fields = ts_stream.fields
    num_samples = len(timestamps)

    if num_samples == 0 or rows.size == 0:
        return issues

    # 按字段名分类索引
    position_indices = [i for i, f in enumerate(fields) if "positions" in f["name"]]
    velocity_indices = [i for i, f in enumerate(fields) if "velocities" in f["name"]]
    temperature_indices = [i for i, f in enumerate(fields) if "temperatures" in f["name"]]

    # ---- 1. position 有限值检查 ----
    for col_idx in position_indices:
        col = rows[:, col_idx]
        finite_mask = np.isfinite(col)
        non_finite = ~finite_mask

        if non_finite.any():
            nf_count = int(non_finite.sum())
            first_idx = int(np.where(non_finite)[0][0])
            issues.append(QualityIssue(
                issue_type="joint_position_non_finite",
                stream_id=stream_id,
                start_ns=int(timestamps[first_idx]),
                end_ns=int(timestamps[first_idx]),
                severity="warning",
                decision="keep_with_flag",
                details={
                    "field": fields[col_idx]["name"],
                    "non_finite_count": nf_count,
                    "first_index": first_idx,
                    "check": "position_finite",
                },
            ))

    # ---- 2. 温度检查 ----
    for col_idx in temperature_indices:
        col = rows[:, col_idx]
        finite = col[np.isfinite(col)]

        if len(finite) > 0:
            # 温度范围合理性 (0°C – 150°C 保守范围)
            too_low = finite < 0
            too_high = finite > 150

            if too_low.any() or too_high.any():
                extreme_count = int(too_low.sum() + too_high.sum())
                issues.append(QualityIssue(
                    issue_type="temperature_out_of_range",
                    stream_id=stream_id,
                    start_ns=int(timestamps[0]),
                    end_ns=int(timestamps[-1]),
                    severity="warning",
                    decision="keep_with_flag",
                    details={
                        "field": fields[col_idx]["name"],
                        "out_of_range_count": extreme_count,
                        "min_value": float(finite.min()),
                        "max_value": float(finite.max()),
                        "check": "temperature_range",
                    },
                ))

            # 温度全 NaN
            nan_ratio = (len(col) - len(finite)) / max(len(col), 1)
            if nan_ratio > 0.9:
                issues.append(QualityIssue(
                    issue_type="temperature_excessive_nan",
                    stream_id=stream_id,
                    start_ns=int(timestamps[0]),
                    end_ns=int(timestamps[-1]),
                    severity="warning",
                    decision="keep_with_flag",
                    details={
                        "field": fields[col_idx]["name"],
                        "nan_ratio": round(nan_ratio, 4),
                        "check": "temperature_nan_ratio",
                    },
                ))

    # ---- 3. 速度极值 (默认关闭) ----
    if velocity_check:
        for col_idx in velocity_indices:
            col = rows[:, col_idx]
            finite = col[np.isfinite(col)]
            if len(finite) > 0:
                abs_max = float(np.max(np.abs(finite)))
                # 标记 > 20 rad/s 的值
                if abs_max > 20.0:
                    extreme = np.abs(col) > 20.0
                    first_idx = int(np.where(extreme)[0][0]) if extreme.any() else 0
                    issues.append(QualityIssue(
                        issue_type="joint_velocity_extreme",
                        stream_id=stream_id,
                        start_ns=int(timestamps[first_idx]),
                        end_ns=int(timestamps[first_idx]),
                        severity="warning",
                        decision="keep_with_flag",
                        details={
                            "field": fields[col_idx]["name"],
                            "max_abs_velocity_rad_s": round(abs_max, 2),
                            "extreme_count": int(extreme.sum()),
                            "threshold_rad_s": 20.0,
                            "check": "velocity_extreme",
                        },
                    ))

    # ---- 4. 信号冻结检测 ----
    if len(position_indices) > 0 and num_samples >= 2:
        freeze_min_samples = max(2, int(freeze_min_s * (ts_stream.expected_rate_hz or 30)))
        for col_idx in position_indices:
            col = rows[:, col_idx]
            finite = np.isfinite(col)
            if not finite.any():
                continue

            # 检测连续不变区域
            changes = np.abs(np.diff(col)) > freeze_eps
            # 找出连续不变的 run
            run_start = None
            for i in range(len(changes)):
                if not changes[i]:
                    if run_start is None:
                        run_start = i
                else:
                    if run_start is not None:
                        run_len = i - run_start + 1
                        if run_len >= freeze_min_samples:
                            start_ns = int(timestamps[run_start])
                            end_ns = int(timestamps[min(i, num_samples - 1)])
                            duration_s = (end_ns - start_ns) / 1e9
                            issues.append(QualityIssue(
                                issue_type="joint_signal_frozen",
                                stream_id=stream_id,
                                start_ns=start_ns,
                                end_ns=end_ns,
                                severity="warning",
                                decision="keep_with_flag",
                                details={
                                    "field": fields[col_idx]["name"],
                                    "frozen_samples": run_len,
                                    "duration_s": round(duration_s, 2),
                                    "value": round(float(col[run_start]), 6),
                                    "check": "signal_frozen",
                                },
                            ))
                        run_start = None

            # 尾部冻结
            if run_start is not None and len(changes) >= run_start:
                run_len = len(changes) - run_start + 1
                if run_len >= freeze_min_samples:
                    start_ns = int(timestamps[run_start])
                    end_ns = int(timestamps[-1])
                    duration_s = (end_ns - start_ns) / 1e9
                    issues.append(QualityIssue(
                        issue_type="joint_signal_frozen",
                        stream_id=stream_id,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        severity="warning",
                        decision="keep_with_flag",
                        details={
                            "field": fields[col_idx]["name"],
                            "frozen_samples": run_len,
                            "duration_s": round(duration_s, 2),
                            "value": round(float(col[run_start]), 6),
                            "check": "signal_frozen",
                        },
                    ))

    return issues


__all__ = ["detect_joint_quality"]
