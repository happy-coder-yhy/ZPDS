"""
A2D State Alignment Report — 验证 aligned_joints.h5 是否确实来自 raw_joints.h5。

不重复写 Robot State，而是审计 aligned 与 raw 的一致性：
  - 时间覆盖对比
  - 消息数量对比
  - 关节顺序验证
  - 状态值统计分布对比
  - Action 值统计分布对比
  - Command-State 延迟测量

用法:
    from segment.a2d_state_alignment import generate_alignment_report

    report = generate_alignment_report(
        aligned_path="E:/datasets/真机/A2D/aligned_joints.h5",
        raw_path="E:/datasets/真机/A2D/record/raw_joints.h5",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# 已知关节名（用于交叉验证）
KNOWN_JOINT_NAMES = [
    "joint31", "joint32", "joint33", "joint34",
    "joint51", "joint52", "joint53", "joint54",
    "joint55", "joint56", "joint57",
    "joint61", "joint62", "joint63", "joint64",
    "joint65", "joint66", "joint67",
]

# robot state/action 中参与比较的字段组
STATE_FIELDS = ["positions", "velocities", "efforts", "temperatures"]
ACTION_FIELDS = ["positions", "velocities", "accelerations",
                 "decelerations", "efforts", "torque_rates"]


# ======================================================================
# 主入口
# ======================================================================

def generate_alignment_report(
    aligned_path: str,
    raw_path: str,
) -> dict:
    """生成 aligned vs raw 对齐审计报告。

    Args:
        aligned_path: aligned_joints.h5 路径。
        raw_path: record/raw_joints.h5 路径。

    Returns:
        state_alignment_report dict。
    """
    aligned = Path(aligned_path)
    raw = Path(raw_path)

    if not aligned.is_file():
        return {"error": f"aligned_joints.h5 不存在: {aligned_path}"}
    if not raw.is_file():
        return {"error": f"raw_joints.h5 不存在: {raw_path}"}

    report: dict = {
        "report_type": "state_alignment",
        "schema_version": "a2d_state_alignment.v1",
        "compared": {
            "aligned": str(aligned),
            "raw": str(raw),
        },
    }

    with h5py.File(str(aligned), "r") as af, h5py.File(str(raw), "r") as rf:
        # ---- 1. 时间覆盖 ----
        report["time_coverage"] = _compare_time_coverage(af, rf)

        # ---- 2. 消息数量 ----
        report["message_counts"] = _compare_message_counts(af, rf)

        # ---- 3. 关节顺序 ----
        report["joint_order"] = _check_joint_order(af, rf)

        # ---- 4. 状态值差异 ----
        report["state_value_differences"] = _compare_value_stats(
            af, rf, "state/robot", STATE_FIELDS,
        )

        # ---- 5. Action 值差异 ----
        report["action_value_differences"] = _compare_value_stats(
            af, rf, "action/robot", ACTION_FIELDS,
        )

        # ---- 6. Command-State 延迟 ----
        report["command_state_delay"] = _measure_delay(rf)

        # ---- 7. 结论 ----
        report["conclusion"] = _derive_conclusion(report)

    return report


# ======================================================================
# 1. 时间覆盖
# ======================================================================

def _compare_time_coverage(af: h5py.File, rf: h5py.File) -> dict:
    """对比 aligned 与 raw 各子组的时间覆盖。"""
    result: dict = {}

    # aligned 统一时间轴
    if "timestamp" in af:
        ats = af["timestamp"][:]
        result["aligned"] = {
            "start_ns": int(ats[0]),
            "end_ns": int(ats[-1]),
            "duration_s": round(float(ats[-1] - ats[0]) / 1e9, 3),
            "samples": len(ats),
        }

    # raw 各子组独立时间轴
    raw_groups = {}
    for subgroup in ["state/robot", "action/robot",
                     "state/gripper", "action/gripper"]:
        ts_path = f"{subgroup}/timestamp"
        if ts_path in rf:
            rts = rf[ts_path][:]
            raw_groups[subgroup.replace("/", "_")] = {
                "start_ns": int(rts[0]),
                "end_ns": int(rts[-1]),
                "duration_s": round(float(rts[-1] - rts[0]) / 1e9, 3),
                "samples": len(rts),
            }
            # overlap with aligned
            if "timestamp" in af:
                ats = af["timestamp"][:]
                overlap_start = max(int(ats[0]), int(rts[0]))
                overlap_end = min(int(ats[-1]), int(rts[-1]))
                raw_groups[subgroup.replace("/", "_")]["overlap_with_aligned_s"] = (
                    round((overlap_end - overlap_start) / 1e9, 3)
                    if overlap_end > overlap_start else 0.0
                )
    result["raw"] = raw_groups

    return result


# ======================================================================
# 2. 消息数量
# ======================================================================

def _compare_message_counts(af: h5py.File, rf: h5py.File) -> dict:
    """对比 aligned 与 raw 各字段组的样本数。"""
    counts: dict = {"aligned": {}, "raw": {}}

    for subgroup in ["state/robot", "action/robot"]:
        for field in (STATE_FIELDS if "state" in subgroup else ACTION_FIELDS):
            ds_path = f"{subgroup}/{field}"
            key = f"{subgroup.replace('/', '_')}_{field}"

            if ds_path in af:
                counts["aligned"][key] = af[ds_path].shape[0]
            if ds_path in rf:
                counts["raw"][key] = rf[ds_path].shape[0]

    # 压缩比
    counts["resample_ratios"] = {}
    for subgroup in ["state_robot", "action_robot"]:
        a_key = f"{subgroup}_positions"
        r_key = a_key
        if a_key in counts["aligned"] and r_key in counts["raw"]:
            ratio = counts["raw"][r_key] / counts["aligned"][a_key]
            counts["resample_ratios"][subgroup] = round(ratio, 2)

    return counts


# ======================================================================
# 3. 关节顺序
# ======================================================================

def _check_joint_order(af: h5py.File, rf: h5py.File) -> dict:
    """验证 aligned 与 raw 的关节维度顺序一致。"""
    # 通过比较 DOF 列数验证
    result: dict = {"consistent": True, "checks": {}}

    for subgroup in ["state/robot", "action/robot"]:
        for field in (STATE_FIELDS if "state" in subgroup else ACTION_FIELDS):
            ds_path = f"{subgroup}/{field}"
            key = f"{subgroup.replace('/', '_')}_{field}"

            a_dof = af[ds_path].shape[1] if ds_path in af else None
            r_dof = rf[ds_path].shape[1] if ds_path in rf else None

            ok = (a_dof == r_dof) if (a_dof is not None and r_dof is not None) else None
            result["checks"][key] = {
                "aligned_dof": a_dof,
                "raw_dof": r_dof,
                "match": ok,
            }
            if ok is False:
                result["consistent"] = False

    result["expected_joints"] = KNOWN_JOINT_NAMES
    result["expected_count"] = len(KNOWN_JOINT_NAMES)

    return result


# ======================================================================
# 4 & 5. 值统计对比
# ======================================================================

def _compare_value_stats(
    af: h5py.File,
    rf: h5py.File,
    subgroup: str,
    field_names: list[str],
) -> dict:
    """按 joint 对比 aligned 与 raw 的统计分布（均值、标准差、极值）。

    由于 aligned 与 raw 采样率不同，不进行逐行对比，
    而是比较每个 joint 的分布统计量。
    """
    result: dict = {}
    short_name = subgroup.replace("/", "_")

    for field in field_names:
        ds_path = f"{subgroup}/{field}"
        if ds_path not in af or ds_path not in rf:
            continue

        a_arr = af[ds_path][:]  # (N_aligned, DOF)
        r_arr = rf[ds_path][:]  # (N_raw, DOF)

        # 跳过全 NaN 字段（如 action 的 velocities/efforts）
        if np.all(np.isnan(a_arr)) or np.all(np.isnan(r_arr)):
            result[field] = {
                "dof": a_arr.shape[1],
                "skipped": "all_nan",
                "aligned_nan_ratio": 1.0,
                "raw_nan_ratio": 1.0,
            }
            continue

        # NaN-safe 统计
        a_mean = np.nanmean(a_arr, axis=0)
        r_mean = np.nanmean(r_arr, axis=0)

        dof = a_arr.shape[1]
        per_joint = []
        abs_diffs = []

        for j in range(dof):
            diff_mean = abs(a_mean[j] - r_mean[j]) if not (np.isnan(a_mean[j]) or np.isnan(r_mean[j])) else float("nan")

            if not np.isnan(diff_mean):
                abs_diffs.append(diff_mean)

            joint_name = KNOWN_JOINT_NAMES[j] if j < len(KNOWN_JOINT_NAMES) else f"joint_{j}"
            per_joint.append({
                "joint": joint_name,
                "aligned_mean": round(float(a_mean[j]), 6) if not np.isnan(a_mean[j]) else None,
                "raw_mean": round(float(r_mean[j]), 6) if not np.isnan(r_mean[j]) else None,
                "mean_abs_diff": round(float(diff_mean), 6) if not np.isnan(diff_mean) else None,
            })

        result[field] = {
            "dof": dof,
            "max_abs_mean_diff": round(float(max(abs_diffs)), 6) if abs_diffs else None,
            "mean_abs_mean_diff": round(float(np.mean(abs_diffs)), 6) if abs_diffs else None,
            "aligned_nan_ratio": round(float(np.isnan(a_arr).sum()) / a_arr.size, 6),
            "raw_nan_ratio": round(float(np.isnan(r_arr).sum()) / r_arr.size, 6),
            "per_joint": per_joint[:5] + ["..."] if len(per_joint) > 5 else per_joint,
        }

    return result


# ======================================================================
# 6. Command-State 延迟
# ======================================================================

def _measure_delay(rf: h5py.File) -> dict:
    """在 raw_joints.h5 中测量 action 指令到 state 观测的延迟。

    对于每个 action 时间戳，找到最近的 state 时间戳，
    计算 delay = state_ts - action_ts（正值 = action 先于 state）。
    """
    if "action/robot/timestamp" not in rf or "state/robot/timestamp" not in rf:
        return {"error": "raw 数据中缺少 action/robot/timestamp 或 state/robot/timestamp"}

    action_ts = rf["action/robot/timestamp"][:]
    state_ts = rf["state/robot/timestamp"][:]

    # 对于每个 action 时间，找最近的 state 时间
    delays_ns = []
    for at in action_ts:
        idx = np.argmin(np.abs(state_ts - at))
        delay = int(state_ts[idx]) - int(at)
        delays_ns.append(delay)

    delays = np.array(delays_ns, dtype=np.float64)
    abs_delays = np.abs(delays)

    return {
        "method": "nearest_state_to_action_raw",
        "measurement": "state_timestamp - action_timestamp",
        "positive_means_state_after_action": True,
        "count": len(delays),
        "mean_ns": round(float(np.mean(delays)), 1),
        "median_ns": round(float(np.median(delays)), 1),
        "p95_ns": round(float(np.percentile(abs_delays, 95)), 1),
        "p99_ns": round(float(np.percentile(abs_delays, 99)), 1),
        "max_abs_ns": round(float(np.max(abs_delays)), 1),
        "mean_abs_s": round(float(np.mean(abs_delays)) / 1e9, 6),
    }


# ======================================================================
# 7. 结论
# ======================================================================

def _derive_conclusion(report: dict) -> str:
    """基于所有检查结果推导结论。"""
    issues = []

    # 检查时间覆盖
    tc = report.get("time_coverage", {})
    raw_groups = tc.get("raw", {})
    for group_name, info in raw_groups.items():
        if info.get("overlap_with_aligned_s", 0) <= 0:
            issues.append(f"{group_name} 与 aligned 无时间重叠")

    # 检查关节顺序
    jo = report.get("joint_order", {})
    if not jo.get("consistent", True):
        issues.append("关节维度不一致")

    # 检查值差异（使用领域合理阈值）
    # positions: > 0.1 rad (~5.7°) 均值偏差视为显著
    # velocities: > 0.05 rad/s
    # efforts: > 100 N·m（力矩噪声大，阈值放宽）
    # temperatures: > 1°C
    thresholds = {
        "positions": 0.1,
        "velocities": 0.05,
        "efforts": 100.0,
        "temperatures": 1.0,
        "accelerations": 0.5,
        "decelerations": 0.5,
        "torque_rates": 50.0,
    }

    for section in ["state_value_differences", "action_value_differences"]:
        diffs = report.get(section, {})
        for field, info in diffs.items():
            if info.get("skipped"):
                continue  # 全 NaN 字段跳过
            max_diff = info.get("max_abs_mean_diff")
            threshold = thresholds.get(field, 0.1)
            if max_diff is not None and max_diff > threshold:
                issues.append(
                    f"{section}.{field}: max_abs_mean_diff={max_diff:.4f} "
                    f"(threshold={threshold})"
                )

    if not issues:
        return "aligned_consistent_with_raw"
    elif len(issues) <= 2:
        return f"minor_discrepancies: {'; '.join(issues)}"
    else:
        return f"significant_discrepancies: {'; '.join(issues[:3])}... ({len(issues)} total)"


# ======================================================================
# 写出
# ======================================================================

def write_alignment_report(report: dict, output_dir: str) -> str:
    """写出 state_alignment_report.json。

    Returns:
        输出文件路径。
    """
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "state_alignment_report.json"

    # 自定义 JSON encoder 处理 numpy 类型
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                if np.isnan(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    return str(out_path)


__all__ = [
    "generate_alignment_report",
    "write_alignment_report",
]
