"""
探测 A2D Episode 的 HDF5 文件（aligned_joints.h5 + raw_joints.h5）。

遍历所有 Dataset，记录 path / shape / dtype / attributes，
并做关键验证：
  - timestamp 单位与时钟
  - 所有 Dataset 第一维是否一致
  - state / action 的 DOF
  - 是否存在 NaN / Inf
  - aligned 与 raw 的差异

用法:
    python scripts/inspect_a2d_hdf5.py \\
        --aligned "E:/datasets/真机/A2D/aligned_joints.h5" \\
        --raw "E:/datasets/真机/A2D/record/raw_joints.h5" \\
        --output output/a2d/8032/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# 已知机器人关节名（用于与 joint_map 交叉验证）
KNOWN_JOINT_NAMES = [
    "joint31", "joint32", "joint33", "joint34",
    "joint51", "joint52", "joint53", "joint54",
    "joint55", "joint56", "joint57",
    "joint61", "joint62", "joint63", "joint64",
    "joint65", "joint66", "joint67",
]


# ---------------------------------------------------------------------------
# 核心探测
# ---------------------------------------------------------------------------

def inspect_hdf5(path: str) -> list[dict]:
    """遍历 HDF5 中所有 Dataset，返回结构化信息列表。

    Returns:
        [{path, shape, dtype, attributes, nan_count, inf_count}, ...]
    """
    result: list[dict] = []

    with h5py.File(path, "r") as file:

        def visitor(name: str, obj):
            if isinstance(obj, h5py.Dataset):
                shape = list(obj.shape)
                dtype = str(obj.dtype)
                attrs = {
                    str(key): _safe_str(value)
                    for key, value in obj.attrs.items()
                }

                # NaN / Inf 统计
                nan_count = 0
                inf_count = 0
                try:
                    arr = obj[:]
                    if arr.dtype.kind == "f":
                        nan_count = int(np.isnan(arr).sum())
                        inf_count = int(np.isinf(arr).sum())
                except Exception:
                    nan_count = -1
                    inf_count = -1

                result.append({
                    "path": name,
                    "shape": shape,
                    "dtype": dtype,
                    "attributes": attrs,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                })

        file.visititems(visitor)

    return result


def _safe_str(value) -> str:
    """将 HDF5 attribute 安全转为字符串（截断过长值）。"""
    s = str(value)
    if len(s) > 200:
        s = s[:200] + "..."
    return s


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------

def analyze(inspection: list[dict], joint_map: dict | None = None) -> dict:
    """基于 inspect 结果生成分析报告。

    Returns:
        {
            timestamp_unit: "ns" | "unknown",
            timestamp_clock: "unix_epoch" | "unknown",
            timestamp_status: "confirmed" | "needs_verification",
            aligned_timestamp_unit: ...,
            aligned_timestamp_clock: ...,
            robot_state_dof: int | None,
            robot_action_dof: int | None,
            gripper_dof: int | None,
            first_dim_consistent: bool,
            nan_summary: dict,
            warnings: [...],
        }
    """
    analysis: dict = {
        "timestamp_unit": None,
        "timestamp_clock": None,
        "timestamp_status": "needs_verification",
        "robot_state_dof": None,
        "robot_action_dof": None,
        "gripper_dof": None,
        "joint_map_cross_check": None,
        "first_dim_consistent": None,
        "first_dim_values": {},
        "nan_summary": {},
        "warnings": [],
    }

    by_path = {item["path"]: item for item in inspection}

    # ---- timestamp 单位判定 ----
    if "timestamp" in by_path:
        ts_info = by_path["timestamp"]
        # 用典型的 2020-2030 Unix epoch ns 范围判断：
        # 2020-01-01 00:00:00 UTC = 1,577,836,800,000,000,000 ns
        # 2030-01-01 00:00:00 UTC = 1,893,456,000,000,000,000 ns
        ts_first = ts_info.get("_first_value")
        ts_last = ts_info.get("_last_value")
        ts_min = ts_info.get("_min_value")
        ts_max = ts_info.get("_max_value")

        if ts_first is not None:
            if 1.5e18 < ts_first < 2.0e18:
                analysis["timestamp_unit"] = "ns"
                analysis["timestamp_clock"] = "unix_epoch"
                analysis["timestamp_status"] = "confirmed"
                analysis["timestamp_range_human"] = {
                    "first_utc": _ns_to_utc_str(int(ts_first)),
                    "last_utc": _ns_to_utc_str(int(ts_last)) if ts_last is not None else None,
                }
            elif 1.5e15 < ts_first < 2.0e15:
                analysis["timestamp_unit"] = "us"
                analysis["timestamp_status"] = "needs_verification"
            elif 1.5e12 < ts_first < 2.0e12:
                analysis["timestamp_unit"] = "ms"
                analysis["timestamp_status"] = "needs_verification"
            elif 1.5e9 < ts_first < 2.0e9:
                analysis["timestamp_unit"] = "s"
                analysis["timestamp_status"] = "needs_verification"
            else:
                analysis["warnings"].append(
                    f"timestamp 不在已知 Unix epoch 范围内: {ts_first}"
                )
    else:
        analysis["warnings"].append("未找到 timestamp dataset")

    # ---- DOF 推断 ----
    for prefix, key in [
        ("state/robot/", "robot_state_dof"),
        ("action/robot/", "robot_action_dof"),
        ("state/gripper/", "gripper_dof"),
    ]:
        for ds_path, info in by_path.items():
            if ds_path.startswith(prefix) and len(info["shape"]) == 2:
                dof = info["shape"][1]
                analysis[key] = dof
                break

    # ---- 第一维一致性 ----
    dims = {}
    for ds_path, info in by_path.items():
        if len(info["shape"]) >= 1:
            dims[ds_path] = info["shape"][0]

    analysis["first_dim_values"] = dims
    if dims:
        values = set(dims.values())
        analysis["first_dim_consistent"] = len(values) == 1

    # ---- NaN 汇总 ----
    nan_entries = {}
    for ds_path, info in by_path.items():
        if info["nan_count"] != 0:
            nan_entries[ds_path] = {
                "nan_count": info["nan_count"],
                "total_elements": (
                    info["shape"][0] * info["shape"][1]
                    if len(info["shape"]) == 2 else info["shape"][0]
                ),
                "nan_ratio": None,  # computed below
            }
    for ds_path, entry in nan_entries.items():
        total = entry["total_elements"]
        entry["nan_ratio"] = round(entry["nan_count"] / total, 6) if total > 0 else None

    analysis["nan_summary"] = nan_entries

    # ---- joint_map 交叉验证 ----
    if joint_map:
        # 检查 joint_map 值 0..17 是否对应 18 个关节
        mapped_indices = sorted(
            [v for v in joint_map.values() if v >= 0]
        )
        expected = list(range(len(mapped_indices)))
        analysis["joint_map_cross_check"] = {
            "num_joints": len(mapped_indices),
            "mapped_range": f"{min(mapped_indices)}–{max(mapped_indices)}",
            "matches_0_to_n": mapped_indices == expected,
        }

    return analysis


# ---------------------------------------------------------------------------
# 补充数值信息（需要实际读取数据）
# ---------------------------------------------------------------------------

def enrich_inspection(inspection: list[dict], h5_path: str) -> None:
    """向 inspection 条目补充 _first_value / _last_value / _min_value / _max_value。

    仅对 1-D timestamp 和关键 2-D 数组的前几行采样。
    """
    with h5py.File(h5_path, "r") as f:
        for item in inspection:
            ds_path = item["path"]
            try:
                ds = f[ds_path]
                arr = ds[:]

                if arr.ndim == 1 and len(arr) > 0:
                    item["_first_value"] = _safe_number(arr[0])
                    item["_last_value"] = _safe_number(arr[-1])
                    item["_min_value"] = _safe_number(arr.min())
                    item["_max_value"] = _safe_number(arr.max())
                    # mean delta for timestamp-like arrays
                    if len(arr) > 1 and "timestamp" in ds_path.lower():
                        diffs = np.diff(arr)
                        item["_mean_delta"] = _safe_number(diffs.mean())
                        item["_min_delta"] = _safe_number(diffs.min())
                        item["_max_delta"] = _safe_number(diffs.max())
                elif arr.ndim == 2:
                    item["_first_row"] = [
                        _safe_number(v) for v in arr[0, :5]
                    ]
                    item["_last_row"] = [
                        _safe_number(v) for v in arr[-1, :5]
                    ]
            except Exception:
                pass


def _safe_number(val) -> int | float:
    """将 numpy 数值转为 Python 原生类型。"""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _ns_to_utc_str(ns: int) -> str:
    """纳秒 Unix epoch → ISO 8601 UTC 字符串。"""
    try:
        dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return f"ns={ns}"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def probe(
    aligned_path: str,
    raw_path: str,
    joint_map_path: str | None = None,
) -> dict:
    """完整探测：读取两个 HDF5 文件 + 可选 joint_map，返回 schema 报告。"""
    report: dict = {
        "schema_version": "a2d_hdf5_schema.v1",
    }

    # --- joint_map ---
    joint_map: dict | None = None
    if joint_map_path:
        jp = Path(joint_map_path)
        if jp.is_file():
            with open(jp, encoding="utf-8") as f:
                joint_map = json.load(f)
            report["joint_map"] = {
                "source": joint_map_path,
                "entries": joint_map,
            }

    # --- aligned_joints.h5 ---
    print("探测 aligned_joints.h5 ...")
    aligned_inspect = inspect_hdf5(aligned_path)
    enrich_inspection(aligned_inspect, aligned_path)
    aligned_analysis = analyze(aligned_inspect, joint_map)

    report["aligned_joints"] = {
        "file": aligned_path,
        "datasets": aligned_inspect,
        "analysis": aligned_analysis,
        "dataset_count": len(aligned_inspect),
    }

    # --- raw_joints.h5 ---
    print("探测 raw_joints.h5 ...")
    raw_inspect = inspect_hdf5(raw_path)
    enrich_inspection(raw_inspect, raw_path)
    raw_analysis = analyze(raw_inspect, joint_map)

    report["raw_joints"] = {
        "file": raw_path,
        "datasets": raw_inspect,
        "analysis": raw_analysis,
        "dataset_count": len(raw_inspect),
    }

    # --- 对比 aligned vs raw ---
    aligned_paths = {item["path"] for item in aligned_inspect}
    raw_paths = {item["path"] for item in raw_inspect}

    report["comparison"] = {
        "common_paths": sorted(aligned_paths & raw_paths),
        "aligned_only": sorted(aligned_paths - raw_paths),
        "raw_only": sorted(raw_paths - aligned_paths),
        "aligned_unique_first_dim": (aligned_analysis.get("first_dim_consistent")),
        "raw_first_dim_consistent": (raw_analysis.get("first_dim_consistent")),
        "note": (
            "aligned_joints.h5: 所有 Dataset 采样到统一时间轴（common timestamp）。"
            "raw_joints.h5: 各子组独立时间轴（state/robot、action/robot、"
            "state/gripper、action/gripper 各不同频率）。"
        ),
    }

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="探测 A2D Episode 的 HDF5 文件 Schema",
    )
    parser.add_argument(
        "--aligned",
        required=True,
        help="aligned_joints.h5 路径",
    )
    parser.add_argument(
        "--raw",
        required=True,
        help="raw_joints.h5 路径",
    )
    parser.add_argument(
        "--joint-map",
        default=None,
        help="joint_map.json 路径（默认自动查找 parameters/meshes/joint_map.json）",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/a2d/",
        help="输出根目录",
    )
    parser.add_argument(
        "--episode-id",
        default="unknown",
        help="Episode ID（用于输出子目录命名）",
    )
    args = parser.parse_args()

    # 自动推断 joint_map 路径
    joint_map_path = args.joint_map
    if joint_map_path is None:
        # 从 aligned 路径推断 episode 根
        aligned_p = Path(args.aligned)
        # aligned_joints.h5 通常在 episode 根目录
        candidate = aligned_p.parent / "parameters" / "meshes" / "joint_map.json"
        if candidate.is_file():
            joint_map_path = str(candidate)
            print(f"自动发现 joint_map: {joint_map_path}")

    report = probe(
        aligned_path=args.aligned,
        raw_path=args.raw,
        joint_map_path=joint_map_path,
    )

    # 写入
    out_dir = Path(args.output) / args.episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hdf5_schema.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n写入: {out_path}")

    # 打印摘要
    aa = report["aligned_joints"]["analysis"]
    ra = report["raw_joints"]["analysis"]

    print()
    print("=" * 50)
    print("  HDF5 Schema 摘要")
    print("=" * 50)

    print(f"\n  Timestamp (aligned):")
    print(f"    单位:       {aa['timestamp_unit']}")
    print(f"    时钟:       {aa['timestamp_clock']}")
    print(f"    状态:       {aa['timestamp_status']}")
    if "timestamp_range_human" in aa:
        tr = aa["timestamp_range_human"]
        print(f"    首条 UTC:   {tr['first_utc']}")
        print(f"    末条 UTC:   {tr['last_utc']}")

    print(f"\n  DOF:")
    print(f"    robot_state:  {aa['robot_state_dof']}")
    print(f"    robot_action: {aa['robot_action_dof']}")
    print(f"    gripper:      {aa['gripper_dof']}")

    print(f"\n  Aligned:")
    print(f"    Dataset 数:        {report['aligned_joints']['dataset_count']}")
    print(f"    第一维一致:          {aa['first_dim_consistent']}")
    print(f"    共同长度:            {max(aa['first_dim_values'].values()) if aa['first_dim_values'] else 'N/A'}")

    print(f"\n  Raw:")
    print(f"    Dataset 数:        {report['raw_joints']['dataset_count']}")
    print(f"    第一维一致:          {ra['first_dim_consistent']}")
    if ra["first_dim_values"]:
        print(f"    各 Dataset 长度:")
        for k, v in sorted(ra["first_dim_values"].items()):
            print(f"      {k}: {v}")

    print(f"\n  Joint Map 交叉验证:")
    if aa.get("joint_map_cross_check"):
        jc = aa["joint_map_cross_check"]
        print(f"    关节数:        {jc['num_joints']}")
        print(f"    索引范围:      {jc['mapped_range']}")
        print(f"    匹配 0..n-1:   {jc['matches_0_to_n']}")

    nan_ds = {k: v for k, v in aa["nan_summary"].items() if v["nan_count"] > 0}
    if nan_ds:
        print(f"\n  NaN 分布 (aligned):")
        for ds_path, info in sorted(nan_ds.items()):
            print(f"    {ds_path}: {info['nan_count']}/{info['total_elements']} "
                  f"({info['nan_ratio']:.1%})")

    print(f"\n  Aligned vs Raw 差异:")
    comp = report["comparison"]
    print(f"    共同路径: {len(comp['common_paths'])}")
    print(f"    Aligned 独有: {comp['aligned_only']}")
    print(f"    Raw 独有:     {comp['raw_only']}")
    print(f"    {comp['note']}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
