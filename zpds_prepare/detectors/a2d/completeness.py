"""B6: A2D 完整性矩阵。

检查 A2D Episode 全部资产的完整性，输出交叉核验矩阵。

检查项：
  1. meta_info.json — 存在性、可解析性、关键字段完整性
  2. 6 路相机（head/left/right × rgb/depth）— 帧目录、帧文件、帧数一致性
  3. aligned_joints.h5 — 存在性、dataset 完整性、shape 一致性
  4. raw_joints.h5 — 可选，记录存在性和基本结构
  5. 标定文件 — parameters/camera/*_intrinsic_params.json
  6. joint_map.json — parameters/meshes/joint_map.json
  7. ROS2 MCAP — 存在性（可选）
  8. 设备日志 — 存在性（可选，若存在记录路径）
  9. Review 标注 — 存在性（可选）
  10. 交叉验证 — meta duration vs camera frames vs HDF5 timestamps

原则：
  - 不按行号配对——所有映射记录方法、误差和不确定度
  - 缺失必需资产 → reject 对应视图
  - 可选资产缺失 → keep_with_flag
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# ---- A2D 预期文件清单 ----

# 必需：缺一不可
_REQUIRED_ASSETS = frozenset({
    "meta_info.json",
    "head_rgb",
    "hand_left_rgb",
    "hand_right_rgb",
    "aligned_joints.h5",
    "camera_calibration",
})

# 可选：缺失标记但不阻断
_OPTIONAL_ASSETS = frozenset({
    "head_depth",
    "hand_left_depth",
    "hand_right_depth",
    "raw_joints.h5",
    "joint_map.json",
    "mcap",
    "device_logs",
    "review_annotations",
})

# HDF5 必需 dataset（aligned_joints.h5 内部）
_REQUIRED_HDF5_DATASETS = frozenset({
    "timestamp",
    "state/robot/positions",
    "state/robot/velocities",
    "state/robot/efforts",
    "state/gripper/positions",
    "action/robot/positions",
    "action/gripper/positions",
})

# 深度帧文件名
_DEPTH_FILES = {
    "head_depth": "head_depth.png",
    "hand_left_depth": "hand_left_depth.png",
    "hand_right_depth": "hand_right_depth.png",
}

# RGB 帧文件名
_RGB_FILES = {
    "head_rgb": "head_color.jpg",
    "hand_left_rgb": "hand_left_color.jpg",
    "hand_right_rgb": "hand_right_color.jpg",
}

# 标定文件
_CALIB_FILES = {
    "head_rgb": "head_intrinsic_params.json",
    "hand_left_rgb": "hand_left_intrinsic_params.json",
    "hand_right_rgb": "hand_right_intrinsic_params.json",
}


# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------


@dataclass
class HDF5DatasetStatus:
    """单个 HDF5 dataset 的状态。"""

    path: str
    present: bool
    shape: tuple[int, ...] | None = None
    dtype: str = ""
    issues: list[str] = field(default_factory=list)


@dataclass
class A2DAssetStatus:
    """单个资产的状态。"""

    asset_id: str
    asset_type: str  # "metadata" | "camera_rgb" | "camera_depth" | "hdf5" | "calibration" | "mcap" | "log" | "annotation"
    required: bool
    present: bool
    frame_count: int = 0
    width: int = 0
    height: int = 0
    issues: list[str] = field(default_factory=list)
    disposition: str = "pass"  # "pass" | "keep_with_flag" | "reject"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class A2DCompletenessReport:
    """A2D Episode 完整性矩阵。

    一次性核验全部 8 类资产，交叉验证 duration/clip/frame_count，
    并记录所有矛盾和不确定度。
    """

    episode_id: str
    source_path: str
    source_sha256: str
    schema_version: str = "zpds.a2d_completeness.v1"

    # 资产列表
    assets: dict[str, A2DAssetStatus] = field(default_factory=dict)

    # HDF5 内部 dataset
    hdf5_datasets: list[HDF5DatasetStatus] = field(default_factory=list)
    hdf5_sample_count: int = 0
    hdf5_timestamps_valid: bool = False
    hdf5_timestamp_start_ns: int = 0
    hdf5_timestamp_end_ns: int = 0

    # 交叉验证
    cross_validation: dict[str, Any] = field(default_factory=dict)

    # 聚合
    overall_disposition: str = "pass"
    required_present: int = 0
    required_total: int = 0
    optional_present: int = 0
    optional_total: int = 0

    @property
    def all_required_present(self) -> bool:
        return self.required_present == self.required_total


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def check_a2d_completeness(
    episode_root: str | Path,
) -> A2DCompletenessReport:
    """检查 A2D Episode 全部资产的完整性。

    Args:
        episode_root: Episode 根目录路径（含 meta_info.json / camera/ / aligned_joints.h5 等）

    Returns:
        A2DCompletenessReport 包含每资产状态、HDF5 内容、
        交叉核验和聚合 disposition。
    """
    root = Path(episode_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Episode 目录不存在: {root}")

    report = A2DCompletenessReport(
        episode_id=root.name,
        source_path=str(root),
        source_sha256=_dir_sha256(root),
    )

    # ---- 1. meta_info.json ----
    report.assets["meta_info.json"] = _check_meta_info(root)

    # ---- 2. 6 路相机 ----
    camera_root = root / "camera"
    frame_dirs = _scan_frame_dirs(camera_root)
    sorted_indices = sorted(frame_dirs.keys())

    for asset_id, filename in _RGB_FILES.items():
        report.assets[asset_id] = _check_camera_images(
            camera_root, frame_dirs, sorted_indices, asset_id, filename,
        )

    for asset_id, filename in _DEPTH_FILES.items():
        report.assets[asset_id] = _check_camera_images(
            camera_root, frame_dirs, sorted_indices, asset_id, filename,
        )

    # ---- 3. aligned_joints.h5 ----
    h5_path = root / "aligned_joints.h5"
    _check_aligned_h5(h5_path, report)

    # ---- 4. raw_joints.h5 ----
    report.assets["raw_joints.h5"] = _check_raw_h5(root)

    # ---- 5. 标定 ----
    report.assets["camera_calibration"] = _check_a2d_calibration(root)

    # ---- 6. joint_map ----
    report.assets["joint_map.json"] = _check_joint_map(root)

    # ---- 7. MCAP ----
    report.assets["mcap"] = _check_mcap(root)

    # ---- 8. 设备日志 ----
    report.assets["device_logs"] = _check_device_logs(root)

    # ---- 9. Review 标注 ----
    report.assets["review_annotations"] = _check_review_annotations(root)

    # ---- 10. 交叉验证 ----
    report.cross_validation = _cross_validate(report)

    # ---- 聚合 ----
    _aggregate(report)

    return report


# ---------------------------------------------------------------------------
# 各资产检查
# ---------------------------------------------------------------------------


def _check_meta_info(root: Path) -> A2DAssetStatus:
    """检查 meta_info.json。"""
    asset = A2DAssetStatus(
        asset_id="meta_info.json",
        asset_type="metadata",
        required=True,
        present=False,
    )
    path = root / "meta_info.json"
    if not path.is_file():
        asset.issues.append("meta_info.json 不存在")
        asset.disposition = "reject"
        return asset

    asset.present = True
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        asset.issues.append(f"meta_info.json 解析失败: {exc}")
        asset.disposition = "reject"
        return asset

    if not isinstance(raw, dict):
        asset.issues.append("meta_info.json 顶层不是对象")
        asset.disposition = "reject"
        return asset

    # 关键字段检查
    key_fields = ["episode_id", "duration", "camera_list", "robot_type"]
    missing = [k for k in key_fields if k not in raw]
    if missing:
        asset.issues.append(f"缺少关键字段: {missing}")
        asset.disposition = "keep_with_flag"

    asset.details = {
        "episode_id": raw.get("episode_id", ""),
        "duration_s": raw.get("duration"),
        "camera_list": raw.get("camera_list", []),
        "robot_type": raw.get("robot_type", ""),
        "is_aligned": raw.get("is_aligned", False),
        "integrity": raw.get("integrity", ""),
    }
    return asset


def _scan_frame_dirs(camera_root: Path) -> dict[int, Path]:
    """扫描 camera/ 目录。"""
    frame_dirs: dict[int, Path] = {}
    if not camera_root.is_dir():
        return frame_dirs
    for d in camera_root.iterdir():
        if not d.is_dir():
            continue
        try:
            idx = int(d.name)
        except ValueError:
            continue
        frame_dirs[idx] = d
    return frame_dirs


def _check_camera_images(
    camera_root: Path,
    frame_dirs: dict[int, Path],
    sorted_indices: list[int],
    asset_id: str,
    filename: str,
) -> A2DAssetStatus:
    """检查单路相机图像序列。"""
    is_rgb = "rgb" in asset_id
    asset_type = "camera_rgb" if is_rgb else "camera_depth"
    required = asset_id in _REQUIRED_ASSETS

    asset = A2DAssetStatus(
        asset_id=asset_id,
        asset_type=asset_type,
        required=required,
        present=False,
    )

    if not sorted_indices:
        asset.issues.append(f"camera/ 目录无可解析帧目录")
        asset.disposition = "reject" if required else "keep_with_flag"
        return asset

    present_indices: list[int] = []
    for idx in sorted_indices:
        if (frame_dirs[idx] / filename).is_file():
            present_indices.append(idx)

    asset.present = len(present_indices) > 0
    asset.frame_count = len(present_indices)

    if not asset.present:
        asset.issues.append(f"未找到任何 {filename} 帧文件")
        asset.disposition = "reject" if required else "keep_with_flag"
        return asset

    # 记录帧范围
    asset.details = {
        "frame_range": (
            (present_indices[0], present_indices[-1])
            if present_indices else None
        ),
        "missing_frames": len(sorted_indices) - len(present_indices),
        "total_dirs": len(sorted_indices),
        "expected_filename": filename,
    }

    # 深度流可选
    if not required and asset.present:
        asset.disposition = "pass"

    return asset


def _check_aligned_h5(
    h5_path: Path, report: A2DCompletenessReport,
) -> None:
    """检查 aligned_joints.h5 及其内部 dataset。"""
    asset = A2DAssetStatus(
        asset_id="aligned_joints.h5",
        asset_type="hdf5",
        required=True,
        present=False,
    )
    report.assets["aligned_joints.h5"] = asset

    if not h5_path.is_file():
        asset.issues.append("aligned_joints.h5 不存在")
        asset.disposition = "reject"
        return

    asset.present = True
    try:
        f = h5py.File(h5_path, "r")
    except Exception as exc:
        asset.issues.append(f"aligned_joints.h5 无法打开: {exc}")
        asset.disposition = "reject"
        return

    try:
        # 检查所有必需 dataset
        datasets: list[HDF5DatasetStatus] = []
        sample_counts: set[int] = set()

        for ds_path in sorted(_REQUIRED_HDF5_DATASETS):
            ds = HDF5DatasetStatus(path=ds_path, present=False)
            if ds_path in f:
                dset = f[ds_path]
                ds.present = True
                ds.shape = tuple(dset.shape)
                ds.dtype = str(dset.dtype)
                sample_counts.add(dset.shape[0])
            else:
                ds.issues.append(f"dataset 缺失: {ds_path}")
            datasets.append(ds)

        # 时间戳特殊处理
        ts_ds = HDF5DatasetStatus(path="timestamp", present=False)
        if "timestamp" in f:
            ts_arr = f["timestamp"][:]
            ts_ds.present = True
            ts_ds.shape = tuple(ts_arr.shape)
            ts_ds.dtype = str(ts_arr.dtype)
            report.hdf5_sample_count = len(ts_arr)
            sample_counts.add(len(ts_arr))

            # 时间戳有效性
            ts_diff = np.diff(ts_arr)
            report.hdf5_timestamps_valid = bool(np.all(ts_diff > 0))
            if not report.hdf5_timestamps_valid:
                issues = []
                neg_mask = ts_diff <= 0
                if neg_mask.any():
                    issues.append(
                        f"{neg_mask.sum()} 处非递增时间戳"
                    )
                ds.issues = issues

            report.hdf5_timestamp_start_ns = int(ts_arr[0])
            report.hdf5_timestamp_end_ns = int(ts_arr[-1])
        else:
            ts_ds.issues.append("timestamp dataset 缺失")
            ds.issues.append("timestamp dataset 缺失")

        datasets.append(ts_ds)
        report.hdf5_datasets = datasets

        # shape 一致性
        if len(sample_counts) > 1:
            asset.issues.append(
                f"HDF5 dataset 行数不一致: {sorted(sample_counts)}"
            )
            asset.disposition = "keep_with_flag"
        elif len(sample_counts) == 1:
            report.hdf5_sample_count = next(iter(sample_counts))
    finally:
        f.close()


def _check_raw_h5(root: Path) -> A2DAssetStatus:
    """检查 raw_joints.h5（可选）。"""
    asset = A2DAssetStatus(
        asset_id="raw_joints.h5",
        asset_type="hdf5",
        required=False,
        present=False,
    )
    path = root / "raw_joints.h5"
    if path.is_file():
        asset.present = True
        try:
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                asset.details = {
                    "top_level_keys": keys,
                    "has_timestamp": "timestamp" in f,
                }
                if "timestamp" in f:
                    asset.frame_count = f["timestamp"].shape[0]
        except Exception as exc:
            asset.issues.append(f"raw_joints.h5 无法打开: {exc}")
            asset.present = False
            asset.disposition = "keep_with_flag"
    else:
        asset.issues.append("raw_joints.h5 不存在（可选，非阻断）")
    return asset


def _check_a2d_calibration(root: Path) -> A2DAssetStatus:
    """检查标定文件。"""
    asset = A2DAssetStatus(
        asset_id="camera_calibration",
        asset_type="calibration",
        required=True,
        present=False,
    )
    calib_dir = root / "parameters" / "camera"
    if not calib_dir.is_dir():
        asset.issues.append("parameters/camera/ 目录不存在")
        asset.disposition = "reject"
        return asset

    present: list[str] = []
    missing: list[str] = []
    for cam_name, filename in _CALIB_FILES.items():
        if (calib_dir / filename).is_file():
            present.append(cam_name)
        else:
            missing.append(cam_name)

    asset.present = len(present) > 0
    asset.details = {
        "present_calibrations": present,
        "missing_calibrations": missing,
    }

    if missing:
        asset.issues.append(f"缺少标定的相机: {missing}")
        asset.disposition = "keep_with_flag"
    else:
        asset.disposition = "pass"

    return asset


def _check_joint_map(root: Path) -> A2DAssetStatus:
    """检查 joint_map.json。"""
    asset = A2DAssetStatus(
        asset_id="joint_map.json",
        asset_type="metadata",
        required=False,
        present=False,
    )
    path = root / "parameters" / "meshes" / "joint_map.json"
    if path.is_file():
        asset.present = True
        try:
            with open(path, encoding="utf-8") as f:
                jm = json.load(f)
            active = {k: v for k, v in jm.items() if v >= 0}
            asset.details = {
                "total_joints": len(jm),
                "active_joints": len(active),
            }
        except Exception as exc:
            asset.issues.append(f"joint_map.json 解析失败: {exc}")
    else:
        asset.issues.append("parameters/meshes/joint_map.json 不存在")
    return asset


def _check_mcap(root: Path) -> A2DAssetStatus:
    """检查 ROS2 MCAP 文件。"""
    asset = A2DAssetStatus(
        asset_id="mcap",
        asset_type="mcap",
        required=False,
        present=False,
    )
    # 在 episode 目录及上级查找 .mcap 文件
    candidates = list(root.glob("*.mcap")) + list(root.parent.glob("*.mcap"))
    if candidates:
        mcap_path = candidates[0]
        asset.present = True
        asset.details = {
            "path": str(mcap_path),
            "size_bytes": mcap_path.stat().st_size if mcap_path.is_file() else 0,
        }
    else:
        asset.issues.append("未找到 ROS2 MCAP 文件（可选，非阻断）")
    return asset


def _check_device_logs(root: Path) -> A2DAssetStatus:
    """检查设备日志。"""
    asset = A2DAssetStatus(
        asset_id="device_logs",
        asset_type="log",
        required=False,
        present=False,
    )
    log_dir = root / "logs"
    if log_dir.is_dir():
        log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("*.txt"))
        if log_files:
            asset.present = True
            asset.details = {
                "log_count": len(log_files),
                "paths": [str(p.relative_to(root)) for p in log_files[:5]],
            }
            return asset
    asset.issues.append("未找到设备日志（可选，非阻断）")
    return asset


def _check_review_annotations(root: Path) -> A2DAssetStatus:
    """检查 review 标注。"""
    asset = A2DAssetStatus(
        asset_id="review_annotations",
        asset_type="annotation",
        required=False,
        present=False,
    )
    # review_*.json 模式
    candidates = list(root.glob("review_*.json")) + list(root.glob("*.review.json"))
    if candidates:
        asset.present = True
        asset.details = {
            "sources": [str(p.name) for p in candidates],
        }
    else:
        asset.issues.append("未找到 review 标注文件（可选，非阻断）")
    return asset


# ---------------------------------------------------------------------------
# 交叉验证
# ---------------------------------------------------------------------------


def _cross_validate(report: A2DCompletenessReport) -> dict[str, Any]:
    """交叉验证 meta duration、camera frames、HDF5 timestamps。"""
    cv: dict[str, Any] = {
        "discrepancies": [],
        "warnings": [],
    }

    meta = report.assets.get("meta_info.json")
    head_rgb = report.assets.get("head_rgb")

    # 1. meta duration vs camera frame count
    if (
        meta
        and meta.present
        and head_rgb
        and head_rgb.present
    ):
        meta_duration_s = meta.details.get("duration_s")
        if meta_duration_s is not None:
            expected_frames = float(meta_duration_s) * 30.0  # 30fps 标称
            actual_frames = head_rgb.frame_count
            if actual_frames > 0 and expected_frames > 0:
                ratio = actual_frames / expected_frames
                if ratio < 0.5 or ratio > 2.0:
                    cv["discrepancies"].append({
                        "type": "duration_vs_frames",
                        "meta_duration_s": meta_duration_s,
                        "expected_frames_at_30fps": round(expected_frames),
                        "actual_head_rgb_frames": actual_frames,
                        "ratio": round(ratio, 3),
                        "note": "相机帧数与 meta duration 差异过大",
                    })

    # 2. HDF5 timestamp range vs meta clip range
    if report.hdf5_sample_count > 0:
        h5_duration_s = (
            (report.hdf5_timestamp_end_ns - report.hdf5_timestamp_start_ns)
            / 1_000_000_000
        )
        cv["hdf5_duration_s"] = round(h5_duration_s, 3)

        if meta and meta.present:
            meta_duration_s = meta.details.get("duration_s")
            if meta_duration_s is not None:
                ratio = h5_duration_s / meta_duration_s if meta_duration_s > 0 else 0
                if ratio < 0.5 or ratio > 2.0:
                    cv["discrepancies"].append({
                        "type": "hdf5_vs_meta_duration",
                        "hdf5_duration_s": round(h5_duration_s, 3),
                        "meta_duration_s": meta_duration_s,
                        "ratio": round(ratio, 3),
                        "note": "HDF5 时间范围与 meta duration 差异过大",
                    })

    # 3. 相机间帧数一致性
    rgb_assets = [
        a for a_id, a in report.assets.items()
        if "rgb" in a_id and a.present
    ]
    if len(rgb_assets) > 1:
        frame_counts = {a.asset_id: a.frame_count for a in rgb_assets}
        unique_counts = set(frame_counts.values())
        if len(unique_counts) > 1:
            cv["discrepancies"].append({
                "type": "camera_frame_count_mismatch",
                "frame_counts": frame_counts,
                "note": "RGB 相机间帧数不一致",
            })

    # 4. HDF5 sample count vs camera frame count
    if report.hdf5_sample_count > 0 and head_rgb and head_rgb.present:
        camera_frames = head_rgb.frame_count
        h5_samples = report.hdf5_sample_count
        if abs(camera_frames - h5_samples) > max(camera_frames, h5_samples) * 0.1:
            cv["warnings"].append({
                "type": "hdf5_vs_camera_count",
                "hdf5_samples": h5_samples,
                "camera_frames": camera_frames,
                "diff": abs(camera_frames - h5_samples),
                "note": "HDF5 状态样本数与相机帧数差异 > 10%",
            })

    # 5. HDF5 时间戳有效性
    cv["hdf5_timestamps_monotonic"] = report.hdf5_timestamps_valid
    if not report.hdf5_timestamps_valid:
        cv["discrepancies"].append({
            "type": "hdf5_timestamps_non_monotonic",
            "note": "HDF5 时间戳存在非递增",
        })

    return cv


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def _aggregate(report: A2DCompletenessReport) -> None:
    """聚合 disposition 和统计。"""
    required = {
        aid: a for aid, a in report.assets.items() if a.required
    }
    optional = {
        aid: a for aid, a in report.assets.items() if not a.required
    }

    report.required_total = len(required)
    report.required_present = sum(1 for a in required.values() if a.present)
    report.optional_total = len(optional)
    report.optional_present = sum(1 for a in optional.values() if a.present)

    # HDF5 内部缺失也算 reject
    h5_asset = report.assets.get("aligned_joints.h5")
    has_h5_issue = (
        h5_asset
        and h5_asset.present
        and any(
            not ds.present
            for ds in report.hdf5_datasets
            if ds.path in _REQUIRED_HDF5_DATASETS
        )
    )

    dispositions = [a.disposition for a in report.assets.values()]
    if "reject" in dispositions or has_h5_issue:
        report.overall_disposition = "reject"
    elif "keep_with_flag" in dispositions or report.cross_validation.get("discrepancies"):
        report.overall_disposition = "keep_with_flag"
    else:
        report.overall_disposition = "pass"


def _dir_sha256(root: Path) -> str:
    """计算 Episode 目录的轻量 SHA-256（基于文件清单和大小，非全量哈希）。"""
    digest = hashlib.sha256()
    # 只取文件路径和大小作为 proxy，避免全量扫描大文件
    entries: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            entries.append(f"{p.relative_to(root)}:{size}")
    digest.update("\n".join(entries).encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "A2DAssetStatus",
    "A2DCompletenessReport",
    "HDF5DatasetStatus",
    "check_a2d_completeness",
]
