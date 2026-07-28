"""
EPIC-KITCHENS-100 数据资产扫描与索引。

扫描三类文件并建立 video_id ↔ 视频/标注的对应关系：
  - 原始视频 (.mp4/.MP4/.mkv)
  - Hand-object Pickle (.pkl, hand-objects/)
  - Mask Pickle (.pkl, masks/)

职责：扫描、ID 提取、文件匹配、构建 inventory。
不做解码、不做清洗。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone


# ---- ID 解析 ----

# EPIC-KITCHENS 视频 ID 格式: P{digits}_{digits} 如 P01_01
VIDEO_ID_PATTERN = re.compile(r"^(P\d+_\d+)$")


def parse_epic_id(path: Path) -> tuple[str, str]:
    """从文件名解析 participant_id 和 video_id。

    Args:
        path: 文件路径，如 P01/P01_01.pkl 或 P01_01.mp4

    Returns:
        (participant_id, video_id) 如 ("P01", "P01_01")

    Raises:
        ValueError: 文件名不符合 EPIC 视频 ID 格式
    """
    video_id = path.stem

    if not VIDEO_ID_PATTERN.match(video_id):
        raise ValueError(f"无法识别 EPIC 视频 ID：{path}")

    participant_id = video_id.split("_", maxsplit=1)[0]

    return participant_id, video_id


# ---- 文件扫描 ----

def scan_files(
    videos_root: Path,
    annotations_root: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """扫描三类文件。

    Args:
        videos_root: 视频文件根目录
        annotations_root: 标注根目录（含 hand-objects/ 和 masks/ 子目录）

    Returns:
        (video_files, hand_object_files, mask_files)
    """
    # 视频（Windows NTFS 大小写不敏感，glob *.mp4 和 *.MP4 会重复匹配，需去重）
    video_files: list[Path] = []
    for suffix in ("*.mp4", "*.MP4", "*.mkv"):
        video_files.extend(videos_root.rglob(suffix))
    video_files = list(dict.fromkeys(video_files))  # 保序去重

    # Hand-object Pickle（排除 .html 等非 pickle 文件）
    ho_dir = annotations_root / "hand-objects"
    hand_object_files: list[Path] = []
    if ho_dir.is_dir():
        for p in ho_dir.rglob("*.pkl"):
            hand_object_files.append(p)

    # Mask Pickle
    mask_dir = annotations_root / "masks"
    mask_files: list[Path] = []
    if mask_dir.is_dir():
        for p in mask_dir.rglob("*.pkl"):
            mask_files.append(p)

    return video_files, hand_object_files, mask_files


# ---- 建立索引 ----

def index_by_video_id(paths: list[Path]) -> dict[str, Path]:
    """按 video_id 建立唯一索引。

    Raises:
        ValueError: 发现重复 video_id
    """
    result: dict[str, Path] = {}

    for path in paths:
        _, video_id = parse_epic_id(path)

        if video_id in result:
            raise ValueError(
                f"发现重复 ID：{video_id}\n"
                f"  {result[video_id]}\n"
                f"  {path}"
            )

        result[video_id] = path

    return result


# ---- 文件属性 ----

def sha256_file(path: Path) -> str:
    """计算文件 SHA-256 哈希。"""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def file_stat(path: Path, *, compute_hash: bool = True) -> dict:
    """获取单个文件的属性。

    Args:
        path: 文件路径
        compute_hash: 是否计算 SHA-256

    Returns:
        {"size_bytes": int, "sha256": str, "modified_time": str}
    """
    stat = path.stat()
    record: dict = {
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }
    if compute_hash:
        record["sha256"] = sha256_file(path)
    return record


# ---- 合并资产信息 ----

def build_inventory(
    video_index: dict[str, Path],
    hand_index: dict[str, Path],
    mask_index: dict[str, Path],
    *,
    compute_hash: bool = True,
) -> list[dict]:
    """合并三类索引为 inventory 记录列表。

    Args:
        video_index: {video_id: video_path}
        hand_index:  {video_id: hand_object_path}
        mask_index:  {video_id: mask_path}
        compute_hash: 是否计算文件 SHA-256

    Returns:
        按 video_id 排序的记录列表
    """
    all_ids = sorted(
        set(video_index)
        | set(hand_index)
        | set(mask_index)
    )

    records: list[dict] = []

    for video_id in all_ids:
        participant_id = video_id.split("_")[0]

        video_path = video_index.get(video_id)
        hand_path = hand_index.get(video_id)
        mask_path = mask_index.get(video_id)

        # 判定状态
        if video_path and hand_path:
            status = "ready"
        elif video_path:
            status = "video_only"
        else:
            status = "annotation_only"

        record: dict = {
            "participant_id": participant_id,
            "video_id": video_id,
            "video_uri": str(video_path) if video_path else None,
            "hand_object_uri": str(hand_path) if hand_path else None,
            "mask_uri": str(mask_path) if mask_path else None,
            "status": status,
        }

        # 附加文件属性
        if video_path:
            record["video_attr"] = file_stat(video_path, compute_hash=compute_hash)
        if hand_path:
            record["hand_object_attr"] = file_stat(hand_path, compute_hash=compute_hash)
        if mask_path:
            record["mask_attr"] = file_stat(mask_path, compute_hash=compute_hash)

        records.append(record)

    return records


# ---- 统计 ----

def compute_statistics(records: list[dict]) -> dict:
    """计算 inventory 汇总统计。

    Returns:
        {
            "total_ids": int,
            "ready": int,
            "video_only": int,
            "annotation_only": int,
            "has_hand_object": int,
            "has_mask": int,
            "total_videos": int,
            "total_hand_objects": int,
            "total_masks": int,
            "duplicate_ids": int,
        }
    """
    stats = {
        "total_ids": len(records),
        "ready": 0,
        "video_only": 0,
        "annotation_only": 0,
        "has_hand_object": 0,
        "has_mask": 0,
    }

    for r in records:
        status = r["status"]
        if status == "ready":
            stats["ready"] += 1
        elif status == "video_only":
            stats["video_only"] += 1
        elif status == "annotation_only":
            stats["annotation_only"] += 1

        if r["hand_object_uri"]:
            stats["has_hand_object"] += 1
        if r["mask_uri"]:
            stats["has_mask"] += 1

    return stats


# ---- 主入口 ----

def scan_inventory(
    videos_root: str,
    annotations_root: str,
    *,
    compute_hash: bool = True,
) -> dict:
    """扫描并生成 EPIC-KITCHENS-100 inventory。

    Args:
        videos_root: 视频文件根目录
        annotations_root: 标注根目录
        compute_hash: 是否计算 SHA-256

    Returns:
        完整的 inventory dict（schema_version + statistics + records）
    """
    video_files, ho_files, mask_files = scan_files(
        Path(videos_root), Path(annotations_root)
    )

    # 建立索引（重复 ID 会直接抛异常）
    video_index = index_by_video_id(video_files)
    ho_index = index_by_video_id(ho_files)
    mask_index = index_by_video_id(mask_files)

    # 合并
    records = build_inventory(
        video_index, ho_index, mask_index,
        compute_hash=compute_hash,
    )

    statistics = compute_statistics(records)

    return {
        "schema_version": "epic_inventory.v1",
        "statistics": statistics,
        "records": records,
    }


__all__ = [
    "VIDEO_ID_PATTERN",
    "parse_epic_id",
    "scan_files",
    "index_by_video_id",
    "sha256_file",
    "file_stat",
    "build_inventory",
    "compute_statistics",
    "scan_inventory",
]
