"""
扫描一个 A2D Episode，生成 inventory.json。

输入:  A2D episode 目录路径
输出:  output/a2d/{episode_id}/inventory.json

用法:
    python scripts/inspect_a2d_episode.py "E:/datasets/真机/A2D/"
    python scripts/inspect_a2d_episode.py "E:/datasets/真机/A2D/" --output output/a2d/
    python scripts/inspect_a2d_episode.py "E:/datasets/真机/A2D/" --episode-id 8032
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

# 每个 frame_idx 目录预期包含的 6 个文件
EXPECTED_CAMERA_FILES: list[str] = [
    "head_color.jpg",
    "head_depth.png",
    "hand_left_color.jpg",
    "hand_left_depth.png",
    "hand_right_color.jpg",
    "hand_right_depth.png",
]

# 三组相机内参
CAMERA_INTRINSIC_FILES: dict[str, str] = {
    "head": "parameters/camera/head_intrinsic_params.json",
    "hand_left": "parameters/camera/hand_left_intrinsic_params.json",
    "hand_right": "parameters/camera/hand_right_intrinsic_params.json",
}

# 四组 ROS2 MCAP
MCAP_FILES: dict[str, str] = {
    "gripper_commands": "record/gripper-commands/gripper-commands_0.mcap",
    "gripper_states": "record/gripper-states/gripper-states_0.mcap",
    "joint_commands": "record/joint-commands/joint-commands_0.mcap",
    "joint_states": "record/joint-states/joint-states_0.mcap",
}

# 日志文件
LOG_FILES: list[str] = [
    "logs/data-acquire-camera.log",
    "logs/data-acquire-mcap.log",
    "logs/data-coordinator.log",
    "logs/data-uploader.log",
    "logs/recorder.log",
    "logs/voice_manager.log",
]


# ---------------------------------------------------------------------------
# 扫描函数
# ---------------------------------------------------------------------------

def scan_camera_frames(camera_root: Path) -> list[dict]:
    """扫描 camera/ 下所有 frame_idx 目录，返回每条目录的完整性记录。

    Returns:
        按 frame_index 升序排列的记录列表。
    """
    records: list[dict] = []

    if not camera_root.is_dir():
        return records

    for frame_dir in camera_root.iterdir():
        if not frame_dir.is_dir():
            continue

        try:
            frame_index = int(frame_dir.name)
        except ValueError:
            continue

        present = {
            filename: (frame_dir / filename).exists()
            for filename in EXPECTED_CAMERA_FILES
        }

        records.append({
            "frame_index": frame_index,
            "directory": str(frame_dir),
            "files": present,
            "complete": all(present.values()),
        })

    return sorted(records, key=lambda row: row["frame_index"])


def _exists(episode_root: Path, relative_path: str) -> dict:
    """检查单个文件/目录是否存在，返回 {uri, exists}。"""
    full = episode_root / relative_path
    return {
        "uri": relative_path,
        "exists": full.exists(),
    }


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def build_inventory(episode_path: str, episode_id: str | None = None) -> dict:
    """扫描整个 A2D Episode，生成 inventory dict。

    Args:
        episode_path: Episode 根目录路径。
        episode_id:   Episode ID；为 None 时自动从 review 文件名或目录名提取。

    Returns:
        符合 a2d_inventory.v1 schema 的 dict。
    """
    root = Path(episode_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    # --- 自动推断 episode_id ---
    if episode_id is None:
        episode_id = _infer_episode_id(root)

    inventory: dict = {
        "schema_version": "a2d_inventory.v1",
        "episode_id": episode_id,
        "episode_path": str(root),
        "assets": {},
        "camera": {},
        "mcap": {},
        "logs": {},
        "review": {},
    }

    # --- 顶层资产 ---
    assets: dict[str, dict] = {}

    assets["meta_info"] = _exists(root, "meta_info.json")
    assets["aligned_joints"] = _exists(root, "aligned_joints.h5")
    assets["raw_joints"] = _exists(root, "record/raw_joints.h5")

    # 相机内参
    for cam_name, rel_path in CAMERA_INTRINSIC_FILES.items():
        assets[f"intrinsics_{cam_name}"] = _exists(root, rel_path)

    inventory["assets"] = assets

    # --- 相机帧扫描 ---
    camera_root = root / "camera"
    frame_records = scan_camera_frames(camera_root)

    complete_count = sum(1 for r in frame_records if r["complete"])
    incomplete_count = len(frame_records) - complete_count

    rgb_file_count = sum(
        1 for r in frame_records
        for k, v in r["files"].items() if v and k.endswith("_color.jpg")
    )
    depth_file_count = sum(
        1 for r in frame_records
        for k, v in r["files"].items() if v and k.endswith("_depth.png")
    )

    # 找出不完整的 frame_idx
    incomplete_frames = [
        {
            "frame_index": r["frame_index"],
            "missing": [fname for fname, ok in r["files"].items() if not ok],
        }
        for r in frame_records if not r["complete"]
    ]

    inventory["camera"] = {
        "frame_directory_count": len(frame_records),
        "complete_frame_count": complete_count,
        "incomplete_frame_count": incomplete_count,
        "min_frame_index": frame_records[0]["frame_index"] if frame_records else None,
        "max_frame_index": frame_records[-1]["frame_index"] if frame_records else None,
        "rgb_file_count": rgb_file_count,
        "depth_file_count": depth_file_count,
        "rgb_depth_pairing_rate": (
            round(depth_file_count / rgb_file_count, 4)
            if rgb_file_count > 0 else 0.0
        ),
        "incomplete_frames": incomplete_frames,
    }

    # --- MCAP ---
    mcap_status: dict[str, dict] = {}
    for mcap_name, rel_path in MCAP_FILES.items():
        mcap_status[mcap_name] = _exists(root, rel_path)
    inventory["mcap"] = mcap_status

    # --- Review ---
    review_dir = root / "review"
    review_files = sorted(review_dir.glob("review_*.json")) if review_dir.is_dir() else []
    inventory["review"] = {
        "files": [str(f.relative_to(root)) for f in review_files],
        "exists": len(review_files) > 0,
    }

    # --- Logs ---
    log_status: dict[str, dict] = {}
    for log_rel in LOG_FILES:
        log_name = Path(log_rel).name
        log_status[log_name] = _exists(root, log_rel)
    inventory["logs"] = log_status
    inventory["logs"]["log_directory_exists"] = (root / "logs").is_dir()

    # --- 整体完成度摘要 ---
    all_asset_ok = all(v["exists"] for v in assets.values())
    all_mcap_ok = all(v["exists"] for v in mcap_status.values())
    all_log_ok = all(
        v["exists"] for k, v in log_status.items()
        if isinstance(v, dict) and "exists" in v
    )

    inventory["summary"] = {
        "all_assets_present": all_asset_ok,
        "all_mcap_present": all_mcap_ok,
        "all_camera_frames_complete": incomplete_count == 0,
        "all_logs_present": all_log_ok,
        "review_present": inventory["review"]["exists"],
        "is_fully_complete": (
            all_asset_ok and all_mcap_ok and (incomplete_count == 0)
            and all_log_ok and inventory["review"]["exists"]
        ),
    }

    return inventory


def _infer_episode_id(root: Path) -> str:
    """从 review JSON 文件名或目录名推断 episode_id。"""
    review_dir = root / "review"
    if review_dir.is_dir():
        for f in review_dir.glob("review_*.json"):
            # e.g. "review_8032.json" → "8032"
            stem = f.stem  # review_8032
            if stem.startswith("review_"):
                return stem[len("review_"):]

    # 回退: 用目录名
    return root.name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="扫描一个 A2D Episode，生成 inventory.json",
    )
    parser.add_argument(
        "episode_path",
        help="A2D Episode 根目录路径，例如 E:/datasets/真机/A2D/",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/a2d/",
        help="输出根目录 (默认: output/a2d/)。inventory.json 将写入 "
             "{output}/{episode_id}/inventory.json",
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Episode ID；不指定时自动从 review_*.json 或目录名推断",
    )
    args = parser.parse_args()

    episode_path = Path(args.episode_path)
    if not episode_path.is_dir():
        print(f"错误: 目录不存在 — {episode_path}", file=sys.stderr)
        sys.exit(1)

    print(f"扫描 A2D Episode: {episode_path}")
    print()

    inventory = build_inventory(str(episode_path), args.episode_id)

    # 写入输出
    episode_id = inventory["episode_id"]
    out_dir = Path(args.output) / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "inventory.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    print(f"  写入: {out_path}")

    # 打印摘要
    print()
    print("=" * 50)
    print(f"  Episode {episode_id} Inventory 摘要")
    print("=" * 50)

    c = inventory["camera"]
    s = inventory["summary"]
    a = inventory["assets"]

    print(f"\n  顶层资产:")
    for name, asset in a.items():
        icon = "✓" if asset["exists"] else "✗"
        print(f"    {icon} {asset['uri']}")

    print(f"\n  相机帧:")
    print(f"    frame_idx 目录数:     {c['frame_directory_count']}")
    print(f"    完整帧数:             {c['complete_frame_count']}")
    print(f"    不完整帧数:           {c['incomplete_frame_count']}")
    print(f"    frame_idx 范围:       {c['min_frame_index']} – {c['max_frame_index']}")
    print(f"    RGB 文件数:           {c['rgb_file_count']}")
    print(f"    Depth 文件数:         {c['depth_file_count']}")
    print(f"    RGB-Depth 配对率:     {c['rgb_depth_pairing_rate']:.2%}")
    if c["incomplete_frames"]:
        for inc in c["incomplete_frames"][:5]:
            print(f"      ✗ frame {inc['frame_index']}: 缺失 {inc['missing']}")
        if len(c["incomplete_frames"]) > 5:
            print(f"      ... 还有 {len(c['incomplete_frames']) - 5} 个不完整帧")

    print(f"\n  MCAP:")
    for name, mcap in inventory["mcap"].items():
        icon = "✓" if mcap["exists"] else "✗"
        print(f"    {icon} {mcap['uri']}")

    print(f"\n  Review: {'✓' if inventory['review']['exists'] else '✗'}")
    print(f"  Logs:   {'✓' if s['all_logs_present'] else '✗'} "
          f"({sum(1 for v in inventory['logs'].values() if isinstance(v, dict) and v.get('exists'))} "
          f"/ {len(LOG_FILES)} 存在)")

    print(f"\n{'─' * 50}")
    print(f"  整体完成度: {'✓ 完全' if s['is_fully_complete'] else '✗ 不完整'}")
    print(f"{'=' * 50}")

    return 0 if s["is_fully_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
