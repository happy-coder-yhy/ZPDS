#!/usr/bin/env python3
"""
扫描 EPIC-KITCHENS-100 数据集，生成 inventory.json。

建立 video_id ↔ 原始视频 ↔ hand-object.pkl ↔ mask.pkl 对应关系。

用法:
    python scripts/inspect_epic_dataset.py \
      --videos-root "E:/datasets/epic-videos" \
      --annotations-root "E:/datasets/epic-kitchens-100" \
      --output "output/epic/inventory.json"

    # 跳过 SHA-256 计算（快速扫描）
    python scripts/inspect_epic_dataset.py \
      --videos-root "E:/datasets/epic-videos" \
      --annotations-root "E:/datasets/epic-kitchens-100" \
      --output "output/epic/inventory.json" \
      --skip-hash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保能导入 zpds_prepare
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from zpds_prepare.readers.epic_inventory import scan_inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EPIC-KITCHENS-100 数据资产扫描"
    )
    parser.add_argument(
        "--videos-root",
        required=True,
        help="视频文件根目录",
    )
    parser.add_argument(
        "--annotations-root",
        required=True,
        help="标注根目录（含 hand-objects/ 和 masks/ 子目录）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 inventory.json 路径",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="跳过 SHA-256 计算（快速扫描）",
    )

    args = parser.parse_args()

    # 校验输入路径
    videos_root = Path(args.videos_root)
    annotations_root = Path(args.annotations_root)

    if not videos_root.is_dir():
        print(f"[warn] 视频目录不存在: {videos_root}")
        print(f"       将只扫描标注文件")

    if not annotations_root.is_dir():
        print(f"[error] 标注目录不存在: {annotations_root}")
        sys.exit(1)

    # 扫描
    print(f"视频根目录:     {videos_root}")
    print(f"标注根目录:     {annotations_root}")
    print(f"计算 SHA-256:   {not args.skip_hash}")
    print()

    inventory = scan_inventory(
        str(videos_root),
        str(annotations_root),
        compute_hash=not args.skip_hash,
    )

    stats = inventory["statistics"]
    print(f"扫描完成:")
    print(f"  总 video_id:        {stats['total_ids']}")
    print(f"  ready:              {stats['ready']}  (视频 + 标注 齐全)")
    print(f"  video_only:         {stats['video_only']}  (仅有视频)")
    print(f"  annotation_only:    {stats['annotation_only']}  (仅有标注)")
    print(f"  含 hand-object:     {stats['has_hand_object']}")
    print(f"  含 mask:            {stats['has_mask']}")

    # 写出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    print(f"\n已写出: {output_path}")


if __name__ == "__main__":
    main()
