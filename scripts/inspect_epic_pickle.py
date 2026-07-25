#!/usr/bin/env python3
"""
探测 EPIC-KITCHENS-100 Pickle 文件结构。

安全地在子进程中反序列化单个 Pickle，输出结构摘要 JSON。
不做完整打印，只输出类型、Key、形状、长度等元信息。

用法:
    python scripts/inspect_epic_pickle.py \
      --input "hand-objects/P01/P01_01.pkl" \
      --output "output/epic/pkl_probe_P01_01.json"

    # 可选: 导出指定帧号的预览数据 (帧索引列表)
    python scripts/inspect_epic_pickle.py \
      --input "hand-objects/P01/P01_01.pkl" \
      --output "output/epic/pkl_probe_P01_01.json" \
      --sample-frames 0,100,200
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


# ---- 摘要函数 ----

def summarize(value, depth: int = 0) -> dict:
    """递归输出值的结构摘要，深度限制避免打印大数组。

    Args:
        value: 任意 Python 对象
        depth: 当前递归深度

    Returns:
        摘要 dict，含 type / length / keys / shape / repr 等
    """
    if depth > 3:
        return {
            "type": type(value).__name__,
            "summary": "max_depth_reached",
        }

    if isinstance(value, dict):
        keys = list(value.keys())
        sample_keys = keys[:3]
        return {
            "type": "dict",
            "length": len(value),
            "keys_range": f"{str(keys[0])} .. {str(keys[-1])}" if len(keys) > 3 else None,
            "keys_preview": [str(k) for k in sample_keys],
            "sample_values": {
                str(k): summarize(value[k], depth + 1)
                for k in sample_keys
            },
        }

    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first_item": (
                summarize(value[0], depth + 1)
                if value else None
            ),
            "last_item": (
                summarize(value[-1], depth + 1)
                if len(value) > 1 else None
            ),
        }

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    return {
        "type": type(value).__name__,
        "repr": repr(value)[:500],
    }


# ---- 帧预览 ----

def extract_frame_samples(data, frame_indices: list[int]) -> dict:
    """提取指定帧的详细内容（含 BBox / 手 / 物体完整数据）。

    Args:
        data: 反序列化后的 pickle 对象
        frame_indices: 要完整导出的帧索引列表 (0-based)

    Returns:
        {
            "root_key_type": "int_str" | "str" | ...,
            "frames": { frame_idx: {完整帧数据} }
        }
    """
    if isinstance(data, dict):
        # 检测 key 类型
        str_keys = [k for k in data.keys()]
        key_type = "int_str" if all(str(k).isdigit() for k in str_keys) else type(str_keys[0]).__name__

        frames = {}
        for idx in frame_indices:
            key = str(idx)  # 尝试字符串 key
            if key in data:
                frames[idx] = _safe_serialize(data[key])
            elif idx in data:
                frames[idx] = _safe_serialize(data[idx])

        return {"root_key_type": key_type, "frames": frames}

    return {"root_key_type": type(data).__name__, "frames": {}}


def _safe_serialize(value) -> dict:
    """将帧级数据安全序列化为可 JSON 输出的 dict。

    处理 numpy 数组、标量、嵌套结构。
    """
    import numpy as np

    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            result[str(k)] = _safe_serialize(v)
        return result
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return {"_type": type(value).__name__, "_length": 0}
        return [_safe_serialize(v) for v in value]
    elif isinstance(value, np.ndarray):
        return {
            "_type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.flatten()[:20].tolist(),
        }
    elif isinstance(value, (np.integer,)):
        return int(value)
    elif isinstance(value, (np.floating,)):
        return float(value)
    elif hasattr(value, "item"):
        return value.item()
    elif isinstance(value, (int, float, str, bool, type(None))):
        return value
    else:
        return {"_type": type(value).__name__, "repr": repr(value)[:200]}


# ---- 主入口 ----

def main() -> None:
    parser = argparse.ArgumentParser(
        description="探测 EPIC-KITCHENS-100 Pickle 结构"
    )
    parser.add_argument(
        "--input", required=True,
        help="输入 .pkl 文件路径",
    )
    parser.add_argument(
        "--output", required=True,
        help="输出结构摘要 .json 路径",
    )
    parser.add_argument(
        "--sample-frames",
        default="",
        help="要完整导出的帧索引，逗号分隔，如 '0,100,200'",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[error] 文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 解析 sample-frames
    sample_frames: list[int] = []
    if args.sample_frames.strip():
        sample_frames = [int(x.strip()) for x in args.sample_frames.split(",") if x.strip()]

    # ---- 反序列化 ----
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    # ---- 构建报告 ----
    report: dict = {
        "source_file": input_path.name,
        "root": summarize(data),
    }

    # 如果有 sample-frames，加入帧预览
    if sample_frames:
        report["sample_frames"] = extract_frame_samples(data, sample_frames)

    # 额外探测顶层类型
    report["meta"] = {
        "root_type": type(data).__name__,
    }
    if isinstance(data, dict):
        keys = list(data.keys())
        report["meta"]["total_entries"] = len(data)
        report["meta"]["key_type"] = type(keys[0]).__name__ if keys else "empty"
        report["meta"]["first_key"] = str(keys[0]) if keys else None
        report["meta"]["last_key"] = str(keys[-1]) if keys else None

    # ---- 写出 ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"已写出: {output_path}")
    print(f"根类型:  {report['meta']['root_type']}")
    if isinstance(data, dict):
        print(f"条目数:  {report['meta']['total_entries']}")
        print(f"Key 范围: {report['meta']['first_key']} → {report['meta']['last_key']}")


if __name__ == "__main__":
    main()
