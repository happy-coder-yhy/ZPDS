"""
读取墨现 (Guida) 数据集的所有原始文件。

复用已有的 reader.py，提供更结构化的接口。
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def read_meta(dataset_path: str) -> dict[str, Any]:
    """读取 meta.json，返回扁平化的元数据字典。"""
    meta_path = Path(dataset_path) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "device": meta["device"]["name"],
        "fps": meta["streams"]["color"]["fps"],
        "frame_count": meta["recording_stats"]["total_frames"],
        "width": meta["streams"]["color"]["width"],
        "height": meta["streams"]["color"]["height"],
        "dropped_frames": meta["recording_stats"]["dropped_frames"],
        "imu_sample_rate": meta["imu"]["sample_rate_hz"],
    }


def read_index_frames(dataset_path: str) -> list[dict]:
    """读取 index.jsonl 中所有 type=frame 的行。

    Returns:
        [{seq, timestamp_ns, type, ...}, ...]  按 seq 排序
    """
    index_path = Path(dataset_path) / "index.jsonl"
    frames = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "frame":
                frames.append(item)
    frames.sort(key=lambda f: f["seq"])
    return frames


def read_index_timestamps(dataset_path: str) -> list[int]:
    """读取 index.jsonl，只返回 type=frame 的纳秒时间戳列表（已排序）。"""
    frames = read_index_frames(dataset_path)
    return [f["timestamp_ns"] for f in frames]


def read_imu(dataset_path: str, imu_filename: str = "imu_000000.csv") -> pd.DataFrame:
    """读取 IMU CSV 文件。

    Returns:
        pd.DataFrame，列: timestamp_ns, ax, ay, az, gx, gy, gz
    """
    imu_path = Path(dataset_path) / "imu" / imu_filename
    return pd.read_csv(imu_path)


def get_color_mkv(dataset_path: str) -> str:
    """获取 RGB 原始 MKV 路径。"""
    return str(Path(dataset_path) / "color_000000.mkv")


def get_session_id(dataset_path: str) -> str:
    """从数据集路径推导 session_id。"""
    folder = Path(dataset_path).name
    return f"guida_{folder}"
