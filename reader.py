"""
读取墨现 (Guida) 数据集的所有原始文件。
"""

import os
import json
import pandas as pd


def read_meta(dataset_path: str) -> dict:
    """读取 meta.json，返回扁平化的元数据字典。

    Returns:
        {
            "device": str,
            "fps": int,
            "frame_count": int,
            "width": int,
            "height": int,
            "dropped_frames": int,
            "imu_sample_rate": int,
        }
    """
    meta_path = os.path.join(dataset_path, "meta.json")
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


def read_index(dataset_path: str) -> dict:
    """读取 index.jsonl，只提取 type=frame 行的时间戳。

    Returns:
        {
            "timestamps": list[int],   # timestamp_ns 列表
            "frame_count": int,
        }
    """
    index_path = os.path.join(dataset_path, "index.jsonl")
    timestamps = []

    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "frame":
                timestamps.append(item["timestamp_ns"])

    return {
        "timestamps": timestamps,
        "frame_count": len(timestamps),
    }


def read_imu(dataset_path: str, imu_filename: str = "imu_000000.csv") -> pd.DataFrame:
    """读取 IMU CSV 文件。

    Args:
        dataset_path: 数据集根目录
        imu_filename: IMU 文件名，默认 imu_000000.csv

    Returns:
        pd.DataFrame，列: timestamp_ns, ax, ay, az, gx, gy, gz
    """
    imu_path = os.path.join(dataset_path, "imu", imu_filename)
    return pd.read_csv(imu_path)
