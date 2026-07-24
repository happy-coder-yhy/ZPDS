"""
IMU 数据规范化：裁剪到 Segment 范围，转换为相对时间，写出 Parquet。
"""

import pandas as pd
from pathlib import Path


def normalize_imu(
    imu_path: str,
    source_start_ns: int,
    source_end_ns: int,
) -> pd.DataFrame:
    """规范化 IMU 数据。

    - 只保留 Segment 时间范围内的行
    - 原始时间戳转换为 Segment 相对时间
    - 列名标准化为 linear_acceleration_x/y/z, angular_velocity_x/y/z
    - 保留原始采样率，不做重采样

    Args:
        imu_path: 原始 IMU CSV 文件路径
        source_start_ns: 源时间戳起始
        source_end_ns: 源时间戳结束

    Returns:
        规范化后的 DataFrame
    """
    imu = pd.read_csv(imu_path)

    # 裁剪到 Segment 范围
    imu = imu[
        (imu["timestamp_ns"] >= source_start_ns)
        & (imu["timestamp_ns"] < source_end_ns)
    ].copy()

    if len(imu) == 0:
        raise ValueError("Segment 范围内没有 IMU 数据")

    # 保留原始时间戳 → 计算相对时间
    imu["source_timestamp_ns"] = imu["timestamp_ns"].astype("int64")
    imu["timestamp_ns"] = imu["source_timestamp_ns"] - source_start_ns

    # 列名标准化
    imu = imu.rename(columns={
        "ax": "linear_acceleration_x",
        "ay": "linear_acceleration_y",
        "az": "linear_acceleration_z",
        "gx": "angular_velocity_x",
        "gy": "angular_velocity_y",
        "gz": "angular_velocity_z",
    })

    # 重排列顺序
    cols = [
        "timestamp_ns",
        "source_timestamp_ns",
        "linear_acceleration_x",
        "linear_acceleration_y",
        "linear_acceleration_z",
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
    ]
    imu = imu[cols]

    return imu


def normalize_imu_df(
    imu: pd.DataFrame,
    source_start_ns: int,
    source_end_ns: int,
) -> pd.DataFrame:
    """规范化 IMU 数据 — 直接接受 DataFrame。

    用于 IMU 数据已从非 CSV 来源（如 MCAP）加载的场景。

    Args:
        imu: 原始 IMU DataFrame，列: timestamp_ns, ax, ay, az, gx, gy, gz
        source_start_ns: 源时间戳起始
        source_end_ns: 源时间戳结束

    Returns:
        规范化后的 DataFrame
    """
    imu = imu[
        (imu["timestamp_ns"] >= source_start_ns)
        & (imu["timestamp_ns"] < source_end_ns)
    ].copy()

    if len(imu) == 0:
        raise ValueError("Segment 范围内没有 IMU 数据")

    imu["source_timestamp_ns"] = imu["timestamp_ns"].astype("int64")
    imu["timestamp_ns"] = imu["source_timestamp_ns"] - source_start_ns

    imu = imu.rename(columns={
        "ax": "linear_acceleration_x",
        "ay": "linear_acceleration_y",
        "az": "linear_acceleration_z",
        "gx": "angular_velocity_x",
        "gy": "angular_velocity_y",
        "gz": "angular_velocity_z",
    })

    cols = [
        "timestamp_ns",
        "source_timestamp_ns",
        "linear_acceleration_x",
        "linear_acceleration_y",
        "linear_acceleration_z",
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
    ]
    return imu[cols]


def write_imu(imu: pd.DataFrame, output_dir: str,
              stream_id: str = "ego_imu") -> str:
    """写出规范化 IMU 为 Parquet。

    Args:
        imu: 规范化后的 IMU DataFrame
        output_dir: Prepared Segment 根目录
        stream_id: IMU 流标识，文件名生成为 {stream_id}.parquet

    Returns:
        输出文件路径
    """
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"{stream_id}.parquet"
    imu.to_parquet(str(output_path), index=False)
    return str(output_path)
