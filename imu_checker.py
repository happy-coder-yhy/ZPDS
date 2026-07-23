"""
IMU 数据检查：读取信息、中断检测。
"""

import numpy as np
import pandas as pd


def check_imu_gaps(
    imu: pd.DataFrame,
    gap_threshold_s: float = 0.06,
    max_print: int = 10,
) -> dict:
    """检测 IMU 时间戳中断。

    IMU 同一时间戳可能有多行（加速度/陀螺仪交错写入），
    先去重再检测间隔。阈值默认 0.06s（≈3× 标称 50Hz 周期）。

    Args:
        imu: IMU DataFrame（含 timestamp_ns 列）
        gap_threshold_s: 超过此秒数视为中断
        max_print: 打印前 N 条异常

    Returns:
        {
            "unique_count": int,
            "normal_gap_s": float,
            "gap_count": int,
            "gap_list": [(sample_idx, gap_s), ...],
            "max_print": int,
        }
    """
    unique_times = np.sort(imu["timestamp_ns"].unique())
    unique_count = len(unique_times)

    # 计算正常间隔（取中位数）
    if unique_count <= 1:
        normal_gap_s = 0.0
    else:
        normal_gap_ns = float(np.median(np.diff(unique_times)))
        normal_gap_s = normal_gap_ns / 1_000_000_000

    gap_list = []
    for i in range(1, unique_count):
        gap_ns = unique_times[i] - unique_times[i - 1]
        gap_s = gap_ns / 1_000_000_000
        if gap_s > gap_threshold_s:
            gap_list.append((i, gap_s))

    return {
        "unique_count": unique_count,
        "normal_gap_s": normal_gap_s,
        "gap_count": len(gap_list),
        "gap_list": gap_list,
        "max_print": max_print,
    }
