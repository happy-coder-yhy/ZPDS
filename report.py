"""
生成数据清洗报告。
"""

import os


def generate_report(results: dict, output_path: str) -> str:
    """根据各模块检查结果生成清洗报告。

    Args:
        results: 包含所有检查结果的字典，结构见 clean.py
        output_path: 报告输出路径

    Returns:
        "PASS" | "FAIL"
    """
    # 汇总问题
    issues = []
    if results["ts_error"] > 0:
        issues.append(f"帧数不一致 (差{results['ts_error']})")
    if results["ts_gap_count"] > 0:
        issues.append(f"时间戳异常 ({results['ts_gap_count']}处)")
    if results["bad_frame"] > 0:
        issues.append(f"坏帧 ({results['bad_frame']})")
    if results["black_frame"] > 0:
        issues.append(f"黑屏 ({results['black_frame']})")
    if results["imu_gap_count"] > 0:
        issues.append(f"IMU中断 ({results['imu_gap_count']}处)")

    result = "PASS" if len(issues) == 0 else "FAIL"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("======== 数据清洗报告 ========\n\n")
        f.write(f"Device:            {results['device']}\n")
        f.write(f"Meta FPS:          {results['fps']}\n")
        f.write(f"Meta Resolution:   {results['width']}×{results['height']}\n")
        f.write(f"Meta Frame Count:  {results['frame_count_meta']}\n")
        f.write(f"Dropped Frames:    {results['dropped_frames']}\n")
        f.write(f"IMU Sample Rate:   {results['imu_sample_rate']} Hz\n\n")
        f.write("---------------------------------\n\n")
        f.write(f"RGB Frame:         {results['rgb_frame_count']}\n")
        f.write(f"Depth Frame:       {results['depth_frame_count']}\n")
        f.write(f"Index Frame Count: {results['index_frame_count']}\n")
        f.write(f"Timestamp Error:   {results['ts_error']}\n")
        f.write(f"Timestamp Gaps:    {results['ts_gap_count']}\n")
        f.write(f"Bad Frame:         {results['bad_frame']}\n")
        f.write(f"Black Frame:       {results['black_frame']}\n")
        f.write(f"IMU Gap:           {results['imu_gap_count']}\n\n")
        if issues:
            f.write("Issues:\n")
            for issue in issues:
                f.write(f"  - {issue}\n")
            f.write("\n")
        f.write(f"Result:            {result}\n")

    return result
