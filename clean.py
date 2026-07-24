"""
墨现 (Guida) 数据清洗 — 主程序

检查项：
  1. 读取 meta.json
  2. 统计 index.jsonl 帧数
  3. 帧数一致性
  4. 时间戳间隔
  5. 视频基本信息
  6. 坏帧
  7. 黑屏
  8. 读取 IMU
  9. IMU 中断
 10. 生成清洗报告
"""

import os
import reader
from timestamp_checker import check_frame_count, check_timestamp_gaps
from video_checker import check_video_info, check_bad_frames, check_black_frames
from imu_checker import check_imu_gaps
from report import generate_report

# ============================================================
# 配置
# ============================================================
DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "egos", "墨现")
COLR_PATH = os.path.join(DATASET, "color.mp4")
DEPTH_PATH = os.path.join(DATASET, "depth.mp4")


def main():
    # ---- Step 1: 读取 meta.json ----
    print("=" * 50)
    print("Step 1: 读取 meta.json")
    print("=" * 50)

    meta = reader.read_meta(DATASET)
    print(f"Device:            {meta['device']}")
    print(f"FPS:               {meta['fps']}")
    print(f"Resolution:        {meta['width']}×{meta['height']}")
    print(f"Frame Count (meta):{meta['frame_count']}")
    print(f"Dropped Frames:    {meta['dropped_frames']}")
    print(f"IMU Sample Rate:   {meta['imu_sample_rate']} Hz")

    # ---- Step 2: 统计 index.jsonl ----
    print("\n" + "=" * 50)
    print("Step 2: 统计 index.jsonl")
    print("=" * 50)

    index = reader.read_index(DATASET)
    print(f"Index 帧数: {index['frame_count']}")

    # ---- Step 3: 帧数比较 ----
    print("\n" + "=" * 50)
    print("Step 3: 帧数比较")
    print("=" * 50)

    fc_result = check_frame_count(index["frame_count"], meta["frame_count"])
    if fc_result["equal"]:
        print(f"帧数一致: Index {fc_result['index_count']} == Meta {fc_result['meta_count']}")
    else:
        print(f"帧数不一致! Index: {fc_result['index_count']}, Meta: {fc_result['meta_count']}, 差: {fc_result['error']}")

    # ---- Step 4: 时间戳间隔检测 ----
    print("\n" + "=" * 50)
    print("Step 4: 时间戳间隔检测")
    print("=" * 50)

    ts_result = check_timestamp_gaps(index["timestamps"])
    if ts_result["gap_count"] == 0:
        print("时间戳正常，无异常间隔")
    else:
        for idx, gap_ms in ts_result["gap_list"][:ts_result["max_print"]]:
            print(f"异常: Frame {idx}, Gap = {gap_ms:.1f} ms")
        print(f"共发现 {ts_result['gap_count']} 处时间戳异常")

    # ---- Step 5: 视频基本信息 ----
    print("\n" + "=" * 50)
    print("Step 5: 视频完整性检查")
    print("=" * 50)

    rgb_info = check_video_info(COLR_PATH)
    print(f"[RGB]   Frame Count: {rgb_info['frame_count']}")
    print(f"[RGB]   FPS:         {rgb_info['fps']}")
    print(f"[RGB]   Resolution:  {rgb_info['width']}×{rgb_info['height']}")

    depth_info = check_video_info(DEPTH_PATH)
    if depth_info["frame_count"] > 0:
        print(f"[Depth] Frame Count: {depth_info['frame_count']}")
        print(f"[Depth] FPS:         {depth_info['fps']}")
        print(f"[Depth] Resolution:  {depth_info['width']}×{depth_info['height']}")
    else:
        print("[Depth] 文件不存在")

    # ---- Step 6: 坏帧检测 ----
    print("\n" + "=" * 50)
    print("Step 6: 坏帧检测")
    print("=" * 50)

    bad = check_bad_frames(COLR_PATH)
    print(f"坏帧: {bad}")

    # ---- Step 7: 黑屏检测 ----
    print("\n" + "=" * 50)
    print("Step 7: 黑屏检测")
    print("=" * 50)

    black_result = check_black_frames(COLR_PATH)
    if black_result["black_count"] == 0:
        print("无黑屏帧")
    else:
        for idx in black_result["black_list"][:black_result["max_print"]]:
            print(f"Frame {idx}: Black Frame")
        print(f"共 {black_result['black_count']} 帧黑屏")

    # ---- Step 8: 读取 IMU ----
    print("\n" + "=" * 50)
    print("Step 8: 读取 IMU")
    print("=" * 50)

    imu = reader.read_imu(DATASET)
    print(imu.head())
    print(f"\nIMU 总行数: {len(imu)}")
    print(f"列: {list(imu.columns)}")

    # ---- Step 9: IMU 中断检测 ----
    print("\n" + "=" * 50)
    print("Step 9: IMU 中断检测")
    print("=" * 50)

    imu_result = check_imu_gaps(imu)
    print(f"IMU 唯一时间戳数: {imu_result['unique_count']}")
    print(f"IMU 正常间隔: {imu_result['normal_gap_s']:.4f} s")

    if imu_result["gap_count"] == 0:
        print("IMU 无中断")
    else:
        for idx, gap_s in imu_result["gap_list"][:imu_result["max_print"]]:
            print(f"IMU异常: Sample {idx}, Gap = {gap_s:.4f} s")
        print(f"共 {imu_result['gap_count']} 处 IMU 中断")

    # ---- Step 10: 生成清洗报告 ----
    print("\n" + "=" * 50)
    print("Step 10: 生成清洗报告")
    print("=" * 50)

    results = {
        "device": meta["device"],
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "frame_count_meta": meta["frame_count"],
        "dropped_frames": meta["dropped_frames"],
        "imu_sample_rate": meta["imu_sample_rate"],
        "rgb_frame_count": rgb_info["frame_count"],
        "depth_frame_count": depth_info["frame_count"],
        "index_frame_count": index["frame_count"],
        "ts_error": fc_result["error"],
        "ts_gap_count": ts_result["gap_count"],
        "bad_frame": bad,
        "black_frame": black_result["black_count"],
        "imu_gap_count": imu_result["gap_count"],
    }

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_report.txt")
    result = generate_report(results, report_path)

    print(f"\n报告已生成: {report_path}")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
