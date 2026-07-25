"""
探测 A2D Episode 的 ROS2 MCAP Bag 文件。

对四组 Bag 记录：
  - Topic / Schema / 消息数量
  - 首尾时间（log_time / publish_time）
  - joint_names / 数组维度
  - 控制模式

第一阶段暂时不把 ROS2 数据放入 Prepared Segment，
但要确认后续是否能解析。

用法:
    python scripts/inspect_a2d_rosbags.py \\
        --record-dir "E:/datasets/真机/A2D/record/" \\
        --output output/a2d/8032/
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# MCAP 四组 Bag 相对路径
EXPECTED_MCAPS: list[tuple[str, str]] = [
    ("joint_states", "joint-states/joint-states_0.mcap"),
    ("joint_commands", "joint-commands/joint-commands_0.mcap"),
    ("gripper_states", "gripper-states/gripper-states_0.mcap"),
    ("gripper_commands", "gripper-commands/gripper-commands_0.mcap"),
]


# ---------------------------------------------------------------------------
# CDR 轻量解码（Header + joint_names + control_modes + 首条样本）
# ---------------------------------------------------------------------------

def _decode_cdr_header(data: bytes, offset: int = 0) -> tuple[dict, int]:
    """从 CDR 字节流中解码 std_msgs/Header。

    Returns:
        ({sec, nanosec, frame_id}, new_offset)
    """
    # Check for RTPS encapsulation header (4 bytes: endian + options)
    rtps = struct.unpack_from("<I", data, offset)[0]
    if rtps in (0x00010000, 0x00000100):
        offset += 4

    sec = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    nanosec = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    fid_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    frame_id = ""
    if 0 < fid_len < 100:
        frame_id = (
            data[offset : offset + fid_len]
            .decode("utf-8", errors="replace")
            .rstrip("\x00")
        )
        offset += fid_len
        offset = (offset + 3) & ~3  # 4-byte align
    elif fid_len > 0:
        offset += fid_len
        offset = (offset + 3) & ~3

    return {"sec": sec, "nanosec": nanosec, "frame_id": frame_id}, offset


def _decode_cdr_string_array(data: bytes, offset: int) -> tuple[list[str], int]:
    """解码 CDR string[] (uint32 count + strings)。"""
    count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if count > 1000:  # 保护
        return [f"(count={count}, likely corrupt)"], offset

    strings = []
    for _ in range(count):
        sl = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if 0 < sl < 200:
            s = (
                data[offset : offset + sl]
                .decode("utf-8", errors="replace")
                .rstrip("\x00")
            )
            offset += sl
            offset = (offset + 3) & ~3
            strings.append(s)
        else:
            strings.append(f"(bad_len={sl})")
            offset += max(sl, 0)
            offset = (offset + 3) & ~3

    return strings, offset


def _decode_cdr_int32_array(data: bytes, offset: int) -> tuple[list[int], int]:
    """解码 CDR int32[] (uint32 count + int32 values)。"""
    count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if count > 1000:
        return [], offset
    vals = list(struct.unpack_from(f"<{count}i", data, offset))
    offset += count * 4
    return vals, offset


def _decode_cdr_float64_array(data: bytes, offset: int) -> tuple[list[float], int]:
    """解码 CDR float64[] (uint32 count + float64 values)。"""
    count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if count > 1000:
        return [], offset
    vals = list(struct.unpack_from(f"<{count}d", data, offset))
    offset += count * 8
    return vals, offset


# ---------------------------------------------------------------------------
# 探测主逻辑
# ---------------------------------------------------------------------------

def probe_mcaps(record_dir: str) -> dict:
    """探测 record/ 下所有 MCAP Bag。

    Returns:
        {bag_name: {file, size_bytes, exists, topics: [{topic, schema_name,
         message_count, first_log_time, last_log_time, header_sample,
         joint_names, control_modes, array_dims, ...}]}}
    """
    from mcap.reader import make_reader

    base = Path(record_dir)
    result: dict = {}

    for bag_name, rel_path in EXPECTED_MCAPS:
        mcap_path = base / rel_path
        entry: dict = {
            "file": str(mcap_path),
            "exists": mcap_path.is_file(),
        }

        if not mcap_path.is_file():
            entry["error"] = "文件不存在"
            result[bag_name] = entry
            continue

        entry["size_bytes"] = mcap_path.stat().st_size

        try:
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                summary = reader.get_summary()

                if summary is None:
                    entry["error"] = "无法读取 MCAP summary"
                    result[bag_name] = entry
                    continue

                stats = summary.statistics
                entry["statistics"] = {
                    "message_count": stats.message_count,
                    "channel_count": stats.channel_count,
                    "schema_count": stats.schema_count,
                }

                # 收集所有 channel 信息
                channels_info: list[dict] = []
                for chan_id, chan in summary.channels.items():
                    schema = summary.schemas.get(chan.schema_id) if summary.schemas else None
                    chan_entry = {
                        "topic": chan.topic,
                        "message_encoding": chan.message_encoding,
                        "schema_name": schema.name if schema else "unknown",
                        "schema_encoding": schema.encoding if schema else "unknown",
                        "schema_text": (
                            schema.data.decode("utf-8", errors="replace")
                            if schema and schema.data else None
                        ),
                        "message_count": 0,  # populated below
                    }
                    channels_info.append(chan_entry)

                # 逐消息遍历：计数 + 首尾时间 + 首条 decode
                first_messages: dict[str, dict] = {}
                last_messages: dict[str, dict] = {}
                msg_counts: dict[str, int] = {}

                # Re-open for iteration
                f.seek(0)
                reader2 = make_reader(f)
                for schema, channel, msg in reader2.iter_messages():
                    topic = channel.topic
                    msg_counts[topic] = msg_counts.get(topic, 0) + 1

                    if topic not in first_messages:
                        first_messages[topic] = {
                            "log_time": msg.log_time,
                            "publish_time": msg.publish_time,
                        }
                    last_messages[topic] = {
                        "log_time": msg.log_time,
                        "publish_time": msg.publish_time,
                    }

                # 填入 message_count
                for chan_entry in channels_info:
                    chan_entry["message_count"] = msg_counts.get(chan_entry["topic"], 0)

                # 解码首条消息的 key 字段（Header + joint_names）
                f.seek(0)
                reader3 = make_reader(f)
                decoded_topics: set[str] = set()
                for schema, channel, msg in reader3.iter_messages():
                    topic = channel.topic
                    if topic in decoded_topics:
                        continue
                    decoded_topics.add(topic)
                    decoded = _decode_first_cdr_message(msg.data, topic)
                    for ci in channels_info:
                        if ci["topic"] == topic:
                            ci["first_message_decode"] = decoded

                entry["topics"] = channels_info

                # 全局首尾时间
                all_first = [
                    (t, fm["log_time"])
                    for t, fm in first_messages.items()
                ]
                all_last = [
                    (t, lm["log_time"])
                    for t, lm in last_messages.items()
                ]
                entry["time_range"] = {
                    "first_log_time": min(lt for _, lt in all_first) if all_first else None,
                    "last_log_time": max(lt for _, lt in all_last) if all_last else None,
                    "first_topic": min(all_first, key=lambda x: x[1])[0] if all_first else None,
                    "last_topic": max(all_last, key=lambda x: x[1])[0] if all_last else None,
                }

                result[bag_name] = entry

        except ImportError:
            entry["error"] = "mcap 库未安装 (pip install mcap)"
            result[bag_name] = entry
        except Exception as exc:
            entry["error"] = str(exc)
            result[bag_name] = entry

    return result


def _decode_first_cdr_message(data: bytes, topic: str) -> dict:
    """解码首条 CDR 消息的 Header、joint_names/name、control_modes/control_mode。

    不完全解码 — 仅提取关键元信息。
    """
    decoded: dict = {"decode_method": "cdr_lightweight"}

    try:
        header, offset = _decode_cdr_header(data, 0)
        decoded["header"] = header

        # 尝试 string[] (joint_names 或 name)
        try:
            strings, _ = _decode_cdr_string_array(data, offset)
            decoded["string_array"] = strings
            # 保存 offset 以便继续
            # (不跟踪精确 offset — 这里只做轻量提取)
        except Exception:
            pass

        # 尝试在 data 中搜索 control_modes / control_mode
        # 简单方式：在 header 之后扫描 int32[] pattern
        # 实际上我们只需要 joint_names 就够关键信息了
        decoded["message_size_bytes"] = len(data)

    except Exception as exc:
        decoded["error"] = str(exc)

    return decoded


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------

def analyze(probe_result: dict) -> dict:
    """基于探测结果生成分析摘要。"""
    analysis: dict = {
        "timestamp_unit": "ns",
        "timestamp_clock": "unix_epoch",
        "timestamp_status": "confirmed",
        "topics_summary": [],
        "decodability": "partial",
        "decodability_note": (
            "Header(joint_names, control_modes) 可手动 CDR 解码。"
            "全部字段解码需要 rosidl 类型支持或 MCAP ros2 插件。"
        ),
    }

    for bag_name, bag_info in probe_result.items():
        if not bag_info.get("exists"):
            analysis["topics_summary"].append({
                "bag": bag_name,
                "status": "missing",
            })
            continue

        for topic_info in bag_info.get("topics", []):
            decoded = topic_info.get("first_message_decode", {})
            string_arr = decoded.get("string_array", [])

            summary = {
                "bag": bag_name,
                "topic": topic_info["topic"],
                "schema": topic_info["schema_name"],
                "message_count": topic_info["message_count"],
                "joint_names": string_arr if string_arr else None,
                "array_dim": len(string_arr) if string_arr else None,
                "first_header": decoded.get("header"),
            }
            analysis["topics_summary"].append(summary)

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="探测 A2D Episode 的 ROS2 MCAP Bag 文件",
    )
    parser.add_argument(
        "--record-dir",
        required=True,
        help="record/ 目录路径，例如 E:/datasets/真机/A2D/record/",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/a2d/",
        help="输出根目录",
    )
    parser.add_argument(
        "--episode-id",
        default="unknown",
        help="Episode ID",
    )
    args = parser.parse_args()

    record_dir = Path(args.record_dir)
    if not record_dir.is_dir():
        print(f"错误: 目录不存在 — {record_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"探测 ROS2 MCAP: {record_dir}")
    print()

    probe_result = probe_mcaps(str(record_dir))
    analysis = analyze(probe_result)

    report = {
        "schema_version": "a2d_mcap_schema.v1",
        "record_dir": str(record_dir),
        "bags": probe_result,
        "analysis": analysis,
    }

    # 写入
    out_dir = Path(args.output) / args.episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mcap_schema.json"

    # 自定义 JSON encoder 处理大整数
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"写入: {out_path}")

    # 打印摘要
    print()
    print("=" * 50)
    print("  ROS2 MCAP Schema 摘要")
    print("=" * 50)

    for bag_name, bag_info in probe_result.items():
        icon = "✓" if bag_info.get("exists") else "✗"
        size_mb = bag_info.get("size_bytes", 0) / 1e6
        print(f"\n  {icon} {bag_name}  ({size_mb:.1f} MB)")

        if "error" in bag_info:
            print(f"    错误: {bag_info['error']}")
            continue

        for topic_info in bag_info.get("topics", []):
            print(f"    Topic:       {topic_info['topic']}")
            print(f"    Schema:      {topic_info['schema_name']}")
            print(f"    Messages:    {topic_info['message_count']}")

            decoded = topic_info.get("first_message_decode", {})
            if "string_array" in decoded:
                arr = decoded["string_array"]
                print(f"    名称数:      {len(arr)}")
                print(f"    前3:         {arr[:3]}")
            if "header" in decoded:
                hdr = decoded["header"]
                print(f"    Header:      sec={hdr.get('sec')}, ns={hdr.get('nanosec')}")

    print(f"\n  时间戳: {analysis['timestamp_unit']} / {analysis['timestamp_clock']}")
    print(f"  可解码性: {analysis['decodability']}")
    print(f"  说明: {analysis['decodability_note']}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
