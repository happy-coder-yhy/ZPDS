"""
EPIC-KITCHENS-100 Reader — 读取视频 + hand-object/mask Pickle 标注。

Pickle 格式:
  - list[bytes] — 每条是 protobuf 编码的帧级检测结果
  - 见 scripts/inspect_epic_pickle.py 探测结论

用法:
    session = read_session(
        source="path/to/P01_01.mp4",
        config={
            "hand_object_path": "path/to/hand-objects/P01/P01_01.pkl",
            "mask_path": "path/to/masks/P01/P01_01.pkl",   # 可选
        },
    )
"""

from __future__ import annotations

import json
import pickle
import struct
import subprocess
from pathlib import Path

import numpy as np

from zpds_prepare.readers.session_model import Session, VideoStream, AnnotationStream
from zpds_prepare.readers.epic_inventory import parse_epic_id


# ---- protobuf 解析 ----

def _parse_varint(b: bytes, offset: int) -> tuple[int, int]:
    """解析 protobuf varint，返回 (value, new_offset)。"""
    val = 0
    shift = 0
    while offset < len(b):
        byte = b[offset]
        val |= (byte & 0x7F) << shift
        offset += 1
        if (byte & 0x80) == 0:
            break
        shift += 7
    return val, offset


def _parse_proto_entry(entry: bytes) -> dict:
    """递归解析一条 protobuf 编码的检测记录。

    wire type 0 → varint
    wire type 2 → length-delimited (递归解析或 UTF-8 解码)
    wire type 5 → 32-bit float

    Returns:
        嵌套 dict，key 为 field_number (int)
    """
    offset = 0
    result: dict = {}

    while offset < len(entry):
        tag = entry[offset]
        field_num = tag >> 3
        wire_type = tag & 7
        offset += 1

        if wire_type == 0:  # varint
            val, offset = _parse_varint(entry, offset)
            result[field_num] = val

        elif wire_type == 5:  # 32-bit fixed (float)
            if offset + 4 <= len(entry):
                val = struct.unpack("<f", entry[offset:offset + 4])[0]
                offset += 4
                result[field_num] = val

        elif wire_type == 2:  # length-delimited
            length, offset = _parse_varint(entry, offset)
            payload = entry[offset:offset + length]
            offset += length

            # 尝试 UTF-8 解码
            try:
                text = payload.decode("utf-8")
                if text.isprintable():
                    result[field_num] = text
                    continue
            except UnicodeDecodeError:
                pass

            # 递归解析为嵌套消息
            try:
                nested = _parse_proto_entry(payload)
                # 空 dict 表示递归解析未发现任何 protobuf 字段
                # → 这是二进制数据 (如 RLE counts)，保留原始 bytes
                result[field_num] = nested if nested else payload
            except Exception:
                # 解析失败 → 保留原始 bytes
                result[field_num] = payload

    return result


# ---- 标注记录提取 ----

def _extract_bbox(msg: dict | None, default: float = 0.0) -> list[float]:
    """从 protobuf 子消息提取 bbox [x1, y1, x2, y2]。

    field 1→x1, field 2→y1, field 3→x2, field 4→y2.
    缺失字段填充 default。
    """
    if msg is None:
        return [default] * 4
    return [
        float(msg.get(i, default)) for i in (1, 2, 3, 4)
    ]


def _decode_hand_object_pickle(pkl_path: str) -> list[dict]:
    """解析 hand-object Pickle 为帧级记录列表。

    Returns:
        [{frame_index (0-based), hand_bbox, hand_score,
          object_bbox, object_score, object_class, contact_offset}, ...]
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    records: list[dict] = []

    for entry_bytes in raw:
        parsed = _parse_proto_entry(entry_bytes)

        # frame_number (1-based in pickle)
        frame_1based = parsed.get(2, 0)

        # hand (field_4)
        hand_msg = parsed.get(4, {})
        hand_bbox_msg = hand_msg.get(1, {}) if isinstance(hand_msg, dict) else {}
        hand_bbox = _extract_bbox(hand_bbox_msg)
        hand_score = float(hand_msg.get(2, 0.0)) if isinstance(hand_msg, dict) else 0.0

        record: dict = {
            "frame_index": frame_1based - 1,  # 转为 0-based
            "hand_bbox": hand_bbox,
            "hand_score": hand_score,
        }

        # object (field_3, 可选 — 仅交互帧存在)
        obj_msg = parsed.get(3)
        if obj_msg and isinstance(obj_msg, dict):
            obj_bbox_msg = obj_msg.get(1, {}) if isinstance(obj_msg.get(1), dict) else {}
            obj_bbox = _extract_bbox(obj_bbox_msg)
            obj_score = float(obj_msg.get(2, 0.0))
            obj_class = obj_msg.get(3, 0)
            contact_msg = obj_msg.get(4, {}) if isinstance(obj_msg.get(4), dict) else {}
            contact_offset = [
                float(contact_msg.get(1, 0.0)),
                float(contact_msg.get(2, 0.0)),
            ] if contact_msg else [0.0, 0.0]

            record.update({
                "object_bbox": obj_bbox,
                "object_score": obj_score,
                "object_class": obj_class,
                "contact_offset": contact_offset,
            })

        records.append(record)

    return records


def _decode_mask_pickle(pkl_path: str) -> list[dict]:
    """解析 mask Pickle 为帧级记录列表。

    Returns:
        [{frame_index (0-based), mask_bbox, mask_rle_bytes, mask_score, mask_class}, ...]
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    records: list[dict] = []

    for entry_bytes in raw:
        parsed = _parse_proto_entry(entry_bytes)

        frame_1based = parsed.get(2, 0)

        mask_msg = parsed.get(3)
        if mask_msg and isinstance(mask_msg, dict):
            mask_bbox_msg = mask_msg.get(1, {}) if isinstance(mask_msg.get(1), dict) else {}
            mask_bbox = _extract_bbox(mask_bbox_msg)
            mask_rle_raw = mask_msg.get(2, b"")
            mask_score = float(mask_msg.get(3, 0.0))
            mask_class = mask_msg.get(4, 0)

            records.append({
                "frame_index": frame_1based - 1,
                "mask_bbox": mask_bbox,
                "mask_rle_bytes": mask_rle_raw,   # raw COCO RLE counts bytes
                "mask_score": mask_score,
                "mask_class": mask_class,
            })

    return records


# ---- ffprobe 视频探测 ----

def probe_video(video_path: Path) -> dict:
    """获取视频流元信息（优先 ffprobe，回退 OpenCV）。

    Returns:
        {"width": int, "height": int, "fps": float, "nb_frames": int, "duration_s": float}
    """
    import shutil

    # 优先 ffprobe（帧数/时长精确）
    if shutil.which("ffprobe"):
        return _probe_with_ffprobe(video_path)

    # 回退 OpenCV（跨平台，无外部依赖）
    return _probe_with_opencv(video_path)


def _probe_with_ffprobe(video_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {result.stderr.strip()}")

    info = json.loads(result.stdout)
    stream = info.get("streams", [{}])[0]

    width = stream.get("width", 0)
    height = stream.get("height", 0)
    nb_frames = int(stream.get("nb_frames", 0))

    fps_str = stream.get("avg_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    duration_s = float(stream.get("duration", 0))

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "nb_frames": nb_frames,
        "duration_s": duration_s,
    }


def _probe_with_opencv(video_path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = nb_frames / fps if fps > 0 else 0.0
    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "nb_frames": nb_frames,
        "duration_s": duration_s,
    }


def read_frame_pts_ns(video_path: Path, fast: bool = False) -> list[int]:
    """使用 ffprobe 逐帧提取 PTS (Presentation TimeStamp)，返回纳秒列表。

    若 ffprobe 提取失败（如文件无索引），回退为基于帧率和帧数的推算。
    fast=True 时直接使用推算（EPIC CFR 视频无需逐帧提取）。
    """
    # fast 模式：EPIC CFR 视频直接用帧率推算（秒级，跳过 ffprobe 空等）
    if fast:
        probe = probe_video(video_path)
        nb = probe["nb_frames"]
        fps = probe["fps"]
        if nb <= 0 or fps <= 0:
            return []
        return [int(i * 1e9 / fps) for i in range(nb)]

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0",
        str(video_path),
    ]
    # ffprobe 逐帧 PTS 对大视频（30min+ 59.94fps）极慢，设 30s 超时回退推算
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        result = None

    if result is not None and result.returncode == 0 and result.stdout.strip():
        timestamps_ns = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seconds = float(line)
                timestamps_ns.append(round(seconds * 1_000_000_000))
            except ValueError:
                continue
        return timestamps_ns

    # 回退: 基于 frame_count + fps 推算
    probe = probe_video(video_path)
    nb_frames = probe["nb_frames"]
    fps = probe["fps"]
    if nb_frames <= 0 or fps <= 0:
        return []

    interval_ns = int(1e9 / fps)
    return [i * interval_ns for i in range(nb_frames)]


# ---- 主入口 ----

def read_session(source: str, config: dict | None = None) -> Session:
    """读取 EPIC-KITCHENS-100 单条视频 + 标注，返回统一 Session。

    Args:
        source: 视频文件路径 (.mp4)
        config: 可选配置字典，支持:
            - hand_object_path: hand-object .pkl 路径
            - mask_path: mask .pkl 路径

    Returns:
        Session 对象:
          - video_streams: 1 个 ("ego_rgb")
          - imu_streams: 0 个
          - annotation_streams: 1-2 个 ("hand_objects" / "masks")
    """
    if config is None:
        config = {}
    fast_pts = config.get("fast_pts", True)  # EPIC CFR 视频默认快速推算

    video_path = Path(source)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {source}")

    # ---- 解析 video_id ----
    participant_id, video_id = parse_epic_id(video_path)

    # ---- 探测视频 ----
    probe = probe_video(video_path)
    width = probe["width"]
    height = probe["height"]
    fps = probe["fps"]
    nb_frames = probe["nb_frames"]

    # ---- 提取帧时间戳 ----
    timestamps_ns = read_frame_pts_ns(video_path, fast=fast_pts)
    if not timestamps_ns:
        raise RuntimeError(f"无法提取视频帧时间戳: {video_path}")

    # ---- 构建 index_frames ----
    index_frames = [
        {
            "seq": i,
            "timestamp_ns": ts,
        }
        for i, ts in enumerate(timestamps_ns)
    ]

    # ---- 构建 VideoStream ----
    video_streams = {
        "ego_rgb": VideoStream(
            stream_id="ego_rgb",
            timestamps_ns=timestamps_ns,
            index_frames=index_frames,
            video_path=str(video_path),
            fps=fps,
            width=width,
            height=height,
            frame_count=nb_frames or len(timestamps_ns),
        ),
    }

    # ---- 构建 AnnotationStream(s) ----
    annotation_streams: dict[str, AnnotationStream] = {}

    # Hand-object
    ho_path = config.get("hand_object_path")
    if ho_path:
        ho_records = _decode_hand_object_pickle(ho_path)
        annotation_streams["hand_objects"] = AnnotationStream(
            stream_id="hand_objects",
            annotation_type="hand_object_detection",
            source_path=Path(ho_path),
            source_video_stream_id="ego_rgb",
            frame_index_base=0,
            bbox_format="xyxy_normalized",
            records=ho_records,
            metadata={
                "ground_truth_status": "model_generated",
                "source_dataset": "EPIC-KITCHENS-100-derived",
                "total_frames": len(ho_records),
                "interaction_frames": sum(
                    1 for r in ho_records if "object_bbox" in r
                ),
            },
        )

    # Mask
    mask_path = config.get("mask_path")
    if mask_path:
        mask_records = _decode_mask_pickle(mask_path)
        annotation_streams["masks"] = AnnotationStream(
            stream_id="masks",
            annotation_type="instance_segmentation",
            source_path=Path(mask_path),
            source_video_stream_id="ego_rgb",
            frame_index_base=0,
            bbox_format="xyxy_normalized",
            records=mask_records,
            metadata={
                "ground_truth_status": "model_generated",
                "source_dataset": "EPIC-KITCHENS-100-derived",
                "total_frames_with_mask": len(mask_records),
            },
        )

    # ---- 构建 Session ----
    return Session(
        session_id=f"epic_{video_id}",
        source_path=str(video_path),
        meta={
            "device": "EPIC-KITCHENS-100",
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": nb_frames or len(timestamps_ns),
            "participant_id": participant_id,
            "video_id": video_id,
        },
        video_streams=video_streams,
        imu_streams={},
        annotation_streams=annotation_streams,
    )


__all__ = [
    "probe_video",
    "read_frame_pts_ns",
    "read_session",
]
