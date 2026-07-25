"""
Session 统一数据模型。

read_session() 返回一个 Session 对象，包含 video_streams、imu_streams 和
annotation_streams 字典。调用方不再分别调用 read_index_frames() / read_imu() /
get_color_*()，而是从 Session 中按流 ID 获取所需数据。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoStream:
    """单个视频流的数据。

    Attributes:
        stream_id: 流标识，如 "ego_rgb", "camera0", "camera1"
        timestamps_ns: 有序纳秒时间戳列表
        index_frames: 帧索引列表 [{seq, timestamp_ns, ...}, ...]
        video_path: 视频文件路径
        fps: 标称帧率
        width: 帧宽
        height: 帧高
        frame_count: 总帧数
    """
    stream_id: str
    timestamps_ns: list[int]
    index_frames: list[dict]
    video_path: str
    fps: float
    width: int = 0
    height: int = 0
    frame_count: int = 0


@dataclass
class ImuStream:
    """单个 IMU 流的数据。

    Attributes:
        stream_id: 流标识，如 "ego_imu", "robot0_imu"
        dataframe: pandas DataFrame (timestamp_ns, ax, ay, az, gx, gy, gz)
        sample_rate_hz: 标称采样率
    """
    stream_id: str
    dataframe: Any  # pd.DataFrame
    sample_rate_hz: float


@dataclass
class AnnotationStream:
    """单个标注流的数据。

    承载 EPIC-KITCHENS-100 等数据源的帧级标注（bbox、类别、掩码等）。

    Attributes:
        stream_id: 流标识，如 "hand_objects", "masks"
        annotation_type: 标注类型，如 "hand_object_detection", "mask"
        source_path: 标注 Pickle 文件路径
        source_video_stream_id: 关联的 VideoStream.stream_id
        frame_index_base: 标注帧号的基数 (0 = 0-based, 1 = 1-based)
        bbox_format: bbox 格式，如 "xyxy_normalized", "xyxy_pixels"
        records: 帧级标注记录列表
        metadata: 附加元数据
    """
    stream_id: str
    annotation_type: str
    source_path: Path
    source_video_stream_id: str
    frame_index_base: int
    bbox_format: str | None
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """一次采集 Session 的全部流数据。

    Attributes:
        session_id: 会话标识
        source_path: 原始数据路径
        meta: 扁平化元数据 dict (device, fps, width, height, frame_count, ...)
        video_streams: {stream_id: VideoStream}
        imu_streams: {stream_id: ImuStream}
        annotation_streams: {stream_id: AnnotationStream}
    """
    session_id: str
    source_path: str
    meta: dict
    video_streams: dict[str, VideoStream] = field(default_factory=dict)
    imu_streams: dict[str, ImuStream] = field(default_factory=dict)
    annotation_streams: dict[str, AnnotationStream] = field(default_factory=dict)

    @property
    def primary_video(self) -> VideoStream:
        """返回第一个视频流（检测器默认使用的 RGB 流）。"""
        if not self.video_streams:
            raise ValueError("Session 中没有视频流")
        return next(iter(self.video_streams.values()))

    @property
    def primary_imu(self) -> ImuStream:
        """返回第一个 IMU 流。"""
        if not self.imu_streams:
            raise ValueError("Session 中没有 IMU 流")
        return next(iter(self.imu_streams.values()))

    @property
    def session_start_ns(self) -> int:
        """Session 起始时间（基于主视频流）。"""
        return self.primary_video.timestamps_ns[0]

    @property
    def session_end_ns(self) -> int:
        """Session 结束时间（基于主视频流）。"""
        return self.primary_video.timestamps_ns[-1]
