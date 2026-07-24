"""
Session 统一数据模型。

read_session() 返回一个 Session 对象，包含 video_streams 和 imu_streams 字典。
调用方不再分别调用 read_index_frames() / read_imu() / get_color_*()，
而是从 Session 中按流 ID 获取所需数据。
"""

from dataclasses import dataclass, field
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
class Session:
    """一次采集 Session 的全部流数据。

    Attributes:
        session_id: 会话标识
        source_path: 原始数据路径
        meta: 扁平化元数据 dict (device, fps, width, height, frame_count, ...)
        video_streams: {stream_id: VideoStream}
        imu_streams: {stream_id: ImuStream}
    """
    session_id: str
    source_path: str
    meta: dict
    video_streams: dict[str, VideoStream] = field(default_factory=dict)
    imu_streams: dict[str, ImuStream] = field(default_factory=dict)

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
