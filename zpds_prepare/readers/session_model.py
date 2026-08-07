"""
Session 统一数据模型。

read_session() 返回一个 Session 对象，包含 video_streams、imu_streams、
annotation_streams 和 time_series_streams 字典。
调用方不再分别调用 read_index_frames() / read_imu() / get_color_*()，
而是从 Session 中按流 ID 获取所需数据。
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
class DepthStream:
    """原始深度流。

    深度不放入 ``video_streams``，避免通用 RGB 转码器将 uint16 深度静默转换
    为 8 位彩色视频。Prepared 写出阶段会把该流保存为无损 PNG 序列，并为
    每个输出帧生成可追溯的 sample map。
    """

    stream_id: str
    timestamps_ns: list[int]
    index_frames: list[dict[str, Any]]
    source_files: list[Path]
    source_kind: str
    fps: float
    width: int = 0
    height: int = 0
    frame_count: int = 0
    dtype: str = "unknown"
    unit: str = "unknown"
    invalid_value: int | float | None = None
    frame_id: str = "depth_optical_frame"
    depth_frames: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
class TimeSeriesStream:
    """通用时序流——机器人关节状态、力传感器、末端位姿、控制指令等。

    与 VideoStream / ImuStream / AnnotationStream 并列。
    不绑定特定模态，供 A2D / 磁编码器 / 力传感器 / VIO 等数据源复用。

    Attributes:
        stream_id: 流标识，如 "robot_state", "robot_action", "gripper_state"
        modality:   模态标签，如 "joint_state", "joint_command", "force", "pose"
        role:       角色标签，如 "state", "action", "sensor"
        source_path: 原始数据文件路径
        timestamps_ns: 有序纳秒时间戳列表
        rows:       行数据（list[list] 或 numpy ndarray），按 timestamp 对齐
        fields:     字段描述 [{name, dtype, unit?}, ...]
        expected_rate_hz: 标称采样率
        frame_id:   坐标系 frame（如 "robot_base", "left_gripper"）
        clock_id:   时间基准标识（默认 "source_clock"）
        metadata:   附加元数据
    """
    stream_id: str
    modality: str
    role: str

    source_path: Path
    timestamps_ns: list[int]
    rows: Any  # list[list[float]] | np.ndarray

    fields: list[dict[str, Any]] = field(default_factory=list)
    expected_rate_hz: float | None = None
    frame_id: str | None = None
    clock_id: str = "source_clock"

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        """样本数。"""
        return len(self.timestamps_ns)

    @property
    def num_fields(self) -> int:
        """字段数（每行维度）。"""
        return len(self.fields)


@dataclass
class AudioStream:
    """单个音频流的数据（foxglove.CompressedAudio → 统一 AudioStream）。

    Attributes:
        stream_id: 流标识，如 "ego_audio", "robot0_audio"
        packets: 解码后的音频包列表 [{timestamp_ns, data, format, log_time_ns}, ...]
        sample_rate_hz: 源采样率（Opus 内部固定 48000）
        channels: 声道数（1=mono, 2=stereo）
        format: 压缩格式（如 "opus"）
    """

    stream_id: str
    packets: list[dict[str, Any]] = field(default_factory=list)
    sample_rate_hz: int = 48000
    channels: int = 1
    format: str = "opus"

    @property
    def num_packets(self) -> int:
        return len(self.packets)

    @property
    def duration_ns(self) -> int:
        if len(self.packets) < 2:
            return 0
        return self.packets[-1]["timestamp_ns"] - self.packets[0]["timestamp_ns"]


@dataclass
class Session:
    """一次采集 Session 的全部流数据。

    Attributes:
        session_id: 会话标识
        source_path: 原始数据路径
        meta: 扁平化元数据 dict (device, fps, width, height, frame_count, ...)
        video_streams: {stream_id: VideoStream}
        depth_streams: {stream_id: DepthStream}
        imu_streams: {stream_id: ImuStream}
        annotation_streams: {stream_id: AnnotationStream}
        time_series_streams: {stream_id: TimeSeriesStream}
    """
    session_id: str
    source_path: str
    meta: dict
    video_streams: dict[str, VideoStream] = field(default_factory=dict)
    depth_streams: dict[str, DepthStream] = field(default_factory=dict)
    imu_streams: dict[str, ImuStream] = field(default_factory=dict)
    audio_streams: dict[str, AudioStream] = field(default_factory=dict)
    annotation_streams: dict[str, AnnotationStream] = field(default_factory=dict)
    time_series_streams: dict[str, TimeSeriesStream] = field(default_factory=dict)

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
