"""逐帧 Pipeline 与状态/BBox Writer 的编排边界。"""

from __future__ import annotations

from dataclasses import dataclass

from zpds.hands.contracts import BBoxWriter, FrameStatusWriter


class FrameWriterUnavailableError(RuntimeError):
    """WiLoR 逐帧正式 Writer 尚未接入。"""


class NullInferenceWriter:
    """MediaPipe 兼容路径使用的无状态 Writer。"""

    def write(self, _record: object) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class InferenceWriterBundle:
    frame_status: FrameStatusWriter
    bbox: BBoxWriter

    def __post_init__(self) -> None:
        if not isinstance(self.frame_status, FrameStatusWriter):
            raise TypeError(
                "frame_status Writer 必须实现 write() 和 close()"
            )
        if not isinstance(self.bbox, BBoxWriter):
            raise TypeError("bbox Writer 必须实现 write() 和 close()")

    def close(self) -> None:
        try:
            self.frame_status.close()
        finally:
            self.bbox.close()


def create_inference_writers(
    primary_model: str,
    *,
    frame_status_path: str | None,
    bbox_path: str | None,
) -> InferenceWriterBundle:
    """创建逐帧 Writer；WiLoR 正式 Writer 未接入时明确失败。"""
    if primary_model == "mediapipe":
        return InferenceWriterBundle(
            frame_status=NullInferenceWriter(),
            bbox=NullInferenceWriter(),
        )
    if primary_model == "wilor":
        raise FrameWriterUnavailableError(
            "WiLoR frame-status/BBox 正式 Writer 尚未接入；"
            f"目标路径为 frame_status={frame_status_path}, bbox={bbox_path}"
        )
    raise ValueError(f"未知 Hands 主模型: {primary_model!r}")


__all__ = [
    "FrameWriterUnavailableError",
    "InferenceWriterBundle",
    "NullInferenceWriter",
    "create_inference_writers",
]
