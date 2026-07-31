"""逐帧 Pipeline 与状态/BBox Writer 的编排边界。"""

from __future__ import annotations

from dataclasses import dataclass

from zpds.hands.contracts import BBoxWriter, FrameStatusWriter
from zpds.hands.frame_artifacts import (
    InferenceArtifactContext,
    ParquetBBoxWriter,
    ParquetFrameStatusWriter,
)


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
    context: InferenceArtifactContext | None = None,
) -> InferenceWriterBundle:
    """创建 MediaPipe 空 Writer 或 WiLoR 正式 Parquet Writer。"""
    if primary_model == "mediapipe":
        return InferenceWriterBundle(
            frame_status=NullInferenceWriter(),
            bbox=NullInferenceWriter(),
        )
    if primary_model == "wilor":
        if frame_status_path is None or bbox_path is None:
            raise ValueError("WiLoR frame-status 和 BBox 输出路径不能为空")
        if context is None:
            raise ValueError("WiLoR Writer 缺少 InferenceArtifactContext")
        return InferenceWriterBundle(
            frame_status=ParquetFrameStatusWriter(
                frame_status_path,
                context,
            ),
            bbox=ParquetBBoxWriter(bbox_path, context),
        )
    raise ValueError(f"未知 Hands 主模型: {primary_model!r}")


__all__ = [
    "InferenceWriterBundle",
    "NullInferenceWriter",
    "create_inference_writers",
]
