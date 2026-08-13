"""逐帧 Pipeline 与状态/BBox Writer 的编排边界（WiLoR 单后端）。"""

from __future__ import annotations

from dataclasses import dataclass

from zpds.hands.contracts import BBoxWriter, FrameStatusWriter
from zpds.hands.frame_artifacts import (
    InferenceArtifactContext,
    ParquetBBoxWriter,
    ParquetFrameStatusWriter,
)


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
    *,
    frame_status_path: str | None,
    bbox_path: str | None,
    context: InferenceArtifactContext | None = None,
) -> InferenceWriterBundle:
    """创建 WiLoR 正式 Parquet Writer（单后端，无 MediaPipe 空 Writer）。"""
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


__all__ = [
    "InferenceWriterBundle",
    "create_inference_writers",
]
