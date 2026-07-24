"""ffprobe 解析：帧数、码流、元数据提取。"""


import cv2

from zpds.adapters.base import BaseAdapter
from zpds.adapters.common import require_file, source_asset
from zpds.adapters.contracts import IssueLevel, ValidationIssue, ValidationReport
from zpds.core.types import (
    ClockDescriptor,
    ClockDomain,
    SessionInventory,
    SourceStream,
    StreamKind,
)

from .decoder import VideoDecoder


class VideoInspector(BaseAdapter):
    """视频容器探测器。"""

    def inspect(self, path: str) -> SessionInventory:
        source = require_file(path)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Unable to open video: {source}")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((codec_value >> (8 * index)) & 0xFF) for index in range(4))
        finally:
            capture.release()
        duration = frames / fps if fps > 0 else 0.0
        root = source.parent
        return SessionInventory(
            session_id=source.stem,
            source_profile="video",
            session_uri=str(source),
            assets=[source_asset(source, root, required=True)],
            streams=[
                SourceStream(
                    kind=StreamKind.COLOR,
                    stream_id="video",
                    role="observation",
                    clock_id="container_time",
                    width=width,
                    height=height,
                    fps=fps if fps > 0 else None,
                    codec=codec.strip("\x00") or None,
                    container=source.suffix.lower().lstrip("."),
                )
            ],
            clocks=[
                ClockDescriptor(
                    clock_id="container_time",
                    domain=ClockDomain.CUSTOM_EPOCH,
                    source="container presentation timestamp",
                )
            ],
            total_frames=frames,
            duration_s=duration,
            clock_domain=ClockDomain.CUSTOM_EPOCH,
        )

    def validate(self, path: str) -> ValidationReport:
        try:
            source = require_file(path)
            with VideoDecoder(str(source)) as decoder:
                decoder.read_frame(0)
        except (FileNotFoundError, OSError, IndexError) as error:
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="video_open_or_decode_failed",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(path),
                    ),
                )
            )
        return ValidationReport(checked_assets=1, decoded_records=1)

    def scan(self, path: str) -> ValidationReport:
        decoded = 0
        issues: list[ValidationIssue] = []
        try:
            for decoded, _, _ in VideoDecoder(path).iter_frames():
                pass
            decoded = decoded + 1 if decoded or self.inspect(path).total_frames else 0
        except OSError as error:
            issues.append(
                ValidationIssue(
                    code="video_decode_failed",
                    level=IssueLevel.ERROR,
                    message=str(error),
                    path=str(path),
                )
            )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=1,
            checked_records=decoded,
            decoded_records=decoded,
        )
