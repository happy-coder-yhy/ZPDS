"""MCAP info/doctor、topic inventory。"""

import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

from zpds.adapters.base import BaseAdapter
from zpds.adapters.common import (
    infer_stream_kind,
    require_file,
    require_optional_module,
    source_asset,
)
from zpds.adapters.contracts import IssueLevel, ValidationIssue, ValidationReport
from zpds.core.types import (
    CalibrationDescriptor,
    ClockDescriptor,
    ClockDomain,
    SessionInventory,
    SourceStream,
)

from .reader import McapReader

MCAP_MAGIC = b"\x89MCAP0\r\n"


class McapInspector(BaseAdapter):
    """MCAP 容器探测器。"""

    def inspect(self, path: str) -> SessionInventory:
        source = require_file(path)
        reader_module = require_optional_module("mcap.reader", "mcap")
        with source.open("rb") as file:
            summary = reader_module.make_reader(file).get_summary()
        if summary is None:
            raise ValueError(f"MCAP summary is missing: {source}")
        streams = [
            SourceStream(
                kind=infer_stream_kind(channel.topic),
                stream_id=channel.topic,
                role="observation",
                clock_id="mcap_log_time",
                topic=channel.topic,
                encoding=channel.message_encoding,
                container="mcap",
                metadata={
                    "schema_id": channel.schema_id,
                    "metadata": dict(channel.metadata),
                },
            )
            for channel in sorted(summary.channels.values(), key=lambda item: item.topic)
        ]
        calibrations = [
            CalibrationDescriptor(
                calibration_id=channel.topic.strip("/").replace("/", "_"),
                kind=(
                    "camera_intrinsics"
                    if "camera_info" in channel.topic
                    else "calibration_message"
                ),
                uri=f"mcap://{source.name}#{channel.topic}",
                format=channel.message_encoding,
                metadata={"topic": channel.topic, "schema_id": channel.schema_id},
            )
            for channel in sorted(summary.channels.values(), key=lambda item: item.topic)
            if "camera_info" in channel.topic or "calibration" in channel.topic
        ]
        statistics = summary.statistics
        message_count = int(statistics.message_count) if statistics is not None else 0
        start_ns = int(statistics.message_start_time) if statistics is not None else 0
        end_ns = int(statistics.message_end_time) if statistics is not None else 0
        duration = max(end_ns - start_ns, 0) / 1_000_000_000
        return SessionInventory(
            session_id=source.stem,
            source_profile="mcap",
            session_uri=str(source),
            assets=[source_asset(source, source.parent)],
            streams=streams,
            clocks=[
                ClockDescriptor(
                    clock_id="mcap_log_time",
                    domain=ClockDomain.DEVICE_MONOTONIC,
                    source="MCAP message.log_time",
                    authoritative=True,
                ),
                ClockDescriptor(
                    clock_id="mcap_publish_time",
                    domain=ClockDomain.DEVICE_MONOTONIC,
                    source="MCAP message.publish_time",
                    notes="Preserved separately; not silently merged with log_time",
                ),
            ],
            calibrations=calibrations,
            duration_s=duration,
            clock_domain=ClockDomain.DEVICE_MONOTONIC,
            metadata={"message_count": message_count},
        )

    def validate(self, path: str) -> ValidationReport:
        source = Path(path)
        if not source.is_file():
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="mcap_missing",
                        level=IssueLevel.FATAL,
                        message=f"MCAP file not found: {source}",
                        path=str(source),
                    ),
                )
            )
        if source.stat().st_size < len(MCAP_MAGIC) * 2:
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="mcap_too_small",
                        level=IssueLevel.ERROR,
                        message="MCAP file is too small to contain leading and trailing magic",
                        path=str(source),
                    ),
                ),
                checked_assets=1,
            )
        with source.open("rb") as file:
            header = file.read(len(MCAP_MAGIC))
            file.seek(-len(MCAP_MAGIC), 2)
            footer = file.read(len(MCAP_MAGIC))
        if header != MCAP_MAGIC or footer != MCAP_MAGIC:
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="mcap_magic_invalid",
                        level=IssueLevel.ERROR,
                        message="MCAP leading or trailing magic is invalid",
                        path=str(source),
                    ),
                ),
                checked_assets=1,
            )
        try:
            inventory = self.inspect(str(source))
        except (ImportError, OSError, ValueError) as error:
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="mcap_summary_failed",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(source),
                    ),
                ),
                checked_assets=1,
            )
        return ValidationReport(
            checked_assets=1,
            checked_records=int(inventory.metadata["message_count"]),
        )

    def scan(self, path: str) -> ValidationReport:
        checked = 0
        decoded = 0
        issues: list[ValidationIssue] = []
        try:
            for _ in McapReader(path).iter_messages():
                checked += 1
            for _ in McapReader(path).iter_decoded():
                decoded += 1
        except (ImportError, OSError, ValueError) as error:
            issues.append(
                ValidationIssue(
                    code="mcap_full_read_failed",
                    level=IssueLevel.ERROR,
                    message=str(error),
                    path=str(path),
                )
            )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=1,
            checked_records=checked,
            decoded_records=decoded,
            metadata={"raw_records": checked, "decoded_records": decoded},
        )

    def analyze_time(self, path: str) -> ValidationReport:
        """分别检查每个 topic 的 log time 与 publish time，不合并两个时钟。"""
        previous: dict[tuple[str, str], int] = {}
        regressions = {"log_time": 0, "publish_time": 0}
        equal_timestamps = {"log_time": 0, "publish_time": 0}
        checked = 0
        issues: list[ValidationIssue] = []
        try:
            for message in McapReader(path).iter_messages():
                checked += 1
                values = {
                    "log_time": message.log_time_ns,
                    "publish_time": message.publish_time_ns,
                }
                for clock_name, timestamp_ns in values.items():
                    if timestamp_ns is None:
                        continue
                    key = (message.stream_id, clock_name)
                    last = previous.get(key)
                    if last is not None:
                        if timestamp_ns < last:
                            regressions[clock_name] += 1
                        elif timestamp_ns == last:
                            equal_timestamps[clock_name] += 1
                    previous[key] = timestamp_ns
        except (ImportError, OSError, ValueError) as error:
            issues.append(
                ValidationIssue(
                    code="mcap_time_read_failed",
                    level=IssueLevel.ERROR,
                    message=str(error),
                    path=str(path),
                )
            )
        for clock_name, count in regressions.items():
            if count:
                issues.append(
                    ValidationIssue(
                        code=f"mcap_{clock_name}_regression",
                        level=IssueLevel.ERROR,
                        message=f"{clock_name} regressed {count} times within topics",
                        path=str(path),
                        details={"count": count},
                    )
                )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=1,
            checked_records=checked,
            metadata={
                "clock_domains_checked": ["log_time", "publish_time"],
                "regressions": regressions,
                "equal_timestamps": equal_timestamps,
                "topic_clock_pairs": len(previous),
            },
        )

    def scan_embedded_media(self, path: str) -> ValidationReport:
        """解码压缩图像和 H264/H265 码流，而非只解码 Protobuf 外壳。"""
        source = require_file(path)
        media_messages = 0
        decoded_images = 0
        decoded_video_frames = 0
        issues: list[ValidationIssue] = []
        packet_counts: dict[str, int] = {}
        formats: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="zpds-mcap-media-") as temporary:
            temporary_root = Path(temporary)
            bitstreams: dict[str, Path] = {}
            with ExitStack() as stack:
                handles: dict[str, BinaryIO] = {}
                try:
                    for _, channel, _, decoded in McapReader(str(source)).iter_decoded():
                        payload = getattr(decoded, "data", None)
                        media_format = str(getattr(decoded, "format", "")).lower()
                        if not isinstance(payload, bytes) or not payload:
                            continue
                        topic = str(channel.topic)
                        media_messages += 1
                        packet_counts[topic] = packet_counts.get(topic, 0) + 1
                        formats[topic] = media_format
                        if "h264" in media_format or "h265" in media_format:
                            extension = ".h265" if "h265" in media_format else ".h264"
                            bitstream = bitstreams.setdefault(
                                topic,
                                temporary_root / f"{_safe_topic_name(topic)}{extension}",
                            )
                            handle = handles.get(topic)
                            if handle is None:
                                handle = stack.enter_context(bitstream.open("wb"))
                                handles[topic] = handle
                            handle.write(payload)
                            continue
                        if any(name in media_format for name in ("png", "jpeg", "jpg")):
                            image = cv2.imdecode(
                                np.frombuffer(payload, dtype=np.uint8),
                                cv2.IMREAD_UNCHANGED,
                            )
                            if image is None:
                                issues.append(
                                    ValidationIssue(
                                        code="mcap_image_payload_decode_failed",
                                        level=IssueLevel.ERROR,
                                        message=f"Unable to decode {media_format} payload",
                                        path=str(source),
                                        stream_id=topic,
                                    )
                                )
                            else:
                                decoded_images += 1
                except (ImportError, OSError, ValueError) as error:
                    issues.append(
                        ValidationIssue(
                            code="mcap_media_extract_failed",
                            level=IssueLevel.ERROR,
                            message=str(error),
                            path=str(source),
                        )
                    )
            for topic, bitstream in bitstreams.items():
                decoded_for_topic = 0
                capture = cv2.VideoCapture(str(bitstream))
                try:
                    if not capture.isOpened():
                        issues.append(
                            ValidationIssue(
                                code="mcap_video_bitstream_open_failed",
                                level=IssueLevel.ERROR,
                                message=f"Unable to open reconstructed {formats[topic]} bitstream",
                                path=str(source),
                                stream_id=topic,
                            )
                        )
                        continue
                    while True:
                        ok, _ = capture.read()
                        if not ok:
                            break
                        decoded_for_topic += 1
                finally:
                    capture.release()
                decoded_video_frames += decoded_for_topic
                if decoded_for_topic == 0:
                    issues.append(
                        ValidationIssue(
                            code="mcap_video_payload_decode_failed",
                            level=IssueLevel.ERROR,
                            message="Reconstructed video produced no decodable frames",
                            path=str(source),
                            stream_id=topic,
                        )
                    )
                elif decoded_for_topic != packet_counts[topic]:
                    issues.append(
                        ValidationIssue(
                            code="mcap_video_packet_frame_count_differs",
                            level=IssueLevel.WARN,
                            message=(
                                f"{packet_counts[topic]} encoded packets produced "
                                f"{decoded_for_topic} frames"
                            ),
                            path=str(source),
                            stream_id=topic,
                        )
                    )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=1,
            checked_records=media_messages,
            decoded_records=decoded_images + decoded_video_frames,
            metadata={
                "media_messages": media_messages,
                "decoded_images": decoded_images,
                "decoded_video_frames": decoded_video_frames,
                "packet_counts": packet_counts,
                "formats": formats,
                "payload_decode": True,
            },
        )


def _safe_topic_name(topic: str) -> str:
    value = "".join(character if character.isalnum() else "_" for character in topic)
    return value.strip("_") or "stream"
