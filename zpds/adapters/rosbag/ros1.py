"""ROS1 BAG 适配器。"""

from pathlib import Path

from zpds.adapters.base import BaseAdapter
from zpds.adapters.common import require_file, require_optional_module, source_asset
from zpds.adapters.contracts import IssueLevel, ValidationIssue, ValidationReport
from zpds.core.types import ClockDescriptor, ClockDomain, SessionInventory, SourceStream


class Ros1BagAdapter(BaseAdapter):
    """ROS1 .bag 适配器。"""

    def inspect(self, path: str) -> SessionInventory:
        source = require_file(path)
        rosbag1 = require_optional_module("rosbags.rosbag1", "ros1")
        streams: list[SourceStream] = []
        message_count = 0
        with rosbag1.Reader(source) as reader:
            message_count = int(reader.message_count)
            streams = [
                SourceStream(
                    kind=self._kind(connection.topic),
                    stream_id=connection.topic,
                    role="observation",
                    clock_id="ros_time",
                    topic=connection.topic,
                    encoding="ros1",
                    container="ros1_bag",
                    metadata={"message_type": connection.msgtype},
                )
                for connection in reader.connections
            ]
        return SessionInventory(
            session_id=source.stem,
            source_profile="ros1_bag",
            session_uri=str(source),
            assets=[source_asset(source, source.parent)],
            streams=streams,
            clocks=[
                ClockDescriptor(
                    clock_id="ros_time",
                    domain=ClockDomain.ROS_TIME,
                    source="ROS1 bag timestamp",
                    authoritative=True,
                )
            ],
            clock_domain=ClockDomain.ROS_TIME,
            metadata={"message_count": message_count},
        )

    def validate(self, path: str) -> ValidationReport:
        source = Path(path)
        if not source.is_file():
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="ros1_bag_missing",
                        level=IssueLevel.FATAL,
                        message=f"ROS1 bag not found: {source}",
                    ),
                )
            )
        with source.open("rb") as file:
            magic = file.readline(32)
        if not magic.startswith(b"#ROSBAG V2.0"):
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="ros1_bag_magic_invalid",
                        level=IssueLevel.ERROR,
                        message="ROS1 bag magic is invalid",
                        path=str(source),
                    ),
                ),
                checked_assets=1,
            )
        try:
            inventory = self.inspect(str(source))
        except ImportError as error:
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="ros1_dependency_missing",
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

    @staticmethod
    def _kind(topic: str):
        from zpds.adapters.common import infer_stream_kind

        return infer_stream_kind(topic)
