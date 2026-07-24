"""ROS2 DB3 (sqlite3) 适配器。"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from zpds.adapters.base import BaseAdapter
from zpds.adapters.common import infer_stream_kind, require_file, source_asset
from zpds.adapters.contracts import (
    ContainerMessage,
    IssueLevel,
    ValidationIssue,
    ValidationReport,
)
from zpds.core.types import (
    ClockDescriptor,
    ClockDomain,
    SessionInventory,
    SourceStream,
)


class Ros2Db3Adapter(BaseAdapter):
    """ROS2 DB3 适配器。"""

    def inspect(self, path: str) -> SessionInventory:
        source = require_file(path)
        with _connect_read_only(source) as connection:
            topics = connection.execute(
                "SELECT id, name, type, serialization_format FROM topics ORDER BY name"
            ).fetchall()
            count, start_ns, end_ns = connection.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages"
            ).fetchone()
        streams = [
            SourceStream(
                kind=infer_stream_kind(name),
                stream_id=name,
                role="observation",
                clock_id="ros_timestamp",
                topic=name,
                encoding=serialization_format,
                container="ros2_db3",
                metadata={"message_type": message_type, "topic_id": topic_id},
            )
            for topic_id, name, message_type, serialization_format in topics
        ]
        duration = (
            max(int(end_ns) - int(start_ns), 0) / 1_000_000_000
            if start_ns is not None and end_ns is not None
            else 0.0
        )
        return SessionInventory(
            session_id=source.stem,
            source_profile="ros2_db3",
            session_uri=str(source),
            assets=[source_asset(source, source.parent)],
            streams=streams,
            clocks=[
                ClockDescriptor(
                    clock_id="ros_timestamp",
                    domain=ClockDomain.ROS_TIME,
                    source="messages.timestamp",
                    authoritative=True,
                )
            ],
            duration_s=duration,
            clock_domain=ClockDomain.ROS_TIME,
            metadata={"message_count": int(count)},
        )

    def validate(self, path: str) -> ValidationReport:
        source = Path(path)
        if not source.is_file():
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="ros2_db3_missing",
                        level=IssueLevel.FATAL,
                        message=f"ROS2 DB3 file not found: {source}",
                        path=str(source),
                    ),
                )
            )
        with source.open("rb") as file:
            if file.read(16) != b"SQLite format 3\x00":
                return ValidationReport(
                    issues=(
                        ValidationIssue(
                            code="ros2_db3_magic_invalid",
                            level=IssueLevel.ERROR,
                            message="SQLite header is invalid",
                            path=str(source),
                        ),
                    ),
                    checked_assets=1,
                )
        try:
            inventory = self.inspect(str(source))
        except sqlite3.DatabaseError as error:
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="ros2_db3_schema_invalid",
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

    def iter_messages(self, path: str, topic: str | None = None) -> Iterator[ContainerMessage]:
        source = require_file(path)
        query = (
            "SELECT t.name, m.timestamp, m.data "
            "FROM messages m JOIN topics t ON t.id=m.topic_id"
        )
        parameters: tuple[str, ...] = ()
        if topic is not None:
            query += " WHERE t.name=?"
            parameters = (topic,)
        query += " ORDER BY m.timestamp, m.id"
        with _connect_read_only(source) as connection:
            for sequence, (name, timestamp, payload) in enumerate(
                connection.execute(query, parameters)
            ):
                yield ContainerMessage(
                    stream_id=name,
                    log_time_ns=int(timestamp),
                    publish_time_ns=None,
                    sequence=sequence,
                    payload=bytes(payload),
                    encoding="cdr",
                )

    def scan(self, path: str) -> ValidationReport:
        checked = sum(1 for _ in self.iter_messages(path))
        return ValidationReport(
            checked_assets=1,
            checked_records=checked,
            decoded_records=0,
            metadata={
                "payload_read": True,
                "cdr_decoded": False,
                "reason": "DB3 does not embed the complete ROS2 type definition",
            },
        )


def _connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
