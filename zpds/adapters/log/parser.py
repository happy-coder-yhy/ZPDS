"""设备日志解析器。"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIMESTAMP_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+|[0-9]{16,19})"
)


@dataclass(frozen=True)
class LogEvent:
    line_number: int
    raw: str
    timestamp: str | int | None = None
    level: str = ""
    fields: dict[str, Any] | None = None


class LogParser:
    """录制/设备日志解析器。"""

    def iter_events(self, path: str) -> Iterator[LogEvent]:
        source = Path(path)
        with source.open(encoding="utf-8", errors="replace") as file:
            for line_number, raw_line in enumerate(file, start=1):
                raw = raw_line.rstrip("\r\n")
                if not raw:
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, dict):
                    timestamp = value.get("timestamp_ns", value.get("timestamp"))
                    level = str(value.get("level", value.get("severity", "")))
                    yield LogEvent(line_number, raw, timestamp, level, value)
                    continue
                match = TIMESTAMP_PATTERN.search(raw)
                timestamp_value: str | int | None = None
                if match:
                    text = match.group("timestamp")
                    timestamp_value = int(text) if text.isdigit() else text
                level = next(
                    (name for name in ("FATAL", "ERROR", "WARN", "INFO", "DEBUG") if name in raw),
                    "",
                )
                yield LogEvent(line_number, raw, timestamp_value, level)

    def parse(self, path: str) -> dict[str, Any]:
        """流式解析日志并返回轻量汇总。"""
        levels: dict[str, int] = {}
        samples: list[LogEvent] = []
        event_count = 0
        for event in self.iter_events(path):
            event_count += 1
            if len(samples) < 100:
                samples.append(event)
            if event.level:
                levels[event.level] = levels.get(event.level, 0) + 1
        return {
            "path": str(Path(path)),
            "events": samples,
            "event_count": event_count,
            "events_truncated": event_count > len(samples),
            "levels": levels,
        }
