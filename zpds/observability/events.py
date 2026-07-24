"""Pipeline 结构化事件与 JSON Lines 输出。"""

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TextIO, runtime_checkable


@dataclass(frozen=True)
class PipelineEvent:
    """一条可机器解析的 Pipeline 生命周期事件。"""

    event: str
    timestamp: datetime
    run_id: str
    session_id: str
    stage_id: int | None = None
    stage_name: str | None = None
    attempt: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event:
            raise ValueError("event must not be empty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("event timestamp must be timezone-aware")
        if not self.run_id or not self.session_id:
            raise ValueError("event run_id and session_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "event": self.event,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        if self.stage_id is not None:
            value["stage_id"] = self.stage_id
        if self.stage_name is not None:
            value["stage_name"] = self.stage_name
        if self.attempt is not None:
            value["attempt"] = self.attempt
        value.update(self.details)
        return value


@runtime_checkable
class PipelineObserver(Protocol):
    """Runner 生命周期事件接收接口。"""

    def emit(self, event: PipelineEvent) -> None:
        ...


class NullObserver:
    """未配置可观测性时的无副作用观察器。"""

    def emit(self, event: PipelineEvent) -> None:
        del event


class CompositeObserver:
    """按注册顺序将事件发送给多个观察器。"""

    def __init__(self, observers: Sequence[PipelineObserver]) -> None:
        self._observers = tuple(observers)

    def emit(self, event: PipelineEvent) -> None:
        for observer in self._observers:
            observer.emit(event)


class JsonLinesObserver:
    """将每条事件立即写为单行 JSON，并 flush 以保留中断前日志。"""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def emit(self, event: PipelineEvent) -> None:
        line = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            self._stream.write(f"{line}\n")
            self._stream.flush()
