"""MCAP 消息解码、schema 解析。"""

from collections.abc import Iterator
from typing import Any

from zpds.adapters.common import require_file, require_optional_module
from zpds.adapters.contracts import ContainerMessage


class McapReader:
    """MCAP 消息读取器。"""

    def __init__(self, path: str):
        self.path = require_file(path)

    def iter_messages(self, topic: str | None = None) -> Iterator[ContainerMessage]:
        """按 topic 迭代消息。"""
        reader_module = require_optional_module("mcap.reader", "mcap")
        with self.path.open("rb") as file:
            reader = reader_module.make_reader(file)
            topics = [topic] if topic is not None else None
            for schema, channel, message in reader.iter_messages(topics=topics):
                yield ContainerMessage(
                    stream_id=channel.topic,
                    log_time_ns=int(message.log_time),
                    publish_time_ns=int(message.publish_time),
                    sequence=int(message.sequence),
                    payload=bytes(message.data),
                    schema_name=schema.name if schema is not None else "",
                    encoding=channel.message_encoding,
                )

    def iter_decoded(self, topic: str | None = None) -> Iterator[Any]:
        """使用 MCAP Protobuf decoder 流式解码消息，不保留整文件内容。"""
        decoder_module = require_optional_module("mcap_protobuf.decoder", "mcap")
        reader_module = require_optional_module("mcap.reader", "mcap")
        topics = [topic] if topic is not None else None
        with self.path.open("rb") as file:
            reader = reader_module.make_reader(
                file,
                decoder_factories=[decoder_module.DecoderFactory()],
            )
            yield from reader.iter_decoded_messages(topics=topics)

    def topics(self) -> list[str]:
        """列出所有 topic。"""
        reader_module = require_optional_module("mcap.reader", "mcap")
        with self.path.open("rb") as file:
            summary = reader_module.make_reader(file).get_summary()
        if summary is None:
            return []
        return sorted(channel.topic for channel in summary.channels.values())
