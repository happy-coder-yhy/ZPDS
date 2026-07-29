"""读取 Prepared Segment 的 RGB 帧及 Sample Map 来源信息。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from zpds.hands.schemas import PreparedFrame

REQUIRED_SAMPLE_MAP_COLUMNS = (
    "output_frame_index",
    "output_timestamp_ns",
    "source_frame_index",
    "source_timestamp_ns",
)


class PreparedSegmentError(Exception):
    """Prepared Segment 无法作为 Hands Pipeline 输入。"""


class StreamNotFoundError(PreparedSegmentError):
    """找不到或无法唯一确定 RGB Stream。"""


class SampleMapValidationError(PreparedSegmentError):
    """Sample Map 缺失、格式错误或数据不满足约定。"""


class VideoDecodeError(PreparedSegmentError):
    """Prepared RGB 视频无法打开、解码或与 Sample Map 对齐。"""


class PreparedSegmentReader:
    """逐帧读取一个 Prepared Segment 的指定 RGB Stream。

    每次迭代都会重新打开视频，因此同一个 Reader 可以重复遍历。Reader 只负责
    解码和 Sample Map 对齐，不运行手部模型，也不写出标注。
    """

    def __init__(
        self,
        segment_dir: str | Path,
        video_stream_id: str | None = None,
    ) -> None:
        self._segment_dir = Path(segment_dir).expanduser().resolve()
        if not self._segment_dir.is_dir():
            raise PreparedSegmentError(f"Prepared Segment 目录不存在: {self._segment_dir}")

        self._segment_path = self._segment_dir / "segment.json"
        self._segment = self._read_segment_json()
        self._segment_id = self._read_segment_id()
        self._stream = self._select_rgb_stream(video_stream_id)
        self._video_stream_id = str(self._stream["stream_id"])
        self._video_path = self._resolve_member_path(
            self._stream.get("uri"),
            field_name=f"RGB Stream {self._video_stream_id!r} 的 uri",
        )
        if not self._video_path.is_file():
            raise VideoDecodeError(
                f"RGB 视频不存在: segment={self._segment_id}, "
                f"stream={self._video_stream_id}, path={self._video_path}"
            )

        origin = self._stream.get("origin")
        sample_map_uri = origin.get("sample_map_uri") if isinstance(origin, dict) else None
        self._sample_map_path = self._resolve_member_path(
            sample_map_uri,
            field_name=f"RGB Stream {self._video_stream_id!r} 的 origin.sample_map_uri",
        )
        self._sample_map = self._read_sample_map()

    @property
    def segment_dir(self) -> Path:
        return self._segment_dir

    @property
    def segment_id(self) -> str:
        return self._segment_id

    @property
    def video_stream_id(self) -> str:
        return self._video_stream_id

    @property
    def video_path(self) -> Path:
        return self._video_path

    @property
    def sample_map_path(self) -> Path:
        return self._sample_map_path

    def __len__(self) -> int:
        return len(self._sample_map)

    def __iter__(self) -> Iterator[PreparedFrame]:
        capture = cv2.VideoCapture(str(self._video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoDecodeError(
                f"无法打开 RGB 视频: segment={self._segment_id}, "
                f"stream={self._video_stream_id}, path={self._video_path}"
            )

        try:
            for row_number, row in enumerate(
                self._sample_map.itertuples(index=False, name="SampleMapRow")
            ):
                readable, frame_bgr = capture.read()
                if not readable or frame_bgr is None:
                    raise VideoDecodeError(
                        f"RGB 视频帧数少于 Sample Map: segment={self._segment_id}, "
                        f"stream={self._video_stream_id}, "
                        f"无法解码 output_frame_index={row.output_frame_index}, "
                        f"sample_map_rows={len(self._sample_map)}, decoded_frames={row_number}"
                    )

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield PreparedFrame(
                    frame_rgb=frame_rgb,
                    output_frame_index=int(row.output_frame_index),
                    timestamp_ns=int(row.output_timestamp_ns),
                    source_frame_index=self._optional_int(row.source_frame_index),
                    source_timestamp_ns=self._optional_int(row.source_timestamp_ns),
                )

            has_extra_frame, _ = capture.read()
            if has_extra_frame:
                raise VideoDecodeError(
                    f"RGB 视频帧数多于 Sample Map: segment={self._segment_id}, "
                    f"stream={self._video_stream_id}, sample_map_rows={len(self._sample_map)}"
                )
        finally:
            capture.release()

    def _read_segment_json(self) -> dict[str, Any]:
        if not self._segment_path.is_file():
            raise PreparedSegmentError(f"segment.json 不存在: {self._segment_path}")
        try:
            with self._segment_path.open(encoding="utf-8") as file:
                segment = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise PreparedSegmentError(
                f"无法读取 segment.json: {self._segment_path}: {error}"
            ) from error
        if not isinstance(segment, dict):
            raise PreparedSegmentError(f"segment.json 顶层必须是对象: {self._segment_path}")
        return segment

    def _read_segment_id(self) -> str:
        segment_id = self._segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise PreparedSegmentError(f"segment.json 缺少有效 segment_id: {self._segment_path}")
        return segment_id

    def _select_rgb_stream(self, requested_stream_id: str | None) -> dict[str, Any]:
        streams = self._segment.get("streams")
        if not isinstance(streams, list):
            raise StreamNotFoundError(
                f"segment.json 的 streams 必须是数组: segment={self._segment_id}"
            )

        if requested_stream_id is not None:
            matches = [
                stream
                for stream in streams
                if isinstance(stream, dict) and stream.get("stream_id") == requested_stream_id
            ]
            if not matches:
                raise StreamNotFoundError(
                    f"找不到指定 Stream: segment={self._segment_id}, stream={requested_stream_id!r}"
                )
            stream = matches[0]
            if stream.get("modality") != "rgb":
                raise StreamNotFoundError(
                    f"指定 Stream 不是 RGB: segment={self._segment_id}, "
                    f"stream={requested_stream_id!r}, modality={stream.get('modality')!r}"
                )
            return stream

        rgb_streams = [
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("modality") == "rgb"
        ]
        if not rgb_streams:
            raise StreamNotFoundError(f"Prepared Segment 中没有 RGB Stream: {self._segment_id}")
        if len(rgb_streams) > 1:
            stream_ids = [str(stream.get("stream_id")) for stream in rgb_streams]
            raise StreamNotFoundError(
                f"Prepared Segment 中存在多个 RGB Stream，请指定 video_stream_id: "
                f"segment={self._segment_id}, streams={stream_ids}"
            )
        return rgb_streams[0]

    def _resolve_member_path(self, uri: Any, field_name: str) -> Path:
        if not isinstance(uri, str) or not uri.strip():
            raise PreparedSegmentError(f"{field_name} 不能为空: segment={self._segment_id}")
        relative_path = Path(uri)
        if relative_path.is_absolute():
            raise PreparedSegmentError(f"{field_name} 必须是 Segment 内相对路径: {uri!r}")
        resolved = (self._segment_dir / relative_path).resolve()
        try:
            resolved.relative_to(self._segment_dir)
        except ValueError as error:
            raise PreparedSegmentError(
                f"{field_name} 不能指向 Segment 目录之外: {uri!r}"
            ) from error
        return resolved

    def _read_sample_map(self) -> pd.DataFrame:
        if not self._sample_map_path.is_file():
            raise SampleMapValidationError(
                f"Sample Map 不存在: segment={self._segment_id}, "
                f"stream={self._video_stream_id}, path={self._sample_map_path}"
            )
        try:
            sample_map = pd.read_parquet(self._sample_map_path)
        except (OSError, ValueError) as error:
            raise SampleMapValidationError(
                f"无法读取 Sample Map: {self._sample_map_path}: {error}"
            ) from error

        missing_columns = [
            column for column in REQUIRED_SAMPLE_MAP_COLUMNS if column not in sample_map.columns
        ]
        if missing_columns:
            raise SampleMapValidationError(
                f"Sample Map 缺少必需列: segment={self._segment_id}, "
                f"stream={self._video_stream_id}, missing={missing_columns}"
            )
        if sample_map.empty:
            raise SampleMapValidationError(
                f"Sample Map 不能为空: segment={self._segment_id}, stream={self._video_stream_id}"
            )

        self._validate_integer_column(sample_map, "output_frame_index", allow_null=False)
        self._validate_integer_column(sample_map, "output_timestamp_ns", allow_null=False)
        self._validate_integer_column(sample_map, "source_frame_index", allow_null=True)
        self._validate_integer_column(sample_map, "source_timestamp_ns", allow_null=True)

        output_indices = sample_map["output_frame_index"].astype(np.int64).tolist()
        expected_indices = list(range(len(sample_map)))
        if output_indices != expected_indices:
            raise SampleMapValidationError(
                f"output_frame_index 必须从 0 连续递增: segment={self._segment_id}, "
                f"stream={self._video_stream_id}, actual={output_indices[:10]}"
            )

        output_timestamps = sample_map["output_timestamp_ns"].to_numpy(dtype=np.int64)
        if np.any(output_timestamps < 0):
            raise SampleMapValidationError("output_timestamp_ns 不能包含负值")
        if len(output_timestamps) > 1 and np.any(np.diff(output_timestamps) <= 0):
            raise SampleMapValidationError("output_timestamp_ns 必须严格递增")

        for column in ("source_frame_index", "source_timestamp_ns"):
            non_null = sample_map[column].dropna()
            if not non_null.empty and (non_null < 0).any():
                raise SampleMapValidationError(f"{column} 不能包含负值")

        return sample_map.loc[:, REQUIRED_SAMPLE_MAP_COLUMNS].copy()

    def _validate_integer_column(
        self,
        sample_map: pd.DataFrame,
        column: str,
        *,
        allow_null: bool,
    ) -> None:
        values = sample_map[column]
        if not allow_null and values.isna().any():
            raise SampleMapValidationError(f"{column} 不能包含空值")
        if not is_integer_dtype(values.dtype):
            raise SampleMapValidationError(f"{column} 必须使用整数 dtype，实际为 {values.dtype}")

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if pd.isna(value) else int(value)


__all__ = [
    "REQUIRED_SAMPLE_MAP_COLUMNS",
    "PreparedSegmentError",
    "PreparedSegmentReader",
    "SampleMapValidationError",
    "StreamNotFoundError",
    "VideoDecodeError",
]
