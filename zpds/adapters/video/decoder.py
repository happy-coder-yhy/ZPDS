"""视频解码、seek、逐帧读取。"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2


class VideoDecoder:
    """视频解码器封装。"""

    def __init__(self, path: str, *, convert_rgb: bool = True):
        self.path = str(Path(path))
        self.convert_rgb = convert_rgb
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoDecoder":  # noqa: PYI034 - Python 3.10 has no typing.Self
        self._capture = self._open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read_frame(self, idx: int) -> Any:
        """读取指定帧。"""
        if idx < 0:
            raise ValueError("frame index must be non-negative")
        capture = self._capture or self._open()
        owned = self._capture is None
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if not ok:
                raise IndexError(f"Unable to decode frame {idx}: {self.path}")
            return frame
        finally:
            if owned:
                capture.release()

    def iter_frames(self) -> Iterator[tuple[int, int, Any]]:
        """逐帧迭代器。"""
        capture = self._open()
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ns = int(capture.get(cv2.CAP_PROP_POS_MSEC) * 1_000_000)
                yield index, timestamp_ns, frame
                index += 1
        finally:
            capture.release()

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Unable to open video: {self.path}")
        if not self.convert_rgb:
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        return capture
