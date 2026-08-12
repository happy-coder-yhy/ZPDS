"""SharedFrameSource — 全管线共享帧源（一次解码、多处消费）。

问题 16 背景：单次主流程中同一个 MKV 被独立全量解码最多 10 次
（hands 推理 / hands 清洗 / privacy / scene / stage3 过曝 / stage3 模糊 /
black_frame / bad_frame 子进程等）。本模块提供：

- 首次顺序迭代时解码一次，逐帧写 JPEG (q95) 磁盘缓存 + frames.json 元数据
- 之后所有消费者通过 ``Sequence`` 接口随机访问（``source[i]``）或
  重放迭代（``for frame in source``），从缓存解压而不是重新解码
- 缓存键 = 源文件 (mtime_ns, size)，不匹配自动重建（最坏退化为一次解码）

帧均为 BGR（与 cv2.VideoCapture 语义一致），由各消费者按需转换
（如 hands 推理需 BGR→RGB）。

注意 bad_frame 检测（MJPEG stderr 捕获）仍保留独立子进程解码：
它需要真实解码器 stderr 输出，无法由本缓存覆盖。
"""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np


def _cache_key(video_path: Path) -> str:
    """缓存键：源文件 (mtime_ns, size)，变化即重建。"""
    st = video_path.stat()
    return f"{st.st_mtime_ns:x}-{st.st_size:x}"


class SharedFrameSource(Sequence[np.ndarray]):
    """一次解码的只读帧序列（BGR），支持随机访问与重放迭代。

    首次访问触发解码 + JPEG 磁盘缓存；后续访问从缓存解压。
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        cache_dir: str | Path,
        jpeg_quality: int = 95,
        lru_size: int = 64,
    ) -> None:
        self._video_path = Path(video_path)
        self._cache_dir = Path(cache_dir)
        self._jpeg_quality = jpeg_quality
        self._lru_size = lru_size
        self._cache_root = self._cache_dir / "frames" / _cache_key(self._video_path)
        self._frames_json = self._cache_root / "frames.json"
        self._meta: dict | None = None
        self._lru: OrderedDict[int, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return float(self._load_meta()["fps"])

    @property
    def width(self) -> int:
        return int(self._load_meta()["width"])

    @property
    def height(self) -> int:
        return int(self._load_meta()["height"])

    @property
    def video_path(self) -> Path:
        return self._video_path

    @property
    def cache_root(self) -> Path:
        """缓存目录（诊断/清理用）。"""
        return self._cache_root

    def _load_meta(self) -> dict:
        if self._meta is None:
            self._ensure_cache()
        return self._meta

    # ------------------------------------------------------------------
    # Sequence 接口（随机访问）
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._load_meta()["decoded_count"])

    def __getitem__(self, index: int) -> np.ndarray:
        count = len(self)
        if index < 0:
            index += count
        if index < 0 or index >= count:
            raise IndexError(f"frame index out of range: {index}")
        if self._lru_size > 0 and index in self._lru:
            frame = self._lru.pop(index)  # LRU 触摸
            self._lru[index] = frame
            return frame
        path = self._cache_root / f"frame_{index:06d}.jpg"
        data = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"缓存帧损坏: {path}")
        if self._lru_size > 0:
            self._lru[index] = frame
            while len(self._lru) > self._lru_size:
                self._lru.popitem(last=False)
        return frame

    # ------------------------------------------------------------------
    # 迭代（首次触发解码 + 写缓存）
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        if not self._is_cache_valid():
            yield from self._decode_and_cache()
        else:
            self._meta = self._read_meta()
            for index in range(len(self)):
                yield self[index]

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------

    def _read_meta(self) -> dict:
        return json.loads(self._frames_json.read_text(encoding="utf-8"))

    def _is_cache_valid(self) -> bool:
        if not self._frames_json.is_file():
            return False
        try:
            meta = self._read_meta()
        except (OSError, ValueError):
            return False
        try:
            st = self._video_path.stat()
        except OSError:
            return False
        if meta.get("source_mtime_ns") != st.st_mtime_ns:
            return False
        if meta.get("source_size") != st.st_size:
            return False
        count = int(meta.get("decoded_count", 0))
        if count <= 0:
            return False
        if len(list(self._cache_root.glob("frame_*.jpg"))) != count:
            return False
        return True

    def _ensure_cache(self) -> None:
        """保证缓存已完整解码（只触发，不返回帧）。"""
        if not self._is_cache_valid():
            for _ in self._decode_and_cache():
                pass
        self._meta = self._read_meta()

    def _decode_and_cache(self) -> Iterator[np.ndarray]:
        """顺序解码视频，逐帧写 JPEG 缓存并 yield BGR 帧。

        解码中断（消费者提前退出 / read 失败）时不写 frames.json，
        下次迭代视为缓存无效重新解码——只影响极端情况，不引入坏数据。
        """
        print(
            f"  共享帧源: 首次解码 {self._video_path.name} "
            f"(缓存: {self._cache_root})"
        )
        if self._cache_root.exists():
            shutil.rmtree(self._cache_root)
        self._cache_root.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {self._video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        st = self._video_path.stat()
        meta = {
            "fps": fps,
            "width": 0,
            "height": 0,
            "decoded_count": 0,
            "source_path": str(self._video_path),
            "source_mtime_ns": int(st.st_mtime_ns),
            "source_size": int(st.st_size),
            "jpeg_quality": self._jpeg_quality,
        }
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index == 0:
                    h, w = frame.shape[:2]
                    meta["width"], meta["height"] = int(w), int(h)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                )
                if not ok:
                    raise RuntimeError(
                        f"JPEG 编码失败: {self._video_path} 第 {index} 帧"
                    )
                (self._cache_root / f"frame_{index:06d}.jpg").write_bytes(
                    encoded.tobytes()
                )
                yield frame
                index += 1
        finally:
            cap.release()

        meta["decoded_count"] = index
        self._frames_json.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._meta = meta
        if index == 0:
            raise ValueError(f"视频没有可解码帧: {self._video_path}")
