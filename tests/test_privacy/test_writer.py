"""write_redacted_video 等长写出测试（丢帧 / 首帧遮挡缺陷回归）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from zpds.privacy.contracts import FrameRedactionRecord
from zpds.privacy.writer import write_redacted_video


def _write_source(path: Path, n_frames: int, w: int = 64, h: int = 48) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
    )
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.full((h, w, 3), i, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _read_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"无法打开产物视频: {path}"
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def _make_records(
    n_frames: int,
    *,
    redacted_idxs: set[int] | None = None,
    first_redacted: bool = True,
) -> list[FrameRedactionRecord]:
    """构造逐帧记录：redacted_idxs 中的帧带 redacted_frame，其余为 None。"""
    records = []
    for i in range(n_frames):
        has = (i in (redacted_idxs or ())) or (first_redacted and i == 0)
        frame = (
            np.full((48, 64, 3), 200 + i, dtype=np.uint8) if has else None
        )
        records.append(
            FrameRedactionRecord(
                frame_index=i,
                timestamp_ns=i * 33_333_333,
                redacted_frame=frame,
            )
        )
    return records


def test_empty_records_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="records 不能为空"):
        write_redacted_video([], tmp_path / "out.mp4")


def test_no_source_first_frame_missing_raises(tmp_path: Path) -> None:
    records = _make_records(3, first_redacted=False)
    with pytest.raises(ValueError, match="无法确定视频尺寸"):
        write_redacted_video(records, tmp_path / "out.mp4")


def test_no_source_missing_frames_are_skipped(tmp_path: Path) -> None:
    """旧行为（无 source_video）：无遮挡帧跳过，输出短于 records。"""
    records = _make_records(5, redacted_idxs={0, 2, 4})
    out = write_redacted_video(
        records, tmp_path / "out.mp4", recode_h264=False
    )
    assert _read_frame_count(out) == 3


def test_source_fills_missing_frames_equal_length(tmp_path: Path) -> None:
    """核心回归：source_video 补齐无遮挡帧，输出与源等长。"""
    src = tmp_path / "src.mp4"
    _write_source(src, 5)
    records = _make_records(5, redacted_idxs={2})

    out = write_redacted_video(
        records, tmp_path / "out.mp4", source_video=src, recode_h264=False
    )
    assert _read_frame_count(out) == 5


def test_source_first_frame_missing_ok(tmp_path: Path) -> None:
    """核心回归：首帧无遮挡时以源为首帧，不再报错。"""
    src = tmp_path / "src.mp4"
    _write_source(src, 4)
    records = _make_records(4, first_redacted=False)

    out = write_redacted_video(
        records, tmp_path / "out.mp4", source_video=src, recode_h264=False
    )
    assert _read_frame_count(out) == 4


def test_source_all_frames_missing_equal_length(tmp_path: Path) -> None:
    """全无遮挡帧 + source：输出等长，内容为源帧。"""
    src = tmp_path / "src.mp4"
    _write_source(src, 6)
    records = _make_records(6, first_redacted=False)

    out = write_redacted_video(
        records, tmp_path / "out.mp4", source_video=src, recode_h264=False
    )
    assert _read_frame_count(out) == 6


def test_source_short_raises(tmp_path: Path) -> None:
    """源帧数不足 → 显式报错（暴露脱敏流与源不一致）。"""
    src = tmp_path / "src.mp4"
    _write_source(src, 3)
    records = _make_records(5, redacted_idxs={0})

    with pytest.raises(ValueError, match="帧数不足"):
        write_redacted_video(
            records, tmp_path / "out.mp4", source_video=src, recode_h264=False
        )


def test_source_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="无法打开源视频"):
        write_redacted_video(
            _make_records(2),
            tmp_path / "out.mp4",
            source_video=tmp_path / "nope.mp4",
            recode_h264=False,
        )
