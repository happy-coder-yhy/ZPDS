from pathlib import Path

import cv2
import numpy as np
import pytest

from segment import video_transcoder
from segment.a2d_video_transcoder import transcode_image_sequence


def test_image_sequence_transcoder_applies_frame_transform(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    image = np.full((4, 6, 3), 80, dtype=np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", image)
    assert encoded
    source_path.write_bytes(jpeg.tobytes())

    transformed_frames = []

    def transform(frame: np.ndarray) -> np.ndarray:
        transformed_frames.append(frame.copy())
        return frame

    result = transcode_image_sequence(
        index_frames=[{"source_timestamp_ns": 0, "source_path": str(source_path)}],
        output_mp4=str(tmp_path / "output.mp4"),
        source_start_ns=0,
        source_end_ns=100_000_000,
        target_fps=10.0,
        width=6,
        height=4,
        frame_transform=transform,
    )

    assert result["output_frames"] == 1
    assert len(transformed_frames) == 1


def test_valid_cached_video_is_reused(tmp_path, monkeypatch):
    output = tmp_path / "cached.mp4"
    output.write_bytes(b"valid-video-placeholder")
    monkeypatch.setattr(
        video_transcoder,
        "_probe_video",
        lambda path: {"width": 640, "height": 480, "frame_count": 12},
    )

    result = video_transcoder.transcode_rgb(
        source_video="unused.mp4",
        output_mp4=str(output),
        source_start_ns=0,
        source_end_ns=399_999_996,
        index_frames=[{"seq": 0, "timestamp_ns": 0}],
    )

    assert result["cached"] is True
    assert result["output_frames"] == 12


def test_frame_transform_uses_opencv_path_and_bypasses_cache(tmp_path, monkeypatch):
    output = tmp_path / "cached.mp4"
    output.write_bytes(b"old-output")
    transformed = []

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == video_transcoder.cv2.CAP_PROP_FRAME_WIDTH:
                return 6
            if prop == video_transcoder.cv2.CAP_PROP_FRAME_HEIGHT:
                return 4
            return 0

        def set(self, prop, value):
            return True

        def read(self):
            return True, np.full((4, 6, 3), 10, dtype=np.uint8)

        def release(self):
            pass

    class FakeWriter:
        def __init__(self, path, *args):
            Path(path).write_bytes(b"new-output")

        def isOpened(self):
            return True

        def write(self, frame):
            transformed.append(frame.copy())

        def release(self):
            pass

    monkeypatch.setattr(video_transcoder.cv2, "VideoCapture", lambda path: FakeCapture())
    monkeypatch.setattr(video_transcoder.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(
        video_transcoder,
        "_probe_video",
        lambda path: {"width": 6, "height": 4, "frame_count": 1},
    )
    monkeypatch.setattr(
        video_transcoder,
        "_transcode_with_ffmpeg",
        lambda *args, **kwargs: pytest.fail("frame transform must not use ffmpeg"),
    )

    result = video_transcoder.transcode_rgb(
        source_video="source.mp4",
        output_mp4=str(output),
        source_start_ns=0,
        source_end_ns=100_000_000,
        index_frames=[{"seq": 0, "timestamp_ns": 0}],
        target_fps=10.0,
        frame_transform=lambda frame: frame + 1,
    )

    assert result["output_frames"] == 1
    assert len(transformed) == 1
    assert np.all(transformed[0] == 11)


def test_invalid_cached_video_is_removed_before_retry(tmp_path, monkeypatch):
    output = tmp_path / "broken.mp4"
    output.write_bytes(b"moov-only")
    monkeypatch.setattr(video_transcoder, "_probe_video", lambda path: None)
    monkeypatch.setattr(
        video_transcoder.shutil,
        "which",
        lambda name: "ffmpeg.exe" if name == "ffmpeg" else None,
    )

    def fake_ffmpeg(*args, **kwargs):
        assert not output.exists()
        return {
            "output_frames": 30,
            "output_fps": 30.0,
            "width": 640,
            "height": 480,
            "codec": "h264",
            "output_path": str(output),
        }

    monkeypatch.setattr(
        video_transcoder,
        "_transcode_with_ffmpeg",
        fake_ffmpeg,
    )

    result = video_transcoder.transcode_rgb(
        source_video="source.mp4",
        output_mp4=str(output),
        source_start_ns=0,
        source_end_ns=1_000_000_000,
        index_frames=[{"seq": 0, "timestamp_ns": 0}],
    )

    assert result["output_frames"] == 30


def test_absolute_source_clock_uses_source_frame_positions(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output.mp4"
    received = {}
    source_origin_ns = 1_782_441_093_231_751_000

    monkeypatch.setattr(
        video_transcoder.shutil,
        "which",
        lambda name: "ffmpeg.exe" if name == "ffmpeg" else None,
    )

    def fake_ffmpeg(*args, **kwargs):
        received.update(kwargs)
        return {
            "output_frames": 30,
            "output_fps": 30.0,
            "width": 640,
            "height": 480,
            "codec": "h264",
            "output_path": str(output),
        }

    monkeypatch.setattr(
        video_transcoder,
        "_transcode_with_ffmpeg",
        fake_ffmpeg,
    )

    video_transcoder.transcode_rgb(
        source_video="source.mp4",
        output_mp4=str(output),
        source_start_ns=source_origin_ns,
        source_end_ns=source_origin_ns + 1_000_000_000,
        index_frames=[
            {"seq": 0, "timestamp_ns": source_origin_ns},
            {"seq": 1, "timestamp_ns": source_origin_ns + 33_333_333},
        ],
    )

    assert received["start_frame"] == 0
    assert received["end_frame"] == 2
    assert received["expected_frames"] == 31
    assert received["source_fps"] == pytest.approx(30.0, rel=0.01)


def test_ffmpeg_path_does_not_require_ffprobe(tmp_path, monkeypatch):
    output = tmp_path / "output.mp4"
    commands = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stderr = iter(())
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(
        video_transcoder.shutil,
        "which",
        lambda name: "ffmpeg.exe" if name == "ffmpeg" else None,
    )
    monkeypatch.setattr(video_transcoder.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        video_transcoder,
        "_probe_video",
        lambda path: {"width": 1280, "height": 720, "frame_count": 30},
    )

    result = video_transcoder._transcode_with_ffmpeg(
        source_video="source.mp4",
        output_mp4=str(output),
        source_start_ns=0,
        source_end_ns=1_000_000_000,
        target_fps=30.0,
    )

    assert commands[0][0] == "ffmpeg.exe"
    assert "ffprobe" not in commands[0]
    assert result["output_frames"] == 30


def test_zero_frame_opencv_output_is_deleted(tmp_path, monkeypatch):
    output = tmp_path / "empty.mp4"

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == video_transcoder.cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            if prop == video_transcoder.cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            return 0

        def set(self, prop, value):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    class FakeWriter:
        def __init__(self, path):
            Path(path).write_bytes(b"empty-container")

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            pass

    monkeypatch.setattr(video_transcoder.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        video_transcoder.cv2,
        "VideoCapture",
        lambda path: FakeCapture(),
    )
    monkeypatch.setattr(
        video_transcoder.cv2,
        "VideoWriter",
        lambda path, *args: FakeWriter(path),
    )
    monkeypatch.setattr(video_transcoder, "_probe_video", lambda path: None)

    with pytest.raises(RuntimeError, match="未生成任何可解码帧"):
        video_transcoder.transcode_rgb(
            source_video="source.mp4",
            output_mp4=str(output),
            source_start_ns=0,
            source_end_ns=1_000_000_000,
            index_frames=[{"seq": 0, "timestamp_ns": 0}],
        )

    assert not output.exists()
