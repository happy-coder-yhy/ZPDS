"""不依赖模型的合成场景视频 fixture。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticBoundary:
    frame_index: int
    kind: str
    tolerance_frames: int = 1


@dataclass(frozen=True)
class SyntheticSceneFixture:
    name: str
    frames: tuple[np.ndarray, ...]
    fps: float
    boundaries: tuple[SyntheticBoundary, ...]


def _textured_frame(seed: int, *, width: int = 96, height: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(20, 220, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (8, 8), (35, 35), (20, 230, 60), -1)
    cv2.circle(frame, (70, 40), 13, (230, 40, 30), -1)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def hard_cut_fixture(*, fps: float = 10.0) -> SyntheticSceneFixture:
    first = _textured_frame(1)
    second = _textured_frame(99)
    second[:, :, 0] = np.clip(second[:, :, 0].astype(np.int16) + 80, 0, 255)
    frames = tuple([first.copy() for _ in range(10)] + [second.copy() for _ in range(10)])
    return SyntheticSceneFixture("hard_cut", frames, fps, (SyntheticBoundary(10, "hard_cut"),))


def gradual_fixture(*, fps: float = 10.0) -> SyntheticSceneFixture:
    first = _textured_frame(2)
    second = _textured_frame(77)
    frames: list[np.ndarray] = [first.copy() for _ in range(5)]
    for alpha in np.linspace(0.0, 1.0, 11)[1:]:
        frames.append(cv2.addWeighted(first, 1.0 - float(alpha), second, float(alpha), 0.0))
    frames.extend(second.copy() for _ in range(5))
    return SyntheticSceneFixture("gradual", tuple(frames), fps, (SyntheticBoundary(10, "gradual", 4),))


def black_frame_fixture(*, fps: float = 10.0) -> SyntheticSceneFixture:
    base = _textured_frame(3)
    black = np.zeros_like(base)
    frames = tuple(
        [base.copy() for _ in range(6)]
        + [black.copy() for _ in range(5)]
        + [base.copy() for _ in range(6)]
    )
    return SyntheticSceneFixture(
        "black_frames",
        frames,
        fps,
        (SyntheticBoundary(6, "black_enter"), SyntheticBoundary(11, "black_exit")),
    )


def freeze_fixture(*, fps: float = 10.0) -> SyntheticSceneFixture:
    base = _textured_frame(4)

    def shifted(offset: int) -> np.ndarray:
        matrix = np.asarray([[1, 0, offset], [0, 1, 0]], dtype=np.float32)
        return cv2.warpAffine(
            base,
            matrix,
            (base.shape[1], base.shape[0]),
            dst=None,
            borderMode=cv2.BORDER_REFLECT,
        )

    moving_before = [shifted(index * 2) for index in range(6)]
    frozen_frame = moving_before[-1]
    frozen = [frozen_frame.copy() for _ in range(7)]
    moving_after = [shifted(12 + index * 2) for index in range(1, 7)]
    frames = tuple(moving_before + frozen + moving_after)
    return SyntheticSceneFixture("freeze", frames, fps, (SyntheticBoundary(6, "freeze", 1),))


def ego_translation_fixture(*, fps: float = 10.0) -> SyntheticSceneFixture:
    base = _textured_frame(5, width=128, height=80)
    frames = []
    for index in range(15):
        matrix = np.asarray(
            [[1, 0, index], [0, 1, index // 4]],
            dtype=np.float32,
        )
        frames.append(
            cv2.warpAffine(
                base,
                matrix,
                (base.shape[1], base.shape[0]),
                dst=None,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return SyntheticSceneFixture("ego_translation", tuple(frames), fps, ())


def semantic_task_switch_fixture(*, fps: float = 10.0) -> SyntheticSceneFixture:
    base = _textured_frame(6)
    first = base.copy()
    second = base.copy()
    cv2.rectangle(first, (40, 18), (55, 45), (0, 0, 255), -1)
    cv2.circle(second, (48, 31), 12, (0, 0, 255), -1)
    frames = tuple([first.copy() for _ in range(10)] + [second.copy() for _ in range(10)])
    return SyntheticSceneFixture(
        "semantic_task_switch",
        frames,
        fps,
        (SyntheticBoundary(10, "semantic_change"),),
    )


def all_stage_a_fixtures() -> tuple[SyntheticSceneFixture, ...]:
    return (
        hard_cut_fixture(),
        gradual_fixture(),
        black_frame_fixture(),
        freeze_fixture(),
        ego_translation_fixture(),
    )


__all__ = [
    "SyntheticBoundary",
    "SyntheticSceneFixture",
    "all_stage_a_fixtures",
    "black_frame_fixture",
    "ego_translation_fixture",
    "freeze_fixture",
    "gradual_fixture",
    "hard_cut_fixture",
    "semantic_task_switch_fixture",
]
