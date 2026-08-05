"""人员 B：场景首/中/尾代表帧的可视化预览。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from zpds.scene.sampling import extract_representative_frames
from zpds.scene.schemas import SceneProposal


def write_scene_previews(
    output_dir: str | Path,
    frames: Sequence[np.ndarray],
    scenes: Sequence[SceneProposal],
    *,
    fps: float,
    start_ns: int = 0,
) -> list[Path]:
    """为每个 scene 写一张三帧并排 PNG，返回写出路径。"""

    root = Path(output_dir).expanduser().resolve()
    preview_dir = root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scene in scenes:
        representative = extract_representative_frames(
            frames,
            scene,
            fps=fps,
            segment_start_ns=start_ns,
        )
        montage = np.hstack(representative)
        path = preview_dir / f"{scene.scene_id}.png"
        cv2.imwrite(str(path), montage)
        written.append(path)
    return written


__all__ = ["write_scene_previews"]
