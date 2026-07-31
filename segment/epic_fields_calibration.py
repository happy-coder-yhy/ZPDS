"""EPIC-Fields 相机标定加载器。

EPIC-Fields JSON 使用 COLMAP/OpenCV ``OPENCV`` 相机模型，参数顺序为
``[fx, fy, cx, cy, k1, k2, p1, p2]``。本模块只读取外置参考资产，不将
EPIC-Fields 数据复制到 Prepared Segment。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EPIC_FIELDS_SAMPLE_URL = (
    "https://raw.githubusercontent.com/epic-kitchens/epic-Fields-code/"
    "2f25448b59138ec5a20c9d7ab70f587b4f2f5be8/example_data/P28_101.json"
)
EPIC_FIELDS_CODE_COMMIT = "2f25448b59138ec5a20c9d7ab70f587b4f2f5be8"
EPIC_FIELDS_RELEASE_URL = "https://thor.robots.ox.ac.uk/epic-fields/json-format.tar.gz"
EPIC_FIELDS_SHA512_URL = "https://thor.robots.ox.ac.uk/epic-fields/SHA512SUMS"


def load_epic_fields_calibration(epic_fields_root: str | Path, video_id: str) -> dict:
    """加载一个 EPIC-Fields 视频的标准化相机标定。

    ``epic_fields_root`` 可以是完整解压目录，也可以是仅包含少量 JSON 验证
    样本的目录。找不到匹配视频时抛出 :class:`FileNotFoundError`。
    """
    path = find_epic_fields_json(epic_fields_root, video_id)
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)

    camera = document.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(f"EPIC-Fields 标定缺少 camera: {path}")
    if camera.get("model") != "OPENCV":
        raise ValueError(
            f"EPIC-Fields 相机模型必须为 OPENCV，实际为 {camera.get('model')!r}: {path}"
        )

    params = camera.get("params")
    if not isinstance(params, list) or len(params) != 8:
        raise ValueError(f"EPIC-Fields OPENCV params 必须为 8 个数值: {path}")
    try:
        fx, fy, cx, cy, k1, k2, p1, p2 = [float(value) for value in params]
        width = int(camera["width"])
        height = int(camera["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"EPIC-Fields 相机参数无效: {path}") from exc

    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError(f"EPIC-Fields 相机尺寸或焦距无效: {path}")

    return {
        "calibration_id": f"epic_fields_{video_id}",
        "source": {
            "kind": "external_reference",
            "producer": "EPIC-Fields",
            "uri": str(path),
            "reference_url": (
                EPIC_FIELDS_SAMPLE_URL
                if video_id == "P28_101"
                else f"https://github.com/epic-kitchens/epic-Fields-code/tree/{EPIC_FIELDS_CODE_COMMIT}"
            ),
            "git_commit": EPIC_FIELDS_CODE_COMMIT,
            "sha256": _sha256(path),
        },
        "coverage": {"status": "covered", "video_id": video_id},
        "cameras": [
            {
                "stream_id": "ego_rgb",
                "model": "pinhole",
                "source_camera_model": "OPENCV",
                "distortion_model": "plumb_bob",
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "D": [k1, k2, p1, p2],
                "resolution": {"width": width, "height": height},
            }
        ],
        "transforms": [],
        "extrinsics_status": "available_per_frame",
    }


def missing_epic_fields_calibration(video_id: str, epic_fields_root: str | Path | None) -> dict:
    """构造未覆盖 EPIC 视频的显式标定状态。"""
    root = str(epic_fields_root) if epic_fields_root is not None else ""
    return {
        "calibration_id": f"epic_fields_{video_id}",
        "source": {"kind": "external_reference", "producer": "EPIC-Fields", "uri": root},
        "coverage": {"status": "missing_calibration", "video_id": video_id},
        "cameras": [],
        "transforms": [],
        "extrinsics_status": "unavailable",
    }


def find_epic_fields_json(epic_fields_root: str | Path, video_id: str) -> Path:
    """在完整或样本 EPIC-Fields 目录中确定性查找 ``<video_id>.json``。"""
    root = Path(epic_fields_root)
    if not root.is_dir():
        raise FileNotFoundError(f"EPIC-Fields 标定目录不存在: {root}")

    direct_candidates = [
        root / f"{video_id}.json",
        root / "example_data" / f"{video_id}.json",
        root / video_id / f"{video_id}.json",
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(root.rglob(f"{video_id}.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"EPIC-Fields 标定存在多个同名候选: {matches}")
    raise FileNotFoundError(f"EPIC-Fields 未覆盖视频: {video_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "EPIC_FIELDS_SAMPLE_URL",
    "EPIC_FIELDS_CODE_COMMIT",
    "EPIC_FIELDS_RELEASE_URL",
    "EPIC_FIELDS_SHA512_URL",
    "find_epic_fields_json",
    "load_epic_fields_calibration",
    "missing_epic_fields_calibration",
]
