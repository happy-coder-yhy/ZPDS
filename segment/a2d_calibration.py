"""
A2D 标定提取：三路相机内参，无外参。

从 Session.meta["calibration"] 中提取 camera intrinsics，
不推测相机间外参。
"""

from __future__ import annotations

import json
from pathlib import Path

# 相机 frame_id 映射
CAMERA_FRAME_IDS = {
    "head_rgb": "head_camera_optical",
    "hand_left_rgb": "hand_left_camera_optical",
    "hand_right_rgb": "hand_right_camera_optical",
}


def extract_a2d_calibration(
    session_meta: dict,
    calibration_id: str | None = None,
) -> dict:
    """从 Session.meta 提取 A2D 标定。

    Args:
        session_meta: Session.meta dict（含 calibration 字段）。
        calibration_id: 标定 ID，默认从 session_id 推导。

    Returns:
        calibration dict（可序列化为 JSON）。
    """
    calib_data = session_meta.get("calibration", {})
    session_id = session_meta.get("episode_id", "unknown")

    if calibration_id is None:
        calibration_id = f"a2d_{session_id}"

    cameras = []
    for cam_entry in calib_data.get("cameras", []):
        camera_id = cam_entry.get("camera_id", "")
        intrinsics = cam_entry.get("intrinsics", {})

        camera = {
            "stream_id": camera_id,
            "frame_id": CAMERA_FRAME_IDS.get(camera_id, f"{camera_id}_optical"),
            "model": intrinsics.get("model", "pinhole"),
            "distortion_model": intrinsics.get(
                "distortion_model", "brown_conrady"
            ),
            "intrinsics": {
                "fx": intrinsics.get("fx", 0),
                "fy": intrinsics.get("fy", 0),
                "cx": intrinsics.get("cx", 0),
                "cy": intrinsics.get("cy", 0),
            },
            "resolution": {
                "width": cam_entry.get("resolution", {}).get("width") or 640,
                "height": cam_entry.get("resolution", {}).get("height") or 480,
            },
        }
        distortion_coeffs = intrinsics.get("distortion_coeffs")
        if distortion_coeffs is not None:
            values = list(distortion_coeffs)
            if len(values) == 5:
                camera["distortion"] = {
                    "k1": values[0],
                    "k2": values[1],
                    "k3": values[2],
                    "p1": values[3],
                    "p2": values[4],
                }
        cameras.append(camera)

    return {
        "calibration_id": calibration_id,
        "cameras": cameras,
        "transforms": [],
        "extrinsics_status": "unavailable",
    }


def write_calibration(calib: dict, output_dir: str) -> str:
    """写出 calibration.json。

    Returns:
        输出文件路径。
    """
    calib_dir = Path(output_dir) / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    output_path = calib_dir / "calibration.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    return str(output_path)


__all__ = ["extract_a2d_calibration", "write_calibration"]
