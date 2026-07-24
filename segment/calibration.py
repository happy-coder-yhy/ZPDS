"""
从 meta.json 提取标定信息，生成 calibration.json。
"""

import json
from pathlib import Path


def extract_calibration(
    meta_path: str,
    calibration_id: str = "calib_guida_001",
) -> dict:
    """从 Guida meta.json 提取标定数据。

    Args:
        meta_path: meta.json 文件路径
        calibration_id: 标定 ID

    Returns:
        calibration dict（可序列化为 JSON）
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    color = meta["streams"]["color"]
    depth = meta["streams"]["depth"]
    imu_cfg = meta["imu"]

    calib = {
        "calibration_id": calibration_id,
        "source": {
            "uri": "meta.json",
            "kind": "source_recorded",
        },
        "frames": [
            {
                "frame_id": "ego_camera_optical",
                "parent_frame_id": None,
            },
            {
                "frame_id": "imu",
                "parent_frame_id": "ego_camera_optical",
            },
        ],
        "cameras": [
            {
                "stream_id": "ego_rgb",
                "frame_id": "ego_camera_optical",
                "model": "pinhole",
                "resolution": [color["width"], color["height"]],
                "intrinsics": {
                    "fx": color["intrinsics"]["fx"],
                    "fy": color["intrinsics"]["fy"],
                    "cx": color["intrinsics"]["cx"],
                    "cy": color["intrinsics"]["cy"],
                },
            }
        ],
        "depth_to_color": {
            "rotation": depth.get("extrinsics_to_color", {}).get("rotation", []),
            "translation": depth.get("extrinsics_to_color", {}).get("translation", []),
            "translation_unit": depth.get("extrinsics_to_color", {}).get("translation_unit", "mm"),
        },
        "imu_extrinsics": {
            "to_frame": "depth",
            "rotation": imu_cfg.get("extrinsics_to_depth", {}).get("rotation", []),
            "translation": imu_cfg.get("extrinsics_to_depth", {}).get("translation", []),
            "translation_unit": imu_cfg.get("extrinsics_to_depth", {}).get("translation_unit", "mm"),
            "status": "available" if imu_cfg.get("extrinsics_to_depth") else "unavailable",
        },
    }

    return calib


def extract_calibration_from_mcap(
    calib_data: dict,
    calibration_id: str = "calib_dunjia_001",
    multi_cam: dict | None = None,
) -> dict:
    """从 MCAP foxglove.CameraCalibration 消息构建标定 JSON。

    Args:
        calib_data: camera0 的标定数据 (主相机)
        calibration_id: 标定 ID
        multi_cam: 多相机标定 dict {cam_name: calib_data, ...}，可选

    Returns:
        与 extract_calibration() 兼容的 calibration dict
    """
    k = calib_data.get("K", [0]*9)
    cameras = [
        {
            "stream_id": "ego_rgb_center" if multi_cam else "ego_rgb",
            "frame_id": calib_data.get("frame_id", "headcam_center_optical_frame"),
            "model": "pinhole",
            "resolution": [calib_data.get("width", 0), calib_data.get("height", 0)],
            "intrinsics": {
                "fx": k[0] if len(k) > 0 else 0,
                "fy": k[4] if len(k) > 4 else 0,
                "cx": k[2] if len(k) > 2 else 0,
                "cy": k[5] if len(k) > 5 else 0,
            },
        }
    ]

    # 添加额外相机
    if multi_cam:
        for cam_name in ["camera1", "camera2"]:
            if cam_name in multi_cam:
                cb = multi_cam[cam_name]
                ck = cb.get("K", [0]*9)
                stream_id = "ego_rgb_left" if cam_name == "camera1" else "ego_rgb_right"
                cameras.append({
                    "stream_id": stream_id,
                    "frame_id": cb.get("frame_id", ""),
                    "model": "pinhole",
                    "resolution": [cb.get("width", 0), cb.get("height", 0)],
                    "intrinsics": {
                        "fx": ck[0] if len(ck) > 0 else 0,
                        "fy": ck[4] if len(ck) > 4 else 0,
                        "cx": ck[2] if len(ck) > 2 else 0,
                        "cy": ck[5] if len(ck) > 5 else 0,
                    },
                })

    return {
        "calibration_id": calibration_id,
        "source": {
            "uri": "MCAP camera_info topics",
            "kind": "source_recorded",
            "format": "foxglove.CameraCalibration",
        },
        "calibrations": multi_cam,  # 传递完整多相机标定
        "frames": [
            {
                "frame_id": calib_data.get("frame_id", "headcam_center_optical_frame"),
                "parent_frame_id": None,
            },
            {
                "frame_id": "imu",
                "parent_frame_id": calib_data.get("frame_id", "headcam_center_optical_frame"),
            },
        ],
        "cameras": cameras,
        "depth_to_color": {
            "status": "unavailable",
        },
        "imu_extrinsics": {
            "status": "unavailable",
        },
    }


def write_calibration(calib: dict, output_dir: str) -> str:
    """写出 calibration.json。

    Returns:
        输出文件路径
    """
    calib_dir = Path(output_dir) / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    output_path = calib_dir / "calibration.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    return str(output_path)
