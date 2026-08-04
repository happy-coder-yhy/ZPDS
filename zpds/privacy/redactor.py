"""组合流水线:一张图同时完成 人脸模糊 + 私密文本模糊。

输入 BGR 图像数组,输出脱敏后的 BGR 图像数组;也提供文件级 / 目录级批处理接口。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .face_detector import blur_faces, blur_regions
from .text_pipeline import detect_private_text

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def redact_image(
    image: np.ndarray,
    *,
    redact_faces: bool = True,
    redact_text: bool = True,
    face_threshold: float = config.FACE_CONFIDENCE,
    yolo_conf: float = config.YOLO_CONF,
    llm_api_key: Optional[str] = None,
) -> np.ndarray:
    """对单张图像执行隐私脱敏:人脸模糊 + 私密文本模糊。

    :param image: BGR 图像数组
    :param redact_faces: 是否进行人脸模糊
    :param redact_text: 是否进行私密文本检测与模糊
    :return: 脱敏后的 BGR 图像数组
    """
    result = image.copy()

    if redact_faces:
        result = blur_faces(result, face_threshold)

    if redact_text:
        text_boxes = detect_private_text(result, yolo_conf=yolo_conf, llm_api_key=llm_api_key)
        private_boxes = [tb.bbox for tb in text_boxes if tb.is_private]
        result = blur_regions(result, private_boxes)

    return result


def _imread(path: Path) -> np.ndarray:
    """读取图片,绕过 cv2.imread 对中文路径的支持问题。"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return image


def _imwrite(path: Path, image: np.ndarray) -> None:
    """保存图片,绕过 cv2.imwrite 对中文路径的支持问题。"""
    suffix = path.suffix or ".jpg"
    ok, buf = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"编码图片失败: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def redact_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    **kwargs,
) -> Path:
    """脱敏单张图片文件,默认输出到 <原名>_redacted.jpg。"""
    input_path = Path(input_path)
    image = _imread(input_path)
    result = redact_image(image, **kwargs)
    out = (
        Path(output_path)
        if output_path
        else input_path.with_name(f"{input_path.stem}_redacted.jpg")
    )
    _imwrite(out, result)
    return out


def redact_directory(
    input_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    **kwargs,
) -> list[Path]:
    """批量脱敏目录下所有 jpg/jpeg/png 图片,输出到 <目录>/redacted/。"""
    input_dir = Path(input_dir)
    out_dir = Path(output_dir) if output_dir else input_dir / "redacted"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for img in sorted(input_dir.iterdir()):
        if img.is_file() and img.suffix.lower() in _IMAGE_SUFFIXES:
            saved.append(redact_file(img, out_dir / img.name, **kwargs))
    return saved
