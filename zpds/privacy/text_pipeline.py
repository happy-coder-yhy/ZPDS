"""文本隐私检测流水线:YOLO 检测文本 → 中英 OCR 抽取 → LLM 判断。

对应项目一(Text-detection-and-classification-on-images)的整条链路,
修复了原版的问题:中英混合 OCR、LLM 按编号结构化判断、可复用的函数库形态。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from . import config
from .llm import classify_text_blocks

# 确定性规则层:标准格式的证件/号码直接判定为隐私,不依赖 LLM
_STRONG_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{17}[\dXx]\b"), "身份证号"),  # 18 位身份证(可带 X)
    (re.compile(r"\b\d{15}\b"), "身份证号"),        # 15 位旧版身份证
    (re.compile(r"\b1[3-9]\d{9}\b"), "手机号"),     # 11 位手机号
    (re.compile(r"\b\d{13,19}\b"), "银行卡号"),     # 银行卡/账户号
]


def _match_strong_pii(text: str) -> str:
    """命中强规则返回隐私类别,否则返回空字符串。"""
    for pattern, label in _STRONG_PII_PATTERNS:
        if pattern.search(text):
            return label
    return ""

# 模型/OCR reader 单例,避免每次调用重复加载
_yolo_model: Optional[Any] = None
_ocr_reader = None


@dataclass
class TextBox:
    """一个 YOLO 检测出的文本区域及其判定结果。"""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) 原图坐标
    text: str                        # OCR 识别出的文本
    confidence: float                # OCR 最高置信度
    is_private: bool = False         # LLM 是否判定为私密
    privacy_type: str = ""           # LLM 给出的隐私类别


def get_yolo() -> Any:
    global _yolo_model
    if _yolo_model is None:
        # Ultralytics 会在导入时全局替换 cv2.imread。延迟到真正需要
        # 文本检测模型时再导入，避免普通 privacy/schema 导入污染深度流水线。
        from ultralytics import YOLO

        if not config.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO 模型不存在: {config.YOLO_MODEL_PATH}\n"
                "可通过环境变量 PRIVACY_YOLO_PATH 指定其它路径。"
            )
        _yolo_model = YOLO(str(config.YOLO_MODEL_PATH))
    return _yolo_model


def get_ocr_reader():
    """懒加载 EasyOCR reader(首次调用会下载中英文模型)。"""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(config.OCR_LANGS)
    return _ocr_reader


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """灰度 + 锐化,提升 OCR 识别率。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, sharpen_kernel)


def _zoomed_crop(image: np.ndarray, box, zoom: float = 1.5) -> tuple[int, int, int, int]:
    """以框中心放大裁剪,并 clamp 到图像边界。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nw, nh = int((x2 - x1) * zoom), int((y2 - y1) * zoom)
    nx1 = max(0, cx - nw // 2)
    ny1 = max(0, cy - nh // 2)
    nx2 = min(w, cx + nw // 2)
    ny2 = min(h, cy + nh // 2)
    return nx1, ny1, nx2, ny2


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """两个矩形框的交并比(用于去除 YOLO 与全图 OCR 的重复检测)。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def detect_private_text(
    image: np.ndarray,
    *,
    yolo_conf: float = config.YOLO_CONF,
    min_ocr_conf: float = config.OCR_MIN_CONFIDENCE,
    zoom: float = 1.5,
    llm_api_key: Optional[str] = None,
    full_image_ocr: bool = True,
    fallback_ocr_conf: float = 0.3,
    ocr_upscale: float = 2.0,
) -> list[TextBox]:
    """检测图片中的私密文本区域。

    双路径:
    1. YOLO 检测文本区域 → 放大裁剪 → OCR(定位精准,但对小图/模糊图可能漏检)
    2. 全图 OCR 兜底(默认开启):先放大再整图识别,自带 bbox,与 YOLO 结果按 IOU 去重。
       小图上 EasyOCR 识别不稳定,放大后置信度显著提升。

    :param image: BGR 图像数组
    :param full_image_ocr: 是否启用全图 OCR 兜底(图片越大越耗时)
    :param fallback_ocr_conf: 兜底路径的 OCR 置信度下限
    :param ocr_upscale: 兜底路径 OCR 前的放大倍数(bbox 自动映射回原图)
    :return: TextBox 列表,is_private=True 表示 LLM 判定含隐私信息
    :raises RuntimeError: 识别出文本但未配置 LLM API key
    """
    yolo = get_yolo()
    reader = get_ocr_reader()

    text_boxes: list[TextBox] = []

    # ---- 路径 1:YOLO 定位 + 裁剪 OCR ----
    results = yolo(image, conf=yolo_conf, verbose=False)
    for box in results[0].boxes.xyxy.cpu().numpy():
        nx1, ny1, nx2, ny2 = _zoomed_crop(image, box, zoom)
        crop = image[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            continue

        ocr_results = reader.readtext(
            _preprocess_crop(crop),
            text_threshold=config.OCR_TEXT_THRESHOLD,
            low_text=config.OCR_LOW_TEXT,
        )

        texts, confs = [], []
        for _, text, conf in ocr_results:
            if conf >= min_ocr_conf and text.strip():
                texts.append(text.strip())
                confs.append(float(conf))
        if not texts:
            continue

        text_boxes.append(
            TextBox(
                bbox=(nx1, ny1, nx2, ny2),
                text=" ".join(texts),
                confidence=max(confs),
            )
        )

    # ---- 路径 2:全图 OCR 兜底(YOLO 漏检时仍能发现文字) ----
    if full_image_ocr:
        # 小图上 EasyOCR 识别不稳定,先放大再识别,bbox 坐标映射回原图
        up = image
        if ocr_upscale > 1:
            up = cv2.resize(
                image, None, fx=ocr_upscale, fy=ocr_upscale, interpolation=cv2.INTER_CUBIC
            )
        for bbox4, text, conf in reader.readtext(
            up,
            text_threshold=config.OCR_TEXT_THRESHOLD,
            low_text=config.OCR_LOW_TEXT,
        ):
            if conf < fallback_ocr_conf or not text.strip():
                continue
            # 单字符块多为模糊噪声,过滤
            if len(text.strip()) < 2:
                continue
            # EasyOCR 返回 4 点四边形,转成矩形并映射回原图坐标
            xs = [p[0] / ocr_upscale for p in bbox4]
            ys = [p[1] / ocr_upscale for p in bbox4]
            box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            # 与 YOLO 已检出的区域高度重叠则跳过(避免重复块)
            if any(_iou(box, tb.bbox) > 0.5 for tb in text_boxes):
                continue
            text_boxes.append(
                TextBox(bbox=box, text=text.strip(), confidence=float(conf))
            )

    # LLM 批量判断,按编号对齐回 bbox
    if text_boxes:
        for index, privacy_type in classify_text_blocks(
            [tb.text for tb in text_boxes], api_key=llm_api_key
        ):
            text_boxes[index].is_private = True
            text_boxes[index].privacy_type = privacy_type

    # 规则层兜底:标准格式证件/号码直接标记(即使 LLM 漏判)
    for tb in text_boxes:
        if not tb.is_private:
            rule_type = _match_strong_pii(tb.text)
            if rule_type:
                tb.is_private = True
                tb.privacy_type = rule_type

    return text_boxes
