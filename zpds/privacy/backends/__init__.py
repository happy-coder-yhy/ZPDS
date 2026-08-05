"""Privacy 后端实现包。

- face_yolo11n.py: YOLOv11n-face 人脸检测器
- text_easyocr.py: EasyOCR 文本识别器（+ 可选 YOLO 区域提议）
- pii_llm.py: Qwen LLM PII 分类器（OpenAI 兼容接口，含 text hash 缓存）
"""

from zpds.privacy.backends.face_yolo11n import YOLOFaceDetector, blur_faces, blur_regions
from zpds.privacy.backends.pii_llm import LLMPIIClassifier, classify_text_blocks
from zpds.privacy.backends.text_easyocr import EasyOCRTextDetector, get_ocr_reader, get_yolo

__all__ = [
    "EasyOCRTextDetector",
    "LLMPIIClassifier",
    "YOLOFaceDetector",
    "blur_faces",
    "blur_regions",
    "classify_text_blocks",
    "get_ocr_reader",
    "get_yolo",
]
