"""图像隐私脱敏库:人脸模糊 + 私密文本检测模糊。

核心入口:
    from privacy_protection import redact_image, redact_file, redact_directory

也提供底层模块:
    - text_pipeline.detect_private_text: YOLO → OCR → LLM 文本隐私检测
    - face_detector.detect_faces / blur_faces: 人脸检测与模糊
    - llm.classify_text_blocks: 直接调用 LLM 判断文本块
"""
from .redactor import redact_image, redact_file, redact_directory

__version__ = "0.1.0"
__all__ = ["redact_image", "redact_file", "redact_directory"]
