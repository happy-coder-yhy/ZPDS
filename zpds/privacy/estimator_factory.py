"""Privacy 后端工厂 — 根据配置与路由组装检测器/分类器/遮挡器。

对标 hands/ 的 HandEstimator 注入模式：factory 负责实例化后端，
pipeline 只依赖 Protocol，不直接导入后端类。
"""

from __future__ import annotations

from dataclasses import dataclass

from zpds.privacy.backend_router import PrivacyBackendPolicy
from zpds.privacy.config import PrivacyConfig
from zpds.privacy.contracts import FaceDetector, PIIClassifier, TextDetector


class PrivacyEstimatorError(RuntimeError):
    """后端不可用或配置错误。"""


@dataclass(frozen=True)
class EstimatorRuntime:
    """已解析的后端运行时信息（用于 manifest 追溯）。"""

    face_backend: str = ""
    face_model_path: str = ""
    face_applicable: bool = False

    text_backend: str = ""
    text_ocr_langs: tuple[str, ...] = ()
    text_applicable: bool = False

    pii_backend: str = ""
    pii_llm_url: str = ""
    pii_llm_model: str = ""

    llm_available: bool = False

    def to_dict(self) -> dict:
        return {
            "face": {
                "backend": self.face_backend,
                "model_path": self.face_model_path,
                "applicable": self.face_applicable,
            },
            "text": {
                "backend": self.text_backend,
                "ocr_langs": list(self.text_ocr_langs),
                "applicable": self.text_applicable,
            },
            "pii": {
                "backend": self.pii_backend,
                "llm_url": self.pii_llm_url,
                "llm_model": self.pii_llm_model,
                "llm_available": self.llm_available,
            },
        }


class PrivacyEstimatorFactory:
    """根据 PrivacyConfig + PrivacyBackendPolicy 生产后端实例。

    所有重依赖（torch、ultralytics、easyocr）的导入仅在工厂方法内发生，
    基础 ``import zpds.privacy`` 不受影响。
    """

    def __init__(
        self,
        config: PrivacyConfig,
        policy: PrivacyBackendPolicy,
    ) -> None:
        self._config = config
        self._policy = policy

    # ---- face ----

    def create_face_detector(self) -> FaceDetector | None:
        """创建人脸检测器。不适用时返回 None。"""
        if not self._policy.face_enabled:
            return None

        from zpds.privacy.backends.face_yolo11n import YOLOFaceDetector

        return YOLOFaceDetector(
            confidence_threshold=self._config.face_confidence,
        )

    # ---- text ----

    def create_text_detector(self) -> TextDetector | None:
        """创建文本检测器。不适用时返回 None。"""
        if not self._policy.text_enabled:
            return None

        from zpds.privacy.backends.text_easyocr import EasyOCRTextDetector

        return EasyOCRTextDetector(
            yolo_conf=0.25,  # 来自 config 的默认值
            min_ocr_conf=self._config.text_confidence,
        )

    # ---- pii ----

    def create_pii_classifier(self) -> PIIClassifier:
        """创建 PII 分类器。LLM 不可用时抛异常。"""
        api_key = self._config.llm_api_key
        if not api_key:
            raise PrivacyEstimatorError(
                "LLM API key 未配置。请设置 DASHSCOPE_API_KEY 环境变量 "
                "或在项目根目录创建 .env 文件。"
            )

        from zpds.privacy.backends.pii_llm import LLMPIIClassifier

        return LLMPIIClassifier(
            api_key=api_key,
            base_url=self._config.pii_llm_url,
            model=self._config.pii_llm_model,
            timeout=self._config.pii_llm_timeout_s,
            cache_enabled=self._config.pii_cache_enabled,
        )

    # ---- runtime info ----

    def build_runtime(self) -> EstimatorRuntime:
        """构建运行时信息（用于 manifest）。"""
        llm_available = bool(self._config.llm_api_key)
        return EstimatorRuntime(
            face_backend=self._policy.face_backend,
            face_model_path=self._config.face_model_path,
            face_applicable=self._policy.face_enabled,

            text_backend=self._policy.text_backend,
            text_ocr_langs=self._config.text_ocr_langs,
            text_applicable=self._policy.text_enabled,

            pii_backend=self._policy.pii_backend,
            pii_llm_url=self._config.pii_llm_url,
            pii_llm_model=self._config.pii_llm_model,

            llm_available=llm_available,
        )


__all__ = [
    "EstimatorRuntime",
    "PrivacyEstimatorError",
    "PrivacyEstimatorFactory",
]
