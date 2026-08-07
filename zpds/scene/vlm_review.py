"""OpenAI 兼容 API 的 VLM 场景-动作一致性复核器（人员 B）。

密钥只从 ``scene.vlm.api_key_env`` 指定的环境变量读取，不写入配置、
日志或任何产物。API 不可用、超时、鉴权失败或响应不合法时明确失败，
禁止伪造复核结果。
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from zpds.scene.config import VLMConfig
from zpds.scene.schemas import SceneProposal, VLMReviewResult

DEFAULT_DASHSCOPE_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
VLM_MODEL_ENV = "VLM_MODEL"


class VLMUnavailableError(RuntimeError):
    """VLM API 不可用、超时、鉴权失败或响应不合法。"""


@dataclass(frozen=True)
class SceneLabels:
    """closed-set scene/task 标签集。"""

    scene_labels: tuple[str, ...]
    task_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, values in (
            ("scene_labels", self.scene_labels),
            ("task_labels", self.task_labels),
        ):
            if not values or any(not str(value).strip() for value in values):
                raise ValueError(f"{field_name} 不能为空且不能包含空标签")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} 不能包含重复标签")


def load_scene_labels(path: str | Path) -> SceneLabels:
    """从 YAML 加载 closed-set 标签集。"""

    import yaml

    labels_path = Path(path).expanduser()
    if not labels_path.is_file():
        raise FileNotFoundError(f"Scene 标签配置文件不存在: {labels_path}")
    with labels_path.open(encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if not isinstance(document, dict):
        raise TypeError("Scene 标签配置顶层必须是对象")
    return SceneLabels(
        scene_labels=tuple(
            str(value).strip()
            for value in document.get("scene_labels", ())
        ),
        task_labels=tuple(
            str(value).strip()
            for value in document.get("task_labels", ())
        ),
    )


def _encode_frame_data_url(frame_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", frame_bgr)
    if not success or encoded.size == 0:
        raise ValueError("代表帧 JPEG 编码失败")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    *,
    timeout_s: float,
    retries: int = 3,
    backoff_s: float = 1.0,
) -> dict[str, object]:
    """向 OpenAI 兼容端点发送请求并返回 JSON 对象。

    对瞬时网络错误（5xx / 超时 / 连接失败）退避重试 ``retries`` 次；
    4xx 与响应解析错误不重试（重试无意义）。
    """

    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        pass
    else:
        context.load_verify_locations(cafile=certifi.where())
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_s,
                context=context,
            ) as response:
                body = response.read()
            break
        except urllib.error.HTTPError as error:
            # 4xx 重试无意义（请求本身被拒），直接抛
            if error.code < 500:
                raise VLMUnavailableError(
                    f"VLM API 请求失败: HTTP {error.code}"
                ) from error
            last_error = error
        except urllib.error.URLError as error:
            last_error = error
        except TimeoutError as error:
            last_error = error
        if attempt < retries - 1:
            time.sleep(backoff_s * (attempt + 1))
    else:
        raise VLMUnavailableError(
            f"VLM API 请求失败（{retries} 次尝试后放弃）: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VLMUnavailableError(
            "VLM API 返回内容不是合法 JSON"
        ) from error
    if not isinstance(document, dict):
        raise VLMUnavailableError("VLM API 返回 JSON 顶层必须是对象")
    return document


def _parse_content(raw_content: object) -> dict[str, object]:
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise VLMUnavailableError("VLM 响应缺少文本内容")
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise VLMUnavailableError(
            "VLM 返回内容不是可解析的 JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise VLMUnavailableError("VLM 返回 JSON 顶层必须是对象")
    return parsed


class OpenAICompatibleVLMReviewer:
    """通过 OpenAI 兼容 API 复核场景-动作一致性。"""

    def __init__(
        self,
        config: VLMConfig,
        labels: SceneLabels | None = None,
        *,
        config_hash: str = "",
    ) -> None:
        self._config = config
        self._labels = labels
        self._config_hash = config_hash
        self._cache: dict[str, VLMReviewResult] = {}

    def _resolve_model(self) -> str:
        model = self._config.model.strip()
        if model:
            return model
        model = os.environ.get(VLM_MODEL_ENV, "").strip()
        if not model:
            raise VLMUnavailableError(
                f"未配置 VLM 模型：请在 scene.vlm.model 或环境变量 "
                f"{VLM_MODEL_ENV} 中指定"
            )
        return model

    def _resolve_api_key(self) -> str:
        env_name = self._config.api_key_env.strip() or "DASHSCOPE_API_KEY"
        api_key = os.environ.get(env_name, "").strip()
        if not api_key:
            raise VLMUnavailableError(
                f"环境变量 {env_name} 未设置 VLM API key"
            )
        return api_key

    def _build_prompt(
        self,
        scene: SceneProposal,
    ) -> str:
        scene_labels = (
            "、".join(self._labels.scene_labels)
            if self._labels
            else "未限定（自由描述）"
        )
        task_labels = (
            "、".join(self._labels.task_labels)
            if self._labels
            else "未限定（自由描述）"
        )
        return (
            "你是场景-动作一致性审核员。请观察给定场景的首、中、尾三帧，"
            "判断画面中的动作是否与场景一致。\n"
            f"scene 只能从以下闭集选择: {scene_labels}\n"
            f"task 只能从以下闭集选择: {task_labels}\n"
            "decision 只能是 consistent / inconsistent / unknown 之一。\n"
            "只输出 JSON，格式为: "
            '{"scene_label": "...", "task_label": "...", '
            '"decision": "...", "confidence": 0.0~1.0, "reasons": "..."}'
        )

    def _normalise_result(
        self,
        scene: SceneProposal,
        parsed: dict[str, object],
    ) -> VLMReviewResult:
        scene_label = str(parsed.get("scene_label", "")).strip()
        task_label = str(parsed.get("task_label", "")).strip()
        decision = str(parsed.get("decision", "")).strip()
        reasons = str(parsed.get("reasons", "")).strip()
        raw_confidence = parsed.get("confidence", 0.0)
        if (
            not isinstance(raw_confidence, (int, float))
            or isinstance(raw_confidence, bool)
        ):
            raise VLMUnavailableError(
                "VLM 返回的 confidence 不是数值"
            )
        confidence = float(raw_confidence)
        if decision not in {"consistent", "inconsistent", "unknown"}:
            raise VLMUnavailableError(f"VLM 返回非法 decision: {decision!r}")
        if not reasons:
            reasons = "VLM 未提供理由"
        label_issues: list[str] = []
        if self._labels is not None:
            if scene_label not in self._labels.scene_labels:
                label_issues.append(
                    f"scene_label={scene_label!r} 不在闭集内"
                )
                scene_label = "unknown"
            if task_label not in self._labels.task_labels:
                label_issues.append(f"task_label={task_label!r} 不在闭集内")
                task_label = "unknown"
        if label_issues:
            decision = "unknown"
            reasons = f"{reasons}；{'；'.join(label_issues)}"
        return VLMReviewResult(
            scene_id=scene.scene_id,
            scene_label=scene_label,
            task_label=task_label,
            decision=decision,  # type: ignore[arg-type]
            confidence=confidence,
            reasons=reasons,
            evidence_frame_uris=scene.evidence_uris,
            producer="zpds.scene.vlm",
            version="v1",
            config_hash=self._config_hash,
        )

    def review(
        self,
        scene: SceneProposal,
        representative_frames_rgb: Sequence[np.ndarray],
    ) -> VLMReviewResult:
        cached = self._cache.get(scene.scene_id)
        if cached is not None:
            return cached
        if len(representative_frames_rgb) != 3:
            raise ValueError("representative_frames_rgb 必须恰好 3 帧")
        base_url = (
            self._config.base_url.strip() or DEFAULT_DASHSCOPE_BASE_URL
        ).rstrip("/")
        model = self._resolve_model()
        api_key = self._resolve_api_key()
        data_urls = tuple(
            _encode_frame_data_url(frame_bgr)
            for frame_bgr in representative_frames_rgb
        )
        content: list[dict[str, object]] = [
            {"type": "text", "text": self._build_prompt(scene)}
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": data_url}}
            for data_url in data_urls
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "只输出符合格式要求的 JSON，不输出其他文字。",
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
        }
        document = _post_json(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_s=self._config.timeout_s,
        )
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VLMUnavailableError("VLM 响应缺少 choices")
        message = choices[0]
        if not isinstance(message, dict):
            raise VLMUnavailableError("VLM choices[0] 不是对象")
        parsed = _parse_content(message.get("message", {}).get("content"))
        result = self._normalise_result(scene, parsed)
        self._cache[scene.scene_id] = result
        return result


def select_review_queue(
    results: Sequence[VLMReviewResult],
    *,
    confidence_threshold: float,
) -> list[VLMReviewResult]:
    """inconsistent 或置信度低于阈值的复核结果进入人工复核队列。"""

    return [
        result
        for result in results
        if result.decision == "inconsistent"
        or result.confidence < confidence_threshold
    ]


__all__ = [
    "DEFAULT_DASHSCOPE_BASE_URL",
    "VLM_MODEL_ENV",
    "OpenAICompatibleVLMReviewer",
    "SceneLabels",
    "VLMUnavailableError",
    "load_scene_labels",
    "select_review_queue",
]
