"""PII 分类后端 — Qwen LLM（OpenAI 兼容接口）。

实现 ``PIIClassifier`` Protocol。LLM 是 PII 分类的**唯一后端**；
不可用时流程明确失败，不产出脱敏产物。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

import requests

from zpds.privacy import config as _cfg
from zpds.privacy.schemas import PIIClassification, TextDetection

_SYSTEM_PROMPT = """你是一名隐私信息审查专家。用户会给你一张图片中 OCR 识别出的文本块列表,每个文本块有编号和内容。

你的任务:判断每个文本块是否包含个人隐私信息(PII)。隐私信息包括但不限于:
- 姓名、人名(如"患者:张三"中的"张三")
- 身份证号、护照号、港澳通行证号、驾驶证号、车牌号
- 手机号、座机号、邮箱地址
- 家庭住址、可定位到个人的完整地址
- 银行卡号、账号
- 出生日期、完整生日
- 病历号、住院号、体检单号、工号、学号、员工号等编号
- 社交账号(微信号、QQ 号等)

保守原则:
- 无明确含义的纯数字串(4 位及以上, 如 "123450"),无法判断是普通数字还是证件号/账号编号时,一律按隐私处理(宁严勿松,因为脱敏场景下漏掉比多模糊代价大)
- 上下文明确是普通数字的除外(如 "共 12 人"、"2026 年"、"准确率 98.5%")

注意(不算隐私):
- 普通日期(如期刊发行日期)、机构名称、药品名、疾病名、疾病症状描述
- "X 主任医师" 中"主任医师"是职称不算,但人名算
- 不含个人标识的学术内容、统计数字不算

只回复 JSON,不要多余文字。格式:
{"private_blocks": [{"index": 编号, "privacy_type": "类别(如:姓名/身份证号/手机号)", "reason": "简短理由"}]}
若都不含隐私,返回 {"private_blocks": []}"""


def _chat_completion(messages: list[dict], *, api_key: str, base_url: str, model: str, timeout: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM 未返回有效 JSON,回复前 200 字符: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class LLMPIIClassifier:
    """Qwen LLM PII 分类器，实现 ``PIIClassifier`` Protocol。

    LLM 不可用时流程失败（PRIVACY_LLM_UNAVAILABLE），不产出脱敏产物。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 30,
        cache_enabled: bool = True,
    ) -> None:
        self._api_key = api_key or _cfg.DASHSCOPE_API_KEY
        self._base_url = base_url or _cfg.LLM_BASE_URL
        self._model = model or _cfg.LLM_MODEL
        self._timeout = timeout
        self._cache_enabled = cache_enabled
        self._text_cache: dict[str, tuple[str, str]] = {}  # text_hash → (category, decision)

    # ---- PIIClassifier Protocol ----

    def classify(self, texts: list[TextDetection]) -> list[PIIClassification]:
        """对文本块做隐私分类。

        Returns:
            与输入一一对应的 PIIClassification 列表。
        """
        if not texts:
            return []

        if not self._api_key:
            raise RuntimeError(
                "未找到 DASHSCOPE_API_KEY。LLM 不可用，脱敏流程失败。\n"
                "  设置环境变量: set DASHSCOPE_API_KEY=sk-xxxx\n"
                "  或创建 .env 文件: DASHSCOPE_API_KEY=sk-xxxx"
            )

        results: list[PIIClassification] = []
        uncached_indices: list[int] = []
        uncached_texts: list[TextDetection] = []

        for i, td in enumerate(texts):
            if self._cache_enabled:
                cache_key = self._text_hash(td.text)
                cached = self._text_cache.get(cache_key)
                if cached is not None:
                    category, decision = cached
                    results.append(PIIClassification(
                        text=td,
                        category=category,  # type: ignore[arg-type]
                        decision=decision,    # type: ignore[arg-type]
                        confidence=0.95,
                        classifier="llm",
                    ))
                    continue
            uncached_indices.append(i)
            uncached_texts.append(td)

        if uncached_texts:
            llm_results = self._classify_batch(uncached_texts)
            for batch_idx, (category, decision) in enumerate(llm_results):
                original_idx = uncached_indices[batch_idx]
                td = texts[original_idx]
                result = PIIClassification(
                    text=td,
                    category=category,  # type: ignore[arg-type]
                    decision=decision,   # type: ignore[arg-type]
                    confidence=0.9,
                    classifier="llm",
                )
                # 插入正确位置
                results.insert(original_idx, result)
                # 缓存
                if self._cache_enabled:
                    self._text_cache[self._text_hash(td.text)] = (category, decision)

        return results

    def close(self) -> None:
        """释放 LLM 连接（无状态）。"""
        self._text_cache.clear()

    # ---- internal ----

    def _classify_batch(self, texts: list[TextDetection]) -> list[tuple[str, str]]:
        """调用 LLM 批量分类。"""
        blocks = [{"index": i, "text": td.text} for i, td in enumerate(texts)]
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(blocks, ensure_ascii=False)},
        ]
        reply = _chat_completion(
            messages,
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._model,
            timeout=self._timeout,
        )
        llm_result = _extract_json(reply)

        # 默认全为 keep
        result_map: dict[int, tuple[str, str]] = {
            i: ("unknown", "keep") for i in range(len(texts))
        }
        for item in llm_result.get("private_blocks", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(texts):
                result_map[idx] = (
                    str(item.get("privacy_type", "unknown")),
                    "mask",
                )
        return [result_map[i] for i in range(len(texts))]

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---- 向后兼容函数（旧 llm.py API） ----

def classify_text_blocks(
    texts: list[str],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> list[tuple[int, str]]:
    """判断每个文本块是否含私密信息（旧 API）。

    Returns:
        [(块索引, 隐私类别)] 列表
    """
    key = api_key or _cfg.DASHSCOPE_API_KEY
    if not key:
        raise RuntimeError(
            "未找到 DASHSCOPE_API_KEY。请先配置(二选一):\n"
            "  1. 设置环境变量:  set DASHSCOPE_API_KEY=sk-xxxx\n"
            "  2. 调用时传入: classify_text_blocks(texts, api_key=\"sk-xxxx\")"
        )

    blocks = [{"index": i, "text": t} for i, t in enumerate(texts)]
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(blocks, ensure_ascii=False)},
    ]
    reply = _chat_completion(
        messages,
        api_key=key,
        base_url=base_url or _cfg.LLM_BASE_URL,
        model=model or _cfg.LLM_MODEL,
        timeout=30,
    )
    result = _extract_json(reply)
    private = []
    for item in result.get("private_blocks", []):
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(texts):
            private.append((index, str(item.get("privacy_type", ""))))
    return private


__all__ = [
    "LLMPIIClassifier",
    "classify_text_blocks",
]
