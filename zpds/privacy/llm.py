"""LLM 检验环节:给定 OCR 文本块,调用 Qwen API 判断哪些包含私密信息。

通过 OpenAI 兼容接口调用,默认指向阿里云 DashScope,无需本地推理。
返回结构化结果(按文本块编号),可直接对齐回 YOLO 的 bbox。
"""
from __future__ import annotations

import json
import re
from typing import Optional

import requests

from . import config

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
- 无明确含义的纯数字串(4 位及以上,如 "123450"),无法判断是普通数字还是证件号/账号编号时,一律按隐私处理(宁严勿松,因为脱敏场景下漏掉比多模糊代价大)
- 上下文明确是普通数字的除外(如 "共 12 人"、"2026 年"、"准确率 98.5%")

注意(不算隐私):
- 普通日期(如期刊发行日期)、机构名称、药品名、疾病名、疾病症状描述
- "X 主任医师" 中"主任医师"是职称不算,但人名算
- 不含个人标识的学术内容、统计数字不算

只回复 JSON,不要多余文字。格式:
{"private_blocks": [{"index": 编号, "privacy_type": "类别(如:姓名/身份证号/手机号)", "reason": "简短理由"}]}
若都不含隐私,返回 {"private_blocks": []}"""


def _chat_completion(messages: list[dict], *, api_key: str, base_url: str, model: str) -> str:
    """调用 OpenAI 兼容的 /chat/completions 接口,返回回复文本。"""
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
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict:
    """从 LLM 回复中稳健提取 JSON 对象(容忍 ```json 代码块包裹、前后杂文)。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM 未返回有效 JSON,回复前 200 字符: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def classify_text_blocks(
    texts: list[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> list[tuple[int, str]]:
    """判断每个文本块是否含私密信息。

    :param texts: 文本块列表,顺序与检测框一一对应
    :return: [(块索引, 隐私类别)] 列表,按输入顺序
    :raises RuntimeError: 未配置 API key 时
    """
    key = api_key or config.DASHSCOPE_API_KEY
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
        base_url=base_url or config.LLM_BASE_URL,
        model=model or config.LLM_MODEL,
    )
    result = _extract_json(reply)

    private = []
    for item in result.get("private_blocks", []):
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(texts):
            private.append((index, str(item.get("privacy_type", ""))))
    return private
