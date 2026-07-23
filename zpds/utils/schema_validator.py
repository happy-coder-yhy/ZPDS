"""JSON Schema 校验工具。"""

import json


def validate_json(data: dict, schema: dict) -> list[str]:
    """校验 data 是否符合 schema，返回错误列表。"""
    raise NotImplementedError


def load_schema(name: str) -> dict:
    """加载内置 schema。"""
    raise NotImplementedError
