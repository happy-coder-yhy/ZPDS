"""文件哈希工具。"""

import hashlib


def sha256_hex(path: str, chunk_size: int = 8192) -> str:
    """计算文件的 SHA-256 十六进制摘要。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
