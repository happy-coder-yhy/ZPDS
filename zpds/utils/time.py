"""纳秒时间工具。"""


def ns_to_s(ns: int) -> float:
    """纳秒 → 秒。"""
    return ns / 1_000_000_000


def ns_to_ms(ns: int) -> float:
    """纳秒 → 毫秒。"""
    return ns / 1_000_000


def s_to_ns(s: float) -> int:
    """秒 → 纳秒。"""
    return int(s * 1_000_000_000)
