"""
时间戳检查：帧数一致性、间隔异常检测。
"""


def check_frame_count(index_count: int, meta_count: int) -> dict:
    """比较 index.jsonl 帧数与 meta.json 声明的帧数。

    Returns:
        {"equal": bool, "error": int, "index_count": int, "meta_count": int}
    """
    equal = index_count == meta_count
    error = 0 if equal else abs(index_count - meta_count)
    return {
        "equal": equal,
        "error": error,
        "index_count": index_count,
        "meta_count": meta_count,
    }


def check_timestamp_gaps(
    timestamps: list,
    max_gap_ms: float = 40.0,
    max_print: int = 10,
) -> dict:
    """检测时间戳间隔异常（丢帧 / 中断）。

    Args:
        timestamps: 纳秒时间戳列表（已排序）
        max_gap_ms: 超过此毫秒数视为异常，默认 40ms（30fps 下正常~33ms）
        max_print: 打印前 N 条异常

    Returns:
        {"gap_count": int, "gap_list": [(frame_idx, gap_ms), ...]}
    """
    gap_list = []

    for i in range(1, len(timestamps)):
        gap_ns = timestamps[i] - timestamps[i - 1]
        gap_ms = gap_ns / 1_000_000
        if gap_ms > max_gap_ms:
            gap_list.append((i, gap_ms))

    return {
        "gap_count": len(gap_list),
        "gap_list": gap_list,
        "max_print": max_print,
    }
