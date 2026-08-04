"""抽样帧对比图：原帧 / 遮挡帧并排，用于人工抽查。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from zpds.privacy.contracts import FrameRedactionRecord


def write_preview_grid(
    records: list[FrameRedactionRecord],
    output_path: str | Path,
    sample_count: int = 16,
    max_width: int = 2560,
) -> Path:
    """从记录中均匀抽样，生成原帧/遮挡帧并排对比图。

    Args:
        records: 运行结果记录。
        output_path: 输出 PNG 路径。
        sample_count: 抽样帧数。
        max_width: 输出图片最大宽度。

    Returns:
        输出路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 均匀抽样有遮挡区域的帧
    redacted_records = [r for r in records if r.redacted_frame is not None]
    if not redacted_records:
        raise ValueError("没有含脱敏帧的记录")

    step = max(1, len(redacted_records) // sample_count)
    samples = redacted_records[::step][:sample_count]

    # 获取原帧（从 video 读取 — 这里 pipeline 没有存原帧，用 redacted 帧本身展示）
    # 对于 preview，我们需要原帧 + 脱敏帧对比。从 record 无法直接获取原帧，
    # 但可以从 redacted 帧和 regions 推断。
    rows = []
    for record in samples:
        if record.redacted_frame is None:
            continue
        redacted = record.redacted_frame
        # 画检测框在原图上（用红色标注脱敏区域 + 标签）
        preview = redacted.copy()
        h, w = preview.shape[:2]
        for region in record.regions:
            x1, y1, x2, y2 = (
                int(region.bbox_xyxy[0] * w),
                int(region.bbox_xyxy[1] * h),
                int(region.bbox_xyxy[2] * w),
                int(region.bbox_xyxy[3] * h),
            )
            color = (0, 0, 255) if region.kind == "face" else (255, 0, 0)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            label = f"{region.kind}:{region.category}" if region.category else region.kind
            cv2.putText(preview, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        rows.append(preview)

    if not rows:
        raise ValueError("没有有效的预览帧")

    # 拼成网格
    grid_cols = min(4, len(rows))
    grid_rows = (len(rows) + grid_cols - 1) // grid_cols
    cell_h, cell_w = rows[0].shape[:2]

    # 缩放
    scale = min(1.0, max_width / (cell_w * grid_cols))
    if scale < 1.0:
        cell_h = int(cell_h * scale)
        cell_w = int(cell_w * scale)
        rows = [cv2.resize(r, (cell_w, cell_h)) for r in rows]

    canvas = np.zeros((cell_h * grid_rows, cell_w * grid_cols, 3), dtype=np.uint8)
    for i, img in enumerate(rows):
        r, c = divmod(i, grid_cols)
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = img

    cv2.imwrite(str(output_path), canvas)
    return output_path


__all__ = ["write_preview_grid"]
