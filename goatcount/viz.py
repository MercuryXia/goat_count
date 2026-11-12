# -*- coding: utf-8 -*-
"""可视化：BEV 平面图 + 原图画框。只依赖 numpy / cv2。

产出的两张图对应论文里的"计数带与迟滞区示意"和"事件快照"。
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .counting import CrossingEvent
from .geometry import apply_homography, obb_from_xywhr
from .tracking import Track

try:  # cv2 在 yolo 环境里有；缺失时 draw_* 抛清晰错误
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("需要 opencv-python（cv2）才能画图")


COLORS = {
    "goat": (60, 180, 75),
    "dog": (30, 120, 220),
    "person": (200, 120, 30),
}


def bev_canvas(
    cfg,
    tracks: Sequence[Track],
    size: Tuple[int, int] = (640, 640),
    meters: float = 12.0,
) -> np.ndarray:
    """画 BEV 俯视图：计数线、迟滞带、轨迹点。"""
    _require_cv2()
    w, h = size
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    scale = min(w, h) / meters
    cx, cy = w / 2.0, h / 2.0

    def to_px(p: np.ndarray) -> Tuple[int, int]:
        return int(cx + p[0] * scale), int(cy - p[1] * scale)

    p1 = np.array(cfg.line_p1, dtype=float)
    p2 = np.array(cfg.line_p2, dtype=float)
    # 迟滞带（两条平行虚线）
    d = p2 - p1
    n = np.array([-d[1], d[0]], dtype=float)
    n = n / (np.linalg.norm(n) + 1e-9)
    for k, color in ((0, (0, 0, 0)), (1, (0, 0, 255)), (-1, (0, 0, 255))):
        off = n * cfg.hysteresis_m * k
        a = to_px(p1 + off)
        b = to_px(p2 + off)
        cv2.line(img, a, b, color, 2 if k == 0 else 1, cv2.LINE_AA)

    for trk in tracks:
        c = COLORS.get(trk.cls_name, (120, 120, 120))
        p = to_px(trk.pos)
        cv2.circle(img, p, 4, c, -1, cv2.LINE_AA)
        cv2.putText(img, f"{trk.id}", (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

    cv2.putText(
        img,
        f"BEV  gate  hysteresis={cfg.hysteresis_m:.2f}m  margin={cfg.margin_m:.2f}m",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return img


def draw_frame(
    frame: np.ndarray,
    cfg,
    tracks: Sequence[Track],
    events: Optional[Sequence[CrossingEvent]] = None,
) -> np.ndarray:
    """在原图上画 OBB、ID、方向与当前计数。"""
    _require_cv2()
    out = frame.copy()
    for trk in tracks:
        if trk.corners is None:
            continue
        c = COLORS.get(trk.cls_name, (120, 120, 120))
        pts = np.asarray(trk.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, c, 2, cv2.LINE_AA)
        label = f"#{trk.id} {trk.cls_name} {trk.conf:.2f}"
        org = (int(trk.corners[:, 0].min()), int(trk.corners[:, 1].min()) - 4)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

    if events:
        out_n = sum(1 for e in events if e.direction == "out")
        in_n = sum(1 for e in events if e.direction == "in")
        cv2.putText(
            out,
            f"OUT {out_n}   IN {in_n}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def save_snapshot(frame: np.ndarray, path: str) -> None:
    _require_cv2()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, frame)


def contact_overlay(frame: np.ndarray, cfg, points: Iterable[np.ndarray]) -> np.ndarray:
    """把 BEV 点反投影回图像，检查单应是否标定正确。"""
    _require_cv2()
    out = frame.copy()
    Hinv = np.linalg.inv(cfg.H)
    for p in points:
        q = apply_homography(Hinv, np.asarray(p, dtype=float).reshape(1, 2))[0]
        cv2.circle(out, (int(q[0]), int(q[1])), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return out
