# -*- coding: utf-8 -*-
"""几何：单应矫正（图像 <-> BEV）、着地点提取、线与带的判定。

坐标约定：
* 图像坐标 (u, v)，原点左上，v 向下。
* BEV 坐标 (x, y)，单位米，x 向右、y 向前（可自由定义，只要与标定点一致）。
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def homography_from_points(
    image_points: Sequence[Sequence[float]],
    bev_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """由 4 组对应点求单应矩阵 H（图像 -> BEV），H 作用于齐次坐标。

    直接解 8 元线性方程组，比 cv2.findHomography 少一个依赖，且点恰好 4 个时是精确解。
    """
    src = np.asarray(image_points, dtype=np.float64).reshape(4, 2)
    dst = np.asarray(bev_points, dtype=np.float64).reshape(4, 2)
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i] = u
        b[2 * i + 1] = v
    try:
        h = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:  # 四点共线/重合
        raise ValueError("标定点退化（三点共线或重合），无法求单应矩阵") from exc
    H = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
    return H


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """把一个或一批点 (..., 2) 从图像坐标变换到 BEV。"""
    pts = np.asarray(points, dtype=np.float64)
    shape = pts.shape
    flat = pts.reshape(-1, 2)
    homo = np.concatenate([flat, np.ones((len(flat), 1))], axis=1)
    proj = homo @ np.asarray(H, dtype=np.float64).T
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    out = proj[:, :2] / w
    return out.reshape(shape)


def contact_point(corners: np.ndarray) -> np.ndarray:
    """OBB 四角 -> 地面着地点（最低那条边的中点）。

    斜俯视下，羊与地面的接触线是轮廓最低处；用四角里 y 最大的那条边中点近似，
    比用框中心更接近真实落点，单应矫正后才不会随体高漂移。
    """
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    mids = np.stack(
        [(c[0] + c[1]) / 2, (c[1] + c[2]) / 2, (c[2] + c[3]) / 2, (c[3] + c[0]) / 2]
    )
    return mids[int(np.argmax(mids[:, 1]))]


def contact_points(corners: np.ndarray) -> np.ndarray:
    """批量版 contact_point：corners (N,4,2) -> (N,2)。"""
    c = np.asarray(corners, dtype=np.float64)
    if c.ndim != 3:
        raise ValueError("corners 形状应为 (N,4,2)")
    mids = np.stack(
        [(c[:, 0] + c[:, 1]) / 2, (c[:, 1] + c[:, 2]) / 2,
         (c[:, 2] + c[:, 3]) / 2, (c[:, 3] + c[:, 0]) / 2],
        axis=1,
    )  # (N,4,2)
    idx = np.argmax(mids[:, :, 1], axis=1)
    return mids[np.arange(len(c)), idx]


def line_signed_distance(p: np.ndarray, p1: Sequence[float], p2: Sequence[float]) -> np.ndarray:
    """点到有向直线的带符号距离（米）。

    正负号表示位于直线哪一侧：以 p1->p2 为前进方向，左侧为正。
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    d = p2 - p1
    n = np.array([-d[1], d[0]], dtype=np.float64)  # 左法向
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("计数线两端点重合")
    n = n / norm
    return (np.asarray(p, dtype=np.float64) - p1) @ n


def polygon_area(corners: np.ndarray) -> np.ndarray:
    """Shoelace 面积，支持 (N,4,2)。"""
    c = np.asarray(corners, dtype=np.float64)
    x, y = c[..., 0], c[..., 1]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)
    )


def obb_from_xywhr(xywhr: Iterable[float]) -> np.ndarray:
    """(cx, cy, w, h, r) -> 4 个角点 (4,2)，与 cv2.boxPoints 同序。"""
    cx, cy, w, h, r = [float(v) for v in xywhr]
    cos, sin = np.cos(r), np.sin(r)
    dx, dy = w / 2.0, h / 2.0
    base = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    rot = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    return base @ rot.T + np.array([cx, cy])


def box_iou_bev(a: np.ndarray, b: np.ndarray) -> float:
    """BEV 轴对齐框 IoU，a/b = (x1,y1,x2,y2)，用于跟踪代价的辅助项。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0
