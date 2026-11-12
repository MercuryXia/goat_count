# -*- coding: utf-8 -*-
"""合成场景：没有真实视频时，用"世界坐标 -> 单应 -> 图像"的方式生成带真值的过栏序列。

这样做的好处是能真正验证几何链路：羊在世界坐标里走直线，经单应投影到图像，
检测器只在图像里工作，流水线再把检测结果投影回 BEV 做计数。若单应或着地点算错，
计数会立刻对不上真值。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SceneConfig
from .data import degrade_low_light
from .geometry import apply_homography

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


# 世界坐标（米）：x 正向 = 出栏方向；y 为横向
WORLD_W, WORLD_H = 9.6, 5.4

# 图像里的梯形（近大远小），与上面的世界矩形对应
IMAGE_QUAD = [(70, 505), (890, 505), (700, 130), (260, 130)]
BEV_QUAD = [(-4.8, -2.7), (4.8, -2.7), (4.8, 2.7), (-4.8, 2.7)]


@dataclass
class Actor:
    kind: str                # goat / dog / person
    y: float                 # 横向位置（米）
    x: float                 # 当前纵向位置（米）
    speed: float             # m/s
    start_x: float
    end_x: float
    gait_hz: float
    phase: float = 0.0
    body_len: float = 0.70
    body_h: float = 0.55
    crossed: bool = False
    crossed_t: Optional[float] = None


@dataclass
class DemoResult:
    frames: int
    gt_out: int
    gt_in: int
    gt_events: List[dict] = field(default_factory=list)


COLORS_BGR = {"goat": (60, 170, 60), "dog": (200, 140, 40), "person": (40, 120, 200)}


def _to_image(cfg: SceneConfig, p: np.ndarray) -> np.ndarray:
    Hinv = np.linalg.inv(cfg.H)
    q = apply_homography(Hinv, np.asarray(p, dtype=np.float64).reshape(1, 2))[0]
    return q


def _px_per_meter(cfg: SceneConfig, p: np.ndarray) -> float:
    a = _to_image(cfg, p)
    b = _to_image(cfg, p + np.array([0.2, 0.0]))
    return float(np.linalg.norm(b - a) / 0.2)


def make_scene(
    cfg: SceneConfig,
    out_dir: str,
    n_out: int = 12,
    n_in: int = 10,
    fps: float = 25.0,
    duration: Optional[float] = None,
    dogs: int = 2,
    persons: int = 2,
    night_from: float = 999.0,
    seed: int = 0,
    video: Optional[str] = None,
    size: Tuple[int, int] = (960, 540),
) -> DemoResult:
    """生成视频 + 真值事件。night_from 之后叠加低照度退化（模拟暮归）。"""
    if cv2 is None:
        raise RuntimeError("需要 opencv-python")
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 出栏：从栏内（x<0）走到栏外（x>0）；归栏反向
    # 相邻个体错开 2 s（约 1.5–2 m 间距），否则同色椭圆会粘成一条轮廓
    actors: List[Actor] = []
    interval = 2.0
    for i in range(n_out):
        t0 = 1.0 + i * interval
        actors.append(
            Actor(
                kind="goat",
                y=float(rng.uniform(-2.3, 2.3)),
                x=-4.6,
                speed=float(rng.uniform(0.75, 1.05)),
                start_x=-4.6,
                end_x=4.6,
                gait_hz=float(rng.uniform(1.5, 1.9)),
                phase=float(rng.uniform(0, 6.28)),
            )
        )
        actors[-1].crossed_t = None  # 用 phase 调度出场时刻
        actors[-1].phase = t0

    in_base = 1.0 + n_out * interval + 4.0
    for i in range(n_in):
        t0 = in_base + i * interval
        a = Actor(
            kind="goat",
            y=float(rng.uniform(-2.3, 2.3)),
            x=6.2 + 0.45 * i,
            speed=float(rng.uniform(0.7, 1.0)),
            start_x=6.2 + 0.45 * i,
            end_x=-4.6,
            gait_hz=float(rng.uniform(1.5, 1.9)),
            phase=t0,
        )
        actors.append(a)
    # 狗：只在栏内徘徊，不过线
    for i in range(dogs):
        a = Actor(
            kind="dog",
            y=float(rng.uniform(-2.0, 2.0)),
            x=float(rng.uniform(-4.2, -1.2)),
            speed=0.0,
            start_x=-4.2,
            end_x=-1.2,
            gait_hz=3.0,
            phase=float(rng.uniform(0, 6.28)),
            body_len=0.55,
            body_h=0.40,
        )
        actors.append(a)
    for i in range(persons):
        a = Actor(
            kind="person",
            y=float(rng.uniform(-2.2, 2.2)),
            x=-3.5,
            speed=0.0,
            start_x=-3.5,
            end_x=1.5,
            gait_hz=1.9,
            phase=float(rng.uniform(0, 6.28)),
            body_len=0.45,
            body_h=1.70,
        )
        actors.append(a)

    dur = duration or (in_base + max(0, n_in - 1) * interval + 14.0)
    n_frames = int(dur * fps)
    writer = None
    if video:
        os.makedirs(os.path.dirname(os.path.abspath(video)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video, fourcc, fps, size)

    gt: List[dict] = []
    for fi in range(n_frames):
        t = fi / fps
        frame = _background(size, cfg)
        illum = 120.0
        if t >= night_from:
            illum = 8.0
        for a in actors:
            _advance(a, t, n_frames)
            lo, hi = min(a.start_x, a.end_x) - 0.5, max(a.start_x, a.end_x) + 0.5
            if a.x < lo or a.x > hi:
                continue
            bob = 0.06 * np.sin(2 * np.pi * a.gait_hz * (t + a.phase))
            p = np.array([a.x, a.y + 0.05 * np.sin(1.3 * t + a.phase)])
            q = _to_image(cfg, p)
            s = _px_per_meter(cfg, p)
            w_px = max(8.0, a.body_len * s * 1.35)
            h_px = max(8.0, a.body_h * s * (1.0 + bob * 2))
            _draw_actor(frame, q, w_px, h_px, a.kind, bob)
            # 真值事件：中线 x=0 穿越
            if not a.crossed and a.kind == "goat":
                if a.start_x < 0 <= a.x:
                    a.crossed = True
                    a.crossed_t = t
                    gt.append({"t": round(t, 3), "direction": "out", "y": round(a.y, 3)})
                elif a.start_x > 0 >= a.x:
                    a.crossed = True
                    a.crossed_t = t
                    gt.append({"t": round(t, 3), "direction": "in", "y": round(a.y, 3)})

        if t >= night_from:
            frame = degrade_low_light(frame, rng, gamma=2.4, gain=0.22, noise_sigma=6.0, motion_px=5, jpeg_q=55)
        if writer is not None:
            writer.write(frame)
        if fi == 0:
            cv2.imwrite(os.path.join(out_dir, "first_frame.png"), frame)

    if writer is not None:
        writer.release()
    with open(os.path.join(out_dir, "gt_events.json"), "w", encoding="utf-8") as fh:
        json.dump(gt, fh, ensure_ascii=False, indent=2)
    return DemoResult(
        frames=n_frames,
        gt_out=sum(1 for e in gt if e["direction"] == "out"),
        gt_in=sum(1 for e in gt if e["direction"] == "in"),
        gt_events=gt,
    )


def _advance(a: Actor, t: float, n_frames: int) -> None:
    if a.kind in ("dog", "person"):
        # 栏内来回走
        period = 9.0
        u = (t / period + a.phase) % 1.0
        tri = 2 * u if u < 0.5 else 2 * (1 - u)
        a.x = a.start_x + (a.end_x - a.start_x) * tri
        return
    t0 = a.phase  # 出场时刻
    if t < t0:
        return
    a.x = a.start_x + a.speed * (t - t0) * (1 if a.end_x > a.start_x else -1)


def _background(size: Tuple[int, int], cfg: SceneConfig) -> np.ndarray:
    w, h = size
    # 背景必须低饱和（灰度），否则 HSV 颜色桩会把地面误检成羊
    img = np.full((h, w, 3), (78, 78, 78), dtype=np.uint8)
    quad = np.array(IMAGE_QUAD, dtype=np.int32)
    cv2.fillPoly(img, [quad], (118, 118, 118))
    # 栏门中线（世界 x=0）
    p1 = _to_image(cfg, np.array([0.0, -2.7]))
    p2 = _to_image(cfg, np.array([0.0, 2.7]))
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (200, 200, 200), 2)
    return img


def _draw_actor(img: np.ndarray, q: np.ndarray, w_px: float, h_px: float, kind: str, bob: float) -> None:
    color = COLORS_BGR[kind]
    center = (int(q[0]), int(q[1] - h_px * 0.5 + bob * w_px * 2))
    axes = (int(max(3, w_px * 0.5)), int(max(3, h_px * 0.5)))
    cv2.ellipse(img, center, axes, 0, 0, 360, color, -1, cv2.LINE_AA)
    if kind == "goat":
        head = (int(center[0] + axes[0] * 0.9), int(center[1] - axes[1] * 0.3))
        # 用粗线把头连到躯干，保证颜色桩只产生一条轮廓（否则一只羊会被拆成两个目标）
        cv2.line(img, center, head, color, max(3, int(axes[1] * 0.8)), cv2.LINE_AA)
        cv2.circle(img, head, max(2, int(axes[1] * 0.35)), color, -1, cv2.LINE_AA)
    elif kind == "person":
        cv2.line(img, center, (center[0], center[1] - axes[1]), color, max(3, int(axes[0] * 0.6)), cv2.LINE_AA)
        cv2.circle(img, (center[0], center[1] - axes[1]), max(3, int(axes[1] * 0.32)), color, -1, cv2.LINE_AA)
