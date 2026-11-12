# -*- coding: utf-8 -*-
"""第 6 章：单路视频的端到端流水线。

帧序列 -> 检测（infer.py）-> 单应投影到 BEV -> 跟踪（tracking.py）
      -> 轨迹级过滤 -> 过栏判决（counting.py）-> 事件表 / 计数 / 守恒校验

跟踪与计数只依赖 numpy，因此没有检测权重时也能用桩检测器跑通（cli demo）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .config import SceneConfig
from .counting import ConservationLedger, CrossingEvent, GateCounter, save_events
from .geometry import apply_homography, contact_point, obb_from_xywhr
from .tracking import LiteTracker, Observation, Track, TrackFilter


@dataclass
class FrameObs:
    """一帧的检测结果（图像坐标）。"""

    frame: int
    t: float
    boxes: np.ndarray            # (N,5) cx, cy, w, h, angle(rad)
    classes: np.ndarray          # (N,)
    confs: np.ndarray            # (N,)
    illuminance: Optional[float] = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 5)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        self.confs = np.asarray(self.confs, dtype=np.float64).reshape(-1)


@dataclass
class PipelineResult:
    events: List[CrossingEvent] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    filter_stats: Dict[str, int] = field(default_factory=dict)
    frames: int = 0
    tracks_total: int = 0

    def summary(self) -> str:
        return (
            f"frames={self.frames} tracks={self.tracks_total} "
            f"events={len(self.events)} out={self.counts.get('out', 0)} "
            f"in={self.counts.get('in', 0)} filter={self.filter_stats}"
        )


class CountingPipeline:
    def __init__(self, cfg: SceneConfig, fps: Optional[float] = None):
        self.cfg = cfg
        self.fps = float(fps or cfg.fps)
        self.tracker = LiteTracker(cfg, self.fps)
        self.filter = TrackFilter(cfg)
        self.counter = GateCounter(cfg)
        self.illuminance: Optional[float] = None
        self._n_frames = 0
        self._n_tracks = 0
        self._seen_tracks = set()
        self._decisions: Dict[int, bool] = {}

    # ---------- 单帧 ----------
    def project(self, fo: FrameObs) -> List[Observation]:
        """图像 OBB -> BEV 着地点 + 足迹尺度。"""
        cfg = self.cfg
        obs: List[Observation] = []
        for box, cls, conf in zip(fo.boxes, fo.classes, fo.confs):
            corners = obb_from_xywhr(box)                 # 图像坐标
            bev = apply_homography(cfg.H, corners)        # BEV 坐标
            # 着地点先在图像里取最低边中点，再投影——比"投影后在 BEV 取最低点"稳
            contact = apply_homography(cfg.H, contact_point(corners).reshape(1, 2))[0]
            w = float(np.max(bev[:, 0]) - np.min(bev[:, 0]))
            h = float(np.max(bev[:, 1]) - np.min(bev[:, 1]))
            span = max(w, h)
            if span > cfg.body_length_m * 1.8:
                scale = cfg.body_length_m * 1.8 / span
                w, h = w * scale, h * scale
            obs.append(
                Observation(
                    pos=contact,
                    size=np.array([max(w, h), max(min(w, h), 1e-3)]),
                    cls=int(cls),
                    conf=float(conf),
                    corners=corners,
                    box_h_px=float(box[3]),
                    w_img=float(box[2]),
                    h_img=float(box[3]),
                )
            )
        return obs

    def step(self, frame: int, t: float, fo: FrameObs) -> List[Track]:
        self._n_frames += 1
        if fo.illuminance is not None:
            self.illuminance = float(fo.illuminance)
            self.counter.set_illuminance(self.illuminance)
        obs = self.project(fo)
        tracks = self.tracker.update(t, frame, obs)

        goats: List[Track] = []
        for trk in tracks:
            if not trk.confirmed:
                continue
            if trk.box_h_px and trk.box_h_px < self.cfg.min_box_h_px:
                continue
            ok = self._decisions.get(trk.id)
            if ok is None:
                ok, reason = self.filter.decide(trk)
                self._decisions[trk.id] = bool(ok)
                if not ok:
                    trk.tag = reason
            if not ok:
                continue
            goats.append(trk)
            if trk.id not in self._seen_tracks:
                self._seen_tracks.add(trk.id)
                self._n_tracks += 1
        self.counter.update(t, frame, goats)
        return tracks

    # ---------- 全序列 ----------
    def run(self, frames: Iterable[FrameObs]) -> PipelineResult:
        for fo in frames:
            self.step(fo.frame, fo.t, fo)
        return self.finish()

    def finish(self) -> PipelineResult:
        counts = dict(self.counter.counts)
        return PipelineResult(
            events=list(self.counter.events),
            counts=counts,
            filter_stats=dict(self.filter.stats),
            frames=self._n_frames,
            tracks_total=self._n_tracks,
        )


def run_video(
    cfg: SceneConfig,
    detector,
    source: str,
    out_events: Optional[str] = None,
) -> PipelineResult:
    """读取视频，逐帧检测 + 计数。detector 需实现 __call__(frame, t) -> FrameObs。"""
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
    pipe = CountingPipeline(cfg, fps)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            fo = detector(frame, t)
            fo.frame = idx
            fo.t = t
            pipe.step(idx, t, fo)
            idx += 1
    finally:
        cap.release()
    result = pipe.finish()
    if out_events:
        save_events(result.events, out_events)
    return result
