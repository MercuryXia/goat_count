# -*- coding: utf-8 -*-
"""第 5 章：双向过栏计数（迟滞 + 位移余量 + 多帧确认 + 往返/重复合并 + 守恒校验）。

判决逻辑（对应开题报告 3.3.2）：

1. 每条轨迹在 BEV 中相对计数线求带符号距离 s；|s| <= hysteresis 视为"带内"。
2. 从带的一侧进入、从另一侧离开，且离开时 |s| >= margin（默认 0.5 个体长），
   且带内累计帧数 >= confirm_frames（按白天/夜间分别取值），才判为一次过栏。
3. 出栏与归栏方向各自使用一套确认帧数与余量，避免"晨出密集遮挡"与"暮归低光漏检"
   两种相反的误差机理共用一套阈值。
4. 同一只羊可能被两条轨迹（ID 切换）各计一次：同方向、时间差 < merge_window 且
   BEV 距离 < merge_radius 的事件合并为一条，只计一次并留下 merged 标记。
5. 短时间内反向穿越（出去了又回来）记为 reentry，原始计数保留，净存栏可另行配对。
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import SceneConfig
from .geometry import line_signed_distance


@dataclass
class CrossingEvent:
    t: float
    frame: int
    track_id: int
    direction: str          # "in"（归栏）/ "out"（出栏）
    cls_name: str
    conf: float
    be_xy: Tuple[float, float]
    speed_mps: float = 0.0
    reentry: bool = False
    merged: int = 0         # 被合并掉的重叠事件数
    illuminance: Optional[float] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["be_xy"] = [round(float(self.be_xy[0]), 3), round(float(self.be_xy[1]), 3)]
        d["t"] = round(float(self.t), 3)
        d["conf"] = round(float(self.conf), 4)
        d["speed_mps"] = round(float(self.speed_mps), 3)
        return d


@dataclass
class _GateState:
    """单条轨迹的过线状态。

    side      最近一次"稳定所在侧"（+1 / -1），只在确认或取消后更新
    pending_* 已离开迟滞带、但还没走够位移余量/确认帧数的候选过栏
    """

    side: int = 0                 # 上一次稳定所在侧：+1 / -1
    band_entry_side: int = 0      # 进带时所在侧
    band_since: Optional[float] = None
    band_entry_pt: Optional[np.ndarray] = None
    band_frames: int = 0
    last_s: float = 0.0
    pending_side: int = 0         # 候选过栏的出口侧
    pending_frames: int = 0       # 候选过栏在带内的帧数
    pending_t: Optional[float] = None
    crossed_at: Optional[float] = None
    crossed_dir: Optional[str] = None


class GateCounter:
    """双向过栏计数器。逐帧喂入已确认轨迹，吐出过栏事件。"""

    def __init__(self, cfg: SceneConfig, illuminance: Optional[float] = None):
        self.cfg = cfg
        self.illuminance = illuminance
        self.states: Dict[int, _GateState] = {}
        self.events: List[CrossingEvent] = []
        self.counts = {"in": 0, "out": 0}
        self._recent: List[CrossingEvent] = []
        self._last_seen: Dict[int, float] = {}

    # ---------------- 对外接口 ----------------
    def update(self, t: float, frame: int, tracks: Iterable) -> List[CrossingEvent]:
        """返回本帧新产生（未被合并）的事件。"""
        new_events: List[CrossingEvent] = []
        alive = set()
        for trk in tracks:
            if not getattr(trk, "confirmed", False):
                continue
            if getattr(trk, "cls_name", "") not in self.cfg.class_names:
                continue
            if getattr(trk, "box_h_px", 1e9) < self.cfg.min_box_h_px:
                continue
            alive.add(trk.id)
            ev = self._step_track(t, frame, trk)
            if ev is not None:
                new_events.append(ev)
        # 清理久未出现的轨迹状态，避免长视频里字典无限增长
        for tid in list(self.states):
            if tid not in alive and t - self._last_seen.get(tid, t) > 10 * self.cfg.max_age_s:
                self.states.pop(tid, None)
                self._last_seen.pop(tid, None)
        return new_events

    def set_illuminance(self, lux: Optional[float]) -> None:
        self.illuminance = lux

    # ---------------- 内部 ----------------
    def _confirm_frames(self) -> int:
        lux = self.illuminance
        if lux is None:
            return self.cfg.confirm_frames_day
        return self.cfg.confirm_frames_night if lux < 10.0 else self.cfg.confirm_frames_day

    def _direction_of(self, s: float) -> str:
        positive_is_in = self.cfg.in_side == "positive"
        toward_in = s > 0 if positive_is_in else s < 0
        return "in" if toward_in else "out"

    def _step_track(self, t: float, frame: int, trk) -> Optional[CrossingEvent]:
        cfg = self.cfg
        pt = np.asarray(trk.pos, dtype=np.float64)
        s = float(line_signed_distance(pt, cfg.line_p1, cfg.line_p2))
        st = self.states.setdefault(trk.id, _GateState())
        self._last_seen[trk.id] = t
        st.last_s = s

        in_band = abs(s) <= cfg.hysteresis_m
        clear_side = 1 if s > 0 else -1

        # ---- 在带内：累计帧数，并取消尚未确认的候选 ----
        if in_band:
            if st.band_since is None:
                st.band_since = t
                st.band_entry_side = st.side if st.side != 0 else clear_side
                st.band_entry_pt = pt.copy()
                st.band_frames = 0
            st.band_frames += 1
            st.pending_side = 0
            st.pending_frames = 0
            st.pending_t = None
            return None

        # ---- 刚离开带：登记候选过栏（只有"从另一侧出去"才算）----
        if st.band_since is not None:
            entry_side = st.band_entry_side
            st.band_since = None
            if st.side == 0:
                st.side = entry_side
            if clear_side != st.side:
                st.pending_side = clear_side
                st.pending_frames = st.band_frames
                st.pending_t = t
            st.band_frames = 0
            st.band_entry_pt = None

        # ---- 没有候选：只更新所在侧 ----
        if st.pending_side == 0:
            st.side = clear_side
            return None

        # ---- 候选期内又回到原侧：撤销 ----
        if clear_side != st.pending_side:
            st.pending_side = 0
            st.pending_frames = 0
            st.pending_t = None
            st.side = clear_side
            return None

        # ---- 位移余量 + 确认帧数都满足，才落成事件 ----
        if abs(s) < cfg.margin_m or st.pending_frames < self._confirm_frames():
            return None

        direction = self._direction_of(s)
        ev = CrossingEvent(
            t=t,
            frame=frame,
            track_id=trk.id,
            direction=direction,
            cls_name=getattr(trk, "cls_name", "goat"),
            conf=float(getattr(trk, "conf", 1.0)),
            be_xy=(float(pt[0]), float(pt[1])),
            speed_mps=float(np.linalg.norm(getattr(trk, "vel", (0.0, 0.0)))),
            illuminance=self.illuminance,
        )
        st.side = clear_side
        st.pending_side = 0
        st.pending_frames = 0
        st.pending_t = None

        merged_into = self._merge_lookup(ev)
        if merged_into is not None:
            merged_into.merged += 1
            st.crossed_at = t
            st.crossed_dir = direction
            return None

        # 往返判定：同一条轨迹短时间内的反向事件（注意要在覆盖 crossed_* 之前比）
        if st.crossed_at is not None and st.crossed_dir and st.crossed_dir != direction:
            if t - st.crossed_at <= cfg.reentry_window_s:
                ev.reentry = True
        st.crossed_at = t
        st.crossed_dir = direction

        self.events.append(ev)
        self._recent.append(ev)
        self.counts[direction] += 1
        return ev

    def _mark_crossing(self, st: _GateState, ev: CrossingEvent) -> None:
        st.crossed_at = ev.t
        st.crossed_dir = ev.direction

    def _merge_lookup(self, ev: CrossingEvent) -> Optional[CrossingEvent]:
        cfg = self.cfg
        for prev in reversed(self._recent):
            if prev.direction != ev.direction:
                continue
            if abs(prev.t - ev.t) > cfg.merge_window_s:
                break
            if prev.track_id == ev.track_id:
                continue
            d = math.dist(prev.be_xy, ev.be_xy)
            if d <= cfg.merge_radius_m:
                return prev
        return None


class ConservationLedger:
    """晨出暮归守恒：S_t = S_{t-1} - E_t + R_t，偏差超限即报警。"""

    def __init__(self, tol_abs: int = 5, tol_rel: float = 0.03):
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
        self.rows: List[Dict] = []

    def add_day(self, date: str, out_count: int, in_count: int, stock_open: int) -> Dict:
        stock_close = stock_open - out_count + in_count
        # 归栏应约等于出栏；按相对 + 绝对双阈值取大者
        diff = in_count - out_count
        tol = max(self.tol_abs, self.tol_rel * max(1, out_count))
        anomaly = abs(diff) > tol
        row = {
            "date": date,
            "out": out_count,
            "in": in_count,
            "stock_open": stock_open,
            "stock_close": stock_close,
            "diff_in_minus_out": diff,
            "anomaly": bool(anomaly),
            "reason": "" if not anomaly else ("归栏显著少于出栏：疑走失/漏检" if diff < 0 else "归栏显著多于出栏：疑重复计数或串栏"),
        }
        self.rows.append(row)
        return row

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, ensure_ascii=False, indent=2)


def save_events(events: Sequence[CrossingEvent], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")


def load_events(path: str) -> List[CrossingEvent]:
    out: List[CrossingEvent] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["be_xy"] = tuple(d.get("be_xy", (0.0, 0.0)))
            out.append(CrossingEvent(**d))
    return out


def counts_by_period(events: Sequence[CrossingEvent], period_s: float = 86400.0) -> List[Dict]:
    """按天（或任意周期）汇总 in/out，供守恒校验与报表使用。"""
    if not events:
        return []
    t0 = min(e.t for e in events)
    buckets: Dict[int, Dict[str, int]] = {}
    for e in events:
        idx = int((e.t - t0) // period_s)
        b = buckets.setdefault(idx, {"in": 0, "out": 0})
        b[e.direction] += 1
    rows = []
    for idx in sorted(buckets):
        rows.append({"period": idx, "start_s": t0 + idx * period_s, **buckets[idx]})
    return rows


def write_counts_csv(rows: Sequence[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
