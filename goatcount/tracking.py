# -*- coding: utf-8 -*-
"""第 4 章：自由流多目标跟踪 + 轨迹级干扰物过滤（步态频谱）。

* LiteTracker：ByteTrack 的两阶段关联思想 + 常速预测 + 丢失后按位置/类别再关联。
  在 BEV（米）里跟踪，尺度一致，门限不随远近变化。
* gait_features：由轨迹的图像框宽/高/质心速度时序做频谱分析，取主频、谐波比、
  自相关峰等，作为"羊 vs 狗"的软证据（狗步频更高、背线起伏更规律）。
* TrackFilter：轨迹投票 + 步态 + 业务规则三级过滤；因为狗不随群出栏，过滤的作用
  主要是抑制"门口徘徊误触发"，而不是细粒度物种识别。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import SceneConfig


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """一帧中的一个检测（已投影到 BEV）。"""

    pos: np.ndarray            # (2,) BEV 米，着地点
    size: np.ndarray           # (2,) BEV 足迹 (长, 宽) 米，粗略估计
    cls: int
    conf: float
    corners: np.ndarray        # (4,2) 图像中心点，用于可视化
    box_h_px: float = 0.0
    w_img: float = 0.0
    h_img: float = 0.0

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(2)
        self.size = np.asarray(self.size, dtype=np.float64).reshape(2)
        self.corners = np.asarray(self.corners, dtype=np.float64).reshape(4, 2)


@dataclass
class Track:
    """一条轨迹。history 存 (t, w_img, h_img, speed_mps) 供步态分析。"""

    id: int
    pos: np.ndarray
    vel: np.ndarray
    size: np.ndarray
    cls: int
    cls_name: str
    conf: float
    hits: int = 1
    misses: int = 0
    first_t: float = 0.0
    last_t: float = 0.0
    confirmed: bool = False
    box_h_px: float = 0.0
    corners: Optional[np.ndarray] = None
    class_votes: Dict[int, int] = field(default_factory=dict)
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    gait: Dict[str, float] = field(default_factory=dict)
    tag: str = ""

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def vote_ratio(self, cls: int) -> float:
        total = sum(self.class_votes.values())
        if total <= 0:
            return 0.0
        return self.class_votes.get(cls, 0) / total


# --------------------------------------------------------------------------- #
# 跟踪器
# --------------------------------------------------------------------------- #
class LiteTracker:
    """轻量在线跟踪器（不依赖 scipy：关联用贪心，代价可解释）。"""

    def __init__(self, cfg: SceneConfig, fps: float = 25.0):
        self.cfg = cfg
        self.fps = float(fps)
        self.tracks: List[Track] = []
        self._next_id = 1
        self._lost: List[Tuple[Track, float]] = []   # (track, t_lost)
        self.reid_window_s = max(3.0, cfg.max_age_s)
        self.reid_radius_m = cfg.gate_m

    # ---------- public ----------
    def update(self, t: float, frame: int, obs: Sequence[Observation]) -> List[Track]:
        obs = [o for o in obs if o.conf >= self.cfg.low_conf]
        for trk in self.tracks:
            trk.pos = trk.pos + trk.vel * (1.0 / self.fps)

        high = [o for o in obs if o.conf >= self.cfg.high_conf]
        low = [o for o in obs if o.conf < self.cfg.high_conf]

        pairs1, free_tracks, free_dets = self._associate(self.tracks, high, t)
        for ti, di in pairs1:
            self._update_track(self.tracks[ti], high[di], t)
        remaining_trks = [self.tracks[i] for i in free_tracks]

        if low:
            pairs2, still_free_tracks, free_low = self._associate(remaining_trks, low, t)
            for ti, di in pairs2:
                self._update_track(remaining_trks[ti], low[di], t)
                if remaining_trks[ti] in self.tracks:
                    pass
            for trk in remaining_trks:
                if trk.last_t != t:
                    trk.misses += 1
                    trk.confirmed = trk.confirmed and trk.misses <= 1
        else:
            for trk in remaining_trks:
                trk.misses += 1
                trk.confirmed = trk.confirmed and trk.misses <= 1

        # 未被任何阶段匹配的高分检测 -> 新建 / 再关联
        for di in (free_dets if not low else self._unmatched_high(high, pairs1)):
            o = high[di]
            if o.conf < self.cfg.high_conf:
                continue
            revived = self._try_reid(o, t)
            if revived is not None:
                self._update_track(revived, o, t)
            else:
                self._spawn(o, t)

        # 老化删除
        alive: List[Track] = []
        for trk in self.tracks:
            if trk.misses * (1.0 / self.fps) > self.cfg.max_age_s:
                self._lost.append((trk, t))
                continue
            alive.append(trk)
        self.tracks = alive

        # 再关联缓存老化
        self._lost = [(trk, tl) for trk, tl in self._lost if t - tl <= self.reid_window_s]
        return self.tracks

    @property
    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]

    # ---------- internal ----------
    def _unmatched_high(self, high: Sequence[Observation], pairs: Sequence[Tuple[int, int]]) -> List[int]:
        used = {di for _, di in pairs}
        return [i for i in range(len(high)) if i not in used]

    def _cost(self, trk: Track, o: Observation) -> float:
        d = float(np.linalg.norm(trk.pos - o.pos))
        class_pen = 0.0 if trk.cls == o.cls else 0.6
        return d + class_pen

    def _associate(
        self, tracks: Sequence[Track], dets: Sequence[Observation], t: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """贪心关联：按代价升序，一条轨迹/一个检测只用一次。"""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cand: List[Tuple[float, int, int]] = []
        for ti, trk in enumerate(tracks):
            for di, o in enumerate(dets):
                if trk.cls != o.cls and trk.vote_ratio(trk.cls) > 0.8:
                    # 类别很稳定时，不允许跨类关联（防止羊狗 ID 互换）
                    continue
                cost = self._cost(trk, o)
                if cost <= self.cfg.gate_m:
                    cand.append((cost, ti, di))
        cand.sort()
        used_t, used_d = set(), set()
        pairs: List[Tuple[int, int]] = []
        for _, ti, di in cand:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            pairs.append((ti, di))
        return (
            pairs,
            [i for i in range(len(tracks)) if i not in used_t],
            [i for i in range(len(dets)) if i not in used_d],
        )

    def _update_track(self, trk: Track, o: Observation, t: float) -> None:
        # dt 必须夹住：首帧 t=0 或时间戳抖动都会让速度估计爆炸
        dt = t - trk.last_t if trk.last_t is not None else 1.0 / self.fps
        dt = float(min(max(dt, 0.5 / self.fps), 1.0))
        new_vel = (o.pos - trk.pos) / dt
        speed = float(np.linalg.norm(new_vel))
        if speed > 6.0:  # 羊跑不出 6 m/s，超出必是关联错误或标注错误
            new_vel = new_vel * (6.0 / speed)
        trk.vel = 0.6 * trk.vel + 0.4 * new_vel if trk.hits > 1 else new_vel
        trk.pos = o.pos
        trk.size = 0.7 * trk.size + 0.3 * o.size
        trk.conf = 0.7 * trk.conf + 0.3 * o.conf
        trk.box_h_px = o.box_h_px
        trk.corners = o.corners
        trk.class_votes[o.cls] = trk.class_votes.get(o.cls, 0) + 1
        trk.cls = max(trk.class_votes, key=trk.class_votes.get)
        trk.cls_name = self._name(trk.cls)
        trk.hits += 1
        trk.misses = 0
        trk.last_t = t
        if trk.hits >= self.cfg.n_init:
            trk.confirmed = True
        trk.history.append((t, o.w_img, o.h_img, float(np.linalg.norm(trk.vel))))
        if len(trk.history) > 400:
            trk.history = trk.history[-400:]

    def _spawn(self, o: Observation, t: float) -> Track:
        trk = Track(
            id=self._next_id,
            pos=o.pos.copy(),
            vel=np.zeros(2),
            size=o.size.copy(),
            cls=o.cls,
            cls_name=self._name(o.cls),
            conf=o.conf,
            hits=1,
            first_t=t,
            last_t=t,
            box_h_px=o.box_h_px,
            corners=o.corners,
            class_votes={o.cls: 1},
            history=[(t, o.w_img, o.h_img, 0.0)],
        )
        self._next_id += 1
        self.tracks.append(trk)
        return trk

    def _try_reid(self, o: Observation, t: float) -> Optional[Track]:
        """新检测落在刚丢失轨迹附近且类别一致 -> 复用旧 ID（缓解 ID 切换重复计数）。"""
        best: Optional[Track] = None
        best_d = self.reid_radius_m
        for trk, tl in self._lost:
            if trk.cls != o.cls:
                continue
            d = float(np.linalg.norm(trk.pos - o.pos))
            # 允许目标在丢失期间继续运动
            travel = trk.speed * max(0.0, t - tl)
            if d <= best_d + travel:
                best_d = d
                best = trk
        if best is None:
            return None
        self._lost = [(trk, tl) for trk, tl in self._lost if trk is not best]
        best.misses = 0
        best.last_t = t
        best.confirmed = True
        if best not in self.tracks:
            self.tracks.append(best)
        return best

    def _name(self, cls: int) -> str:
        names = self.cfg.class_names
        return names[cls] if 0 <= cls < len(names) else f"cls{cls}"


# --------------------------------------------------------------------------- #
# 步态特征
# --------------------------------------------------------------------------- #
def _resample(times: np.ndarray, values: np.ndarray, fps: float) -> np.ndarray:
    if len(times) < 2:
        return np.asarray(values, dtype=np.float64)
    t0, t1 = float(times[0]), float(times[-1])
    if t1 <= t0:
        return np.asarray(values, dtype=np.float64)
    n = max(8, int(round((t1 - t0) * fps)))
    grid = np.linspace(t0, t1, n)
    return np.interp(grid, times, values)


def gait_features(
    history: Sequence[Tuple[float, float, float, float]],
    fps: float = 25.0,
    min_len: int = 12,
) -> Dict[str, float]:
    """由 (t, w_img, h_img, speed) 序列提取步态频谱特征。

    返回：主频 dominant_hz、谱峰占比 peak_ratio、谐波比 harmonic_ratio、
    自相关峰 acf_peak、节律熵 rhythm_entropy、平均速度、体长宽比、样本时长。
    """
    if len(history) < min_len:
        return {}
    arr = np.asarray(history, dtype=np.float64)
    times, w, h, speed = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    dur = float(times[-1] - times[0])
    if dur < 1.0:
        return {}

    fs = 25.0
    w_r = _resample(times, w, fs)
    h_r = _resample(times, h, fs)
    s_r = _resample(times, speed, fs)

    def _detrend(x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n < 3:
            return x - x.mean()
        k = np.arange(n)
        A = np.stack([k, np.ones(n)], axis=1)
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        return x - A @ coef

    def _spectrum(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = _detrend(x) * np.hanning(len(x))
        spec = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)
        return freqs, spec

    freqs, spec_h = _spectrum(h_r)
    _, spec_w = _spectrum(w_r)
    # 只保留 0.5–6 Hz（步频合理区间），排除直流与高频噪声
    band = (freqs >= 0.5) & (freqs <= 6.0)
    feats: Dict[str, float] = {"duration_s": dur}
    if band.sum() < 3:
        return feats
    fb, sb = freqs[band], spec_h[band]
    k = int(np.argmax(sb))
    dominant = float(fb[k])
    total = float(np.sum(sb)) + 1e-9
    peak_ratio = float(sb[k] / total)

    # 谐波比：二倍频能量 / 基频能量（狗步态更"尖"，谐波更明显）
    def _at(f: float) -> float:
        idx = int(np.argmin(np.abs(freqs - f)))
        return float(spec_h[idx])

    harmonic_ratio = float(_at(2 * dominant) / (_at(dominant) + 1e-9))
    # 自相关峰：节律规整度
    x = _detrend(h_r)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo = int(fs / 6.0)
    hi = min(len(ac) - 1, int(fs / 0.5))
    acf_peak = float(np.max(ac[lo:hi])) if hi > lo else 0.0
    # 节律熵：谱归一化后的熵，越接近均匀越"乱"
    p = sb / total
    rhythm_entropy = float(-np.sum(p * np.log(p + 1e-12)) / math.log(len(p)))

    feats.update(
        {
            "dominant_hz": dominant,
            "peak_ratio": peak_ratio,
            "harmonic_ratio": harmonic_ratio,
            "acf_peak": acf_peak,
            "rhythm_entropy": rhythm_entropy,
            "mean_speed_mps": float(np.mean(s_r)),
            "aspect_mean": float(np.mean(w_r / np.maximum(h_r, 1e-6))),
            "aspect_std": float(np.std(w_r / np.maximum(h_r, 1e-6))),
        }
    )
    return feats


def gait_says_dog(feats: Dict[str, float], cfg: SceneConfig) -> Optional[bool]:
    """规则化判据：主频高 + 节律规整 + 体尺偏小 -> 倾向狗。返回 None 表示无意见。"""
    if not feats or "dominant_hz" not in feats:
        return None
    dog_hz = float(cfg.extra.get("filter", {}).get("gait", {}).get("dog_freq_hz", 2.6))
    goat_hz = float(cfg.extra.get("filter", {}).get("gait", {}).get("goat_freq_hz", 1.6))
    dom = feats["dominant_hz"]
    if dom >= dog_hz and feats.get("acf_peak", 0.0) >= 0.35:
        return True
    if dom <= goat_hz:
        return False
    return None


# --------------------------------------------------------------------------- #
# 轨迹级过滤
# --------------------------------------------------------------------------- #
class TrackFilter:
    """轨道级判别：类别投票为主，步态为辅，规则兜底（宁漏勿错与宁错勿漏可配）。"""

    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.stats = {"goat": 0, "dog": 0, "person": 0, "ambiguous": 0}

    def decide(self, trk: Track) -> Tuple[bool, str]:
        cfg = self.cfg
        votes = trk.class_votes
        total = sum(votes.values()) or 1
        goat_vote = votes.get(cfg.goat_class, 0) / total
        name = trk.cls_name

        if name == "goat" and goat_vote >= cfg.goat_vote:
            self.stats["goat"] += 1
            return True, f"vote={goat_vote:.2f}"
        if name == "person":
            self.stats["person"] += 1
            return False, "class=person"

        # 剩下的主要是 goat/dog 混淆
        if cfg.use_gait and trk.history:
            feats = gait_features(trk.history, fps=cfg.fps)
            trk.gait = feats
            if feats:
                dog = gait_says_dog(feats, cfg)
                if dog is True and name != "goat":
                    self.stats["dog"] += 1
                    return False, f"gait_dog(f={feats['dominant_hz']:.2f}Hz)"
                if dog is True and goat_vote < 0.5:
                    self.stats["dog"] += 1
                    return False, "gait_dog+weak_vote"

        if name == "dog":
            self.stats["dog"] += 1
            return False, "class=dog"
        # 类别含糊但也不是明确干扰物 -> 按羊处理（保召回），并打标签便于复核
        self.stats["ambiguous"] += 1
        trk.tag = "ambiguous"
        return True, f"ambiguous(goat_vote={goat_vote:.2f})"
