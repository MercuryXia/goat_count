# -*- coding: utf-8 -*-
"""配置读取：YAML -> 带默认值的 dataclass。

本机环境没有 PyYAML，但 ultralytics 8.4 自带一个最小 YAML 读写器
（ultralytics.utils.YAML），这里直接复用它，避免额外依赖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML；优先用 ultralytics 自带的解析器。"""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    try:
        from ultralytics.utils import YAML  # type: ignore

        data = YAML.load(path)
    except Exception:  # pragma: no cover - 仅在没有 ultralytics 时走到
        import json

        if not path.endswith((".json", ".JSON")):
            raise RuntimeError(
                "当前环境既没有 PyYAML 也没有 ultralytics，无法解析 YAML。"
                "可把配置改存为同名 .json，或 pip install pyyaml。"
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    return data or {}


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    try:
        from ultralytics.utils import YAML  # type: ignore

        YAML.save(path, data)
    except Exception:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """支持 "a.b.c" 的点路径取值。"""
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class SceneConfig:
    """场景与判决参数。所有长度单位是米（BEV），时间单位是秒。"""

    fps: float = 25.0
    # --- 单应矫正：图像 4 点 -> BEV 4 点（米） ---
    image_points: List[List[float]] = field(
        default_factory=lambda: [[0, 0], [1, 0], [1, 1], [0, 1]]
    )
    bev_points: List[List[float]] = field(
        default_factory=lambda: [[0, 0], [4, 0], [4, 4], [0, 4]]
    )
    # --- 计数线（BEV） ---
    line_p1: Tuple[float, float] = (2.0, 0.0)
    line_p2: Tuple[float, float] = (2.0, 4.0)
    in_side: str = "positive"  # 哪一侧算"归栏"

    # --- 判决参数 ---
    hysteresis_m: float = 0.30          # 迟滞带半宽
    body_length_m: float = 0.70         # 石山羊体长先验
    margin_body: float = 0.5            # 过线后位移余量（个体长）
    confirm_frames_day: int = 5         # 出栏（白天）确认帧数
    confirm_frames_night: int = 8       # 归栏（低照度）确认帧数
    reentry_window_s: float = 6.0       # 往返判定窗口
    merge_window_s: float = 1.5         # 同一只羊被两条轨迹重复计数的合并窗口
    merge_radius_m: float = 1.2
    min_box_h_px: float = 12.0          # 太小的框不参与计数

    # --- 跟踪 ---
    gate_m: float = 1.6
    max_age_s: float = 2.5
    n_init: int = 3
    high_conf: float = 0.5
    low_conf: float = 0.10

    # --- 类别 ---
    class_names: List[str] = field(default_factory=lambda: ["goat", "dog", "person"])
    goat_class: int = 0
    vote_window: int = 15
    goat_vote: float = 0.70
    use_gait: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneConfig":
        c = cls()
        c.fps = float(_get(d, "fps", c.fps))
        homo = _get(d, "homography", {}) or {}
        c.image_points = homo.get("image_points", c.image_points)
        c.bev_points = homo.get("bev_points", c.bev_points)
        cnt = _get(d, "counting", {}) or {}
        line = cnt.get("line")
        if line and len(line) == 2:
            c.line_p1 = tuple(float(v) for v in line[0])
            c.line_p2 = tuple(float(v) for v in line[1])
        c.in_side = cnt.get("in_side", c.in_side)
        c.hysteresis_m = float(cnt.get("hysteresis_m", c.hysteresis_m))
        c.body_length_m = float(cnt.get("body_length_m", c.body_length_m))
        c.margin_body = float(cnt.get("margin_body", c.margin_body))
        cf = cnt.get("confirm_frames", {}) or {}
        c.confirm_frames_day = int(cf.get("day", c.confirm_frames_day))
        c.confirm_frames_night = int(cf.get("night", c.confirm_frames_night))
        c.reentry_window_s = float(cnt.get("reentry_window_s", c.reentry_window_s))
        c.merge_window_s = float(cnt.get("merge_window_s", c.merge_window_s))
        c.merge_radius_m = float(cnt.get("merge_radius_m", c.merge_radius_m))
        c.min_box_h_px = float(cnt.get("min_box_h_px", c.min_box_h_px))
        trk = _get(d, "tracking", {}) or {}
        c.gate_m = float(trk.get("gate_m", c.gate_m))
        c.max_age_s = float(trk.get("max_age_s", c.max_age_s))
        c.n_init = int(trk.get("n_init", c.n_init))
        c.high_conf = float(trk.get("high_conf", c.high_conf))
        c.low_conf = float(trk.get("low_conf", c.low_conf))
        flt = _get(d, "filter", {}) or {}
        c.class_names = list(flt.get("classes", c.class_names))
        c.goat_class = int(flt.get("goat_class", c.goat_class))
        c.vote_window = int(flt.get("vote_window", c.vote_window))
        c.goat_vote = float(flt.get("goat_vote", c.goat_vote))
        c.use_gait = bool(_get(flt, "gait.enabled", c.use_gait))
        c.extra = d
        return c

    @classmethod
    def load(cls, path: Optional[str] = None) -> "SceneConfig":
        path = path or os.path.join(CONFIG_DIR, "scene_gate.yaml")
        return cls.from_dict(load_yaml(path))

    # --- 便捷派生属性 ---
    @property
    def H(self) -> "object":
        """图像 -> BEV 的单应矩阵（按 image_points / bev_points 现算，带缓存）。"""
        from .geometry import homography_from_points

        key = (tuple(map(tuple, self.image_points)), tuple(map(tuple, self.bev_points)))
        cached = _H_CACHE.get(key)
        if cached is None:
            cached = homography_from_points(self.image_points, self.bev_points)
            _H_CACHE[key] = cached
        return cached

    @property
    def confirm_frames(self) -> int:
        return self.confirm_frames_day

    @property
    def margin_m(self) -> float:
        return self.margin_body * self.body_length_m


_H_CACHE: Dict[Any, Any] = {}

DEFAULT_SCENE = SceneConfig()


def resolve(path: str) -> str:
    """把相对路径解析到项目根目录下。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)
