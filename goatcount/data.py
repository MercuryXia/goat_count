# -*- coding: utf-8 -*-
"""第 3 章数据侧：OBB 标签读写、按天划分、低照度物理退化合成。

约定（与 ultralytics OBB 一致）：
* 标签行：``cls x1 y1 x2 y2 x3 y3 x4 y4``，坐标为归一化多边形（顺时针）。
* 目录：``images/{train,val,test}`` 与 ``labels/{train,val,test}``。
* 划分单位是"采集日"而不是帧——同一天的相邻帧高度相似，按帧划分会高估精度。
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # 只有做退化合成时才需要 cv2 之外的重活
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

CLS_NAMES = ["goat", "dog", "person"]


# --------------------------------------------------------------------------- #
# 几何：多边形 <-> (cx, cy, w, h, angle)
# --------------------------------------------------------------------------- #
def poly_to_xywhr(pts: np.ndarray) -> Tuple[float, float, float, float, float]:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts.astype(np.float32)) if cv2 is not None else _min_area_rect(pts)
    return float(cx), float(cy), float(w), float(h), float(math.radians(ang))


def _min_area_rect(pts: np.ndarray):
    """cv2 缺失时的最小外接矩形（用四边中最长边定向，够用且可解释）。"""
    pts = np.asarray(pts, dtype=np.float64)
    # 用 PCA 主轴定向
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c)
    axis = vt[0]
    ang = math.degrees(math.atan2(axis[1], axis[0]))
    proj = (pts - c) @ vt.T
    w = float(proj[:, 0].max() - proj[:, 0].min())
    h = float(proj[:, 1].max() - proj[:, 1].min())
    return (float(c[0]), float(c[1])), (w, h), ang


def xywhr_to_poly(cx: float, cy: float, w: float, h: float, ang: float) -> np.ndarray:
    ca, sa = math.cos(ang), math.sin(ang)
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
    x = cx + dx * ca - dy * sa
    y = cy + dx * sa + dy * ca
    return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------- #
# 标签读写
# --------------------------------------------------------------------------- #
@dataclass
class ObbLabel:
    cls: int
    poly: np.ndarray  # (4,2) 归一化

    def to_line(self) -> str:
        flat = " ".join(f"{v:.6f}" for v in self.poly.reshape(-1))
        return f"{self.cls} {flat}"


def read_labels(path: str) -> List[ObbLabel]:
    out: List[ObbLabel] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 9:
                continue
            cls = int(float(parts[0]))
            poly = np.array([float(x) for x in parts[1:]], dtype=np.float64).reshape(4, 2)
            out.append(ObbLabel(cls, poly))
    return out


def write_labels(path: str, labels: Sequence[ObbLabel]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for lb in labels:
            fh.write(lb.to_line() + "\n")


# --------------------------------------------------------------------------- #
# 按天划分
# --------------------------------------------------------------------------- #
_DATE_RE = re.compile(r"(20\d{6})")


def day_of(path: str) -> str:
    """从文件名或父目录名里取采集日（YYYYMMDD）；取不到就用父目录名。"""
    base = os.path.basename(path)
    m = _DATE_RE.search(base)
    if m:
        return m.group(1)
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    m = _DATE_RE.search(parent)
    return m.group(1) if m else parent


def split_by_day(
    images: Sequence[str], ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 0
) -> Dict[str, List[str]]:
    days = sorted({day_of(p) for p in images})
    rng = random.Random(seed)
    rng.shuffle(days)
    n = len(days)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(0, int(round(n * ratios[1])))
    train_days = set(days[:n_train])
    val_days = set(days[n_train:n_train + n_val])
    split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for p in images:
        d = day_of(p)
        key = "train" if d in train_days else ("val" if d in val_days else "test")
        split[key].append(p)
    # 只有一个采集日时，退化为按帧切分（并在日志里说明这是无奈之举）
    if len(days) == 1:
        imgs = sorted(images)
        k1 = max(1, int(len(imgs) * ratios[0]))
        k2 = max(k1 + 1, int(len(imgs) * (ratios[0] + ratios[1])))
        split = {"train": imgs[:k1], "val": imgs[k1:k2], "test": imgs[k2:]}
    return split


# --------------------------------------------------------------------------- #
# 低照度退化合成
# --------------------------------------------------------------------------- #
def degrade_low_light(
    img: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    gamma: float = 2.2,
    gain: float = 0.18,
    noise_sigma: float = 6.0,
    motion_px: int = 0,
    jpeg_q: int = 60,
    tint: Tuple[float, float, float] = (0.95, 1.0, 1.08),
) -> np.ndarray:
    """物理上"说得通"的夜间合成：先光照衰减（幂律），再读出噪声，再运动模糊与压缩。

    与 style-transfer 式增强的区别：这里的顺序与相机成像链一致，噪声随信号变化，
    因此增强网络学到的不是"把暗部拉亮"，而是"在低信噪比下仍然可分"。
    """
    rng = rng or np.random.default_rng()
    x = img.astype(np.float64) / 255.0
    x = np.power(np.clip(x, 0, 1), gamma) * gain
    # 信号相关噪声（光子散粒噪声近似 + 读出噪声）
    shot = rng.normal(0, 1, x.shape) * np.sqrt(np.maximum(x, 1e-6)) * (noise_sigma / 255.0) * 0.5
    read = rng.normal(0, noise_sigma / 255.0, x.shape)
    x = np.clip(x + shot + read, 0, 1)
    x = x * np.asarray(tint, dtype=np.float64).reshape(1, 1, 3)
    out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    if motion_px and motion_px > 1 and cv2 is not None:
        k = np.zeros((motion_px, motion_px), dtype=np.float32)
        k[motion_px // 2, :] = 1.0 / motion_px
        out = cv2.filter2D(out, -1, k)
    if jpeg_q and cv2 is not None:
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def synthesize_pairs(
    images: Sequence[str], out_dir: str, per_image: int = 1, seed: int = 0
) -> List[Tuple[str, str, float]]:
    """为"增强—检测"辅助任务生成 (夜间图, 白天图, 照度) 配对清单。"""
    if cv2 is None:
        raise RuntimeError("需要 opencv-python 才能合成")
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "night"), exist_ok=True)
    manifest: List[Tuple[str, str, float]] = []
    for src in images:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            continue
        for k in range(per_image):
            g = float(rng.uniform(1.8, 3.0))
            gain = float(rng.uniform(0.08, 0.28))
            night = degrade_low_light(
                img,
                rng,
                gamma=g,
                gain=gain,
                noise_sigma=float(rng.uniform(3.0, 10.0)),
                motion_px=int(rng.integers(0, 13)),
                jpeg_q=int(rng.integers(45, 80)),
            )
            name = f"{os.path.splitext(os.path.basename(src))[0]}_{k:02d}.jpg"
            dst = os.path.join(out_dir, "night", name)
            cv2.imwrite(dst, night)
            manifest.append((dst, src, gain))
    with open(os.path.join(out_dir, "pairs.jsonl"), "w", encoding="utf-8") as fh:
        for n, d, g in manifest:
            fh.write(json.dumps({"night": n, "day": d, "gain": g}, ensure_ascii=False) + "\n")
    return manifest


# --------------------------------------------------------------------------- #
# 数据集整理
# --------------------------------------------------------------------------- #
def prepare_dataset(
    src_root: str,
    out_root: str,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    names: Sequence[str] = CLS_NAMES,
) -> Dict[str, object]:
    """把"图片 + 同名 txt"的平铺目录整理成 ultralytics OBB 数据集结构，并按天划分。"""
    images = sorted(
        p for p in glob.glob(os.path.join(src_root, "**", "*.jpg"), recursive=True)
    ) + sorted(glob.glob(os.path.join(src_root, "**", "*.png"), recursive=True))
    if not images:
        raise FileNotFoundError(f"{src_root} 下没有 jpg/png")
    split = split_by_day(images, ratios, seed)
    stats = {"train": 0, "val": 0, "test": 0, "missing_labels": 0, "instances": 0}
    for key, paths in split.items():
        img_dir = os.path.join(out_root, "images", key)
        lbl_dir = os.path.join(out_root, "labels", key)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            lab_src = os.path.join(os.path.dirname(p), stem + ".txt")
            if os.path.exists(lab_src):
                labels = read_labels(lab_src)
            else:
                labels = []
                stats["missing_labels"] += 1
            write_labels(os.path.join(lbl_dir, stem + ".txt"), labels)
            stats["instances"] += len(labels)
            # 只建软链接/复制，避免翻倍占盘
            dst = os.path.join(img_dir, os.path.basename(p))
            if not os.path.exists(dst):
                try:
                    os.link(p, dst)
                except OSError:
                    import shutil

                    shutil.copy2(p, dst)
            stats[key] += 1
    data_yaml = os.path.join(out_root, "data.yaml")
    write_data_yaml(data_yaml, out_root, names)
    stats["data_yaml"] = data_yaml
    return stats


def write_data_yaml(path: str, root: str, names: Sequence[str] = CLS_NAMES) -> None:
    payload = {
        "path": os.path.abspath(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    _dump_yaml(path, payload)


def _dump_yaml(path: str, payload: Dict) -> None:
    try:
        from ultralytics.utils import YAML

        YAML.save(path, payload)
        return
    except Exception:
        pass
    # 极简 YAML 兜底（只覆盖本项目的扁平结构）
    lines = []
    for k, v in payload.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k}: {v}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def dataset_report(root: str) -> Dict[str, object]:
    """统计各类别实例数、空标签数、图片/标签配对情况。"""
    rep: Dict[str, object] = {"splits": {}, "class_counts": {n: 0 for n in CLS_NAMES}}
    for key in ("train", "val", "test"):
        img_dir = os.path.join(root, "images", key)
        lbl_dir = os.path.join(root, "labels", key)
        imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
        empty = 0
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            labels = read_labels(os.path.join(lbl_dir, stem + ".txt"))
            if not labels:
                empty += 1
            for lb in labels:
                nm = CLS_NAMES[lb.cls] if 0 <= lb.cls < len(CLS_NAMES) else str(lb.cls)
                rep["class_counts"][nm] = rep["class_counts"].get(nm, 0) + 1
        rep["splits"][key] = {"images": len(imgs), "empty_labels": empty}
    return rep
