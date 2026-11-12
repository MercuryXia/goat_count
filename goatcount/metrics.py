# -*- coding: utf-8 -*-
"""评价指标：计数准确率、方向判别、CLEAR-MOT（MOTA/IDSW）、IDF1 近似。

没有 scipy，关联一律用贪心 + 距离阈值；对本文这种"少量目标、强空间约束"的场景够用，
但要在论文里如实说明 IDF1 是近似值（不做全局最优指派）。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .counting import CrossingEvent


# --------------------------------------------------------------------------- #
# 计数指标
# --------------------------------------------------------------------------- #
def count_metrics(pairs: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """pairs = [(pred, gt), ...]，按天/场次给出。

    CA   = 1 - |pred-gt| / gt 的均值（gt>0）
    MAE  = mean(|pred-gt|)
    MAPE = mean(|pred-gt|/gt) * 100
    bias = mean(pred-gt)  —— 正数表示系统性多计
    """
    vals = [(float(p), float(g)) for p, g in pairs if g > 0]
    if not vals:
        return {"CA": float("nan"), "MAE": float("nan"), "MAPE": float("nan"), "bias": float("nan"), "n": 0}
    diffs = np.array([p - g for p, g in vals])
    rel = np.array([abs(p - g) / g for p, g in vals])
    return {
        "CA": float(np.mean(1.0 - np.minimum(rel, 1.0))),
        "MAE": float(np.mean(np.abs(diffs))),
        "MAPE": float(np.mean(rel) * 100.0),
        "bias": float(np.mean(diffs)),
        "n": len(vals),
    }


def direction_accuracy(pred: Sequence[CrossingEvent], gt: Sequence[CrossingEvent], tol_s: float = 2.0) -> Dict[str, float]:
    """按时间最近邻匹配事件，再看方向是否一致。"""
    if not gt:
        return {"matched": 0, "direction_acc": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    used = set()
    correct = 0
    matched = 0
    for p in pred:
        best_i, best_dt = -1, 1e9
        for i, g in enumerate(gt):
            if i in used:
                continue
            dt = abs(g.t - p.t)
            if dt < best_dt:
                best_dt, best_i = dt, i
        if best_i >= 0 and best_dt <= tol_s:
            used.add(best_i)
            matched += 1
            if gt[best_i].direction == p.direction:
                correct += 1
    precision = matched / max(1, len(pred))
    recall = matched / max(1, len(gt))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "matched": matched,
        "direction_acc": correct / matched if matched else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------------------------------------------------- #
# 跟踪指标
# --------------------------------------------------------------------------- #
def _greedy_match(
    pred: Dict[int, np.ndarray], gt: Dict[int, np.ndarray], thresh: float
) -> List[Tuple[int, int]]:
    """按距离升序贪心匹配 pred_id <-> gt_id（BEV 米）。"""
    cand = []
    for pid, p in pred.items():
        for gid, g in gt.items():
            d = float(np.linalg.norm(np.asarray(p) - np.asarray(g)))
            if d <= thresh:
                cand.append((d, pid, gid))
    cand.sort()
    used_p, used_g, pairs = set(), set(), []
    for _, pid, gid in cand:
        if pid in used_p or gid in used_g:
            continue
        used_p.add(pid)
        used_g.add(gid)
        pairs.append((pid, gid))
    return pairs


def mot_metrics(
    frames_pred: Sequence[Dict[int, np.ndarray]],
    frames_gt: Sequence[Dict[int, np.ndarray]],
    thresh: float = 1.0,
    idf1_approx: bool = True,
) -> Dict[str, float]:
    """CLEAR-MOT：MOTA = 1 - (FN+FP+IDSW)/GT；IDF1 用"ID 对共现"近似。"""
    assert len(frames_pred) == len(frames_gt)
    tp = fp = fn = idsw = 0
    gt_total = 0
    # id 对共现计数，用于近似 IDF1
    pair_hits: Dict[Tuple[int, int], int] = defaultdict(int)
    last_match: Dict[int, int] = {}   # gt_id -> pred_id

    for pred, gt in zip(frames_pred, frames_gt):
        gt_total += len(gt)
        pairs = _greedy_match(pred, gt, thresh)
        matched_gt = {g for _, g in pairs}
        tp += len(pairs)
        fp += len(pred) - len(pairs)
        fn += len(gt) - len(pairs)
        for pid, gid in pairs:
            pair_hits[(gid, pid)] += 1
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid

    mota = 1.0 - (fn + fp + idsw) / max(1, gt_total)
    motp = float("nan")
    res = {
        "MOTA": float(mota),
        "IDSW": int(idsw),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "GT": gt_total,
        "ID_precision": float(tp / max(1, tp + fp)),
        "ID_recall": float(tp / max(1, tp + fn)),
    }
    if idf1_approx and pair_hits:
        # 每个 gt 与 pred 各取最大共现 ID 作为主匹配
        best_for_gt: Dict[int, Tuple[int, int]] = {}
        best_for_pred: Dict[int, Tuple[int, int]] = {}
        for (gid, pid), n in pair_hits.items():
            if gid not in best_for_gt or n > best_for_gt[gid][0]:
                best_for_gt[gid] = (n, pid)
            if pid not in best_for_pred or n > best_for_pred[pid][0]:
                best_for_pred[pid] = (n, gid)
        idtp = sum(n for n, _ in best_for_gt.values())
        res["IDF1"] = float(2 * idtp / max(1, (tp + fn) + (tp + fp)))
        res["IDP"] = float(idtp / max(1, tp + fp))
        res["IDR"] = float(idtp / max(1, tp + fn))
    else:
        res["IDF1"] = float("nan")
    return res


def summarize(metrics: Dict[str, float]) -> str:
    items = []
    for k, v in metrics.items():
        if isinstance(v, float):
            if math.isnan(v):
                continue
            items.append(f"{k}={v:.4f}")
        else:
            items.append(f"{k}={v}")
    return "  ".join(items)
