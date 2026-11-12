# -*- coding: utf-8 -*-
"""生成一个极小的合成 OBB 数据集，只为把训练链路跑通（验证数据/标签/增强/损失）。

真实训练请用 ``goatcount.data.prepare_dataset`` 整理实拍数据：按天划分、OBB 多边形标签。

用法::

    python scripts/make_synthetic_dataset.py --out data/synthetic --frames 120
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goatcount.config import SceneConfig, resolve  # noqa: E402
from goatcount.demo import _background, _draw_actor, _px_per_meter, _to_image  # noqa: E402

NAMES = ["goat", "dog", "person"]
BODY = {"goat": (0.70, 0.62), "dog": (0.55, 0.40), "person": (0.45, 1.70)}


def _poly(cx: float, cy: float, w: float, h: float, W: int, H: int) -> str:
    x1, x2 = (cx - w / 2) / W, (cx + w / 2) / W
    y1, y2 = (cy - h / 2) / H, (cy + h / 2) / H
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return " ".join(f"{v:.6f}" for p in pts for v in p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--val-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = SceneConfig.load(resolve("configs/scene_gate.yaml"))
    W, H = (int(v) for v in args.size.split("x"))
    rng = random.Random(args.seed)
    root = resolve(args.out)

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    n_val = max(1, int(args.frames * args.val_ratio))
    stats = {k: 0 for k in NAMES}
    for fi in range(args.frames):
        split = "val" if fi < n_val else "train"
        img = _background((W, H), cfg)
        lines = []
        for _ in range(rng.randint(1, 6)):
            kind = rng.choices(NAMES, weights=[0.72, 0.16, 0.12])[0]
            bev = np.array([rng.uniform(-4.3, 4.3), rng.uniform(-2.4, 2.4)])
            q = _to_image(cfg, bev)
            if not (0 <= q[0] < W and 0 <= q[1] < H):
                continue
            s = _px_per_meter(cfg, bev)
            body_len, body_h = BODY[kind]
            w_px = max(10.0, body_len * s * 1.35)
            h_px = max(10.0, body_h * s)
            bob = 0.0
            _draw_actor(img, q, w_px, h_px, kind, bob)
            cy = q[1] - h_px * 0.5
            lines.append(f"{NAMES.index(kind)} " + _poly(q[0], cy, w_px, h_px, W, H))
            stats[kind] += 1

        name = f"syn_{fi:04d}"
        cv2.imwrite(os.path.join(root, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(root, "labels", split, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))

    # 训练用的 data.yaml（ultralytics 需要一个 yaml，键名固定）
    yaml_text = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(yaml_text)

    print(f"[synth] {args.frames} 帧 -> {root}")
    print("[synth] 目标数", stats)
    print(yaml_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
