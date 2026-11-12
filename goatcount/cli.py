# -*- coding: utf-8 -*-
"""命令行入口：``python -m goatcount <command>``。

commands
--------
demo    合成视频 -> 端到端计数（不需要任何权重，用来验证链路）
infer   真实视频 -> OBB 检测 -> 计数，输出事件表与标注视频
check   检查 OBB 数据集（类别分布、空标签、划分）
train   转给 train.py（三阶段训练）
selftest 跑单元测试
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from .config import SceneConfig, resolve
from .counting import CrossingEvent, save_events
from .metrics import count_metrics, direction_accuracy, summarize
from .pipeline import CountingPipeline

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


# --------------------------------------------------------------------------- #
def _open_writer(path: Optional[str], frame, fps: float):
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h, w = frame.shape[:2]
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def cmd_demo(args) -> int:
    from .demo import make_scene
    from .infer import ColorGateDetector
    from .viz import bev_canvas, draw_frame

    cfg = SceneConfig.load(args.config)
    out_dir = resolve(args.out)
    os.makedirs(out_dir, exist_ok=True)
    video = os.path.join(out_dir, "demo.mp4")
    print(f"[demo] 合成场景 -> {video}")
    scene = make_scene(
        cfg,
        out_dir,
        n_out=args.out_goats,
        n_in=args.in_goats,
        fps=cfg.fps,
        duration=args.duration,
        dogs=args.dogs,
        persons=args.persons,
        night_from=(args.duration * 0.55 if args.night else 1e9),
        video=video,
        seed=args.seed,
    )

    detector = ColorGateDetector()
    pipe = CountingPipeline(cfg, cfg.fps)
    cap = cv2.VideoCapture(video)
    writer = _open_writer(os.path.join(out_dir, "demo_annotated.mp4") if not args.no_video else None, np.zeros((540, 960, 3), np.uint8), cfg.fps)
    idx = 0
    last_tracks = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / cfg.fps
        fo = detector(frame, t)
        last_tracks = pipe.step(idx, t, fo)
        if writer is not None:
            writer.write(draw_frame(frame, cfg, [x for x in last_tracks if x.confirmed], pipe.counter.events))
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()

    result = pipe.finish()
    save_events(result.events, os.path.join(out_dir, "events.jsonl"))
    cv2.imwrite(os.path.join(out_dir, "bev.png"), bev_canvas(cfg, [x for x in last_tracks if x.confirmed]))

    gt = [CrossingEvent(t=e["t"], frame=0, track_id=-1, direction=e["direction"], cls_name="goat", conf=1.0, be_xy=(0.0, e["y"])) for e in scene.gt_events]
    c_metrics = count_metrics([(result.counts["out"], scene.gt_out), (result.counts["in"], scene.gt_in)])
    d_metrics = direction_accuracy(result.events, gt, tol_s=2.5)
    summary = {
        "pred": result.counts,
        "gt": {"out": scene.gt_out, "in": scene.gt_in},
        "frames": result.frames,
        "tracks": result.tracks_total,
        "filter": result.filter_stats,
        "count": c_metrics,
        "direction": d_metrics,
        "events": [e.to_dict() for e in result.events],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[demo] pred={result.counts} gt={{'out': {scene.gt_out}, 'in': {scene.gt_in}}}")
    print(f"[demo] count  {summarize(c_metrics)}")
    print(f"[demo] dir    {summarize(d_metrics)}")
    print(f"[demo] 输出目录 {out_dir}")
    return 0


def cmd_infer(args) -> int:
    from .infer import UltralyticsObbDetector
    from .viz import draw_frame

    if cv2 is None:
        print("需要 opencv-python", file=sys.stderr)
        return 2
    cfg = SceneConfig.load(args.config)
    out_dir = resolve(args.out)
    os.makedirs(out_dir, exist_ok=True)
    det = UltralyticsObbDetector(args.weights, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"打不开视频：{args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
    pipe = CountingPipeline(cfg, fps)
    writer = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        fo = det(frame, t)
        tracks = pipe.step(idx, t, fo)
        if args.save_video:
            if writer is None:
                writer = _open_writer(os.path.join(out_dir, "annotated.mp4"), frame, fps)
            writer.write(draw_frame(frame, cfg, [x for x in tracks if x.confirmed], pipe.counter.events))
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
    result = pipe.finish()
    save_events(result.events, os.path.join(out_dir, "events.jsonl"))
    payload = {
        "counts": result.counts,
        "events": len(result.events),
        "frames": result.frames,
        "filter": result.filter_stats,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_check(args) -> int:
    from .data import dataset_report

    root = resolve(args.data)
    rep = dataset_report(root)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_train(args) -> int:
    from .train import main as train_main

    return train_main(args.rest)


def cmd_selftest(args) -> int:
    import unittest

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    suite = unittest.defaultTestLoader.discover(os.path.join(here, "tests"), top_level_dir=here)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return 0 if res.wasSuccessful() else 1


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="goatcount", description="桥头石山羊进出栏视觉计数（论文配套工程）")
    p.add_argument("--config", default=None, help="场景配置，默认 configs/scene_gate.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="合成视频端到端演示（无需权重）")
    d.add_argument("--out", default="out/demo")
    d.add_argument("--out-goats", type=int, default=12)
    d.add_argument("--in-goats", type=int, default=10)
    d.add_argument("--dogs", type=int, default=2)
    d.add_argument("--persons", type=int, default=2)
    d.add_argument("--duration", type=float, default=0.0, help="秒；0 表示按羊只数量自动估算")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--night", action="store_true", help="后半段叠加低照度退化")
    d.add_argument("--no-video", action="store_true", help="不写标注视频（更快）")
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("infer", help="真实视频推理")
    i.add_argument("--video", required=True)
    i.add_argument("--weights", required=True)
    i.add_argument("--out", default="out/infer")
    i.add_argument("--imgsz", type=int, default=640)
    i.add_argument("--conf", type=float, default=0.25)
    i.add_argument("--iou", type=float, default=0.5)
    i.add_argument("--device", default=None)
    i.add_argument("--save-video", action="store_true")
    i.set_defaults(func=cmd_infer)

    c = sub.add_parser("check", help="检查 OBB 数据集")
    c.add_argument("--data", default="data/goat_gate")
    c.set_defaults(func=cmd_check)

    t = sub.add_parser("train", help="训练（参数转交 train.py）")
    t.set_defaults(func=cmd_train, rest=[])

    s = sub.add_parser("selftest", help="跑单元测试")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    # train 子命令把参数整体转交给 train.py，因此这里用 parse_known_args
    args, extra = p.parse_known_args(argv)
    if extra:
        if args.cmd != "train":
            p.error("unrecognized arguments: " + " ".join(extra))
        args.rest = list(getattr(args, "rest", []) or []) + extra
    if getattr(args, "config", None):
        os.environ["GOATCOUNT_CONFIG"] = args.config
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
