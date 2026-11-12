# -*- coding: utf-8 -*-
"""第 3 章训练入口：三阶段。

stage=enh   ：用"夜间-白天"配对训练照度估计 + Retinex 重建（辅助任务，导出时裁掉）
stage=det   ：YOLO11-OBB 主训练（--film 时接上 FiLM 调制，端到端）
stage=all   ：先 enh 再 det

所有配置走 ultralytics 自己的 YAML 读写（环境里没有 PyYAML）。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import torch

from .models import attach_joint_enhancement, build_obb_model


# --------------------------------------------------------------------------- #
# stage=enh：自监督低光恢复
# --------------------------------------------------------------------------- #
def train_enhancer(
    pairs_jsonl: str,
    epochs: int = 5,
    imgsz: int = 320,
    batch: int = 8,
    lr: float = 1e-3,
    device: str = "cuda",
    out: str = "runs/enh/best.pt",
) -> str:
    """极简 Retinex-ish 训练：估计照度图 L，重建 R=I/L，约束 R 接近白天图。

    只训练 IllumEstimator + 一个小解码器；训练完把权重交给 det 阶段的 --enh-weights
    （det 阶段只用照度估计器，解码器丢弃）。
    """
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader, Dataset

    from .models import IllumEstimator

    class PairDS(Dataset):
        def __init__(self, items, size):
            self.items = items
            self.size = size

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            night = cv2.imread(it["night"], cv2.IMREAD_COLOR)
            day = cv2.imread(it["day"], cv2.IMREAD_COLOR)
            night = cv2.resize(night, (self.size, self.size))
            day = cv2.resize(day, (self.size, self.size))
            n = torch.from_numpy(cv2.cvtColor(night, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            d = torch.from_numpy(cv2.cvtColor(day, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            return n, d

    items = [json.loads(l) for l in open(pairs_jsonl, "r", encoding="utf-8") if l.strip()]
    if not items:
        raise ValueError("pairs.jsonl 为空")
    ds = PairDS(items, imgsz)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    est = IllumEstimator().to(dev)
    dec = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.SiLU(), torch.nn.Conv2d(16, 3, 1)
    ).to(dev)
    opt = torch.optim.Adam(list(est.parameters()) + list(dec.parameters()), lr=lr)

    for ep in range(epochs):
        ep_loss = 0.0
        for night, day in dl:
            night, day = night.to(dev), day.to(dev)
            L = est(night)
            L_up = torch.nn.functional.interpolate(L, size=night.shape[-2:], mode="bilinear", align_corners=False)
            R = night / (L_up + 1e-3)
            enhanced = torch.clamp(dec(torch.cat([R, L_up], dim=1)), 0, 1)
            # 平滑 + 灰度世界 + 重建
            tv = (L[:, :, 1:] - L[:, :, :-1]).abs().mean() + (L[:, :, :, 1:] - L[:, :, :, :-1]).abs().mean()
            gw = (L_up.mean(dim=(2, 3)) - 0.35).pow(2).mean()
            rec = torch.nn.functional.l1_loss(enhanced, day)
            loss = rec + 0.1 * tv + 0.5 * gw
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
        print(f"[enh] epoch {ep+1}/{epochs}  loss={ep_loss/max(1,len(dl)):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"estimator": est.state_dict(), "decoder": dec.state_dict()}, out)
    print(f"[enh] saved {out}")
    return out


# --------------------------------------------------------------------------- #
# stage=det：YOLO11-OBB
# --------------------------------------------------------------------------- #
class _SerialPool:
    """受限环境下的 ThreadPool 替身。

    ultralytics 在缓存标签时会用 ``multiprocessing.pool.ThreadPool``，它的
    SimpleQueue 需要创建命名管道；在沙箱/受限账户下会抛
    ``PermissionError: [WinError 5]``。这里换成串行 map，语义等价、只是慢一点。
    """

    def __init__(self, processes=None, *a, **k):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, func=None, iterable=None, **k):
        return map(func, iterable or [])

    def map(self, func=None, iterable=None, **k):
        return list(map(func, iterable or []))


def patch_ultralytics_pool() -> None:
    try:
        import ultralytics.data.dataset as uds

        uds.ThreadPool = _SerialPool  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover
        print(f"[det] 跳过 ThreadPool 替换：{exc}")


def make_obb_trainer(aux_weight: float = 0.1, enh_weights: Optional[str] = None):
    """返回一个在 get_model() 里挂 FiLM 的 OBBTrainer 子类。

    关键点：ultralytics 训练时会用 cfg **重新构建**模型，之前挂在传进去那个
    模型上的 hook / 子模块会被丢掉（权重不会进 checkpoint）。所以必须在
    trainer.get_model() 里挂，才能保证 FiLM 参数被优化并随权重保存。
    """
    from ultralytics.models.yolo.obb import OBBTrainer

    class FilmOBBTrainer(OBBTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            adapter = attach_joint_enhancement(model, aux_weight=aux_weight)
            if enh_weights and os.path.exists(enh_weights):
                ckpt = torch.load(enh_weights, map_location="cpu")
                if "estimator" in ckpt:
                    adapter.estimator.load_state_dict(ckpt["estimator"])
                    print(f"[det] 载入照度估计器：{enh_weights}（解码器已丢弃）")
            self.film_adapter = adapter
            print(f"[det] FiLM 已挂到 trainer 模型：params={sum(p.numel() for p in adapter.parameters())/1e3:.1f}k")
            return model

    return FilmOBBTrainer


def train_detector(
    data: str,
    model: str = "yolo11-obb.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    workers: int = 0,
    lr0: float = 0.01,
    project: str = "runs/obb",
    name: str = "goat_gate",
    film: bool = True,
    aux_weight: float = 0.1,
    enh_weights: Optional[str] = None,
    resume: bool = False,
    seed: int = 0,
    patience: int = 30,
):
    yolo = build_obb_model(model, nc=0, verbose=True)
    patch_ultralytics_pool()
    trainer_cls = None
    if film:
        trainer_cls = make_obb_trainer(aux_weight=aux_weight, enh_weights=enh_weights)
    results = yolo.train(
        trainer=trainer_cls,
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        lr0=lr0,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        seed=seed,
        patience=patience,
        plots=True,
        val=True,
    )
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="goat_count 训练入口")
    ap.add_argument("--stage", choices=["enh", "det", "all"], default="det")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--pairs", default="data/pairs/pairs.jsonl")
    ap.add_argument("--model", default="yolo11-obb.yaml", help="yaml 结构或 .pt 权重")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--enh-epochs", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--project", default="runs/obb")
    ap.add_argument("--name", default="goat_gate")
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--enh-weights", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="只跑 1 个 epoch / 极小尺寸")
    args = ap.parse_args(argv)

    if args.smoke:
        args.epochs = max(1, min(args.epochs, 1))
        args.imgsz = min(args.imgsz, 320)
        args.batch = max(1, min(args.batch, 2))

    enh_out = args.enh_weights
    if args.stage in ("enh", "all"):
        enh_out = train_enhancer(
            args.pairs,
            epochs=args.enh_epochs,
            device="cuda" if args.device not in ("cpu",) else "cpu",
        )
    if args.stage in ("det", "all"):
        train_detector(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            lr0=args.lr0,
            project=args.project,
            name=args.name,
            film=not args.no_film,
            aux_weight=args.aux_weight,
            enh_weights=enh_out,
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
