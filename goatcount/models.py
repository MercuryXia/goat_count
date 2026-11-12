# -*- coding: utf-8 -*-
"""第 3 章模型侧：YOLO11-OBB 基线 + 照度条件 FiLM 调制。

设计要点（对应开题 3.3.1）：
* 不做"先增强、再检测"的串联，而是在检测头读取颈部特征之前，用照度估计把特征
  做一次仿射调制：F' = (1+γ(l))·F + β(l)。γ、β 由一个小 MLP 产生，随检测损失端到端训练。
* 照度估计器 IllumEstimator 只看输入图的低频分量，参数量极小（约 2k），推理延迟可忽略。
* 增强解码器（Retinex 重建）在训练阶段作为辅助任务单独训练（见 train.py --stage enh），
  导出时裁掉，因此线上仍是单网络——这是"部署优势"的由来。
* 挂载方式：在检测头（model.model[-1]）上做 forward_pre_hook，改的是它收到的特征列表，
  这样不依赖 YOLO 具体版本的颈部层名，升级 ultralytics 也不易失效。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 照度估计
# --------------------------------------------------------------------------- #
class IllumEstimator(nn.Module):
    """输入 RGB（0–1），输出单通道相对照度图（0–1）。参数量约 2k。"""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 期望 0–1；若输入是 0–255 的 uint8 浮点，先归一化
        if x.max() > 2.0:
            x = x / 255.0
        return torch.sigmoid(self.net(x))


# --------------------------------------------------------------------------- #
# FiLM 适配器
# --------------------------------------------------------------------------- #
class FiLMAdapter(nn.Module):
    """把照度条件注入检测头输入的三级特征（P3/P4/P5）。"""

    def __init__(self, levels: int = 3, hidden: int = 32, aux_weight: float = 0.1):
        super().__init__()
        self.levels = levels
        self.aux_weight = float(aux_weight)
        self.estimator = IllumEstimator()
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, 2))
                for _ in range(levels)
            ]
        )
        # 初始化成恒等映射附近，避免一开始就破坏预训练特征
        for mlp in self.mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        self.enabled = True
        self._input: Optional[torch.Tensor] = None
        self._illum: Optional[torch.Tensor] = None
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    # ---- hooks ----
    def grab_input(self, module, args):
        if args and torch.is_tensor(args[0]):
            self._input = args[0]
        return None

    def modulate(self, module, args):
        if not self.enabled or not args:
            return None
        feats = args[0]
        if not isinstance(feats, (list, tuple)) or not torch.is_tensor(feats[0]):
            return None
        if self._input is None:
            return None
        illum = self.estimator(self._input)          # (B,1,H/4,W/4)
        self._illum = illum
        l = illum.mean(dim=(2, 3))                    # (B,1) 全局照度标量
        out = []
        n = min(len(feats), self.levels)
        for i in range(len(feats)):
            f = feats[i]
            if i >= n:
                out.append(f)
                continue
            gb = self.mlps[i](l.to(dtype=f.dtype))    # (B,2)
            gamma = gb[:, 0].view(-1, 1, 1, 1)
            beta = gb[:, 1].view(-1, 1, 1, 1)
            out.append(f * (1.0 + gamma) + beta)
        return (out,) + tuple(args[1:])

    # ---- aux loss（L_ill：平滑 + 曝光先验）----
    def aux_loss(self, target_illum: float = 0.35) -> torch.Tensor:
        if self._illum is None:
            return torch.zeros((), device=next(self.parameters()).device)
        m = self._illum
        tv = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean() + (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
        exposure = (m.mean() - target_illum) ** 2
        return tv + 0.5 * exposure

    def extra_loss(self) -> torch.Tensor:
        return self.aux_weight * self.aux_loss()


def attach_joint_enhancement(
    model,
    levels: int = 3,
    hidden: int = 32,
    aux_weight: float = 0.1,
) -> FiLMAdapter:
    """把 FiLMAdapter 挂到 DetectionModel 上，并注册为子模块（参数进入优化器）。"""
    det = model
    if not (hasattr(det, "model") and isinstance(getattr(det, "model"), torch.nn.Sequential)):
        inner = getattr(model, "model", None)
        if hasattr(inner, "model") and isinstance(getattr(inner, "model"), torch.nn.Sequential):
            det = inner  # 传进来的是 YOLO 包装器
    if not (hasattr(det, "model") and isinstance(getattr(det, "model"), torch.nn.Sequential)):
        raise TypeError("需要 ultralytics 的 DetectionModel（或 YOLO 包装器）")
    adapter = FiLMAdapter(levels=levels, hidden=hidden, aux_weight=aux_weight)
    det.add_module("film_adapter", adapter)
    h1 = det.model[0].register_forward_pre_hook(adapter.grab_input)
    h2 = det.model[-1].register_forward_pre_hook(adapter.modulate)
    adapter.handles = [h1, h2]
    return adapter


def detach_joint_enhancement(model, adapter: FiLMAdapter) -> None:
    for h in adapter.handles:
        h.remove()
    adapter.handles = []


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #
def build_obb_model(model: str = "yolo11-obb.yaml", nc: int = 3, verbose: bool = True):
    """离线可构建：无预训练权重时用 yaml 结构 + 随机初始化（便于冒烟训练）。

    注意：只改 ``head.nc`` 而不重建分类分支会让输出通道与 nc 不一致
    （shape '[B, nc, -1]' is invalid），所以这里用 OBBModel(cfg, nc=nc) 直接建。
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import OBBModel, yaml_model_load

    if model.endswith((".yaml", ".yml")):
        cfg = dict(yaml_model_load(model))   # 支持 "yolo11-obb.yaml" 这种内置名
        det = OBBModel(cfg, ch=3, nc=nc)
        yolo = YOLO(model)          # 仅用于拿到 YOLO 封装（trainer 会自行重建）
        yolo.model = det
    else:
        yolo = YOLO(model)
        det = yolo.model
    if verbose:
        n = sum(p.numel() for p in det.parameters())
        print(f"[models] {model}  params={n/1e6:.2f}M  nc={getattr(det, 'nc', nc)}")
    return yolo


def smoke_test(imgsz: int = 320, nc: int = 3) -> dict:
    """前向 + 反向冒烟：确认 FiLM 已接入计算图，且梯度能回到 MLP。"""
    m = build_obb_model("yolo11-obb.yaml", nc=nc, verbose=False)
    det = m.model
    adapter = attach_joint_enhancement(det, aux_weight=0.1)
    det.train()
    x = torch.rand(2, 3, imgsz, imgsz)
    out = det(x)
    preds = out
    if isinstance(out, dict):                 # 训练态可能返回 {"one2many": ...}
        for v in out.values():
            if isinstance(v, torch.Tensor):
                preds = v
                break
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                preds = v[0]
                break
    elif isinstance(out, (list, tuple)):
        preds = out[0]
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    loss = preds.float().mean() + adapter.extra_loss()
    loss.backward()
    grad_ok = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in adapter.mlps.parameters()
    )
    return {
        "out_shape": tuple(preds.shape),
        "loss": float(loss.detach()),
        "film_grad_ok": bool(grad_ok),
        "adapter_params": sum(p.numel() for p in adapter.parameters()),
    }


if __name__ == "__main__":  # pragma: no cover
    print(smoke_test())
