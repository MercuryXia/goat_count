# -*- coding: utf-8 -*-
"""检测器封装：真实权重（ultralytics）与彩色桩（无权重也能端到端冒烟）。

* UltralyticsObbDetector：读 yolo11-obb 权重，输出 OBB（cx,cy,w,h,angle）。
* ColorGateDetector：按 HSV 颜色找色块，用于 `cli demo`——没有训练权重时验证
  单应、跟踪、迟滞判决、守恒校验这条链路是否真的通。
* JsonDetector：从逐帧 JSON 回放检测结果，便于复现实验与回归测试。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np

from .pipeline import FrameObs

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


class UltralyticsObbDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.5,
        device: Optional[str] = None,
        max_det: int = 100,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.max_det = max_det
        self.names = self.model.names

    def __call__(self, frame: np.ndarray, t: float) -> FrameObs:
        r = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )[0]
        if r.obb is None or len(r.obb) == 0:
            return FrameObs(0, t, np.zeros((0, 5)), np.zeros((0,), dtype=np.int64), np.zeros((0,)))
        boxes = r.obb.xywhr.cpu().numpy().astype(np.float64)
        cls = r.obb.cls.cpu().numpy().astype(np.int64)
        conf = r.obb.conf.cpu().numpy().astype(np.float64)
        return FrameObs(0, t, boxes, cls, conf)


# HSV 阈值（OpenCV: H 0–179, S/V 0–255），对应合成视频里三种颜色
DEFAULT_HSV = {
    "goat": [(35, 60, 60), (85, 255, 255)],
    "dog": [(95, 60, 60), (135, 255, 255)],
    "person": [(5, 80, 80), (25, 255, 255)],
}


class ColorGateDetector:
    """颜色桩检测器：只用于演示/自测，不参与论文精度结论。"""

    def __init__(self, hsv: Optional[Dict[str, list]] = None, min_area: int = 120):
        if cv2 is None:
            raise RuntimeError("需要 opencv-python")
        self.hsv = hsv or DEFAULT_HSV
        self.min_area = min_area
        self.names = ["goat", "dog", "person"]

    def __call__(self, frame: np.ndarray, t: float) -> FrameObs:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        boxes, classes, confs = [], [], []
        for idx, name in enumerate(self.names):
            lo, hi = self.hsv[name]
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < self.min_area:
                    continue
                (cx, cy), (w, h), ang = cv2.minAreaRect(c)
                if w < 4 or h < 4:
                    continue
                boxes.append([cx, cy, w, h, np.deg2rad(ang)])
                classes.append(idx)
                # 用面积做置信度的粗糙代理
                confs.append(float(min(0.95, 0.5 + cv2.contourArea(c) / 5000.0)))
        if not boxes:
            return FrameObs(0, t, np.zeros((0, 5)), np.zeros((0,), dtype=np.int64), np.zeros((0,)))
        return FrameObs(
            0, t, np.asarray(boxes, dtype=np.float64), np.asarray(classes, dtype=np.int64), np.asarray(confs)
        )


class JsonDetector:
    """回放：JSON 每行 {"frame": i, "boxes": [[...]], "classes": [...], "confs": [...]}"""

    def __init__(self, path: str, fps: float = 25.0):
        self.by_frame: Dict[int, dict] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.by_frame[int(rec["frame"])] = rec
        self.fps = fps
        self._i = 0

    def __call__(self, frame: np.ndarray, t: float) -> FrameObs:
        idx = self._i
        self._i += 1
        rec = self.by_frame.get(idx, {"boxes": [], "classes": [], "confs": []})
        return FrameObs(
            idx,
            t,
            np.asarray(rec.get("boxes", []), dtype=np.float64).reshape(-1, 5),
            np.asarray(rec.get("classes", []), dtype=np.int64).reshape(-1),
            np.asarray(rec.get("confs", []), dtype=np.float64).reshape(-1),
        )
