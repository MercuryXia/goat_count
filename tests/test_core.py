# -*- coding: utf-8 -*-
"""单元测试：几何、计数判决、跟踪、步态、指标、标签读写。

运行：``python -m goatcount selftest`` 或 ``python -m unittest discover -s tests -t .``
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np

from goatcount.config import SceneConfig
from goatcount.counting import CrossingEvent, GateCounter
from goatcount.data import ObbLabel, degrade_low_light, read_labels, split_by_day, write_labels
from goatcount.demo import BEV_QUAD, IMAGE_QUAD
from goatcount.geometry import (
    apply_homography,
    contact_point,
    homography_from_points,
    line_signed_distance,
    obb_from_xywhr,
)
from goatcount.metrics import count_metrics, direction_accuracy
from goatcount.tracking import LiteTracker, Observation, Track, gait_features, gait_says_dog


def _cfg() -> SceneConfig:
    cfg = SceneConfig()
    cfg.image_points = [list(p) for p in IMAGE_QUAD]
    cfg.bev_points = [list(p) for p in BEV_QUAD]
    cfg.line_p1 = (0.0, -2.7)
    cfg.line_p2 = (0.0, 2.7)
    cfg.in_side = "positive"
    return cfg


def _track(i: int, x: float, y: float = 0.0, cls_name: str = "goat") -> Track:
    t = Track(id=i, pos=np.array([x, y], dtype=float), vel=np.zeros(2), size=np.array([0.7, 0.4]),
              cls=0, cls_name=cls_name, conf=0.9)
    t.confirmed = True
    t.box_h_px = 60.0
    return t


class TestGeometry(unittest.TestCase):
    def test_homography_roundtrip(self):
        cfg = _cfg()
        for img_pt, bev_pt in zip(cfg.image_points, cfg.bev_points):
            got = apply_homography(cfg.H, np.asarray(img_pt, dtype=float).reshape(1, 2))[0]
            self.assertLess(float(np.linalg.norm(got - np.asarray(bev_pt))), 1e-6)

    def test_inverse_maps_back(self):
        cfg = _cfg()
        Hinv = np.linalg.inv(cfg.H)
        p = np.array([[120.0, 400.0]])
        bev = apply_homography(cfg.H, p)
        back = apply_homography(Hinv, bev)
        self.assertLess(float(np.linalg.norm(back - p)), 1e-6)

    def test_line_sign(self):
        cfg = _cfg()
        s_inside = float(line_signed_distance(np.array([-1.0, 0.0]), cfg.line_p1, cfg.line_p2))
        s_outside = float(line_signed_distance(np.array([1.0, 0.0]), cfg.line_p1, cfg.line_p2))
        self.assertGreater(s_inside, 0)
        self.assertLess(s_outside, 0)

    def test_contact_point_is_lowest_edge(self):
        # 图像坐标里 v 越大越靠下
        corners = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=float)
        c = contact_point(corners)
        self.assertAlmostEqual(c[0], 5.0)
        self.assertAlmostEqual(c[1], 20.0)

    def test_obb_from_xywhr_area(self):
        poly = obb_from_xywhr([100, 100, 40, 20, 0.0])
        area = 0.5 * abs(
            np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
        )
        self.assertAlmostEqual(float(area), 800.0, places=3)


class TestCounter(unittest.TestCase):
    def _run(self, xs, cfg=None, dt=0.04, track_id=1):
        cfg = cfg or _cfg()
        counter = GateCounter(cfg)
        events = []
        for k, x in enumerate(xs):
            events.extend(counter.update(k * dt, k, [_track(track_id, x)]))
        return counter, events

    def test_single_out_crossing(self):
        _, ev = self._run([-2.0 + 0.05 * i for i in range(81)])
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].direction, "out")

    def test_single_in_crossing(self):
        counter, ev = self._run([2.0 - 0.05 * i for i in range(81)])
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].direction, "in")
        self.assertEqual(counter.counts["in"], 1)

    def test_jitter_inside_band_not_counted(self):
        xs = [0.2 * math.sin(0.8 * i) for i in range(80)]
        _, ev = self._run(xs)
        self.assertEqual(ev, [])

    def test_no_margin_no_event(self):
        xs = [-1.0 + 0.02 * i for i in range(60)]  # 只走到 +0.18，未超过 margin
        _, ev = self._run(xs)
        self.assertEqual(ev, [])

    def test_shallow_crossing_rejected(self):
        # 过线但只走 0.1 m（< 0.35 m 余量）
        xs = [-0.5 + 0.02 * i for i in range(40)]
        _, ev = self._run(xs)
        self.assertEqual(ev, [])

    def test_duplicate_tracks_merged(self):
        cfg = _cfg()
        counter = GateCounter(cfg)
        ev = []
        for k in range(60):
            x = -1.5 + 0.05 * k
            ev.extend(counter.update(k * 0.04, k, [_track(1, x), _track(2, x + 0.02)]))
        self.assertEqual(len(ev), 1)
        self.assertEqual(counter.counts["out"], 1)

    def test_night_needs_more_frames(self):
        cfg = _cfg()
        cfg.confirm_frames_night = 12
        cfg.confirm_frames_day = 3
        cfg.hysteresis_m = 0.3
        counter = GateCounter(cfg, illuminance=5.0)  # 夜间
        xs = [-0.4 + 0.12 * i for i in range(8)]     # 带内约 5 帧
        ev = []
        for k, x in enumerate(xs):
            ev.extend(counter.update(k * 0.04, k, [_track(1, x)]))
        self.assertEqual(ev, [], "夜间应因确认帧数不足而放弃")

    def test_reentry_flagged(self):
        cfg = _cfg()
        counter = GateCounter(cfg)
        ev = []
        xs = [-2.0 + 0.05 * i for i in range(70)] + [1.45 - 0.06 * i for i in range(1, 46)]
        for k, x in enumerate(xs):
            ev.extend(counter.update(k * 0.04, k, [_track(1, x)]))
        self.assertEqual(len(ev), 2)
        self.assertTrue(any(e.reentry for e in ev))

    def test_conservation_ledger(self):
        from goatcount.counting import ConservationLedger

        led = ConservationLedger(tol_abs=3, tol_rel=0.03)
        row = led.add_day("2026-11-01", out_count=100, in_count=98, stock_open=120)
        self.assertFalse(row["anomaly"])
        self.assertEqual(row["stock_close"], 118)
        bad = led.add_day("2026-11-02", out_count=100, in_count=80, stock_open=118)
        self.assertTrue(bad["anomaly"])


class TestTracker(unittest.TestCase):
    def test_ids_stay_stable(self):
        cfg = _cfg()
        trk = LiteTracker(cfg, fps=25)
        ids = set()
        for k in range(40):
            x = -3.0 + 0.08 * k
            o = Observation(pos=np.array([x, 0.3]), size=np.array([0.7, 0.4]), cls=0, conf=0.9,
                            corners=np.zeros((4, 2)), box_h_px=50, w_img=40, h_img=50)
            tracks = trk.update(k * 0.04, k, [o])
            for t in tracks:
                if t.confirmed:
                    ids.add(t.id)
        self.assertEqual(len(ids), 1)

    def test_two_targets_keep_two_ids(self):
        cfg = _cfg()
        trk = LiteTracker(cfg, fps=25)
        ids = {0: set(), 1: set()}
        for k in range(40):
            o1 = Observation(np.array([-3.0 + 0.05 * k, -1.0]), np.array([0.7, 0.4]), 0, 0.9, np.zeros((4, 2)), 50, 40, 50)
            o2 = Observation(np.array([-3.0 + 0.05 * k, 1.2]), np.array([0.7, 0.4]), 0, 0.9, np.zeros((4, 2)), 50, 40, 50)
            tracks = trk.update(k * 0.04, k, [o1, o2])
            for t in tracks:
                if t.confirmed:
                    ids[0 if t.pos[1] < 0 else 1].add(t.id)
        self.assertEqual(len(ids[0]), 1)
        self.assertEqual(len(ids[1]), 1)
        self.assertNotEqual(ids[0], ids[1])

    def test_reid_reuses_id_after_gap(self):
        cfg = _cfg()
        cfg.max_age_s = 0.5
        trk = LiteTracker(cfg, fps=25)
        first = None
        for k in range(10):
            o = Observation(np.array([-2.0, 0.0]), np.array([0.7, 0.4]), 0, 0.9, np.zeros((4, 2)), 50, 40, 50)
            for t in trk.update(k * 0.04, k, [o]):
                first = t.id
        # 中断 3 帧后在同一位置附近重新出现
        for k in range(10, 13):
            trk.update(k * 0.04, k, [])
        o = Observation(np.array([-1.98, 0.0]), np.array([0.7, 0.4]), 0, 0.9, np.zeros((4, 2)), 50, 40, 50)
        tracks = trk.update(13 * 0.04, 13, [o])
        self.assertIn(first, [t.id for t in tracks])


class TestGait(unittest.TestCase):
    def _history(self, hz: float, n: int = 100, fps: float = 25.0):
        ts = np.arange(n) / fps
        h = 50 + 4 * np.sin(2 * np.pi * hz * ts)
        w = 40 + 1.5 * np.sin(2 * np.pi * hz * ts + 0.4)
        return [(float(t), float(ww), float(hh), 0.9) for t, ww, hh in zip(ts, w, h)]

    def test_fast_gait_reads_as_dog(self):
        cfg = _cfg()
        feats = gait_features(self._history(3.0), fps=25)
        self.assertGreater(feats["dominant_hz"], 2.4)
        self.assertTrue(gait_says_dog(feats, cfg))

    def test_slow_gait_reads_as_goat(self):
        cfg = _cfg()
        feats = gait_features(self._history(1.5), fps=25)
        self.assertLess(feats["dominant_hz"], 1.9)
        self.assertFalse(gait_says_dog(feats, cfg))

    def test_too_short_returns_empty(self):
        self.assertEqual(gait_features(self._history(2.0, n=5), fps=25), {})


class TestMetrics(unittest.TestCase):
    def test_count_metrics_exact(self):
        m = count_metrics([(10, 10), (9, 10)])
        self.assertAlmostEqual(m["CA"], 0.95)
        self.assertAlmostEqual(m["MAE"], 0.5)

    def test_direction_accuracy(self):
        pred = [
            CrossingEvent(t=1.0, frame=0, track_id=1, direction="out", cls_name="goat", conf=0.9, be_xy=(0, 0)),
            CrossingEvent(t=5.0, frame=0, track_id=2, direction="in", cls_name="goat", conf=0.9, be_xy=(0, 0)),
        ]
        gt = [
            CrossingEvent(t=1.1, frame=0, track_id=9, direction="out", cls_name="goat", conf=1.0, be_xy=(0, 0)),
            CrossingEvent(t=5.2, frame=0, track_id=8, direction="in", cls_name="goat", conf=1.0, be_xy=(0, 0)),
        ]
        m = direction_accuracy(pred, gt, tol_s=1.0)
        self.assertEqual(m["matched"], 2)
        self.assertAlmostEqual(m["direction_acc"], 1.0)


class TestData(unittest.TestCase):
    def test_label_roundtrip(self):
        import shutil

        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_labels")
        os.makedirs(d, exist_ok=True)
        try:
            p = os.path.join(d, "a.txt")
            poly = np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.4], [0.1, 0.4]])
            write_labels(p, [ObbLabel(0, poly), ObbLabel(1, poly)])
            got = read_labels(p)
            self.assertEqual(len(got), 2)
            self.assertEqual(got[0].cls, 0)
            self.assertLess(float(np.abs(got[0].poly - poly).max()), 1e-5)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_split_by_day_groups(self):
        imgs = [f"2026110{i}/img_{j}.jpg" for i in range(1, 6) for j in range(4)]
        split = split_by_day(imgs, (0.6, 0.2, 0.2), seed=1)
        days = {k: {p.split("/")[0] for p in v} for k, v in split.items()}
        self.assertFalse(days["train"] & days["val"])
        self.assertFalse(days["train"] & days["test"])
        self.assertEqual(sum(len(v) for v in split.values()), len(imgs))

    def test_degrade_darkens(self):
        img = np.full((60, 80, 3), 180, dtype=np.uint8)
        out = degrade_low_light(img, np.random.default_rng(0), gamma=2.2, gain=0.2, noise_sigma=4, jpeg_q=0)
        self.assertEqual(out.shape, img.shape)
        self.assertLess(float(out.mean()), float(img.mean()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
