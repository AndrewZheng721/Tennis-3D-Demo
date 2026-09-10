import os
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from .court3d import estimate_camera, ray_at_z
from .geom import H_inv_of, NET_X, NET_Y, REF_KPS, image_to_court

ANKLE_L, ANKLE_R = 15, 16


def default_pose_weights() -> Optional[str]:
    for p in (
        "weights/yolo26m-pose.pt",
        "weights/yolo26n-pose.pt",
        os.path.join(os.path.dirname(__file__), "..", "..", "weights", "yolo26m-pose.pt"),
    ):
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(players, thr: float = 0.45) -> List[dict]:
    keep = []
    for p in sorted(players, key=lambda x: x["confidence"], reverse=True):
        if all(_iou(p["bbox"], q["bbox"]) < thr for q in keep):
            keep.append(p)
    return keep


def _in_play(court_xy) -> bool:
    x, y = court_xy
    x0 = float(REF_KPS[0, 0]) - 90
    x1 = float(REF_KPS[1, 0]) + 90
    y0 = float(REF_KPS[0, 1]) - 280
    y1 = float(REF_KPS[2, 1]) + 280
    return x0 <= x <= x1 and y0 <= y <= y1


def _select_players(players, det, w: int, h: int) -> List[dict]:
    if not players:
        return []
    H = H_inv_of(det)
    kps = None
    if det is not None and getattr(det, "keypoints_xy", None) is not None:
        kps = np.asarray(det.keypoints_xy, dtype=np.float64).reshape(-1, 2)
    net_y_img = None
    if kps is not None and len(kps) >= 14:
        net_y_img = 0.5 * (float(kps[12, 1]) + float(kps[13, 1]))
    buckets = {"far": [], "near": []}
    for p in players:
        uv = foot_uv(p.get("keypoints") or [], p.get("bbox"))
        if uv is None:
            continue
        court_xy = image_to_court(uv, H) if H is not None else None
        if court_xy is not None:
            if not _in_play(court_xy):
                continue
            side = "far" if court_xy[1] < NET_Y else "near"
            dist = abs(court_xy[0] - NET_X)
        elif net_y_img is not None and kps is not None:
            side = "far" if uv[1] < net_y_img else "near"
            idx = (0, 1, 4, 6, 8, 9, 12) if side == "far" else (2, 3, 5, 7, 10, 11, 13)
            dist = min(
                ((uv[0] - kps[i, 0]) ** 2 + (uv[1] - kps[i, 1]) ** 2) ** 0.5 for i in idx
            )
        else:
            continue
        q = dict(p)
        q["_dist"] = dist
        buckets[side].append(q)
    out = []
    for side, tid in (("far", 1), ("near", 2)):
        if not buckets[side]:
            continue
        p = min(buckets[side], key=lambda x: x["_dist"])
        p["track_id"] = tid
        p.pop("_dist", None)
        out.append(p)
    return out


def foot_uv(kpts, bbox=None) -> Optional[tuple]:
    pts = []
    for i in (ANKLE_L, ANKLE_R):
        if i < len(kpts):
            x, y = float(kpts[i][0]), float(kpts[i][1])
            if np.isfinite(x) and np.isfinite(y) and x > 1 and y > 1:
                pts.append((x, y))
    if pts:
        return float(sum(p[0] for p in pts) / len(pts)), float(sum(p[1] for p in pts) / len(pts))
    if bbox and len(bbox) >= 4:
        return float((bbox[0] + bbox[2]) * 0.5), float(bbox[3])
    return None


class PlayerTracker:
    def __init__(self, weights: str, conf: float = 0.05, imgsz: int = 1280):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.prev = {"far": None, "near": None}
        self.miss = {"far": 0, "near": 0}

    def reset(self):
        try:
            self.model.predictor = None
        except Exception:
            pass
        self.prev = {"far": None, "near": None}
        self.miss = {"far": 0, "near": 0}

    def _run(self, image) -> List[dict]:
        result = self.model.predict(
            image,
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=40,
            verbose=False,
        )[0]
        if result.boxes is None or result.keypoints is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.xy.cpu().numpy()
        out = []
        for i, (box, conf, kp) in enumerate(zip(boxes, confs, kpts)):
            out.append(
                {
                    "track_id": int(i),
                    "bbox": [float(x) for x in box],
                    "confidence": float(conf),
                    "keypoints": kp.astype(float).tolist(),
                }
            )
        return out

    def detect(self, frame, court_det=None) -> List[dict]:
        h, w = frame.shape[:2]
        raw = self._run(frame)
        y1 = int(h * 0.48)
        raw.extend(self._run(frame[:y1, :]))
        picked = _select_players(_nms(raw), court_det, w, h)
        by_id = {p["track_id"]: p for p in picked}
        out = []
        for side, tid in (("far", 1), ("near", 2)):
            if tid in by_id:
                self.prev[side] = by_id[tid]
                self.miss[side] = 0
                out.append(by_id[tid])
            elif self.prev[side] is not None and self.miss[side] < 18:
                self.miss[side] += 1
                out.append(self.prev[side])
            else:
                self.miss[side] += 1
        return out


def lift_players(players, det, image_shape, cam=None):
    if det is not None and getattr(det, "quality_ok", False):
        got = estimate_camera(det.keypoints_xy, image_shape)
        if got is not None:
            cam = got
    h = image_shape[0]
    out = []
    for p in players:
        uv = foot_uv(p.get("keypoints") or [], p.get("bbox"))
        xyz = ray_at_z(cam, uv, 0.0) if uv is not None else None
        if xyz is not None:
            side = "far" if xyz[1] >= 0 else "near"
        elif uv is not None:
            side = "near" if uv[1] > h * 0.55 else "far"
        else:
            side = None
        q = dict(p)
        q["foot_xy"] = None if uv is None else [uv[0], uv[1]]
        q["xyz"] = None if xyz is None else [xyz[0], xyz[1], xyz[2]]
        q["side"] = side
        out.append(q)
    return out, cam


def draw_players(img, players):
    import cv2

    for p in players:
        box = p.get("bbox") or []
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        side = p.get("side") or ""
        color = (0, 165, 255) if side == "near" else (255, 0, 180)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{side or 'P'} #{p.get('track_id', 0)}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return img
