import os
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from .court3d import estimate_camera, ray_at_z

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


def _cy(p):
    b = p["bbox"]
    return 0.5 * (b[1] + b[3])


def _cx(p):
    b = p["bbox"]
    return 0.5 * (b[0] + b[2])


def _area(p):
    b = p["bbox"]
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


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


def _shift(p, dy: float):
    q = dict(p)
    q["bbox"] = [p["bbox"][0], p["bbox"][1] + dy, p["bbox"][2], p["bbox"][3] + dy]
    q["keypoints"] = [[xy[0], xy[1] + dy] for xy in p.get("keypoints") or []]
    return q


def _pick_near_far(players, w: int, h: int) -> List[dict]:
    if not players:
        return []
    far_c = [p for p in players if _cy(p) < h * 0.52]
    near_c = [p for p in players if _cy(p) >= h * 0.48]
    out = []
    if far_c:
        p = min(far_c, key=lambda x: abs(_cx(x) - w * 0.5))
        p = dict(p)
        p["track_id"] = 1
        out.append(p)
    if near_c:
        p = max(near_c, key=_area)
        p = dict(p)
        p["track_id"] = 2
        if not out or abs(_cy(p) - _cy(out[0])) > 20:
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

    def reset(self):
        try:
            self.model.predictor = None
        except Exception:
            pass

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

    def detect(self, frame) -> List[dict]:
        h, w = frame.shape[:2]
        raw = self._run(frame)
        y1 = int(h * 0.48)
        for p in self._run(frame[:y1, :]):
            raw.append(_shift(p, 0.0))
        return _pick_near_far(_nms(raw), w, h)


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
