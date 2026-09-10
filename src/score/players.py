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


def foot_uv(kpts) -> Optional[tuple]:
    pts = []
    for i in (ANKLE_L, ANKLE_R):
        if i < len(kpts):
            x, y = float(kpts[i][0]), float(kpts[i][1])
            if np.isfinite(x) and np.isfinite(y) and x > 1 and y > 1:
                pts.append((x, y))
    if not pts:
        return None
    return float(sum(p[0] for p in pts) / len(pts)), float(sum(p[1] for p in pts) / len(pts))


class PlayerTracker:
    def __init__(self, weights: str, conf: float = 0.12, imgsz: int = 1280):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def reset(self):
        try:
            self.model.predictor = None
        except Exception:
            pass

    def detect(self, frame) -> List[dict]:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=30,
            verbose=False,
        )[0]
        players = []
        if result.boxes is None or result.keypoints is None:
            return players
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.xy.cpu().numpy()
        h, w = frame.shape[:2]
        raw = []
        for i, (box, conf, kp) in enumerate(zip(boxes, confs, kpts)):
            raw.append(
                {
                    "track_id": int(i),
                    "bbox": [float(x) for x in box],
                    "confidence": float(conf),
                    "keypoints": kp.astype(float).tolist(),
                }
            )
        return _pick_near_far(raw, w, h)


def lift_players(players, det, image_shape, cam=None):
    if det is not None and getattr(det, "quality_ok", False):
        got = estimate_camera(det.keypoints_xy, image_shape)
        if got is not None:
            cam = got
    h = image_shape[0]
    out = []
    for p in players:
        uv = foot_uv(p.get("keypoints") or [])
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
