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
    def __init__(self, weights: str, conf: float = 0.35):
        self.model = YOLO(weights)
        self.conf = conf

    def reset(self):
        try:
            self.model.predictor = None
        except Exception:
            pass

    def detect(self, frame) -> List[dict]:
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            verbose=False,
        )[0]
        players = []
        if result.boxes is None or result.keypoints is None:
            return players
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.xy.cpu().numpy()
        ids = result.boxes.id
        ids = ids.cpu().numpy() if ids is not None else np.arange(len(boxes))
        for box, conf, tid, kp in zip(boxes, confs, ids, kpts):
            players.append(
                {
                    "track_id": int(tid),
                    "bbox": [float(x) for x in box],
                    "confidence": float(conf),
                    "keypoints": kp.astype(float).tolist(),
                }
            )
        return players


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
