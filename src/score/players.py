import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .court3d import estimate_camera, ray_at_z
from .geom import NET_X, NET_Y, REF_KPS, image_to_court


def default_person_weights() -> Optional[str]:
    for p in (
        "weights/yolo26m.pt",
        "weights/yolov8x.pt",
        "weights/yolo26m-pose.pt",
        os.path.join(os.path.dirname(__file__), "..", "..", "weights", "yolo26m.pt"),
    ):
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def default_pose_weights() -> Optional[str]:
    return default_person_weights()


def _H_ref(det) -> Optional[np.ndarray]:
    if det is None:
        return None
    return getattr(det, "homography_ref_to_image", None)


def _H_inv(det) -> Optional[np.ndarray]:
    if det is None:
        return None
    return getattr(det, "homography_image_to_ref", None)


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


def _nms(players, thr: float = 0.5) -> List[dict]:
    keep = []
    for p in sorted(players, key=lambda x: x["confidence"], reverse=True):
        if all(_iou(p["bbox"], q["bbox"]) < thr for q in keep):
            keep.append(p)
    return keep


def foot_uv(kpts=None, bbox=None) -> Optional[Tuple[float, float]]:
    if bbox and len(bbox) >= 4:
        return float((bbox[0] + bbox[2]) * 0.5), float(bbox[3])
    return None


def _ref_half_masks(pad: int = 120):
    x0 = int(REF_KPS[0, 0]) - pad
    x1 = int(REF_KPS[1, 0]) + pad
    y_far0 = int(REF_KPS[0, 1]) - pad
    y_net = int(NET_Y)
    y_near1 = int(REF_KPS[2, 1]) + pad
    w = max(x1 + 1, int(REF_KPS[1, 0]) + pad + 1)
    h = max(y_near1 + 1, int(REF_KPS[2, 1]) + pad + 1)
    far = np.zeros((h, w), np.uint8)
    near = np.zeros((h, w), np.uint8)
    cv2.rectangle(far, (x0, y_far0), (x1, y_net), 1, -1)
    cv2.rectangle(near, (x0, y_net), (x1, y_near1), 1, -1)
    return far, near


_REF_FAR, _REF_NEAR = _ref_half_masks()


def _warp_masks(det, shape) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    H = _H_ref(det)
    if H is None:
        return None
    h, w = shape[:2]
    far = cv2.warpPerspective(_REF_FAR, H.astype(np.float64), (w, h))
    near = cv2.warpPerspective(_REF_NEAR, H.astype(np.float64), (w, h))
    return far, near


def _far_roi(det, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    kps = getattr(det, "keypoints_xy", None) if det is not None else None
    if kps is None:
        return None
    pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 14:
        return None
    idx = (0, 1, 4, 6, 8, 9, 12)
    xs = pts[[i for i in idx], 0]
    ys = pts[[i for i in idx], 1]
    x0 = int(max(0, xs.min() - 40))
    x1 = int(min(w, xs.max() + 40))
    y0 = int(max(0, ys.min() - 80))
    y1 = int(min(h, ys.max() + 120))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None
    return x0, y0, x1, y1


def _in_mask(mask, uv) -> bool:
    x, y = int(round(uv[0])), int(round(uv[1]))
    if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return mask[y, x] > 0


def _half_center(det, side: str) -> Optional[Tuple[float, float]]:
    H = _H_ref(det)
    if H is None:
        return None
    if side == "far":
        p = np.array([[[NET_X, (REF_KPS[0, 1] + NET_Y) * 0.5]]], dtype=np.float32)
    else:
        p = np.array([[[NET_X, (REF_KPS[2, 1] + NET_Y) * 0.5]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, H.astype(np.float32))[0, 0]
    return float(q[0]), float(q[1])


def _select_players(players, det, shape) -> List[dict]:
    if not players:
        return []
    masks = _warp_masks(det, shape)
    H_inv = _H_inv(det)
    buckets = {"far": [], "near": []}
    for p in players:
        uv = foot_uv(bbox=p.get("bbox"))
        if uv is None:
            continue
        side = None
        if masks is not None:
            if _in_mask(masks[0], uv):
                side = "far"
            elif _in_mask(masks[1], uv):
                side = "near"
        if side is None and H_inv is not None:
            cxy = image_to_court(uv, H_inv)
            if cxy is None:
                continue
            x0 = float(REF_KPS[0, 0]) - 120
            x1 = float(REF_KPS[1, 0]) + 120
            y0 = float(REF_KPS[0, 1]) - 200
            y1 = float(REF_KPS[2, 1]) + 200
            if not (x0 <= cxy[0] <= x1 and y0 <= cxy[1] <= y1):
                continue
            side = "far" if cxy[1] < NET_Y else "near"
        if side is None:
            continue
        center = _half_center(det, side)
        if center is not None:
            dist = (uv[0] - center[0]) ** 2 + (uv[1] - center[1]) ** 2
        else:
            dist = abs(uv[0] - shape[1] * 0.5)
        q = dict(p)
        q["_dist"] = dist
        buckets[side].append(q)
    out = []
    for side, tid in (("far", 1), ("near", 2)):
        if not buckets[side]:
            continue
        p = min(buckets[side], key=lambda x: x["_dist"])
        p["track_id"] = tid
        p["side"] = side
        p.pop("_dist", None)
        out.append(p)
    return out


class PlayerTracker:
    def __init__(self, weights: str, conf: float = 0.15, imgsz: int = 1280):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.is_pose = "pose" in os.path.basename(weights).lower()
        self.prev = {"far": None, "near": None}
        self.miss = {"far": 0, "near": 0}

    def reset(self):
        try:
            self.model.predictor = None
        except Exception:
            pass
        self.prev = {"far": None, "near": None}
        self.miss = {"far": 0, "near": 0}

    def _run(self, image, ox: float = 0.0, oy: float = 0.0) -> List[dict]:
        kwargs = dict(
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=50,
            verbose=False,
        )
        if not self.is_pose:
            kwargs["classes"] = [0]
        result = self.model.predict(image, **kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = None
        if self.is_pose and result.keypoints is not None:
            kpts = result.keypoints.xy.cpu().numpy()
        out = []
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            b = [
                float(box[0] + ox),
                float(box[1] + oy),
                float(box[2] + ox),
                float(box[3] + oy),
            ]
            item = {
                "track_id": int(i),
                "bbox": b,
                "confidence": float(conf),
                "keypoints": None,
            }
            if kpts is not None and i < len(kpts):
                item["keypoints"] = [
                    [float(xy[0] + ox), float(xy[1] + oy)] for xy in kpts[i]
                ]
            out.append(item)
        return out

    def detect(self, frame, court_det=None) -> List[dict]:
        h, w = frame.shape[:2]
        raw = self._run(frame)
        roi = _far_roi(court_det, w, h)
        if roi is not None:
            x0, y0, x1, y1 = roi
            crop = frame[y0:y1, x0:x1]
            if crop.size > 0:
                raw.extend(self._run(crop, ox=x0, oy=y0))
        picked = _select_players(_nms(raw), court_det, frame.shape)
        by_side = {p["side"]: p for p in picked if p.get("side")}
        out = []
        for side, tid in (("far", 1), ("near", 2)):
            if side in by_side:
                p = by_side[side]
                p["track_id"] = tid
                self.prev[side] = p
                self.miss[side] = 0
                out.append(p)
            elif self.prev[side] is not None and self.miss[side] < 20:
                self.miss[side] += 1
                hold = dict(self.prev[side])
                hold["track_id"] = tid
                hold["side"] = side
                out.append(hold)
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
        uv = foot_uv(p.get("keypoints"), p.get("bbox"))
        xyz = ray_at_z(cam, uv, 0.0) if uv is not None else None
        side = p.get("side")
        if side is None:
            if xyz is not None:
                side = "far" if xyz[1] >= 0 else "near"
            elif uv is not None:
                side = "near" if uv[1] > h * 0.55 else "far"
        q = dict(p)
        q["foot_xy"] = None if uv is None else [uv[0], uv[1]]
        q["xyz"] = None if xyz is None else [xyz[0], xyz[1], xyz[2]]
        q["side"] = side
        out.append(q)
    return out, cam


def draw_players(img, players):
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
