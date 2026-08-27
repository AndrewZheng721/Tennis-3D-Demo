"""TrackNet 网球追踪：连续 3 帧热力图，而不是 YOLO 单帧框。

YOLO 在高位转播里球只有几个像素，置信度乱跳。
TrackNet 看连续三帧的运动，输出球的热力图峰值。
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .ball_tracker import BallTracker
from .tracknet_models import TrackNetV1, TrackNetV4

V1_W, V1_H = 640, 360
V4_W, V4_H = 512, 288

TRACKNET_V1_URL = (
    "https://huggingface.co/vishnushenoy09/tracknet-v1-tennis/resolve/main/tracknet_weights.pth"
)
TRACKNET_V1_DRIVE = "https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view"


def _load_state(path: str):
    size = os.path.getsize(path)
    if size < 5_000_000:
        raise RuntimeError(
            f"{path} 只有 {size} 字节，权重没下完整。完整文件大约 40MB。\n"
            f"重新下载:\n  wget -O weights/tracknet.pth \"{TRACKNET_V1_URL}\"\n"
            f"或网盘: {TRACKNET_V1_DRIVE}"
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(str(k).startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def detect_tracknet_kind(state: dict) -> str:
    keys = list(state.keys())
    joined = " ".join(keys)
    if "conv18" in joined:
        return "v1"
    if "motion" in joined or "fusion" in joined or "c24" in joined:
        return "v4"
    out = state.get("c24.weight")
    if out is not None and tuple(out.shape)[:2] == (3, 64):
        return "v4"
    raise RuntimeError(
        "无法识别 TrackNet 权重结构。需要 yastrebksv V1（conv18）或 V4 Type A（c24/motion）。"
    )


def _weighted_blob(hm: np.ndarray, min_val: float, hint=None, max_area: int = 250):
    binary = (hm >= min_val).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    best_score = -1e18
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2 or area > max_area:
            continue
        ys, xs = np.where(labels == i)
        vals = hm[ys, xs].astype(np.float64)
        mass = float(vals.sum())
        if mass <= 0:
            continue
        cx = float((xs * vals).sum() / mass)
        cy = float((ys * vals).sum() / mass)
        score = mass
        if hint is not None:
            score = mass / (1.0 + 0.12 * float(np.hypot(cx - hint[0], cy - hint[1])))
        if score > best_score:
            best_score = score
            best = (cx, cy, float(min(1.0, mass / 4000.0)))
    return best


def decode_v1_from_argmax(hm: np.ndarray, orig_w: int, orig_h: int, hint=None):
    hx = hy = None
    if hint is not None:
        hx = hint[0] * V1_W / float(orig_w)
        hy = hint[1] * V1_H / float(orig_h)
    blob = _weighted_blob(hm.astype(np.float32), 127.0, hint=(hx, hy) if hx is not None else None)
    if blob is None:
        return None
    x, y, conf = blob
    sx, sy = orig_w / float(V1_W), orig_h / float(V1_H)
    return (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5, max(conf, 0.5)


def decode_float_heatmap(hm: np.ndarray, orig_w: int, orig_h: int, in_w: int, in_h: int, thresh: float, hint=None):
    if float(hm.max()) < thresh:
        return None
    hx = hy = None
    if hint is not None:
        hx = hint[0] * in_w / float(orig_w)
        hy = hint[1] * in_h / float(orig_h)
    blob = _weighted_blob(hm, thresh, hint=(hx, hy) if hx is not None else None, max_area=400)
    if blob is None:
        return None
    x, y, conf = blob
    sx, sy = orig_w / float(in_w), orig_h / float(in_h)
    return (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5, float(conf)


class TrackNetBallTracker:
    def __init__(self, model_path: str, device: Optional[str] = None, threshold: float = 0.5):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"找不到 TrackNet 权重: {model_path}\n"
                f"网球 V1（推荐，高位转播能用）:\n  {TRACKNET_V1_URL}\n"
                f"或 {TRACKNET_V1_DRIVE}\n"
                "保存为 weights/tracknet.pth\n"
                "V4 Type A 的 PyTorch 权重放到 weights/tracknet_v4.pth"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold
        state = _load_state(model_path)
        self.kind = detect_tracknet_kind(state)
        if self.kind == "v1":
            self.model = TrackNetV1()
            self.in_w, self.in_h = V1_W, V1_H
        else:
            self.model = TrackNetV4()
            self.in_w, self.in_h = V4_W, V4_H
        self.model.load_state_dict(state, strict=True if self.kind == "v1" else False)
        self.model.to(self.device)
        self.model.eval()
        self._buf: List[np.ndarray] = []
        self._orig_size = None
        self._prev_center = None
        self._misses = 0
        self._court = None
        self.diag = None

    def reset(self):
        self._buf = []
        self._orig_size = None
        self._prev_center = None
        self._misses = 0
        self.diag = None

    def set_court(self, keypoints_xy: np.ndarray):
        corners = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1, 2)[[0, 1, 3, 2]]
        c = corners.mean(axis=0)
        self._court = (corners - c) * 1.45 + c

    def _stack_triplet(self, frames: List[np.ndarray]) -> torch.Tensor:
        seq = frames[::-1] if self.kind == "v1" else frames
        chips = []
        for fr in seq:
            img = cv2.resize(fr, (self.in_w, self.in_h))
            if self.kind == "v4":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            chips.append(img.astype(np.float32) / 255.0)
        stacked = np.concatenate(chips, axis=2)
        tensor = torch.from_numpy(np.transpose(stacked, (2, 0, 1))).unsqueeze(0)
        return tensor.to(self.device)

    def _predict_triplet(self, frames: List[np.ndarray]) -> Optional[Tuple[float, float, float]]:
        h, w = frames[-1].shape[:2]
        inp = self._stack_triplet(frames)
        hint = self._prev_center
        with torch.no_grad():
            out = self.model(inp)
        if self.kind == "v1":
            hm = out.argmax(dim=1).detach().cpu().numpy()[0].reshape(self.in_h, self.in_w)
            return decode_v1_from_argmax(hm, w, h, hint=hint)
        heatmaps = out.detach().cpu().numpy()[0]
        return decode_float_heatmap(
            heatmaps[2], w, h, self.in_w, self.in_h, self.threshold, hint=hint
        )

    def detect_frame(self, frame: np.ndarray) -> Dict[int, List[float]]:
        h, w = frame.shape[:2]
        self.diag = float(np.hypot(w, h))
        self._buf.append(frame)
        if len(self._buf) < 3:
            return {}
        self._buf = self._buf[-3:]
        pred = self._predict_triplet(self._buf)
        if pred is None:
            self._misses += 1
            if self._misses > 8:
                self._prev_center = None
            return {}
        cx, cy, conf = pred
        if self._court is not None and cv2.pointPolygonTest(self._court, (cx, cy), False) < 0:
            self._misses += 1
            if self._misses > 8:
                self._prev_center = None
            return {}
        if self._prev_center is not None:
            jump = float(np.hypot(cx - self._prev_center[0], cy - self._prev_center[1]))
            if jump > 0.22 * self.diag:
                self._misses += 1
                if self._misses > 8:
                    self._prev_center = None
                return {}
        self._prev_center = (cx, cy)
        self._misses = 0
        r = max(6.0, 0.004 * float(max(h, w)))
        return {1: [cx - r, cy - r, cx + r, cy + r, float(conf)]}

    interpolate_ball_positions = BallTracker.interpolate_ball_positions
    get_ball_shot_frames = BallTracker.get_ball_shot_frames
    to_json = BallTracker.to_json
    draw_frame = BallTracker.draw_frame


def create_ball_tracker(
    weights: str,
    backend: str = "auto",
    conf: float = 0.15,
    imgsz: int = 1280,
    coco_sports_ball: bool = False,
):
    if backend == "auto":
        name = os.path.basename(weights).lower()
        if "tracknet" in name or name.endswith(".pth"):
            backend = "tracknet"
        else:
            backend = "yolo"
    if backend in ("tracknet", "tracknetv4", "v4", "v1"):
        return TrackNetBallTracker(weights)
    return BallTracker(weights, conf=conf, imgsz=imgsz, coco_sports_ball=coco_sports_ball)


def default_ball_weights():
    candidates = [
        ("weights/tracknet_v4.pth", "tracknet"),
        ("weights/tracknet.pth", "tracknet"),
        ("weights/tracknet_weights.pth", "tracknet"),
        ("weights/yolo5_last.pt", "yolo"),
    ]
    for path, backend in candidates:
        if os.path.isfile(path):
            return path, backend
    return "weights/tracknet.pth", "tracknet"
