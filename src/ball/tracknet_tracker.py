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
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
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


def _largest_blob_center(binary: np.ndarray):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    if stats[idx, cv2.CC_STAT_AREA] < 3:
        return None
    ys, xs = np.where(labels == idx)
    return float(xs.mean()), float(ys.mean()), float(min(1.0, stats[idx, cv2.CC_STAT_AREA] / 40.0))


def decode_v1_from_argmax(hm: np.ndarray, orig_w: int, orig_h: int):
    hm_u8 = hm.astype(np.uint8)
    _, binary = cv2.threshold(hm_u8, 127, 255, cv2.THRESH_BINARY)
    blob = _largest_blob_center(binary)
    if blob is None:
        return None
    x, y, conf = blob
    sx, sy = orig_w / float(V1_W), orig_h / float(V1_H)
    return (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5, conf


def decode_float_heatmap(hm: np.ndarray, orig_w: int, orig_h: int, in_w: int, in_h: int, thresh: float):
    if float(hm.max()) < thresh:
        return None
    binary = (hm >= thresh).astype(np.uint8) * 255
    blob = _largest_blob_center(binary)
    if blob is None:
        y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
        conf = float(hm[y, x])
        blob = (float(x), float(y), conf)
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

    def reset(self):
        self._buf = []
        self._orig_size = None

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
        with torch.no_grad():
            out = self.model(inp)
        if self.kind == "v1":
            hm = out.argmax(dim=1).detach().cpu().numpy()[0].reshape(self.in_h, self.in_w)
            return decode_v1_from_argmax(hm, w, h)
        heatmaps = out.detach().cpu().numpy()[0]
        return decode_float_heatmap(
            heatmaps[2], w, h, self.in_w, self.in_h, self.threshold
        )

    def detect_frame(self, frame: np.ndarray) -> Dict[int, List[float]]:
        self._buf.append(frame)
        if len(self._buf) < 3:
            return {}
        self._buf = self._buf[-3:]
        pred = self._predict_triplet(self._buf)
        if pred is None:
            return {}
        cx, cy, conf = pred
        r = max(6.0, 0.004 * float(max(frame.shape[0], frame.shape[1])))
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
