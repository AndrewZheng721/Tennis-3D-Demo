"""TennisCourtDetector 热力图球场检测。

和 ResNet 的差别
----------------
ResNet：把整张图压成 224×224，最后吐出 28 个数字。透视一变就全错。
热力图：图缩到 640×360（比 224 清晰得多），为每个角点各画一张「亮斑图」，
在亮斑里找圆/最大值，才得到坐标。点是在图上搜出来的，不是猜出来的。

权重来自 yastrebksv/TennisCourtDetector，数据仍以高位转播为主。
水平机位会比 ResNet 稳一些，但不保证完美。点不够时用单应性把标准场投回去。
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .heatmap_net import CourtHeatmapNet

INPUT_W = 640
INPUT_H = 360

REF_KPS = np.array(
    [
        [286, 561],
        [1379, 561],
        [286, 2935],
        [1379, 2935],
        [423, 561],
        [423, 2935],
        [1242, 561],
        [1242, 2935],
        [423, 1110],
        [1242, 1110],
        [423, 2386],
        [1242, 2386],
        [832, 1110],
        [832, 2386],
    ],
    dtype=np.float32,
)

HOMO_CFGS = [
    [0, 1, 2, 3],
    [4, 6, 5, 7],
    [4, 6, 10, 11],
    [0, 1, 5, 7],
    [4, 1, 5, 3],
    [8, 9, 10, 11],
    [8, 9, 5, 7],
    [0, 4, 2, 5],
    [6, 1, 7, 3],
    [8, 12, 10, 13],
    [12, 9, 13, 11],
]


def _load_state(path: str):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def heatmap_peak(heatmap: np.ndarray, scale_x: float, scale_y: float, low_thresh: int = 170):
    """从一张热力图里取出最可能的角点。

    先用霍夫圆（Hough）找亮斑，找不到就用最亮像素。
    热力图是 360×640，要乘 scale 才能回到原图像素坐标。
    """
    hm = heatmap
    if hm.max() <= 1.5:
        hm = hm * 255.0
    hm_u8 = np.clip(hm, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(hm_u8, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=2,
        minRadius=10,
        maxRadius=30,
    )
    if circles is not None:
        x, y = float(circles[0][0][0]), float(circles[0][0][1])
        return x * scale_x, y * scale_y
    if hm_u8.max() < low_thresh:
        return None, None
    y, x = np.unravel_index(int(np.argmax(hm_u8)), hm_u8.shape)
    return float(x) * scale_x, float(y) * scale_y


def snap_with_partial_points(points: List[Optional[Tuple[float, float]]]):
    """部分点缺失时，用 4 个可靠点算单应性，把 14 个标准点一起投过去。"""
    pts = []
    for p in points:
        if p is None or p[0] is None or p[1] is None:
            pts.append(None)
        else:
            pts.append((float(p[0]), float(p[1])))

    best_H = None
    best_err = 1e9
    best_proj = None
    for idx in HOMO_CFGS:
        four = [pts[i] for i in idx]
        if any(p is None for p in four):
            continue
        src = REF_KPS[idx]
        dst = np.array(four, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(REF_KPS.reshape(-1, 1, 2), H).reshape(14, 2)
        errs = []
        for i, p in enumerate(pts):
            if p is None or i in idx:
                continue
            errs.append(float(np.hypot(proj[i, 0] - p[0], proj[i, 1] - p[1])))
        err = float(np.mean(errs)) if errs else 0.0
        if err < best_err:
            best_err = err
            best_H = H
            best_proj = proj
    if best_H is None:
        return None, None, None
    try:
        H_inv = np.linalg.inv(best_H)
    except np.linalg.LinAlgError:
        H_inv = None
    return best_proj.flatten().astype(np.float32), best_H, H_inv


class HeatmapCourtDetector:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"找不到热力图权重: {model_path}\n"
                "请下载 https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view\n"
                "保存为 weights/court_heatmap.pth"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CourtHeatmapNet(out_channels=15)
        self.model.load_state_dict(_load_state(model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict_points(self, image: np.ndarray) -> List[Optional[Tuple[float, float]]]:
        h, w = image.shape[:2]
        img = cv2.resize(image, (INPUT_W, INPUT_H))
        inp = img.astype(np.float32) / 255.0
        inp = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            out = self.model(inp.to(self.device))[0]
            pred = torch.sigmoid(out).cpu().numpy()
        sx, sy = w / float(INPUT_W), h / float(INPUT_H)
        points = []
        for k in range(14):
            x, y = heatmap_peak(pred[k], sx, sy)
            points.append(None if x is None else (x, y))
        return points

    def predict(self, image: np.ndarray):
        """返回 (keypoints_flat, H, H_inv)。点太少时 keypoints 可能含 nan。"""
        points = self.predict_points(image)
        snapped, H, H_inv = snap_with_partial_points(points)
        if snapped is not None:
            return snapped, H, H_inv
        flat = []
        for p in points:
            if p is None:
                flat.extend([np.nan, np.nan])
            else:
                flat.extend([p[0], p[1]])
        return np.array(flat, dtype=np.float32), None, None
