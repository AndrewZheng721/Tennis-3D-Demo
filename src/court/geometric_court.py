"""水平机位球场检测：不靠 ResNet，靠场地颜色 + 标准球场比例。

为什么 ResNet 在这种视频里会失败
--------------------------------
ResNet 权重是用「高位转播机位」训练的：摄像机在看台上方，整片场都摊在地面上。
水平机位（齐胸、站在底线后面）时，远端底线在画面里几乎贴着幕墙，
近端底线在脚边。网络没见过这种透视，就会把远端点标到幕布上，
整张绿框看起来像悬在空中。这不是权重下错，是任务已经变了。

微调也救不了：要换数据、换结构。短期能用的办法是几何法：
1. 找出「场地颜色」那一大块（蓝场、绿场、红土）。
2. 用这块区域的外轮廓得到 4 个角。
3. 用标准网球场的真实长宽比，做单应性（homography）把 14 个点投到画面上。

单应性可以想成：把一张标准俯视球场图，按透视贴到当前照片上。
只要 4 个外角大致对，发球线、T 点会按真实比例自动出现。
"""

from typing import Optional, Tuple

import cv2
import numpy as np

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


COLOR_PRESETS = [
    ("blue", np.array([85, 40, 40]), np.array([135, 255, 255])),
    ("green", np.array([35, 40, 40]), np.array([85, 255, 255])),
    ("clay", np.array([0, 50, 40]), np.array([25, 255, 255])),
]


def order_quad(points: np.ndarray) -> np.ndarray:
    """把 4 个点排成：左上、右上、右下、左下。

    原理：左上 x+y 最小，右下 x+y 最大；
    右上 x-y 最小（x 大 y 小），左下 x-y 最大。
    """
    pts = np.array(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _mask_from_range(hsv, lo, hi):
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    a = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), hi)
    b = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]]))
    return cv2.bitwise_or(a, b)


def adaptive_surface_mask(image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """在画面中下部找面积最大、最像一片场地的颜色块。

    HSV 是把颜色拆成：色相 H（什么颜色）、饱和度 S（鲜不鲜）、明度 V（亮不亮）。
    场地通常是一大片比较纯的蓝/绿/土红，白线饱和度低，不会进这块掩膜。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    kernel = np.ones((9, 9), np.uint8)
    best_mask = None
    best_score = -1.0
    best_name = "none"

    candidates = list(COLOR_PRESETS)
    roi = hsv[int(h * 0.2) : int(h * 0.98), int(w * 0.08) : int(w * 0.92)]
    colored = (roi[:, :, 1] > 40) & (roi[:, :, 2] > 40)
    if colored.sum() > 1000:
        peak = int(np.argmax(np.bincount(roi[:, :, 0][colored].ravel(), minlength=180)))
        lo = np.array([(peak - 12) % 180, 40, 40], dtype=np.int32)
        hi = np.array([(peak + 12) % 180, 255, 255], dtype=np.int32)
        candidates.append(("adaptive", lo, hi))

    for name, lo, hi in candidates:
        mask = _mask_from_range(hsv, lo, hi)
        mask[: int(h * 0.08), :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        area = float(cv2.countNonZero(mask))
        if area < 0.04 * h * w:
            continue
        ys, xs = np.where(mask > 0)
        cy = float(ys.mean()) / h
        top_ratio = float(mask[: int(h * 0.18)].sum()) / (area + 1.0)
        score = area * (0.35 + 0.65 * cy) * (1.0 - 0.75 * top_ratio)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_name = name
    return best_mask, best_name


def quad_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """场地掩膜所有像素的凸包，再收成四边形。

    用全部像素而不是最大连通块，是因为人站在场上会把场地戳出洞，
    凸包（橡皮筋箍一圈）不受内部空洞影响。
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 2000:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect)
    if quad is None or len(quad) != 4:
        return None
    ordered = order_quad(quad)
    top_w = np.linalg.norm(ordered[0] - ordered[1])
    bot_w = np.linalg.norm(ordered[3] - ordered[2])
    if bot_w < 8 or top_w < 8:
        return None
    return ordered


def keypoints_on_mask(pts: np.ndarray, mask: np.ndarray, radius: int = 18) -> int:
    """统计有多少关键点落在场地颜色上（允许几个像素偏差）。"""
    h, w = mask.shape[:2]
    hit = 0
    for x, y in pts.reshape(-1, 2):
        xi, yi = int(round(x)), int(round(y))
        x1, y1 = max(0, xi - radius), max(0, yi - radius)
        x2, y2 = min(w, xi + radius + 1), min(h, yi + radius + 1)
        if mask[y1:y2, x1:x2].any():
            hit += 1
    return hit


def detect_geometric(image: np.ndarray):
    """几何法主入口。成功返回 (keypoints_flat, H, H_inv, mask, color_name)，失败返回 None。"""
    mask, color_name = adaptive_surface_mask(image)
    if mask is None:
        return None
    quad = quad_from_mask(mask)
    if quad is None:
        return None
    src = REF_KPS[[0, 1, 3, 2]].astype(np.float32)
    H, _ = cv2.findHomography(src, quad, method=0)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(REF_KPS.reshape(-1, 1, 2), H).reshape(14, 2)
    h, w = image.shape[:2]
    if np.any(proj[:, 0] < -0.2 * w) or np.any(proj[:, 0] > 1.2 * w):
        return None
    if np.any(proj[:, 1] < -0.2 * h) or np.any(proj[:, 1] > 1.2 * h):
        return None
    hit = keypoints_on_mask(proj[[0, 1, 2, 3, 4, 5, 6, 7]], mask)
    if hit < 5:
        return None
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = None
    return proj.flatten().astype(np.float32), H, H_inv, mask, color_name
