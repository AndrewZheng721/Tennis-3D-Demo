"""球场 14 个角点检测。

这个文件要解决的问题
--------------------
看网球比赛视频时，我们需要知道「球场画在画面的哪里」。
后面判断球有没有出界、人站在哪半场，都依赖这件事。

电脑怎么「看见」球场？
--------------------
我们不让模型去找每一条白线（线又细又容易被人和影子挡住），
而是让它直接回答：14 个固定角点分别在像素图的哪个位置。

这 14 个点是国际标准网球场上线与线的交点，例如：
双打底线角落、单打底线角落、发球线与边线的交点、发球区 T 字交叉点。
只要这 14 个点对了，整张球场的几何关系就能算出来。

模型是什么：ResNet50
--------------------
ResNet50 是一种卷积神经网络（CNN）。
你可以把它想成：很多层「小过滤器」叠在一起，从图片里一层层提取特征。
浅层看边缘和颜色，深层看「这大概是一片球场」。

它不是 YOLO。YOLO 的工作是「框出物体」；
这里的 ResNet 被改成了「回归」：最后输出 28 个数字（14 个点 × 每个点的 x、y）。

输入为什么要缩成 224×224？
--------------------------
这个权重是按 224×224 训练的。训练和推理必须用同一套预处理，
否则坐标会对不上。缩小时画面会被拉扁，这是原模型的设计，不是 bug。
预测完后再按原图宽高把坐标乘回去。

这个模型的能力边界
------------------
训练数据几乎全是「高位、从底线后方拍的转播机位」。
水平机位会把远端点投到幕墙上。检测到这种情况时，
自动改用几何法：场地颜色块 → 四个角 → 按标准球场比例投影 14 个点。
不要微调这个 ResNet 去适应水平机位。
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models

from .geometric_court import (
    adaptive_surface_mask,
    detect_geometric,
    keypoints_on_mask,
)


COURT_KEYPOINT_NAMES = [
    "远底线-左双打角",
    "远底线-右双打角",
    "近底线-左双打角",
    "近底线-右双打角",
    "远底线-左单打角",
    "近底线-左单打角",
    "远底线-右单打角",
    "近底线-右单打角",
    "远发球线-左单打交点",
    "远发球线-右单打交点",
    "近发球线-左单打交点",
    "近发球线-右单打交点",
    "远侧发球区T点",
    "近侧发球区T点",
]

COURT_LINES = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (4, 6),
    (5, 7),
    (12, 13),
    (8, 10),
    (9, 11),
]

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

HORIZ_SEGMENTS = [(0, 1), (2, 3), (8, 9), (10, 11), (4, 6), (5, 7)]
VERT_SEGMENTS = [(0, 2), (1, 3), (4, 5), (6, 7), (8, 10), (9, 11)]


def white_line_mask(image: np.ndarray) -> np.ndarray:
    """转播画面里的球场线往往带一点蓝/绿，不能只靠 gray>155。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    by_hsv = cv2.inRange(hsv, (0, 0, 165), (180, 70, 255))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    by_gray = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_or(by_hsv, by_gray)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _sample_line_offsets(mask: np.ndarray, pts: np.ndarray, pairs, along_x: bool):
    h, w = mask.shape[:2]
    band = 32
    offsets = []
    for a, b in pairs:
        p0, p1 = pts[a], pts[b]
        for t in np.linspace(0.12, 0.88, 18):
            px = float(p0[0] * (1.0 - t) + p1[0] * t)
            py = float(p0[1] * (1.0 - t) + p1[1] * t)
            xi, yi = int(round(px)), int(round(py))
            if along_x:
                if not (0 <= yi < h):
                    continue
                x0, x1 = max(0, xi - band), min(w, xi + band + 1)
                strip = mask[yi, x0:x1]
                hits = np.where(strip > 0)[0]
                if hits.size == 0:
                    continue
                wx = x0 + int(hits[np.argmin(np.abs(hits - (xi - x0)))])
                offsets.append(wx - px)
            else:
                if not (0 <= xi < w):
                    continue
                y0, y1 = max(0, yi - band), min(h, yi + band + 1)
                strip = mask[y0:y1, xi]
                hits = np.where(strip > 0)[0]
                if hits.size == 0:
                    continue
                wy = y0 + int(hits[np.argmin(np.abs(hits - (yi - y0)))])
                offsets.append(wy - py)
    return offsets


def align_keypoints_to_white_lines(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """整体平移网格，吸到真实白线上。

    高位转播里形状对、整网下沉几像素，是热力图峰值/霍夫圆的系统偏差，
    不是「画在白线内沿」的设计。这里用白线掩膜估计 (dx, dy)。
    """
    pts = keypoints.reshape(14, 2).astype(np.float32)
    mask = white_line_mask(image)
    dys = _sample_line_offsets(mask, pts, HORIZ_SEGMENTS, along_x=False)
    dxs = _sample_line_offsets(mask, pts, VERT_SEGMENTS, along_x=True)
    dx = float(np.median(dxs)) if len(dxs) >= 10 else 0.0
    dy = float(np.median(dys)) if len(dys) >= 10 else 0.0
    dx = float(np.clip(dx, -30.0, 30.0))
    dy = float(np.clip(dy, -30.0, 30.0))
    if abs(dx) < 0.4 and abs(dy) < 0.4:
        return keypoints
    pts[:, 0] += dx
    pts[:, 1] += dy
    return pts.flatten().astype(np.float32)


@dataclass
class CourtDetection:
    """一次球场检测的完整结果，方便后面阶段直接用。"""

    keypoints: np.ndarray
    keypoints_xy: np.ndarray
    homography_ref_to_image: Optional[np.ndarray]
    homography_image_to_ref: Optional[np.ndarray]
    quality_ok: bool
    quality_reason: str
    frame_id: int = 0
    method: str = "resnet"
    names: List[str] = field(default_factory=lambda: list(COURT_KEYPOINT_NAMES))

    def to_json(self):
        H = self.homography_ref_to_image
        H_inv = self.homography_image_to_ref
        return {
            "frame_id": int(self.frame_id),
            "method": self.method,
            "quality_ok": bool(self.quality_ok),
            "quality_reason": self.quality_reason,
            "keypoints": [
                {
                    "id": i,
                    "name": COURT_KEYPOINT_NAMES[i],
                    "x": float(self.keypoints_xy[i, 0]),
                    "y": float(self.keypoints_xy[i, 1]),
                }
                for i in range(14)
            ],
            "keypoints_flat": [float(v) for v in self.keypoints.tolist()],
            "homography_ref_to_image": None if H is None else H.tolist(),
            "homography_image_to_ref": None if H_inv is None else H_inv.tolist(),
        }


def _line_intersection(l1, l2):
    """求两条线段所在直线的交点。

    线段用 (x1, y1, x2, y2) 表示。
    两条线平行时分母接近 0，返回 None。
    """
    x1, y1, x2, y2 = map(float, l1)
    x3, y3, x4, y4 = map(float, l2)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def _intersect_from_hough(lines):
    """从一堆短线段里挑两条方向差得足够大的线，求它们的交点。

    霍夫变换（Hough）会在一小块图里找出很多「像直线」的片段。
    球场角点正好是两条线交叉的地方，所以交点比 CNN 直接回归的坐标更贴白线。
    """
    lines = lines.reshape(-1, 4)
    best, best_score = None, -1.0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1 = np.arctan2(lines[i, 3] - lines[i, 1], lines[i, 2] - lines[i, 0])
            a2 = np.arctan2(lines[j, 3] - lines[j, 1], lines[j, 2] - lines[j, 0])
            da = abs(a1 - a2) % np.pi
            da = min(da, np.pi - da)
            if da < np.deg2rad(25):
                continue
            inter = _line_intersection(lines[i], lines[j])
            if inter is None:
                continue
            score = da * (
                np.hypot(lines[i, 2] - lines[i, 0], lines[i, 3] - lines[i, 1])
                + np.hypot(lines[j, 2] - lines[j, 0], lines[j, 3] - lines[j, 1])
            )
            if score > best_score:
                best_score = score
                best = inter
    return best


class CourtLineDetector:
    """球场检测：优先热力图，其次几何法，ResNet 只作高位转播兜底。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        heatmap_path: str = "weights/court_heatmap.pth",
    ):
        """加载权重并切到推理模式。

        heatmap_path：TennisCourtDetector 热力图权重，水平/高位都优先用它。
        model_path：原来的 ResNet50 回归权重，热力图不可用时才考虑。
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = None
        self.transform = None
        self.heatmap = None

        if heatmap_path and os.path.isfile(heatmap_path):
            from .heatmap_court import HeatmapCourtDetector
            self.heatmap = HeatmapCourtDetector(heatmap_path, device=device)

        if model_path and os.path.isfile(model_path):
            try:
                self.model = models.resnet50(weights=None)
            except TypeError:
                self.model = models.resnet50(pretrained=False)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
            state = self._load_state(model_path)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        if self.heatmap is None and self.model is None:
            raise FileNotFoundError(
                "请至少提供一个球场权重。\n"
                "热力图（推荐）: weights/court_heatmap.pth\n"
                "https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view\n"
                "ResNet: weights/keypoints_model.pth"
            )

    def _load_state(self, model_path: str):
        """兼容不同 PyTorch 版本和 checkpoint 打包方式。"""
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return state

    def _predict_raw(self, image: np.ndarray) -> np.ndarray:
        """CNN 前向推理，得到原图像素坐标系下的 28 个数。

        步骤：
        1. OpenCV 读到的是 BGR，训练用的是 RGB，所以要转换。
        2. 缩到 224×224，并做 ImageNet 那套均值方差归一化（权重是在这套预处理上训的）。
        3. 网络输出的是 224×224 画布上的坐标，再乘回原图宽高。
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        return keypoints.astype(np.float32)

    def _refine_keypoint(
        self, image: np.ndarray, x: float, y: float, crop_size: int = 40
    ) -> Tuple[float, float]:
        """在预测点周围裁一小块，用白线交点把点吸到真正的角落。

        CNN 在 224×224 上差 1 个像素，回到 1920 宽的图上就差大约 8 个像素。
        球场线是白的、交点是几何确定的，所以局部找线比继续信任 CNN 更准。
        找不到两条交叉线时，退回 CNN 原来的点。
        """
        h, w = image.shape[:2]
        x, y = int(round(x)), int(round(y))
        x1, y1 = max(x - crop_size, 0), max(y - crop_size, 0)
        x2, y2 = min(x + crop_size, w), min(y + crop_size, h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return float(x), float(y)
        binary = white_line_mask(crop)
        lines = cv2.HoughLinesP(
            binary, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30
        )
        if lines is None or len(lines) < 2:
            return float(x), float(y)
        inter = _intersect_from_hough(lines)
        if inter is None:
            return float(x), float(y)
        rx, ry = inter
        if not (0 <= rx < crop.shape[1] and 0 <= ry < crop.shape[0]):
            return float(x), float(y)
        return x1 + rx, y1 + ry

    def _refine(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """对 14 个点逐个做白线吸附。"""
        out = keypoints.copy()
        for i in range(14):
            x, y = self._refine_keypoint(image, out[2 * i], out[2 * i + 1])
            out[2 * i], out[2 * i + 1] = x, y
        return out

    def _homography_snap(self, keypoints: np.ndarray):
        """用标准球场的几何约束，把 14 个点整体对齐成「像一个真球场」。

        单应性（homography）是一张 3×3 的矩阵。
        它描述的是：把「标准俯视球场」上的点，投到「当前摄像机拍到的画面」上。
        只要 4 个对应点可靠，其余点都可以由这张矩阵算出来。

        这样做的好处：个别点被球员挡住时，仍能被标准几何「拉回去」。
        如果整组点已经很离谱（例如水平机位），误差会超过阈值，我们宁可不吸附。
        """
        pts = keypoints.reshape(14, 2).astype(np.float32)
        best_err = 1e9
        best_proj = None
        best_H = None
        for idx in HOMO_CFGS:
            src = REF_KPS[idx]
            dst = pts[idx]
            H, _ = cv2.findHomography(src, dst, method=0)
            if H is None:
                continue
            proj = cv2.perspectiveTransform(REF_KPS.reshape(-1, 1, 2), H).reshape(14, 2)
            err = float(np.median(np.linalg.norm(proj - pts, axis=1)))
            if err < best_err:
                best_err = err
                best_proj = proj
                best_H = H
        if best_proj is None or best_err > 80:
            return keypoints, None, None
        try:
            H_inv = np.linalg.inv(best_H)
        except np.linalg.LinAlgError:
            H_inv = None
        return best_proj.flatten().astype(np.float32), best_H, H_inv

    def assess_quality(self, keypoints: np.ndarray, image_shape) -> Tuple[bool, str]:
        """粗查这组点像不像一个能用的球场。"""
        h, w = image_shape[:2]
        pts = keypoints.reshape(14, 2)
        if np.any(~np.isfinite(pts)):
            return False, "存在无效坐标"
        if np.any(pts[:, 0] < -0.1 * w) or np.any(pts[:, 0] > 1.1 * w):
            return False, "关键点明显超出画面"
        if np.any(pts[:, 1] < -0.1 * h) or np.any(pts[:, 1] > 1.1 * h):
            return False, "关键点明显超出画面"
        quad = pts[[0, 1, 3, 2]].astype(np.float32)
        area = abs(cv2.contourArea(quad))
        img_area = float(w * h)
        if area < 0.04 * img_area:
            return False, "球场四边形过小"
        if (pts[:, 0].max() - pts[:, 0].min()) < 0.15 * w:
            return False, "关键点横向挤在一起"
        far_y = float((pts[0, 1] + pts[1, 1]) / 2)
        near_y = float((pts[2, 1] + pts[3, 1]) / 2)
        if near_y <= far_y + 8:
            return False, "远近底线上下关系不对"
        return True, "ok"

    def _resnet_usable(self, keypoints: np.ndarray, image: np.ndarray, mask) -> Tuple[bool, str]:
        """ResNet 结果必须几何合理，并且预测出的球场多边形要盖在场地颜色上。

        你那张水平机位图里 ResNet 仍报 ok，是因为不少点碰巧落在蓝色场地区域附近。
        加上「整块四边形和场地掩膜的重合比例」后，漂在空中的框会被拒掉。
        """
        ok, reason = self.assess_quality(keypoints, image.shape)
        if not ok:
            return False, reason
        if mask is None:
            return False, "没有场地颜色块，无法确认点是否在场上"
        pts = keypoints.reshape(14, 2)
        hit = keypoints_on_mask(pts, mask)
        if hit < 10:
            return False, f"只有{hit}/14个点落在场地上"
        iou = self._quad_on_mask(pts, mask)
        if iou < 0.45:
            return False, f"球场框和场地颜色重合太低({iou:.2f})，像水平机位误检"
        return True, "ok"

    def _quad_on_mask(self, pts: np.ndarray, mask: np.ndarray) -> float:
        h, w = mask.shape[:2]
        poly = np.zeros((h, w), np.uint8)
        quad = np.round(pts[[0, 1, 3, 2]]).astype(np.int32)
        try:
            cv2.fillConvexPoly(poly, quad, 255)
        except cv2.error:
            return 0.0
        inter = cv2.countNonZero(cv2.bitwise_and(poly, mask))
        area = cv2.countNonZero(poly)
        if area < 1:
            return 0.0
        return inter / float(area)

    def _pack(
        self,
        keypoints: np.ndarray,
        H,
        H_inv,
        image: np.ndarray,
        frame_id: int,
        method: str,
        quality_ok: bool,
        reason: str,
    ) -> CourtDetection:
        return CourtDetection(
            keypoints=keypoints,
            keypoints_xy=keypoints.reshape(14, 2),
            homography_ref_to_image=H,
            homography_image_to_ref=H_inv,
            quality_ok=quality_ok,
            quality_reason=reason,
            frame_id=frame_id,
            method=method,
        )

    def _geometric_detection(self, image: np.ndarray, frame_id: int) -> Optional[CourtDetection]:
        geo = detect_geometric(image)
        if geo is None:
            return None
        keypoints, H, H_inv, mask, color_name = geo
        ok, reason = self.assess_quality(keypoints, image.shape)
        hit = keypoints_on_mask(keypoints.reshape(14, 2), mask)
        if hit < 5:
            ok = False
            reason = f"几何投影只有{hit}个点落在{color_name}场地上"
        elif ok:
            reason = f"geometric/{color_name}"
        return self._pack(
            keypoints, H, H_inv, image, frame_id, "geometric", ok, reason
        )

    def _heatmap_detection(self, image: np.ndarray, frame_id: int) -> Optional[CourtDetection]:
        if self.heatmap is None:
            return None
        keypoints, H, H_inv = self.heatmap.predict(image)
        if keypoints is None or np.any(~np.isfinite(keypoints)):
            return None
        keypoints = align_keypoints_to_white_lines(image, keypoints)
        keypoints, H, H_inv = self._homography_snap(keypoints)
        keypoints = self._refine(image, keypoints)
        keypoints, H2, H_inv2 = self._homography_snap(keypoints)
        if H2 is not None:
            H, H_inv = H2, H_inv2
        mask, _ = adaptive_surface_mask(image)
        ok, reason = self.assess_quality(keypoints, image.shape)
        if mask is not None:
            iou = self._quad_on_mask(keypoints.reshape(14, 2), mask)
            if iou < 0.35:
                ok = False
                reason = f"热力图球场框和场地重合低({iou:.2f})"
            elif ok:
                reason = f"heatmap iou={iou:.2f}"
        return self._pack(
            keypoints, H, H_inv, image, frame_id, "heatmap", ok, reason
        )

    def _resnet_detection(self, image: np.ndarray, frame_id: int, mask) -> Optional[CourtDetection]:
        if self.model is None:
            return None
        keypoints = self._predict_raw(image)
        keypoints = self._refine(image, keypoints)
        keypoints, H, H_inv = self._homography_snap(keypoints)
        usable, reason = self._resnet_usable(keypoints, image, mask)
        return self._pack(
            keypoints, H, H_inv, image, frame_id, "resnet", usable, reason
        )

    def predict(self, image: np.ndarray, frame_id: int = 0) -> CourtDetection:
        """优先热力图，其次几何法，最后才是 ResNet。"""
        heat = self._heatmap_detection(image, frame_id)
        if heat is not None and heat.quality_ok:
            return heat
        geo = self._geometric_detection(image, frame_id)
        if geo is not None and geo.quality_ok:
            return geo
        mask, _ = adaptive_surface_mask(image)
        resnet = self._resnet_detection(image, frame_id, mask)
        if resnet is not None and resnet.quality_ok:
            return resnet
        for cand in (heat, geo, resnet):
            if cand is not None:
                return cand
        raise RuntimeError("球场检测失败：热力图、几何法、ResNet 都不可用")

    def predict_video_sampled(
        self,
        read_frame_fn,
        sample_ids: List[int],
    ) -> CourtDetection:
        """在抽到的几帧上跑，优先热力图。"""
        frames = []
        for fid in sample_ids:
            image = read_frame_fn(fid)
            if image is None:
                continue
            frames.append((fid, image))
        if not frames:
            raise RuntimeError("没有读到可用于球场检测的帧")

        best = None
        for fid, image in frames:
            det = self.predict(image, frame_id=fid)
            if det is None:
                continue
            area = abs(cv2.contourArea(det.keypoints_xy[[0, 1, 3, 2]].astype(np.float32)))
            score = area + (1e12 if det.quality_ok else 0)
            if best is None or score > best[0]:
                best = (score, det)
        if best is None:
            raise RuntimeError("球场检测失败")
        return best[1]

    def draw(self, image: np.ndarray, detection: CourtDetection) -> np.ndarray:
        """把点和线画到图上，方便肉眼检查对不对。"""
        output = image.copy()
        pts = detection.keypoints_xy
        color = (0, 255, 0) if detection.quality_ok else (0, 0, 255)
        for a, b in COURT_LINES:
            p1 = (int(pts[a, 0]), int(pts[a, 1]))
            p2 = (int(pts[b, 0]), int(pts[b, 1]))
            cv2.line(output, p1, p2, color, 2)
        for i in range(14):
            x, y = int(pts[i, 0]), int(pts[i, 1])
            cv2.circle(output, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                output,
                str(i),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        tag = f"{detection.method} ok" if detection.quality_ok else f"{detection.method} warn"
        cv2.putText(
            output,
            tag[:70],
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        return output


def read_frame_at(cap, frame_id):
    """跳到指定帧号再读一张。比把整段视频装进内存安全。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def choose_sample_ids(total_frames, n_samples):
    """在视频前半段均匀抽帧，避免只信可能是特写的第 0 帧。"""
    if total_frames <= 0:
        return [0]
    n_samples = max(1, n_samples)
    span = min(total_frames - 1, max(total_frames // 3, 1), 150)
    if span < n_samples:
        return list(range(max(1, span + 1)))
    return [
        int(i * span / (n_samples - 1)) if n_samples > 1 else 0
        for i in range(n_samples)
    ]
